"""kira-ai-plugin-arxiv arXiv 核心客户端

从 kira-ai-plugin-arxiv-search 合并而来：查询/详情/PDF 下载/源码下载，
遵守 arXiv API 礼貌间隔，结果带 TTL 缓存，ID 白名单正则校验防路径穿越。
"""
import asyncio
import gzip
import io
import os
import re
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx

ATOM_NS = "http://www.w3.org/2005/Atom"
ARXIV_NS = "http://arxiv.org/schemas/atom"

API_BASE = "https://export.arxiv.org/api/query"
PDF_BASE = "https://arxiv.org/pdf/"

# arXiv API 官方要求请求携带 User-Agent（无 UA 易被限流/封禁）；可用配置 user_agent 覆盖
DEFAULT_USER_AGENT = "KiraAI-arxiv-plugin/2.0 (arxiv search plugin for KiraAI bot; contact: bot-admin)"

# arXiv API 官方建议两次请求间隔 >= 3 秒
MIN_API_INTERVAL = 3.0
# PDF 并发下载上限
MAX_DOWNLOAD_CONCURRENCY = 3
# 结果缓存 TTL（秒）与最大条数
CACHE_TTL = 600.0
MAX_CACHE_SIZE = 200

# 新旧两种 arXiv ID 格式：2101.00001 或 math/0101011（可带 vN 版本号）
ARXIV_ID_RE = re.compile(
    r"^(?:\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?$",
    re.IGNORECASE,
)

# 模块级节流状态（跨实例共享，防止多用户同时打爆 API）
_api_lock = asyncio.Lock()
_last_api_call = 0.0
_download_sem = asyncio.Semaphore(MAX_DOWNLOAD_CONCURRENCY)


class ArxivApiError(Exception):
    """arXiv API 请求 / 解析 / 下载相关错误"""


class ArxivClient:
    """arXiv 查询与下载客户端（配置通过参数注入，不依赖插件上下文）"""

    def __init__(
        self,
        download_dir: Path,
        source_dir: Path,
        timeout: float = 15.0,
        sort_by: str = "relevance",
        max_results: int = 5,
        user_agent: str = "",
    ):
        self.download_dir = download_dir
        self.source_dir = source_dir
        self.timeout = timeout
        self.user_agent = (user_agent or "").strip() or DEFAULT_USER_AGENT
        self.sort_by = sort_by if sort_by in ("relevance", "submittedDate", "lastUpdatedDate") else "relevance"
        self.max_results = max(1, min(int(max_results or 5), 20))
        self._cache: Dict[str, Tuple[float, List[dict]]] = {}

    # ── 缓存 ──────────────────────────────────────────────
    def _cache_get(self, key: str) -> Optional[List[dict]]:
        item = self._cache.get(key)
        if item and time.monotonic() - item[0] < CACHE_TTL:
            return item[1]
        return None

    def _cache_set(self, key: str, value: List[dict]) -> None:
        if len(self._cache) >= MAX_CACHE_SIZE:
            now = time.monotonic()
            for k in [k for k, v in self._cache.items() if now - v[0] >= CACHE_TTL]:
                self._cache.pop(k, None)
        self._cache[key] = (time.monotonic(), value)

    # ── 解析 ──────────────────────────────────────────────
    def _parse_entry(self, entry: ET.Element) -> dict:
        def _text(tag: str, ns: str = ATOM_NS) -> str:
            node = entry.find(f"{{{ns}}}{tag}")
            if node is None:
                return ""
            return " ".join("".join(node.itertext()).split())

        authors = []
        for author in entry.findall(f"{{{ATOM_NS}}}author"):
            name = author.findtext(f"{{{ATOM_NS}}}name") or ""
            name = " ".join(name.split())
            if name:
                authors.append(name)

        pdf_url = ""
        abs_url = ""
        for link in entry.findall(f"{{{ATOM_NS}}}link"):
            if link.get("title") == "pdf" and not pdf_url:
                pdf_url = link.get("href") or ""
            if link.get("rel") == "alternate" and not abs_url:
                abs_url = link.get("href") or ""

        entry_id = _text("id")
        paper_id = ""
        if entry_id:
            paper_id = entry_id.rstrip("/").rsplit("/", 1)[-1]

        primary = entry.find(f"{{{ARXIV_NS}}}primary_category")
        categories = [
            c.get("term")
            for c in entry.findall(f"{{{ATOM_NS}}}category")
            if c.get("term")
        ]
        if primary is not None and primary.get("term") not in categories:
            categories.insert(0, primary.get("term"))

        return {
            "id": paper_id,
            "title": _text("title") or "（无标题）",
            "summary": _text("summary"),
            "authors": authors,
            "published": _text("published"),
            "updated": _text("updated"),
            "primary_category": primary.get("term") if primary is not None else "",
            "categories": categories,
            "comment": _text("comment", ARXIV_NS),
            "doi": _text("doi", ARXIV_NS),
            "pdf_url": pdf_url or (f"{PDF_BASE}{paper_id}" if paper_id else ""),
            "abs_url": abs_url or (f"https://arxiv.org/abs/{paper_id}" if paper_id else ""),
        }

    # ── API ───────────────────────────────────────────────
    async def _api_query(self, params: dict) -> List[dict]:
        global _last_api_call
        timeout = self.timeout
        # 限流/服务端错误重试：429/5xx → 指数退避，最多 3 次
        max_retries = 3
        last_exc: Optional[Exception] = None
        async with _api_lock:
            now = time.monotonic()
            wait = MIN_API_INTERVAL - (now - _last_api_call)
            if wait > 0:
                await asyncio.sleep(wait)
            for attempt in range(max_retries):
                try:
                    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True,
                                                 headers={"User-Agent": self.user_agent}) as client:
                        resp = await client.get(API_BASE, params=params)
                    if resp.status_code in (429,) or resp.status_code >= 500:
                        retry_after = 0.0
                        ra = resp.headers.get("Retry-After")
                        if ra:
                            try:
                                retry_after = float(ra)
                            except ValueError:
                                retry_after = 0.0
                        backoff = max(retry_after, 2 ** (attempt + 1))  # 2s/4s/8s 指数退避
                        print(f"[arxiv] HTTP {resp.status_code}，{backoff:.0f}s 后重试 "
                              f"({attempt + 1}/{max_retries})", flush=True)
                        await asyncio.sleep(backoff)
                        continue
                    resp.raise_for_status()
                    content = resp.content
                    _last_api_call = time.monotonic()
                    break
                except httpx.HTTPError as e:
                    last_exc = e
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue
            else:
                raise ArxivApiError(
                    f"arXiv API 请求失败（重试 {max_retries} 次后仍失败）: {last_exc}"
                ) from last_exc
        try:
            root = ET.fromstring(content)
        except ET.ParseError as e:
            raise ArxivApiError(f"arXiv API 返回解析失败: {e}") from e
        return [self._parse_entry(entry) for entry in root.findall(f"{{{ATOM_NS}}}entry")]

    @staticmethod
    def _normalize_search_query(raw: str) -> str:
        query = raw.strip()
        if not query:
            raise ValueError("搜索关键词不能为空")
        if re.match(r"^(?:all|ti|au|abs|co|jr|cat|rn):", query, re.IGNORECASE):
            return query
        return f"all:{query}"

    async def search(self, query: str, max_results: int = 5) -> List[dict]:
        search_query = self._normalize_search_query(query)
        limit = max(1, min(int(max_results or self.max_results), 20))
        cache_key = f"search|{search_query}|{limit}|{self.sort_by}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached
        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": limit,
            "sortBy": self.sort_by,
        }
        results = await self._api_query(params)
        self._cache_set(cache_key, results)
        return results

    async def get_by_id(self, arxiv_id: str) -> Optional[dict]:
        cache_key = f"id|{arxiv_id.lower()}"
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached[0] if cached else None
        results = await self._api_query({"id_list": arxiv_id})
        paper = results[0] if results else None
        self._cache_set(cache_key, [paper] if paper else [])
        return paper

    @staticmethod
    def sanitize_id(arxiv_id: str) -> str:
        """校验并规范化 arXiv ID，仅保留安全字符，防止路径穿越。"""
        raw = (arxiv_id or "").strip().lower()
        if not raw:
            raise ValueError("arXiv ID 不能为空")
        if not ARXIV_ID_RE.match(raw):
            raise ValueError(
                f"无效的 arXiv ID: {arxiv_id!r}（格式示例：1706.03762 或 math/0101011v1）"
            )
        return raw.replace("/", "_")

    # ── 下载 ──────────────────────────────────────────────
    async def download_pdf(self, arxiv_id: str) -> Tuple[str, int]:
        """下载 PDF 到下载目录，返回 (本地绝对路径, 字节数)。临时文件 + os.replace 原子落盘。"""
        safe_id = self.sanitize_id(arxiv_id)
        save_dir = self.download_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        final_path = save_dir / f"{safe_id}.pdf"
        url = f"{PDF_BASE}{arxiv_id}"
        timeout = self.timeout * 2

        tmp_path = None
        async with _download_sem:
            try:
                async with httpx.AsyncClient(timeout=timeout, follow_redirects=True,
                                             headers={"User-Agent": self.user_agent}) as client:
                    async with client.stream("GET", url) as resp:
                        resp.raise_for_status()
                        fd, tmp_name = tempfile.mkstemp(
                            prefix=f".{safe_id}.", suffix=".part", dir=str(save_dir)
                        )
                        tmp_path = Path(tmp_name)
                        size = 0
                        with os.fdopen(fd, "wb") as fh:
                            async for chunk in resp.aiter_bytes(65536):
                                fh.write(chunk)
                                size += len(chunk)
            except httpx.HTTPError as e:
                if tmp_path:
                    tmp_path.unlink(missing_ok=True)
                raise ArxivApiError(f"PDF 下载失败（{arxiv_id}）: {e}") from e
            except OSError as e:
                if tmp_path:
                    tmp_path.unlink(missing_ok=True)
                raise ArxivApiError(f"PDF 写入失败（{arxiv_id}）: {e}") from e

        try:
            with open(tmp_path, "rb") as fh:
                head = fh.read(5)
        except OSError as e:
            tmp_path.unlink(missing_ok=True)
            raise ArxivApiError(f"PDF 校验失败（{arxiv_id}）: {e}") from e
        if head[:4] != b"%PDF":
            tmp_path.unlink(missing_ok=True)
            raise ArxivApiError(
                f"下载内容不是有效的 PDF 文件（{arxiv_id}），可能是 ID 不存在或 arXiv 返回了错误页"
            )
        os.replace(tmp_path, final_path)
        return str(final_path), size

    @staticmethod
    def _infer_src_extension(content_type: str, head: bytes) -> str:
        ct = (content_type or "").lower()
        if "eprint-tar" in ct or ("tar" in ct and "gzip" in ct):
            return ".tar.gz"
        if "x-eprint" in ct or "gzip" in ct:
            return ".tex.gz"
        if "tar" in ct:
            return ".tar"
        if "tex" in ct or "plain" in ct:
            return ".tex"
        if head[:2] == b"\x1f\x8b":
            try:
                with gzip.GzipFile(fileobj=io.BytesIO(head)) as gz:
                    decompressed = gz.read(1024)
                if decompressed[257:262] == b"ustar":
                    return ".tar.gz"
            except (OSError, EOFError, ValueError):
                pass
            return ".tex.gz"
        if head[257:262] == b"ustar":
            return ".tar"
        return ".tex"

    async def download_src(self, arxiv_id: str) -> Tuple[str, int]:
        """下载 LaTeX 源码包（e-print）到源码目录，返回 (本地绝对路径, 字节数)。"""
        safe_id = self.sanitize_id(arxiv_id)
        save_dir = self.source_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        url = f"https://arxiv.org/e-print/{arxiv_id}"
        timeout = self.timeout * 2

        tmp_path = None
        content_type = ""
        size = 0
        async with _download_sem:
            try:
                async with httpx.AsyncClient(timeout=timeout, follow_redirects=True,
                                             headers={"User-Agent": self.user_agent}) as client:
                    async with client.stream("GET", url) as resp:
                        resp.raise_for_status()
                        content_type = resp.headers.get("content-type", "")
                        fd, tmp_name = tempfile.mkstemp(
                            prefix=f".{safe_id}.", suffix=".part", dir=str(save_dir)
                        )
                        tmp_path = Path(tmp_name)
                        with os.fdopen(fd, "wb") as fh:
                            async for chunk in resp.aiter_bytes(65536):
                                fh.write(chunk)
                                size += len(chunk)
            except httpx.HTTPError as e:
                if tmp_path:
                    tmp_path.unlink(missing_ok=True)
                raise ArxivApiError(f"源码下载失败（{arxiv_id}）: {e}") from e
            except OSError as e:
                if tmp_path:
                    tmp_path.unlink(missing_ok=True)
                raise ArxivApiError(f"源码写入失败（{arxiv_id}）: {e}") from e

        try:
            with open(tmp_path, "rb") as fh:
                head = fh.read(1024)
        except OSError as e:
            tmp_path.unlink(missing_ok=True)
            raise ArxivApiError(f"源码校验失败（{arxiv_id}）: {e}") from e
        if size <= 0:
            tmp_path.unlink(missing_ok=True)
            raise ArxivApiError(f"下载内容为空（{arxiv_id}），可能该论文没有公开的 LaTeX 源码")
        stripped = head.lstrip()[:256].lower()
        if stripped[:5] in (b"<html", b"<!doc") or b"404" in stripped:
            tmp_path.unlink(missing_ok=True)
            raise ArxivApiError(
                f"下载到的是错误页面而非源码包（{arxiv_id}），可能是 ID 不存在或该论文未公开源码"
            )
        ext = self._infer_src_extension(content_type, head)
        final_path = save_dir / f"{safe_id}{ext}"
        os.replace(tmp_path, final_path)
        return str(final_path), size

    # ── 格式化（供插件层与 LLM 工具复用） ──────────────────
    @staticmethod
    def _truncate(text: str, length: int) -> str:
        text = re.sub(r"\s+", " ", text or "").strip()
        if len(text) <= length:
            return text
        return text[: length - 1].rstrip() + "…"

    @staticmethod
    def _fmt_date(value: str) -> str:
        if not value:
            return ""
        return value[:10]

    @staticmethod
    def _fmt_size(size: int) -> str:
        if size < 1024:
            return f"{size} B"
        if size < 1024 * 1024:
            return f"{size / 1024:.1f} KB"
        return f"{size / 1024 / 1024:.2f} MB"

    def format_search_results(self, query: str, papers: List[dict]) -> str:
        if not papers:
            return f"❌ 未在 arXiv 找到与「{query}」相关的论文"
        lines = [f"📚 arXiv 搜索结果：{query}（共 {len(papers)} 条）", ""]
        for i, paper in enumerate(papers, 1):
            authors = ", ".join(paper["authors"][:3])
            if len(paper["authors"]) > 3:
                authors += f" 等{len(paper['authors'])}人"
            category = paper["primary_category"] or (",".join(paper["categories"][:2]) if paper["categories"] else "-")
            lines.append(f"{i}. {paper['title']}")
            lines.append(f"   📎 arXiv:{paper['id']} | 🏷 {category}")
            if authors:
                lines.append(f"   👤 {authors}")
            lines.append(f"   🗓 {self._fmt_date(paper['published']) or '-'}")
            lines.append(f"   📝 {self._truncate(paper['summary'], 180)}")
            lines.append(f"   🔗 PDF: {paper['pdf_url']}")
            lines.append("")
        return "\n".join(lines).rstrip()

    def format_paper_detail(self, paper: dict) -> str:
        lines = [f"📄 {paper['title']}", ""]
        lines.append(f"🔖 arXiv ID: {paper['id']}")
        if paper["authors"]:
            lines.append(f"👥 作者({len(paper['authors'])}): {', '.join(paper['authors'])}")
        published = self._fmt_date(paper["published"])
        updated = self._fmt_date(paper["updated"])
        if published and updated and updated != published:
            lines.append(f"🗓 发布: {published} | 更新: {updated}")
        elif published:
            lines.append(f"🗓 发布: {published}")
        if paper["categories"]:
            lines.append(f"🏷 分类: {', '.join(paper['categories'])}")
        if paper.get("comment"):
            lines.append(f"💬 备注: {self._truncate(paper['comment'], 120)}")
        if paper.get("doi"):
            lines.append(f"🔗 DOI: {paper['doi']}")
        lines.append("")
        lines.append(f"📝 摘要: {paper['summary']}")
        if paper["pdf_url"]:
            lines.append(f"🔗 PDF: {paper['pdf_url']}")
        if paper["abs_url"]:
            lines.append(f"🌐 页面: {paper['abs_url']}")
        return "\n".join(lines)

    def format_download_result(self, paper: dict, local_path: str, size: int) -> str:
        lines = ["✅ 论文下载成功", ""]
        lines.append(f"📄 {paper['title']}")
        lines.append(f"🔖 arXiv ID: {paper['id']}")
        lines.append(f"📁 本地路径: {local_path}")
        lines.append(f"📦 文件大小: {self._fmt_size(size)}")
        lines.append(f"🔗 在线: {paper['pdf_url']}")
        return "\n".join(lines)

    def format_src_result(self, arxiv_id: str, local_path: str, size: int) -> str:
        lines = ["✅ 源码下载成功", ""]
        lines.append(f"🔖 arXiv ID: {arxiv_id}")
        lines.append(f"📁 本地路径: {local_path}")
        lines.append(f"📦 文件大小: {self._fmt_size(size)}")
        lines.append(f"🔗 在线: https://arxiv.org/e-print/{arxiv_id}")
        return "\n".join(lines)
