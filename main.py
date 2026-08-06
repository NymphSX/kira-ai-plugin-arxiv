"""
KiraAI arXiv 学术助手插件 (kira-ai-plugin-arxiv) v2.0.0

由三个插件合并而来：
- kira-ai-plugin-arxiv-search（arXiv 查询/下载/摘要翻译）
- kira-ai-plugin-translate（多后端文本翻译引擎）
- kira-ai-plugin-pdf-gen（PDF 生成）

功能：
- /arxiv search/get/tr/dl/src 斜杠命令（前缀可配置，默认 /arxiv）
- LLM 工具：arxiv_search / arxiv_get / arxiv_translate / arxiv_download / arxiv_src
           / parse_arxiv_command / translate / generate_pdf
- 摘要翻译默认走快速模型，可切换到内置多后端翻译引擎（百度/DeepL/Google/阿里/本地）

实现要点：
- arXiv API 礼貌间隔（>=3s）节流 + TTL 缓存 + 原子落盘下载
- 翻译引擎带配额/限流/缓存/后端自动回退
- 输出目录统一：data/files/arxiv_pdf、data/files/arxiv_src
"""

import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

import httpx

from core.plugin import BasePlugin, logger, register, on, Priority
from core.chat.message_utils import MessageChain, KiraMessageEvent
from core.chat.message_elements import Text, At
from core.provider import LLMRequest

from .arxiv_core import ArxivClient, ArxivApiError
from .translate_engine import TranslationEngine

log = logging.getLogger(__name__)


class ArxivPlugin(BasePlugin):
    """arXiv 论文查询、翻译与下载，内置多后端翻译与 PDF 生成"""

    SELF_PLUGIN_ID = "kira-ai-plugin-arxiv"

    # ---------------------------------------------------------------
    # 生命周期
    # ---------------------------------------------------------------

    def __init__(self, ctx, cfg: dict):
        super().__init__(ctx, cfg)
        self.download_dir = self._resolve_dir("download_dir", "data/files/arxiv_pdf")
        self.source_dir = self._resolve_dir("source_dir", "data/files/arxiv_src")
        timeout = float(self._cfg("request_timeout", 15) or 15)
        sort_by = self._cfg("sort_by", "relevance") or "relevance"
        max_results = int(self._cfg("max_results", 5) or 5)
        self.client = ArxivClient(
            download_dir=self.download_dir,
            source_dir=self.source_dir,
            timeout=timeout,
            sort_by=sort_by,
            max_results=max_results,
        )
        engine_cfg = {
            "default_backend": self._cfg("default_backend", "auto"),
            "baidu_appid": self._cfg("baidu_appid", ""),
            "baidu_secret_key": self._cfg("baidu_secret_key", ""),
            "deepl_api_key": self._cfg("deepl_api_key", ""),
            "deepl_pro": self._cfg("deepl_pro", False),
            "google_api_key": self._cfg("google_api_key", ""),
            "aliyun_access_key_id": self._cfg("aliyun_access_key_id", ""),
            "aliyun_access_key_secret": self._cfg("aliyun_access_key_secret", ""),
            "aliyun_region": self._cfg("aliyun_region", "cn-hangzhou"),
            "local_backend_url": self._cfg("local_backend_url", ""),
            "local_model": self._cfg("local_model", ""),
            "local_timeout": self._cfg("local_timeout", 120),
            "max_chars_per_call": self._cfg("max_chars_per_call", 5000),
            "max_chars_per_day": self._cfg("max_chars_per_day", 10000),
            "max_queries_per_min": self._cfg("max_queries_per_min", 30),
            "enable_cache": self._cfg("enable_cache", True),
        }
        self.engine = TranslationEngine(engine_cfg)
        self._pdf_dir = self._resolve_dir("pdf_dir", "data/files")

    def _cfg(self, key: str, default=None):
        """读取配置：优先顶层字段，其次扫描各 section 下的字段。"""
        cfg = self.plugin_cfg or {}
        if key in cfg:
            return cfg.get(key, default)
        for value in cfg.values():
            if isinstance(value, dict) and key in value:
                return value.get(key, default)
        return default

    def _resolve_dir(self, key: str, default: str) -> Path:
        cfg_dir = (self._cfg(key, "") or "").strip()
        base = Path(cfg_dir) if cfg_dir else Path(default)
        if not base.is_absolute():
            base = Path.cwd() / base
        return base

    async def on_load(self):
        logger.info("arXiv 插件已加载，PDF 目录: %s，源码目录: %s", self.download_dir, self.source_dir)
        for _dir in (self.download_dir, self.source_dir, self._pdf_dir):
            try:
                _dir.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                logger.error("创建目录失败（%s）: %s", _dir, e)

    async def on_unload(self):
        logger.info("arXiv 插件已卸载")

    async def initialize(self):
        await self.engine.initialize()
        await self.on_load()

    async def terminate(self):
        await self.engine.close()
        await self.on_unload()

    # ---------------------------------------------------------------
    # 摘要翻译（fast=快速模型；其他走内置翻译引擎）
    # ---------------------------------------------------------------

    async def _translate_lines(
        self, lines: List[str], target: str = "zh", client=None, fallback: bool = True
    ) -> Optional[List[str]]:
        """批量翻译多行文本（每行一条）。

        默认走快速 LLM（原 arxiv-search 行为）；若配置 translate_backend 指定了
        翻译引擎后端（auto/baidu/.../local），则改走内置 TranslationEngine。
        """
        if not lines:
            return lines
        if not self._cfg("translate_enabled", True):
            return lines if fallback else None

        backend = (self._cfg("translate_backend", "fast") or "fast").strip()
        if backend != "fast":
            try:
                sid = "arxiv-summary"
                result = await self.engine.translate(
                    "\n".join(lines), target, "auto", backend, sid=sid
                )
            except Exception as e:
                logger.warning("arXiv 摘要翻译引擎失败: %s", e)
                return lines if fallback else None
            if result.startswith("✅"):
                body = result[1:].split("\n（后端:", 1)[0].strip()
                translated = [x for x in (body or "").splitlines() if x.strip()]
                if len(translated) == len(lines):
                    return translated
            return lines if fallback else None

        # 默认：快速 LLM
        if client is None:
            client = self.ctx.get_default_fast_llm_client()
        if not client:
            return lines if fallback else None
        try:
            numbered = "\n".join(f"{i+1}. {line}" for i, line in enumerate(lines))
            prompt = (
                f"请将以下 {len(lines)} 条文本逐条翻译成{target}，"
                f"严格保持编号格式，每条一行，只输出翻译结果，不要任何解释。\n\n{numbered}"
            )
            request = LLMRequest(messages=[{"role": "user", "content": prompt}])
            response = await client.chat(request)
            result = (response.text_response or "").strip()
            translated: List[str] = []
            for line in result.splitlines():
                line = line.strip()
                m = re.match(r"^\d+[.、:：]\s*(.*)$", line)
                translated.append(m.group(1).strip() if m else line)
            if len(translated) != len(lines) or any(not x for x in translated):
                return lines if fallback else None
            return translated
        except Exception as e:
            logger.warning("arXiv 翻译失败: %s", e)
            return lines if fallback else None

    # ---------------------------------------------------------------
    # LLM 工具 1：搜索
    # ---------------------------------------------------------------

    @register.tool(
        "arxiv_search",
        "在 arXiv 学术预印本库中按关键词搜索论文，返回标题、摘要、作者、arXiv ID、分类和 PDF 下载链接。"
        "支持 arXiv 高级查询语法（如 au:vaswani、ti:attention、cat:cs.CL、abs:deep learning），"
        "多个词默认按 AND 组合，可通过 AND/OR/NOT 与括号组合。",
        {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词，例如 'large language model' 或高级语法 'au:vaswani AND ti:attention'"
                },
                "max_results": {
                    "type": "integer",
                    "description": "返回条数，默认 5，最大 20",
                    "default": 5
                },
                "translate": {
                    "type": "boolean",
                    "description": "是否将各论文标题翻译为目标语言（默认 false）",
                    "default": False
                }
            },
            "required": ["query"]
        }
    )
    async def tool_arxiv_search(self, event, query: str, max_results: int = 5, translate: bool = False):
        """按关键词搜索 arXiv 论文。"""
        try:
            papers = await self.client.search(query, max_results)
        except ArxivApiError as e:
            return f"❌ {e}"
        except Exception as e:
            logger.exception("arxiv_search 未预期异常")
            return f"❌ 搜索失败：{type(e).__name__}: {e}"
        result = self.client.format_search_results(query, papers)
        if translate:
            titles = [p["title"] for p in papers]
            translated = await self._translate_lines(titles)
            if translated and len(translated) == len(titles):
                lines = ["", "【标题译文】"]
                for i, t in enumerate(translated, 1):
                    lines.append(f"{i}. {t}")
                result += "\n" + "\n".join(lines)
        return result

    # ---------------------------------------------------------------
    # LLM 工具 2：获取单篇详情
    # ---------------------------------------------------------------

    @register.tool(
        "arxiv_get",
        "根据 arXiv ID 获取单篇论文详情，包含完整标题、全部作者、摘要、发布日期、分类、PDF 链接。"
        "arXiv ID 示例：1706.03762 或 math/0101011。",
        {
            "type": "object",
            "properties": {
                "arxiv_id": {
                    "type": "string",
                    "description": "arXiv ID，例如 1706.03762"
                },
                "translate": {
                    "type": "boolean",
                    "description": "是否将标题与摘要翻译为目标语言（默认 false）",
                    "default": False
                }
            },
            "required": ["arxiv_id"]
        }
    )
    async def tool_arxiv_get(self, event, arxiv_id: str, translate: bool = False):
        """根据 arXiv ID 获取单篇论文详情。"""
        try:
            self.client.sanitize_id(arxiv_id)
        except ValueError as e:
            return f"❌ {e}"
        try:
            paper = await self.client.get_by_id(arxiv_id.strip())
        except ArxivApiError as e:
            return f"❌ {e}"
        except Exception as e:
            logger.exception("arxiv_get 未预期异常")
            return f"❌ 获取失败：{type(e).__name__}: {e}"
        if not paper:
            return f"❌ 未找到 arXiv 论文：{arxiv_id.strip()}"
        result = self.client.format_paper_detail(paper)
        if translate:
            translated = await self._translate_lines([paper["title"], paper["summary"]])
            if translated and len(translated) == 2:
                t_title, t_summary = translated
                result += "\n\n【标题译文】" + t_title
                result += "\n\n【摘要译文】" + t_summary
        return result

    # ---------------------------------------------------------------
    # LLM 工具 3：下载 PDF
    # ---------------------------------------------------------------

    @register.tool(
        "arxiv_download",
        "根据 arXiv ID 下载论文 PDF 到本地 data/files/arxiv_pdf/ 目录，返回本地文件路径、大小与在线链接。"
        "支持一次传入多个 ID（空格或逗号分隔），将并发下载。arXiv ID 示例：1706.03762。",
        {
            "type": "object",
            "properties": {
                "arxiv_id": {
                    "type": "string",
                    "description": "arXiv ID，或空格/逗号分隔的多个 ID，例如 '1706.03762 2105.02723'"
                }
            },
            "required": ["arxiv_id"]
        }
    )
    async def tool_arxiv_download(self, event, arxiv_id: str):
        """下载论文 PDF 到本地。"""
        import asyncio as _asyncio
        ids = [x.strip() for x in re.split(r"[\s,]+", arxiv_id or "") if x.strip()]
        if not ids:
            return "❌ 请提供 arXiv ID，例如：/arxiv dl 1706.03762"
        for raw in ids:
            try:
                self.client.sanitize_id(raw)
            except ValueError as e:
                return f"❌ {e}"
        if len(ids) == 1:
            return await self._download_one(ids[0])
        results = await _asyncio.gather(
            *(self._download_one(pid) for pid in ids),
            return_exceptions=True,
        )
        blocks = []
        for pid, res in zip(ids, results):
            if isinstance(res, BaseException):
                blocks.append(f"❌ {pid}: 下载失败（{type(res).__name__}: {res}）")
            else:
                blocks.append(res)
        return "\n\n".join(blocks)

    async def _download_one(self, arxiv_id: str) -> str:
        paper = None
        try:
            paper = await self.client.get_by_id(arxiv_id)
        except ArxivApiError as e:
            return f"❌ {e}"
        if not paper:
            return f"❌ 未找到 arXiv 论文：{arxiv_id}（ID 可能不存在，请先 /arxiv search 确认）"
        try:
            local_path, size = await self.client.download_pdf(paper["id"])
        except ArxivApiError as e:
            return f"❌ {e}"
        except Exception as e:
            logger.exception("PDF 下载未预期异常")
            return f"❌ PDF 下载失败：{type(e).__name__}: {e}"
        return self.client.format_download_result(paper, local_path, size)

    # ---------------------------------------------------------------
    # LLM 工具 5：翻译标题/摘要
    # ---------------------------------------------------------------

    @register.tool(
        "arxiv_translate",
        "根据 arXiv ID 获取单篇论文，并将标题与摘要翻译成中文，便于快速了解论文大意。"
        "翻译默认使用快速模型；若翻译服务不可用会返回友好提示，可改用 arxiv_get 获取原文详情。"
        "arXiv ID 示例：1706.03762 或 math/0101011。",
        {
            "type": "object",
            "properties": {
                "arxiv_id": {
                    "type": "string",
                    "description": "arXiv ID，例如 1706.03762"
                }
            },
            "required": ["arxiv_id"]
        }
    )
    async def tool_arxiv_translate(self, event, arxiv_id: str):
        """根据 arXiv ID 将单篇论文的标题与摘要翻译成中文。"""
        try:
            self.client.sanitize_id(arxiv_id)
        except ValueError as e:
            return f"❌ {e}"
        try:
            paper = await self.client.get_by_id(arxiv_id.strip())
        except ArxivApiError as e:
            return f"❌ {e}"
        except Exception as e:
            logger.exception("arxiv_translate 未预期异常")
            return f"❌ 获取论文失败：{type(e).__name__}: {e}"
        if not paper:
            return f"❌ 未找到 arXiv 论文：{arxiv_id.strip()}"
        if not self._cfg("translate_enabled", True):
            return "❌ 翻译功能已在配置中关闭（translate_enabled=false），可先用 /arxiv get 查看原文"
        backend = (self._cfg("translate_backend", "fast") or "fast").strip()
        if backend == "fast":
            client = self.ctx.get_default_llm_client()
            if not client:
                return "❌ 翻译服务不可用，先试试 /arxiv get"
        else:
            client = None
        try:
            translated = await self._translate_lines(
                [paper["title"], paper["summary"]],
                target=self._cfg("translate_lang", "zh") or "zh",
                client=client,
                fallback=False,
            )
        except Exception as e:
            logger.warning("arxiv_translate LLM 调用异常: %s", e)
            return "❌ 翻译服务不可用，先试试 /arxiv get"
        if not translated or len(translated) != 2:
            return "❌ 翻译服务不可用，先试试 /arxiv get"
        t_title, t_summary = translated
        return "\n".join([
            f"📄 标题：{paper['title']}",
            "",
            f"🀄 译文标题：{t_title}",
            "",
            f"📝 摘要：{paper['summary']}",
            "",
            f"🀄 译文摘要：{t_summary}",
            "",
            f"🔖 arXiv ID: {paper['id']}",
        ])

    # ---------------------------------------------------------------
    # LLM 工具 6：下载 LaTeX 源码
    # ---------------------------------------------------------------

    @register.tool(
        "arxiv_src",
        "根据 arXiv ID 下载论文的 LaTeX 源码包（e-print，格式为 .tar.gz / .tex.gz / .tex）"
        "到本地 data/files/arxiv_src/ 目录，返回本地文件路径、大小与在线 e-print 链接。"
        "arXiv ID 示例：1706.03762。",
        {
            "type": "object",
            "properties": {
                "arxiv_id": {
                    "type": "string",
                    "description": "arXiv ID，例如 1706.03762"
                }
            },
            "required": ["arxiv_id"]
        }
    )
    async def tool_arxiv_src(self, event, arxiv_id: str):
        """下载论文 LaTeX 源码包（e-print）到本地。"""
        try:
            self.client.sanitize_id(arxiv_id)
        except ValueError as e:
            return f"❌ {e}"
        try:
            local_path, size = await self.client.download_src(arxiv_id.strip())
        except ArxivApiError as e:
            return f"❌ {e}"
        except Exception as e:
            logger.exception("arxiv_src 未预期异常")
            return f"❌ 源码下载失败：{type(e).__name__}: {e}"
        return self.client.format_src_result(arxiv_id.strip(), local_path, size)

    # ---------------------------------------------------------------
    # LLM 工具 4：代为执行斜杠命令
    # ---------------------------------------------------------------

    @register.tool(
        "parse_arxiv_command",
        "解析并执行 arXiv 插件的斜杠命令（默认前缀 /arxiv，可在插件配置中自定义）。"
        "当用户消息中出现斜杠命令（如 /arxiv search transformer）时调用本工具。"
        "支持子命令：search <关键词> 搜索论文；get <ID> 获取单篇详情；dl <ID> [多个] 下载 PDF；tr <ID> 翻译；src <ID> 下载源码；help 帮助。",
        {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "完整的斜杠命令文本，例如 '/arxiv search attention is all you need' 或 '/arxiv dl 1706.03762'"
                }
            },
            "required": ["command"]
        }
    )
    async def tool_parse_arxiv_command(self, event, command: str):
        """LLM 可调用本工具代为执行 /arxiv 斜杠命令。"""
        return await self._parse_and_execute(command or "", event)

    # ---------------------------------------------------------------
    # LLM 工具 7：多后端翻译（合并自 kira-ai-plugin-translate）
    # ---------------------------------------------------------------

    @register.tool(
        name="translate",
        description=(
            "将文本翻译成目标语言。自动检测源语言；支持多后端（百度/DeepL/Google/阿里云/本地模型），"
            "默认按配置自动回退。适用于对话翻译、长文翻译、术语查译。"
        ),
        params={
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "需要翻译的文本"},
                "target_lang": {
                    "type": "string",
                    "description": "目标语言代码：zh(中文) en(英语) ja(日语) ko(韩语) fr(法语) "
                                   "de(德语) es(西语) ru(俄语) pt(葡语) it(意语) nl(荷语) "
                                   "ar(阿语) hi(印地语) th(泰语) vi(越语) id(印尼语)",
                },
                "source_lang": {"type": "string", "description": "源语言代码，auto=自动检测（默认 auto）"},
                "backend": {"type": "string", "description": "指定后端：auto(默认，按配置回退)/baidu/deepl/google/aliyun/local"},
            },
            "required": ["text", "target_lang"],
        },
    )
    async def tool_translate(
        self,
        event,
        *_,
        text: str,
        target_lang: str,
        source_lang: str = "auto",
        backend: str = "auto",
    ) -> str:
        """翻译文本（含额度/限流/缓存/后端回退）"""
        if not self._cfg("enable_tool", True):
            return "❌ translate 工具已关闭（可在插件配置页开启）。"
        sid = getattr(event, "sid", None) or (
            event.session.session_id if event and event.session else "unknown"
        )
        return await self.engine.translate(text, target_lang, source_lang, backend, sid=sid)

    # ---------------------------------------------------------------
    # LLM 工具 8：PDF 生成（合并自 kira-ai-plugin-pdf-gen）
    # ---------------------------------------------------------------

    @register.tool(
        name="generate_pdf",
        description="将文本内容生成PDF文件，支持标题、正文、题目选项和答案等排版。返回 PDF 文件名与路径。",
        params={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "PDF标题"},
                "body": {"type": "string", "description": "正文内容"},
            },
            "required": ["title", "body"],
        },
    )
    async def tool_generate_pdf(self, event, title: str, body: str) -> str:
        """生成 PDF 文件（基于 fpdf，正文自动换行分页）。"""
        if not self._cfg("enable_pdf_tool", True):
            return "❌ generate_pdf 工具已关闭（可在插件配置页开启）。"
        try:
            from fpdf import FPDF
        except ImportError:
            return "❌ PDF 生成失败：缺少 fpdf 库，请先 pip install fpdf"
        try:
            self._pdf_dir.mkdir(parents=True, exist_ok=True)
            pdf = FPDF()
            pdf.add_page()
            pdf.set_auto_page_break(auto=True, margin=20)
            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT", align="C")
            pdf.ln(5)
            pdf.set_font("Helvetica", "", 11)
            pdf.multi_cell(0, 5.5, body or "")
            filename = f"pdf_{abs(hash(title)) % 10000}.pdf"
            filepath = self._pdf_dir / filename
            pdf.output(str(filepath))
            return f"✅ PDF 已生成：{filename}（{filepath}）"
        except Exception as e:
            logger.error("PDF 生成失败: %s", e)
            return f"❌ PDF 生成失败：{e}"

    # ---------------------------------------------------------------
    # 斜杠命令支持（/arxiv ...）
    # ---------------------------------------------------------------

    @staticmethod
    def _extract_text(event) -> str:
        def _iter_text_parts(chain):
            for ele in chain or []:
                if isinstance(ele, At):
                    continue
                if isinstance(ele, Text):
                    yield ele.text or ""

        msg = getattr(event, "message", None)
        chain = getattr(msg, "chain", None) if msg is not None else None
        if chain is not None:
            text = " ".join(_iter_text_parts(chain)).strip()
        else:
            parts = []
            for m in getattr(event, "messages", []) or []:
                parts.extend(_iter_text_parts(getattr(m, "chain", None)))
            text = " ".join(parts).strip()
        return re.sub(r"^(?:@\S+\s*)+", "", text).strip()

    @staticmethod
    def _get_sid(event) -> str:
        session = getattr(event, "session", None)
        if session is not None:
            sid = getattr(session, "sid", None)
            if sid:
                return sid
        msg = getattr(event, "message", None)
        if msg is not None:
            sender = getattr(msg, "sender", None)
            adapter = getattr(event, "adapter", None)
            adapter_name = getattr(adapter, "name", "unknown") if adapter else "unknown"
            if sender is not None:
                group = getattr(msg, "group", None)
                if group is not None and getattr(group, "group_id", None):
                    return f"{adapter_name}:gm:{group.group_id}"
                if getattr(sender, "user_id", None):
                    return f"{adapter_name}:dm:{sender.user_id}"
        return ""

    @staticmethod
    def _get_user_id(event) -> str:
        msg = getattr(event, "message", None)
        if msg is not None:
            sender = getattr(msg, "sender", None)
            if sender is not None and getattr(sender, "user_id", None):
                return str(sender.user_id)
        for m in getattr(event, "messages", []) or []:
            sender = getattr(m, "sender", None)
            if sender is not None and getattr(sender, "user_id", None):
                return str(sender.user_id)
        return ""

    @staticmethod
    def _is_group_message(event) -> bool:
        is_group = getattr(event, "is_group_message", None)
        if callable(is_group):
            try:
                return bool(is_group())
            except Exception:
                pass
        msg = getattr(event, "message", None)
        if msg is not None and getattr(msg, "group", None) is not None:
            return True
        messages = getattr(event, "messages", None) or []
        if messages and getattr(messages[-1], "group", None) is not None:
            return True
        return False

    @staticmethod
    def _bot_self_id(event) -> str:
        msg = getattr(event, "message", None)
        if msg is not None:
            sid = getattr(msg, "self_id", None)
            if sid:
                return str(sid)
        for m in getattr(event, "messages", []) or []:
            sid = getattr(m, "self_id", None)
            if sid:
                return str(sid)
        return ""

    @staticmethod
    def _is_bot_at(event, bot_id: str) -> bool:
        if not bot_id:
            return False
        bot_id = str(bot_id)
        msg = getattr(event, "message", None)
        chain = getattr(msg, "chain", None) if msg is not None else None
        if chain is not None:
            return any(
                getattr(ele, "pid", None) == bot_id
                for ele in chain
                if isinstance(ele, At)
            )
        for m in getattr(event, "messages", []) or []:
            for ele in getattr(m, "chain", None) or []:
                if isinstance(ele, At) and getattr(ele, "pid", None) == bot_id:
                    return True
        return False

    def _command_prefix(self) -> str:
        return (self._cfg("command_prefix", "/arxiv") or "/arxiv").strip()

    async def _check_slash_allowed(self, event) -> Tuple[bool, str]:
        whitelist = self._cfg("slash_whitelist") or []
        if not whitelist:
            return True, ""
        allowed = {str(x).strip() for x in whitelist if str(x).strip()}
        uid = self._get_user_id(event)
        if not uid:
            return False, "❌ 无法识别发送者 QQ 号，斜杠命令已拒绝执行。"
        if uid in allowed:
            return True, ""
        return False, f"❌ 您不在白名单内，无权使用 {self._command_prefix()} 斜杠命令（你的 QQ：{uid}）。"

    async def _reply(self, event, content: str, at_uid: str = ""):
        sid = self._get_sid(event)
        if not sid:
            logger.warning("无法确定会话 ID，arXiv 斜杠命令结果未发送")
            return
        chain = [Text(content)]
        if at_uid and self._is_group_message(event):
            chain = [At(at_uid), Text(content)]
        try:
            await self.ctx.message_processor.send_message_chain(
                session=sid, chain=MessageChain(chain)
            )
        except Exception as e:
            logger.error(f"发送 arXiv 斜杠命令回复失败（尝试退化纯文本）: {e}")
            if len(chain) > 1:
                try:
                    await self.ctx.message_processor.send_message_chain(
                        session=sid, chain=MessageChain([Text(content)])
                    )
                except Exception as e2:
                    logger.error(f"发送 arXiv 斜杠命令回复（纯文本）失败: {e2}")

    def _help_text(self) -> str:
        prefix = self._command_prefix()
        return (
            f"📚 arXiv 学术助手使用说明\n\n"
            f"🔍 {prefix} search <关键词> — 搜索论文（默认 5 条），"
            f"支持高级语法如 au:作者 ti:标题 cat:分类\n"
            f"📄 {prefix} get <arXiv ID> — 获取单篇论文详情\n"
            f"🀄 {prefix} tr <arXiv ID> — 将单篇论文的标题与摘要翻译成中文\n"
            f"⬇️  {prefix} dl <arXiv ID> [多个ID] — 下载 PDF 到 data/files/arxiv_pdf/\n"
            f"📦 {prefix} src <arXiv ID> — 下载 LaTeX 源码包到 data/files/arxiv_src/\n"
            f"ℹ️  {prefix} help — 查看帮助\n\n"
            f"示例：\n"
            f"  {prefix} search large language model\n"
            f"  {prefix} get 1706.03762\n"
            f"  {prefix} tr 1706.03762\n"
            f"  {prefix} dl 1706.03762\n"
            f"  {prefix} src 1706.03762"
        )

    async def _parse_and_execute(self, text: str, event) -> str:
        parts = (text or "").split()
        if not parts:
            return ""
        sub = parts[1].lower() if len(parts) > 1 else ""
        args = parts[2:]

        if sub in ("", "help", "-h", "--help"):
            return self._help_text()

        if sub == "search":
            if not args:
                return "❌ 用法：/arxiv search <关键词> [-t]，例如 /arxiv search large language model"
            translate = any(a in ("-t", "--translate") for a in args)
            q = " ".join(a for a in args if a not in ("-t", "--translate"))
            if not q:
                return "❌ 用法：/arxiv search <关键词> [-t]，例如 /arxiv search large language model"
            return await self.tool_arxiv_search(event, q, translate=translate)

        if sub == "get":
            if not args:
                return "❌ 用法：/arxiv get <arXiv ID> [-t]，例如 /arxiv get 1706.03762"
            translate = any(a in ("-t", "--translate") for a in args)
            ids = [a for a in args if a not in ("-t", "--translate")]
            if not ids:
                return "❌ 用法：/arxiv get <arXiv ID> [-t]，例如 /arxiv get 1706.03762"
            return await self.tool_arxiv_get(event, ids[0], translate=translate)

        if sub == "tr":
            if not args:
                return "❌ 用法：/arxiv tr <arXiv ID>，例如 /arxiv tr 1706.03762"
            return await self.tool_arxiv_translate(event, args[0])

        if sub == "src":
            if not args:
                return "❌ 用法：/arxiv src <arXiv ID>，例如 /arxiv src 1706.03762"
            return await self.tool_arxiv_src(event, args[0])

        if sub == "dl":
            if not args:
                return "❌ 用法：/arxiv dl <arXiv ID> [更多ID...]，例如 /arxiv dl 1706.03762"
            return await self.tool_arxiv_download(event, " ".join(args))

        return f"❌ 未知子命令：{sub}\n\n{self._help_text()}"

    @on.im_message(priority=Priority.HIGH)
    async def handle_arxiv_commands(self, event: KiraMessageEvent):
        """拦截斜杠命令开头的消息（前缀可配置，默认 /arxiv），直接执行，不再进入 LLM 流程。"""
        matched = False
        try:
            if not self._cfg("enable_commands", True):
                return
            text = self._extract_text(event)
            if not text:
                return
            if self._is_group_message(event):
                bot_id = self._bot_self_id(event)
                if not bot_id or not self._is_bot_at(event, bot_id):
                    return
            stripped = text.strip()
            prefix = self._command_prefix()
            if not re.match(rf"^{re.escape(prefix)}(\s|$)", stripped, re.IGNORECASE):
                return
            matched = True
            logger.info("拦截到 arXiv 斜杠命令: %s", stripped[:100])

            allowed, denied = await self._check_slash_allowed(event)
            if not allowed:
                await self._reply(event, denied, at_uid=self._get_user_id(event))
                return

            try:
                result = await self._parse_and_execute(stripped, event)
            except Exception as e:
                logger.exception("arXiv 斜杠命令执行异常: %s", e)
                result = f"❌ 斜杠命令执行出错：{type(e).__name__}: {e}"

            if result and result.strip():
                await self._reply(event, result, at_uid=self._get_user_id(event))
        except Exception as e:
            logger.exception("arXiv 斜杠命令钩子异常: %s", e)
            try:
                await self._reply(
                    event, f"❌ arXiv 斜杠命令处理失败：{type(e).__name__}: {e}"
                )
            except Exception:
                logger.exception("arXiv 斜杠命令钩子异常后回复失败")
        finally:
            if matched:
                try:
                    event.discard(force=True)
                    event.stop()
                except Exception as e:
                    logger.warning("丢弃 arXiv 斜杠命令消息失败: %s", e)


# ── 插件入口 ──────────────────────────────────────────────
plugin_class = ArxivPlugin
