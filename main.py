"""
KiraAI arXiv 学术助手插件 (kira-ai-plugin-arxiv) v2.0.0

由两个插件合并而来：
- kira-ai-plugin-arxiv-search（arXiv 查询/下载/摘要翻译）
- pdf_translator（论文翻译成中文 PDF：源码优先翻译 + PDF 直接翻译 + 后台任务）

功能：
- /arxiv search/get/tr/dl/src 斜杠命令（前缀可配置，默认 /arxiv）
- LLM 工具：arxiv_search / arxiv_get / arxiv_translate / arxiv_download / arxiv_src
           / parse_arxiv_command / pdf_translate / query_pdf_translate_task
- 摘要翻译走默认快速模型（ctx.get_default_fast_llm_client，不依赖翻译插件）
- PDF 翻译两条路线：源码优先（arxiv_id/tex_path → xelatex 编译）与 PDF 直接翻译
  （提取→分块→翻译→重组 Markdown→xelatex 编译），长 PDF 自动转后台任务

实现要点：
- arXiv API 礼貌间隔（>=3s）节流 + TTL 缓存 + 原子落盘下载 + User-Agent
- 输出目录统一：data/files/arxiv_pdf、data/files/arxiv_src、data/files/pdf_translator
"""

import asyncio
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

from core.plugin import BasePlugin, logger, register, on, Priority
from core.chat.message_utils import MessageChain, KiraMessageEvent
from core.chat.message_elements import Text, At
from core.provider import LLMRequest

from .arxiv_core import ArxivClient, ArxivApiError
from .engine import PdfTranslatorEngine

log = logging.getLogger(__name__)

# PDF 翻译默认模型（model_select 未配置时兜底，与 engine.DEFAULT_MODEL 一致）
DEFAULT_TRANSLATE_MODEL = "deepseek-v4-flash"
# 默认翻译模型 = 快速模型（R4，model_select 下拉默认值，provider_id:model_id）
DEFAULT_FAST_MODEL = "3937f0fdf6b7:deepseek-v4-flash-0731"

# ── 后台翻译任务注册表（模块级，跨实例共享）──
# task_id → 任务状态 dict；asyncio.Lock 保证同一事件循环内对字典的读写串行化
_TASKS: dict = {}
_TASKS_LOCK = asyncio.Lock()
# 持有后台 asyncio.Task 引用，防止任务被垃圾回收而意外取消
_BG_TASKS: set = set()


class ArxivPlugin(BasePlugin):
    """arXiv 论文查询、翻译与下载，内置论文翻译中文 PDF 引擎"""

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
            user_agent=self._cfg("user_agent", ""),
        )

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
        for _dir in (self.download_dir, self.source_dir):
            try:
                _dir.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                logger.error("创建目录失败（%s）: %s", _dir, e)

    async def on_unload(self):
        logger.info("arXiv 插件已卸载")

    async def initialize(self):
        await self.on_load()

    async def terminate(self):
        # 取消仍挂起的后台翻译任务，在途任务标记为 failed
        for _t in list(_BG_TASKS):
            if not _t.done():
                _t.cancel()
        _BG_TASKS.clear()
        async with _TASKS_LOCK:
            for _t in _TASKS.values():
                if _t.get("status") in ("pending", "running"):
                    _t["status"] = "failed"
                    _t["error"] = "任务被取消（插件卸载或系统关闭）"
                    _t["updated_at"] = time.time()
        await self.on_unload()

    # ---------------------------------------------------------------
    # 摘要翻译（默认快速模型，不依赖翻译插件）
    # ---------------------------------------------------------------

    def _get_translation_client(self):
        """解析翻译模型选择：translation_model（provider:model）→ ctx.get_llm_client；
        未配置或解析失败回退默认快速模型。"""
        val = (self._cfg("translation_model", "") or "").strip()
        if val:
            try:
                picked = self.ctx.get_llm_client(model_uuid=val)
                if picked is not None:
                    return picked
            except Exception as e:
                logger.warning("translation_model 解析失败（%s），回退快速模型: %s", val, e)
        return self.ctx.get_default_fast_llm_client() or self.ctx.get_default_llm_client()

    async def _translate_lines(
        self, lines: List[str], target: str = "zh", client=None, fallback: bool = True
    ) -> Optional[List[str]]:
        """批量翻译多行文本（每行一条）。默认走快速 LLM；失败/禁用时回退原文。"""
        if not lines:
            return lines
        if not self._cfg("translate_enabled", True):
            return lines if fallback else None
        if client is None:
            client = self._get_translation_client()
        if not client:
            return lines if fallback else None
        try:
            numbered = "\n".join(f"{i+1}. {line}" for i, line in enumerate(lines))
            prompt = (
                f"请将以下 {len(lines)} 条文本逐条翻译成{target}，"
                f"严格保持编号格式，每条一行，只输出翻译结果，不要任何解释。\n\n{numbered}"
            )
            request = LLMRequest(messages=[{"role": "user", "content": prompt}], tools=[])
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

    async def _run_tex_task(self, task_id, engine, arxiv_id, tex_path, target_lang, limit, sid):
        """后台执行源码优先翻译（run_tex）：下载→提取→逐文件翻译→编译，完成后推送结果。"""
        try:
            await self._update_task(task_id, status="running")
            _loop = asyncio.get_running_loop()

            def _on_stage(stage):
                try:
                    asyncio.run_coroutine_threadsafe(
                        self._update_task(task_id, stage=stage), _loop)
                except Exception as e:
                    logger.warning("更新后台任务 %s 阶段 %s 失败: %s", task_id, stage, e)

            def _on_progress(done, total):
                try:
                    asyncio.run_coroutine_threadsafe(
                        self._update_task(
                            task_id, done_blocks=int(done), total_blocks=int(total),
                            stage="translate",
                        ),
                        _loop,
                    )
                except Exception as e:
                    logger.warning("更新后台任务 %s 进度失败: %s", task_id, e)
                step = 1 if total <= 10 else 5
                if done == total or done % step == 0:
                    try:
                        asyncio.run_coroutine_threadsafe(
                            self._send_to_session(sid, f"📖 翻译进度 {done}/{total} 个 tex 文件"),
                            _loop,
                        )
                    except Exception as e:
                        logger.warning("推送后台任务 %s 进度失败: %s", task_id, e)

            summary = await asyncio.to_thread(
                engine.run_tex,
                arxiv_id=arxiv_id or None,
                tex_path=tex_path or None,
                limit=limit,
                lang=target_lang,
                on_stage=_on_stage,
                on_progress=_on_progress,
            )
            result_pdf = ""
            for line in str(summary).splitlines():
                line = line.strip()
                if line.startswith("- PDF: "):
                    cand = line.split("- PDF: ", 1)[1].split(" (")[0].strip()
                    if cand and os.path.exists(cand):
                        result_pdf = cand
                    break
            await self._update_task(
                task_id, status="done", stage="done", summary=summary,
                result_md="", result_pdf=result_pdf,
            )
            lines = ["✅ 源码优先翻译完成", f"🔖 任务ID：{task_id}"]
            if result_pdf:
                lines.append(f"📕 PDF：{result_pdf}")
            if summary:
                lines.append(f"📊 统计：{summary}")
            await self._send_to_session(sid, "\n".join(lines))
        except asyncio.CancelledError:
            logger.info("后台源码翻译任务 %s 被取消", task_id)
            await self._update_task(task_id, status="failed", error="任务被取消（插件卸载或系统关闭）")
            raise
        except Exception as e:
            logger.exception("后台源码翻译任务 %s 失败: %s", task_id, e)
            err_msg = f"{type(e).__name__}: {e}"
            await self._update_task(task_id, status="failed", error=err_msg)
            try:
                await self._send_to_session(
                    sid,
                    f"❌ 源码优先翻译任务 {task_id} 失败：{err_msg}\n"
                    f"（可用 query_pdf_translate_task(task_id=\"{task_id}\") 查询详情）",
                )
            except Exception:
                logger.exception("发送翻译失败通知失败")

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
        results = await asyncio.gather(
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
        client = self.ctx.get_default_llm_client()
        if not client:
            return "❌ 翻译服务不可用，先试试 /arxiv get"
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
    # PDF 翻译（合并自 pdf_translator：源码优先 + PDF 直接翻译 + 后台任务）
    # ---------------------------------------------------------------

    def _engine(self) -> PdfTranslatorEngine:
        """根据插件配置（schema section_pdf_translate）构造引擎实例。
        翻译模型支持 model_select（translation_model 存 provider_id:model_id），
        空则回退旧字段 model（兼容），再回退默认快速模型。"""
        s = self.plugin_cfg.get("section_pdf_translate", {}) or {}
        translation_model = (s.get("translation_model") or "").strip()
        model = (s.get("model") or "").strip() or DEFAULT_TRANSLATE_MODEL
        provider = "deepseek-main"
        if translation_model:
            if ":" in translation_model:
                provider, model = translation_model.split(":", 1)
            else:
                model = translation_model
        return PdfTranslatorEngine(
            root=None,  # engine 自动向上定位 KiraAI 根目录
            model=model or DEFAULT_TRANSLATE_MODEL,
            provider=provider or "deepseek-main",
            base_url=(s.get("base_url") or "").strip() or None,
            api_key=(s.get("api_key") or "").strip() or None,
            chunk_size=int(s.get("chunk_size") or 1800),
            output_dir=(s.get("output_dir") or "").strip() or None,
            enable_mineru=bool(s.get("enable_mineru", False)),
        )

    @staticmethod
    def _new_task_id() -> str:
        """生成后台翻译任务 ID：PDFTR + 时间戳 + uuid 短码（如 PDFTR1700000000A1B2C3）。"""
        return f"PDFTR{int(time.time())}{uuid.uuid4().hex[:6].upper()}"

    async def _update_task(self, task_id: str, **fields):
        async with _TASKS_LOCK:
            task = _TASKS.get(task_id)
            if task is None:
                return
            task.update(fields)
            task["updated_at"] = time.time()

    def _schedule_background(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        _BG_TASKS.add(task)
        task.add_done_callback(_BG_TASKS.discard)
        return task

    async def _send_to_session(self, sid: str, content: str):
        if not sid:
            return
        try:
            await self.ctx.message_processor.send_message_chain(
                session=sid, chain=MessageChain([Text(content)])
            )
        except Exception as e:
            logger.warning("向会话 %s 发送消息失败: %s", sid, e)

    def _format_task(self, task_id: str) -> str:
        task = _TASKS.get(task_id)
        if task is None:
            return f"❌ 未找到翻译任务：{task_id}"
        status = task.get("status", "unknown")
        icon = {
            "pending": "⏳ 排队中",
            "running": "🔄 进行中",
            "done": "✅ 已完成",
            "failed": "❌ 已失败",
        }.get(status, f"❓ {status}")
        stage_name = {
            "queued": "排队",
            "download": "下载源码",
            "extract": "提取文本",
            "chunk": "清洗分块",
            "translate": "分块翻译",
            "compile": "编译 PDF",
            "done": "完成",
        }.get(task.get("stage", ""), task.get("stage", "") or "-")
        lines = [
            f"{icon} PDF 翻译任务 {task_id}",
            f"📄 PDF：{task.get('pdf_path', '-')}",
            f"🛠 当前阶段：{stage_name}",
        ]
        total = task.get("total_blocks", 0)
        done = task.get("done_blocks", 0)
        if total:
            lines.append(f"📖 翻译进度：{done}/{total} 块")
        elif status in ("running", "pending"):
            lines.append("📖 翻译进度：尚未开始分块")
        if task.get("result_md"):
            lines.append(f"📝 Markdown：{task['result_md']}")
        if task.get("result_pdf"):
            lines.append(f"📕 PDF：{task['result_pdf']}")
        if task.get("summary"):
            lines.append(f"📊 统计：{task['summary']}")
        if status == "failed" and task.get("error"):
            lines.append(f"🚨 错误：{task['error']}")
        elapsed = time.time() - task.get("created_at", time.time())
        lines.append(f"⏱ 耗时：{elapsed:.0f}s")
        return "\n".join(lines)

    async def _run_translate_task(self, task_id, engine, pdf_path, target_lang, limit, sid):
        """后台执行 PDF 翻译：全程更新任务状态并推送进度，完成后推送结果路径。"""
        try:
            await self._update_task(task_id, status="running")
            _loop = asyncio.get_running_loop()

            def _on_stage(stage):
                try:
                    asyncio.run_coroutine_threadsafe(
                        self._update_task(task_id, stage=stage), _loop)
                except Exception as e:
                    logger.warning("更新后台任务 %s 阶段 %s 失败: %s", task_id, stage, e)

            def _on_progress(done, total):
                try:
                    asyncio.run_coroutine_threadsafe(
                        self._update_task(
                            task_id, done_blocks=int(done), total_blocks=int(total),
                            stage="translate",
                        ),
                        _loop,
                    )
                except Exception as e:
                    logger.warning("更新后台任务 %s 进度失败: %s", task_id, e)
                step = 1 if total <= 10 else 5
                if done == total or done % step == 0:
                    try:
                        asyncio.run_coroutine_threadsafe(
                            self._send_to_session(sid, f"📖 翻译进度 {done}/{total} 块"),
                            _loop,
                        )
                    except Exception as e:
                        logger.warning("推送后台任务 %s 进度失败: %s", task_id, e)

            summary = await asyncio.to_thread(
                engine.run, pdf_path, limit, target_lang,
                on_stage=_on_stage, on_progress=_on_progress, chunks=None,
            )

            stem = os.path.splitext(os.path.basename(pdf_path))[0] or "document"
            output_dir = getattr(engine, "output_dir", "") or ""
            result_md = os.path.join(output_dir, f"{stem}_zh.md")
            result_pdf = os.path.join(output_dir, f"{stem}_zh.pdf")
            result_md = result_md if os.path.exists(result_md) else ""
            result_pdf = result_pdf if os.path.exists(result_pdf) else ""

            await self._update_task(
                task_id, status="done", stage="done", summary=summary,
                result_md=result_md, result_pdf=result_pdf,
            )
            lines = ["✅ PDF 翻译完成", f"🔖 任务ID：{task_id}"]
            if result_md:
                lines.append(f"📝 Markdown：{result_md}")
            if result_pdf:
                lines.append(f"📕 PDF：{result_pdf}")
            if summary:
                lines.append(f"📊 统计：{summary}")
            await self._send_to_session(sid, "\n".join(lines))
        except asyncio.CancelledError:
            logger.info("后台翻译任务 %s 被取消", task_id)
            await self._update_task(task_id, status="failed", error="任务被取消（插件卸载或系统关闭）")
            raise
        except Exception as e:
            logger.exception("后台翻译任务 %s 失败: %s", task_id, e)
            err_msg = f"{type(e).__name__}: {e}"
            await self._update_task(task_id, status="failed", error=err_msg)
            try:
                await self._send_to_session(
                    sid,
                    f"❌ PDF 翻译任务 {task_id} 失败：{err_msg}\n"
                    f"（可用 query_pdf_translate_task(task_id=\"{task_id}\") 查询详情）",
                )
            except Exception:
                logger.exception("发送翻译失败通知失败")

    @register.tool(
        name="pdf_translate",
        description=(
            "把学术论文翻译成中文 PDF，支持两条路线："
            "① 源码优先翻译（推荐）：传 arxiv_id（arXiv 编号，如 2401.00001）或 tex_path"
            "（本地 TeX 源码/源码包 .tex/.tex.gz/.tar.gz/.tgz），引擎下载/读取源码→xelatex 编译中文 PDF；"
            "② PDF 翻译：传 pdf_path 走 PDF→分块→翻译→xelatex 编译。"
            "长 PDF（块数超过后台阈值 background_threshold，默认 20）自动转后台任务：立即返回任务 ID，"
            "执行期间推送进度，完成后自动发送结果，可用 query_pdf_translate_task 查询。"
            "两条路线输出均到 data/files/pdf_translator/。"
        ),
        params={
            "type": "object",
            "properties": {
                "pdf_path": {"type": "string", "description": "待翻译 PDF 的本地路径（PDF 翻译路线；走源码优先翻译时可不传）。超过后台阈值时自动转为后台任务并返回任务 ID"},
                "arxiv_id": {"type": "string", "description": "arXiv 编号（如 2401.00001），提供则走源码优先翻译"},
                "tex_path": {"type": "string", "description": "本地 TeX 源码/源码包（.tex/.tex.gz/.tar.gz/.tgz）路径，提供则走源码优先翻译"},
                "target_lang": {"type": "string", "description": "目标语言，默认 zh（可选）"},
                "limit": {"type": "integer", "description": "只翻译前 N 块（测试用，可选）"}
            },
            "required": []
        }
    )
    async def pdf_translate(self, event, pdf_path: str = "", arxiv_id: str = "",
                            tex_path: str = "", target_lang: str = "zh", limit: int = 0) -> str:
        if not self._cfg("enable_tool", True):
            return "❌ pdf_translate 工具已关闭（可在插件配置页开启）。"
        pdf_path = (pdf_path or "").strip()
        arxiv_id = (arxiv_id or "").strip()
        tex_path = (tex_path or "").strip()
        if not arxiv_id and not tex_path and not pdf_path:
            return ("pdf_translate 参数错误：请至少提供 pdf_path（PDF 翻译路线）"
                    "或 arxiv_id / tex_path（源码优先翻译路线）")
        engine = self._engine()
        try:
            if arxiv_id or tex_path:
                sid = self._get_sid(event)
                task_id = self._new_task_id()
                record = {
                    "task_id": task_id,
                    "pdf_path": tex_path or f"arxiv:{arxiv_id}",
                    "sid": sid,
                    "target_lang": target_lang,
                    "limit": int(limit or 0),
                    "status": "pending",
                    "stage": "queued",
                    "total_blocks": 0,
                    "done_blocks": 0,
                    "result_md": "",
                    "result_pdf": "",
                    "summary": "",
                    "error": "",
                    "created_at": time.time(),
                    "updated_at": time.time(),
                }
                async with _TASKS_LOCK:
                    _TASKS[task_id] = record
                self._schedule_background(
                    self._run_tex_task(
                        task_id, engine, arxiv_id or None, tex_path or None,
                        target_lang, int(limit or 0), sid,
                    )
                )
                logger.info("源码优先翻译任务已提交后台: %s (tex=%s, sid=%s)",
                            task_id, tex_path or arxiv_id, sid)
                return (
                    f"📖 源码优先翻译任务已提交，任务ID：{task_id}\n"
                    f"📨 执行期间推送进度，完成后自动发送结果\n"
                    f"🔎 查询进度：query_pdf_translate_task(task_id=\"{task_id}\")"
                )
            else:
                chunks = await asyncio.to_thread(engine.prepare, pdf_path)
                try:
                    threshold = int(
                        (self.plugin_cfg.get("section_pdf_translate", {}) or {})
                        .get("background_threshold", 20) or 20
                    )
                except (TypeError, ValueError):
                    threshold = 20
                if len(chunks) > threshold:
                    sid = self._get_sid(event)
                    if not sid:
                        summary = await asyncio.to_thread(
                            engine.run, pdf_path, int(limit or 0), target_lang, chunks=chunks)
                        return summary
                    async with _TASKS_LOCK:
                        for _tid, _t in _TASKS.items():
                            if (_t.get("sid") == sid
                                    and _t.get("pdf_path") == pdf_path
                                    and _t.get("status") in ("pending", "running")):
                                return (
                                    f"⚠️ 该会话已有同一 PDF 的翻译任务进行中（任务ID：{_tid}）。\n"
                                    f"可用 query_pdf_translate_task(task_id=\"{_tid}\") 查询进度。"
                                )
                    task_id = self._new_task_id()
                    record = {
                        "task_id": task_id,
                        "pdf_path": pdf_path,
                        "sid": sid,
                        "target_lang": target_lang,
                        "limit": int(limit or 0),
                        "status": "pending",
                        "stage": "queued",
                        "total_blocks": len(chunks),
                        "done_blocks": 0,
                        "result_md": "",
                        "result_pdf": "",
                        "summary": "",
                        "error": "",
                        "created_at": time.time(),
                        "updated_at": time.time(),
                    }
                    async with _TASKS_LOCK:
                        _TASKS[task_id] = record
                    self._schedule_background(
                        self._run_translate_task(
                            task_id, engine, pdf_path, target_lang, int(limit or 0), sid))
                    n = len(chunks)
                    logger.info("PDF 翻译任务已提交后台: %s (pdf=%s, blocks=%d, sid=%s)",
                                task_id, pdf_path, n, sid)
                    return (
                        f"📖 翻译任务已提交，任务ID：{task_id}\n"
                        f"共 {n} 块，超过阈值 {threshold}，已转入后台执行（不阻塞）\n"
                        f"📨 执行期间推送进度，完成后自动发送结果\n"
                        f"🔎 查询进度：query_pdf_translate_task(task_id=\"{task_id}\")"
                    )
                summary = await asyncio.to_thread(
                    engine.run, pdf_path, int(limit or 0), target_lang, chunks=chunks)
            return summary
        except Exception as e:
            logger.error("pdf_translate 失败: %s", e)
            return f"PDF 翻译失败：{e}"

    @register.tool(
        name="query_pdf_translate_task",
        description=(
            "查询 pdf_translate 提交的长 PDF 后台翻译任务的状态。返回任务状态"
            "（pending 排队中 / running 进行中 / done 已完成 / failed 已失败）、当前阶段"
            "（提取文本/清洗分块/分块翻译/编译 PDF）、分块翻译进度（已翻译块数/总块数）、"
            "结果 Markdown/PDF 路径或错误信息。task_id 由 pdf_translate 提交长任务时返回的任务 ID。"
        ),
        params={
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "由 pdf_translate 提交长任务时返回的任务 ID，可查状态/阶段/进度/结果"
                }
            },
            "required": ["task_id"]
        }
    )
    async def query_pdf_translate_task(self, event, task_id: str) -> str:
        """查询 PDF 翻译后台任务状态。"""
        task_id = (task_id or "").strip()
        if not task_id:
            return "❌ 请提供任务 ID，例如：query_pdf_translate_task(task_id=\"PDFTR...\")"
        return self._format_task(task_id)

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
            if not self._cfg("enable_commands", False):
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
