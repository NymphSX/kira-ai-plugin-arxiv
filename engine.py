#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF → 中文 PDF 翻译管线引擎（纯 Python，不依赖 KiraAI core，可独立命令行运行）。

管线: PDF 提取(PyMuPDF / Mineru 后端) → 文本清洗 → 分块(≤1200字符) → DeepSeek 翻译(断点续传)
      → Markdown 重组 → xelatex 编译中文 PDF。

命令行用法:
    python3 data/plugins/pdf_translator/engine.py <pdf_path> [--lang zh] [--limit N] [--out DIR] [--model M]

DeepSeek 配置读取优先级: 显式参数 > 环境变量(KIRAAI_BASE_URL/KIRAAI_API_KEY/KIRAAI_ROOT)
    > <KiraAI根>/data/config/system_config.json 的 providers.deepseek-main.provider_config。
根目录通过从 __file__ 向上逐级查找含 data/config/system_config.json 的目录自动定位。
v0.2: 新增源码优先翻译分支（--arxiv-id/--tex 走 run_tex），xelatex 三遍稳健编译并清理辅助文件，LaTeX 输出前经 sanitize_unicode 清洗。

依赖: pymupdf(fitz)、系统 xelatex(TeXLive，含 ctex 宏包)。
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from abc import ABC, abstractmethod

# ----------------------------- 常量 -----------------------------
DEFAULT_MODEL = "deepseek-v4-flash"
CHUNK_LIMIT = 1200        # 清洗后单个 chunk 上限字符数
DEFAULT_BLOCK = 1800      # 翻译时合并后每块目标字符数
RETRIES = 3               # 单块翻译失败重试次数

SYSTEM = (
    "你是一位专业的中文学术论文翻译，把英文 LaTeX 论文正文翻译成通顺准确的中文学术语言。"
    "规则：1) 保留全部 LaTeX 命令与数学结构不翻译（\\cite \\ref \\label $...$ \\frac 等），编号/数字/单位原样保留；"
    "2) 纯中文表述不加英文括号；中英混排时术语首次用「中文（English）」，之后只用中文；"
    "3) 学术缩写（WER、PCC、SSL、CNN、LLM 等）保持原样，不翻译不展开；"
    "4) \\caption{} 内容必须全部翻译；5) 长句拆分为短句，避免长定语堆叠；"
    "6) 被动语态转为主动语态；7) 全文术语一致：翻译前先建立术语表，同一术语全文统一译法；"
    "8) 保持段落/行结构，不增删内容；只输出译文，不解释。"
)

# ----------------------------- 配置定位 -----------------------------
def find_kiraai_root():
    """从环境变量或 __file__ 向上逐级查找 KiraAI 根目录（含 data/config/system_config.json）。"""
    env = os.environ.get("KIRAAI_ROOT") or os.environ.get("KIRAAI_HOME")
    if env and os.path.exists(os.path.join(env, "data/config/system_config.json")):
        return env
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(8):
        if os.path.exists(os.path.join(d, "data/config/system_config.json")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    if os.path.exists(os.path.join(os.getcwd(), "data/config/system_config.json")):
        return os.getcwd()
    return None


def load_cfg(root=None, base_url=None, api_key=None):
    """读取 DeepSeek base_url / api_key。优先级: 显式传入 > 环境变量 > system_config.json。"""
    base_url = (base_url or "").strip() or os.environ.get("KIRAAI_BASE_URL", "").strip()
    api_key = (api_key or "").strip() or os.environ.get("KIRAAI_API_KEY", "").strip()
    if not base_url or not api_key:
        root = root or find_kiraai_root()
        if not root:
            raise RuntimeError(
                "找不到 KiraAI 根目录（data/config/system_config.json）。"
                "请设置环境变量 KIRAAI_ROOT 或 KIRAAI_BASE_URL/KIRAAI_API_KEY。"
            )
        d = json.load(open(os.path.join(root, "data/config/system_config.json"), encoding="utf-8"))
        cfg = d["providers"]["deepseek-main"]["provider_config"]
        base_url = base_url or cfg["base_url"]
        api_key = api_key or cfg["api_key"]
    if not base_url or not api_key:
        raise RuntimeError("DeepSeek base_url/api_key 为空，请检查配置")
    return base_url.rstrip("/"), api_key


def chat(base, key, model, text):
    """调用 DeepSeek chat/completions，返回译文（temperature=0.2, max_tokens=3000）。"""
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=json.dumps({
            "model": model,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": text}],
            "temperature": 0.2,
            "max_tokens": 3000,
        }).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        data = json.loads(r.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"].strip()


# ----------------------------- 文本提取（预留 mineru 后端） -----------------------------
class Extractor(ABC):
    """PDF 文本提取器抽象接口。新增后端只需实现 extract(pdf_path) -> str。"""
    name = "base"

    @abstractmethod
    def extract(self, pdf_path):
        raise NotImplementedError


class PdfExtractor(Extractor):
    """PyMuPDF 提取：按页提取文本，页间插入 '===== PAGE N =====' 便于后续去页眉页脚。"""
    name = "pymupdf"

    def extract(self, pdf_path):
        import fitz
        doc = fitz.open(pdf_path)
        pages = []
        try:
            for i, page in enumerate(doc):
                pages.append(f"===== PAGE {i + 1} =====\n{page.get_text('text')}")
        finally:
            doc.close()
        return "\n".join(pages)


class MineruExtractor(Extractor):
    """Mineru 后端占位：接口同 Extractor；enable_mineru=True 时被调用，尚未实现即报错提示。"""
    name = "mineru"

    def extract(self, pdf_path):
        raise NotImplementedError(
            "Mineru 提取后端尚未实现，请将 enable_mineru 设为 false，使用默认 PyMuPDF 提取器。"
        )


def get_extractor(enable_mineru=False):
    return MineruExtractor() if enable_mineru else PdfExtractor()


# ----------------------------- 文本清洗 / 分块（移植 chunk.py） -----------------------------
def clean_text(raw, header_footer_lines=(), header_footer_substrings=()):
    """行级清洗：去分页标记、可配置页眉页脚、孤立数字行；规范化连字/弯引号/软连字符断行。
    header_footer_lines / header_footer_substrings 为空即关闭页眉页脚过滤（通用 PDF 默认关闭）。"""
    hf_lines = set(header_footer_lines or ())
    hf_subs = tuple(header_footer_substrings or ())
    clean = []
    for ln in raw.split("\n"):
        t = ln.strip()
        if t.startswith("===== PAGE") and t.endswith("====="):
            continue
        if hf_lines and t in hf_lines:
            continue
        if hf_subs and any(s in t for s in hf_subs):
            continue
        if re.fullmatch(r"[\d\s]+", t):          # 纯页码行
            continue
        if re.fullmatch(r"\d+[−>]\d+", t):       # 图注残留
            continue
        clean.append(t if t else "")              # 保留空行作为段落分界
    text = "\n".join(clean)
    # 连字与特殊字符规范化
    text = text.replace("\ufb01", "fi").replace("\ufb02", "fl").replace("\ufb00", "ff")
    text = text.replace("\ufb03", "ffi").replace("\ufb04", "ffl")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"¨([aou])", lambda m: {"a": "ä", "o": "ö", "u": "ü"}[m.group(1)], text)
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    # 合并 PDF 分栏造成的软连字符断行
    text = re.sub(r"-\n(?=[a-z])", "", text)
    # 多余空行归一化
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


HEAD_RE = re.compile(r"^([0-9]+(?:\.[0-9]+)*)\s+[A-Z]")
CAP_RE = re.compile(r"^(Figure|Table)\s+([0-9]+):")


def classify_and_group(text):
    """行分类（heading/caption/body/blank）→ 按空行与标题合并为段落 paras。"""
    BODY_LINES = []
    for ln in text.split("\n"):
        t = ln.strip()
        if not t:
            BODY_LINES.append(("blank", t))
        elif HEAD_RE.match(t) or t in ("Abstract", "Keywords", "Acknowledgments", "References"):
            BODY_LINES.append(("heading", t))
        elif CAP_RE.match(t):
            BODY_LINES.append(("caption", t))
        else:
            BODY_LINES.append(("body", t))

    paras = []
    i = 0
    while i < len(BODY_LINES):
        kind, t = BODY_LINES[i]
        if kind == "blank":
            i += 1
            continue
        if kind == "heading":
            m = HEAD_RE.match(t)
            depth = len(m.group(1).split(".")) if m else 0
            paras.append({"type": "heading", "level": depth, "text": t})
            i += 1
            continue
        if kind == "caption":
            paras.append({"type": "caption", "level": 0, "text": t})
            i += 1
            continue
        buf = []
        while i < len(BODY_LINES) and BODY_LINES[i][0] in ("body",):
            buf.append(BODY_LINES[i][1])
            i += 1
        para = re.sub(r"\s+", " ", " ".join(buf)).strip()
        if para:
            paras.append({"type": "para", "level": 0, "text": para})
    return paras


def split_para(text, limit=CHUNK_LIMIT):
    """把长段落按句边界拆分为 ≤limit 字符的片段（单句过长时按逗号/空格硬切）。"""
    if len(text) <= limit:
        return [text]
    parts = []
    cur = ""
    for sent in re.split(r"(?<=[.!?;:])\s+", text):
        if len(cur) + len(sent) + 1 <= limit:
            cur = (cur + " " + sent).strip()
        else:
            if cur:
                parts.append(cur)
            if len(sent) > limit:
                while len(sent) > limit:
                    cut = sent.rfind(", ", 0, limit)
                    if cut == -1:
                        cut = sent.rfind(" ", 0, limit)
                    if cut == -1:
                        cut = limit
                    parts.append(sent[:cut].strip())
                    sent = sent[cut:].strip()
                cur = sent
            else:
                cur = sent
    if cur:
        parts.append(cur)
    return parts


def build_chunks(raw, header_footer_lines=(), header_footer_substrings=()):
    """清洗 + 分类 + 分块，返回 chunk 列表 [{"id","type","level","text"}]。"""
    text = clean_text(raw, header_footer_lines, header_footer_substrings)
    paras = classify_and_group(text)
    chunks = []
    idx = 0
    for p in paras:
        if p["type"] == "para":
            for piece in split_para(p["text"], CHUNK_LIMIT):
                idx += 1
                chunks.append({"id": idx, "type": "para", "level": 0, "text": piece})
        else:
            idx += 1
            chunks.append({"id": idx, "type": p["type"], "level": p.get("level", 0), "text": p["text"]})
    return chunks


def merge_chunks(chunks, block=DEFAULT_BLOCK):
    """按 heading 分节，para 合并为 ≤block 字符的块。返回 [(type, text)]。"""
    merged, buf, buf_len = [], "", 0
    for c in chunks:
        t, txt = c["type"], (c.get("text") or "").strip()
        if not txt:
            continue
        if t == "heading":
            if buf:
                merged.append(("para", buf))
                buf, buf_len = "", 0
            merged.append(("heading", txt))
        else:
            if buf and buf_len + len(txt) > block:
                merged.append(("para", buf))
                buf, buf_len = "", 0
            buf += ("\n\n" if buf else "") + txt
            buf_len += len(txt)
    if buf:
        merged.append(("para", buf))
    return merged


# ----------------------------- 引擎 -----------------------------
_MATH_UNI = {
    "≥": "$\\ge$", "≤": "$\\le$", "≠": "$\\neq$", "×": "$\\times$", "÷": "$\\div$",
    "±": "$\\pm$", "∞": "$\\infty$", "°": "$\\textdegree{}", "→": "$\\rightarrow$",
    "←": "$\\leftarrow$", "·": "$\\cdot$", "…": "\\ldots{}", "–": "--", "—": "---",
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
}

def sanitize_unicode(text):
    """编译前清洗进入 LaTeX 的文本：数学符号→LaTeX、引号规范；emoji/私用区/不可打印字符删除或替换并告警。"""
    if not text:
        return text
    out, warn = [], 0
    for ch in text:
        if ch in _MATH_UNI:
            out.append(_MATH_UNI[ch]); continue
        o = ord(ch)
        if o < 32 and ch not in "\n\t":
            warn += 1; continue
        if (0x1F600 <= o <= 0x1F64F or 0x1F300 <= o <= 0x1F5FF or 0x1F680 <= o <= 0x1F6FF
                or 0xFE00 <= o <= 0xFE0F or o == 0xFEFF or 0xE000 <= o <= 0xF8FF
                or 0x10000 <= o <= 0x10FFFF):
            warn += 1
            out.append("{\\bfseries 注意：}")
            continue
        out.append(ch)
    if warn:
        print(f"[sanitize_unicode] 替换/删除 {warn} 个特殊 Unicode 字符", flush=True)
    return "".join(out)


class PdfTranslatorEngine:
    """PDF → 中文 PDF 翻译引擎。可被 main.py 的 LLM 工具调用，也可被 CLI 直接驱动。"""

    def __init__(self, root=None, model=DEFAULT_MODEL, base_url=None, api_key=None,
                 chunk_size=DEFAULT_BLOCK, output_dir=None, enable_mineru=False,
                 header_footer_lines=None, header_footer_substrings=None):
        self.root = root or find_kiraai_root()
        if not self.root:
            raise RuntimeError("无法定位 KiraAI 根目录（缺少 data/config/system_config.json）")
        self.model = model or DEFAULT_MODEL
        self.base_url = base_url
        self.api_key = api_key
        self.chunk_size = int(chunk_size or DEFAULT_BLOCK)
        self.output_dir = output_dir or os.path.join(self.root, "data/files/pdf_translator")
        self.enable_mineru = enable_mineru
        self.header_footer_lines = list(header_footer_lines or [])
        self.header_footer_substrings = list(header_footer_substrings or [])

    # ---- 预提取分块（main.py 后台阈值判断用） ----
    def prepare(self, pdf_path):
        """预提取+清洗分块：把 raw.txt / chunks.json 写入 work_dir，返回 chunks 列表。
        供 main.py 在 PDF 翻译路线先做块数阈值判断（块数 > background_threshold 转后台任务）。"""
        pdf_path = os.path.abspath(pdf_path)
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF 不存在: {pdf_path}")
        stem = os.path.splitext(os.path.basename(pdf_path))[0] or "document"
        work_dir = os.path.join(self.root, "data/temp/pdf_translator_work", stem)
        os.makedirs(work_dir, exist_ok=True)
        extractor = get_extractor(self.enable_mineru)
        print(f"[prepare] 提取文本: {extractor.name} <- {pdf_path}", flush=True)
        raw = extractor.extract(pdf_path)
        with open(os.path.join(work_dir, "raw.txt"), "w", encoding="utf-8") as f:
            f.write(raw)
        print("[prepare] 清洗分块", flush=True)
        chunks = build_chunks(raw, self.header_footer_lines, self.header_footer_substrings)
        with open(os.path.join(work_dir, "chunks.json"), "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=1)
        return chunks

    # ---- 源码优先翻译（v0.2：arXiv 源码 / 本地 tex → 中文 PDF） ----
    _TEX_INPUT_RE = re.compile(r"\\(?:input|include)\s*\{([^}]+)\}")
    _TEX_CAP_RE = re.compile(r"(\\caption(?:\[[^\]]*\])?\{)([^}]*)(\})")

    @staticmethod
    def _sanitize_tex_id(tex_id):
        """把 arXiv id / 路径片段转成安全目录名。"""
        s = re.sub(r"[^A-Za-z0-9._-]", "_", str(tex_id))
        return s or "tex"

    @staticmethod
    def _extract_source(archive, dest_dir):
        """解压 arXiv e-print 源码包到 dest_dir（防路径穿越：拒绝绝对路径与含 .. 的成员）。"""
        import tarfile
        dest_dir = os.path.abspath(dest_dir)
        os.makedirs(dest_dir, exist_ok=True)
        a = str(archive)
        if a.endswith((".tar.gz", ".tgz")):
            with tarfile.open(a, "r:gz") as tf:
                safe = [m for m in tf.getmembers()
                        if not m.name.startswith("/") and ".." not in m.name.split("/")]
                tf.extractall(dest_dir, members=safe)
        elif a.endswith(".tex.gz"):
            import gzip as _gz
            out = os.path.join(dest_dir, os.path.basename(a)[:-3])
            with _gz.open(a, "rt", encoding="utf-8", errors="ignore") as f:
                open(out, "w", encoding="utf-8").write(f.read())
        else:
            shutil.copy(a, os.path.join(dest_dir, os.path.basename(a)))

    @staticmethod
    def _find_root_tex(src_dir):
        """在目录中定位根 .tex（前 2000 字符含 \\documentclass 者优先）。"""
        from pathlib import Path
        tex_files = sorted(Path(src_dir).rglob("*.tex"))
        for f in tex_files:
            try:
                head = f.read_text(encoding="utf-8", errors="ignore")[:2000]
            except Exception:
                continue
            if "\\documentclass" in head:
                return f
        return tex_files[0] if tex_files else None

    @classmethod
    def _collect_tex_files(cls, main_tex, src_dir):
        """递归收集根 tex 及其 \\input/\\include 引用的 .tex（去重，含根）。.bib/.cls/.sty/图片不收集。"""
        files, seen = [], set()

        def walk(path):
            path = os.path.abspath(path)
            if path in seen:
                return
            seen.add(path)
            files.append(path)
            try:
                text = open(path, encoding="utf-8", errors="ignore").read()
            except Exception:
                return
            base_dir = os.path.dirname(path)
            for m in cls._TEX_INPUT_RE.finditer(text):
                ref = m.group(1).strip()
                if ref.endswith(".tex"):
                    ref = ref[:-4]
                for cand in (os.path.join(base_dir, ref + ".tex"),
                             os.path.join(src_dir, ref + ".tex")):
                    if os.path.exists(cand):
                        walk(cand)
                        break

        walk(main_tex)
        return files

    @staticmethod
    def _inject_preamble(main_tex):
        """在根 tex 的 \\documentclass 行后注入 ctex + hyperref(unicode)，重复调用去重。"""
        p = str(main_tex)
        text = open(p, encoding="utf-8", errors="ignore").read()
        lines = text.split("\n")
        have = set(lines)
        inject = [
            "\\usepackage[UTF8,fontset=fandol]{ctex}",
            "\\usepackage[unicode=true,pdfencoding=auto,psdextra]{hyperref}",
        ]
        add = [x for x in inject if not any(x in h for h in have)]
        if add:
            for i, ln in enumerate(lines):
                if "\\documentclass" in ln:
                    lines[i + 1:i + 1] = add
                    break
        out = "\n".join(lines)
        open(p, "w", encoding="utf-8").write(out)
        return out

    @staticmethod
    def _protect_latex_line(line):
        """把一行中的公式/命令/花括号结构替换为占位符，返回 (纯文本, 占位符列表)。"""
        ph = []

        def _mk(m):
            ph.append(m.group(0))
            return f"__KIRA_PH_{len(ph) - 1}__"

        s = re.sub(r"\$\$.*?\$\$|\$.*?\$", _mk, line, flags=re.S)
        s = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^}]*\})*", _mk, s)
        s = re.sub(r"\{[^}]*\}", _mk, s)
        return s, ph

    @staticmethod
    def _restore_placeholders(text, ph):
        """将译文中的占位符还原为原始 LaTeX 片段。"""
        def _rp(m):
            idx = int(m.group(1))
            return ph[idx] if idx < len(ph) else m.group(0)
        return re.sub(r"__KIRA_PH_(\d+)__", _rp, text)

    def _chat_retry(self, base, key, model, text):
        """调 DeepSeek 翻译，失败重试 RETRIES 次；全部失败返回 None。"""
        last_err = None
        for attempt in range(RETRIES):
            try:
                return chat(base, key, model, text)
            except Exception as e:
                last_err = e
                print(f"[RETRY {attempt + 1}] {type(e).__name__}: {e}", flush=True)
                time.sleep(2 * (attempt + 1))
        print(f"[FAIL] {last_err}", flush=True)
        return None

    def _translate_tex_content(self, tex_text, base, key, model, limit=0):
        """翻译 .tex 正文：保护 LaTeX 结构，\\caption 内容强制翻译。返回 (译文或 None, 说明)。"""
        out_lines = []
        pending = []    # (out_idx, protected_text, ph)
        captions = []   # (out_idx, holder, prefix, content, suffix)

        for line in tex_text.split("\n"):
            s = line.strip()
            if (not s or s.startswith("%")
                    or s.startswith("\\begin{") or s.startswith("\\end{")):
                out_lines.append(line)
                continue
            # 挖出 \\caption{...}（含 \\caption[...]{...}）内容，保留命令前后缀
            parts, pos = [], 0
            for m in self._TEX_CAP_RE.finditer(line):
                parts.append(line[pos:m.start(1)])
                holder = f"__KIRA_CAP_{len(captions)}__"
                captions.append((len(out_lines), holder, m.group(1), m.group(2), m.group(3)))
                parts.append(holder)
                pos = m.end(3)
            parts.append(line[pos:])
            masked = "".join(parts)
            protected, ph = self._protect_latex_line(masked)
            body = re.sub(r"__KIRA_(?:PH|CAP)_\d+__", "", protected)
            if body.strip() and not protected.strip().startswith("\\"):
                pending.append((len(out_lines), protected, ph))
                out_lines.append(None)
            else:
                out_lines.append(masked)

        # ---- 翻译正文行（分块 ≤4500 字符，limit>0 只翻前 limit 块） ----
        if limit:
            skipped = pending[limit:]   # 被截断不译的块：还原原文占位，避免 out_lines 残留 None（join 崩溃）
            pending = pending[:limit]
            for idx, text, ph in skipped:
                out_lines[idx] = self._restore_placeholders(text, ph)
        rows = {}
        if pending:
            blocks, cur, cur_len = [], [], 0
            for (idx, text, _ph) in pending:
                ln = len(text) + 1
                if cur and cur_len + ln > 4500:
                    blocks.append(cur)
                    cur, cur_len = [], 0
                cur.append((idx, text))
                cur_len += ln
            if cur:
                blocks.append(cur)
            for blk in blocks:
                block_text = "\n".join(t for _, t in blk)
                tr = self._chat_retry(base, key, model, block_text)
                if tr is None:
                    return None, f"正文翻译失败：{block_text[:120]}"
                bt = tr.split("\n")
                if len(bt) != len(blk):
                    bt = []
                    for _, t in blk:
                        one = self._chat_retry(base, key, model, t)
                        if one is None:
                            return None, f"逐行翻译失败：{t[:120]}"
                        bt.append(one)
                for (idx, _t), tline in zip(blk, bt):
                    rows[idx] = tline
            for idx, text, ph in pending:
                out_lines[idx] = self._restore_placeholders(rows[idx], ph)

        # ---- 翻译 caption 内容（合并一块，行数不符回退逐条） ----
        if captions:
            blocks, cur, cur_len = [], [], 0
            for (idx, holder, prefix, content, suffix) in captions:
                if cur and cur_len + len(content) + 1 > 4500:
                    blocks.append(cur)
                    cur, cur_len = [], 0
                cur.append((idx, holder, content))
                cur_len += len(content) + 1
            if cur:
                blocks.append(cur)
            for blk in blocks:
                block_text = "\n".join(c for _, _, c in blk)
                tr = self._chat_retry(base, key, model, block_text)
                if tr is None:
                    return None, f"caption 翻译失败：{block_text[:120]}"
                ct = tr.split("\n")
                if len(ct) != len(blk):
                    ct = []
                    for _, _, c in blk:
                        one = self._chat_retry(base, key, model, c)
                        if one is None:
                            return None, f"caption 逐条翻译失败：{c[:120]}"
                        ct.append(one)
                for (idx, holder, _c), tline in zip(blk, ct):
                    if out_lines[idx] is None:
                        out_lines[idx] = ""
                    out_lines[idx] = out_lines[idx].replace(holder, tline)

        safe = ["\n" if ln is None else ln for ln in out_lines]  # join 前兜底过滤 None
        return "\n".join(safe), f"翻译 {len(pending)} 正文行 / {len(captions)} 条 caption"

    def _compile_tex_multi(self, work_dir, main_tex, out_pdf):
        """xelatex + bibtex 多遍编译；编译前清缓存；以 'Output written on N pages' 判定成功。
        返回 (页数, PDF路径)。"""
        if not shutil.which("xelatex"):
            raise RuntimeError("未找到 xelatex（需要 TeX Live），请先安装")
        work_dir = os.path.abspath(work_dir)
        for ext in (".aux", ".log", ".bbl", ".blg", ".toc", ".out",
                    ".lof", ".lot", ".xdv", ".bcf", ".run.xml", ".fls", ".fdb_latexmk"):
            for root, _dirs, fnames in os.walk(work_dir):
                for fn in fnames:
                    if fn.endswith(ext):
                        try:
                            os.remove(os.path.join(root, fn))
                        except OSError:
                            pass
        stem = os.path.splitext(os.path.basename(main_tex))[0]
        tex_name = stem + ".tex"
        old = os.getcwd()
        os.chdir(work_dir)
        log = ""
        pages = None
        try:
            def _run_xelatex():
                return subprocess.run(
                    ["xelatex", "-interaction=nonstopmode", "-halt-on-error", tex_name],
                    capture_output=True, text=True, timeout=900)

            r = _run_xelatex()                      # 第 1 遍
            log = (r.stdout or "") + (r.stderr or "")
            aux = os.path.join(work_dir, stem + ".aux")
            if os.path.exists(aux):
                aux_text = open(aux, encoding="utf-8", errors="ignore").read()
                if "\\bibdata" in aux_text and shutil.which("bibtex"):
                    subprocess.run(["bibtex", stem], capture_output=True, text=True, timeout=300)
            for _i in range(2):                     # 第 2-3 遍（含引用/目录稳定）
                r = _run_xelatex()
                log = (r.stdout or "") + (r.stderr or "")
                if "Output written on" in log:
                    m = re.search(r"Output written on [^(\n]*\((\d+) page", log)
                    if m:
                        pages = int(m.group(1))
                    break
            if "Output written on" not in log:
                raise RuntimeError("xelatex 编译失败（未输出 'Output written on'）：\n" + log[-1500:])
        finally:
            os.chdir(old)
        src_pdf = os.path.join(work_dir, stem + ".pdf")
        if not os.path.exists(src_pdf):
            raise RuntimeError("xelatex 未生成 PDF 文件")
        shutil.copy(src_pdf, out_pdf)
        return pages, out_pdf

    def run_tex(self, arxiv_id=None, tex_path=None, limit=0, lang="zh",
                on_stage=None, on_progress=None):
        """源码优先翻译：arXiv ID 或本地 tex 源码 → 定位根 tex → 保留原文档类注入
        ctex + hyperref[unicode] → 只翻译 \\input/\\include 的正文 tex（.bib/图片原样）
        → xelatex + bibtex 多遍编译 → 输出 {stem}_zh.pdf。"""
        if lang and lang != "zh":
            raise NotImplementedError(f"当前仅支持目标语言 zh，收到: {lang}")

        base, key = load_cfg(self.root, self.base_url, self.api_key)

        # ---- 1. 获取/定位源码 → 工作副本 src/（绝不覆盖用户源文件） ----
        if on_stage:
            on_stage("download")
        os.makedirs(self.output_dir, exist_ok=True)
        if arxiv_id:
            arxiv_id = str(arxiv_id).strip()
            work_dir = os.path.join(self.root, "data/temp/pdf_translator_work",
                                    "tex_" + self._sanitize_tex_id(arxiv_id))
            os.makedirs(work_dir, exist_ok=True)
            archive = os.path.join(work_dir, "e-print.tar.gz")
            if not os.path.exists(archive):
                url = f"https://export.arxiv.org/e-print/{arxiv_id}"
                print(f"[run_tex] 下载源码: {url}", flush=True)
                try:
                    urllib.request.urlretrieve(url, archive)
                except Exception as e:
                    raise RuntimeError(f"下载 arXiv e-print 失败: {type(e).__name__}: {e}")
            src_dir = os.path.join(work_dir, "src")
            os.makedirs(src_dir, exist_ok=True)
            self._extract_source(archive, src_dir)
            main_tex = self._find_root_tex(src_dir)
            if main_tex is None:
                raise RuntimeError(f"源码包中未找到 .tex 文件: {archive}")
        elif tex_path:
            tex_path = os.path.abspath(str(tex_path))
            if not os.path.exists(tex_path):
                raise FileNotFoundError(f"TeX 源码不存在: {tex_path}")
            work_dir = os.path.join(self.root, "data/temp/pdf_translator_work",
                                    "tex_" + self._sanitize_tex_id(os.path.basename(tex_path)) or "tex_local")
            src_dir = os.path.join(work_dir, "src")
            if os.path.isdir(tex_path):
                shutil.copytree(tex_path, src_dir, dirs_exist_ok=True)
                main_tex = self._find_root_tex(src_dir)
            elif tex_path.endswith((".tar.gz", ".tgz")):
                os.makedirs(src_dir, exist_ok=True)
                self._extract_source(tex_path, src_dir)
                main_tex = self._find_root_tex(src_dir)
            elif tex_path.endswith(".tex"):
                os.makedirs(src_dir, exist_ok=True)
                shutil.copy(tex_path, os.path.join(src_dir, os.path.basename(tex_path)))
                main_tex = os.path.join(src_dir, os.path.basename(tex_path))
            else:
                raise ValueError(f"不支持的 tex 源码类型: {tex_path}")
            if main_tex is None:
                raise RuntimeError(f"未找到根 tex（含 \\documentclass）: {tex_path}")
        else:
            raise ValueError("必须提供 arxiv_id 或 tex_path 之一")

        if on_stage:
            on_stage("extract")

        # ---- 2. 保留原文档类，注入 ctex + hyperref(unicode) ----
        self._inject_preamble(main_tex)

        # ---- 3. 收集正文 tex（递归 \\input/\\include，.bib 不收集） ----
        tex_files = self._collect_tex_files(main_tex, str(src_dir))
        print("[run_tex] 待翻译 tex: "
              + ", ".join(os.path.relpath(f, src_dir) for f in tex_files), flush=True)

        # ---- 4. 逐文件翻译（写回工作副本；.bib/图片原样保留） ----
        if on_stage:
            on_stage("translate")
        total = len(tex_files)
        for fi, tf in enumerate(tex_files):
            text = open(tf, encoding="utf-8", errors="ignore").read()
            translated, note = self._translate_tex_content(text, base, key, self.model, limit=limit)
            if translated is None:
                raise RuntimeError(f"翻译失败: {tf}: {note}")
            cleaned = "\n".join(sanitize_unicode(ln) for ln in translated.split("\n"))
            with open(tf, "w", encoding="utf-8") as f:
                f.write(cleaned)
            print(f"[run_tex] 已翻译 {fi + 1}/{total}: {os.path.relpath(tf, src_dir)} ({note})",
                  flush=True)
            if on_progress:
                on_progress(fi + 1, total)

        # ---- 5. xelatex + bibtex 多遍编译 ----
        if on_stage:
            on_stage("compile")
        stem = os.path.splitext(os.path.basename(str(main_tex)))[0]
        out_pdf = os.path.join(self.output_dir, f"{stem}_zh.pdf")
        pages, out_pdf = self._compile_tex_multi(str(src_dir), str(main_tex), out_pdf)

        pages_txt = f", {pages} 页" if pages else ""
        return (
            f"源码优先翻译完成（run_tex）\n"
            f"- 翻译 tex: {total} 个文件\n"
            f"- PDF: {out_pdf} ({os.path.getsize(out_pdf)} bytes{pages_txt})\n"
            f"- 说明: 保留原文档类，注入 ctex + hyperref[unicode=true]，"
            f"xelatex + bibtex 多遍编译"
        )

    # ---- 主流程 ----
    def run(self, pdf_path, limit=0, lang="zh", on_stage=None, on_progress=None, chunks=None):
        if lang and lang != "zh":
            raise NotImplementedError(f"当前仅支持目标语言 zh，收到: {lang}")
        pdf_path = os.path.abspath(pdf_path)
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF 不存在: {pdf_path}")

        stem = os.path.splitext(os.path.basename(pdf_path))[0] or "document"
        work_dir = os.path.join(self.root, "data/temp/pdf_translator_work", stem)
        os.makedirs(work_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

        if chunks:
            print(f"[1/5] 复用 prepare 分块结果（{len(chunks)} 块）", flush=True)
            with open(os.path.join(work_dir, "chunks.json"), "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=1)
        else:
            if on_stage:
                on_stage("extract")
            extractor = get_extractor(self.enable_mineru)
            print(f"[1/5] 提取文本: {extractor.name} <- {pdf_path}", flush=True)
            raw = extractor.extract(pdf_path)
            with open(os.path.join(work_dir, "raw.txt"), "w", encoding="utf-8") as f:
                f.write(raw)

            if on_stage:
                on_stage("chunk")
            print("[2/5] 清洗分块", flush=True)
            chunks = build_chunks(raw, self.header_footer_lines, self.header_footer_substrings)
            with open(os.path.join(work_dir, "chunks.json"), "w", encoding="utf-8") as f:
                json.dump(chunks, f, ensure_ascii=False, indent=1)

        # 3. 合并翻译（断点续传）
        if on_stage:
            on_stage("translate")
        merged = merge_chunks(chunks, block=self.chunk_size)
        if not merged:
            raise RuntimeError("未从 PDF 提取到可翻译文本（可能是扫描版/图片型 PDF，需 Mineru OCR 后端）")
        md_path = os.path.join(self.output_dir, f"{stem}_zh.md")
        prog_path = os.path.join(work_dir, "progress.json")
        print(f"[3/5] 翻译 {len(merged)} 块 (limit={limit or '全部'})", flush=True)
        result = self._translate_merged(merged, md_path, prog_path, limit=limit, on_progress=on_progress)

        if not os.path.exists(md_path) or os.path.getsize(md_path) < 10:
            raise RuntimeError(f"Markdown 为空，无法编译 PDF: {md_path}")
        print(f"[4/5] 重组 Markdown -> {md_path}", flush=True)
        pdf_out = os.path.join(self.output_dir, f"{stem}_zh.pdf")
        if on_stage:
            on_stage("compile")
        print("[5/5] xelatex 编译 PDF", flush=True)
        self._build_pdf(md_path, stem, work_dir, pdf_out)

        summary = (
            f"PDF 翻译完成\n"
            f"- Markdown: {md_path} ({os.path.getsize(md_path)} bytes)\n"
            f"- PDF: {pdf_out} ({os.path.getsize(pdf_out)} bytes)\n"
            f"- 统计: {result['ok']} 成功 / {result['fail']} 失败 / 共 {result['total']} 块"
        )
        if result["failed_ids"]:
            summary += f"\n- 失败块索引: {result['failed_ids'][:10]}"
        return summary

    # ---- 翻译（断点续传） ----
    def _translate_merged(self, merged, md_path, prog_path, limit=0, on_progress=None):
        base, key = load_cfg(self.root, self.base_url, self.api_key)
        prog = {}
        if os.path.exists(prog_path):
            try:
                prog = json.load(open(prog_path, encoding="utf-8"))
            except Exception:
                prog = {}
        if not prog and os.path.exists(md_path):
            open(md_path, "w", encoding="utf-8").write("")

        total = min(len(merged), limit) if limit else len(merged)
        ok = fail = 0
        failed_ids = []

        def commit(key_id, out):
            with open(md_path, "a", encoding="utf-8") as f:
                f.write(out)
            prog[key_id] = out
            with open(prog_path, "w", encoding="utf-8") as f:
                json.dump(prog, f, ensure_ascii=False, indent=1)

        for i in range(total):
            if on_progress:
                on_progress(i + 1, total)
            typ, txt = merged[i]
            key_id = hashlib.md5(f"{typ}\n{txt}".encode("utf-8")).hexdigest()
            if key_id in prog:
                ok += 1
                continue
            if typ == "heading":
                commit(key_id, f"\n\n## {txt}\n\n")
                ok += 1
                continue
            last_err = None
            for attempt in range(RETRIES):
                try:
                    tr = chat(base, key, self.model, txt)
                    commit(key_id, tr + "\n\n")
                    ok += 1
                    print(f"[OK] {i + 1}/{total} len={len(tr)}", flush=True)
                    break
                except Exception as e:
                    last_err = e
                    print(f"[RETRY {attempt + 1}] block {i + 1}: {type(e).__name__}: {e}", flush=True)
                    time.sleep(2 * (attempt + 1))
            else:
                fail += 1
                failed_ids.append(i)
                print(f"[FAIL] block {i + 1}: {last_err}", flush=True)
            time.sleep(0.5)
        print(f"\n完成: OK={ok} FAIL={fail} 总块数={total}")
        return {"ok": ok, "fail": fail, "total": total, "failed_ids": failed_ids}

    # ---- MD → LaTeX → PDF（移植 build_pdf.py） ----
    @staticmethod
    def _esc(t):
        t = str(t).replace("\\", "\\textbackslash{}")
        for a, b in [("&", "\\&"), ("%", "\\%"), ("$", "\\$"), ("#", "\\#"),
                     ("_", "\\_"), ("{", "\\{"), ("}", "\\}"),
                     ("~", "\\textasciitilde{}"), ("^", "\\textasciicircum{}")]:
            t = t.replace(a, b)
        return t

    def _latex_document(self, stem, secs):
        parts = [r"""\documentclass[11pt]{ctexart}
\usepackage[margin=2.4cm]{geometry}
\usepackage{amsmath,amssymb}
\usepackage[unicode=true,pdfencoding=auto,psdextra]{hyperref}
\usepackage{enumitem}
\setlist{nosep}
\title{\textit{%s} 中文翻译}
\author{KiraAI PDF 翻译插件}
\date{\today}
\begin{document}
\maketitle
\tableofcontents
\newpage
""" % self._esc(stem)]
        for title, paras in secs:
            if not title:
                for p in paras:
                    parts.append("\\par %s" % sanitize_unicode(self._esc(p)))
                continue
            parts.append("\\section*{%s}" % sanitize_unicode(self._esc(title)))
            parts.append("\\addcontentsline{toc}{section}{%s}" % sanitize_unicode(self._esc(title)))
            for p in paras:
                parts.append("\\par %s" % sanitize_unicode(self._esc(p)))
        parts.append("\\end{document}\n")
        return "\n".join(parts)

    def _build_pdf(self, md_path, stem, work_dir, pdf_out):
        lines = open(md_path, encoding="utf-8").read().splitlines()
        secs, cur = [], None
        for ln in lines:
            s = ln.strip()
            if not s:
                continue
            if s.startswith("## "):
                cur = [s[3:].strip(), []]
                secs.append(cur)
            else:
                if cur is None:
                    cur = ["", []]
                    secs.append(cur)
                cur[1].append(s)

        tex = os.path.join(work_dir, f"{stem}_zh.tex")
        with open(tex, "w", encoding="utf-8") as f:
            f.write(self._latex_document(stem, secs))

        for ext in (".aux", ".log", ".bbl", ".blg", ".toc", ".out", ".lof", ".lot"):
            p = os.path.join(work_dir, f"{stem}_zh{ext}")
            if os.path.exists(p):
                os.remove(p)

        old_cwd = os.getcwd()
        os.chdir(work_dir)
        try:
            log = ""
            ok = False
            for i in range(3):   # 最多三遍编译生成目录
                r = subprocess.run(
                    ["xelatex", "-interaction=nonstopmode", os.path.basename(tex)],
                    capture_output=True, text=True, timeout=600)
                log = (r.stdout or "") + (r.stderr or "")
                if "Output written on" in log:
                    ok = True
                    break
            if not ok:
                raise RuntimeError(f"xelatex 编译失败(3遍均未生成 PDF):\n" + log[-1500:])
        finally:
            os.chdir(old_cwd)

        src_pdf = os.path.join(work_dir, f"{stem}_zh.pdf")
        if not os.path.exists(src_pdf):
            raise RuntimeError("xelatex 未生成 PDF")
        shutil.copy(src_pdf, pdf_out)
        print(f"PDF OK: {pdf_out} {os.path.getsize(pdf_out)} bytes")


# ----------------------------- CLI 入口 -----------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="pdf_translator.engine",
        description="PDF → 中文 PDF 翻译管线（KiraAI pdf_translator 插件引擎，任意 PDF 可直接命令行复用）")
    ap.add_argument("pdf_path", help="待翻译 PDF 路径（传 --arxiv-id/--tex 时可为空）")
    ap.add_argument("--arxiv-id", default=None, help="arXiv 编号：提供则走源码优先翻译（run_tex）")
    ap.add_argument("--tex", default=None, help="本地 TeX 源码/源码包路径：提供则走 run_tex")
    ap.add_argument("--lang", default="zh", help="目标语言（当前仅支持 zh）")
    ap.add_argument("--limit", type=int, default=0, help="只翻译前 N 块（测试用）")
    ap.add_argument("--out", default=None, help="输出目录（默认 <KiraAI>/data/files/pdf_translator）")
    ap.add_argument("--model", default=DEFAULT_MODEL, help=f"DeepSeek 模型（默认 {DEFAULT_MODEL}）")
    ap.add_argument("--enable-mineru", action="store_true", help="启用 Mineru 提取后端（未实现，会报错提示）")
    args = ap.parse_args(argv)
    try:
        engine = PdfTranslatorEngine(root=None, model=args.model, output_dir=args.out,
                                     enable_mineru=args.enable_mineru)
        if args.arxiv_id or args.tex:
            summary = engine.run_tex(arxiv_id=args.arxiv_id, tex_path=args.tex,
                                     limit=args.limit, lang=args.lang)
        else:
            summary = engine.run(args.pdf_path, limit=args.limit, lang=args.lang)
        print(summary)
        return 0
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
