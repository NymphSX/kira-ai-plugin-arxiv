"""kira-ai-plugin-arxiv 内置多后端翻译引擎

从 kira-ai-plugin-translate 合并而来：百度/DeepL/Google/阿里云/本地模型，
自动语言检测、后端自动回退、按会话额度控制、翻译缓存。
"""
import asyncio
import hashlib
import json
import logging
import random
import time
from collections import OrderedDict
from datetime import date
from pathlib import Path
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)

# ── 语言代码归一化（统一 ISO 639-1 小写，兼容各别名） ──────────
_LANG_ALIASES = {
    "zh": "zh", "zh-cn": "zh", "zh_cn": "zh", "cn": "zh", "chs": "zh",
    "zh-hans": "zh", "zh-tw": "zh", "zh-hant": "zh", "cht": "zh", "中文": "zh",
    "en": "en", "english": "en", "英语": "en",
    "ja": "ja", "jp": "ja", "japanese": "ja", "日语": "ja",
    "ko": "ko", "kor": "ko", "korean": "ko", "韩语": "ko",
    "fr": "fr", "fra": "fr", "法语": "fr",
    "de": "de", "ger": "de", "德语": "de",
    "es": "es", "spa": "es", "西班牙语": "es",
    "ru": "ru", "rus": "ru", "俄语": "ru",
    "pt": "pt", "葡萄牙语": "pt", "it": "it", "意大利语": "it",
    "nl": "nl", "荷兰语": "nl", "ar": "ar", "ara": "ar", "阿拉伯语": "ar",
    "hi": "hi", "印地语": "hi", "th": "th", "泰语": "th",
    "vi": "vi", "vie": "vi", "越南语": "vi", "id": "id", "印尼语": "id",
}
_SUPPORTED = {"auto", "zh", "en", "ja", "ko", "fr", "de", "es", "ru",
              "pt", "it", "nl", "ar", "hi", "th", "vi", "id"}

# canonical -> 各家后端语言代码
_VENDOR_LANG = {
    "baidu":  {"zh": "zh", "en": "en", "ja": "jp", "ko": "kor", "fr": "fra",
               "de": "de", "es": "spa", "ru": "ru", "pt": "pt", "it": "it",
               "nl": "nl", "ar": "ara", "hi": "hi", "th": "th", "vi": "vie"},
    "deepl":  {"zh": "ZH", "en": "EN", "ja": "JA", "ko": "KO", "fr": "FR",
               "de": "DE", "es": "ES", "ru": "RU", "pt": "PT", "it": "IT",
               "nl": "NL", "ar": "AR"},
    "google": {"zh": "zh-CN", "en": "en", "ja": "ja", "ko": "ko", "fr": "fr",
               "de": "de", "es": "es", "ru": "ru", "pt": "pt", "it": "it",
               "nl": "nl", "ar": "ar", "hi": "hi", "th": "th", "vi": "vi", "id": "id"},
    "aliyun": {"zh": "zh", "en": "en", "ja": "ja", "ko": "ko", "fr": "fr",
               "de": "de", "es": "es", "ru": "ru", "pt": "pt", "it": "it",
               "nl": "nl", "ar": "ar", "hi": "hi", "th": "th", "vi": "vi", "id": "id"},
}
# 自动回退顺序：百度→阿里云→DeepL→Google→本地模型
_FALLBACK_CHAIN = ["baidu", "aliyun", "deepl", "google", "local"]

# 阿里云 SDK 探测结果缓存（None=未探测）
_ALIYUN_SDK_OK = None


def _aliyun_sdk_ready() -> bool:
    """探测阿里云 SDK 是否可导入（只探测一次）"""
    global _ALIYUN_SDK_OK
    if _ALIYUN_SDK_OK is None:
        try:
            import aliyun_python_sdk_core  # noqa: F401
            import aliyun_python_sdk_alimt  # noqa: F401
            _ALIYUN_SDK_OK = True
        except ImportError:
            _ALIYUN_SDK_OK = False
    return _ALIYUN_SDK_OK


def _safe_int(value, default: int) -> int:
    """容错的 int 转换，失败返回默认值"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_bool(value, default: bool) -> bool:
    """容错的 bool 转换：正确处理字符串 \"false\"/\"0\"/\"off\" 等"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("1", "true", "yes", "on"):
            return True
        if v in ("0", "false", "no", "off", ""):
            return False
    try:
        return bool(int(value))
    except (TypeError, ValueError):
        return default


def _norm_lang(code: str) -> str:
    """把用户/LLM 传入的语言代码规范化为 ISO 639-1 小写；非法则抛 ValueError"""
    if not code:
        return "auto"
    key = code.strip().lower().replace("_", "-")
    key = _LANG_ALIASES.get(key, key)
    if key not in _SUPPORTED:
        raise ValueError(f"不支持的语言代码: {code}")
    return key


def _to_vendor(backend: str, lang: str, vendor_map: dict) -> str:
    if lang == "auto":
        return "auto"
    return vendor_map.get(lang, lang)


class TranslationEngine:
    """多后端翻译引擎（不依赖插件上下文，配置通过 dict 注入）"""

    def __init__(self, cfg: dict, data_dir: Optional[Path] = None):
        self.cfg = cfg
        self._http: Optional[aiohttp.ClientSession] = None
        self._cache: "OrderedDict[str, str]" = OrderedDict()
        self._cache_max = 1024
        self._quota: dict = {}          # session_id -> {"date": str, "chars": int}
        self._window: dict = {}         # session_id -> {"ts": float, "n": int}
        self._lock = asyncio.Lock()
        self._data_dir = data_dir

    # ── 生命周期 ──────────────────────────────────────────
    async def initialize(self):
        self._http = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30),
            headers={"User-Agent": "KiraAI-Arxiv/2.0"},
        )
        _aliyun_sdk_ready()
        available = [b for b in _FALLBACK_CHAIN if self._backend_ready(b)]
        if not available:
            logger.warning("翻译引擎：未配置任何后端，translate 不可用（可只配本地模型实现零成本离线翻译）")
        else:
            logger.info("翻译引擎已初始化，可用后端: %s", ", ".join(available))

    async def close(self):
        if self._http:
            await self._http.close()
            self._http = None

    # ── 配置 ──────────────────────────────────────────────
    def _get_config(self) -> dict:
        return {
            "default_backend": self.cfg.get("default_backend", "auto"),
            "baidu_appid": self.cfg.get("baidu_appid", ""),
            "baidu_secret_key": self.cfg.get("baidu_secret_key", ""),
            "deepl_api_key": self.cfg.get("deepl_api_key", ""),
            "deepl_pro": _safe_bool(self.cfg.get("deepl_pro", False), False),
            "google_api_key": self.cfg.get("google_api_key", ""),
            "aliyun_ak": self.cfg.get("aliyun_access_key_id", ""),
            "aliyun_sk": self.cfg.get("aliyun_access_key_secret", ""),
            "aliyun_region": self.cfg.get("aliyun_region", "cn-hangzhou"),
            "local_url": self.cfg.get("local_backend_url", ""),
            "local_model": self.cfg.get("local_model", ""),
            "local_timeout": _safe_int(self.cfg.get("local_timeout", 120), 120),
            "max_chars_per_call": _safe_int(self.cfg.get("max_chars_per_call", 5000), 5000),
            "max_chars_per_day": _safe_int(self.cfg.get("max_chars_per_day", 10000), 10000),
            "max_qpm": _safe_int(self.cfg.get("max_queries_per_min", 30), 30),
            "enable_cache": _safe_bool(self.cfg.get("enable_cache", True), True),
        }

    def _backend_ready(self, backend: str) -> bool:
        cfg = self._get_config()
        if backend == "baidu":
            return bool(cfg["baidu_appid"] and cfg["baidu_secret_key"])
        if backend == "deepl":
            return bool(cfg["deepl_api_key"])
        if backend == "google":
            return bool(cfg["google_api_key"])
        if backend == "aliyun":
            return bool(cfg["aliyun_ak"] and cfg["aliyun_sk"] and _aliyun_sdk_ready())
        if backend == "local":
            return bool(cfg["local_url"] and cfg["local_model"])
        return False

    # ── 翻译入口 ──────────────────────────────────────────
    async def translate(
        self,
        text: str,
        target_lang: str,
        source_lang: str = "auto",
        backend: str = "auto",
        sid: str = "unknown",
    ) -> str:
        """翻译文本（含额度/限流/缓存/后端回退），返回带结果前缀的字符串。"""
        if not text or not text.strip():
            return "❌ 翻译失败：文本为空。"
        cfg = self._get_config()
        try:
            tgt = _norm_lang(target_lang)
            src = _norm_lang(source_lang)
        except ValueError as e:
            return f"❌ {e}"
        if tgt == "auto":
            return "❌ 翻译失败：目标语言不能为 auto。"
        if len(text) > cfg["max_chars_per_call"]:
            return f"❌ 翻译失败：文本超过单次上限 {cfg['max_chars_per_call']} 字符（当前 {len(text)}）。"

        # 缓存命中（不消耗额度）
        cache_key = f"{backend}|{src}|{tgt}|{text}"
        if cfg["enable_cache"] and cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return f"✅ {self._cache[cache_key]}\n（缓存命中，未消耗额度）"

        # 额度与限流（预扣制，防并发超卖）
        ok, err = await self._reserve_quota(sid, len(text), cfg)
        if not ok:
            return f"❌ {err}"

        # 选择后端执行顺序：显式 backend 优先，其次默认后端，最后回退链
        if backend not in ("", "auto"):
            chain = [backend]
        else:
            default = cfg["default_backend"]
            chain = [default] if default not in ("", "auto") else _FALLBACK_CHAIN

        errors = []
        for b in chain:
            if not self._backend_ready(b):
                errors.append(f"{b}未配置")
                continue
            try:
                result = await self._translate_with(b, text, src, tgt, cfg)
            except Exception as e:
                logger.error("翻译后端 %s 失败: %s", b, e)
                errors.append(f"{b}: {e}")
                continue
            if result is None or not result.strip():
                errors.append(f"{b}: 无返回结果")
                continue
            if cfg["enable_cache"]:
                self._cache[cache_key] = result
                self._cache.move_to_end(cache_key)
                while len(self._cache) > self._cache_max:
                    self._cache.popitem(last=False)
            return f"✅ {result}\n（后端: {b}）"

        # 全部失败：回滚预扣额度
        await self._release_quota(sid, len(text))
        return "❌ 翻译失败：" + "；".join(errors)

    async def translate_lines(
        self,
        lines,
        target: str = "zh",
        backend: str = "fast",
        sid: str = "unknown",
        fallback: bool = True,
    ) -> Optional[list]:
        """批量翻译多行文本（每行一条）。backend=fast 走 LLM 快速模型，其余走翻译引擎。

        fallback=True 回退原文；fallback=False 返回 None（供需要区分「翻译失败」的调用方使用）。
        """
        if not lines:
            return lines
        if backend == "fast":
            raise NotImplementedError("fast backend handled by plugin layer")
        joined = "\n".join(lines)
        result = await self.translate(joined, target, "auto", backend, sid=sid)
        if result.startswith("✅"):
            body = result[1:].split("\n（后端:", 1)[0].strip()
            translated = [x for x in (body or "").splitlines() if x.strip()]
            if len(translated) == len(lines):
                return translated
        return lines if fallback else None

    # ── 后端适配器 ────────────────────────────────────────
    async def _translate_with(self, backend: str, text: str, src: str, tgt: str, cfg: dict) -> Optional[str]:
        if backend == "baidu":
            return await self._baidu(text, src, tgt, cfg)
        if backend == "deepl":
            return await self._deepl(text, src, tgt, cfg)
        if backend == "google":
            return await self._google(text, src, tgt, cfg)
        if backend == "aliyun":
            return await self._aliyun(text, src, tgt, cfg)
        if backend == "local":
            return await self._local(text, src, tgt, cfg)
        raise ValueError(f"未知后端: {backend}")

    async def _baidu(self, text, src, tgt, cfg) -> str:
        appid, secret = cfg["baidu_appid"], cfg["baidu_secret_key"]
        salt = str(random.randint(0, 2 ** 31))
        sign = hashlib.md5(f"{appid}{text}{salt}{secret}".encode()).hexdigest()
        data = {
            "q": text, "from": _to_vendor("baidu", src, _VENDOR_LANG["baidu"]),
            "to": _to_vendor("baidu", tgt, _VENDOR_LANG["baidu"]),
            "appid": appid, "salt": salt, "sign": sign,
        }
        async with self._http.post("https://fanyi-api.baidu.com/api/trans/vip/translate",
                                   data=data) as resp:
            body = await resp.json()
        if "error_code" in body:
            raise RuntimeError(f"百度翻译错误 {body['error_code']}: {body.get('error_msg')}")
        parts = [item.get("dst", "") for item in body.get("trans_result", [])]
        if not any(parts):
            raise RuntimeError("百度翻译返回空结果")
        detected = body.get("from", src)
        return f"[{detected}→{tgt}] " + "\n".join(parts)

    async def _deepl(self, text, src, tgt, cfg) -> str:
        host = "api.deepl.com" if cfg["deepl_pro"] else "api-free.deepl.com"
        vendor_tgt = _VENDOR_LANG["deepl"].get(tgt)
        if not vendor_tgt:
            raise RuntimeError(f"DeepL 不支持目标语言: {tgt}")
        data = {"text": text, "target_lang": vendor_tgt}
        if src != "auto":
            vendor_src = _VENDOR_LANG["deepl"].get(src)
            if not vendor_src:
                raise RuntimeError(f"DeepL 不支持源语言: {src}")
            data["source_lang"] = vendor_src
        headers = {"Authorization": f"DeepL-Auth-Key {cfg['deepl_api_key']}"}
        async with self._http.post(f"https://{host}/v2/translate",
                                   data=data, headers=headers) as resp:
            body = await resp.json()
        if "message" in body:
            raise RuntimeError(f"DeepL 错误: {body['message']}")
        try:
            translations = body["translations"]
        except (KeyError, TypeError):
            raise RuntimeError(f"DeepL 响应结构异常: {body}")
        detected = translations[0].get("detected_source_language", src)
        return f"[{detected}→{tgt}] " + translations[0]["text"]

    async def _google(self, text, src, tgt, cfg) -> str:
        payload = {
            "q": text, "target": _to_vendor("google", tgt, _VENDOR_LANG["google"]),
            "format": "text",
        }
        if src != "auto":
            payload["source"] = _to_vendor("google", src, _VENDOR_LANG["google"])
        params = {"key": cfg["google_api_key"]}
        async with self._http.post("https://translation.googleapis.com/language/translate/v2",
                                   params=params, json=payload) as resp:
            body = await resp.json()
        if "error" in body:
            err = body["error"]
            raise RuntimeError(f"Google 错误: {err.get('message') if isinstance(err, dict) else err}")
        try:
            data = body["data"]["translations"][0]
        except (KeyError, IndexError, TypeError):
            raise RuntimeError(f"Google 响应结构异常: {body}")
        detected = data.get("detectedSourceLanguage", src)
        return f"[{detected}→{tgt}] " + data["translatedText"]

    async def _aliyun(self, text, src, tgt, cfg) -> str:
        if not _aliyun_sdk_ready():
            raise RuntimeError("阿里云 SDK 未安装（缺少 aliyun-python-sdk-core / aliyun-python-sdk-alimt）")
        from aliyun_python_sdk_core.auth.credentials import AccessKeyCredential
        from aliyun_python_sdk_core.client import AcsClient
        import aliyun_python_sdk_alimt.request.v20181012 as alimt_api

        cred = AccessKeyCredential(cfg["aliyun_ak"], cfg["aliyun_sk"])
        client = AcsClient(region_id=cfg["aliyun_region"], credential=cred)
        req = alimt_api.TranslateGeneralRequest.TranslateGeneralRequest()
        req.set_FormatType("text")
        req.set_SourceLanguage(_to_vendor("aliyun", src, _VENDOR_LANG["aliyun"]))
        req.set_TargetLanguage(_to_vendor("aliyun", tgt, _VENDOR_LANG["aliyun"]))
        req.set_SourceText(text)
        req.set_Scene("general")
        body = await asyncio.to_thread(client.do_action_with_exception, req)
        result = json.loads(body)
        if result.get("Code") != "200":
            raise RuntimeError(f"阿里云错误: {result.get('Message')}")
        translated = (result.get("Data") or {}).get("Translated")
        if not translated:
            raise RuntimeError("阿里云返回空结果")
        return f"[{src}→{tgt}] " + translated

    async def _local(self, text, src, tgt, cfg) -> str:
        prompt = f"Translate the following text from {src or 'auto'} to {tgt}. " \
                 f"Reply with ONLY the translation, no explanation.\n\n{text}"
        base = cfg["local_url"].rstrip("/")
        timeout = aiohttp.ClientTimeout(total=cfg["local_timeout"])
        if base.endswith("/v1"):
            url = base + "/chat/completions"
            payload = {
                "model": cfg["local_model"],
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            }
            async with self._http.post(url, json=payload, timeout=timeout) as resp:
                body = await resp.json()
            if "error" in body:
                raise RuntimeError(f"本地模型错误: {body['error']}")
            try:
                content = body["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                raise RuntimeError(f"本地模型响应结构异常: {body}")
            result = (content or "").strip()
        else:
            url = base + "/api/generate"
            payload = {"model": cfg["local_model"], "prompt": prompt, "stream": False}
            async with self._http.post(url, json=payload, timeout=timeout) as resp:
                body = await resp.json()
            if "error" in body:
                raise RuntimeError(f"本地模型错误: {body['error']}")
            result = (body.get("response") or "").strip()
        if not result:
            raise RuntimeError("本地模型返回空结果")
        return f"[{src}→{tgt}] " + result

    # ── 额度与限流（预扣制，防并发超卖） ────────────────────
    async def _reserve_quota(self, sid: str, chars: int, cfg: dict) -> tuple:
        async with self._lock:
            now = time.monotonic()
            today = date.today().isoformat()
            if len(self._quota) > 512:
                for k in [k for k, v in self._quota.items() if v.get("date") != today]:
                    self._quota.pop(k, None)
                    self._window.pop(k, None)
            q = self._quota.setdefault(sid, {"date": today, "chars": 0})
            if q["date"] != today:
                q.update(date=today, chars=0)
            if q["chars"] + chars > cfg["max_chars_per_day"]:
                return False, f"今日翻译额度已用完（{cfg['max_chars_per_day']} 字符/日），请明日再试或提高配置。"
            w = self._window.setdefault(sid, {"ts": now, "n": 0})
            if now - w["ts"] > 60:
                w.update(ts=now, n=0)
            if w["n"] >= cfg["max_qpm"]:
                return False, "请求过于频繁，请稍后再试。"
            q["chars"] += chars
            w["n"] += 1
            return True, ""

    async def _release_quota(self, sid: str, chars: int):
        async with self._lock:
            today = date.today().isoformat()
            q = self._quota.get(sid)
            if q and q.get("date") == today:
                q["chars"] = max(0, q["chars"] - chars)
            w = self._window.get(sid)
            if w and w["n"] > 0:
                w["n"] -= 1
