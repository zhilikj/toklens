"""token 计数与结构分析。

依赖 tiktoken。计数结果只对使用同一套 BPE 词表的模型准确——
见 README「这套数字对谁有效」一节。
"""

from __future__ import annotations

import re
from functools import lru_cache

DEFAULT_ENCODING = "o200k_base"
FALLBACK_ENCODING = "cl100k_base"

# 常用模型的编码器映射。没列出的用 DEFAULT_ENCODING。
_ENCODING_HINTS = {
    "gpt-4o": "o200k_base",
    "gpt-4o-mini": "o200k_base",
    "gpt-4.1": "o200k_base",
    "gpt-5": "o200k_base",
    "o1": "o200k_base",
    "o3": "o200k_base",
    "o4": "o200k_base",
    "gpt-4-turbo": "cl100k_base",
    "gpt-4": "cl100k_base",
    "gpt-3.5": "cl100k_base",
    "text-embedding-3": "cl100k_base",
}


@lru_cache(maxsize=8)
def _get_encoding(name: str):
    try:
        import tiktoken
    except ImportError as e:  # pragma: no cover - 环境问题
        raise SystemExit(
            "缺少依赖 tiktoken。安装：pip install tiktoken\n"
            "（这个工具刻意不做「估算」——估算出来的数不能用来做成本判断。）"
        ) from e
    try:
        return tiktoken.get_encoding(name)
    except Exception:
        return tiktoken.get_encoding(FALLBACK_ENCODING)


def encoding_for_model(model: str | None) -> str:
    """按模型名猜编码器名。猜不到就用默认。"""
    if not model:
        return DEFAULT_ENCODING
    m = model.lower()
    # 从长到短匹配，避免 "gpt-4" 抢在 "gpt-4o" 前面
    for key in sorted(_ENCODING_HINTS, key=len, reverse=True):
        if key in m:
            return _ENCODING_HINTS[key]
    return DEFAULT_ENCODING


def encode(text: str, encoding: str = DEFAULT_ENCODING) -> list[int]:
    return _get_encoding(encoding).encode(text or "")


def count_tokens(text: str, encoding: str = DEFAULT_ENCODING) -> int:
    return len(encode(text, encoding))


# ── 字符构成 ──────────────────────────────────────────────────────

_HAN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_LATIN = re.compile(r"[A-Za-z]")
_DIGIT = re.compile(r"[0-9]")


def char_kind_stats(text: str) -> dict:
    """汉字 / 拉丁字母 / 数字 / 其他 的字符数。

    「每汉字几个 token」比「每字符几个 token」有意义得多——
    这两者在混排文本里能差一倍以上。
    """
    text = text or ""
    han = len(_HAN.findall(text))
    latin = len(_LATIN.findall(text))
    digit = len(_DIGIT.findall(text))
    return {
        "chars": len(text),
        "han": han,
        "latin": latin,
        "digit": digit,
        "other": len(text) - han - latin - digit,
    }


# ── OpenAI 风格消息 ───────────────────────────────────────────────

# 每条消息的固定开销。不同厂商不一样，这里用 OpenAI 的量级做默认值，
# 可以用 --overhead 覆盖。它的影响在短消息很多时很明显。
DEFAULT_PER_MESSAGE_OVERHEAD = 4
DEFAULT_REPLY_PRIMING = 3


def _content_text(content) -> str:
    """content 可能是字符串，也可能是分块数组。"""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                # 常见形态：{"type":"text","text":"..."}
                parts.append(str(block.get("text") or block.get("content") or ""))
        return "\n".join(parts)
    return str(content)


def message_tokens(msg: dict, encoding: str = DEFAULT_ENCODING,
                   overhead: int = DEFAULT_PER_MESSAGE_OVERHEAD) -> int:
    """单条消息的 token 数（含固定开销）。"""
    total = overhead
    for key in ("role", "name"):
        if msg.get(key):
            total += count_tokens(str(msg[key]), encoding)
    total += count_tokens(_content_text(msg.get("content")), encoding)
    if msg.get("tool_calls"):
        for call in msg["tool_calls"]:
            fn = (call or {}).get("function") or {}
            total += count_tokens(str(fn.get("name") or ""), encoding)
            total += count_tokens(str(fn.get("arguments") or ""), encoding)
    return total


def request_tokens(messages: list[dict], encoding: str = DEFAULT_ENCODING,
                   overhead: int = DEFAULT_PER_MESSAGE_OVERHEAD,
                   priming: int = DEFAULT_REPLY_PRIMING) -> int:
    """一次请求的输入 token 总数。"""
    return sum(message_tokens(m, encoding, overhead) for m in messages) + priming


def split_messages(messages: list[dict], encoding: str = DEFAULT_ENCODING,
                   overhead: int = DEFAULT_PER_MESSAGE_OVERHEAD) -> dict:
    """按角色拆开，看固定部分和对话部分各占多少。"""
    out = {"system": 0, "user": 0, "assistant": 0, "tool": 0, "other": 0}
    for m in messages:
        role = (m.get("role") or "other").lower()
        out[role if role in out else "other"] += message_tokens(m, encoding, overhead)
    fixed = out["system"]
    total = sum(out.values())
    out["total"] = total
    out["fixed_share"] = (fixed / total) if total else 0.0
    return out


def common_prefix_tokens(sequences: list[list[int]]) -> int:
    """多个 token 序列的公共前缀长度。

    这是这个工具最有用的一个数：把同一次会话的每一轮请求都喂进来，
    得到的长度就是「每一轮都在原样重发的那段」——系统提示、知识库、
    少样本示例。它是纯浪费，也是最容易被忽略的成本。
    """
    if not sequences:
        return 0
    n = min(len(s) for s in sequences)
    first = sequences[0]
    for i in range(n):
        t = first[i]
        for s in sequences[1:]:
            if s[i] != t:
                return i
    return n


def flatten_request_tokens(messages: list[dict], encoding: str = DEFAULT_ENCODING) -> list[int]:
    """把一次请求压成单个 token 序列，用于算公共前缀。

    角色名也要算进去——否则两条 role 不同但正文相同的消息会被误判成相同。
    """
    seq: list[int] = []
    for m in messages:
        seq += encode(f"<{m.get('role') or ''}>", encoding)
        seq += encode(_content_text(m.get("content")), encoding)
        seq += encode("</>", encoding)
    return seq
