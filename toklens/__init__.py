"""toklens —— 看清大模型请求里的 token 花在哪。

只做三件事：
  1. 数 token（含中文密度）
  2. 找出「每一轮都在重发」的固定前缀，算它占了多少
  3. 模拟不同的上下文裁剪策略，比较累计输入
"""

from .tokens import (
    count_tokens,
    encode,
    encoding_for_model,
    common_prefix_tokens,
    message_tokens,
    request_tokens,
    split_messages,
    char_kind_stats,
    flatten_request_tokens,
    DEFAULT_ENCODING,
)
from .simulate import Turn, simulate, summarize, STRATEGIES, SimResult

__version__ = "0.1.0"

__all__ = [
    "count_tokens",
    "encode",
    "encoding_for_model",
    "common_prefix_tokens",
    "message_tokens",
    "request_tokens",
    "split_messages",
    "char_kind_stats",
    "flatten_request_tokens",
    "DEFAULT_ENCODING",
    "Turn",
    "simulate",
    "summarize",
    "SimResult",
    "STRATEGIES",
    "__version__",
]
