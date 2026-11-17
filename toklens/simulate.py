"""上下文策略模拟：把一段对话按不同裁剪策略重放一遍，比较累计输入 token。

重要的一点（很多人算错）：

    摘要不是「替换」窗口内容，而是在窗口之外「额外加上去」的。

所以加摘要之后每轮的输入一定比纯窗口大。摘要买的不是省钱，
是「早期上下文不丢」。它的省钱对比对象是「全量历史」，不是「纯窗口」。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .tokens import count_tokens, DEFAULT_ENCODING

# 策略名 -> 说明
STRATEGIES = {
    "full": "全量历史（每轮把所有历史都发一遍）",
    "window": "滑动窗口（只带最近 N 轮原文）",
    "summary": "窗口 + 摘要（摘要额外加上，不替代窗口）",
    "none": "不带历史（对照组，会丢失上下文）",
}


@dataclass
class Turn:
    """一轮对话：用户一句 + 助手一句。"""

    user: str
    assistant: str

    def tokens(self, encoding: str = DEFAULT_ENCODING) -> tuple[int, int]:
        return count_tokens(self.user, encoding), count_tokens(self.assistant, encoding)


@dataclass
class SimResult:
    strategy: str
    total: int
    per_turn: list[int] = field(default_factory=list)
    note: str = ""


def simulate(
    turns: list[Turn],
    fixed_tokens: int = 0,
    summary_tokens: int = 0,
    window: int = 3,
    encoding: str = DEFAULT_ENCODING,
    priming: int = 0,
) -> dict[str, SimResult]:
    """把 turns 按四种策略各重放一遍。

    fixed_tokens   每一轮都重发的固定部分（系统提示 + 知识库）的 token 数
    summary_tokens 摘要本身的 token 数（仅 summary 策略使用）
    window         窗口策略保留最近几轮
    """
    results: dict[str, SimResult] = {}

    # 注意：这里的 hist 是「每轮一个元素」（一轮 = 用户 + 助手），
    # 所以窗口切片是 hist[-window:]，不是 hist[-window*2:]。
    # 写成 *2 会变成保留 2 倍轮数——这个 bug 会让「窗口策略」看起来
    # 比实际更贵，从而把结论整个带偏。

    # 全量历史
    total, per, hist = 0, [], []
    for t in turns:
        u, _ = t.tokens(encoding)
        cost = fixed_tokens + sum(hist) + u + priming
        total += cost
        per.append(cost)
        hist.append(sum(t.tokens(encoding)))
    results["full"] = SimResult("full", total, per)

    # 滑动窗口
    total, per, hist = 0, [], []
    for t in turns:
        u, _ = t.tokens(encoding)
        keep = hist[-window:]
        cost = fixed_tokens + sum(keep) + u + priming
        total += cost
        per.append(cost)
        hist.append(sum(t.tokens(encoding)))
    results["window"] = SimResult("window", total, per, note=f"窗口 {window} 轮")

    # 窗口 + 摘要
    total, per, hist = 0, [], []
    for t in turns:
        u, _ = t.tokens(encoding)
        keep = hist[-window:]
        dropped = len(hist) > len(keep)
        cost = fixed_tokens + (summary_tokens if dropped else 0) + sum(keep) + u + priming
        total += cost
        per.append(cost)
        hist.append(sum(t.tokens(encoding)))
    results["summary"] = SimResult("summary", total, per, note=f"窗口 {window} 轮 + 摘要")

    # 不带历史
    total, per = 0, []
    for t in turns:
        u, _ = t.tokens(encoding)
        cost = fixed_tokens + u + priming
        total += cost
        per.append(cost)
    results["none"] = SimResult("none", total, per)

    return results


def summarize(results: dict[str, SimResult]) -> str:
    """把模拟结果排成一张可读的表。"""
    base = results["full"].total or 1
    lines = []
    for key in ("full", "window", "summary", "none"):
        r = results[key]
        if key == "full":
            tag = "基准"
        else:
            tag = f"{(1 - r.total / base) * 100:+.1f}%"
        note = f"  {r.note}" if r.note else ""
        lines.append(f"  {key:<8} {r.total:>8}  {tag:>8}{note}")
    return "\n".join(lines)
