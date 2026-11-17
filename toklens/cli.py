"""命令行入口。"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .tokens import (
    DEFAULT_ENCODING,
    DEFAULT_PER_MESSAGE_OVERHEAD,
    DEFAULT_REPLY_PRIMING,
    char_kind_stats,
    common_prefix_tokens,
    count_tokens,
    encoding_for_model,
    flatten_request_tokens,
    request_tokens,
    split_messages,
)
from .simulate import STRATEGIES, Turn, simulate, summarize
from . import __version__

LINE = "─" * 68


def _read_text(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _load_json_or_jsonl(path: str):
    """支持三种输入：JSON 数组、单个 JSON 对象、JSONL。"""
    text = _read_text(path).strip()
    if not text:
        return []
    if text.startswith("["):
        return json.loads(text)
    if text.startswith("{"):
        # 可能是单个对象，也可能是 JSONL
        if "\n" in text:
            items = []
            for i, line in enumerate(text.splitlines(), 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    # 整个文件其实是一个格式化过的 JSON 对象
                    return json.loads(text)
            return items
        return json.loads(text)
    raise SystemExit(f"看不懂的输入格式：{path}")


# ── count ────────────────────────────────────────────────────────


def cmd_count(args) -> int:
    enc = args.encoding or encoding_for_model(args.model)
    print(f"编码器 {enc}   （可用 --encoding / --model 改）")
    print(LINE)
    print(f"  {'来源':<28}{'token':>8}{'字符':>8}{'汉字':>8}{'token/汉字':>11}")
    for path in args.files:
        text = _read_text(path)
        n = count_tokens(text, enc)
        s = char_kind_stats(text)
        per_han = f"{n / s['han']:.2f}" if s["han"] else "—"
        name = os.path.basename(path) if path != "-" else "(stdin)"
        print(f"  {name:<28}{n:>8}{s['chars']:>8}{s['han']:>8}{per_han:>11}")
    print()
    print("  提示：token/汉字 比 token/字符 有意义——中英混排时两者能差一倍。")
    return 0


# ── audit ────────────────────────────────────────────────────────


def cmd_audit(args) -> int:
    data = _load_json_or_jsonl(args.requests)
    if not isinstance(data, list) or not data:
        raise SystemExit("audit 需要一组请求（JSON 数组或 JSONL，每项含 messages）")

    enc = args.encoding or encoding_for_model(args.model)
    overhead = args.overhead

    totals, seqs = [], []
    for i, req in enumerate(data, 1):
        messages = req.get("messages") if isinstance(req, dict) else None
        if not messages:
            raise SystemExit(f"第 {i} 项没有 messages 字段")
        totals.append(request_tokens(messages, enc, overhead, args.priming))
        seqs.append(flatten_request_tokens(messages, enc))

    prefix = common_prefix_tokens(seqs)
    grand = sum(totals)
    fixed_total = prefix * len(totals)
    share = fixed_total / grand if grand else 0

    print(f"请求数 {len(totals)}   编码器 {enc}")
    print(LINE)
    print(f"  输入 token 合计        {grand:>10}")
    if totals[0]:
        print(f"  单轮输入               {totals[0]:>10}  →  {totals[-1]:>10}"
              f"   末轮是首轮的 {totals[-1] / totals[0]:.1f} 倍")
    print()
    print(f"  ★ 公共前缀长度         {prefix:>10}  token")
    print(f"    也就是每一轮都在原样重发的部分（含角色标记）")
    print(f"    累计                 {fixed_total:>10}  占全部输入 {share:.1%}")

    if args.show_breakdown and isinstance(data[0], dict):
        print()
        print("  第一轮的构成：")
        bd = split_messages(data[0]["messages"], enc, overhead)
        for role in ("system", "user", "assistant", "tool", "other"):
            if bd.get(role):
                print(f"    {role:<10}{bd[role]:>10}")
        print(f"    {'合法固定部分占比':<10}{bd['fixed_share']:>9.1%}"
              f"   （按 role=system 统计）")

    print()
    print(LINE)
    print("  结论")
    if share >= 0.5:
        print(f"  ★ 固定部分占了 {share:.0%}——裁历史最多只能影响剩下那 {1 - share:.0%}。")
        print("    先动固定部分：缩短它、检索得更准、或者让它命中 prompt 缓存。")
    elif share >= 0.25:
        print(f"  固定部分占 {share:.0%}，和对话部分量级相当，两边都值得看。")
    else:
        print(f"  固定部分只占 {share:.0%}，裁历史是这里收益最大的方向。")
    if prefix == 0:
        print("  注意：公共前缀是 0，说明每次请求的开头都不一样。")
        print("  这会同时让 prompt 缓存无法命中——检查是不是把时间戳或随机 ID 放进了开头。")
    return 0


# ── sim ──────────────────────────────────────────────────────────


def cmd_sim(args) -> int:
    data = _load_json_or_jsonl(args.conversation)
    if isinstance(data, dict):
        spec = data
    elif isinstance(data, list):
        spec = {"turns": data}
    else:
        raise SystemExit("sim 需要 {turns:[{user,assistant}...]} 或 turns 数组")

    enc = args.encoding or encoding_for_model(args.model)
    raw_turns = spec.get("turns") or []
    if not raw_turns:
        raise SystemExit("没有 turns")

    turns = [Turn(str(t.get("user", "")), str(t.get("assistant", ""))) for t in raw_turns]

    system = str(spec.get("system") or "")
    knowledge = str(spec.get("knowledge") or "")
    summary = str(spec.get("summary") or "")

    fixed = count_tokens(system, enc) + count_tokens(knowledge, enc)
    summary_tokens = count_tokens(summary, enc)

    print(f"轮数 {len(turns)}   编码器 {enc}")
    print(LINE)
    print(f"  固定部分（system + knowledge）  {fixed:>8} token")
    print(f"  摘要（仅 summary 策略使用）     {summary_tokens:>8} token")
    print(f"  对话本身合计                    "
          f"{sum(sum(t.tokens(enc)) for t in turns):>8} token")
    print()
    print("  累计输入 token")
    results = simulate(turns, fixed, summary_tokens, args.window, enc)
    print(summarize(results))

    full = results["full"].total or 1
    win = results["window"].total
    smy = results["summary"].total
    print()
    print(LINE)
    print("  怎么读")
    print(f"    纯窗口 {args.window} 轮          省 {(1 - win / full):.1%}（相对全量历史）")
    print(f"    窗口 + 摘要                省 {(1 - smy / full):.1%}")
    print(f"  ★ 摘要比纯窗口贵 {(smy / win - 1):+.1%}——它是在窗口之外额外加上去的，")
    print("    买的不是省钱，是「早期上下文不丢」。")
    print(f"    不带历史                   省 {(1 - results['none'].total / full):.1%}"
          f"（会丢失上下文，只是下限参考）")
    return 0


# ── 入口 ─────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="toklens",
        description="看清大模型请求里的 token 花在哪",
        epilog="文档与示例：见仓库 README",
    )
    p.add_argument("--version", action="version", version=f"toklens {__version__}")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--encoding", default=None,
                        help=f"tiktoken 编码器名（默认 {DEFAULT_ENCODING}）")
    common.add_argument("--model", default=None,
                        help="按模型名自动选编码器，如 gpt-4o / gpt-4-turbo")
    common.add_argument("--overhead", type=int, default=DEFAULT_PER_MESSAGE_OVERHEAD,
                        help=f"每条消息的固定开销（默认 {DEFAULT_PER_MESSAGE_OVERHEAD}）")
    common.add_argument("--priming", type=int, default=DEFAULT_REPLY_PRIMING,
                        help=f"回复引导开销（默认 {DEFAULT_REPLY_PRIMING}）")

    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("count", parents=[common], help="数 token 与中文密度")
    c.add_argument("files", nargs="+", help="文件路径，- 表示 stdin")
    c.set_defaults(func=cmd_count)

    a = sub.add_parser("audit", parents=[common],
                       help="找出每轮重发的固定前缀，算它占多少")
    a.add_argument("requests", help="JSON 数组 / JSONL，每项是含 messages 的请求体")
    a.add_argument("--show-breakdown", action="store_true", help="额外按 role 列出构成")
    a.set_defaults(func=cmd_audit)

    s = sub.add_parser("sim", parents=[common], help="比较不同的上下文裁剪策略")
    s.add_argument("conversation", help="JSON：{system, knowledge, summary, turns:[...]}")
    s.add_argument("--window", type=int, default=3, help="窗口保留几轮（默认 3）")
    s.set_defaults(func=cmd_sim)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
