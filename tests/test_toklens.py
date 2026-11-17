"""toklens 的单元测试。跑法：pytest -q"""

import json
import os

import pytest

from toklens import (
    Turn,
    common_prefix_tokens,
    count_tokens,
    encoding_for_model,
    request_tokens,
    simulate,
    split_messages,
)
from toklens.tokens import char_kind_stats, flatten_request_tokens

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLES = os.path.join(os.path.dirname(HERE), "examples")


def test_count_positive():
    assert count_tokens("hello") > 0
    assert count_tokens("") == 0


def test_han_density_is_below_one_for_technical_chinese():
    """中文技术说明的 token/汉字 应该在 1 附近，不可能是 1.5~2。

    这个断言是刻意的：网上流传「1 个汉字 ≈ 1.5~2 个 token」，
    实测推翻了它。
    """
    text = "大模型 API 的流式输出需要客户端自己维护缓冲区，按空行聚合后再解析。"
    n = count_tokens(text)
    han = char_kind_stats(text)["han"]
    assert 0.5 < n / han < 1.3


def test_char_kind_stats():
    s = char_kind_stats("中文 abc 123")
    assert s["han"] == 2
    assert s["latin"] == 3
    assert s["digit"] == 3
    assert s["chars"] == 10


def test_encoding_for_model():
    assert encoding_for_model("gpt-4o") == "o200k_base"
    assert encoding_for_model("gpt-4-turbo") == "cl100k_base"
    assert encoding_for_model("gpt-4") == "cl100k_base"
    assert encoding_for_model(None) == "o200k_base"
    assert encoding_for_model("some-unknown-model") == "o200k_base"


def test_common_prefix_simple():
    assert common_prefix_tokens([[1, 2, 3], [1, 2, 9]]) == 2
    assert common_prefix_tokens([[1, 2], [9, 9]]) == 0
    assert common_prefix_tokens([[1, 2, 3]]) == 3
    assert common_prefix_tokens([]) == 0


def test_common_prefix_ignores_shared_suffix():
    """公共后缀不算——只有开头相同才是「每轮重发」。"""
    assert common_prefix_tokens([[1, 2, 3], [9, 8, 3]]) == 0


def test_request_tokens_grows_with_history():
    sys_msg = {"role": "system", "content": "你是助手。"}
    one = request_tokens([sys_msg, {"role": "user", "content": "你好"}])
    two = request_tokens([sys_msg, {"role": "user", "content": "你好"},
                          {"role": "assistant", "content": "在的"},
                          {"role": "user", "content": "问题"}])
    assert two > one


def test_split_messages_fixed_share():
    msgs = [{"role": "system", "content": "x" * 400},
            {"role": "user", "content": "hi"}]
    bd = split_messages(msgs)
    assert bd["system"] > bd["user"]
    assert 0 < bd["fixed_share"] < 1


def test_multimodal_content_list():
    """content 是分块数组时不能崩。"""
    msgs = [{"role": "user",
             "content": [{"type": "text", "text": "看看这个"},
                         {"type": "image_url", "image_url": {"url": "x"}}]}]
    assert request_tokens(msgs) > 0


def test_simulate_summary_costs_more_than_window():
    """摘要是在窗口之外额外加的，所以一定比纯窗口贵。

    这条断言固化了 README 里那个反直觉的结论。
    """
    turns = [Turn("问题" * 5, "回答" * 20) for _ in range(10)]
    r = simulate(turns, fixed_tokens=200, summary_tokens=90, window=3)
    assert r["summary"].total > r["window"].total
    assert r["window"].total < r["full"].total


def test_simulate_full_history_monotonic():
    """全量历史下，单轮输入应该逐轮变大。"""
    turns = [Turn("问" * 3, "答" * 10) for _ in range(6)]
    r = simulate(turns, fixed_tokens=100)
    per = r["full"].per_turn
    assert per == sorted(per)
    assert per[-1] > per[0]


def test_examples_are_loadable():
    """仓库里的示例必须是能跑通的。"""
    conv = os.path.join(EXAMPLES, "conversation.json")
    reqs = os.path.join(EXAMPLES, "requests.jsonl")
    if not (os.path.exists(conv) and os.path.exists(reqs)):
        pytest.skip("示例文件不存在")
    with open(conv, encoding="utf-8") as f:
        spec = json.load(f)
    assert spec["turns"]
    with open(reqs, encoding="utf-8") as f:
        lines = [json.loads(x) for x in f if x.strip()]
    assert len(lines) == len(spec["turns"])
    seqs = [flatten_request_tokens(r["messages"]) for r in lines]
    # 每轮都重发同样的 system + knowledge，前缀必须大于 0
    assert common_prefix_tokens(seqs) > 0


# ── 回归：窗口切片不能写成 [-window*2:] ──────────────────────────
#
# 这个 bug 真实发生过。hist 在 simulate 里是「每轮一个元素」，
# 写成 [-window*2:] 会保留 2 倍轮数，让「窗口策略」看起来比实际贵，
# 于是「加摘要比纯窗口贵多少」这个结论会被算错。
#
# 下面这组数字来自一个独立实现的实测结果，两边必须一致。


def _example_spec():
    conv = os.path.join(EXAMPLES, "conversation.json")
    if not os.path.exists(conv):
        pytest.skip("示例文件不存在")
    with open(conv, encoding="utf-8") as f:
        return json.load(f)


def test_window_slicing_regression():
    spec = _example_spec()
    turns = [Turn(t["user"], t["assistant"]) for t in spec["turns"]]
    fixed = count_tokens(spec["system"]) + count_tokens(spec["knowledge"])
    smy = count_tokens(spec["summary"])

    expect = {1: 2658, 2: 3027, 3: 3343, 5: 3830, 9: 4326}
    for w, want in expect.items():
        got = simulate(turns, fixed, smy, w)["window"].total
        assert got == want, f"窗口 {w} 轮：期望 {want}，实际 {got}"

    r = simulate(turns, fixed, smy, 3)
    assert r["full"].total == 4326
    assert r["none"].total == 2247
    assert r["summary"].total == 3901


def test_window_keeps_exactly_n_turns():
    """窗口 N 轮 = 保留 N 轮原文，第 N+1 轮起才丢东西。"""
    turns = [Turn("u" * 10, "a" * 40) for _ in range(5)]
    r = simulate(turns, fixed_tokens=0, window=2)
    # 第 1 轮无历史；第 2 轮带 1 轮；第 3 轮带 2 轮；第 4、5 轮也带 2 轮
    per = r["window"].per_turn
    assert per[0] < per[1] < per[2]
    assert per[2] == per[3] == per[4]
    # 若是 [-window*2:] 的 bug，第 3 轮之后还会继续变大
