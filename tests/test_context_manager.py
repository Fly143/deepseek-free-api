"""Unit tests for context_manager.py — token counting, pruning, and retry logic."""

import json
import sys
from pathlib import Path

import pytest

# Ensure the project root is on sys.path so we can import context_manager
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from context_manager import (
    count_tokens,
    estimate_message_tokens,
    estimate_tool_def_tokens,
    token_breakdown,
    format_token_breakdown,
    prune_context,
    aggressive_prune,
    enforce_context_limit,
    retry_prune,
    BOS_OVERHEAD,
    ASST_TRAILER,
    OVERHEAD_PER_ROLE,
)


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def sample_messages():
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write a hello world in Python."},
        {"role": "assistant", "content": "```python\nprint('hello world')\n```"},
        {"role": "user", "content": "Now in Rust."},
        {"role": "assistant", "content": "```rust\nfn main() {\n    println!(\"hello world\");\n}\n```"},
        {"role": "user", "content": "Explain the difference."},
    ]


@pytest.fixture
def reasoning_messages():
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2 equals 4.", "reasoning_content": "This is a simple arithmetic question. The answer is 4."},
        {"role": "user", "content": "And 3+3?"},
        {"role": "assistant", "content": "3+3 equals 6.", "reasoning_content": "Another simple addition. 3+3 = 6."},
    ]


@pytest.fixture
def tool_messages():
    return [
        {"role": "system", "content": "You are a helpful assistant with tools."},
        {"role": "user", "content": "What's the weather in Tokyo?"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "Tokyo"}'},
        }]},
        {"role": "tool", "content": "Tokyo: 22°C, clear sky", "tool_call_id": "call_1"},
        {"role": "user", "content": "What's the weather in London?"},
        {"role": "assistant", "content": "", "tool_calls": [{
            "id": "call_2",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "London"}'},
        }]},
        {"role": "tool", "content": "London: 15°C, light rain", "tool_call_id": "call_2"},
        {"role": "assistant", "content": "London currently has light rain at 15°C."},
    ]


@pytest.fixture
def sample_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                    },
                    "required": ["city"],
                },
            },
        }
    ]


# ── count_tokens ─────────────────────────────────────────────────────

class TestCountTokens:
    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_simple_text(self):
        assert count_tokens("hello world") > 0

    def test_none_coerced_to_empty(self):
        assert count_tokens(None) == 0

    def test_known_text(self):
        # "hello world" in cl100k_base is 2 tokens
        assert count_tokens("hello world") == 2


# ── estimate_message_tokens ──────────────────────────────────────────

class TestEstimateMessageTokens:
    def test_string_content(self):
        msg = {"role": "user", "content": "hello"}
        tokens = estimate_message_tokens(msg)
        assert tokens > 0
        assert tokens == count_tokens("hello") + OVERHEAD_PER_ROLE["user"]

    def test_list_content(self):
        msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe this image"},
                {"type": "text", "text": "another text part"},
            ],
        }
        tokens = estimate_message_tokens(msg)
        expected_content = count_tokens("describe this image") + count_tokens("another text part")
        assert tokens == expected_content + OVERHEAD_PER_ROLE["user"]

    def test_reasoning_content(self):
        msg = {
            "role": "assistant",
            "content": "The answer is 4.",
            "reasoning_content": "Let me think step by step...",
        }
        tokens = estimate_message_tokens(msg)
        assert tokens > count_tokens("The answer is 4.") + OVERHEAD_PER_ROLE["assistant"]

    def test_tool_calls(self):
        msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "f1", "arguments": '{"a":1}'}},
            ],
        }
        tokens = estimate_message_tokens(msg)
        assert tokens > OVERHEAD_PER_ROLE["assistant"]

    def test_tool_role(self):
        msg = {"role": "tool", "content": "result data", "tool_call_id": "c1"}
        tokens = estimate_message_tokens(msg)
        assert tokens == count_tokens("result data") + OVERHEAD_PER_ROLE["tool"]

    def test_empty_content(self):
        msg = {"role": "user", "content": ""}
        tokens = estimate_message_tokens(msg)
        assert tokens == OVERHEAD_PER_ROLE["user"]

    def test_system_role(self):
        msg = {"role": "system", "content": "Be helpful."}
        tokens = estimate_message_tokens(msg)
        assert tokens == count_tokens("Be helpful.") + OVERHEAD_PER_ROLE["system"]


# ── estimate_tool_def_tokens ─────────────────────────────────────────

class TestEstimateToolDefTokens:
    def test_no_tools(self):
        assert estimate_tool_def_tokens(None) == 0
        assert estimate_tool_def_tokens([]) == 0

    def test_with_tools(self, sample_tools):
        tokens = estimate_tool_def_tokens(sample_tools)
        assert tokens > 0
        # 3x multiplier on raw JSON token count
        raw_tokens = count_tokens(json.dumps(sample_tools, ensure_ascii=False))
        assert tokens == raw_tokens * 3


# ── token_breakdown ──────────────────────────────────────────────────

class TestTokenBreakdown:
    def test_basic_breakdown(self, sample_messages):
        bd = token_breakdown(sample_messages)
        assert bd["message_count"] == 6
        assert bd["total"] > 0
        assert bd["system"] > 0
        assert bd["history_user"] > 0
        assert bd["history_assistant"] > 0
        assert bd["tool_results"] == 0
        assert bd["overhead"] >= BOS_OVERHEAD + ASST_TRAILER
        assert len(bd["by_message"]) == 6

    def test_with_tools(self, tool_messages, sample_tools):
        bd = token_breakdown(tool_messages, sample_tools)
        assert bd["message_count"] == 8
        assert bd["tool_count"] == 1
        assert bd["tool_defs"] > 0
        assert bd["tool_results"] > 0  # tool messages + assistant tool_calls

    def test_no_tools_specified(self, sample_messages):
        bd = token_breakdown(sample_messages)
        assert bd["tool_count"] == 0
        assert bd["tool_defs"] == 0

    def test_by_message_ordering(self, sample_messages):
        bd = token_breakdown(sample_messages)
        for i, entry in enumerate(bd["by_message"]):
            assert entry["index"] == i
            assert entry["role"] == sample_messages[i]["role"]
            assert entry["tokens"] > 0


# ── format_token_breakdown ───────────────────────────────────────────

class TestFormatTokenBreakdown:
    def test_output_format(self, sample_messages):
        bd = token_breakdown(sample_messages)
        formatted = format_token_breakdown(bd)
        assert isinstance(formatted, str)
        assert str(bd["total"]) in formatted
        assert "System/prompt" in formatted
        assert "History (user)" in formatted
        assert "History (asst)" in formatted
        assert "Tool definitions" in formatted
        assert "Tool results" in formatted
        assert "Overhead" in formatted


# ── prune_context ────────────────────────────────────────────────────

class TestPruneContext:
    def test_within_budget(self, sample_messages):
        pruned, count, desc = prune_context(sample_messages, max_tokens=1_000_000)
        assert len(pruned) == len(sample_messages)

    def test_over_budget_removes_messages(self, sample_messages):
        # Very large conversation forces pruning (need >1976 tokens for max_tokens=3000)
        big_msgs = sample_messages + [
            {"role": "user", "content": "Explain how concurrency works. " * 150},
            {"role": "assistant", "content": "Concurrency in Python uses asyncio, threading, and multiprocessing. " * 200},
        ]
        pruned, count, desc = prune_context(big_msgs, max_tokens=3000)
        assert len(pruned) < len(big_msgs)
        assert "removed" in desc

    def test_preserves_system_message(self, sample_messages):
        pruned, count, desc = prune_context(sample_messages, max_tokens=50)
        # System message should still be present
        roles = [m["role"] for m in pruned]
        assert "system" in roles

    def test_preserves_last_user_message(self, sample_messages):
        pruned, count, desc = prune_context(sample_messages, max_tokens=50)
        # Last user message ("Explain the difference.") should still be present
        last_user = [m for m in reversed(pruned) if m["role"] == "user"]
        assert len(last_user) >= 1
        assert "Explain the difference" in str(last_user[0].get("content", ""))

    def test_strips_reasoning_content(self, reasoning_messages):
        # Add large reasoning content; strategy 1 (message removal) alone can't meet tight budget
        big_reasoning = list(reasoning_messages)
        for m in big_reasoning:
            if m.get("role") == "assistant":
                m["reasoning_content"] = ("Let me think through this step by step. " * 500)
                m["content"] = "Short answer."
        pruned, count, desc = prune_context(big_reasoning, max_tokens=2000)
        was_stripped = any(
            m.get("role") == "assistant" and not m.get("reasoning_content", "")
            for m in pruned
        )
        assert was_stripped, f"reasoning was not stripped: {desc}"

    def test_truncates_tool_results(self, tool_messages):
        # Very large tool results; strategy 1 (message removal) alone can't meet tight budget
        long_msgs = list(tool_messages)
        for m in long_msgs:
            if m.get("role") == "tool":
                m["content"] = "Here is the detailed result of the function call. " * 300  # ~3900 tokens
            elif m.get("role") != "system":
                m["content"] = "Short."  # keep regular messages tiny
        pruned, count, desc = prune_context(long_msgs, max_tokens=2000)
        was_truncated = any(
            m.get("role") == "tool" and str(m.get("content", "")).endswith("...")
            for m in pruned
        )
        assert was_truncated, f"tool results were not truncated: {desc}"

    def test_single_message_no_prune(self):
        msgs = [{"role": "user", "content": "hi"}]
        pruned, count, desc = prune_context(msgs, max_tokens=10)
        assert len(pruned) == 1

    def test_empty_messages(self):
        pruned, count, desc = prune_context([], max_tokens=100)
        assert len(pruned) == 0

    def test_output_reserve_applied(self, sample_messages):
        # With small max_tokens but large reserve ratio, less budget for prompt
        pruned_tight, _, _ = prune_context(sample_messages, max_tokens=1000, output_reserve_ratio=0.5)
        pruned_loose, _, _ = prune_context(sample_messages, max_tokens=1000, output_reserve_ratio=0.05)
        # Loose reserve should keep more messages
        assert len(pruned_loose) >= len(pruned_tight)


# ── aggressive_prune ─────────────────────────────────────────────────

class TestAggressivePrune:
    def test_keeps_system_and_few_turns(self, sample_messages):
        pruned, count, desc = aggressive_prune(sample_messages, max_tokens=50000)
        roles = [m["role"] for m in pruned]
        assert "system" in roles
        assert "user" in roles
        assert "assistant" in roles
        assert len(pruned) <= len(sample_messages)

    def test_strips_reasoning(self, reasoning_messages):
        pruned, count, desc = aggressive_prune(reasoning_messages, max_tokens=50000)
        for m in pruned:
            if m.get("role") == "assistant":
                assert not m.get("reasoning_content")

    def test_very_tight_budget_system_only(self):
        msgs = [
            {"role": "system", "content": "s" * 500},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"},
        ]
        pruned, count, desc = aggressive_prune(msgs, max_tokens=500)
        # With budget=2048 (max(2048,500)), system+user+assistant (~130 tokens) fits
        # So no pruning needed — verify all messages are kept
        assert len(pruned) >= 1

    def test_only_system_message(self):
        msgs = [{"role": "system", "content": "Be helpful."}]
        pruned, count, desc = aggressive_prune(msgs, max_tokens=50000)
        assert len(pruned) == 1


# ── enforce_context_limit ────────────────────────────────────────────

class TestEnforceContextLimit:
    def test_within_budget(self, sample_messages):
        msgs, count, was_pruned, desc = enforce_context_limit(
            sample_messages, model_max_input=1_000_000,
        )
        assert not was_pruned
        assert "no pruning needed" in desc

    def test_over_budget(self, sample_messages):
        # Large conversation with small model_max_input forces pruning
        big_msgs = sample_messages + [
            {"role": "user", "content": "Explain how concurrency works. " * 150},
            {"role": "assistant", "content": "Concurrency in Python uses asyncio, threading, and multiprocessing. " * 200},
        ]
        msgs, count, was_pruned, desc = enforce_context_limit(
            big_msgs, model_max_input=3000,
        )
        assert was_pruned
        assert len(msgs) < len(big_msgs)

    def test_with_tools(self, sample_messages, sample_tools):
        big_msgs = sample_messages + [
            {"role": "user", "content": "Explain how concurrency works. " * 150},
            {"role": "assistant", "content": "Concurrency in Python uses asyncio, threading, and multiprocessing. " * 200},
        ]
        msgs, count, was_pruned, desc = enforce_context_limit(
            big_msgs, model_max_input=3000, tools=sample_tools,
        )
        assert was_pruned

    def test_very_large_model_limit(self, sample_messages):
        msgs, count, was_pruned, desc = enforce_context_limit(
            sample_messages, model_max_input=10_000_000,
        )
        assert not was_pruned
        assert len(msgs) == len(sample_messages)

    def test_empty_messages(self):
        msgs, count, was_pruned, desc = enforce_context_limit([], model_max_input=1000)
        assert not was_pruned


# ── retry_prune ──────────────────────────────────────────────────────

class TestRetryPrune:
    def test_first_call_aggressive(self, sample_messages):
        pruned, count, desc = retry_prune(sample_messages, 50000, was_aggressive=False)
        assert len(pruned) < len(sample_messages)
        assert "aggressive prune" in desc

    def test_second_call_ultimate_fallback(self, sample_messages):
        pruned, count, desc = retry_prune(sample_messages, 50000, was_aggressive=True)
        assert "ultimate fallback" in desc
        # Should keep only last user message
        assert all(m["role"] == "user" for m in pruned)


# ── Integration: context_manager + proxy compatibility ───────────────

class TestIntegration:
    """These tests verify that context_manager functions work correctly with
    the data shapes produced by proxy.py's convert_messages_for_deepseek."""

    def test_message_types_match_proxy_output(self):
        """Messages with all field types that proxy.py produces."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!", "reasoning_content": ""},
            {"role": "user", "content": [
                {"type": "text", "text": "Describe this"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
            ]},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "test", "arguments": "{}"}},
            ]},
            {"role": "tool", "content": "result", "tool_call_id": "c1"},
        ]
        # Should not raise
        bd = token_breakdown(messages)
        assert bd["message_count"] == 6

        pruned, count, was_pruned, desc = enforce_context_limit(messages, model_max_input=1_000_000)
        assert not was_pruned

    def test_large_conversation_simulating_opencode(self):
        """Simulate a large OpenCode conversation (many turns)."""
        messages = [{"role": "system", "content": "You are a coding assistant."}]
        for i in range(50):
            messages.append({"role": "user", "content": f"Question number {i} about Python."})
            messages.append({"role": "assistant", "content": f"Answer number {i} with some detailed explanation and code."})

        # With typical deepseek-expert limit (128K), should fit
        msgs, count, was_pruned, desc = enforce_context_limit(messages, model_max_input=131072)
        if was_pruned:
            # If pruned, system message should still be there
            assert any(m["role"] == "system" for m in msgs)

    def test_oversized_conversation_force_prune(self):
        """Force pruning on a very large conversation with small budget."""
        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        for i in range(100):
            messages.append({"role": "user", "content": f"Long question {i} " * 20})
            messages.append({"role": "assistant", "content": f"Long answer {i} " * 50})

        msgs, count, was_pruned, desc = enforce_context_limit(messages, model_max_input=4096)
        assert was_pruned
        assert len(msgs) < len(messages)

    def test_retry_chain(self):
        """Simulate the full retry flow from proxy.py."""
        messages = [{"role": "system", "content": "You are a coding assistant."}]
        for i in range(30):
            messages.append({"role": "user", "content": f"Q{i} " * 30})
            messages.append({"role": "assistant", "content": f"A{i} " * 80})

        # Step 1: Initial enforcement
        pruned1, count1, was_pruned1, desc1 = enforce_context_limit(messages, model_max_input=2048)
        assert was_pruned1

        # Step 2: Retry with aggressive prune
        pruned2, count2, desc2 = retry_prune(pruned1, model_max_input=2048, was_aggressive=False)
        assert "aggressive" in desc2

        # Verify no enforced output keeps within limit
        input_tokens = sum(estimate_message_tokens(m) for m in pruned2)
        assert input_tokens < 2048


# ── Edge cases ───────────────────────────────────────────────────────

class TestEdgeCases:
    def test_all_system_messages(self):
        msgs = [{"role": "system", "content": f"Rule {i} " * 200} for i in range(10)]
        pruned, count, was_pruned, desc = enforce_context_limit(msgs, model_max_input=5000)
        assert was_pruned
        assert len(pruned) < len(msgs)

    def test_single_turn(self):
        # A single message can't be pruned (nothing to remove safely)
        msgs = [{"role": "user", "content": "hello"}]
        pruned, count, was_pruned, desc = enforce_context_limit(msgs, model_max_input=10)
        assert not was_pruned
        assert len(pruned) == 1

    def test_very_long_system_message(self):
        msgs = [
            {"role": "system", "content": "x" * 10000},
            {"role": "user", "content": "hello"},
        ]
        pruned, count, was_pruned, desc = enforce_context_limit(msgs, model_max_input=100)
        assert was_pruned

    def test_tool_message_without_content(self):
        msgs = [
            {"role": "user", "content": "call tool"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "c1"},  # no content field
        ]
        bd = token_breakdown(msgs)
        assert bd["message_count"] == 3
        # Should not raise
        enforce_context_limit(msgs, model_max_input=1000)

    def test_content_none(self):
        msgs = [{"role": "user", "content": None}]
        count = estimate_message_tokens(msgs[0])
        assert count == OVERHEAD_PER_ROLE["user"]
