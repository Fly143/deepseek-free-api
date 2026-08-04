"""
上下文管理器 — DeepSeek Free API

在请求发出前检查 token 数，超长时自动裁剪，避免 "Content is too long" 错误。

裁剪策略（按优先级递进）：
  1. 删除最早的非关键对话轮次
  2. 清除 assistant 消息的 reasoning_content
  3. 截断 tool result 内容
  4. 截断 system message 中的工具定义部分
  5. 激进模式：只保留 system + 最后 1-2 轮
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import tiktoken

logger = logging.getLogger("context_manager")

# ── 分词器 ─────────────────────────────────────────────────

_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """用 cl100k_base 编码估算 token 数（对 DeepSeek 是合理近似）。"""
    return len(_enc.encode(text or ""))


# ── 每条消息的格式开销（DeepSeek 原生特殊 token）──────────

_OVERHEAD: Dict[str, int] = {
    "system": 2,      # <｜System｜> ... <｜end▁of▁instructions｜>
    "user": 1,        # <｜User｜>
    "assistant": 2,   # <｜Assistant｜> ... <｜end▁of▁sentence｜>
    "tool": 2,        # <｜Tool｜> ... <｜end▁of▁toolresults｜>
}
_BOS_OVERHEAD = 1     # <｜begin▁of▁sentence｜>
_ASST_TRAILER = 1     # 末尾 <｜Assistant｜>

# ── 预算参数 ───────────────────────────────────────────────

_OUTPUT_RESERVE_RATIO = 0.2
_MIN_OUTPUT_RESERVE = 1024
_MAX_OUTPUT_RESERVE = 65536
_DEFAULT_MAX_INPUT = 65536


# ── Token 估算 ─────────────────────────────────────────────

def estimate_message_tokens(msg: Dict[str, Any]) -> int:
    """估算单条消息在 DeepSeek 格式下的 token 数。"""
    role = msg.get("role", "")
    tokens = 0

    content = msg.get("content")
    if isinstance(content, str) and content:
        tokens += count_tokens(content)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                tokens += count_tokens(part.get("text", ""))

    if role == "assistant":
        rc = msg.get("reasoning_content")
        if rc and isinstance(rc, str):
            tokens += count_tokens(rc)
        tool_calls = msg.get("tool_calls")
        if tool_calls and isinstance(tool_calls, list):
            for tc in tool_calls:
                func = tc.get("function", {})
                tokens += count_tokens(func.get("name", ""))
                tokens += count_tokens(func.get("arguments", ""))

    tokens += _OVERHEAD.get(role, 0)
    return tokens


def estimate_tool_tokens(tools: Optional[List[Dict[str, Any]]]) -> int:
    """估算工具定义的 token 开销（DSML 包裹约 3 倍膨胀）。"""
    if not tools:
        return 0
    return count_tokens(json.dumps(tools, ensure_ascii=False)) * 3


# ── 裁剪策略 ───────────────────────────────────────────────

def _remove_oldest_turn(
    messages: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], bool]:
    """删除最早的一条非关键消息。

    保护：system 消息、最后一条 user 消息、最后一条 user 之后的所有消息。
    """
    if len(messages) <= 1:
        return messages, False

    last_user = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            last_user = i
            break

    for i in range(len(messages)):
        if messages[i].get("role") == "system":
            continue
        if i == last_user:
            continue
        if last_user >= 0 and i > last_user:
            continue
        messages.pop(i)
        return messages, True

    # 所有非 system 都受保护，尝试删 system
    for i in range(len(messages)):
        if i == last_user or (last_user >= 0 and i > last_user):
            continue
        messages.pop(i)
        return messages, True

    return messages, False


def prune_context(
    messages: List[Dict[str, Any]],
    max_tokens: int,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], int, str]:
    """渐进式裁剪消息以适配模型上下文限制。

    返回 (裁剪后消息, token 数, 操作描述)。
    """
    output_reserve = max(
        _MIN_OUTPUT_RESERVE,
        min(_MAX_OUTPUT_RESERVE, int(max_tokens * _OUTPUT_RESERVE_RATIO)),
    )
    budget = max_tokens - output_reserve
    tool_overhead = estimate_tool_tokens(tools)
    effective_budget = max(budget - tool_overhead, 1024)

    pruned = list(messages)
    actions: List[str] = []

    def _current() -> int:
        return sum(estimate_message_tokens(m) for m in pruned) + tool_overhead

    # 策略 1：逐条删除最早的对话轮次
    while _current() > effective_budget:
        pruned, ok = _remove_oldest_turn(pruned)
        if not ok:
            break
    removed = len(messages) - len(pruned)
    if removed > 0:
        actions.append(f"删除 {removed} 条消息")

    # 策略 2：清除 reasoning_content
    if _current() > effective_budget:
        saved = 0
        for m in pruned:
            if m.get("role") == "assistant" and m.get("reasoning_content"):
                saved += count_tokens(m["reasoning_content"])
                m["reasoning_content"] = ""
                if _current() <= effective_budget:
                    break
        if saved > 0:
            actions.append(f"清除推理链 ({saved} tokens)")

    # 策略 3：截断 tool result
    if _current() > effective_budget:
        saved = 0
        for m in pruned:
            if m.get("role") == "tool":
                content = str(m.get("content", ""))
                if len(content) > 100:
                    old = count_tokens(content)
                    m["content"] = content[:100] + "..."
                    saved += old - count_tokens(m["content"])
                    if _current() <= effective_budget:
                        break
        if saved > 0:
            actions.append(f"截断工具结果 ({saved} tokens)")

    # 策略 4：截断 system message 中的工具定义
    if _current() > effective_budget:
        for m in pruned:
            if m.get("role") == "system":
                content = str(m.get("content", ""))
                if "TOOL CALL FORMAT" in content:
                    old = count_tokens(content)
                    before = content.split("TOOL CALL FORMAT")[0].strip()
                    m["content"] = before + "\n\n[工具定义因上下文限制被截断]"
                    saved = old - count_tokens(m["content"])
                    if saved > 0:
                        actions.append(f"截断工具提示 ({saved} tokens)")
                        break

    desc = ", ".join(actions) if actions else "无需裁剪"
    return pruned, sum(estimate_message_tokens(m) for m in pruned), desc


def aggressive_prune(
    messages: List[Dict[str, Any]],
    max_tokens: int,
) -> Tuple[List[Dict[str, Any]], int, str]:
    """激进裁剪：只保留 system + 最后 1-2 轮对话。用于重试。"""
    system_msgs = [m for m in messages if m.get("role") == "system"]
    non_system = [m for m in messages if m.get("role") != "system"]

    keep = non_system[-4:] if len(non_system) > 4 else non_system
    for m in keep:
        if m.get("role") == "assistant":
            m["reasoning_content"] = ""

    pruned = system_msgs + keep

    # 还是太长？只保留 system + 最后一条 user
    if sum(estimate_message_tokens(m) for m in pruned) > max(2048, max_tokens):
        users = [m for m in pruned if m.get("role") == "user"]
        pruned = system_msgs + [users[-1]] if users else []

    # 还是太长？连 system 也删
    if sum(estimate_message_tokens(m) for m in pruned) > max(2048, max_tokens):
        pruned = [m for m in pruned if m.get("role") != "system"]

    count = sum(estimate_message_tokens(m) for m in pruned)
    removed = len(messages) - len(pruned)
    return pruned, count, f"激进裁剪：删除 {removed} 条消息"


# ── 主入口 ─────────────────────────────────────────────────

def enforce_context_limit(
    messages: List[Dict[str, Any]],
    max_input_tokens: int = _DEFAULT_MAX_INPUT,
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Dict[str, Any]], int, bool, str]:
    """主入口：在请求发出前检查并裁剪上下文。

    返回 (消息, token数, 是否裁剪了, 描述)。
    """
    tool_overhead = estimate_tool_tokens(tools)
    total = sum(estimate_message_tokens(m) for m in messages) + tool_overhead

    output_reserve = max(
        _MIN_OUTPUT_RESERVE,
        min(_MAX_OUTPUT_RESERVE, int(max_input_tokens * _OUTPUT_RESERVE_RATIO)),
    )
    budget = max_input_tokens - output_reserve
    if budget <= 0:
        budget = max(1024, max_input_tokens // 2)

    if total <= budget:
        return messages, total, False, "未超限"

    pruned, count, desc = prune_context(messages, max_input_tokens, tools)
    actually_pruned = len(messages) > len(pruned) or count < total

    if actually_pruned:
        logger.info("上下文裁剪: %d→%d tokens, %s", total, count, desc)

    return pruned, count, actually_pruned, desc


def retry_prune(
    messages: List[Dict[str, Any]],
    max_input_tokens: int,
    was_aggressive: bool = False,
) -> Tuple[List[Dict[str, Any]], int, str]:
    """重试时的更激进裁剪。"""
    if was_aggressive:
        # 终极兜底：只保留最后一条 user 消息
        users = [m for m in messages if m.get("role") == "user"]
        pruned = [users[-1]] if users else []
        return pruned, sum(estimate_message_tokens(m) for m in pruned), "终极兜底：仅保留最后 user 消息"
    return aggressive_prune(messages, max_input_tokens)
