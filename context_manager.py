"""
Context manager for DeepSeek Free API Proxy.

Handles token counting, context pruning, and retry logic to prevent
"Content is too long. Please shorten it and try again." errors from DeepSeek's upstream API.

Pruning strategies (applied in order):
1. Remove oldest user-assistant conversation turns (preserving system message & last user message)
2. Strip `reasoning_content` from assistant messages
3. Truncate tool result content further (past the existing 500-char limit)
4. Truncate system message (tool definitions section)
5. Aggressive: keep only system message + last 1-2 turns
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import tiktoken

logger = logging.getLogger("context_manager")

# ── Tokenizer ──────────────────────────────────────────────────────

_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    """Count tokens using OpenAI's cl100k_base encoding (reasonable approximation for DeepSeek V4)."""
    return len(_enc.encode(text or ""))


# ── Per-message overhead (DeepSeek native tokens added by convert_messages_for_deepseek) ──

# These are the tokens prepended/appended by convert_messages_for_deepseek in tool_call.py
OVERHEAD_PER_ROLE: Dict[str, int] = {
    "system": 2,     # <｜System｜> ... <｜end▁of▁instructions｜>
    "user": 1,       # <｜User｜>
    "assistant": 2,  # <｜Assistant｜> ... <｜end▁of▁sentence｜>
    "tool": 2,       # <｜Tool｜> ... <｜end▁of▁toolresults｜>
}

BOS_OVERHEAD = 1    # <｜begin▁of▁sentence｜>
ASST_TRAILER = 1    # Trailing <｜Assistant｜> appended by convert_messages_for_deepseek

# ── Default budget parameters ──────────────────────────────────────

DEFAULT_OUTPUT_RESERVE_RATIO = 0.2   # Reserve 20% of max_input_tokens for the model's response
MIN_OUTPUT_RESERVE = 1024            # At minimum reserve 1K tokens for output
MAX_OUTPUT_RESERVE = 65536           # At most reserve 64K tokens for output

DEFAULT_MAX_INPUT_TOKENS = 1_048_576  # Fallback if model_info unavailable
DEFAULT_MAX_OUTPUT_TOKENS = 1_048_576


# ── Token estimation ───────────────────────────────────────────────

def estimate_message_tokens(message: Dict[str, Any]) -> int:
    """Estimate the token count for a single message in DeepSeek native format.

    Accounts for content, reasoning_content, tool_calls, and role overhead tokens.
    """
    role = message.get("role", "")
    tokens = 0

    # Content (string or list of content parts)
    content = message.get("content")
    if isinstance(content, str) and content:
        tokens += count_tokens(content)
    elif isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                tokens += count_tokens(part.get("text", ""))

    # Reasoning content (assistant only)
    if role == "assistant":
        rc = message.get("reasoning_content")
        if rc and isinstance(rc, str):
            tokens += count_tokens(rc)

    # Tool calls (assistant only)
    if role == "assistant":
        tool_calls = message.get("tool_calls")
        if tool_calls and isinstance(tool_calls, list):
            for tc in tool_calls:
                func = tc.get("function", {})
                tokens += count_tokens(func.get("name", ""))
                tokens += count_tokens(func.get("arguments", ""))

    # Role overhead
    tokens += OVERHEAD_PER_ROLE.get(role, 0)
    return tokens


def estimate_tool_def_tokens(tools: List[Dict[str, Any]]) -> int:
    """Estimate the token overhead of tool definitions in the prompt.

    The build_dsml_tool_prompt() function wraps tool definitions in DSML markdown 
    with extensive instructions. We use a 3x multiplier on raw JSON size as an approximation.
    """
    if not tools:
        return 0
    raw_text = json.dumps(tools, ensure_ascii=False)
    raw_tokens = count_tokens(raw_text)
    return raw_tokens * 3  # DSML wrapping expands ~3x


# ── Token breakdown diagnostics ────────────────────────────────────

def token_breakdown(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Return a structured breakdown of token usage by category.

    Categories: system, tool_defs, history, tool_results, overhead.
    """
    breakdown: Dict[str, Any] = {
        "total": 0,
        "system": 0,
        "tool_defs": 0,
        "history_user": 0,
        "history_assistant": 0,
        "tool_results": 0,
        "overhead": BOS_OVERHEAD + ASST_TRAILER,
        "message_count": len(messages),
        "tool_count": len(tools) if tools else 0,
        "by_message": [],
    }

    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        tokens = estimate_message_tokens(msg)
        breakdown["total"] += tokens
        breakdown["by_message"].append({"index": i, "role": role, "tokens": tokens})

        if role == "system":
            breakdown["system"] += tokens
        elif role == "user":
            breakdown["history_user"] += tokens
        elif role == "assistant":
            if msg.get("tool_calls"):
                breakdown["tool_results"] += tokens
            else:
                breakdown["history_assistant"] += tokens
        elif role == "tool":
            breakdown["tool_results"] += tokens

    # Tool definition overhead
    if tools:
        tool_tokens = estimate_tool_def_tokens(tools)
        breakdown["tool_defs"] = tool_tokens
        breakdown["total"] += tool_tokens

    # Message role overhead
    for msg in messages:
        breakdown["overhead"] += OVERHEAD_PER_ROLE.get(msg.get("role", ""), 0)
    breakdown["total"] += breakdown["overhead"]

    return breakdown


def format_token_breakdown(breakdown: Dict[str, Any]) -> str:
    """Format token breakdown as a human-readable string."""
    lines = [
        f"Token breakdown ({breakdown['total']} total, {breakdown['message_count']} messages, "
        f"{breakdown['tool_count']} tools):",
        f"  System/prompt:   {breakdown['system']:>8} tokens",
        f"  Tool definitions: {breakdown['tool_defs']:>8} tokens",
        f"  History (user):  {breakdown['history_user']:>8} tokens",
        f"  History (asst):  {breakdown['history_assistant']:>8} tokens",
        f"  Tool results:    {breakdown['tool_results']:>8} tokens",
        f"  Overhead:        {breakdown['overhead']:>8} tokens",
    ]
    return "\n".join(lines)


# ── Pruning strategies ─────────────────────────────────────────────

def _prune_single_pass(
    messages: List[Dict[str, Any]],
    max_prompt_tokens: int,
) -> Tuple[List[Dict[str, Any]], int, bool]:
    """One pruning pass: remove the oldest non-essential message.

    Removal priority:
    1. Old non-system, non-last-user, non-after-last-user messages
    2. System message (last resort — only when nothing else can be removed)

    Protected (never removed): last user message, messages after last user.

    Returns (pruned, new_count, was_able_to_prune).
    Returns was_able_to_prune=False when all remaining messages are protected.
    """
    if len(messages) <= 1:
        return list(messages), sum(estimate_message_tokens(m) for m in messages), False

    last_user_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            last_user_idx = i
            break

    removed = False
    for i in range(len(messages)):
        if messages[i].get("role") == "system":
            continue
        if i == last_user_idx:
            continue
        if last_user_idx >= 0 and i > last_user_idx:
            continue
        messages.pop(i)
        removed = True
        break

    if not removed:
        for i in range(len(messages)):
            if i == last_user_idx:
                continue
            if last_user_idx >= 0 and i > last_user_idx:
                continue
            messages.pop(i)
            removed = True
            break

    if not removed:
        return messages, sum(estimate_message_tokens(m) for m in messages), False

    new_count = sum(estimate_message_tokens(m) for m in messages)
    return messages, new_count, True


def prune_context(
    messages: List[Dict[str, Any]],
    max_tokens: int,
    tools: Optional[List[Dict[str, Any]]] = None,
    output_reserve_ratio: float = DEFAULT_OUTPUT_RESERVE_RATIO,
) -> Tuple[List[Dict[str, Any]], int, str]:
    """Prune messages to fit within model context limit.

    Strategy (applied incrementally):
    1. Remove oldest non-system user-assistant turns one by one
    2. Strip reasoning_content from all assistant messages
    3. Truncate tool result content to 100 chars
    4. Truncate system message (remove tool prompt section)

    Returns: (pruned_messages, pruned_token_count, action_description)
    """
    # Calculate budget
    output_reserve = max(
        MIN_OUTPUT_RESERVE,
        min(MAX_OUTPUT_RESERVE, int(max_tokens * output_reserve_ratio)),
    )
    budget = max_tokens - output_reserve

    # Include tool definition overhead in budget check
    tool_overhead = estimate_tool_def_tokens(tools) if tools else 0
    effective_budget = max(budget - tool_overhead, 1024)

    pruned = list(messages)
    current = sum(estimate_message_tokens(m) for m in pruned) + tool_overhead
    actions: List[str] = []

    if current <= effective_budget:
        return pruned, current - tool_overhead, "within budget, no pruning"

    # Strategy 1: Remove oldest conversation turns
    while len(pruned) > 1:
        current = sum(estimate_message_tokens(m) for m in pruned) + tool_overhead
        if current <= effective_budget:
            break
        pruned, new_count, could_prune = _prune_single_pass(pruned, effective_budget)
        if not could_prune:
            break

    removed = len(messages) - len(pruned)
    if removed > 0:
        actions.append(f"removed {removed} messages")

    current = sum(estimate_message_tokens(m) for m in pruned) + tool_overhead

    # Strategy 2: Strip reasoning_content from assistant messages
    if current > effective_budget:
        saved = 0
        for m in pruned:
            if m.get("role") == "assistant" and m.get("reasoning_content"):
                saved += count_tokens(m["reasoning_content"])
                m["reasoning_content"] = ""
                current = sum(estimate_message_tokens(m) for m in pruned) + tool_overhead
                if current <= effective_budget:
                    break
        if saved > 0:
            actions.append(f"stripped reasoning ({saved} tokens)")

    # Strategy 3: Truncate tool result content to 100 chars
    if current > effective_budget:
        saved = 0
        for m in pruned:
            if m.get("role") == "tool":
                content = str(m.get("content", ""))
                if len(content) > 100:
                    old_tok = count_tokens(content)
                    m["content"] = content[:100] + "..."
                    saved += old_tok - count_tokens(m["content"])
                    current = sum(estimate_message_tokens(m) for m in pruned) + tool_overhead
                    if current <= effective_budget:
                        break
        if saved > 0:
            actions.append(f"trimmed tool results ({saved} tokens)")

    # Strategy 4: Truncate system message (tool prompt injection section)
    if current > effective_budget:
        for m in pruned:
            if m.get("role") == "system":
                content = str(m.get("content", ""))
                if "TOOL CALL FORMAT" in content:
                    old_tok = count_tokens(content)
                    # Keep only the part before tool instructions
                    before = content.split("TOOL CALL FORMAT")[0].strip()
                    m["content"] = before + "\n\n[Tool definitions truncated due to context limit.]"
                    saved = old_tok - count_tokens(str(m["content"]))
                    if saved > 0:
                        actions.append(f"truncated tool prompt ({saved} tokens)")
                        current = sum(estimate_message_tokens(m) for m in pruned) + tool_overhead
                        break

    final_count = sum(estimate_message_tokens(m) for m in pruned)
    action_desc = ", ".join(actions) if actions else "no pruning needed"
    return pruned, final_count, action_desc


def aggressive_prune(
    messages: List[Dict[str, Any]],
    max_tokens: int,
) -> Tuple[List[Dict[str, Any]], int, str]:
    """Aggressive pruning for retry after upstream 'Content is too long' rejection.

    Keeps only: system message + last 1-2 conversation turns.
    Strips all reasoning_content.
    """
    budget = max(2048, max_tokens)

    system_msgs = [m for m in messages if m.get("role") == "system"]
    non_system = [m for m in messages if m.get("role") != "system"]

    # Keep at most 4 non-system messages (≈ 2 user-assistant turns)
    keep_count = min(len(non_system), 4)
    keep = non_system[-keep_count:] if keep_count > 0 else []

    # Remove reasoning from all assistant messages
    for m in keep:
        if m.get("role") == "assistant":
            m["reasoning_content"] = ""

    pruned = system_msgs + keep

    # If still over budget, reduce to just system + last user
    current = sum(estimate_message_tokens(m) for m in pruned)
    if current > budget:
        users = [m for m in pruned if m.get("role") == "user"]
        if users:
            pruned = system_msgs + [users[-1]]
        current = sum(estimate_message_tokens(m) for m in pruned)

    # If STILL over budget (system too long), remove system too
    if current > budget and system_msgs:
        pruned = [m for m in pruned if m.get("role") != "system"]

    final_count = sum(estimate_message_tokens(m) for m in pruned)
    removed = len(messages) - len(pruned)
    return pruned, final_count, f"aggressive prune: removed {removed} messages ({current}→{final_count} tokens)"


# ── Main entry point ───────────────────────────────────────────────

def enforce_context_limit(
    messages: List[Dict[str, Any]],
    model_max_input: int,
    tools: Optional[List[Dict[str, Any]]] = None,
    output_reserve_ratio: float = DEFAULT_OUTPUT_RESERVE_RATIO,
) -> Tuple[List[Dict[str, Any]], int, bool, str]:
    """Main entry point: enforce the context limit for a set of messages.

    Steps:
    1. Estimate total tokens (including tool definition overhead)
    2. If within budget, return unchanged
    3. If over budget, apply :func:`prune_context`
    4. Return pruned messages, final token count, whether pruning was needed, and description

    Args:
        messages: List of OpenAI-format message dicts
        model_max_input: The model's max_input_tokens from discovery
        tools: Optional list of tool definitions (OpenAI format)
        output_reserve_ratio: Fraction of context to reserve for model output (default 0.2)

    Returns:
        Tuple of (messages, token_count, was_pruned, description)
    """
    # Current estimate including tool overhead
    tool_overhead = estimate_tool_def_tokens(tools) if tools else 0
    total = sum(estimate_message_tokens(m) for m in messages) + tool_overhead

    output_reserve = max(
        MIN_OUTPUT_RESERVE,
        min(MAX_OUTPUT_RESERVE, int(model_max_input * output_reserve_ratio)),
    )
    max_prompt_tokens = model_max_input - output_reserve
    # If model limit is so small that reserve consumes it all, use the bare minimum
    if max_prompt_tokens <= 0:
        max_prompt_tokens = max(1024, model_max_input // 2)

    logger.debug(
        "Context check: total_est=%d max_input=%d reserve=%d budget=%d",
        total, model_max_input, output_reserve, max_prompt_tokens,
    )

    if total <= max_prompt_tokens:
        return messages, total, False, "within budget, no pruning needed"

    # Prune
    pruned, pruned_count, action_desc = prune_context(
        messages, model_max_input, tools, output_reserve_ratio,
    )
    removed = len(messages) - len(pruned)
    desc = f"pruned {removed} messages ({total}→{pruned_count} tokens): {action_desc}"

    # Only report as pruned if something actually changed
    actually_pruned = removed > 0 or "stripped" in action_desc or "trimmed" in action_desc or "truncated" in action_desc

    if actually_pruned:
        logger.info("Context pruned: %s", desc)
    else:
        desc = f"need pruning but could not: {desc}"

    return pruned, pruned_count, actually_pruned, desc


def retry_prune(
    messages: List[Dict[str, Any]],
    model_max_input: int,
    was_aggressive: bool = False,
) -> Tuple[List[Dict[str, Any]], int, str]:
    """Even more aggressive pruning for retry attempts.

    First call (was_aggressive=False): aggressive_prune
    Second call (was_aggressive=True): keep only last user message, strip everything else
    """
    if was_aggressive:
        # Ultimate fallback: keep only last user message
        users = [m for m in messages if m.get("role") == "user"]
        pruned = [users[-1]] if users else []
        count = sum(estimate_message_tokens(m) for m in pruned)
        return pruned, count, "ultimate fallback: only last user message"

    return aggressive_prune(messages, model_max_input)
