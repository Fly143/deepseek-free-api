"""上下文压缩器 — DeepSeek free API

两种模式，回复前检测超长触发：
  - truncation：滑动窗口裁剪，保留 head + tail，丢弃 middle
  - compress：三段式 LLM 摘要，保留 head + tail + middle 摘要

默认阈值基于字符数；裁剪兜底走 context_manager.enforce_context_limit。
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

MAX_QUERY_CHARS = int(os.getenv("DS_MAX_QUERY_CHARS", "1048576"))
COMPRESS_THRESHOLD = 0.80
TAIL_TOKEN_BUDGET = 20000
TAIL_CHAR_BUDGET = TAIL_TOKEN_BUDGET * 4  # ~80K 字符

DEFAULT_COMPRESSION_MODE = os.getenv("DS_COMPRESSION_MODE", "compress")


def _msg_text(msg: Dict[str, Any]) -> str:
    content = msg.get("content") or ""
    if isinstance(content, list):
        parts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        return " ".join(parts)
    return str(content)


def estimate_chars(messages: List[Dict[str, Any]]) -> int:
    total = 0
    for msg in messages:
        total += len(_msg_text(msg)) + len(str(msg.get("role", ""))) + 2
    return total


def should_compress(messages: List[Dict[str, Any]]) -> bool:
    return estimate_chars(messages) > MAX_QUERY_CHARS * COMPRESS_THRESHOLD


def split_messages(messages: List[Dict[str, Any]]):
    if not messages:
        return [], [], []

    system_msgs = [m for m in messages if m.get("role") == "system"]
    other_msgs = [m for m in messages if m.get("role") != "system"]

    head = list(system_msgs)
    head_turns = []
    saw_user = False
    for msg in other_msgs:
        head_turns.append(msg)
        if msg.get("role") == "user":
            saw_user = True
        if saw_user and msg.get("role") == "assistant":
            break
    head.extend(head_turns)

    tail = []
    tail_chars = 0
    for msg in reversed(other_msgs):
        msg_chars = len(_msg_text(msg)) + len(str(msg.get("role", ""))) + 2
        if tail_chars + msg_chars > TAIL_CHAR_BUDGET and tail:
            break
        tail.insert(0, msg)
        tail_chars += msg_chars

    head_ids = {id(m) for m in head}
    tail_ids = {id(m) for m in tail}
    middle = [m for m in messages if id(m) not in head_ids and id(m) not in tail_ids]
    return head, middle, tail


def truncate_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    head, middle, tail = split_messages(messages)
    if not middle:
        return messages
    result = head + tail
    print(f"[ContextCompressor:truncation] {len(messages)} 条 → {len(result)} 条 (丢弃 {len(middle)} 条)")
    return result


def build_summary_prompt(messages: List[Dict[str, Any]]) -> str:
    lines = [f"{m.get('role', '')}: {_msg_text(m)}" for m in messages]
    conversation_text = "\n".join(lines)
    return (
        "请将以下对话历史压缩为一段简洁的中文摘要，保留关键信息：\n"
        "- 已完成的任务和决策\n"
        "- 待解决的问题\n"
        "- 用户偏好和重要上下文\n"
        "- 关键数据和结论\n\n"
        f"对话历史：\n{conversation_text}\n\n"
        "摘要（中文）："
    )


async def compress_messages(
    messages: List[Dict[str, Any]],
    model: str,
    call_fn,
) -> Tuple[Optional[str], List[Dict[str, Any]]]:
    """call_fn(prompt, model) -> 可 await 的 content 字符串或 (content, ...) 元组。"""
    head, middle, tail = split_messages(messages)
    if not middle:
        return None, messages

    prompt = build_summary_prompt(middle)
    try:
        result = await call_fn(prompt, model)
        content = result[0] if isinstance(result, (tuple, list)) else result
    except Exception as e:
        print(f"[ContextCompressor:compress] 摘要调用失败，回退到截断: {e}")
        return None, truncate_messages(messages)

    if not content:
        return None, truncate_messages(messages)

    summary_msg = {
        "role": "system",
        "content": f"[对话摘要] 原对话已压缩，请基于以下摘要继续对话：\n{content}",
    }
    compressed = head + [summary_msg] + tail
    print(f"[ContextCompressor:compress] {len(messages)} 条 → {len(compressed)} 条 (middle {len(middle)} 条 → 摘要)")
    return content, compressed
