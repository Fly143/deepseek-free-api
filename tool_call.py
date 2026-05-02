"""
工具调用模块 — deepseek-free-api (DSML 格式)

基于 ds2api 的 DSML XML + CDATA 工具调用格式。
流式筛分 + DSML 解析 + 工具历史格式化。
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

from tool_dsml import (
    parse_dsml_tool_calls,
    format_tool_calls_for_prompt,
    build_dsml_tool_prompt,
    sanitize_loose_cdata,
)

__all__ = [
    "build_tool_prompt",
    "get_tool_names",
    "extract_tool_call",
    "normalize_tool_call",
    "clean_tool_text",
    "convert_messages_for_deepseek",
    "parse_dsml_tool_calls",
    "format_tool_calls_for_prompt",
    "build_dsml_tool_prompt",
    "sanitize_loose_cdata",
]


# ─── 安全取值 ─────────────────────────────────────────────────

def _safe_get(d: Any, key: str, default: Any = None) -> Any:
    """安全取值 — 兼容 dict、pydantic model、任意对象。"""
    if d is None:
        return default
    if isinstance(d, dict):
        return d.get(key, default)
    return getattr(d, key, default)


# ─── 构建工具提示词 ──────────────────────────────────────────

def build_tool_prompt(tools: List[Dict[str, Any]]) -> str:
    """构建 DSML 格式的工具调用提示词。"""
    return build_dsml_tool_prompt(tools)


# ─── 提取工具名列表 ──────────────────────────────────────────

def get_tool_names(tools: List[Dict[str, Any]]) -> List[str]:
    """从 tools 列表提取所有 function name。"""
    names = []
    for tool in tools or []:
        fn = tool.get("function", tool)
        name = fn.get("name", None)
        if name:
            names.append(str(name))
    return names


# ─── 主入口：从文本中提取工具调用 ────────────────────────────

def extract_tool_call(
    text: str, tool_names: List[str]
) -> Tuple[Optional[List[Dict[str, Any]]], str]:
    """从文本中提取 DSML 工具调用。

    Returns:
        (tool_calls_list_or_None, cleaned_text)
    """
    if not text or not tool_names:
        return None, clean_tool_text(text) if text else text

    text = text.replace("\x00", "")

    # 尝试 DSML 解析
    tool_calls, cleaned = parse_dsml_tool_calls(text, tool_names)

    # 如果 DSML 解析失败，尝试 CDATA 修复后再试
    if not tool_calls:
        repaired = sanitize_loose_cdata(text)
        if repaired != text:
            tool_calls, cleaned = parse_dsml_tool_calls(repaired, tool_names)

    if tool_calls:
        # 标准化为 OpenAI 格式
        normalized = [normalize_tool_call(tc) for tc in tool_calls]
        normalized = [tc for tc in normalized if tc]
        return (normalized if normalized else None), clean_tool_text(text)

    return None, clean_tool_text(text)


# ─── 标准化工具调用为 OpenAI 格式 ────────────────────────────

def normalize_tool_call(raw: Any) -> Optional[Dict[str, Any]]:
    """将各种格式的 tool_call 标准化为 OpenAI 格式。

    OpenAI 格式:
        {
            "id": "call_xxx",
            "type": "function",
            "function": {
                "name": "...",
                "arguments": "{...}"   # JSON 字符串
            }
        }
    """
    if not raw:
        return None

    if isinstance(raw, list):
        raw = raw[0] if raw else {}
    if not isinstance(raw, dict):
        return None

    # 已经是标准格式
    if "function" in raw and isinstance(raw.get("function"), dict):
        func = raw["function"]
        if "name" in func and func["name"]:
            if "id" not in raw:
                raw["id"] = f"call_{uuid.uuid4().hex[:24]}"
            if "type" not in raw:
                raw["type"] = "function"
            if "arguments" in func and not isinstance(func["arguments"], str):
                func["arguments"] = json.dumps(func["arguments"], ensure_ascii=False)
            elif "arguments" not in func:
                func["arguments"] = "{}"
            return raw

    # 扁平格式: {"name": "xxx", "arguments": {...}}
    name = raw.get("name")
    if not name:
        return None

    args = raw.get("arguments") or raw.get("parameters") or raw.get("args") or {}
    if not isinstance(args, str):
        args = json.dumps(args, ensure_ascii=False)

    return {
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": args,
        },
    }


# ─── 清理工具文本 ────────────────────────────────────────────

def clean_tool_text(text: str) -> str:
    """清理文本中的 DSML 工具调用残留标签。"""
    if not text:
        return text

    # DSML 标签
    text = re.sub(r"</?\|?DSML\|?(?:tool_calls|invoke|parameter)[^>]*>", "", text, flags=re.IGNORECASE)
    # 无 DSML 前缀的 XML 标签
    text = re.sub(r"<tool_calls?>.*?</tool_calls?>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<invoke[^>]*>.*?</invoke>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<parameter[^>]*>.*?</parameter>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # CDATA
    text = re.sub(r"<!\[CDATA\[.*?\]\]>", "", text, flags=re.DOTALL)
    # DeepSeek search citation tags
    text = re.sub(r"\[citation:\d+\]", "", text)
    # 多余空行
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ─── 消息转换为 DeepSeek 格式 ──────────────────────────────

def convert_messages_for_deepseek(messages, tools=None):
    """将 OpenAI 消息列表转换为 DeepSeek 原生 prompt 格式。

    参照 ds2api 的 MessagesPrepare:
      <｜begin▁of▁sentence｜>系统消息<｜User｜>用户消息<｜Assistant｜>
    """
    # DeepSeek V3/R1 原生对话标记
    BOS = "<｜begin▁of▁sentence｜>"
    SYS = "<｜System｜>"
    USER = "<｜User｜>"
    ASST = "<｜Assistant｜>"
    TOOL = "<｜Tool｜>"
    EOS = "<｜end▁of▁sentence｜>"
    TOOL_END = "<｜end▁of▁toolresults｜>"
    SYS_END = "<｜end▁of▁instructions｜>"

    parts = [BOS]
    last_role = ""
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            text = str(content) if content else ""
            if text.strip():
                parts.append(SYS + text + SYS_END)
            last_role = "system"
        elif role == "user":
            if isinstance(content, list):
                text = " ".join(
                    p.get("text", "") for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            else:
                text = str(content)
            parts.append(USER + text)
            last_role = "user"
        elif role == "assistant":
            segs = []
            reasoning = msg.get("reasoning_content", "")
            if reasoning:
                segs.append(reasoning)
            if content and str(content).strip():
                segs.append(str(content).strip())
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                dsml = format_tool_calls_for_prompt(tool_calls)
                if dsml:
                    segs.append(dsml)
            if segs:
                parts.append(ASST + "\n\n".join(segs) + EOS)
            elif content:
                parts.append(ASST + str(content) + EOS)
            last_role = "assistant"
        elif role == "tool":
            result = str(content) if content else ""
            if result:
                try:
                    rd = json.loads(result)
                    if isinstance(rd, dict):
                        extracted = []
                        for k in ("output", "error", "result", "content"):
                            v = rd.get(k)
                            if v is not None and str(v).strip():
                                extracted.append(str(v).strip())
                        if extracted:
                            result = "\n".join(extracted)
                except (json.JSONDecodeError, ValueError):
                    pass
                parts.append(TOOL + result[:500] + TOOL_END)
            last_role = "tool"

    # 确保最后是 Assistant 开头
    if last_role != "assistant":
        parts.append(ASST)

    return "".join(parts)
