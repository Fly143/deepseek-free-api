"""
DSML — DeepSeek Model Language 解析器
基于 ds2api (Go) 的 DSML XML + CDATA 工具调用格式。

格式：
  <|DSML|tool_calls>
    <|DSML|invoke name="TOOL_NAME">
      <|DSML|parameter name="ARG_NAME"><![CDATA[VALUE]]></|DSML|parameter>
    </|DSML|invoke>
  </|DSML|tool_calls>
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple


_TOOL_MARKUP_NAMES = {"tool_calls", "invoke", "parameter"}
_CDATA_OPEN = "<![CDATA["
_CDATA_CLOSE = "]]>"


# ─── DSML 前缀剥离 ──────────────────────────────────────

def strip_dsml_markup(text: str) -> str:
    """去除 DSML 前缀，参照 ds2api 的字符级扫描逻辑。"""
    if not text:
        return text

    result_parts = []
    i = 0
    n = len(text)

    while i < n:
        c = text[i]

        # CDATA 块 → 原样保留
        if text[i:].startswith(_CDATA_OPEN):
            close = text.find(_CDATA_CLOSE, i + len(_CDATA_OPEN))
            if close == -1:
                result_parts.append(text[i:])
                break
            result_parts.append(text[i:close + len(_CDATA_CLOSE)])
            i = close + len(_CDATA_CLOSE)
            continue

        if c != '<':
            result_parts.append(c)
            i += 1
            continue

        end = text.find('>', i)
        if end == -1:
            result_parts.append(text[i:])
            break

        inner = text[i + 1 : end]

        closing = inner.startswith('/')
        rest = inner[1:] if closing else inner

        j = 0
        dsml = False
        while j < len(rest):
            ch = rest[j]
            if ch == '|':
                j += 1
                dsml = True
            elif ch in (' ', '\t', '\r', '\n'):
                j += 1
                dsml = True
            elif rest[j:j+4].lower() == 'dsml':
                j += 4
                dsml = True
            else:
                break

        if dsml:
            name_end = j
            while name_end < len(rest) and (rest[name_end].isalnum() or rest[name_end] == '_'):
                name_end += 1
            tag_name = rest[j:name_end].lower()

            if tag_name in _TOOL_MARKUP_NAMES:
                prefix = '</' if closing else '<'
                result_parts.append(prefix)
                result_parts.append(rest[j:])
                result_parts.append('>')
                i = end + 1
                continue

        result_parts.append(text[i : end + 1])
        i = end + 1

    return ''.join(result_parts)


# ─── CDATA ──────────────────────────────────────────────

def sanitize_loose_cdata(text: str) -> str:
    """修复未闭合的 CDATA 段。"""
    if _CDATA_OPEN not in text:
        return text

    result = ""
    start = 0
    while True:
        pos = text.find(_CDATA_OPEN, start)
        if pos == -1:
            result += text[start:]
            break
        result += text[start:pos]
        close_pos = text.find(_CDATA_CLOSE, pos + len(_CDATA_OPEN))
        if close_pos == -1:
            result += text[pos:] + _CDATA_CLOSE
            break
        else:
            result += text[pos:close_pos + len(_CDATA_CLOSE)]
            start = close_pos + len(_CDATA_CLOSE)

    return result


def extract_cdata(text: str) -> str:
    text = text.strip()
    if text.startswith(_CDATA_OPEN) and text.endswith(_CDATA_CLOSE):
        inner = text[len(_CDATA_OPEN):-len(_CDATA_CLOSE)]
        inner = inner.replace("]]]]><![CDATA[>", "]]>")
        return inner
    return text


# ─── 类型 ──────────────────────────────────────────────

def _auto_type(val: str) -> Any:
    if val.lower() == "true":
        return True
    if val.lower() == "false":
        return False
    if val.lower() in ("null", "none"):
        return None
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        pass
    return val


def _resolve_tool_name(name: str, tool_names: List[str]) -> str:
    if not tool_names:
        return name
    if name in tool_names:
        return name
    for tn in tool_names:
        if tn.lower() == name.lower():
            return tn
    snake = re.sub(r'(?<=[a-z0-9])([A-Z])', r'_\1', name).lower()
    if snake in tool_names:
        return snake
    for tn in tool_names:
        if tn.lower() == snake:
            return tn
    return name


# ─── DSML 解析 ─────────────────────────────────────────

def parse_dsml_tool_calls(
    text: str,
    tool_names: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], str]:
    if not text:
        return [], text

    normalized = strip_dsml_markup(text)
    tool_calls = []

    tc_pattern = re.compile(r"<tool_calls>(.*?)</tool_calls>", re.DOTALL | re.IGNORECASE)
    tc_single = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL | re.IGNORECASE)

    blocks = [(m.group(1), m.start(), m.end()) for m in tc_pattern.finditer(normalized)]
    if not blocks:
        blocks = [(m.group(1), m.start(), m.end()) for m in tc_single.finditer(normalized)]

    # 也处理裸 <invoke>（无 <tool_calls> 包裹），模型有时省略外层 wrapper
    if not blocks:
        invoke_bare = re.compile(r"<invoke\s+name=[\"']([^\"']+)[\"']>(.*?)</invoke>", re.DOTALL | re.IGNORECASE)
        for m in invoke_bare.finditer(normalized):
            name = m.group(1).strip()
            inner = m.group(2)
            args = _parse_parameters(inner)
            resolved = _resolve_tool_name(name, tool_names or [])
            tc = _format_openai_tool_call(resolved, args)
            if tc:
                tool_calls.append(tc)

    for block_text, _, _ in blocks:
        invoke_pattern = re.compile(
            r"<invoke\s+name=[\"']([^\"']+)[\"']>(.*?)</invoke>",
            re.DOTALL | re.IGNORECASE,
        )
        for m in invoke_pattern.finditer(block_text):
            name = m.group(1).strip()
            inner = m.group(2)
            args = _parse_parameters(inner)
            resolved = _resolve_tool_name(name, tool_names or [])
            tc = _format_openai_tool_call(resolved, args)
            if tc:
                tool_calls.append(tc)

    cleaned = _clean_dsml_text(normalized)
    return tool_calls, cleaned


def _parse_parameters(inner_text: str) -> Dict[str, Any]:
    args: Dict[str, Any] = {}
    param_pattern = re.compile(
        r"<parameter\s+name=[\"']([^\"']+)[\"']>(.*?)</parameter>",
        re.DOTALL | re.IGNORECASE,
    )
    for m in param_pattern.finditer(inner_text):
        key = m.group(1).strip()
        val_raw = m.group(2).strip()
        val_raw = extract_cdata(val_raw)
        try:
            val = json.loads(val_raw)
        except (json.JSONDecodeError, ValueError):
            val = _auto_type(val_raw)
        args[key] = val
    return args


def _format_openai_tool_call(name: str, args: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not name:
        return None
    return {
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
    }


def _clean_dsml_text(text: str) -> str:
    text = re.sub(r"<tool_calls?>.*?</tool_calls?>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<invoke[^>]*>.*?</invoke>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<parameter[^>]*>.*?</parameter>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<!\[CDATA\[.*?\]\]>", "", text, flags=re.DOTALL)
    text = re.sub(r"\[citation:\d+\]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ─── 工具历史格式化 ────────────────────────────────────

def format_tool_calls_for_prompt(tool_calls_raw: Any) -> str:
    if isinstance(tool_calls_raw, str):
        try:
            tool_calls_raw = json.loads(tool_calls_raw)
        except (json.JSONDecodeError, ValueError):
            return ""
    if not isinstance(tool_calls_raw, list) or not tool_calls_raw:
        return ""

    blocks = []
    for tc in tool_calls_raw:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function", {})
        name = tc.get("name") or fn.get("name", "")
        if not name:
            continue
        args = tc.get("arguments") or tc.get("input") or fn.get("arguments") or "{}"
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except (json.JSONDecodeError, ValueError):
                pass
        params = _format_params_dsml(args)
        block = f'  <|DSML|invoke name="{_escape_xml_attr(name)}">'
        if params.strip():
            block += "\n" + params + "\n  </|DSML|invoke>"
        else:
            block += "</|DSML|invoke>"
        blocks.append(block)

    if not blocks:
        return ""
    return "<|DSML|tool_calls>\n" + "\n".join(blocks) + "\n</|DSML|tool_calls>"


def _format_params_dsml(args: Any, indent: str = "    ") -> str:
    if isinstance(args, dict):
        if not args:
            return ""
        return "\n".join(_format_param_node(k, v, indent) for k, v in sorted(args.items()))
    elif isinstance(args, list):
        return "\n".join(_format_param_node("item", item, indent) for item in args)
    elif isinstance(args, str):
        return f'{indent}<|DSML|parameter name="content">{_cdata(args)}</|DSML|parameter>'
    else:
        return f'{indent}<|DSML|parameter name="value">{str(args)}</|DSML|parameter>'


def _format_param_node(name: str, value: Any, indent: str) -> str:
    open_tag = f'<|DSML|parameter name="{_escape_xml_attr(name)}">'
    close = "</|DSML|parameter>"
    if value is None:
        return f"{indent}{open_tag}{close}"
    elif isinstance(value, dict):
        inner = "\n".join(_format_xml_node(k, v, indent + "  ") for k, v in sorted(value.items()))
        return f"{indent}{open_tag}\n{inner}\n{indent}{close}" if inner.strip() else f"{indent}{open_tag}{close}"
    elif isinstance(value, list):
        inner = "\n".join(_format_xml_node("item", item, indent + "  ") for item in value)
        return f"{indent}{open_tag}\n{inner}\n{indent}{close}"
    elif isinstance(value, (bool, int, float)):
        return f"{indent}{open_tag}{str(value)}{close}"
    elif isinstance(value, str):
        return f"{indent}{open_tag}{_cdata(value)}{close}"
    return f"{indent}{open_tag}{_cdata(str(value))}{close}"


def _format_xml_node(name: str, value: Any, indent: str) -> str:
    if value is None:
        return f"{indent}<{name}></{name}>"
    elif isinstance(value, dict):
        inner = "\n".join(_format_xml_node(k, v, indent + "  ") for k, v in sorted(value.items()))
        return f"{indent}<{name}>\n{inner}\n{indent}</{name}>" if inner.strip() else f"{indent}<{name}></{name}>"
    elif isinstance(value, list):
        return "\n".join(_format_xml_node(name, item, indent) for item in value)
    elif isinstance(value, (bool, int, float)):
        return f"{indent}<{name}>{value}</{name}>"
    elif isinstance(value, str):
        return f"{indent}<{name}>{_cdata(value)}</{name}>"
    return f"{indent}<{name}>{_cdata(str(value))}</{name}>"


def _cdata(text: str) -> str:
    if "]]>" in text:
        text = text.replace("]]>", "]]]]><![CDATA[>")
    return f"<![CDATA[{text}]]>"


def _escape_xml_attr(text: str) -> str:
    return text.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")


# ─── 工具提示词 ────────────────────────────────────────

def build_dsml_tool_prompt(tools: List[Dict[str, Any]]) -> str:
    if not tools:
        return ""

    prompt = """TOOL CALL FORMAT — FOLLOW EXACTLY:

<|DSML|tool_calls>
  <|DSML|invoke name="TOOL_NAME_HERE">
    <|DSML|parameter name="PARAMETER_NAME"><![CDATA[PARAMETER_VALUE]]></|DSML|parameter>
  </|DSML|invoke>
</|DSML|tool_calls>

RULES:
1) Use the <|DSML|tool_calls> wrapper format.
2) Put one or more <|DSML|invoke> entries under a single <|DSML|tool_calls> root.
3) Put the tool name in the invoke name attribute: <|DSML|invoke name="TOOL_NAME">.
4) All string values must use <![CDATA[...]]>, even short ones.
5) Every top-level argument must be a <|DSML|parameter name="ARG_NAME">...</|DSML|parameter> node.
6) Objects use nested XML elements. Arrays may repeat <item> children.
7) Numbers, booleans, and null stay plain text.
8) Use only the parameter names in the tool schema. Do not invent fields.
9) Do NOT wrap XML in markdown fences. Do NOT output explanations or role markers.
10) If you call a tool, the first non-whitespace characters must be exactly <|DSML|tool_calls>.
11) Never omit the opening <|DSML|tool_calls> tag.

【WRONG — Do NOT do these】:

Wrong 1 — mixed text after XML:
  <|DSML|tool_calls>...</|DSML|tool_calls> I hope this helps.

Wrong 2 — Markdown code fences:
  ```xml
  <|DSML|tool_calls>...</|DSML|tool_calls>
  ```

Wrong 3 — missing opening wrapper:
  <|DSML|invoke name="TOOL_NAME">...</|DSML|invoke>
  </|DSML|tool_calls>

Remember: The ONLY valid way to use tools is the <|DSML|tool_calls>...</|DSML|tool_calls> block.

"""

    prompt += "Available tools:\n"
    for tool in tools:
        fn = tool.get("function", tool)
        name = fn.get("name", "")
        desc = fn.get("description", "").split("\n")[0].strip()
        if name:
            prompt += f"  - {name}: {desc}\n" if desc else f"  - {name}\n"

    return prompt
