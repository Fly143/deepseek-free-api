"""
工具调用模块 — deepseek-free-api (DSML 格式)

基于 ds2api 的 DSML XML + CDATA 工具调用格式。
流式筛分 + DSML 解析 + 工具历史格式化。
"""

from __future__ import annotations

import json
import math
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

from tool_dsml import (
    parse_dsml_tool_calls,
    format_tool_calls_for_prompt,
    build_dsml_tool_prompt,
    sanitize_loose_cdata,
    extract_cdata,
)

_MARKER_BREAK_RE = re.compile(r"<(?=[|｜])")


def escape_session_markers(text: str) -> str:
    """把工具输出里的 <| / <｜ 打断，防止伪造 <｜System｜> 等会话分隔标记注入提示词。"""
    return _MARKER_BREAK_RE.sub("<\u200b", text)


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

def build_tool_prompt(tools: List[Dict[str, Any]], passthrough: bool = False) -> str:
    """构建 DSML 格式的工具调用提示词。
    passthrough=True 时跳过格式说明书，直接嵌入原始工具定义。
    """
    return build_dsml_tool_prompt(tools, passthrough=passthrough)


# ─── 提取工具名列表 ──────────────────────────────────────────

def get_tool_names(tools: List[Dict[str, Any]]) -> List[str]:
    """从 tools 列表提取所有 function name。
    兼容两种格式：Chat Completions (tool.function.name) 和 Responses API (tool.name)。
    """
    names = []
    for tool in tools or []:
        fn = tool.get("function", {})
        name = fn.get("name") or tool.get("name")
        if name:
            names.append(str(name))
    return names


# ─── 主入口：从文本中提取工具调用 ────────────────────────────

class ToolFormatError(ValueError):
    pass


def _valid_schema_metadata(schema):
    if "$schema" in schema:
        dialect = schema["$schema"]
        if not isinstance(dialect, str) or not re.fullmatch(
            r"https?://json-schema\.org/(?:draft-0[467]/schema|draft/(?:2019-09|2020-12)/schema)#?", dialect
        ):
            return False
    return "$id" not in schema or isinstance(schema["$id"], str)


def _safe_schema_value(value, schema):
    if not isinstance(schema, dict):
        return False
    allowed = {"type", "title", "description", "default", "examples", "$schema", "$id", "$comment", "enum", "const", "minLength", "maxLength"}
    if set(schema) - allowed or not _valid_schema_metadata(schema):
        return False
    kind = schema.get("type")
    kinds = {
        "string": lambda x: isinstance(x, str),
        "integer": lambda x: isinstance(x, int) and not isinstance(x, bool),
        "number": lambda x: isinstance(x, (int, float)) and not isinstance(x, bool),
        "boolean": lambda x: isinstance(x, bool),
        "null": lambda x: x is None,
    }
    if kind not in kinds or not kinds[kind](value):
        return False
    if isinstance(value, float) and not math.isfinite(value):
        return False
    def same(candidate):
        if isinstance(value, bool) or isinstance(candidate, bool):
            return type(value) is type(candidate) and value == candidate
        return value == candidate
    if "enum" in schema and (not isinstance(schema["enum"], list) or not any(same(item) for item in schema["enum"])):
        return False
    if "const" in schema and not same(schema["const"]):
        return False
    if "minLength" in schema or "maxLength" in schema:
        if kind != "string":
            return False
        for key, compare in (("minLength", lambda n: len(value) >= n), ("maxLength", lambda n: len(value) <= n)):
            if key in schema and (not isinstance(schema[key], int) or isinstance(schema[key], bool) or schema[key] < 0 or not compare(schema[key])):
                return False
    return True


def _placeholder_call(name, command, tool_names, tools):
    if name != "bash" or name not in tool_names or not isinstance(tools, list):
        raise ToolFormatError("工具格式无法安全解析，请按 do JSON 格式重新生成调用。")
    matches = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        if fn.get("name") == name:
            matches.append(fn)
    if len(matches) != 1:
        raise ToolFormatError("工具定义无法确认，请按 do JSON 格式重新生成调用。")
    schema = matches[0].get("parameters")
    allowed = {"type", "properties", "required", "additionalProperties", "title", "description", "default", "examples", "$schema", "$id", "$comment"}
    if not isinstance(schema, dict) or set(schema) - allowed or not _valid_schema_metadata(schema) or schema.get("type") != "object":
        raise ToolFormatError("工具参数结构无法安全验证，请按 do JSON 格式重新生成调用。")
    properties = schema.get("properties")
    required = schema.get("required", [])
    if (not isinstance(properties, dict) or not isinstance(required, list)
            or any(not isinstance(key, str) for key in required)
            or not isinstance(schema.get("additionalProperties", True), bool)):
        raise ToolFormatError("工具参数结构无法安全验证，请按 do JSON 格式重新生成调用。")
    if not command or not _safe_schema_value(command, properties.get("command")):
        raise ToolFormatError("命令无法匹配工具参数，请按 do JSON 格式重新生成调用。")
    args = {"command": command}
    for key in required:
        if key == "command":
            continue
        field = properties.get(key)
        if not isinstance(field, dict) or "default" not in field or not _safe_schema_value(field["default"], field):
            raise ToolFormatError("缺少必要工具参数，请按 do JSON 格式重新生成调用。")
        args[key] = field["default"]
    return normalize_tool_call({"name": name, "arguments": args})


def _is_string_schema(spec) -> bool:
    """判断参数 schema 是否声明为 string（含 ["string","null"] 这类 nullable/联合形式）。"""
    if not isinstance(spec, dict):
        return False
    declared = spec.get("type")
    if declared == "string":
        return True
    if isinstance(declared, list) and "string" in declared:
        return True
    for key in ("anyOf", "oneOf"):
        options = spec.get(key)
        if isinstance(options, list) and any(_is_string_schema(opt) for opt in options):
            return True
    return False


def _code_ranges(text: str) -> list:
    """返回代码区行范围：围栏块（≤4 空格/tab 缩进）+ 4 空格/tab 缩进的缩进代码块。"""
    ranges = []
    fence_start = None
    fence_char = ""
    fence_len = 0
    offset = 0
    indented_start = None
    for line in text.splitlines(keepends=True):
        stripped = line.rstrip("\r\n")
        marker = re.match(r"^[ \t]{0,4}(`{3,}|~{3,})", line)
        if fence_start is not None:
            if (marker and marker.group(1)[0] == fence_char and len(marker.group(1)) >= fence_len
                    and not line[marker.end():].strip()):
                ranges.append((fence_start, offset + len(line)))
                fence_start = None
            offset += len(line)
            continue
        if marker:
            fence_start = offset
            fence_char = marker.group(1)[0]
            fence_len = len(marker.group(1))
            if indented_start is not None:
                ranges.append((indented_start, offset))
                indented_start = None
        elif stripped.startswith("    ") or stripped.startswith("\t"):
            if indented_start is None:
                indented_start = offset
        else:
            if indented_start is not None:
                ranges.append((indented_start, offset))
                indented_start = None
        offset += len(line)
    if fence_start is not None:
        ranges.append((fence_start, len(text)))
    if indented_start is not None:
        ranges.append((indented_start, len(text)))
    return ranges


def _declared_string_keys(tools) -> dict:
    """按工具名收集声明为 string 的参数名：{工具名: {参数名}}，供 DSML 解析保留原始文本。"""
    mapping: dict = {}
    if not isinstance(tools, list):
        return mapping
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        function = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = function.get("name")
        params = function.get("parameters")
        props = params.get("properties") if isinstance(params, dict) else None
        if not isinstance(name, str) or not isinstance(props, dict):
            continue
        keys = {key for key, spec in props.items() if _is_string_schema(spec)}
        if keys:
            mapping[name] = keys
    return mapping


def _validate_tool_calls(calls, tools):
    if tools is None:
        return calls
    from jsonschema import Draft202012Validator
    from jsonschema.validators import validator_for
    from referencing import Registry

    definitions = {}
    for tool in tools:
        if not isinstance(tool, dict):
            raise ToolFormatError("工具定义无效，请检查工具参数结构后重试。")
        function = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = function.get("name")
        if not isinstance(name, str) or name in definitions:
            raise ToolFormatError("工具定义无效，请检查工具参数结构后重试。")
        definitions[name] = function

    def check_references(node):
        if isinstance(node, dict):
            if "$schema" in node and not _valid_schema_metadata(node):
                raise ValueError("unsupported schema")
            for key, value in node.items():
                if key in ("$ref", "$dynamicRef", "$recursiveRef"):
                    if not isinstance(value, str) or not value.startswith("#"):
                        raise ValueError("external reference")
                check_references(value)
        elif isinstance(node, list):
            for value in node:
                check_references(value)

    for call in calls:
        try:
            function = call["function"]
            definition = definitions[function["name"]]
            declared_params = definition.get("parameters")
            schema = {} if declared_params is None else declared_params
            if not isinstance(schema, (dict, bool)):
                raise ValueError("invalid schema")
            check_references(schema)
            validator_class = validator_for(schema, default=Draft202012Validator)
            validator_class.check_schema(schema)
            arguments = json.loads(function["arguments"], parse_constant=_reject_json_constant)
            validator_class(schema, registry=Registry()).validate(arguments)
        except Exception as exc:
            raise ToolFormatError("工具参数不符合声明的结构或无法验证，请按工具定义重新生成调用。") from exc
    return calls


def _reject_json_constant(value):
    raise ValueError(value)


def extract_tool_call(
    text: str, tool_names: List[str], tools: Optional[List[Dict[str, Any]]] = None
) -> Tuple[Optional[List[Dict[str, Any]]], str]:
    """从文本中提取 DSML 工具调用。

    Returns:
        (tool_calls_list_or_None, cleaned_text)
    """
    if not text or not tool_names:
        return None, clean_tool_text(text) if text else text

    text = text.replace("\x00", "")

    fence_ranges = _code_ranges(text)

    native_start = "<|begin_of_tool_calls_begin|>"
    native_call = "<|begin_of_tool_call_begin|>"
    native_sep = "<|begin_of_tool_sep|>"
    native_key = "<|begin_of_tool_call_arg_key|>"
    native_call_end = "<|begin_of_tool_call_end|>"
    native_end = "<|begin_of_tool_calls_end|>"
    for header in re.finditer(r"(?m)^ {0,3}" + re.escape(native_start), text):
        if any(start <= header.start() < end for start, end in fence_ranges):
            continue
        if not isinstance(tools, list):
            raise ToolFormatError("无法验证工具定义，请按 do JSON 格式重新生成调用。")
        definitions = {}
        for tool in tools:
            if not isinstance(tool, dict):
                raise ToolFormatError("工具定义无效，请检查工具参数结构后重试。")
            function = tool.get("function") if isinstance(tool.get("function"), dict) else tool
            name = function.get("name")
            if not isinstance(name, str) or name in definitions:
                raise ToolFormatError("工具定义无效，请检查工具参数结构后重试。")
            definitions[name] = function
        position = header.end()
        calls = []
        while True:
            while position < len(text) and text[position].isspace():
                position += 1
            if text.startswith(native_end, position):
                if not calls:
                    raise ToolFormatError("工具调用为空，请按 do JSON 格式重新生成调用。")
                position += len(native_end)
                validated = _validate_tool_calls(calls, tools)
                cleaned = text[:header.start()] + text[position:]
                cleaned = re.sub(r"<\s*\|{1,2}DSML\|{1,2}\s*calls\s*>\s*</\s*\|{1,2}DSML\|{1,2}\s*calls\s*>", "", cleaned, flags=re.IGNORECASE)
                return validated, cleaned if cleaned.strip() else ""
            if not text.startswith(native_call, position):
                raise ToolFormatError("工具调用格式不完整，请按 do JSON 格式重新生成调用。")
            position += len(native_call)
            name_end = text.find(native_sep, position)
            if name_end < 0:
                raise ToolFormatError("工具调用格式不完整，请按 do JSON 格式重新生成调用。")
            name = text[position:name_end].strip()
            if name not in tool_names or name not in definitions:
                raise ToolFormatError("工具未声明，请按工具定义重新生成调用。")
            schema = definitions[name].get("parameters")
            properties = schema.get("properties") if isinstance(schema, dict) else None
            if properties is None:
                properties = {}  # 零参数工具：parameters 省略 / null / 空对象
            if not isinstance(properties, dict):
                raise ToolFormatError("无法验证工具参数，请按 do JSON 格式重新生成调用。")
            position = name_end + len(native_sep)
            arguments = {}
            while not text.startswith(native_call_end, position):
                if not text.startswith(native_key, position):
                    raise ToolFormatError("工具参数格式不完整，请按 do JSON 格式重新生成调用。")
                position += len(native_key)
                key_end = text.find(native_sep, position)
                if key_end < 0:
                    raise ToolFormatError("工具参数格式不完整，请按 do JSON 格式重新生成调用。")
                key = text[position:key_end].strip()
                field = properties.get(key)
                if not key or key in arguments or not isinstance(field, dict):
                    raise ToolFormatError("工具参数未声明或重复，请按工具定义重新生成调用。")
                position = key_end + len(native_sep)
                next_key = text.find(native_key, position)
                next_end = text.find(native_call_end, position)
                if next_end < 0:
                    raise ToolFormatError("工具调用格式不完整，请按 do JSON 格式重新生成调用。")
                value_end = min(next_key, next_end) if next_key >= 0 else next_end
                value = text[position:value_end]
                kind = field.get("type")
                if _is_string_schema(field):
                    arguments[key] = extract_cdata(sanitize_loose_cdata(value)).strip()
                elif kind in ("integer", "number", "boolean", "null", "object", "array"):
                    raw_value = extract_cdata(sanitize_loose_cdata(value)).strip()
                    try:
                        arguments[key] = json.loads(raw_value, parse_constant=_reject_json_constant)
                    except (json.JSONDecodeError, ValueError, RecursionError) as exc:
                        raise ToolFormatError("工具参数格式无效，请按工具定义重新生成调用。") from exc
                else:
                    raise ToolFormatError("工具参数类型无法确认，请按 do JSON 格式重新生成调用。")
                position = value_end
            position += len(native_call_end)
            calls.append(normalize_tool_call({"name": name, "arguments": arguments}))

    decoder = json.JSONDecoder(parse_constant=_reject_json_constant)
    json_header = re.compile(r"(?im)^ {0,3}<[|｜]\s*DSML\s*[|｜]\s*tool_calls\s*>")
    json_closer = re.compile(r"</[|｜]?\s*DSML\s*[|｜]\s*tool_calls\s*>", re.IGNORECASE)
    for header in json_header.finditer(text):
        if any(start <= header.start() < end for start, end in fence_ranges):
            continue
        position = header.end()
        while position < len(text) and text[position].isspace():
            position += 1
        if position >= len(text) or text[position] != "{":
            continue
        calls = []
        while position < len(text):
            try:
                obj, end = decoder.raw_decode(text, position)
            except (json.JSONDecodeError, ValueError, RecursionError):
                return None, text
            if (not isinstance(obj, dict) or set(obj) != {"name", "arguments"}
                    or not isinstance(obj["name"], str) or obj["name"] not in tool_names
                    or not isinstance(obj["arguments"], dict)):
                return None, text
            calls.append(normalize_tool_call(obj))
            position = end
            while position < len(text) and text[position].isspace():
                position += 1
            if position == len(text):
                cleaned = text[:header.start()]
                return _validate_tool_calls(calls, tools), cleaned if cleaned.strip() else ""
            closer = json_closer.match(text, position)
            if closer:
                cleaned = text[:header.start()] + text[closer.end():]
                return _validate_tool_calls(calls, tools), cleaned if cleaned.strip() else ""
            if text[position] != "{" or "\n" not in text[end:position]:
                return None, text

    header_pattern = re.compile(r"<\s*\|{1,2}DSML\|{1,2}\s*calls\s*>", re.IGNORECASE)
    marker_pattern = re.compile(r"<\s*\|place_holder_no_1\|\s*>", re.IGNORECASE)
    for header in header_pattern.finditer(text):
        if any(start <= header.start() < end for start, end in fence_ranges):
            continue
        if re.match(r"\s*</\s*\|{1,2}DSML\|{1,2}\s*calls\s*>", text[header.end():], re.IGNORECASE):
            continue
        first = marker_pattern.search(text, header.end())
        second = marker_pattern.search(text, first.end()) if first else None
        third = marker_pattern.search(text, second.end()) if second else None
        if not first or not second or not third or text[header.end():first.start()].strip():
            raise ToolFormatError("工具调用格式不完整，请按 do JSON 格式重新生成调用。")
        name = text[first.end():second.start()].strip()
        command = text[second.end():third.start()].strip()
        call = _placeholder_call(name, command, tool_names, tools)
        cleaned = text[:header.start()] + text[third.end():]
        return _validate_tool_calls([call], tools), cleaned if cleaned.strip() else ""

    parsed = []
    spans = []
    for match in re.finditer(r"(?im)^ {0,3}do:[ \t]*", text):
        if any(start <= match.start() < end for start, end in fence_ranges):
            continue
        if spans and match.start() < spans[-1][1]:
            continue
        try:
            obj, end = decoder.raw_decode(text, match.end())
        except (json.JSONDecodeError, ValueError, RecursionError):
            continue
        line_end = text.find("\n", end)
        if line_end < 0:
            line_end = len(text)
        if text[end:line_end].strip():
            continue
        if (not isinstance(obj, dict) or set(obj) != {"name", "arguments"}
                or not isinstance(obj["name"], str) or obj["name"] not in tool_names
                or not isinstance(obj["arguments"], dict)):
            continue
        parsed.append(normalize_tool_call(obj))
        spans.append((match.start(), line_end))
    if parsed:
        segments = []
        last = 0
        for start, end in spans:
            segments.append(text[last:start])
            last = end
        segments.append(text[last:])
        remainder = "".join(segments)
        return _validate_tool_calls(parsed, tools), remainder if remainder.strip() else ""

    # DSML XML 路径不带行首缩进限制：只要 DSML 标记落在代码区内，就整体按文本处理，避免执行示例
    for _marker in re.finditer(r"(?i)<\s*[|｜]{1,2}\s*DSML\s*[|｜]{1,2}|<invoke\b|<tool_calls?\b", text):
        if any(start <= _marker.start() < end for start, end in fence_ranges):
            return None, clean_tool_text(text)

    # 尝试 DSML 解析（按声明的参数类型保留字符串原值）
    string_keys = _declared_string_keys(tools)
    tool_calls, cleaned = parse_dsml_tool_calls(text, tool_names, string_keys)

    # 如果 DSML 解析失败，尝试 CDATA 修复后再试
    if not tool_calls:
        repaired = sanitize_loose_cdata(text)
        if repaired != text:
            tool_calls, cleaned = parse_dsml_tool_calls(repaired, tool_names, string_keys)

    if tool_calls:
        # 标准化为 OpenAI 格式
        normalized = [normalize_tool_call(tc) for tc in tool_calls]
        normalized = [tc for tc in normalized if tc and tc.get("function", {}).get("name") in tool_names]
        if normalized:
            return _validate_tool_calls(normalized, tools), clean_tool_text(cleaned)
        return None, text

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

    return text  # 不 strip — 流式分块中 \n 需要保留


# ─── 消息转换为 DeepSeek 格式 ──────────────────────────────

def convert_messages_for_deepseek(messages, tools=None, passthrough=False, tool_required=False):
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
    has_system = False
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            has_system = True
            text = str(content) if content else ""
            if tools:
                tool_text = build_dsml_tool_prompt(tools, passthrough=passthrough)
                if tool_required:
                    tool_text += "\nYou must call an available tool; do not answer without calling one."
                if tool_text:
                    if text.strip():
                        text = text + "\n\n" + tool_text
                    else:
                        text = tool_text
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
                    ids = [tc.get("id", "") if isinstance(tc, dict) else "" for tc in tool_calls]
                    if any(ids):
                        segs.append("Tool call IDs in order: " + json.dumps(ids, ensure_ascii=False))
            if segs:
                parts.append(ASST + "\n\n".join(segs) + EOS)
            elif content:
                parts.append(ASST + str(content) + EOS)
            last_role = "assistant"
        elif role == "tool":
            result = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            result = escape_session_markers(result)
            call_id = msg.get("tool_call_id", "")
            payload = json.dumps({"tool_call_id": call_id, "content": result}, ensure_ascii=False)
            parts.append(TOOL + payload + TOOL_END)
            last_role = "tool"

    # 如果没有 system 消息但有 tools，将 tool prompt 作为 system 消息插入
    if tools and not has_system:
        tool_text = build_dsml_tool_prompt(tools, passthrough=passthrough)
        if tool_required:
            tool_text += "\nYou must call an available tool; do not answer without calling one."
        if tool_text:
            parts.insert(1, SYS + tool_text + SYS_END)

    # 确保最后是 Assistant 开头
    if last_role != "assistant":
        parts.append(ASST)

    return "".join(parts)
