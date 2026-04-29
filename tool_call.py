"""
工具调用模块 — MiMo2API

将 OpenAI function calling 格式转译为 MiMo 可理解的纯文本提示词，
并从 MiMo 的纯文本响应中解析回结构化 tool_call。

6 重提取策略 + camelCase 全链路匹配 + 防御性编程。
"""

from __future__ import annotations

import re
import json
import uuid
from typing import Any, Dict, List, Optional, Tuple, Set

__all__ = [
    "build_tool_prompt",
    "get_tool_names",
    "extract_tool_call",
    "normalize_tool_call",
    "clean_tool_text",
    "convert_messages_for_deepseek",
]

# ─── 内部常量 ─────────────────────────────────────────────────

THINK_OPEN = chr(60) + "thought" + chr(62)
THINK_CLOSE = chr(60) + "/thought" + chr(62)


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
    """构建极简工具提示词，动态提取客户端 tools 的名称和描述。"""
    if not tools:
        return ""
    parts = []
    for tool in tools:
        func = _safe_get(tool, "function", default={})
        name = _safe_get(func, "name", default="")
        if not name:
            continue
        desc = _safe_get(func, "description", default="")
        short_desc = desc.split("\n")[0].strip()
        if short_desc:
            parts.append(f"{name}({short_desc})")
        else:
            parts.append(name)
    if not parts:
        return ""
    tool_list = ", ".join(parts)
    return (
        "你有工具可以使用: " + tool_list + "\n"
        "当你需要执行操作时，必须输出以下格式（不要用其他任何格式）：\n"
        "TOOL_CALL: 工具名(参数名=\"参数值\")\n\n"
        "示例：\n"
        "TOOL_CALL: terminal(command=\"ls -la\")\n"
        "TOOL_CALL: terminal(command=\"date\")\n\n"
        "规则：\n"
        "1. 需要执行操作时，只输出 TOOL_CALL 行，不要加任何解释文字\n"
        "2. 拿到结果后用自然语言回复用户\n"
        "3. 不要输出 TOOL_RESULT、[SYS]、<execute_operation> 等标签"
    )


# ─── 提取工具名列表 ──────────────────────────────────────────

def get_tool_names(tools: List[Dict[str, Any]]) -> List[str]:
    """从 tools 列表提取所有 function name。"""
    names = []
    for tool in tools or []:
        func = _safe_get(tool, "function", default={})
        name = _safe_get(func, "name", default=None)
        if name:
            names.append(str(name))
    return names


# ─── camelCase 工具名解析 ────────────────────────────────────

def _resolve_tool_name(name: str, tool_names: List[str]) -> Optional[str]:
    """将任意形式的工具名解析为规范的 snake_case。

    4 级匹配：
      1. 直接匹配 name in tool_names
      2. 大小写不敏感匹配
      3. camelCase -> snake_case 转换（getTimeInfo -> get_time_info）
      4. 转换后大小写不敏感匹配
    """
    if not name or not tool_names:
        return None

    # 1. 直接匹配
    if name in tool_names:
        return name

    # 2. 大小写不敏感
    name_lower = name.lower()
    for tn in tool_names:
        if tn.lower() == name_lower:
            return tn

    # 3. camelCase -> snake_case
    snake = re.sub(r'(?<=[a-z0-9])([A-Z])', r'_\1', name).lower()
    if snake in tool_names:
        return snake

    # 4. snake_case 大小写不敏感
    for tn in tool_names:
        if tn.lower() == snake:
            return tn

    return None


# ─── 主入口：从文本中提取工具调用 ──────────────────────────

def extract_tool_call(
    text: str, tool_names: List[str]
) -> Tuple[Optional[List[Dict[str, Any]]], str]:
    """从 MiMo 输出文本中提取工具调用。

    7 重策略（按优先级）：
      1. TOOL_CALL: name(args)  — 标准格式
      2. JSON {"name":"x","arguments":{...}} — 内嵌 JSON
      3. <tool_call> XML — MiMo 原生格式
      4. <function_call> JSON+XML
      4.5. <function=name> 裸 XML 标签（DeepSeek 常用格式）
      5. [调用工具: NAME] — 中文格式
      6. name(args) 自由文本 — 低优先级

    Returns:
        (tool_calls_list_or_None, cleaned_text)
    """
    if not text or not tool_names:
        return None, clean_tool_text(text) if text else text

    text = text.replace("\x00", "")

    # 策略1: TOOL_CALL: name(args)
    tc = _extract_tool_call_pattern(text, tool_names)
    if tc:
        return tc, clean_tool_text(text)

    # 策略2: JSON 格式
    tc = _extract_json_tool_call(text, tool_names)
    if tc:
        return tc, clean_tool_text(text)

    # 策略3: <tool_call> XML
    tc = _extract_xml_tool_call(text, tool_names)
    if tc:
        return tc, clean_tool_text(text)

    # 策略4: <function_call> JSON+XML
    tc = _extract_function_call_json(text, tool_names)
    if tc:
        return tc, clean_tool_text(text)

    # 策略4.5: 裸 <function=name> XML 标签（无包裹）
    tc = _extract_bare_function_xml(text, tool_names)
    if tc:
        return tc, clean_tool_text(text)

    # 策略5: [调用工具: NAME] 中文格式
    tc = _extract_chinese_tool_call(text, tool_names)
    if tc:
        return tc, clean_tool_text(text)

    # 策略5.5: <execute_operation> XML (DeepSeek 自由格式)
    tc = _extract_execute_operation_xml(text, tool_names)
    if tc:
        return tc, clean_tool_text(text)

    # 策略6: 自由文本 name(args)（低优先级）
    tc = _extract_freeform_tool_call(text, tool_names)
    if tc:
        return tc, clean_tool_text(text)

    # No tool call found — still clean any echoed prompt tags
    return None, clean_tool_text(text)


# ─── 策略1: TOOL_CALL: name(...) ────────────────────────────

def _extract_tool_call_pattern(
    text: str, tool_names: List[str]
) -> Optional[List[Dict[str, Any]]]:
    """匹配 TOOL_CALL: name(args) 或 TOOL_CALL: name{...}"""
    results = []
    idx = 0
    while idx < len(text):
        m = re.search(
            r"(?:^|\n)\s*[A-Z_]*CALL:\s*(\w+)\s*\(",
            text[idx:], re.IGNORECASE
        )
        if not m:
            break

        fname = m.group(1)
        if _is_inside_think(text, idx + m.start()):
            idx += m.end()
            continue

        paren = idx + m.end() - 1
        depth = 1
        in_s = False
        esc = False
        end = -1
        for i in range(paren + 1, len(text)):
            c = text[i]
            if esc:
                esc = False
                continue
            if c == "\\" and in_s:
                esc = True
                continue
            if c == '"':
                in_s = not in_s
                continue
            if in_s:
                continue
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0:
                    end = i
                    break

        if end == -1:
            break

        args_raw = text[paren + 1:end]
        args = _parse_function_args(args_raw)

        resolved = _resolve_tool_name(fname, tool_names)
        if resolved:
            results.append({"name": resolved, "arguments": args})

        idx = end + 1

    if results:
        return [normalize_tool_call(tc) for tc in results if normalize_tool_call(tc)]
    return None


# ─── 策略2: JSON {"name":"x","arguments":{...}} ─────────────

def _extract_json_tool_call(
    text: str, tool_names: List[str]
) -> Optional[List[Dict[str, Any]]]:
    """从文本中提取 JSON 格式的工具调用。"""
    # 先尝试用 _find_balanced_json 找到所有可能的 JSON 对象
    start = 0
    while True:
        brace = text.find("{", start)
        if brace == -1:
            break

        if _is_inside_think(text, brace):
            start = brace + 1
            continue

        js = _find_balanced_json(text, brace)
        if not js:
            start = brace + 1
            continue

        try:
            obj = json.loads(js)
        except (json.JSONDecodeError, ValueError):
            start = brace + 1
            continue

        # 检查是否是工具调用 JSON
        name = obj.get("name") or _safe_get(obj.get("function", {}), "name")
        resolved = _resolve_tool_name(name, tool_names) if name else None
        if resolved:
            args = obj.get("arguments") or obj.get("parameters") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (json.JSONDecodeError, ValueError):
                    pass
            tc = normalize_tool_call({"name": resolved, "arguments": args})
            if tc:
                return [tc]

        start = text.find("}", brace + len(js)) + 1


# ─── 策略3: name(args) 自由文本 ─────────────────────────────

def _extract_freeform_tool_call(
    text: str, tool_names: List[str]
) -> Optional[List[Dict[str, Any]]]:
    """匹配 name(args) 模式（无 TOOL_CALL 前缀）。

    支持 snake_case 和 camelCase 变体，大小写不敏感。
    """
    for name in tool_names:
        # 构建工具名变体
        variants = [re.escape(name)]
        if '_' in name:
            # 生成 camelCase 变体
            camel = re.sub(r'_([a-z])', lambda m: m.group(1).upper(), name)
            variants.append(re.escape(camel))

        escaped = '|'.join(variants)
        # word boundary 前缀避免句子中间误匹配
        pat = rf"(?<!\w)({escaped})\s*\((.*?)\)"
        for m in re.finditer(pat, text, re.DOTALL | re.IGNORECASE):
            if _is_inside_think(text, m.start()):
                continue
            resolved = _resolve_tool_name(m.group(1).strip(), tool_names)
            if resolved:
                args_raw = m.group(2).strip()
                args = _parse_function_args(args_raw)
                tc = normalize_tool_call({"name": resolved, "arguments": args})
                if tc:
                    return [tc]

    return None


# ─── 策略4: <tool_call> XML（MiMo 原生） ───────────────────

def _extract_xml_tool_call(
    text: str, tool_names: List[str]
) -> Optional[List[Dict[str, Any]]]:
    """匹配 <tool_call><function=NAME><parameter=K>V</parameter>...</function></tool_call>"""
    tc_pattern = r"<tool_call>(.*?)</tool_call>"
    m = re.search(tc_pattern, text, re.DOTALL | re.IGNORECASE)
    if not m:
        return None

    if _is_inside_think(text, m.start()):
        return None

    inner = m.group(1)

    func_pattern = r"<function=(\w+)>(.*?)</function>"
    fm = re.search(func_pattern, inner, re.DOTALL | re.IGNORECASE)
    if not fm:
        return None

    name = fm.group(1).strip()
    resolved = _resolve_tool_name(name, tool_names)
    if not resolved:
        return None

    func_body = fm.group(2)

    # 提取 <parameter=KEY>VALUE</parameter>
    args = {}
    param_pattern = r"<parameter=(\w+)>(.*?)</parameter>"
    for pm in re.finditer(param_pattern, func_body, re.DOTALL | re.IGNORECASE):
        key = pm.group(1).strip()
        val = pm.group(2).strip()
        args[key] = _auto_type(val)

    tc = normalize_tool_call({"name": resolved, "arguments": args})
    return [tc] if tc else None


# ─── 策略5: <function_call> JSON+XML ────────────────────────

def _extract_function_call_json(
    text: str, tool_names: List[str]
) -> Optional[List[Dict[str, Any]]]:
    """匹配 <function_call>{"name":"x","arguments":{...}}</function_call>"""
    fc_pat = r"<function_calls?>(.*?)</function_calls?>"
    fc_m = re.search(fc_pat, text, re.DOTALL)
    if not fc_m:
        return None

    if _is_inside_think(text, fc_m.start()):
        return None

    inner = fc_m.group(1)
    for block in re.split(r"</function_call>", inner):
        if not block.strip():
            continue
        block = re.sub(r"^.*?<function_call>", "", block, flags=re.DOTALL).strip()
        if not block:
            continue
        js_start = block.find("{")
        if js_start == -1:
            continue
        js = _find_balanced_json(block, js_start)
        if js:
            try:
                data = json.loads(js)
                name = data.get("name", "")
                resolved = _resolve_tool_name(name, tool_names) if name else None
                if resolved:
                    args = data.get("arguments", {})
                    tc = normalize_tool_call({"name": resolved, "arguments": args})
                    if tc:
                        return [tc]
            except (json.JSONDecodeError, AttributeError):
                pass

    return None


# --- Strategy 4.5: bare XML function tags ---
def _extract_bare_function_xml(text, tool_names):
    """Match bare XML function tags without  wrapper.

    Handles: text TAG_NAME(args) or text TAG_NAME>content TAG_NAME
    """
    if not tool_names:
        return None

    # Strategy A: match TAG(args) bare format
    for name in tool_names:
        variants = [re.escape(name)]
        if "_" in name:
            camel = re.sub(r"_([a-z])", lambda m: m.group(1).upper(), name)
            variants.append(re.escape(camel))
        escaped = "|".join(variants)
        pat = r"(?<!\w)(" + escaped + r")\s*\(([^)]*)\)"
        for m in re.finditer(pat, text, re.DOTALL | re.IGNORECASE):
            if _is_inside_think(text, m.start()):
                continue
            prefix = text[:m.start()].rstrip()
            if prefix.endswith("TOOL_CALL:"):
                continue
            resolved = _resolve_tool_name(m.group(1).strip(), tool_names)
            if not resolved:
                continue
            args_raw = m.group(2).strip()
            args = _parse_function_args(args_raw) if args_raw else {}
            tc = normalize_tool_call({"name": resolved, "arguments": args})
            if tc:
                return [tc]

    # Strategy B: match bare XML tags without  wrapper
    # Build XML patterns using chr() to avoid parser issues
    _L = chr(60)  # <
    _G = chr(62)  # >
    _SL = chr(47)  # /
    _FNAME = "function"
    _FPARAM = "parameter"

    for name in tool_names:
        variants = [name]
        if "_" in name:
            camel = re.sub(r"_([a-z])", lambda m: m.group(1).upper(), name)
            variants.append(camel)
        for vname in variants:
            open_tag = _L + _FNAME + chr(61) + vname + _G
            close_tag = _L + _SL + _FNAME + _G
            full_pat = re.escape(open_tag) + r"(.*?)" + re.escape(close_tag)
            for m in re.finditer(full_pat, text, re.DOTALL | re.IGNORECASE):
                if _is_inside_think(text, m.start()):
                    continue
                resolved = _resolve_tool_name(name, tool_names)
                if not resolved:
                    continue
                inner = m.group(1)
                args = {}
                pp = re.escape(_L + _FPARAM + chr(61)) + r"(\w+)" + re.escape(_L + _SL + _FPARAM + _G) + r"(.*?)" + re.escape(_L + _SL + _FPARAM + _G)
                for pm in re.finditer(pp, inner, re.DOTALL | re.IGNORECASE):
                    args[pm.group(1).strip()] = _auto_type(pm.group(2).strip())
                tc = normalize_tool_call({"name": resolved, "arguments": args})
                if tc:
                    return [tc]

    return None

# ─── 策略6: [调用工具: NAME] 中文格式 ───────────────────────

def _extract_chinese_tool_call(
    text: str, tool_names: List[str]
) -> Optional[List[Dict[str, Any]]]:
    """匹配 [调用工具: NAME] 中文格式（模型从历史中学到的格式）。"""
    pat = r"\[\u8c03\u7528\u5de5\u5177:\s*(\w+(?:,\s*\w+)*)\]"
    m = re.search(pat, text)
    if not m:
        return None

    if _is_inside_think(text, m.start()):
        return None

    names = [n.strip() for n in m.group(1).split(",")]
    found_name = None
    for n in names:
        resolved = _resolve_tool_name(n, tool_names)
        if resolved:
            found_name = resolved
            break

    if not found_name:
        return None

    after = text[m.end():].strip()
    args = {}

    if after:
        if after.startswith("{"):
            js = _find_balanced_json(after, 0)
            if js:
                try:
                    args = json.loads(js)
                except json.JSONDecodeError:
                    pass
        else:
            first_line = after.split("\n")[0].strip()
            if first_line and not first_line.startswith("["):
                args = {"input": first_line}

    tc = normalize_tool_call({"name": found_name, "arguments": args})
    return [tc] if tc else None


# ─── 策略5.5: <execute_operation> XML (DeepSeek 自由格式) ──────

def _extract_execute_operation_xml(
    text: str, tool_names: List[str]
) -> Optional[List[Dict[str, Any]]]:
    """匹配 DeepSeek 输出的 XML 格式:
    <execute_operation>
      <operation>描述</operation>
      <command>date '+%Y'</command>
    </execute_operation>
    """
    m = re.search(r"<execute_operation>(.*?)</execute_operation>", text, re.DOTALL)
    if not m:
        return None
    if _is_inside_think(text, m.start()):
        return None

    inner = m.group(1)

    # 提取 <command> 内容
    cmd_m = re.search(r"<command>(.*?)</command>", inner, re.DOTALL)
    if not cmd_m:
        return None
    command = cmd_m.group(1).strip()
    if not command:
        return None

    # 尝试匹配到已知工具名（通常就是 terminal）
    resolved = None
    for tn in tool_names:
        if tn.lower() in ("terminal", "shell", "exec", "run_command", "execute"):
            resolved = tn
            break
    if not resolved and tool_names:
        resolved = tool_names[0]  # fallback to first tool

    if not resolved:
        return None

    tc = normalize_tool_call({"name": resolved, "arguments": {"command": command}})
    return [tc] if tc else None


# ─── 策略7: 纯文本中的 shell 命令（兜底） ──────────────────────

# 常见 shell 命令前缀（优先匹配这些）
_SHELL_CMD_RE = re.compile(
    r"^\s*(?:"
    r"ls|cd|pwd|cat|head|tail|grep|find|wc|du|df|mkdir|rm|cp|mv|chmod|chown"
    r"|echo|date|whoami|hostname|uname|env|export|source"
    r"|ps|top|kill|nohup|bg|fg|jobs"
    r"|git|npm|node|python|python3|pip|curl|wget"
    r"|tar|zip|unzip|gzip|gunzip"
    r"|ssh|scp|rsync"
    r"|apt|pkg|pacman|yum|brew"
    r"|docker|kubectl|systemctl"
    r"|sed|awk|sort|uniq|cut|tr|xargs"
    r")\b",
    re.MULTILINE,
)

def _extract_plain_command(
    text: str, tool_names: List[str]
) -> Optional[List[Dict[str, Any]]]:
    """Fallback: detect plain shell commands in text when model didn't use
    any standard tool call format.

    Only activates when:
    - Text is short (< 300 chars, no long explanation)
    - Contains a line starting with a known shell command
    - Has a terminal-like tool available
    """
    if not tool_names or len(text) > 300:
        return None

    # Find first shell command line
    for line in text.split("\n"):
        line = line.strip().strip("`").strip()
        if not line:
            continue
        # Strip markdown code block markers
        if line.startswith("```"):
            continue
        m = _SHELL_CMD_RE.match(line)
        if m:
            # Pick the terminal-like tool
            resolved = None
            for tn in tool_names:
                if tn.lower() in ("terminal", "shell", "exec", "run_command", "execute"):
                    resolved = tn
                    break
            if not resolved:
                resolved = tool_names[0]

            tc = normalize_tool_call({"name": resolved, "arguments": {"command": line}})
            return [tc] if tc else None

    return None


# ─── 标准化工具调用为 OpenAI 格式 ──────────────────────────

def _flatten_tool_args(args: Any) -> Any:
    """Flatten malformed nested arguments from DeepSeek.

    DeepSeek sometimes wraps real args inside {"input": "{...json...}"}.
    Unwrap them so tools receive the correct flat dict.
    """
    if not isinstance(args, dict):
        return args
    # Single "input" key with dict-like string value
    if len(args) == 1 and "input" in args:
        inner = args["input"]
        if isinstance(inner, str):
            inner = inner.strip()
            if inner.startswith("{"):
                # Try parsing directly, then with single→double quote swap
                for attempt in [inner, inner.replace("'", '"')]:
                    try:
                        return json.loads(attempt)
                    except (json.JSONDecodeError, ValueError):
                        continue
            # Not JSON — use as command directly
            return {"command": inner} if inner else args
    # Check all values: if any is a string containing JSON dict, unwrap
    for k, v in list(args.items()):
        if isinstance(v, str) and v.strip().startswith("{"):
            try:
                args[k] = json.loads(v)
            except (json.JSONDecodeError, ValueError):
                pass
    return args


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
            # 确保 arguments 是字符串
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
        args = _flatten_tool_args(args)
        args = json.dumps(args, ensure_ascii=False)

    return {
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": args,
        },
    }


# ─── 清理工具文本 ──────────────────────────────────────────

def clean_tool_text(text: str) -> str:
    """清理文本中的工具调用残留痕迹。

    移除所有已知格式的标签，保留纯自然语言内容。
    """
    if not text:
        return text

    # TOOL_CALL: xxx 行
    text = re.sub(r"[A-Z_]*CALL:.*$", "", text, flags=re.MULTILINE | re.IGNORECASE)
    # [TOOL_RESULT] / [SYS] echo cleanup (model may echo prompt tags)
    text = re.sub(r"\[TOOL_RESULT\].*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\[SYS\]\s*工具已执行完毕.*?(?=\n\[|\Z)", "", text, flags=re.DOTALL)
    # <execute_operation> XML echo
    text = re.sub(r"<execute_operation>.*?</execute_operation>", "", text, flags=re.DOTALL)
    # DeepSeek search citation tags: [citation:1][citation:2]
    text = re.sub(r"\[citation:\d+\]", "", text)
    # TOOL_CALL: name(...) 内联
    text = re.sub(
        r"TOOL_CALL:\s*\w+\s*\([^)]*(?:\([^)]*\)[^)]*)*\)",
        "", text, flags=re.IGNORECASE
    )
    # <function_call> / <function_calls> 标签
    text = re.sub(r"</?function_calls?>", "", text)
    # <tool_call>...</tool_call>
    text = re.sub(
        r"<tool_call>.*?</tool_call>", "",
        text, flags=re.DOTALL | re.IGNORECASE
    )
    # <function=xxx>...</function>
    text = re.sub(
        r"<function=\w+>.*?</function>", "",
        text, flags=re.DOTALL | re.IGNORECASE
    )
    # <parameter=xxx>...</parameter>
    text = re.sub(
        r"<parameter=\w+>.*?</parameter>", "",
        text, flags=re.DOTALL | re.IGNORECASE
    )
    # python调用工具(xxx) 残留
    text = re.sub(r"</?function=\w+>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<parameter=\w+>", "", text)
    text = re.sub(r"</parameter>", "", text)
    # [调用工具: xxx] 中文格式
    text = re.sub(
        r"\[\s*\u8c03\u7528\u5de5\u5177\s*:\s*\w+(?:\s*,\s*\w+)*\s*\].*",
        "", text, flags=re.MULTILINE
    )
    # JSON tool_call 块
    text = re.sub(
        r"```(?:json)?\s*\n?\s*\{.*?\"tool_call\".*?\}\s*\n?\s*```",
        "", text, flags=re.DOTALL
    )
    # 空代码块
    text = re.sub(r"```\w*\s*\n?\s*```", "", text)
    # 多余空行
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# ═══════════════════════════════════════════════════════════
# 内部辅助函数
# ═══════════════════════════════════════════════════════════

def _find_balanced_json(text: str, start: int) -> str:
    """从 start 位置查找配对的 JSON {}，处理字符串转义。"""
    if start >= len(text) or text[start] != "{":
        return ""

    depth = 0
    in_str = False
    esc = False

    for i in range(start, len(text)):
        c = text[i]
        if esc:
            esc = False
            continue
        if c == "\\" and in_str:
            esc = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    return ""


def _parse_function_args(raw: str) -> Dict[str, Any]:
    """解析函数参数字符串到 dict。

    支持格式:
      key="value", key2=123
      key=value
      {"json": "object"}
    """
    raw = raw.strip()
    if not raw:
        return {}

    # 已经是 JSON 对象
    if raw.startswith("{") and raw.endswith("}"):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            pass

    # key=value 格式，用智能分割处理嵌套
    args = {}
    for pair in _smart_split(raw, ","):
        pair = pair.strip()
        if not pair or "=" not in pair:
            continue
        k, v = pair.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k:
            args[k] = _auto_type(v)

    if args:
        return args

    # 无法解析，返回原文本作为 input
    return {"input": raw}


def _smart_split(text: str, sep: str) -> List[str]:
    """智能分割字符串，正确处理括号嵌套和引号。"""
    parts = []
    current = []
    dp = db = dbr = 0  # 括号深度
    in_str = False
    esc = False

    for ch in text:
        if esc:
            current.append(ch)
            esc = False
            continue
        if ch == "\\" and in_str:
            current.append(ch)
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            current.append(ch)
            continue
        if in_str:
            current.append(ch)
            continue
        if ch == "(":
            dp += 1
        elif ch == ")":
            dp -= 1
        elif ch == "[":
            db += 1
        elif ch == "]":
            db -= 1
        elif ch == "{":
            dbr += 1
        elif ch == "}":
            dbr -= 1
        elif ch == sep and dp == 0 and db == 0 and dbr == 0:
            parts.append("".join(current).strip())
            current = []
            continue
        current.append(ch)

    if current:
        parts.append("".join(current).strip())

    return parts


def _auto_type(val: str) -> Any:
    """自动推断值类型。"""
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


def _is_inside_think(text: str, pos: int) -> bool:
    """检查 pos 是否在 <think>...</think> 块内部。"""
    sf = 0
    while True:
        s = text.find(THINK_OPEN, sf)
        if s == -1:
            break
        e = text.find(THINK_CLOSE, s + 7)
        if e == -1:
            return pos >= s
        if s <= pos < e + 8:
            return True
        sf = e + 8
    return False


def convert_messages_for_deepseek(messages, tools=None):
    out = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            out.append("[SYS]\n" + str(content) + "\n")
        elif role == "user":
            if isinstance(content, list):
                text = " ".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
            else:
                text = str(content)
            out.append("[USER]\n" + text + "\n")
        elif role == "assistant":
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                # Skip user-facing text like "Tool" / "Tool 'terminal' called..." —
                # these confuse DeepSeek into thinking the call already succeeded
                tc_lines = []
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    name = fn.get("name", "")
                    args_str = fn.get("arguments", "{}")
                    try:
                        args = json.loads(args_str) if isinstance(args_str, str) else args_str
                        # Flatten: if args has a single "input" key containing a
                        # dict-like string, unwrap it
                        if isinstance(args, dict) and len(args) == 1 and "input" in args:
                            inner = args["input"]
                            if isinstance(inner, str):
                                try:
                                    args = json.loads(inner.replace("'", '"'))
                                except (json.JSONDecodeError, ValueError):
                                    args = {"command": inner}
                        kv = ", ".join(str(k) + "=" + str(v) for k, v in args.items())
                    except (json.JSONDecodeError, AttributeError):
                        kv = args_str
                    tc_lines.append("TOOL_CALL: " + name + "(" + kv + ")")
                out.append("[ASST]\n" + "\n".join(tc_lines) + "\n")
            elif content:
                out.append("[ASST]\n" + str(content) + "\n")
        elif role == "tool":
            # Parse tool result: extract actual output from JSON envelope
            # to prevent DeepSeek from echoing raw [TOOL_RESULT] format
            result = str(content) if content else ""
            if result:
                actual_output = result[:500]
                try:
                    rd = json.loads(result)
                    if isinstance(rd, dict):
                        parts = []
                        for k in ("output", "error", "result", "content"):
                            v = rd.get(k)
                            if v is not None and str(v).strip():
                                parts.append(str(v).strip())
                        if parts:
                            actual_output = "\n".join(parts)
                except (json.JSONDecodeError, ValueError):
                    pass
                # Use [SYS] format — avoid [TOOL_RESULT] which the model echoes
                out.append("[SYS]\n工具已执行完毕，以下是输出:\n" + actual_output + "\n")
    return "\n".join(out)