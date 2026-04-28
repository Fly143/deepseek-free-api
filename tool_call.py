# -*- coding: utf-8 -*-
"""工具调用模块 — DeepSeek Free API"""

from __future__ import annotations
import re, json, uuid
from typing import Any, Dict, List, Optional, Set, Tuple


def _safe_get(d, key, default=None):
    if d is None: return default
    if isinstance(d, dict): return d.get(key, default)
    return getattr(d, key, default)


THINK_OPEN = chr(60) + "thought" + chr(62)
THINK_CLOSE = chr(60) + "/thought" + chr(62)


def _is_inside_think(text, pos):
    sf = 0
    while True:
        s = text.find(THINK_OPEN, sf)
        if s == -1: break
        e = text.find(THINK_CLOSE, s + 7)
        if e == -1: return pos >= s
        if s <= pos < e + 8: return True
        sf = e + 8
    return False


def _find_balanced_json(text, start):
    if start >= len(text) or text[start] != "{": return ""
    depth = 0; in_str = False; esc = False
    for i in range(start, len(text)):
        c = text[i]
        if esc: esc = False; continue
        if c == chr(92) and in_str: esc = True; continue
        if c == chr(34): in_str = not in_str; continue
        if in_str: continue
        if c == "{": depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0: return text[start:i+1]
    return ""


def build_tool_prompt(tools):
    if not tools: return ""
    tl = []
    for tool in tools:
        func = _safe_get(tool, "function", default={})
        name = _safe_get(func, "name", default="unknown")
        desc = _safe_get(func, "description", default="")
        params = _safe_get(func, "parameters", default=None)
        pt = ""
        if params and isinstance(params, dict):
            req = set(params.get("required") or [])
            props = params.get("properties") or {}
            for pn, pd in props.items():
                if not isinstance(pd, dict): pd = {}
                t = pd.get("type", "string")
                d2 = pd.get("description", "")
                m = "*" if pn in req else ""
                s = f" ({d2})" if d2 else ""
                pt += chr(10) + "    " + pn + m + "(" + t + ")" + s
        tl.append("- " + name + ": " + desc + pt)
    NL = chr(10)
    L = []
    L.append("## 可用工具")
    L.append("当用户请求需要调用工具时，你必须在回复中包含一行 TOOL_CALL 指令。")
    L.append("格式: TOOL_CALL: 工具名(参数1=值1, 参数2=\"值2\")")
    L.append("")
    L.append("规则:")
    L.append("- TOOL_CALL 必须在单独一行")
    L.append("- 括号内参数用逗号分隔，字符串值用引号包裹")
    L.append("- 整数/布尔值不要加引号")
    L.append("- 如果不需要调用工具，直接回答，不输出 TOOL_CALL")
    L.append("")
    L.append("示例:")
    L.append('TOOL_CALL: get_weather(city="北京")')
    L.append('TOOL_CALL: search_web(query="latest AI news", page=1)')
    L.append('TOOL_CALL: send_email(to="user@example.com", subject="通知", body="你好")')
    L.append("")
    L.append("可用工具列表:")
    L.extend(tl)
    return NL.join(L)


def get_tool_names(tools):
    names = set()
    for tool in tools or []:
        func = _safe_get(tool, "function", default={})
        name = _safe_get(func, "name", default=None)
        if name: names.add(str(name))
    return names


def _parse_function_args(raw):
    raw = raw.strip()
    if not raw: return {}
    args = {}
    for pair in _smart_split(raw, ","):
        pair = pair.strip()
        if not pair or "=" not in pair: continue
        k, v = pair.split("=", 1)
        k = k.strip()
        v = v.strip().strip(chr(34)).strip(chr(39))
        if k: args[k] = _auto_type(v)
    return args


def _smart_split(text, sep):
    parts = []; current = []
    dp = db = dbr = 0; in_str = False; esc = False
    for ch in text:
        if esc: current.append(ch); esc = False; continue
        if ch == chr(92) and in_str: current.append(ch); esc = True; continue
        if ch == chr(34): in_str = not in_str; current.append(ch); continue
        if in_str: current.append(ch); continue
        if ch == "(": dp += 1
        elif ch == ")": dp -= 1
        elif ch == "[": db += 1
        elif ch == "]": db -= 1
        elif ch == "{": dbr += 1
        elif ch == "}": dbr -= 1
        elif ch == sep and dp == 0 and db == 0 and dbr == 0:
            parts.append("".join(current).strip()); current = []; continue
        current.append(ch)
    if current: parts.append("".join(current).strip())
    return parts


def _auto_type(val):
    if val.lower() == "true": return True
    if val.lower() == "false": return False
    if val.lower() in ("null", "none"): return None
    try: return int(val)
    except ValueError: pass
    try: return float(val)
    except ValueError: pass
    return val


def normalize_tool_call(tc):
    if isinstance(tc, list): tc = tc[0] if tc else {}
    if not isinstance(tc, dict): return None
    name = tc.get("name", "")
    if not name: return None
    args = tc.get("arguments", {})
    args_str = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
    return {
        "id": "call_" + uuid.uuid4().hex[:24],
        "type": "function",
        "function": {"name": name, "arguments": args_str},
    }


def extract_tool_call(content, tool_names=None):
    if not content: return None, content or ""
    content = content.replace("\x00", "")

    # Strategy 1: TOOL_CALL: name(args)
    results = []
    idx = 0
    while idx < len(content):
        m = re.search(r"(?:^|\n)\s*TOOL_CALL:\s*(\w+)\s*\(", content[idx:], re.IGNORECASE)
        if not m: break
        sp = idx + m.start()
        if _is_inside_think(content, sp):
            idx += m.end(); continue
        fname = m.group(1)
        paren = idx + m.end() - 1
        depth = 1; in_s = False; esc2 = False; end = -1
        for i in range(paren + 1, len(content)):
            c = content[i]
            if esc2: esc2 = False; continue
            if c == chr(92) and in_s: esc2 = True; continue
            if c == chr(34): in_s = not in_s; continue
            if in_s: continue
            if c == "(": depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0: end = i; break
        if end == -1: break
        args = _parse_function_args(content[paren+1:end])
        if not tool_names or fname in tool_names:
            results.append({"name": fname, "arguments": args})
        idx = end + 1

    if results:
        normed = [r for r in (normalize_tool_call(tc) for tc in results) if r]
        if normed: return normed, clean_tool_text(content)

    # Strategy 2: JSON code blocks
    for cb in re.finditer(r"```(?:json)?\s*\n?(.*?)\n?\s*```", content, re.DOTALL):
        bt = cb.group(1).strip()
        if not bt.startswith("{"): continue
        js = _find_balanced_json(bt, 0)
        if not js: continue
        if _is_inside_think(content, cb.start()): continue
        try:
            data = json.loads(js)
            if "tool_call" in data:
                result = normalize_tool_call(data["tool_call"])
                if result and (not tool_names or result["function"]["name"] in tool_names):
                    return [result], clean_tool_text(content)
        except (json.JSONDecodeError, AttributeError): pass

    # Strategy 3: <function=name> XML tags
    fm = re.search(r"<function=(\\w+)>", content)
    if fm and not _is_inside_think(content, fm.start()):
        fname = fm.group(1)
        if not tool_names or fname in tool_names:
            args = {}
            pat = r"<parameter=(\\w+)>(.*?)</parameter>"
            for pm in re.finditer(pat, content, re.DOTALL):
                args[pm.group(1)] = pm.group(2).strip()
            if not args:
                ja = re.search(r"<function=" + re.escape(fname) + ">\\s*(\\{.*?\\})", content, re.DOTALL)
                if ja:
                    try: args = json.loads(ja.group(1))
                    except json.JSONDecodeError: pass
            result = normalize_tool_call({"name": fname, "arguments": args})
            if result: return [result], clean_tool_text(content)

    # Strategy 4: [调用工具: NAME] Chinese bracket format (cross-contamination from MiMo API history)
    cm = re.search(r"\[调用工具:\s*(\w+)\]", content)
    if cm and not _is_inside_think(content, cm.start()):
        fname = cm.group(1)
        if not tool_names or fname in tool_names:
            args = {}
            after = content[cm.end():].strip()
            if after:
                if after.startswith("{"):
                    js = _find_balanced_json(after, 0)
                    if js:
                        try: args = json.loads(js)
                        except json.JSONDecodeError: pass
                else:
                    first_line = after.split("\n")[0].strip()
                    if first_line:
                        args = {"input": first_line}
            result = normalize_tool_call({"name": fname, "arguments": args})
            if result: return [result], clean_tool_text(content)

    return None, content


def clean_tool_text(content):
    content = re.sub(r"^[ \t]*TOOL_CALL:.*$", "", content, flags=re.MULTILINE | re.IGNORECASE)
    content = re.sub(r"TOOL_CALL:\s*\w+\s*\([^)]*(?:\([^)]*\)[^)]*)*\)", "", content, flags=re.IGNORECASE)
    content = re.sub(r"<function=\w+>.*?</function>", "", content, flags=re.DOTALL)
    content = re.sub(r"</?function[^>]*>", "", content)
    content = re.sub(r"<parameter=\w+>.*?</parameter>", "", content, flags=re.DOTALL)
    content = re.sub(r"<parameter=\w+>", "", content)
    content = re.sub(r"</parameter>", "", content)
    content = re.sub(r"```(?:json)?\s*\n?\s*\{.*?\"tool_call\".*?\}\s*\n?\s*```", "", content, flags=re.DOTALL)
    content = re.sub(r"```\w*\s*\n?\s*```", "", content)
    content = re.sub(r"^\s*webSearch\b", "", content, flags=re.MULTILINE)
    content = re.sub(r"\[\s*调用工具\s*:\s*\w+(?:\s*,\s*\w+)*\s*\].*", "", content, flags=re.MULTILINE)
    content = re.sub(r"\n{3,}", "\n\n", content)
    return content.strip()


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
                tc_lines = []
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    name = fn.get("name", "")
                    args_str = fn.get("arguments", "{}")
                    try:
                        args = json.loads(args_str) if isinstance(args_str, str) else args_str
                        kv = ", ".join(str(k) + "=" + str(v) for k, v in args.items())
                    except (json.JSONDecodeError, AttributeError):
                        kv = args_str
                    tc_lines.append("TOOL_CALL: " + name + "(" + kv + ")")
                text_content = content if content else ""
                assistant_text = str(text_content) + "\n" if text_content else ""
                assistant_text += "\n".join(tc_lines)
                out.append("[ASST]\n" + assistant_text + "\n")
            elif content:
                out.append("[ASST]\n" + str(content) + "\n")
        elif role == "tool":
            # Skip tool results — the assistant's summary already follows,
            # and including raw JSON causes DeepSeek to echo/duplicate it.
            pass
    return "\n".join(out)