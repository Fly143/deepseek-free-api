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
_DSML_NS = set("| \t\r\n")
_DSML_NS.add(chr(0xff5c))  # full-width pipe
_DSML_HYPHENATED = {"dsml-tool_calls":"tool_calls","dsml-invoke":"invoke","dsml-parameter":"parameter"}


# ─── DSML 前缀剥离 ──────────────────────────────────────

def strip_dsml_markup(text: str) -> str:
    """去除 DSML 前缀，支持噪声容错和围栏代码块跳过。"""
    if not text: return text
    r = []; i = 0; n = len(text)
    while i < n:
        if text[i:].startswith(_CDATA_OPEN):
            cl = text.find(_CDATA_CLOSE, i + len(_CDATA_OPEN))
            if cl == -1: r.append(text[i:]); break
            r.append(text[i:cl + len(_CDATA_CLOSE)]); i = cl + len(_CDATA_CLOSE); continue
        i2, sk = _skip_fence_dsml(text, i)
        if sk: r.append(sk); i = i2; continue
        ch = text[i]
        if ch != '<': r.append(ch); i += 1; continue
        end = text.find('>', i)
        if end == -1: r.append(text[i:]); break
        inner = text[i + 1 : end]
        cl = inner.startswith('/')
        rest = inner[1:] if cl else inner
        j, isd = _consume_dsml_noise(rest)
        if isd:
            tn = _match_dsml_tag(rest, j)
            if tn:
                ra = rest[tn[1]:]; ae = end
                if ra.startswith('|') or ra.startswith(chr(0xff5c)):
                    rest = rest[:tn[1]] + ra[1:]
                    ne = text.find('>', i)
                    if ne != -1 and ne < end: ae = ne
                pre = '</' if cl else '<'
                r.append(pre); r.append(rest[j:]); r.append('>'); i = ae + 1; continue
        r.append(text[i : end + 1]); i = end + 1
    return ''.join(r)


def _consume_dsml_noise(rest):
    j = 0; isd = False; rl = len(rest)
    while j < rl:
        ch = rest[j]
        if ch == '<': j += 1; isd = True; continue
        if ch in _DSML_NS: j += 1; isd = True; continue
        if rest[j:j+4].lower() == 'dsml': j += 4; isd = True; continue
        break
    return j, isd


def _match_dsml_tag(rest, j):
    rl = len(rest)
    for h, cn in _DSML_HYPHENATED.items():
        if rest[j:j+len(h)].lower() == h: return (cn, j + len(h))
    ne = j
    while ne < rl and (rest[ne].isalnum() or rest[ne] == '_'): ne += 1
    if ne == j: return None
    tn = rest[j:ne].lower()
    if tn in _TOOL_MARKUP_NAMES: return (tn, ne)
    return None


def _skip_fence_dsml(text, i):
    n = len(text)
    for ch in ('`','~'):
        fl = 0
        while i + fl < n and text[i + fl] == ch: fl += 1
        if fl >= 3:
            ep = _find_fence_close_dsml(text, i + fl, ch, fl)
            if ep >= 0: return ep, text[i:ep]
    return i, None


def _find_fence_close_dsml(text, start, ch, ml):
    i = start
    nl = text.find(chr(10), i)
    if nl >= 0: i = nl + 1
    else: return -1
    while i < len(text):
        nl = text.find(chr(10), i)
        if nl < 0: return -1
        ls = nl + 1
        fl = 0
        while ls + fl < len(text) and text[ls + fl] == ch: fl += 1
        if fl >= ml:
            af = ls + fl
            if af >= len(text) or text[af] in (chr(10), chr(13)):
                return ls + fl + (1 if af < len(text) and text[af] == chr(10) else 0)
        i = ls + 1
    return -1


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

    if _CDATA_OPEN in text: text = sanitize_loose_cdata(text)
    hc = ("</|DSML|tool_calls>" in text or "</dsml-tool_calls>" in text.lower() or "</tool_calls>" in text.lower())
    ho = ("<|DSML|tool_calls" in text.replace(' ','').replace(chr(0xff5c),'|').replace('||','|') or "<dsml-tool_calls" in text.lower() or "<tool_calls>" in text.lower())
    if hc and not ho: text = "<|DSML|tool_calls>\n" + text

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
        wcd = val_raw.startswith(_CDATA_OPEN) and val_raw.endswith(_CDATA_CLOSE)
        if wcd: val_raw = _extract_cdata_safe_dsml(val_raw)
        else: val_raw = extract_cdata(val_raw)
        if wcd and _preserves_cdata_dsml(key):
            val = _html_unescape_dsml(val_raw)
        else:
            if wcd: val_raw = _normalize_br_dsml(val_raw)
            try: val = json.loads(val_raw)
            except (json.JSONDecodeError, ValueError):
                rep = _repair_loose_json_dsml(val_raw)
                if rep != val_raw:
                    try: val = json.loads(rep)
                    except (json.JSONDecodeError, ValueError): val = _auto_type(val_raw)
                else: val = _auto_type(val_raw)
            if isinstance(val, str): val = _html_unescape_dsml(val)
        if key in args:
            ex = args[key]
            if isinstance(ex, list): ex.append(val)
            else: args[key] = [ex, val]
        else: args[key] = val
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

def build_dsml_tool_prompt(tools: List[Dict[str, Any]], passthrough: bool = False) -> str:
    if not tools:
        return ""

    if passthrough:
        # 透传模式：跳过 DSML 格式说明书，直接嵌入原始工具定义
        import json
        tools_json = json.dumps(tools, indent=2, ensure_ascii=False)
        return (
            "You have the following tools available. "
            "Use your native tool-calling format when you need to invoke one.\n\n"
            f"<tools>\n{tools_json}\n</tools>\n\n"
            "When you use a tool, use whatever tool call format you normally use "
            "(<|DSML|tool_calls>, TOOL_CALL:, or the standard format you prefer)."
        )

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
        fn = tool.get("function", {})
        name = fn.get("name") or tool.get("name", "")
        desc = (fn.get("description") or tool.get("description", "")).split("\n")[0].strip()
        if name:
            prompt += f"  - {name}: {desc}\n" if desc else f"  - {name}\n"

    return prompt


# ─── 泄漏输出清理 ────────────────────────────────────────

# 匹配 DeepSeek BOS 标记（两种形式：ASCII 下划线和 U+2581 变体）
#   - ASCII underscore: <｜begin_of_sentence｜>
#   - U+2581 variant:   <｜begin▁of▁sentence｜>
_LEAKED_BOS_MARKER_PATTERN = re.compile(
    r"(?i)<[｜\|]\s*begin[_▁]of[_▁]sentence\s*[｜\|]>"
)

# 匹配剩余的 DeepSeek 特殊标记（两种形式）
#   - ASCII underscore: <｜end_of_sentence｜>, <｜end_of_toolresults｜>, <｜end_of_instructions｜>
#   - U+2581 variant:   <｜end▁of▁sentence｜>, <｜end▁of▁toolresults｜>, <｜end▁of▁instructions｜>
_LEAKED_META_MARKER_PATTERN = re.compile(
    r"(?i)<[｜\|]\s*(?:assistant|tool|end[_▁]of[_▁]sentence|end[_▁]of[_▁]thinking|end[_▁]of[_▁]toolresults|end[_▁]of[_▁]instructions)\s*[｜\|]>"
)


def sanitize_leaked_output(text: str) -> str:
    """清理泄漏的 DeepSeek 特殊标记。

    移除模型可能输出的 DSML 对话格式标记，如：
    - <｜begin_of_sentence｜> / <｜begin▁of▁sentence｜>
    - <｜end_of_sentence｜> / <｜end▁of▁sentence｜>
    - <｜tool｜>
    - <｜assistant｜>
    - <｜end_of_thinking｜> / <｜end▁of▁thinking｜>
    - <｜end_of_toolresults｜> / <｜end▁of▁toolresults｜>
    - <｜end_of_instructions｜> / <｜end▁of▁instructions｜>
    """
    if not text:
        return text

    # 移除 BOS 标记
    text = _LEAKED_BOS_MARKER_PATTERN.sub("", text)
    # 移除其他元标记
    text = _LEAKED_META_MARKER_PATTERN.sub("", text)
    # 移除联网搜索引用标记
    text = re.sub(r"\[citation:\d+\]", "", text)

    return text


# ─── JSON修复 + CDATA保护 + br归一化 ─────────────────

_UNQUOTED_DSML = re.compile(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:')
_MISSING_ARR_DSML = re.compile(r'(:\s*)(\{(?:[^{}]|\{[^{}]*\})*\}(?:\s*,\s*\{(?:[^{}]|\{[^{}]*\})*\})+)')
_BR_DSML = re.compile(r'<br\s*/?>', re.IGNORECASE)
_CDATA_STR_DSML = {"content","file_content","text","prompt","query","command","cmd","script","code","old_string","new_string","pattern","path","file_path","result","input"}

def _repair_loose_json_dsml(s):
    s=s.strip()
    if not s: return s
    s=_UNQUOTED_DSML.sub(r'\1"\2":',s)
    s=_MISSING_ARR_DSML.sub(r'\1[\2]',s)
    return _rjb_dsml(s)

def _rjb_dsml(s):
    if '\\' not in s: return s
    r=[]; i=0
    while i<len(s):
        if s[i]=='\\':
            if i+1<len(s):
                n=s[i+1]
                if n in ('"','\\','/','b','f','n','r','t'):
                    r.append('\\');r.append(n);i+=2;continue
                if n=='u' and i+5<len(s):
                    h=s[i+2:i+6]
                    if all(c in '0123456789abcdefABCDEF' for c in h):
                        r.append('\\u');r.append(h);i+=6;continue
            r.append('\\\\');i+=1
        else: r.append(s[i]);i+=1
    return ''.join(r)

def _has_meaningful_value_dsml(v):
    if v is None: return False
    if isinstance(v,str): return v.strip()!=""
    if isinstance(v,(int,float,bool)): return True
    if isinstance(v,dict): return bool(v) and any(_has_meaningful_value_dsml(c) for c in v.values())
    if isinstance(v,list): return bool(v) and any(_has_meaningful_value_dsml(c) for c in v)
    return True

def _preserves_cdata_dsml(name):
    return name.strip().lower() in _CDATA_STR_DSML

def _extract_cdata_safe_dsml(text):
    if not text: return text
    lw=text.lower(); st=lw.find("<![cdata[")
    if st<0: return text
    cs=st+len("<![CDATA[")
    lines=text[cs:].split(chr(10))
    inf=False; fm=""; col=[]
    for ln in lines:
        sl=ln.lstrip()
        if not inf:
            if sl.startswith('```') or sl.startswith('~~~'):
                fm=sl[:3]; inf=True
        elif sl.startswith(fm): inf=False
        if not inf and ']]>' in ln:
            ei=ln.index(']]>')
            col.append(ln[:ei])
            return chr(10).join(col)
        col.append(ln)
    return chr(10).join(col)

def _normalize_br_dsml(text):
    if not text or '<br' not in text.lower(): return text
    return _BR_DSML.sub(chr(10),text).replace(chr(13)+chr(10),chr(10))

def _html_unescape_dsml(text):
    if not isinstance(text,str): return text
    for e,c in {"&lt;":"<","&gt;":">","&amp;":"&","&quot;":'"',"&apos;":"'"}.items():
        text=text.replace(e,c)
    return text


def _coerce_string_params_dsml(tcs, tools=None):
    if not tools or not tcs: return tcs
    si = {}
    for t in tools:
        fn = t.get("function", {}) or t
        n = fn.get("name", "") or t.get("name", "")
        p = fn.get("parameters") or t.get("parameters")
        if n and isinstance(p, dict): si[n] = p
    for tc in tcs:
        fn = tc.get("function", {})
        nm = fn.get("name", "")
        a = fn.get("arguments", "{}")
        if isinstance(a, dict): args = a
        else:
            try: args = json.loads(a) if isinstance(a, str) else a
            except: continue
        sc = si.get(nm)
        if not sc: continue
        pp = sc.get("properties", {})
        if not pp: continue
        for k, v in args.items():
            pr = pp.get(k, {})
            if isinstance(pr, dict) and pr.get("type") == "string" and not isinstance(v, str) and v is not None:
                args[k] = str(v)
        fn["arguments"] = json.dumps(args, ensure_ascii=False)
    return tcs
