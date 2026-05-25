"""
DeepSeek 网页 → API 代理（纯 HTTP 转发，无浏览器依赖）
用法: python proxy.py → 打开 http://localhost:8000/admin → 粘贴 cURL → 保存 → 用
"""
import json, os, shlex, time, uuid, webbrowser, base64, re, secrets, threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tiktoken
from curl_cffi import requests as cffi_requests
from app.config import config_manager, DsAccount
from app.batch import init_batch_storage as anthropic_init_batch_storage

# ── Tokenizer ───────────────────────────────────
_enc = tiktoken.get_encoding("cl100k_base")

def _count_tokens(text: str) -> int:
    return len(_enc.encode(text or ""))

# ── 用量统计 ───────────────────────────────────
from usage_store import add_usage, get_usage, clear_usage

# ── 会话管理 ───────────────────────────────────
from session_store import needs_renewal, on_new_session, add_tokens, get_expired_sessions, remove_old_session

# ── 响应存储 ───────────────────────────────────
from response_store import save_response_record, get_response_record, delete_response_record, update_response_record

# ── PoW (Proof of Work) Solver — 纯 Python 实现（无 WASM 依赖）────────
from pow_native import DeepSeekPOW

# Initialize PoW solver
pow_solver = DeepSeekPOW()

BASE_DIR = Path(__file__).parent
CONFIG_FILE = BASE_DIR / "token.json"
VISION_LOG = BASE_DIR / "vision.log"
_DEBUG = os.getenv("DS_DEBUG", "").lower() in ("1", "true", "yes")

# ── DeepSeek API 通用 Headers ─────────────────────
DS_HEADERS = {
    "content-type": "application/json",
    "origin": "https://chat.deepseek.com",
    "referer": "https://chat.deepseek.com/",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/134.0.0.0 Safari/537.36",
    "x-client-version": "2.0.2",
    "x-client-platform": "web",
}

def _vlog(msg: str):
    """Log vision-related messages. File logging only when DS_DEBUG=1."""
    ts = time.strftime("%H:%M:%S")
    if _DEBUG:
        with open(VISION_LOG, "a") as f:
            f.write(f"[{ts}] {msg}\n")
    print(f"[Vision] {msg}", flush=True)
PROXY_PORT = int(os.getenv("PROXY_PORT", "8000"))

# ── Responses API 辅助 ──────────────────────────


def _gen_response_id() -> str:
    return f"resp_{uuid.uuid4().hex}"


def _ensure_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _normalize_response_tool_output(output: Any) -> str:
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    if isinstance(output, (dict, list)):
        return json.dumps(output, ensure_ascii=False)
    return str(output)


def _normalize_response_tool(tool: Any) -> dict | None:
    if not isinstance(tool, dict):
        return None
    tid = tool.get("id") or tool.get("tool_use_id") or ""
    name = tool.get("name") or tool.get("function", {}).get("name", "")
    inp = tool.get("input") or tool.get("arguments") or tool.get("function", {}).get("arguments", "")
    if not tid or not name:
        return None
    inp_str = _normalize_response_tool_output(inp)
    return {
        "type": "function",
        "id": tid or "",
        "function": {"name": name, "arguments": inp_str},
    }


def _normalize_input_file_part(part: dict) -> dict:
    """Normalize input_items file parts for DeepSeek vision."""
    fid = part.get("file_id") or part.get("file", {}).get("file_id", "") or part.get("file", {}).get("file_data", "")
    if fid:
        return {"type": "file", "file_id": fid, "model_type": "vision"}
    return {}


def _extract_response_messages_and_tools(input_items: Any) -> tuple[list[dict], list[dict] | None]:
    if isinstance(input_items, str):
        return [{"role": "user", "content": input_items}], None
    if not isinstance(input_items, list):
        return [], None

    messages = []
    all_tools: list[dict] = []
    for item in input_items:
        role = item.get("role", "user")
        content = item.get("content", "")

        if role == "developer":
            role = "system"

        if isinstance(content, list):
            text_parts = []
            tool_results = []
            for block in content:
                if isinstance(block, dict):
                    bt = block.get("type", "")
                    if bt == "input_text":
                        text_parts.append(block.get("text", ""))
                    elif bt == "input_file":
                        fp = _normalize_input_file_part(block)
                        if fp:
                            text_parts.append(f"[file: {fp.get('file_id', '')}]")
            if text_parts:
                messages.append({"role": role, "content": "\n".join(text_parts)})
            continue

        if isinstance(content, str):
            messages.append({"role": role, "content": content})

    return messages, all_tools or None


def _merge_previous_response_context(messages: list[dict], previous_response_id: str | None) -> list[dict]:
    if not previous_response_id:
        return messages
    prev = get_response_record(previous_response_id)
    if not prev:
        raise HTTPException(404, detail={"error": {"message": f"response {previous_response_id} not found"}})
    merged = list(messages)
    prev_output = prev.get("output", [])
    for out in prev_output if isinstance(prev_output, list) else [prev_output]:
        if isinstance(out, dict) and out.get("type") == "message":
            mc = out.get("content", [])
            for c in mc if isinstance(mc, list) else [mc]:
                if isinstance(c, dict):
                    merged.append({"role": "assistant", "content": c.get("text", "")})
    return merged


def _normalize_response_tools(body: dict, parsed_tools: list[dict] | None) -> list[dict] | None:
    tools = body.get("tools")
    merged: list[dict] = []
    seen: set[str] = set()
    for source in (parsed_tools or []) + (tools if isinstance(tools, list) else []):
        normalized = _normalize_response_tool(source)
        if normalized and normalized.get("function", {}).get("name", "") not in seen:
            seen.add(normalized["function"]["name"])
            merged.append(normalized)
    return merged or None


def _has_web_search_tool(body: dict) -> bool:
    tools = body.get("tools", [])
    if isinstance(tools, list):
        for t in tools:
            if isinstance(t, dict) and t.get("type") == "web_search":
                return True
    return False


def _resolve_responses_model(body: dict) -> str:
    model = body.get("model", "deepseek-default")
    if not _has_web_search_tool(body) or "search" in model:
        return model

    candidates = []
    if model.endswith("-reasoner"):
        candidates.append(f"{model}-search")
    candidates.append(f"{model}-search")
    if model == "deepseek-default":
        candidates.append("deepseek-search")
    if model == "deepseek-reasoner":
        candidates.append("deepseek-reasoner-search")

    models = get_models()
    for candidate in candidates:
        if candidate in models:
            return candidate
    return model


def _messages_from_responses_request(body: dict) -> tuple[list[dict], list[dict] | None]:
    input_items = body.get("input", [])
    if isinstance(input_items, str):
        messages, tools = [{"role": "user", "content": input_items}], None
    else:
        messages, tools = _extract_response_messages_and_tools(input_items)

    instructions = body.get("instructions", "")
    if instructions:
        messages.insert(0, {"role": "system", "content": instructions})

    return messages, tools


def _build_responses_record(body: dict) -> dict:
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    return {
        "id": body.get("_response_id", _gen_response_id()),
        "object": "response",
        "created_at": now,
        "status": "completed",
        "model": body.get("model", "deepseek-default"),
        "output": [],
        "error": None,
        "incomplete_details": None,
    }


# ── cURL 解析 ──────────────────────────────────────────
def parse_curl(curl: str) -> dict:
    try:
        tokens = shlex.split(curl)
    except ValueError:
        tokens = curl.replace("\\\n", " ").split()
    out = {"url": "", "headers": {}, "body": ""}
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t == "curl": i += 1; continue
        if t in ("-H", "--header") and i + 1 < len(tokens):
            line = tokens[i + 1]
            if ":" in line:
                k, _, v = line.partition(":")
                out["headers"][k.strip().lower()] = v.strip()
            i += 2
        elif t in ("--data-raw", "--data", "--data-binary", "-d") and i + 1 < len(tokens):
            out["body"] = tokens[i + 1]; i += 2
        elif t in ("-X", "--request"): i += 2 if i + 1 < len(tokens) else 1
        elif t.startswith("-"): i += 1
        else: out["url"] = t; i += 1
    return out


def build_config(parsed: dict) -> dict:
    h = parsed["headers"]
    token = ""
    ah = h.get("authorization", "")
    if ah.startswith("Bearer "): token = ah[7:]

    session_id = ""
    for src in [parsed.get("url", ""), parsed.get("body", "")]:
        m = re.search(r"[sS]ession[_-]?[iI]d[=:\"]+([a-f0-9-]{36})", src)
        if m: session_id = m.group(1); break
    ref = h.get("referer", "")
    m = re.search(r"/a/chat/s/([a-f0-9-]+)", ref)
    if m: session_id = m.group(1)

    return {
        "token": token,
        "session_id": session_id,
        "headers": h,
        "cookie": h.get("cookie", ""),
        "url": parsed.get("url", ""),
    }


app = FastAPI(title="DeepSeek Proxy")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_discover():
    """启动时自动刷新模型列表，延迟清理过期会话（后台线程，避免风控）。"""
    print("[启动] 探测模型列表...")
    _discover_models()
    # 后台延迟清理过期会话
    def _bg_cleanup():
        import time as _t
        _t.sleep(10)
        try:
            cleanup_old_sessions()
        except Exception:
            pass
    import threading as _th
    _th.Thread(target=_bg_cleanup, daemon=True).start()

# ── 管理页面 ─────────────────────────────────────────────
ADMIN = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>DeepSeek Proxy</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:system-ui,sans-serif;background:#0f172a;color:#e2e8f0;min-height:100vh;display:flex;justify-content:center;align-items:flex-start;padding-top:40px}
.c{background:#1e293b;border-radius:16px;padding:32px;width:600px;max-width:95vw;border:1px solid #334155;position:relative}
h1{font-size:22px;margin-bottom:20px}
.s{display:flex;align-items:center;gap:8px;padding:12px 16px;border-radius:10px;margin-bottom:20px;font-size:14px}
.ok{background:#064e3b;color:#6ee7b7}.no{background:#1e293b;color:#94a3b8}.err{background:#450a0a;color:#fca5a5}
.d{width:10px;height:10px;border-radius:50%;display:inline-block}
.dg{background:#22c55e}.dy{background:#64748b}.dr{background:#ef4444}
.step{margin-bottom:18px}.sl{font-size:13px;color:#94a3b8;margin-bottom:6px}
.btn{padding:10px 20px;border-radius:8px;border:none;cursor:pointer;font-size:14px;font-weight:500}
.bp{background:#2563eb;color:#fff;width:100%}.bp:hover{background:#1d4ed8}
.bp:disabled{background:#1e3a5f;color:#64748b;cursor:not-allowed}
input[type=text],input[type=password],input[type=tel],input[type=email]{width:100%;padding:12px 14px;background:#0f172a;border:1px solid #334155;border-radius:8px;color:#e2e8f0;font-size:14px;font-family:inherit}
input:focus{outline:none;border-color:#3b82f6}
.row{display:flex;gap:12px;margin-bottom:14px}
.row .ac{width:90px;flex-shrink:0}
.row .ph{flex:1}
.pw-row{margin-bottom:14px}
.pw-row input{width:100%}
.tab-bar{display:flex;gap:0;margin-bottom:16px;border-radius:8px;overflow:hidden;border:1px solid #334155}
.tab{flex:1;padding:10px;text-align:center;font-size:13px;cursor:pointer;background:#0f172a;color:#94a3b8;transition:all .2s}
.tab.active{background:#2563eb;color:#fff}
.tab:hover:not(.active){background:#1e293b}
.panel{display:none}.panel.active{display:block}
hr{border:none;border-top:1px solid #334155;margin:24px 0}
.cfg{background:#0f172a;border-radius:10px;padding:16px}
.cr{display:flex;justify-content:space-between;align-items:center;padding:6px 0;font-size:13px}
.cr code{background:#1e293b;padding:2px 8px;border-radius:4px;font-size:13px;color:#7dd3fc;cursor:pointer}
.info{font-size:12px;color:#94a3b8;margin-top:8px;padding:8px 12px;background:#0f172a;border-radius:8px;border-left:3px solid #3b82f6;display:none}
.toast{position:fixed;top:20px;right:20px;padding:12px 20px;border-radius:8px;font-size:14px;z-index:999;display:none}
.ts{display:block;background:#064e3b;color:#6ee7b7}.te{display:block;background:#7f1d1d;color:#fca5a5}
/* Usage table */
.ut{width:100%;border-collapse:collapse;font-size:13px;margin-top:12px}
.ut th,.ut td{padding:10px 12px;text-align:right;border-bottom:1px solid #334155}
.ut th{color:#94a3b8;font-weight:500;font-size:11px;white-space:nowrap;position:sticky;top:0;background:#0f172a;z-index:1}
.ut td{font-variant-numeric:tabular-nums}
.ut tr:last-child td{border-bottom:none}
.ut .ml{text-align:left}
.ut .tr{font-weight:600;border-top:2px solid #2563eb}
.ut .tr td{padding-top:14px;color:#93c5fd;background:#0f172a;position:sticky;bottom:0}
.us{max-height:440px;overflow-y:auto}
.ue{text-align:center;color:#64748b;padding:40px 20px}
/* Period buttons */
.pb{padding:8px 16px;border-radius:8px;border:1px solid #334155;background:transparent;color:#e2e8f0;font-size:13px;cursor:pointer}
.pb:hover{background:#1e293b;border-color:#2563eb}
.pb.ac{background:#2563eb;color:#fff;border-color:#2563eb}
.period-btn.active{background:#2563eb;color:#fff}
a{color:#7dd3fc}
/* Account management */
.acct-tbl{width:100%;border-collapse:collapse;font-size:13px;margin-top:12px}
.acct-tbl th,.acct-tbl td{padding:8px 10px;text-align:left;border-bottom:1px solid #334155}
.acct-tbl th{color:#94a3b8;font-weight:500;font-size:11px;white-space:nowrap}
.acct-tbl td{font-variant-numeric:tabular-nums}
.acct-tbl td:nth-child(3){max-width:140px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.acct-tbl td:last-child{white-space:nowrap}
.acct-st{width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:6px;vertical-align:middle}
.acct-st.ok{background:#22c55e}.acct-st.no{background:#64748b}.acct-st.er{background:#ef4444}
.acct-btn{padding:4px 10px;border-radius:4px;border:none;cursor:pointer;font-size:12px;font-weight:500}
.acct-btn.rm{background:#7f1d1d;color:#fca5a5}.acct-btn.rm:hover{background:#991b1b}
.acct-btn.rl{background:#1e3a5f;color:#7dd3fc}.acct-btn.rl:hover{background:#1e40af}
.acct-btn.batch{background:#2563eb;color:#fff;width:100%;margin-top:12px;padding:10px;border-radius:8px;border:none;cursor:pointer;font-size:13px;font-weight:500}
.acct-btn.batch:hover{background:#1d4ed8}
.acct-add{display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap}
.acct-add input{flex:1;min-width:100px;padding:8px 10px;background:#0f172a;border:1px solid #334155;border-radius:6px;color:#e2e8f0;font-size:13px}
.acct-add button{padding:8px 16px;background:#2563eb;color:#fff;border:none;border-radius:6px;cursor:pointer;font-size:13px;font-weight:500}
.acct-add button:hover{background:#1d4ed8}
.acct-empty{text-align:center;color:#64748b;padding:30px 0;font-size:13px}
.acct-stat{font-size:12px;color:#94a3b8;margin-bottom:8px}
</style>
</head>
<body>
<div class="c">
<h1>DeepSeek Proxy</h1>
<div style="position:absolute;top:32px;right:32px">
<button onclick="toggleLang()" id="langBtn" style="padding:6px 14px;background:#1e293b;color:#94a3b8;border:1px solid #334155;border-radius:6px;cursor:pointer;font-size:13px;transition:all .2s">🌐 EN</button>
</div>
<div id="s" class="s no"><span id="sd" class="d dy"></span><span id="st" data-i18n="waitingCfg">等待配置</span></div>

<div class="tab-bar">
<div class="tab active" onclick="switchTab('phone')" data-i18n="phoneLogin">手机号登录</div>
<div class="tab" onclick="switchTab('email')" data-i18n="emailLogin">邮箱登录</div>
<div class="tab" onclick="switchTab('usage')" data-i18n="usage">用量统计</div>
<div class="tab" onclick="switchTab('accounts')" data-i18n="accounts">账号管理</div>
<div class="tab" onclick="switchTab('settings')" data-i18n="settings">设置</div>
</div>

<div id="phonePanel" class="panel active">
<div class="row">
<input class="ac" type="tel" id="area_code" value="+86" placeholder="+86">
<input class="ph" type="tel" id="mobile" data-i18n-ph="phonePlaceholder" placeholder="手机号" autocomplete="tel">
</div>
<div class="pw-row"><input type="password" id="pw1" data-i18n-ph="pwdPlaceholder" placeholder="密码" autocomplete="current-password"></div>
<button class="btn bp" id="btn1" onclick="doLogin('phone')" data-i18n="loginBtn">登录</button>
</div>

<div id="emailPanel" class="panel">
<div class="pw-row"><input type="email" id="email" data-i18n-ph="emailPlaceholder" placeholder="邮箱地址" autocomplete="email"></div>
<div class="pw-row"><input type="password" id="pw2" data-i18n-ph="pwdPlaceholder" placeholder="密码" autocomplete="current-password"></div>
<button class="btn bp" id="btn2" onclick="doLogin('email')" data-i18n="loginBtn">登录</button>
</div>

<div class="info" id="info"></div>

<div id="apiSection">
<div class="sl" style="font-weight:600;color:#e2e8f0;margin-bottom:10px" data-i18n="curlTitle">📋 cURL 导入</div>
<div class="curl-help" style="font-size:12px;color:#94a3b8;margin-bottom:12px;line-height:1.7">
  <div style="margin-bottom:6px;color:#7dd3fc;font-weight:500" data-i18n="curlSteps">导入步骤：</div>
  <div data-i18n="curlStep1">1. 打开 chat.deepseek.com 并登录</div>
  <div data-i18n="curlStep2">2. 按 F12 → Network 面板</div>
  <div data-i18n="curlStep3">3. 发送任意消息，找到 completion 请求</div>
  <div data-i18n="curlStep4">4. 右键 → Copy as cURL，粘贴到下方</div>
</div>
<textarea id="curl" data-i18n-ph="pasteCurl" placeholder="粘贴 cURL ..." style="width:100%;height:120px;background:#0f172a;border:1px solid #334155;border-radius:8px;color:#e2e8f0;padding:12px;font-family:monospace;font-size:11px;resize:vertical"></textarea>
<button class="btn bp" id="btn3" onclick="saveCurl()" data-i18n="saveCurlBtn" style="margin-top:8px">保存 cURL</button>

<hr>
<div class="step">
<div class="sl" style="font-weight:600;color:#e2e8f0;" data-i18n="apiConfig">API 配置</div>
<div class="cfg">
<div class="cr"><span data-i18n="apiAddr">API 地址</span><code onclick="cp(this)">http://localhost:""" + str(PROXY_PORT) + """/v1</code></div>
<div class="cr"><span data-i18n="apiKey">API Key</span><code onclick="cp(this)" data-i18n="apiKeyVal">任意填写</code></div>

</div>
</div>
<div class="step" style="margin-top:16px">
<button class="btn" style="background:#334155;color:#e2e8f0;width:100%;font-size:13px" onclick="refreshModels()" id="refreshBtn" data-i18n="refreshModels">🔄 刷新模型列表</button>
<div id="modelsInfo" style="margin-top:8px;font-size:12px;color:#64748b;display:none"></div>
</div>
</div>

<div id="usagePanel" class="panel">
<div id="usageContent"></div>
<div style="margin-top:14px">
<button class="pb ac" onclick="switchPeriod('total')" id="pbTotal" data-i18n="periodAll">全部</button>
<button class="pb" onclick="switchPeriod('week')" id="pbWeek" data-i18n="periodWeek">本周</button>
<button class="pb" onclick="switchPeriod('today')" id="pbToday" data-i18n="periodToday">今日</button>
<button class="btn" style="background:#334155;color:#e2e8f0;font-size:12px;padding:6px 12px;margin-left:8px" onclick="loadUsage()" data-i18n="refreshBtn">刷新</button>
<button class="btn" style="background:#7f1d1d;color:#fca5a5;font-size:12px;padding:6px 12px;margin-left:4px" onclick="clearUsage()" data-i18n="clearBtn">清空</button>
</div>
</div>

<div id="accountsPanel" class="panel">
<div class="acct-stat" id="acctStat" data-i18n="loadingAccounts">加载中...</div>
<div class="acct-add">
<input type="tel" id="acctAreaCode" value="+86" placeholder="+86" style="width:70px;flex:none">
<input type="tel" id="acctPhone" placeholder="手机号" data-i18n-ph="phonePlaceholder">
<input type="password" id="acctPw" placeholder="密码" data-i18n-ph="pwdPlaceholder">
<button onclick="addAccount()" data-i18n="addAcctBtn">添加</button>
</div>
<div id="acctList"><div class="acct-empty" data-i18n="noAccounts">暂无账号，请先添加</div></div>
<button class="acct-btn batch" onclick="reloginAll()" data-i18n="reloginAllBtn">全部重新登录</button>
</div>

<div id="settingsPanel" class="panel">
<div class="sl" style="font-weight:600;color:#e2e8f0;" data-i18n="proxyTitle">代理配置</div>
<div class="cr" style="margin-top:12px">
  <span style="color:#94a3b8;font-size:13px" data-i18n="proxyHint">绕过 AWS WAF 拦截。格式：http://127.0.0.1:7890 或 socks5://127.0.0.1:7891</span>
</div>
<div class="pw-row" style="margin-top:12px">
  <input type="text" id="proxyUrl" placeholder="http://127.0.0.1:7890" style="width:100%;background:#0f172a;border:1px solid #334155;border-radius:8px;color:#e2e8f0;padding:12px;font-size:14px">
</div>
<button class="btn bp" onclick="saveProxy()" data-i18n="proxySaveBtn" style="margin-top:8px">保存代理设置</button>
<div id="proxyStatus" style="margin-top:8px;font-size:12px;color:#64748b"></div>
</div>
</div>

<div id="toast" class="toast"></div>
<script>
// === i18n ===
var _lang=localStorage.getItem('ds_lang')||'zh';
var _I={
zh:{phoneLogin:'手机号登录',emailLogin:'邮箱登录',usage:'用量统计',settings:'设置',
phonePlaceholder:'手机号',pwdPlaceholder:'密码',loginBtn:'登录',loginBtnDoing:'登录中...',
emailPlaceholder:'邮箱地址',waitingCfg:'等待配置',configured:'已配置',connFail:'连接失败',
loggingDS:'正在登录 DeepSeek...',loginOk:'登录成功',loginFail:'失败:',
error:'错误:',saveCurlBtn:'保存 cURL',
parsing:'解析中...',saved:'已保存',apiConfig:'API 配置',apiAddr:'API 地址',
apiKey:'API Key',apiKeyVal:'任意填写',refreshModels:'🔄 刷新模型列表',
refreshingModels:'刷新中...',foundModels:'✅ 发现',foundModelsSuffix:'个模型:',
refreshOk:'刷新成功',refreshFail:'刷新失败',
periodAll:'全部',periodWeek:'本周',periodToday:'今日',refreshBtn:'刷新',clearBtn:'清空',
noData:'📊 暂无用量数据',loadFail:'加载失败: ',modelHeader:'模型',reqHeader:'请求',
inputHeader:'输入',outputHeader:'输出',totalHeader:'总计',sumLabel:'📋 合计',
clearConfirm:'确定清空全部用量数据？',cleared:'已清空',clearFail:'清空失败',
accounts:'账号管理',loadingAccounts:'加载中...',noAccounts:'暂无账号，请先添加',
addAcctBtn:'添加',reloginAllBtn:'全部重新登录',
accountHeader:'账号',statusHeader:'状态',tokenHeader:'Token',
loginTimeHeader:'登录时间',opHeader:'操作',valid:'有效',notLogin:'未登录',
reloginBtn:'重登',deleteBtn:'删除',
deleteConfirm:'确定删除账号',deleted:'已删除',deleteFail:'删除失败:',
reloginOk:'重新登录成功',reloginFail:'重登失败: ',
addFail:'添加失败: ',
allRelogining:'全部重新登录中...',allReloginDone:'重登完成:',allReloginFail:'失败:',
proxyTitle:'代理配置',proxyHint:'绕过 AWS WAF 拦截。格式：http://127.0.0.1:7890 或 socks5://127.0.0.1:7891',proxySaveBtn:'保存代理设置',proxySaved:'已保存',proxySaveFail:'保存失败: ',proxyLoadFail:'加载失败: ',
phoneRequired:'请输入手机号和密码',emailRequired:'请输入邮箱和密码',
pasteCurl:'粘贴 cURL ...',modelCountSuffix:' 个模型: ',unknownErr:'未知错误',
curlTitle:'📋 cURL 导入',curlSteps:'导入步骤：',
curlStep1:'1. 打开 chat.deepseek.com 并登录',curlStep2:'2. 按 F12 → Network 面板',
curlStep3:'3. 发送任意消息，找到 completion 请求',curlStep4:'4. 右键 → Copy as cURL，粘贴到下方'},
en:{phoneLogin:'Phone Login',emailLogin:'Email Login',usage:'Usage',settings:'Settings',
phonePlaceholder:'Phone Number',pwdPlaceholder:'Password',loginBtn:'Login',loginBtnDoing:'Logging in...',
emailPlaceholder:'Email Address',waitingCfg:'Awaiting Config',configured:'Configured',connFail:'Connection Failed',
loggingDS:'Logging into DeepSeek...',loginOk:'Login Successful',loginFail:'Failed:',
error:'Error:',saveCurlBtn:'Save cURL',
parsing:'Parsing...',saved:'Saved',apiConfig:'API Config',apiAddr:'API Endpoint',
apiKey:'API Key',apiKeyVal:'Any value',refreshModels:'🔄 Refresh Models',
refreshingModels:'Refreshing...',foundModels:'✅ Found',foundModelsSuffix:' model(s):',
refreshOk:'Refreshed',refreshFail:'Refresh Failed',
periodAll:'All',periodWeek:'This Week',periodToday:'Today',refreshBtn:'Refresh',clearBtn:'Clear',
noData:'📊 No Usage Data',loadFail:'Load failed: ',modelHeader:'Model',reqHeader:'Requests',
inputHeader:'Input',outputHeader:'Output',totalHeader:'Total',sumLabel:'📋 Total',
clearConfirm:'Clear all usage data?',cleared:'Cleared',clearFail:'Clear Failed',
accounts:'Accounts',loadingAccounts:'Loading...',noAccounts:'No accounts, add one first',
addAcctBtn:'Add',reloginAllBtn:'Relogin All',
accountHeader:'Account',statusHeader:'Status',tokenHeader:'Token',
loginTimeHeader:'Login Time',opHeader:'Action',valid:'Valid',notLogin:'Not Logged In',
reloginBtn:'Relogin',deleteBtn:'Delete',
deleteConfirm:'Delete account',deleted:'Deleted',deleteFail:'Delete failed:',
reloginOk:'Relogin successful',reloginFail:'Relogin failed: ',
addFail:'Add failed: ',
allRelogining:'Relogging all...',allReloginDone:'Done:',allReloginFail:'Failed:',
proxyTitle:'Proxy Config',proxyHint:'Bypass AWS WAF. Format: http://127.0.0.1:7890 or socks5://127.0.0.1:7891',proxySaveBtn:'Save Proxy',proxySaved:'Saved',proxySaveFail:'Save failed: ',proxyLoadFail:'Load failed: ',
phoneRequired:'Phone number and password required',emailRequired:'Email and password required',
pasteCurl:'Paste cURL ...',modelCountSuffix:' model(s): ',unknownErr:'Unknown error',
curlTitle:'📋 cURL Import',curlSteps:'Steps:',
curlStep1:'1. Open chat.deepseek.com and log in',curlStep2:'2. Press F12 → Network tab',
curlStep3:'3. Send any message, find the completion request',curlStep4:'4. Right-click → Copy as cURL, paste below'}};
function _(k){return (_I[_lang]||_I.zh)[k]||k}
function toggleLang(){_lang=_lang==='zh'?'en':'zh';localStorage.setItem('ds_lang',_lang);Q('langBtn').textContent=_lang==='zh'?'🌐 EN':'🌐 中';applyI18n()}
function applyI18n(){
Qs('[data-i18n]').forEach(function(el){var k=el.getAttribute('data-i18n');if(k){el.textContent=_(k)}});
Qs('[data-i18n-ph]').forEach(function(el){var k=el.getAttribute('data-i18n-ph');if(k){el.placeholder=_(k)}});
Qs('[data-i18n-val]').forEach(function(el){var k=el.getAttribute('data-i18n-val');if(k){el.value=_(k)}});
loadUsage();cs();
}
function Qs(s){return document.querySelectorAll(s)}
function Q(id){return document.getElementById(id)}
function switchTab(type){
var ti={'phone':0,'email':1,'usage':2,'settings':3};
document.querySelectorAll('.tab').forEach((t,i)=>{t.className='tab'+(i===ti[type]?' active':'');});
if(Q('phonePanel'))Q('phonePanel').className='panel'+(type==='phone'?' active':'');
if(Q('emailPanel'))Q('emailPanel').className='panel'+(type==='email'?' active':'');
if(Q('usagePanel'))Q('usagePanel').className='panel'+(type==='usage'?' active':'');
if(Q('accountsPanel'))Q('accountsPanel').className='panel'+(type==='accounts'?' active':'');
if(Q('settingsPanel'))Q('settingsPanel').className='panel'+(type==='settings'?' active':'');
var as=Q('apiSection');if(as)as.style.display=(type==='usage'||type==='accounts'||type==='settings')?'none':'';
if(type==='usage')loadUsage();
if(type==='accounts')loadAccounts();
if(type==='settings')loadProxy();
}
async function cs(){
try{const r=await fetch('/api/config');const d=await r.json()
if(d.configured){Q('s').className='s ok';Q('sd').className='d dg';Q('st').textContent=_('configured')+' | '+d.masked}
else{Q('s').className='s no';Q('sd').className='d dy';Q('st').textContent=d.error||_('waitingCfg')}
}catch(e){Q('s').className='s err';Q('st').textContent=_('connFail')}
}
async function doLogin(type){
let body={}
if(type==='phone'){
const m=Q('mobile').value.trim();const p=Q('pw1').value;const a=Q('area_code').value.trim()
if(!m||!p){t(_('phoneRequired'),1);return}
body={mobile:m,password:p,area_code:a,login_type:'phone'}
var btn=Q('btn1')
}else{
const e=Q('email').value.trim();const p=Q('pw2').value
if(!e||!p){t(_('emailRequired'),1);return}
body={email:e,password:p,login_type:'email'}
var btn=Q('btn2')
}
btn.disabled=true;btn.textContent=_('loginBtnDoing')
Q('info').style.display='block';Q('info').innerHTML=_('loggingDS')
try{
const r=await fetch('/api/login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})
const d=await r.json()
if(d.ok){Q('info').innerHTML=_('loginOk')+' | Token: '+d.masked+' | Session: '+d.session_id;t(_('loginOk'));cs()}
else{Q('info').innerHTML=_('loginFail')+d.error;t(d.error,1)}
}catch(e){Q('info').innerHTML=_('error')+e.message;t(e.message,1)}
btn.disabled=false;btn.textContent=_('loginBtn')
}
async function saveCurl(){
const c=Q('curl').value.trim();if(!c){t(_('pasteCurl'),1);return}
const b=Q('btn3');b.disabled=true;b.textContent=_('parsing')
Q('info').style.display='block';Q('info').innerHTML=_('parsing')
try{
const r=await fetch('/api/config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({curl:c})})
const d=await r.json()
if(d.ok){Q('info').innerHTML='OK | '+d.masked+' | Session '+d.session_id;t(_('saved'));cs()}
else{Q('info').innerHTML=_('loginFail')+d.error;t(d.error,1)}
}catch(e){Q('info').innerHTML=_('error')+e.message;t(e.message,1)}
b.disabled=false;b.textContent=_('saveCurlBtn')
}
function cp(el){navigator.clipboard.writeText(el.textContent);t('Copied')}
function t(m,e){const x=Q('toast');x.textContent=m;x.className='toast t'+(e?'e':'s');setTimeout(()=>x.className='toast',2500)}
async function refreshModels(){
const btn=Q('refreshBtn');const info=Q('modelsInfo')
btn.disabled=true;btn.textContent=_('refreshingModels');info.style.display='none'
try{
const r=await fetch('/v1/models/refresh',{method:'POST'})
const d=await r.json()
const names=d.data.map(m=>m.id).join(', ')
info.style.display='block';info.innerHTML=_('foundModels')+d.data.length+_('modelCountSuffix')+names;t(_('refreshOk'))
}catch(e){info.style.display='block';info.innerHTML='❌ '+_('loginFail')+e.message;t(_('refreshFail'),1)}
btn.disabled=false;btn.textContent=_('refreshModels')
}
// === 用量统计 ===
var _up='total';
function f(n){return n.toLocaleString()}
async function loadUsage(){
try{
const r=await fetch('/api/usage');const d=await r.json();
const p=d[_up]||d.total||{};const m=p.models||{};const t=p.total||{};
const e=Object.entries(m).sort((a,b)=>b[1].total_tokens-a[1].total_tokens);
if(!e.length&&!t.requests){Q('usageContent').innerHTML='<div class=ue>'+_('noData')+'</div>';return}
let h='<div class=us><table class=ut><thead><tr><th class=ml>'+_('modelHeader')+'</th><th>'+_('reqHeader')+'</th><th>'+_('inputHeader')+'</th><th>'+_('outputHeader')+'</th><th>'+_('totalHeader')+'</th></tr></thead><tbody>';
for(const[k,v]of e){h+=`<tr><td class=ml>${k}</td><td>${f(v.requests)}</td><td>${f(v.prompt_tokens)}</td><td>${f(v.completion_tokens)}</td><td>${f(v.total_tokens)}</td></tr>`}
h+='<tr class=tr><td class=ml>'+_('sumLabel')+'</td><td>'+f(t.requests)+'</td><td>'+f(t.prompt_tokens)+'</td><td>'+f(t.completion_tokens)+'</td><td>'+f(t.total_tokens)+'</td></tr></tbody></table></div>';
Q('usageContent').innerHTML=h
}catch(e){Q('usageContent').innerHTML='<div class=ue>'+_('loadFail')+e.message+'</div>'}
}
function switchPeriod(p){
_up=p;
['total','week','today'].forEach(x=>{var b=Q('pb'+x.charAt(0).toUpperCase()+x.slice(1));if(b)b.className='pb'+(x===p?' ac':'')});
loadUsage()
}
async function clearUsage(){
if(!confirm(_('clearConfirm')))return;
try{await fetch('/api/usage',{method:'DELETE'});t(_('cleared'));loadUsage()}catch(e){t(_('clearFail'),1)}
}
// === 账号管理 ===
async function loadAccounts(){
try{
const r=await fetch('/api/accounts');const d=await r.json();
Q('acctStat').textContent=d.total+' '+_('accounts')+', '+d.valid+' '+_('valid');
if(d.accounts&&d.accounts.length){
var h='<table class="acct-tbl"><tr><th>'+_('accountHeader')+'</th><th>'+_('statusHeader')+'</th><th>'+_('tokenHeader')+'</th><th>'+_('loginTimeHeader')+'</th><th>'+_('opHeader')+'</th></tr>';
for(const a of d.accounts){
var st=a.is_valid?'ok':'no';
var stl=a.is_valid?_('valid'):_('notLogin');
var tk=a.token_masked||'***';
var lt=a.login_time||'-';
var l=encodeURIComponent(a.account_label);
h+='<tr><td>'+a.account_label+'</td><td><span class="acct-st '+st+'"></span>'+stl+'</td><td>'+tk+'</td><td>'+lt+'</td>';
h+='<td><button class="acct-btn rl" onclick="reloginAccount(\''+l+'\')">'+_('reloginBtn')+'</button><button class="acct-btn rm" onclick="removeAccount(\''+l+'\')">'+_('deleteBtn')+'</button></td></tr>';
}
h+='</table>';
Q('acctList').innerHTML=h;
}else{Q('acctList').innerHTML='<div class="acct-empty">'+_('noAccounts')+'</div>'}
}catch(e){Q('acctList').innerHTML='<div class="acct-empty">'+_('loadFail')+e.message+'</div>'}
}
async function addAccount(){
var area=Q('acctAreaCode').value.trim()||'+86';
var phone=Q('acctPhone').value.trim();
var pw=Q('acctPw').value;
if(!phone||!pw){t(_('phoneRequired'),1);return}
try{
const r=await fetch('/api/accounts',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({login_type:'phone',area_code:area,mobile:phone,password:pw})});
const d=await r.json();
if(d.ok){t(_('saved'));Q('acctPhone').value='';Q('acctPw').value='';loadAccounts()}
else{t(_('addFail')+(d.detail||d.error||''),1)}
}catch(e){t(_('addFail')+e.message,1)}
}
async function removeAccount(label){
if(!confirm(_('deleteConfirm')+' '+decodeURIComponent(label)+'？'))return;
try{
const r=await fetch('/api/accounts/'+label,{method:'DELETE'});
const d=await r.json();
if(d.ok){t(_('deleted'));loadAccounts()}
else{t(_('deleteFail')+(d.detail||''),1)}
}catch(e){t(_('deleteFail')+e.message,1)}
}
async function reloginAccount(label){
try{
const r=await fetch('/api/accounts/'+label+'/relogin',{method:'POST'});
const d=await r.json();
if(d.ok){t(_('reloginOk'));loadAccounts()}
else{t(_('reloginFail')+(d.error||''),1)}
}catch(e){t(_('reloginFail')+e.message,1)}
}
async function reloginAll(){
var btn=document.querySelector('.acct-btn.batch');
btn.disabled=true;btn.textContent=_('allRelogining');
try{
const r=await fetch('/api/accounts/relogin-all',{method:'POST'});
const d=await r.json();
t(_('allReloginDone')+' '+d.success+'/'+d.total+(_('allReloginFail')?', '+_('allReloginFail')+' '+(d.total-d.success):''));
loadAccounts()
}catch(e){t(_('allReloginFail')+e.message,1)}
btn.disabled=false;btn.textContent=_('reloginAllBtn');
}
// === 代理配置 ===
async function loadProxy(){
try{
const r=await fetch('/api/proxy');const d=await r.json();
Q('proxyUrl').value=d.proxy||'';
var st=Q('proxyStatus');
st.textContent=d.proxy?_('proxySaved'):'';
st.style.color=d.proxy?'#22c55e':'#64748b';
}catch(e){Q('proxyStatus').textContent=_('proxyLoadFail')+e.message}
}
async function saveProxy(){
var url=Q('proxyUrl').value.trim();
try{
const r=await fetch('/api/proxy',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({proxy:url})});
const d=await r.json();
if(d.ok){Q('proxyStatus').textContent=_('proxySaved');Q('proxyStatus').style.color='#22c55e';t(_('proxySaved'))}
else{Q('proxyStatus').textContent=_('proxySaveFail')+(d.msg||'');t(_('proxySaveFail')+(d.msg||''),1)}
}catch(e){Q('proxyStatus').textContent=_('proxySaveFail')+e.message;t(_('proxySaveFail')+e.message,1)}
}
cs()
</script>
</body>
</html>"""


from starlette.responses import RedirectResponse

@app.get("/")
async def root():
    return RedirectResponse(url="/admin")


@app.get("/admin", response_class=HTMLResponse)
async def admin():
    from starlette.responses import Response
    html = ADMIN
    return Response(content=html, media_type="text/html", headers={
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
    })


# ── 配置 API ─────────────────────────────────────────────

def _load_config_sync() -> dict:
    """同步加载 token.json 原始数据（供非 async 上下文使用）。"""
    if not CONFIG_FILE.exists():
        return {}
    return json.loads(CONFIG_FILE.read_text("utf-8"))


@app.get("/api/config")
async def get_config():
    if not CONFIG_FILE.exists():
        return {"configured": False, "error": "未配置"}
    d = _load_config_sync()
    t = d.get("token", "")
    return {
        "configured": True,
        "masked": t[:20] + "..." + t[-8:] if len(t) > 30 else "***",
        "session_id": d.get("session_id", "N/A"),
    }


@app.post("/api/config")
async def save_config(data: dict):
    curl = data.get("curl", "").strip()
    if not curl: raise HTTPException(400, "请提供 cURL")
    parsed = parse_curl(curl)
    cfg = build_config(parsed)
    if not cfg["token"]: return {"ok": False, "error": "未从 cURL 提取到 Token，请确认 Authorization header"}
    if not cfg["session_id"]: return {"ok": False, "error": "未从 cURL 提取到 Session ID"}
    CONFIG_FILE.write_text(json.dumps(cfg, ensure_ascii=False), "utf-8")
    t = cfg["token"]
    return {"ok": True, "masked": t[:20] + "..." + t[-8:], "session_id": cfg["session_id"]}


# ── DeepSeek 登录 API ─────────────────────────────────────
@app.post("/api/login")
async def deepseek_login(data: dict):
    login_type = data.get("login_type", "phone")
    password = data.get("password", "").strip()
    if not password:
        raise HTTPException(400, "请提供密码")

    # 构造登录 payload（参考 NIyueeE/ds-free-api: email 和 mobile 二选一）
    login_payload = {"password": password, "device_id": secrets.token_hex(16), "os": "web"}
    account_label = ""
    email, mobile, area_code = "", "", "+86"

    if login_type == "email":
        email = data.get("email", "").strip()
        if not email:
            raise HTTPException(400, "请提供邮箱")
        login_payload["email"] = email
        login_payload["mobile"] = ""
        login_payload["area_code"] = ""
        account_label = email
    else:
        mobile = data.get("mobile", "").strip()
        area_code = data.get("area_code", "+86").strip()
        if not mobile:
            raise HTTPException(400, "请提供手机号")
        login_payload["mobile"] = mobile
        login_payload["area_code"] = area_code
        login_payload["email"] = ""
        account_label = f"{area_code} {mobile}"

    DS_HEADERS = {
        "content-type": "application/json",
        "origin": "https://chat.deepseek.com",
        "referer": "https://chat.deepseek.com/",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/134.0.0.0 Safari/537.36",
        "x-client-version": "2.0.2",
        "x-client-platform": "web",
    }

    try:
        # 0. 创建 Session + 预访问首页获取 WAF Cookie
        session = cffi_requests.Session()
        session.impersonate = "chrome120"
        proxy_dict = _get_proxy_dict()
        if proxy_dict:
            session.proxies = proxy_dict
        try:
            session.get(
                "https://chat.deepseek.com/",
                headers={"user-agent": DS_HEADERS.get("user-agent", "")},
                timeout=15,
            )
        except Exception:
            pass

        # 1. 登录
        login_resp = session.post(
            "https://chat.deepseek.com/api/v0/users/login",
            json=login_payload,
            headers=DS_HEADERS,
            timeout=30,
        )

        # WAF 检测
        if login_resp.status_code == 202 and login_resp.headers.get("x-amzn-waf-action"):
            return {"ok": False, "error": "登录被 AWS WAF 拦截 (HTTP 202)。请配置代理（设置 → 代理配置）绕过 WAF。"}

        raw_text = (login_resp.text or "").strip()
        if not raw_text:
            return {"ok": False, "error": f"登录失败: 服务器返回空响应 (HTTP {login_resp.status_code})，可能是 IP 被风控或需要完成人机验证"}

        try:
            login_data = login_resp.json()
        except Exception:
            preview = raw_text[:200]
            return {"ok": False, "error": f"登录失败: 服务器返回非 JSON 响应 (HTTP {login_resp.status_code}): {preview}"}

        outer_code = login_data.get("code", 0)
        data_block = login_data.get("data") or {}
        biz_code = data_block.get("biz_code", 0)
        biz_msg = data_block.get("biz_msg", "")

        if login_resp.status_code != 200 or outer_code != 0 or biz_code != 0:
            err_msg = biz_msg or login_data.get("msg") or f"HTTP {login_resp.status_code}/code={outer_code}/biz_code={biz_code}"
            return {"ok": False, "error": f"登录失败: {err_msg}"}

        biz_data = data_block.get("biz_data") or {}
        token = biz_data.get("user", {}).get("token", "")
        if not token:
            return {"ok": False, "error": f"登录失败: biz_data 中无 token（biz_msg={biz_msg}）"}

        print(f"[Login] Token acquired for {account_label}: {token[:20]}...{token[-8:]}")

        # 2. 创建会话
        auth_headers = {**DS_HEADERS, "authorization": f"Bearer {token}"}
        session_resp = session.post(
            "https://chat.deepseek.com/api/v0/chat_session/create",
            json={},
            headers=auth_headers,
            timeout=15,
        )

        session_id = ""
        if session_resp.status_code == 200:
            session_data = session_resp.json()
            biz = session_data.get("data", {}).get("biz_data", {})
            session_id = biz.get("chat_session", {}).get("id", "") or biz.get("id", "")
            print(f"[Login] Session created: {session_id}")
        else:
            print(f"[Login] Session creation failed: {session_resp.status_code} {session_resp.text[:200]}")

        # 3. 保存配置
        cfg = {
            "token": token,
            "session_id": session_id,
            "headers": {**DS_HEADERS, "authorization": f"Bearer {token}"},
            "cookie": "",
            "account": account_label,
            "login_type": login_type,
            "_password": password,
            "_email": email if login_type == "email" else "",
            "_mobile": mobile if login_type == "phone" else "",
            "_area_code": area_code if login_type == "phone" else "+86",
        }
        CONFIG_FILE.write_text(json.dumps(cfg, ensure_ascii=False), "utf-8")

        masked = token[:20] + "..." + token[-8:]
        return {"ok": True, "masked": masked, "session_id": session_id}

    except Exception as e:
        print(f"[Login] Error: {e}")
        return {"ok": False, "error": str(e)}


# ── Health ───────────────────────────────────────────────
@app.get("/health")
async def health():
    if CONFIG_FILE.exists(): return {"status": "ok", "configured": True}
    return {"status": "waiting", "configured": False}


# ─── 用量统计 API ─────────────────────────────────────────────

@app.get("/api/usage")
async def usage_stats():
    return get_usage()


@app.delete("/api/usage")
async def clear_usage_stats():
    clear_usage()
    return {"ok": True}


# ── 代理配置 ────────────────────────────────────────


def _get_proxy_dict() -> dict | None:
    """从 ConfigManager 读取代理配置，返回 curl_cffi 兼容格式。
    返回 None 表示未配置代理，应走直连。"""
    url = config_manager.get_proxy()
    if not url:
        return None
    return {"http": url, "https": url}


@app.get("/api/proxy")
async def get_proxy():
    """获取当前代理配置"""
    proxy_url = config_manager.get_proxy()
    return {"proxy": proxy_url or ""}


@app.put("/api/proxy")
async def set_proxy(data: dict):
    """设置代理地址。传 {"proxy": "http://127.0.0.1:7890"} 或 {"proxy": ""} 清除。"""
    url = data.get("proxy", "").strip()
    config_manager.set_proxy(url)
    return {"ok": True, "proxy": url}


# ─── 多账号管理 API ───────────────────────────────────────


@app.get("/api/accounts")
async def list_accounts():
    """获取所有账号列表"""
    return {
        "accounts": config_manager.get_all_accounts(),
        "total": config_manager.count(),
        "valid": config_manager.count_valid(),
    }


@app.post("/api/accounts")
async def add_account(data: dict):
    """手动添加账号"""
    login_type = data.get("login_type", "phone")
    password = data.get("password", "").strip()
    if not password:
        raise HTTPException(400, "请提供密码")

    if login_type == "email":
        email = data.get("email", "").strip()
        if not email:
            raise HTTPException(400, "请提供邮箱")
        account_label = email
    else:
        mobile = data.get("mobile", "").strip()
        area_code = data.get("area_code", "+86").strip()
        if not mobile:
            raise HTTPException(400, "请提供手机号")
        account_label = f"{area_code} {mobile}"

    existing = config_manager.get_account_by_label(account_label)
    if existing:
        if existing.token:
            return {"ok": True, "account_label": account_label, "exist": True}

    ds_account = DsAccount(
        account_label=account_label,
        login_type=login_type,
        _password=password,
        _mobile=mobile if login_type == "phone" else "",
        _area_code=area_code if login_type == "phone" else "+86",
        _email=email if login_type == "email" else "",
        login_time="",
        is_valid=False,
    )
    added = config_manager.add_account(ds_account)
    return {"ok": True, "account_label": account_label, "added": added}


@app.delete("/api/accounts/{account_label}")
async def remove_account(account_label: str):
    """删除账号"""
    from urllib.parse import unquote
    label = unquote(account_label)
    if config_manager.remove_account(label):
        return {"ok": True, "account_label": label}
    raise HTTPException(404, f"账号 {label} 不存在")


@app.post("/api/accounts/{account_label}/relogin")
async def relogin_account(account_label: str):
    """重新登录指定账号"""
    from urllib.parse import unquote
    label = unquote(account_label)
    account = config_manager.get_account_by_label(label)
    if not account:
        raise HTTPException(404, f"账号 {label} 不存在")

    login_type = account.login_type
    password = account._password
    if not password:
        raise HTTPException(400, f"账号 {label} 无保存密码，无法自动登录")

    cfg = {
        "login_type": login_type,
        "_password": password,
        "_email": account._email,
        "_mobile": account._mobile,
        "_area_code": account._area_code,
        "account": label,
    }

    new_cfg = relogin(cfg)
    if new_cfg:
        return {"ok": True, "account_label": label, "token_masked": new_cfg.get("token", "")[:20] + "..."}
    return {"ok": False, "error": "重新登录失败"}


@app.post("/api/accounts/relogin-all")
async def relogin_all():
    """重新登录所有有效账号"""
    accounts = config_manager.get_all_accounts()
    results = []
    for acc in accounts:
        label = acc.get("account_label", "")
        account = config_manager.get_account_by_label(label)
        if not account or not account._password:
            results.append({"label": label, "ok": False, "error": "无密码"})
            continue

        cfg = {
            "login_type": account.login_type,
            "_password": account._password,
            "_email": account._email,
            "_mobile": account._mobile,
            "_area_code": account._area_code,
            "account": label,
        }
        new_cfg = relogin(cfg)
        results.append({"label": label, "ok": bool(new_cfg), "error": None if new_cfg else "登录失败"})

    return {"results": results, "total": len(results), "success": sum(1 for r in results if r["ok"])}


@app.post("/api/cleanup")
async def manual_cleanup():
    """手动触发会话清理。"""
    try:
        cleanup_old_sessions()
        return {"ok": True, "msg": "清理完成"}
    except Exception as e:
        return {"ok": False, "msg": str(e)}


# ─── 模型列表（免鉴权，供管理页面使用） ───────────────────────

@app.get("/api/models")
async def admin_models():
    return {"models": list(get_models().keys())}


# ── 模型映射（动态从 DeepSeek 探测）─────────────────
MODEL_CONFIG_URL = "https://chat.deepseek.com/api/v0/client/settings?scope=model"

_models_cache = {}       # model_id → (thinking, search, max_in, max_out)
_models_cache_time = 0
_MODELS_TTL = 3600       # 缓存1小时


def _discover_models() -> dict:
    """从 DeepSeek /api/v0/client/settings?scope=model 动态获取模型配置。

    返回: {model_id: (thinking_enabled, search_enabled, max_input, max_output), ...}
    失败返回 None。
    """
    global _models_cache, _models_cache_time

    cfg = _load_config_sync()
    if not cfg:
        return None

    token = cfg.get("token", "")
    ua = cfg.get("headers", {}).get("user-agent", "Mozilla/5.0")

    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": ua,
        "X-Client-Version": "2.0.0",
        "X-Client-Platform": "web",
    }

    try:
        resp = cffi_requests.get(MODEL_CONFIG_URL, headers=headers, timeout=10, proxies=_get_proxy_dict())
        data = resp.json()
        biz_data = data.get("data", {}).get("biz_data", {})
        settings = biz_data.get("settings", {})
        model_configs = settings.get("model_configs", {}).get("value", [])

        if not model_configs:
            print(f"[模型发现] model_configs 为空")
            return None

        models = {}
        for mc in model_configs:
            mt = mc.get("model_type")
            if not mt or not mc.get("enabled"):
                continue

            # 上下文大小：优先从 input_character_limit 推算 (V4 系列 ≈ 1M tokens)，
            # 对 Expert 等 UI 限制偏小的模型硬编码 1M
            icl = mc.get("input_character_limit", 0) or 0
            if icl >= 1_000_000:
                max_in = int(icl * 0.4)      # 2621440 × 0.4 ≈ 1048576 (1M)
            else:
                max_in = 1_048_576            # Expert 等硬编码 1M
            max_out = max_in                  # DeepSeek V4 输出上限即上下文大小
            has_think = mc.get("think_feature") is not None
            has_search = mc.get("search_feature") is not None

            # 基础模型
            name = f"deepseek-{mt}" if mt != "default" else "deepseek-default"
            models[name] = (False, False, max_in, max_out)
            print(f"[模型发现]   {name}: in={max_in}, out={max_out}, think={has_think}, search={has_search}")

            # 思维链变体
            if has_think:
                tname = "deepseek-reasoner" if mt == "default" else f"deepseek-{mt}-reasoner"
                models[tname] = (True, False, max_in, max_out)

            # 搜索变体
            if has_search:
                sname = "deepseek-search" if mt == "default" else f"deepseek-{mt}-search"
                models[sname] = (False, True, max_in, max_out)

            # 思考+联网 组合变体
            if has_think and has_search:
                cname = "deepseek-reasoner-search" if mt == "default" else f"deepseek-{mt}-reasoner-search"
                models[cname] = (True, True, max_in, max_out)

        if models:
            # 模型名称为纯英文ID，中文对照见 README.md
            _models_cache = models
            _models_cache_time = time.time()
            print(f"[模型发现] 发现 {len(models)} 个模型: {list(models.keys())}")
            return models

    except Exception as e:
        print(f"[模型发现] 失败: {e}")

    return None


def get_models() -> dict:
    """获取模型映射（缓存优先，过期自动刷新。发现失败返回 {}）。"""
    global _models_cache, _models_cache_time

    if _models_cache and time.time() - _models_cache_time < _MODELS_TTL:
        return _models_cache

    discovered = _discover_models()
    if discovered:
        return discovered

    # 探测失败 → 返回空（不骗人）
    print("[模型发现] 探测失败，模型列表为空")
    return {}


# ── Token 自动刷新 ─────────────────────────────────────────
def relogin(cfg: dict) -> dict | None:
    """用保存的凭证重新登录，返回新 cfg 或 None"""
    login_type = cfg.get("login_type", "")
    password = cfg.get("_password", "")
    if not password:
        print("[Token] 无保存密码，无法自动刷新")
        return None

    login_payload = {"password": password, "device_id": secrets.token_hex(16), "os": "web"}
    account_label = cfg.get("account", "")

    if login_type == "email":
        email = cfg.get("_email", "")
        if not email:
            return None
        login_payload["email"] = email
        login_payload["mobile"] = ""
        login_payload["area_code"] = ""
    elif login_type == "phone":
        mobile = cfg.get("_mobile", "")
        area_code = cfg.get("_area_code", "+86")
        if not mobile:
            return None
        login_payload["mobile"] = mobile
        login_payload["area_code"] = area_code
        login_payload["email"] = ""
    else:
        return None

    DS_HEADERS = {
        "content-type": "application/json",
        "origin": "https://chat.deepseek.com",
        "referer": "https://chat.deepseek.com/",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/134.0.0.0 Safari/537.36",
        "x-client-version": "2.0.2",
        "x-client-platform": "web",
    }

    try:
        print(f"[Token] 自动重新登录 {account_label}...")

        # 0. 创建 Session + 预访问首页获取 WAF Cookie
        session = cffi_requests.Session()
        session.impersonate = "chrome120"
        proxy_dict = _get_proxy_dict()
        if proxy_dict:
            session.proxies = proxy_dict
        try:
            session.get(
                "https://chat.deepseek.com/",
                headers={"user-agent": DS_HEADERS.get("user-agent", "")},
                timeout=15,
            )
        except Exception:
            pass

        # 1. 登录
        login_resp = session.post(
            "https://chat.deepseek.com/api/v0/users/login",
            json=login_payload,
            headers=DS_HEADERS,
            timeout=30,
        )
        # WAF 检测
        if login_resp.status_code == 202 and login_resp.headers.get("x-amzn-waf-action"):
            print(f"[Token] 自动登录被 AWS WAF 拦截 (HTTP 202)")
            return None

        raw_text = (login_resp.text or "").strip()
        if not raw_text:
            print(f"[Token] 自动登录失败: 服务器返回空响应 (HTTP {login_resp.status_code})")
            return None
        try:
            login_data = login_resp.json()
        except Exception:
            print(f"[Token] 自动登录失败: 非 JSON 响应: {raw_text[:200]}")
            return None
        outer_code = login_data.get("code", 0)
        data_block = login_data.get("data") or {}
        biz_code = data_block.get("biz_code", 0)
        biz_msg = data_block.get("biz_msg", "")

        if login_resp.status_code != 200 or outer_code != 0 or biz_code != 0:
            err_msg = biz_msg or login_data.get("msg") or f"HTTP {login_resp.status_code}/code={outer_code}/biz_code={biz_code}"
            print(f"[Token] 自动登录失败: {err_msg}")
            return None

        biz_data = data_block.get("biz_data") or {}
        token = biz_data.get("user", {}).get("token", "")
        if not token:
            print(f"[Token] 登录失败: biz_data 中无 token（biz_msg={biz_msg}）")
            return None

        print(f"[Token] 新 token: {token[:20]}...{token[-8:]}")

        # 2. 创建新会话
        auth_headers = {**DS_HEADERS, "authorization": f"Bearer {token}"}
        session_resp = session.post(
            "https://chat.deepseek.com/api/v0/chat_session/create",
            json={},
            headers=auth_headers,
            timeout=15,
        )
        session_id = ""
        if session_resp.status_code == 200:
            session_data = session_resp.json()
            biz = session_data.get("data", {}).get("biz_data", {})
            session_id = biz.get("chat_session", {}).get("id", "") or biz.get("id", "")
            print(f"[Token] 新 session: {session_id}")
        else:
            print(f"[Token] Session 创建失败: {session_resp.status_code}")

        new_cfg = {
            "token": token,
            "session_id": session_id,
            "headers": {**DS_HEADERS, "authorization": f"Bearer {token}"},
            "cookie": "",
            "account": account_label,
            "login_type": login_type,
            # 保留凭证供下次刷新
            "_password": password,
            "_email": cfg.get("_email", ""),
            "_mobile": cfg.get("_mobile", ""),
            "_area_code": cfg.get("_area_code", "+86"),
        }
        CONFIG_FILE.write_text(json.dumps(new_cfg, ensure_ascii=False), "utf-8")
        return new_cfg

    except Exception as e:
        print(f"[Token] 自动登录异常: {e}")
        return None


def load_config_with_refresh() -> dict:
    """加载配置，如果 token 失效则自动刷新"""
    if not CONFIG_FILE.exists():
        return {}
    cfg = json.loads(CONFIG_FILE.read_text("utf-8"))
    return cfg


# ── OpenAI 兼容 API ──────────────────────────────────────
@app.get("/v1/models")
async def models():
    data = []
    for mid, (think, search, mi, mo) in get_models().items():
        data.append({
            "id": mid, "object": "model", "created": 1704067200,
            "owned_by": "deepseek",
            "max_input_tokens": mi, "max_output_tokens": mo,
            "context_length": mi, "context_window": mi,
            "supported_parameters": ["tools", "tool_choice", "temperature", "max_tokens", "stream"],
        })
    return {"object": "list", "data": data}


@app.get("/v1/models/{model_id}")
async def model_detail(model_id: str):
    info = get_models().get(model_id)
    if not info:
        raise HTTPException(404, f"模型 {model_id} 不存在")
    think, search, mi, mo = info
    return {
        "id": model_id, "object": "model", "created": 1704067200,
        "owned_by": "deepseek",
        "max_input_tokens": mi, "max_output_tokens": mo,
        "context_length": mi, "context_window": mi,
    }


@app.post("/v1/models/refresh")
async def refresh_models():
    """强制刷新模型列表"""
    global _models_cache_time
    _models_cache_time = 0  # 让下次 get_models() 重新探测
    models = get_models()
    data = []
    for mid, (think, search, mi, mo) in models.items():
        data.append({
            "id": mid, "object": "model", "created": 1704067200,
            "owned_by": "deepseek",
            "max_input_tokens": mi, "max_output_tokens": mo,
            "context_length": mi, "context_window": mi,
            "supported_parameters": ["tools", "tool_choice", "temperature", "max_tokens", "stream"],
        })
    return {"object": "list", "data": data}


def build_request_headers(cfg: dict, session_id: str) -> dict:
    """Build headers for DeepSeek API request, excluding stale PoW and conflict headers."""
    # Start from saved headers
    req_headers = dict(cfg.get("headers", {}))

    # Remove stale PoW - we'll generate fresh one
    req_headers.pop("x-ds-pow-response", None)

    # Remove headers that curl_cffi manages or that conflict
    for h in ("host", "content-length", "transfer-encoding", "accept-encoding",
              "content-type"):
        req_headers.pop(h, None)

    # Ensure required headers
    req_headers["content-type"] = "application/json"
    req_headers["origin"] = "https://chat.deepseek.com"
    req_headers["referer"] = f"https://chat.deepseek.com/a/chat/s/{session_id}"

    return req_headers


def get_pow_response(target_path: str = "/api/v0/chat/completion") -> str | None:
    """Get fresh PoW response from DeepSeek."""
    try:
        cfg = json.loads(CONFIG_FILE.read_text("utf-8"))
        headers = build_request_headers(cfg, cfg["session_id"])

        resp = cffi_requests.post(
            "https://chat.deepseek.com/api/v0/chat/create_pow_challenge",
            headers=headers,
            json={"target_path": target_path},
            impersonate="chrome120",
            timeout=15,
            proxies=_get_proxy_dict(),
        )
        if resp.status_code == 200:
            data = resp.json()
            challenge = data.get("data", {}).get("biz_data", {}).get("challenge", {})
            if challenge:
                pow_response = pow_solver.solve_challenge(challenge)
                print(f"[PoW] Solved: {pow_response[:50]}...")
                return pow_response
            else:
                print(f"[PoW] No challenge: {data}")
        else:
                print(f"[PoW] Request failed {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        print(f"[PoW] Error: {e}")
    return None


# ── 文件上传（Vision 模型支持）──────────────────────────────

def upload_file_to_deepseek(file_data: bytes, filename: str, content_type: str = "image/png") -> str | None:
    """Upload a file to DeepSeek and return the file_id.

    Uses the /api/v0/file/upload_file endpoint with PoW authentication.
    Returns file_id string or None on failure.
    """
    if not CONFIG_FILE.exists():
        _vlog("upload: no config")
        return None
    cfg = json.loads(CONFIG_FILE.read_text("utf-8"))
    session_id = cfg["session_id"]

    # Get PoW for upload_file scene
    pow_response = get_pow_response(target_path="/api/v0/file/upload_file")

    req_headers = build_request_headers(cfg, session_id)
    if pow_response:
        req_headers["x-ds-pow-response"] = pow_response

    # Remove content-type, let requests/curl set multipart boundary
    req_headers.pop("content-type", None)

    # curl_cffi doesn't support `files` param; use standard requests for upload
    import requests as req
    try:
        resp = req.post(
            "https://chat.deepseek.com/api/v0/file/upload_file",
            headers=req_headers,
            files={"file": (filename, file_data, content_type)},
            timeout=60,
            proxies=_get_proxy_dict(),
        )
        if resp.status_code == 200:
            data = resp.json()
            file_id = (data.get("data", {})
                            .get("biz_data", {})
                            .get("id", "")
                       or data.get("data", {})
                              .get("id", ""))
            if file_id:
                _vlog(f"upload OK: {filename} -> {file_id}")
                return file_id
            _vlog(f"upload: no file_id in response: {resp.text[:300]}")
        else:
            _vlog(f"upload HTTP {resp.status_code}: {resp.text[:300]}")
    except Exception as e:
        _vlog(f"upload error: {e}")
    return None


def fork_file_to_vision(cfg: dict, file_id: str) -> str | None:
    """Fork an uploaded file to the vision model type.

    DeepSeek requires forking files to a specific model before they can be
    referenced in chat. Returns the new forked file_id or None.
    """
    import requests as req
    try:
        headers = build_request_headers(cfg, cfg["session_id"])
        resp = req.post(
            "https://chat.deepseek.com/api/v0/file/fork_file_task",
            headers=headers,
            json={"file_id": file_id, "to_model_type": "vision"},
            timeout=15,
            proxies=_get_proxy_dict(),
        )
        if resp.status_code == 200:
            data = resp.json()
            biz_data = data.get("data", {}).get("biz_data", {})
            forked_id = biz_data.get("id") or biz_data.get("file_id")
            if forked_id and forked_id != file_id:
                _vlog(f"fork OK: {file_id} -> {forked_id}")
                return forked_id
        _vlog(f"fork failed: {resp.status_code} {resp.text[:200]}")
    except Exception as e:
        _vlog(f"fork error: {e}")
    return None


def wait_for_file_parsing(cfg: dict, file_ids: list[str], timeout: int = 30) -> list[str]:
    """Wait for DeepSeek to finish parsing uploaded files.

    Polls /api/v0/file/fetch_files until all files are parsed or timeout.
    Returns list of successfully parsed file_ids.
    """
    import time as _time
    if not file_ids:
        return []
    start = _time.time()
    while _time.time() - start < timeout:
        statuses = _fetch_file_statuses(cfg, file_ids)
        if statuses is None:
            _time.sleep(1)
            continue
        all_done = True
        parsed_ids = []
        for fid in file_ids:
            s = statuses.get(fid, {})
            status = str(s.get("status", "")).upper()
            # Terminal states: file is processed (success or not, just done)
            if status in ("SUCCESS", "COMPLETED", "CONTENT_EMPTY", "FAILED", "ERROR", "PARSE_FAILED"):
                if status == "SUCCESS":
                    parsed_ids.append(fid)
                # Even non-success states mean the file is done processing
            elif status in ("PENDING", "PARSING", "UPLOADING", "QUEUED"):
                all_done = False
                # If it's been more than 5s and still PARSING, accept it anyway
                if _time.time() - start > 5:
                    _vlog(f"file {fid} still {status} after 5s, accepting")
                    parsed_ids.append(fid)
            else:
                # Unknown status — assume done
                _vlog(f"file {fid} unknown status={status}, accepting")
                parsed_ids.append(fid)
        if all_done and parsed_ids:
            print(f"[Vision] Files parsed: {parsed_ids}")
            return parsed_ids
        if parsed_ids and _time.time() - start > 5:
            # Some files parsed, others still processing — return what we have
            if parsed_ids:
                return parsed_ids
        _time.sleep(1)
    print(f"[Vision] Parse timeout, got 0/{len(file_ids)} files")
    return []


def _fetch_file_statuses(cfg: dict, file_ids: list[str]) -> dict | None:
    """Fetch parse status for uploaded files from DeepSeek."""
    import requests as req
    try:
        session_id = cfg["session_id"]
        headers = build_request_headers(cfg, session_id)
        resp = req.get(
            "https://chat.deepseek.com/api/v0/file/fetch_files",
            headers=headers,
            params={"file_ids": file_ids},
            timeout=15,
            proxies=_get_proxy_dict(),
        )
        if resp.status_code == 200:
            data = resp.json()
            files = (data.get("data", {}).get("biz_data", {}).get("files", [])
                     or data.get("data", {}).get("files", []))
            if not files:
                # Sometimes response wraps differently
                biz = data.get("data", {}).get("biz_data", {})
                for key in ("file_statuses", "file_list", "items"):
                    if key in biz:
                        files = biz[key]
                        break
            statuses = {}
            for f in files:
                fid = f.get("id") or f.get("file_id") or f.get("_id")
                if fid and fid in file_ids:
                    statuses[fid] = f
            return statuses if statuses else None
        print(f"[Vision] fetch_files HTTP {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        print(f"[Vision] fetch_files error: {e}")
    return None


# ── 会话清理 ─────────────────────────────────────


def _delete_deepseek_session(token: str, session_id: str) -> bool:
    """调用 DeepSeek API 删除指定会话。"""
    try:
        headers = {**DS_HEADERS, "authorization": f"Bearer {token}"}
        resp = cffi_requests.post(
            "https://chat.deepseek.com/api/v0/chat_session/delete",
            json={"chat_session_id": session_id},
            headers=headers,
            impersonate="chrome120",
            timeout=15,
            proxies=_get_proxy_dict(),
        )
        if resp.status_code == 200:
            data = resp.json()
            return data.get("data", {}).get("biz_code") == 0
        return False
    except Exception as e:
        print(f"[Cleanup] Delete session {session_id} failed: {e}")
        return False


def cleanup_old_sessions():
    """清理所有账号中过期的旧会话。每次删除后等待 3 秒，避免触发风控。"""
    expired = get_expired_sessions()
    if not expired:
        return

    print(f"[Cleanup] Found {len(expired)} expired sessions, deleting with 10s delay...")
    deleted = 0
    for account_label, session_id, model, days_ago in expired:
        token = config_manager.get_token(account_label)
        if not token:
            continue
        if _delete_deepseek_session(token, session_id):
            remove_old_session(account_label, session_id)
            deleted += 1
            print(f"[Cleanup] Deleted: {session_id[:12]}... ({days_ago}d old)")
        time.sleep(10)
    if deleted:
        print(f"[Cleanup] Done: {deleted}/{len(expired)} deleted")


def extract_images_from_messages(messages: list) -> list[dict]:
    """Extract image URLs/bytes from OpenAI-format messages.

    Returns list of dicts: {data: bytes, content_type: str, filename: str}
    Supports: image_url (url/base64), images (list), content array
    """
    import base64 as b64
    images = []
    for msg in messages:
        content = msg.get("content", "")
        # OpenAI multi-content format
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "image_url":
                        url = part.get("image_url", {}).get("url", "")
                        images.append(_parse_image_url(url))
                    elif part.get("type") == "image":
                        data = part.get("data", "") or part.get("source", {}).get("data", "")
                        if data:
                            images.append(_parse_image_url(data))
        elif isinstance(content, str):
            # Check for images array in msg
            imgs = msg.get("images", [])
            for img in imgs:
                if isinstance(img, str):
                    images.append(_parse_image_url(img))
                elif isinstance(img, dict):
                    data = img.get("data", "") or img.get("url", "")
                    if data:
                        images.append(_parse_image_url(data))
    return [img for img in images if img is not None]


def extract_text_files_from_messages(messages: list) -> list[dict]:
    """Extract text files from OpenAI-format messages.

    Returns list of dicts: {data: bytes, filename: str, content_type: str}
    Handles type="file" content parts with base64 file_data or data fields.
    """
    import base64 as b64
    files = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "file":
                    file_obj = part.get("file", {})
                    if isinstance(file_obj, dict):
                        filename = file_obj.get("filename", "file.txt")
                        file_data = file_obj.get("file_data", "") or file_obj.get("data", "")
                        if file_data:
                            try:
                                data = b64.b64decode(file_data)
                                ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "txt"
                                ct_map = {
                                    "md": "text/markdown", "py": "text/x-python",
                                    "json": "application/json", "yaml": "text/yaml",
                                    "yml": "text/yaml", "txt": "text/plain",
                                    "csv": "text/csv", "xml": "text/xml",
                                    "html": "text/html", "js": "text/javascript",
                                    "css": "text/css", "sh": "text/x-shellscript",
                                }
                                content_type = ct_map.get(ext, "text/plain")
                                files.append({
                                    "data": data,
                                    "filename": filename,
                                    "content_type": content_type,
                                })
                            except Exception:
                                continue
    return files


def _parse_image_url(url_or_data: str) -> dict | None:
    """Parse an image URL or base64 data string."""
    import base64 as b64
    if not url_or_data:
        return None
    s = url_or_data.strip()
    # base64 data URI
    if s.startswith("data:"):
        header, encoded = s.split(",", 1)
        ct = "image/png"
        for part in header.split(";")[0].split(":")[1:]:
            ct = part
        try:
            data = b64.b64decode(encoded)
            ext = ct.split("/")[-1] if "/" in ct else "png"
            return {"data": data, "content_type": ct, "filename": f"image.{ext}"}
        except Exception:
            print(f"[Vision] Failed to decode base64 image")
            return None
    # HTTP URL
    if s.startswith("http://") or s.startswith("https://"):
        try:
            resp = cffi_requests.get(s, timeout=30, impersonate="chrome120", proxies=_get_proxy_dict())
            if resp.status_code == 200:
                ct = resp.headers.get("content-type", "image/png")
                ext = ct.split("/")[-1] if "/" in ct else "png"
                return {"data": resp.content, "content_type": ct, "filename": f"image.{ext}"}
        except Exception as e:
            print(f"[Vision] Failed to download image: {e}")
    return None










# ── 消息格式转换 ──────────────────────────────────────────

def _convert_messages(messages, tools=None):
    """将 OpenAI 消息列表转换为 DeepSeek 原生 prompt 格式（无工具版本）。"""
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
            if segs:
                parts.append(ASST + "\n\n".join(segs) + EOS)
            elif content:
                parts.append(ASST + str(content) + EOS)
            last_role = "assistant"
        elif role == "tool":
            result = str(content) if content else ""
            if result:
                parts.append(TOOL + result[:500] + TOOL_END)
            last_role = "tool"

    if last_role != "assistant":
        parts.append(ASST)
    return "".join(parts)


@app.post("/v1/chat/completions")
async def chat(request: Request):
    if not CONFIG_FILE.exists():
        raise HTTPException(503, detail="请先访问 http://localhost:{}/admin 登录账号".format(PROXY_PORT))

    body = await request.json()
    messages = body.get("messages", [])
    model = body.get("model", "deepseek-default")
    stream = body.get("stream", False)
    tools = body.get("tools", None)

    # Log client info for debugging
    ua = request.headers.get("user-agent", "?")[:60]
    msg = f"[REQ] model={model} stream={stream} msgs={len(messages)} tools={bool(tools)} ua={ua}"
    print(msg, flush=True)
    _vlog(msg)

    # 模型映射
    model_info = get_models().get(model, get_models().get("deepseek-default"))
    thinking_enabled, search_enabled, _, _ = model_info

    cfg = json.loads(CONFIG_FILE.read_text("utf-8"))
    ref_file_ids = []
    import time as _vtime

    # 文本文件：上传到 DeepSeek（不 fork，等解析完直接用原始 file_id）
    text_files = extract_text_files_from_messages(messages)
    if text_files:
        _t0 = _vtime.time()
        _vlog(f"TEXT_FILES: found {len(text_files)} files")
        raw_ids = []
        for i, tf in enumerate(text_files):
            _t1 = _vtime.time()
            orig_fid = upload_file_to_deepseek(tf["data"], tf["filename"], tf["content_type"])
            _vlog(f"text_upload #{i} -> {orig_fid} ({_vtime.time()-_t1:.1f}s)")
            if orig_fid:
                raw_ids.append(orig_fid)
        if raw_ids:
            text_ids = wait_for_file_parsing(cfg, raw_ids, timeout=30)
            ref_file_ids.extend(text_ids)
            _vlog(f"TEXT_DONE: {len(text_ids)}/{len(raw_ids)} ready ({_vtime.time()-_t0:.1f}s)")

    # Vision 模型：提取、上传、fork 图片
    is_vision = "vision" in model
    if is_vision:
        _t0 = _vtime.time()
        _vlog(f"START model={model} msgs={len(messages)}")
        images = extract_images_from_messages(messages)
        _vlog(f"extracted {len(images)} images ({_vtime.time()-_t0:.1f}s)")
        for i, img in enumerate(images):
            _t1 = _vtime.time()
            orig_fid = upload_file_to_deepseek(img["data"], img["filename"], img["content_type"])
            _vlog(f"upload #{i} -> {orig_fid} ({_vtime.time()-_t1:.1f}s)")
            if orig_fid:
                _t2 = _vtime.time()
                forked_fid = fork_file_to_vision(cfg, orig_fid)
                _vlog(f"fork #{i} -> {forked_fid} ({_vtime.time()-_t2:.1f}s)")
                if forked_fid:
                    ref_file_ids.append(forked_fid)
        if ref_file_ids:
            _t3 = _vtime.time()
            ref_file_ids = wait_for_file_parsing(cfg, ref_file_ids, timeout=10)
            _vlog(f"parse_check -> {len(ref_file_ids)} ready ({_vtime.time()-_t3:.1f}s)")
        _vlog(f"DONE: {len(images)} images -> {len(ref_file_ids)} ready ({_vtime.time()-_t0:.1f}s)")

        # Create a FRESH session for vision to avoid parallel_chat_limit_by_queue
        # from any lingering requests on the main session
        try:
            token = cfg.get("token", "")
            if token:
                auth_h = {**cfg.get("headers", {}), "authorization": f"Bearer {token}"}
                sess_resp = cffi_requests.post(
                    "https://chat.deepseek.com/api/v0/chat_session/create",
                    json={}, headers=auth_h, impersonate="chrome120", timeout=15,
                    proxies=_get_proxy_dict())
                if sess_resp.status_code == 200:
                    biz = sess_resp.json().get("data", {}).get("biz_data", {})
                    new_sid = biz.get("chat_session", {}).get("id", "") or biz.get("id", "")
                    if new_sid:
                        cfg = dict(cfg)
                        cfg["session_id"] = new_sid
                        _vlog(f"vision fresh session: {new_sid}")
        except Exception as e:
            _vlog(f"fresh session failed: {e}")

    prompt = _convert_messages(messages)
    prompt_tokens = _count_tokens(prompt)
    has_tools = False

    # Try streaming for all models including vision with images.
    # Old issue: vision stream put everything in thinking_content, but the new
    # fragments format (THINK/RESPONSE) should handle this correctly now.

    result = _do_chat(cfg, prompt, model, thinking_enabled, search_enabled, stream,
                    is_retry=False, has_tools=has_tools, tools=tools,
                    ref_file_ids=ref_file_ids)

    # (Vision SSE wrapper removed — all models now stream directly via fragments format)

    # 用量统计：非流式直接计数，流式包装生成器
    if stream and hasattr(result, 'body_iterator'):
        orig_iter = result.body_iterator
        async def _counted_stream():
            completion_text = ""
            async for chunk in orig_iter:
                s = chunk.decode("utf-8", errors="ignore") if isinstance(chunk, bytes) else str(chunk)
                if s.startswith("data: ") and not s.startswith("data: [DONE]"):
                    try:
                        obj = json.loads(s[6:])
                        delta = obj.get("choices", [{}])[0].get("delta", {})
                        c = delta.get("content", "") or ""
                        r = delta.get("reasoning_content", "") or ""
                        completion_text += (c + r)
                    except: pass
                yield chunk
            add_usage(model, prompt_tokens, _count_tokens(completion_text))
        result.body_iterator = _counted_stream()
    else:
        add_usage(model, prompt_tokens, 0)
    return result


def _do_chat(cfg, prompt, model, thinking_enabled, search_enabled, stream, is_retry=False, has_tools=False, tools=None, ref_file_ids=None):
    """核心聊天逻辑，支持 token 过期后重试
    
    DeepSeek SSE 流结构（thinking_enabled=True 时）：
    - data: {"v":{"response":{...}}} → 元数据，跳过
    - data: {"p":"response/thinking_content","v":"嗯"} → thinking 第一段（有p）
    - data: {"o":"APPEND","v":"，"} → thinking 后续段（无p，有o=APPEND）
    - data: {"v":"用户"} → thinking 更多后续（只有v）
    - data: {"p":"response/content","o":"APPEND","v":"你好"} → 正式内容第一段
    - data: {"v":"！"} → 正式内容后续
    - data: {"p":"response/status","v":"FINISHED"} → 状态，跳过
    - event: title → 对话标题，跳过
    - event: toast → 错误提示（如版本过低）
    """
    session_id = cfg["session_id"]
    req_headers = build_request_headers(cfg, session_id)
    pow_response = get_pow_response()
    if pow_response:
        req_headers["x-ds-pow-response"] = pow_response

    # model_type 字段：DeepSeek 根据此值路由到不同模型后端。
    # 映射：模型名含 "vision" → "vision"，含 "expert" → "expert"，其余 → "default"
    req_body = {
        "chat_session_id": session_id,
        "parent_message_id": None,
        "prompt": prompt,
        "ref_file_ids": ref_file_ids if ref_file_ids else [],
        "thinking_enabled": thinking_enabled,
        "search_enabled": search_enabled,
    }
    if "vision" in model:
        req_body["model_type"] = "vision"
        if ref_file_ids:
            _vlog(f"chat request files={ref_file_ids} thinking={thinking_enabled}")
    elif "expert" in model:
        if ref_file_ids:
            # 专家模式不支持文件上传，自动降级到快速模式
            print(f"[Chat] 专家模式不支持文件上传，自动降级到快速模式 (files={len(ref_file_ids)})")
            req_body["model_type"] = "default"
        else:
            req_body["model_type"] = "expert"
    else:
        req_body["model_type"] = "default"


    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    def _parse_sse(resp):
        """Shared SSE parser — yields (type, value) tuples.
        type: "content" | "thinking" | "error" | "done"
        value: string content or error dict

        Handles two SSE formats:
        1. Old format: response/thinking_content + response/content
        2. New format: response/fragments/-1/content with fragment type tracking
           (fragments have type THINK or RESPONSE)
        """
        # Pre-flight: check Content-Type — if DeepSeek returns HTML/text instead of SSE,
        # treat the entire response as an error to avoid silent data loss
        ct = resp.headers.get("content-type", "")
        if ct and "text/event-stream" not in ct and "application/json" not in ct:
            body_sample = ""
            try:
                body_sample = resp.text[:300] if hasattr(resp, "text") else ""
            except Exception:
                pass
            yield ("error", {
                "message": f"DeepSeek returned non-SSE response (Content-Type: {ct}): {body_sample}",
                "code": "bad_content_type"
            })
            return

        # Track non-JSON lines for error detection
        non_json_line_count = 0
        phase = "thinking"
        # New format: track fragment type (THINK/RESPONSE) from metadata events
        fragment_type = None  # None = old format (use phase), "THINK"/"RESPONSE" = new format
        _line_buf = b""
        def _read_lines():
            nonlocal _line_buf
            for chunk in resp.iter_content(chunk_size=4096):
                if not chunk:
                    continue
                _line_buf += chunk
                while b"\n" in _line_buf:
                    raw_line, _line_buf = _line_buf.split(b"\n", 1)
                    yield raw_line.decode("utf-8", errors="ignore").strip()
            # flush remaining buffer
            if _line_buf.strip():
                yield _line_buf.decode("utf-8", errors="ignore").strip()

        for line in _read_lines():
            if not line:
                continue
            # Debug: log raw SSE lines for thinking models
            if thinking_enabled and line.startswith("data:") and "fragments" in line:
                _vlog(f"SSE_LINE: {line[:500]}")

            # Skip event: lines (title, update_session, etc.)
            if line.startswith("event:"):
                if line.startswith("event: hint"):
                    continue  # handled below via raw line processing
                continue

            # Detect raw text/HTML error responses
            if line.startswith("<!DOCTYPE") or line.startswith("<html") or line.startswith("<HTML"):
                yield ("error", {
                    "message": f"DeepSeek returned HTML error: {line[:200]}",
                    "code": "html_response"
                })
                return

            if non_json_line_count >= 3:
                yield ("error", {
                    "message": f"DeepSeek returned non-SSE text (too many non-JSON lines): first={line[:200]}",
                    "code": "non_sse_response"
                })
                return

            # DeepSeek non-SSE error JSON
            if line.startswith("{"):
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict) and "code" in obj and obj.get("code", 0) >= 40000:
                        yield ("error", {"message": obj.get("msg", "unknown"), "code": obj.get("code")})
                        return
                except json.JSONDecodeError:
                    pass
                continue

            ds = line[6:] if line.startswith("data: ") else line
            if ds.strip() == "[DONE]":
                yield ("done", "")
                return

            try:
                obj = json.loads(ds)
                if not isinstance(obj, dict):
                    continue

                # Error object: {"type": "error", "content": "...", "finish_reason": "..."}
                obj_type = obj.get("type", "")
                if obj_type == "error":
                    content = obj.get("content", "")
                    fr = obj.get("finish_reason", "")
                    yield ("error", {"message": content, "code": fr})
                    return

                val = obj.get("v")

                # Toast error (v is dict with type=error)
                if isinstance(val, dict):
                    # Check for error
                    t_type = val.get("type", "")
                    t_content = val.get("content", "")
                    fr = val.get("finish_reason", "")
                    if t_type == "error" and fr:
                        yield ("error", {"message": t_content, "code": fr})
                        return
                    # New format: metadata with response.fragments → extract fragment type
                    resp_data = val.get("response", {})
                    if isinstance(resp_data, dict):
                        frags = resp_data.get("fragments", [])
                        if frags and isinstance(frags, list):
                            last_frag = frags[-1]
                            if isinstance(last_frag, dict) and last_frag.get("type"):
                                fragment_type = last_frag["type"]
                                if thinking_enabled:
                                    _vlog(f"SSE: fragment_type={fragment_type}")
                    continue

                path = obj.get("p", "")

                # ── New format: response/fragments ──────────────────
                # Fragment append event: {"p":"response/fragments","o":"APPEND","v":[{"id":N,"type":"RESPONSE","content":"...",...}]}
                if path == "response/fragments" and obj.get("o") == "APPEND" and isinstance(val, list):
                    if val:
                        last_frag = val[-1] if isinstance(val[-1], dict) else {}
                        new_type = last_frag.get("type", "")
                        if new_type:
                            fragment_type = new_type
                            if thinking_enabled:
                                _vlog(f"SSE: new fragment type={new_type}")
                        # Extract initial content from fragment object
                        frag_content = last_frag.get("content", "")
                        if frag_content and isinstance(frag_content, str):
                            if fragment_type == "THINK":
                                yield ("thinking", frag_content)
                            else:
                                yield ("content", frag_content)
                    continue

                # Fragment content: {"p":"response/fragments/-1/content","o":"APPEND","v":"..."}
                # or without "o": {"p":"response/fragments/-1/content","v":"..."}
                if path == "response/fragments/-1/content":
                    if fragment_type == "THINK":
                        phase = "thinking"
                        if isinstance(val, str) and val:
                            yield ("thinking", val)
                    else:  # RESPONSE or unknown
                        phase = "content"
                        if isinstance(val, str) and val:
                            yield ("content", val)
                    continue

                # ── Old format: response/thinking_content + response/content ──
                if path == "response/content" and obj.get("o") == "APPEND":
                    phase = "content"
                    if isinstance(val, str) and val:
                        yield ("content", val)
                elif path == "response/thinking_content" and thinking_enabled:
                    phase = "thinking"
                    if isinstance(val, str) and val:
                        yield ("thinking", val)
                elif path:
                    continue  # other metadata (status, elapsed_secs, BATCH, etc.)
                elif isinstance(val, str) and val:
                    # Pathless continuation lines: use fragment_type if new format, else phase
                    if fragment_type is not None:
                        # New format: use fragment type
                        if fragment_type == "THINK":
                            yield ("thinking", val)
                        else:
                            yield ("content", val)
                    else:
                        # Old format: use phase
                        if phase == "thinking" and thinking_enabled:
                            yield ("thinking", val)
                        else:
                            yield ("content", val)
            except json.JSONDecodeError:
                non_json_line_count += 1
                continue

    def do_stream():
        """SSE streaming for OpenAI-compatible clients."""
        try:
            resp = cffi_requests.post(
                "https://chat.deepseek.com/api/v0/chat/completion",
                headers=req_headers,
                json=req_body,
                impersonate="chrome120",
                stream=True,
                timeout=120,
                proxies=_get_proxy_dict(),
            )

            if ref_file_ids or thinking_enabled:
                _vlog(f"chat stream response: status={resp.status_code} ct={resp.headers.get('content-type','?')} model={model} thinking={thinking_enabled}")

            if resp.status_code == 401 and not is_retry:
                print("[Token] 401, trying refresh...")
                new_cfg = relogin(cfg)
                if new_cfg:
                    for chunk in _do_chat_stream_only(new_cfg, prompt, model, thinking_enabled, search_enabled, has_tools, tools, ref_file_ids):
                        yield chunk
                    return
                yield f'data: {json.dumps({"error": {"message": "Token expired", "type": "auth_error", "code": 401}})}\n\n'
                yield "data: [DONE]\n\n"
                return

            if resp.status_code != 200:
                error_msg = f"DeepSeek returned {resp.status_code}: {resp.text[:300]}"
                print(f"[Error] {error_msg}")
                yield f'data: {json.dumps({"error": {"message": error_msg, "type": "server_error", "code": resp.status_code}})}\n\n'
                yield "data: [DONE]\n\n"
                return

            # No tools: normal streaming
            _stream_think_count = 0
            _stream_content_count = 0
            # Send role delta first — many clients need this to start rendering
            r = {"id": chat_id, "object": "chat.completion.chunk", "created": created, "model": model,
                 "choices": [{"index": 0, "delta": {"role": "assistant", "content": None}, "finish_reason": None}]}
            yield f'data: {json.dumps(r, ensure_ascii=False)}\n\n'
            for etype, val in _parse_sse(resp):
                if etype == "content":
                    _stream_content_count += 1
                    r = {"id": chat_id, "object": "chat.completion.chunk", "created": created, "model": model,
                         "choices": [{"index": 0, "delta": {"content": val}, "finish_reason": None}]}
                    yield f'data: {json.dumps(r, ensure_ascii=False)}\n\n'
                elif etype == "thinking":
                    _stream_think_count += 1
                    r = {"id": chat_id, "object": "chat.completion.chunk", "created": created, "model": model,
                         "choices": [{"index": 0, "delta": {"reasoning_content": val}, "finish_reason": None}]}
                    yield f'data: {json.dumps(r, ensure_ascii=False)}\n\n'
                elif etype == "error":
                    yield f'data: {json.dumps({"error": {"message": val["message"], "type": "server_error", "code": val.get("code")}})}\n\n'
                    yield "data: [DONE]\n\n"
                    return
                elif etype == "done":
                    if thinking_enabled:
                        _vlog(f"STREAM_DONE: thinking_chunks={_stream_think_count} content_chunks={_stream_content_count}")
                    yield "data: [DONE]\n\n"
                    return

        except Exception as e:
            print(f"[Error] do_stream failed: {e}")
            yield f'data: {json.dumps({"error": {"message": str(e), "type": "server_error"}})}\n\n'
            yield "data: [DONE]\n\n"

    def do_nonstream():
        """Non-streaming: use stream=True internally (curl_cffi stream=False
        returns incomplete SSE), buffer all events, return complete JSON response."""
        full_content = ""
        full_thinking = ""

        try:
            resp = cffi_requests.post(
                "https://chat.deepseek.com/api/v0/chat/completion",
                headers=req_headers,
                json=req_body,
                impersonate="chrome120",
                stream=True,  # Always stream — curl_cffi stream=False truncates SSE
                timeout=120,
                proxies=_get_proxy_dict(),
            )

            if ref_file_ids or thinking_enabled:
                _vlog(f"chat nonstream(stream-internal) response: status={resp.status_code} ct={resp.headers.get('content-type','?')}")

            if resp.status_code == 401 and not is_retry:
                print("[Token] 401 in nonstream, trying refresh...")
                new_cfg = relogin(cfg)
                if new_cfg:
                    return _do_chat(new_cfg, prompt, model, thinking_enabled, search_enabled, False, is_retry=True, has_tools=has_tools, tools=tools, ref_file_ids=ref_file_ids)

            if resp.status_code != 200:
                body_sample = ""
                try:
                    body_sample = resp.text[:500] if hasattr(resp, "text") else f"(no body, status={resp.status_code})"
                except Exception:
                    body_sample = f"(body unreadable, status={resp.status_code})"
                print(f"[nonstream] DeepSeek {resp.status_code}: {body_sample[:200]}")
                raise HTTPException(502, detail={
                    "error": {
                        "message": f"DeepSeek returned {resp.status_code}: {body_sample[:200]}",
                        "type": "server_error",
                        "code": resp.status_code
                    }
                })

            # Buffer all events from stream using _parse_sse
            for etype, val in _parse_sse(resp):
                if etype == "content":
                    full_content += val
                elif etype == "thinking":
                    full_thinking += val
                elif etype == "error":
                    raise HTTPException(502, detail={"error": {
                        "message": val["message"],
                        "type": "server_error",
                        "code": val.get("code", "")
                    }})
                elif etype == "done":
                    break

        except HTTPException:
            raise
        except Exception as e:
            print(f"[nonstream] Error: {e}")
            raise HTTPException(502, detail={"error": {"message": str(e), "type": "server_error"}})

        # Debug: log extracted thinking/content for thinking models
        if thinking_enabled:
            _vlog(f"NONSTREAM_RESULT: thinking={len(full_thinking)} chars, content={len(full_content)} chars")
            _vlog(f"NONSTREAM_THINKING[:500]: {full_thinking[:500]}")
            _vlog(f"NONSTREAM_CONTENT[:500]: {full_content[:500]}")

        finish_reason = "stop"
        final_content = full_content

        msg = {"role": "assistant", "content": final_content}
        if full_thinking:
            msg["reasoning_content"] = full_thinking

        # Build and validate response — pre-serialize to catch any issues early
        response_body = {
            "id": chat_id, "object": "chat.completion", "created": created, "model": model,
            "choices": [{"index": 0, "message": msg, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        try:
            # Validate JSON serializability
            json.dumps(response_body, ensure_ascii=False)
        except (TypeError, ValueError) as serr:
            print(f"[nonstream] JSON serialization failed: {serr}")
            # Sanitize: replace non-serializable values with their string repr
            safe_msg = {}
            for k, v in msg.items():
                try:
                    json.dumps({k: v}, ensure_ascii=False)
                    safe_msg[k] = v
                except (TypeError, ValueError):
                    safe_msg[k] = str(v)
            response_body["choices"][0]["message"] = safe_msg
        return JSONResponse(response_body)

    if stream:
        return StreamingResponse(do_stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})
    return do_nonstream()


def _do_chat_stream_only(cfg, prompt, model, thinking_enabled, search_enabled, has_tools=False, tools=None, ref_file_ids=None):
    """Token 刷新重试专用的流式生成器"""
    result = _do_chat(cfg, prompt, model, thinking_enabled, search_enabled, stream=True, is_retry=True, has_tools=has_tools, tools=tools, ref_file_ids=ref_file_ids)
    if isinstance(result, StreamingResponse):
        yield from result.body_iterator
    else:
        yield f"data: {json.dumps({'error': {'message': 'Retry returned non-stream', 'type': 'server_error'}})}\n\n"
        yield "data: [DONE]\n\n"


# ── 路由挂载 ─────────────────────────────────────
from app.anthropic_routes import router as _anthropic_router
app.include_router(_anthropic_router)

# ── 启动 ─────────────────────────────────────────────────
if __name__ == "__main__":
    import os as _anthropic_os
    import uvicorn
    anthropic_init_batch_storage(_anthropic_os.path.join(_anthropic_os.path.dirname(_anthropic_os.path.abspath(__file__)), ".anthropic_batches"))
    print(f" Anthropic: /v1/messages, /v1/messages/count_tokens, /v1/messages/batches, /v1/messages/{{id}}")
    print(f"DeepSeek Proxy\n Admin: http://localhost:{PROXY_PORT}/admin\n API: http://localhost:{PROXY_PORT}/v1")
    uvicorn.run(app, host="0.0.0.0", port=PROXY_PORT, log_level="info")
