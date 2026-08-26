"""批量注册 DeepSeek 账号 + 自动登录入池

注册流程（对应 platform.deepseek.com 的 auth-api，即用户提供的两个接口）：

1. POST /auth-api/v0/users/create_guest_challenge
   body: {"target_path": "/api/v0/users/register"}
   -> 返回 DeepSeekHashV1 PoW challenge（algorithm/challenge/salt/signature/difficulty/expire_at）

2. 求解 challenge（复用 pow_native.DeepSeekPOW），结果 base64 后放入请求头
   x-ds-guest-pow-response（注意：不是聊天接口用的 x-ds-pow-response）

3. POST /auth-api/v0/users/create_email_verification_code
   body: {"email", "turnstile_token": "", "locale", "device_id", "scenario": "register"}
   -> biz_msg=SENT_EMAIL_SUCCESS，biz_data.send_window_secs 为同邮箱发送间隔

4. 读取邮箱验证码：IMAP 收件箱 / 临时邮箱 JSON API / 手动输入（UI 提交）

5. POST /auth-api/v0/users/register
   body: {"region", "locale", "device_id",
          "payload": {"email", "email_verification_code", "password"}, "os"}
   头: x-ds-guest-pow-response
   -> 注册成功直接返回 data.biz_data.user.token（账号即已登录）

6. 自动登录：用 token 创建聊天 session，把账号写入 config.json 多账号池，
   之后代理的轮询负载均衡会自动使用该账号；密码同时保存，401 时可自动刷新。

注意：
- 注册接口对大陆 IP 直接返回 biz_code=6 REGISTER_FROM_MAINLAND，
  需要在管理面板配置海外代理（设置 → 代理配置）后由代理转发请求。
- 密码要求至少 8 位且包含字母和数字（DeepSeek 规则）。
"""

import base64
import imaplib
import json
import os
import random
import re
import secrets
import threading
import time
from email import policy
from email.parser import BytesParser
from typing import Any, Dict, List, Optional, Tuple

from curl_cffi import requests as cffi_requests

from app.config import config_manager, DsAccount
from app.device_ids import get_device_id as _ds_real_device_id
from pow_native import DeepSeekPOW

AUTH_BASE = "https://platform.deepseek.com/auth-api/v0/users"
# 与 create_guest_challenge 的 target_path 保持一致
REGISTER_TARGET = "/api/v0/users/register"
CHAT_BASE = "https://chat.deepseek.com"
MAILCX_BASE = "https://api.mail.cx/v1"

_pow_solver = DeepSeekPOW()

_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36 Edg/151.0.0.0")

# 随机 iOS 客户端指纹池：每次请求随机组合，避免固定指纹被 WAF/风控聚类识别
_IOS_UAS = [
    "DeepSeek/2 CFNetwork/1568.100.1 Darwin/24.0.0",
    "DeepSeek/2.1 CFNetwork/1568.100.1 Darwin/24.0.0",
    "DeepSeek/2 CFNetwork/1490.0.4 Darwin/23.4.0",
    "DeepSeek/2 CFNetwork/1410.0.3 Darwin/22.4.0",
    "DeepSeek/2.0 CFNetwork/1568.100.1 Darwin/24.1.0",
    "DeepSeek/2 CFNetwork/1474.4 Darwin/23.2.0",
]
_IOS_VERSIONS = ["2.0.4", "2.0.2", "2.1.0", "2.0.1", "1.10.1", "2.0.0"]
_IOS_TZ_OFFSETS = ["28800", "3600", "0", "7200", "32400", "10800", "-14400", "25200"]


def _random_ios_headers(locale: str = "zh_CN") -> Dict[str, str]:
    """生成随机 iOS 客户端指纹请求头。UA/版本/时区/rangers-id 每次随机。"""
    return {
        "accept": "application/json",
        "content-type": "application/json",
        "user-agent": random.choice(_IOS_UAS),
        "x-client-bundle-id": "com.deepseek.chat",
        "x-client-locale": locale,
        "x-client-platform": "ios",
        "x-client-timezone-offset": random.choice(_IOS_TZ_OFFSETS),
        "x-client-version": random.choice(_IOS_VERSIONS),
        "x-rangers-id": str(random.randint(10 ** 18, 10 ** 19 - 1)),
    }


def random_ios_headers(locale: str = "zh_CN") -> Dict[str, str]:
    """公开的随机 iOS 指纹生成器（供 proxy.py 登录复用，每次随机组合）。"""
    return _random_ios_headers(locale)


# ── 基础工具 ────────────────────────────────────────────────

def new_device_id() -> str:
    """生成 device_id。

    优先从真实 iOS 设备指纹库随机选取（aiodeepseek 收集的已知设备 ID，
    可显著降低 DeepSeek 风控 RISK_DEVICE_DETECTED 拦截概率），
    库不可用时回退为 32 字节随机数 base64。
    """
    try:
        return _ds_real_device_id()
    except Exception:
        return base64.b64encode(os.urandom(32)).decode()


def generate_password(length: int = 14) -> str:
    """生成随机密码（大小写字母+数字，满足 DeepSeek 至少 8 位且含字母+数字的规则）。"""
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    while True:
        pw = "".join(secrets.choice(alphabet) for _ in range(length))
        if re.search(r"[A-Za-z]", pw) and re.search(r"\d", pw):
            return pw


def generate_local_part(length: int = 10) -> str:
    """生成满足 mail.cx 本地部分规则的随机邮箱前缀（小写字母+数字，字母开头，2-20 位）。"""
    letters = "abcdefghijklmnopqrstuvwxyz"
    digits = "0123456789"
    n = max(2, min(20, int(length)))
    head = secrets.choice(letters)
    body = "".join(secrets.choice(letters + digits) for _ in range(n - 1))
    return head + body


def _proxy() -> Optional[dict]:
    """读取管理面板配置的代理（绕过 WAF / 大陆 IP 注册限制）。"""
    url = config_manager.get_proxy()
    if not url:
        return None
    return {"http": url, "https": url}


def _guest_headers(locale: str = "zh_CN") -> Dict[str, str]:
    """注册流程专用请求头。x-rangers-id 为 auth-api 必填头（缺失报 Missing Header）。"""
    return {
        "accept": "application/json",
        "content-type": "application/json",
        "user-agent": _UA,
        "origin": "https://platform.deepseek.com",
        "referer": "https://platform.deepseek.com/sign_up",
        "x-client-bundle-id": "com.deepseek.chat",
        "x-client-locale": locale,
        "x-client-platform": "web",
        "x-client-timezone-offset": "28800",
        "x-client-version": "1.0.0",
        "x-rangers-id": str(random.randint(10 ** 18, 10 ** 19 - 1)),
    }


def _register_headers(locale: str = "zh_CN") -> Dict[str, str]:
    """注册接口专用 iOS 客户端请求头（随机指纹，每次请求都不同，降低风控拦截）。"""
    return _random_ios_headers(locale)


def _post(path: str, body: dict, extra_headers: Optional[dict] = None,
          proxy: Optional[dict] = None, timeout: int = 30,
          ios: bool = False) -> Any:
    if ios:
        headers = _register_headers(body.get("locale", "zh_CN") if isinstance(body, dict) else "zh_CN")
    else:
        headers = _guest_headers(body.get("locale", "zh_CN") if isinstance(body, dict) else "zh_CN")
    if extra_headers:
        headers.update(extra_headers)
    return cffi_requests.post(
        AUTH_BASE + path, json=body, headers=headers,
        impersonate="chrome120", timeout=timeout, proxies=proxy,
    )


def _biz_error(data: dict, fallback: str) -> str:
    """从 auth-api 响应中提取可读错误。"""
    inner = data.get("data") or {}
    return inner.get("biz_msg") or data.get("msg") or fallback


# ── 注册流程 API ────────────────────────────────────────────

def send_verification_code(email: str, device_id: Optional[str] = None,
                           locale: str = "zh_CN", proxy: Optional[dict] = None) -> Tuple[bool, Dict]:
    """发送注册验证码。返回 (ok, {send_window_secs} 或 {error})。"""
    payload = {
        "email": email,
        "turnstile_token": "",
        "locale": locale,
        "device_id": device_id or new_device_id(),
        "scenario": "register",
    }
    try:
        resp = _post("/create_email_verification_code", payload, proxy=proxy)
        data = resp.json()
    except Exception as e:
        return False, {"error": f"请求失败: {e}"}
    if resp.status_code != 200 or data.get("code") != 0:
        return False, {"error": _biz_error(data, f"HTTP {resp.status_code}")}
    inner = data.get("data") or {}
    return True, {"send_window_secs": (inner.get("biz_data") or {}).get("send_window_secs", 60)}


def fetch_guest_challenge(target_path: str = REGISTER_TARGET,
                          proxy: Optional[dict] = None) -> Dict:
    """获取注册 PoW challenge。"""
    resp = _post("/create_guest_challenge", {"target_path": target_path}, proxy=proxy)
    data = resp.json()
    if resp.status_code != 200 or data.get("code") != 0:
        raise RuntimeError(_biz_error(data, f"HTTP {resp.status_code}"))
    ch = (data.get("data") or {}).get("biz_data") or {}
    ch = ch.get("guest_challenge")
    if not ch:
        raise RuntimeError("create_guest_challenge 未返回 challenge")
    return ch


def solve_guest_challenge(challenge: Dict, target_path: str = REGISTER_TARGET) -> str:
    """求解 challenge，返回 x-ds-guest-pow-response 头值。"""
    cfg = dict(challenge)
    cfg["target_path"] = target_path
    return _pow_solver.solve_challenge(cfg)


def register_account(email: str, password: str, code: str,
                     device_id: Optional[str] = None, region: str = "US",
                     locale: str = "zh_CN", proxy: Optional[dict] = None) -> Tuple[bool, Dict]:
    """完成注册。返回 (ok, {token, user} 或 {error})。"""
    try:
        challenge = fetch_guest_challenge(proxy=proxy)
        pow_resp = solve_guest_challenge(challenge)
    except Exception as e:
        return False, {"error": f"获取/求解 challenge 失败: {e}"}
    payload = {
        "region": region,
        "locale": locale,
        "device_id": device_id or new_device_id(),
        "payload": {
            "email": email,
            "email_verification_code": code,
            "password": password,
        },
        "os": "ios",
    }
    try:
        resp = _post("/register", payload,
                     extra_headers={"x-ds-guest-pow-response": pow_resp}, proxy=proxy,
                     ios=True)
        data = resp.json()
    except Exception as e:
        return False, {"error": f"请求失败: {e}"}
    if resp.status_code != 200 or data.get("code") != 0:
        return False, {"error": _biz_error(data, f"HTTP {resp.status_code}")}
    user = ((data.get("data") or {}).get("biz_data") or {}).get("user") or {}
    token = user.get("token", "")
    if not token:
        return False, {"error": f"注册响应中无 token: {json.dumps(data, ensure_ascii=False)[:300]}"}
    return True, {"token": token, "user": user}


# ── 自动登录并入池 ──────────────────────────────────────────

def login_chat_account(email: str, password: str,
                       proxy: Optional[dict] = None) -> Tuple[bool, str, str, str]:
    """用注册的邮箱密码在 chat.deepseek.com 登录，返回 (ok, token, session_id, error)。

    使用随机 iOS 客户端指纹（UA/版本/时区/rangers-id/device_id 每次随机），
    绕过 chat.deepseek.com 对 Web 流量触发的 AWS WAF challenge。
    """
    login_payload = {
        "email": email, "password": password,
        "device_id": new_device_id(), "os": "ios",
        "mobile": "", "area_code": "",
    }
    headers = _random_ios_headers()
    try:
        # 1. 登录（iOS 指纹无需 WAF cookie）
        login_resp = cffi_requests.post(
            "https://chat.deepseek.com/api/v0/users/login",
            json=login_payload, headers=headers, timeout=30, proxies=proxy)
        if login_resp.status_code == 202 and login_resp.headers.get("x-amzn-waf-action"):
            return False, "", "", "登录被 AWS WAF 拦截 (HTTP 202)，请配置海外代理后重试"
        raw_text = (login_resp.text or "").strip()
        if not raw_text:
            return False, "", "", f"登录返回空响应 (HTTP {login_resp.status_code})"
        try:
            login_data = login_resp.json()
        except Exception:
            return False, "", "", f"登录返回非 JSON 响应: {raw_text[:150]}"
        outer_code = login_data.get("code", 0)
        data_block = login_data.get("data") or {}
        if login_resp.status_code != 200 or outer_code != 0 or data_block.get("biz_code", 0) != 0:
            err = data_block.get("biz_msg") or login_data.get("msg") or f"HTTP {login_resp.status_code}"
            return False, "", "", f"登录失败: {err}"
        token = (data_block.get("biz_data") or {}).get("user", {}).get("token", "")
        if not token:
            return False, "", "", "登录响应中无 token"

        # 2. 创建聊天 session
        session_id = ""
        auth_headers = {**headers, "authorization": f"Bearer {token}"}
        try:
            sresp = cffi_requests.post("https://chat.deepseek.com/api/v0/chat_session/create",
                                       json={}, headers=auth_headers, timeout=15, proxies=proxy)
            if sresp.status_code == 200:
                sd = sresp.json()
                biz = (sd.get("data") or {}).get("biz_data") or {}
                session_id = (biz.get("chat_session") or {}).get("id", "") or biz.get("id", "")
        except Exception:
            pass
        return True, token, session_id, ""
    except Exception as e:
        return False, "", "", f"登录异常: {e}"


def save_registered_account(email: str, password: str, token: str,
                            session_id: str = "", is_valid: bool = True) -> None:
    """把新注册账号写入多账号池（config.json），headers 使用随机 iOS 指纹。"""
    headers = _random_ios_headers()
    headers["authorization"] = f"Bearer {token}"
    ds = DsAccount(
        account_label=email,
        login_type="email",
        _password=password,
        _email=email,
        token=token,
        session_id=session_id,
        headers=headers,
        login_time=time.strftime("%Y-%m-%d %H:%M:%S"),
        is_valid=is_valid,
    )
    config_manager.add_account(ds)
    print(f"[Registrar] 账号 {email} 已写入账号池 (session={session_id or 'N/A'}, valid={is_valid})")


# ── 验证码读取 ──────────────────────────────────────────────

CODE_RE = re.compile(r"(?<![A-Za-z0-9_])(\d{6})(?![A-Za-z0-9_])")


def _decode_part(part) -> str:
    try:
        payload = part.get_payload(decode=True)
        if payload is None:
            return ""
        charset = part.get_content_charset() or "utf-8"
        return payload.decode(charset, errors="ignore")
    except Exception:
        return ""


def extract_code_from_text(text: str) -> Optional[str]:
    """从邮件文本中提取 6 位验证码。优先关键词附近的数字，否则取最后一组独立 6 位数字。

    使用词边界匹配，避免把收件人地址/文件名中的数字当成验证码。
    """
    if not text:
        return None
    text = text.replace("\r", " ").replace("\n", " ")
    kw = r"(code|verification|verify|otp|验证码|验证|确认码|安全码|校验码)"
    for m in re.finditer(kw + r"[^\d]{0,40}?(\d{6})", text, re.I):
        return m.group(1)
    for m in re.finditer(r"(\d{6})[^\d]{0,20}?" + kw, text, re.I):
        return m.group(1)
    nums = CODE_RE.findall(text)
    return nums[-1] if nums else None


def read_code_imap(host: str, port: int, user: str, password: str, folder: str = "INBOX",
                   target_email: Optional[str] = None, seen_uids: Optional[set] = None,
                   timeout: int = 30) -> Tuple[Optional[str], Optional[str]]:
    """从 IMAP 收件箱读取最新验证码。

    返回 (code, uid)；无新邮件返回 (None, None)。seen_uids 用于跨轮询去重。
    """
    try:
        M = imaplib.IMAP4_SSL(host, int(port), timeout=timeout)
        M.login(user, password)
        M.select(folder)
        typ, data = M.search(None, "ALL")
        if typ == "OK":
            ids = data[0].split()
            for num in reversed(ids[-30:]):  # 只看最近 30 封
                uid = num.decode()
                if seen_uids is not None and uid in seen_uids:
                    continue
                typ2, msg_data = M.fetch(num, "(RFC822)")
                if typ2 != "OK" or not msg_data or msg_data[0] is None:
                    continue
                msg = BytesParser(policy=policy.default).parsebytes(msg_data[0][1])
                if target_email:
                    to = str(msg.get("To", "") or "")
                    if target_email.lower() not in to.lower():
                        continue
                text = ""
                for part in msg.walk():
                    if part.get_content_type() in ("text/plain", "text/html"):
                        text += _decode_part(part) + "\n"
                code = extract_code_from_text(str(msg.get("subject", "") or "") + "\n" + text)
                if code:
                    if seen_uids is not None:
                        seen_uids.add(uid)
                    M.logout()
                    return code, uid
        M.logout()
    except Exception as e:
        print(f"[Registrar] IMAP 读取失败: {e}")
    return None, None


def read_code_temp_api(url: str, timeout: int = 30) -> Optional[str]:
    try:
        resp = cffi_requests.get(url, impersonate="chrome120", timeout=timeout)
        text = resp.text or ""
        try:
            obj = resp.json()
        except Exception:
            obj = None

        def walk(o, depth: int = 0) -> Optional[str]:
            if depth > 6:
                return None
            if isinstance(o, dict):
                for k, v in o.items():
                    if isinstance(v, (str, int)) and re.fullmatch(r"\d{6}", str(v)):
                        return str(v)
                    r = walk(v, depth + 1)
                    if r:
                        return r
            elif isinstance(o, list):
                for v in o:
                    r = walk(v, depth + 1)
                    if r:
                        return r
            return None

        if isinstance(obj, dict):
            for key in ("verification_code", "verify_code", "email_code", "otp", "code"):
                if key in obj and obj[key]:
                    nums = re.findall(r"\d{6}", str(obj[key]))
                    if nums:
                        return nums[0]
            return walk(obj)
        nums = re.findall(r"\d{6}", text)
        return nums[0] if nums else None
    except Exception as e:
        print(f"[Registrar] 临时邮箱 API 读取失败: {e}")
        return None


def list_mailcx_domains(api_key: str) -> Tuple[bool, Any]:
    """获取 mail.cx 可用邮箱后缀（自定义已验证域名 + 系统域名）。返回 (ok, 域名列表|错误)。"""
    try:
        domains: List[str] = []
        resp = cffi_requests.get(MAILCX_BASE + "/domains", headers={"x-api-token": api_key},
                                 impersonate="chrome120", timeout=15)
        if resp.status_code == 200:
            for d in (resp.json().get("domains") or []):
                if d.get("verify_status") == "verified":
                    domains.append(d.get("domain", ""))
        resp2 = cffi_requests.get(MAILCX_BASE + "/config", impersonate="chrome120", timeout=15)
        if resp2.status_code == 200:
            for d in (resp2.json().get("system_domains") or []):
                domains.append(d.get("domain", ""))
        domains = [d for d in domains if d]
        if not domains:
            return False, "未获取到可用域名（请检查 API Key）"
        return True, domains
    except Exception as e:
        return False, f"获取域名失败: {e}"


def clear_mailcx_inbox(api_key: str, address: str) -> None:
    """清空 mail.cx 邮箱，保证验证码是全新邮件。"""
    try:
        cffi_requests.delete(MAILCX_BASE + f"/inbox/{address}", headers={"x-api-token": api_key},
                             impersonate="chrome120", timeout=15)
    except Exception as e:
        print(f"[Registrar] mail.cx 清空邮箱失败: {e}")


def read_code_mailcx(api_key: str, address: str, seen_ids: Optional[set] = None,
                     timeout: int = 35) -> Optional[str]:
    """mail.cx long-poll 读取验证码（服务端最长保持 25s，204 表示暂无新邮件）。

    返回 6 位验证码或 None。seen_ids 用于跨轮询去重。
    """
    try:
        resp = cffi_requests.get(MAILCX_BASE + f"/inbox/{address}", headers={"x-api-token": api_key},
                                 impersonate="chrome120", timeout=timeout)
        if resp.status_code != 200:
            return None
        for em in (resp.json().get("emails") or []):
            eid = em.get("id")
            if seen_ids is not None and eid in seen_ids:
                continue
            full = cffi_requests.get(MAILCX_BASE + f"/email/{eid}", headers={"x-api-token": api_key},
                                     impersonate="chrome120", timeout=15)
            if full.status_code != 200:
                continue
            fdata = full.json()
            text = (fdata.get("text_body") or "") + "\n" + re.sub(r"<[^>]+>", " ", fdata.get("html_body") or "")
            # 剔除收件人地址，避免把地址里的数字误认为验证码（to 可能是字符串或数组）
            to_field = fdata.get("to") or []
            if isinstance(to_field, str):
                to_field = [to_field]
            for to_addr in to_field:
                if isinstance(to_addr, dict):
                    to_addr = to_addr.get("email") or ""
                if to_addr:
                    text = text.replace(str(to_addr), "")
            code = extract_code_from_text(text)
            if code:
                if seen_ids is not None:
                    seen_ids.add(eid)
                return code
            if seen_ids is not None:
                seen_ids.add(eid)   # 该邮件无验证码，跳过避免重复处理
    except Exception as e:
        print(f"[Registrar] mail.cx 读取失败: {e}")
    return None


def expand_email_patterns(lines: List[str]) -> List[str]:
    """把输入行展开为邮箱列表。支持 {N-M} 或 {N-M:位数} 数字展开。

    例: "user+{1-3}@example.com" -> user+1@example.com ... user+3@example.com
         "a{01-03}@x.com"        -> a01@x.com ... a03@x.com
    """
    out: List[str] = []
    for raw in lines:
        line = str(raw).strip()
        if not line or line.startswith("#"):
            continue
        m = re.match(r"^(.*)\{(\d+)-(\d+)(?::(\d+))?\}(.*)$", line)
        if m:
            prefix, a, b, pad, suffix = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4), m.group(5)
            pad = int(pad) if pad else 0
            for i in range(a, b + 1):
                num = str(i).zfill(pad) if pad else str(i)
                out.append(prefix + num + suffix)
        else:
            out.append(line)
    return out


# ── 批量任务管理 ────────────────────────────────────────────

class BatchRegistrar:
    """批量注册任务管理器（后台线程执行，线程安全）。"""

    def __init__(self):
        self._jobs: Dict[str, dict] = {}
        self._lock = threading.RLock()
        self._seq = 0

    def start(self, emails: List[str], password: str, options: dict) -> str:
        """启动一个批量注册任务，返回 job_id。

        password 为空或 options.random_password 为真时，为每个账号随机生成独立密码；
        否则使用统一密码（需满足 DeepSeek 至少 8 位且含字母+数字）。
        """
        emails = [str(e).strip() for e in emails if str(e).strip()]
        # 配置了临时邮箱域名时，把纯本地部分补全为完整邮箱
        domain = (options.get("domain") or "").strip().lstrip("@")
        if domain:
            emails = [e if "@" in e else f"{e}@{domain}" for e in emails]
        emails = [e for e in emails if e]
        if not emails:
            raise ValueError("邮箱列表为空")
        if any("@" not in e for e in emails):
            raise ValueError("存在无效邮箱（缺少 @）")

        password = (password or "").strip()
        random_pw = bool(options.get("random_password", False)) or not password
        if not random_pw and (len(password) < 8 or not re.search(r"[A-Za-z]", password) or not re.search(r"\d", password)):
            raise ValueError("密码至少 8 位，且需同时包含字母和数字")
        # 每个账号一个密码（随机模式各自独立，统一模式共享同一密码）
        passwords = {e: (generate_password() if random_pw else password) for e in emails}

        with self._lock:
            self._seq += 1
            job_id = f"reg_{int(time.time())}_{self._seq}"
            job = {
                "id": job_id,
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "status": "running",
                "emails": emails,
                "password_masked": "随机生成" if random_pw else ((password[:2] + "***" + password[-1:]) if len(password) > 3 else "***"),
                "password_mode": "random" if random_pw else "uniform",
                "options": options,
                "_password": password,   # 私有字段，不出现在 API 返回中
                "passwords": passwords,  # email -> 密码（随机模式下每账号独立）
                "results": {e: {"status": "pending", "error": "", "token_masked": "", "elapsed": 0}
                            for e in emails},
                "logs": [],
                "codes": dict(options.get("codes") or {}),   # 预置/手动提交的验证码 {email: code}
                "seen_uids": set(),
                "seen_ids": set(),
                "stop_flag": False,
            }
            self._jobs[job_id] = job
        t = threading.Thread(target=self._worker, args=(job_id,), daemon=True)
        t.start()
        return job_id

    def get(self, job_id: str) -> Optional[dict]:
        """返回任务的 API 快照（不含明文密码等私有字段）。"""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            snap = {
                "id": job["id"],
                "created": job["created"],
                "status": job["status"],
                "emails": list(job["emails"]),
                "password_masked": job["password_masked"],
                "password_mode": job.get("password_mode", "uniform"),
                "options": dict(job["options"]),
                "results": {e: dict(r) for e, r in job["results"].items()},
                "logs": list(job["logs"]),
                "codes": dict(job["codes"]),
                "stop_flag": job["stop_flag"],
            }
            return snap

    def export_data(self, job_id: str) -> Optional[dict]:
        """导出指定任务的账号密码数据（email -> password + 状态）。"""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            return {
                "emails": list(job["emails"]),
                "passwords": dict(job["passwords"]),
                "results": {e: dict(r) for e, r in job["results"].items()},
                "password_mode": job.get("password_mode", "uniform"),
            }

    def list_jobs(self) -> List[dict]:
        with self._lock:
            return [{
                "id": j["id"], "status": j["status"], "created": j["created"],
                "total": len(j["emails"]),
                "done": sum(1 for r in j["results"].values() if r["status"] == "done"),
                "failed": sum(1 for r in j["results"].values() if r["status"] == "failed"),
            } for j in self._jobs.values()]

    def submit_code(self, job_id: str, email: str, code: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or email not in job["results"]:
                return False
            job["codes"][email] = code.strip()
            return True

    def stop(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return False
            job["stop_flag"] = True
            return True

    # ── 内部 ──
    def _log(self, job: dict, msg: str) -> None:
        job["logs"].append(f"[{time.strftime('%H:%M:%S')}] {msg}")
        if len(job["logs"]) > 200:
            job["logs"] = job["logs"][-200:]

    def _set(self, job: dict, email: str, **kw) -> None:
        r = job["results"].get(email)
        if r:
            r.update(kw)

    def _worker(self, job_id: str) -> None:
        job = self._jobs[job_id]
        opts = job["options"]
        reader = opts.get("reader") or {"mode": "manual"}
        reader_mode = reader.get("mode", "manual")
        wait_timeout = int(opts.get("wait_timeout", 180) or 180)
        poll_interval = float(opts.get("poll_interval", 5) or 5)
        proxy = _proxy()
        total = len(job["emails"])

        for idx, email in enumerate(job["emails"], 1):
            if job["stop_flag"]:
                job["status"] = "stopped"
                self._log(job, "任务已停止")
                return
            start_ts = time.time()
            try:
                self._process_one(job, idx, total, email, opts, reader, reader_mode,
                                  wait_timeout, poll_interval, proxy)
            except Exception as e:
                self._set(job, email, status="failed", error=f"内部异常: {e}",
                          elapsed=round(time.time() - start_ts, 1))
                self._log(job, f"{email} 内部异常: {e}")

        done = sum(1 for r in job["results"].values() if r["status"] == "done")
        failed = sum(1 for r in job["results"].values() if r["status"] == "failed")
        job["status"] = "done"
        self._log(job, f"批量注册完成：成功 {done}/{total}，失败 {failed}")

    def _process_one(self, job: dict, idx: int, total: int, email: str, opts: dict,
                     reader: dict, reader_mode: str, wait_timeout: int,
                     poll_interval: float, proxy: Optional[dict]) -> None:
        start_ts = time.time()
        self._log(job, f"[{idx}/{total}] 开始处理 {email}")
        device_id = new_device_id()

        # 0. mail.cx 模式：清空该邮箱旧邮件，保证验证码是全新的
        if reader_mode == "mailcx":
            mailcx_key = reader.get("api_key") or config_manager.get_mailcx_api_key()
            if mailcx_key:
                clear_mailcx_inbox(mailcx_key, email)

        # 1. 发送验证码（遇到限流 EMAIL_REQUEST_TOO_FREQUENT 等待后重试一次）
        self._set(job, email, status="sending_code")
        ok, info = send_verification_code(email, device_id=device_id,
                                          locale=opts.get("locale", "zh_CN"), proxy=proxy)
        if not ok and "TOO_FREQUENT" in str(info.get("error", "")):
            self._log(job, f"{email} 发送太频繁，等待 65s 后重试...")
            time.sleep(65)
            ok, info = send_verification_code(email, device_id=device_id,
                                              locale=opts.get("locale", "zh_CN"), proxy=proxy)
        if not ok:
            self._set(job, email, status="failed", error=info.get("error", "发送验证码失败"),
                      elapsed=round(time.time() - start_ts, 1))
            self._log(job, f"{email} 发送验证码失败: {info.get('error')}")
            return
        self._log(job, f"{email} 验证码已发送（间隔 {info.get('send_window_secs', 60)}s）")

        # 2. 获取验证码
        code = job["codes"].get(email) or ""
        if not code and reader_mode == "manual":
            self._set(job, email, status="waiting_code")
            code = self._wait_manual_code(job, email, wait_timeout)
        elif not code:
            self._set(job, email, status="waiting_code")
            code = self._read_code_loop(job, email, reader_mode, reader, wait_timeout, poll_interval)
        if job["stop_flag"]:
            job["status"] = "stopped"
            return
        if not code:
            self._set(job, email, status="failed", error="未获取到验证码（超时）",
                      elapsed=round(time.time() - start_ts, 1))
            self._log(job, f"{email} 验证码获取超时")
            return
        self._log(job, f"{email} 已获取验证码")

        # 3. 注册（注册成功即返回 token）
        self._set(job, email, status="registering")
        pw = job["passwords"].get(email) or job.get("_password") or ""
        ok, info = register_account(email, pw, code,
                                    device_id=device_id,
                                    region=opts.get("region", "US"),
                                    locale=opts.get("locale", "zh_CN"),
                                    proxy=proxy)
        if not ok:
            self._set(job, email, status="failed", error=info.get("error", "注册失败"),
                      elapsed=round(time.time() - start_ts, 1))
            self._log(job, f"{email} 注册失败: {info.get('error')}")
            return
        token = info["token"]

        # 4. 自动登录：注册 token 仅 platform 域可用，需走标准密码登录获取聊天 token
        if opts.get("auto_login", True):
            self._set(job, email, status="logging_in")
            ok2, chat_token, session_id, lerr = login_chat_account(email, pw, proxy=proxy)
            if not ok2:
                # 注册成功但登录被拦（如 WAF/IP 风控）：仍保存账号（凭证齐全），供稍后重登
                save_registered_account(email, pw, token, "", is_valid=False)
                self._set(job, email, status="failed",
                          error=f"注册成功但自动登录失败: {lerr}",
                          elapsed=round(time.time() - start_ts, 1))
                self._log(job, f"{email} 注册成功，但自动登录失败: {lerr}（账号已保存，可在账号管理中重登）")
                return
            save_registered_account(email, pw, chat_token, session_id, is_valid=True)
            masked = chat_token[:20] + "..." + chat_token[-8:]
        else:
            session_id = ""
            save_registered_account(email, pw, token, session_id, is_valid=False)
            masked = token[:20] + "..." + token[-8:]
        self._set(job, email, status="done", token_masked=masked,
                  elapsed=round(time.time() - start_ts, 1))
        self._log(job, f"{email} 注册成功，已加入账号池 Token={masked}")

        # 账号间隔，避免限流
        if idx < total:
            time.sleep(2)

    def _wait_manual_code(self, job: dict, email: str, timeout: int) -> Optional[str]:
        """manual 模式：轮询等待用户在 UI 提交验证码。"""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if job["stop_flag"]:
                return None
            code = job["codes"].get(email)
            if code:
                return code
            time.sleep(1)
        return None

    def _read_code_loop(self, job: dict, email: str, mode: str, cfg: dict,
                        timeout: int, poll_interval: float) -> Optional[str]:
        """imap / temp_api 模式：轮询读取新邮件验证码。"""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if job["stop_flag"]:
                return None
            code = job["codes"].get(email)   # 预置/手动提交优先
            if code:
                return code
            if mode == "imap":
                code, _uid = read_code_imap(
                    cfg.get("host", ""), cfg.get("port", 993),
                    cfg.get("user", ""), cfg.get("password", ""),
                    cfg.get("folder", "INBOX"), target_email=email,
                    seen_uids=job["seen_uids"])
            elif mode == "temp_api":
                code = read_code_temp_api(cfg.get("url", ""))
            elif mode == "mailcx":
                api_key = cfg.get("api_key", "") or config_manager.get_mailcx_api_key()
                code = read_code_mailcx(api_key, email,
                                        seen_ids=job["seen_ids"], timeout=30)
            else:
                return None
            if code:
                return code
            time.sleep(poll_interval)
        return None


# 全局批量注册管理器
registrar = BatchRegistrar()
