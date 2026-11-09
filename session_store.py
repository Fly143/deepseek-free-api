"""会话管理 — DeepSeek session 超限自动续期 + 旧会话跟踪

DeepSeek 已有固定的 session_id（登录时创建），每次 token 超限或
vision 模式会创建新 session。此模块追踪新旧 session 以便后续清理。

- 当前 session：token 累计 + 超限续期
- 旧 session：记录被替换的 session_id + 时间，供定时清理
"""

import atexit
import json
import os
import threading
import time
from pathlib import Path

SESSION_FILE = Path(__file__).parent / "sessions.json"

# DeepSeek V4 系列 1M 上下文，留 10% 余量
TOKEN_THRESHOLD = 900_000
# 旧 session 保留天数（超期后清理）
SESSION_TTL_DAYS = 3


def _load() -> dict:
    if not SESSION_FILE.exists():
        return {}
    try:
        return json.loads(SESSION_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return {}


_file_lock = threading.RLock()


def _save(data: dict) -> None:
    """原子写入：先写临时文件再替换，避免并发读/写读到半个文件。"""
    with _file_lock:  # RLock：允许在已持锁的调用方内部重入
        tmp = SESSION_FILE.with_name(SESSION_FILE.name + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, SESSION_FILE)


def needs_renewal(account_id: str = "default") -> bool:
    """检查当前 session 是否需要续期（token 超限）。"""
    db = _load()
    key = f"ds_{account_id}"
    s = db.get(key, {})
    return s.get("prompt_tokens", 0) > TOKEN_THRESHOLD


def get_usage_status(account_id: str = "default") -> dict:
    """返回当前 session 的用量状态。"""
    db = _load()
    key = f"ds_{account_id}"
    s = db.get(key, {})
    return {
        "prompt_tokens": s.get("prompt_tokens", 0),
        "threshold": TOKEN_THRESHOLD,
        "remaining": max(0, TOKEN_THRESHOLD - s.get("prompt_tokens", 0)),
    }


def on_new_session(account_id: str, session_id: str, model: str) -> None:
    """新建 session 时重置 token 计数。

    如果之前有旧 session，自动归档。
    """
    db = _load()
    key = f"ds_{account_id}"
    old = db.get(key, {})
    now = time.time()

    # 归档旧 session
    old_sid = old.get("session_id")
    if old_sid and old_sid != session_id:
        old_sessions = old.get("old_sessions", [])
        old_sessions.append({
            "session_id": old_sid,
            "model": old.get("model", ""),
            "prompt_tokens": old.get("prompt_tokens", 0),
            "created": old.get("created", now),
            "last_used": old.get("last_used", now),
            "replaced_at": now,
        })
        # 限制旧 session 记录数
        if len(old_sessions) > 200:
            old_sessions = old_sessions[-200:]
        db[key] = {
            "session_id": session_id,
            "prompt_tokens": 0,
            "model": model,
            "created": now,
            "last_used": now,
            "old_sessions": old_sessions,
        }
    else:
        db[key] = {
            "session_id": session_id,
            "prompt_tokens": 0,
            "model": model,
            "created": now,
            "last_used": now,
        }

    _save(db)


def add_tokens(account_id: str, session_id: str, prompt_tokens: int) -> None:
    """记录会话的上下文 token 峰值。

    传入的 prompt_tokens 是「本次请求的完整上下文长度」（已含历史），并非增量；
    累加会随轮次线性放大，导致真实上下文远未达上限就触碰 TOKEN_THRESHOLD
    而提前续期。这里取峰值。
    """
    if not prompt_tokens:
        return
    db = _load()
    key = f"ds_{account_id}"
    s = db.get(key, {})
    if s.get("session_id") == session_id:
        s["prompt_tokens"] = max(s.get("prompt_tokens", 0), prompt_tokens)
        s["last_used"] = time.time()
    else:
        # 新 session 或首次使用：初始化
        now = time.time()
        s = {
            "session_id": session_id,
            "prompt_tokens": prompt_tokens,
            "model": "",
            "created": now,
            "last_used": now,
        }
    _save({**db, key: s})


# ── 请求级会话登记（每次聊天新建，不参与 token 计数 / needs_renewal）──
TRANSIENT_MAX = 2000
_transient_buffer: list = []
_transient_lock = threading.Lock()


def _flush_transient_locked(batch: list) -> None:
    """在文件锁内做整文件读改写，避免并发 flush 相互覆盖。"""
    with _file_lock:
        db = _load()
        items = db.get("transient_sessions") or []
        items.extend(batch)
        if len(items) > TRANSIENT_MAX:
            # 只保留最新的记录；更旧的会话已无法追踪，打日志便于排查
            print(f"[Session] transient 记录超过上限 {TRANSIENT_MAX}，丢弃最旧 {len(items) - TRANSIENT_MAX} 条")
        db["transient_sessions"] = items[-TRANSIENT_MAX:]
        _save(db)


def register_transient_session(account_id: str, session_id: str) -> None:
    """登记请求级会话；攒够 20 条再落盘，避免每个请求都写 sessions.json。"""
    with _transient_lock:
        _transient_buffer.append({"account": account_id, "session_id": session_id, "created": time.time()})
        if len(_transient_buffer) < 20:
            return
        batch = _transient_buffer[:]
        _transient_buffer.clear()
    _flush_transient_locked(batch)


def flush_transient_sessions() -> None:
    """把缓冲中的请求级会话立即写盘（进程退出/测试用）。"""
    with _transient_lock:
        if not _transient_buffer:
            return
        batch = _transient_buffer[:]
        _transient_buffer.clear()
    _flush_transient_locked(batch)


def get_expired_transient_sessions(ttl_days: int = SESSION_TTL_DAYS) -> list:
    """返回超过 TTL 的请求级会话 [(account, session_id, "", days_ago), ...]。"""
    db = _load()
    now = time.time()
    threshold = ttl_days * 86400
    out = []
    for item in db.get("transient_sessions") or []:
        created = item.get("created")
        created = now if created is None else float(created)
        age = now - created
        if age > threshold:
            out.append((item.get("account", ""), item.get("session_id", ""), "", round(age / 86400, 1)))
    return out


def remove_transient_session(session_id: str) -> None:
    with _file_lock:
        db = _load()
        items = db.get("transient_sessions") or []
        db["transient_sessions"] = [i for i in items if i.get("session_id") != session_id]
        _save(db)


atexit.register(flush_transient_sessions)


def get_expired_sessions(account_id: str = None, ttl_days: int = SESSION_TTL_DAYS) -> list:
    """获取过期的旧 session 列表。

    Args:
        account_id: None=所有账号, str=指定账号
        ttl_days: 多少天未使用算过期

    Returns:
        [(account_label, session_id, model, last_used_days_ago), ...]
    """
    db = _load()
    now = time.time()
    threshold = ttl_days * 86400
    expired = []

    keys = [f"ds_{account_id}"] if account_id else list(db.keys())
    for key in keys:
        if not key.startswith("ds_"):
            continue
        s = db.get(key, {})
        account_label = key[3:]  # strip "ds_" prefix

        # 检查旧 session
        for old in s.get("old_sessions", []):
            age = now - old.get("last_used", old.get("replaced_at", now))
            if age > threshold:
                expired.append((
                    account_label,
                    old["session_id"],
                    old.get("model", ""),
                    round(age / 86400, 1),
                ))

    return expired


def remove_old_session(account_id: str, session_id: str) -> None:
    """从存储中移除指定的旧 session 记录。"""
    db = _load()
    key = f"ds_{account_id}"
    s = db.get(key, {})
    if not s:
        return
    old_sessions = s.get("old_sessions", [])
    s["old_sessions"] = [o for o in old_sessions if o.get("session_id") != session_id]
    _save({**db, key: s})


def get_current_session_id(account_id: str) -> str:
    """获取当前活跃的 session_id。"""
    db = _load()
    key = f"ds_{account_id}"
    return db.get(key, {}).get("session_id", "")
