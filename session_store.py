"""会话管理 — DeepSeek session 超限自动续期

与 MiMo2API 不同，DeepSeek 已有固定的 session_id（登录时创建一直用），
不需要指纹匹配。只需要：
1. 追踪累计 prompt_tokens
2. 超限时标记需要新 session

参考实现：
  GoblinHonest/mimo2api_mimoapi — session.ts / session-marker.ts
  (https://github.com/GoblinHonest/mimo2api_mimoapi)
"""

import json
import time
from pathlib import Path

SESSION_FILE = Path(__file__).parent / "sessions.json"

# DeepSeek V4 系列 1M 上下文，留 10% 余量
TOKEN_THRESHOLD = 900_000


def _load() -> dict:
    if not SESSION_FILE.exists():
        return {}
    try:
        return json.loads(SESSION_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save(data: dict) -> None:
    SESSION_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2))


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
    """新建 session 时重置 token 计数。"""
    db = _load()
    key = f"ds_{account_id}"
    now = time.time()
    db[key] = {
        "session_id": session_id,
        "prompt_tokens": 0,
        "model": model,
        "created": now,
        "last_used": now,
    }
    _save(db)


def add_tokens(account_id: str, session_id: str, prompt_tokens: int) -> None:
    """累加 prompt_tokens。

    如果该 account 尚无 session 记录，自动初始化（首次使用）。
    """
    if not prompt_tokens:
        return
    db = _load()
    key = f"ds_{account_id}"
    s = db.get(key, {})
    if s.get("session_id") == session_id:
        # 正常续接：累加 token
        s["prompt_tokens"] = s.get("prompt_tokens", 0) + prompt_tokens
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
