"""Lock-avoidance helpers: global pacing + concurrent cap for upstream chat."""
from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager

# Conservative defaults for agent clients (ZCode/OpenCode) that burst tool calls.
# Raise interval / lower concurrency further if accounts still get muted.
MIN_INTERVAL_SEC = float(os.environ.get("DEEPSEEK_MIN_INTERVAL_SEC", "2.5"))
MAX_CONCURRENT = max(1, int(os.environ.get("DEEPSEEK_MAX_CONCURRENT", "1")))
ERROR_COOLDOWN_SEC = float(os.environ.get("DEEPSEEK_ERROR_COOLDOWN_SEC", "60"))
# Cooldown after a failed auto-relogin (session invalidated / auto-logout).
# Longer than ERROR_COOLDOWN_SEC so risk flags can cool down before the next
# attempt; keeps the account "alive" so it self-heals instead of going invalid.
SESSION_INVALID_COOLDOWN_SEC = float(os.environ.get("DEEPSEEK_SESSION_INVALID_COOLDOWN_SEC", "300"))

# 等待上游槽位的上限（秒）：槽被卡死的流占住时，后来者快速失败而不是永久挂起。
SLOT_WAIT_TIMEOUT_SEC = float(os.environ.get("DEEPSEEK_SLOT_WAIT_TIMEOUT_SEC", "600"))


class UpstreamBusy(RuntimeError):
    """上游通道长时间不可用（槽位等待超时）。"""


_lock = threading.RLock()
_slots = threading.Semaphore(MAX_CONCURRENT)
_last_start = 0.0


@contextmanager
def upstream_slot():
    """Serialize/pace calls to chat.deepseek.com to reduce mute/ban signals."""
    global _last_start
    if not _slots.acquire(timeout=SLOT_WAIT_TIMEOUT_SEC):
        raise UpstreamBusy(f"上游通道繁忙（等待超过 {SLOT_WAIT_TIMEOUT_SEC:.0f}s），请稍后重试")
    try:
        # Sleep OUTSIDE the lock so concurrent waiters are not serialized on sleep.
        while True:
            with _lock:
                now = time.time()
                wait = (_last_start + MIN_INTERVAL_SEC) - now
                if wait <= 0:
                    _last_start = time.time()
                    break
            time.sleep(min(wait, MIN_INTERVAL_SEC))
        yield
    finally:
        _slots.release()
