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

_lock = threading.RLock()
_slots = threading.Semaphore(MAX_CONCURRENT)
_last_start = 0.0


@contextmanager
def upstream_slot():
    """Serialize/pace calls to chat.deepseek.com to reduce mute/ban signals."""
    global _last_start
    _slots.acquire()
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
