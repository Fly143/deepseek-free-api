#!/usr/bin/env python3
"""Start deepseek-free-api on 8000 (idempotent) and refresh login."""
from __future__ import annotations

import base64
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PID = ROOT / "bridge8000.pid"
OUT = ROOT / "bridge8000.out.log"
ERR = ROOT / "bridge8000.err.log"
PORT = 8000
PY = Path(os.environ.get("DEEPSEEK_BRIDGE_PYTHON") or sys.executable)


def port_open(port: int = PORT) -> bool:
    s = socket.socket()
    s.settimeout(0.5)
    try:
        s.connect(("127.0.0.1", port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
        if h:
            ctypes.windll.kernel32.CloseHandle(h)
            return True
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def relogin() -> None:
    auth = base64.b64encode(os.environ.get("DEEPSEEK_BRIDGE_ADMIN", "admin:admin").encode()).decode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/api/accounts/relogin-all",
        data=b"{}",
        method="POST",
        headers={"Authorization": f"Basic {auth}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as r:
            print("relogin", r.status, r.read()[:200].decode("utf-8", "replace"))
    except Exception as e:
        print(f"relogin_skip: {e}")


def ensure_running() -> int:
    if port_open():
        print(f"already listening on {PORT}")
        return 0
    if PID.exists():
        try:
            old = int(PID.read_text().strip())
            if alive(old) and port_open():
                print(f"already running pid={old}")
                return 0
        except Exception:
            pass
    py = str(PY if PY.exists() else sys.executable)
    flags = (0x00000008 | 0x08000000) if os.name == "nt" else 0
    with OUT.open("ab") as out, ERR.open("ab") as err:
        p = subprocess.Popen(
            [py, "-u", "proxy.py"],
            cwd=str(ROOT),
            stdout=out,
            stderr=err,
            creationflags=flags,
            close_fds=True,
        )
    PID.write_text(str(p.pid))
    for _ in range(40):
        time.sleep(0.25)
        if port_open():
            print(f"started pid={p.pid}")
            return 0
    print(f"started pid={p.pid} but port {PORT} not open yet", file=sys.stderr)
    return 1


def main() -> int:
    code = ensure_running()
    if port_open():
        relogin()
    return code


if __name__ == "__main__":
    raise SystemExit(main())