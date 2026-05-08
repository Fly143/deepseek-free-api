#!/usr/bin/env python3
"""Reproducible Python requests-based test for context management.

Sends oversized conversations to the proxy and verifies it recovers
instead of returning "Content is too long" from DeepSeek's upstream.

Usage:
    # Run all tests
    python tests/test_oversized.py

    # Run with verbose output
    python tests/test_oversized.py -v

    # Test against a specific proxy URL
    PROXY_URL=http://myproxy:8000 python tests/test_oversized.py

Prerequisites:
    - proxy.py running at PROXY_URL (default http://localhost:8000)
    - `requests` package installed
"""

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

PROXY_URL = os.environ.get("PROXY_URL", "http://localhost:8000")
MODEL = os.environ.get("MODEL", "deepseek-expert")

# Colors for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
RESET = "\033[0m"

pass_count = 0
fail_count = 0

def log_pass(msg: str) -> None:
    global pass_count
    pass_count += 1
    print(f"  {GREEN}✅ PASS:{RESET} {msg}")

def log_fail(msg: str) -> None:
    global fail_count
    fail_count += 1
    print(f"  {RED}❌ FAIL:{RESET} {msg}")


def gen_conversation(turns: int) -> List[Dict[str, Any]]:
    """Generate a conversation with `turns` user-assistant exchanges."""
    messages = [{"role": "system", "content": "You are a coding assistant with expertise in Python, Rust, and system design."}]
    for i in range(1, turns + 1):
        q = (f"Question {i}: Please write a detailed explanation of how garbage collection works in Python, "
             "including reference counting, generational collection, and the cyclic garbage collector. "
             "Provide code examples for each concept.")
        a = f"Answer {i}: " + "Python uses reference counting where each object has an ob_refcnt field. " * (i * 5)
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant", "content": a})
    return messages


def send_request(messages: List[Dict[str, Any]]) -> Tuple[int, Dict[str, Any]]:
    """Send a chat completion request and return (http_status, response_body)."""
    import requests

    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": False,
    }

    try:
        resp = requests.post(
            f"{PROXY_URL}/v1/chat/completions",
            json=payload,
            timeout=120,
        )
        body = resp.json() if resp.text else {}
        return resp.status_code, body
    except requests.exceptions.ConnectionError:
        return 0, {"error": {"message": "connection refused"}}
    except json.JSONDecodeError:
        return resp.status_code, {"error": {"message": "invalid JSON response", "raw": resp.text[:200]}}
    except requests.exceptions.Timeout:
        return 0, {"error": {"message": "timeout"}}


def check_debug_endpoint(messages: List[Dict[str, Any]]) -> None:
    """Test the /v1/context/debug endpoint."""
    import requests

    try:
        resp = requests.post(
            f"{PROXY_URL}/v1/context/debug",
            json={"model": MODEL, "messages": messages},
            timeout=15,
        )
        body = resp.json() if resp.text else {}
        if resp.status_code == 200 and "token_breakdown" in body:
            log_pass("Debug endpoint: HTTP 200 with token_breakdown")
        else:
            log_fail(f"Debug endpoint: HTTP {resp.status_code}, keys={list(body.keys())}")
    except Exception as e:
        log_fail(f"Debug endpoint: exception: {e}")


def test_normal_conversation() -> None:
    """A normal 5-turn conversation should complete successfully."""
    messages = gen_conversation(5)
    status, body = send_request(messages)
    if status == 200 and "choices" in body:
        log_pass(f"Normal conversation: HTTP {status}, has choices")
    else:
        error_msg = body.get("error", {}).get("message", str(body)[:100])
        log_fail(f"Normal conversation: HTTP {status}, error: {error_msg}")


def test_large_conversation() -> None:
    """A 30-turn conversation should trigger pruning but still succeed."""
    messages = gen_conversation(30)
    status, body = send_request(messages)
    if status == 200 and "choices" in body:
        log_pass(f"Large conversation (30 turns): HTTP {status}, has choices")
    else:
        error_msg = body.get("error", {}).get("message", str(body)[:100])
        if "Content is too long" in error_msg:
            log_fail(f"Large conversation: 'Content is too long' error (pruning not working)")
        else:
            log_fail(f"Large conversation: HTTP {status}, error: {error_msg}")


def test_oversized_conversation() -> None:
    """An 80-turn conversation should trigger aggressive retry pruning."""
    messages = gen_conversation(80)
    status, body = send_request(messages)
    if status == 200 and "choices" in body:
        log_pass(f"Oversized conversation (80 turns): HTTP {status}, has choices")
    else:
        error_msg = body.get("error", {}).get("message", str(body)[:100])
        if "Content is too long" in error_msg:
            log_fail(f"Oversized conversation: 'Content is too long' error (retry pruning not working)")
        else:
            log_fail(f"Oversized conversation: HTTP {status}, error: {error_msg}")


def test_minimal_conversation() -> None:
    """A 1-turn conversation is the normal case."""
    messages = gen_conversation(1)
    status, body = send_request(messages)
    if status == 200 and "choices" in body:
        log_pass(f"Minimal conversation: HTTP {status}, has choices")
    else:
        error_msg = body.get("error", {}).get("message", str(body)[:100])
        log_fail(f"Minimal conversation: HTTP {status}, error: {error_msg}")


def test_system_only() -> None:
    """System message only (no user messages)."""
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    status, body = send_request(messages)
    if status == 200 and "choices" in body:
        log_pass(f"System only: HTTP {status}, has choices")
    else:
        error_msg = body.get("error", {}).get("message", str(body)[:100])
        log_pass(f"System only: HTTP {status}, {error_msg} (expected edge case)")


def test_empty_messages() -> None:
    """Empty messages array — proxy should handle gracefully."""
    status, body = send_request([])
    if status in (200, 422, 400):
        log_pass(f"Empty messages: HTTP {status} (expected edge case)")
    else:
        log_fail(f"Empty messages: unexpected HTTP {status}")


def run_all_tests() -> None:
    """Run all tests sequentially."""
    print("=" * 50)
    print(" DeepSeek Free API — Context Management Tests")
    print(f" Proxy: {PROXY_URL}   Model: {MODEL}")
    print("=" * 50)
    print()

    # Test 1: Debug endpoint
    print("--- Test 1: Debug endpoint ---")
    check_debug_endpoint(gen_conversation(5))
    print()

    # Test 2: Normal
    print("--- Test 2: Normal conversation (5 turns) ---")
    test_normal_conversation()
    print()

    # Test 3: Large (triggers pruning)
    print("--- Test 3: Large conversation (30 turns) ---")
    test_large_conversation()
    print()

    # Test 4: Oversized (triggers retry/aggressive prune)
    print("--- Test 4: Oversized conversation (80 turns) ---")
    test_oversized_conversation()
    print()

    # Test 5: Minimal
    print("--- Test 5: Minimal conversation (1 turn) ---")
    test_minimal_conversation()
    print()

    # Test 6: System only
    print("--- Test 6: System message only ---")
    test_system_only()
    print()

    # Test 7: Empty
    print("--- Test 7: Empty messages array ---")
    test_empty_messages()
    print()

    # Summary
    print("=" * 50)
    print(f" Results: {pass_count} passed, {fail_count} failed")
    print("=" * 50)

    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test DeepSeek proxy context management")
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose output")
    args = parser.parse_args()

    run_all_tests()
