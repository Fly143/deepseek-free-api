import asyncio
import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app.config as config
import app.lock_guard as guard
import proxy


class Response:
    status_code = 200
    headers = {"content-type": "text/event-stream"}
    text = ""

    def __init__(self, body):
        self.body = body

    def iter_content(self, chunk_size=4096):
        yield self.body


class StreamGuardTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        original = config.CONFIG_FILE
        config.CONFIG_FILE = Path(self.tmp.name) / "config.json"
        self.manager = config.ConfigManager()
        config.CONFIG_FILE = original
        self.account = config.DsAccount(account_label="test", login_type="email", token="token", session_id="session", is_valid=True)
        self.manager.add_account(self.account)
        self.patches = [
            patch.object(proxy, "config_manager", self.manager),
            patch.object(proxy, "get_pow_response", return_value=""),
            patch.object(guard, "MIN_INTERVAL_SEC", 0),
            patch.object(guard, "_slots", threading.Semaphore(1)),
        ]
        for item in self.patches:
            item.start()

    async def asyncTearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.tmp.cleanup()

    def response(self):
        cfg = {"token": "token", "session_id": "session", "headers": {}, "account_label": "test"}
        return proxy._do_chat(cfg, "hello", "deepseek-default", False, False, True)

    async def consume(self, response):
        return "".join([chunk async for chunk in response.body_iterator])

    async def test_waiting_stream_rechecks_cooldown(self):
        entered = threading.Event()
        release = threading.Event()
        calls = []

        def post(*args, **kwargs):
            calls.append(1)
            entered.set()
            release.wait(5)
            return Response(b'data: {"p":"response/content","v":"ok"}\n\ndata: [DONE]\n\n')

        with patch.object(proxy.cffi_requests, "post", side_effect=post):
            first = asyncio.create_task(self.consume(self.response()))
            self.assertTrue(await asyncio.to_thread(entered.wait, 2))
            second = asyncio.create_task(self.consume(self.response()))
            await asyncio.sleep(0.1)
            self.assertEqual(len(calls), 1)
            self.manager.mark_account_cooldown("test", 60)
            release.set()
            first_result, second_result = await asyncio.wait_for(asyncio.gather(first, second), 5)
            self.assertIn("ok", first_result)
            self.assertIn("account_unavailable", second_result)
            self.assertEqual(len(calls), 1)
            self.assertGreater(self.account.cooldown_until, __import__("time").time())

    async def test_error_sets_cooldown(self):
        body = json.dumps({"v": {"type": "error", "content": "上游失败", "finish_reason": "error"}}).encode()
        with patch.object(proxy.cffi_requests, "post", return_value=Response(b"data: " + body + b"\n\n")):
            output = await asyncio.wait_for(self.consume(self.response()), 5)
        self.assertIn('"error"', output)
        self.assertGreater(self.account.cooldown_until, __import__("time").time())

    async def test_success_clears_previous_errors(self):
        self.account.consecutive_errors = 2
        with patch.object(proxy.cffi_requests, "post", return_value=Response(b'data: {"p":"response/content","v":"ok"}\n\ndata: [DONE]\n\n')):
            output = await asyncio.wait_for(self.consume(self.response()), 5)
        self.assertIn("ok", output)
        self.assertEqual(self.account.consecutive_errors, 0)


if __name__ == "__main__":
    unittest.main()
