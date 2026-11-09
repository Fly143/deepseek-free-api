import asyncio
import json
import unittest
import threading
import time
from contextlib import ExitStack, contextmanager, nullcontext
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

import proxy
from app import anthropic_routes


class Request:
    headers = {}

    def __init__(self, body):
        self.body = body

    async def json(self):
        return self.body


class AnthropicRequestSessionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.account = SimpleNamespace(token="token", session_id="old", headers={}, cookie="", account_label="a", login_type="email", _password="", _email="", _mobile="", _area_code="")
        self.account.is_valid = True
        self.account.muted_until = 0
        self.account.cooldown_until = 0
        manager = SimpleNamespace(get_next_account=lambda: self.account, get_account_by_label=lambda label: self.account, lock=threading.RLock())
        self.stack.enter_context(patch.object(proxy, "config_manager", manager))
        self.stack.enter_context(patch.object(proxy, "get_models", return_value={"deepseek-default": (False, False, 10000, 1000), "deepseek-vision": (False, False, 10000, 1000)}))
        self.stack.enter_context(patch.object(proxy, "add_usage"))
        self.stack.enter_context(patch.object(proxy, "upstream_slot", side_effect=nullcontext))
        self.calls = []
        self.count = 0

        def create(cfg):
            self.count += 1
            return dict(cfg, session_id=f"new-{self.count}")

        self.stack.enter_context(patch.object(proxy, "create_request_session", side_effect=create))

        def chat(cfg, prompt, model, *args, **kwargs):
            self.calls.append((cfg["session_id"], prompt, kwargs.get("ref_file_ids")))
            if kwargs.get("stream"):
                async def generate():
                    yield 'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":null}]}\n\n'
                    yield 'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
                    yield 'data: [DONE]\n\n'
                return StreamingResponse(generate())
            return JSONResponse({"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}], "usage": {}})

        self.stack.enter_context(patch.object(proxy, "_do_chat", side_effect=chat))

    async def test_each_request_keeps_history_and_uses_new_session(self):
        body = {"model": "deepseek-default", "messages": [{"role": "user", "content": "first"}, {"role": "assistant", "content": "answer"}, {"role": "user", "content": "second"}]}
        await anthropic_routes.anthropic_messages(Request(body))
        await anthropic_routes.anthropic_messages(Request(body))
        self.assertEqual([item[0] for item in self.calls], ["new-1", "new-2"])
        self.assertEqual(self.account.session_id, "old")
        for text in ("first", "answer", "second"):
            self.assertIn(text, self.calls[0][1])

    async def test_attachment_uses_same_session_as_chat(self):
        seen = []
        image = {"data": b"image", "filename": "a.png", "content_type": "image/png"}
        with patch.object(proxy, "extract_images_from_messages", return_value=[image]), patch.object(proxy, "upload_file_to_deepseek", side_effect=lambda *a, cfg, **k: seen.append(cfg["session_id"]) or "file"), patch.object(proxy, "fork_file_to_vision", side_effect=lambda cfg, fid: seen.append(cfg["session_id"]) or "fork"), patch.object(proxy, "wait_for_file_parsing", side_effect=lambda cfg, ids, timeout: seen.append(cfg["session_id"]) or ids):
            await anthropic_routes.anthropic_messages(Request({"model": "deepseek-vision", "messages": [{"role": "user", "content": "look"}]}))
        self.assertEqual(seen, ["new-1", "new-1", "new-1"])
        self.assertEqual(self.calls[0][0], "new-1")
        self.assertEqual(self.calls[0][2], ["fork"])

    async def test_creation_failure_stops_before_upstream_chat(self):
        with patch.object(proxy, "create_request_session", side_effect=HTTPException(502, "failed")), patch.object(proxy, "upload_file_to_deepseek") as upload:
            with self.assertRaises(HTTPException):
                await anthropic_routes.anthropic_messages(Request({"messages": [{"role": "user", "content": "hello"}]}))
        upload.assert_not_called()
        self.assertEqual(self.calls, [])

    async def test_nonstream_rechecks_account_after_waiting_for_slot(self):
        @contextmanager
        def slot():
            self.account.cooldown_until = time.time() + 60
            yield

        with patch.object(proxy, "upstream_slot", slot):
            with self.assertRaises(HTTPException) as error:
                await anthropic_routes.anthropic_messages(Request({"messages": [{"role": "user", "content": "hello"}]}))
        self.assertEqual(error.exception.status_code, 503)
        self.assertEqual(self.calls, [])

    async def test_stream_uses_new_session(self):
        response = await anthropic_routes.anthropic_messages(Request({"stream": True, "messages": [{"role": "user", "content": "hello"}]}))
        output = ''.join([chunk async for chunk in response.body_iterator])
        self.assertIn("ok", output)
        self.assertEqual(self.calls[0][0], "new-1")

    async def test_batch_items_create_distinct_sessions(self):
        finished = asyncio.Event()
        results = []

        async def process(batch_id, fn):
            try:
                results.append(await fn({"params": {"messages": [{"role": "user", "content": "first"}]}}))
                results.append(await fn({"body": {"messages": [{"role": "user", "content": "second"}]}}))
            finally:
                finished.set()

        with patch.object(anthropic_routes, "_anthropic_create_batch", return_value={"id": "batch"}), patch.object(anthropic_routes, "_anthropic_process_batch_requests", side_effect=process):
            await anthropic_routes.anthropic_create_batch_ep(Request({"requests": []}))
            await asyncio.wait_for(finished.wait(), 3)
        self.assertEqual(len(results), 2)
        self.assertEqual([item[0] for item in self.calls], ["new-1", "new-2"])
        self.assertIn("first", self.calls[0][1])
        self.assertIn("second", self.calls[1][1])


if __name__ == "__main__":
    unittest.main()
