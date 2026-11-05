import asyncio
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import app.config as config
import app.lock_guard as guard
import proxy


class UpstreamResponse:
    status_code = 200
    headers = {"content-type": "text/event-stream"}
    text = ""

    def __init__(self, parts):
        self.body = b"".join(
            b"data: " + json.dumps({"p": "response/content", "v": part}, ensure_ascii=False).encode() + b"\n\n"
            for part in parts
        ) + b"data: [DONE]\n\n"

    def iter_content(self, chunk_size=4096):
        for start in range(0, len(self.body), 13):
            yield self.body[start:start + 13]


async def request(body):
    received = False
    blocked = asyncio.Event()
    events = []

    async def receive():
        nonlocal received
        if not received:
            received = True
            return {"type": "http.request", "body": json.dumps(body, ensure_ascii=False).encode(), "more_body": False}
        await blocked.wait()

    async def send(event):
        events.append(event)

    scope = {
        "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"}, "http_version": "1.1",
        "method": "POST", "scheme": "http", "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions", "query_string": b"", "root_path": "",
        "headers": [(b"content-type", b"application/json"), (b"authorization", b"Bearer test-key")],
        "client": ("127.0.0.1", 12345), "server": ("127.0.0.1", 8000),
    }
    await asyncio.wait_for(proxy.app(scope, receive, send), timeout=5)
    status = next(e["status"] for e in events if e["type"] == "http.response.start")
    data = b"".join(e.get("body", b"") for e in events if e["type"] == "http.response.body")
    return status, data.decode()


class DoApiIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        with patch.object(config, "CONFIG_FILE", Path(self.tmp.name) / "config.json"), patch.object(config, "LEGACY_FILE", Path(self.tmp.name) / "token.json"):
            self.manager = config.ConfigManager()
        self.manager.add_account(config.DsAccount(account_label="test", login_type="email", token="token", session_id="session", is_valid=True))
        self.patches = [
            patch.object(proxy, "config_manager", self.manager),
            patch.object(proxy, "get_pow_response", return_value=""),
            patch.object(proxy, "get_models", return_value={"deepseek-default": (False, False, 64000, 8000)}),
            patch.object(proxy, "needs_renewal", return_value=False),
            patch.object(guard, "MIN_INTERVAL_SEC", 0),
            patch.object(guard, "_slots", threading.Semaphore(1)),
            patch.dict(os.environ, {"DEEPSEEK_PROXY_API_KEY": "test-key"}),
        ]
        for p in self.patches:
            p.start()

    async def asyncTearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    async def call(self, output, stream):
        tools = [{"type": "function", "function": {"name": "read", "description": "Read a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "meta": {"type": "object"}}}}}]
        body = {"model": "deepseek-default", "messages": [{"role": "user", "content": "读取文件"}], "tools": tools, "stream": stream}
        parts = [output[:1], output[1:3], output[3:8], output[8:]]
        calls = []

        def post(url, **kwargs):
            if url.endswith("/chat_session/create"):
                response = UpstreamResponse([])
                response.json = lambda: {"data": {"biz_data": {"id": "request-session"}}}
                return response
            calls.append((url, kwargs))
            self.assertEqual(url, "https://chat.deepseek.com/api/v0/chat/completion")
            return UpstreamResponse(parts)

        with patch.object(proxy.cffi_requests, "post", side_effect=post), patch.object(proxy.cffi_requests, "get", side_effect=AssertionError("unexpected network access")):
            status, data = await request(body)
        self.assertEqual(status, 200, data)
        self.assertEqual(len(calls), 1)
        self.assertIn("do:", calls[0][1]["json"]["prompt"].lower())
        return data

    async def test_nonstream_tool_call(self):
        args = {"path": "目录/文件\n\"x\"", "meta": {"nested": [1, True, None]}}
        output = "do: " + json.dumps({"name": "read", "arguments": args}, ensure_ascii=False)
        result = json.loads(await self.call(output, False))
        choice = result["choices"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertIsNone(choice["message"]["content"])
        call = choice["message"]["tool_calls"][0]
        self.assertEqual(call["type"], "function")
        self.assertEqual(call["function"]["name"], "read")
        self.assertEqual(json.loads(call["function"]["arguments"]), args)

    async def test_stream_tool_call(self):
        args = {"path": "目录/文件\n\"x\"", "meta": {"nested": [1, True, None]}}
        output = "do: " + json.dumps({"name": "read", "arguments": args}, ensure_ascii=False)
        raw = await self.call(output, True)
        records = [line[6:] for line in raw.splitlines() if line.startswith("data: ")]
        self.assertEqual(records[-1], "[DONE]")
        events = [json.loads(item) for item in records[:-1]]
        choices = [event["choices"][0] for event in events]
        self.assertEqual(choices[-1]["finish_reason"], "tool_calls")
        calls = [delta for choice in choices for delta in choice["delta"].get("tool_calls", [])]
        self.assertTrue(calls)
        self.assertTrue(all(call["index"] == 0 for call in calls))
        self.assertEqual(next(call["function"]["name"] for call in calls if "name" in call["function"]), "read")
        arguments = "".join(call["function"].get("arguments", "") for call in calls)
        self.assertEqual(json.loads(arguments), args)
        self.assertFalse(any(choice["delta"].get("content") for choice in choices))

    async def test_nonstream_plain_text(self):
        output = "请继续 do 这件事。"
        choice = json.loads(await self.call(output, False))["choices"][0]
        self.assertEqual(choice["finish_reason"], "stop")
        self.assertEqual(choice["message"]["content"], output)
        self.assertNotIn("tool_calls", choice["message"])

    async def test_stream_plain_text(self):
        output = "请继续 do 这件事。"
        raw = await self.call(output, True)
        records = [line[6:] for line in raw.splitlines() if line.startswith("data: ")]
        self.assertEqual(records[-1], "[DONE]")
        choices = [json.loads(item)["choices"][0] for item in records[:-1]]
        self.assertEqual(choices[-1]["finish_reason"], "stop")
        self.assertEqual("".join(c["delta"].get("content") or "" for c in choices), output)
        self.assertFalse(any(c["delta"].get("tool_calls") for c in choices))
