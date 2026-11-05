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
from test_do_protocol_integration import UpstreamResponse, request


class MultiFragmentResponse(UpstreamResponse):
    def __init__(self, fragments, continuation=None):
        events = [{"p": "response/fragments", "o": "APPEND", "v": fragments}]
        if continuation is not None:
            events.append({"p": "response/fragments/-1/content", "o": "APPEND", "v": continuation})
        self.body = b"".join(b"data: " + json.dumps(event, ensure_ascii=False).encode() + b"\n\n" for event in events) + b"data: [DONE]\n\n"


class MultiFragmentTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        with patch.object(config, "CONFIG_FILE", Path(self.tmp.name) / "config.json"), patch.object(config, "LEGACY_FILE", Path(self.tmp.name) / "token.json"):
            self.manager = config.ConfigManager()
        self.manager.add_account(config.DsAccount(account_label="test", login_type="email", token="token", session_id="session", is_valid=True))
        self.patches = [
            patch.object(proxy, "config_manager", self.manager),
            patch.object(proxy, "get_pow_response", return_value=""),
            patch.object(proxy, "get_models", return_value={"deepseek-default": (True, False, 64000, 8000)}),
            patch.object(guard, "MIN_INTERVAL_SEC", 0),
            patch.object(guard, "_slots", threading.Semaphore(1)),
            patch.dict(os.environ, {"DEEPSEEK_PROXY_API_KEY": "test-key"}),
        ]
        for item in self.patches:
            item.start()

    async def asyncTearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.tmp.cleanup()

    async def call(self, body, fragments, continuation):
        self.manager.mark_account_ok("test")
        def post(url, **kwargs):
            if url.endswith("/chat_session/create"):
                response = UpstreamResponse()
                response.json = lambda: {"data": {"biz_data": {"id": "fresh"}}}
                return response
            return MultiFragmentResponse(fragments, continuation)
        with patch.object(proxy.cffi_requests, "post", side_effect=post):
            return await request("/v1/chat/completions", body)

    async def test_all_fragment_content_and_continuation_in_order(self):
        fragments = [{"type": "THINK", "content": "思考一"}, {"content": "思考二"}, None, {"type": "RESPONSE", "content": "甲"}, {"content": "乙"}]
        for stream in (False, True):
            body = {"model": "deepseek-default", "messages": [{"role": "user", "content": "hello"}], "stream": stream}
            status, data = await self.call(body, fragments, "丙")
            self.assertEqual(status, 200, data)
            if stream:
                events = [json.loads(line[6:]) for line in data.splitlines() if line.startswith("data: ") and line[6:] != "[DONE]"]
                deltas = [choice.get("delta", {}) for event in events for choice in event.get("choices", [])]
                content = "".join(delta.get("content") or "" for delta in deltas)
                thinking = "".join(delta.get("reasoning_content") or "" for delta in deltas)
            else:
                message = json.loads(data)["choices"][0]["message"]
                content = message["content"]
                thinking = message["reasoning_content"]
            self.assertEqual(thinking, "思考一思考二")
            self.assertEqual(content, "甲乙丙")

    async def test_tool_call_split_across_fragments(self):
        fragments = [{"type": "THINK", "content": "准备"}, {"type": "RESPONSE", "content": "开始。\n"}, {"content": '<｜DSML|tool_calls>\n{"name":"bash",'}, {"content": '"arguments":{"command":"ls'}]
        tools = [{"type": "function", "function": {"name": "bash", "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]
        for stream in (False, True):
            body = {"model": "deepseek-default", "messages": [{"role": "user", "content": "hello"}], "tools": tools, "stream": stream}
            status, data = await self.call(body, fragments, ' -la"}}\n')
            self.assertEqual(status, 200, data)
            if stream:
                events = [json.loads(line[6:]) for line in data.splitlines() if line.startswith("data: ") and line[6:] != "[DONE]"]
                choices = [choice for event in events for choice in event.get("choices", [])]
                calls = [call for choice in choices for call in choice.get("delta", {}).get("tool_calls", [])]
                content = "".join(choice.get("delta", {}).get("content") or "" for choice in choices)
                finish = choices[-1]["finish_reason"]
            else:
                choice = json.loads(data)["choices"][0]
                calls = choice["message"]["tool_calls"]
                content = choice["message"]["content"]
                finish = choice["finish_reason"]
            self.assertEqual(content, "开始。\n")
            self.assertEqual(finish, "tool_calls")
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["function"]["name"], "bash")
            self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"command": "ls -la"})
