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
from app.tool_choice import select_tools
from tool_call import convert_messages_for_deepseek
from test_do_protocol_integration import UpstreamResponse, request


TOOLS = [
    {"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}},
    {"type": "function", "function": {"name": "bash", "parameters": {"type": "object"}}},
]


class ToolChoiceUnitTests(unittest.TestCase):
    def test_choices_and_validation(self):
        self.assertEqual(select_tools(TOOLS, "none"), (None, False))
        self.assertEqual(select_tools(TOOLS, "auto"), (TOOLS, False))
        self.assertEqual(select_tools(TOOLS, "required"), (TOOLS, True))
        for choice in ({"type": "function", "name": "bash"}, {"type": "function", "function": {"name": "bash"}}, {"type": "tool", "name": "bash"}):
            self.assertEqual(select_tools(TOOLS, choice), ([TOOLS[1]], True))
        for tools, choice in [(TOOLS, {"type": "function", "name": "missing"}), ([], "required"), (TOOLS, "wrong")]:
            with self.assertRaises(proxy.HTTPException):
                select_tools(tools, choice)

    def test_tool_results_remain_complete_and_identified(self):
        output = "x" * 501 + "终点"
        messages = [
            {"role": "assistant", "tool_calls": [
                {"id": "call_a", "function": {"name": "read", "arguments": "{}"}},
                {"id": "call_b", "function": {"name": "bash", "arguments": "{}"}},
            ]},
            {"role": "tool", "tool_call_id": "call_b", "content": output},
            {"role": "tool", "tool_call_id": "call_a", "content": '{"result":"ok","extra":"keep"}'},
        ]
        prompt = convert_messages_for_deepseek(messages, TOOLS)
        self.assertIn('Tool call IDs in order: ["call_a", "call_b"]', prompt)
        self.assertIn(json.dumps({"tool_call_id": "call_b", "content": output}, ensure_ascii=False), prompt)
        self.assertIn(json.dumps({"tool_call_id": "call_a", "content": messages[-1]["content"]}, ensure_ascii=False), prompt)


class ToolChoiceIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        with patch.object(config, "CONFIG_FILE", Path(self.tmp.name) / "config.json"), patch.object(config, "LEGACY_FILE", Path(self.tmp.name) / "token.json"):
            self.manager = config.ConfigManager()
        self.manager.add_account(config.DsAccount(account_label="test", login_type="email", token="token", session_id="session", is_valid=True))
        self.patches = [
            patch.object(proxy, "config_manager", self.manager),
            patch.object(proxy, "get_pow_response", return_value=""),
            patch.object(proxy, "get_models", return_value={"deepseek-default": (False, False, 64000, 8000)}),
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

    async def call(self, path, body, output):
        self.manager.mark_account_ok("test")
        prompts = []
        def post(url, **kwargs):
            if url.endswith("/chat_session/create"):
                response = UpstreamResponse()
                response.json = lambda: {"data": {"biz_data": {"id": "fresh"}}}
                return response
            prompts.append(kwargs["json"]["prompt"])
            return UpstreamResponse([output])
        with patch.object(proxy.cffi_requests, "post", side_effect=post):
            status, result = await request(path, body)
        return status, result, prompts

    async def test_chat_none_and_required_stream_and_nonstream(self):
        for stream in (False, True):
            body = {"model": "deepseek-default", "messages": [{"role": "user", "content": "hello"}], "tools": TOOLS, "tool_choice": "none", "stream": stream}
            status, result, prompts = await self.call("/v1/chat/completions", body, "normal answer")
            self.assertEqual(status, 200)
            self.assertNotIn("Available tools", prompts[0])
            self.assertIn("normal answer", result)
            body["tool_choice"] = "required"
            status, result, prompts = await self.call("/v1/chat/completions", body, "normal answer")
            self.assertIn("You must call", prompts[0])
            if stream:
                self.assertEqual(status, 200)
                self.assertIn("upstream_error", result)
            else:
                self.assertEqual(status, 502)

    async def test_specific_tool_responses_and_anthropic(self):
        output = 'do: {"name":"bash","arguments":{"command":"pwd"}}'
        response_body = {"model": "deepseek-default", "input": "hello", "tools": [{"type": "function", "name": "read", "parameters": {"type": "object"}}, {"type": "function", "name": "bash", "parameters": {"type": "object"}}], "tool_choice": {"type": "function", "name": "bash"}, "store": False}
        anthropic_body = {"model": "deepseek-default", "messages": [{"role": "user", "content": "hello"}], "tools": [{"name": "read", "input_schema": {"type": "object"}}, {"name": "bash", "input_schema": {"type": "object"}}], "tool_choice": {"type": "tool", "name": "bash"}, "max_tokens": 50}
        for path, body in [("/v1/responses", response_body), ("/v1/messages", anthropic_body)]:
            for stream in (False, True):
                body["stream"] = stream
                status, result, prompts = await self.call(path, body, output)
                self.assertEqual(status, 200, result)
                self.assertIn("bash", result)
                self.assertIn("pwd", result)
                self.assertNotIn('"name":"read"', prompts[0])
                self.assertIn("You must call", prompts[0])

    async def test_required_success_and_specific_rejects_other_tool(self):
        for stream in (False, True):
            body = {"model": "deepseek-default", "messages": [{"role": "user", "content": "hello"}], "tools": TOOLS, "tool_choice": "required", "stream": stream}
            output = 'do: {"name":"read","arguments":{}}'
            status, result, _ = await self.call("/v1/chat/completions", body, output)
            self.assertEqual(status, 200, result)
            self.assertIn("tool_calls", result)
            body["tool_choice"] = {"type": "function", "function": {"name": "bash"}}
            status, result, _ = await self.call("/v1/chat/completions", body, output)
            if stream:
                self.assertEqual(status, 200)
                self.assertIn("upstream_error", result)
            else:
                self.assertEqual(status, 502)
            self.assertNotIn('"name": "read"', result)

    async def test_stream_distinct_indices_for_separate_calls(self):
        body = {"model": "deepseek-default", "messages": [{"role": "user", "content": "hello"}], "tools": TOOLS, "stream": True}
        output = '<invoke name="read"></invoke><invoke name="bash"></invoke>'
        status, result, _ = await self.call("/v1/chat/completions", body, output)
        self.assertEqual(status, 200)
        events = [json.loads(line[6:]) for line in result.splitlines() if line.startswith("data: ") and line[6:] != "[DONE]"]
        calls = [call for event in events for choice in event.get("choices", []) for call in choice.get("delta", {}).get("tool_calls", [])]
        self.assertEqual([(call["index"], call["function"]["name"]) for call in calls], [(0, "read"), (1, "bash")])

    async def test_unsafe_placeholder_fails_without_inventing_required_argument(self):
        output = '<||DSML|| calls><|place_holder_no_1|>bash<|place_holder_no_1|>ls -la<|place_holder_no_1|>'
        schema = {"type": "object", "properties": {"command": {"type": "string"}, "description": {"type": "string"}}, "required": ["command", "description"]}
        for stream in (False, True):
            body = {"model": "deepseek-default", "messages": [{"role": "user", "content": "hello"}], "tools": [{"type": "function", "function": {"name": "bash", "parameters": schema}}], "stream": stream}
            status, result, _ = await self.call("/v1/chat/completions", body, output)
            self.assertEqual(status, 200 if stream else 502)
            self.assertIn("tool_format_error", result)
            self.assertNotIn("place_holder_no_1", result)
            self.assertNotIn('"tool_calls"', result)
