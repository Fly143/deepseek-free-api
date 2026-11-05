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

    def __init__(self, parts=()):
        self.body = b"".join(
            b"data: " + json.dumps({"p": "response/content", "v": part}, ensure_ascii=False).encode() + b"\n\n"
            for part in parts
        ) + b"data: [DONE]\n\n"

    def iter_content(self, chunk_size=4096):
        for index in range(0, len(self.body), 11):
            yield self.body[index:index + 11]


async def request(path, body):
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
        "method": "POST", "scheme": "http", "path": path, "raw_path": path.encode(),
        "query_string": b"", "root_path": "", "headers": [(b"content-type", b"application/json"),
        (b"authorization", b"Bearer test-key"), (b"x-api-key", b"test-key"), (b"anthropic-version", b"2023-06-01")],
        "client": ("127.0.0.1", 12345), "server": ("127.0.0.1", 8000),
    }
    await asyncio.wait_for(proxy.app(scope, receive, send), timeout=5)
    status = next(event["status"] for event in events if event["type"] == "http.response.start")
    data = b"".join(event.get("body", b"") for event in events if event["type"] == "http.response.body")
    return status, data.decode()


class ProtocolIntegrationTests(unittest.IsolatedAsyncioTestCase):
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
            patch.object(proxy, "add_usage"),
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

    async def call(self, path, stream, prefix, supplied_output=None, tool_name="read", chunk_size=1):
        arguments = {"path": "目录/文件", "meta": {"nested": [1, True, None]}}
        output = supplied_output if supplied_output is not None else prefix + "do: " + json.dumps({"name": "read", "arguments": arguments}, ensure_ascii=False)
        parts = [output[i:i + chunk_size] for i in range(0, len(output), chunk_size)]
        schema = {"type": "object", "properties": {"path": {"type": "string"}, "meta": {"type": "object"}}}
        if tool_name == "bash":
            schema = {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}
        if path == "/v1/responses":
            body = {"model": "deepseek-default", "input": "读取文件", "tools": [{"type": "function", "name": tool_name, "parameters": schema}], "stream": stream, "store": False}
        elif path == "/v1/messages":
            body = {"model": "deepseek-default", "messages": [{"role": "user", "content": "读取文件"}], "max_tokens": 100, "tools": [{"name": tool_name, "input_schema": schema}], "stream": stream}
        else:
            body = {"model": "deepseek-default", "messages": [{"role": "user", "content": "读取文件"}], "tools": [{"type": "function", "function": {"name": tool_name, "parameters": schema}}], "stream": stream}

        def post(url, **kwargs):
            if url.endswith("/chat_session/create"):
                response = UpstreamResponse()
                response.json = lambda: {"data": {"biz_data": {"id": "request-session"}}}
                return response
            self.assertTrue(url.endswith("/chat/completion"), url)
            return UpstreamResponse(parts)

        with patch.object(proxy.cffi_requests, "post", side_effect=post), patch.object(proxy.cffi_requests, "get", side_effect=AssertionError("unexpected network access")):
            status, data = await request(path, body)
        self.assertEqual(status, 200, data)
        return data, arguments

    async def test_responses_nonstream(self):
        data, arguments = await self.call("/v1/responses", False, "")
        calls = [item for item in json.loads(data)["output"] if item["type"] == "function_call"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "read")
        self.assertEqual(json.loads(calls[0]["arguments"]), arguments)

    async def test_responses_stream(self):
        data, arguments = await self.call("/v1/responses", True, "")
        events = [json.loads(line[6:]) for line in data.splitlines() if line.startswith("data: ") and line[6:] != "[DONE]"]
        done = [event for event in events if event["type"] == "response.output_item.done" and event["item"]["type"] == "function_call"]
        self.assertEqual(len(done), 1)
        self.assertEqual(done[0]["item"]["name"], "read")
        self.assertEqual(json.loads(done[0]["item"]["arguments"]), arguments)

    async def test_anthropic_nonstream(self):
        data, arguments = await self.call("/v1/messages", False, "")
        result = json.loads(data)
        self.assertEqual(result["stop_reason"], "tool_use")
        calls = [item for item in result["content"] if item["type"] == "tool_use"]
        self.assertEqual(len(calls), 1)
        self.assertEqual((calls[0]["name"], calls[0]["input"]), ("read", arguments))

    async def test_anthropic_stream(self):
        data, arguments = await self.call("/v1/messages", True, "")
        events = [json.loads(line[6:]) for line in data.splitlines() if line.startswith("data: ")]
        blocks = [event["content_block"] for event in events if event["type"] == "content_block_start" and event["content_block"]["type"] == "tool_use"]
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["name"], "read")
        fragments = [event["delta"]["partial_json"] for event in events if event["type"] == "content_block_delta" and event["delta"]["type"] == "input_json_delta"]
        self.assertEqual(json.loads("".join(fragments)), arguments)
        self.assertTrue(any(event["type"] == "message_delta" and event["delta"]["stop_reason"] == "tool_use" for event in events))

    async def test_responses_mixed_prefix(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                data, arguments = await self.call("/v1/responses", stream, "准备读取。\n")
                if stream:
                    events = [json.loads(line[6:]) for line in data.splitlines() if line.startswith("data: ") and line[6:] != "[DONE]"]
                    completed = next(event["response"] for event in events if event["type"] == "response.completed")
                else:
                    completed = json.loads(data)
                calls = [item for item in completed["output"] if item["type"] == "function_call"]
                self.assertEqual(len(calls), 1)
                self.assertEqual(json.loads(calls[0]["arguments"]), arguments)
                text = "".join(part["text"] for item in completed["output"] if item["type"] == "message" for part in item["content"] if part["type"] == "output_text")
                self.assertEqual(text, "准备读取。\n")

    async def test_anthropic_mixed_prefix(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                data, arguments = await self.call("/v1/messages", stream, "准备读取。\n")
                if stream:
                    events = [json.loads(line[6:]) for line in data.splitlines() if line.startswith("data: ")]
                    blocks = [event["content_block"] for event in events if event["type"] == "content_block_start" and event["content_block"]["type"] == "tool_use"]
                    fragments = [event["delta"]["partial_json"] for event in events if event["type"] == "content_block_delta" and event["delta"]["type"] == "input_json_delta"]
                    text = "".join(event["delta"]["text"] for event in events if event["type"] == "content_block_delta" and event["delta"]["type"] == "text_delta")
                    self.assertEqual(len(blocks), 1)
                    self.assertEqual(json.loads("".join(fragments)), arguments)
                else:
                    result = json.loads(data)
                    blocks = [item for item in result["content"] if item["type"] == "tool_use"]
                    text = "".join(item["text"] for item in result["content"] if item["type"] == "text")
                    self.assertEqual(len(blocks), 1)
                    self.assertEqual(blocks[0]["input"], arguments)
                self.assertEqual(text, "准备读取。\n")


    async def test_multiple_calls_with_prefix(self):
        expected = [{"path": "目录/甲", "meta": {"nested": [1, {"enabled": True}]}}, {"path": "目录/乙", "meta": {"nested": [None, "do: inside"]}}]
        output = "开始读取。\n" + "\n".join("do: " + json.dumps({"name": "read", "arguments": item}, ensure_ascii=False) for item in expected)
        for path in ("/v1/responses", "/v1/messages"):
            for stream in (False, True):
                with self.subTest(path=path, stream=stream):
                    data, _ = await self.call(path, stream, "", output)
                    if stream:
                        events = [json.loads(line[6:]) for line in data.splitlines() if line.startswith("data: ") and line[6:] != "[DONE]"]
                        if path == "/v1/responses":
                            items = next(event["response"]["output"] for event in events if event["type"] == "response.completed")
                            calls = [json.loads(item["arguments"]) for item in items if item["type"] == "function_call"]
                        else:
                            starts = [event for event in events if event["type"] == "content_block_start" and event["content_block"]["type"] == "tool_use"]
                            calls = []
                            for start in starts:
                                fragments = [event["delta"]["partial_json"] for event in events if event["type"] == "content_block_delta" and event["index"] == start["index"] and event["delta"]["type"] == "input_json_delta"]
                                calls.append(json.loads("".join(fragments)))
                    elif path == "/v1/responses":
                        calls = [json.loads(item["arguments"]) for item in json.loads(data)["output"] if item["type"] == "function_call"]
                    else:
                        calls = [item["input"] for item in json.loads(data)["content"] if item["type"] == "tool_use"]
                    self.assertEqual(calls, expected)

    async def test_raw_placeholder_bash_call(self):
        for path in ("/v1/chat/completions", "/v1/responses", "/v1/messages"):
            for stream in (False, True):
                for chunk_size in (1, 7, 1000):
                    with self.subTest(path=path, stream=stream, chunk_size=chunk_size):
                        raw = "<||DSML|| calls><|place_holder_no_1|>bash<|place_holder_no_1|>ls -la<|place_holder_no_1|>"
                        data, _ = await self.call(path, stream, "", raw, "bash", chunk_size)
                        if stream:
                            events = [json.loads(line[6:]) for line in data.splitlines() if line.startswith("data: ") and line[6:] != "[DONE]"]
                            if path == "/v1/responses":
                                items = next(event["response"]["output"] for event in events if event["type"] == "response.completed")
                                calls = [(item["name"], json.loads(item["arguments"])) for item in items if item["type"] == "function_call"]
                            elif path == "/v1/chat/completions":
                                chunks = [call for event in events for choice in event.get("choices", []) for call in choice.get("delta", {}).get("tool_calls", [])]
                                names = [call["function"]["name"] for call in chunks if call.get("function", {}).get("name")]
                                arguments = "".join(call.get("function", {}).get("arguments", "") for call in chunks)
                                calls = [(names[0], json.loads(arguments))] if names else []
                            else:
                                starts = [event for event in events if event["type"] == "content_block_start" and event["content_block"]["type"] == "tool_use"]
                                calls = []
                                for start in starts:
                                    fragments = [event["delta"]["partial_json"] for event in events if event["type"] == "content_block_delta" and event["index"] == start["index"] and event["delta"]["type"] == "input_json_delta"]
                                    calls.append((start["content_block"]["name"], json.loads("".join(fragments))))
                        elif path == "/v1/responses":
                            calls = [(item["name"], json.loads(item["arguments"])) for item in json.loads(data)["output"] if item["type"] == "function_call"]
                        elif path == "/v1/chat/completions":
                            items = json.loads(data)["choices"][0]["message"].get("tool_calls", [])
                            calls = [(item["function"]["name"], json.loads(item["function"]["arguments"])) for item in items]
                        else:
                            calls = [(item["name"], item["input"]) for item in json.loads(data)["content"] if item["type"] == "tool_use"]
                        self.assertEqual(calls, [("bash", {"command": "ls -la"})])
                        self.assertNotIn("place_holder_no_1", data)

    async def test_raw_unclosed_dsml_bash_call(self):
        for path in ("/v1/chat/completions", "/v1/responses", "/v1/messages"):
            for stream in (False, True):
                for chunk_size in (1, 7, 1000):
                    with self.subTest(path=path, stream=stream, chunk_size=chunk_size):
                        raw = '\n\n<｜DSML|tool_calls>\n{"name": "bash", "arguments": {"command": "ls -la"}}\n'
                        data, _ = await self.call(path, stream, "", raw, "bash", chunk_size)
                        if stream:
                            events = [json.loads(line[6:]) for line in data.splitlines() if line.startswith("data: ") and line[6:] != "[DONE]"]
                            if path == "/v1/responses":
                                items = next(event["response"]["output"] for event in events if event["type"] == "response.completed")
                                calls = [(item["name"], json.loads(item["arguments"])) for item in items if item["type"] == "function_call"]
                            elif path == "/v1/chat/completions":
                                chunks = [call for event in events for choice in event.get("choices", []) for call in choice.get("delta", {}).get("tool_calls", [])]
                                names = [call["function"]["name"] for call in chunks if call.get("function", {}).get("name")]
                                arguments = "".join(call.get("function", {}).get("arguments", "") for call in chunks)
                                calls = [(names[0], json.loads(arguments))] if names else []
                            else:
                                starts = [event for event in events if event["type"] == "content_block_start" and event["content_block"]["type"] == "tool_use"]
                                calls = []
                                for start in starts:
                                    fragments = [event["delta"]["partial_json"] for event in events if event["type"] == "content_block_delta" and event["index"] == start["index"] and event["delta"]["type"] == "input_json_delta"]
                                    calls.append((start["content_block"]["name"], json.loads("".join(fragments))))
                        elif path == "/v1/responses":
                            calls = [(item["name"], json.loads(item["arguments"])) for item in json.loads(data)["output"] if item["type"] == "function_call"]
                        elif path == "/v1/chat/completions":
                            items = json.loads(data)["choices"][0]["message"].get("tool_calls", [])
                            calls = [(item["function"]["name"], json.loads(item["function"]["arguments"])) for item in items]
                        else:
                            calls = [(item["name"], item["input"]) for item in json.loads(data)["content"] if item["type"] == "tool_use"]
                        self.assertEqual(calls, [("bash", {"command": "ls -la"})])
                        self.assertNotIn("DSML", data)

    async def test_begin_tool_call_bash(self):
        for path in ("/v1/chat/completions", "/v1/responses", "/v1/messages"):
            for stream in (False, True):
                for chunk_size in (1, 7, 1000):
                    with self.subTest(path=path, stream=stream, chunk_size=chunk_size):
                        raw = '<|begin_of_tool_calls_begin|><|begin_of_tool_call_begin|>bash<|begin_of_tool_sep|><|begin_of_tool_call_arg_key|>command<|begin_of_tool_sep|>mkdir -p /tmp/new-folder && ls -la /tmp/new-folder<|begin_of_tool_call_end|><|begin_of_tool_calls_end|>'
                        data, _ = await self.call(path, stream, "", raw, "bash", chunk_size)
                        if stream:
                            events = [json.loads(line[6:]) for line in data.splitlines() if line.startswith("data: ") and line[6:] != "[DONE]"]
                            if path == "/v1/responses":
                                items = next(event["response"]["output"] for event in events if event["type"] == "response.completed")
                                calls = [(item["name"], json.loads(item["arguments"])) for item in items if item["type"] == "function_call"]
                            elif path == "/v1/chat/completions":
                                chunks = [call for event in events for choice in event.get("choices", []) for call in choice.get("delta", {}).get("tool_calls", [])]
                                names = [call["function"]["name"] for call in chunks if call.get("function", {}).get("name")]
                                arguments = "".join(call.get("function", {}).get("arguments", "") for call in chunks)
                                calls = [(names[0], json.loads(arguments))] if names else []
                            else:
                                starts = [event for event in events if event["type"] == "content_block_start" and event["content_block"]["type"] == "tool_use"]
                                calls = []
                                for start in starts:
                                    fragments = [event["delta"]["partial_json"] for event in events if event["type"] == "content_block_delta" and event["index"] == start["index"] and event["delta"]["type"] == "input_json_delta"]
                                    calls.append((start["content_block"]["name"], json.loads("".join(fragments))))
                        elif path == "/v1/responses":
                            calls = [(item["name"], json.loads(item["arguments"])) for item in json.loads(data)["output"] if item["type"] == "function_call"]
                        elif path == "/v1/chat/completions":
                            items = json.loads(data)["choices"][0]["message"].get("tool_calls", [])
                            calls = [(item["function"]["name"], json.loads(item["function"]["arguments"])) for item in items]
                        else:
                            calls = [(item["name"], item["input"]) for item in json.loads(data)["content"] if item["type"] == "tool_use"]
                        self.assertEqual(calls, [("bash", {"command": 'mkdir -p /tmp/new-folder && ls -la /tmp/new-folder'})])
                        self.assertNotIn("begin_of_tool_call", data)

    async def test_undeclared_and_fenced_calls_are_text(self):
        examples = [
            '示例：\n```text\ndo: {"name":"read","arguments":{"path":"demo"}}\n```',
            'do: {"name":"undeclared","arguments":{"path":"secret"}}',
            '请 do: 这件事',
        ]
        for path in ("/v1/responses", "/v1/messages"):
            for stream in (False, True):
                for output in examples:
                    with self.subTest(path=path, stream=stream, output=output):
                        data, _ = await self.call(path, stream, "", output)
                        if stream:
                            events = [json.loads(line[6:]) for line in data.splitlines() if line.startswith("data: ") and line[6:] != "[DONE]"]
                            if path == "/v1/responses":
                                items = next(event["response"]["output"] for event in events if event["type"] == "response.completed")
                                self.assertFalse(any(item["type"] == "function_call" for item in items))
                                text = "".join(part["text"] for item in items if item["type"] == "message" for part in item["content"] if part["type"] == "output_text")
                            else:
                                self.assertFalse(any(event["type"] == "content_block_start" and event["content_block"]["type"] == "tool_use" for event in events))
                                text = "".join(event["delta"]["text"] for event in events if event["type"] == "content_block_delta" and event["delta"]["type"] == "text_delta")
                        elif path == "/v1/responses":
                            items = json.loads(data)["output"]
                            self.assertFalse(any(item["type"] == "function_call" for item in items))
                            text = "".join(part["text"] for item in items if item["type"] == "message" for part in item["content"] if part["type"] == "output_text")
                        else:
                            items = json.loads(data)["content"]
                            self.assertFalse(any(item["type"] == "tool_use" for item in items))
                            text = "".join(item["text"] for item in items if item["type"] == "text")
                        self.assertEqual(text, output)

if __name__ == "__main__":
    unittest.main()
