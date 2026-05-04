import json
import shutil
import asyncio
import time
from pathlib import Path

from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.testclient import TestClient

import proxy


def _sse_line(obj: dict) -> bytes:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode("utf-8")


def main() -> None:
    base = Path(__file__).parent / ".tmp_responses_smoke"
    if base.exists():
        shutil.rmtree(base, ignore_errors=True)
    base.mkdir(parents=True, exist_ok=True)
    try:
        token_path = base / "token.json"
        token_path.write_text(json.dumps({"token": "t", "session_id": "s", "headers": {}}), encoding="utf-8")

        old_config = proxy.CONFIG_FILE
        old_discover = proxy._discover_models
        original_chat = proxy.chat

        proxy.CONFIG_FILE = token_path
        import response_store
        old_response_store_file = response_store._STORE_FILE
        response_store._STORE_FILE = base / "responses.json"
        proxy._discover_models = lambda: None

        async def fake_chat(req):
            body = req._body
            if "cancel me" in json.dumps(body.get("messages", []), ensure_ascii=False):
                await asyncio.sleep(0.2)
            if body.get("stream"):
                async def gen():
                    yield _sse_line({
                        "choices": [{"delta": {"content": "```json\n{\"value\":1}", "reasoning_content": "think ", "tool_calls": [{
                            "index": 0,
                            "id": "call_1",
                            "function": {"name": "lookup", "arguments": "{\"x\":1"},
                        }]}}]
                    })
                    yield _sse_line({
                        "choices": [{"delta": {"content": "\n```", "tool_calls": [{
                            "index": 0,
                            "function": {"arguments": "}"},
                        }], "refusal": "cannot comply"}, "finish_reason": "stop"}],
                        "model": body.get("model", "deepseek-default"),
                    })
                    yield b"data: [DONE]\n\n"

                return StreamingResponse(gen(), media_type="text/event-stream")

            response_json = {
                "id": "chatcmpl_test",
                "object": "chat.completion",
                "created": 123,
                "model": body.get("model", "deepseek-default"),
                "choices": [{
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "```json\n{\"ok\":true}\n```",
                        "tool_calls": [{
                            "id": "call_flat",
                            "type": "function",
                            "function": {"name": "lookup", "arguments": "{\"city\":\"sh\"}"},
                        }],
                    },
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
            }
            if "invalid json" in json.dumps(body.get("messages", []), ensure_ascii=False):
                response_json["choices"][0]["message"]["content"] = "not json"
                response_json["choices"][0]["message"].pop("tool_calls", None)
            return JSONResponse(response_json)

        proxy.chat = fake_chat
        client = TestClient(proxy.app)

        non_stream_body = {
            "model": "deepseek-default",
            "text": {"format": {"type": "json_object"}},
            "tools": [{
                "type": "function",
                "name": "lookup",
                "description": "lookup city",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
                "strict": True,
            }],
            "input": [{
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hello"}],
            }, {
                "type": "function_call_output",
                "call_id": "call_prev",
                "output": {"status": "ok"},
            }],
        }
        resp = client.post("/v1/responses", json=non_stream_body)
        assert resp.status_code == 200, resp.text
        data = resp.json()
        assert data["text"]["format"]["type"] == "json_object"
        assert data["output_text"] == "{\"ok\": true}" or data["output_text"] == "{\"ok\":true}"
        assert data["tools"][0]["type"] == "function"
        for field in ("id", "object", "created_at", "completed_at", "status", "model", "output", "output_text", "usage", "error", "incomplete_details", "metadata"):
            assert field in data
        assert not any(key.startswith("_") for key in data)
        message_items = [item for item in data["output"] if item["type"] == "message"]
        assert message_items
        assert message_items[0]["status"] == "completed"
        assert message_items[0]["role"] == "assistant"
        assert message_items[0]["content"][0]["text"] == data["output_text"]

        response_id = data["id"]

        schema_resp = client.post("/v1/responses", json={
            "model": "deepseek-default",
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "ok_schema",
                    "schema": {
                        "type": "object",
                        "required": ["ok"],
                        "properties": {"ok": {"type": "boolean"}},
                        "additionalProperties": False,
                    },
                }
            },
            "input": "schema success",
        })
        assert schema_resp.status_code == 200, schema_resp.text
        schema_data = schema_resp.json()
        assert schema_data["status"] == "completed"
        assert json.loads(schema_data["output_text"]) == {"ok": True}

        invalid_resp = client.post("/v1/responses", json={
            "model": "deepseek-default",
            "text": {"format": {"type": "json_object"}},
            "input": "invalid json",
        })
        assert invalid_resp.status_code == 200, invalid_resp.text
        invalid_data = invalid_resp.json()
        assert invalid_data["status"] == "failed"
        assert invalid_data["error"]["code"] == "invalid_json"

        token_count = client.post("/v1/responses/input_tokens", json={
            "input": non_stream_body["input"],
            "tools": non_stream_body["tools"],
        })
        assert token_count.status_code == 200, token_count.text
        assert token_count.json()["input_tokens"] > 0

        page1 = client.get(f"/v1/responses/{response_id}/input_items", params={"limit": 1})
        assert page1.status_code == 200, page1.text
        page1_data = page1.json()
        assert len(page1_data["data"]) == 1
        assert page1_data["first_id"]
        assert page1_data["has_more"] is True
        assert page1_data["data"][0]["id"].endswith("_in_1")

        page2 = client.get(
            f"/v1/responses/{response_id}/input_items",
            params={"limit": 5, "after": page1_data["last_id"]},
        )
        assert page2.status_code == 200, page2.text
        page2_data = page2.json()
        assert page2_data["data"]
        assert page2_data["data"][0]["id"].endswith("_in_0")

        background = client.post("/v1/responses", json={
            "model": "deepseek-default",
            "background": True,
            "input": "background",
        })
        assert background.status_code == 200, background.text
        background_data = background.json()
        assert background_data["status"] in ("queued", "in_progress")
        background_id = background_data["id"]

        polled = None
        deadline = time.time() + 2
        while time.time() < deadline:
            poll = client.get(f"/v1/responses/{background_id}")
            assert poll.status_code == 200, poll.text
            polled = poll.json()
            if polled["status"] in ("completed", "failed", "incomplete", "cancelled"):
                break
            time.sleep(0.05)
        assert polled and polled["status"] == "completed"

        with client.stream("GET", f"/v1/responses/{background_id}", params={"stream": "true"}) as replayed:
            assert replayed.status_code == 200
            replay_events = []
            for line in replayed.iter_lines():
                if not line or not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                replay_events.append(json.loads(line[6:]))
        assert replay_events[0]["type"] == "response.created"
        assert replay_events[-1]["type"] == "response.completed"
        assert replay_events[-1]["response"] == client.get(f"/v1/responses/{background_id}").json()
        replay_cursor = replay_events[1]["sequence_number"]
        with client.stream(
            "GET",
            f"/v1/responses/{background_id}",
            params={"stream": "true", "starting_after": replay_cursor},
        ) as replayed_after:
            after_events = []
            for line in replayed_after.iter_lines():
                if not line or not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                after_events.append(json.loads(line[6:]))
        assert after_events
        assert all(event["sequence_number"] > replay_cursor for event in after_events)

        cancellable = client.post("/v1/responses", json={
            "model": "deepseek-default",
            "background": True,
            "input": "cancel me",
        })
        assert cancellable.status_code == 200, cancellable.text
        cancel_id = cancellable.json()["id"]
        cancelled = client.post(f"/v1/responses/{cancel_id}/cancel")
        assert cancelled.status_code == 200, cancelled.text
        assert cancelled.json()["status"] == "cancelled"
        cancelled_again = client.post(f"/v1/responses/{cancel_id}/cancel")
        assert cancelled_again.status_code == 200, cancelled_again.text
        assert cancelled_again.json()["status"] == "cancelled"

        compacted = client.post("/v1/responses/compact", json={"response_id": response_id})
        assert compacted.status_code == 200, compacted.text
        compacted_data = compacted.json()
        assert compacted_data["previous_response_id"] == response_id
        continued = client.post("/v1/responses", json={
            "model": "deepseek-default",
            "previous_response_id": compacted_data["id"],
            "input": "next",
        })
        assert continued.status_code == 200, continued.text

        stream_body = {
            "model": "deepseek-default",
            "stream": True,
            "text": {"format": {"type": "json_object"}},
            "tools": [{
                "type": "function",
                "function": {
                    "name": "lookup",
                    "description": "lookup city",
                    "parameters": {"type": "object", "properties": {"x": {"type": "integer"}}},
                },
            }],
            "input": [{"type": "input_text", "text": "stream"}],
        }

        with client.stream("POST", "/v1/responses", json=stream_body) as streamed:
            assert streamed.status_code == 200
            events = []
            for line in streamed.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                payload = json.loads(line[6:])
                events.append(payload)

        assert events[0]["type"] == "response.created"
        assert all("sequence_number" in event for event in events)
        seqs = [event["sequence_number"] for event in events]
        assert seqs == sorted(seqs)
        assert any(event["type"] == "response.function_call_arguments.done" for event in events)
        assert any(event["type"] == "response.refusal.done" for event in events)
        assert events[-1]["type"] == "response.completed"
        assert "metadata" in events[-1]["response"]
        assert not any(key.startswith("_") for key in events[-1]["response"])

        missing = client.get("/v1/responses/not_found/input_items")
        assert missing.status_code == 404

        proxy.chat = original_chat
        proxy.CONFIG_FILE = old_config
        proxy._discover_models = old_discover
        response_store._STORE_FILE = old_response_store_file
    finally:
        shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    main()
