"""
Anthropic Messages API ↔ OpenAI Chat Completions 格式转换器。

独立模块，不依赖项目内部逻辑，可在 deepseek-free-api 和 MiMo2API 中共用。

转换路径:
  Anthropic Request → convert_request() → OpenAI body
    → 现有 pipeline (chat completions)
  OpenAI Response → convert_response() → Anthropic Response

用法:
  from anthropic_adapter import convert_request, convert_response, stream_response

  # 非流式
  openai_body = convert_request(anthropic_body)
  openai_result = await chat(openai_body)  # 调现有 pipeline
  anthropic_result = convert_response(openai_json, model, msg_id)

  # 流式
  openai_body = convert_request(anthropic_body)
  async for chunk in existing_stream:
      anthropic_sse = convert_stream_chunk(chunk, state)
      if anthropic_sse:
          yield anthropic_sse
  # 还有最终的 flush
"""

import json
import uuid
import re
from typing import Any, AsyncIterator, Optional


# ─── 请求转换（Anthropic → OpenAI）───────────────────────────────────────────

def convert_messages(
    messages: list,
    system: Optional[str] = None,
) -> list:
    """
    将 Anthropic Messages API 格式的消息列表转换为 OpenAI Chat Completions 格式。

    Anthropic 格式:
      {"role": "user", "content": [{"type": "text", "text": "hello"}]}
      {"role": "assistant", "content": [{"type": "text", "text": "hi"}, {"type": "tool_use", ...}]}
      {"role": "user", "content": [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "..."}}]}

    OpenAI 格式:
      {"role": "user", "content": "hello"}
      {"role": "user", "content": [{"type": "text", "text": "hello"}, {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}]}

    system 参数作为 system 消息插入到列表最前面。
    """
    result = []

    # System 消息
    if system:
        result.append({"role": "system", "content": system})

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        # Anthropic 的 tool_use/tool_result 消息
        if role == "assistant" and isinstance(content, list):
            # assistant 消息可能有 text + tool_use 混合
            text_parts = []
            tool_calls = []
            for block in content:
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == "tool_use":
                    tool_calls.append({
                        "id": block.get("id", f"tu_{uuid.uuid4().hex[:24]}"),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": json.dumps(block.get("input", {}), ensure_ascii=False),
                        }
                    })
                elif block.get("type") == "thinking":
                    text_parts.append(block.get("thinking", ""))
                elif block.get("type") == "redacted_thinking":
                    text_parts.append(f"[REDACTED THINKING: {block.get('data', '')}]")

            combined_text = "\n".join(t for t in text_parts if t)
            obj = {"role": "assistant"}
            if combined_text:
                obj["content"] = combined_text
            else:
                obj["content"] = None
            if tool_calls:
                obj["tool_calls"] = tool_calls
            result.append(obj)

        elif role == "user" and isinstance(content, list):
            # 用户消息可能有 text + image + tool_result 混合
            new_blocks = []
            text_parts = []
            tool_results = []
            for block in content:
                if block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
                elif block.get("type") == "image":
                    source = block.get("source", {})
                    img_type = source.get("media_type", "image/png")
                    img_data = source.get("data", "")
                    if source.get("type") == "base64":
                        new_blocks.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{img_type};base64,{img_data}"
                            }
                        })
                    elif source.get("type") == "url":
                        new_blocks.append({
                            "type": "image_url",
                            "image_url": {"url": source.get("url", "")}
                        })
                elif block.get("type") == "tool_result":
                    tool_results.append(block)

            # 如果有 tool_result，以 tool role 追加
            if tool_results:
                for tr in tool_results:
                    tc = tr.get("content", "")
                    tool_text = ""
                    if isinstance(tc, list):
                        tool_text = "\n".join(
                            c.get("text", "") for c in tc if isinstance(c, dict) and c.get("type") == "text"
                        )
                    else:
                        tool_text = str(tc)
                    result.append({
                        "role": "tool",
                        "tool_call_id": tr.get("tool_use_id", ""),
                        "content": tool_text,
                    })

            # 如果有 text 或 image，还要追加 user 消息
            combined_text = "\n".join(t for t in text_parts if t)
            if combined_text:
                new_blocks.insert(0, {"type": "text", "text": combined_text})

            if new_blocks:
                result.append({"role": "user", "content": new_blocks})
            elif combined_text and not tool_results:
                result.append({"role": "user", "content": combined_text})

        elif role == "tool" and isinstance(content, list):
            # tool_result 消息 → tool role
            tool_content = ""
            tool_call_id = ""
            for block in content:
                if block.get("type") == "tool_result":
                    tool_call_id = block.get("tool_use_id", "")
                    tc = block.get("content", "")
                    if isinstance(tc, list):
                        tool_content = "\n".join(
                            c.get("text", "") for c in tc if isinstance(c, dict) and c.get("type") == "text"
                        )
                    else:
                        tool_content = str(tc)
            if tool_content:
                result.append({
                    "role": "tool",
                    "content": tool_content,
                    "tool_call_id": tool_call_id,
                })

        elif isinstance(content, str):
            # 纯字符串（兼容）
            result.append({"role": role, "content": content})

        else:
            # fallback
            text = str(content) if content else ""
            result.append({"role": role, "content": text})

    return result


def convert_tools(tools: list) -> list:
    """将 Anthropic 工具格式转换为 OpenAI 格式。

    Anthropic: {"name": "get_time", "description": "...", "input_schema": {"type": "object", ...}}
    OpenAI:   {"type": "function", "function": {"name": "get_time", "description": "...", "parameters": {"type": "object", ...}}}
    """
    if not tools:
        return None

    result = []
    for t in tools:
        if isinstance(t, dict):
            fn = {
                "name": t.get("name", ""),
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {}),
            }
            result.append({"type": "function", "function": fn})
    return result if result else None


def convert_request(body: dict) -> dict:
    """将完整的 Anthropic Messages API 请求体转换为 OpenAI Chat Completions 格式。"""
    model = body.get("model", "deepseek-default")
    stream = body.get("stream", False)
    max_tokens = body.get("max_tokens", 4096)
    system = body.get("system", None)
    messages = body.get("messages", [])
    tools = body.get("tools", None)
    temperature = body.get("temperature", None)
    top_p = body.get("top_p", None)
    stop_sequences = body.get("stop_sequences", None)

    openai_msg = convert_messages(messages, system)
    openai_tools = convert_tools(tools)

    result = {
        "model": model,
        "messages": openai_msg,
        "stream": stream,
        "max_tokens": max_tokens,
    }

    if openai_tools:
        result["tools"] = openai_tools
    if temperature is not None:
        result["temperature"] = temperature
    if top_p is not None:
        result["top_p"] = top_p
    if stop_sequences:
        result["stop"] = stop_sequences

    return result


# ─── 响应转换（OpenAI → Anthropic 非流式）─────────────────────────────────

def _make_msg_id():
    return f"msg_{uuid.uuid4().hex[:24]}"


def _reasoning_to_thinking(content: str) -> list:
    """将 reasoning_content 字符串拆解为 thinking block + redacted_thinking block（如有）。"""
    blocks = []
    if not content:
        return blocks

    # 如果有 <think> 标签，提取 thinking 内容
    # 但 reasoning_content 通常已经是纯 thinking 文本
    blocks.append({
        "type": "thinking",
        "thinking": content,
        "signature": "",
    })
    return blocks


def convert_response(
    openai_body: dict,
    model: str,
    msg_id: Optional[str] = None,
) -> dict:
    """将 OpenAI Chat Completions 响应转换为 Anthropic Messages API 格式。"""
    msg_id = msg_id or _make_msg_id()
    choice = openai_body.get("choices", [{}])[0]
    message = choice.get("message", {})

    content = message.get("content", "") or ""
    reasoning = message.get("reasoning_content", "") or ""
    tool_calls = message.get("tool_calls", None)

    # 构建 content blocks
    content_blocks = []

    # 1. Thinking block（优先）
    if reasoning:
        content_blocks.extend(_reasoning_to_thinking(reasoning))

    # 2. Text block
    text = content.strip()
    if tool_calls:
        # 有工具调用时，可能也有文本
        if text:
            content_blocks.append({"type": "text", "text": text})
    else:
        # 无工具调用，text 就是响应
        if text:
            content_blocks.append({"type": "text", "text": text})
        elif not reasoning:
            # 既无思考也无文本（空响应）
            content_blocks.append({"type": "text", "text": ""})

    # 3. Tool use blocks
    stop_reason = "end_turn"
    if tool_calls:
        stop_reason = "tool_use"
        for tc in tool_calls:
            fn = tc.get("function", {})
            try:
                arguments = json.loads(fn.get("arguments", "{}"))
            except (json.JSONDecodeError, TypeError):
                arguments = {}
            content_blocks.append({
                "type": "tool_use",
                "id": tc.get("id", f"tu_{uuid.uuid4().hex[:24]}"),
                "name": fn.get("name", ""),
                "input": arguments,
            })

    # 4. 如果既无 thinking 也无 text 也无 tool_calls，给空数组
    if not content_blocks:
        content_blocks = [{"type": "text", "text": ""}]

    # Usage
    usage = openai_body.get("usage", {})
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0) or usage.get("total_tokens", 0) - input_tokens

    response = {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "content": content_blocks,
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        },
    }

    return response


# ─── 流式转换（OpenAI SSE → Anthropic SSE）────────────────────────────────

class StreamState:
    """追踪 Anthropic 流式响应的状态。"""
    def __init__(self, model: str, msg_id: str, input_tokens: int = 0):
        self.model = model
        self.msg_id = msg_id or _make_msg_id()
        self.input_tokens = input_tokens
        self.text_index = 0        # 当前文本 block 的 index
        self.thinking_index = None # thinking block 的 index（如果有）
        self.tool_indices = []     # tool_use blocks 的 index 列表
        self.thinking_active = False
        self.text_active = False
        self.tool_active = {}      # call_id → index
        self.thinking_buf = ""
        self.text_buf = ""
        self.tool_buf = {}         # call_id → {"name": "", "arguments": ""}
        self.started = False
        self.finished = False
        self.current_index = 0     # 下一个可用的 block index


def _make_sse(event: str, data: dict) -> str:
    """生成 Anthropic SSE 事件字符串。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _make_message_start(state: StreamState) -> str:
    """发送 message_start 事件。"""
    msg = {
        "id": state.msg_id,
        "type": "message",
        "role": "assistant",
        "content": [],
        "model": state.model,
        "stop_reason": None,
        "stop_sequence": None,
        "usage": {
            "input_tokens": state.input_tokens,
            "output_tokens": 0,
        },
    }
    return _make_sse("message_start", {"type": "message_start", "message": msg})


def _make_thinking_start(state: StreamState) -> str:
    """发送 thinking content_block_start。"""
    idx = state.current_index
    state.current_index += 1
    state.thinking_index = idx
    state.thinking_active = True
    block = {"type": "thinking", "thinking": "", "signature": ""}
    return _make_sse("content_block_start", {
        "type": "content_block_start",
        "index": idx,
        "content_block": block,
    })


def _make_thinking_delta(state: StreamState, text: str) -> str:
    """发送 thinking delta。"""
    return _make_sse("content_block_delta", {
        "type": "content_block_delta",
        "index": state.thinking_index,
        "delta": {"type": "thinking_delta", "thinking": text},
    })


def _make_thinking_stop(state: StreamState) -> str:
    """发送 thinking content_block_stop。"""
    state.thinking_active = False
    state.thinking_buf = ""
    return _make_sse("content_block_stop", {
        "type": "content_block_stop",
        "index": state.thinking_index,
    })


def _make_text_start(state: StreamState) -> str:
    """发送 text content_block_start。"""
    idx = state.current_index
    state.current_index += 1
    state.text_index = idx
    state.text_active = True
    return _make_sse("content_block_start", {
        "type": "content_block_start",
        "index": idx,
        "content_block": {"type": "text", "text": ""},
    })


def _make_text_delta(state: StreamState, text: str) -> str:
    """发送 text delta。"""
    if not text:
        return None
    return _make_sse("content_block_delta", {
        "type": "content_block_delta",
        "index": state.text_index,
        "delta": {"type": "text_delta", "text": text},
    })


def _make_text_stop(state: StreamState) -> str:
    """发送 text content_block_stop。"""
    state.text_active = False
    state.text_buf = ""
    return _make_sse("content_block_stop", {
        "type": "content_block_stop",
        "index": state.text_index,
    })


def _make_tool_use_start(state: StreamState, name: str, call_id: str) -> str:
    """发送 tool_use content_block_start。"""
    idx = state.current_index
    state.current_index += 1
    state.tool_active[call_id] = idx
    state.tool_indices.append(idx)
    block = {"type": "tool_use", "id": call_id, "name": name, "input": {}}
    return _make_sse("content_block_start", {
        "type": "content_block_start",
        "index": idx,
        "content_block": block,
    })


def _make_tool_input_delta(state: StreamState, text: str, call_id: str) -> str:
    """发送 input_json_delta。"""
    if not text:
        return None
    idx = state.tool_active.get(call_id)
    if idx is None:
        return None
    return _make_sse("content_block_delta", {
        "type": "content_block_delta",
        "index": idx,
        "delta": {"type": "input_json_delta", "partial_json": text},
    })


def _make_tool_use_stop(state: StreamState, call_id: str) -> str:
    """发送 tool_use content_block_stop。"""
    idx = state.tool_active.pop(call_id, None)
    if idx is None:
        return None
    return _make_sse("content_block_stop", {
        "type": "content_block_stop",
        "index": idx,
    })


def _make_message_delta(state: StreamState, stop_reason: str = "end_turn", output_tokens: int = 0) -> str:
    """发送 message_delta。"""
    return _make_sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": stop_reason},
        "usage": {"output_tokens": output_tokens},
    })


def _make_message_stop() -> str:
    """发送 message_stop。"""
    return _make_sse("message_stop", {"type": "message_stop"})


async def stream_response(
    openai_stream: AsyncIterator[Any],
    model: str,
    msg_id: Optional[str] = None,
) -> AsyncIterator[str]:
    """
    将 OpenAI Chat Completions SSE 流实时转换为 Anthropic Messages API SSE 事件。

    Args:
        openai_stream: OpenAI SSE 的 async generator（yield bytes/str）
        model: 模型名
        msg_id: 可选的 message ID

    Yields:
        Anthropic SSE 事件字符串（event: xxx\\ndata: {...}\\n\\n）
    """
    state = StreamState(model, msg_id or _make_msg_id())

    # 先发 message_start
    yield _make_message_start(state)

    tool_call_slots = {}      # index → {"id": ..., "name": ..., "arguments": ""}
    any_content_sent = False
    any_thinking_sent = False

    async for raw_chunk in openai_stream:
        if isinstance(raw_chunk, bytes):
            s = raw_chunk.decode("utf-8", errors="replace")
        else:
            s = str(raw_chunk)

        if not s.startswith("data: "):
            continue

        payload = s[6:].strip()
        if payload == "[DONE]":
            break

        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue

        choices = obj.get("choices", [])
        if not choices:
            continue

        delta = choices[0].get("delta", {})
        finish_reason = choices[0].get("finish_reason")

        # --- reasoning_content → thinking block ---
        reasoning = delta.get("reasoning_content", "")
        if reasoning:
            if not any_thinking_sent:
                any_thinking_sent = True
                yield _make_thinking_start(state)
            yield _make_thinking_delta(state, reasoning)

        # --- content → text block ---
        content = delta.get("content", "")
        if content:
            if not any_content_sent:
                # 如果有 thinking 在跑，先结束它
                if state.thinking_active:
                    yield _make_thinking_stop(state)
                yield _make_text_start(state)
                any_content_sent = True
            yield _make_text_delta(state, content)

        # --- tool_calls → tool_use block ---
        tool_calls = delta.get("tool_calls", [])
        if tool_calls:
            for tc in tool_calls:
                idx = tc.get("index", 0)
                fn = tc.get("function", {})

                if idx not in tool_call_slots:
                    # 新的 tool call slot
                    call_id = tc.get("id", f"tu_{uuid.uuid4().hex[:24]}")
                    tool_call_slots[idx] = {
                        "id": call_id,
                        "name": fn.get("name", ""),
                        "arguments": "",
                    }
                else:
                    call_id = tool_call_slots[idx]["id"]
                    # 补充 name（有些流在后续 chunk 才发 name）
                    if fn.get("name"):
                        tool_call_slots[idx]["name"] = fn["name"]

                args_delta = fn.get("arguments", "")
                if args_delta:
                    tool_call_slots[idx]["arguments"] += args_delta

        # --- finish_reason: 需要先结束已开始的 blocks ---
        if finish_reason:
            # 结束还在活跃的 blocks
            if any_thinking_sent and state.thinking_active:
                yield _make_thinking_stop(state)
            if any_content_sent and state.text_active:
                yield _make_text_stop(state)

            # 发送 tool_use blocks（如果有）
            if tool_call_slots:
                for idx in sorted(tool_call_slots.keys()):
                    slot = tool_call_slots[idx]
                    yield _make_tool_use_start(state, slot["name"], slot["id"])
                    if slot["arguments"]:
                        yield _make_tool_input_delta(state, slot["arguments"], slot["id"])
                    yield _make_tool_use_stop(state, slot["id"])

            # 确定 stop_reason
            if tool_call_slots:
                stop_reason = "tool_use"
            elif finish_reason == "tool_calls":
                stop_reason = "tool_use"
            elif finish_reason == "stop":
                stop_reason = "end_turn"
            else:
                stop_reason = finish_reason or "end_turn"

            # Usage
            usage = obj.get("usage", {})
            output_tokens = usage.get("completion_tokens", 0) or usage.get("total_tokens", 0) - (usage.get("prompt_tokens", 0) or 0) or 0

            yield _make_message_delta(state, stop_reason, output_tokens)
            yield _make_message_stop()
            state.finished = True
            return

    # 如果流没有 finish_reason 就结束了（异常）
    if not state.finished:
        if state.thinking_active:
            yield _make_thinking_stop(state)
        if state.text_active:
            yield _make_text_stop(state)
        yield _make_message_delta(state, "end_turn", 0)
        yield _make_message_stop()


# ─── 非流式的 wrapper（将完整响应包装为 SSE 事件）──────────────────────────

def nonstream_to_sse(anthropic_response: dict) -> list:
    """将非流式的 Anthropic 响应包装为 SSE 事件流（pseudo-streaming，一次性发送）。

    用于工具调用场景：先内部用非流式获取完整响应，再包装成 Anthropic SSE 格式。
    """
    events = []

    # message_start
    msg = dict(anthropic_response)
    msg["content"] = []
    msg["stop_reason"] = None
    events.append(_make_sse("message_start", {
        "type": "message_start", "message": msg
    }))

    # content blocks
    content_blocks = anthropic_response.get("content", [])
    for i, block in enumerate(content_blocks):
        block_type = block.get("type", "text")

        if block_type == "thinking":
            events.append(_make_sse("content_block_start", {
                "type": "content_block_start",
                "index": i,
                "content_block": {"type": "thinking", "thinking": "", "signature": ""},
            }))
            events.append(_make_sse("content_block_delta", {
                "type": "content_block_delta",
                "index": i,
                "delta": {"type": "thinking_delta", "thinking": block.get("thinking", "")},
            }))
            events.append(_make_sse("content_block_stop", {
                "type": "content_block_stop", "index": i,
            }))

        elif block_type == "text":
            events.append(_make_sse("content_block_start", {
                "type": "content_block_start",
                "index": i,
                "content_block": {"type": "text", "text": ""},
            }))
            text = block.get("text", "")
            if text:
                events.append(_make_sse("content_block_delta", {
                    "type": "content_block_delta",
                    "index": i,
                    "delta": {"type": "text_delta", "text": text},
                }))
            events.append(_make_sse("content_block_stop", {
                "type": "content_block_stop", "index": i,
            }))

        elif block_type == "tool_use":
            events.append(_make_sse("content_block_start", {
                "type": "content_block_start",
                "index": i,
                "content_block": {
                    "type": "tool_use",
                    "id": block.get("id", ""),
                    "name": block.get("name", ""),
                    "input": {},
                },
            }))
            inp = json.dumps(block.get("input", {}), ensure_ascii=False)
            if inp and inp != "{}":
                events.append(_make_sse("content_block_delta", {
                    "type": "content_block_delta",
                    "index": i,
                    "delta": {"type": "input_json_delta", "partial_json": inp},
                }))
            events.append(_make_sse("content_block_stop", {
                "type": "content_block_stop", "index": i,
            }))

    # message_delta + message_stop
    events.append(_make_sse("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": anthropic_response.get("stop_reason", "end_turn")},
        "usage": {"output_tokens": anthropic_response.get("usage", {}).get("output_tokens", 0)},
    }))
    events.append(_make_sse("message_stop", {"type": "message_stop"}))

    return events


# ─── 错误响应 ──────────────────────────────────────────────────────────────

def error_response(message: str, error_type: str = "api_error") -> dict:
    """生成 Anthropic 格式的错误响应。"""
    return {
        "error": {
            "type": error_type,
            "message": message,
        }
    }


# ═══════════════════════════════════════════════════════════════════════════
# 消息存储（GET /v1/messages/{message_id}）
# ═══════════════════════════════════════════════════════════════════════════
