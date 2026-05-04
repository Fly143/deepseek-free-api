"""
DeepSeek 网页 → API 代理（纯 HTTP 转发，无浏览器依赖）
用法: python proxy.py → 打开 http://localhost:8000/admin → 粘贴 cURL → 保存 → 用
"""
import asyncio, json, os, shlex, time, uuid, webbrowser, base64, re, secrets
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
import tiktoken
from curl_cffi import requests as cffi_requests

# ── Tokenizer ───────────────────────────────────
_enc = tiktoken.get_encoding("cl100k_base")

def _count_tokens(text: str) -> int:
    return len(_enc.encode(text or ""))

# ── 用量统计 ───────────────────────────────────
from usage_store import add_usage, get_usage, clear_usage
from session_store import needs_renewal, on_new_session, add_tokens, get_usage_status
from response_store import save_response_record, get_response_record, delete_response_record, update_response_record

# ── 工具调用处理模块 ─────────────────────────────────
from tool_call import (
    build_tool_prompt,
    extract_tool_call,
    get_tool_names,
    convert_messages_for_deepseek,
)

# ── 流式筛分 + DSML 解析 ────────────────────────────
from tool_sieve import StreamSieve, SieveEvent
from tool_dsml import parse_dsml_tool_calls as _parse_dsml

# ── PoW (Proof of Work) Solver — 纯 Python 实现（无 WASM 依赖）────────
from pow_native import DeepSeekPOW

# Initialize PoW solver
pow_solver = DeepSeekPOW()

BASE_DIR = Path(__file__).parent
CONFIG_FILE = BASE_DIR / "token.json"
VISION_LOG = BASE_DIR / "vision.log"
_DEBUG = os.getenv("DS_DEBUG", "").lower() in ("1", "true", "yes")

def _vlog(msg: str):
    """Log vision-related messages. File logging only when DS_DEBUG=1."""
    ts = time.strftime("%H:%M:%S")
    if _DEBUG:
        with open(VISION_LOG, "a") as f:
            f.write(f"[{ts}] {msg}\n")
    print(f"[Vision] {msg}", flush=True)
PROXY_PORT = int(os.getenv("PROXY_PORT", "8000"))


def _gen_response_id() -> str:
    return f"resp_{uuid.uuid4().hex}"


def _ensure_list(value: Any) -> list:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _safe_json_loads(text: Any, default: Any):
    if not isinstance(text, str):
        return default
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError, TypeError):
        return default


def _response_status_from_finish_reason(finish_reason: str) -> str:
    if finish_reason in ("stop", "tool_calls"):
        return "completed"
    if finish_reason in ("length", "content_filter"):
        return "incomplete"
    return "completed"


def _response_incomplete_details(finish_reason: str) -> dict | None:
    if finish_reason in ("length", "content_filter"):
        return {"reason": finish_reason}
    return None


def _response_terminal_event_type(status: str) -> str:
    if status == "failed":
        return "response.failed"
    if status == "incomplete":
        return "response.incomplete"
    if status == "cancelled":
        return "response.cancelled"
    return "response.completed"


def _build_response_usage(usage: dict | None) -> dict:
    usage = usage or {}
    input_tokens = int(usage.get("prompt_tokens", 0) or 0)
    output_tokens = int(usage.get("completion_tokens", 0) or 0)
    total_tokens = int(usage.get("total_tokens", input_tokens + output_tokens) or 0)
    return {
        "input_tokens": input_tokens,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": total_tokens,
    }


def _response_text_item(text: str, item_id: str | None = None) -> dict:
    return {
        "id": item_id or f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "content": [{
            "type": "output_text",
            "text": text or "",
            "annotations": [],
        }],
    }


def _response_refusal_item(refusal_text: str, item_id: str | None = None) -> dict:
    return {
        "id": item_id or f"rf_{uuid.uuid4().hex[:24]}",
        "type": "refusal",
        "status": "completed",
        "content": [{
            "type": "output_text",
            "text": refusal_text or "",
            "annotations": [],
        }],
    }


def _response_reasoning_item(summary_text: str, item_id: str | None = None) -> dict:
    return {
        "id": item_id or f"rs_{uuid.uuid4().hex[:24]}",
        "type": "reasoning",
        "summary": [{
            "type": "summary_text",
            "text": summary_text or "",
        }],
    }


def _response_function_call_item(tool_call: dict, call_id: str | None = None) -> dict:
    fn = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
    return {
        "id": tool_call.get("id") or f"fc_{uuid.uuid4().hex[:24]}",
        "type": "function_call",
        "call_id": call_id or tool_call.get("id") or f"call_{uuid.uuid4().hex[:24]}",
        "name": fn.get("name", ""),
        "arguments": fn.get("arguments", "{}"),
        "status": "completed",
    }


def _response_text_config(body: dict) -> dict:
    text = body.get("text")
    if isinstance(text, dict):
        cfg = dict(text)
        fmt = cfg.get("format")
        if isinstance(fmt, dict):
            cfg["format"] = dict(fmt)
        elif isinstance(fmt, str):
            cfg["format"] = {"type": fmt}
        else:
            cfg["format"] = {"type": "text"}
        return cfg
    return {"format": {"type": "text"}}


def _extract_structured_json_text(output_text: str) -> tuple[str, Any] | tuple[None, None]:
    if not output_text:
        return None, None
    candidate = output_text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate, flags=re.IGNORECASE)
        candidate = re.sub(r"\s*```$", "", candidate)
    start_candidates = [i for i in (candidate.find("{"), candidate.find("[")) if i != -1]
    if start_candidates:
        start = min(start_candidates)
        end_candidates = [candidate.rfind("}"), candidate.rfind("]")]
        end = max(end_candidates)
        if end > start:
            candidate = candidate[start:end + 1]
    try:
        parsed = json.loads(candidate)
        return json.dumps(parsed, ensure_ascii=False), parsed
    except (json.JSONDecodeError, ValueError, TypeError):
        return None, None


def _normalize_structured_output_text(output_text: str, text_config: dict | None) -> str:
    if not output_text or not isinstance(text_config, dict):
        return output_text
    fmt = text_config.get("format")
    if not isinstance(fmt, dict):
        return output_text
    fmt_type = fmt.get("type")
    if fmt_type not in ("json_object", "json_schema"):
        return output_text

    normalized, _ = _extract_structured_json_text(output_text)
    return normalized if normalized is not None else output_text


def _json_schema_from_text_config(text_config: dict | None) -> dict | None:
    fmt = text_config.get("format") if isinstance(text_config, dict) else None
    if not isinstance(fmt, dict) or fmt.get("type") != "json_schema":
        return None
    schema = fmt.get("schema")
    if isinstance(schema, dict):
        return schema
    json_schema = fmt.get("json_schema")
    if isinstance(json_schema, dict):
        nested = json_schema.get("schema")
        return nested if isinstance(nested, dict) else json_schema
    return None


def _schema_type_matches(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return (isinstance(value, int) or isinstance(value, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    return True


def _validate_json_schema_subset(value: Any, schema: dict | None, path: str = "$") -> str | None:
    if not isinstance(schema, dict):
        return None
    expected_type = schema.get("type")
    if isinstance(expected_type, list):
        if not any(_schema_type_matches(value, t) for t in expected_type if isinstance(t, str)):
            return f"{path} does not match any allowed type"
    elif isinstance(expected_type, str) and not _schema_type_matches(value, expected_type):
        return f"{path} must be {expected_type}"

    if isinstance(value, dict):
        required = schema.get("required")
        if isinstance(required, list):
            for key in required:
                if isinstance(key, str) and key not in value:
                    return f"{path}.{key} is required"
        properties = schema.get("properties")
        if isinstance(properties, dict):
            for key, prop_schema in properties.items():
                if key in value and isinstance(prop_schema, dict):
                    error = _validate_json_schema_subset(value[key], prop_schema, f"{path}.{key}")
                    if error:
                        return error
        additional = schema.get("additionalProperties")
        if additional is False and isinstance(properties, dict):
            extra = [key for key in value.keys() if key not in properties]
            if extra:
                return f"{path}.{extra[0]} is not allowed"
        elif isinstance(additional, dict):
            properties = properties if isinstance(properties, dict) else {}
            for key, item in value.items():
                if key not in properties:
                    error = _validate_json_schema_subset(item, additional, f"{path}.{key}")
                    if error:
                        return error

    if isinstance(value, list):
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for idx, item in enumerate(value):
                error = _validate_json_schema_subset(item, item_schema, f"{path}[{idx}]")
                if error:
                    return error

    return None


def _structured_output_error(output_text: str, text_config: dict | None) -> dict | None:
    fmt = text_config.get("format") if isinstance(text_config, dict) else None
    if not isinstance(fmt, dict):
        return None
    fmt_type = fmt.get("type")
    if fmt_type not in ("json_object", "json_schema"):
        return None
    normalized, parsed = _extract_structured_json_text(output_text)
    if normalized is None:
        return {"message": "response output_text is not valid JSON", "type": "invalid_response_format", "code": "invalid_json"}
    if fmt_type == "json_schema":
        schema_error = _validate_json_schema_subset(parsed, _json_schema_from_text_config(text_config))
        if schema_error:
            return {"message": f"response output_text does not match json_schema: {schema_error}", "type": "invalid_response_format", "code": "schema_validation_failed"}
    return None


def _extract_output_text(output: list[dict]) -> str:
    texts: list[str] = []
    for item in output or []:
        if item.get("type") == "message":
            for content in item.get("content", []) or []:
                if content.get("type") == "output_text":
                    texts.append(content.get("text", "") or "")
    return "".join(texts)


def _normalize_response_tool_output(output: Any) -> str:
    if isinstance(output, str):
        return output
    if output is None:
        return ""
    return json.dumps(output, ensure_ascii=False)


def _normalize_response_tool(tool: Any) -> dict | None:
    if not isinstance(tool, dict):
        return None
    ttype = tool.get("type")
    if ttype == "web_search_preview":
        return None

    fn = tool.get("function") if isinstance(tool.get("function"), dict) else {}
    name = tool.get("name") or fn.get("name")
    if ttype == "function" or name:
        normalized_fn = {
            "name": name or "",
            "description": tool.get("description") or fn.get("description", ""),
            "parameters": tool.get("parameters") or fn.get("parameters") or {"type": "object", "properties": {}},
        }
        if "strict" in tool:
            normalized_fn["strict"] = tool.get("strict")
        elif "strict" in fn:
            normalized_fn["strict"] = fn.get("strict")
        return {"type": "function", "function": normalized_fn}
    return None


def _normalize_input_file_part(part: dict) -> dict:
    if part.get("file_data") or part.get("data"):
        return {
            "type": "input_file",
            "filename": part.get("filename") or "file.txt",
            "file_data": part.get("file_data") or part.get("data") or "",
        }

    out = {"type": "input_file"}
    if part.get("file_id"):
        out["file_id"] = part.get("file_id")
    if part.get("filename"):
        out["filename"] = part.get("filename")
    return out


def _normalize_response_input_item(item: Any) -> dict | None:
    if isinstance(item, str):
        return {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": item}],
        }
    if not isinstance(item, dict):
        return None

    item_type = item.get("type")
    role = item.get("role")

    if item_type in ("input_text", "text"):
        return {"type": "input_text", "text": item.get("text", "")}

    if item_type == "function_call_output":
        normalized = {
            "type": "function_call_output",
            "call_id": item.get("call_id") or item.get("id") or "",
            "output": _normalize_response_tool_output(item.get("output")),
        }
        if item.get("id"):
            normalized["id"] = item.get("id")
        return normalized

    if item_type == "function_call":
        normalized = {
            "type": "function_call",
            "call_id": item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex[:24]}",
            "name": item.get("name", ""),
            "arguments": item.get("arguments", "{}") if isinstance(item.get("arguments", "{}"), str)
            else json.dumps(item.get("arguments", {}), ensure_ascii=False),
        }
        if item.get("id"):
            normalized["id"] = item.get("id")
        if "parameters" in item:
            normalized["parameters"] = item.get("parameters")
        if "description" in item:
            normalized["description"] = item.get("description")
        return normalized

    if item_type == "message" or role in ("system", "user", "assistant", "tool"):
        normalized = {
            "type": "message",
            "role": role or item.get("role", "user"),
        }
        content = item.get("content", "")
        normalized_parts = []
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type")
                if ptype in ("input_text", "output_text", "text"):
                    normalized_parts.append({"type": "input_text", "text": part.get("text", "")})
                elif ptype == "input_image":
                    image_url = part.get("image_url") or part.get("url") or ""
                    item_part = {"type": "input_image"}
                    if isinstance(image_url, dict):
                        item_part["image_url"] = image_url.get("url", "")
                        if image_url.get("detail"):
                            item_part["detail"] = image_url.get("detail")
                    elif image_url:
                        item_part["image_url"] = image_url
                    if part.get("file_id"):
                        item_part["file_id"] = part.get("file_id")
                    normalized_parts.append(item_part)
                elif ptype == "input_file":
                    normalized_parts.append(_normalize_input_file_part(part))
                elif ptype == "function_call" and normalized["role"] == "assistant":
                    normalized_parts.append({
                        "type": "function_call",
                        "call_id": part.get("call_id") or part.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                        "name": part.get("name", ""),
                        "arguments": part.get("arguments", "{}") if isinstance(part.get("arguments", "{}"), str)
                        else json.dumps(part.get("arguments", {}), ensure_ascii=False),
                    })
        elif isinstance(content, str):
            normalized_parts.append({"type": "input_text", "text": content})
        normalized["content"] = normalized_parts
        return normalized

    return item


def _normalize_response_input_items(input_items: Any) -> list[dict]:
    items = _ensure_list(input_items)
    normalized: list[dict] = []
    for item in items:
        normalized_item = _normalize_response_input_item(item)
        if normalized_item is not None:
            normalized.append(normalized_item)
    return normalized


def _assign_response_input_item_ids(items: list[dict], response_id: str) -> list[dict]:
    assigned: list[dict] = []
    for idx, item in enumerate(items or []):
        if not isinstance(item, dict):
            continue
        copy = dict(item)
        if not copy.get("id"):
            copy["id"] = f"{response_id}_in_{idx}"
        assigned.append(copy)
    return assigned


def _response_instructions_item(instructions: str) -> dict:
    return {
        "type": "message",
        "role": "system",
        "content": [{"type": "input_text", "text": instructions}],
    }


def _stored_input_items(body: dict) -> list[dict]:
    items = _normalize_response_input_items(body.get("input"))
    instructions = body.get("instructions")
    if isinstance(instructions, str) and instructions.strip():
        items = [_response_instructions_item(instructions)] + items
    return _assign_response_input_item_ids(items, body.get("_response_id", f"resp_{uuid.uuid4().hex}"))


def _paginate_response_input_items(items: list[dict], *, limit: int, after: str | None, before: str | None, order: str) -> tuple[list[dict], bool]:
    ordered = list(items or [])
    if order == "desc":
        ordered = list(reversed(ordered))

    if after:
        idx = next((i for i, item in enumerate(ordered) if item.get("id") == after), -1)
        ordered = ordered[idx + 1:] if idx != -1 else []
    if before:
        idx = next((i for i, item in enumerate(ordered) if item.get("id") == before), -1)
        ordered = ordered[:idx] if idx != -1 else []

    has_more = len(ordered) > limit
    return ordered[:limit], has_more


def _response_object_payload(record: dict, *, status: str | None = None, usage: dict | None = None,
                              completed_at: int | None | object = Ellipsis, output: list[dict] | None = None,
                              error: dict | None | object = Ellipsis, incomplete_details: dict | None | object = Ellipsis,
                              output_text: str | None = None) -> dict:
    payload = dict(_public_response_record(record))
    if status is not None:
        payload["status"] = status
    if usage is not None or "usage" in payload:
        payload["usage"] = usage
    if completed_at is not Ellipsis:
        payload["completed_at"] = completed_at
    if output is not None:
        payload["output"] = output
    if error is not Ellipsis:
        payload["error"] = error
    if incomplete_details is not Ellipsis:
        payload["incomplete_details"] = incomplete_details
    if output_text is not None:
        payload["output_text"] = output_text
    return payload


def _response_failed_payload(response_id: str, created: int, model_name: str, body: dict,
                             previous_response_id: str | None, error: dict, output_text: str = "") -> dict:
    text_cfg = _response_text_config(body)
    return {
        "id": response_id,
        "object": "response",
        "created_at": created,
        "completed_at": None,
        "status": "failed",
        "error": error,
        "incomplete_details": None,
        "instructions": body.get("instructions"),
        "max_output_tokens": body.get("max_output_tokens"),
        "model": model_name,
        "output": [],
        "parallel_tool_calls": True,
        "previous_response_id": previous_response_id,
        "reasoning": {"effort": body.get("reasoning", {}).get("effort")} if isinstance(body.get("reasoning"), dict) else {},
        "store": True if body.get("store", True) else False,
        "temperature": body.get("temperature"),
        "text": text_cfg,
        "tool_choice": body.get("tool_choice", "auto"),
        "tools": body.get("tools", []),
        "top_p": body.get("top_p"),
        "truncation": body.get("truncation", "disabled"),
        "usage": None,
        "user": body.get("user"),
        "metadata": body.get("metadata", {}),
        "output_text": _normalize_structured_output_text(output_text, text_cfg),
    }


def _extract_response_messages_and_tools(input_items: Any) -> tuple[list[dict], list[dict] | None]:
    items = _ensure_list(input_items)
    messages: list[dict] = []
    tools_from_input: list[dict] = []

    for item in items:
        if isinstance(item, str):
            messages.append({"role": "user", "content": item})
            continue
        if not isinstance(item, dict):
            continue

        item_type = item.get("type")
        role = item.get("role")

        if role in ("system", "user", "assistant", "tool"):
            content = item.get("content", "")
            if isinstance(content, list):
                parts = []
                assistant_tool_calls = []
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    ptype = part.get("type")
                    if ptype in ("input_text", "output_text", "text"):
                        parts.append({"type": "text", "text": part.get("text", "")})
                    elif ptype == "input_image":
                        image_url = part.get("image_url") or part.get("url") or ""
                        if image_url:
                            parts.append({"type": "image_url", "image_url": {"url": image_url}})
                    elif ptype == "input_file":
                        file_obj = {
                            "filename": part.get("filename") or "file.txt",
                            "file_data": part.get("file_data") or part.get("data") or "",
                        }
                        parts.append({"type": "file", "file": file_obj})
                    elif ptype == "function_call" and role == "assistant":
                        assistant_tool_calls.append({
                            "id": part.get("call_id") or part.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                            "type": "function",
                            "function": {
                                "name": part.get("name", ""),
                                "arguments": part.get("arguments", "{}") if isinstance(part.get("arguments", "{}"), str)
                                else json.dumps(part.get("arguments", {}), ensure_ascii=False),
                            }
                        })
                msg = {"role": role, "content": parts if parts else ""}
                if assistant_tool_calls:
                    msg["tool_calls"] = assistant_tool_calls
                    if not parts:
                        msg["content"] = None
                messages.append(msg)
            else:
                msg = {"role": role, "content": content}
                if role == "assistant" and item_type == "function_call":
                    msg["tool_calls"] = [{
                        "id": item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                        "type": "function",
                        "function": {
                            "name": item.get("name", ""),
                            "arguments": item.get("arguments", "{}") if isinstance(item.get("arguments", "{}"), str)
                            else json.dumps(item.get("arguments", {}), ensure_ascii=False),
                        }
                    }]
                    if content in ("", None):
                        msg["content"] = None
                messages.append(msg)
            continue

        if item_type == "message":
            content = item.get("content", [])
            role = item.get("role", "user")
            normalized_parts = []
            assistant_tool_calls = []
            if isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    ptype = part.get("type")
                    if ptype in ("input_text", "output_text", "text"):
                        normalized_parts.append({"type": "text", "text": part.get("text", "")})
                    elif ptype == "input_image":
                        image_url = part.get("image_url") or part.get("url") or ""
                        if image_url:
                            normalized_parts.append({"type": "image_url", "image_url": {"url": image_url}})
                    elif ptype == "input_file":
                        normalized_parts.append({
                            "type": "file",
                            "file": {
                                "filename": part.get("filename") or "file.txt",
                                "file_data": part.get("file_data") or part.get("data") or "",
                            }
                        })
                    elif ptype == "function_call" and role == "assistant":
                        assistant_tool_calls.append({
                            "id": part.get("call_id") or part.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                            "type": "function",
                            "function": {
                                "name": part.get("name", ""),
                                "arguments": part.get("arguments", "{}") if isinstance(part.get("arguments", "{}"), str)
                                else json.dumps(part.get("arguments", {}), ensure_ascii=False),
                            }
                        })
            msg = {"role": role, "content": normalized_parts if normalized_parts else ""}
            if assistant_tool_calls:
                msg["tool_calls"] = assistant_tool_calls
                if not normalized_parts:
                    msg["content"] = None
            messages.append(msg)
            continue

        if item_type == "function_call_output":
            output = _normalize_response_tool_output(item.get("output"))
            tool_message = {"role": "tool", "content": output}
            if item.get("call_id"):
                tool_message["tool_call_id"] = item.get("call_id")
            messages.append(tool_message)
            continue

        if item_type == "function_call":
            if item.get("name") and ("parameters" in item or "description" in item):
                tools_from_input.append({
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "description": item.get("description", ""),
                        "parameters": item.get("parameters", {"type": "object", "properties": {}}),
                    }
                })
            else:
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                        "type": "function",
                        "function": {
                            "name": item.get("name", ""),
                            "arguments": item.get("arguments", "{}") if isinstance(item.get("arguments", "{}"), str)
                            else json.dumps(item.get("arguments", {}), ensure_ascii=False),
                        }
                    }]
                })
            continue

        if item_type in ("input_text", "text"):
            messages.append({"role": "user", "content": item.get("text", "")})

    return messages, (tools_from_input or None)


def _merge_previous_response_context(messages: list[dict], previous_response_id: str | None) -> list[dict]:
    if not previous_response_id:
        return messages
    prev = get_response_record(previous_response_id)
    if not prev:
        raise HTTPException(404, detail={"error": {"message": f"response {previous_response_id} not found", "type": "invalid_request_error"}})

    previous_messages = prev.get("_messages", [])
    if not isinstance(previous_messages, list):
        previous_messages = []
    else:
        previous_messages = list(previous_messages)
    if messages and messages[0].get("role") == "system":
        while previous_messages and isinstance(previous_messages[0], dict) and previous_messages[0].get("role") == "system":
            previous_messages.pop(0)
    return previous_messages + messages


def _normalize_response_tools(body: dict, parsed_tools: list[dict] | None) -> list[dict] | None:
    tools = body.get("tools")
    merged: list[dict] = []
    seen: set[str] = set()
    for source in (parsed_tools or []) + (tools if isinstance(tools, list) else []):
        normalized = _normalize_response_tool(source)
        if not normalized:
            continue
        name = normalized.get("function", {}).get("name", "")
        if name and name in seen:
            continue
        if name:
            seen.add(name)
        merged.append(normalized)
    return merged or None


def _has_web_search_tool(body: dict) -> bool:
    tools = body.get("tools")
    if not isinstance(tools, list):
        return False
    for tool in tools:
        if isinstance(tool, dict) and tool.get("type") == "web_search_preview":
            return True
    return False


def _resolve_responses_model(body: dict) -> str:
    model = body.get("model", "deepseek-default")
    if not _has_web_search_tool(body) or "search" in model:
        return model

    candidates = []
    if model.endswith("-reasoner"):
        candidates.append(f"{model}-search")
    candidates.append(f"{model}-search")
    if model == "deepseek-default":
        candidates.append("deepseek-search")
    if model == "deepseek-reasoner":
        candidates.append("deepseek-reasoner-search")

    models = get_models()
    for candidate in candidates:
        if candidate in models:
            return candidate
    return model


def _messages_from_responses_request(body: dict) -> tuple[list[dict], list[dict] | None]:
    input_items = body.get("input", [])
    if isinstance(input_items, str):
        messages, tools = [{"role": "user", "content": input_items}], None
    else:
        messages, tools = _extract_response_messages_and_tools(input_items)

    instructions = body.get("instructions")
    if isinstance(instructions, str) and instructions.strip():
        messages = [{"role": "system", "content": instructions}] + messages
    return messages, tools


def _build_responses_record(
    response_id: str,
    body: dict,
    model: str,
    created: int,
    completed_at: int | None,
    output: list[dict],
    usage: dict,
    messages: list[dict],
    status: str = "completed",
    incomplete_details: dict | None = None,
) -> dict:
    text_config = _response_text_config(body)
    text = _normalize_structured_output_text(_extract_output_text(output), text_config)
    record = {
        "id": response_id,
        "object": "response",
        "created_at": created,
        "completed_at": completed_at,
        "status": status,
        "error": None,
        "incomplete_details": incomplete_details,
        "instructions": body.get("instructions"),
        "max_output_tokens": body.get("max_output_tokens"),
        "model": model,
        "output": output,
        "parallel_tool_calls": True,
        "previous_response_id": body.get("previous_response_id"),
        "reasoning": {"effort": body.get("reasoning", {}).get("effort")} if isinstance(body.get("reasoning"), dict) else {},
        "store": True if body.get("store", True) else False,
        "temperature": body.get("temperature"),
        "text": text_config,
        "tool_choice": body.get("tool_choice", "auto"),
        "tools": body.get("tools", []),
        "top_p": body.get("top_p"),
        "truncation": body.get("truncation", "disabled"),
        "usage": _build_response_usage(usage),
        "user": body.get("user"),
        "metadata": body.get("metadata", {}),
        "output_text": text,
        "_messages": messages,
        "_input": _stored_input_items(body),
    }
    return record


_RESPONSE_PUBLIC_DEFAULTS = {
    "object": "response",
    "completed_at": None,
    "error": None,
    "incomplete_details": None,
    "instructions": None,
    "max_output_tokens": None,
    "parallel_tool_calls": True,
    "previous_response_id": None,
    "reasoning": {},
    "store": True,
    "temperature": None,
    "text": {"format": {"type": "text"}},
    "tool_choice": "auto",
    "tools": [],
    "top_p": None,
    "truncation": "disabled",
    "usage": None,
    "user": None,
    "metadata": {},
    "output": [],
    "output_text": "",
}


def _normalized_response_output_item(item: Any) -> dict:
    if not isinstance(item, dict):
        return _response_text_item("")
    item_type = item.get("type")
    if item_type == "message":
        normalized = dict(item)
        normalized.setdefault("id", f"msg_{uuid.uuid4().hex[:24]}")
        normalized.setdefault("status", "completed")
        normalized.setdefault("role", "assistant")
        content = []
        for part in normalized.get("content", []) or []:
            if not isinstance(part, dict):
                continue
            p = dict(part)
            p.setdefault("type", "output_text")
            if p.get("type") == "output_text":
                p.setdefault("text", "")
                p.setdefault("annotations", [])
            content.append(p)
        normalized["content"] = content
        return normalized
    if item_type == "reasoning":
        normalized = dict(item)
        normalized.setdefault("id", f"rs_{uuid.uuid4().hex[:24]}")
        normalized.setdefault("summary", [])
        return normalized
    if item_type == "refusal":
        normalized = dict(item)
        normalized.setdefault("id", f"rf_{uuid.uuid4().hex[:24]}")
        normalized.setdefault("status", "completed")
        normalized.setdefault("content", [])
        return normalized
    if item_type == "function_call":
        normalized = dict(item)
        normalized.setdefault("id", normalized.get("call_id") or f"fc_{uuid.uuid4().hex[:24]}")
        normalized.setdefault("call_id", normalized.get("id"))
        normalized.setdefault("name", "")
        normalized.setdefault("arguments", "{}")
        normalized.setdefault("status", "completed")
        return normalized
    return dict(item)


def _sync_output_text_to_message_items(output: list[dict], output_text: str) -> list[dict]:
    synced: list[dict] = []
    replaced = False
    for item in output or []:
        normalized = _normalized_response_output_item(item)
        if normalized.get("type") == "message" and not replaced:
            for part in normalized.get("content", []) or []:
                if part.get("type") == "output_text":
                    part["text"] = output_text or ""
                    replaced = True
                    break
        synced.append(normalized)
    return synced


def _public_response_record(record: dict) -> dict:
    payload = {k: v for k, v in record.items() if not k.startswith("_")}
    for key, value in _RESPONSE_PUBLIC_DEFAULTS.items():
        if key not in payload:
            payload[key] = dict(value) if isinstance(value, dict) else list(value) if isinstance(value, list) else value
    payload["object"] = "response"
    payload["output"] = [_normalized_response_output_item(item) for item in _ensure_list(payload.get("output"))]
    if not isinstance(payload.get("metadata"), dict):
        payload["metadata"] = {}
    if not isinstance(payload.get("reasoning"), dict):
        payload["reasoning"] = {}
    if not isinstance(payload.get("text"), dict):
        payload["text"] = {"format": {"type": "text"}}
    if payload.get("usage") is not None and not isinstance(payload.get("usage"), dict):
        payload["usage"] = None
    payload["output_text"] = payload.get("output_text") or _extract_output_text(payload["output"])
    return payload


def _apply_structured_output_contract(record: dict) -> dict:
    text_config = record.get("text") if isinstance(record.get("text"), dict) else {"format": {"type": "text"}}
    output_text = _extract_output_text(record.get("output", []))
    normalized_text = _normalize_structured_output_text(output_text, text_config)
    record = dict(record)
    record["output_text"] = normalized_text
    record["output"] = _sync_output_text_to_message_items(record.get("output", []), normalized_text)
    error = _structured_output_error(normalized_text, text_config)
    if error and record.get("status") == "completed":
        record["status"] = "failed"
        record["completed_at"] = None
        record["error"] = error
        record["incomplete_details"] = None
    return record


def _response_output_from_chat_message(msg: dict) -> list[dict]:
    output: list[dict] = []
    reasoning = msg.get("reasoning_content", "")
    if reasoning:
        output.append(_response_reasoning_item(reasoning))
    refusal = msg.get("refusal", "")
    if isinstance(refusal, str) and refusal:
        output.append(_response_refusal_item(refusal))
    content = msg.get("content", "")
    if isinstance(content, str) and content:
        output.append(_response_text_item(content))
    tool_calls = msg.get("tool_calls") or []
    for tc in tool_calls:
        output.append(_response_function_call_item(tc))
    if not output:
        output.append(_response_text_item(""))
    return output


def _assistant_message_from_chat_message(msg: dict) -> dict:
    assistant = {
        "role": "assistant",
        "content": msg.get("content"),
    }
    if msg.get("reasoning_content"):
        assistant["reasoning_content"] = msg.get("reasoning_content")
    if msg.get("refusal"):
        assistant["refusal"] = msg.get("refusal")
    if msg.get("tool_calls"):
        assistant["tool_calls"] = msg.get("tool_calls")
    return assistant


def _chat_completion_to_response_record(body: dict, response_id: str, response_json: dict, messages: list[dict]) -> dict:
    choice = (response_json.get("choices") or [{}])[0]
    msg = choice.get("message", {}) or {}
    finish_reason = choice.get("finish_reason", "stop")
    created = int(response_json.get("created", int(time.time())))
    model = response_json.get("model") or body.get("model", "deepseek-default")
    output = _response_output_from_chat_message(msg)
    full_messages = messages + [_assistant_message_from_chat_message(msg)]
    record = _build_responses_record(
        response_id=response_id,
        body=body,
        model=model,
        created=created,
        completed_at=created if _response_status_from_finish_reason(finish_reason) == "completed" else None,
        output=output,
        usage=response_json.get("usage") or {},
        messages=full_messages,
        status=_response_status_from_finish_reason(finish_reason),
        incomplete_details=_response_incomplete_details(finish_reason),
    )
    return _apply_structured_output_contract(record)


_RESPONSE_TERMINAL_STATUSES = {"completed", "failed", "incomplete", "cancelled"}


def _runtime_metadata(kind: str, status: str, *, source_response_id: str | None = None) -> dict:
    now = int(time.time())
    runtime = {
        "kind": kind,
        "status": status,
        "cancel_requested": False,
        "queued_at": now if status == "queued" else None,
        "started_at": now if status == "in_progress" else None,
        "completed_at": now if status in _RESPONSE_TERMINAL_STATUSES else None,
        "cancelled_at": now if status == "cancelled" else None,
        "source_response_id": source_response_id,
    }
    return {k: v for k, v in runtime.items() if v is not None}


def _with_runtime(record: dict, runtime: dict | None = None, events: list[dict] | None = None) -> dict:
    copy = dict(record)
    current = copy.get("_runtime") if isinstance(copy.get("_runtime"), dict) else {}
    merged = dict(current)
    if runtime:
        merged.update(runtime)
    if copy.get("status") in _RESPONSE_TERMINAL_STATUSES and "completed_at" not in merged:
        merged["completed_at"] = int(time.time())
    copy["_runtime"] = merged
    if events is not None:
        copy["_events"] = events
    return copy


def _response_cancelled_record(record: dict) -> dict:
    now = int(time.time())
    cancelled = dict(record)
    cancelled["status"] = "cancelled"
    cancelled["completed_at"] = now
    cancelled["error"] = None
    cancelled["incomplete_details"] = None
    runtime = dict(cancelled.get("_runtime") or {})
    runtime.update({
        "status": "cancelled",
        "cancel_requested": True,
        "cancelled_at": now,
        "completed_at": now,
    })
    cancelled["_runtime"] = runtime
    cancelled["_events"] = _response_replay_events(cancelled, persistable=True)
    return cancelled


def _response_failed_record(response_id: str, body: dict, model_name: str, messages: list[dict],
                            previous_response_id: str | None, error: dict) -> dict:
    now = int(time.time())
    failed = _response_failed_payload(response_id, now, model_name, body, previous_response_id, error)
    failed["_messages"] = messages
    failed["_input"] = _stored_input_items(body)
    return _with_runtime(failed, _runtime_metadata("background", "failed"), _response_replay_events(failed, persistable=True))


def _count_response_input_tokens(input_value: Any, instructions: str | None = None, tools: list[dict] | None = None) -> int:
    body = {"input": input_value}
    if instructions:
        body["instructions"] = instructions
    messages, parsed_tools = _messages_from_responses_request(body)
    normalized_tools = _normalize_response_tools({"tools": tools or []}, parsed_tools)
    return _count_tokens(convert_messages_for_deepseek(messages, normalized_tools))


def _response_replay_events(record: dict, *, persistable: bool = False, starting_after: int | str | None = None) -> list[dict]:
    if not persistable and isinstance(record.get("_events"), list) and record.get("_events"):
        events = [dict(event) for event in record.get("_events", []) if isinstance(event, dict)]
    else:
        status = record.get("status", "completed")
        terminal_record = _public_response_record(record)
        sequence_number = 0

        def event(payload: dict) -> dict:
            nonlocal sequence_number
            sequence_number += 1
            copy = dict(payload)
            copy["sequence_number"] = sequence_number
            return copy

        events = [
            event({
                "type": "response.created",
                "response": _response_object_payload(record, status="in_progress", completed_at=None, usage=None),
            }),
            event({
                "type": "response.in_progress",
                "response": _response_object_payload(record, status="in_progress", completed_at=None, usage=None),
            }),
        ]
        for output_index, item in enumerate(record.get("output", []) or []):
            events.append(event({
                "type": "response.output_item.added",
                "output_index": output_index,
                "item": item,
            }))
            if item.get("type") == "reasoning":
                summary = item.get("summary", []) or []
                text = summary[0].get("text", "") if summary and isinstance(summary[0], dict) else ""
                if text:
                    events.append(event({
                        "type": "response.reasoning_text.delta",
                        "item_id": item.get("id"),
                        "output_index": output_index,
                        "content_index": 0,
                        "delta": text,
                    }))
                    events.append(event({
                        "type": "response.reasoning_text.done",
                        "item_id": item.get("id"),
                        "output_index": output_index,
                        "content_index": 0,
                        "text": text,
                    }))
            elif item.get("type") == "message":
                for content_index, content in enumerate(item.get("content", []) or []):
                    events.append(event({
                        "type": "response.content_part.added",
                        "item_id": item.get("id"),
                        "output_index": output_index,
                        "content_index": content_index,
                        "part": content,
                    }))
                    if content.get("type") == "output_text":
                        text = content.get("text", "") or ""
                        if text:
                            events.append(event({
                                "type": "response.output_text.delta",
                                "item_id": item.get("id"),
                                "output_index": output_index,
                                "content_index": content_index,
                                "delta": text,
                            }))
                            events.append(event({
                                "type": "response.output_text.done",
                                "item_id": item.get("id"),
                                "output_index": output_index,
                                "content_index": content_index,
                                "text": text,
                            }))
                    events.append(event({
                        "type": "response.content_part.done",
                        "item_id": item.get("id"),
                        "output_index": output_index,
                        "content_index": content_index,
                        "part": content,
                    }))
            elif item.get("type") == "refusal":
                for content_index, content in enumerate(item.get("content", []) or []):
                    text = content.get("text", "") or ""
                    if text:
                        events.append(event({
                            "type": "response.refusal.delta",
                            "item_id": item.get("id"),
                            "output_index": output_index,
                            "content_index": content_index,
                            "delta": text,
                        }))
                        events.append(event({
                            "type": "response.refusal.done",
                            "item_id": item.get("id"),
                            "output_index": output_index,
                            "content_index": content_index,
                            "text": text,
                        }))
            elif item.get("type") == "function_call":
                events.append(event({
                    "type": "response.function_call_arguments.delta",
                    "item_id": item.get("id"),
                    "output_index": output_index,
                    "delta": item.get("arguments", "{}"),
                }))
                events.append(event({
                    "type": "response.function_call_arguments.done",
                    "item_id": item.get("id"),
                    "output_index": output_index,
                    "arguments": item.get("arguments", "{}"),
                }))
            events.append(event({
                "type": "response.output_item.done",
                "output_index": output_index,
                "item": item,
            }))
        events.append(event({
            "type": _response_terminal_event_type(status),
            "response": terminal_record,
        }))

    if starting_after is None:
        return events
    try:
        cursor = int(starting_after)
    except (TypeError, ValueError):
        cursor = -1
    return [event for event in events if int(event.get("sequence_number", 0) or 0) > cursor]


async def _response_replay_stream(record: dict, starting_after: int | str | None = None):
    for event in _response_replay_events(record, starting_after=starting_after):
        yield _sse_json(event)
    yield "data: [DONE]\n\n"


def _responses_error(message: str, code: int | None = None, err_type: str = "server_error") -> dict:
    err = {"message": message, "type": err_type}
    if code is not None:
        err["code"] = code
    return {"error": err}


def _sse_json(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def _json_from_response(resp: JSONResponse) -> dict:
    body = resp.body.decode("utf-8", errors="ignore") if isinstance(resp.body, (bytes, bytearray)) else str(resp.body)
    return json.loads(body)


async def _single_response_stream(record: dict):
    sequence_number = 0

    def _event_payload(payload: dict) -> dict:
        nonlocal sequence_number
        sequence_number += 1
        payload["sequence_number"] = sequence_number
        return payload

    yield _sse_json(_event_payload({
        "type": "response.created",
        "response": _response_object_payload(record, status="in_progress", completed_at=None, usage=None)
    }))
    yield _sse_json(_event_payload({
        "type": "response.in_progress",
        "response": _response_object_payload(record, status="in_progress", completed_at=None, usage=None)
    }))
    output = record.get("output", [])
    for output_index, item in enumerate(output):
        yield _sse_json(_event_payload({
            "type": "response.output_item.added",
            "output_index": output_index,
            "item": item,
        }))
        if item.get("type") == "reasoning":
            text = ""
            summary = item.get("summary", [])
            if summary and isinstance(summary, list):
                text = summary[0].get("text", "") or ""
            if text:
                yield _sse_json(_event_payload({
                    "type": "response.reasoning_text.delta",
                    "item_id": item.get("id"),
                    "output_index": output_index,
                    "content_index": 0,
                    "delta": text,
                }))
                yield _sse_json(_event_payload({
                    "type": "response.reasoning_text.done",
                    "item_id": item.get("id"),
                    "output_index": output_index,
                    "content_index": 0,
                    "text": text,
                }))
        elif item.get("type") == "message":
            for content_index, content in enumerate(item.get("content", []) or []):
                yield _sse_json(_event_payload({
                    "type": "response.content_part.added",
                    "item_id": item.get("id"),
                    "output_index": output_index,
                    "content_index": content_index,
                    "part": content,
                }))
                if content.get("type") == "output_text":
                    text = content.get("text", "") or ""
                    if text:
                        yield _sse_json(_event_payload({
                            "type": "response.output_text.delta",
                            "item_id": item.get("id"),
                            "output_index": output_index,
                            "content_index": content_index,
                            "delta": text,
                        }))
                        yield _sse_json(_event_payload({
                            "type": "response.output_text.done",
                            "item_id": item.get("id"),
                            "output_index": output_index,
                            "content_index": content_index,
                            "text": text,
                        }))
                yield _sse_json(_event_payload({
                    "type": "response.content_part.done",
                    "item_id": item.get("id"),
                    "output_index": output_index,
                    "content_index": content_index,
                    "part": content,
                }))
        elif item.get("type") == "refusal":
            for content_index, content in enumerate(item.get("content", []) or []):
                text = content.get("text", "") or ""
                if text:
                    yield _sse_json(_event_payload({
                        "type": "response.refusal.delta",
                        "item_id": item.get("id"),
                        "output_index": output_index,
                        "content_index": content_index,
                        "delta": text,
                    }))
                    yield _sse_json(_event_payload({
                        "type": "response.refusal.done",
                        "item_id": item.get("id"),
                        "output_index": output_index,
                        "content_index": content_index,
                        "text": text,
                    }))
        elif item.get("type") == "function_call":
            yield _sse_json(_event_payload({
                "type": "response.function_call_arguments.delta",
                "item_id": item.get("id"),
                "output_index": output_index,
                "delta": item.get("arguments", "{}"),
            }))
            yield _sse_json(_event_payload({
                "type": "response.function_call_arguments.done",
                "item_id": item.get("id"),
                "output_index": output_index,
                "arguments": item.get("arguments", "{}"),
            }))
        yield _sse_json(_event_payload({
            "type": "response.output_item.done",
            "output_index": output_index,
            "item": item,
        }))
    yield _sse_json(_event_payload({
        "type": _response_terminal_event_type(record.get("status", "completed")),
        "response": _public_response_record(record),
    }))


class _SyntheticRequest:
    def __init__(self, source_request: Request, body: dict):
        self._body = body
        self.headers = source_request.headers

    async def json(self):
        return self._body


async def _run_background_response(source_request: Request, body: dict, chat_body: dict, messages: list[dict],
                                   response_id: str, model: str, previous_response_id: str | None) -> None:
    def mark_started(record: dict) -> dict:
        if record.get("status") == "cancelled":
            return record
        runtime = dict(record.get("_runtime") or {})
        runtime.update({"status": "in_progress", "started_at": int(time.time())})
        record["status"] = "in_progress"
        record["_runtime"] = runtime
        return record

    started = update_response_record(response_id, mark_started)
    if started and started.get("status") == "cancelled":
        return

    try:
        chat_result = await chat(_SyntheticRequest(source_request, chat_body))
        if isinstance(chat_result, JSONResponse):
            response_json = _json_from_response(chat_result)
            final_record = _chat_completion_to_response_record(body, response_id, response_json, messages)
            runtime = _runtime_metadata("background", final_record.get("status", "completed"))
            runtime["started_at"] = (started.get("_runtime") or {}).get("started_at", int(time.time())) if started else int(time.time())
            final_record = _with_runtime(final_record, runtime)
            final_record["_events"] = _response_replay_events(final_record, persistable=True)
        else:
            final_record = _response_failed_record(
                response_id,
                body,
                model,
                messages,
                previous_response_id,
                {"message": "unexpected non-JSON response", "type": "server_error"},
            )
    except Exception as exc:
        final_record = _response_failed_record(
            response_id,
            body,
            model,
            messages,
            previous_response_id,
            {"message": str(exc), "type": "server_error"},
        )

    def finish(record: dict) -> dict:
        runtime = dict(record.get("_runtime") or {})
        if record.get("status") == "cancelled" or runtime.get("cancel_requested"):
            return _response_cancelled_record(record)
        return final_record

    update_response_record(response_id, finish)

# ── cURL 解析 ──────────────────────────────────────────
def parse_curl(curl: str) -> dict:
    try:
        tokens = shlex.split(curl)
    except ValueError:
        tokens = curl.replace("\\\n", " ").split()
    out = {"url": "", "headers": {}, "body": ""}
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t == "curl": i += 1; continue
        if t in ("-H", "--header") and i + 1 < len(tokens):
            line = tokens[i + 1]
            if ":" in line:
                k, _, v = line.partition(":")
                out["headers"][k.strip().lower()] = v.strip()
            i += 2
        elif t in ("--data-raw", "--data", "--data-binary", "-d") and i + 1 < len(tokens):
            out["body"] = tokens[i + 1]; i += 2
        elif t in ("-X", "--request"): i += 2 if i + 1 < len(tokens) else 1
        elif t.startswith("-"): i += 1
        else: out["url"] = t; i += 1
    return out


def build_config(parsed: dict) -> dict:
    h = parsed["headers"]
    token = ""
    ah = h.get("authorization", "")
    if ah.startswith("Bearer "): token = ah[7:]

    session_id = ""
    for src in [parsed.get("url", ""), parsed.get("body", "")]:
        m = re.search(r"[sS]ession[_-]?[iI]d[=:\"]+([a-f0-9-]{36})", src)
        if m: session_id = m.group(1); break
    ref = h.get("referer", "")
    m = re.search(r"/a/chat/s/([a-f0-9-]+)", ref)
    if m: session_id = m.group(1)

    return {
        "token": token,
        "session_id": session_id,
        "headers": h,
        "cookie": h.get("cookie", ""),
        "url": parsed.get("url", ""),
    }


app = FastAPI(title="DeepSeek Proxy")


@app.on_event("startup")
async def startup_discover():
    """启动时自动刷新模型列表。"""
    print("[启动] 探测模型列表...")
    _discover_models()

# ── 管理页面 ─────────────────────────────────────────────
ADMIN = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>DeepSeek Proxy</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:system-ui,sans-serif;background:#0f172a;color:#e2e8f0;min-height:100vh;display:flex;justify-content:center;align-items:flex-start;padding-top:40px}
.c{background:#1e293b;border-radius:16px;padding:32px;width:600px;max-width:95vw;border:1px solid #334155}
h1{font-size:22px;margin-bottom:20px}
.s{display:flex;align-items:center;gap:8px;padding:12px 16px;border-radius:10px;margin-bottom:20px;font-size:14px}
.ok{background:#064e3b;color:#6ee7b7}.no{background:#1e293b;color:#94a3b8}.err{background:#450a0a;color:#fca5a5}
.d{width:10px;height:10px;border-radius:50%;display:inline-block}
.dg{background:#22c55e}.dy{background:#64748b}.dr{background:#ef4444}
.step{margin-bottom:18px}.sl{font-size:13px;color:#94a3b8;margin-bottom:6px}
.btn{padding:10px 20px;border-radius:8px;border:none;cursor:pointer;font-size:14px;font-weight:500}
.bp{background:#2563eb;color:#fff;width:100%}.bp:hover{background:#1d4ed8}
.bp:disabled{background:#1e3a5f;color:#64748b;cursor:not-allowed}
input[type=text],input[type=password],input[type=tel],input[type=email]{width:100%;padding:12px 14px;background:#0f172a;border:1px solid #334155;border-radius:8px;color:#e2e8f0;font-size:14px;font-family:inherit}
input:focus{outline:none;border-color:#3b82f6}
.row{display:flex;gap:12px;margin-bottom:14px}
.row .ac{width:90px;flex-shrink:0}
.row .ph{flex:1}
.pw-row{margin-bottom:14px}
.pw-row input{width:100%}
.tab-bar{display:flex;gap:0;margin-bottom:16px;border-radius:8px;overflow:hidden;border:1px solid #334155}
.tab{flex:1;padding:10px;text-align:center;font-size:13px;cursor:pointer;background:#0f172a;color:#94a3b8;transition:all .2s}
.tab.active{background:#2563eb;color:#fff}
.tab:hover:not(.active){background:#1e293b}
.panel{display:none}.panel.active{display:block}
hr{border:none;border-top:1px solid #334155;margin:24px 0}
.cfg{background:#0f172a;border-radius:10px;padding:16px}
.cr{display:flex;justify-content:space-between;align-items:center;padding:6px 0;font-size:13px}
.cr code{background:#1e293b;padding:2px 8px;border-radius:4px;font-size:13px;color:#7dd3fc;cursor:pointer}
.info{font-size:12px;color:#94a3b8;margin-top:8px;padding:8px 12px;background:#0f172a;border-radius:8px;border-left:3px solid #3b82f6;display:none}
.toast{position:fixed;top:20px;right:20px;padding:12px 20px;border-radius:8px;font-size:14px;z-index:999;display:none}
.ts{display:block;background:#064e3b;color:#6ee7b7}.te{display:block;background:#7f1d1d;color:#fca5a5}
/* Usage table */
.ut{width:100%;border-collapse:collapse;font-size:13px;margin-top:12px}
.ut th,.ut td{padding:10px 12px;text-align:right;border-bottom:1px solid #334155}
.ut th{color:#94a3b8;font-weight:500;font-size:11px;white-space:nowrap;position:sticky;top:0;background:#0f172a;z-index:1}
.ut td{font-variant-numeric:tabular-nums}
.ut tr:last-child td{border-bottom:none}
.ut .ml{text-align:left}
.ut .tr{font-weight:600;border-top:2px solid #2563eb}
.ut .tr td{padding-top:14px;color:#93c5fd;background:#0f172a;position:sticky;bottom:0}
.us{max-height:440px;overflow-y:auto}
.ue{text-align:center;color:#64748b;padding:40px 20px}
/* Period buttons */
.pb{padding:8px 16px;border-radius:8px;border:1px solid #334155;background:transparent;color:#e2e8f0;font-size:13px;cursor:pointer}
.pb:hover{background:#1e293b;border-color:#2563eb}
.pb.ac{background:#2563eb;color:#fff;border-color:#2563eb}
.period-btn.active{background:#2563eb;color:#fff}
a{color:#7dd3fc}
.collapse{cursor:pointer;user-select:none;color:#64748b;font-size:12px;margin-top:8px}
.collapse:hover{color:#94a3b8}
.curl-box{display:none;margin-top:10px}
</style>
</head>
<body>
<div class="c">
<h1>DeepSeek Proxy</h1>
<div id="s" class="s no"><span id="sd" class="d dy"></span><span id="st">等待配置</span></div>

<div class="tab-bar">
<div class="tab active" onclick="switchTab('phone')">手机号登录</div>
<div class="tab" onclick="switchTab('email')">邮箱登录</div>
<div class="tab" onclick="switchTab('usage')">用量统计</div>
</div>

<div id="phonePanel" class="panel active">
<div class="row">
<input class="ac" type="tel" id="area_code" value="+86" placeholder="+86">
<input class="ph" type="tel" id="mobile" placeholder="手机号" autocomplete="tel">
</div>
<div class="pw-row"><input type="password" id="pw1" placeholder="密码" autocomplete="current-password"></div>
<button class="btn bp" id="btn1" onclick="doLogin('phone')">登录</button>
</div>

<div id="emailPanel" class="panel">
<div class="pw-row"><input type="email" id="email" placeholder="邮箱地址" autocomplete="email"></div>
<div class="pw-row"><input type="password" id="pw2" placeholder="密码" autocomplete="current-password"></div>
<button class="btn bp" id="btn2" onclick="doLogin('email')">登录</button>
</div>

<div class="info" id="info"></div>

<div id="apiSection">
<div class="collapse" onclick="toggleCurl()">高级: 手动粘贴 cURL ▾</div>
<div class="curl-box" id="curlBox">
<textarea id="curl" placeholder="粘贴 cURL ..." style="width:100%;height:120px;background:#0f172a;border:1px solid #334155;border-radius:8px;color:#e2e8f0;padding:12px;font-family:monospace;font-size:11px;resize:vertical;margin-top:8px"></textarea>
<button class="btn bp" id="btn3" onclick="saveCurl()" style="margin-top:8px">保存 cURL</button>
</div>

<hr>
<div class="step">
<div class="sl" style="font-weight:600;color:#e2e8f0;">API 配置</div>
<div class="cfg">
<div class="cr"><span>API 地址</span><code onclick="cp(this)">http://localhost:""" + str(PROXY_PORT) + """/v1</code></div>
<div class="cr"><span>API Key</span><code onclick="cp(this)">任意填写</code></div>

</div>
</div>
<div class="step" style="margin-top:16px">
<button class="btn" style="background:#334155;color:#e2e8f0;width:100%;font-size:13px" onclick="refreshModels()" id="refreshBtn">🔄 刷新模型列表</button>
<div id="modelsInfo" style="margin-top:8px;font-size:12px;color:#64748b;display:none"></div>
</div>
</div>

<div id="usagePanel" class="panel">
<div id="usageContent"></div>
<div style="margin-top:14px">
<button class="pb ac" onclick="switchPeriod('total')" id="pbTotal">全部</button>
<button class="pb" onclick="switchPeriod('week')" id="pbWeek">本周</button>
<button class="pb" onclick="switchPeriod('today')" id="pbToday">今日</button>
<button class="btn" style="background:#334155;color:#e2e8f0;font-size:12px;padding:6px 12px;margin-left:8px" onclick="loadUsage()">刷新</button>
<button class="btn" style="background:#7f1d1d;color:#fca5a5;font-size:12px;padding:6px 12px;margin-left:4px" onclick="clearUsage()">清空</button>
</div>
</div>
</div>
<div id="toast" class="toast"></div>
<script>
function Q(id){return document.getElementById(id)}
function switchTab(type){
var ti={'phone':0,'email':1,'usage':2};
document.querySelectorAll('.tab').forEach((t,i)=>{t.className='tab'+(i===ti[type]?' active':'')});
Q('phonePanel').className='panel'+(type==='phone'?' active':'');
Q('emailPanel').className='panel'+(type==='email'?' active':'');
if(Q('usagePanel'))Q('usagePanel').className='panel'+(type==='usage'?' active':'');
var as=Q('apiSection');if(as)as.style.display=type==='usage'?'none':'';
if(type==='usage')loadUsage();
}
async function cs(){
try{const r=await fetch('/api/config');const d=await r.json()
if(d.configured){Q('s').className='s ok';Q('sd').className='d dg';Q('st').textContent='已配置 | '+d.masked}
else{Q('s').className='s no';Q('sd').className='d dy';Q('st').textContent=d.error||'等待配置'}
}catch(e){Q('s').className='s err';Q('st').textContent='连接失败'}
}
async function doLogin(type){
let body={}
if(type==='phone'){
const m=Q('mobile').value.trim();const p=Q('pw1').value;const a=Q('area_code').value.trim()
if(!m||!p){t('请输入手机号和密码',1);return}
body={mobile:m,password:p,area_code:a,login_type:'phone'}
var btn=Q('btn1')
}else{
const e=Q('email').value.trim();const p=Q('pw2').value
if(!e||!p){t('请输入邮箱和密码',1);return}
body={email:e,password:p,login_type:'email'}
var btn=Q('btn2')
}
btn.disabled=true;btn.textContent='登录中...'
Q('info').style.display='block';Q('info').innerHTML='正在登录 DeepSeek...'
try{
const r=await fetch('/api/login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})
const d=await r.json()
if(d.ok){Q('info').innerHTML='登录成功 | Token: '+d.masked+' | Session: '+d.session_id;t('登录成功');cs()}
else{Q('info').innerHTML='失败: '+d.error;t(d.error,1)}
}catch(e){Q('info').innerHTML='错误: '+e.message;t(e.message,1)}
btn.disabled=false;btn.textContent='登录'
}
function toggleCurl(){const b=Q('curlBox');b.style.display=b.style.display==='block'?'none':'block'}
async function saveCurl(){
const c=Q('curl').value.trim();if(!c){t('请先粘贴 cURL',1);return}
const b=Q('btn3');b.disabled=true;b.textContent='保存中...'
Q('info').style.display='block';Q('info').innerHTML='解析中...'
try{
const r=await fetch('/api/config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({curl:c})})
const d=await r.json()
if(d.ok){Q('info').innerHTML='OK | '+d.masked+' | Session '+d.session_id;t('已保存');cs()}
else{Q('info').innerHTML='失败: '+d.error;t(d.error,1)}
}catch(e){Q('info').innerHTML='错误: '+e.message;t(e.message,1)}
b.disabled=false;b.textContent='保存 cURL'
}
function cp(el){navigator.clipboard.writeText(el.textContent);t('已复制')}
function t(m,e){const x=Q('toast');x.textContent=m;x.className='toast t'+(e?'e':'s');setTimeout(()=>x.className='toast',2500)}
async function refreshModels(){
const btn=Q('refreshBtn');const info=Q('modelsInfo')
btn.disabled=true;btn.textContent='刷新中...';info.style.display='none'
try{
const r=await fetch('/v1/models/refresh',{method:'POST'})
const d=await r.json()
const names=d.data.map(m=>m.id).join(', ')
info.style.display='block';info.innerHTML='✅ 发现 '+d.data.length+' 个模型: '+names;t('刷新成功')
}catch(e){info.style.display='block';info.innerHTML='❌ 失败: '+e.message;t('刷新失败',1)}
btn.disabled=false;btn.textContent='🔄 刷新模型列表'
}
// === 用量统计 ===
var _up='total';
function f(n){return n.toLocaleString()}
async function loadUsage(){
try{
const r=await fetch('/api/usage');const d=await r.json();
const p=d[_up]||d.total||{};const m=p.models||{};const t=p.total||{};
const e=Object.entries(m).sort((a,b)=>b[1].total_tokens-a[1].total_tokens);
if(!e.length&&!t.requests){Q('usageContent').innerHTML='<div class=ue>📊 暂无用量数据</div>';return}
let h='<div class=us><table class=ut><thead><tr><th class=ml>模型</th><th>请求</th><th>输入</th><th>输出</th><th>总计</th></tr></thead><tbody>';
for(const[k,v]of e){h+=`<tr><td class=ml>${k}</td><td>${f(v.requests)}</td><td>${f(v.prompt_tokens)}</td><td>${f(v.completion_tokens)}</td><td>${f(v.total_tokens)}</td></tr>`}
h+=`<tr class=tr><td class=ml>📋 合计</td><td>${f(t.requests)}</td><td>${f(t.prompt_tokens)}</td><td>${f(t.completion_tokens)}</td><td>${f(t.total_tokens)}</td></tr></tbody></table></div>`;
Q('usageContent').innerHTML=h
}catch(e){Q('usageContent').innerHTML='<div class=ue>加载失败: '+e.message+'</div>'}
}
function switchPeriod(p){
_up=p;
['total','week','today'].forEach(x=>{var b=Q('pb'+x.charAt(0).toUpperCase()+x.slice(1));if(b)b.className='pb'+(x===p?' ac':'')});
loadUsage()
}
async function clearUsage(){
if(!confirm('确定清空全部用量数据？'))return;
try{await fetch('/api/usage',{method:'DELETE'});t('已清空');loadUsage()}catch(e){t('清空失败',1)}
}
cs()
</script>
</body>
</html>"""


from starlette.responses import RedirectResponse

@app.get("/")
async def root():
    return RedirectResponse(url="/admin")


@app.get("/admin", response_class=HTMLResponse)
async def admin():
    from starlette.responses import Response
    html = ADMIN
    return Response(content=html, media_type="text/html", headers={
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
    })


# ── 配置 API ─────────────────────────────────────────────

def _load_config_sync() -> dict:
    """同步加载 token.json 原始数据（供非 async 上下文使用）。"""
    if not CONFIG_FILE.exists():
        return {}
    return json.loads(CONFIG_FILE.read_text("utf-8"))


@app.get("/api/config")
async def get_config():
    if not CONFIG_FILE.exists():
        return {"configured": False, "error": "未配置"}
    d = _load_config_sync()
    t = d.get("token", "")
    return {
        "configured": True,
        "masked": t[:20] + "..." + t[-8:] if len(t) > 30 else "***",
        "session_id": d.get("session_id", "N/A"),
    }


@app.post("/api/config")
async def save_config(data: dict):
    curl = data.get("curl", "").strip()
    if not curl: raise HTTPException(400, "请提供 cURL")
    parsed = parse_curl(curl)
    cfg = build_config(parsed)
    if not cfg["token"]: return {"ok": False, "error": "未从 cURL 提取到 Token，请确认 Authorization header"}
    if not cfg["session_id"]: return {"ok": False, "error": "未从 cURL 提取到 Session ID"}
    CONFIG_FILE.write_text(json.dumps(cfg, ensure_ascii=False), "utf-8")
    t = cfg["token"]
    return {"ok": True, "masked": t[:20] + "..." + t[-8:], "session_id": cfg["session_id"]}


# ── DeepSeek 登录 API ─────────────────────────────────────
@app.post("/api/login")
async def deepseek_login(data: dict):
    login_type = data.get("login_type", "phone")
    password = data.get("password", "").strip()
    if not password:
        raise HTTPException(400, "请提供密码")

    # 构造登录 payload（参考 NIyueeE/ds-free-api: email 和 mobile 二选一）
    login_payload = {"password": password, "device_id": secrets.token_hex(16), "os": "web"}
    account_label = ""
    email, mobile, area_code = "", "", "+86"

    if login_type == "email":
        email = data.get("email", "").strip()
        if not email:
            raise HTTPException(400, "请提供邮箱")
        login_payload["email"] = email
        login_payload["mobile"] = ""
        login_payload["area_code"] = ""
        account_label = email
    else:
        mobile = data.get("mobile", "").strip()
        area_code = data.get("area_code", "+86").strip()
        if not mobile:
            raise HTTPException(400, "请提供手机号")
        login_payload["mobile"] = mobile
        login_payload["area_code"] = area_code
        login_payload["email"] = ""
        account_label = f"{area_code} {mobile}"

    DS_HEADERS = {
        "content-type": "application/json",
        "origin": "https://chat.deepseek.com",
        "referer": "https://chat.deepseek.com/",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/134.0.0.0 Safari/537.36",
        "x-client-version": "2.0.2",
        "x-client-platform": "web",
    }

    try:
        # 1. 登录
        login_resp = cffi_requests.post(
            "https://chat.deepseek.com/api/v0/users/login",
            json=login_payload,
            headers=DS_HEADERS,
            impersonate="chrome120",
            timeout=30,
        )

        login_data = login_resp.json()
        outer_code = login_data.get("code", 0)
        data_block = login_data.get("data") or {}
        biz_code = data_block.get("biz_code", 0)
        biz_msg = data_block.get("biz_msg", "")

        if login_resp.status_code != 200 or outer_code != 0 or biz_code != 0:
            err_msg = biz_msg or login_data.get("msg") or f"HTTP {login_resp.status_code}/code={outer_code}/biz_code={biz_code}"
            return {"ok": False, "error": f"登录失败: {err_msg}"}

        biz_data = data_block.get("biz_data") or {}
        token = biz_data.get("user", {}).get("token", "")
        if not token:
            return {"ok": False, "error": f"登录失败: biz_data 中无 token（biz_msg={biz_msg}）"}

        print(f"[Login] Token acquired for {account_label}: {token[:20]}...{token[-8:]}")

        # 2. 创建会话
        auth_headers = {**DS_HEADERS, "authorization": f"Bearer {token}"}
        session_resp = cffi_requests.post(
            "https://chat.deepseek.com/api/v0/chat_session/create",
            json={},
            headers=auth_headers,
            impersonate="chrome120",
            timeout=15,
        )

        session_id = ""
        if session_resp.status_code == 200:
            session_data = session_resp.json()
            biz = session_data.get("data", {}).get("biz_data", {})
            session_id = biz.get("chat_session", {}).get("id", "") or biz.get("id", "")
            print(f"[Login] Session created: {session_id}")
        else:
            print(f"[Login] Session creation failed: {session_resp.status_code} {session_resp.text[:200]}")

        # 3. 保存配置（含凭证供自动刷新）
        cfg = {
            "token": token,
            "session_id": session_id,
            "headers": {**DS_HEADERS, "authorization": f"Bearer {token}"},
            "cookie": "",
            "account": account_label,
            "login_type": login_type,
            # 保存凭证用于 token 过期后自动刷新
            "_password": password,
            "_email": email if login_type == "email" else "",
            "_mobile": mobile if login_type == "phone" else "",
            "_area_code": area_code if login_type == "phone" else "+86",
        }
        CONFIG_FILE.write_text(json.dumps(cfg, ensure_ascii=False), "utf-8")

        masked = token[:20] + "..." + token[-8:]
        return {"ok": True, "masked": masked, "session_id": session_id}

    except Exception as e:
        print(f"[Login] Error: {e}")
        return {"ok": False, "error": str(e)}


# ── Health ───────────────────────────────────────────────
@app.get("/health")
async def health():
    if CONFIG_FILE.exists(): return {"status": "ok", "configured": True}
    return {"status": "waiting", "configured": False}


# ─── 用量统计 API ─────────────────────────────────────────────

@app.get("/api/usage")
async def usage_stats():
    return get_usage()


@app.delete("/api/usage")
async def clear_usage_stats():
    clear_usage()
    return {"ok": True}


# ─── 模型列表（免鉴权，供管理页面使用） ───────────────────────

@app.get("/api/models")
async def admin_models():
    return {"models": list(get_models().keys())}


# ── 模型映射（动态从 DeepSeek 探测）─────────────────
MODEL_CONFIG_URL = "https://chat.deepseek.com/api/v0/client/settings?scope=model"

_models_cache = {}       # model_id → (thinking, search, max_in, max_out)
_models_cache_time = 0
_MODELS_TTL = 3600       # 缓存1小时


def _discover_models() -> dict:
    """从 DeepSeek /api/v0/client/settings?scope=model 动态获取模型配置。

    返回: {model_id: (thinking_enabled, search_enabled, max_input, max_output), ...}
    失败返回 None。
    """
    global _models_cache, _models_cache_time

    cfg = _load_config_sync()
    if not cfg:
        return None

    token = cfg.get("token", "")
    ua = cfg.get("headers", {}).get("user-agent", "Mozilla/5.0")

    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": ua,
        "X-Client-Version": "2.0.0",
        "X-Client-Platform": "web",
    }

    try:
        resp = cffi_requests.get(MODEL_CONFIG_URL, headers=headers, timeout=10)
        data = resp.json()
        biz_data = data.get("data", {}).get("biz_data", {})
        settings = biz_data.get("settings", {})
        model_configs = settings.get("model_configs", {}).get("value", [])

        if not model_configs:
            print(f"[模型发现] model_configs 为空")
            return None

        models = {}
        for mc in model_configs:
            mt = mc.get("model_type")
            if not mt or not mc.get("enabled"):
                continue

            # 上下文大小：优先从 input_character_limit 推算 (V4 系列 ≈ 1M tokens)，
            # 对 Expert 等 UI 限制偏小的模型硬编码 1M
            icl = mc.get("input_character_limit", 0) or 0
            if icl >= 1_000_000:
                max_in = int(icl * 0.4)      # 2621440 × 0.4 ≈ 1048576 (1M)
            else:
                max_in = 1_048_576            # Expert 等硬编码 1M
            max_out = max_in                  # DeepSeek V4 输出上限即上下文大小
            has_think = mc.get("think_feature") is not None
            has_search = mc.get("search_feature") is not None

            # 基础模型
            name = f"deepseek-{mt}" if mt != "default" else "deepseek-default"
            models[name] = (False, False, max_in, max_out)
            print(f"[模型发现]   {name}: in={max_in}, out={max_out}, think={has_think}, search={has_search}")

            # 思维链变体
            if has_think:
                tname = "deepseek-reasoner" if mt == "default" else f"deepseek-{mt}-reasoner"
                models[tname] = (True, False, max_in, max_out)

            # 搜索变体
            if has_search:
                sname = "deepseek-search" if mt == "default" else f"deepseek-{mt}-search"
                models[sname] = (False, True, max_in, max_out)

            # 思考+联网 组合变体
            if has_think and has_search:
                cname = "deepseek-reasoner-search" if mt == "default" else f"deepseek-{mt}-reasoner-search"
                models[cname] = (True, True, max_in, max_out)

        if models:
            # 模型名称为纯英文ID，中文对照见 README.md
            _models_cache = models
            _models_cache_time = time.time()
            print(f"[模型发现] 发现 {len(models)} 个模型: {list(models.keys())}")
            return models

    except Exception as e:
        print(f"[模型发现] 失败: {e}")

    return None


def get_models() -> dict:
    """获取模型映射（缓存优先，过期自动刷新。发现失败返回 {}）。"""
    global _models_cache, _models_cache_time

    if _models_cache and time.time() - _models_cache_time < _MODELS_TTL:
        return _models_cache

    discovered = _discover_models()
    if discovered:
        return discovered

    # 探测失败 → 返回空（不骗人）
    print("[模型发现] 探测失败，模型列表为空")
    return {}


# ── Token 自动刷新 ─────────────────────────────────────────
def relogin(cfg: dict) -> dict | None:
    """用保存的凭证重新登录，返回新 cfg 或 None"""
    login_type = cfg.get("login_type", "")
    password = cfg.get("_password", "")
    if not password:
        print("[Token] 无保存密码，无法自动刷新")
        return None

    login_payload = {"password": password, "device_id": secrets.token_hex(16), "os": "web"}
    account_label = cfg.get("account", "")

    if login_type == "email":
        email = cfg.get("_email", "")
        if not email:
            return None
        login_payload["email"] = email
        login_payload["mobile"] = ""
        login_payload["area_code"] = ""
    elif login_type == "phone":
        mobile = cfg.get("_mobile", "")
        area_code = cfg.get("_area_code", "+86")
        if not mobile:
            return None
        login_payload["mobile"] = mobile
        login_payload["area_code"] = area_code
        login_payload["email"] = ""
    else:
        return None

    DS_HEADERS = {
        "content-type": "application/json",
        "origin": "https://chat.deepseek.com",
        "referer": "https://chat.deepseek.com/",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/134.0.0.0 Safari/537.36",
        "x-client-version": "2.0.2",
        "x-client-platform": "web",
    }

    try:
        print(f"[Token] 自动重新登录 {account_label}...")
        login_resp = cffi_requests.post(
            "https://chat.deepseek.com/api/v0/users/login",
            json=login_payload,
            headers=DS_HEADERS,
            impersonate="chrome120",
            timeout=30,
        )
        login_data = login_resp.json()
        outer_code = login_data.get("code", 0)
        data_block = login_data.get("data") or {}
        biz_code = data_block.get("biz_code", 0)
        biz_msg = data_block.get("biz_msg", "")

        if login_resp.status_code != 200 or outer_code != 0 or biz_code != 0:
            err_msg = biz_msg or login_data.get("msg") or f"HTTP {login_resp.status_code}/code={outer_code}/biz_code={biz_code}"
            print(f"[Token] 自动登录失败: {err_msg}")
            return None

        biz_data = data_block.get("biz_data") or {}
        token = biz_data.get("user", {}).get("token", "")
        if not token:
            print(f"[Token] 登录失败: biz_data 中无 token（biz_msg={biz_msg}）")
            return None

        print(f"[Token] 新 token: {token[:20]}...{token[-8:]}")

        # 创建新会话
        auth_headers = {**DS_HEADERS, "authorization": f"Bearer {token}"}
        session_resp = cffi_requests.post(
            "https://chat.deepseek.com/api/v0/chat_session/create",
            json={},
            headers=auth_headers,
            impersonate="chrome120",
            timeout=15,
        )
        session_id = ""
        if session_resp.status_code == 200:
            session_data = session_resp.json()
            biz = session_data.get("data", {}).get("biz_data", {})
            session_id = biz.get("chat_session", {}).get("id", "") or biz.get("id", "")
            print(f"[Token] 新 session: {session_id}")
        else:
            print(f"[Token] Session 创建失败: {session_resp.status_code}")

        new_cfg = {
            "token": token,
            "session_id": session_id,
            "headers": {**DS_HEADERS, "authorization": f"Bearer {token}"},
            "cookie": "",
            "account": account_label,
            "login_type": login_type,
            # 保留凭证供下次刷新
            "_password": password,
            "_email": cfg.get("_email", ""),
            "_mobile": cfg.get("_mobile", ""),
            "_area_code": cfg.get("_area_code", "+86"),
        }
        CONFIG_FILE.write_text(json.dumps(new_cfg, ensure_ascii=False), "utf-8")
        return new_cfg

    except Exception as e:
        print(f"[Token] 自动登录异常: {e}")
        return None


def load_config_with_refresh() -> dict:
    """加载配置，如果 token 失效则自动刷新"""
    if not CONFIG_FILE.exists():
        return {}
    cfg = json.loads(CONFIG_FILE.read_text("utf-8"))
    return cfg


# ── OpenAI 兼容 API ──────────────────────────────────────
@app.get("/v1/models")
async def models():
    data = []
    for mid, (think, search, mi, mo) in get_models().items():
        data.append({
            "id": mid, "object": "model", "created": 1704067200,
            "owned_by": "deepseek",
            "max_input_tokens": mi, "max_output_tokens": mo,
            "context_length": mi, "context_window": mi,
            "supported_parameters": ["tools", "tool_choice", "temperature", "max_tokens", "stream"],
        })
    return {"object": "list", "data": data}


@app.get("/v1/models/{model_id}")
async def model_detail(model_id: str):
    info = get_models().get(model_id)
    if not info:
        raise HTTPException(404, f"模型 {model_id} 不存在")
    think, search, mi, mo = info
    return {
        "id": model_id, "object": "model", "created": 1704067200,
        "owned_by": "deepseek",
        "max_input_tokens": mi, "max_output_tokens": mo,
        "context_length": mi, "context_window": mi,
    }


@app.post("/v1/models/refresh")
async def refresh_models():
    """强制刷新模型列表"""
    global _models_cache_time
    _models_cache_time = 0  # 让下次 get_models() 重新探测
    models = get_models()
    data = []
    for mid, (think, search, mi, mo) in models.items():
        data.append({
            "id": mid, "object": "model", "created": 1704067200,
            "owned_by": "deepseek",
            "max_input_tokens": mi, "max_output_tokens": mo,
            "context_length": mi, "context_window": mi,
            "supported_parameters": ["tools", "tool_choice", "temperature", "max_tokens", "stream"],
        })
    return {"object": "list", "data": data}


def build_request_headers(cfg: dict, session_id: str) -> dict:
    """Build headers for DeepSeek API request, excluding stale PoW and conflict headers."""
    # Start from saved headers
    req_headers = dict(cfg.get("headers", {}))

    # Remove stale PoW - we'll generate fresh one
    req_headers.pop("x-ds-pow-response", None)

    # Remove headers that curl_cffi manages or that conflict
    for h in ("host", "content-length", "transfer-encoding", "accept-encoding",
              "content-type"):
        req_headers.pop(h, None)

    # Ensure required headers
    req_headers["content-type"] = "application/json"
    req_headers["origin"] = "https://chat.deepseek.com"
    req_headers["referer"] = f"https://chat.deepseek.com/a/chat/s/{session_id}"

    return req_headers


def get_pow_response(target_path: str = "/api/v0/chat/completion") -> str | None:
    """Get fresh PoW response from DeepSeek."""
    try:
        cfg = json.loads(CONFIG_FILE.read_text("utf-8"))
        headers = build_request_headers(cfg, cfg["session_id"])

        resp = cffi_requests.post(
            "https://chat.deepseek.com/api/v0/chat/create_pow_challenge",
            headers=headers,
            json={"target_path": target_path},
            impersonate="chrome120",
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            challenge = data.get("data", {}).get("biz_data", {}).get("challenge", {})
            if challenge:
                pow_response = pow_solver.solve_challenge(challenge)
                print(f"[PoW] Solved: {pow_response[:50]}...")
                return pow_response
            else:
                print(f"[PoW] No challenge: {data}")
        else:
                print(f"[PoW] Request failed {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        print(f"[PoW] Error: {e}")
    return None


# ── 文件上传（Vision 模型支持）──────────────────────────────

def upload_file_to_deepseek(file_data: bytes, filename: str, content_type: str = "image/png") -> str | None:
    """Upload a file to DeepSeek and return the file_id.

    Uses the /api/v0/file/upload_file endpoint with PoW authentication.
    Returns file_id string or None on failure.
    """
    if not CONFIG_FILE.exists():
        _vlog("upload: no config")
        return None
    cfg = json.loads(CONFIG_FILE.read_text("utf-8"))
    session_id = cfg["session_id"]

    # Get PoW for upload_file scene
    pow_response = get_pow_response(target_path="/api/v0/file/upload_file")

    req_headers = build_request_headers(cfg, session_id)
    if pow_response:
        req_headers["x-ds-pow-response"] = pow_response

    # Remove content-type, let requests/curl set multipart boundary
    req_headers.pop("content-type", None)

    # curl_cffi doesn't support `files` param; use standard requests for upload
    import requests as req
    try:
        resp = req.post(
            "https://chat.deepseek.com/api/v0/file/upload_file",
            headers=req_headers,
            files={"file": (filename, file_data, content_type)},
            timeout=60,
        )
        if resp.status_code == 200:
            data = resp.json()
            file_id = (data.get("data", {})
                            .get("biz_data", {})
                            .get("id", "")
                       or data.get("data", {})
                              .get("id", ""))
            if file_id:
                _vlog(f"upload OK: {filename} -> {file_id}")
                return file_id
            _vlog(f"upload: no file_id in response: {resp.text[:300]}")
        else:
            _vlog(f"upload HTTP {resp.status_code}: {resp.text[:300]}")
    except Exception as e:
        _vlog(f"upload error: {e}")
    return None


def fork_file_to_vision(cfg: dict, file_id: str) -> str | None:
    """Fork an uploaded file to the vision model type.

    DeepSeek requires forking files to a specific model before they can be
    referenced in chat. Returns the new forked file_id or None.
    """
    import requests as req
    try:
        headers = build_request_headers(cfg, cfg["session_id"])
        resp = req.post(
            "https://chat.deepseek.com/api/v0/file/fork_file_task",
            headers=headers,
            json={"file_id": file_id, "to_model_type": "vision"},
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            biz_data = data.get("data", {}).get("biz_data", {})
            forked_id = biz_data.get("id") or biz_data.get("file_id")
            if forked_id and forked_id != file_id:
                _vlog(f"fork OK: {file_id} -> {forked_id}")
                return forked_id
        _vlog(f"fork failed: {resp.status_code} {resp.text[:200]}")
    except Exception as e:
        _vlog(f"fork error: {e}")
    return None


def wait_for_file_parsing(cfg: dict, file_ids: list[str], timeout: int = 30) -> list[str]:
    """Wait for DeepSeek to finish parsing uploaded files.

    Polls /api/v0/file/fetch_files until all files are parsed or timeout.
    Returns list of successfully parsed file_ids.
    """
    import time as _time
    if not file_ids:
        return []
    start = _time.time()
    while _time.time() - start < timeout:
        statuses = _fetch_file_statuses(cfg, file_ids)
        if statuses is None:
            _time.sleep(1)
            continue
        all_done = True
        parsed_ids = []
        for fid in file_ids:
            s = statuses.get(fid, {})
            status = str(s.get("status", "")).upper()
            # Terminal states: file is processed (success or not, just done)
            if status in ("SUCCESS", "COMPLETED", "CONTENT_EMPTY", "FAILED", "ERROR", "PARSE_FAILED"):
                if status == "SUCCESS":
                    parsed_ids.append(fid)
                # Even non-success states mean the file is done processing
            elif status in ("PENDING", "PARSING", "UPLOADING", "QUEUED"):
                all_done = False
                # If it's been more than 5s and still PARSING, accept it anyway
                if _time.time() - start > 5:
                    _vlog(f"file {fid} still {status} after 5s, accepting")
                    parsed_ids.append(fid)
            else:
                # Unknown status — assume done
                _vlog(f"file {fid} unknown status={status}, accepting")
                parsed_ids.append(fid)
        if all_done and parsed_ids:
            print(f"[Vision] Files parsed: {parsed_ids}")
            return parsed_ids
        if parsed_ids and _time.time() - start > 5:
            # Some files parsed, others still processing — return what we have
            if parsed_ids:
                return parsed_ids
        _time.sleep(1)
    print(f"[Vision] Parse timeout, got 0/{len(file_ids)} files")
    return []


def _fetch_file_statuses(cfg: dict, file_ids: list[str]) -> dict | None:
    """Fetch parse status for uploaded files from DeepSeek."""
    import requests as req
    try:
        session_id = cfg["session_id"]
        headers = build_request_headers(cfg, session_id)
        resp = req.get(
            "https://chat.deepseek.com/api/v0/file/fetch_files",
            headers=headers,
            params={"file_ids": file_ids},
            timeout=15,
        )
        if resp.status_code == 200:
            data = resp.json()
            files = (data.get("data", {}).get("biz_data", {}).get("files", [])
                     or data.get("data", {}).get("files", []))
            if not files:
                # Sometimes response wraps differently
                biz = data.get("data", {}).get("biz_data", {})
                for key in ("file_statuses", "file_list", "items"):
                    if key in biz:
                        files = biz[key]
                        break
            statuses = {}
            for f in files:
                fid = f.get("id") or f.get("file_id") or f.get("_id")
                if fid and fid in file_ids:
                    statuses[fid] = f
            return statuses if statuses else None
        print(f"[Vision] fetch_files HTTP {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        print(f"[Vision] fetch_files error: {e}")
    return None


def extract_images_from_messages(messages: list) -> list[dict]:
    """Extract image URLs/bytes from OpenAI-format messages.

    Returns list of dicts: {data: bytes, content_type: str, filename: str}
    Supports: image_url (url/base64), images (list), content array
    """
    import base64 as b64
    images = []
    for msg in messages:
        content = msg.get("content", "")
        # OpenAI multi-content format
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "image_url":
                        url = part.get("image_url", {}).get("url", "")
                        images.append(_parse_image_url(url))
                    elif part.get("type") == "image":
                        data = part.get("data", "") or part.get("source", {}).get("data", "")
                        if data:
                            images.append(_parse_image_url(data))
        elif isinstance(content, str):
            # Check for images array in msg
            imgs = msg.get("images", [])
            for img in imgs:
                if isinstance(img, str):
                    images.append(_parse_image_url(img))
                elif isinstance(img, dict):
                    data = img.get("data", "") or img.get("url", "")
                    if data:
                        images.append(_parse_image_url(data))
    return [img for img in images if img is not None]


def extract_text_files_from_messages(messages: list) -> list[dict]:
    """Extract text files from OpenAI-format messages.

    Returns list of dicts: {data: bytes, filename: str, content_type: str}
    Handles type="file" content parts with base64 file_data or data fields.
    """
    import base64 as b64
    files = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "file":
                    file_obj = part.get("file", {})
                    if isinstance(file_obj, dict):
                        filename = file_obj.get("filename", "file.txt")
                        file_data = file_obj.get("file_data", "") or file_obj.get("data", "")
                        if file_data:
                            try:
                                data = b64.b64decode(file_data)
                                ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "txt"
                                ct_map = {
                                    "md": "text/markdown", "py": "text/x-python",
                                    "json": "application/json", "yaml": "text/yaml",
                                    "yml": "text/yaml", "txt": "text/plain",
                                    "csv": "text/csv", "xml": "text/xml",
                                    "html": "text/html", "js": "text/javascript",
                                    "css": "text/css", "sh": "text/x-shellscript",
                                }
                                content_type = ct_map.get(ext, "text/plain")
                                files.append({
                                    "data": data,
                                    "filename": filename,
                                    "content_type": content_type,
                                })
                            except Exception:
                                continue
    return files


def _parse_image_url(url_or_data: str) -> dict | None:
    """Parse an image URL or base64 data string."""
    import base64 as b64
    if not url_or_data:
        return None
    s = url_or_data.strip()
    # base64 data URI
    if s.startswith("data:"):
        header, encoded = s.split(",", 1)
        ct = "image/png"
        for part in header.split(";")[0].split(":")[1:]:
            ct = part
        try:
            data = b64.b64decode(encoded)
            ext = ct.split("/")[-1] if "/" in ct else "png"
            return {"data": data, "content_type": ct, "filename": f"image.{ext}"}
        except Exception:
            print(f"[Vision] Failed to decode base64 image")
            return None
    # HTTP URL
    if s.startswith("http://") or s.startswith("https://"):
        try:
            resp = cffi_requests.get(s, timeout=30, impersonate="chrome120")
            if resp.status_code == 200:
                ct = resp.headers.get("content-type", "image/png")
                ext = ct.split("/")[-1] if "/" in ct else "png"
                return {"data": resp.content, "content_type": ct, "filename": f"image.{ext}"}
        except Exception as e:
            print(f"[Vision] Failed to download image: {e}")
    return None








@app.post("/v1/chat/completions")
async def chat(request: Request):
    if not CONFIG_FILE.exists():
        raise HTTPException(503, detail="请先访问 http://localhost:{}/admin 登录账号".format(PROXY_PORT))

    body = await request.json()
    messages = body.get("messages", [])
    model = body.get("model", "deepseek-default")
    stream = body.get("stream", False)
    tools = body.get("tools", None)

    # Log client info for debugging
    ua = request.headers.get("user-agent", "?")[:60]
    msg = f"[REQ] model={model} stream={stream} msgs={len(messages)} tools={bool(tools)} ua={ua}"
    print(msg, flush=True)
    _vlog(msg)

    # 模型映射
    model_info = get_models().get(model, get_models().get("deepseek-default"))
    thinking_enabled, search_enabled, _, _ = model_info

    cfg = json.loads(CONFIG_FILE.read_text("utf-8"))
    ref_file_ids = []
    import time as _vtime

    # 文本文件：上传到 DeepSeek（不 fork，等解析完直接用原始 file_id）
    text_files = extract_text_files_from_messages(messages)
    if text_files:
        _t0 = _vtime.time()
        _vlog(f"TEXT_FILES: found {len(text_files)} files")
        raw_ids = []
        for i, tf in enumerate(text_files):
            _t1 = _vtime.time()
            orig_fid = upload_file_to_deepseek(tf["data"], tf["filename"], tf["content_type"])
            _vlog(f"text_upload #{i} -> {orig_fid} ({_vtime.time()-_t1:.1f}s)")
            if orig_fid:
                raw_ids.append(orig_fid)
        if raw_ids:
            text_ids = wait_for_file_parsing(cfg, raw_ids, timeout=30)
            ref_file_ids.extend(text_ids)
            _vlog(f"TEXT_DONE: {len(text_ids)}/{len(raw_ids)} ready ({_vtime.time()-_t0:.1f}s)")

    # Vision 模型：提取、上传、fork 图片
    is_vision = "vision" in model
    if is_vision:
        _t0 = _vtime.time()
        _vlog(f"START model={model} msgs={len(messages)}")
        images = extract_images_from_messages(messages)
        _vlog(f"extracted {len(images)} images ({_vtime.time()-_t0:.1f}s)")
        for i, img in enumerate(images):
            _t1 = _vtime.time()
            orig_fid = upload_file_to_deepseek(img["data"], img["filename"], img["content_type"])
            _vlog(f"upload #{i} -> {orig_fid} ({_vtime.time()-_t1:.1f}s)")
            if orig_fid:
                _t2 = _vtime.time()
                forked_fid = fork_file_to_vision(cfg, orig_fid)
                _vlog(f"fork #{i} -> {forked_fid} ({_vtime.time()-_t2:.1f}s)")
                if forked_fid:
                    ref_file_ids.append(forked_fid)
        if ref_file_ids:
            _t3 = _vtime.time()
            ref_file_ids = wait_for_file_parsing(cfg, ref_file_ids, timeout=10)
            _vlog(f"parse_check -> {len(ref_file_ids)} ready ({_vtime.time()-_t3:.1f}s)")
        _vlog(f"DONE: {len(images)} images -> {len(ref_file_ids)} ready ({_vtime.time()-_t0:.1f}s)")

        # Create a FRESH session for vision to avoid parallel_chat_limit_by_queue
        # from any lingering requests on the main session
        try:
            token = cfg.get("token", "")
            if token:
                auth_h = {**cfg.get("headers", {}), "authorization": f"Bearer {token}"}
                sess_resp = cffi_requests.post(
                    "https://chat.deepseek.com/api/v0/chat_session/create",
                    json={}, headers=auth_h, impersonate="chrome120", timeout=15)
                if sess_resp.status_code == 200:
                    biz = sess_resp.json().get("data", {}).get("biz_data", {})
                    new_sid = biz.get("chat_session", {}).get("id", "") or biz.get("id", "")
                    if new_sid:
                        cfg = dict(cfg)
                        cfg["session_id"] = new_sid
                        _vlog(f"vision fresh session: {new_sid}")
        except Exception as e:
            _vlog(f"fresh session failed: {e}")

    # 构建 prompt：使用 convert_messages_for_deepseek 处理完整多轮对话
    prompt = convert_messages_for_deepseek(messages, tools)
    prompt_tokens = _count_tokens(prompt)

    # 会话管理：token 超限时自动建新 DeepSeek session
    if needs_renewal():
        status = get_usage_status()
        print(f"[Session] Tokens {status['prompt_tokens']}/{status['threshold']} exceeded, creating new session...")
        try:
            token = cfg.get("token", "")
            if token:
                auth_h = {**cfg.get("headers", {}), "authorization": f"Bearer {token}"}
                sess_resp = cffi_requests.post(
                    "https://chat.deepseek.com/api/v0/chat_session/create",
                    json={}, headers=auth_h, impersonate="chrome120", timeout=15)
                if sess_resp.status_code == 200:
                    biz = sess_resp.json().get("data", {}).get("biz_data", {})
                    new_sid = biz.get("chat_session", {}).get("id", "") or biz.get("id", "")
                    if new_sid:
                        cfg = dict(cfg)
                        cfg["session_id"] = new_sid
                        CONFIG_FILE.write_text(json.dumps(cfg, indent=2))
                        on_new_session("default", new_sid, model)
                        print(f"[Session] New session: {new_sid}")
        except Exception as e:
            print(f"[Session] Failed to create new session: {e}")

    # 如果有 tools 定义，将工具提示词注入到最后一个 USER 标记之前
    tool_prompt = build_tool_prompt(tools) if tools else ""
    if tool_prompt:
        # 原生格式：找最后一个 <｜User｜>
        last_user_idx = prompt.rfind("<｜User｜>")
        if last_user_idx != -1:
            prompt = prompt[:last_user_idx] + tool_prompt + "\n" + prompt[last_user_idx:]
        else:
            prompt = tool_prompt + "\n" + prompt

    has_tools = bool(tools)

    # Try streaming for all models including vision with images.
    # Old issue: vision stream put everything in thinking_content, but the new
    # fragments format (THINK/RESPONSE) should handle this correctly now.

    result = _do_chat(cfg, prompt, model, thinking_enabled, search_enabled, stream,
                    is_retry=False, has_tools=has_tools, tools=tools,
                    ref_file_ids=ref_file_ids)

    # (Vision SSE wrapper removed — all models now stream directly via fragments format)

    # 用量统计：非流式直接计数，流式包装生成器
    if stream and hasattr(result, 'body_iterator'):
        orig_iter = result.body_iterator
        async def _counted_stream():
            completion_text = ""
            async for chunk in orig_iter:
                s = chunk.decode("utf-8", errors="ignore") if isinstance(chunk, bytes) else str(chunk)
                if s.startswith("data: ") and not s.startswith("data: [DONE]"):
                    try:
                        obj = json.loads(s[6:])
                        delta = obj.get("choices", [{}])[0].get("delta", {})
                        c = delta.get("content", "") or ""
                        r = delta.get("reasoning_content", "") or ""
                        completion_text += (c + r)
                    except: pass
                yield chunk
            add_usage(model, prompt_tokens, _count_tokens(completion_text))
            add_tokens("default", cfg.get("session_id", ""), prompt_tokens)
        result.body_iterator = _counted_stream()
    else:
        add_usage(model, prompt_tokens, 0)
        add_tokens("default", cfg.get("session_id", ""), prompt_tokens)
    return result


@app.post("/v1/responses")
async def responses(request: Request):
    if not CONFIG_FILE.exists():
        raise HTTPException(503, detail="请先访问 http://localhost:{}/admin 登录账号".format(PROXY_PORT))

    body = await request.json()
    model = _resolve_responses_model(body)
    stream = body.get("stream", False)
    previous_response_id = body.get("previous_response_id")
    body["_response_id"] = _gen_response_id()

    messages, parsed_tools = _messages_from_responses_request(body)
    messages = _merge_previous_response_context(messages, previous_response_id)
    tools = _normalize_response_tools(body, parsed_tools)

    chat_body = {
        "model": model,
        "messages": messages,
        "stream": stream,
    }
    if tools:
        chat_body["tools"] = tools
    if "tool_choice" in body:
        chat_body["tool_choice"] = body.get("tool_choice")

    response_id = body["_response_id"]

    if body.get("background") is True:
        created = int(time.time())
        shell = _build_responses_record(
            response_id=response_id,
            body=body,
            model=model,
            created=created,
            completed_at=None,
            output=[],
            usage={},
            messages=messages,
            status="queued",
            incomplete_details=None,
        )
        shell["usage"] = None
        shell = _with_runtime(shell, _runtime_metadata("background", "queued"))
        if shell.get("store", True):
            save_response_record(shell)
            asyncio.create_task(_run_background_response(request, body, dict(chat_body, stream=False), messages, response_id, model, previous_response_id))
        return JSONResponse(_response_object_payload(shell, status="queued", completed_at=None, usage=None))

    if not stream:
        chat_result = await chat(_SyntheticRequest(request, chat_body))
        if isinstance(chat_result, JSONResponse):
            response_json = _json_from_response(chat_result)
            record = _chat_completion_to_response_record(body, response_id, response_json, messages)
            if record.get("store", True):
                save_response_record(record)
            return JSONResponse(_public_response_record(record))
        raise HTTPException(502, detail={"error": {"message": "unexpected non-JSON response", "type": "server_error"}})

    chat_stream = await chat(_SyntheticRequest(request, chat_body))
    if not isinstance(chat_stream, StreamingResponse):
        if isinstance(chat_stream, JSONResponse):
            response_json = _json_from_response(chat_stream)
            record = _chat_completion_to_response_record(body, response_id, response_json, messages)
            if record.get("store", True):
                save_response_record(record)
            return StreamingResponse(_single_response_stream(record), media_type="text/event-stream",
                                     headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})
        raise HTTPException(502, detail={"error": {"message": "unexpected non-stream response", "type": "server_error"}})

    async def _responses_stream():
        created = int(time.time())
        model_name = model
        reasoning_parts: list[str] = []
        text_parts: list[str] = []
        refusal_parts: list[str] = []
        tool_calls: dict[int, dict] = {}
        reasoning_item_id = "rs_0"
        message_item_id = "msg_0"
        refusal_item_id = "rf_0"
        output_indices: dict[str, int] = {}
        output_started: set[str] = set()
        content_started = False
        sequence_number = 0

        def _event_payload(payload: dict) -> dict:
            nonlocal sequence_number
            sequence_number += 1
            payload["sequence_number"] = sequence_number
            return payload

        def _start_output_item(item: dict) -> tuple[int, dict | None]:
            item_id = item.get("id") or f"out_{len(output_indices)}"
            if item_id not in output_indices:
                output_indices[item_id] = len(output_indices)
            output_index = output_indices[item_id]
            if item_id not in output_started:
                output_started.add(item_id)
                return_event = {
                    "type": "response.output_item.added",
                    "output_index": output_index,
                    "item": item,
                }
                return output_index, return_event
            return output_index, None

        def _ensure_reasoning_started() -> list[dict]:
            if not reasoning_parts:
                return []
            item = _response_reasoning_item("".join(reasoning_parts), reasoning_item_id)
            output_index, event = _start_output_item(item)
            events = []
            if event:
                events.append(event)
            return events

        def _ensure_refusal_started() -> list[dict]:
            if not refusal_parts:
                return []
            item = _response_refusal_item("".join(refusal_parts), refusal_item_id)
            output_index, event = _start_output_item(item)
            events = []
            if event:
                events.append(event)
            return events

        def _ensure_message_started() -> list[dict]:
            nonlocal content_started
            if not text_parts and not content_started:
                return []
            item = _response_text_item("".join(text_parts), message_item_id)
            output_index, event = _start_output_item(item)
            events = []
            if event:
                events.append(event)
            if not content_started:
                content_started = True
                events.append({
                    "type": "response.content_part.added",
                    "item_id": message_item_id,
                    "output_index": output_index,
                    "content_index": 0,
                    "part": item["content"][0],
                })
            return events

        created_record = _build_responses_record(
            response_id=response_id,
            body=body,
            model=model_name,
            created=created,
            completed_at=None,
            output=[],
            usage={},
            messages=messages,
            status="in_progress",
            incomplete_details=None,
        )

        yield _sse_json(_event_payload({
            "type": "response.created",
            "response": _response_object_payload(created_record, status="in_progress", completed_at=None, usage=None)
        }))
        yield _sse_json(_event_payload({
            "type": "response.in_progress",
            "response": _response_object_payload(created_record, status="in_progress", completed_at=None, usage=None)
        }))

        try:
            async for chunk in chat_stream.body_iterator:
                s = chunk.decode("utf-8", errors="ignore") if isinstance(chunk, bytes) else str(chunk)
                if not s.startswith("data: "):
                    continue
                raw = s[6:].strip()
                if raw == "[DONE]":
                    break
                try:
                    obj = json.loads(raw)
                except (json.JSONDecodeError, ValueError):
                    continue

                if "error" in obj:
                    err = obj.get("error", {})
                    failed_payload = _response_failed_payload(
                        response_id,
                        created,
                        model_name,
                        body,
                        previous_response_id,
                        err,
                        _normalize_structured_output_text("".join(text_parts), _response_text_config(body)),
                    )
                    yield _sse_json(_event_payload({
                        "type": "response.failed",
                        "response": _public_response_record(failed_payload),
                    }))
                    return

                model_name = obj.get("model", model_name)
                delta = (obj.get("choices") or [{}])[0].get("delta", {}) or {}
                finish_reason = (obj.get("choices") or [{}])[0].get("finish_reason")

                reasoning_delta = delta.get("reasoning_content")
                if isinstance(reasoning_delta, str) and reasoning_delta:
                    reasoning_parts.append(reasoning_delta)
                    for event in _ensure_reasoning_started():
                        yield _sse_json(_event_payload(event))
                    yield _sse_json(_event_payload({
                        "type": "response.reasoning_text.delta",
                        "item_id": reasoning_item_id,
                        "output_index": output_indices.get(reasoning_item_id, 0),
                        "content_index": 0,
                        "delta": reasoning_delta,
                    }))

                content_delta = delta.get("content")
                if isinstance(content_delta, str) and content_delta:
                    text_parts.append(content_delta)
                    for event in _ensure_message_started():
                        yield _sse_json(_event_payload(event))
                    yield _sse_json(_event_payload({
                        "type": "response.output_text.delta",
                        "item_id": message_item_id,
                        "output_index": output_indices.get(message_item_id, 0),
                        "content_index": 0,
                        "delta": content_delta,
                    }))

                refusal_delta = delta.get("refusal")
                if isinstance(refusal_delta, str) and refusal_delta:
                    refusal_parts.append(refusal_delta)
                    for event in _ensure_refusal_started():
                        yield _sse_json(_event_payload(event))
                    yield _sse_json(_event_payload({
                        "type": "response.refusal.delta",
                        "item_id": refusal_item_id,
                        "output_index": output_indices.get(refusal_item_id, 0),
                        "content_index": 0,
                        "delta": refusal_delta,
                    }))

                tc_list = delta.get("tool_calls") or []
                if isinstance(tc_list, list):
                    for tc in tc_list:
                        if not isinstance(tc, dict):
                            continue
                        idx = int(tc.get("index", 0) or 0)
                        slot = tool_calls.setdefault(idx, {
                            "id": tc.get("id") or f"call_{uuid.uuid4().hex[:24]}",
                            "name": "",
                            "arguments": "",
                        })
                        fn = tc.get("function", {}) or {}
                        if fn.get("name"):
                            slot["name"] = fn.get("name")
                        if fn.get("arguments"):
                            slot["arguments"] += fn.get("arguments")
                            function_item = {
                                "id": slot["id"],
                                "type": "function_call",
                                "call_id": slot["id"],
                                "name": slot["name"],
                                "arguments": slot["arguments"] or "{}",
                                "status": "in_progress",
                            }
                            output_index, event = _start_output_item(function_item)
                            if event:
                                yield _sse_json(_event_payload(event))
                            yield _sse_json(_event_payload({
                                "type": "response.function_call_arguments.delta",
                                "item_id": slot["id"],
                                "output_index": output_index,
                                "delta": fn.get("arguments"),
                            }))

                if finish_reason:
                    output_by_id: dict[str, dict] = {}
                    if reasoning_parts:
                        output_by_id[reasoning_item_id] = _response_reasoning_item("".join(reasoning_parts), reasoning_item_id)
                    if refusal_parts:
                        output_by_id[refusal_item_id] = _response_refusal_item("".join(refusal_parts), refusal_item_id)
                    if text_parts:
                        output_by_id[message_item_id] = _response_text_item(
                            _normalize_structured_output_text("".join(text_parts), _response_text_config(body)),
                            message_item_id,
                        )
                    for idx in sorted(tool_calls.keys()):
                        tc = tool_calls[idx]
                        output_by_id[tc["id"]] = {
                            "id": tc["id"],
                            "type": "function_call",
                            "call_id": tc["id"],
                            "name": tc["name"],
                            "arguments": tc["arguments"] or "{}",
                            "status": "completed",
                        }
                    if not output_by_id:
                        output_by_id[message_item_id] = _response_text_item("", message_item_id)
                        for event in _ensure_message_started():
                            yield _sse_json(_event_payload(event))
                    output = [
                        item for _, item in sorted(
                            output_by_id.items(),
                            key=lambda pair: output_indices.get(pair[0], len(output_indices))
                        )
                    ]

                    assistant_msg = {
                        "role": "assistant",
                        "content": _normalize_structured_output_text("".join(text_parts), _response_text_config(body)) if text_parts else None,
                    }
                    if reasoning_parts:
                        assistant_msg["reasoning_content"] = "".join(reasoning_parts)
                    if refusal_parts:
                        assistant_msg["refusal"] = "".join(refusal_parts)
                    if tool_calls:
                        assistant_msg["tool_calls"] = [{
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": tc["arguments"] or "{}",
                            }
                        } for _, tc in sorted(tool_calls.items())]

                    approx_completion_tokens = _count_tokens("".join(reasoning_parts) + "".join(text_parts))
                    approx_prompt_tokens = _count_tokens(convert_messages_for_deepseek(messages, tools))
                    status = _response_status_from_finish_reason(finish_reason)
                    incomplete_details = _response_incomplete_details(finish_reason)
                    completed_at = int(time.time()) if status == "completed" else None

                    record = _build_responses_record(
                        response_id=response_id,
                        body=body,
                        model=model_name,
                        created=created,
                        completed_at=completed_at,
                        output=output,
                        usage={
                            "prompt_tokens": approx_prompt_tokens,
                            "completion_tokens": approx_completion_tokens,
                            "total_tokens": approx_prompt_tokens + approx_completion_tokens,
                        },
                        messages=messages + [assistant_msg],
                        status=status,
                        incomplete_details=incomplete_details,
                    )
                    record = _apply_structured_output_contract(record)
                    status = record.get("status", status)
                    if record.get("store", True):
                        save_response_record(record)

                    if reasoning_parts:
                        yield _sse_json(_event_payload({
                            "type": "response.reasoning_text.done",
                            "item_id": reasoning_item_id,
                            "output_index": output_indices.get(reasoning_item_id, 0),
                            "content_index": 0,
                            "text": "".join(reasoning_parts),
                        }))
                    if refusal_parts:
                        yield _sse_json(_event_payload({
                            "type": "response.refusal.done",
                            "item_id": refusal_item_id,
                            "output_index": output_indices.get(refusal_item_id, 0),
                            "content_index": 0,
                            "text": "".join(refusal_parts),
                        }))
                    if text_parts:
                        normalized_text = _normalize_structured_output_text("".join(text_parts), _response_text_config(body))
                        yield _sse_json(_event_payload({
                            "type": "response.output_text.done",
                            "item_id": message_item_id,
                            "output_index": output_indices.get(message_item_id, 0),
                            "content_index": 0,
                            "text": normalized_text,
                        }))
                        yield _sse_json(_event_payload({
                            "type": "response.content_part.done",
                            "item_id": message_item_id,
                            "output_index": output_indices.get(message_item_id, 0),
                            "content_index": 0,
                            "part": _response_text_item(normalized_text, message_item_id)["content"][0],
                        }))
                    for idx in sorted(tool_calls.keys()):
                        tc = tool_calls[idx]
                        yield _sse_json(_event_payload({
                            "type": "response.function_call_arguments.done",
                            "item_id": tc["id"],
                            "output_index": output_indices.get(tc["id"], 0),
                            "arguments": tc["arguments"] or "{}",
                        }))
                    for idx, item in enumerate(output):
                        yield _sse_json(_event_payload({
                            "type": "response.output_item.done",
                            "output_index": idx,
                            "item": item,
                        }))
                    yield _sse_json(_event_payload({
                        "type": _response_terminal_event_type(status),
                        "response": _public_response_record(record),
                    }))
                    return

            output_by_id: dict[str, dict] = {}
            if reasoning_parts:
                output_by_id[reasoning_item_id] = _response_reasoning_item("".join(reasoning_parts), reasoning_item_id)
            if refusal_parts:
                output_by_id[refusal_item_id] = _response_refusal_item("".join(refusal_parts), refusal_item_id)
            normalized_text = _normalize_structured_output_text("".join(text_parts), _response_text_config(body)) if text_parts else ""
            output_by_id[message_item_id] = _response_text_item(normalized_text, message_item_id)
            for idx in sorted(tool_calls.keys()):
                tc = tool_calls[idx]
                output_by_id[tc["id"]] = {
                    "id": tc["id"],
                    "type": "function_call",
                    "call_id": tc["id"],
                    "name": tc["name"],
                    "arguments": tc["arguments"] or "{}",
                    "status": "completed",
                }
            output = [
                item for _, item in sorted(
                    output_by_id.items(),
                    key=lambda pair: output_indices.get(pair[0], len(output_indices))
                )
            ]
            if message_item_id not in output_indices:
                for event in _ensure_message_started():
                    yield _sse_json(_event_payload(event))

            assistant_msg = {"role": "assistant", "content": normalized_text if text_parts else None}
            if reasoning_parts:
                assistant_msg["reasoning_content"] = "".join(reasoning_parts)
            if refusal_parts:
                assistant_msg["refusal"] = "".join(refusal_parts)
            if tool_calls:
                assistant_msg["tool_calls"] = [{
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": tc["arguments"] or "{}",
                    }
                } for _, tc in sorted(tool_calls.items())]

            approx_completion_tokens = _count_tokens("".join(reasoning_parts) + "".join(text_parts))
            approx_prompt_tokens = _count_tokens(convert_messages_for_deepseek(messages, tools))
            record = _build_responses_record(
                response_id=response_id,
                body=body,
                model=model_name,
                created=created,
                completed_at=int(time.time()),
                output=output,
                usage={
                    "prompt_tokens": approx_prompt_tokens,
                    "completion_tokens": approx_completion_tokens,
                    "total_tokens": approx_prompt_tokens + approx_completion_tokens,
                },
                messages=messages + [assistant_msg],
                status="completed",
                incomplete_details=None,
            )
            record = _apply_structured_output_contract(record)
            status = record.get("status", "completed")
            if record.get("store", True):
                save_response_record(record)

            if reasoning_parts:
                yield _sse_json(_event_payload({
                    "type": "response.reasoning_text.done",
                    "item_id": reasoning_item_id,
                    "output_index": output_indices.get(reasoning_item_id, 0),
                    "content_index": 0,
                    "text": "".join(reasoning_parts),
                }))
            if refusal_parts:
                yield _sse_json(_event_payload({
                    "type": "response.refusal.done",
                    "item_id": refusal_item_id,
                    "output_index": output_indices.get(refusal_item_id, 0),
                    "content_index": 0,
                    "text": "".join(refusal_parts),
                }))
            if text_parts:
                yield _sse_json(_event_payload({
                    "type": "response.output_text.done",
                    "item_id": message_item_id,
                    "output_index": output_indices.get(message_item_id, 0),
                    "content_index": 0,
                    "text": normalized_text,
                }))
                yield _sse_json(_event_payload({
                    "type": "response.content_part.done",
                    "item_id": message_item_id,
                    "output_index": output_indices.get(message_item_id, 0),
                    "content_index": 0,
                    "part": _response_text_item(normalized_text, message_item_id)["content"][0],
                }))
            for idx in sorted(tool_calls.keys()):
                tc = tool_calls[idx]
                yield _sse_json(_event_payload({
                    "type": "response.function_call_arguments.done",
                    "item_id": tc["id"],
                    "output_index": output_indices.get(tc["id"], 0),
                    "arguments": tc["arguments"] or "{}",
                }))
            for idx, item in enumerate(output):
                yield _sse_json(_event_payload({
                    "type": "response.output_item.done",
                    "output_index": idx,
                    "item": item,
                }))
            yield _sse_json(_event_payload({
                "type": _response_terminal_event_type(status),
                "response": _public_response_record(record),
            }))
        except Exception as e:
            failed_payload = _response_failed_payload(
                response_id,
                created,
                model_name,
                body,
                previous_response_id,
                {"message": str(e), "type": "server_error"},
                _normalize_structured_output_text("".join(text_parts), _response_text_config(body)),
            )
            yield _sse_json(_event_payload({
                "type": "response.failed",
                "response": _public_response_record(failed_payload),
            }))

    return StreamingResponse(_responses_stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})


@app.post("/v1/responses/input_tokens")
async def count_response_input_tokens(request: Request):
    body = await request.json()
    input_value = body.get("input")
    if input_value is None and body.get("response_id"):
        record = get_response_record(str(body.get("response_id")))
        if not record:
            raise HTTPException(404, detail={"error": {"message": f"response {body.get('response_id')} not found", "type": "invalid_request_error"}})
        input_value = record.get("_input", [])
    token_count = _count_response_input_tokens(
        input_value,
        body.get("instructions") if isinstance(body.get("instructions"), str) else None,
        body.get("tools") if isinstance(body.get("tools"), list) else None,
    )
    return {"object": "response.input_tokens", "input_tokens": token_count}


def _compact_response_record(source: dict, body: dict) -> dict:
    response_id = _gen_response_id()
    now = int(time.time())
    source_text = source.get("output_text") or _extract_output_text(source.get("output", []))
    compact_text = body.get("summary") if isinstance(body.get("summary"), str) else source_text
    compact_item = {
        "type": "message",
        "role": "assistant",
        "content": [{"type": "output_text", "text": compact_text or ""}],
    }
    compact_messages = [{"role": "assistant", "content": compact_text or ""}]
    compact_body = {
        "_response_id": response_id,
        "input": [compact_item],
        "model": body.get("model") or source.get("model", "deepseek-default"),
        "previous_response_id": source.get("id"),
        "metadata": dict(source.get("metadata") or {}),
        "store": True,
    }
    compact_body["metadata"].update({
        "compacted": True,
        "source_response_id": source.get("id"),
    })
    record = _build_responses_record(
        response_id=response_id,
        body=compact_body,
        model=compact_body["model"],
        created=now,
        completed_at=now,
        output=[_response_text_item(compact_text or "")],
        usage={
            "prompt_tokens": _count_tokens(json.dumps(source.get("_input", []), ensure_ascii=False)),
            "completion_tokens": _count_tokens(compact_text or ""),
            "total_tokens": _count_tokens(json.dumps(source.get("_input", []), ensure_ascii=False)) + _count_tokens(compact_text or ""),
        },
        messages=compact_messages,
        status="completed",
        incomplete_details=None,
    )
    record["_lineage"] = {
        "type": "compaction",
        "source_response_id": source.get("id"),
        "source_created_at": source.get("created_at"),
    }
    record = _with_runtime(record, _runtime_metadata("compaction", "completed", source_response_id=source.get("id")))
    record["_events"] = _response_replay_events(record, persistable=True)
    return record


@app.post("/v1/responses/compact")
async def compact_response(request: Request):
    body = await request.json()
    response_id = body.get("response_id") or body.get("previous_response_id")
    if not response_id:
        raise HTTPException(400, detail={"error": {"message": "response_id is required", "type": "invalid_request_error"}})
    source = get_response_record(str(response_id))
    if not source:
        raise HTTPException(404, detail={"error": {"message": f"response {response_id} not found", "type": "invalid_request_error"}})
    record = _compact_response_record(source, body)
    save_response_record(record)
    return JSONResponse(_public_response_record(record))


@app.post("/v1/responses/{response_id}/compact")
async def compact_response_by_id(response_id: str, request: Request):
    body = await request.json()
    body["response_id"] = response_id
    source = get_response_record(response_id)
    if not source:
        raise HTTPException(404, detail={"error": {"message": f"response {response_id} not found", "type": "invalid_request_error"}})
    record = _compact_response_record(source, body)
    save_response_record(record)
    return JSONResponse(_public_response_record(record))


@app.post("/v1/responses/{response_id}/cancel")
async def cancel_response(response_id: str):
    record = get_response_record(response_id)
    if not record:
        raise HTTPException(404, detail={"error": {"message": f"response {response_id} not found", "type": "invalid_request_error"}})
    if record.get("status") == "cancelled":
        return JSONResponse(_public_response_record(record))

    if record.get("status") in _RESPONSE_TERMINAL_STATUSES:
        return JSONResponse(_public_response_record(record))

    cancelled = update_response_record(response_id, _response_cancelled_record)
    return JSONResponse(_public_response_record(cancelled or _response_cancelled_record(record)))


@app.get("/v1/responses/{response_id}")
async def get_response(response_id: str, request: Request):
    record = get_response_record(response_id)
    if not record:
        raise HTTPException(404, detail={"error": {"message": f"response {response_id} not found", "type": "invalid_request_error"}})
    if (request.query_params.get("stream") or "").lower() in ("1", "true", "yes"):
        starting_after = request.query_params.get("starting_after")
        return StreamingResponse(_response_replay_stream(record, starting_after), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})
    return JSONResponse(_public_response_record(record))


@app.get("/v1/responses/{response_id}/input_items")
async def get_response_input_items(response_id: str, request: Request):
    record = get_response_record(response_id)
    if not record:
        raise HTTPException(404, detail={"error": {"message": f"response {response_id} not found", "type": "invalid_request_error"}})
    stored_items = _ensure_list(record.get("_input"))
    limit_raw = request.query_params.get("limit")
    try:
        limit = max(1, min(int(limit_raw), 100)) if limit_raw is not None else 20
    except ValueError:
        raise HTTPException(400, detail={"error": {"message": "invalid limit", "type": "invalid_request_error"}})
    after = request.query_params.get("after")
    before = request.query_params.get("before")
    order = (request.query_params.get("order") or "desc").lower()
    if order not in ("asc", "desc"):
        raise HTTPException(400, detail={"error": {"message": "invalid order", "type": "invalid_request_error"}})
    page_items, has_more = _paginate_response_input_items(
        stored_items,
        limit=limit,
        after=after,
        before=before,
        order=order,
    )
    return {
        "object": "list",
        "data": page_items,
        "first_id": page_items[0].get("id") if page_items else None,
        "last_id": page_items[-1].get("id") if page_items else None,
        "has_more": has_more,
    }


@app.delete("/v1/responses/{response_id}")
async def delete_response(response_id: str):
    if not delete_response_record(response_id):
        raise HTTPException(404, detail={"error": {"message": f"response {response_id} not found", "type": "invalid_request_error"}})
    return {"id": response_id, "object": "response", "deleted": True}


def _do_chat(cfg, prompt, model, thinking_enabled, search_enabled, stream, is_retry=False, has_tools=False, tools=None, ref_file_ids=None):
    """核心聊天逻辑，支持 token 过期后重试
    
    DeepSeek SSE 流结构（thinking_enabled=True 时）：
    - data: {"v":{"response":{...}}} → 元数据，跳过
    - data: {"p":"response/thinking_content","v":"嗯"} → thinking 第一段（有p）
    - data: {"o":"APPEND","v":"，"} → thinking 后续段（无p，有o=APPEND）
    - data: {"v":"用户"} → thinking 更多后续（只有v）
    - data: {"p":"response/content","o":"APPEND","v":"你好"} → 正式内容第一段
    - data: {"v":"！"} → 正式内容后续
    - data: {"p":"response/status","v":"FINISHED"} → 状态，跳过
    - event: title → 对话标题，跳过
    - event: toast → 错误提示（如版本过低）
    """
    session_id = cfg["session_id"]
    req_headers = build_request_headers(cfg, session_id)
    pow_response = get_pow_response()
    if pow_response:
        req_headers["x-ds-pow-response"] = pow_response

    # model_type 字段：DeepSeek 根据此值路由到不同模型后端。
    # 映射：模型名含 "vision" → "vision"，含 "expert" → "expert"，其余 → "default"
    req_body = {
        "chat_session_id": session_id,
        "parent_message_id": None,
        "prompt": prompt,
        "ref_file_ids": ref_file_ids if ref_file_ids else [],
        "thinking_enabled": thinking_enabled,
        "search_enabled": search_enabled,
    }
    if "vision" in model:
        req_body["model_type"] = "vision"
        if ref_file_ids:
            _vlog(f"chat request files={ref_file_ids} thinking={thinking_enabled}")
    elif "expert" in model:
        req_body["model_type"] = "expert"
    else:
        req_body["model_type"] = "default"


    chat_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())

    def _parse_sse(resp):
        """Shared SSE parser — yields (type, value) tuples.
        type: "content" | "thinking" | "error" | "done"
        value: string content or error dict

        Handles two SSE formats:
        1. Old format: response/thinking_content + response/content
        2. New format: response/fragments/-1/content with fragment type tracking
           (fragments have type THINK or RESPONSE)
        """
        # Pre-flight: check Content-Type — if DeepSeek returns HTML/text instead of SSE,
        # treat the entire response as an error to avoid silent data loss
        ct = resp.headers.get("content-type", "")
        if ct and "text/event-stream" not in ct and "application/json" not in ct:
            body_sample = ""
            try:
                body_sample = resp.text[:300] if hasattr(resp, "text") else ""
            except Exception:
                pass
            yield ("error", {
                "message": f"DeepSeek returned non-SSE response (Content-Type: {ct}): {body_sample}",
                "code": "bad_content_type"
            })
            return

        # Track non-JSON lines for error detection
        non_json_line_count = 0
        phase = "thinking"
        # New format: track fragment type (THINK/RESPONSE) from metadata events
        fragment_type = None  # None = old format (use phase), "THINK"/"RESPONSE" = new format
        _line_buf = b""
        def _read_lines():
            nonlocal _line_buf
            for chunk in resp.iter_content(chunk_size=4096):
                if not chunk:
                    continue
                _line_buf += chunk
                while b"\n" in _line_buf:
                    raw_line, _line_buf = _line_buf.split(b"\n", 1)
                    yield raw_line.decode("utf-8", errors="ignore").strip()
            # flush remaining buffer
            if _line_buf.strip():
                yield _line_buf.decode("utf-8", errors="ignore").strip()

        for line in _read_lines():
            if not line:
                continue
            # Debug: log raw SSE lines for thinking models
            if thinking_enabled and line.startswith("data:") and "fragments" in line:
                _vlog(f"SSE_LINE: {line[:500]}")

            # Skip event: lines (title, update_session, etc.)
            if line.startswith("event:"):
                if line.startswith("event: hint"):
                    continue  # handled below via raw line processing
                continue

            # Detect raw text/HTML error responses
            if line.startswith("<!DOCTYPE") or line.startswith("<html") or line.startswith("<HTML"):
                yield ("error", {
                    "message": f"DeepSeek returned HTML error: {line[:200]}",
                    "code": "html_response"
                })
                return

            if non_json_line_count >= 3:
                yield ("error", {
                    "message": f"DeepSeek returned non-SSE text (too many non-JSON lines): first={line[:200]}",
                    "code": "non_sse_response"
                })
                return

            # DeepSeek non-SSE error JSON
            if line.startswith("{"):
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict) and "code" in obj and obj.get("code", 0) >= 40000:
                        yield ("error", {"message": obj.get("msg", "unknown"), "code": obj.get("code")})
                        return
                except json.JSONDecodeError:
                    pass
                continue

            ds = line[6:] if line.startswith("data: ") else line
            if ds.strip() == "[DONE]":
                yield ("done", "")
                return

            try:
                obj = json.loads(ds)
                if not isinstance(obj, dict):
                    continue

                # Error object: {"type": "error", "content": "...", "finish_reason": "..."}
                obj_type = obj.get("type", "")
                if obj_type == "error":
                    content = obj.get("content", "")
                    fr = obj.get("finish_reason", "")
                    yield ("error", {"message": content, "code": fr})
                    return

                val = obj.get("v")

                # Toast error (v is dict with type=error)
                if isinstance(val, dict):
                    # Check for error
                    t_type = val.get("type", "")
                    t_content = val.get("content", "")
                    fr = val.get("finish_reason", "")
                    if t_type == "error" and fr:
                        yield ("error", {"message": t_content, "code": fr})
                        return
                    # New format: metadata with response.fragments → extract fragment type & content
                    resp_data = val.get("response", {})
                    if isinstance(resp_data, dict):
                        frags = resp_data.get("fragments", [])
                        if frags and isinstance(frags, list):
                            for frag in frags:
                                if isinstance(frag, dict):
                                    ftype = frag.get("type", "")
                                    if ftype:
                                        fragment_type = ftype
                                        if thinking_enabled:
                                            _vlog(f"SSE: fragment_type={fragment_type}")
                                    fcontent = frag.get("content", "")
                                    if fcontent and isinstance(fcontent, str):
                                        if fragment_type == "THINK":
                                            yield ("thinking", fcontent)
                                        else:
                                            yield ("content", fcontent)
                    continue

                path = obj.get("p", "")

                # ── New format: response/fragments ──────────────────
                # Fragment append event: {"p":"response/fragments","o":"APPEND","v":[{"id":N,"type":"RESPONSE","content":"...",...}]}
                if path == "response/fragments" and obj.get("o") == "APPEND" and isinstance(val, list):
                    if val:
                        last_frag = val[-1] if isinstance(val[-1], dict) else {}
                        new_type = last_frag.get("type", "")
                        if new_type:
                            fragment_type = new_type
                            if thinking_enabled:
                                _vlog(f"SSE: new fragment type={new_type}")
                        # Extract initial content from fragment object
                        frag_content = last_frag.get("content", "")
                        if frag_content and isinstance(frag_content, str):
                            if fragment_type == "THINK":
                                yield ("thinking", frag_content)
                            else:
                                yield ("content", frag_content)
                    continue

                # Fragment content: {"p":"response/fragments/-1/content","o":"APPEND","v":"..."}
                # or without "o": {"p":"response/fragments/-1/content","v":"..."}
                if path == "response/fragments/-1/content":
                    if fragment_type == "THINK":
                        phase = "thinking"
                        if isinstance(val, str) and val:
                            yield ("thinking", val)
                    else:  # RESPONSE or unknown
                        phase = "content"
                        if isinstance(val, str) and val:
                            yield ("content", val)
                    continue

                # ── Old format: response/content + response/thinking_content ──
                if path == "response/content":
                    o_val = obj.get("o")
                    if o_val is None or o_val == "APPEND":
                        phase = "content"
                        if isinstance(val, str) and val:
                            yield ("content", val)
                elif path == "response/thinking_content" and thinking_enabled:
                    o_val = obj.get("o")
                    if o_val is None or o_val == "APPEND":
                        phase = "thinking"
                        if isinstance(val, str) and val:
                            yield ("thinking", val)
                elif path:
                    continue  # other metadata (status, elapsed_secs, BATCH, etc.)
                elif isinstance(val, str) and val:
                    # Pathless continuation lines: use fragment_type if new format, else phase
                    if fragment_type is not None:
                        # New format: use fragment type
                        if fragment_type == "THINK":
                            yield ("thinking", val)
                        else:
                            yield ("content", val)
                    else:
                        # Old format: use phase
                        if phase == "thinking" and thinking_enabled:
                            yield ("thinking", val)
                        else:
                            yield ("content", val)
            except json.JSONDecodeError:
                non_json_line_count += 1
                continue

    def do_stream():
        """SSE streaming for OpenAI-compatible clients."""
        try:
            resp = cffi_requests.post(
                "https://chat.deepseek.com/api/v0/chat/completion",
                headers=req_headers,
                json=req_body,
                impersonate="chrome120",
                stream=True,
                timeout=120,
            )

            if ref_file_ids or thinking_enabled:
                _vlog(f"chat stream response: status={resp.status_code} ct={resp.headers.get('content-type','?')} model={model} thinking={thinking_enabled}")

            if resp.status_code == 401 and not is_retry:
                print("[Token] 401, trying refresh...")
                new_cfg = relogin(cfg)
                if new_cfg:
                    for chunk in _do_chat_stream_only(new_cfg, prompt, model, thinking_enabled, search_enabled, has_tools, tools, ref_file_ids):
                        yield chunk
                    return
                yield f'data: {json.dumps({"error": {"message": "Token expired", "type": "auth_error", "code": 401}})}\n\n'
                yield "data: [DONE]\n\n"
                return

            if resp.status_code != 200:
                error_msg = f"DeepSeek returned {resp.status_code}: {resp.text[:300]}"
                print(f"[Error] {error_msg}")
                yield f'data: {json.dumps({"error": {"message": error_msg, "type": "server_error", "code": resp.status_code}})}\n\n'
                yield "data: [DONE]\n\n"
                return

            if has_tools:
                # 输出 tool_calls SSE 事件的辅助函数
                def _emit_tool_calls(tc_result, _cid, _created, _model):
                    if tc_result:
                        for i, tc in enumerate(tc_result):
                            delta = {"role": "assistant", "content": None,
                                     "tool_calls": [{"index": i, "id": tc["id"], "type": "function",
                                                     "function": {"name": tc["function"]["name"], "arguments": ""}}]}
                            r = {"id": _cid, "object": "chat.completion.chunk", "created": _created, "model": _model,
                                 "choices": [{"index": 0, "delta": delta, "finish_reason": None}]}
                            yield f'data: {json.dumps(r, ensure_ascii=False)}\n\n'
                            args = tc["function"]["arguments"]
                            r = {"id": _cid, "object": "chat.completion.chunk", "created": _created, "model": _model,
                                 "choices": [{"index": 0, "delta": {"tool_calls": [{"index": i, "function": {"arguments": args}}]}, "finish_reason": None}]}
                            yield f'data: {json.dumps(r, ensure_ascii=False)}\n\n'
                        r = {"id": _cid, "object": "chat.completion.chunk", "created": _created, "model": _model,
                             "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}
                        yield f'data: {json.dumps(r, ensure_ascii=False)}\n\n'
                    else:
                        r = {"id": _cid, "object": "chat.completion.chunk", "created": _created, "model": _model,
                             "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                        yield f'data: {json.dumps(r, ensure_ascii=False)}\n\n'

                # 流式筛分 + 并行缓冲：筛分实时播正文，同时攒完整内容做 fallback
                def _parse_fn(text):
                    return extract_tool_call(text, get_tool_names(tools) if tools else [])

                sieve = StreamSieve(parse_fn=_parse_fn)
                _role_sent = False
                _full_buf = ""  # 并行缓冲完整内容，flush 时 fallback 解析

                for etype, val in _parse_sse(resp):
                    if etype == "content":
                        _full_buf += val
                        for evt in sieve.feed(val):
                            if evt.type == "text":
                                if isinstance(evt.data, str) and evt.data:
                                    if not _role_sent:
                                        r = {"id": chat_id, "object": "chat.completion.chunk", "created": created, "model": model,
                                             "choices": [{"index": 0, "delta": {"role": "assistant", "content": None}, "finish_reason": None}]}
                                        yield f'data: {json.dumps(r, ensure_ascii=False)}\n\n'
                                        _role_sent = True
                                    r = {"id": chat_id, "object": "chat.completion.chunk", "created": created, "model": model,
                                         "choices": [{"index": 0, "delta": {"content": evt.data}, "finish_reason": None}]}
                                    yield f'data: {json.dumps(r, ensure_ascii=False)}\n\n'
                    elif etype == "thinking":
                        r = {"id": chat_id, "object": "chat.completion.chunk", "created": created, "model": model,
                             "choices": [{"index": 0, "delta": {"reasoning_content": val}, "finish_reason": None}]}
                        yield f'data: {json.dumps(r, ensure_ascii=False)}\n\n'
                    elif etype == "error":
                        yield f'data: {json.dumps({"error": {"message": val["message"], "type": "server_error", "code": val.get("code")}})}\n\n'
                        yield "data: [DONE]\n\n"
                        return
                    elif etype == "done":
                        break

                # Flush + fallback：筛分没抓到就用全量解析
                _had_tool_calls = False
                for evt in sieve.flush():
                    if evt.type == "text":
                        if isinstance(evt.data, str) and evt.data:
                            if not _role_sent:
                                r = {"id": chat_id, "object": "chat.completion.chunk", "created": created, "model": model,
                                     "choices": [{"index": 0, "delta": {"role": "assistant", "content": None}, "finish_reason": None}]}
                                yield f'data: {json.dumps(r, ensure_ascii=False)}\n\n'
                                _role_sent = True
                            r = {"id": chat_id, "object": "chat.completion.chunk", "created": created, "model": model,
                                 "choices": [{"index": 0, "delta": {"content": evt.data}, "finish_reason": None}]}
                            yield f'data: {json.dumps(r, ensure_ascii=False)}\n\n'
                    elif evt.type == "tool_calls":
                        _had_tool_calls = True
                        for chunk in _emit_tool_calls(evt.data, chat_id, created, model):
                            yield chunk

                # Fallback: 筛分没抓到，用全量缓冲重试
                if not _had_tool_calls and _full_buf:
                    tc_result, _ = extract_tool_call(_full_buf, get_tool_names(tools) if tools else [])
                    if tc_result:
                        if not _role_sent:
                            r = {"id": chat_id, "object": "chat.completion.chunk", "created": created, "model": model,
                                 "choices": [{"index": 0, "delta": {"role": "assistant", "content": None}, "finish_reason": None}]}
                            yield f'data: {json.dumps(r, ensure_ascii=False)}\n\n'
                            _role_sent = True
                        _had_tool_calls = True
                        for chunk in _emit_tool_calls(tc_result, chat_id, created, model):
                            yield chunk

                if not _role_sent:
                    r = {"id": chat_id, "object": "chat.completion.chunk", "created": created, "model": model,
                         "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                    yield f'data: {json.dumps(r, ensure_ascii=False)}\n\n'
                elif not _had_tool_calls:
                    r = {"id": chat_id, "object": "chat.completion.chunk", "created": created, "model": model,
                         "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                    yield f'data: {json.dumps(r, ensure_ascii=False)}\n\n'
                yield "data: [DONE]\n\n"
                return

            # No tools: normal streaming
            _stream_think_count = 0
            _stream_content_count = 0
            # Send role delta first — many clients need this to start rendering
            r = {"id": chat_id, "object": "chat.completion.chunk", "created": created, "model": model,
                 "choices": [{"index": 0, "delta": {"role": "assistant", "content": None}, "finish_reason": None}]}
            yield f'data: {json.dumps(r, ensure_ascii=False)}\n\n'
            for etype, val in _parse_sse(resp):
                if etype == "content":
                    _stream_content_count += 1
                    r = {"id": chat_id, "object": "chat.completion.chunk", "created": created, "model": model,
                         "choices": [{"index": 0, "delta": {"content": val}, "finish_reason": None}]}
                    yield f'data: {json.dumps(r, ensure_ascii=False)}\n\n'
                elif etype == "thinking":
                    _stream_think_count += 1
                    r = {"id": chat_id, "object": "chat.completion.chunk", "created": created, "model": model,
                         "choices": [{"index": 0, "delta": {"reasoning_content": val}, "finish_reason": None}]}
                    yield f'data: {json.dumps(r, ensure_ascii=False)}\n\n'
                elif etype == "error":
                    yield f'data: {json.dumps({"error": {"message": val["message"], "type": "server_error", "code": val.get("code")}})}\n\n'
                    yield "data: [DONE]\n\n"
                    return
                elif etype == "done":
                    if thinking_enabled:
                        _vlog(f"STREAM_DONE: thinking_chunks={_stream_think_count} content_chunks={_stream_content_count}")
                    yield "data: [DONE]\n\n"
                    return

        except Exception as e:
            print(f"[Error] do_stream failed: {e}")
            yield f'data: {json.dumps({"error": {"message": str(e), "type": "server_error"}})}\n\n'
            yield "data: [DONE]\n\n"

    def do_nonstream():
        """Non-streaming: use stream=True internally (curl_cffi stream=False
        returns incomplete SSE), buffer all events, return complete JSON response."""
        full_content = ""
        full_thinking = ""

        try:
            resp = cffi_requests.post(
                "https://chat.deepseek.com/api/v0/chat/completion",
                headers=req_headers,
                json=req_body,
                impersonate="chrome120",
                stream=True,  # Always stream — curl_cffi stream=False truncates SSE
                timeout=120,
            )

            if ref_file_ids or thinking_enabled:
                _vlog(f"chat nonstream(stream-internal) response: status={resp.status_code} ct={resp.headers.get('content-type','?')}")

            if resp.status_code == 401 and not is_retry:
                print("[Token] 401 in nonstream, trying refresh...")
                new_cfg = relogin(cfg)
                if new_cfg:
                    return _do_chat(new_cfg, prompt, model, thinking_enabled, search_enabled, False, is_retry=True, has_tools=has_tools, tools=tools, ref_file_ids=ref_file_ids)

            if resp.status_code != 200:
                body_sample = ""
                try:
                    body_sample = resp.text[:500] if hasattr(resp, "text") else f"(no body, status={resp.status_code})"
                except Exception:
                    body_sample = f"(body unreadable, status={resp.status_code})"
                print(f"[nonstream] DeepSeek {resp.status_code}: {body_sample[:200]}")
                raise HTTPException(502, detail={
                    "error": {
                        "message": f"DeepSeek returned {resp.status_code}: {body_sample[:200]}",
                        "type": "server_error",
                        "code": resp.status_code
                    }
                })

            # Buffer all events from stream using _parse_sse
            for etype, val in _parse_sse(resp):
                if etype == "content":
                    full_content += val
                elif etype == "thinking":
                    full_thinking += val
                elif etype == "error":
                    raise HTTPException(502, detail={"error": {
                        "message": val["message"],
                        "type": "server_error",
                        "code": val.get("code", "")
                    }})
                elif etype == "done":
                    break

        except HTTPException:
            raise
        except Exception as e:
            print(f"[nonstream] Error: {e}")
            raise HTTPException(502, detail={"error": {"message": str(e), "type": "server_error"}})

        # Debug: log extracted thinking/content for thinking models
        if thinking_enabled:
            _vlog(f"NONSTREAM_RESULT: thinking={len(full_thinking)} chars, content={len(full_content)} chars")
            _vlog(f"NONSTREAM_THINKING[:500]: {full_thinking[:500]}")
            _vlog(f"NONSTREAM_CONTENT[:500]: {full_content[:500]}")

        # 如果有 tools，检查 content 中是否包含 tool_call 标签
        finish_reason = "stop"
        tc_result = None
        final_content = full_content
        if has_tools:
            tc_result, final_content = extract_tool_call(full_content, get_tool_names(tools) if tools else None)
            if tc_result:
                finish_reason = "tool_calls"

        msg = {"role": "assistant", "content": final_content}
        if full_thinking:
            msg["reasoning_content"] = full_thinking
        if tc_result:
            msg["tool_calls"] = tc_result
            if final_content is None:
                msg["content"] = None

        # Build and validate response — pre-serialize to catch any issues early
        response_body = {
            "id": chat_id, "object": "chat.completion", "created": created, "model": model,
            "choices": [{"index": 0, "message": msg, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        try:
            # Validate JSON serializability
            json.dumps(response_body, ensure_ascii=False)
        except (TypeError, ValueError) as serr:
            print(f"[nonstream] JSON serialization failed: {serr}")
            # Sanitize: replace non-serializable values with their string repr
            safe_msg = {}
            for k, v in msg.items():
                try:
                    json.dumps({k: v}, ensure_ascii=False)
                    safe_msg[k] = v
                except (TypeError, ValueError):
                    safe_msg[k] = str(v)
            response_body["choices"][0]["message"] = safe_msg
        return JSONResponse(response_body)

    if stream:
        return StreamingResponse(do_stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})
    return do_nonstream()


def _do_chat_stream_only(cfg, prompt, model, thinking_enabled, search_enabled, has_tools=False, tools=None, ref_file_ids=None):
    """Token 刷新重试专用的流式生成器"""
    result = _do_chat(cfg, prompt, model, thinking_enabled, search_enabled, stream=True, is_retry=True, has_tools=has_tools, tools=tools, ref_file_ids=ref_file_ids)
    if isinstance(result, StreamingResponse):
        yield from result.body_iterator
    else:
        yield f"data: {json.dumps({'error': {'message': 'Retry returned non-stream', 'type': 'server_error'}})}\n\n"
        yield "data: [DONE]\n\n"


# ── 启动 ─────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    print(f"DeepSeek Proxy\n Admin: http://localhost:{PROXY_PORT}/admin\n API: http://localhost:{PROXY_PORT}/v1")
    uvicorn.run(app, host="0.0.0.0", port=PROXY_PORT, log_level="info")
