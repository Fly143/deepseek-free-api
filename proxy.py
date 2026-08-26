"""
DeepSeek 网页 → API 代理（纯 HTTP 转发，无浏览器依赖）
用法: python proxy.py → 打开 http://localhost:8000/admin → 粘贴 cURL → 保存 → 用
"""
import asyncio, json, os, shlex, time, uuid, webbrowser, base64, re, secrets
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.security import HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
import tiktoken
from curl_cffi import requests as cffi_requests
from app.batch import init_batch_storage as anthropic_init_batch_storage

# ── Tokenizer ───────────────────────────────────
_enc = tiktoken.get_encoding("cl100k_base")

def _count_tokens(text: str) -> int:
    return len(_enc.encode(text or ""))

# ── 用量统计 ───────────────────────────────────
from usage_store import add_usage, get_usage, clear_usage
from session_store import needs_renewal, on_new_session, add_tokens, get_usage_status, get_expired_sessions, remove_old_session
from response_store import save_response_record, get_response_record, delete_response_record, update_response_record

# ── 工具调用处理模块 ─────────────────────────────────
from tool_call import (
    extract_tool_call,
    get_tool_names,
    convert_messages_for_deepseek,
    clean_tool_text,
)

# ── 流式筛分 + DSML 解析 ────────────────────────────
from tool_sieve import StreamSieve, SieveEvent
from tool_dsml import parse_dsml_tool_calls as _parse_dsml, sanitize_leaked_output

# ── PoW (Proof of Work) Solver — 纯 Python 实现（无 WASM 依赖）────────
from pow_native import DeepSeekPOW

# Initialize PoW solver
pow_solver = DeepSeekPOW()

BASE_DIR = Path(__file__).parent
CONFIG_FILE = BASE_DIR / "config.json"

# 多账号管理
from app.config import config_manager, DsAccount
from app.auth import verify_admin
from app.registrar import (BatchRegistrar, registrar as _registrar,
                           send_verification_code, expand_email_patterns,
                           list_mailcx_domains, generate_local_part,
                           random_ios_headers, new_device_id)
VISION_LOG = BASE_DIR / "vision.log"
_DEBUG = os.getenv("DS_DEBUG", "").lower() in ("1", "true", "yes")

# ── DeepSeek API 通用 Headers ─────────────────────
DS_HEADERS = {
    "content-type": "application/json",
    "origin": "https://chat.deepseek.com",
    "referer": "https://chat.deepseek.com/",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/134.0.0.0 Safari/537.36",
    "x-client-version": "2.4.2",
    "x-client-platform": "web",
}

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
        "status": "completed",
        "summary": [{
            "type": "summary_text",
            "text": summary_text or "",
        }],
    }


def _response_function_call_item(tool_call: dict, call_id: str | None = None) -> dict:
    fn = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
    return {
        "id": f"fc_{uuid.uuid4().hex[:24]}",
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
        "reasoning": {"effort": body.get("reasoning", {}).get("effort")} if isinstance(body.get("reasoning"), dict) and body.get("reasoning", {}).get("effort") else None,
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
        "reasoning": {"effort": body.get("reasoning", {}).get("effort")} if isinstance(body.get("reasoning"), dict) and body.get("reasoning", {}).get("effort") else None,
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
    "reasoning": None,
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
    if payload.get("reasoning") is not None and not isinstance(payload.get("reasoning"), dict):
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
        # 安全防护：剥除 content 中残留的 <think> 标签（SSE 解析可能遗漏）
        content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        if content:
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
    # When cURL uses \\\n (backslash-newline) continuation (e.g. Chrome
    # DevTools "Copy as cURL" on macOS), shlex.split preserves '\n' as
    # the first character of the following token, producing "\n-H" or
    # "\n--data-raw".  Strip only leading '\n' — not all whitespace —
    # so intentional whitespace in values is preserved.
    tokens = [t.lstrip('\n') for t in tokens]
    tokens = [t for t in tokens if t]
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── 修复 /v1/v1/ 双前缀（Claude Code CLI 等客户端 base URL 带 /v1）──
@app.middleware("http")
async def _rewrite_double_v1(request: Request, call_next):
    path = request.scope.get("path", "")
    if path.startswith("/v1/v1/"):
        request.scope["path"] = path[3:]
    elif path == "/v1/v1":
        request.scope["path"] = "/v1"
    return await call_next(request)

from app.anthropic_routes import router as _anthropic_router
app.include_router(_anthropic_router)


@app.on_event("startup")
async def startup_discover():
    """启动时自动刷新模型列表，延迟清理过期会话（后台线程，避免风控）。"""
    print("[启动] 探测模型列表...")
    _discover_models()
    print("[启动] 后台清理过期会话...")
    import threading
    threading.Thread(target=cleanup_old_sessions, daemon=True).start()

# ── 管理页面 ─────────────────────────────────────────────
ADMIN = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>DeepSeek Proxy</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{font-family:system-ui,sans-serif;background:#0f172a;color:#e2e8f0;min-height:100vh;display:flex;justify-content:center;align-items:flex-start;padding-top:40px}
.c{background:#1e293b;border-radius:16px;padding:32px;width:600px;max-width:95vw;border:1px solid #334155;position:relative}
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
/* Toggle switch */
.switch input:checked + span{background:#2563eb!important}
.switch input:checked + span + span{transform:translateX(18px)!important;background:#fff!important}
/* Account management */
.acct-tbl{width:100%;border-collapse:collapse;font-size:13px;margin-top:12px}
.acct-tbl th,.acct-tbl td{padding:8px 10px;text-align:left;border-bottom:1px solid #334155}
.acct-tbl th{color:#94a3b8;font-weight:500;font-size:11px;white-space:nowrap}
.acct-tbl td{font-variant-numeric:tabular-nums}
.acct-tbl td:nth-child(3){max-width:140px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
@media(max-width:640px){.acct-tbl td:nth-child(3){max-width:80px}.acct-tbl td:first-child{white-space:nowrap;max-width:100px;overflow:hidden;text-overflow:ellipsis}.acct-tbl td:nth-child(2){white-space:nowrap}.acct-tbl td{padding:10px 6px}.acct-btn{margin:3px 0}}
.acct-tbl td:last-child{white-space:nowrap}
.acct-st{width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:6px;vertical-align:middle}
.acct-st.ok{background:#22c55e}.acct-st.no{background:#64748b}.acct-st.er{background:#ef4444}
.acct-btn{padding:4px 10px;border-radius:4px;border:none;cursor:pointer;font-size:12px;font-weight:500}
.acct-btn.rm{background:#7f1d1d;color:#fca5a5}.acct-btn.rm:hover{background:#991b1b}
.acct-btn.rl{background:#1e3a5f;color:#7dd3fc}.acct-btn.rl:hover{background:#1e40af}
.acct-btn.batch{background:#2563eb;color:#fff;width:100%;margin-top:12px;padding:10px;border-radius:8px;border:none;cursor:pointer;font-size:13px;font-weight:500}
.acct-btn.batch:hover{background:#1d4ed8}
.acct-add{display:flex;gap:8px;margin-bottom:12px;flex-wrap:wrap}
.acct-add input{flex:1;min-width:100px;padding:8px 10px;background:#0f172a;border:1px solid #334155;border-radius:6px;color:#e2e8f0;font-size:13px}
.acct-add select{padding:8px 10px;background:#0f172a;border:1px solid #334155;border-radius:6px;color:#e2e8f0;font-size:13px}
.acct-add button{padding:8px 16px;background:#2563eb;color:#fff;border:none;border-radius:6px;cursor:pointer;font-size:13px;font-weight:500}
.acct-add button:hover{background:#1d4ed8}
.acct-empty{text-align:center;color:#64748b;padding:30px 0;font-size:13px}
.acct-stat{font-size:12px;color:#94a3b8;margin-bottom:8px}
</style>
</head>
<body>
<div class="c" id="mainContent">
<h1>DeepSeek Proxy</h1>
<div style="position:absolute;top:32px;right:32px">
<button onclick="toggleLang()" id="langBtn" style="padding:6px 14px;background:#1e293b;color:#94a3b8;border:1px solid #334155;border-radius:6px;cursor:pointer;font-size:13px;transition:all .2s">🌐 EN</button>
</div>
<div id="s" class="s no"><span id="sd" class="d dy"></span><span id="st" data-i18n="waitingCfg">等待配置</span></div>

<div class="tab-bar">
<div class="tab active" onclick="switchTab('phone')">手机号登录</div>
<div class="tab" onclick="switchTab('email')">邮箱登录</div>
<div class="tab" onclick="switchTab('usage')">用量统计</div>
<div class="tab" onclick="switchTab('accounts')">账号管理</div>
<div class="tab" onclick="switchTab('settings')" data-i18n="settings">设置</div>
<div class="tab" onclick="switchTab('register')">批量注册</div>
</div>

<div id="phonePanel" class="panel active">
<div class="row">
<input class="ac" type="tel" id="area_code" value="+86" placeholder="+86">
<input class="ph" type="tel" id="mobile" data-i18n-ph="phonePlaceholder" placeholder="手机号" autocomplete="tel">
</div>
<div class="pw-row"><input type="password" id="pw1" data-i18n-ph="pwdPlaceholder" placeholder="密码" autocomplete="current-password"></div>
<button class="btn bp" id="btn1" onclick="doLogin('phone')" data-i18n="loginBtn">登录</button>
</div>

<div id="emailPanel" class="panel">
<div class="pw-row"><input type="email" id="email" data-i18n-ph="emailPlaceholder" placeholder="邮箱地址" autocomplete="email"></div>
<div class="pw-row"><input type="password" id="pw2" data-i18n-ph="pwdPlaceholder" placeholder="密码" autocomplete="current-password"></div>
<button class="btn bp" id="btn2" onclick="doLogin('email')" data-i18n="loginBtn">登录</button>
</div>

<div class="info" id="info"></div>

<div id="apiSection">
<div class="sl" style="font-weight:600;color:#e2e8f0;margin-bottom:10px" data-i18n="curlTitle">📋 cURL 导入</div>
<div class="curl-help" style="font-size:12px;color:#94a3b8;margin-bottom:12px;line-height:1.7">
  <div style="margin-bottom:6px;color:#7dd3fc;font-weight:500" data-i18n="curlSteps">导入步骤：</div>
  <div data-i18n="curlStep1">1. 打开 chat.deepseek.com 并登录</div>
  <div data-i18n="curlStep2">2. 按 F12 → Network 面板</div>
  <div data-i18n="curlStep3">3. 发送任意消息，找到 completion 请求</div>
  <div data-i18n="curlStep4">4. 右键 → Copy as cURL，粘贴到下方</div>
</div>
<textarea id="curl" data-i18n-ph="pasteCurl" placeholder="粘贴 cURL ..." style="width:100%;height:120px;background:#0f172a;border:1px solid #334155;border-radius:8px;color:#e2e8f0;padding:12px;font-family:monospace;font-size:11px;resize:vertical"></textarea>
<button class="btn bp" id="btn3" onclick="saveCurl()" data-i18n="saveCurlBtn" style="margin-top:8px">保存 cURL</button>

<hr>
<div class="step">
<div class="sl" style="font-weight:600;color:#e2e8f0;" data-i18n="apiConfig">API 配置</div>
<div class="cfg">
<div class="cr"><span data-i18n="apiAddr">API 地址</span><code onclick="cp(this)">http://localhost:""" + str(PROXY_PORT) + """/v1</code></div>
<div class="cr"><span data-i18n="apiKey">API Key</span><code onclick="cp(this)" data-i18n="apiKeyVal">任意填写</code></div>

</div>
</div>
<div class="step" style="margin-top:16px">
<button class="btn" style="background:#334155;color:#e2e8f0;width:100%;font-size:13px" onclick="refreshModels()" id="refreshBtn" data-i18n="refreshModels">🔄 刷新模型列表</button>
<div id="modelsInfo" style="margin-top:8px;font-size:12px;color:#64748b;display:none"></div>
</div>
</div>

<div id="usagePanel" class="panel">
<div id="usageContent"></div>
<div style="margin-top:14px">
<button class="pb ac" onclick="switchPeriod('total')" id="pbTotal" data-i18n="periodAll">全部</button>
<button class="pb" onclick="switchPeriod('week')" id="pbWeek" data-i18n="periodWeek">本周</button>
<button class="pb" onclick="switchPeriod('today')" id="pbToday" data-i18n="periodToday">今日</button>
<button class="btn" style="background:#334155;color:#e2e8f0;font-size:12px;padding:6px 12px;margin-left:8px" onclick="loadUsage()" data-i18n="refreshBtn">刷新</button>
<button class="btn" style="background:#7f1d1d;color:#fca5a5;font-size:12px;padding:6px 12px;margin-left:4px" onclick="clearUsage()" data-i18n="clearBtn">清空</button>
</div>
</div>

<div id="accountsPanel" class="panel">
<div class="acct-stat" id="acctStat" data-i18n="loadingAccounts">加载中...</div>

<div id="acctList"><div class="acct-empty">暂无账号，请先添加</div></div>
<button class="acct-btn batch" onclick="reloginAll()" data-i18n="reloginAllBtn">全部重新登录</button>
<button class="acct-btn batch" onclick="cleanupSessions()" data-i18n="cleanupSessionsBtn" style="background:#7c3aed;color:#fff">清理过期会话</button>
</div>

<div id="settingsPanel" class="panel">
<div class="sl" style="font-weight:600;color:#e2e8f0;" data-i18n="proxyTitle">代理配置</div>
<div class="cr" style="margin-top:12px">
  <span style="color:#94a3b8;font-size:13px" data-i18n="proxyHint">绕过 AWS WAF 拦截。格式：http://127.0.0.1:7890 或 socks5://127.0.0.1:7891</span>
</div>
<div class="pw-row" style="margin-top:12px">
  <input type="text" id="proxyUrl" placeholder="http://127.0.0.1:7890" style="width:100%;background:#0f172a;border:1px solid #334155;border-radius:8px;color:#e2e8f0;padding:12px;font-size:14px">
</div>
<button class="btn bp" onclick="saveProxy()" data-i18n="proxySaveBtn" style="margin-top:8px">保存</button>
<div id="proxyStatus" style="margin-top:8px;font-size:12px;color:#64748b"></div>
<hr>
<div class="sl" style="font-weight:600;color:#e2e8f0;margin-bottom:8px" data-i18n="passthroughTitle">工具透传模式</div>
<div class="cr">
  <span style="color:#94a3b8;font-size:13px" data-i18n="passthroughHint">跳过 DSML 格式说明书，直接嵌入原始工具定义（适合 Roo Code / Cline）</span>
</div>
<div style="display:flex;align-items:center;gap:12px;margin-top:8px">
  <label style="font-size:13px;color:#e2e8f0" id="passthroughLabel" data-i18n="passthroughToggle">关闭</label>
  <label class="switch" style="position:relative;display:inline-block;width:40px;height:22px">
    <input type="checkbox" id="passthroughToggle" onchange="updatePassthroughLabel()" style="opacity:0;width:0;height:0">
    <span style="position:absolute;cursor:pointer;top:0;left:0;right:0;bottom:0;background:#334155;border-radius:22px;transition:.3s"></span>
    <span style="position:absolute;height:16px;width:16px;left:3px;bottom:3px;background:#94a3b8;border-radius:50%;transition:.3s" class="slider-knob"></span>
  </label>
  <button class="btn bp" onclick="savePassthrough()" data-i18n="passthroughSaveBtn" style="font-size:12px;padding:6px 14px;width:auto">保存</button>
</div>
<div id="passthroughStatus" style="margin-top:4px;font-size:12px;color:#64748b"></div>
<hr>
<div class="sl" style="font-weight:600;color:#e2e8f0;margin-bottom:8px" data-i18n="changePwdTitle">修改管理员密码</div>
<div class="pw-row" style="margin-top:12px">
  <input type="password" id="newPwd" data-i18n-ph="newPwdPlaceholder" placeholder="输入新密码" style="width:100%;background:#0f172a;border:1px solid #334155;border-radius:8px;color:#e2e8f0;padding:12px;font-size:14px">
</div>
<button class="btn bp" onclick="changePwd()" data-i18n="changePwdBtn" style="margin-top:8px">保存新密码</button>
<div id="pwdStatus" style="margin-top:8px;font-size:12px;color:#64748b"></div>
</div>

<div id="registerPanel" class="panel">
<div class="sl" style="font-weight:600;color:#e2e8f0;margin-bottom:10px">批量注册账号（注册后自动登录并入池）</div>
<div style="font-size:12px;color:#94a3b8;line-height:1.7;margin-bottom:10px">
  流程：发送验证码 → mail.cx 临时邮箱收码 → 注册 → 自动登录并入账号池，之后由轮询负载均衡自动使用。<br>
  只需填写注册数量，邮箱前缀与密码均随机生成；也可在「高级」里自定义邮箱列表。<br>
  ⚠️ 注册接口对大陆 IP 直接返回 <code>REGISTER_FROM_MAINLAND</code>，需先在「设置」中配置海外代理。
</div>
<div class="pw-row" style="display:flex;gap:8px;align-items:center">
  <input type="number" id="regCount" value="1" min="1" max="500" placeholder="注册数量" style="flex:1;background:#0f172a;border:1px solid #334155;border-radius:8px;color:#e2e8f0;padding:12px;font-size:14px">
  <span style="font-size:13px;color:#94a3b8;white-space:nowrap">个账号（邮箱前缀随机生成）</span>
</div>
<details style="margin-top:6px">
  <summary style="cursor:pointer;font-size:12px;color:#64748b">高级：自定义邮箱列表（留空则按数量随机生成）</summary>
  <div class="pw-row" style="margin-top:6px">
    <textarea id="regEmails" placeholder="每行一个邮箱，或 user{1-5}@example.com 批量展开" style="width:100%;height:70px;background:#0f172a;border:1px solid #334155;border-radius:8px;color:#e2e8f0;padding:12px;font-size:13px;resize:vertical"></textarea>
  </div>
</details>
<div class="pw-row" style="margin-top:8px;display:flex;gap:8px;align-items:center">
  <input type="password" id="regPassword" placeholder="注册密码（至少8位，含字母+数字）" style="flex:1;background:#0f172a;border:1px solid #334155;border-radius:8px;color:#e2e8f0;padding:12px;font-size:14px">
  <label style="display:flex;align-items:center;gap:6px;font-size:13px;color:#e2e8f0;white-space:nowrap;cursor:pointer">
    <input type="checkbox" id="regRandomPw" checked onchange="regPwToggle()"> 随机生成密码
  </label>
</div>
<div class="row" style="margin-top:8px;display:flex;align-items:center;gap:8px">
  <span style="font-size:12px;color:#94a3b8">Region:</span>
  <select id="regRegion" style="background:#0f172a;border:1px solid #334155;border-radius:8px;color:#e2e8f0;padding:8px;font-size:13px">
    <option value="US">US</option><option value="JP">JP</option><option value="SG">SG</option>
    <option value="GB">GB</option><option value="DE">DE</option><option value="AU">AU</option>
  </select>
  <span style="font-size:12px;color:#94a3b8">验证码来源:</span>
  <select id="regReader" onchange="regReaderChange()" style="background:#0f172a;border:1px solid #334155;border-radius:8px;color:#e2e8f0;padding:8px;font-size:13px">
    <option value="mailcx">mail.cx 临时邮箱（推荐）</option>
    <option value="manual">手动输入</option>
    <option value="imap">IMAP 收件箱</option>
    <option value="temp_api">通用临时邮箱 API</option>
  </select>
</div>
<div id="regMailcx" style="margin-top:8px">
  <div class="pw-row" style="display:flex;gap:6px;align-items:center">
    <input id="regMailcxKey" placeholder="mail.cx API Key (tm_live_...)" style="flex:3;background:#0f172a;border:1px solid #334155;border-radius:8px;color:#e2e8f0;padding:10px;font-size:13px">
    <button class="btn" style="background:#2563eb;color:#fff;font-size:12px;padding:10px 14px" onclick="loadMailcxDomains()">获取域名</button>
  </div>
  <div class="pw-row" style="margin-top:6px;display:flex;gap:6px;align-items:center">
    <select id="regMailcxDomain" style="flex:3;background:#0f172a;border:1px solid #334155;border-radius:8px;color:#e2e8f0;padding:10px;font-size:13px">
      <option value="">（先获取域名）</option>
    </select>
    <button class="btn" style="background:#334155;color:#e2e8f0;font-size:12px;padding:10px 14px" onclick="saveMailcxConfig()">保存配置</button>
  </div>
  <div style="font-size:11px;color:#64748b;margin-top:4px">mail.cx 本地部分仅允许小写字母/数字/._-，建议用 <code>user{01-10}</code> 展开（不要用 + 号）；纯本地部分会自动补全 @域名</div>
</div>
<div id="regImap" style="display:none;margin-top:8px">
  <div class="row" style="display:flex;gap:6px">
    <input class="ph" id="regImapHost" placeholder="IMAP 服务器 (imap.example.com)" style="flex:2;background:#0f172a;border:1px solid #334155;border-radius:8px;color:#e2e8f0;padding:10px;font-size:13px">
    <input class="ph" id="regImapPort" placeholder="端口" value="993" style="flex:1;background:#0f172a;border:1px solid #334155;border-radius:8px;color:#e2e8f0;padding:10px;font-size:13px">
  </div>
  <div class="pw-row" style="margin-top:6px"><input id="regImapUser" placeholder="IMAP 账号（通常为邮箱）" style="width:100%;background:#0f172a;border:1px solid #334155;border-radius:8px;color:#e2e8f0;padding:10px;font-size:13px"></div>
  <div class="pw-row" style="margin-top:6px"><input type="password" id="regImapPass" placeholder="IMAP 密码 / 授权码" style="width:100%;background:#0f172a;border:1px solid #334155;border-radius:8px;color:#e2e8f0;padding:10px;font-size:13px"></div>
</div>
<div id="regTempApi" style="display:none;margin-top:8px">
  <div class="pw-row"><input id="regTempUrl" placeholder="临时邮箱 JSON API URL（返回含 6 位验证码的 JSON）" style="width:100%;background:#0f172a;border:1px solid #334155;border-radius:8px;color:#e2e8f0;padding:10px;font-size:13px"></div>
</div>
<div class="row" style="margin-top:10px;display:flex;gap:8px;align-items:center">
  <button class="btn bp" onclick="startBatch()">开始批量注册</button>
  <button class="btn" style="background:#7f1d1d;color:#fca5a5" onclick="stopBatch()">停止任务</button>
  <button class="btn" style="background:#334155;color:#e2e8f0" onclick="loadRegStatus()">刷新状态</button>
  <button class="btn" style="background:#0d9488;color:#fff" onclick="exportAccounts()">下载账号列表</button>
</div>
<div id="regStatus" style="margin-top:12px;font-size:12px;color:#64748b"></div>
</div>
</div>

<div id="toast" class="toast"></div>
<script>
// === i18n ===
var _lang=localStorage.getItem('ds_lang')||'zh';
var _I={
zh:{phoneLogin:'手机号登录',emailLogin:'邮箱登录',usage:'用量统计',accounts:'账号管理',settings:'设置',
phonePlaceholder:'手机号',pwdPlaceholder:'密码',loginBtn:'登录',loginBtnDoing:'登录中...',
emailPlaceholder:'邮箱地址',waitingCfg:'等待配置',configured:'已配置',connFail:'连接失败',
loggingDS:'正在登录 DeepSeek...',loginOk:'登录成功',loginFail:'失败:',
error:'错误:',saveCurlBtn:'保存 cURL',
parsing:'解析中...',saved:'已保存',apiConfig:'API 配置',apiAddr:'API 地址',
apiKey:'API Key',apiKeyVal:'任意填写',refreshModels:'🔄 刷新模型列表',
refreshingModels:'刷新中...',foundModels:'✅ 发现',foundModelsSuffix:'个模型:',
refreshOk:'刷新成功',refreshFail:'刷新失败',
periodAll:'全部',periodWeek:'本周',periodToday:'今日',refreshBtn:'刷新',clearBtn:'清空',
noData:'📊 暂无用量数据',loadFail:'加载失败: ',modelHeader:'模型',reqHeader:'请求',
inputHeader:'输入',outputHeader:'输出',totalHeader:'总计',sumLabel:'📋 合计',
clearConfirm:'确定清空全部用量数据？',cleared:'已清空',clearFail:'清空失败',
loadingAccounts:'加载中...',noAccounts:'暂无账号，请先添加',accountHeader:'账号',
statusHeader:'状态',tokenHeader:'Token',loginTimeHeader:'登录时间',opHeader:'操作',
valid:'有效',notLogin:'未登录',reloginBtn:'重登',deleteBtn:'删除',
reloginAllBtn:'全部重新登录',cleanupSessionsBtn:'清理过期会话',
allRelogining:'登录中...',allReloginDone:'重登完成:',allReloginFail:'失败:',
deleteConfirm:'确定删除账号',deleted:'已删除',deleteFail:'删除失败:',
reloginOk:'重新登录成功',reloginFail:'重登失败: ',
acctCount:'共',acctCount2:'个账号，',acctCount3:'个有效',
cleanupDone:'清理完成',cleanupFail:'清理失败: ',
addAcctTitle:'添加账号（手机号登录）',addAcctBtn:'添加',
phoneRequired:'请输入手机号和密码',emailRequired:'请输入邮箱和密码',
addOk:'已添加，需登录获取token',addFail:'失败: ',
pleaseAdd:'暂无账号，请在上方添加',pasteCurl:'粘贴 cURL ...',
modelCountSuffix:' 个模型: ',acctAddFail:'添加失败: ',unknownErr:'未知错误',
curlTitle:'📋 cURL 导入',curlSteps:'导入步骤：',
curlStep1:'1. 打开 chat.deepseek.com 并登录',curlStep2:'2. 按 F12 → Network 面板',
curlStep3:'3. 发送任意消息，找到 completion 请求',curlStep4:'4. 右键 → Copy as cURL，粘贴到下方',
cleanupBtnDoing:'清理中...',unknown:'未知',
proxyTitle:'代理配置',proxyHint:'绕过 AWS WAF 拦截。格式：http://127.0.0.1:7890 或 socks5://127.0.0.1:7891',proxySaveBtn:'保存代理设置',proxySaved:'已保存',proxySaveFail:'保存失败: ',proxyLoadFail:'加载失败: ',
passthroughTitle:'工具透传模式',passthroughHint:'跳过 DSML 格式说明书，直接嵌入原始工具定义（适合 Roo Code / Cline）',passthroughToggle:'关闭',passthroughSaveBtn:'保存',passthroughSaved:'已保存',passthroughSaveFail:'保存失败: ',passthroughLoadFail:'加载失败: ',
changePwdTitle:'修改管理员密码',newPwdPlaceholder:'输入新密码',changePwdBtn:'保存新密码',changePwdOk:'密码已更新',changePwdFail:'修改失败: ',changePwdEmpty:'密码不能为空'},
en:{phoneLogin:'Phone Login',emailLogin:'Email Login',usage:'Usage',accounts:'Accounts',
phonePlaceholder:'Phone Number',pwdPlaceholder:'Password',loginBtn:'Login',loginBtnDoing:'Logging in...',
emailPlaceholder:'Email Address',waitingCfg:'Awaiting Config',configured:'Configured',connFail:'Connection Failed',
loggingDS:'Logging into DeepSeek...',loginOk:'Login Successful',loginFail:'Failed:',
error:'Error:',saveCurlBtn:'Save cURL',
parsing:'Parsing...',saved:'Saved',apiConfig:'API Config',apiAddr:'API Endpoint',
apiKey:'API Key',apiKeyVal:'Any value',refreshModels:'🔄 Refresh Models',
refreshingModels:'Refreshing...',foundModels:'✅ Found',foundModelsSuffix:'model(s):',
refreshOk:'Refreshed',refreshFail:'Refresh Failed',
periodAll:'All',periodWeek:'This Week',periodToday:'Today',refreshBtn:'Refresh',clearBtn:'Clear',
noData:'📊 No Usage Data',loadFail:'Load failed: ',modelHeader:'Model',reqHeader:'Requests',
inputHeader:'Input',outputHeader:'Output',totalHeader:'Total',sumLabel:'📋 Total',
clearConfirm:'Clear all usage data?',cleared:'Cleared',clearFail:'Clear Failed',
loadingAccounts:'Loading...',noAccounts:'No accounts. Add one above.',accountHeader:'Account',
statusHeader:'Status',tokenHeader:'Token',loginTimeHeader:'Login Time',opHeader:'Actions',
valid:'Active',notLogin:'Not Logged In',reloginBtn:'Relogin',deleteBtn:'Delete',
reloginAllBtn:'Relogin All',cleanupSessionsBtn:'Cleanup Old Sessions',
allRelogining:'Relogging...',allReloginDone:'Relogin done:',allReloginFail:'Failed:',
deleteConfirm:'Delete account',deleted:'Deleted',deleteFail:'Delete failed:',
reloginOk:'Relogin Successful',reloginFail:'Relogin failed: ',
acctCount:'',acctCount2:' account(s), ',acctCount3:' active',
cleanupDone:'Cleanup done',cleanupFail:'Cleanup failed: ',
addAcctTitle:'Add Account (Phone Login)',addAcctBtn:'Add',
phoneRequired:'Phone number and password required',emailRequired:'Email and password required',
addOk:'Added. Login needed to get token.',addFail:'Failed: ',
pleaseAdd:'No accounts. Add one above.',pasteCurl:'Paste cURL ...',
modelCountSuffix:' model(s): ',acctAddFail:'Add failed: ',unknownErr:'Unknown error',
curlTitle:'📋 cURL Import',curlSteps:'Steps:',
curlStep1:'1. Open chat.deepseek.com and log in',curlStep2:'2. Press F12 → Network tab',
curlStep3:'3. Send any message, find the completion request',curlStep4:'4. Right-click → Copy as cURL, paste below',
cleanupBtnDoing:'Cleaning...',unknown:'Unknown',
proxyTitle:'Proxy Config',proxyHint:'Bypass AWS WAF. Format: http://127.0.0.1:7890 or socks5://127.0.0.1:7891',proxySaveBtn:'Save Proxy',proxySaved:'Saved',proxySaveFail:'Save Failed: ',proxyLoadFail:'Load Failed: ',
passthroughTitle:'Tool Passthrough Mode',passthroughHint:'Skip DSML format spec, embed raw tool definitions (suitable for Roo Code / Cline)',passthroughToggle:'Off',passthroughSaveBtn:'Save',passthroughSaved:'Saved',passthroughSaveFail:'Save Failed: ',passthroughLoadFail:'Load Failed: ',
changePwdTitle:'Change Admin Password',newPwdPlaceholder:'Enter new password',changePwdBtn:'Save New Password',changePwdOk:'Password updated',changePwdFail:'Failed: ',changePwdEmpty:'Password cannot be empty'}};
function _(k){return (_I[_lang]||_I.zh)[k]||k}
function toggleLang(){_lang=_lang==='zh'?'en':'zh';localStorage.setItem('ds_lang',_lang);Q('langBtn').textContent=_lang==='zh'?'🌐 EN':'🌐 中';applyI18n()}
function applyI18n(){
Qs('[data-i18n]').forEach(function(el){var k=el.getAttribute('data-i18n');if(k){el.textContent=_(k)}});
Qs('[data-i18n-ph]').forEach(function(el){var k=el.getAttribute('data-i18n-ph');if(k){el.placeholder=_(k)}});
Qs('[data-i18n-val]').forEach(function(el){var k=el.getAttribute('data-i18n-val');if(k){el.value=_(k)}});
Qs('[data-i18n-confirm]').forEach(function(el){el.setAttribute('data-i18n-confirm-msg',_(el.getAttribute('data-i18n-confirm')))});
// Update tab texts
var tabs=document.querySelectorAll('.tab');var tkeys=['phoneLogin','emailLogin','usage','accounts'];
for(var i=0;i<4&&i<tabs.length;i++){if(tabs[i]!==Q('langBtn'))tabs[i].textContent=_(tkeys[i])}
loadUsage();loadAccounts();cs();
}
function Qs(s){return document.querySelectorAll(s)}
document.addEventListener('DOMContentLoaded',function(){Q('langBtn').textContent=_lang==='zh'?'🌐 EN':'🌐 中';applyI18n()});
function Q(id){return document.getElementById(id)}
function switchTab(type){
var ti={'phone':0,'email':1,'usage':2,'accounts':3,'settings':4,'register':5};
document.querySelectorAll('.tab').forEach((t,i)=>{t.className='tab'+(i===ti[type]?' active':'');});
Q('phonePanel').className='panel'+(type==='phone'?' active':'');
Q('emailPanel').className='panel'+(type==='email'?' active':'');
if(Q('usagePanel'))Q('usagePanel').className='panel'+(type==='usage'?' active':'');
if(Q('accountsPanel'))Q('accountsPanel').className='panel'+(type==='accounts'?' active':'');
if(Q('settingsPanel'))Q('settingsPanel').className='panel'+(type==='settings'?' active':'');
if(Q('registerPanel'))Q('registerPanel').className='panel'+(type==='register'?' active':'');
var as=Q('apiSection');if(as)as.style.display=(type==='usage'||type==='accounts'||type==='settings'||type==='register')?'none':'';
if(type==='usage')loadUsage();
if(type==='accounts')loadAccounts();
if(type==='settings'){loadProxy();loadPassthrough();}
if(type==='register'){loadRegStatus();regPwToggle();}
}
async function cs(){
try{const r=await fetch('/api/config');const d=await r.json()
if(d.configured){Q('s').className='s ok';Q('sd').className='d dg';Q('st').textContent=_('configured')+' | '+d.masked}
else{Q('s').className='s no';Q('sd').className='d dy';Q('st').textContent=d.error||_('waitingCfg')}
}catch(e){Q('s').className='s err';Q('st').textContent=_('connFail')}
}
async function doLogin(type){
let body={}
if(type==='phone'){
const m=Q('mobile').value.trim();const p=Q('pw1').value;const a=Q('area_code').value.trim()
if(!m||!p){t(_('phoneRequired'),1);return}
body={mobile:m,password:p,area_code:a,login_type:'phone'}
var btn=Q('btn1')
}else{
const e=Q('email').value.trim();const p=Q('pw2').value
if(!e||!p){t(_('emailRequired'),1);return}
body={email:e,password:p,login_type:'email'}
var btn=Q('btn2')
}
btn.disabled=true;btn.textContent=_('loginBtnDoing')
Q('info').style.display='block';Q('info').innerHTML=_('loggingDS')
try{
const r=await fetch('/api/login',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})
const d=await r.json()
if(d.ok){Q('info').innerHTML=_('loginOk')+' | Token: '+d.masked+' | Session: '+d.session_id;t(_('loginOk'));cs()}
else{Q('info').innerHTML=_('loginFail')+d.error;t(d.error,1)}
}catch(e){Q('info').innerHTML=_('error')+e.message;t(e.message,1)}
btn.disabled=false;btn.textContent=_('loginBtn')
}
async function saveCurl(){
const c=Q('curl').value.trim();if(!c){t(_('pasteCurl'),1);return}
const b=Q('btn3');b.disabled=true;b.textContent=_('saveCurlBtn')+'...'
Q('info').style.display='block';Q('info').innerHTML=_('parsing')
try{
const r=await fetch('/api/config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({curl:c})})
const d=await r.json()
if(d.ok){Q('info').innerHTML='OK | '+d.masked+' | Session '+d.session_id;t(_('saved'));cs()}
else{Q('info').innerHTML=_('loginFail')+d.error;t(d.error,1)}
}catch(e){Q('info').innerHTML=_('error')+e.message;t(e.message,1)}
b.disabled=false;b.textContent=_('saveCurlBtn')
}
function cp(el){navigator.clipboard.writeText(el.textContent);t('已复制')}
function t(m,e){const x=Q('toast');x.textContent=m;x.className='toast t'+(e?'e':'s');setTimeout(()=>x.className='toast',2500)}
async function refreshModels(){
const btn=Q('refreshBtn');const info=Q('modelsInfo')
btn.disabled=true;btn.textContent=_('refreshingModels');info.style.display='none'
try{
const r=await fetch('/v1/models/refresh',{method:'POST'})
const d=await r.json()
const names=d.data.map(m=>m.id).join(', ')
info.style.display='block';info.innerHTML=_('foundModels')+' '+d.data.length+_('modelCountSuffix')+names;t(_('refreshOk'))
}catch(e){info.style.display='block';info.innerHTML='❌ '+_('refreshFail')+': '+e.message;t(_('refreshFail'),1)}
btn.disabled=false;btn.textContent=_('refreshModels')
}
// === 账号管理 ===
async function loadAccounts(){
try{
const r=await fetch('/api/accounts');const d=await r.json();
var h='';
if(d.accounts&&d.accounts.length>0){
Q('acctStat').innerHTML=_('acctCount')+d.total+_('acctCount2')+d.valid+_('acctCount3');
h+='<table class="acct-tbl"><tr><th>'+_('accountHeader')+'</th><th>'+_('statusHeader')+'</th><th>'+_('tokenHeader')+'</th><th>'+_('loginTimeHeader')+'</th><th>'+_('opHeader')+'</th></tr>';
for(var a of d.accounts){
var st=a.is_valid?'ok':'no';
var stT=a.is_valid?_('valid'):_('notLogin');
var l=encodeURIComponent(a.account_label);
h+='<tr><td>'+a.account_label+'</td><td><span class="acct-st '+st+'"></span>'+stT+'</td><td>'+(a.token_masked||'***')+'</td><td>'+(a.login_time||'-')+'</td>';
h+=`<td><button class="acct-btn rl" onclick="reloginAccount('${l}')">`+_('reloginBtn')+`</button><br><button class="acct-btn rm" onclick="removeAccount('${l}')">`+_('deleteBtn')+`</button></td>`;
}
h+='</table>';
}else{h='<div class="acct-empty">'+_('noAccounts')+'</div>'}
Q('acctList').innerHTML=h;
}catch(e){Q('acctList').innerHTML='<div class="acct-empty">'+_('loadFail')+e.message+'</div>'}
}
async function addAccount(){
var phone=Q('acctPhone').value.trim();
var code=Q('acctCode').value.trim()||'+86';
var pw=Q('acctPw').value;
if(!phone||!pw){t(_('phoneRequired'),1);return}
try{
var r=await fetch('/api/accounts',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mobile:phone,area_code:code,password:pw,login_type:'phone'})});
var d=await r.json();
if(d.ok){t(_('addOk'));Q('acctPhone').value='';Q('acctPw').value='';loadAccounts()}
else{t(_('addFail')+(d.error||_('unknownErr')),1)}
}catch(e){t(_('acctAddFail')+e.message,1)}
}
async function removeAccount(label){
if(!confirm(_('deleteConfirm')+' '+decodeURIComponent(label)+'？'))return;
try{
var r=await fetch('/api/accounts/'+label,{method:'DELETE'});
var d=await r.json();
if(d.ok){t(_('deleted'));loadAccounts()}
else{t(_('deleteFail')+(d.error||_('unknownErr')),1)}
}catch(e){t(_('deleteFail')+e.message,1)}
}
async function reloginAccount(label){
var btn=event&&event.target;if(btn){btn.disabled=true;btn.textContent='...'}
try{
var r=await fetch('/api/accounts/'+label+'/relogin',{method:'POST'});
var d=await r.json();
if(d.ok){t(_('reloginOk'));loadAccounts()}
else{t(_('reloginFail')+(d.error||_('unknownErr')),1)}
}catch(e){t(_('reloginFail')+e.message,1)}
if(btn){btn.disabled=false;btn.textContent=_('reloginBtn')}
}
async function reloginAll(){
var btn=event&&event.target;if(btn){btn.disabled=true;btn.textContent=_('allRelogining')}
try{
var r=await fetch('/api/accounts/relogin-all',{method:'POST'});
var d=await r.json();
if(d.results){
var ok=d.results.filter(x=>x.ok).length;
t(_('allReloginDone')+' '+ok+'/'+d.total+' '+(ok===d.total?_('loginOk'):''));
loadAccounts();
}else{t(_('allReloginFail')+(d.error||_('unknown')),1)}
}catch(e){t(_('allReloginFail')+e.message,1)}
if(btn){btn.disabled=false;btn.textContent=_('reloginAllBtn')}
}
async function cleanupSessions(){
var btn=event&&event.target;if(btn){btn.disabled=true;btn.textContent=_('cleanupBtnDoing')}
try{
var r=await fetch('/api/cleanup',{method:'POST'});
var d=await r.json();
t(d.ok?d.msg:_('cleanupFail')+(d.msg||_('unknown')),d.ok?0:1)
}catch(e){t(_('cleanupFail')+e.message,1)}
if(btn){btn.disabled=false;btn.textContent=_('cleanupSessionsBtn')}
}
// === 批量注册 ===
var _regJob='';
function regReaderChange(){
  var m=Q('regReader').value;
  Q('regImap').style.display=(m==='imap')?'':'none';
  Q('regTempApi').style.display=(m==='temp_api')?'':'none';
  Q('regMailcx').style.display=(m==='mailcx')?'':'none';
  if(m==='mailcx')loadMailcxConfig();
}
async function loadMailcxConfig(){
  try{
    const r=await fetch('/api/register/mailcx-config');
    const d=await r.json();
    if(d.api_key){Q('regMailcxKey').value=d.api_key;loadMailcxDomains();}
    if(d.domain)Q('regMailcxDomain').value=d.domain;
  }catch(e){}
}
async function loadMailcxDomains(){
  var key=Q('regMailcxKey').value.trim();
  if(!key){t('请先填写 mail.cx API Key',1);return}
  try{
    const r=await fetch('/api/register/mailcx-domains?api_key='+encodeURIComponent(key));
    const d=await r.json();
    if(!d.ok){t(d.error||'获取域名失败',1);return}
    var sel=Q('regMailcxDomain');
    sel.innerHTML='<option value="">（选择邮箱后缀）</option>'+d.domains.map(x=>'<option value="'+x+'">'+x+'</option>').join('');
    var cur=Q('regMailcxDomain').value;
    t('获取到 '+d.domains.length+' 个可用后缀');
    saveMailcxConfig();
  }catch(e){t('错误: '+e.message,1)}
}
async function saveMailcxConfig(){
  var key=Q('regMailcxKey').value.trim();
  if(!key)return;
  try{
    await fetch('/api/register/mailcx-config',{method:'PUT',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({api_key:key,domain:Q('regMailcxDomain').value})});
  }catch(e){}
}
function regPwToggle(){
  Q('regPassword').disabled=Q('regRandomPw').checked;
  Q('regPassword').style.opacity=Q('regRandomPw').checked?'0.5':'1';
}
async function startBatch(){
  var emails=(Q('regEmails').value||'').split(String.fromCharCode(10)).map(s=>s.trim()).filter(Boolean);
  var count=parseInt(Q('regCount').value)||0;
  var randomPw=Q('regRandomPw').checked;
  var password=randomPw?'':Q('regPassword').value;
  if(!emails.length&&!count){t('请填写注册数量',1);return}
  if(!randomPw&&!password){t('请填写注册密码或勾选随机生成',1);return}
  var reader={mode:Q('regReader').value};
  var domain='';
  if(reader.mode==='mailcx'){
    reader.api_key=Q('regMailcxKey').value.trim();
    domain=Q('regMailcxDomain').value;
    if(!reader.api_key){t('请填写 mail.cx API Key',1);return}
    if(!domain){t('请选择邮箱后缀',1);return}
    saveMailcxConfig();
  }else if(reader.mode==='imap'){
    reader.host=Q('regImapHost').value.trim();
    reader.port=parseInt(Q('regImapPort').value)||993;
    reader.user=Q('regImapUser').value.trim();
    reader.password=Q('regImapPass').value;
    reader.folder='INBOX';
    if(!reader.host||!reader.user||!reader.password){t('请填写完整 IMAP 配置',1);return}
  }else if(reader.mode==='temp_api'){
    reader.url=Q('regTempUrl').value.trim();
    if(!reader.url){t('请填写临时邮箱 API URL',1);return}
  }
  try{
    const r=await fetch('/api/register/start',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({emails:emails,count:count,password:password,random_password:randomPw,region:Q('regRegion').value,
        auto_login:true,reader:reader,wait_timeout:300,domain:domain})});
    const d=await r.json();
    if(d.ok){_regJob=d.job_id;t('任务已启动: '+d.job_id);
      Q('regStatus').innerHTML='<div style="color:#7dd3fc">任务 '+d.job_id+' 运行中，共 '+d.emails.length+' 个邮箱...</div>';
      pollRegStatus();
    }else t(d.error||d.detail||'启动失败',1);
  }catch(e){t('错误: '+e.message,1)}
}
function pollRegStatus(){
  if(!_regJob)return;
  loadRegStatus();
  setTimeout(pollRegStatus,2000);
}
async function loadRegStatus(){
  if(!_regJob){Q('regStatus').innerHTML='';return}
  try{
    const r=await fetch('/api/register/status?job_id='+encodeURIComponent(_regJob));
    const d=await r.json();
    renderRegStatus(d);
    if(d.status==='done'||d.status==='stopped')_regJob='';
  }catch(e){}
}
function renderRegStatus(job){
  var stColor=job.status==='done'?'#4ade80':(job.status==='failed'||job.status==='stopped'?'#f87171':'#7dd3fc');
  var h='<div style="color:'+stColor+';margin-bottom:6px;font-weight:600">任务 '+job.id+' ['+job.status+']'+
        ' · 密码：'+(job.password_masked||'-')+' · 完成 '+
        Object.values(job.results).filter(function(r){return r.status==='done'}).length+'/'+job.emails.length+'</div>';
  h+='<table class="acct-tbl"><tr><th>邮箱</th><th>状态</th><th>Token</th><th>耗时(s)</th><th>错误</th></tr>';
  for(var email of job.emails){
    var r=job.results[email]||{};
    var c=r.status==='done'?'#4ade80':(r.status==='failed'?'#f87171':'#94a3b8');
    h+='<tr><td>'+email+'</td><td style="color:'+c+'">'+r.status+'</td><td>'+(r.token_masked||'')+'</td><td>'+(r.elapsed||'')+'</td><td style="color:#f87171;font-size:11px">'+(r.error||'')+'</td></tr>';
  }
  h+='</table>';
  h+='<div id="regLogBox" style="margin-top:6px;max-height:160px;overflow:auto;font-family:monospace;font-size:11px;color:#94a3b8;background:#0f172a;border:1px solid #334155;border-radius:8px;padding:8px">'+job.logs.join('<br>')+'</div>';
  if(job.status==='running'){
    h+='<div style="margin-top:8px;display:flex;gap:6px;align-items:center">'+
       '<input id="regCodeEmail" placeholder="邮箱" style="flex:2;background:#0f172a;border:1px solid #334155;border-radius:8px;color:#e2e8f0;padding:8px;font-size:12px">'+
       '<input id="regCode" placeholder="6位验证码" style="flex:1;background:#0f172a;border:1px solid #334155;border-radius:8px;color:#e2e8f0;padding:8px;font-size:12px">'+
       '<button class="btn" style="background:#2563eb;color:#fff;font-size:12px;padding:8px 14px" onclick="submitBatchCode()">提交验证码</button></div>';
  }else{
    h+='<div style="margin-top:8px;display:flex;gap:8px">'+
       '<button class="btn" style="background:#0d9488;color:#fff;font-size:12px;padding:8px 14px" onclick="exportAccounts()">下载账号密码列表</button>'+
       '<button class="btn" style="background:#334155;color:#e2e8f0;font-size:12px;padding:8px 14px" onclick="exportPool()">导出账号池全部账号</button></div>';
  }
  Q('regStatus').innerHTML=h;
  var lb=Q('regLogBox');if(lb)lb.scrollTop=lb.scrollHeight;
}
async function exportAccounts(){
  if(!_regJob){t('无任务，先启动批量注册');return}
  try{
    const r=await fetch('/api/register/export?job_id='+encodeURIComponent(_regJob));
    if(!r.ok){let d={};try{d=await r.json()}catch(e){};t(d.detail||('导出失败 HTTP '+r.status),1);return}
    const blob=await r.blob();
    const url=URL.createObjectURL(blob);
    const a=document.createElement('a');
    a.href=url;a.download='deepseek_accounts_'+_regJob+'.csv';
    document.body.appendChild(a);a.click();a.remove();
    URL.revokeObjectURL(url);
    t('已导出账号密码列表');
  }catch(e){t('下载失败: '+e.message,1)}
}
async function exportPool(){
  try{
    const r=await fetch('/api/accounts/export');
    if(!r.ok){t('导出失败',1);return}
    const blob=await r.blob();
    const url=URL.createObjectURL(blob);
    const a=document.createElement('a');
    a.href=url;a.download='deepseek_accounts_pool.csv';
    document.body.appendChild(a);a.click();a.remove();
    URL.revokeObjectURL(url);
    t('已导出账号池');
  }catch(e){t('下载失败: '+e.message,1)}
}
async function submitBatchCode(){
  var email=Q('regCodeEmail').value.trim();
  var code=Q('regCode').value.trim();
  if(!email||!code){t('请填写邮箱和验证码',1);return}
  try{
    const r=await fetch('/api/register/code',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({job_id:_regJob,email:email,code:code})});
    const d=await r.json();
    if(d.ok){t('验证码已提交');loadRegStatus()}else t(d.error||d.detail||'提交失败',1);
  }catch(e){t('错误: '+e.message,1)}
}
async function stopBatch(){
  if(!_regJob){t('无运行中的任务');return}
  try{
    await fetch('/api/register/stop',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({job_id:_regJob})});
    t('已请求停止');loadRegStatus();
  }catch(e){t('错误: '+e.message,1)}
}
// === 代理配置 ===
async function loadProxy(){
try{
const r=await fetch('/api/proxy');const d=await r.json();
Q('proxyUrl').value=d.proxy||'';
var st=Q('proxyStatus');
st.textContent=d.proxy?_('proxySaved'):'';
st.style.color=d.proxy?'#22c55e':'#64748b';
}catch(e){Q('proxyStatus').textContent=_('proxyLoadFail')+e.message}
}
async function saveProxy(){
var url=Q('proxyUrl').value.trim();
try{
const r=await fetch('/api/proxy',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({proxy:url})});
const d=await r.json();
if(d.ok){Q('proxyStatus').textContent=_('proxySaved');Q('proxyStatus').style.color='#22c55e';t(_('proxySaved'))}
else{Q('proxyStatus').textContent=_('proxySaveFail')+d.msg;t(_('proxySaveFail')+(d.msg||''),1)}
}catch(e){Q('proxyStatus').textContent=_('proxySaveFail')+e.message;t(_('proxySaveFail')+e.message,1)}
}
async function changePwd(){
var newPwd=Q('newPwd').value.trim();
if(!newPwd){Q('pwdStatus').textContent=_('changePwdEmpty');Q('pwdStatus').style.color='#fca5a5';return}
try{
const r=await fetch('/api/admin-password',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({password:newPwd})});
const d=await r.json();
if(d.ok){
Q('pwdStatus').textContent=_('changePwdOk');Q('pwdStatus').style.color='#22c55e';t(_('changePwdOk'));
Q('newPwd').value='';
}else{Q('pwdStatus').textContent=_('changePwdFail')+(d.detail||'');Q('pwdStatus').style.color='#fca5a5'}
}catch(e){Q('pwdStatus').textContent=_('changePwdFail')+e.message;Q('pwdStatus').style.color='#fca5a5'}
}
// === 透传模式 ===
async function loadPassthrough(){
try{
const r=await fetch('/api/passthrough');const d=await r.json();
Q('passthroughToggle').checked=!!d.passthrough;
updatePassthroughLabel();
}catch(e){Q('passthroughStatus').textContent=_('passthroughLoadFail')+e.message}
}
async function savePassthrough(){
var en=Q('passthroughToggle').checked;
try{
const r=await fetch('/api/passthrough',{method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify({passthrough:en})});
const d=await r.json();
if(d.ok){Q('passthroughStatus').textContent=_('passthroughSaved');Q('passthroughStatus').style.color='#22c55e';t(_('passthroughSaved'))}
else{Q('passthroughStatus').textContent=_('passthroughSaveFail')+d.msg;t(_('passthroughSaveFail')+(d.msg||''),1)}
}catch(e){Q('passthroughStatus').textContent=_('passthroughSaveFail')+e.message;t(_('passthroughSaveFail')+e.message,1)}
}
function updatePassthroughLabel(){
Q('passthroughLabel').textContent=Q('passthroughToggle').checked?'ON':'OFF';
Q('passthroughLabel').style.color=Q('passthroughToggle').checked?'#22c55e':'#e2e8f0';
}
// === 用量统计 ===
var _up='total';
function f(n){return n.toLocaleString()}
async function loadUsage(){
try{
const r=await fetch('/api/usage');const d=await r.json();
const p=d[_up]||d.total||{};const m=p.models||{};const t=p.total||{};
const e=Object.entries(m).sort((a,b)=>b[1].total_tokens-a[1].total_tokens);
if(!e.length&&!t.requests){Q('usageContent').innerHTML='<div class=ue>'+_('noData')+'</div>';return}
let h='<div class=us><table class=ut><thead><tr><th class=ml>'+_('modelHeader')+'</th><th>'+_('reqHeader')+'</th><th>'+_('inputHeader')+'</th><th>'+_('outputHeader')+'</th><th>'+_('totalHeader')+'</th></tr></thead><tbody>';
for(const[k,v]of e){h+=`<tr><td class=ml>${k}</td><td>${f(v.requests)}</td><td>${f(v.prompt_tokens)}</td><td>${f(v.completion_tokens)}</td><td>${f(v.total_tokens)}</td></tr>`}
h+=`<tr class=tr><td class=ml>`+_('sumLabel')+`</td><td>${f(t.requests)}</td><td>${f(t.prompt_tokens)}</td><td>${f(t.completion_tokens)}</td><td>${f(t.total_tokens)}</td></tr></tbody></table></div>`;
Q('usageContent').innerHTML=h
}catch(e){Q('usageContent').innerHTML='<div class=ue>'+_('loadFail')+e.message+'</div>'}
}
function switchPeriod(p){
_up=p;
['total','week','today'].forEach(x=>{var b=Q('pb'+x.charAt(0).toUpperCase()+x.slice(1));if(b)b.className='pb'+(x===p?' ac':'')});
loadUsage()
}
async function clearUsage(){
if(!confirm(_('clearConfirm')))return;
try{await fetch('/api/usage',{method:'DELETE'});t(_('cleared'));loadUsage()}catch(e){t(_('clearFail'),1)}
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
async def admin(creds: HTTPBasicCredentials = Depends(verify_admin)):
    from starlette.responses import Response
    html = ADMIN
    return Response(content=html, media_type="text/html", headers={
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
    })


# ── 配置 API ─────────────────────────────────────────────

def _load_config_sync() -> dict:
    """同步加载配置信息（兼容旧接口）。多账号模式下取第一个有效账号。"""
    accounts = config_manager.get_all_accounts()
    if not accounts:
        return {}
    first = accounts[0]
    return {
        "configured": True,
        "masked": first.get("token_masked", "***"),
        "session_id": first.get("session_id", "N/A"),
        "accounts": accounts,
    }


@app.get("/api/config")
async def get_config(creds: HTTPBasicCredentials = Depends(verify_admin)):
    accounts = config_manager.get_all_accounts()
    if not accounts:
        return {"configured": False, "error": "未配置"}
    first = accounts[0]
    return {
        "configured": True,
        "masked": first.get("token_masked", "***"),
        "session_id": first.get("session_id", "N/A"),
        "account_count": len(accounts),
        "valid_count": config_manager.count_valid(),
    }


@app.post("/api/config")
async def save_config(data: dict, creds: HTTPBasicCredentials = Depends(verify_admin)):
    curl = data.get("curl", "").strip()
    if not curl: raise HTTPException(400, "请提供 cURL")
    parsed = parse_curl(curl)
    cfg = build_config(parsed)
    if not cfg["token"]: return {"ok": False, "error": "未从 cURL 提取到 Token，请确认 Authorization header"}
    if not cfg["session_id"]: return {"ok": False, "error": "未从 cURL 提取到 Session ID"}
    # 创建账号并加入池
    account_label = f"curl_import_{cfg['token'][:8]}"
    ds_account = DsAccount(
        account_label=account_label,
        login_type="phone",
        token=cfg["token"],
        session_id=cfg["session_id"],
        headers=cfg.get("headers", {}),
        cookie=cfg.get("cookie", ""),
        login_time=time.strftime("%Y-%m-%d %H:%M:%S"),
        is_valid=True,
    )
    config_manager.add_account(ds_account)
    t = cfg["token"]
    return {"ok": True, "masked": t[:20] + "..." + t[-8:], "session_id": cfg["session_id"], "account_label": account_label}


# ── DeepSeek 登录 API ─────────────────────────────────────
@app.post("/api/login")
async def deepseek_login(data: dict):
    login_type = data.get("login_type", "phone")
    password = data.get("password", "").strip()
    if not password:
        raise HTTPException(400, "请提供密码")

    # 构造登录 payload（参考 NIyueeE/ds-free-api: email 和 mobile 二选一）
    login_payload = {"password": password, "device_id": new_device_id(), "os": "ios"}
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

    DS_HEADERS = random_ios_headers()

    try:
        # 1. 登录（iOS 随机指纹绕过 WAF）
        login_resp = cffi_requests.post(
            "https://chat.deepseek.com/api/v0/users/login",
            json=login_payload,
            headers=DS_HEADERS,
            timeout=30,
            proxies=_get_proxy_dict(),
        )

        # WAF 挑战检测
        if login_resp.status_code == 202 and login_resp.headers.get("x-amzn-waf-action"):
            return {"ok": False, "error": "登录被 AWS WAF 拦截 (HTTP 202)。请配置代理（设置 → 代理配置）绕过 WAF。"}

        raw_text = (login_resp.text or "").strip()
        if not raw_text:
            return {"ok": False, "error": f"登录失败: 服务器返回空响应 (HTTP {login_resp.status_code})，可能是 IP 被风控或需要完成人机验证"}

        try:
            login_data = login_resp.json()
        except Exception:
            preview = raw_text[:200]
            return {"ok": False, "error": f"登录失败: 服务器返回非 JSON 响应 (HTTP {login_resp.status_code}): {preview}"}
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

        # 2. 创建会话（iOS 随机指纹）
        auth_headers = {**DS_HEADERS, "authorization": f"Bearer {token}"}
        session_resp = cffi_requests.post(
            "https://chat.deepseek.com/api/v0/chat_session/create",
            json={},
            headers=auth_headers,
            timeout=15,
            proxies=_get_proxy_dict(),
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
        # 添加到多账号池
        ds_account = DsAccount(
            account_label=account_label,
            login_type=login_type,
            _password=password,
            _mobile=mobile if login_type == "phone" else "",
            _area_code=area_code if login_type == "phone" else "+86",
            _email=email if login_type == "email" else "",
            token=token,
            session_id=session_id,
            headers={**DS_HEADERS, "authorization": f"Bearer {token}"},
            login_time=time.strftime("%Y-%m-%d %H:%M:%S"),
            is_valid=True,
        )
        config_manager.add_account(ds_account)

        masked = token[:20] + "..." + token[-8:]
        return {"ok": True, "masked": masked, "session_id": session_id}

    except Exception as e:
        print(f"[Login] Error: {e}")
        return {"ok": False, "error": str(e)}


# ── Health ───────────────────────────────────────────────
@app.get("/health")
async def health():
    valid = config_manager.count_valid()
    total = config_manager.count()
    return {"status": "ok" if valid else "waiting", "configured": valid > 0, "accounts": total, "valid": valid}


# ─── 账号管理 API ───────────────────────────────────────────

@app.get("/api/accounts")
async def list_accounts(creds: HTTPBasicCredentials = Depends(verify_admin)):
    """获取所有账号列表"""
    return {
        "accounts": config_manager.get_all_accounts(),
        "total": config_manager.count(),
        "valid": config_manager.count_valid(),
    }


@app.post("/api/accounts")
async def add_account(data: dict, creds: HTTPBasicCredentials = Depends(verify_admin)):
    """手动添加账号"""
    login_type = data.get("login_type", "phone")
    password = data.get("password", "").strip()
    if not password:
        raise HTTPException(400, "请提供密码")

    if login_type == "email":
        email = data.get("email", "").strip()
        if not email:
            raise HTTPException(400, "请提供邮箱")
        account_label = email
    else:
        mobile = data.get("mobile", "").strip()
        area_code = data.get("area_code", "+86").strip()
        if not mobile:
            raise HTTPException(400, "请提供手机号")
        account_label = f"{area_code} {mobile}"

    existing = config_manager.get_account_by_label(account_label)
    if existing:
        if not existing.token:
            pass
        else:
            return {"ok": True, "account_label": account_label, "exist": True}

    ds_account = DsAccount(
        account_label=account_label,
        login_type=login_type,
        _password=password,
        _mobile=mobile if login_type == "phone" else "",
        _area_code=area_code if login_type == "phone" else "+86",
        _email=email if login_type == "email" else "",
        login_time="",
        is_valid=False,
    )
    added = config_manager.add_account(ds_account)
    return {"ok": True, "account_label": account_label, "added": added}


@app.delete("/api/accounts/{account_label}")
async def remove_account(account_label: str, creds: HTTPBasicCredentials = Depends(verify_admin)):
    """删除账号"""
    from urllib.parse import unquote
    label = unquote(account_label)
    if config_manager.remove_account(label):
        return {"ok": True, "account_label": label}
    raise HTTPException(404, f"账号 {label} 不存在")


@app.post("/api/accounts/{account_label}/relogin")
async def relogin_account(account_label: str, creds: HTTPBasicCredentials = Depends(verify_admin)):
    """重新登录指定账号"""
    from urllib.parse import unquote
    label = unquote(account_label)
    account = config_manager.get_account_by_label(label)
    if not account:
        raise HTTPException(404, f"账号 {label} 不存在")

    login_type = account.login_type
    password = account._password
    if not password:
        raise HTTPException(400, f"账号 {label} 无保存密码，无法自动登录")

    cfg = {
        "login_type": login_type,
        "_password": password,
        "_email": account._email,
        "_mobile": account._mobile,
        "_area_code": account._area_code,
        "account": label,
    }

    new_cfg = relogin(cfg)
    if new_cfg:
        return {"ok": True, "account_label": label, "token_masked": new_cfg.get("token", "")[:20] + "..."}
    return {"ok": False, "error": "重新登录失败"}


@app.post("/api/accounts/relogin-all")
async def relogin_all(creds: HTTPBasicCredentials = Depends(verify_admin)):
    """重新登录所有有效账号"""
    accounts = config_manager.get_all_accounts()
    results = []
    for acc in accounts:
        label = acc.get("account_label", "")
        account = config_manager.get_account_by_label(label)
        if not account or not account._password:
            results.append({"label": label, "ok": False, "error": "无密码"})
            continue

        cfg = {
            "login_type": account.login_type,
            "_password": account._password,
            "_email": account._email,
            "_mobile": account._mobile,
            "_area_code": account._area_code,
            "account": label,
        }
        new_cfg = relogin(cfg)
        results.append({"label": label, "ok": bool(new_cfg), "error": None if new_cfg else "登录失败"})

    return {"results": results, "total": len(results), "success": sum(1 for r in results if r["ok"])}


# ─── 批量注册 API ─────────────────────────────────────────
# 流程：发送验证码 → 读取邮箱验证码（IMAP/临时邮箱API/手动）→ 注册 → 自动登录并入池。
# 注册接口对大陆 IP 返回 REGISTER_FROM_MAINLAND，需在设置中配置海外代理。

@app.post("/api/register/send-code")
async def register_send_code(data: dict, creds: HTTPBasicCredentials = Depends(verify_admin)):
    """单独发送注册验证码（测试用）。body: {email, locale?}"""
    email = (data.get("email") or "").strip()
    if not email:
        raise HTTPException(400, "请提供邮箱")
    ok, info = send_verification_code(email, locale=data.get("locale", "zh_CN"), proxy=_get_proxy_dict())
    if not ok:
        return {"ok": False, "error": info.get("error", "发送失败")}
    return {"ok": True, "email": email, "send_window_secs": info.get("send_window_secs", 60)}


@app.post("/api/register/start")
async def register_batch_start(data: dict, creds: HTTPBasicCredentials = Depends(verify_admin)):
    """启动批量注册任务。body:
    {
      "count": 5,                                  // 注册数量：随机生成邮箱前缀（推荐，无需 emails）
      "emails": ["a@x.com", "b+{1-3}@x.com"],      // 可选：自定义邮箱列表，支持 {N-M} 展开
      "password": "...",                           // 可留空；留空或 random_password=true 时随机生成独立密码
      "random_password": false,
      "region": "US", "locale": "zh_CN",
      "auto_login": true,                          // 注册后自动登录并入池
      "wait_timeout": 180, "poll_interval": 5,
      "codes": {"a@x.com": "123456"},              // 预置验证码（可选）
      "domain": "ds.mail.zjb.edu.kg",              // 临时邮箱后缀；纯本地部分/随机前缀会自动补全
      "reader": {"mode": "manual"|"imap"|"temp_api"|"mailcx",
                 "api_key": "...",                 // mailcx 模式的 API Key
                 "host": "...", "port": 993, "user": "...", "password": "...",
                 "folder": "INBOX", "url": "..."}
    }
    """
    emails = expand_email_patterns([str(e) for e in data.get("emails", [])])
    count = int(data.get("count") or 0)
    # 未填邮箱列表时，按数量随机生成邮箱前缀 + 域名后缀
    if not emails and count > 0:
        domain = (data.get("domain") or config_manager.get_mailcx_domain() or "").strip().lstrip("@")
        if not domain:
            raise HTTPException(400, "随机生成邮箱需要先选择邮箱后缀")
        seen = set()
        generated = []
        while len(generated) < min(count, 500):
            local = generate_local_part()
            email = f"{local}@{domain}"
            if email in seen:
                continue
            seen.add(email)
            generated.append(email)
        emails = generated
    if not emails:
        raise HTTPException(400, "请填写注册数量或邮箱列表")
    password = (data.get("password") or "").strip()
    reader = data.get("reader") or {"mode": "manual"}
    # mailcx 模式：未显式传 api_key 时，回退到已保存的配置
    if reader.get("mode") == "mailcx" and not reader.get("api_key"):
        reader["api_key"] = config_manager.get_mailcx_api_key()
    try:
        job_id = _registrar.start(emails, password, {
            "region": data.get("region", "US"),
            "locale": data.get("locale", "zh_CN"),
            "auto_login": data.get("auto_login", True),
            "random_password": data.get("random_password", False),
            "wait_timeout": data.get("wait_timeout", 180),
            "poll_interval": data.get("poll_interval", 5),
            "codes": data.get("codes") or {},
            "domain": data.get("domain") or config_manager.get_mailcx_domain(),
            "reader": reader,
        })
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True, "job_id": job_id, "emails": emails}


@app.get("/api/register/status")
async def register_batch_status(job_id: str = "", creds: HTTPBasicCredentials = Depends(verify_admin)):
    """查询批量注册任务状态。不带 job_id 时列出所有任务摘要。"""
    if not job_id:
        return {"jobs": _registrar.list_jobs()}
    job = _registrar.get(job_id)
    if not job:
        raise HTTPException(404, f"任务 {job_id} 不存在")
    return job


@app.post("/api/register/code")
async def register_submit_code(data: dict, creds: HTTPBasicCredentials = Depends(verify_admin)):
    """手动提交验证码（manual 模式）。body: {job_id, email, code}"""
    job_id = data.get("job_id", "")
    email = (data.get("email") or "").strip()
    code = (data.get("code") or "").strip()
    if not job_id or not email or not code:
        raise HTTPException(400, "job_id / email / code 均不能为空")
    if _registrar.submit_code(job_id, email, code):
        return {"ok": True}
    raise HTTPException(404, f"任务或邮箱不存在: {job_id} / {email}")


@app.post("/api/register/stop")
async def register_batch_stop(data: dict, creds: HTTPBasicCredentials = Depends(verify_admin)):
    """停止批量注册任务。body: {job_id}"""
    job_id = data.get("job_id", "")
    if not job_id:
        raise HTTPException(400, "请提供 job_id")
    if _registrar.stop(job_id):
        return {"ok": True}
    raise HTTPException(404, f"任务 {job_id} 不存在")


@app.get("/api/register/mailcx-domains")
async def register_mailcx_domains(api_key: str = "", creds: HTTPBasicCredentials = Depends(verify_admin)):
    """获取 mail.cx 可用邮箱后缀。优先用配置的 API Key，可用 ?api_key= 覆盖。"""
    key = api_key.strip() or config_manager.get_mailcx_api_key()
    if not key:
        return {"ok": False, "error": "未配置 mail.cx API Key"}
    ok, result = list_mailcx_domains(key)
    if not ok:
        return {"ok": False, "error": result}
    return {"ok": True, "domains": result}


@app.get("/api/register/mailcx-config")
async def register_mailcx_config_get(creds: HTTPBasicCredentials = Depends(verify_admin)):
    """读取 mail.cx 配置（API Key / 邮箱后缀）。"""
    return {
        "api_key": config_manager.get_mailcx_api_key(),
        "domain": config_manager.get_mailcx_domain(),
    }


@app.put("/api/register/mailcx-config")
async def register_mailcx_config_set(data: dict, creds: HTTPBasicCredentials = Depends(verify_admin)):
    """保存 mail.cx 配置。body: {api_key, domain?}"""
    key = (data.get("api_key") or "").strip()
    if not key:
        raise HTTPException(400, "API Key 不能为空")
    config_manager.set_mailcx_api_key(key)
    if data.get("domain"):
        config_manager.set_mailcx_domain(data["domain"])
    return {"ok": True}


@app.get("/api/register/export")
async def register_export(job_id: str = "", format: str = "csv",
                          creds: HTTPBasicCredentials = Depends(verify_admin)):
    """导出批量注册任务的账号密码列表。默认 CSV，?format=txt 导出「邮箱\t密码」纯文本。"""
    from fastapi.responses import Response
    data = _registrar.export_data(job_id) if job_id else None
    if not data:
        raise HTTPException(404, f"任务 {job_id} 不存在")

    fmt = "txt" if format == "txt" else "csv"
    rows = []
    for email in data["emails"]:
        pw = data["passwords"].get(email, "")
        st = data["results"].get(email, {}).get("status", "")
        rows.append((email, pw, st))

    if fmt == "txt":
        body = "\n".join(f"{e}\t{p}" for e, p, _s in rows)
        media = "text/plain; charset=utf-8"
        ext = "txt"
    else:
        body = "\ufeff" + "email,password,status\n" + "\n".join(f"{e},{p},{s}" for e, p, s in rows)
        media = "text/csv; charset=utf-8"
        ext = "csv"
    filename = f"deepseek_accounts_{job_id}.{ext}"
    return Response(content=body, media_type=media,
                    headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.get("/api/accounts/export")
async def accounts_export(creds: HTTPBasicCredentials = Depends(verify_admin)):
    """导出账号池中全部账号（邮箱、密码、状态、Token、登录时间）。CSV。"""
    from fastapi.responses import Response
    accounts = config_manager.get_all_accounts()
    rows = ["email,password,login_type,is_valid,login_time"]
    for a in accounts:
        email = a.get("_email") or a.get("account_label") or ""
        pw = a.get("_password") or ""
        lt = a.get("login_type") or ""
        valid = "true" if a.get("is_valid") else "false"
        t = (a.get("login_time") or "").replace(",", " ")
        rows.append(f"{email},{pw},{lt},{valid},{t}")
    body = "\ufeff" + "\n".join(rows)
    return Response(content=body, media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": 'attachment; filename="deepseek_accounts_pool.csv"'})


# ─── 用量统计 API ─────────────────────────────────────────────

@app.get("/api/usage")
async def usage_stats(creds: HTTPBasicCredentials = Depends(verify_admin)):
    return get_usage()


@app.delete("/api/usage")
async def clear_usage_stats(creds: HTTPBasicCredentials = Depends(verify_admin)):
    clear_usage()
    return {"ok": True}


@app.post("/api/cleanup")
async def manual_cleanup(creds: HTTPBasicCredentials = Depends(verify_admin)):
    """手动触发会话清理。"""
    try:
        cleanup_old_sessions()
        return {"ok": True, "msg": "清理完成"}
    except Exception as e:
        return {"ok": False, "msg": str(e)}


# ── 代理配置 ────────────────────────────────────────


def _get_proxy_dict() -> dict | None:
    """从 ConfigManager 读取代理配置，返回 curl_cffi 兼容格式。
    返回 None 表示未配置代理，应走直连。"""
    url = config_manager.get_proxy()
    if not url:
        return None
    return {"http": url, "https": url}


@app.get("/api/proxy")
async def get_proxy(creds: HTTPBasicCredentials = Depends(verify_admin)):
    """获取当前代理配置"""
    proxy_url = config_manager.get_proxy()
    return {"proxy": proxy_url or ""}


@app.put("/api/proxy")
async def set_proxy(data: dict, creds: HTTPBasicCredentials = Depends(verify_admin)):
    """设置代理地址。传 {"proxy": "http://127.0.0.1:7890"} 或 {"proxy": ""} 清除。"""
    url = data.get("proxy", "").strip()
    config_manager.set_proxy(url)
    return {"ok": True, "proxy": url}


@app.get("/api/passthrough")
async def get_passthrough(creds: HTTPBasicCredentials = Depends(verify_admin)):
    """获取透传模式状态。"""
    return {"passthrough": config_manager.get_passthrough()}


@app.put("/api/passthrough")
async def set_passthrough(data: dict, creds: HTTPBasicCredentials = Depends(verify_admin)):
    """设置透传模式。传 {"passthrough": true/false}"""
    enabled = data.get("passthrough", False)
    config_manager.set_passthrough(enabled)
    return {"ok": True, "passthrough": enabled}


@app.put("/api/admin-password")
async def change_admin_password(data: dict, creds: HTTPBasicCredentials = Depends(verify_admin)):
    """修改管理员密码。传 {"password": "新密码"}"""
    new_pwd = data.get("password", "").strip()
    if not new_pwd:
        raise HTTPException(400, "密码不能为空")
    config_manager.set_admin_password(new_pwd)
    return {"ok": True}


# ─── 模型列表（免鉴权，供管理页面使用） ───────────────────────

@app.get("/api/models")
async def admin_models(creds: HTTPBasicCredentials = Depends(verify_admin)):
    return {"models": list(get_models().keys())}


# ── 模型映射（动态从 DeepSeek 探测）─────────────────
MODEL_CONFIG_URL = "https://chat.deepseek.com/api/v0/client/settings"

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
        "X-Client-Version": "2.4.2",
        "X-Client-Platform": "web",
    }

    try:
        params = {"did": str(uuid.uuid4()), "scope": "model"}
        resp = cffi_requests.get(MODEL_CONFIG_URL, params=params, headers=headers, timeout=10, proxies=_get_proxy_dict())
        data = resp.json()
        biz_data = data.get("data", {})
        if not isinstance(biz_data, dict):
            print(f"[模型发现] 响应格式异常: data={type(biz_data).__name__}")
            return None
        biz_data = biz_data.get("biz_data")
        if not biz_data:
            print(f"[模型发现] biz_data 为空 (biz_code={data.get('data', {}).get('biz_code')} {data.get('data', {}).get('biz_msg', '')})")
            return None
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

    login_payload = {"password": password, "device_id": new_device_id(), "os": "ios"}
    account_label = cfg.get("account_label", "") or cfg.get("account", "")

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

    DS_HEADERS = random_ios_headers()

    try:
        print(f"[Token] 自动重新登录 {account_label}...")

        # 1. 登录（iOS 随机指纹绕过 WAF）
        login_resp = cffi_requests.post(
            "https://chat.deepseek.com/api/v0/users/login",
            json=login_payload,
            headers=DS_HEADERS,
            timeout=30,
            proxies=_get_proxy_dict(),
        )

        # WAF 挑战检测
        if login_resp.status_code == 202 and login_resp.headers.get("x-amzn-waf-action"):
            print(f"[Token] 自动登录被 AWS WAF 拦截 (HTTP 202)")
            return None

        raw_text = (login_resp.text or "").strip()
        if not raw_text:
            print(f"[Token] 自动登录失败: 服务器返回空响应 (HTTP {login_resp.status_code})")
            return None
        try:
            login_data = login_resp.json()
        except Exception:
            print(f"[Token] 自动登录失败: 非 JSON 响应: {raw_text[:200]}")
            return None
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

        # 2. 创建新会话（iOS 随机指纹）
        auth_headers = {**DS_HEADERS, "authorization": f"Bearer {token}"}
        session_resp = cffi_requests.post(
            "https://chat.deepseek.com/api/v0/chat_session/create",
            json={},
            headers=auth_headers,
            timeout=15,
            proxies=_get_proxy_dict(),
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
        # 更新多账号池
        config_manager.update_account(account_label,
            token=token, session_id=session_id,
            headers={**DS_HEADERS, "authorization": f"Bearer {token}"},
            is_valid=True, login_time=time.strftime("%Y-%m-%d %H:%M:%S"))
        return new_cfg

    except Exception as e:
        print(f"[Token] 自动登录异常: {e}")
        return None


def load_config_with_refresh() -> dict:
    """加载配置，如果 token 失效则自动刷新（多账号模式：返回第一个有效账号）"""
    accounts = config_manager.get_all_accounts()
    if not accounts:
        return {}
    first = accounts[0]
    return {
        "token": first.get("token", ""),
        "session_id": first.get("session_id", ""),
        "configured": True,
    }


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


def get_pow_response(target_path: str = "/api/v0/chat/completion",
                      cfg: dict | None = None) -> str | None:
    """Get fresh PoW response from DeepSeek."""
    try:
        if cfg is None:
            account = config_manager.get_next_account()
            if not account:
                return None
            cfg = {
                "token": account.token,
                "session_id": account.session_id,
                "headers": dict(account.headers),
            }
        headers = build_request_headers(cfg, cfg["session_id"])

        resp = cffi_requests.post(
            "https://chat.deepseek.com/api/v0/chat/create_pow_challenge",
            headers=headers,
            json={"target_path": target_path},
            impersonate="chrome120",
            timeout=15,
            proxies=_get_proxy_dict(),
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

def upload_file_to_deepseek(file_data: bytes, filename: str, content_type: str = "image/png",
                             cfg: dict | None = None) -> str | None:
    """Upload a file to DeepSeek and return the file_id.

    Uses the /api/v0/file/upload_file endpoint with PoW authentication.
    Returns file_id string or None on failure.
    """
    if cfg is None:
        account = config_manager.get_next_account()
        if not account:
            _vlog("upload: no account available")
            return None
        cfg = {
            "token": account.token,
            "session_id": account.session_id,
            "headers": dict(account.headers),
        }
    session_id = cfg["session_id"]

    # Get PoW for upload_file scene
    pow_response = get_pow_response(target_path="/api/v0/file/upload_file", cfg=cfg)

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
            proxies=_get_proxy_dict(),
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


# ── 会话清理 ──────────────────────────────────────────

def _delete_deepseek_session(token: str, session_id: str) -> bool:
    """调用 DeepSeek API 删除指定会话。"""
    try:
        headers = {**DS_HEADERS, "authorization": f"Bearer {token}"}
        resp = cffi_requests.post(
            "https://chat.deepseek.com/api/v0/chat_session/delete",
            json={"chat_session_id": session_id},
            headers=headers,
            impersonate="chrome120",
            timeout=15,
            proxies=_get_proxy_dict(),
        )
        if resp.status_code == 200:
            data = resp.json()
            return data.get("data", {}).get("biz_code") == 0
        return False
    except Exception as e:
        print(f"[Cleanup] Delete session {session_id} failed: {e}")
        return False


def cleanup_old_sessions():
    """清理所有账号中过期的旧会话。每次删除后等待 3 秒，避免触发风控。"""
    expired = get_expired_sessions()
    if not expired:
        return

    print(f"[Cleanup] Found {len(expired)} expired sessions, deleting with 3s delay...")
    deleted = 0
    for account_label, session_id, model, days_ago in expired:
        token = config_manager.get_token(account_label)
        if not token:
            continue
        if _delete_deepseek_session(token, session_id):
            remove_old_session(account_label, session_id)
            deleted += 1
            print(f"[Cleanup] Deleted: {session_id[:12]}... ({days_ago}d old)")
        time.sleep(10)
    if deleted:
        print(f"[Cleanup] Done: {deleted}/{len(expired)} deleted")


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
            proxies=_get_proxy_dict(),
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
            proxies=_get_proxy_dict(),
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
            resp = cffi_requests.get(s, timeout=30, impersonate="chrome120", proxies=_get_proxy_dict())
            if resp.status_code == 200:
                ct = resp.headers.get("content-type", "image/png")
                ext = ct.split("/")[-1] if "/" in ct else "png"
                return {"data": resp.content, "content_type": ct, "filename": f"image.{ext}"}
        except Exception as e:
            print(f"[Vision] Failed to download image: {e}")
    return None








@app.post("/v1/chat/completions")
async def chat(request: Request):
    # 多账号：获取下一个可用账号
    account = config_manager.get_next_account()
    if not account:
        raise HTTPException(503, detail="没有可用账号，请先访问 /admin 添加并登录账号")

    body = await request.json()
    messages = body.get("messages", [])
    model = body.get("model", "deepseek-default")
    stream = body.get("stream", False)
    tools = body.get("tools", None)
    passthrough = body.get("passthrough", False) or config_manager.get_passthrough()

    # Log client info for debugging
    ua = request.headers.get("user-agent", "?")[:60]
    msg = f"[REQ] account={account.account_label} model={model} stream={stream} msgs={len(messages)} tools={bool(tools)} ua={ua}"
    print(msg, flush=True)
    _vlog(msg)

    # 模型映射
    models = get_models()
    if not models:
        raise HTTPException(503, detail="Model list is empty — DeepSeek model discovery failed")
    model_info = models.get(model, models.get("deepseek-default"))
    if model_info is None:
        raise HTTPException(404, detail=f"Unknown model '{model}' and no fallback available")
    thinking_enabled, search_enabled, _, _ = model_info

    cfg = {
        "token": account.token,
        "session_id": account.session_id,
        "headers": dict(account.headers),
        "cookie": account.cookie,
        "account_label": account.account_label,
    }
    account_label = account.account_label
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
            orig_fid = upload_file_to_deepseek(tf["data"], tf["filename"], tf["content_type"], cfg=cfg)
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
            orig_fid = upload_file_to_deepseek(img["data"], img["filename"], img["content_type"], cfg=cfg)
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
                    json={}, headers=auth_h, impersonate="chrome120", timeout=15,
                    proxies=_get_proxy_dict())
                if sess_resp.status_code == 200:
                    biz = sess_resp.json().get("data", {}).get("biz_data", {})
                    new_sid = biz.get("chat_session", {}).get("id", "") or biz.get("id", "")
                    if new_sid:
                        old_sid = cfg.get("session_id", "")
                        cfg = dict(cfg)
                        cfg["session_id"] = new_sid
                        if old_sid and old_sid != new_sid:
                            on_new_session(account_label, new_sid, model)
                        _vlog(f"vision fresh session: {new_sid}")
        except Exception as e:
            _vlog(f"fresh session failed: {e}")

    # 构建 prompt：使用 convert_messages_for_deepseek 处理完整多轮对话
    prompt = convert_messages_for_deepseek(messages, tools, passthrough=passthrough)
    prompt_tokens = _count_tokens(prompt)

    # 会话管理：token 超限时自动建新 DeepSeek session
    if needs_renewal(account_label):
        status = get_usage_status(account_label)
        print(f"[Session] {account_label} tokens {status['prompt_tokens']}/{status['threshold']} exceeded, creating new session...")
        try:
            token = cfg.get("token", "")
            if token:
                auth_h = {**cfg.get("headers", {}), "authorization": f"Bearer {token}"}
                sess_resp = cffi_requests.post(
                    "https://chat.deepseek.com/api/v0/chat_session/create",
                    json={}, headers=auth_h, impersonate="chrome120", timeout=15,
                    proxies=_get_proxy_dict())
                if sess_resp.status_code == 200:
                    biz = sess_resp.json().get("data", {}).get("biz_data", {})
                    new_sid = biz.get("chat_session", {}).get("id", "") or biz.get("id", "")
                    if new_sid:
                        cfg = dict(cfg)
                        cfg["session_id"] = new_sid
                        config_manager.update_account(account_label, session_id=new_sid)
                        on_new_session(account_label, new_sid, model)
                        print(f"[Session] {account_label} new session: {new_sid}")
        except Exception as e:
            print(f"[Session] Failed to create new session: {e}")

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
            add_tokens(account_label, cfg.get("session_id", ""), prompt_tokens)
        result.body_iterator = _counted_stream()
    else:
        add_usage(model, prompt_tokens, 0)
        add_tokens(account_label, cfg.get("session_id", ""), prompt_tokens)
    return result


@app.post("/v1/responses")
async def responses(request: Request):
    account = config_manager.get_next_account()
    if not account:
        raise HTTPException(503, detail="没有可用账号，请先访问 /admin 添加并登录账号")

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
                    # 跳过纯空 <think></think> 块（可能含空白字符），
                    # 避免客户端显示无内容的空白思维链
                    stripped = re.sub(r'<think>\s*</think>', '', reasoning_delta, flags=re.DOTALL).strip()
                    if not stripped:
                        continue  # 全空，跳过这个 delta
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
                            "fc_id": f"fc_{uuid.uuid4().hex[:24]}",
                            "name": "",
                            "arguments": "",
                        })
                        fn = tc.get("function", {}) or {}
                        if fn.get("name"):
                            slot["name"] = fn.get("name")
                        if fn.get("arguments"):
                            old_args = slot["arguments"]
                            slot["arguments"] += fn.get("arguments")
                            function_item = {
                                "id": slot["fc_id"],
                                "type": "function_call",
                                "call_id": slot["id"],
                                "name": slot["name"],
                                "arguments": "",
                                "status": "in_progress",
                            }
                            output_index, event = _start_output_item(function_item)
                            if event:
                                yield _sse_json(_event_payload(event))
                            yield _sse_json(_event_payload({
                                "type": "response.function_call_arguments.delta",
                                "item_id": slot["fc_id"],
                                "output_index": output_index,
                                "delta": fn.get("arguments"),
                            }))

                if finish_reason:
                    output_by_id: dict[str, dict] = {}
                    if reasoning_parts:
                        full_reasoning = "".join(reasoning_parts)
                        # 剥除空 <think></think> 块（可能含空白字符），避免客户端显示空白思维链
                        cleaned = re.sub(r'<think>\s*</think>', '', full_reasoning, flags=re.DOTALL).strip()
                        if not cleaned:
                            reasoning_parts.clear()  # 清空，跳过后续 reasoning 相关事件
                        else:
                            reasoning_parts = [cleaned]  # 替换为清理后的文本
                            output_by_id[reasoning_item_id] = _response_reasoning_item(cleaned, reasoning_item_id)
                    if refusal_parts:
                        output_by_id[refusal_item_id] = _response_refusal_item("".join(refusal_parts), refusal_item_id)
                    if text_parts:
                        output_by_id[message_item_id] = _response_text_item(
                            _normalize_structured_output_text("".join(text_parts), _response_text_config(body)),
                            message_item_id,
                        )
                    for idx in sorted(tool_calls.keys()):
                        tc = tool_calls[idx]
                        output_by_id[tc["fc_id"]] = {
                            "id": tc["fc_id"],
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
                            "item_id": tc["fc_id"],
                            "output_index": output_indices.get(tc["fc_id"], 0),
                            "arguments": tc["arguments"] or "{}",
                        }))
                    for idx, item in enumerate(output):
                        # 推理项已有 reasoning_text.done 结束，跳过 output_item.done 避免 RikkaHub 重复创建空白思维链
                        if item.get("type") == "reasoning":
                            continue
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
                full_reasoning = "".join(reasoning_parts)
                cleaned = re.sub(r'<think>\s*</think>', '', full_reasoning, flags=re.DOTALL).strip()
                if not cleaned:
                    reasoning_parts.clear()
                else:
                    reasoning_parts = [cleaned]
                    output_by_id[reasoning_item_id] = _response_reasoning_item(cleaned, reasoning_item_id)
            if refusal_parts:
                output_by_id[refusal_item_id] = _response_refusal_item("".join(refusal_parts), refusal_item_id)
            normalized_text = _normalize_structured_output_text("".join(text_parts), _response_text_config(body)) if text_parts else ""
            output_by_id[message_item_id] = _response_text_item(normalized_text, message_item_id)
            for idx in sorted(tool_calls.keys()):
                tc = tool_calls[idx]
                output_by_id[tc["fc_id"]] = {
                    "id": tc["fc_id"],
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
                    "item_id": tc["fc_id"],
                    "output_index": output_indices.get(tc["fc_id"], 0),
                    "arguments": tc["arguments"] or "{}",
                }))
            for idx, item in enumerate(output):
                if item.get("type") == "reasoning":
                    continue
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
    pow_response = get_pow_response(cfg=cfg)
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
        if ref_file_ids:
            # 专家模式不支持文件上传，自动降级到快速模式
            print(f"[Chat] 专家模式不支持文件上传，自动降级到快速模式 (files={len(ref_file_ids)})")
            req_body["model_type"] = "default"
        else:
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

            # Skip SSE comment lines (: — used for keepalive/comments)
            if line.startswith(":") or line == ":":
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

            # Skip SSE comment lines (: — used for keepalive/comments)
            if line.startswith(":") or line == ":":
                continue

            ds = line[6:] if line.startswith("data: ") else line
            # Skip data: : (SSE empty/commented data events)
            if ds.strip() == ":":
                continue
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
                proxies=_get_proxy_dict(),
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
                else:
                    al = cfg.get("account_label", "") or cfg.get("account", "")
                    if al:
                        config_manager.update_account(al, is_valid=False)
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
                content_buffer = []  # 缓冲 text content 用于 fallback 解析

                for etype, val in _parse_sse(resp):
                    if etype == "content":
                        _full_buf += val
                        for evt in sieve.feed(val):
                            if evt.type == "text":
                                if isinstance(evt.data, str) and evt.data:
                                    chunk = clean_tool_text(sanitize_leaked_output(evt.data))
                                    content_buffer.append(chunk)
                                    # 边流式边缓冲：RikkaHub 需要实时收到 content delta
                                    if not _role_sent:
                                        r = {"id": chat_id, "object": "chat.completion.chunk", "created": created, "model": model,
                                             "choices": [{"index": 0, "delta": {"role": "assistant", "content": None}, "finish_reason": None}]}
                                        yield f'data: {json.dumps(r, ensure_ascii=False)}\n\n'
                                        _role_sent = True
                                    r = {"id": chat_id, "object": "chat.completion.chunk", "created": created, "model": model,
                                         "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}]}
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
                            chunk = clean_tool_text(sanitize_leaked_output(evt.data))
                            content_buffer.append(chunk)
                            # 边流式边缓冲
                            if not _role_sent:
                                r = {"id": chat_id, "object": "chat.completion.chunk", "created": created, "model": model,
                                     "choices": [{"index": 0, "delta": {"role": "assistant", "content": None}, "finish_reason": None}]}
                                yield f'data: {json.dumps(r, ensure_ascii=False)}\n\n'
                                _role_sent = True
                            r = {"id": chat_id, "object": "chat.completion.chunk", "created": created, "model": model,
                                 "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}]}
                            yield f'data: {json.dumps(r, ensure_ascii=False)}\n\n'
                    elif evt.type == "tool_calls":
                        _had_tool_calls = True
                        for chunk in _emit_tool_calls(evt.data, chat_id, created, model):
                            yield chunk

                # Fallback: 筛分没抓到，用全量缓冲重试
                if not _had_tool_calls and _full_buf:
                    tc_result, _ = extract_tool_call(_full_buf, get_tool_names(tools) if tools else [])
                    if tc_result:
                        _had_tool_calls = True
                        for chunk in _emit_tool_calls(tc_result, chat_id, created, model):
                            yield chunk

                if _had_tool_calls:
                    # 有工具调用 → content 已在流中发出（如有），直接 DONE
                    if not _role_sent:
                        r = {"id": chat_id, "object": "chat.completion.chunk", "created": created, "model": model,
                             "choices": [{"index": 0, "delta": {"role": "assistant", "content": None}, "finish_reason": None}]}
                        yield f'data: {json.dumps(r, ensure_ascii=False)}\n\n'
                    yield "data: [DONE]\n\n"
                    return
                elif content_buffer:
                    # 无工具调用，content 已在流中发出，只需发 stop + DONE
                    if not _role_sent:
                        r = {"id": chat_id, "object": "chat.completion.chunk", "created": created, "model": model,
                             "choices": [{"index": 0, "delta": {"role": "assistant", "content": None}, "finish_reason": None}]}
                        yield f'data: {json.dumps(r, ensure_ascii=False)}\n\n'
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
                         "choices": [{"index": 0, "delta": {"content": clean_tool_text(sanitize_leaked_output(val))}, "finish_reason": None}]}
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
                proxies=_get_proxy_dict(),
            )

            if ref_file_ids or thinking_enabled:
                _vlog(f"chat nonstream(stream-internal) response: status={resp.status_code} ct={resp.headers.get('content-type','?')}")

            if resp.status_code == 401 and not is_retry:
                print("[Token] 401 in nonstream, trying refresh...")
                new_cfg = relogin(cfg)
                if new_cfg:
                    return _do_chat(new_cfg, prompt, model, thinking_enabled, search_enabled, False, is_retry=True, has_tools=has_tools, tools=tools, ref_file_ids=ref_file_ids)
                else:
                    al = cfg.get("account_label", "") or cfg.get("account", "")
                    if al:
                        config_manager.update_account(al, is_valid=False)

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
        full_content = sanitize_leaked_output(full_content)
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
            if not final_content:
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
    import os as _anthropic_os
    import uvicorn
    anthropic_init_batch_storage(_anthropic_os.path.join(_anthropic_os.path.dirname(_anthropic_os.path.abspath(__file__)), ".anthropic_batches"))
    print(f" Anthropic: /v1/messages, /v1/messages/count_tokens, /v1/messages/batches, /v1/messages/{{id}}")
    print(f"DeepSeek Proxy\n Admin: http://localhost:{PROXY_PORT}/admin\n API: http://localhost:{PROXY_PORT}/v1")
    uvicorn.run(app, host="0.0.0.0", port=PROXY_PORT, log_level="info")
