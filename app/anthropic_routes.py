"""
Anthropic Messages API 路由处理。

依赖 proxy.py 的全局函数（_do_chat、get_models 等），通过局部导入避免循环依赖。
"""
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
import json, uuid, asyncio, re
from .anthropic import (
    convert_request as _anthropic_convert_request,
    convert_response as _anthropic_convert_response,
    stream_response as _anthropic_stream_response,
    error_response as _anthropic_error_response,
)
from .batch import (
    count_tokens as _anthropic_count_tokens,
    store_message as _anthropic_store_message,
    get_message as _anthropic_get_message,
    get_message_or_error as _anthropic_get_message_or_error,
    create_batch as _anthropic_create_batch,
    get_batch as _anthropic_get_batch,
    get_batch_or_error as _anthropic_get_batch_or_error,
    list_batches as _anthropic_list_batches,
    cancel_batch as _anthropic_cancel_batch,
    get_batch_results as _anthropic_get_batch_results,
    delete_batch as _anthropic_delete_batch,
    process_batch_requests as _anthropic_process_batch_requests,
)

router = APIRouter()

# ── Anthropic 模型名 → DeepSeek 内部模型名映射 ──
# Claude Code CLI 等工具期望 Anthropic 风格的模型名（如 claude-sonnet-4-6），
# 无法直接使用 deepseek-* 原生名。此映射表在请求时自动转换。
ANTHROPIC_MODEL_ALIASES = {
    # Claude 4.7
    "claude-opus-4-7": "deepseek-expert-reasoner",
    "claude-sonnet-4-7": "deepseek-reasoner",
    "claude-haiku-4-7": "deepseek-default",
    # Claude 4.6
    "claude-opus-4-6": "deepseek-expert-reasoner",
    "claude-sonnet-4-6": "deepseek-reasoner",
    "claude-haiku-4-6": "deepseek-default",
    # Claude 4.5
    "claude-opus-4-5": "deepseek-expert-reasoner",
    "claude-sonnet-4-5": "deepseek-reasoner",
    "claude-haiku-4-5": "deepseek-default",
    # Claude 4.x 历史
    "claude-opus-4-1": "deepseek-expert-reasoner",
    "claude-opus-4-0": "deepseek-expert-reasoner",
    "claude-sonnet-4-0": "deepseek-reasoner",
    # Claude 3.x
    "claude-3-7-sonnet": "deepseek-reasoner",
    "claude-3-5-sonnet": "deepseek-default",
    "claude-3-opus": "deepseek-expert-reasoner",
    "claude-3-sonnet": "deepseek-default",
    "claude-3-haiku": "deepseek-default",
    # Search 变体
    "claude-opus-4-7-search": "deepseek-expert-reasoner-search",
    "claude-opus-4-6-search": "deepseek-expert-reasoner-search",
    "claude-sonnet-4-7-search": "deepseek-reasoner-search",
    "claude-sonnet-4-6-search": "deepseek-reasoner-search",
    # No-thinking 变体
    "claude-sonnet-4-7-nothinking": "deepseek-default",
    "claude-sonnet-4-6-nothinking": "deepseek-default",
    "claude-haiku-4-5-nothinking": "deepseek-default",
    # Thinking 变体
    "claude-sonnet-4-7-thinking": "deepseek-reasoner",
    "claude-opus-4-7-thinking": "deepseek-expert-reasoner",
}


def _resolve_anthropic_model(model: str) -> str:
    """将 Anthropic 风格模型名映射为 DeepSeek 内部模型名。

    - 已是 deepseek-* → 原样返回
    - 表内精确匹配 → 对应型号
    - 带日期后缀（claude-sonnet-4-6-20250929）→ 去掉日期再匹配
    - 去掉 -latest / @latest
    - 未知 claude-* 启发式：含 opus → expert-reasoner，其余 → reasoner
    """
    if not model:
        return model
    m = model.lower().strip()
    if m.startswith("deepseek-"):
        return m

    if m in ANTHROPIC_MODEL_ALIASES:
        return ANTHROPIC_MODEL_ALIASES[m]

    # 去掉 -YYYYMMDD / -YYYY-MM-DD 日期后缀
    base = re.sub(r"-\d{8}$", "", m)
    base = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", base)
    if base in ANTHROPIC_MODEL_ALIASES:
        return ANTHROPIC_MODEL_ALIASES[base]

    # 去掉 -latest / @latest
    base = re.sub(r"[-@]latest$", "", base)
    if base in ANTHROPIC_MODEL_ALIASES:
        return ANTHROPIC_MODEL_ALIASES[base]

    # 未知 Claude 名：启发式
    if m.startswith("claude-"):
        return "deepseek-expert-reasoner" if "opus" in m else "deepseek-reasoner"

    return model


@router.post("/v1/messages")
async def anthropic_messages(request: Request):
    """Anthropic Messages API 兼容端点（main 分支，支持工具调用+多账号+多模态）"""
    from proxy import (
        _count_tokens, convert_messages_for_deepseek,
        get_models, create_request_session, select_tools,
        add_usage, _do_chat, JSONResponse, run_chat_with_slot,
        config_manager, extract_text_files_from_messages, extract_images_from_messages,
        upload_file_to_deepseek, fork_file_to_vision, wait_for_file_parsing,
    )
    body = await request.json()
    model = body.get("model", "deepseek-default")
    model = _resolve_anthropic_model(model)  # Anthropic 别名映射

    # 多账号轮询
    account = config_manager.get_next_account()
    if not account:
        raise HTTPException(status_code=503, detail=_anthropic_error_response("No available account, please visit /admin to add and login"))

    openai_body = _anthropic_convert_request(body)
    messages = openai_body.get("messages", [])
    tools, tool_required = select_tools(openai_body.get("tools"), body.get("tool_choice"))
    stream = openai_body.get("stream", False)
    has_tools = bool(tools)

    model_info = get_models().get(model, get_models().get("deepseek-default"))
    if not model_info:
        raise HTTPException(status_code=400, detail=_anthropic_error_response(f"Unknown model: {model}"))
    thinking_enabled, search_enabled, _, _ = model_info

    cfg = {
        "token": account.token,
        "session_id": account.session_id,
        "headers": dict(account.headers),
        "cookie": account.cookie,
        "account_label": account.account_label,
        # 凭证必须带入请求路径：会话被风控登出时自动重登（relogin）依赖这些字段
        "login_type": account.login_type,
        "_password": account._password,
        "_email": account._email,
        "_mobile": account._mobile,
        "_area_code": account._area_code,
    }
    cfg = await asyncio.to_thread(create_request_session, cfg)
    ref_file_ids = []

    # 文本文件上传
    text_files = extract_text_files_from_messages(messages)
    if text_files:
        raw_ids = []
        for tf in text_files:
            orig_fid = upload_file_to_deepseek(tf["data"], tf["filename"], tf["content_type"], cfg=cfg)
            if orig_fid:
                raw_ids.append(orig_fid)
        if raw_ids:
            text_ids = wait_for_file_parsing(cfg, raw_ids, timeout=30)
            ref_file_ids.extend(text_ids)

    # 多模态：提取、上传、fork 图片
    is_vision = "vision" in model
    if is_vision:
        images = extract_images_from_messages(messages)
        for img in images:
            orig_fid = upload_file_to_deepseek(img["data"], img["filename"], img["content_type"], cfg=cfg)
            if orig_fid:
                forked_fid = fork_file_to_vision(cfg, orig_fid)
                if forked_fid:
                    ref_file_ids.append(forked_fid)
        if ref_file_ids:
            ref_file_ids = wait_for_file_parsing(cfg, ref_file_ids, timeout=10)
    prompt = convert_messages_for_deepseek(messages, tools, tool_required=tool_required)
    prompt_tokens = _count_tokens(prompt)

    msg_id = f"msg_{uuid.uuid4().hex[:24]}"

    if stream:
        result = _do_chat(cfg, prompt, model, thinking_enabled, search_enabled, stream=True, is_retry=False, has_tools=has_tools, tools=tools, ref_file_ids=ref_file_ids, tool_required=tool_required)

        async def _wrap():
            orig_iter = result.body_iterator
            async def _gen():
                async for chunk in orig_iter:
                    yield chunk
            async for event in _anthropic_stream_response(_gen(), model, msg_id):
                yield event
            add_usage(model, prompt_tokens, 0)

        return StreamingResponse(_wrap(), media_type="text/event-stream", headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"})
    else:
        result = await asyncio.to_thread(run_chat_with_slot, cfg, prompt, model, thinking_enabled, search_enabled, stream=False, is_retry=False, has_tools=has_tools, tools=tools, ref_file_ids=ref_file_ids, tool_required=tool_required)
        add_usage(model, prompt_tokens, 0)

        if isinstance(result, JSONResponse):
            openai_body_resp = json.loads(result.body)
        elif isinstance(result, dict):
            openai_body_resp = result
        else:
            raise HTTPException(status_code=500, detail=_anthropic_error_response("Internal error"))

        return _anthropic_convert_response(openai_body_resp, model, msg_id)


@router.post("/v1/messages/count_tokens")
async def anthropic_count_tokens_ep(request: Request):
    body = await request.json()
    return _anthropic_count_tokens(body, _get_encoder())


@router.post("/v1/messages/batches")
async def anthropic_create_batch_ep(request: Request):
    from proxy import _SyntheticRequest
    body = await request.json()
    requests_data = body.get("requests", [])
    model = body.get("model", "deepseek-default")
    batch = _anthropic_create_batch(requests_data, model)

    async def _process_one(req):
        params = dict(req.get("params") or req.get("body") or {})
        params.setdefault("model", model)
        params["stream"] = False
        return await anthropic_messages(_SyntheticRequest(request, params))

    asyncio.create_task(_anthropic_process_batch_requests(batch["id"], _process_one))
    return batch


@router.get("/v1/messages/batches")
async def anthropic_list_batches_ep(status: str = None, limit: int = 20, after_id: str = None):
    return _anthropic_list_batches(status, min(limit, 100), after_id)


@router.get("/v1/messages/batches/{batch_id}")
async def anthropic_get_batch_ep(batch_id: str):
    b = _anthropic_get_batch(batch_id)
    if b is None:
        raise HTTPException(status_code=404, detail=_anthropic_error_response(f"Batch {batch_id} not found", "not_found_error"))
    return b


@router.post("/v1/messages/batches/{batch_id}/cancel")
async def anthropic_cancel_batch_ep(batch_id: str):
    b = _anthropic_cancel_batch(batch_id)
    if b is None:
        raise HTTPException(status_code=404, detail=_anthropic_error_response(f"Batch {batch_id} not found", "not_found_error"))
    return b


@router.get("/v1/messages/batches/{batch_id}/results")
async def anthropic_batch_results_ep(batch_id: str):
    results = _anthropic_get_batch_results(batch_id)
    if results is None:
        raise HTTPException(status_code=404, detail=_anthropic_error_response(f"Results for batch {batch_id} not found", "not_found_error"))
    return StreamingResponse(
        iter([json.dumps(r, ensure_ascii=False) + "\n" for r in results]),
        media_type="application/jsonl",
        headers={"Content-Disposition": f"attachment; filename={batch_id}_results.jsonl"})


@router.delete("/v1/messages/batches/{batch_id}")
async def anthropic_delete_batch_ep(batch_id: str):
    _anthropic_delete_batch(batch_id)
    return {"id": batch_id, "type": "message_batch_deleted", "object": "message_batch"}


@router.get("/v1/messages/{message_id}")
async def anthropic_get_msg_ep(message_id: str):
    msg = _anthropic_get_message(message_id)
    if msg is None:
        raise HTTPException(status_code=404, detail=_anthropic_error_response(f"Message {message_id} not found", "not_found_error"))
    return msg


def _get_encoder():
    """延迟获取 tiktoken 编码器以避免循环导入问题。"""
    from proxy import _enc
    return _enc
