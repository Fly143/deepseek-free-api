from .anthropic import error_response, convert_request
_message_store: dict[str, dict] = {}
_MAX_STORED_MESSAGES = 1000


def store_message(msg_id: str, response: dict) -> None:
    """存储消息响应供后续通过 ID 查询。"""
    global _message_store
    _message_store[msg_id] = response
    # 防止内存泄漏：超过上限时清理最旧的
    if len(_message_store) > _MAX_STORED_MESSAGES:
        for k in list(_message_store.keys())[:-_MAX_STORED_MESSAGES]:
            _message_store.pop(k, None)


def get_message(msg_id: str) -> dict | None:
    """获取已存储的消息。"""
    return _message_store.get(msg_id)


def get_message_or_error(msg_id: str) -> dict:
    """获取消息，不存在时返回 404 格式错误。"""
    msg = get_message(msg_id)
    if msg is None:
        return error_response(f"Message {msg_id} not found", "not_found_error")
    return msg


def clear_messages() -> None:
    """清空消息存储（测试用）。"""
    _message_store.clear()


# ═══════════════════════════════════════════════════════════════════════════
# Count Tokens（POST /v1/messages/count_tokens）
# ═══════════════════════════════════════════════════════════════════════════

def count_tokens(body: dict, encoder) -> dict:
    """计算 Anthropic Messages 请求的 token 数。

    使用传入的 tiktoken 编码器（cl100k_base）近似计算。
    Anthropic 返回格式：{"input_tokens": N, "output_tokens": 0}
    """
    openai_body = convert_request(body)
    text = ""
    for msg in openai_body.get("messages", []):
        content = msg.get("content", "")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    text += block.get("text", "") or block.get("content", "") or ""
        elif isinstance(content, str):
            text += content
        text += "\n"
    system = body.get("system", "")
    if system:
        text = system + "\n" + text
    tokens = len(encoder.encode(text))
    return {"input_tokens": tokens, "output_tokens": 0}


# ═══════════════════════════════════════════════════════════════════════════
# Legacy Completions（POST /v1/complete）
# ═══════════════════════════════════════════════════════════════════════════


import os as _os
import time as _time
import asyncio as _asyncio

_BATCH_STORAGE_DIR = None  # 由 init_batch_storage 设置


def init_batch_storage(storage_dir: str) -> None:
    """初始化批量存储目录。在应用启动时调用。"""
    global _BATCH_STORAGE_DIR
    _BATCH_STORAGE_DIR = storage_dir
    _os.makedirs(storage_dir, exist_ok=True)


def _batch_path(batch_id: str = None) -> str:
    """获取批存储文件路径。"""
    d = _BATCH_STORAGE_DIR or "/tmp/anthropic_batches"
    if batch_id:
        return _os.path.join(d, f"{batch_id}.json")
    return d


def _results_path(batch_id: str) -> str:
    """获取批处理结果 JSONL 文件路径。"""
    d = _BATCH_STORAGE_DIR or "/tmp/anthropic_batches"
    return _os.path.join(d, f"{batch_id}_results.jsonl")


def _load_all_batches() -> dict[str, dict]:
    """从磁盘加载所有批处理。"""
    batches = {}
    d = _batch_path()
    if not _os.path.isdir(d):
        return batches
    for fn in _os.listdir(d):
        if fn.endswith(".json") and not fn.endswith("_results.jsonl"):
            try:
                with open(_os.path.join(d, fn)) as f:
                    b = json.loads(f.read())
                    batches[b["id"]] = b
            except Exception:
                pass
    return batches


def _save_batch(batch: dict) -> None:
    """保存单个批处理到磁盘。"""
    fp = _batch_path(batch["id"])
    _os.makedirs(_os.path.dirname(fp), exist_ok=True)
    with open(fp, "w") as f:
        f.write(json.dumps(batch, ensure_ascii=False, default=str))


def _delete_batch_file(batch_id: str) -> None:
    """删除批处理文件。"""
    fp = _batch_path(batch_id)
    if _os.path.exists(fp):
        _os.remove(fp)


def create_batch(requests_data: list, model: str = "deepseek-default") -> dict:
    """创建一个新的批处理。"""
    now = _time.time()
    batch_id = f"batch_msg_{uuid.uuid4().hex[:24]}"
    batch = {
        "id": batch_id,
        "type": "message_batch",
        "processing_status": "in_progress",
        "request_counts": {
            "processing": len(requests_data),
            "succeeded": 0,
            "errored": 0,
            "canceled": 0,
            "expired": 0,
        },
        "created_at": now,
        "expires_at": now + 86400 * 7,  # 7 days
        "cancel_initiated_at": None,
        "ended_at": None,
        "model": model,
        "results_url": None,
        "requests": requests_data,
    }
    _save_batch(batch)
    return batch


def get_batch(batch_id: str) -> dict | None:
    """获取批处理详情。"""
    batches = _load_all_batches()
    return batches.get(batch_id)


def get_batch_or_error(batch_id: str) -> dict:
    """获取批处理，不存在时返回错误。"""
    b = get_batch(batch_id)
    if b is None:
        return error_response(f"Batch {batch_id} not found", "not_found_error")
    return b


def list_batches(status: str = None, limit: int = 20, after_id: str = None) -> dict:
    """列出批处理。"""
    batches = _load_all_batches()
    items = list(batches.values())
    # 按创建时间降序
    items.sort(key=lambda x: x.get("created_at", 0), reverse=True)
    if status:
        items = [b for b in items if b.get("processing_status") == status]
    if after_id:
        idx = next((i for i, b in enumerate(items) if b["id"] == after_id), -1)
        if idx != -1:
            items = items[idx + 1:]
    has_more = len(items) > limit
    items = items[:limit]
    # 清理 requests 字段（列表时不应返回请求体）
    public_items = []
    for b in items:
        pb = {k: v for k, v in b.items() if k != "requests"}
        public_items.append(pb)
    return {
        "data": public_items,
        "has_more": has_more,
        "first_id": items[0]["id"] if items else None,
        "last_id": items[-1]["id"] if items else None,
    }


def cancel_batch(batch_id: str) -> dict | None:
    """取消批处理。"""
    batch = get_batch(batch_id)
    if batch is None:
        return None
    if batch.get("processing_status") in ("ended", "canceled"):
        return batch
    batch["processing_status"] = "canceled"
    batch["cancel_initiated_at"] = _time.time()
    batch["ended_at"] = _time.time()
    _save_batch(batch)
    return batch


def add_batch_result(batch_id: str, result: dict) -> None:
    """向批处理结果文件追加一行结果。"""
    rf = _results_path(batch_id)
    _os.makedirs(_os.path.dirname(rf), exist_ok=True)
    with open(rf, "a") as f:
        f.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
    # 更新计数
    batch = get_batch(batch_id)
    if batch:
        custom_id = result.get("custom_id", "")
        is_success = "result" in result
        is_error = "error" in result
        rc = batch["request_counts"]
        if is_success:
            rc["succeeded"] = rc.get("succeeded", 0) + 1
        elif is_error:
            rc["errored"] = rc.get("errored", 0) + 1
        rc["processing"] = max(0, rc.get("processing", 1) - 1)
        _save_batch(batch)


def finalize_batch(batch_id: str) -> None:
    """结束批处理。"""
    batch = get_batch(batch_id)
    if batch is None:
        return
    if batch.get("processing_status") == "canceled":
        return
    # 将剩余 processing 的标记为 errored（未处理完）
    rc = batch["request_counts"]
    if rc.get("processing", 0) > 0:
        rc["errored"] = rc.get("errored", 0) + rc.get("processing", 0)
        rc["processing"] = 0
    batch["processing_status"] = "ended"
    batch["ended_at"] = _time.time()
    batch["results_url"] = f"/v1/messages/batches/{batch_id}/results"
    # 清理 requests 以节省空间
    batch.pop("requests", None)
    _save_batch(batch)


def get_batch_results(batch_id: str) -> list[dict] | dict | None:
    """获取批处理结果。"""
    rf = _results_path(batch_id)
    if not _os.path.exists(rf):
        return None
    results = []
    with open(rf) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    results.append({"error": "invalid result line"})
    return results


def delete_batch(batch_id: str, results_dir: str = None) -> bool:
    """删除批处理及其结果文件。"""
    _delete_batch_file(batch_id)
    rf = _results_path(batch_id)
    if _os.path.exists(rf):
        _os.remove(rf)
    return True


async def process_batch_requests(
    batch_id: str,
    process_fn,
    batch_store_dir: str = None,
) -> None:
    """后台批处理协程。

    Args:
        batch_id: 批处理 ID
        process_fn: 异步处理函数，接受 (request_dict) 并返回响应 dict
        batch_store_dir: 批存储目录
    """
    batch = get_batch(batch_id)
    if batch is None:
        return
    requests_list = batch.get("requests", [])
    for req in requests_list:
        # 检查是否已取消
        current = get_batch(batch_id)
        if current and current.get("processing_status") == "canceled":
            return
        try:
            resp = await process_fn(req)
            result = {"custom_id": req.get("custom_id", ""), "result": resp}
        except Exception as e:
            result = {
                "custom_id": req.get("custom_id", ""),
                "error": {"type": "api_error", "message": str(e)[:500]},
            }
        add_batch_result(batch_id, result)
    finalize_batch(batch_id)
