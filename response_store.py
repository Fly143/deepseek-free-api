"""Local persistence for Responses API compatibility state."""

import json
import threading
from pathlib import Path

_STORE_FILE = Path(__file__).parent / "responses.json"
_LOCK = threading.Lock()


def _load() -> dict:
    if not _STORE_FILE.exists():
        return {}
    try:
        with open(_STORE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save(data: dict) -> None:
    tmp = _STORE_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(_STORE_FILE)


def save_response_record(record: dict) -> None:
    if not isinstance(record, dict):
        return
    rid = str(record.get("id", "")).strip()
    if not rid:
        return
    with _LOCK:
        data = _load()
        data[rid] = record
        _save(data)


def update_response_record(response_id: str, updater) -> dict | None:
    if not response_id:
        return None
    with _LOCK:
        data = _load()
        record = data.get(response_id)
        if not isinstance(record, dict):
            return None
        updated = updater(dict(record))
        if not isinstance(updated, dict):
            return record
        data[response_id] = updated
        _save(data)
        return updated


def get_response_record(response_id: str) -> dict | None:
    if not response_id:
        return None
    with _LOCK:
        return _load().get(response_id)


def delete_response_record(response_id: str) -> bool:
    if not response_id:
        return False
    with _LOCK:
        data = _load()
        if response_id not in data:
            return False
        del data[response_id]
        _save(data)
        return True
