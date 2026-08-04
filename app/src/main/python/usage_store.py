"""用量统计持久化存储（JSON 文件，按模型分组 + 按日累计）"""

import json
import threading
from pathlib import Path
from datetime import datetime, timezone, timedelta

_USAGE_FILE = Path(__file__).parent / "usage.json"
_LOCK = threading.Lock()


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _load() -> dict:
    if not _USAGE_FILE.exists():
        return _empty_state()
    try:
        with open(_USAGE_FILE, "r") as f:
            data = json.load(f)
        for key in ("models", "total", "daily"):
            if key not in data:
                data[key] = {} if key == "daily" else (_zero_count() if key == "total" else {})
        return data
    except Exception:
        return _empty_state()


def _save(data: dict):
    tmp = _USAGE_FILE.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(_USAGE_FILE)


def _empty_state() -> dict:
    return {"models": {}, "total": _zero_count(), "daily": {}}


def _zero_count() -> dict:
    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "requests": 0}


def add_usage(model: str, prompt_tokens: int, completion_tokens: int):
    """累加一次用量（线程安全）。"""
    with _LOCK:
        data = _load()
        day = _today()

        # 按模型（累计）
        if model not in data["models"]:
            data["models"][model] = _zero_count()
        m = data["models"][model]
        m["prompt_tokens"] += prompt_tokens
        m["completion_tokens"] += completion_tokens
        m["total_tokens"] += prompt_tokens + completion_tokens
        m["requests"] += 1

        # 全部累计
        t = data["total"]
        t["prompt_tokens"] += prompt_tokens
        t["completion_tokens"] += completion_tokens
        t["total_tokens"] += prompt_tokens + completion_tokens
        t["requests"] += 1

        # 按日累计
        if day not in data["daily"]:
            data["daily"][day] = {}
        if model not in data["daily"][day]:
            data["daily"][day][model] = _zero_count()
        dm = data["daily"][day][model]
        dm["prompt_tokens"] += prompt_tokens
        dm["completion_tokens"] += completion_tokens
        dm["total_tokens"] += prompt_tokens + completion_tokens
        dm["requests"] += 1

        _save(data)


def _merge_days(days_data: dict) -> dict:
    """合并多天的模型用量为一个汇总。"""
    merged = {}
    for day, models in days_data.items():
        for model, counts in models.items():
            if model not in merged:
                merged[model] = _zero_count()
            for k in ("prompt_tokens", "completion_tokens", "total_tokens", "requests"):
                merged[model][k] += counts.get(k, 0)
    return merged


def get_usage() -> dict:
    """返回用量统计：今天 / 本周 / 全部，按模型分组。"""
    data = _load()
    today = _today()
    week_start = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%d")

    today_data = data.get("daily", {}).get(today, {})
    week_days = {d: v for d, v in data.get("daily", {}).items() if d >= week_start}

    return {
        "today": {
            "models": today_data,
            "total": _sum_models(today_data),
        },
        "week": {
            "models": _merge_days(week_days),
            "total": _sum_models(_merge_days(week_days)),
        },
        "total": {
            "models": data.get("models", {}),
            "total": data.get("total", _zero_count()),
        },
    }


def _sum_models(models: dict) -> dict:
    total = _zero_count()
    for counts in models.values():
        for k in ("prompt_tokens", "completion_tokens", "total_tokens", "requests"):
            total[k] += counts.get(k, 0)
    return total


def clear_usage():
    """清空全部用量统计数据。"""
    with _LOCK:
        _save(_empty_state())
