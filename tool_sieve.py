"""
StreamSieve (DSML mode) — 流式筛分引擎
逐字符喂入，实时分离正文与 DSML 工具调用。

检测 <|DSML|tool_calls> / <tool_calls> / <invoke 开头的工具调用块。
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class SieveEvent:
    type: str  # 'text' | 'tool_calls'
    data: Any  # str for text, list[dict] for tool_calls


# 安全尾窗：远大于任何工具标记长度，窗前的正文可以安全实时下发
_SAFE_TAIL = 200

_TOOL_HINTS = (
    "<|dsml|",
    "|dsml|",
    "<tool_calls",
    "<tool_call",
    "<invoke",
    "<|begin_of_tool_calls_begin|>",
)


class StreamSieve:
    """流式筛分：尾窗前的正文实时下发；尾窗里出现工具标记就转捕获态，留到 flush 解析。"""

    def __init__(self, parse_fn: Optional[Callable[[str], Tuple[List[Dict], str]]] = None):
        self.parse_fn = parse_fn
        self._pending = ""
        self._capturing = False

    def feed(self, chunk: str) -> List[SieveEvent]:
        self._pending += chunk
        if self._capturing:
            return []
        probe = self._pending.lower().replace("\uff5c", "|").replace("||", "|")
        if any(hint in probe for hint in _TOOL_HINTS):
            self._capturing = True
            return []
        if len(self._pending) <= _SAFE_TAIL:
            return []
        emit = self._pending[:-_SAFE_TAIL]
        newline = emit.rfind("\n")
        if newline < 0:
            return []  # 还没到行边界，继续攒
        emit = emit[: newline + 1]
        if not emit.strip():
            return []
        self._pending = self._pending[len(emit):]
        return [SieveEvent("text", emit)]

    def flush(self) -> List[SieveEvent]:
        pending, self._pending = self._pending, ""
        self._capturing = False
        if not pending:
            return []
        if not self.parse_fn:
            return [SieveEvent("text", pending)]
        calls, cleaned = self.parse_fn(pending)
        events: List[SieveEvent] = []
        if cleaned:
            events.append(SieveEvent("text", cleaned))
        if calls:
            events.append(SieveEvent("tool_calls", calls))
        return events
