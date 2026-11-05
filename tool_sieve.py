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


class StreamSieve:
    def __init__(self, parse_fn: Optional[Callable[[str], Tuple[List[Dict], str]]] = None):
        self.parse_fn = parse_fn
        self._parts = []

    def feed(self, chunk: str) -> List[SieveEvent]:
        self._parts.append(chunk)
        return []

    def flush(self) -> List[SieveEvent]:
        text = "".join(self._parts)
        self._parts = []
        if not text:
            return []
        if not self.parse_fn:
            return [SieveEvent("text", text)]
        calls, cleaned = self.parse_fn(text)
        if not calls:
            return [SieveEvent("text", cleaned)] if cleaned else []
        events = [SieveEvent("text", cleaned)] if cleaned else []
        events.append(SieveEvent("tool_calls", calls))
        return events
