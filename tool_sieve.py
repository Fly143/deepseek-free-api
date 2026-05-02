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
    """DSML 模式流式筛分器"""

    _TOOL_STARTS = [
        "<|DSML|tool_calls>",
        "|DSML|tool_calls>",     # 模型有时丢开头的 <
        "<tool_calls>",
        "<tool_call>",
        "<invoke ",
        "<|DSML|invoke ",
        "|DSML|invoke ",         # 模型有时丢开头的 <
    ]

    def __init__(
        self,
        parse_fn: Optional[Callable[[str], Tuple[List[Dict], str]]] = None,
    ):
        self.parse_fn = parse_fn
        self._pending = ""
        self._capture_buf = ""
        self._capturing = False

    def feed(self, chunk: str) -> List[SieveEvent]:
        events: List[SieveEvent] = []

        if self._capturing:
            self._capture_buf += chunk
            result = self._try_finish_capture()
            if result is not None:
                prefix_text, tool_calls, suffix = result
                if prefix_text:
                    events.append(SieveEvent("text", prefix_text))
                if tool_calls:
                    events.append(SieveEvent("tool_calls", tool_calls))
                if suffix:
                    self._pending = suffix
                self._capture_buf = ""
                self._capturing = False
                if suffix:
                    events.extend(self.feed(""))
            return events

        self._pending += chunk
        start_idx = self._find_tool_start(self._pending)

        if start_idx >= 0:
            prefix = self._pending[:start_idx]
            rest = self._pending[start_idx:]
            self._pending = ""

            if prefix:
                events.append(SieveEvent("text", prefix))

            self._capture_buf = rest
            self._capturing = True

            result = self._try_finish_capture()
            if result is not None:
                prefix_text, tool_calls, suffix = result
                if prefix_text:
                    events.append(SieveEvent("text", prefix_text))
                if tool_calls:
                    events.append(SieveEvent("tool_calls", tool_calls))
                if suffix:
                    self._pending = suffix
                self._capture_buf = ""
                self._capturing = False
        else:
            safe, hold = self._split_safe(self._pending)
            if safe:
                events.append(SieveEvent("text", safe))
            self._pending = hold

        return events

    def flush(self) -> List[SieveEvent]:
        events: List[SieveEvent] = []

        if self._capturing:
            result = self._try_finish_capture()
            if result is not None:
                prefix_text, tool_calls, suffix = result
                if prefix_text:
                    events.append(SieveEvent("text", prefix_text))
                if tool_calls:
                    events.append(SieveEvent("tool_calls", tool_calls))
                if suffix:
                    events.append(SieveEvent("text", suffix))
            else:
                from tool_dsml import sanitize_loose_cdata, parse_dsml_tool_calls
                repaired = sanitize_loose_cdata(self._capture_buf)
                tcs, _ = parse_dsml_tool_calls(repaired)
                if tcs:
                    events.append(SieveEvent("tool_calls", tcs))
                elif self._capture_buf:
                    events.append(SieveEvent("text", self._capture_buf))
            self._capture_buf = ""
            self._capturing = False

        if self._pending:
            events.append(SieveEvent("text", self._pending))
            self._pending = ""

        return events

    def _find_tool_start(self, text: str) -> int:
        for tag in self._TOOL_STARTS:
            pos = text.find(tag)
            if pos >= 0:
                return pos
        for prefix in ("<|DSML|", "|DSML|", "<tool_calls", "<tool_call", "<invoke", "|DSML|invoke"):
            pos = text.find(prefix)
            if pos >= 0:
                return pos
        return -1

    def _split_safe(self, text: str) -> Tuple[str, str]:
        # 先找 <，找不到再找 |
        last_lt = text.rfind("<")
        if last_lt == -1:
            last_lt = text.rfind("|")
        if last_lt == -1:
            return text, ""

        tail = text[last_lt:]
        for tag in self._TOOL_STARTS:
            if tag.startswith(tail) or tail == tag[:len(tail)]:
                return text[:last_lt], tail
        for prefix in ("<|DSML|", "|DSML|", "<tool_calls", "<tool_call", "<invoke", "|DSML|invoke"):
            if prefix.startswith(tail) or (len(tail) <= len(prefix) and tail == prefix[:len(tail)]):
                return text[:last_lt], tail
        return text, ""

    def _try_finish_capture(self):
        if not self._capture_buf or not self.parse_fn:
            return None
        if not self._is_capture_complete():
            return None
        tool_calls, cleaned = self.parse_fn(self._capture_buf)
        if tool_calls:
            return ("", tool_calls, "")
        else:
            return (self._capture_buf, None, "")

    def _is_capture_complete(self) -> bool:
        buf = self._capture_buf
        if "<|DSML|tool_calls>" in buf or "<tool_calls>" in buf:
            return "</|DSML|tool_calls>" in buf or "</tool_calls>" in buf
        if "<tool_call>" in buf or "<|DSML|tool_call>" in buf:
            return "</tool_call>" in buf or "</|DSML|tool_call>" in buf
        if "<invoke " in buf or "<|DSML|invoke " in buf:
            return "</invoke>" in buf or "</|DSML|invoke>" in buf
        return False
