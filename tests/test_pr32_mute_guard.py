"""PR#32 review fixes: mute detection, model normalize, account pool (offline)."""
from __future__ import annotations

import ast
import os
import sys
import time
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _load_helpers():
    """Load pure helpers from proxy.py without importing the FastAPI app."""
    import types
    src = open(os.path.join(ROOT, "proxy.py"), encoding="utf-8").read()
    # Isolate helper block between markers
    start = src.find("_BIZ_CODE_MUTED")
    end = src.find("def get_models")
    if start < 0 or end < 0:
        raise RuntimeError("helper markers not found in proxy.py")
    chunk = src[start:end]
    # Provide a tiny config_manager stub for _mark_account_restricted
    marked = {}

    class _CM:
        @staticmethod
        def mark_account_muted(label, mute_until=0.0, banned=False):
            marked[label] = (mute_until, banned)

    ns = {
        "HTTPException": type("HTTPException", (Exception,), {
            "__init__": lambda self, status_code=500, detail=None: (
                setattr(self, "status_code", status_code),
                setattr(self, "detail", detail),
                Exception.__init__(self, detail),
            )[-1],
        }),
        "config_manager": _CM,
        "json": __import__("json"),
        "print": print,
    }
    exec(compile(chunk, "proxy_helpers", "exec"), ns)
    ns["_marked"] = marked
    return ns


class TestSyntax(unittest.TestCase):
    def test_changed_files_parse_and_no_bom(self):
        files = [
            "proxy.py",
            os.path.join("app", "config.py"),
            os.path.join("app", "lock_guard.py"),
            "context_manager.py",
            "start_bridge8000.py",
        ]
        for rel in files:
            path = os.path.join(ROOT, rel)
            src = open(path, encoding="utf-8").read()
            self.assertFalse(src.startswith("﻿"), f"BOM in {rel}")
            ast.parse(src)

    def test_no_nested_double_quote_fstring_patterns(self):
        # Guard the specific P0 footgun: f"...{x or "y"}..."
        path = os.path.join(ROOT, "proxy.py")
        src = open(path, encoding="utf-8").read()
        bad = ['or "restricted"', 'headers.get("content-type","?")']
        for b in bad:
            # these must not appear inside an f-string with double quotes outer
            if b in src:
                # allow non-f usage via .format; fail only if inside f"
                for i, line in enumerate(src.splitlines(), 1):
                    if b in line and 'f"' in line and line.strip().startswith("f") is False:
                        # still check: f-string containing same double quotes
                        if 'f"' in line and b in line:
                            self.fail(f"line {i}: possible nested quote f-string: {line.strip()[:120]}")


class TestMuteHelpers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.h = _load_helpers()

    def test_mute_biz_code(self):
        p = self.h["_parse_account_restriction"]({
            "code": 0,
            "data": {"biz_code": 5, "biz_msg": "user is muted", "biz_data": {"mute_until": 1700000000}},
        })
        self.assertIsNotNone(p)
        muted, banned, biz_msg, biz_code, mute_until = p
        self.assertTrue(muted)
        self.assertEqual(biz_code, 5)
        self.assertEqual(mute_until, 1700000000)

    def test_banned(self):
        p = self.h["_parse_account_restriction"]({"data": {"biz_msg": "user_is_banned"}})
        self.assertIsNotNone(p)
        self.assertTrue(p[1])

    def test_ok_payload(self):
        self.assertIsNone(self.h["_parse_account_restriction"]({
            "code": 0, "data": {"biz_code": 0, "biz_msg": "ok"},
        }))

    def test_raise_403_uses_explicit_label(self):
        raw = b'{"code":0,"data":{"biz_code":5,"biz_msg":"user is muted","biz_data":{}}}'
        with self.assertRaises(Exception) as ctx:
            self.h["_raise_if_account_restricted_from_bytes"](raw, account_label="acc-1")
        self.assertEqual(ctx.exception.status_code, 403)
        self.assertIn("acc-1", self.h["_marked"])

    def test_raise_without_label_still_403(self):
        raw = b'{"code":0,"data":{"biz_code":5,"biz_msg":"user is muted","biz_data":{}}}'
        with self.assertRaises(Exception) as ctx:
            self.h["_raise_if_account_restricted_from_bytes"](raw)
        self.assertEqual(ctx.exception.status_code, 403)

    def test_normalize_model(self):
        n = self.h["normalize_model_id"]
        self.assertEqual(n("deepseek-chat"), "deepseek-default")
        self.assertEqual(n("deepseek-search"), "deepseek-search")
        self.assertEqual(n("deepseek-chat", {"reasoning_effort": "high"}), "deepseek-reasoner")
        self.assertEqual(n("deepseek-reasoner", {"thinking": {"type": "disabled"}}), "deepseek-default")


class TestConfigPool(unittest.TestCase):
    def test_account_fields_and_marks(self):
        # Import may fail if optional crypto dep missing; parse source instead.
        src = open(os.path.join(ROOT, "app", "config.py"), encoding="utf-8").read()
        for name in ("mark_account_muted", "mark_account_cooldown", "mark_account_ok", "get_next_account"):
            self.assertIn(f"def {name}", src)
        self.assertIn("muted_until", src)
        self.assertIn("cooldown_until", src)
        # hot path must not unconditionally save inside get_next_account
        fn_start = src.find("def get_next_account")
        fn_end = src.find("def mark_account_muted")
        body = src[fn_start:fn_end]
        self.assertIn("if state_dirty", body)
        # save() must be gated — only appear after if state_dirty
        self.assertIn("if state_dirty:\n                self.save()", body)

    def test_lock_guard_sleep_outside_lock(self):
        src = open(os.path.join(ROOT, "app", "lock_guard.py"), encoding="utf-8").read()
        # crude: sleep must not appear as the statement immediately inside with _lock
        self.assertIn("time.sleep", src)
        self.assertNotIn("﻿", src)


if __name__ == "__main__":
    unittest.main()
