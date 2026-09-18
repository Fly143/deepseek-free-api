"""Offline tests for issue #31 register risk-device handling."""
from __future__ import annotations

import ast
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


class TestSyntax(unittest.TestCase):
    def test_changed_files_parse(self):
        for rel in ("app/registrar.py", "app/device_ids.py", "proxy.py"):
            path = os.path.join(ROOT, rel)
            src = open(path, encoding="utf-8").read()
            self.assertFalse(src.startswith("﻿"), rel)
            ast.parse(src)


class TestDeviceIds(unittest.TestCase):
    def setUp(self):
        import app.device_ids as d
        self.d = d
        d.clear_burned()

    def tearDown(self):
        self.d.clear_burned()

    def test_get_skips_burned(self):
        a = self.d.get_device_id()
        self.d.mark_device_burned(a)
        # many draws should never return burned id while pool has others
        for _ in range(50):
            b = self.d.get_device_id()
            self.assertNotEqual(b, a)
            if b in self.d.DEVICE_IDS:
                break

    def test_pool_exhausted_falls_back(self):
        pool = list(self.d.DEVICE_IDS)
        for did in pool:
            self.d.mark_device_burned(did)
        got = self.d.get_device_id()
        self.assertNotIn(got, pool)
        self.assertTrue(len(got) >= 16)

    def test_load_extra_device_ids(self):
        path = os.path.join(ROOT, "tests", "_tmp_device_ids.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("# comment\n\nBxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx==\n")
        before = len(self.d.DEVICE_IDS)
        added = self.d.load_extra_device_ids(path)
        self.assertEqual(added, 1)
        self.assertEqual(len(self.d.DEVICE_IDS), before + 1)
        os.remove(path)


def _load_registrar_helpers():
    """Load pure helpers from registrar.py without importing curl_cffi/cryptography."""
    src = open(os.path.join(ROOT, "app", "registrar.py"), encoding="utf-8").read()
    # Slice from BIZ_CODE constants through _is_risk_device_error
    start = src.find("BIZ_CODE_OK")
    end = src.find("# ── 注册流程 API")
    if start < 0 or end < 0:
        raise RuntimeError("registrar helper markers not found")
    chunk = src[start:end]
    # Also grab _parse_auth_biz/_biz_error if they sit before markers
    pre = src.find("def _parse_auth_biz")
    if pre >= 0 and pre < start:
        # include from _parse_auth_biz if present earlier - actually they're after new_device_id
        pass
    # _parse_auth_biz is before BIZ_CODE_OK in current file order? check
    # Order in file: new_device_id, then BIZ constants, then _parse_auth_biz may be after constants
    # Current layout after edit:
    #   _ensure_extra, new_device_id, BIZ_CODE_*, _parse_auth_biz, _is_risk_device_error
    # So chunk from BIZ_CODE_OK to 注册流程 covers constants + parse + is_risk
    ns = {"Tuple": tuple, "Any": object}
    exec(compile(chunk, "registrar_helpers", "exec"), ns)
    return ns


class TestRegistrarBizParse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.h = _load_registrar_helpers()

    def test_parse_auth_biz_risk_device(self):
        data = {
            "code": 0,
            "msg": "",
            "data": {
                "biz_code": 11,
                "biz_msg": "RISK_DEVICE_DETECTED",
                "biz_data": None,
            },
        }
        code, msg, biz = self.h["_parse_auth_biz"](data)
        self.assertEqual(code, 11)
        self.assertEqual(msg, "RISK_DEVICE_DETECTED")
        self.assertIsNone(biz)
        self.assertTrue(self.h["_is_risk_device_error"]({"biz_code": code, "error": msg}))
        self.assertTrue(self.h["_is_risk_device_error"]({"error": "… RISK_DEVICE_DETECTED …"}))
        self.assertFalse(self.h["_is_risk_device_error"]({"biz_code": 0, "error": "ok"}))
        self.assertFalse(self.h["_is_risk_device_error"]({"biz_code": 6, "error": "REGISTER_FROM_MAINLAND"}))

    def test_new_device_id_from_device_ids_module(self):
        import app.device_ids as d
        d.clear_burned()
        # new_device_id logic: prefer pool via get_device_id
        did = d.get_device_id()
        self.assertTrue(did and isinstance(did, str))

    def test_register_error_message_shape(self):
        data = {
            "code": 0,
            "msg": "",
            "data": {"biz_code": 11, "biz_msg": "RISK_DEVICE_DETECTED", "biz_data": None},
        }
        biz_code, biz_msg, biz_data = self.h["_parse_auth_biz"](data)
        self.assertEqual(biz_code, self.h["BIZ_CODE_RISK_DEVICE"])
        self.assertNotIn(biz_code, (None, self.h["BIZ_CODE_OK"]))
        err = "注册被拒 biz_code={}: {}".format(biz_code, biz_msg or "unknown")
        self.assertIn("RISK_DEVICE_DETECTED", err)
        self.assertIn("11", err)
        self.assertNotIn("注册响应中无 token", err)

    def test_source_has_ios_send_and_risk_retry(self):
        src = open(os.path.join(ROOT, "app", "registrar.py"), encoding="utf-8").read()
        self.assertIn("ios: bool = True", src)
        self.assertIn("_mark_device_burned", src)
        self.assertIn("_is_risk_device_error", src)
        self.assertIn("注册被拒 biz_code=", src)
        # send-code uses ios fingerprint to match register
        self.assertIn("resp = _post(\"/create_email_verification_code\", payload, proxy=proxy, ios=ios)", src)


if __name__ == "__main__":
    unittest.main()
