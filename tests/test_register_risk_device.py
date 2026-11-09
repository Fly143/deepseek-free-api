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
        for rel in ("app/identity.py", "app/device_ids.py", "proxy.py"):
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


if __name__ == '__main__':
    unittest.main()
