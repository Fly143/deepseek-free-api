"""Issue #35 修复验证：会话被风控登出（高风险账号自动登出）时自动重登并重试。

离线测试：mock cffi_requests.post，不访问网络。覆盖：
1. 非流式：SSE 登出错误 → 自动重登（login + 新会话）→ 重试成功 → 账号池/session_store 更新
2. 流式：SSE 登出错误 → 自动重登 → 流式重试成功
3. 重登失败：账号进入冷却（保持 is_valid，自愈）而非永久 invalid，冷却期内被跳过
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import app.config as ac  # noqa: E402
import proxy  # noqa: E402
import session_store  # noqa: E402
import usage_store  # noqa: E402
from app.config import DsAccount  # noqa: E402


class _FakeResp:
    def __init__(self, status_code=200, body=b"", ct="application/json"):
        self.status_code = status_code
        self._body = body
        self.headers = {"content-type": ct}

    @property
    def text(self):
        return self._body.decode("utf-8", "replace")

    def iter_content(self, chunk_size=4096):
        yield self._body

    def json(self):
        return json.loads(self.text)


class _FakeRequest:
    def __init__(self, body):
        self._body = body
        self.headers = {"user-agent": "test"}

    async def json(self):
        return self._body


def _sse_login_error():
    return ('data: {"v":{"type":"error","content":"检测到高风险账号，已自动登出","finish_reason":"error"}}\n\n'
            'data: [DONE]\n\n')


def _sse_ok():
    return 'data: {"p":"response/content","o":"APPEND","v":"ok"}\n\ndata: [DONE]\n\n'


def _make_fake_post(calls, fail_login=False):
    """构造假上游：第一次 chat/completion 返回登出错误，重登/重试按预期响应。"""
    def fake_post(url, **kwargs):
        calls.append(url)
        if url.endswith("/users/login"):
            if fail_login:
                body = json.dumps({"code": 0, "data": {"biz_code": 1001,
                                     "biz_msg": "登录被风控拦截"}})
                return _FakeResp(200, body.encode())
            body = json.dumps({"code": 0, "data": {"biz_code": 0, "biz_msg": "",
                                     "biz_data": {"user": {"token": "T2"}}}})
            return _FakeResp(200, body.encode())
        if url.endswith("/chat_session/create"):
            body = json.dumps({"data": {"biz_data": {"chat_session": {"id": "S2"}}}})
            return _FakeResp(200, body.encode())
        if url.endswith("/chat/completion"):
            n = sum(1 for c in calls if c.endswith("/chat/completion"))
            if n == 1:
                return _FakeResp(200, _sse_login_error().encode(), "text/event-stream")
            return _FakeResp(200, _sse_ok().encode(), "text/event-stream")
        return _FakeResp(404, b"{}")
    return fake_post


def _seed_account(label):
    acc = DsAccount(account_label=label, login_type="email",
                    _password="pw", _email=label,
                    token="OLD", session_id="OLD_SID",
                    headers={}, is_valid=True)
    proxy.config_manager.add_account(acc)
    return acc


class Issue35AutoReloginTest(unittest.TestCase):
    def setUp(self):
        # 隔离所有落盘文件到临时目录
        self.tmp = tempfile.TemporaryDirectory()
        tmpdir = Path(self.tmp.name)
        self._orig_config = ac.CONFIG_FILE
        ac.CONFIG_FILE = tmpdir / "config.json"
        cm = ac.ConfigManager()
        ac.CONFIG_FILE = self._orig_config
        self._orig_cm = proxy.config_manager
        proxy.config_manager = cm
        self._orig_session_file = session_store.SESSION_FILE
        session_store.SESSION_FILE = tmpdir / "sessions.json"
        self._orig_usage_file = usage_store._USAGE_FILE
        usage_store._USAGE_FILE = tmpdir / "usage.json"
        # 预置模型缓存（跳过真实探测）
        self._orig_cache = (proxy._models_cache, proxy._models_cache_time)
        proxy._models_cache = {"deepseek-default": (False, False, 64000, 8000)}
        proxy._models_cache_time = time.time()
        self._orig_post = proxy.cffi_requests.post

    def tearDown(self):
        proxy.cffi_requests.post = self._orig_post
        (proxy._models_cache, proxy._models_cache_time) = self._orig_cache
        proxy.config_manager = self._orig_cm
        session_store.SESSION_FILE = self._orig_session_file
        usage_store._USAGE_FILE = self._orig_usage_file
        self.tmp.cleanup()

    def _chat(self, body):
        return asyncio.run(proxy.chat(_FakeRequest(body)))

    def test_1_nonstream_relogin_retry(self):
        """非流式：SSE 登出错误 → 自动重登 → 重试成功"""
        _seed_account("test@example.com")
        calls = []
        proxy.cffi_requests.post = _make_fake_post(calls)

        result = self._chat({"model": "deepseek-default",
                             "messages": [{"role": "user", "content": "hi"}],
                             "stream": False})
        self.assertIsInstance(result, proxy.JSONResponse)
        self.assertEqual(sum(1 for c in calls if c.endswith("/chat/completion")), 2,
                         f"重试未发生: {calls}")
        self.assertTrue(any(c.endswith("/users/login") for c in calls), "自动重登未发生")
        acc = proxy.config_manager.get_account_by_label("test@example.com")
        self.assertEqual(acc.token, "T2")
        self.assertEqual(acc.session_id, "S2")
        self.assertTrue(acc.is_valid)
        self.assertEqual(session_store.get_current_session_id("test@example.com"), "S2")

    def test_2_stream_relogin_retry(self):
        """流式：SSE 登出错误 → 自动重登 → 流式重试成功"""
        _seed_account("s@example.com")
        calls = []
        proxy.cffi_requests.post = _make_fake_post(calls)

        async def run():
            result = await proxy.chat(_FakeRequest({
                "model": "deepseek-default",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True}))
            chunks = []
            async for c in result.body_iterator:
                chunks.append(c)
            return result, chunks

        result, chunks = asyncio.run(run())
        self.assertIsInstance(result, proxy.StreamingResponse)
        self.assertEqual(sum(1 for c in calls if c.endswith("/chat/completion")), 2,
                         f"重试未发生: {calls}")
        acc = proxy.config_manager.get_account_by_label("s@example.com")
        self.assertEqual(acc.token, "T2")
        self.assertEqual(acc.session_id, "S2")
        self.assertTrue(acc.is_valid)

    def test_3_relogin_failure_cooldown(self):
        """重登失败：账号进入冷却（保持 is_valid）而非永久 invalid"""
        _seed_account("b@example.com")
        calls = []
        proxy.cffi_requests.post = _make_fake_post(calls, fail_login=True)

        with self.assertRaises(proxy.HTTPException):
            self._chat({"model": "deepseek-default",
                        "messages": [{"role": "user", "content": "hi"}],
                        "stream": False})
        acc = proxy.config_manager.get_account_by_label("b@example.com")
        self.assertTrue(acc.is_valid, "重登失败不应把账号永久标记 invalid")
        self.assertGreater(float(acc.cooldown_until or 0), time.time(),
                           "重登失败应进入冷却")
        # 冷却期内 get_next_account 跳过该账号（唯一账号 → None）
        self.assertIsNone(proxy.config_manager.get_next_account())


if __name__ == "__main__":
    unittest.main()
