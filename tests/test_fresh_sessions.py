import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.responses import JSONResponse

import app.config as config
import app.lock_guard as guard
import context_compressor
import proxy


class Request:
    headers = {}

    def __init__(self, body):
        self.body = body

    async def json(self):
        return self.body


class Response:
    status_code = 200

    def __init__(self, session_id):
        self.session_id = session_id

    def json(self):
        return {"code": 0, "data": {"biz_data": {"chat_session": {"id": self.session_id}}}}


class FreshSessionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        with patch.object(config, "CONFIG_FILE", Path(self.tmp.name) / "config.json"), patch.object(config, "LEGACY_FILE", Path(self.tmp.name) / "token.json"):
            self.manager = config.ConfigManager()
        self.manager.add_account(config.DsAccount(account_label="test", login_type="email", token="token", session_id="old", is_valid=True))
        self.patches = [
            patch.object(proxy, "config_manager", self.manager),
            patch.object(guard, "MIN_INTERVAL_SEC", 0),
            patch.object(guard, "_slots", threading.Semaphore(1)),
            patch.object(proxy, "get_models", return_value={"deepseek-default": (False, False, 64000, 8000)}),
            patch.object(proxy, "add_usage"),
            patch.object(context_compressor, "should_compress", return_value=False),
        ]
        for item in self.patches:
            item.start()

    async def asyncTearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.tmp.cleanup()

    async def test_each_request_uses_new_session_and_full_history(self):
        created = iter(["new-1", "new-2"])
        sent = []
        uploads = []
        body = {"messages": [{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "reply"}, {"role": "user", "content": "later"}]}

        def post(url, **kwargs):
            self.assertTrue(url.endswith("/chat_session/create"))
            self.assertEqual(kwargs["headers"]["authorization"], "Bearer token")
            return Response(next(created))

        def upload(data, filename, content_type, cfg):
            uploads.append(cfg["session_id"])
            return "file"

        def chat(cfg, prompt, *args, **kwargs):
            sent.append((cfg["session_id"], prompt, kwargs["ref_file_ids"]))
            return JSONResponse({"choices": [{"message": {"content": "ok"}}]})

        with patch.object(proxy.cffi_requests, "post", side_effect=post), patch.object(proxy, "extract_text_files_from_messages", return_value=[{"data": b"x", "filename": "x.txt", "content_type": "text/plain"}]), patch.object(proxy, "upload_file_to_deepseek", side_effect=upload), patch.object(proxy, "wait_for_file_parsing", return_value=["file"]), patch.object(proxy, "_do_chat", side_effect=chat), patch.object(proxy, "upstream_slot"):
            await proxy.chat(Request(body))
            await proxy.chat(Request(body))
        self.assertEqual(uploads, ["new-1", "new-2"])
        self.assertEqual([x[0] for x in sent], uploads)
        self.assertEqual([x[2] for x in sent], [["file"], ["file"]])
        for _, prompt, _ in sent:
            self.assertIn("earlier", prompt)
            self.assertIn("reply", prompt)
            self.assertIn("later", prompt)
        self.assertEqual(self.manager.get_account_by_label("test").session_id, "old")
        self.assertEqual(body["messages"][0]["content"], "earlier")

    async def test_create_failure_never_uses_old_session(self):
        with patch.object(proxy.cffi_requests, "post", side_effect=RuntimeError("offline")), patch.object(proxy, "_do_chat") as chat, patch.object(proxy, "upload_file_to_deepseek") as upload:
            with self.assertRaises(HTTPException) as error:
                await proxy.chat(Request({"messages": [{"role": "user", "content": "hi"}]}))
        self.assertEqual(error.exception.status_code, 502)
        chat.assert_not_called()
        upload.assert_not_called()
        self.assertEqual(self.manager.get_account_by_label("test").session_id, "old")

    async def test_summary_uses_separate_session(self):
        ids = iter(["main", "summary"])
        sent = []

        async def compress(messages, model, summarize):
            await summarize("summary prompt", model)
            return 1, messages

        def chat(cfg, prompt, *args, **kwargs):
            sent.append((cfg["session_id"], prompt))
            return JSONResponse({"choices": [{"message": {"content": "ok"}}]})

        with patch.object(proxy.cffi_requests, "post", side_effect=lambda *a, **k: Response(next(ids))), patch.object(context_compressor, "should_compress", return_value=True), patch.object(context_compressor, "compress_messages", side_effect=compress), patch.object(self.manager, "get_compression_mode", return_value="compress"), patch.object(proxy, "_do_chat", side_effect=chat), patch.object(proxy, "upstream_slot"):
            await proxy.chat(Request({"messages": [{"role": "user", "content": "hi"}]}))
        self.assertEqual([x[0] for x in sent], ["summary", "main"])

    def test_rejects_invalid_session_responses(self):
        cfg = {"token": "token", "session_id": "old", "headers": {}}
        invalid = [None, [], {"code": 1}, {"code": 0, "data": {"biz_code": 1, "biz_data": {"id": "bad"}}}, {"code": 0, "data": {"biz_code": 0, "biz_data": {"id": "  "}}}, {"code": 0, "data": {"biz_data": {"id": 123}}}]
        for payload in invalid:
            response = Response("unused")
            response.json = lambda: payload
            with self.subTest(payload=payload), patch.object(proxy.cffi_requests, "post", return_value=response), self.assertRaises(HTTPException):
                proxy.create_request_session(cfg)

    def test_creation_failure_sets_appropriate_account_state(self):
        cfg = {"token": "token", "session_id": "old", "headers": {}, "account_label": "test"}
        account = self.manager.get_account_by_label("test")
        for status, expected in [(401, proxy.SESSION_INVALID_COOLDOWN_SEC), (403, proxy.SESSION_INVALID_COOLDOWN_SEC), (429, proxy.ERROR_COOLDOWN_SEC)]:
            response = Response("unused")
            response.status_code = status
            response.json = lambda: {}
            with self.subTest(status=status), patch.object(proxy.cffi_requests, "post", return_value=response), patch.object(self.manager, "mark_account_cooldown", wraps=self.manager.mark_account_cooldown) as cooldown:
                with self.assertRaises(HTTPException):
                    proxy.create_request_session(cfg)
                cooldown.assert_called_once_with("test", expected)
                self.assertTrue(account.is_valid)
                self.manager.mark_account_ok("test")
        response = Response("unused")
        response.json = lambda: {"data": {"biz_code": 5, "biz_msg": "user is muted", "biz_data": {}}}
        with patch.object(proxy.cffi_requests, "post", return_value=response), self.assertRaises(HTTPException) as error:
            proxy.create_request_session(cfg)
        self.assertEqual(error.exception.status_code, 403)
        self.assertGreater(account.muted_until, 0)
        self.assertTrue(account.is_valid)

    def test_request_session_does_not_auto_relogin(self):
        with patch.object(proxy, "relogin") as relogin:
            self.assertIsNone(proxy.auto_relogin({"_request_scoped_session": True}))
        relogin.assert_not_called()
