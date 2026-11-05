import asyncio
import base64
import os
import unittest
from unittest.mock import patch

import proxy


async def request(path, headers=None, method="GET", body=b""):
    received = False
    never = asyncio.Event()
    events = []

    async def receive():
        nonlocal received
        if not received:
            received = True
            return {"type": "http.request", "body": body, "more_body": False}
        await never.wait()

    async def send(event):
        events.append(event)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()],
        "client": ("127.0.0.1", 12345),
        "server": ("127.0.0.1", 8000),
    }
    await asyncio.wait_for(proxy.app(scope, receive, send), timeout=5)
    return next(event["status"] for event in events if event["type"] == "http.response.start")


class ApiAuthTests(unittest.TestCase):
    def test_key_required_for_api_paths_and_double_prefix(self):
        with patch.dict(os.environ, {"DEEPSEEK_PROXY_API_KEY": "test-secret"}):
            for path in ("/v1", "/v1/missing", "/v1/v1/missing"):
                with self.subTest(path=path):
                    self.assertEqual(asyncio.run(request(path)), 401)
                    self.assertEqual(asyncio.run(request(path, {"Authorization": "Bearer wrong"})), 401)
                    self.assertNotEqual(asyncio.run(request(path, {"Authorization": "Bearer test-secret"})), 401)
                    self.assertNotEqual(asyncio.run(request(path, {"x-api-key": "test-secret"})), 401)

    def test_unconfigured_key_preserves_access(self):
        with patch.dict(os.environ, {"DEEPSEEK_PROXY_API_KEY": ""}):
            self.assertNotEqual(asyncio.run(request("/v1/missing")), 401)

    def test_browser_preflight_succeeds_without_credentials(self):
        headers = {"Origin": "https://example.org", "Access-Control-Request-Method": "POST"}
        with patch.dict(os.environ, {"DEEPSEEK_PROXY_API_KEY": "test-secret"}):
            self.assertEqual(asyncio.run(request("/v1/chat/completions", headers, "OPTIONS")), 200)
            self.assertEqual(asyncio.run(request("/v1/chat/completions", {}, "OPTIONS")), 401)

    def test_login_requires_admin_credentials(self):
        body = b"{}"
        headers = {"Content-Type": "application/json"}
        with patch.object(proxy.config_manager, "get_admin_password", return_value="test-password"):
            self.assertEqual(asyncio.run(request("/api/login", headers, "POST", body)), 401)
            encoded = base64.b64encode(b"admin:test-password").decode()
            headers["Authorization"] = f"Basic {encoded}"
            self.assertEqual(asyncio.run(request("/api/login", headers, "POST", body)), 400)


if __name__ == "__main__":
    unittest.main()
