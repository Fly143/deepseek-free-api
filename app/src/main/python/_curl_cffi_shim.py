"""curl_cffi 兼容垫片 —— 在 Android 上用 httpx 提供等价 API。

背景：Chaquopy 无 curl-cffi 预编译轮子（依赖 libcurl-impersonate 原生库）。
已实测 DeepSeek 不校验 TLS 指纹：httpx 裸连 chat.deepseek.com 的
首页 / users/login / users/current / chat_session/create /
create_pow_challenge 全部返回 200，AWS WAF 未触发。

因此本模块吃掉 impersonate 参数，其余行为对齐 curl_cffi.requests，
使业务代码（proxy.py 等）无需任何改动。
"""
from __future__ import annotations

import httpx

__all__ = ["requests", "Session", "get", "post"]

_DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
)


class _Response:
    """包装 httpx.Response，补齐 curl_cffi 用到的接口。"""

    def __init__(self, resp: httpx.Response, client: httpx.Client | None = None,
                 streaming: bool = False):
        self._r = resp
        self._client = client
        self._streaming = streaming

    # --- curl_cffi / requests 通用属性 ---
    @property
    def status_code(self):
        return self._r.status_code

    @property
    def headers(self):
        return self._r.headers

    @property
    def text(self):
        return self._r.text

    @property
    def content(self):
        return self._r.content

    @property
    def cookies(self):
        return self._r.cookies

    @property
    def url(self):
        return str(self._r.url)

    def json(self, **kw):
        return self._r.json(**kw)

    def iter_content(self, chunk_size: int = 4096):
        """对应 curl_cffi 的流式读取；结束后释放连接。"""
        try:
            for chunk in self._r.iter_bytes(chunk_size):
                yield chunk
        finally:
            self.close()

    def iter_lines(self, chunk_size: int = 4096):
        for line in self._r.iter_lines():
            yield line.encode("utf-8") if isinstance(line, str) else line

    def close(self):
        try:
            self._r.close()
        except Exception:
            pass
        # 流式请求使用了临时 client，读完必须关掉避免句柄泄漏
        if self._streaming and self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass

    def raise_for_status(self):
        self._r.raise_for_status()
        return self

    def __getattr__(self, item):
        return getattr(self._r, item)


def _clean_kwargs(kw: dict) -> dict:
    """剔除 curl_cffi 专属参数，转换 requests 风格参数为 httpx 风格。"""
    kw.pop("impersonate", None)
    kw.pop("verify", None)
    kw.pop("allow_redirects", None)
    kw.pop("http_version", None)
    kw.pop("default_headers", None)

    # requests/curl_cffi 的 proxies={"http":..,"https":..} -> httpx proxies
    proxies = kw.pop("proxies", None)
    if proxies:
        kw["_proxies"] = proxies
    return kw


def _build_client(proxies=None, timeout=30, cookies=None, headers=None) -> httpx.Client:
    proxy = None
    if proxies:
        proxy = proxies.get("https") or proxies.get("http") or proxies.get("all")
    return httpx.Client(
        timeout=timeout,
        follow_redirects=True,
        proxy=proxy,
        cookies=cookies,
        headers=headers or {"user-agent": _DEFAULT_UA},
        verify=True,
    )


class Session:
    """对应 curl_cffi.requests.Session —— 保持 cookie 会话。"""

    def __init__(self, **kw):
        _clean_kwargs(kw)
        self.impersonate = None          # 业务代码会赋值，这里吃掉
        self.proxies = None
        self.headers = {"user-agent": _DEFAULT_UA}
        self._client: httpx.Client | None = None

    @property
    def cookies(self):
        """始终返回底层 client 的 cookie jar，保证登录流程能读到 WAF cookie。"""
        return self._ensure().cookies

    def _ensure(self, timeout=30) -> httpx.Client:
        if self._client is None:
            proxy = None
            if self.proxies:
                proxy = (self.proxies.get("https") or self.proxies.get("http")
                         or self.proxies.get("all"))
            self._client = httpx.Client(
                timeout=timeout,
                follow_redirects=True,
                proxy=proxy,
                headers=self.headers,
            )
        return self._client

    def request(self, method: str, url: str, **kw):
        kw = _clean_kwargs(kw)
        kw.pop("_proxies", None)
        stream = kw.pop("stream", False)
        timeout = kw.pop("timeout", 30)
        c = self._ensure(timeout)

        if stream:
            req = c.build_request(method, url, timeout=timeout, **kw)
            resp = c.send(req, stream=True)
            return _Response(resp, c, streaming=False)  # client 复用，不在此关闭

        resp = c.request(method, url, timeout=timeout, **kw)
        return _Response(resp)

    def get(self, url, **kw):
        return self.request("GET", url, **kw)

    def post(self, url, **kw):
        return self.request("POST", url, **kw)

    def put(self, url, **kw):
        return self.request("PUT", url, **kw)

    def delete(self, url, **kw):
        return self.request("DELETE", url, **kw)

    def close(self):
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def request(method: str, url: str, **kw):
    kw = _clean_kwargs(kw)
    proxies = kw.pop("_proxies", None)
    stream = kw.pop("stream", False)
    timeout = kw.pop("timeout", 30)
    headers = kw.get("headers")

    client = _build_client(proxies, timeout, headers=headers)
    if stream:
        req = client.build_request(method, url, timeout=timeout, **kw)
        resp = client.send(req, stream=True)
        # 流式：读完由 _Response.close() 关闭 client
        return _Response(resp, client, streaming=True)

    try:
        resp = client.request(method, url, timeout=timeout, **kw)
        # 一次性请求：先把 body 读进内存再关 client
        _ = resp.content
        return _Response(resp)
    finally:
        if not stream:
            client.close()


def get(url, **kw):
    return request("GET", url, **kw)


def post(url, **kw):
    return request("POST", url, **kw)


def put(url, **kw):
    return request("PUT", url, **kw)


def delete(url, **kw):
    return request("DELETE", url, **kw)


# 让 `from curl_cffi import requests as cffi_requests` 拿到本模块自身
import sys as _sys  # noqa: E402

requests = _sys.modules[__name__]
