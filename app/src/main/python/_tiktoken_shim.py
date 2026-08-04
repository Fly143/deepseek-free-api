"""tiktoken 兼容垫片 —— Android 上的纯 Python token 估算。

背景：tiktoken 是 Rust 扩展，Chaquopy 无预编译轮子。
项目中 tiktoken 仅用于 `len(_enc.encode(text))` 估算 token 数
（用于上下文裁剪与用量统计），不参与请求签名或协议，
因此用近似算法替代不影响功能正确性。

近似规则（贴近 cl100k_base 的实际表现）：
  - CJK 字符：约 1 token / 字
  - 英文单词：约 1 token / 4 字符
  - 数字、标点、空白：单独计
"""
from __future__ import annotations

import re

__all__ = ["get_encoding", "encoding_for_model", "Encoding"]

_CJK = re.compile(
    r"[\u3000-\u303f\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff"
    r"\uf900-\ufaff\uff00-\uffef\uac00-\ud7af]"
)
_WORD = re.compile(r"[A-Za-z]+")
_NUM = re.compile(r"\d")
_OTHER = re.compile(r"[^\sA-Za-z\d]")


class Encoding:
    """最小化的 tiktoken.Encoding 替身。"""

    def __init__(self, name: str = "cl100k_base"):
        self.name = name

    def encode(self, text, *args, **kwargs):
        """返回一个长度≈真实 token 数的列表（内容无意义，仅用于 len()）。"""
        if not text:
            return []
        return [0] * self._count(text)

    def decode(self, tokens, *args, **kwargs):
        # 项目未使用 decode；提供空实现以防调用报错
        return ""

    @staticmethod
    def _count(text: str) -> int:
        cjk = len(_CJK.findall(text))
        # 去掉 CJK 后再统计其余部分，避免重复计数
        rest = _CJK.sub("", text)

        words = _WORD.findall(rest)
        word_tokens = 0
        for w in words:
            # cl100k 下英文大致 4 字符 ≈ 1 token，至少 1
            word_tokens += max(1, (len(w) + 3) // 4)

        digits = len(_NUM.findall(rest))
        num_tokens = max(0, (digits + 2) // 3)

        others = len(_OTHER.findall(rest))

        total = cjk + word_tokens + num_tokens + others
        return max(1, total)


_CACHE: dict[str, Encoding] = {}


def get_encoding(name: str = "cl100k_base") -> Encoding:
    if name not in _CACHE:
        _CACHE[name] = Encoding(name)
    return _CACHE[name]


def encoding_for_model(model: str) -> Encoding:
    return get_encoding("cl100k_base")
