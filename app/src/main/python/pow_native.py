"""
DeepSeek PoW Solver — Node.js WASM bridge (primary) + pure Python fallback
"""

import json
import base64
import subprocess
import os
import hashlib
import struct
import time
from typing import Dict, Any, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
POW_SOLVER_JS = os.path.join(SCRIPT_DIR, "pow_solver.js")

# Android: 使用 APK 内置的 Node 运行时（jniLibs 下的 libnodebin.so 可执行）
# 由 android_boot.py 设置以下环境变量
_NODE_BIN = os.environ.get("DS_NODE_BIN")            # node 可执行文件绝对路径
_NODE_LIBS = os.environ.get("DS_NODE_LIBS")          # 依赖 .so 所在目录
_NODE_ASSETS = os.environ.get("DS_NODE_ASSETS")      # pow_solver.js + wasm 解包目录
_OPENSSL_CONF = os.environ.get("DS_OPENSSL_CONF")    # 空的 openssl.cnf（绕开硬编码 Termux 路径）


class DeepSeekPOW:
    """Solves DeepSeek PoW challenge via Node.js WASM bridge."""

    def solve_challenge(self, config: Dict[str, Any]) -> str:
        """Solve PoW challenge and return base64-encoded response.

        Tries Node.js WASM solver first (fast + correct),
        falls back to pure Python if Node.js unavailable.
        """
        answer = self._solve_via_node(config)
        if answer is None:
            print("[PoW] Node.js solver failed, trying pure Python fallback...")
            answer = self._solve_pure_python(config)

        if answer is None:
            raise RuntimeError("PoW solve failed with both methods")

        result = {
            "algorithm": config["algorithm"],
            "challenge": config["challenge"],
            "salt": config["salt"],
            "answer": answer,
            "signature": config["signature"],
            "target_path": config["target_path"],
        }
        return base64.b64encode(json.dumps(result).encode()).decode()

    def _solve_via_node(self, config: Dict[str, Any]) -> Optional[int]:
        """Call Node.js WASM solver subprocess."""
        try:
            input_json = json.dumps(config)

            node_cmd = _NODE_BIN or "node"
            solver_js = POW_SOLVER_JS
            workdir = SCRIPT_DIR
            env = dict(os.environ)

            if _NODE_ASSETS:
                # APK 内 python 目录只读，pow_solver.js / .wasm 已解包到可写目录
                candidate = os.path.join(_NODE_ASSETS, "pow_solver.js")
                if os.path.exists(candidate):
                    solver_js = candidate
                    workdir = _NODE_ASSETS
            if _NODE_LIBS:
                env["LD_LIBRARY_PATH"] = _NODE_LIBS
            if _OPENSSL_CONF:
                # 该 Node 二进制硬编码了 Termux 的 openssl.cnf 路径，
                # 不覆盖会直接报 BIO_new_file:Permission denied 启动失败
                env["OPENSSL_CONF"] = _OPENSSL_CONF

            result = subprocess.run(
                [node_cmd, solver_js, input_json],
                capture_output=True,
                text=True,
                timeout=300,
                cwd=workdir,
                env=env,
            )
            if result.returncode == 0 and result.stdout.strip():
                # stdout is base64-encoded response, decode to get answer
                decoded = json.loads(base64.b64decode(result.stdout.strip()))
                answer = decoded.get("answer")
                print(f"[PoW-Node] nonce={answer}")
                return answer
            else:
                print(f"[PoW-Node] Failed: {result.stderr[:200]}")
                return None
        except FileNotFoundError:
            print("[PoW-Node] node not found")
            return None
        except subprocess.TimeoutExpired:
            print("[PoW-Node] Timeout (300s)")
            return None
        except Exception as e:
            print(f"[PoW-Node] Error: {e}")
            return None

    def _solve_pure_python(self, config: Dict[str, Any]) -> Optional[int]:
        """Pure Python fallback — may not match WASM algorithm exactly."""
        try:
            challenge = config["challenge"]
            salt = config["salt"]
            difficulty = config["difficulty"]
            expire_at = config["expire_at"]

            prefix = f"{salt}_{expire_at}_"
            threshold = (2**32) // difficulty

            start = time.time()
            for nonce in range(10_000_000):
                data = prefix + str(nonce)
                h = hashlib.sha3_256((challenge + data).encode()).digest()
                value = struct.unpack("<I", h[:4])[0]
                if value < threshold:
                    elapsed = time.time() - start
                    print(f"[PoW-Python] nonce={nonce}, time={elapsed:.2f}s")
                    return nonce

            print("[PoW-Python] No solution found in 10M iterations")
            return None
        except Exception as e:
            print(f"[PoW-Python] Error: {e}")
            return None


# Standalone test
if __name__ == "__main__":
    import sys

    pow = DeepSeekPOW()
    if len(sys.argv) > 1:
        config = json.loads(sys.argv[1])
    else:
        config = {
            "algorithm": "DeepSeekHashV1",
            "challenge": "b0000b22959bad0cc1ecbbfa07f97191b20332fa10d7341ff9c7ba6e7ed927f1",
            "salt": "dde3ed472be5a2494ee0",
            "difficulty": 144000,
            "expire_at": 1777057596443,
            "signature": "test",
            "target_path": "/api/v0/chat/completion",
        }

    response = pow.solve_challenge(config)
    decoded = json.loads(base64.b64decode(response))
    print(f"Answer: {decoded['answer']}")
    print(f"Response: {response[:80]}...")
