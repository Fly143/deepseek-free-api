"""Android 启动器 —— 注册兼容垫片、准备 Node 运行时、启动 uvicorn。

由 ServerService.java 通过 Chaquopy 调用 main(files_dir, native_lib_dir)。
"""
from __future__ import annotations

import os
import shutil
import sys
import traceback

PORT = 8001


def _install_shims():
    """把 curl_cffi / tiktoken 替换为纯 Python 垫片。

    必须在业务模块 import 之前执行。
    """
    import _curl_cffi_shim
    import _tiktoken_shim

    # curl_cffi 包 + curl_cffi.requests 子模块
    sys.modules["curl_cffi"] = _curl_cffi_shim
    sys.modules["curl_cffi.requests"] = _curl_cffi_shim
    sys.modules["tiktoken"] = _tiktoken_shim
    print("[boot] shims installed: curl_cffi->httpx, tiktoken->pure-python")


def _prepare_node(files_dir: str, native_lib_dir: str):
    """准备 Node 运行时：解包依赖库与 solver 资源，设置环境变量。"""
    py_dir = os.path.dirname(os.path.abspath(__file__))

    # 1) node 可执行文件：Android 只允许执行 lib 目录下的文件，
    #    因此它以 libnodebin.so 的名字打包进 jniLibs
    node_bin = os.path.join(native_lib_dir, "libnodebin.so")

    # 2) 依赖库：libcrypto.so.3 这类带版本号的文件名无法放进 jniLibs
    #    （Android 只打包 lib*.so），故放在 assets 里运行时解包，
    #    保持原始文件名以匹配 node 的 SONAME 查找。
    node_libs = os.path.join(files_dir, "noderuntime")
    os.makedirs(node_libs, exist_ok=True)
    _extract_assets("noderuntime", node_libs)

    # 3) solver 资源解包到可写目录（APK 内 python 目录只读）
    assets_dir = os.path.join(files_dir, "nodejs")
    os.makedirs(assets_dir, exist_ok=True)
    for name in ("pow_solver.js", "sha3_wasm_bg.wasm"):
        src = os.path.join(py_dir, name)
        dst = os.path.join(assets_dir, name)
        try:
            if os.path.exists(src) and (
                not os.path.exists(dst)
                or os.path.getsize(src) != os.path.getsize(dst)
            ):
                shutil.copyfile(src, dst)
        except Exception as e:
            print(f"[boot] copy {name} failed: {e}")

    # 4) 空 openssl.cnf —— 该 Node 二进制硬编码了 Termux 路径，
    #    不覆盖 OPENSSL_CONF 会直接启动失败
    ossl = os.path.join(files_dir, "openssl.cnf")
    if not os.path.exists(ossl):
        with open(ossl, "w", encoding="utf-8") as f:
            f.write("")

    os.environ["DS_NODE_BIN"] = node_bin
    os.environ["DS_NODE_LIBS"] = node_libs
    os.environ["DS_NODE_ASSETS"] = assets_dir
    os.environ["DS_OPENSSL_CONF"] = ossl

    print(f"[boot] node bin: {node_bin} exists={os.path.exists(node_bin)}")
    print(f"[boot] node libs: {node_libs} ({len(os.listdir(node_libs))} files)")
    print(f"[boot] node assets: {assets_dir}")


def _extract_assets(asset_subdir: str, dest_dir: str):
    """把 APK assets 下的目录解包到可写目录（已存在且大小一致则跳过）。"""
    try:
        from com.chaquo.python import Python
        from java.io import File  # noqa: F401
    except Exception as e:
        print(f"[boot] chaquopy unavailable, skip asset extract: {e}")
        return

    try:
        ctx = Python.getPlatform().getApplication()
        am = ctx.getAssets()
        names = am.list(asset_subdir)
        for name in names:
            src_path = asset_subdir + "/" + name
            dst_path = os.path.join(dest_dir, str(name))
            try:
                ins = am.open(src_path)
                data = bytes(ins.readAllBytes()) if hasattr(ins, "readAllBytes") else None
                if data is None:
                    # 兼容旧 API：分块读
                    chunks = []
                    buf = bytearray(65536)
                    while True:
                        n = ins.read(buf)
                        if n <= 0:
                            break
                        chunks.append(bytes(buf[:n]))
                    data = b"".join(chunks)
                ins.close()

                if os.path.exists(dst_path) and os.path.getsize(dst_path) == len(data):
                    continue
                with open(dst_path, "wb") as f:
                    f.write(data)
            except Exception as e:
                print(f"[boot] extract {name} failed: {e}")
    except Exception as e:
        print(f"[boot] asset extract failed: {e}")


def _prepare_data(files_dir: str):
    """数据目录重定向 —— APK 内 python 目录只读。"""
    data_dir = os.path.join(files_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    os.environ["DS_DATA_DIR"] = data_dir
    os.environ.setdefault("PROXY_PORT", str(PORT))
    print(f"[boot] data dir: {data_dir}")
    return data_dir


def main(files_dir: str, native_lib_dir: str, port: int = PORT):
    global PORT
    PORT = int(port or PORT)
    try:
        py_dir = os.path.dirname(os.path.abspath(__file__))
        if py_dir not in sys.path:
            sys.path.insert(0, py_dir)

        print("[boot] starting DeepSeek API server…")
        _prepare_data(files_dir)
        _prepare_node(files_dir, native_lib_dir)
        _install_shims()

        import uvicorn
        from proxy import app

        print(f"[boot] uvicorn on 127.0.0.1:{PORT}")
        uvicorn.run(app, host="127.0.0.1", port=PORT,
                    log_level="info", access_log=False)
    except Exception:
        traceback.print_exc()
        raise


# ServerService.java 调用入口别名
def start(files_dir: str, native_lib_dir: str, port: int = PORT):
    main(files_dir, native_lib_dir, port)
    return "started"
