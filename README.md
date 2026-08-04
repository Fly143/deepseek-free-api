# deepseek-free-api Android 版

将 DeepSeek 网页版代理服务打包为独立 Android APK（Chaquopy + WebView）。

## 特性

- **点开即用** — 打开 App 直接进入管理后台（右下角「登录 DEEPSEEK」按钮进入内置浏览器登录）
- **内置 Node 运行时** — 执行原版 PoW WASM 求解（`pow_solver.js`），无需外部依赖
- **无 TLS 指纹依赖** — `curl_cffi` 用纯 Python 垫片替代（httpx），`tiktoken` 用近似计数
- **内置浏览器登录** — 登录 chat.deepseek.com 后自动抓取 token 并导入账号
- **动态模型发现** — 从 DeepSeek 官方 API 实时获取模型列表（含 did 参数）

## 环境要求

| 依赖 | 版本 |
|---|---|
| Android SDK | compileSdk 35 (Android 15) |
| JDK | 17 |
| Gradle | 8.9（wrapper 自动下载，无需预装） |
| Python 依赖 | 纯 Python 包，构建时由 Chaquopy 自动安装 |

## 构建

```bash
# 1. 克隆
git clone -b android https://github.com/Fly143/deepseek-free-api.git
cd deepseek-free-api

# 2. 创建 local.properties（Windows 注意格式，必须用 \: 转义冒号）
#    Windows:
echo sdk.dir=C\:\\Android\\sdk > local.properties
#    macOS/Linux:
echo sdk.dir=/path/to/android/sdk > local.properties

# 3. 构建 release APK（无 keystore 时自动使用 debug 签名）
./gradlew assembleRelease
# 或 debug 版
./gradlew assembleDebug

# 产物
app/build/outputs/apk/release/app-release.apk
```

> **⚠️ Windows 用户注意**：`local.properties` 中的路径必须使用 `\:` 转义冒号，
> 例如 `sdk.dir=C\:\\Android\\sdk`。如果写成 `sdk.dir=C:\Android\sdk`（不转义）
> 会导致 `IOException: 文件名、目录名或卷标语法不正确`。

> **签名**：仓库不含 `release.keystore`（避免密钥泄露）。构建时会自动检测：
> - 有 `release.keystore` → 使用 release 签名
> - 无 `release.keystore` → 回退 debug 签名（可直接安装测试）

## 架构

```
┌─ Android App ────────────────────────────┐
│  MainActivity (WebView 管理后台)          │
│    └─ http://127.0.0.1:8001/admin        │
│  ServerService (前台服务)                 │
│    └─ Chaquopy Python 后端                │
│       └─ uvicorn @ 127.0.0.1:8001        │
│          └─ proxy.py (DeepSeek 代理)     │
│             └─ Node (PoW WASM 求解)      │
└──────────────────────────────────────────┘
```

- 端口：**8001**（与 MiMo2API 的 8000 隔离，可同时运行）
- 管理后台认证：`admin` / `admin`（可在 config.json 修改）

## 与主分支的关系

本分支（`android`）为 Android 打包版。主分支（`main`）是桌面版 Python 服务。
同步流程：主分支更新后，将 `proxy.py` 等 Python 代码同步到
`app/src/main/python/` 并重新构建 APK。

## 主要文件

| 文件 | 说明 |
|---|---|
| `app/src/main/python/proxy.py` | DeepSeek 代理主逻辑 |
| `app/src/main/python/android_boot.py` | Android 启动器（垫片注入、Node 解包、数据重定向） |
| `app/src/main/python/_curl_cffi_shim.py` | curl_cffi → httpx 垫片 |
| `app/src/main/python/_tiktoken_shim.py` | tiktoken 纯 Python 近似 |
| `app/src/main/java/.../ServerService.java` | 前台服务，启动 Python 后端 |
| `app/src/main/java/.../MainActivity.java` | WebView 管理后台 |
| `app/src/main/java/.../LoginActivity.java` | 内置浏览器登录页 |
| `app/src/main/jniLibs/arm64-v8a/libnodebin.so` | arm64 Node 运行时 |
| `app/src/main/assets/noderuntime/` | Node 依赖库（运行时解包） |
