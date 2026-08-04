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
| Gradle | 8.9（wrapper 自动下载） |
| Python 依赖 | 纯 Python 包，构建时由 Chaquopy 自动安装 |

## 构建

```bash
# 1. 配置 SDK 路径（本机路径）
#    编辑 local.properties（不存在则创建）：
#    sdk.dir=C:\\Android\\sdk

# 2. 构建 release APK（无 keystore 时自动使用 debug 签名）
./gradlew assembleRelease
# 或 debug 版
./gradlew assembleDebug

# 产物
app/build/outputs/apk/release/app-release.apk
```

> 如需正式签名：在工程根目录放置 `release.keystore`（别名 `mimo2api`），
> 密码见 `app/build.gradle` 的 `signingConfigs`。没有 keystore 也能构建，
> 会自动回退到 debug 签名（可直接安装测试）。

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
- Python 数据目录：`/data/user/0/com.dsapi.app/files/`（APK 内只读，运行时重定向）

## 与主分支的关系

本分支（`android`）为 Android 打包版。主分支（`main`）是桌面版 Python 服务。
同步流程：主分支更新后，将 `app/`、`proxy.py` 等 Python 代码同步到
`app/src/main/python/` 并重新构建 APK。

## 主要文件

| 文件 | 说明 |
|---|---|
| `app/src/main/python/proxy.py` | DeepSeek 代理主逻辑（v2.3.9 基） |
| `app/src/main/python/android_boot.py` | Android 启动器（垫片注入、Node 解包、数据重定向） |
| `app/src/main/python/_curl_cffi_shim.py` | curl_cffi → httpx 垫片 |
| `app/src/main/python/_tiktoken_shim.py` | tiktoken 纯 Python 近似 |
| `app/src/main/java/.../ServerService.java` | 前台服务，启动 Python 后端 |
| `app/src/main/java/.../MainActivity.java` | WebView 管理后台 |
| `app/src/main/java/.../LoginActivity.java` | 内置浏览器登录页 |
| `app/src/main/jniLibs/arm64-v8a/libnodebin.so` | arm64 Node 运行时 |
| `app/src/main/assets/noderuntime/` | Node 依赖库（运行时解包） |
