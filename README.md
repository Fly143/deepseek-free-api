     1|# DeepSeek Free API Proxy
     2|
     3|[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
     4|[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
     5|[![FastAPI](https://img.shields.io/badge/FastAPI-teal)](https://fastapi.tiangolo.com/)
     6|
     7|将 **DeepSeek 网页端免费对话**（chat.deepseek.com）反代为 **OpenAI 兼容 API**，专注纯文本对话，支持深度思考、Vision 图像理解、联网搜索、动态模型发现、PoW 自动求解、Token 自动刷新。**本分支不含工具调用逻辑。**
     8|
     9|本项目所修改代码均为ai完成，不含任何一句人工代码，望周知！
    10|
    11|[zhangjiabo522](https://github.com/zhangjiabo522) — 大力感谢热心群友为 Vision 功能修改测试提供模型Token算力
    12|
    13|> **参考项目：** [NIyueeE/ds-free-api](https://github.com/NIyueeE/ds-free-api)（Rust 版），本项目为 Python 重写。
    14|> Rust 原版使用浏览器自动化（Playwright/Chrome），本 Python 版改为**纯 HTTP 转发**（curl_cffi 模拟 Chrome TLS 指纹），资源占用更低。
    15|
    16|## 目录
    17|
    18|- [特性](#特性)
    19|- [架构](#架构)
    20|- [快速开始](#快速开始)
    21|  - [一键部署（推荐）](#一键部署推荐)
    22|  - [手动安装](#手动安装)
    23|- [配置凭证](#配置凭证)
    24|  - [方法1：手机号/邮箱登录（推荐）](#方法1手机号邮箱登录推荐)
    25|  - [方法2：cURL 导入](#方法2curl-导入)
    26|  - [方法3：Cookie 导入](#方法3cookie-导入)
    27|- [API 使用](#api-使用)
    28|  - [列出模型](#1-列出模型)
    29|  - [非流式对话](#2-非流式对话)
    30|  - [流式对话](#3-流式对话)
    31|    32|  - [模型刷新](#5-模型刷新)
    33|- [模型系统](#模型系统)
    34|  - [动态模型发现](#动态模型发现)
    35|  - [当前可用模型](#当前可用模型)
    36|    37|- [PoW 求解机制](#pow-求解机制)
    38|- [Token 自动刷新](#token-自动刷新)
    39|- [管理命令](#管理命令)
    40|- [项目结构](#项目结构)
    41|- [配置参考](#配置参考)
    42|- [依赖](#依赖)
    43|- [限制与已知问题](#限制与已知问题)
    44|- [常见问题](#常见问题)
    45|- [许可与致谢](#许可与致谢)
    46|
    47|## 特性
    48|
    49|- **OpenAI 完全兼容** — 标准 `/v1/chat/completions`（流式/非流式）、`/v1/models`、`/v1/models/{id}`、`/v1/models/refresh` 端点
    50|    51|- **动态模型发现** — 启动时从 DeepSeek 官方 API 实时探测模型列表，每小时自动刷新（含上下文大小等完整信息）
    52|- **PoW 自动求解** — Node.js WASM 主求解器 + Python 纯算法回退，请求前自动获取 challenge 并求解
    53|- **Token 自动刷新** — 检测到 401 时自动用保存的密码重新登录，无需人工干预
    54|- **深度思考** — 支持 DeepSeek 的 `<thought>` 标签，流式输出时分离为 `reasoning_content`
    55|- **Vision 图像理解** — 支持图片上传、解析、对话，Vision 模型同时支持工具调用
    56|- **联网搜索** — 支持 search 模型变体的 `search_enabled` 参数
    57|- **管理面板** — 内嵌单文件 Web UI，支持手机号/邮箱登录、cURL 导入
    58|- **纯 HTTP 方案** — 不依赖浏览器/Playwright/Chrome，用 curl_cffi 模拟 Chrome TLS 指纹
    59|
    60|## 架构
    61|
    62|```
    63|┌──────────────────────────────────────────────────────────┐
    64|│                     OpenAI 兼容客户端                        │
    65|│            (ChatBox / LobeChat / curl / Cline)             │
    66|└───────────────┬──────────────────────────────────────────┘
    67|                │  /v1/chat/completions
    68|                ▼
    69|┌──────────────────────────────────────────────────────────┐
    70|│                 DeepSeek Free API Proxy (FastAPI)           │
    71|│  ┌─────────┐  ┌──────────────┐  ┌──────────────────────┐ │
    72|│  │ 路由层   │  │  tool_call   │  │   curl_cffi 客户端    │ │
    73|│  │ /v1/*   │──│ (8策略提取)   │──│ (模拟Chrome指纹)      │ │
    74|│  └─────────┘  └──────────────┘  └──────────────────────┘ │
    75|│  ┌─────────┐  ┌──────────────┐  ┌──────────────────────┐ │
    76|│  │ 模型发现 │  │   PoW 求解   │  │   Token 自动刷新      │ │
    77|│  │ (动态)   │  │ (Node+Python) │  │ (保存密码自动relogin) │ │
    78|│  └─────────┘  └──────────────┘  └──────────────────────┘ │
    79|│  ┌─────────┐  ┌──────────────┐                            │
    80|│  │ Vision  │  │ 文件上传/解析 │                            │
    81|│  │ 图像理解 │  │ (upload→fork) │                            │
    82|│  └─────────┘  └──────────────┘                            │
    83|└───────────────┬──────────────────────────────────────────┘
    84|                │  HTTPS (curl_cffi, Chrome指纹)
    85|                ▼
    86|┌──────────────────────────────────────────────────────────┐
    87|│        DeepSeek API (chat.deepseek.com)                   │
    88|│  /api/v0/chat/completion (SSE)                            │
    89|│  /api/v0/users/login                                     │
    90|│  /api/v0/chat_session/create                             │
    91|│  /api/v0/chat/create_pow_challenge                       │
    92|│  /api/v0/client/settings?scope=model                     │
    93|│  /api/v0/file/upload_file + fork_file_task               │
    94|└──────────────────────────────────────────────────────────┘
    95|```
    96|
    97|## 快速开始
    98|
    99|### 一键部署（推荐）
   100|
   101|```bash
   102|# 先安装 Node.js（PoW 求解器需要）
   103|# Termux:
   104|pkg install nodejs
   105|
   106|# Linux:
   107|# sudo apt install nodejs
   108|
   109|# 解压并部署
   110|tar xzf ds-free-api.tar.gz
   111|cd ds-free-api
   112|chmod +x deploy.sh
   113|
   114|# 前台启动（Ctrl+C 停止）
   115|./deploy.sh
   116|
   117|# 或后台启动
   118|./deploy.sh --bg
   119|
   120|# 查看状态
   121|./deploy.sh --status
   122|
   123|# 停止
   124|./deploy.sh --stop
   125|```
   126|
   127|部署完成后访问：**http://localhost:8000/admin**
   128|
   129|### 手动安装
   130|
   131|```bash
   132|# 1. 确保有 Python 3.10+ 和 Node.js
   133|python3 --version
   134|node --version
   135|
   136|# 2. 安装 Python 依赖
   137|pip install fastapi uvicorn curl-cffi python-dotenv
   138|
   139|# 3. 启动
   140|python3 proxy.py
   141|```
   142|
   143|## 配置凭证
   144|
   145|打开管理面板 http://localhost:8000/admin 进行配置。
   146|
   147|### 方法1：手机号/邮箱登录（推荐）
   148|
   149|最方便的方式，和网页登录体验一样：
   150|
   151|1. 选择 **手机号** 或 **邮箱** 标签
   152|2. 填入手机号（区号默认 +86）或邮箱
   153|3. 填入密码
   154|4. 点击 **登录**
   155|
   156|系统会自动完成：登录获取 Token → 创建聊天 Session → 保存配置到 `token.json`（含密码用于自动刷新）。
   157|
   158|### 方法2：cURL 导入
   159|
   160|1. 登录 chat.deepseek.com
   161|2. 打开**开发者工具** → **Network** 面板
   162|3. 发送一条消息，找到 `completion` 请求
   163|4. 右键 → **Copy as cURL**
   164|5. 在管理面板展开 **高级: 手动粘贴 cURL**，粘贴进去
   165|6. 点击 **保存 cURL**
   166|
   167|### 方法3：Cookie 导入
   168|
   169|1. 登录 chat.deepseek.com
   170|2. 打开**开发者工具** → **Application** → **Cookies**
   171|3. 找到 `chat.deepseek.com` 的 Cookie
   172|4. 导出包含 `userToken` 的 Cookie 字符串
   173|5. 粘贴到管理面板 → 保存
   174|
   175|## API 使用
   176|
   177|### 1. 列出模型
   178|
   179|```bash
   180|curl http://localhost:8000/v1/models
   181|```
   182|
   183|返回动态探测到的所有可用模型，包含 `max_input_tokens`、`max_output_tokens` 等详细信息。
   184|
   185|### 2. 非流式对话
   186|
   187|```bash
   188|curl http://localhost:8000/v1/chat/completions \
   189|  -H "Content-Type: application/json" \
   190|  -d '{
   191|    "model": "deepseek-default",
   192|    "messages": [
   193|      {"role": "user", "content": "用Python写一个快速排序"}
   194|    ]
   195|  }'
   196|```
   197|
   198|### 3. 流式对话
   199|
   200|```bash
   201|curl http://localhost:8000/v1/chat/completions \
   202|  -H "Content-Type: application/json" \
   203|  -d '{
   204|    "model": "deepseek-reasoner",
   205|    "messages": [
   206|      {"role": "user", "content": "解释量子纠缠"}
   207|    ],
   208|    "stream": true
   209|  }'
   210|```
   211|
   212|流式响应中思考内容会出现在 `delta.reasoning_content` 字段，正式内容在 `delta.content`。
   213|
   214### 4. 模型刷新
   267|
   268|```bash
   269|# 强制刷新模型列表（无需等待1小时缓存过期）
   270|curl -X POST http://localhost:8000/v1/models/refresh
   271|```
   272|
   273|## 模型系统
   274|
   275|### 动态模型发现
   276|
   277|启动时自动调用 DeepSeek 官方 API `GET /api/v0/client/settings?scope=model` 获取当前可用模型配置。
   278|
   279|核心发现逻辑（`proxy.py:418`）：
   280|
   281|```python
   282|def _discover_models():
   283|    resp = cffi_requests.get(
   284|        "https://chat.deepseek.com/api/v0/client/settings?scope=model",
   285|        headers={"Authorization": f"Bearer {token}", ...}
   286|    )
   287|    # 解析 model_configs，按 model_type 生成基础/思考/搜索/思考+搜索变体
   288|```
   289|
   290|- **自动探测**：无需手动更新模型列表
   291|- **1小时缓存**：避免频繁请求
   292|- **手动刷新**：`POST /v1/models/refresh`
   293|- **容错**：探测失败不影响已缓存的列表
   294|
   295|每个模型返回的信息包括：
   296|- `max_input_tokens` — 最大输入 token
   297|- `max_output_tokens` — 最大输出 token（含思考）
   298|- `thinking_enabled` — 是否支持深度思考
   299|- `search_enabled` — 是否支持联网搜索
   300|
   301|### 当前可用模型
   302|
   303|模型列表**随 DeepSeek 官方动态变化**。当前探测到 3 个基础模型 × 4 变体 = 12 个模型：
   304|
   305|| 模型 ID | 中文名称 | 说明 | 思考 | 联网 |
   306||---------|---------|------|:----:|:----:|
   307|| `deepseek-default` | DeepSeek V4 Flash 基础版 | V4 Flash 快速基础模型 | ✗ | ✗ |
   308|| `deepseek-reasoner` | DeepSeek V4 Flash 思考 | V4 Flash + 深度思考 | ✓ | ✗ |
   309|| `deepseek-search` | DeepSeek V4 Flash 联网 | V4 Flash + 联网搜索 | ✗ | ✓ |
   310|| `deepseek-reasoner-search` | DeepSeek V4 Flash 思考+联网 | V4 Flash + 思考 + 联网 | ✓ | ✓ |
   311|| `deepseek-expert` | DeepSeek V4 Pro 基础版 | V4 Pro 专家基础模型 | ✗ | ✗ |
   312|| `deepseek-expert-reasoner` | DeepSeek V4 Pro 思考 | V4 Pro + 深度思考 | ✓ | ✗ |
   313|| `deepseek-expert-search` | DeepSeek V4 Pro 联网 | V4 Pro + 联网搜索 | ✗ | ✓ |
   314|| `deepseek-expert-reasoner-search` | DeepSeek V4 Pro 思考+联网 | V4 Pro + 思考 + 联网 | ✓ | ✓ |
   315|| `deepseek-vision` | DeepSeek Vision 基础版 | 图像理解基础模型 | ✗ | ✗ |
   316|| `deepseek-vision-reasoner` | DeepSeek Vision 思考 | 图像理解 + 深度思考 | ✓ | ✗ |
   317|
   318|> **注意：**
   319|> - 如果 DeepSeek 推出新模型，代理会自动发现，无需改代码
   320|> - 所有模型均显式指定 `model_type`（`default` / `expert` / `vision`），确保 DeepSeek 正确路由
   321|> - 模型名称为纯英文 ID，中文对照见上表
   322|
   323## PoW 求解机制
   384|
   385|DeepSeek 对 `/api/v0/chat/completion` 端点要求 **Proof of Work (PoW)** 验证。
   386|
   387|### 流程
   388|
   389|1. 每次请求前调用 `POST /api/v0/chat/create_pow_challenge` 获取 challenge
   390|2. 求解 challenge → 得到 `x-ds-pow-response` header
   391|3. 将 solve 结果附加到聊天请求的 header 中
   392|
   393|### 双求解器
   394|
   395|| 求解器 | 方式 | 速度 | 兼容性 |
   396||--------|------|------|--------|
   397|| Node.js WASM | `node pow_solver.js` 子进程 | 快（秒级） | 算法与官方一致 |
   398|| Python 回退 | `hashlib.sha3_256` 纯 Python | 较慢 | 无 Node.js 时备用 |
   399|
   400|需要 Node.js 安装 + `sha3_wasm_bg.wasm` 文件（已包含在项目中）。
   401|
   402|### 算法
   403|
   404|DeepSeek 使用自定义算法 `DeepSeekHashV1`，本质是 SHA3-256 哈希碰撞。WASM 版（Node.js 调用）的算法与官方完全匹配。
   405|
   406|## Token 自动刷新
   407|
   408|Token 有效期约 **24 小时**。当请求返回 401 时：
   409|
   410|1. 检测到 401 → 触发 `relogin()` 函数
   411|2. 用保存的密码重新调用 `POST /api/v0/users/login`
   412|3. 获取新 Token → 创建新 Session → 保存到 `token.json`
   413|4. 用新 Token **重试当前请求**（用户无感知）
   414|
   415|> **前提：** 首次配置时必须通过**账号密码登录**方式。纯 cURL/Cookie 导入不含密码，无法自动刷新。
   416|
   417|## 管理命令
   418|
   419|```bash
   420|# 前台运行
   421|python3 proxy.py
   422|
   423|# 后台启动
   424|./deploy.sh --bg
   425|
   426|# 查看运行状态
   427|./deploy.sh --status
   428|
   429|# 停止后台进程
   430|./deploy.sh --stop
   431|
   432|# 查看实时日志（后台运行时）
   433|tail -f ~/dsapi.log
   434|
   435|# 指定端口
   436|PROXY_PORT=9000 python3 proxy.py
   437|
   438|# 强制刷新模型列表
   439|curl -X POST http://localhost:8000/v1/models/refresh
   440|
   441|# 健康检查
   442|curl http://localhost:8000/health
   443|```
   444|
   445|**启动后：**
   446|
   447|| 地址 | 说明 |
   448||------|------|
   449|| `http://localhost:8000/admin` | Web 管理后台（登录配置） |
   450|| `http://localhost:8000/v1` | OpenAI 兼容 API 根路径 |
   451|| `http://localhost:8000/health` | 健康检查端点 |
   452|
   453|## 项目结构
   454|
   455|```
   456|ds-free-api/
   457|├── proxy.py              # 主程序：FastAPI 应用、SSE 解析、OpenAI 端点、管理面板
   458|   459|├── pow_native.py         # PoW 求解器：Node.js WASM 主求解 + Python 回退
   460|├── pow_solver.js         # Node.js PoW 求解脚本（调用 WASM）
   461|├── sha3_wasm_bg.wasm     # SHA3 WASM 二进制
   462|├── deploy.sh             # 一键部署脚本（安装依赖、启动/停止/状态管理）
   463|├── requirements.txt      # Python 依赖
   464|├── token.example.json    # 配置文件模板
   465|└── token.json            # 实际配置（.gitignore，含凭证）
   466|```
   467|
   468|### 核心文件说明
   469|
   470|| 文件 | 职责 | 行数 |
   471||------|------|------|
   472|| `proxy.py` | 应用入口、路由、SSE 解析、DeepSeek API 交互、Token 刷新、管理面板 UI | ~1524 |
   473|   474|| `pow_native.py` | PoW 求解器（Node.js 子进程 + Python 纯算法回退） | ~124 |
   475|| `deploy.sh` | 一键部署（环境检查、依赖安装、启动/停止/状态） | ~198 |
   476|
   477|## 配置参考
   478|
   479|`token.json` 完整配置项：
   480|
   481|```json
   482|{
   483|  "token": "***",
   484|  "session_id": "abc-def-123...",
   485|  "headers": {
   486|    "content-type": "application/json",
   487|    "origin": "https://chat.deepseek.com",
   488|    "referer": "https://chat.deepseek.com/",
   489|    "user-agent": "Mozilla/5.0 ...",
   490|    "x-client-version": "2.0.2",
   491|    "x-client-platform": "web",
   492|    "authorization": "Bearer YOUR_TOKEN"
   493|  },
   494|  "account": "+86 138xxxx",
   495|  "login_type": "phone",
   496|  "_password": "your_password",
   497|  "_email": "",
   498|  "_mobile": "138xxxx",
   499|  "_area_code": "+86"
   500|}
   501|```
   502|
   503|| 配置项 | 说明 | 自动生成 |
   504||--------|------|:--------:|
   505|| `token` | Bearer Token（约24小时有效） | ✓ |
   506|| `session_id` | 聊天会话 ID（UUID） | ✓ |
   507|| `headers` | 请求头（含 UA、authorization 等） | ✓ |
   508|| `account` | 账号标识（显示用） | ✓ |
   509|| `login_type` | 登录方式：`phone` / `email` | 首次设置 |
   510|| `_password` | 登录密码（用于自动刷新） | 首次设置 |
   511|| `_mobile` | 手机号（自动刷新用） | 首次设置 |
   512|| `_email` | 邮箱（自动刷新用） | 首次设置 |
   513|| `_area_code` | 区号（默认 +86） | 首次设置 |
   514|
   515|> **安全提示：** `_password` 明文存储在本地文件。请确保 `token.json` 权限正确（`chmod 600`），并在分发/打包时排除（已加入 `.gitignore`）。
   516|
   517|**环境变量：** `PROXY_PORT` — 监听端口（默认 `8000`）
   518|
   519|## 依赖
   520|
   521|### Python（pip）
   522|
   523|```bash
   524|pip install fastapi uvicorn curl-cffi python-dotenv
   525|```
   526|
   527|| 依赖 | 用途 |
   528||------|------|
   529|| `fastapi` | Web 框架 |
   530|| `uvicorn` | ASGI 服务器 |
   531|| `curl-cffi` | HTTP 客户端（模拟 Chrome TLS 指纹，绕过反爬） |
   532|| `python-dotenv` | 环境变量加载 |
   533|
   534|### 系统
   535|
   536|- **Node.js** — PoW 求解器（必需，安装 `pkg install nodejs` 或 `apt install nodejs`）
   537|- Python 3.10+ — 运行环境
   538|
   539|## 限制与已知问题
   540|
   541|| 限制 | 说明 |
   542||------|------|
   543|| Token 有效期 | 约 24 小时过期，需要密码登录来自动刷新 |
   544|| 并发限制 | DeepSeek 免费版每账号限制约 2 并发请求 |
   545|| 仅 Chat Completions | 不支持 Embeddings、Fine-tuning 等端点 |
   546|| PoW 耗时 | 每次请求需要先获取并求解 PoW challenge（Node.js 约 1-3 秒） |
   547|| 非流式走 SSE | DeepSeek 只提供 SSE 流，非流式请求会缓冲全部 SSE 后合并返回 |
   548|| Vision 非流式 | Vision 模型在流式模式下无 content 输出，内部用非流式获取后包装为 SSE |
   549|
   550|## 常见问题
   551|
   552|**Q: 启动后访问 /admin 显示空白？**
   553|A: 管理面板是内嵌在 `proxy.py` 中的单文件 HTML，检查是否有 JavaScript 报错（F12 Console）。确保直接访问 `http://localhost:8000/admin`。
   554|
   555|**Q: 提示 "Update to the latest version to use Expert/Vision"？**
   556|A: `x-client-version` 需要与 DeepSeek 网页端保持一致（当前 `2.0.2`）。代理启动时已自动设置。
   557|
   558|**Q: PoW 求解失败？**
   559|A: 检查 Node.js 是否安装（`node --version`）。如果 Node.js 求解失败，代理会自动回退到 Python 纯算法求解（较慢但无需外部依赖）。
   560|
   561|**Q: 登录时提示密码错误？**
   562|A: 确认密码正确。DeepSeek 密码要求至少 8 位，含字母+数字。某些情况下可能需要先完成人机验证再试。
   563|
   564|**Q: Token 过期后怎么办？**
   565|A: 如果使用**账号密码登录**配置的，代理会在 401 时自动重新登录刷新 Token。如果使用 cURL/Cookie 导入的，需要手动重新导入。
   566|
   567|**Q: 可以部署到服务器公网访问吗？**
   568|A: 可以，但建议使用 Nginx 反向代理 + HTTPS + IP 白名单。API Key 不校验（任意值即可），需要通过其他方式控制访问。
   569|
   570|## 为什么选 no-tools

本分支移除了所有工具调用逻辑（`tool_call.py`，约 1000 行代码），**不注入任何工具 prompt**。

效果：
- 上下文更干净 — 模型注意力 100% 在用户问题上
- 输出质量更高 — 不会因格式指令"分心"
- 代码更简洁 — 减少约 1000 行，部署更快
- 不会幻觉 TOOL_CALL — 模型不会无中生有输出工具调用格式
- Token 利用率更高 — 上下文 100% 用于对话内容

如果后续需要工具调用，切回 `main` 分支即可。

## 许可与致谢
   571|
   572|MIT License
   573|
   574|**参考项目：**
   575|- [NIyueeE/ds-free-api](https://github.com/NIyueeE/ds-free-api) — Rust 原版，提供了 DeepSeek API 逆向思路和 PoW 算法参考
   576|- [xstjmark21-cmyk](https://github.com/xstjmark21-cmyk) — 为 Vision 功能修改测试提供模型Token算力
   577|