# DeepSeek Free API Proxy

## 🚨🚨🚨 封号警告 · BAN WARNING 🚨🚨🚨

<div align="center">

# ⛔ 此项目目前封号问题严重，不建议使用 ⛔

**DeepSeek 官方检测力度持续加强，封号处罚会依据违规次数逐渐升级：**

# 🔴 第 1 次 → 封禁 1 天 ｜ 第 2 次 → 封禁 3 天 ｜ 第 3 次 → 封禁 8 天 → 永久封禁

</div>

---


[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-teal)](https://fastapi.tiangolo.com/)

将 **DeepSeek 网页端免费对话**（chat.deepseek.com）反代为 **OpenAI 兼容 API**，支持动态模型发现、PoW 自动求解、手动更新账号凭证。

本项目所修改代码均为ai完成，不含任何一句人工代码，望周知！

> 📖 [English Version](README_EN.md)

[zhangjiabo522](https://github.com/zhangjiabo522) — 大力感谢热心群友为 Vision 功能修改测试提供模型Token算力

> **⚠️ `no-tools` 分支已停止维护**，不再同步新功能和修复。

> **参考项目：** [NIyueeE/ds-free-api](https://github.com/NIyueeE/ds-free-api)（Rust 版），本项目为 Python 重写。
> Rust 原版使用浏览器自动化（Playwright/Chrome），本 Python 版改为**纯 HTTP 转发**（curl_cffi 模拟 Chrome TLS 指纹），资源占用更低。

## 目录

- [特性](#特性)
- [架构](#架构)
- [快速开始](#快速开始)
  - [一键部署（推荐）](#一键部署推荐)
  - [手动安装](#手动安装)
- [配置凭证](#配置凭证)
  - [方法1：账号密码登录](#方法1账号密码登录)
  - [方法2：cURL 导入](#方法2curl-导入)
- [API 使用](#api-使用)
  - [列出模型](#1-列出模型)
  - [非流式对话](#2-非流式对话)
  - [流式对话](#3-流式对话)
  - [模型刷新](#7-模型刷新)
- [Anthropic Messages API](#6-anthropic-messages-api)
- [Responses API](#5-responses-apiopenai-兼容)
- [模型系统](#模型系统)
  - [动态模型发现](#动态模型发现)
  - [当前可用模型](#当前可用模型)
- [工具调用详解](#工具调用详解)
- [无工具分支 (no-tools，已停更)](#无工具分支-no-tools)
- [PoW 求解机制](#pow-求解机制)
- [会话和凭证](#会话和凭证)
- [管理命令](#管理命令)
- [项目结构](#项目结构)
- [配置参考](#配置参考)
- [依赖](#依赖)
- [限制与已知问题](#限制与已知问题)
- [常见问题](#常见问题)
- [许可与致谢](#许可与致谢)

## 特性

- **OpenAI 完全兼容** — `/v1/chat/completions`（流式/非流式）、`/v1/models`、`/v1/models/refresh`、**`/v1/responses`** 端点
- **OpenAI Responses API** — 新增 `/v1/responses` create/retrieve/delete/input_items/cancel/compact，完整 SSE 生命周期事件，Structured Output 支持
- **纯聊天代理** — 无工具调用 prompt 注入，输出更干净，模型注意力集中在用户问题上
- **动态模型发现** — 启动时从 DeepSeek 官方 API 实时探测模型列表，每小时自动刷新（含上下文大小等完整信息）
- **PoW 自动求解** — Node.js WASM 主求解器 + Python 纯算法回退，请求前自动获取 challenge 并求解
- **账号凭证管理** — 凭证失效后可在管理面板重新登录
- **深度思考** — 支持 DeepSeek 的 `<thought>` 标签，流式输出时分离为 `reasoning_content`
- **Vision 图像理解** — 支持图片上传、解析、对话
- **文本文件上传** — 支持 .txt/.md/.py 等文本文件直接上传对话，走 ref_file_ids（和网页端一致）
- **联网搜索** — 支持 search 模型变体的 `search_enabled` 参数
- **管理面板** — 内嵌单文件 Web UI，支持手机号/邮箱登录、cURL 导入
- **纯 HTTP 方案** — 不依赖浏览器/Playwright/Chrome，用 curl_cffi 模拟 Chrome TLS 指纹
- **无工具分支（已停更）** — `no-tools` 分支已停止维护，不再同步新功能和修复

## 架构

```
┌──────────────────────────────────────────────────────────┐
│                     OpenAI 兼容客户端                        │
│            (ChatBox / LobeChat / curl / Cline)             │
└───────────────┬──────────────────────────────────────────┘
                │  /v1/chat/completions
                ▼
┌──────────────────────────────────────────────────────────┐
│                 DeepSeek Free API Proxy (FastAPI)           │
│  ┌─────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │ 路由层   │  │  tool_call   │  │  tool_sieve  │  │  tool_dsml   │  │   curl_cffi 客户端    │ │
│  │ /v1/*   │──│ (工具提示词) │──│ (流式筛分)   │──│ (DSML解析)   │──│ (模拟Chrome指纹)      │ │
│  └─────────┘  └──────────────┘  └──────────────────────┘ │
│  ┌─────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │ 模型发现 │  │   PoW 求解   │  │   账号凭证管理        │ │
│  │ (动态)   │  │ (Node+Python) │  │ (管理面板重新登录)   │ │
│  └─────────┘  └──────────────┘  └──────────────────────┘ │
│  ┌─────────┐  ┌──────────────────────────┐              │
│  │ Vision  │  │ 文件上传/解析             │              │
│  │ 图像理解 │  │ (图片: upload→fork→wait)  │              │
│  └─────────┘  │ (文本: upload→wait)       │              │
│               └──────────────────────────┘              │
└───────────────┬──────────────────────────────────────────┘
                │  HTTPS (curl_cffi, Chrome指纹)
                ▼
┌──────────────────────────────────────────────────────────┐
│        DeepSeek API (chat.deepseek.com)                   │
│  /api/v0/chat/completion (SSE)                            │
│  /api/v0/users/login                                     │
│  /api/v0/chat_session/create                             │
│  /api/v0/chat/create_pow_challenge                       │
│  /api/v0/client/settings                                │
│  /api/v0/file/upload_file + fork_file_task               │
└──────────────────────────────────────────────────────────┘
```

## 快速开始

### 一键部署（推荐）

```bash
# 先安装 Node.js（PoW 求解器需要）
# Termux:
pkg install nodejs

# Linux:
# sudo apt install nodejs

# 直接克隆（推荐）
git clone https://github.com/Fly143/deepseek-free-api.git
cd deepseek-free-api
chmod +x deploy.sh

# 前台启动（Ctrl+C 停止）
./deploy.sh

# 或后台启动
./deploy.sh --bg

# 查看状态
./deploy.sh --status

# 停止
./deploy.sh --stop
```

部署完成后访问：**http://localhost:8000/admin**（默认账号密码：`admin` / `admin`，可在管理面板设置中修改）

### Docker 部署

```bash
docker run -d -p 8000:8000 -v deepseek-data:/data ghcr.io/fly143/deepseek-free-api:latest sh -c 'ln -sf /data/config.json /app/config.json && ln -sf /data/.secret_key /app/.secret_key && exec python proxy.py'
```

> ⚠️ **`no-tools` 分支已停更。**

### 手动安装

```bash
# 1. 确保有 Python 3.10+ 和 Node.js
python3 --version
node --version

# 2. 安装 Python 依赖
pip install fastapi uvicorn curl-cffi python-dotenv tiktoken cryptography jsonschema

# 3. 启动
python3 proxy.py
```

## 配置凭证

打开管理面板 http://localhost:8000/admin 进行配置。

### 方法1：手机号/邮箱登录（推荐）

最方便的方式，和网页登录体验一样：

1. 选择 **手机号** 或 **邮箱** 标签
2. 填入手机号（区号默认 +86）或邮箱
3. 填入密码
4. 点击 **登录**

系统会自动完成：登录获取 Token → 创建聊天 Session → 保存账号配置到 `config.json`。

### 方法2：cURL 导入

1. 登录 chat.deepseek.com
2. 打开**开发者工具** → **Network** 面板
3. 发送一条消息，找到 `completion` 请求
4. 右键 → **Copy as cURL**
5. 在管理面板展开 **高级: 手动粘贴 cURL**，粘贴进去
6. 点击 **保存 cURL**

## API 使用

### 1. 列出模型

```bash
curl http://localhost:8000/v1/models
```

返回动态探测到的所有可用模型，包含 `max_input_tokens`、`max_output_tokens` 等详细信息。

### 2. 非流式对话

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-default",
    "messages": [
      {"role": "user", "content": "用Python写一个快速排序"}
    ]
  }'
```

### 3. 流式对话

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-reasoner",
    "messages": [
      {"role": "user", "content": "解释量子纠缠"}
    ],
    "stream": true
  }'
```

流式响应中思考内容会出现在 `delta.reasoning_content` 字段，正式内容在 `delta.content`。

### 4. 文件上传（文本 & 图片）

**文本文件上传**（所有模型均支持，不 fork，走 `ref_file_ids`）：

```bash
# 准备文件 base64
FILE_B64=$(base64 -w0 三体简介.txt)

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-default",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "这个文件是什么内容？"},
        {"type": "file", "file": {"filename": "三体简介.txt", "file_data": "'"$FILE_B64"'"}}
      ]
    }]
  }'
```

**Vision 图片上传**（需 Vision 模型，上传后 fork 到 vision 类型）：

```bash
# 准备图片 base64
IMG_B64=$(base64 -w0 photo.png)

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-vision",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "描述这张图片"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,'"$IMG_B64"'"}}
      ]
    }]
  }'
```

> **注意：** 文本文件不 fork，直接等 DeepSeek 解析完成后引用原始 `file_id`；图片需要 fork 到 `"vision"` 才能被 Vision 模型读取。

### 5. Responses API（OpenAI 兼容）

支持 OpenAI 最新的 `/v1/responses` 端点。非流式：

```bash
curl http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek",
    "input": "用Python写一个快速排序",
    "stream": false
  }'
```

流式（带完整 SSE 生命周期事件）：

```bash
curl http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek",
    "input": "解释量子纠缠",
    "stream": true
  }'
```

Events: `response.created` → `response.in_progress` → `response.output_item.added` → `response.content_part.added` → `response.output_text.delta`(逐块) → `response.output_text.done` → `response.content_part.done` → `response.output_item.done` → `response.completed`

其他端点（支持流式 replay）：

```bash
# 查询
curl http://localhost:8000/v1/responses/{response_id}

# 输入项
curl http://localhost:8000/v1/responses/{response_id}/input_items

# 取消
curl -X POST http://localhost:8000/v1/responses/{response_id}/cancel

# 删除
curl -X DELETE http://localhost:8000/v1/responses/{response_id}

# 压缩多轮对话
curl -X POST http://localhost:8000/v1/responses/{response_id}/compact \
  -H "Content-Type: application/json" \
  -d '{"instructions": "请用中文回答接下来的所有问题"}'
```

Structured Output（json_schema）：

```bash
curl http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek",
    "input": "北京今天天气25°C，请返回结构化数据",
    "text": {
      "format": {
        "type": "json_schema",
        "schema": {
          "type": "object",
          "properties": {
            "city": {"type": "string"},
            "temperature": {"type": "integer"},
            "unit": {"type": "string"}
          },
          "required": ["city", "temperature", "unit"]
        }
      }
    }
  }'
```

> Responses API 是对现有 `/v1/chat/completions` 的补充，两者可同时使用。

### 6. Anthropic Messages API

本代理完全兼容 **Anthropic Messages API** 格式，支持 RikkaHub 等客户端无缝接入。

**认证方式**：使用 `x-api-key` 头或 `Authorization: Bearer` 均可：

```bash
# x-api-key 方式（推荐）
curl http://localhost:8000/v1/messages \
  -H "x-api-key: sk-dsapi" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-default",
    "max_tokens": 1024,
    "messages": [
      {"role": "user", "content": "用Python写一个快速排序"}
    ]
  }'
```

**流式（思考链 + 文本）：**

```bash
curl http://localhost:8000/v1/messages \
  -H "x-api-key: sk-dsapi" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-reasoner",
    "max_tokens": 1024,
    "stream": true,
    "messages": [
      {"role": "user", "content": "解释量子纠缠"}
    ]
  }'
```

思考内容以 `thinking` block 形式实时流出，文本以 `text` block 流出。

**可用端点：**

| 方法 | 端点 | 说明 |
|------|------|------|
| POST | `/v1/messages` | 发送消息（文本/思考链/工具调用） |
| POST | `/v1/messages/count_tokens` | 计算 token 数 |
| GET | `/v1/messages/{id}` | 查询已发送的消息 |
| POST | `/v1/messages/batches` | 创建批量请求 |
| GET | `/v1/messages/batches` | 列出批量请求 |
| GET | `/v1/messages/batches/{id}` | 查询批量详情 |
| POST | `.../cancel` | 取消批量 |
| GET | `.../results` | 下载批量结果 |
| DELETE | `/v1/messages/batches/{id}` | 删除批量 |

> **注意：** no-tools 分支的 `/v1/messages` 端点**不支持** `tools` 参数，纯对话场景使用更简洁。

#### Anthropic 模型名映射

Claude Code CLI 等工具期望 Anthropic 风格的模型名（如 `claude-sonnet-4-6`），无法直接使用 `deepseek-*` 原生名。本代理在 Anthropic 端点内部自动映射：

| Claude 模型名 | → DeepSeek 内部 | 思考 | 联网 |
|---|---|---|---|
| `claude-opus-4-6` | `deepseek-expert-reasoner` | ✓ | ✗ |
| `claude-opus-4-6-search` | `deepseek-expert-reasoner-search` | ✓ | ✓ |
| `claude-sonnet-4-6` | `deepseek-reasoner` | ✓ | ✗ |
| `claude-sonnet-4-6-search` | `deepseek-reasoner-search` | ✓ | ✓ |
| `claude-haiku-4-5` | `deepseek-default` | ✗ | ✗ |
| `claude-sonnet-4-6-nothinking` | `deepseek-default` | ✗ | ✗ |
| `claude-3-7-sonnet` | `deepseek-reasoner` | ✓ | ✗ |
| `claude-3-5-sonnet` | `deepseek-default` | ✗ | ✗ |
| `claude-3-opus` | `deepseek-expert-reasoner` | ✓ | ✗ |

也支持 Claude 4.x 历史名（`claude-sonnet-4-5`、`claude-opus-4-1` 等）和 `-nothinking` 变体。DeepSeek 原生名（`deepseek-*`）继续直接使用，`/v1/models` 返回的仍是原生名，不影响其他软件。

```bash
# 用 Claude 模型名同样可用
curl http://localhost:8000/v1/messages \
  -H "x-api-key: sk-dsapi" \
  -d '{"model":"claude-sonnet-4-6","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}'
```

### 7. 模型刷新

```bash
# 强制刷新模型列表（无需等待1小时缓存过期）
curl -X POST http://localhost:8000/v1/models/refresh
```

## 模型系统

### 动态模型发现

启动时自动调用 DeepSeek 官方 API `GET /api/v0/client/settings`（参数：`did`、`scope=model`）获取当前可用模型配置。

核心发现逻辑（`proxy.py:418`）：

```python
def _discover_models():
    resp = cffi_requests.get(
        "https://chat.deepseek.com/api/v0/client/settings",
        params={"did": str(uuid.uuid4()), "scope": "model"},
        headers={"Authorization": f"Bearer {token}", ...}
    )
    # 解析 model_configs，按 model_type 生成基础/思考/搜索/思考+搜索变体
```

- **自动探测**：无需手动更新模型列表
- **1小时缓存**：避免频繁请求
- **手动刷新**：`POST /v1/models/refresh`
- **容错**：探测失败不影响已缓存的列表

每个模型返回的信息包括：
- `max_input_tokens` — 最大输入 token
- `max_output_tokens` — 最大输出 token（含思考）
- `thinking_enabled` — 是否支持深度思考
- `search_enabled` — 是否支持联网搜索

### 当前可用模型

模型列表**随 DeepSeek 官方动态变化**。当前探测到 3 个基础模型 × 4 变体 = 12 个模型：

| 模型 ID | 中文名称 | 说明 | 思考 | 联网 |
|---------|---------|------|:----:|:----:|
| `deepseek-default` | DeepSeek V4 Flash 基础版 | V4 Flash 快速基础模型 | ✗ | ✗ |
| `deepseek-reasoner` | DeepSeek V4 Flash 思考 | V4 Flash + 深度思考 | ✓ | ✗ |
| `deepseek-search` | DeepSeek V4 Flash 联网 | V4 Flash + 联网搜索 | ✗ | ✓ |
| `deepseek-reasoner-search` | DeepSeek V4 Flash 思考+联网 | V4 Flash + 思考 + 联网 | ✓ | ✓ |
| `deepseek-expert` | DeepSeek V4 Pro 基础版 | V4 Pro 专家基础模型 | ✗ | ✗ |
| `deepseek-expert-reasoner` | DeepSeek V4 Pro 思考 | V4 Pro + 深度思考 | ✓ | ✗ |
| `deepseek-expert-search` | DeepSeek V4 Pro 联网 | V4 Pro + 联网搜索 | ✗ | ✓ |
| `deepseek-expert-reasoner-search` | DeepSeek V4 Pro 思考+联网 | V4 Pro + 思考 + 联网 | ✓ | ✓ |
| `deepseek-vision` | DeepSeek Vision 基础版 | 图像理解基础模型 | ✗ | ✗ |
| `deepseek-vision-reasoner` | DeepSeek Vision 思考 | 图像理解 + 深度思考 | ✓ | ✗ |

> **注意：**
> - 如果 DeepSeek 推出新模型，代理会自动发现，无需改代码
> - 所有模型均显式指定 `model_type`（`default` / `expert` / `vision`），确保 DeepSeek 正确路由
> - 模型名称为纯英文 ID，中文对照见上表

## 分支说明

本仓库提供两个分支：

| 分支 | 特点 |
|------|------|
| `main`（当前分支） | 完整功能版 — 支持 DSML 工具调用、流式筛分、会话管理等 |
| `no-tools`（已停更） | 纯对话代理，已停止维护，不再同步新功能和修复 |

> ⚠️ `no-tools` 分支已停更，建议使用 main 分支。


## 工具调用详解

DeepSeek 网页端**不支持** OpenAI function calling 格式。本代理通过 **工具提示词 + 多策略提取**实现工具调用：

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-chat",
    "messages": [{"role": "user", "content": "北京天气怎么样？"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "获取天气信息",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string"}},
          "required": ["city"]
        }
      }
    }]
  }'
```

### 工具调用提示词

默认提示模型使用以下格式，工具透传模式则允许模型选择格式：

```text
do: {"name":"search_file","arguments":{"query":"config.yaml"}}
```

### 提取策略

| 优先级 | 格式 | 说明 |
|--------|------|------|
| DO | `do: {"name":"x","arguments":{}}` | 默认提示格式 |
| DSML | `<\|DSML\|tool_calls><\|DSML\|invoke name="X">...</\|DSML\|invoke></\|DSML\|tool_calls>` | 兼容格式，支持噪声变体 |
| TOOL_CALL | `TOOL_CALL: name(key=value)` | 旧格式兜底 |
| JSON | `{"name":"x","arguments":{...}}` | JSON 块解析 |
| XML | `<tool_call><function=NAME>...</function></tool_call>` | 原生 XML |
| 混合 | `<function_call>{...}</function_call>` | XML+JSON |

### 容错能力

- **噪声容错** — 支持缺管道、重复 `<`、全宽 `｜`、连字符 `dsml-` 等 7 种变体
- **围栏代码块** — 自动跳过 markdown 代码块内的 DSML 示例
- **JSON 修复** — 未加引号 key、缺失数组括号自动修复
- **CDATA 保护** — content/command/prompt 等参数保留原始字符串
- **缺失开标签** — 有关闭标签无开头时自动补回


## PoW 求解机制

DeepSeek 对 `/api/v0/chat/completion` 端点要求 **Proof of Work (PoW)** 验证。

### 流程

1. 每次请求前调用 `POST /api/v0/chat/create_pow_challenge` 获取 challenge
2. 求解 challenge → 得到 `x-ds-pow-response` header
3. 将 solve 结果附加到聊天请求的 header 中

### 双求解器

| 求解器 | 方式 | 速度 | 兼容性 |
|--------|------|------|--------|
| Node.js WASM | `node pow_solver.js` 子进程 | 快（秒级） | 算法与官方一致 |
| Python 回退 | `hashlib.sha3_256` 纯 Python | 较慢 | 无 Node.js 时备用 |

需要 Node.js 安装 + `sha3_wasm_bg.wasm` 文件（已包含在项目中）。

### 算法

DeepSeek 使用自定义算法 `DeepSeekHashV1`，本质是 SHA3-256 哈希碰撞。WASM 版（Node.js 调用）的算法与官方完全匹配。

## 会话和凭证

聊天请求每次创建独立的上游对话；如果凭证失效，当前请求不会自动重新登录或重试。请在管理面板重新登录，或重新导入 cURL/Cookie 凭证。客户端每次请求都应发送所需历史消息，历史过长时仍可能被压缩或裁剪。新建的上游对话不会立即删除，过期（默认 3 天）后随清理任务回收。

账号密码登录时，程序为每个账号保存并复用设备标识和请求头；管理面板也支持手动设置设备标识。设备标识取自仓库共享的真实设备 ID 池，但不是本机指纹，这些措施不能保证避免封号。


## 管理命令

```bash
# 前台运行
python3 proxy.py

# 后台启动
./deploy.sh --bg

# 查看运行状态
./deploy.sh --status

# 停止后台进程
./deploy.sh --stop

# 查看实时日志（后台运行时）
tail -f ~/dsapi.log

# 指定端口
PROXY_PORT=9000 python3 proxy.py

# 强制刷新模型列表
curl -X POST http://localhost:8000/v1/models/refresh

# 健康检查
curl http://localhost:8000/health
```

**启动后：**

| 地址 | 说明 |
|------|------|
| `http://localhost:8000/admin` | Web 管理后台（登录配置） |
| `http://localhost:8000/v1` | OpenAI 兼容 API 根路径 |
| `http://localhost:8000/health` | 健康检查端点 |

## 项目结构

```
ds-free-api/
├── proxy.py              # 主程序：FastAPI 应用、SSE 解析、OpenAI 端点、管理面板
├── response_store.py     # Responses API 本地持久化（JSON 文件）
├── pow_native.py         # PoW 求解器：Node.js WASM 主求解 + Python 回退
├── pow_solver.js         # Node.js PoW 求解脚本（调用 WASM）
├── sha3_wasm_bg.wasm     # SHA3 WASM 二进制
├── deploy.sh             # 一键部署脚本（安装依赖、启动/停止/状态管理）
├── requirements.txt      # Python 依赖
├── token.example.json    # 配置文件模板
├── config.json           # 账号与设置（.gitignore）
└── .secret_key           # 加密密钥（.gitignore）
```

### 核心文件说明

| 文件 | 职责 | 行数 |
|------|------|------|
| `proxy.py` | 应用入口、路由、SSE 解析、DeepSeek API 交互、Token 刷新、管理面板 UI | ~3770 |
| `response_store.py` | Responses API 本地持久化（线程安全 JSON 文件读写） | ~73 |
| `pow_native.py` | PoW 求解器（Node.js 子进程 + Python 纯算法回退） | ~124 |
| `deploy.sh` | 一键部署（环境检查、依赖安装、启动/停止/状态） | ~198 |

## 配置参考

账号与设置保存在 `config.json` 的 `accounts` 列表及其他配置字段中。令牌、密码、Cookie、请求头、设备标识和管理密码等敏感字段使用 Fernet 加密后保存；解密密钥保存在同目录的 `.secret_key`。账号标识等其他字段并非全部加密。

两个文件均已列入 `.gitignore`。请限制访问权限，妥善保存并一起备份；丢失或更换密钥将无法解密现有凭证。首次启动时，如果没有 `config.json`，程序会尝试从旧版 `token.json` 迁移账号并将旧文件重命名为 `token.json.bak`；旧文件可能包含明文凭证，应妥善处理。

Docker 示例使用持久卷保存配置和密钥，重建容器时请保留该卷。

**环境变量：**

| 变量 | 作用和默认值 |
|------|-------------|
| `PROXY_PORT` | 监听端口，默认 `8000` |
| `DEEPSEEK_PROXY_API_KEY` | `/v1` 接口密钥；未设置时不鉴权。客户端使用 Bearer 令牌或 `x-api-key` |
| `DEEPSEEK_MIN_INTERVAL_SEC` | 受控上游操作的最小启动间隔，默认 `2.5` 秒 |
| `DEEPSEEK_MAX_CONCURRENT` | 受控上游操作的最大并发数，默认 `1` |
| `DEEPSEEK_ERROR_COOLDOWN_SEC` | 一般错误后的账号冷却时间，默认 `60` 秒 |
| `DEEPSEEK_SESSION_INVALID_COOLDOWN_SEC` | 会话失效后的账号冷却时间，默认 `300` 秒 |
| `DEEPSEEK_AUTO_RELOGIN` | 自动重登开关，默认启用；v2.5.0 起聊天请求（/v1 全部路径）每次新建上游会话，命中会话失效时不再自动重登，改为冷却账号（默认 300 秒）；手动重登见管理面板 |
| `DEEPSEEK_SLOT_WAIT_TIMEOUT_SEC` | 等待全局上游槽位的上限，默认 `600` 秒；超时返回 503（建会话/非流式阶段），不冷却账号 |
| `DS_STREAM_IDLE_TIMEOUT` | 上游流读取空闲上限，默认 `300` 秒（总时长仍不限）；仅在未设置 `DS_CLIENT_TIMEOUT` 时生效 |
| `DS_CLIENT_TIMEOUT` | 覆盖整个上游请求的超时（秒），默认 `0`=不限；设置后 `DS_STREAM_IDLE_TIMEOUT` 不再生效 |

这些限制不覆盖所有上游操作，例如附件上传。公网部署应设置接口密钥，并配置 HTTPS 和其他访问控制。

## 依赖

### Python（pip）

```bash
pip install fastapi uvicorn curl-cffi python-dotenv tiktoken cryptography jsonschema
```

| 依赖 | 用途 |
|------|------|
| `fastapi` | Web 框架 |
| `uvicorn` | ASGI 服务器 |
| `curl-cffi` | HTTP 客户端（模拟 Chrome TLS 指纹，绕过反爬） |
| `python-dotenv` | 环境变量加载 |
| `tiktoken` | 分词与用量统计 |
| `cryptography` | config.json 敏感字段加密（Fernet） |
| `jsonschema` | 工具参数结构校验 |

### 系统

- **Node.js** — PoW 求解器（必需，安装 `pkg install nodejs` 或 `apt install nodejs`）
- Python 3.10+ — 运行环境

## 限制与已知问题

| 限制 | 说明 |
|------|------|
| Token 有效期 | 约 24 小时过期，失效后需手动重新登录或重新导入凭证 |
| 并发限制 | DeepSeek 免费版每账号限制约 2 并发请求 |
| 仅 Chat Completions + Responses | 不支持 Embeddings、Fine-tuning 等端点 |
| PoW 耗时 | 每次请求需要先获取并求解 PoW challenge（Node.js 约 1-3 秒） |
| 非流式走 SSE | DeepSeek 只提供 SSE 流，非流式请求会缓冲全部 SSE 后合并返回 |
| Vision 非流式 | Vision 模型在流式模式下无 content 输出，内部用非流式获取后包装为 SSE |

## 常见问题

**Q: 启动后访问 /admin 显示空白？**
A: 管理面板是内嵌在 `proxy.py` 中的单文件 HTML，检查是否有 JavaScript 报错（F12 Console）。确保直接访问 `http://localhost:8000/admin`。

**Q: 提示 "Update to the latest version to use Expert/Vision"？**
A: `x-client-version` 需要与 DeepSeek 网页端保持一致（当前 `2.0.2`）。代理启动时已自动设置。

**Q: PoW 求解失败？**
A: 检查 Node.js 是否安装（`node --version`）。如果 Node.js 求解失败，代理会自动回退到 Python 纯算法求解（较慢但无需外部依赖）。

**Q: 登录时提示密码错误？**
A: 确认密码正确。DeepSeek 密码要求至少 8 位，含字母+数字。某些情况下可能需要先完成人机验证再试。

**Q: Token 过期后怎么办？**
A: 请在管理面板重新登录；使用 cURL/Cookie 导入的账号需重新导入凭证。

**Q: 指定 expert 模型但对话记录显示在"快速模式"（default）？**
A: 通常是 Token 或 Session 过期导致的。DeepSeek 在凭证失效时会把请求降级到 default 模型。解决方法：在管理面板 `http://localhost:8000/admin` 用手机号/邮箱**重新登录**一次即可，登录后自动刷新 Token 和 Session。

**Q: 可以部署到服务器公网访问吗？**
A: 请设置 `DEEPSEEK_PROXY_API_KEY` 并配置 HTTPS 和访问控制。未设置该变量时，`/v1` 接口不校验密钥。

## 许可与致谢

MIT License

**参考项目：**
- [NIyueeE/ds-free-api](https://github.com/NIyueeE/ds-free-api) — Rust 原版，提供了 DeepSeek API 逆向思路和 PoW 算法参考
- [CJackHwang/ds2api](https://github.com/CJackHwang/ds2api) — DSML 工具调用格式、流式筛分架构、DeepSeek 原生对话标记 均参考此项目
- [GoblinHonest/mimo2api_mimoapi](https://github.com/GoblinHonest/mimo2api_mimoapi) — 会话管理（消息指纹续接、token 超限自动清屏）设计参考
- [Acidmoon](https://github.com/Acidmoon) — 提交 PR #2，实现 OpenAI Responses API 兼容层
- [xstjmark21-cmyk](https://github.com/xstjmark21-cmyk) — 为 Vision 功能修改测试提供模型Token算力