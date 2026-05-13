# DeepSeek Free API Proxy

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-teal)](https://fastapi.tiangolo.com/)

Reverse-engineer **DeepSeek web chat** (chat.deepseek.com) into an **OpenAI-compatible API**, with dynamic model discovery, automatic PoW solving, token refresh, and a pure chat edition (no-tools branch, zero tool prompt injection).

本项目所修改代码均为ai完成，不含任何一句人工代码，望周知！

> 📖 [中文版本](README.md)

[zhangjiabo522](https://github.com/zhangjiabo522) — Thanks for providing model tokens for Vision feature testing!

> **💡 Need pure chat without tools?** If your use case is pure conversation (writing, translation, coding, Q&A), use the [`no-tools` branch](#branch-info) — no tool prompt injection, cleaner context, higher output quality.

> **Reference project:** [NIyueeE/ds-free-api](https://github.com/NIyueeE/ds-free-api) (Rust). This is a Python rewrite using pure HTTP forwarding (curl_cffi with Chrome TLS fingerprint) instead of browser automation, with lower resource usage.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
  - [One-Click Deploy (Recommended)](#one-click-deploy-recommended)
  - [Manual Install](#manual-install)
- [Authentication](#authentication)
  - [Method 1: Phone/Email Login (Recommended)](#method-1-phoneemail-login-recommended)
  - [Method 2: cURL Import](#method-2-curl-import)
  - [Method 3: Cookie Import](#method-3-cookie-import)
- [API Usage](#api-usage)
  - [List Models](#1-list-models)
  - [Non-streaming Chat](#2-non-streaming-chat)
  - [Streaming Chat](#3-streaming-chat)
  - [Model Refresh](#7-model-refresh)
- [Anthropic Messages API](#6-anthropic-messages-api)
- [Model System](#model-system)
  - [Dynamic Model Discovery](#dynamic-model-discovery)
  - [Currently Available Models](#currently-available-models)
- [Tool Calling](#tool-calling)
- [Branch Info](#branch-info)
- [PoW Solver](#pow-solver)
- [Token Auto-Refresh](#token-auto-refresh)
- [Management Commands](#management-commands)
- [Project Structure](#project-structure)
- [Configuration Reference](#configuration-reference)
- [Dependencies](#dependencies)
- [Limitations & Known Issues](#limitations--known-issues)
- [FAQ](#faq)
- [License & Credits](#license--credits)

## Features

- **OpenAI Fully Compatible** — `/v1/chat/completions` (streaming/non-streaming), `/v1/models`, `/v1/models/refresh`, **`/v1/responses`** endpoints
- **OpenAI Responses API** — `/v1/responses` create/retrieve/delete/input_items/cancel/compact, full SSE lifecycle events, Structured Output support
- **Pure Chat Proxy** — No tool call prompt injection, cleaner output, model attention focused on user queries
- **Dynamic Model Discovery** — Real-time model list fetched from DeepSeek official API at startup, auto-refreshes hourly (includes context size and full metadata)
- **Automatic PoW Solving** — Node.js WASM primary solver + Python pure algorithm fallback, auto-fetches and solves challenges before each request
- **Token Auto-Refresh** — Automatically re-logins with saved password on 401, no manual intervention needed
- **Deep Thinking** — Supports DeepSeek's `<thought>` tags, separated as `reasoning_content` in streaming output
- **Vision** — Image upload, parsing, and conversation
- **Text File Upload** — Upload .txt/.md/.py files for chat via `ref_file_ids` (same as web UI)
- **Web Search** — `search_enabled` parameter for search model variants
- **Multilingual Admin Panel** — Embedded single-file Web UI, Chinese/English toggle, phone/email login, cURL import
- **Pure HTTP Solution** — No browser/Playwright/Chrome dependency, uses curl_cffi to emulate Chrome TLS fingerprint
- **No-Tools Branch** — Dedicated `no-tools` branch with tool calling logic removed, ideal for pure chat scenarios

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                  OpenAI Compatible Client                  │
│            (ChatBox / LobeChat / curl / Cline)             │
└───────────────┬──────────────────────────────────────────┘
                │  /v1/chat/completions
                ▼
┌──────────────────────────────────────────────────────────┐
│             DeepSeek Free API Proxy (FastAPI)              │
│  ┌─────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │ Routes  │  │  tool_call   │  │  tool_sieve │ tool_dsml │ │
│  │ /v1/*   │──│ (DSML prompt)│──│ (streaming  │ (DSML     │ │
│  └─────────┘  │ injection)   │  │ sieve)     │  parser)  │ │
│               └──────────────┘  └──────────────────────┘ │
│  ┌─────────┐  ┌──────────────┐  ┌──────────────────────┐ │
│  │ Model   │  │ PoW Solver   │  │   Token Refresh     │ │
│  │ Disc.   │  │ (Node+Python)│  │ (auto-relogin)      │ │
│  └─────────┘  └──────────────┘  └──────────────────────┘ │
│  ┌─────────┐  ┌──────────────┐                            │
│  │ curl    │  │ Vision/Files │                            │
│  │ (Chrome │  │ (upload→fork │                            │
│  │  TLS)   │  │  →wait)      │                            │
│  └─────────┘  └──────────────┘                            │
└───────────────┬──────────────────────────────────────────┘
                │  HTTPS (curl_cffi, Chrome fingerprint)
                ▼
┌──────────────────────────────────────────────────────────┐
│           DeepSeek API (chat.deepseek.com)                 │
│  /api/v0/chat/completion (SSE)                            │
│  /api/v0/users/login                                     │
│  /api/v0/chat_session/create                             │
│  /api/v0/chat/create_pow_challenge                       │
│  /api/v0/client/settings?scope=model                     │
│  /api/v0/file/upload_file + fork_file_task               │
└──────────────────────────────────────────────────────────┘
```

## Quick Start

### One-Click Deploy (Recommended)

```bash
# Install Node.js first (required by PoW solver)
# Termux:
pkg install nodejs

# Linux:
# sudo apt install nodejs

# Clone the repo
git clone https://github.com/Fly143/deepseek-free-api.git
cd deepseek-free-api
chmod +x deploy.sh

# Foreground (Ctrl+C to stop)
./deploy.sh

# Or background
./deploy.sh --bg

# Check status
./deploy.sh --status

# Stop
./deploy.sh --stop
```

After deployment, visit: **http://localhost:8000/admin**

### Docker

```bash
docker run -d -p 8000:8000 -v $(pwd)/config.json:/app/config.json ghcr.io/fly143/deepseek-free-api:latest
```

> 💡 **No tools needed?** Clone the [`no-tools` branch](https://github.com/Fly143/deepseek-free-api/tree/no-tools) for a cleaner pure chat edition (no prompt injection, higher output quality).

### Manual Install

```bash
# 1. Ensure Python 3.10+ and Node.js
python3 --version
node --version

# 2. Install Python dependencies
pip install fastapi uvicorn curl-cffi python-dotenv

# 3. Start
python3 proxy.py
```

## Authentication

Open the admin panel at http://localhost:8000/admin to configure.

### Method 1: Phone/Email Login (Recommended)

The easiest way, same experience as the web login:

1. Select the **Phone** or **Email** tab
2. Enter your phone number (area code defaults to +86) or email
3. Enter your password
4. Click **Login**

The system automatically completes: login to get Token → create chat Session → save config to `token.json` (including password for auto-refresh).

### Method 2: cURL Import

1. Log in to chat.deepseek.com
2. Open **DevTools** → **Network** panel
3. Send a message, find the `completion` request
4. Right-click → **Copy as cURL**
5. In the admin panel, expand **Advanced: Paste cURL** and paste it
6. Click **Save cURL**

### Method 3: Cookie Import

1. Log in to chat.deepseek.com
2. Open **DevTools** → **Application** → **Cookies**
3. Find cookies for `chat.deepseek.com`
4. Export the cookie string containing `userToken`
5. Paste into the admin panel → Save

## API Usage

### 1. List Models

```bash
curl http://localhost:8000/v1/models
```

Returns all dynamically discovered models with `max_input_tokens`, `max_output_tokens`, and other details.

### 2. Non-streaming Chat

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-default",
    "messages": [
      {"role": "user", "content": "Write a quicksort in Python"}
    ]
  }'
```

### 3. Streaming Chat

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-reasoner",
    "messages": [
      {"role": "user", "content": "Explain quantum entanglement"}
    ],
    "stream": true
  }'
```

In streaming responses, thinking content appears in `delta.reasoning_content`, while actual content is in `delta.content`.

### 4. File Upload (Text & Images)

**Text file upload** (supported by all models, no fork, via `ref_file_ids`):

```bash
# Prepare file base64
FILE_B64=$(base64 -w0 notes.txt)

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-default",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "What is this file about?"},
        {"type": "file", "file": {"filename": "notes.txt", "file_data": "'"$FILE_B64"'"}}
      ]
    }]
  }'
```

**Vision image upload** (requires Vision model, fork to vision type after upload):

```bash
# Prepare image base64
IMG_B64=$(base64 -w0 photo.png)

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-vision",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe this image"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,'"$IMG_B64"'"}}
      ]
    }]
  }'
```

> **Note:** Text files do not fork — wait for DeepSeek to finish parsing and reference the original `file_id`. Images must fork to `"vision"` to be readable by Vision models.

### 5. Responses API (OpenAI Compatible)

Supports OpenAI's latest `/v1/responses` endpoint. Non-streaming:

```bash
curl http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek",
    "input": "Write a quicksort in Python",
    "stream": false
  }'
```

Streaming (with full SSE lifecycle events):

```bash
curl http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek",
    "input": "Explain quantum entanglement",
    "stream": true
  }'
```

Events: `response.created` → `response.in_progress` → `response.output_item.added` → `response.content_part.added` → `response.output_text.delta`(chunks) → `response.output_text.done` → `response.content_part.done` → `response.output_item.done` → `response.completed`

Other endpoints (supports streaming replay):

```bash
# Retrieve
curl http://localhost:8000/v1/responses/{response_id}

# Input items
curl http://localhost:8000/v1/responses/{response_id}/input_items

# Cancel
curl -X POST http://localhost:8000/v1/responses/{response_id}/cancel

# Delete
curl -X DELETE http://localhost:8000/v1/responses/{response_id}

# Compact multi-turn conversations
curl -X POST http://localhost:8000/v1/responses/{response_id}/compact \
  -H "Content-Type: application/json" \
  -d '{"instructions": "Please answer all subsequent questions in Chinese"}'
```

Structured Output (json_schema):

```bash
curl http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek",
    "input": "Beijing weather today 25°C, return structured data",
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

> The Responses API supplements the existing `/v1/chat/completions` — both can be used simultaneously.

### 6. Anthropic Messages API

This proxy is fully compatible with the **Anthropic Messages API** format, supporting seamless integration for clients like RikkaHub.

**Authentication**: Use either `x-api-key` header or `Authorization: Bearer`:

```bash
# x-api-key method (recommended)
curl http://localhost:8000/v1/messages \
  -H "x-api-key: sk-dsapi" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-default",
    "max_tokens": 1024,
    "messages": [
      {"role": "user", "content": "Write a quicksort in Python"}
    ]
  }'
```

**Streaming (thinking + text):**

```bash
curl http://localhost:8000/v1/messages \
  -H "x-api-key: sk-dsapi" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-reasoner",
    "max_tokens": 1024,
    "stream": true,
    "messages": [
      {"role": "user", "content": "Explain quantum entanglement"}
    ]
  }'
```

Thinking content streams as `thinking` blocks, text as `text` blocks.

**Available Endpoints:**

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/v1/messages` | Send message (text / thinking) |
| POST | `/v1/messages/count_tokens` | Count tokens |
| GET | `/v1/messages/{id}` | Retrieve sent message |
| POST | `/v1/messages/batches` | Create batch request |
| GET | `/v1/messages/batches` | List batches |
| GET | `/v1/messages/batches/{id}` | Get batch details |
| POST | `.../cancel` | Cancel batch |
| GET | `.../results` | Download batch results |
| DELETE | `/v1/messages/batches/{id}` | Delete batch |

#### Anthropic Model Name Aliases

Tools like Claude Code CLI expect Anthropic-style model names (e.g., `claude-sonnet-4-6`) and cannot directly use `deepseek-*` native names. This proxy automatically maps them internally on Anthropic endpoints:

| Claude Model | → DeepSeek Internal | Thinking | Search |
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

Also supports Claude 4.x legacy names (`claude-sonnet-4-5`, `claude-opus-4-1`, etc.) and `-nothinking` variants. DeepSeek native names (`deepseek-*`) continue to work directly; `/v1/models` still returns native names, not affecting other software.

```bash
# Claude model names work too
curl http://localhost:8000/v1/messages \
  -H "x-api-key: sk-dsapi" \
  -d '{"model":"claude-sonnet-4-6","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}'
```

### 7. Model Refresh

```bash
# Force refresh model list (no need to wait 1-hour cache expiry)
curl -X POST http://localhost:8000/v1/models/refresh
```

## Model System

### Dynamic Model Discovery

At startup, the proxy automatically calls DeepSeek's official API `GET /api/v0/client/settings?scope=model` to obtain currently available model configurations.

Core discovery logic (`proxy.py:418`):

```python
def _discover_models():
    resp = cffi_requests.get(
        "https://chat.deepseek.com/api/v0/client/settings?scope=model",
        headers={"Authorization": f"Bearer {token}", ...}
    )
    # Parse model_configs, generate base/think/search/think+search variants by model_type
```

- **Auto-detect**: No need to manually update model lists
- **1-hour cache**: Avoids frequent requests
- **Manual refresh**: `POST /v1/models/refresh`
- **Fault-tolerant**: Detection failure does not affect cached lists

Each model returns:
- `max_input_tokens` — maximum input tokens
- `max_output_tokens` — maximum output tokens (including thinking)
- `thinking_enabled` — whether deep thinking is supported
- `search_enabled` — whether web search is supported

### Currently Available Models

The model list **changes dynamically with DeepSeek**. Currently 3 base models × 4 variants = 12 models detected:

| Model ID | Name | Description | Thinking | Search |
|---------|------|-------------|:--------:|:------:|
| `deepseek-default` | DeepSeek V4 Flash | V4 Flash fast base model | ✗ | ✗ |
| `deepseek-reasoner` | DeepSeek V4 Flash Thinking | V4 Flash + deep thinking | ✓ | ✗ |
| `deepseek-search` | DeepSeek V4 Flash Search | V4 Flash + web search | ✗ | ✓ |
| `deepseek-reasoner-search` | DeepSeek V4 Flash Think+Search | V4 Flash + think + search | ✓ | ✓ |
| `deepseek-expert` | DeepSeek V4 Pro | V4 Pro expert base model | ✗ | ✗ |
| `deepseek-expert-reasoner` | DeepSeek V4 Pro Thinking | V4 Pro + deep thinking | ✓ | ✗ |
| `deepseek-expert-search` | DeepSeek V4 Pro Search | V4 Pro + web search | ✗ | ✓ |
| `deepseek-expert-reasoner-search` | DeepSeek V4 Pro Think+Search | V4 Pro + think + search | ✓ | ✓ |
| `deepseek-vision` | DeepSeek Vision | Vision base model | ✗ | ✗ |
| `deepseek-vision-reasoner` | DeepSeek Vision Thinking | Vision + deep thinking | ✓ | ✗ |

> **Notes:**
> - If DeepSeek introduces new models, the proxy auto-discovers them — no code changes needed
> - All models explicitly set `model_type` (`default` / `expert` / `vision`) to ensure correct DeepSeek routing
> - Model names are English-only IDs; Chinese names are in the table above

## Tool Calling

DeepSeek's web UI **does not** support OpenAI function calling format. This proxy implements tool calling via **DSML prompt injection + multi-strategy extraction**:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer sk-dsapi" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-chat",
    "messages": [{"role": "user", "content": "What is the weather in Beijing?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get weather information",
        "parameters": {
          "type": "object",
          "properties": {"city": {"type": "string"}},
          "required": ["city"]
        }
      }
    }]
  }'
```

### DSML Prompt Injection

OpenAI tools definitions are converted to DSML format and injected into the system message:

```xml
<|DSML|tool_calls>
  <|DSML|invoke name="search_file">
    <|DSML|parameter name="query"><![CDATA[config.yaml]]></|DSML|parameter>
  </|DSML|invoke>
</|DSML|tool_calls>
```

### Extraction Strategies

| Priority | Format | Description |
|----------|--------|-------------|
| DSML | `<\|DSML\|tool_calls><\|DSML\|invoke name="X">...</\|DSML\|invoke></\|DSML\|tool_calls>` | Primary format, 7 noise variant tolerances |
| TOOL_CALL | `TOOL_CALL: name(key=value)` | Legacy format fallback |
| JSON | `{"name":"x","arguments":{...}}` | JSON block parsing |
| XML | `<tool_call><function=NAME>...</function></tool_call>` | Native XML |
| Mixed | `<function_call>{...}</function_call>` | XML+JSON |

### Fault Tolerance

- **Noise tolerance** — Supports missing pipes, duplicate `<`, fullwidth `｜`, hyphen `dsml-`, and 5 other variants
- **Fenced code blocks** — Automatically skips DSML examples inside markdown code blocks
- **JSON repair** — Auto-fixes unquoted keys, missing array brackets
- **CDATA protection** — content/command/prompt parameters retain original strings
- **Missing open tags** — Auto-restores `<|DSML|tool_calls>` wrapper when only closing tag is present

## Branch Info

This repository provides two branches:

| Branch | Characteristics |
|--------|----------------|
| `main` (current) | Full-featured — supports DSML tool calling, streaming sieve, session management, etc. Use when tool calling is needed |
| `no-tools` | Pure chat proxy — no tool call prompt injection, cleaner output. Ideal for writing, translation, code generation, etc. |

> You are currently on the `main` branch. For pure chat without tools, switch to `no-tools`:
> ```bash
> git checkout main
> ```

## PoW Solver

DeepSeek requires **Proof of Work (PoW)** verification for the `/api/v0/chat/completion` endpoint.

### Flow

1. Before each request, call `POST /api/v0/chat/create_pow_challenge` to obtain a challenge
2. Solve the challenge → get the `x-ds-pow-response` header
3. Attach the solve result to the chat request header

### Dual Solvers

| Solver | Method | Speed | Compatibility |
|--------|--------|-------|---------------|
| Node.js WASM | `node pow_solver.js` subprocess | Fast (seconds) | Algorithm identical to official |
| Python fallback | `hashlib.sha3_256` pure Python | Slower | Fallback when no Node.js |

Requires Node.js installation + `sha3_wasm_bg.wasm` file (included in the project).

### Algorithm

DeepSeek uses a custom `DeepSeekHashV1` algorithm, essentially SHA3-256 hash collision. The WASM version (via Node.js) matches the official algorithm exactly.

## Token Auto-Refresh

Token validity is approximately **24 hours**. When a request returns 401:

1. Detect 401 → trigger `relogin()` function
2. Re-login using saved password via `POST /api/v0/users/login`
3. Obtain new Token → create new Session → save to `token.json`
4. Retry current request with new Token (transparent to user)

> **Prerequisite:** Initial configuration must use **account password login**. Pure cURL/Cookie imports do not include passwords and cannot auto-refresh.

## Management Commands

```bash
# Foreground
python3 proxy.py

# Background start
./deploy.sh --bg

# Check status
./deploy.sh --status

# Stop background process
./deploy.sh --stop

# Watch real-time logs (background mode)
tail -f ~/dsapi.log

# Custom port
PROXY_PORT=9000 python3 proxy.py

# Force refresh model list
curl -X POST http://localhost:8000/v1/models/refresh

# Health check
curl http://localhost:8000/health
```

**After Startup:**

| Address | Description |
|---------|-------------|
| `http://localhost:8000/admin` | Web admin panel (login config) |
| `http://localhost:8000/v1` | OpenAI compatible API root |
| `http://localhost:8000/health` | Health check endpoint |

## Project Structure

```
ds-free-api/
├── proxy.py              # Main: FastAPI app, SSE parser, OpenAI endpoints, admin panel
├── tool_call.py          # Tool calling aggregation (prompt injection, extraction)
├── tool_dsml.py          # DSML parser (prefix stripping, CDATA, structured params)
├── tool_sieve.py         # Streaming tool call sieve (real-time separation)
├── response_store.py     # Responses API local persistence (JSON file)
├── pow_native.py         # PoW solver: Node.js WASM primary + Python fallback
├── pow_solver.js         # Node.js PoW solve script (calls WASM)
├── sha3_wasm_bg.wasm     # SHA3 WASM binary
├── deploy.sh             # One-click deploy script (install deps, start/stop/status)
├── requirements.txt      # Python dependencies
├── token.example.json    # Config file template
└── token.json            # Actual config (.gitignore, contains credentials)
```

### Core File Descriptions

| File | Responsibility | Lines |
|------|---------------|-------|
| `proxy.py` | App entry, routing, SSE parsing, DeepSeek API interaction, token refresh, admin panel UI | ~3770 |
| `response_store.py` | Responses API local persistence (thread-safe JSON read/write) | ~73 |
| `pow_native.py` | PoW solver (Node.js subprocess + Python pure algorithm fallback) | ~124 |
| `deploy.sh` | One-click deploy (env check, dependency install, start/stop/status) | ~198 |

## Configuration Reference

Full `token.json` configuration:

```json
{
  "token": "eyJ...",
  "session_id": "abc-def-123...",
  "headers": {
    "content-type": "application/json",
    "origin": "https://chat.deepseek.com",
    "referer": "https://chat.deepseek.com/",
    "user-agent": "Mozilla/5.0 ...",
    "x-client-version": "2.0.2",
    "x-client-platform": "web",
    "authorization": "Bearer YOUR_TOKEN"
  },
  "account": "+86 138xxxx",
  "login_type": "phone",
  "_password": "your_password",
  "_email": "",
  "_mobile": "138xxxx",
  "_area_code": "+86"
}
```

| Config Item | Description | Auto-Generated |
|-------------|-------------|:--------------:|
| `token` | Bearer Token (~24h validity) | ✓ |
| `session_id` | Chat session ID (UUID) | ✓ |
| `headers` | Request headers (UA, authorization, etc.) | ✓ |
| `account` | Account identifier (display only) | ✓ |
| `login_type` | Login method: `phone` / `email` | On first setup |
| `_password` | Login password (for auto-refresh) | On first setup |
| `_mobile` | Phone number (for auto-refresh) | On first setup |
| `_email` | Email (for auto-refresh) | On first setup |
| `_area_code` | Area code (default +86) | On first setup |

> **Security note:** `_password` is stored in plaintext locally. Ensure `token.json` has proper permissions (`chmod 600`) and is excluded from distribution/packaging (already in `.gitignore`).

**Environment variable:** `PROXY_PORT` — listening port (default `8000`)

## Dependencies

### Python (pip)

```bash
pip install fastapi uvicorn curl-cffi python-dotenv
```

| Dependency | Purpose |
|------------|---------|
| `fastapi` | Web framework |
| `uvicorn` | ASGI server |
| `curl-cffi` | HTTP client (emulates Chrome TLS fingerprint, bypasses anti-bot) |
| `python-dotenv` | Environment variable loading |

### System

- **Node.js** — PoW solver (required, install with `pkg install nodejs` or `apt install nodejs`)
- Python 3.10+ — runtime

## Limitations & Known Issues

| Limitation | Description |
|------------|-------------|
| Token validity | ~24 hours expiry, requires password login for auto-refresh |
| Concurrency limit | DeepSeek free tier limits ~2 concurrent requests per account |
| Chat Completions + Responses only | Embeddings, Fine-tuning, and other endpoints are not supported |
| PoW latency | Each request requires fetching and solving a PoW challenge (Node.js ~1-3 seconds) |
| Non-streaming via SSE | DeepSeek only provides SSE streams; non-streaming requests buffer all SSE events then merge |
| Vision non-streaming | Vision models have no content output in streaming mode; internally uses non-streaming then wraps as SSE |

## FAQ

**Q: Admin page shows blank after startup?**
A: The admin panel is a single-file HTML embedded in `proxy.py`. Check for JavaScript errors (F12 Console). Make sure you access `http://localhost:8000/admin` directly.

**Q: "Update to the latest version to use Expert/Vision" error?**
A: `x-client-version` must match DeepSeek's web UI version (currently `2.0.2`). The proxy sets this automatically at startup.

**Q: PoW solving fails?**
A: Check if Node.js is installed (`node --version`). If Node.js solving fails, the proxy automatically falls back to Python pure algorithm solving (slower but no external dependencies).

**Q: Wrong password error on login?**
A: Verify your password is correct. DeepSeek passwords require at least 8 characters with letters + numbers. You may need to complete a captcha first in some cases.

**Q: What happens when Token expires?**
A: If configured via **account password login**, the proxy auto-relogins and refreshes the Token on 401. If configured via cURL/Cookie import, manual re-import is required.

**Q: Specified expert model but conversation appears in "Quick Mode" (default)?**
A: Usually caused by expired Token or Session. DeepSeek silently downgrades requests to the default model when credentials are invalid. Solution: **re-login** with phone/email on the admin panel at `http://localhost:8000/admin` — the login will auto-refresh both Token and Session.

**Q: Can this be deployed on a public server?**
A: Yes, but recommend using Nginx reverse proxy + HTTPS + IP whitelist. API Key is not validated (any value works), so access control should be handled through other means.

## License & Credits

MIT License

**Reference projects:**
- [NIyueeE/ds-free-api](https://github.com/NIyueeE/ds-free-api) — Rust original, provided the DeepSeek API reverse-engineering approach and PoW algorithm reference
- [CJackHwang/ds2api](https://github.com/CJackHwang/ds2api) — DSML tool calling format, streaming sieve architecture, DeepSeek native dialogue markers
- [GoblinHonest/mimo2api_mimoapi](https://github.com/GoblinHonest/mimo2api_mimoapi) — Session management design reference
- [Acidmoon](https://github.com/Acidmoon) — Submitted PR #2, implementing the OpenAI Responses API compatibility layer
- [xstjmark21-cmyk](https://github.com/xstjmark21-cmyk) — Provided model tokens for Vision feature testing
