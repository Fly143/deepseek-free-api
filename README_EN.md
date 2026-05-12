# DeepSeek Free API Proxy

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![FastAPI](https://img.shields.io/badge/FastAPI-teal)](https://fastapi.tiangolo.com/)

Reverse-engineer **DeepSeek web chat** (chat.deepseek.com) into an **OpenAI-compatible API**. Features dynamic model discovery, automatic PoW solving, token refresh, and tool calling support.

本项目所修改代码均为ai完成，不含任何一句人工代码，望周知！

> 📖 [中文版本](README.md)

[zhangjiabo522](https://github.com/zhangjiabo522) — Thanks for providing model tokens for Vision feature testing!

> **💡 Need pure chat without tools?** Use the [`no-tools` branch](#no-tools-branch) — no tool prompt injection, cleaner context, higher output quality.

> **Reference project:** [NIyueeE/ds-free-api](https://github.com/NIyueeE/ds-free-api) (Rust). This is a Python rewrite using pure HTTP forwarding (curl_cffi with Chrome TLS fingerprint) instead of browser automation.

## Features

- **OpenAI Compatible** — `/v1/chat/completions` (streaming/non-streaming), `/v1/models`, `/v1/responses`
- **Multilingual Web Admin** — Chinese/English UI with one-click language toggle
- **Dynamic Model Discovery** — Real-time model list fetched from DeepSeek API at startup
- **Automatic PoW Solving** — Pure Python solver, no WASM dependency required
- **Token Auto-Refresh** — Automatically re-login with saved credentials on 401
- **Deep Thinking** — Reasoning content separated as `reasoning_content` in streaming
- **Vision & File Upload** — Image understanding + text file chat via `ref_file_ids`
- **Web Search** — `search_enabled` support for search model variants
- **Pure HTTP Solution** — No browser/Playwright/Chrome, uses curl_cffi for TLS fingerprint
- **CORS Enabled** — Cross-origin requests supported for web clients

## Quick Start

```bash
git clone https://github.com/Fly143/deepseek-free-api.git
cd deepseek-free-api
chmod +x deploy.sh
./deploy.sh
```

Then open: **http://localhost:8000/admin**

## Configuration

Open the admin panel at http://localhost:8000/admin and choose:

1. **Phone/Email Login** (recommended) — Enter credentials, auto-fetches token + session
2. **cURL Import** — Copy `completion` request from browser DevTools → Network
3. **Cookie Import** — Export `userToken` cookie from chat.deepseek.com

## API Usage

### List Models
```bash
curl http://localhost:8000/v1/models
```

### Chat Completions (non-streaming)
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-default",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Chat Completions (streaming)
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-reasoner",
    "messages": [{"role": "user", "content": "Explain quantum entanglement"}],
    "stream": true
  }'
```

### Vision / File Upload
```bash
# Text file upload
FILE_B64=$(base64 -w0 notes.txt)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-default",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "Summarize this file"},
        {"type": "file", "file": {"filename": "notes.txt", "file_data": "'"$FILE_B64"'"}}
      ]
    }]
  }'

# Image upload (Vision models)
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

### Anthropic Messages API
```bash
curl http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: any-value" \
  -d '{
    "model": "claude-sonnet-4-6",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

Supports Claude model name aliases (`claude-sonnet-4-6` → `deepseek-reasoner`, etc.).

## Model System

Models are dynamically discovered from DeepSeek's settings API. Each base model type generates 4 variants (base, reasoner, search, reasoner-search):

| Base Model | Description | Context |
|-----------|-------------|---------|
| deepseek-default | V4 Flash (fast) | 1M tokens |
| deepseek-expert | V4 Pro (powerful) | 1M tokens |
| deepseek-vision | Vision model | 1M tokens |

## no-tools Branch

For pure chat scenarios without tool calling:

```bash
git clone -b no-tools https://github.com/Fly143/deepseek-free-api.git
```

The `no-tools` branch removes all tool calling logic and DSML prompt injection, giving cleaner output for writing, translation, Q&A, and coding.

## Project Structure

| File | Purpose |
|------|---------|
| `proxy.py` | Main proxy server (FastAPI + admin UI) |
| `app/config.py` | Multi-account manager with round-robin |
| `app/anthropic.py` | Anthropic Messages API format conversion |
| `app/anthropic_routes.py` | Anthropic API endpoints |
| `app/batch.py` | Batch processing & message storage |
| `tool_call.py` | Tool calling aggregation |
| `tool_dsml.py` | DSML parser for tool calls |
| `tool_sieve.py` | Streaming tool call sieve |
| `pow_native.py` | Pure Python PoW solver |
| `session_store.py` | Session & token tracking |
| `usage_store.py` | Usage statistics persistence |

## Dependencies

- Python 3.10+
- Node.js (for WASM PoW solver)
- curl_cffi, FastAPI, uvicorn, tiktoken

## FAQ

**Q: Is an API key needed?**  
A: No. The proxy handles authentication with your DeepSeek account credentials.

**Q: Does it support multiple accounts?**  
A: Yes. Add multiple accounts in the admin panel for load balancing.

**Q: Why use no-tools branch?**  
A: If you don't need function calling, the no-tools branch gives cleaner context and better output quality.

**Q: How are models discovered?**  
A: At startup, the proxy fetches model configuration from DeepSeek's settings API and generates all available variants.

## Credits & License

MIT License. This project is a Python rewrite inspired by [NIyueeE/ds-free-api](https://github.com/NIyueeE/ds-free-api).

Thanks to [zhangjiabo522](https://github.com/zhangjiabo522) for Vision testing support.
