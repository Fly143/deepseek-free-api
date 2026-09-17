# Ops notes — DeepSeek Free API bridge (local fork)

## Judgment
- Do **not** push secrets or machine-local paths to `Fly143/deepseek-free-api` (no write access; upstream is theirs).
- This fork keeps **generic fixes** + a short runbook. Account emails, tokens, and `config.json` stay local (gitignored).

## Fixes in this fork (2026-09-18)
1. **Mute/ban detection** — Upstream sometimes returns HTTP 200 + JSON `{ biz_code: 5, biz_msg: "user is muted", mute_until }` instead of SSE. Previously the bridge treated that as an empty completion (ZCode/OpenCode looked "dead"). Now returns **HTTP 403** with a clear `account_muted` error and optional `mute_until` (KST).
2. **Context budget** — `context_manager._DEFAULT_MAX_INPUT` raised **64k → 1,048,576** so agent tool schemas do not prune almost the entire chat history.
3. **Model normalize** — Prefer non-search bridge ids by default; `reasoning_effort` / `thinking` still toggles reasoner. (Web-search-forced aliases were unstable and noisier for risk control.)

## Local run (Windows)
```bat
python -u proxy.py
# or
python -u start_bridge8000.py
```
- API: `http://127.0.0.1:8000/v1`
- Admin: `http://127.0.0.1:8000/admin` (Basic Auth — configure locally)
- Optional env: `DEEPSEEK_BRIDGE_PYTHON`, `DEEPSEEK_BRIDGE_ADMIN` (default `admin:admin` only if you leave it unset locally — do not commit real passwords)

## Client wiring (OpenAI-compatible)
Point ZCode / OpenCode / Hermes at `http://127.0.0.1:8000/v1` with one primary model id such as `deepseek-chat`, and use `reasoning_effort` `off`/`on` (or separate `deepseek-reasoner`) as the UI allows. **Do not commit API keys.**

## When chat returns empty / 403
1. Check admin → accounts: remove muted accounts from the pool.
2. If 403 `account_muted`, wait until `mute_until` or add another disposable account in admin.
3. Ban/mute risk of unofficial web reverse proxies is **not** eliminated by PoW/fingerprint. Prefer official `https://api.deepseek.com` for real agent workloads.

## Not committed
`config.json`, `sessions.json`, logs, pid files, `*.bak`, local helper scratch scripts.
