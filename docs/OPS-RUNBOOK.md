# Ops notes — DeepSeek Free API bridge

## Fixes from PR #32 (reviewed + maintainer fixes)

1. **Mute/ban detection** — Upstream sometimes returns HTTP 200 + JSON `{ biz_code: 5, biz_msg: "user is muted", mute_until }` instead of SSE. Previously treated as empty completion. Now returns **HTTP 403** `account_muted` with optional `mute_until`. Detection covers nonstream buffered body, bare JSON lines, and SSE `data:` JSON wrappers. Account label is passed explicitly (no shared function attribute).
2. **Account pool skip** — Muted / cooldown accounts skipped by `get_next_account()` (LRU among healthy). Mute persisted via `mark_account_muted`. Hot path does **not** write config.json on every request.
3. **Upstream pacing** — `upstream_slot()`: env knobs `DEEPSEEK_MIN_INTERVAL_SEC` (default 2.5), `DEEPSEEK_MAX_CONCURRENT` (default 1), `DEEPSEEK_ERROR_COOLDOWN_SEC` (default 60). Sleep happens outside the pacing lock.
4. **Search default off** — Forced off unless resolved model id contains `search`. Unknown models fall back to `deepseek-default`, never search.
5. **Context budget** — `context_manager._DEFAULT_MAX_INPUT` raised to 1M; chat path also passes discovered `model_max_in` when available.
6. **Model normalize** — Official names map to bridge non-search ids; thinking toggle switches chat/reasoner.
7. **Python 3.10** — No nested same-quote f-strings (PEP 701 is 3.12+).

## What this does **not** do

No WAF bypass, device spoofing, or mass-account farming. Ban/mute risk of unofficial web reverse proxies is **not** eliminated. Prefer official `https://api.deepseek.com` for real agent workloads.

## Local run (Windows)

```bat
python -u proxy.py
# or
python -u start_bridge8000.py
```

- API: `http://127.0.0.1:8000/v1`
- Admin: `http://127.0.0.1:8000/admin` (Basic Auth — configure locally)
- Optional env: `DEEPSEEK_BRIDGE_PYTHON`, `DEEPSEEK_BRIDGE_ADMIN`, `DEEPSEEK_MIN_INTERVAL_SEC`, `DEEPSEEK_MAX_CONCURRENT`, `DEEPSEEK_ERROR_COOLDOWN_SEC`

## When chat returns empty / 403

1. Check admin → accounts: remove muted accounts from the pool.
2. If 403 `account_muted`, wait until `mute_until` or add another account.
3. If clients burst tool calls, raise `DEEPSEEK_MIN_INTERVAL_SEC` (e.g. 4–5) and keep `DEEPSEEK_MAX_CONCURRENT=1`.

## Credit

PR #32 by [@khs0927](https://github.com/khs0927) — mute detection, account pool skip, pacing, context budget. Maintainer review fixes applied on merge branch.

## Not committed

`config.json`, `sessions.json`, logs, pid files, `*.bak`, local helper scratch scripts.
