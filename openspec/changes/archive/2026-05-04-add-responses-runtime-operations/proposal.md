## Why

The proxy now covers the basic `POST /v1/responses` compatibility path, but it still misses several operational behaviors that OpenAI clients expect from the Responses API. In particular, background execution, cancellation, input-token counting, and response compaction are part of the official Responses workflow and are the next highest-value gaps after basic create/retrieve/stream parity.

## What Changes

- Add Responses background execution semantics for `background=true`, including persisted lifecycle states that can be polled through `GET /v1/responses/{response_id}`.
- Add `POST /v1/responses/{response_id}/cancel` for locally tracked background responses with idempotent terminal behavior.
- Add Responses context-management subresources for input-token counting and response compaction.
- Align stored input-item listing semantics with the OpenAI Responses pagination contract, including the default descending order.
- Define which parts of the official runtime workflow are supported locally versus emulated on top of the existing DeepSeek-backed synchronous execution path.

## Capabilities

### New Capabilities
- `responses-runtime-operations`: Background lifecycle, cancellation, input-token counting, and compaction behavior for locally stored Responses objects.

### Modified Capabilities
- `openai-compatible-api`: Extend Responses endpoint behavior to cover background request creation, polling-visible status transitions, and cancellation semantics.
- `responses-state-parity`: Refine stored input-item listing and local response-state transitions to match OpenAI Responses runtime expectations.
- `responses-streaming-parity`: Clarify streamed behavior for background-created responses and cursor-oriented sequence tracking.

## Impact

- Affected code: `proxy.py`, `response_store.py`, and the Responses smoke/integration test surface.
- Affected APIs: `POST /v1/responses`, `GET /v1/responses/{response_id}`, `GET /v1/responses/{response_id}/input_items`, plus new Responses subresource endpoints.
- Dependencies/systems: local JSON response persistence, token counting logic, and streamed event bookkeeping.
