## Why

The Responses implementation now covers creation, streaming, retrieval, background execution, and runtime operations, but its public response formatting is still assembled across several code paths. That makes structured-output failures, terminal streamed payloads, replayed events, and stored objects vulnerable to small shape differences that break OpenAI-compatible clients.

## What Changes

- Consolidate Responses public object and output-item formatting so non-stream, stream terminal, background completion, compaction, retrieval, and replay paths use the same response-shaping contract.
- Strengthen `text.format` handling for `json_object` and `json_schema` so successful responses contain valid structured JSON and failures are surfaced as OpenAI-style Responses failure objects.
- Preserve requested structured-output metadata on stored and returned response objects while validating the generated `output_text` against the supported local subset.
- Align streamed terminal response payloads and replayed lifecycle events with the same public response format used by `GET /v1/responses/{response_id}`.
- Expand focused smoke coverage for response object fields, output item shapes, structured-output success/failure, and stream/non-stream consistency.

## Capabilities

### New Capabilities

None.

### Modified Capabilities

- `responses-structured-outputs`: Structured-output requests gain concrete JSON validation and failure-state behavior instead of metadata preservation only.
- `openai-compatible-api`: Responses objects gain a consistent public format across sync, stream, background, replay, and retrieval paths.
- `responses-streaming-parity`: Stream terminal events and replay events are required to embed the same public response shape as stored retrieval.

## Impact

- Affected code: `proxy.py`, Responses persistence helpers, and Responses smoke tests.
- Affected APIs: `POST /v1/responses`, streamed Responses SSE events, `GET /v1/responses/{response_id}`, background replay streams.
- Dependencies/systems: no new runtime dependencies; JSON schema validation should use a small local validator subset rather than adding a package.
