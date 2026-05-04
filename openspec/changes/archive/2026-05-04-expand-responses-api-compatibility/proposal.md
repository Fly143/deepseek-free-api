## Why

The baseline Responses layer covers create, retrieve, delete, and the common function-call loop, but SDKs still rely on newer contract details for structured outputs, stream sequencing, tool interop, and paginated state access. Those gaps block higher-fidelity OpenAI client compatibility even though the underlying DeepSeek chat path is already usable.

## What Changes

- Add structured-output handling for `text.format` requests such as `json_object` and `json_schema`, and persist the chosen format on stored response objects.
- Expand streamed Responses events with deterministic sequencing and the missing lifecycle events expected by modern SDKs.
- Normalize native Responses tool definitions and `function_call_output` items into the existing DeepSeek-backed tool pipeline.
- Add paginated retrieval for stored Responses input items so state can be consumed incrementally instead of only as a full list.
- Keep the single upstream execution pipeline; extend translation and response shaping rather than introducing a second DeepSeek integration path.

## Capabilities

### New Capabilities
- `responses-structured-outputs`: preserve and surface supported Responses structured-output formats.
- `responses-streaming-parity`: emit the fuller Responses stream lifecycle with stable sequencing.
- `responses-tool-interoperability`: accept native Responses tool definitions and function-call output items.
- `responses-state-parity`: support paginated retrieval of stored Responses input items.

### Modified Capabilities
None.

## Impact

`proxy.py`, `response_store.py`, Responses request normalization, streamed event shaping, HTTP compatibility behavior, and API test coverage.
