## Why

The project already exposes OpenAI-compatible chat completions, but modern OpenAI clients increasingly target the `Responses` API instead of `chat/completions`. Without a `Responses` compatibility layer, this proxy cannot serve those clients even when the underlying DeepSeek-backed behavior is already available.

## What Changes

- Add a `POST /v1/responses` compatibility endpoint that translates Responses-style requests onto the existing DeepSeek chat execution path.
- Add minimal retrieval endpoints for persisted Responses objects and input items.
- Define the supported subset of Responses behavior, especially for streaming events, function calls, reasoning output, and previous-response chaining.
- Preserve reuse of the existing tool-calling, file-handling, and model-routing logic instead of implementing a second independent execution pipeline.

## Capabilities

### New Capabilities

- None.

### Modified Capabilities

- `openai-compatible-api`: extend the public API contract to include Responses endpoints and event shapes.
- `dsml-tool-calling`: define how DSML-extracted tool calls map into Responses function-call output items and stream events.
- `file-and-vision-handling`: define how Responses-style multimodal input is translated into the existing file and image preparation pipeline.

## Impact

- Affects `proxy.py` request routing, request translation, streaming adaptation, and persisted compatibility state.
- Adds a local response persistence layer for `previous_response_id` and retrieval endpoints.
- Extends the externally visible compatibility surface for OpenAI SDK and agent workflows that use the Responses API.
