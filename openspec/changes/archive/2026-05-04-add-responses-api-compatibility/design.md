## Context

The current codebase has one real execution pipeline: `POST /v1/chat/completions` converts OpenAI-style messages into a DeepSeek prompt, forwards the request upstream, and adapts SSE or buffered output back into OpenAI chat-completion responses. Tool calling, reasoning output, file upload, image preparation, token refresh, and session renewal already hang off this path.

Adding `Responses` support should not fork the business logic into a second upstream execution stack. The practical design is to translate Responses requests into the chat-completions shape the proxy already knows how to execute, then translate the resulting chat-completions output into Responses objects and streaming events.

One extra requirement is `previous_response_id`, because Responses clients may treat the server as stateful. The current proxy is mostly stateless apart from token/session and usage JSON, so a small local persistence layer is needed to make response chaining and retrieval work within one proxy instance.

## Goals / Non-Goals

**Goals:**
- Expose a usable `Responses` compatibility subset on top of the existing chat execution path.
- Support both non-stream and stream modes with OpenAI-style response events.
- Support `previous_response_id` within the proxy by persisting enough local state to reconstruct the next turn.
- Reuse the existing tool-calling and file/image preparation pipeline rather than reimplementing it.

**Non-Goals:**
- Perfectly reproduce every field, tool type, and advanced lifecycle behavior from the full OpenAI Responses API.
- Implement OpenAI-hosted tools such as web search or code interpreter semantics.
- Add provider-agnostic conversation state beyond this proxy's local persistence.
- Rework the upstream DeepSeek protocol or split `proxy.py` as part of this change.

## Decisions

Translate Responses input into chat-completions messages.
Rationale: the proxy already has stable handling for messages, tools, text files, images, reasoning, and function calls on the chat-completions path.

Persist response records locally in a JSON store.
Rationale: `previous_response_id`, `GET /v1/responses/{id}`, and `GET /v1/responses/{id}/input_items` need local server state, and the repository already uses small JSON stores for similar operational concerns.

Implement a supported subset, not the full Responses surface.
Rationale: the goal is compatibility for common SDK flows, not a full reimplementation of OpenAI backend semantics the upstream DeepSeek service cannot natively provide.

Map DeepSeek reasoning output into Responses reasoning items and reasoning-text stream events.
Rationale: the existing proxy already exposes `reasoning_content`; Responses clients expect reasoning information through the newer object and stream format.

Map DSML-derived tool calls into Responses function-call output items.
Rationale: function-calling compatibility is already synthetic in this project, so the Responses layer should preserve that synthetic behavior in the newer output format.

## Risks / Trade-offs

[Risk] Responses clients may depend on fields or event variants not implemented in the supported subset.
Mitigation: define the subset explicitly in specs and keep the object shape stable for supported flows.

[Risk] Local `previous_response_id` state can be lost across file deletion, repo resets, or instance changes.
Mitigation: treat response chaining as proxy-local state and persist enough conversation context in `responses.json`.

[Risk] Reusing chat-completions translation may leak chat-specific assumptions into Responses semantics.
Mitigation: keep translation logic isolated and specify the supported mappings clearly.

[Risk] Stream event ordering may diverge from official OpenAI behavior in edge cases.
Mitigation: support the core event family that most SDK consumers rely on: created, text delta/done, reasoning delta/done, function-call argument delta/done, completed, failed.

## Migration Plan

1. Add a small local response store for persisted Responses compatibility state.
2. Add request translation from Responses input to chat-completions messages and tools.
3. Add `POST /v1/responses` non-stream and stream adapters.
4. Add retrieval endpoints for stored response objects and input items.
5. Validate Python syntax and OpenSpec artifacts.

## Open Questions

- Whether a future phase should expose deletion or listing endpoints for stored Responses objects.
- Whether unsupported tool types should remain ignored or become explicit request-validation errors.
