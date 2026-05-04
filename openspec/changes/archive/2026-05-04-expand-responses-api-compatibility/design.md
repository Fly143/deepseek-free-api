## Context

The current Responses implementation is already a thin compatibility layer over `/v1/chat/completions`, with local persistence for response records and input items. The next gap is not a second execution path, but better contract coverage for the OpenAI Responses surface that client SDKs increasingly expect.

## Goals / Non-Goals

**Goals:**
- Preserve the single DeepSeek execution pipeline while expanding Responses compatibility.
- Support structured-output requests, richer streamed lifecycle events, native tool shapes, and paginated input retrieval.
- Keep the contract testable at the HTTP boundary and durable through the existing local JSON store.

**Non-Goals:**
- Reimplement the full OpenAI Responses backend or hosted tools.
- Add a separate upstream execution stack or provider-specific agent runtime.
- Guarantee perfect schema enforcement beyond what the upstream model can actually produce.

## Decisions

- Continue translating Responses requests into the existing chat-completions path.
  Alternative: add a second execution stack. Rejected because it would duplicate all DeepSeek-specific handling and increase drift.

- Treat structured outputs as a preserved contract plus best-effort JSON shaping.
  Alternative: strict schema enforcement. Rejected because upstream DeepSeek output is not guaranteed to satisfy OpenAI schema semantics deterministically.

- Add server-owned sequence numbers for streamed Responses events.
  Alternative: leave ordering implicit. Rejected because client SDK compatibility depends on a stable event order contract.

- Normalize both flat and nested function-tool definitions into the existing DSML tool pipeline.
  Alternative: build a new tool executor. Rejected because tool calling here is synthetic already and should stay centralized.

- Keep input-item pagination in the local response store.
  Alternative: derive pagination from in-memory state only. Rejected because retrieval must survive request boundaries and restarts.

## Risks / Trade-offs

- [Risk] Structured outputs may remain best-effort rather than fully schema-enforced.
  Mitigation: preserve the requested format metadata and fail explicitly when the output cannot be represented cleanly.

- [Risk] Stream event ordering may still diverge from OpenAI in edge cases.
  Mitigation: use deterministic sequencing and test the emitted event contract directly.

- [Risk] Local pagination state can become stale after deletion or file loss.
  Mitigation: document the state as proxy-local and keep retrieval semantics consistent with the local store.

- [Risk] Unsupported tool types still cannot behave like OpenAI-hosted tools.
  Mitigation: scope this change to function-tool interoperability and keep the unsupported surface explicit.
