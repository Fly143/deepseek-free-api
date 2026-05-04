## Context

Responses support has grown through several incremental changes. The current code has separate shaping paths for non-stream completions, streamed terminal records, background completions, replay events, compaction records, and retrieval. These paths mostly agree today, but they duplicate response-object and output-item decisions and only lightly handle `text.format`.

The project remains centered on `proxy.py`, so this change should improve consistency with focused helper functions rather than introducing new modules or dependencies. DeepSeek remains the upstream execution source; structured output validation is necessarily local and limited to the proxy-visible generated text.

## Goals / Non-Goals

**Goals:**
- Make every public Responses object pass through a single public-shape normalizer before it is returned or embedded in terminal stream events.
- Make output item builders responsible for stable `id`, `type`, `status`, content, and tool-call fields across sync, stream, replay, and stored records.
- Validate `json_object` and a small useful subset of `json_schema` before returning successful structured-output Responses.
- Store structured-output failures with `status: failed`, an OpenAI-style error payload, and replayable terminal state.
- Keep stream terminal events and replay events consistent with `GET /v1/responses/{response_id}`.

**Non-Goals:**
- No full JSON Schema implementation or new `jsonschema` dependency.
- No upstream-native structured-output support, because DeepSeek execution is still prompt-backed.
- No broad rewrite of the chat completions surface.
- No change to the local runtime operation semantics implemented in the previous change.

## Decisions

### 1. Centralize public response normalization inside `proxy.py`

Responses records will still be stored as dicts, but public returns should pass through a formatter that fills expected nullable fields and strips internal keys. This preserves the single-file architecture while reducing shape drift.

Alternative considered: create a dedicated model layer or Pydantic models. Rejected because the existing code uses dict shaping throughout and a model layer would be larger than the current compatibility need.

### 2. Treat structured-output validation as a terminal record transformation

The proxy will first build the completion record from upstream output, then apply structured-output validation. If validation passes, the normalized JSON text is retained. If validation fails, the record becomes `failed` with a structured error while preserving request metadata and stored input.

Alternative considered: reject requests before upstream execution. Rejected because the failure condition usually depends on the generated output, not only the requested format.

### 3. Implement only a local JSON Schema subset

The validator should cover common object-schema behavior: `type`, `properties`, `required`, `additionalProperties`, arrays, primitive scalar types, and nested object validation. Unsupported schema keywords should be ignored instead of causing false failures.

Alternative considered: add a `jsonschema` dependency. Rejected to keep runtime dependencies unchanged and avoid broad packaging impact.

### 4. Reuse persisted events when available, rebuild with the normalized public payload otherwise

Replay should continue to use stored lifecycle events for background responses. When events are rebuilt from a final record, the terminal event should embed the same public record returned by retrieval.

Alternative considered: keep stream and replay formatting separate. Rejected because it leaves the compatibility risk this change is intended to remove.

## Risks / Trade-offs

- [Risk] The local JSON Schema subset can differ from OpenAI's full validator. -> Mitigation: document the supported subset in specs and ignore unsupported keywords rather than over-promising.
- [Risk] Changing terminal failure behavior for invalid structured output may expose failures that were previously successful text responses. -> Mitigation: only apply this when clients explicitly request `json_object` or `json_schema`.
- [Risk] Refactoring output shaping could accidentally change chat-completions behavior. -> Mitigation: scope changes to Responses helpers and keep existing chat routes untouched.
- [Risk] Stored records from older versions may lack newer fields. -> Mitigation: public normalization should fill missing fields defensively at retrieval/replay time.
