## Context

The current Responses implementation is intentionally lightweight: `POST /v1/responses` reuses the existing chat execution path, stores a finished response record in `responses.json`, and exposes retrieval, deletion, and input-item listing. That is enough for basic synchronous compatibility, but it does not yet cover the operational lifecycle that many Responses clients expect around background work, polling, cancellation, and context-management subresources.

The repository is still centered on `proxy.py` plus a small JSON-backed `response_store.py`. That single-file architecture is a strength here: the next runtime operations should remain local and explicit rather than introducing queues, databases, or worker systems that would be disproportionate to the project.

## Goals / Non-Goals

**Goals:**
- Add a local background-execution mode for `POST /v1/responses` with persisted status transitions that can be polled.
- Add cancellation semantics for locally tracked background responses.
- Add local Responses subresources for input-token counting and response compaction.
- Clarify stream recovery behavior for stored background responses so clients can resume from persisted sequence state.
- Preserve the current DeepSeek-backed synchronous execution core and avoid broad architectural churn.

**Non-Goals:**
- No distributed job queue, process supervisor, or multi-worker orchestration.
- No attempt to mirror every OpenAI built-in tool or Conversations API feature in this change.
- No upstream asynchronous DeepSeek execution, because the upstream path still fundamentally executes as a proxied chat request.
- No durable event log outside the existing JSON persistence model.

## Decisions

### 1. Background mode will be implemented as local asynchronous completion over the existing synchronous chat path

When `background=true`, the proxy will create and persist a response shell immediately, return a queued or in-progress response object, and complete the actual DeepSeek-backed execution in a local background task. This keeps behavior compatible with polling-oriented clients without pretending that DeepSeek itself offers a matching background API.

Why this approach:
- It reuses the existing `chat()` and Responses-shaping logic.
- It fits FastAPI's in-process background execution model.
- It keeps the current repo simple and avoids new infrastructure.

Alternative considered:
- A real job queue or external worker.
  Rejected because it adds operational complexity far beyond the current project scope.

### 2. Stored response records will carry explicit runtime metadata in addition to public response fields

The public response object is not enough to support background lifecycle recovery, cancellation, compaction lineage, or resumable stream playback. The store should gain internal metadata such as:
- private execution status
- cancellation requested flag
- persisted stream events or replayable output snapshots
- lineage fields for compacted responses
- timestamps for queued, started, completed, cancelled

Why this approach:
- `response_store.py` already stores raw dict records and can evolve without schema migrations beyond best-effort defaults.
- It lets the public API stay OpenAI-shaped while internal execution details remain local.

Alternative considered:
- Derive runtime state only from the public payload.
  Rejected because it cannot represent cancellation intent, replay cursors, or compaction provenance cleanly.

### 3. Cancellation will be local, cooperative, and idempotent

`POST /v1/responses/{response_id}/cancel` will only affect locally managed background responses. If execution has not started, the response moves directly to `cancelled`. If execution is already in flight, the proxy marks cancellation requested and returns a cancelled or cancelling-compatible terminal view once the local execution loop reaches a safe boundary.

Why this approach:
- The upstream DeepSeek flow is a single proxied request, so hard remote preemption is not reliably available.
- Clients mainly need a consistent contract and terminal state, not true upstream kill semantics.

Alternative considered:
- Expose cancel only as a best-effort no-op.
  Rejected because it creates a misleading API surface with poor observability.

### 4. Input-token counting and compaction will be local transformations over stored input state

The repository already has `tiktoken`-based counting utilities and normalized stored input items. `input_tokens` can therefore be served by recounting the stored or supplied input shape locally. `compact` can be implemented as a local reduction step that produces a new stored response derived from an earlier response lineage, using persisted normalized inputs and outputs rather than any upstream-native compaction feature.

Why this approach:
- It is achievable with current project dependencies.
- It aligns with the proxy's role as a compatibility layer.

Alternative considered:
- Omit these endpoints until upstream-native support exists.
  Rejected because these are compatibility features that can be emulated locally with acceptable fidelity.

### 5. Stream resume will replay persisted lifecycle events from local state instead of reconstructing the upstream SSE

Background-created responses that are later streamed should emit a replayable Responses event sequence from stored state, optionally filtered by a client cursor such as `starting_after`. The source of truth is the locally persisted event list or an equivalent replay snapshot.

Why this approach:
- The upstream SSE is not available after the initial request completes.
- Local replay is deterministic and testable.

Alternative considered:
- Regenerate events from the final response object only.
  Rejected because it loses event boundaries and cursor semantics.

## Risks / Trade-offs

- [In-process background execution can be lost on process restart] → Persist intermediate status early and treat orphaned in-progress jobs as failed or cancelled-recovery states on next access.
- [Cancellation cannot guarantee upstream preemption] → Document cancellation as local cooperative termination and ensure the public status is consistent.
- [Persisting replayable stream events grows the JSON store] → Keep event payloads minimal and allow compaction to reduce long-lived lineage size.
- [Compaction fidelity may differ from OpenAI internals] → Specify the externally visible contract in terms of lineage and reduced input state, not exact summarization internals.
- [Single-file implementation can become harder to navigate] → Extract only narrowly scoped helpers if the runtime logic becomes too dense inside `proxy.py`.
