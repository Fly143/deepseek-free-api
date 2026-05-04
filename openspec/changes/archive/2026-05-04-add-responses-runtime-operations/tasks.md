## 1. Background Lifecycle

- [x] 1.1 Extend stored response records with internal runtime metadata for queued, in-progress, terminal, and replayable stream state.
- [x] 1.2 Implement `background: true` handling on `POST /v1/responses` so creation returns immediately and completion proceeds through a local background path.
- [x] 1.3 Update `GET /v1/responses/{response_id}` to expose persisted background lifecycle transitions consistently.

## 2. Runtime Operations

- [x] 2.1 Implement `POST /v1/responses/{response_id}/cancel` with cooperative, idempotent cancellation semantics for local background responses.
- [x] 2.2 Implement the Responses input-token counting subresource using the existing normalized input and token-counting utilities.
- [x] 2.3 Implement the Responses compaction subresource with stored lineage metadata and compacted-state reuse for later chaining.

## 3. Stream Replay And Listing Semantics

- [x] 3.1 Persist or reconstruct replayable lifecycle events for completed background responses and support cursor-based replay ordering.
- [x] 3.2 Confirm `GET /v1/responses/{response_id}/input_items` defaults to descending order and preserves pagination metadata under `after` and `before`.

## 4. Validation

- [x] 4.1 Expand Responses smoke coverage for background creation, polling, cancellation, input-token counting, compaction, and replayed streaming.
- [x] 4.2 Run Python syntax validation and `openspec validate --changes --strict`.
