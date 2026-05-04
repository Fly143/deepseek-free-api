## MODIFIED Requirements

### Requirement: Deterministic Responses stream sequencing
The proxy SHALL emit monotonically increasing `sequence_number` values for streamed `POST /v1/responses` events and persisted replay streams.

#### Scenario: Assign ordered sequence numbers
- **WHEN** the service emits streamed Responses SSE events
- **THEN** each event includes a `sequence_number`
- **AND** each later event has a greater sequence number than the prior event

#### Scenario: Resume from a stored event cursor
- **WHEN** a client requests replay of a stored Responses stream after a previously seen event cursor
- **THEN** the service emits only events after that cursor
- **AND** the replay preserves the original event ordering

### Requirement: Full Responses lifecycle events
The proxy SHALL emit the core Responses stream lifecycle events for streamed requests.

#### Scenario: Emit lifecycle events in order
- **WHEN** a streamed Responses request begins, produces output, and completes successfully
- **THEN** the service emits `response.created`
- **AND** it emits `response.in_progress`
- **AND** it emits `response.output_item.added` before item-specific delta events
- **AND** it emits `response.content_part.added` and `response.content_part.done` around assistant text output
- **AND** it emits `response.output_text.delta` and `response.output_text.done` for text output
- **AND** it emits `response.reasoning_text.delta` and `response.reasoning_text.done` when reasoning text exists
- **AND** it emits `response.function_call_arguments.delta` and `response.function_call_arguments.done` when tool-call arguments stream
- **AND** it emits `response.output_item.done` before the terminal response event
- **AND** it ends with `response.completed`

#### Scenario: Emit refusal events when applicable
- **WHEN** the streamed response contains a refusal output
- **THEN** the service emits `response.refusal.delta`
- **AND** it emits `response.refusal.done` before the terminal response event

#### Scenario: Emit failure terminal events
- **WHEN** a streamed Responses request fails
- **THEN** the service emits `response.failed`
- **AND** the embedded response payload includes an error object

#### Scenario: Replay a completed background response as a stream
- **WHEN** a locally completed background response is later requested as a stream
- **THEN** the service replays the persisted lifecycle event sequence
- **AND** it ends with the stored terminal event for that response
