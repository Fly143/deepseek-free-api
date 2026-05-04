## ADDED Requirements

### Requirement: Background Responses lifecycle
The proxy SHALL support local background execution for `POST /v1/responses` requests when the client sets `background: true`.

#### Scenario: Create a background response shell
- **WHEN** a client sends `POST /v1/responses` with `background: true`
- **THEN** the service returns a persisted response object without waiting for the DeepSeek-backed completion to finish
- **AND** the returned response has a non-terminal status of `queued` or `in_progress`

#### Scenario: Poll a background response to completion
- **WHEN** a client later requests `GET /v1/responses/{response_id}` for a locally tracked background response
- **THEN** the service returns the latest persisted lifecycle state
- **AND** the response eventually becomes `completed`, `failed`, `incomplete`, or `cancelled`

### Requirement: Background response cancellation
The proxy SHALL support cooperative cancellation for locally managed background responses.

#### Scenario: Cancel a queued or active background response
- **WHEN** a client sends `POST /v1/responses/{response_id}/cancel` for a locally managed background response that is not terminal
- **THEN** the service marks that response as cancellation-requested or cancelled
- **AND** the returned response object reflects a terminal cancelled lifecycle once local execution stops

#### Scenario: Treat repeated cancellation idempotently
- **WHEN** a client sends `POST /v1/responses/{response_id}/cancel` for a response that is already cancelled
- **THEN** the service returns the stored cancelled response object
- **AND** it does not create a second cancellation side effect

### Requirement: Responses input-token counting
The proxy SHALL expose local input-token counting for Responses-compatible inputs.

#### Scenario: Count normalized response input tokens
- **WHEN** a client requests the Responses input-token counting subresource for valid input
- **THEN** the service returns a JSON payload containing the counted input-token total
- **AND** the count is derived from the proxy's normalized Responses input representation

### Requirement: Responses compaction lineage
The proxy SHALL support local response compaction for stored Responses state.

#### Scenario: Compact a stored response
- **WHEN** a client invokes the Responses compaction subresource for a stored response
- **THEN** the service creates a new locally stored compacted response object
- **AND** the compacted object records lineage back to the source response
- **AND** later `previous_response_id` chaining can use the compacted response identifier

### Requirement: Replayable background stream state
The proxy SHALL preserve enough local lifecycle data for completed background responses to be replayed as Responses stream events.

#### Scenario: Replay stored lifecycle events
- **WHEN** a client requests a streamed representation of a previously completed background response
- **THEN** the service replays the persisted Responses lifecycle events in deterministic order
- **AND** later events use larger sequence numbers than earlier events
