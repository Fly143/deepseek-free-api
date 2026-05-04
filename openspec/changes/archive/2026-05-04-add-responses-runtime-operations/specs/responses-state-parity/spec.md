## MODIFIED Requirements

### Requirement: Ordered Responses input retrieval
The proxy SHALL return stored Responses input items in stable submission order while honoring the Responses list ordering contract.

#### Scenario: Return items in stored order
- **WHEN** a client requests `GET /v1/responses/{response_id}/input_items`
- **THEN** the service returns the normalized stored input items in the same order they were persisted before list-order transforms are applied

### Requirement: Paginated input item listing
The proxy SHALL support paginated retrieval of stored Responses input items.

#### Scenario: Support list window parameters
- **WHEN** a client supplies `limit`, `after`, `before`, or `order` on `GET /v1/responses/{response_id}/input_items`
- **THEN** the service returns only the requested slice of the stored input items
- **AND** it preserves `first_id`, `last_id`, and `has_more` metadata on the list response

#### Scenario: Default to descending item order
- **WHEN** a client omits `order` on `GET /v1/responses/{response_id}/input_items`
- **THEN** the service returns the newest stored input item first

#### Scenario: Preserve retrieval semantics after storage changes
- **WHEN** a stored response has been deleted or is otherwise missing
- **THEN** the service returns an OpenAI-style invalid-request error payload for input-item retrieval

## ADDED Requirements

### Requirement: Local response runtime state persistence
The proxy SHALL persist enough private state to represent local background and compaction lifecycles beyond the public response payload.

#### Scenario: Persist background lifecycle metadata
- **WHEN** a response is created with local background execution
- **THEN** the stored record includes internal lifecycle metadata sufficient to resume polling and terminal-state resolution after creation

#### Scenario: Persist compaction lineage metadata
- **WHEN** a response is locally compacted
- **THEN** the stored compacted record preserves lineage back to the source response record
