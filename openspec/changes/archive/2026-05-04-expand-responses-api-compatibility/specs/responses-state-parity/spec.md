# responses-state-parity Specification

## Purpose
Define retrieval behavior for stored Responses input items and their pagination contract.

## ADDED Requirements

### Requirement: Ordered Responses input retrieval
The proxy SHALL return stored Responses input items in stable submission order.

#### Scenario: Return items in stored order
- **WHEN** a client requests `GET /v1/responses/{response_id}/input_items`
- **THEN** the service returns the normalized stored input items in the same order they were persisted

### Requirement: Paginated input item listing
The proxy SHALL support paginated retrieval of stored Responses input items.

#### Scenario: Support list window parameters
- **WHEN** a client supplies `limit`, `after`, `before`, or `order` on `GET /v1/responses/{response_id}/input_items`
- **THEN** the service returns only the requested slice of the stored input items
- **AND** it preserves `first_id`, `last_id`, and `has_more` metadata on the list response

#### Scenario: Preserve retrieval semantics after storage changes
- **WHEN** a stored response has been deleted or is otherwise missing
- **THEN** the service returns an OpenAI-style invalid-request error payload for input-item retrieval
