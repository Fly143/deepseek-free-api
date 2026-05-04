## MODIFIED Requirements

### Requirement: Responses compatibility layer
The proxy SHALL expose `POST /v1/responses` as an OpenAI-compatible Responses subset backed by the same DeepSeek chat execution path, and SHALL keep public response objects consistently shaped across creation, retrieval, background completion, replay, and streamed terminal events.

#### Scenario: Support non-stream Responses objects
- **WHEN** a client sends `POST /v1/responses` without streaming
- **THEN** the service returns a JSON object with `object: "response"`
- **AND** the object includes `output` items and `output_text` derived from the DeepSeek-backed completion
- **AND** the object includes stable public lifecycle, model, metadata, usage, error, and incomplete-detail fields

#### Scenario: Support streamed Responses events
- **WHEN** a client sends `POST /v1/responses` with streaming enabled
- **THEN** the service emits a Responses-style SSE sequence
- **AND** the sequence includes `response.created`
- **AND** the sequence includes `response.in_progress`
- **AND** streamed output items are announced with `response.output_item.added`
- **AND** completed output items are finalized with `response.output_item.done`
- **AND** the sequence ends with either `response.completed` or `response.failed`

#### Scenario: Expose content-part lifecycle for streamed assistant text
- **WHEN** a streamed `POST /v1/responses` request yields assistant text output
- **THEN** the service emits `response.content_part.added` before `response.output_text.delta`
- **AND** it emits `response.content_part.done` after `response.output_text.done`

#### Scenario: Persist response objects for later retrieval
- **WHEN** the proxy returns a Responses object
- **THEN** it persists enough local state to support `GET /v1/responses/{response_id}` and `GET /v1/responses/{response_id}/input_items`

#### Scenario: Retrieve public response shape consistently
- **WHEN** a client retrieves a stored response using `GET /v1/responses/{response_id}`
- **THEN** the returned object uses the same public response field contract as the creation response
- **AND** it does not expose internal metadata keys

#### Scenario: Continue a stored response conversation
- **WHEN** a client sends `POST /v1/responses` with a known `previous_response_id`
- **THEN** the proxy prepends the stored conversation context from that prior response before forwarding the next turn upstream

#### Scenario: Reject unknown previous response
- **WHEN** a client sends `POST /v1/responses` with a `previous_response_id` that is not stored locally
- **THEN** the proxy returns HTTP 404 with an OpenAI-style invalid-request error payload

#### Scenario: Delete a stored response object
- **WHEN** a client sends `DELETE /v1/responses/{response_id}` for a locally stored response
- **THEN** the service returns a JSON object with `deleted: true`
- **AND** subsequent retrieval for that identifier returns HTTP 404 with an OpenAI-style invalid-request error payload

#### Scenario: Create background responses
- **WHEN** a client sends `POST /v1/responses` with `background: true`
- **THEN** the service returns an immediately persisted response object
- **AND** the object has a non-terminal background lifecycle status rather than waiting for the final completion
- **AND** later polling returns the same public response field contract with updated lifecycle state

#### Scenario: Cancel a background response
- **WHEN** a client sends `POST /v1/responses/{response_id}/cancel`
- **THEN** the service returns a locally stored response object for that identifier
- **AND** the object reflects a cancelled lifecycle when cancellation succeeds
