## MODIFIED Requirements

### Requirement: Responses compatibility layer
The proxy SHALL expose `POST /v1/responses` as an OpenAI-compatible Responses subset backed by the same DeepSeek chat execution path.

#### Scenario: Support non-stream Responses objects
- **WHEN** a client sends `POST /v1/responses` without streaming
- **THEN** the service returns a JSON object with `object: "response"`
- **AND** the object includes `output` items and `output_text` derived from the DeepSeek-backed completion

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

#### Scenario: Cancel a background response
- **WHEN** a client sends `POST /v1/responses/{response_id}/cancel`
- **THEN** the service returns a locally stored response object for that identifier
- **AND** the object reflects a cancelled lifecycle when cancellation succeeds

### Requirement: OpenAI-style error shaping for upstream failures
The proxy SHALL convert upstream chat failures into OpenAI-compatible error payloads or HTTP errors rather than returning raw DeepSeek SSE directly.

#### Scenario: Handle streaming upstream error
- **WHEN** the upstream streaming chat request returns a non-200 status after the proxy has started processing a streamed completion
- **THEN** the service emits an SSE error object containing an error message and code
- **AND** the stream still terminates with `data: [DONE]`

#### Scenario: Handle non-stream upstream error
- **WHEN** the upstream chat request fails during a non-stream completion
- **THEN** the service returns an HTTP 502 error payload containing an error message and upstream status code when available

#### Scenario: Handle streamed Responses upstream error
- **WHEN** an error occurs while producing a streamed Responses request
- **THEN** the service emits `response.failed`
- **AND** the embedded response object includes an `error` payload

#### Scenario: Surface incomplete streamed Responses status
- **WHEN** a streamed or non-stream `POST /v1/responses` request ends with an upstream completion reason of `length` or `content_filter`
- **THEN** the resulting response object has `status: "incomplete"`
- **AND** it includes `incomplete_details.reason`
