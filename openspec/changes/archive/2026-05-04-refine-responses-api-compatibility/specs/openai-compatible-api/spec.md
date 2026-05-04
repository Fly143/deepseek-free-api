## MODIFIED Requirements

### Requirement: Chat completions entrypoint
The proxy SHALL expose OpenAI-compatible text-generation endpoints backed by DeepSeek chat completion requests, including both `POST /v1/chat/completions` and a Responses compatibility layer at `POST /v1/responses`.

#### Scenario: Emit richer streamed Responses lifecycle events
- **WHEN** a client sends `POST /v1/responses` with streaming enabled
- **THEN** the service emits `response.created` followed by `response.in_progress`
- **AND** it emits `response.output_item.added` before item-specific delta events
- **AND** it emits `response.output_item.done` before the terminal response event

#### Scenario: Expose content-part lifecycle for streamed assistant text
- **WHEN** a streamed `POST /v1/responses` request yields assistant text output
- **THEN** the service emits `response.content_part.added` before `response.output_text.delta`
- **AND** it emits `response.content_part.done` after `response.output_text.done`

#### Scenario: Delete a stored response object
- **WHEN** a client sends `DELETE /v1/responses/{response_id}` for a locally stored response
- **THEN** the service returns a JSON object with `deleted: true`
- **AND** subsequent retrieval for that identifier returns HTTP 404 with an OpenAI-style invalid-request error payload

### Requirement: OpenAI-style error shaping for upstream failures
The proxy SHALL convert upstream chat failures into OpenAI-compatible error payloads or response-failure events rather than returning raw DeepSeek SSE directly.

#### Scenario: Surface incomplete streamed Responses status
- **WHEN** a streamed or non-stream `POST /v1/responses` request ends with an upstream completion reason of `length` or `content_filter`
- **THEN** the resulting response object has `status: "incomplete"`
- **AND** it includes `incomplete_details.reason`
