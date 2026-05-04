## MODIFIED Requirements

### Requirement: Chat completions entrypoint
The proxy SHALL expose OpenAI-compatible text-generation endpoints backed by DeepSeek chat completion requests, including both `POST /v1/chat/completions` and a Responses compatibility layer at `POST /v1/responses`.

#### Scenario: Require prior configuration
- **WHEN** a client calls `POST /v1/chat/completions` or `POST /v1/responses` before local credentials have been configured
- **THEN** the service returns HTTP 503
- **AND** the error instructs the operator to configure the account through the local admin flow

#### Scenario: Map requests onto DeepSeek prompt format
- **WHEN** a configured client submits chat messages or Responses input to the proxy
- **THEN** the service converts the request into the DeepSeek prompt markers expected by the upstream chat endpoint
- **AND** the upstream request includes the resolved DeepSeek `model_type`

#### Scenario: Preserve chat-completions stream behavior
- **WHEN** a client sets `stream: true` on `POST /v1/chat/completions`
- **THEN** the service responds as an SSE stream using OpenAI-style `chat.completion.chunk` objects
- **AND** the stream terminates with `data: [DONE]`

#### Scenario: Preserve chat-completions non-stream behavior
- **WHEN** a client omits `stream` or sets it to false on `POST /v1/chat/completions`
- **THEN** the service internally consumes the upstream SSE response
- **AND** the client receives a single JSON `chat.completion` response

#### Scenario: Support non-stream Responses objects
- **WHEN** a client sends `POST /v1/responses` without streaming
- **THEN** the service returns a JSON object with `object: "response"`
- **AND** the object includes `output` items and `output_text` derived from the DeepSeek-backed completion

#### Scenario: Support streamed Responses events
- **WHEN** a client sends `POST /v1/responses` with streaming enabled
- **THEN** the service emits a Responses-style SSE sequence
- **AND** the sequence includes `response.created`
- **AND** the sequence ends with either `response.completed` or `response.failed`

#### Scenario: Persist response objects for later retrieval
- **WHEN** the proxy returns a Responses object
- **THEN** it persists enough local state to support `GET /v1/responses/{response_id}` and `GET /v1/responses/{response_id}/input_items`

### Requirement: Reasoning content compatibility
The proxy SHALL surface DeepSeek thinking output through OpenAI-compatible reasoning fields for both chat-completions and Responses clients when the selected model variant enables thinking.

#### Scenario: Emit reasoning content in chat-completions streaming responses
- **WHEN** the upstream SSE includes thinking fragments for a thinking-enabled model on `POST /v1/chat/completions`
- **THEN** the service emits them in `choices[0].delta.reasoning_content`

#### Scenario: Emit reasoning content in chat-completions non-stream responses
- **WHEN** a non-stream chat-completions response accumulates thinking output for a thinking-enabled model
- **THEN** the final assistant message includes a `reasoning_content` field

#### Scenario: Emit reasoning output in non-stream Responses objects
- **WHEN** a non-stream `POST /v1/responses` request accumulates thinking output
- **THEN** the returned response object includes a reasoning output item alongside any assistant text output

#### Scenario: Emit reasoning stream events for Responses clients
- **WHEN** a streamed `POST /v1/responses` request receives thinking output
- **THEN** the service emits `response.reasoning_text.delta`
- **AND** it emits `response.reasoning_text.done` before completion when reasoning text exists

### Requirement: OpenAI-style error shaping for upstream failures
The proxy SHALL convert upstream chat failures into OpenAI-compatible error payloads or response-failure events rather than returning raw DeepSeek SSE directly.

#### Scenario: Handle streaming chat-completions upstream error
- **WHEN** the upstream streaming chat request returns a non-200 status after the proxy has started processing a streamed chat-completions request
- **THEN** the service emits an SSE error object containing an error message and code
- **AND** the stream still terminates with `data: [DONE]`

#### Scenario: Handle non-stream chat-completions upstream error
- **WHEN** the upstream chat request fails during a non-stream chat-completions request
- **THEN** the service returns an HTTP 502 error payload containing an error message and upstream status code when available

#### Scenario: Handle streamed Responses upstream error
- **WHEN** an error occurs while producing a streamed Responses request
- **THEN** the service emits `response.failed`
- **AND** the embedded response object includes an `error` payload

#### Scenario: Reject unknown stored Responses identifiers
- **WHEN** a client requests `GET /v1/responses/{response_id}` or `GET /v1/responses/{response_id}/input_items` for a missing stored response
- **THEN** the service returns HTTP 404 with an OpenAI-style invalid-request error payload

## ADDED Requirements

### Requirement: Responses conversation chaining
The proxy SHALL support `previous_response_id` for Responses clients by reusing locally persisted conversation state from earlier Responses objects.

#### Scenario: Continue a stored response conversation
- **WHEN** a client sends `POST /v1/responses` with a known `previous_response_id`
- **THEN** the proxy prepends the stored conversation context from that prior response before forwarding the next turn upstream

#### Scenario: Reject unknown previous response
- **WHEN** a client sends `POST /v1/responses` with a `previous_response_id` that is not stored locally
- **THEN** the proxy returns HTTP 404 with an OpenAI-style invalid-request error payload
