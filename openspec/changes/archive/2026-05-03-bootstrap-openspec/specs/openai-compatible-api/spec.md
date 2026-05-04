## ADDED Requirements

### Requirement: OpenAI-compatible model listing
The proxy SHALL expose OpenAI-style model listing endpoints that return dynamically discovered DeepSeek-backed models with compatibility metadata.

#### Scenario: List models
- **WHEN** a client sends `GET /v1/models`
- **THEN** the service returns a JSON object with `object: "list"`
- **AND** each returned model entry includes an `id`, `object`, `owned_by`, `max_input_tokens`, `max_output_tokens`, `context_length`, and `context_window`
- **AND** each returned model entry advertises supported parameters for chat completion compatibility

#### Scenario: Fetch model detail
- **WHEN** a client sends `GET /v1/models/{model_id}` for a discovered model
- **THEN** the service returns a JSON object describing that model
- **AND** the response includes the same context-window metadata used in model listing

#### Scenario: Reject unknown model detail lookup
- **WHEN** a client sends `GET /v1/models/{model_id}` for a model that is not in the discovered model set
- **THEN** the service returns an HTTP 404 response

### Requirement: Chat completions entrypoint
The proxy SHALL expose `POST /v1/chat/completions` as the primary OpenAI-compatible chat endpoint backed by DeepSeek chat completion requests.

#### Scenario: Require prior configuration
- **WHEN** a client calls `POST /v1/chat/completions` before local credentials have been configured
- **THEN** the service returns HTTP 503
- **AND** the error instructs the operator to configure the account through the local admin flow

#### Scenario: Map requests onto DeepSeek prompt format
- **WHEN** a configured client submits chat messages to `POST /v1/chat/completions`
- **THEN** the service converts the message history into DeepSeek prompt markers before sending the upstream request
- **AND** the upstream request includes the resolved DeepSeek `model_type`

#### Scenario: Preserve stream flag behavior
- **WHEN** a client sets `stream: true`
- **THEN** the service responds as an SSE stream using OpenAI-style `chat.completion.chunk` objects
- **AND** the stream terminates with `data: [DONE]`

#### Scenario: Support buffered non-stream responses
- **WHEN** a client omits `stream` or sets it to false
- **THEN** the service internally consumes the upstream SSE response
- **AND** the client receives a single JSON `chat.completion` response

### Requirement: Reasoning content compatibility
The proxy SHALL surface DeepSeek thinking output through the OpenAI-compatible `reasoning_content` field when the selected model variant enables thinking.

#### Scenario: Emit reasoning content in streaming responses
- **WHEN** the upstream SSE includes thinking fragments for a thinking-enabled model
- **THEN** the service emits them in `choices[0].delta.reasoning_content`

#### Scenario: Emit reasoning content in non-stream responses
- **WHEN** a non-stream completion accumulates thinking output for a thinking-enabled model
- **THEN** the final assistant message includes a `reasoning_content` field

### Requirement: OpenAI-style error shaping for upstream failures
The proxy SHALL convert upstream chat failures into OpenAI-compatible error payloads or HTTP errors rather than returning raw DeepSeek SSE directly.

#### Scenario: Handle streaming upstream error
- **WHEN** the upstream streaming chat request returns a non-200 status after the proxy has started processing a streamed completion
- **THEN** the service emits an SSE error object containing an error message and code
- **AND** the stream still terminates with `data: [DONE]`

#### Scenario: Handle non-stream upstream error
- **WHEN** the upstream chat request fails during a non-stream completion
- **THEN** the service returns an HTTP 502 error payload containing an error message and upstream status code when available
