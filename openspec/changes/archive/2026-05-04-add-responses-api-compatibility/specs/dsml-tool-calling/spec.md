## MODIFIED Requirements

### Requirement: Extract tool calls from model output
The proxy SHALL convert DSML-formatted tool invocations in model output into OpenAI-compatible function-call structures for both chat-completions and Responses clients.

#### Scenario: Parse non-stream chat-completions tool calls
- **WHEN** a non-stream chat-completions response contains DSML tool-call markup
- **THEN** the service extracts one or more tool calls
- **AND** it returns them in OpenAI `tool_calls` format
- **AND** it sets the completion `finish_reason` to `tool_calls`

#### Scenario: Normalize tool-call arguments
- **WHEN** DSML parsing yields tool invocations with structured parameters
- **THEN** the service serializes arguments as JSON strings within the OpenAI-style `function.arguments` field

#### Scenario: Map non-stream tool calls into Responses output items
- **WHEN** a non-stream Responses request yields DSML-derived tool calls
- **THEN** the returned response object includes `function_call` output items with serialized argument strings

### Requirement: Stream text and tool calls separately
The proxy SHALL separate normal assistant text from DSML tool-call markup during streamed tool-enabled completions for both chat-completions and Responses compatibility.

#### Scenario: Emit regular streamed text for chat-completions
- **WHEN** streamed model output contains normal assistant text before or around a tool call on the chat-completions path
- **THEN** the service emits that text as OpenAI `delta.content` chunks

#### Scenario: Emit streamed tool calls for chat-completions
- **WHEN** streamed model output contains a complete DSML tool call on the chat-completions path
- **THEN** the service emits OpenAI-style streamed `tool_calls` deltas
- **AND** it ends the streamed completion with `finish_reason: "tool_calls"`

#### Scenario: Emit streamed function-call argument deltas for Responses
- **WHEN** a streamed Responses request yields DSML-derived tool calls
- **THEN** the service emits `response.function_call_arguments.delta` events as arguments stream in
- **AND** it emits `response.function_call_arguments.done` for each completed function call before `response.completed`

#### Scenario: Fallback to buffered parse if sieve misses
- **WHEN** the streaming sieve does not emit a tool call during incremental processing
- **AND** the fully buffered streamed content contains a valid DSML invocation
- **THEN** the service performs a fallback full-buffer parse and emits the recovered tool call
