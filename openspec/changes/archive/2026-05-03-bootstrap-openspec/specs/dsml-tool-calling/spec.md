## ADDED Requirements

### Requirement: DSML tool prompt injection
The proxy SHALL translate OpenAI tool definitions into a DSML prompt format before forwarding a tool-enabled request to DeepSeek.

#### Scenario: Inject tool prompt before user turn
- **WHEN** a chat completion request includes `tools`
- **THEN** the service builds a DSML tool prompt describing the available tool names and invocation format
- **AND** it injects that prompt before the final user marker in the generated DeepSeek prompt when possible

### Requirement: Extract tool calls from model output
The proxy SHALL convert DSML-formatted tool invocations in model output into OpenAI-style `tool_calls`.

#### Scenario: Parse non-stream tool calls
- **WHEN** a non-stream completion response contains DSML tool-call markup
- **THEN** the service extracts one or more tool calls
- **AND** it returns them in OpenAI `tool_calls` format
- **AND** it sets the completion `finish_reason` to `tool_calls`

#### Scenario: Normalize tool-call arguments
- **WHEN** DSML parsing yields tool invocations with structured parameters
- **THEN** the service serializes arguments as JSON strings within the OpenAI-style `function.arguments` field

### Requirement: Stream text and tool calls separately
The proxy SHALL separate normal assistant text from DSML tool-call markup during streamed tool-enabled completions.

#### Scenario: Emit regular streamed text
- **WHEN** streamed model output contains normal assistant text before or around a tool call
- **THEN** the service emits that text as OpenAI `delta.content` chunks

#### Scenario: Emit streamed tool calls
- **WHEN** streamed model output contains a complete DSML tool call
- **THEN** the service emits OpenAI-style streamed `tool_calls` deltas
- **AND** it ends the streamed completion with `finish_reason: "tool_calls"`

#### Scenario: Fallback to buffered parse if sieve misses
- **WHEN** the streaming sieve does not emit a tool call during incremental processing
- **AND** the fully buffered streamed content contains a valid DSML invocation
- **THEN** the service performs a fallback full-buffer parse and emits the recovered tool call

### Requirement: Preserve multi-turn tool context
The proxy SHALL convert prior assistant tool calls and tool-role messages into a DeepSeek-understandable prompt history for subsequent turns.

#### Scenario: Forward prior tool-call history
- **WHEN** the chat history includes assistant `tool_calls` and tool result messages
- **THEN** the service reformats that history into the DeepSeek prompt markers and DSML-compatible tool history representation before sending the next upstream request
