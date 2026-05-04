# dsml-tool-calling Specification

## Purpose
Define the compatibility layer that simulates OpenAI-style function calling on top of DeepSeek by using DSML prompt injection, parsing, and streamed tool-call extraction for both chat-completions and Responses clients.
## Requirements
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

#### Scenario: Map non-stream tool calls into Responses output items
- **WHEN** a non-stream Responses request yields DSML-derived tool calls
- **THEN** the returned response object includes `function_call` output items with serialized argument strings

### Requirement: Stream text and tool calls separately
The proxy SHALL separate normal assistant text from DSML tool-call markup during streamed tool-enabled completions.

#### Scenario: Emit regular streamed text
- **WHEN** streamed model output contains normal assistant text before or around a tool call
- **THEN** the service emits that text as OpenAI `delta.content` chunks

#### Scenario: Emit streamed tool calls
- **WHEN** streamed model output contains a complete DSML tool call
- **THEN** the service emits OpenAI-style streamed `tool_calls` deltas
- **AND** it ends the streamed completion with `finish_reason: "tool_calls"`

#### Scenario: Emit streamed function-call argument deltas for Responses
- **WHEN** a streamed Responses request yields DSML-derived tool calls
- **THEN** the service emits `response.function_call_arguments.delta` events as arguments stream in
- **AND** it emits `response.function_call_arguments.done` for each completed function call before `response.completed`

#### Scenario: Preserve tool-call output ordering in Responses streams
- **WHEN** a streamed Responses request emits one or more function-call argument delta events
- **THEN** each function call is first announced with `response.output_item.added`
- **AND** the final stored `response.output` array preserves the same item ordering observed during the stream

#### Scenario: Fallback to buffered parse if sieve misses
- **WHEN** the streaming sieve does not emit a tool call during incremental processing
- **AND** the fully buffered streamed content contains a valid DSML invocation
- **THEN** the service performs a fallback full-buffer parse and emits the recovered tool call

### Requirement: Preserve multi-turn tool context
The proxy SHALL convert prior assistant tool calls and tool-role messages into a DeepSeek-understandable prompt history for subsequent turns.

#### Scenario: Forward prior tool-call history
- **WHEN** the chat history includes assistant `tool_calls` and tool result messages
- **THEN** the service reformats that history into the DeepSeek prompt markers and DSML-compatible tool history representation before sending the next upstream request

#### Scenario: Preserve tool result linkage from Responses input items
- **WHEN** a Responses request includes `function_call_output` items with a `call_id`
- **THEN** the translated tool-role messages retain that linkage metadata for the next upstream turn
