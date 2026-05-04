# responses-tool-interoperability Specification

## Purpose
Define how native Responses tool definitions and function-call outputs map into the existing DeepSeek-backed tool pipeline.

## ADDED Requirements

### Requirement: Normalize Responses function tools
The proxy SHALL accept Responses-style function tool definitions and translate them into the existing internal tool representation.

#### Scenario: Accept flat function tools
- **WHEN** a client sends a tool entry with `type: "function"` and top-level `name`, `description`, and `parameters`
- **THEN** the service normalizes that tool into the existing function-tool path

#### Scenario: Accept nested function tools
- **WHEN** a client sends a tool entry with `type: "function"` and a nested `function` object
- **THEN** the service normalizes that tool into the existing function-tool path
- **AND** it preserves the function name and argument schema

### Requirement: Preserve function-call output linkage
The proxy SHALL accept `function_call_output` input items and preserve their linkage to the originating function call.

#### Scenario: Forward function_call_output to the next turn
- **WHEN** a Responses request includes `function_call_output` input items
- **THEN** the service forwards their output as tool-role content to the upstream chat path
- **AND** it preserves the associated `call_id`

#### Scenario: Store function-call linkage in response state
- **WHEN** a response object contains assistant function calls
- **THEN** the stored response state preserves the function-call linkage metadata for later retrieval

### Requirement: Surface Responses function-call output items
The proxy SHALL emit `function_call` output items for assistant tool calls in Responses objects and streamed events.

#### Scenario: Return function-call items in non-stream responses
- **WHEN** a non-stream Responses request produces assistant tool calls
- **THEN** the returned response object includes `function_call` output items

#### Scenario: Stream function-call arguments
- **WHEN** a streamed Responses request yields tool-call arguments
- **THEN** the service emits `response.function_call_arguments.delta`
- **AND** it emits `response.function_call_arguments.done` before completion
