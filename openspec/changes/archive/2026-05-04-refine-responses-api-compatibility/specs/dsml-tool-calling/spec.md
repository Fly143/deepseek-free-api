## MODIFIED Requirements

### Requirement: Stream text and tool calls separately
The proxy SHALL separate normal assistant text from DSML tool-call markup during streamed tool-enabled completions for both chat-completions and Responses compatibility.

#### Scenario: Preserve tool-call output ordering in Responses streams
- **WHEN** a streamed Responses request emits one or more function-call argument delta events
- **THEN** each function call is first announced with `response.output_item.added`
- **AND** the final stored `response.output` array preserves the same item ordering observed during the stream

#### Scenario: Preserve tool result linkage from Responses input items
- **WHEN** a Responses request includes `function_call_output` items with a `call_id`
- **THEN** the translated tool-role messages retain that linkage metadata for the next upstream turn
