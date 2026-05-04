## 1. Structured Outputs

- [x] 1.1 Normalize `text.format` request data and persist supported structured-output metadata in stored response records.
- [x] 1.2 Add best-effort handling for `json_object` and `json_schema` Responses outputs without breaking the existing chat-completions path.

## 2. Streaming Parity

- [x] 2.1 Add monotonically increasing `sequence_number` fields to streamed Responses events.
- [x] 2.2 Emit the missing Responses lifecycle events for completion, refusal, and terminal failure states.

## 3. Tool Interoperability

- [x] 3.1 Normalize flat and nested Responses function-tool definitions into the existing DSML tool pipeline.
- [x] 3.2 Preserve `function_call_output` linkage and keep function-call output items stable across retrieval and streaming.

## 4. State Parity

- [x] 4.1 Add pagination support for `GET /v1/responses/{response_id}/input_items` with stable ordering.
- [x] 4.2 Preserve list metadata and invalid-request behavior for missing stored Responses input items.

## 5. Validation

- [x] 5.1 Add or update API smoke tests covering structured outputs, streaming sequencing, tool interop, and paginated input retrieval.
- [x] 5.2 Run OpenSpec validation and targeted Python syntax checks for the touched modules.
