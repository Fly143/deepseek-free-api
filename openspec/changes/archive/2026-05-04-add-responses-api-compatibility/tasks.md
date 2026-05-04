## 1. Request Translation

- [x] 1.1 Add local persistence for Responses compatibility state.
- [x] 1.2 Translate Responses input, instructions, tools, and previous-response chaining into the existing chat request shape.

## 2. Endpoint Compatibility

- [x] 2.1 Implement `POST /v1/responses` for non-stream responses.
- [x] 2.2 Implement `POST /v1/responses` streaming event adaptation.
- [x] 2.3 Implement retrieval endpoints for stored response objects and input items.

## 3. Capability Alignment

- [x] 3.1 Update OpenSpec API compatibility requirements for Responses endpoints and event shapes.
- [x] 3.2 Update tool-calling requirements for Responses function-call output mapping.
- [x] 3.3 Update file-and-vision requirements for Responses-style multimodal input handling.

## 4. Validation

- [x] 4.1 Run Python syntax validation for the modified modules.
- [x] 4.2 Validate the OpenSpec change with the CLI.
