## 1. Public Response Formatting

- [x] 1.1 Add a centralized Responses public-object formatter that fills expected nullable fields and strips internal metadata.
- [x] 1.2 Update synchronous creation, retrieval, background completion, compaction, cancellation, and replay terminal payloads to use the centralized formatter.
- [x] 1.3 Normalize output item builders so message, reasoning, refusal, and function-call items use stable public fields across code paths.

## 2. Structured Output Validation

- [x] 2.1 Implement JSON extraction and validation for `text.format.type=json_object` before returning a successful response.
- [x] 2.2 Implement a local JSON Schema subset validator for `json_schema` covering `type`, `properties`, `required`, `additionalProperties`, `items`, and primitive scalar types.
- [x] 2.3 Convert structured-output validation failures into stored Responses failure objects with OpenAI-style error payloads.

## 3. Stream And Replay Consistency

- [x] 3.1 Ensure streamed terminal `response.completed`, `response.failed`, and `response.incomplete` events embed the normalized public response object.
- [x] 3.2 Ensure persisted or reconstructed replay events use the same terminal public payload as `GET /v1/responses/{response_id}`.

## 4. Validation

- [x] 4.1 Expand Responses smoke coverage for public object fields, output item shape consistency, JSON object success, JSON schema success, and structured-output failure.
- [x] 4.2 Run Python syntax validation, Responses smoke tests, and `openspec validate --changes --strict`.
