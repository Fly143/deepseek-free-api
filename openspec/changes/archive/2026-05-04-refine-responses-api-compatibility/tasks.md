## 1. Responses State Refinement

- [x] 1.1 Normalize stored input items and preserve function-call output linkage metadata.
- [x] 1.2 Ensure `previous_response_id` chaining replaces prior system instructions when new `instructions` are supplied.

## 2. Streaming Lifecycle Alignment

- [x] 2.1 Emit `response.in_progress` and output-item lifecycle events during streamed Responses requests.
- [x] 2.2 Preserve stable output ordering between streamed item events and the final stored response object.

## 3. Stored Response Management

- [x] 3.1 Add `DELETE /v1/responses/{response_id}` for locally stored response objects.
- [x] 3.2 Return normalized stored input items from `GET /v1/responses/{response_id}/input_items`.

## 4. Validation

- [x] 4.1 Run Python syntax validation for the modified modules.
- [x] 4.2 Run a mocked Responses smoke test covering stream lifecycle, retrieval, chaining, and deletion.
