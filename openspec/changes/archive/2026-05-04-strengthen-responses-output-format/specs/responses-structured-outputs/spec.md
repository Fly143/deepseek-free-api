## MODIFIED Requirements

### Requirement: Responses structured output formatting
The proxy SHALL accept supported `text.format` structured-output requests on `POST /v1/responses`, preserve the requested format on stored response objects, and validate generated output before returning a successful Responses object.

#### Scenario: Accept JSON object format
- **WHEN** a client sends `text.format.type` as `json_object`
- **THEN** the service preserves the requested format on the stored response object
- **AND** the returned `output_text` is valid JSON text
- **AND** the assistant message output item contains the same normalized JSON text

#### Scenario: Accept JSON schema format
- **WHEN** a client sends `text.format.type` as `json_schema` with a schema definition
- **THEN** the service preserves the schema metadata on the stored response object
- **AND** the response object reflects the requested structured-output format
- **AND** the returned `output_text` validates against the proxy-supported JSON Schema subset

#### Scenario: Validate common JSON schema keywords
- **WHEN** a requested `json_schema` includes `type`, `properties`, `required`, `additionalProperties`, `items`, or primitive scalar constraints
- **THEN** the service validates generated JSON against those supported keywords before returning a successful response

#### Scenario: Surface structured-output failure
- **WHEN** the requested structured-output format cannot be satisfied by the upstream completion
- **THEN** the service returns a Responses failure or incomplete response with an OpenAI-style error payload
- **AND** the stored response object records the failure state
- **AND** replaying or retrieving the stored response preserves that failure state
