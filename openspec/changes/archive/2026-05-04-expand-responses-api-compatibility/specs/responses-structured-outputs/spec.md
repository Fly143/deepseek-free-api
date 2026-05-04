# responses-structured-outputs Specification

## Purpose
Define the supported Responses structured-output contract for `text.format` and stored response metadata.

## ADDED Requirements

### Requirement: Responses structured output formatting
The proxy SHALL accept supported `text.format` structured-output requests on `POST /v1/responses` and preserve the requested format on stored response objects.

#### Scenario: Accept JSON object format
- **WHEN** a client sends `text.format.type` as `json_object`
- **THEN** the service preserves the requested format on the stored response object
- **AND** the returned `output_text` remains the model-generated JSON payload

#### Scenario: Accept JSON schema format
- **WHEN** a client sends `text.format.type` as `json_schema` with a schema definition
- **THEN** the service preserves the schema metadata on the stored response object
- **AND** the response object reflects the requested structured-output format

#### Scenario: Surface structured-output failure
- **WHEN** the requested structured-output format cannot be satisfied by the upstream completion
- **THEN** the service returns a Responses failure or incomplete response with an OpenAI-style error payload
- **AND** the stored response object records the failure state
