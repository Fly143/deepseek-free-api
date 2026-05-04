## ADDED Requirements

### Requirement: Text file ingestion for chat requests
The proxy SHALL accept OpenAI-style file content parts containing inline file data and forward successfully parsed files to DeepSeek as referenced files.

#### Scenario: Upload and wait for text files
- **WHEN** a chat completion message contains one or more `type: "file"` parts with decodable inline file data
- **THEN** the service uploads those files to DeepSeek
- **AND** it waits for file parsing readiness before sending the chat request
- **AND** it forwards the resulting ready file identifiers through `ref_file_ids`

### Requirement: Vision image ingestion
The proxy SHALL accept OpenAI-style inline or URL-based image inputs for vision models and prepare them for DeepSeek vision access.

#### Scenario: Parse image input formats
- **WHEN** a vision request contains `image_url`, inline image data, or message-level image entries
- **THEN** the service extracts image bytes from supported data-URI or HTTP URL sources before upload

#### Scenario: Fork uploaded images to vision model type
- **WHEN** an image is uploaded for a vision model request
- **THEN** the service forks the uploaded file into DeepSeek's vision model type before referencing it in chat

#### Scenario: Wait for vision file readiness
- **WHEN** one or more forked vision file identifiers exist
- **THEN** the service polls file status until files are ready, timed out, or otherwise terminal
- **AND** it forwards the ready identifiers as `ref_file_ids`

### Requirement: Fresh vision session isolation
The proxy SHALL attempt to create a fresh DeepSeek session for vision requests after file preparation.

#### Scenario: Create fresh session for vision completion
- **WHEN** a vision request is about to be forwarded and a valid token is available
- **THEN** the service attempts to create a new DeepSeek chat session for that request path
- **AND** it uses the new session identifier for the outgoing vision chat request when session creation succeeds

### Requirement: Best-effort file readiness behavior
The proxy SHALL treat file parsing as a best-effort readiness check rather than blocking indefinitely.

#### Scenario: Accept partially ready file set after waiting
- **WHEN** some uploaded files are ready while others remain pending beyond the short readiness window
- **THEN** the service may proceed with the subset of ready file identifiers

#### Scenario: Ignore undecodable inline file data
- **WHEN** a file or image content part cannot be decoded into usable bytes
- **THEN** that item is skipped instead of crashing the request parser
