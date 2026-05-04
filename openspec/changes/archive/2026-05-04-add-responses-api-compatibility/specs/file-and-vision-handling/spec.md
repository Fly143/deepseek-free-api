## MODIFIED Requirements

### Requirement: Text file ingestion for chat requests
The proxy SHALL accept both chat-completions-style and Responses-style inline file inputs and forward successfully parsed files to DeepSeek as referenced files.

#### Scenario: Upload and wait for text files from chat-completions input
- **WHEN** a chat-completions message contains one or more `type: "file"` parts with decodable inline file data
- **THEN** the service uploads those files to DeepSeek
- **AND** it waits for file parsing readiness before sending the chat request
- **AND** it forwards the resulting ready file identifiers through `ref_file_ids`

#### Scenario: Upload and wait for text files from Responses input
- **WHEN** a Responses request contains one or more decodable file input items that map to inline text-file content
- **THEN** the service translates them into the existing file-upload preparation path
- **AND** it forwards the resulting ready file identifiers through `ref_file_ids`

### Requirement: Vision image ingestion
The proxy SHALL accept both chat-completions-style and Responses-style image inputs for vision-compatible requests and prepare them for DeepSeek vision access.

#### Scenario: Parse image input formats from chat-completions requests
- **WHEN** a vision request on the chat-completions path contains `image_url`, inline image data, or message-level image entries
- **THEN** the service extracts image bytes from supported data-URI or HTTP URL sources before upload

#### Scenario: Parse image input formats from Responses requests
- **WHEN** a Responses request contains supported image input items
- **THEN** the service translates them into the existing image preparation path before upload

#### Scenario: Fork uploaded images to vision model type
- **WHEN** an image is uploaded for a vision-compatible request
- **THEN** the service forks the uploaded file into DeepSeek's vision model type before referencing it in chat

#### Scenario: Wait for vision file readiness
- **WHEN** one or more forked vision file identifiers exist
- **THEN** the service polls file status until files are ready, timed out, or otherwise terminal
- **AND** it forwards the ready identifiers as `ref_file_ids`
