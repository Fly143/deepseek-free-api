# credential-session-management Specification

## Purpose
Define how the proxy is configured with DeepSeek credentials, how it persists local auth/session state, and how it refreshes or renews that state during operation.
## Requirements
### Requirement: Interactive account configuration
The service SHALL support local account configuration through JSON persistence and admin-facing login helpers.

#### Scenario: Save configuration from copied cURL
- **WHEN** an operator submits a DeepSeek browser request as cURL to `POST /api/config`
- **THEN** the service extracts the bearer token, session identifier, headers, and request origin data
- **AND** it persists the resulting configuration to `token.json`

#### Scenario: Reject incomplete cURL configuration
- **WHEN** the submitted cURL does not contain a bearer token or session identifier
- **THEN** `POST /api/config` returns an unsuccessful result explaining which required value could not be extracted

#### Scenario: Login with phone or email
- **WHEN** an operator submits credentials to `POST /api/login`
- **THEN** the service authenticates against the DeepSeek login endpoint
- **AND** it attempts to create a chat session
- **AND** it stores the resulting token, session identifier, and login metadata in `token.json`

### Requirement: Automatic token refresh
The proxy SHALL attempt automatic re-login when an upstream chat request fails with HTTP 401 and saved credentials are available.

#### Scenario: Refresh token during streaming request
- **WHEN** an upstream streaming chat request returns HTTP 401
- **AND** the local configuration contains reusable login credentials
- **THEN** the service re-authenticates with DeepSeek, persists the refreshed token and session information, and retries the completion once

#### Scenario: Refresh token during non-stream request
- **WHEN** an upstream non-stream completion path encounters HTTP 401
- **AND** the local configuration contains reusable login credentials
- **THEN** the service re-authenticates with DeepSeek, persists the refreshed token and session information, and retries the completion once

#### Scenario: Fail cleanly without saved credentials
- **WHEN** an upstream chat request returns HTTP 401
- **AND** the local configuration does not contain reusable login credentials
- **THEN** the service does not attempt automatic refresh
- **AND** it surfaces an authentication failure to the caller

### Requirement: Prompt-token-based session renewal
The proxy SHALL track prompt-token usage per local session and renew the DeepSeek session when the tracked threshold is exceeded.

#### Scenario: Accumulate prompt-token usage
- **WHEN** a chat completion request is processed
- **THEN** the service records prompt-token usage in local session state keyed to the current session identifier

#### Scenario: Renew session after threshold breach
- **WHEN** the tracked prompt-token usage exceeds the configured renewal threshold
- **THEN** the service creates a fresh DeepSeek chat session before continuing normal request handling
- **AND** it persists the new session identifier for subsequent requests

### Requirement: Local operational credential introspection
The admin surface SHALL expose whether the service is configured without leaking the full stored bearer token.

#### Scenario: Read masked configuration state
- **WHEN** an operator requests `GET /api/config`
- **THEN** the service indicates whether configuration exists
- **AND** a configured response includes only a masked token view plus the stored session identifier
