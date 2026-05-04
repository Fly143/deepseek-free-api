# operations-and-deployment Specification

## Purpose
Define the operator-facing runtime behavior of the service, including admin access, health, local usage visibility, PoW solving expectations, and deployment-time configuration.
## Requirements
### Requirement: Local admin and health surface
The service SHALL expose a local operator surface for basic setup and runtime visibility.

#### Scenario: Redirect root to admin
- **WHEN** an operator requests `GET /`
- **THEN** the service redirects the browser to `/admin`

#### Scenario: Serve embedded admin UI
- **WHEN** an operator requests `GET /admin`
- **THEN** the service returns the embedded HTML admin interface

#### Scenario: Expose health state
- **WHEN** an operator requests `GET /health`
- **THEN** the service reports whether local configuration exists
- **AND** it distinguishes between configured and waiting states

### Requirement: Local usage statistics
The proxy SHALL maintain local usage aggregates for operational visibility.

#### Scenario: Read usage statistics
- **WHEN** an operator requests `GET /api/usage`
- **THEN** the service returns usage summaries for today, the trailing week, and total history
- **AND** the summaries include per-model token/request counts

#### Scenario: Clear usage statistics
- **WHEN** an operator requests `DELETE /api/usage`
- **THEN** the service clears the locally persisted usage aggregates

### Requirement: PoW challenge solving for upstream-protected endpoints
The proxy SHALL obtain and solve DeepSeek PoW challenges for upstream operations that require them.

#### Scenario: Solve PoW for chat completion
- **WHEN** the service prepares an upstream chat completion request
- **THEN** it requests a fresh PoW challenge for the chat completion target path
- **AND** it attaches the solved response header when solving succeeds

#### Scenario: Solve PoW for file upload
- **WHEN** the service uploads a file to DeepSeek
- **THEN** it requests a fresh PoW challenge for the file upload target path
- **AND** it attaches the solved response header when solving succeeds

#### Scenario: Use fallback solver path
- **WHEN** the primary Node.js/WASM PoW solver path is unavailable or fails
- **THEN** the service attempts a pure Python PoW fallback before giving up

### Requirement: Configurable listen port
The service SHALL support port configuration through environment variables.

#### Scenario: Read proxy port from environment
- **WHEN** the process starts with `PROXY_PORT` set
- **THEN** the HTTP service listens on that configured port instead of the default port
