## ADDED Requirements

### Requirement: Upstream-driven model discovery
The proxy SHALL discover available models from DeepSeek upstream settings instead of relying on a hardcoded local model list.

#### Scenario: Discover models from configured account
- **WHEN** the service has a configured token and invokes model discovery
- **THEN** it requests DeepSeek client settings for model scope
- **AND** it builds local model entries only from enabled upstream model types

#### Scenario: Derive model variants from upstream capabilities
- **WHEN** an upstream model type advertises thinking or search features
- **THEN** the service creates corresponding local model variants for base, reasoning, search, and combined reasoning-plus-search cases as supported

### Requirement: Cache discovered model metadata
The proxy SHALL cache discovered models locally to avoid probing the upstream settings endpoint on every request.

#### Scenario: Reuse unexpired model cache
- **WHEN** the model cache is populated and the cache TTL has not expired
- **THEN** model listing and lookup reuse the cached data

#### Scenario: Refresh expired model cache
- **WHEN** the model cache is missing or expired
- **THEN** the next model lookup triggers a new discovery attempt

### Requirement: Manual model refresh endpoint
The proxy SHALL expose an administrative refresh endpoint for model metadata.

#### Scenario: Force refresh model cache
- **WHEN** a client sends `POST /v1/models/refresh`
- **THEN** the service invalidates the current model cache
- **AND** it immediately re-discovers the available models
- **AND** it returns the refreshed model list in the same OpenAI-style format as `GET /v1/models`

### Requirement: Fail without inventing models
The proxy SHALL not fabricate a non-empty model list when discovery fails.

#### Scenario: Discovery failure after cache miss
- **WHEN** model discovery fails and there is no valid cached model set
- **THEN** the service returns an empty model list rather than a guessed or hardcoded fallback set
