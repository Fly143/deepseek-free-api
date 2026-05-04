## Why

The repository has working code and documentation, but it does not yet have an OpenSpec baseline that captures the current system as a maintainable contract for future development. Initializing OpenSpec now reduces ambiguity before further features and refactors accumulate.

## What Changes

- Add OpenSpec project configuration tailored to this repository and its actual runtime constraints.
- Capture the current system as baseline capabilities in OpenSpec rather than leaving only code and README as the source of truth.
- Record this initialization as a completed bootstrap change so future work can build on a clean spec-driven foundation.

## Capabilities

### New Capabilities
- `openai-compatible-api`: OpenAI-style model listing and chat completion behavior exposed by the proxy.
- `credential-session-management`: Local credential storage, login, token refresh, and session renewal behavior.
- `dynamic-model-discovery`: Upstream model probing, caching, and refresh behavior.
- `dsml-tool-calling`: DSML prompt injection and tool-call extraction compatibility layer.
- `file-and-vision-handling`: Text file ingestion, image upload, vision forking, and file parsing readiness flow.
- `operations-and-deployment`: Health, admin surface, local operational state, and deployment/runtime expectations.

### Modified Capabilities

- None.

## Impact

- Adds `openspec/config.yaml`.
- Adds baseline spec deltas under `openspec/changes/bootstrap-openspec/specs/`.
- Establishes future planning and implementation work around the current FastAPI proxy, helper modules, and local JSON state files.
