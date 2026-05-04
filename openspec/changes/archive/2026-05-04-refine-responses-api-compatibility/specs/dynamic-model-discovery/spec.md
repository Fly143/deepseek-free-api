## MODIFIED Requirements

### Requirement: Upstream-driven model discovery
The proxy SHALL discover available models from DeepSeek upstream settings instead of relying on a hardcoded local model list.

#### Scenario: Route Responses web-search requests to search-capable variants
- **WHEN** a Responses request includes the `web_search_preview` tool type
- **AND** the requested model does not already name a search-capable variant
- **THEN** the service resolves the request onto a discovered `*-search` or `*-reasoner-search` model variant when one exists
