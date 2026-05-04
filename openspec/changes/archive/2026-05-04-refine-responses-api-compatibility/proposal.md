## Why

The first Responses compatibility change established a usable baseline, but several details still affect real SDK interoperability: streamed lifecycle events are narrower than expected, `previous_response_id` chaining can preserve outdated system instructions, and persisted input items do not consistently reflect the normalized Responses request shape.

## What Changes

- Refine chained Responses state handling so new `instructions` replace earlier stored system context.
- Expand the streamed Responses event lifecycle to include `response.in_progress`, `response.output_item.*`, and `response.content_part.*` events alongside the existing text, reasoning, and function-call deltas.
- Add deletion support for locally stored response objects and tighten retrieval/input-item normalization behavior.
- Route `web_search_preview` requests onto existing DeepSeek search-capable model variants when available.

## Impact

- Affects `proxy.py` request translation, streaming event shaping, and model selection for Responses requests.
- Extends `response_store.py` with record deletion support.
- Clarifies the supported Responses subset in OpenSpec without broadening the upstream execution model.
