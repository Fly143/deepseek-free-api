## Context

This project is a compact Python service whose behavior is concentrated in `proxy.py`, with a few helper modules for DSML parsing, streaming separation, PoW solving, usage tracking, and session tracking. The repository already has meaningful product behavior, but the development contract currently lives only in code and README prose.

OpenSpec initialization for this repository should therefore do two things:

1. Install project-level OpenSpec configuration that reflects the actual architecture and constraints.
2. Convert current behavior into baseline capabilities that future changes can modify with deltas instead of restating from scratch.

The bootstrap should avoid speculative redesign. It should describe what the system does today, including important compatibility boundaries and operational caveats.

## Goals / Non-Goals

**Goals:**
- Create a valid OpenSpec setup that the CLI can use immediately in this repository.
- Define baseline capabilities along domain boundaries that match the current implementation.
- Preserve an auditable bootstrap change that explains why the baseline exists.
- Keep future spec work focused on externally meaningful behavior.

**Non-Goals:**
- Refactor the application code.
- Normalize all implementation quirks into idealized behavior.
- Add tests, new endpoints, or feature changes unrelated to OpenSpec initialization.
- Split `proxy.py` during this bootstrap.

## Decisions

Use a bootstrap change and archive it into main specs.
Rationale: this leaves a traceable initialization history and produces normal main specs under `openspec/specs/`, which is what later changes should target.

Model the system as multiple capabilities rather than a single monolith.
Rationale: the repository has clear behavioral clusters with different change rates and risks: API surface, auth/session lifecycle, model discovery, tool calling, file/vision behavior, and operations.

Write requirements around observable behavior rather than internal helper functions.
Rationale: future changes should be able to evolve internals without rewriting baseline specs unless the user-visible contract changes.

Capture implementation caveats when they materially affect behavior.
Rationale: this codebase includes compatibility shims such as DSML function calling and internal streaming for non-stream requests. Those are not minor internals; they shape the contract and must be represented.

Add a minimal `.gitignore` for runtime state files.
Rationale: the repository currently stores secrets and runtime JSON in the working tree layout, so the bootstrap should prevent these files from becoming part of spec-driven workflow noise.

## Risks / Trade-offs

[Risk] The baseline may overstate guarantees that the current implementation only approximates.
Mitigation: phrase requirements around what the current code actually does and explicitly avoid promising native DeepSeek support for OpenAI-only features.

[Risk] A spec split that is too granular will create unnecessary future coordination overhead.
Mitigation: use six broad but coherent capability areas aligned with current architecture and user-facing surfaces.

[Risk] A spec split that is too broad will make future deltas noisy and hard to review.
Mitigation: separate independently evolving areas such as model discovery and DSML tool calling.

[Risk] OpenSpec bootstrap artifacts may drift from the code if not validated against the implementation.
Mitigation: derive all baseline requirements from the repository contents and run OpenSpec validation after writing and after archiving.

## Migration Plan

1. Initialize OpenSpec structure in the repository.
2. Add `openspec/config.yaml` with project context and artifact rules.
3. Write bootstrap proposal, design, tasks, and delta specs.
4. Mark bootstrap tasks complete because the initialization is executed as part of this change.
5. Archive the bootstrap change to generate main specs under `openspec/specs/`.
6. Validate both active artifacts and resulting main specs with the OpenSpec CLI.

## Open Questions

- Whether future work should keep all specs in the default schema or introduce a project-local schema with repository-specific artifacts.
- Whether `session_store.py` behavior should become a stronger external operational contract or remain documented as internal lifecycle management.
