# Docs Authority and Freshness (2026-02-13)

Use this file to decide which documents are authoritative when sources disagree.

## Authority Order
1. `docs/RUNTIME_TRUTH_AND_DOC_GROUNDING_2026-02-13.md`
2. Runtime code in `orchestrator.py`, `guardrails/`, `hooks/`, `tools/`
3. `docs/VERSION_TRUTH.md`
4. `docs/SDK_ENFORCEMENT_PROTOCOL.md`
5. `docs/AI_OPERATION_MANUAL.md`
6. Planning and postmortem docs (`*_2026-01-*.md`, `*_2026-02-06*.md`, etc.)

## Current Canonical Set (LLM Prompt Context)
- `docs/RUNTIME_TRUTH_AND_DOC_GROUNDING_2026-02-13.md`
- `docs/VERSION_TRUTH.md`
- `docs/SDK_ENFORCEMENT_PROTOCOL.md`
- `docs/AI_OPERATION_MANUAL.md`
- `docs/MASTER_ROADMAP_2026-01-26.md` (planning only, not runtime truth)

## Docs Marked Historical Snapshot
- `docs/AGENT_SPECIFICATION.md`
- `docs/PROJECT_STATE_ANALYSIS_2026-02-06.md`
- `docs/LLM_PRIMER_2026-01-26.md` (onboarding snapshot)
- `docs/MULTI_AGENT_ARCHITECTURE_ADDENDUM_2026-01-29.md` (target architecture)

## Grounding Rules to Preserve
- Never accept sentinel `doc_refs` (`doc_search_empty`, `doc_search_unavailable`, etc.).
- Never accept null/blank/filename-only `doc_refs`.
- Require at least one API doc reference (`bpy.types.*` or `bpy.ops.*`) for API-intent outputs.
- Treat Blender 4.x attributes as invalid unless explicitly verified in Blender 5 docs.

## Maintenance Rule
When changing runtime behavior, update:
1. `docs/RUNTIME_TRUTH_AND_DOC_GROUNDING_2026-02-13.md`
2. `docs/VERSION_TRUTH.md` or `docs/SDK_ENFORCEMENT_PROTOCOL.md` (if API/SDK truth changed)
3. This file if authority/freshness status changed.
