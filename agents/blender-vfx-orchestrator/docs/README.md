# Docs Index

Status: authoritative
Last verified: 2026-03-14
Purpose: first-stop navigation for the Blender VFX Orchestrator docs

## Start Here

Use these documents as the current source of truth before reading older plans, reviews, or prompts:

1. `CURRENT_STATE.md`
2. `CURRENT_ROADMAP.md`
3. `MISSION_STATEMENT_2026-02-22.md`
4. `VERSION_TRUTH.md`
5. `RUNTIME_TRUTH_AND_DOC_GROUNDING_2026-02-13.md`
6. `DOCS_AUTHORITY_AND_FRESHNESS_2026-02-13.md`
7. `AI_OPERATION_MANUAL.md`
8. `SDK_ENFORCEMENT_PROTOCOL.md`

## Directory Layout

- `docs/`
  Root-level canonical and long-lived reference docs only.
- `docs/reviews/`
  Architecture reviews, external analyses, response series, and roadmap correspondence.
- `docs/postmortems/`
  Incident reports and failure write-ups.
- `docs/prompts/`
  Reusable review and analysis prompts.
- `docs/worklogs/`
  Implementation logs and change notes.
- `docs/reports/`
  Investigations, audits, and synthesized reports.
- `docs/research/`
  background research and topic-specific supporting material.
- `docs/archive/`
  superseded historical material that is no longer active planning input.

## Canonical vs Historical

Treat docs as belonging to one of these classes:

- `authoritative`
  Use directly for planning and implementation.
- `active`
  Current working material that may still change.
- `historical`
  Useful context, but not current truth.
- `superseded`
  Kept for record only; do not plan from it.
- `draft`
  Incomplete working material.

If a doc does not declare a status yet, assume it is non-authoritative until verified.

## Current March 2026 Review Chain

The recent architecture-review correspondence has been moved out of the root to:

- `docs/reviews/2026-03/`

The most useful current documents in that set are:

- `docs/reviews/2026-03/20260314_DEEP_AUTONOMY_ANALYSIS_GPT_54_XHIGH.md`
- `docs/reviews/2026-03/20260314_ARCHITECTURE_REVIEW_MODIFY_CODE_RESPONSE.md`
- `docs/reviews/2026-03/20260312_WAVE2_STATUS_AND_PATH_FORWARD.md`

Use them as supporting analysis, not as a replacement for `CURRENT_STATE.md` and `CURRENT_ROADMAP.md`.

## Writing Rules Going Forward

- Put new review or cross-LLM analysis docs under `docs/reviews/<yyyy-mm>/`.
- Put new postmortems under `docs/postmortems/<yyyy-mm>/`.
- Put new reusable prompts under `docs/prompts/`.
- Put new work logs under `docs/worklogs/`.
- Do not add new root-level docs unless they are intended to become canonical.
- When a review changes the plan, update `CURRENT_STATE.md` or the relevant canonical doc instead of leaving the change trapped inside a review thread.

## Next Cleanup Pass

This is a first-pass cleanup only. Likely follow-up work:

- add explicit status headers to more docs
- move remaining dated plans and proposals into category folders
- create ADRs for durable architecture decisions
- reduce root-level docs further once the canonical set is stable
