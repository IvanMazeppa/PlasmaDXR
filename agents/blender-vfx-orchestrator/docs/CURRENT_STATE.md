# Current State

Status: authoritative
Last verified: 2026-03-19
Purpose: current high-level state of the Blender VFX Orchestrator after Phases 1-3 and Phase 3.5 Tasks 1-8

## Mission Context

The project goal remains: an autonomous, self-learning Blender VFX system that can generate complete, high-quality scenes from natural language while staying honest about uncertainty and improving through evidence.

The current question is no longer "can the system ever work?" It is "can the system adapt correctly, learn from trustworthy evidence, and earn higher autonomy without drifting into guesswork?"

## What Is Now Considered Implemented

- `TechniqueContract` exists and has already proven it can bind at least one previously weak technique path.
- The stack is on GPT-5.4 and the newer OpenAI Agents SDK line.
- Runtime truth-pack validation remains a core reliability layer.
- Stale-render reuse is no longer accepted as normal behavior; execution/render validity is expected to be iteration-scoped.
- Section patching infrastructure exists.
- Code-grounded QA feedback exists.
- `call_model_input_filter` and artifact-first handoffs are part of the direction of travel.

## What Is No Longer the Main Blocker

Do not treat these as the primary active problems unless new evidence shows regression:

- technique monotony as the central architecture failure
- hallucinated Blender API usage as the dominant runtime risk
- missing QA-to-code grounding as a completely absent capability
- stale render reuse as an unresolved architecture question

These may still regress, but they are not the best explanation for the current plateau.

## Current Primary Blockers

1. ~~Iteration-state authority is still split.~~ **RESOLVED (2026-03-15).** `record_iteration()` is now the sole path. `StuckDetectionState` computes `plateau_count`, `same_issue_count`, and `escape_level` deterministically. Quality Gate no longer overwrites.

2. ~~Repair routing receives wrong inputs.~~ **RESOLVED (2026-03-15).** `choose_repair_intent()` now reads deterministic stuck-state values directly.

3. ~~The evaluator-to-repair boundary is still too prose-heavy.~~ **RESOLVED (2026-03-16).** `structured_issues` are enforced when `passed=False`, `QualityIssue.kind` / `repair_mode_hint` are constrained, and routing now prioritizes typed issues over keyword heuristics.

4. ~~`modify_code` and section patching are present but need stronger end-to-end proof in live runs.~~ **RESOLVED (2026-03-16).** A live `modify_code` path has been exercised end-to-end, and the hot path now validates scripts before execution.

5. Evaluator calibration and learning provenance are not yet strong enough to justify larger autonomy claims.

6. Scene aesthetic realism has advanced significantly (Phase 3.5 Tasks 1-8 complete). The pipeline now has:
   hero-object refinement recipes, camera/readability enforcement, look-dev coverage metrics, aesthetic benchmarks,
   `hero_object`/`lookdev` issue kinds routed to structural repair, and iteration manifests with script hashes.
   The remaining question is whether these changes produce measurable visual gains (Task 9 benchmark pass).

7. A specialist section-builder workstream exists as a planned quality-ceiling extension (Task 10), gated on
   benchmark evidence from Task 9 before implementation begins.

## Current Planning Priorities

In order:

1. ~~Make repair-mode selection authoritative.~~ **DONE.** State authority fixed, repair routing receives correct inputs.
2. ~~Unify iteration manifests and state mutation.~~ **DONE.** `record_iteration()` is the sole path.
3. ~~Scene aesthetic quick wins (Tasks 1-8).~~ **DONE.** Hero refinement, camera hints, look-dev metrics, aesthetic benchmarks, iteration manifests all shipped.
4. **Run focused benchmark pass (Task 9)** — baseline vs post-quick-win comparison to measure real visual impact. This is the current next step.
5. Finish evaluator calibration and provenance-safe learning before making bigger autonomy claims. (Roadmap Phase 4 Jobs 3-7)
6. Start hero-only specialist builder (Task 10) once Task 9 confirms the current approach is working. (Roadmap Phase 6A first milestone)

The canonical implementation plan and ordered job list for these priorities now lives in `CURRENT_ROADMAP.md`.

## Supporting Analysis

Use these as the main supporting March 2026 review documents:

- `docs/reviews/2026-03/20260314_DEEP_AUTONOMY_ANALYSIS_GPT_54_XHIGH.md`
- `docs/reviews/2026-03/20260314_ARCHITECTURE_REVIEW_MODIFY_CODE_RESPONSE.md`
- `docs/reviews/2026-03/20260312_WAVE2_STATUS_AND_PATH_FORWARD.md`
- `docs/reviews/2026-03/20260317_SPECIALIZED_SECTION_BUILDERS_IMPLEMENTATION_PLAN.md` (candidate final plan)
- `docs/reviews/2026-03/20260317_SECTION_BUILDERS_REVIEW_AND_REFINEMENTS.md`

Use these as supplemental external/opposing views, not canonical truth:

- `docs/reviews/2026-03/20260313_COMPREHENSIVE_MULTIAGENT_SYSTEM_ANALYSIS.md`
- `docs/reviews/2026-03/archive/20260314_DEEP_ANALYSIS_MULTIAGENT_SYSTEM_GEMINI.md`

## How To Use This File

- Read this first when planning.
- If a new review changes the roadmap, update this file.
- If a claim in another doc conflicts with this file, verify it against live code and then update whichever document is wrong.
