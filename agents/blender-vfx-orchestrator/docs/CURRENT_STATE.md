# Current State

Status: authoritative
Last verified: 2026-03-17
Purpose: current high-level state of the Blender VFX Orchestrator after Wave 1 and early Wave 2 work

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

6. Scene aesthetic realism is still below target even though the craft/intention layer has improved.
   The current bottleneck is no longer just "pick the right technique" but "consistently build convincing hero objects, materials, and environments."

7. A specialist section-builder workstream now exists as a planned quality-ceiling extension, but it is still gated and pending final external review before implementation begins.

## Current Planning Priorities

In order:

1. ~~Make repair-mode selection authoritative.~~ **DONE.** State authority fixed, repair routing receives correct inputs.
2. ~~Unify iteration manifests and state mutation.~~ **DONE.** `record_iteration()` is the sole path.
3. Finish evaluator calibration and provenance-safe learning before making bigger autonomy claims. (Roadmap Phase 4-5)
4. Continue the scene aesthetic realism workstream so hero objects, materials, and environments stop looking underbuilt. (Roadmap Phase 3.5)
5. Keep specialist section builders as a planned, gated extension until the first milestone is externally cleared. (Roadmap Phase 6A)

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
