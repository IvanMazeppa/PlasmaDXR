# Current State

Status: authoritative
Last verified: 2026-03-14
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

1. Repair routing is still not authoritative enough.
   Parameter-first fast paths can still become the effective repair policy before structural repair has had a clean chance to act.

2. Iteration-state authority is still split.
   Session truth, stuck-state, and iteration recording are not yet cleanly unified into one canonical mutation path.

3. The evaluator-to-repair boundary is still too prose-heavy.
   The runtime still has to infer whether an issue is structural, parametric, or technique-level.

4. `modify_code` and section patching are present but need stronger end-to-end proof in live runs.

5. Evaluator calibration and learning provenance are not yet strong enough to justify larger autonomy claims.

## Current Planning Priorities

In order:

1. Make repair-mode selection authoritative.
2. Unify iteration manifests and state mutation.
3. Add typed repair semantics at the QA boundary.
4. Prove `modify_code -> patch section -> execute -> evaluate` on real cases.
5. Calibrate evaluators and learning evidence before expanding autonomy.

The canonical implementation plan for these priorities now lives in `CURRENT_ROADMAP.md`.

## Supporting Analysis

Use these as the main supporting March 2026 review documents:

- `docs/reviews/2026-03/20260314_DEEP_AUTONOMY_ANALYSIS_GPT_54_XHIGH.md`
- `docs/reviews/2026-03/20260314_ARCHITECTURE_REVIEW_MODIFY_CODE_RESPONSE.md`
- `docs/reviews/2026-03/20260312_WAVE2_STATUS_AND_PATH_FORWARD.md`

Use these as supplemental external/opposing views, not canonical truth:

- `docs/reviews/2026-03/20260313_COMPREHENSIVE_MULTIAGENT_SYSTEM_ANALYSIS.md`
- `docs/reviews/2026-03/archive/20260314_DEEP_ANALYSIS_MULTIAGENT_SYSTEM_GEMINI.md`

## How To Use This File

- Read this first when planning.
- If a new review changes the roadmap, update this file.
- If a claim in another doc conflicts with this file, verify it against live code and then update whichever document is wrong.
