# Codex Response to S1 Upgrade Status and Latest Kitchen Leak Test (2026-02-14)

## Scope Reviewed
- `agents/blender-vfx-orchestrator/docs/S1_UPGRADE_STATUS_AND_NEXT_STEPS_2026-02-14.md`
- `agents/blender-vfx-orchestrator/docs/DEEP_ANALYSIS_INVESTIGATION_2026-02-12.md`
- `agents/blender-vfx-orchestrator/docs/CLAUDE_RESPONSE_TO_REASSESSMENT_2026-02-14.md`
- Runtime validation in:
  - `agents/blender-vfx-orchestrator/orchestrator.py`
  - `agents/blender-vfx-orchestrator/guardrails/research_guardrails.py`
  - `agents/blender-vfx-orchestrator/tools/blender_api_fixer.py`
  - `agents/blender-vfx-orchestrator/tools/script_generator_tools.py`
  - `agents/blender-vfx-orchestrator/utils/quality_parameter_map.py`

## Executive Position
S1 delivered real infrastructure improvements, especially deterministic execution.  
However, the latest quality plateau (22 -> 22 -> 30) is not only prompt weakness. There are still fail-open paths and missing deterministic liquid-specific corrections that can keep quality flat even when API hallucinations are reduced.

Recommendation: insert an **S1.5 stabilization sprint** before P1 autonomy work.

## Confirmed S1 Improvements (Code-Verified)
1. Deterministic executor is live in main and recovery paths:
   - `agents/blender-vfx-orchestrator/orchestrator.py:3180`
   - `agents/blender-vfx-orchestrator/orchestrator.py:3424`
2. Parallel preflight now re-raises `OutputGuardrailTripwireTriggered`:
   - `agents/blender-vfx-orchestrator/orchestrator.py:985`
3. Research guardrail accepts empty `doc_refs` when `api_modules` has `bpy.types.*`/`bpy.ops.*`:
   - `agents/blender-vfx-orchestrator/guardrails/research_guardrails.py:120`
4. Technique-switch tripwire now aborts switch and reverts state:
   - `agents/blender-vfx-orchestrator/orchestrator.py:4111`
5. Effect-type guardrail uses enum-derived values:
   - `agents/blender-vfx-orchestrator/guardrails/script_guardrails.py:33`

These are meaningful gains and should be retained.

## Remaining High-Impact Gaps

### 1) Spec-first can still fail open to legacy writer
When API Spec agent fails, pipeline falls back to original Script Writer:
- `agents/blender-vfx-orchestrator/orchestrator.py:1516`

Impact: Blender-5 legality is no longer guaranteed in failure cases.

### 2) Technique+Spec parallel path does not re-raise tripwire/critical exceptions
Parallel gather uses `return_exceptions=True` and treats failures as `None`:
- `agents/blender-vfx-orchestrator/orchestrator.py:1372`

Impact: can silently degrade strategy quality and continue.

### 3) Execution success can still be force-overridden by file discovery
If a render exists, success is flipped true even after non-zero execution state:
- `agents/blender-vfx-orchestrator/orchestrator.py:3267`
- `agents/blender-vfx-orchestrator/orchestrator.py:3465`

Impact: learning loop receives contaminated success/failure signal.

### 4) Script modification path still bypasses tool hook lifecycle
`_apply_script_modifications` calls `_modify_script_impl` directly:
- `agents/blender-vfx-orchestrator/orchestrator.py:838`

Impact: less observability/guardrail consistency on highest-risk mutation path.

### 5) Liquid material determinism is still missing
API fixer intentionally skips volume-material injection for liquid domains (correct), but there is no equivalent deterministic liquid-surface material assignment safety net:
- `agents/blender-vfx-orchestrator/tools/blender_api_fixer.py:996`

Impact: "water not readable / white-on-white" can persist even with valid APIs.

### 6) Deterministic quality mapping remains volume-biased for some issues
For overexposure, default deterministic params are `emission_strength` + `density`:
- `agents/blender-vfx-orchestrator/utils/quality_parameter_map.py:31`

Impact: useful for smoke/fire, often weak for liquid readability/exposure problems.

## Root-Cause Model for the Latest Plateau
The S1 report is right that prompt quality mattered in this run.  
But plateau likely comes from **combined factors**:

1. Under-specified prompt caused weak initial lighting/camera/material choices.
2. Missing deterministic liquid material assignment kept water readability low.
3. Iteration loop relied on conservative or non-targeted parameter edits.
4. Success override logic may hide partial failures and mislead adaptation.

## Proposed Next Step: S1.5 Stabilization Sprint

### Workstream A: Fail-Closed Correctness Containment (1-2 days)
1. Add strict runtime mode (`ORCHESTRATOR_STRICT_STABILITY=1`):
   - No fallback from spec-first to legacy writer.
   - Mandatory phase exceptions/tripwires abort phase deterministically.
2. In Technique+Spec parallel path, re-raise guardrail tripwires and treat API-Spec failure as blocking in strict mode.
3. Disable render-exists success override in strict mode (keep diagnostic-only flag if needed).

Acceptance gates:
- `generation_mode=spec_first` for all strict runs.
- `phase_abort_reason` explicit, never silent degrade.
- `execution_success_override_count == 0` in strict runs.

### Workstream B: Deterministic Liquid Visual Baseline (1-2 days)
1. Implement `_inject_liquid_material_setup(...)` in API fixer:
   - Detect LIQUID domain scripts.
   - Ensure generated liquid mesh gets a water-like Principled BSDF baseline (IOR/transmission/roughness).
   - Do not overwrite explicit user-authored water material if present.
2. Add deterministic liquid visibility checks:
   - Mesh enabled (`use_mesh` true for liquid).
   - At least one mesh object with non-empty material slot before evaluation.
3. Add optional velocity sanity clamp relative to domain scale to prevent "all fluid exits domain" starts.

Acceptance gates:
- `liquid_material_missing_count == 0`
- `liquid_mesh_missing_count == 0`

### Workstream C: Quality Loop Hardening (1 day)
1. Promote deterministic corrections to first-class for quantifiable issues:
   - Exposure: explicit stop-based reductions for light/world/shader intensity.
   - Readability: deterministic liquid shader targets before free-form LLM tuning.
2. Add `modification_match_rate` telemetry from `_modify_script_impl`:
   - if low across 2 iterations, escalate to regeneration instead of micro-tweaks.

Acceptance gates:
- `modification_match_rate >= 0.6` median on iteration 2+ runs.
- No 3-iteration flatlines without escalation action.

### Workstream D: Dual-Lane Benchmarking (1 day)
Run both lanes, do not pick one:
1. **Capability lane (enhanced prompts)**: measures max quality ceiling.
2. **Robustness lane (bare prompts)**: measures resilience to weak input.

Minimum suite:
- `kitchen_leak`, `wine_pour`, `fire_pillar`, `campfire_scene`, `smoke_test`
- 3 runs each lane (30 total) if budget allows.
- If budget constrained, start with 2 liquid + 1 gas scenario in both lanes.

## Direct Answers to S1 Questions

1. Prompt enrichment vs strict schema:
   - Use both: strict input schema + auto-enrichment adapter.
   - Reject malformed requests; enrich minimal valid prompts.

2. Where should liquid material assignment live:
   - Primary in script template/spec transpiler.
   - Safety net in API fixer (must be deterministic and idempotent).

3. Iteration aggressiveness:
   - Aggressive for quantifiable errors (exposure, no-fluid, no-mesh).
   - Conservative for aesthetic style shifts.

4. Benchmark prompt source:
   - Both enhanced and bare prompts are required; they answer different questions.

5. Priority under budget:
   1. Fail-closed containment + success-signal integrity
   2. Deterministic liquid material/visibility baseline
   3. Quality loop hardening
   4. Prompt enrichment polish
   5. Full registry/firewall expansion

## Collaboration Pack for Claude Opus 4.6
Ask Claude to produce three implementation specs that align with this document:

1. **Strict-mode contract spec**
   - Exact phase-by-phase fail/abort matrix.
   - Which fallbacks are disabled in strict mode.

2. **Liquid deterministic baseline spec**
   - Water material default node graph.
   - Safe detection and no-overwrite rules.
   - Validation checks before quality evaluation.

3. **Benchmark protocol spec**
   - Two-lane prompt methodology.
   - Pass/fail scorecard and budget-aware run schedule.

## Exit Criteria to Start P1
Do not start P1 autonomy/self-learning expansion until all are true:
- Deprecated/hallucinated attribute leaks in final scripts: `0` across benchmark corpus.
- Strict-mode runs: `0` silent fallbacks and `0` execution success overrides.
- Liquid scenarios: median score improvement over current S1 baseline and no flatline pattern over 3 iterations.
- All failure outcomes produce explicit diagnosis + fix artifacts.
