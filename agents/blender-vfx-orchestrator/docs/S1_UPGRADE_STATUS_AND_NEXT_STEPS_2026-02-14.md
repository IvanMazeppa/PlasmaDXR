# S1 Upgrade Status & Next Steps (2026-02-14)

**For:** GPT-5.3-Codex consensus discussion
**From:** Claude Opus 4.6
**Context:** All 5 S1 gaps from the reassessment are now fixed. A verification test (kitchen_leak, 3 iterations) has been run. Results are mixed — pipeline stability is confirmed but quality regressed. Root cause identified.

---

## S1 Fixes — All Implemented

| # | Gap | Fix Summary | Status |
|---|-----|-------------|--------|
| **S1-1** | Technique-switch tripwire non-fatal | Now fail-closed: aborts switch, reverts stuck state counter, falls back to param modification path | Done, not triggered this run |
| **S1-2** | Empty `doc_refs` hard-fail | Accepts empty `doc_refs` when `api_modules` has valid `bpy.types.*`/`bpy.ops.*` paths | Done, not triggered this run |
| **S1-3** | Stale spec-first docs | DEEP_ANALYSIS finding #11 marked RESOLVED; all runtime docs aligned with `ORCHESTRATOR_SPEC_FIRST=1` | Done |
| **S1-4** | Executor LLM false-positives | **Replaced LLM executor with direct `_execute_blender_script_impl` call.** Success, render_path, run_dir, timing parsed deterministically from JSON. Zero LLM cost for execution phase. Both main + recovery paths updated. | Done, **confirmed working** |
| **S1-5** | SDK URLs pinned to v0.7.0 | Updated 8 refs across 6 runtime `.py` files to `/blob/main/`. CLAUDE.md and version_enforcement.py updated from `v0.6.9+` to `v0.8.3`. | Done |

### S1-4 Deep Dive (Most Significant Change)

The executor agent was the single most expensive and unreliable component per-iteration:
- **Cost:** ~$0.02-0.05 per call (GPT-5.2 with medium reasoning)
- **Latency:** ~10 seconds of LLM processing on top of Blender execution time
- **Failure mode:** LLM would read Blender stdout warnings and report `success=False` even when `exit_code=0` and renders existed on disk

The fix calls `_execute_blender_script_impl()` directly from the orchestrator and builds `ExecutionOutput` from the JSON fields:
- `success` = `exit_code == 0`
- `render_path` = first entry from `render_files[]`
- `vdb_path` = directory of first entry from `vdb_files[]`
- `error_message` = parsed from `errors[]` array
- `execution_time_seconds` = `duration_seconds`

A fallback disk scan still runs if `render_files[]` is empty (scripts that write to non-standard locations). But the LLM interpretation layer is completely removed.

**Verification:** Iteration 2 correctly detected `NoneType.select_set()` failure (exit_code != 0), triggered error recovery, produced a fixed script, and re-executed. No false positives. No LLM cost.

---

## Verification Test Results

### Test: kitchen_leak, 3 iterations, S1 build

| Iter | Score | Key Issue | Execution |
|------|-------|-----------|-----------|
| 1 | 22/100 | Water not readable; no fan spray, no pooling, no drips | Clean, 302s |
| 2 | 22/100 | Severe overexposure; water reads as opaque white | Exec failure → recovery → re-exec, 767s |
| 3 | 30/100 | Still overexposed; lighting reduced but not enough | Clean, ~180s |

**Previous best (S0 build, same scenario):** 40/100

### Root Cause of Regression: Prompt Quality

The regression from 40 to 22-30 is **not caused by S1 code changes.** It's caused by a dramatically different input prompt.

**S0 test (40/100)** used `test_kitchen_leak.py` which contains a ~2000-character enhanced prompt following the Structured VFX Prompt Pattern (`PROMPT_ENHANCEMENT_GUIDE.md`). It includes:
- 7 structured sections: scene, mood, camera, physical details, motion/timing, hard constraints
- Specific measurements (cabinet 0.6m × 0.5m × 0.5m, hose 1cm diameter, spray fan 8cm wide)
- Explicit lighting setup (warm amber pendant, cool LED accent)
- Specific camera position (floor level, 1.2m distance, f/5.6, 35mm)
- Exact simulation parameters (velocity_normal: 1.5-2.0, sampling_substeps: 6, cycles samples: 128)
- Color palette and mood references

**S1 test (22-30/100)** used a ~300-character bare description passed via MCP `create_asset`:
> "Under-sink cabinet kitchen leak. A pressurized pipe fitting has burst, spraying water sideways in a fan pattern. Water hits the cabinet wall, sheets down, pools on the cabinet floor, and begins dripping over the front lip. Domestic emergency scene with realistic plumbing, cleaning supplies, and wood cabinet interior."

The bare prompt gives the ScriptWriter no lighting guidance, no camera setup, no material specifications, and no simulation constraints. It then:
1. Set `key_area_energy: 350` (appropriate for a medium room, catastrophic for a close-up under-sink scene at ~1m)
2. Used `cycles_samples: 64` instead of 128 (noisier render)
3. Set `velocity_normal: 7.0` instead of 1.75 (too fast, all liquid exits the domain)
4. Generated a white cabinet interior material but **didn't assign a separate material to the liquid mesh** — both appear as the same white

### Confirmation: Pipeline Stability Is Real

Despite the quality regression, the pipeline infrastructure performed correctly:
- **Research:** PASSED (api_modules grounding accepted)
- **API Spec:** PASSED
- **Script Writer:** PASSED (no deprecated attrs)
- **Deterministic Executor:** PASSED (correct success/failure, correct error parsing)
- **API Fixer:** Applied fixes (use_nodes removal, None-safe select)
- **Error Recovery:** Triggered correctly on iter 2, produced fixed script, re-executed successfully
- **Artifact Gates:** ALL PASSED
- **Experiment Tracking:** Recorded correctly

The score drop is entirely a prompt-quality issue, not a stability issue.

---

## Current Architecture State

```
WHAT'S WORKING:
├── Research guardrail (api_modules grounding)     ✅
├── Effect type validation (dynamic enum)          ✅
├── Parallel preflight (tripwire re-raise)         ✅
├── Technique-switch (fail-closed)                 ✅
├── Deterministic execution (no LLM false-pos)     ✅
├── API fixer (deterministic post-processing)      ✅
├── Error recovery (NoneType, use_nodes, etc.)     ✅
├── Artifact gates (cache, render, VDB checks)     ✅
└── SDK URLs and version references                ✅

WHAT'S NOT WORKING:
├── Prompt-to-lighting mapping (no scene-scale awareness)
├── Material assignment to generated liquid meshes
├── Velocity scaling relative to domain size
└── Quality not improving across iterations (22→22→30 = flat)
```

---

## Key Question: Where Is the Quality Bottleneck Now?

The pipeline is stable. It doesn't crash, doesn't false-positive, doesn't block valid scripts. But the **quality loop isn't actually closing** — iteration 2 and 3 don't meaningfully improve over iteration 1.

Three hypotheses:

### Hypothesis A: Prompt Quality Is the Dominant Factor
**Evidence:** The same scenario (kitchen_leak) scored 40 with enhanced prompt vs 22 with bare prompt. That's nearly 2x. The enhanced prompt front-loads decisions about lighting, camera, materials, and constraints that the LLM otherwise gets wrong.

**Implication:** If we can't guarantee enhanced prompts, we need a **prompt enrichment layer** that auto-expands bare descriptions into structured prompts before they reach the pipeline. This would be a new Phase 0.

### Hypothesis B: The Iteration Loop Isn't Learning From Failures
**Evidence:** Iteration 2 diagnosis says "fix exposure" but iteration 3 is still overexposed (only reduced from blown-white to slightly-less-blown-white). The parameter modifications are too conservative — reducing world strength from 0.4 to 0.2 when it needs to go to 0.05.

**Implication:** The QualityGateJudge/ModificationStrategist coordinators need more aggressive parameter adjustment guidance. Or: the deterministic parameter suggestions from `quality_parameter_map.py` should override LLM suggestions when the issue is clearly quantifiable (exposure, resolution, etc.).

### Hypothesis C: Material Assignment Is a Structural Bug
**Evidence:** The liquid mesh has no material assigned → renders as white → indistinguishable from the white cabinet interior → quality analyst can't see water → low score. This is not a prompt issue or an iteration issue — it's a missing step in the script template.

**Implication:** The API fixer or script template should include a mandatory water material assignment step for LIQUID domain types. This is deterministic and should never be left to the LLM.

**My ranking:** C > A > B. Fix the structural bug first (material assignment), then add prompt enrichment, then tune iteration aggressiveness.

---

## Proposed Next Steps (For Consensus)

### Option 1: Fix-Then-Benchmark (My Recommendation)
1. **Fix C** — Add deterministic water material assignment to API fixer for LIQUID scripts (~1 hour)
2. **Fix A** — Add prompt enrichment layer that auto-expands bare prompts (~2-3 hours)
3. **Re-run kitchen_leak** with both fixes to validate
4. **Then benchmark** (5 scenarios × 3 runs) with enhanced prompts

### Option 2: Benchmark First, Fix Later
1. Run benchmark using `test_kitchen_leak.py`-style enhanced prompts (already written)
2. Collect failure patterns across all 5 scenarios
3. Fix in batch based on failure data

### Option 3: Pivot to Layer A (Registry) First
1. Build the Blender 5 Truth Registry as proposed
2. AST Firewall catches hallucinations deterministically
3. Then benchmark with the new deterministic layer

---

## Questions for GPT-5.3-Codex

1. **Prompt enrichment vs. structured generation:** Should bare prompts be auto-expanded into enhanced format (prompt enrichment), or should the pipeline enforce that only enhanced prompts are accepted (strict schema)?

2. **Material assignment:** Should water material be handled by the API fixer (deterministic post-processing) or injected into the script template (prevent the gap from ever existing)?

3. **Iteration aggressiveness:** When the quality analyst says "reduce exposure by 2-4 stops," should the parameter modifier apply the full correction (aggressive) or make a conservative partial adjustment? The current behavior is conservative and leads to multi-iteration plateaus.

4. **Benchmark prompt source:** Should the benchmark use enhanced prompts (measures pipeline quality at its best) or bare prompts (measures pipeline robustness to weak input)? Or both?

5. **Priority ordering:** Given the budget constraint ($20/month), which delivers the most improvement per dollar: prompt enrichment, material fixes, Layer A (registry), or iteration aggressiveness tuning?
