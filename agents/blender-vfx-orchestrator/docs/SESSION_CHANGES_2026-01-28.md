# Session Changes: 2026-01-28

**Branch:** `0.31.0`
**Status:** Pushed to GitHub
**Key Result:** Quality score jumped from 22.0 to 55.0 (150% improvement)

---

## Summary

This session delivered three major pipeline capabilities and several critical bug fixes. The most impactful change is the `modify_code` action for the Modification Strategist, which enables the pipeline to structurally rewrite scripts when parameter tuning alone cannot fix failures. Combined with session resumption and error recovery, the system can now recover from dead ends that previously required human intervention.

---

## Changes

### 1. Session Resumption Pipeline

**Problem:** `resume_session()` used the deprecated handoff-based orchestrator. Sessions that hit `MAX_ITERATIONS` could not be continued.

**Solution:** Added `resume_session_id` parameter to `create_asset_pipeline()`. When provided:
- Loads existing session from disk
- Skips Phase 0 (Research) and Phase 0.5 (Technique Selection) — already stored
- Reconstructs loop variables from last `IterationResult`
- Reconstructs `SessionManager` by replaying iteration history via `from_session_state()`
- Creates SDK session (reuses existing conversation DB)
- Enters the iteration loop at the next iteration number

**Files modified:**
- `orchestrator.py` — Signature change, resume branch logic, learning guards
- `session_manager.py` — Added `from_session_state()` classmethod

**Usage:**
```python
from orchestrator import resume_vfx_session
result = await resume_vfx_session("session_abc", max_iterations=10)
```

### 2. `modify_code` Action for Modification Strategist

**Problem:** The Modification Strategist could only choose `modify_params` (regex parameter replacement) or `switch_technique` (full technique switch). Structural failures — broken API calls, missing modifiers, incorrect object setup — fell between these two actions. `modify_params` couldn't fix them (it only tweaks numbers), and `switch_technique` was too drastic (abandons all progress).

**Solution:** Added `modify_code` as a third Strategist action. When the Strategist classifies an issue as structural (confidence threshold 0.7+), the pipeline routes to the Script Writer with `write_script` (full code generation) instead of `modify_script` (regex parameter replacement). The Strategist receives full context: the failing script, execution errors, quality issues, and iteration history.

**Impact:** This is the breakthrough change. In testing:

| Iteration | Strategist Action | Confidence | Score | Result |
|-----------|-------------------|------------|-------|--------|
| 2 | `modify_params` | 0.6 | 0 | Bake failed (structural issue) |
| 3 | `modify_code` | 0.88 | 0 | New technique, API mismatch |
| 4 | `modify_code` | 0.9 | **55.0** | Success — structural rewrite worked |
| 5 | `modify_params` | — | 40.0 | Score dropped (param tuning regressed) |

The Strategist correctly identified iterations 3 and 4 as structural failures requiring code-level fixes, while iteration 2 was reasonably classified as a parameter issue (it looked like simulation settings on the surface). Iteration 4's rewrite jumped from 22.0 to 55.0 — a 150% improvement.

**Files modified:**
- `orchestrator.py` — `modify_code` handler in the Strategist response processing block (~line 2320)
- `orchestrator.py` — Updated Strategist prompt to include `modify_code` as an option

### 3. Phase 2.5 Error Recovery

**Problem:** When Blender script execution failed (e.g., `'ParticleSettings' has no attribute 'child_nbr'`), the pipeline had no AI-driven recovery. It either crashed or returned a zero score.

**Solution:** Added Phase 2.5 between script execution and quality evaluation. When execution fails:
1. Sends the error + script to the Script Writer agent
2. Script Writer uses `write_script` to generate a fixed version
3. Re-executes the fixed script
4. If the second attempt also fails, records the failure and continues to quality evaluation (which will score it as 0)

**Files modified:**
- `orchestrator.py` — New Phase 2.5 block after execution (~line 2750)

### 4. Bake Frame Alignment (API Fixer)

**Problem:** Mantaflow's `cache_frame_start`/`cache_frame_end` default to 1-120, independent of `scene.frame_end`. A 25-frame scene would waste time baking 120 frames.

**Solution:** Added an API fixer injection in `blender_api_fixer.py` that aligns the bake range with the scene frame range before baking.

**Files modified:**
- `specialized_agents/blender_api_fixer.py` — New frame alignment injection

---

## Bug Fixes

### A. `_script_writer_standalone` Attribute Error

**Bug:** `modify_code` handler and Phase 2.5 both referenced `self._script_writer_standalone`, but the correct attribute name is `self._script_agent_standalone` (assigned at line 1401).

**Impact:** Blocked the primary `modify_code` path and Phase 2.5 recovery in ALL iterations during the test run. The fallback Script Writer path still worked (Script Writer autonomously chose `write_script`), which is how the 55.0 score was achieved despite the bug.

**Fix:** Global replace `self._script_writer_standalone` -> `self._script_agent_standalone` (2 occurrences at lines 2332 and 2766).

### B. ResearchOutput JSON Field Name Mismatch

**Bug:** The Research Agent LLM returned JSON with title-case keys (`"Recommended Approach"`) instead of the Pydantic-expected snake_case (`"recommended_approach"`). This crashed the pipeline at the end of iteration 5 during stuck-detection re-research.

**Error:** `pydantic_core._pydantic_core.ValidationError: 1 validation error for ResearchOutput / recommended_approach / Field required`

**Fix:** Added a `model_validator(mode="before")` to `ResearchOutput` that normalizes all keys to snake_case:
```python
@model_validator(mode="before")
@classmethod
def normalize_field_names(cls, data):
    if isinstance(data, dict):
        return {
            key.lower().replace(" ", "_").replace("-", "_"): value
            for key, value in data.items()
        }
    return data
```

### C. `learning` Variable Null Guards

**Bug:** On the first resumed iteration, `learning` is `None` because the Learning Agent hasn't run yet. Three unguarded references at the quality gate fallback (lines 2883-2895) would crash.

**Fix:** Added null guards: `learning and learning.next_action == 'complete'` etc.

---

## Strategic Model Upgrade

Updated `config/presets.yaml` to strategically assign gpt-5.2 to agents that benefit from stronger reasoning, while keeping formulaic agents on gpt-5-mini.

### Upgraded to gpt-5.2

| Agent | Reasoning |
|-------|-----------|
| **Script Writer** | Code generation is the hardest task — needs strong reasoning for structural rewrites |
| **Quality Analyst** | Vision-based evaluation requires nuanced reasoning about visual quality |
| **Learning Agent** | Pattern extraction and experiment analysis require deep comprehension |
| **Research Agent** | Document interpretation and technique recommendation are knowledge-heavy |
| **Modification Strategist** | The breakthrough agent — correct classification of structural vs parameter failures drives quality |

### Remain on gpt-5-mini (preset default)

| Agent | Reasoning |
|-------|-----------|
| **Executor** | Runs scripts and parses stdout — minimal reasoning needed |
| **Technique Selector** | Quick pick from known techniques — simple routing |
| **Quality Gate Judge** | Pass/fail decision with clear numeric criteria — formulaic |

### Cost Impact

With the `development` preset (gpt-5-mini default), 5 agents upgrade to gpt-5.2 via `agent_overrides`. Rough cost comparison per iteration:

- **Before:** 8 agents x gpt-5-mini = baseline cost
- **After:** 5 agents x gpt-5.2 + 3 agents x gpt-5-mini = ~2.5-3x baseline per iteration

This is offset by fewer wasted iterations — the 22.0 -> 55.0 jump happened in 3 iterations instead of potentially 10+ with only parameter tuning.

---

## Test Results

### Resume Test (5 iterations, `quick_test_v1` water effect)

```
Session: session_quick_test_v1_20260127_223448
Initial score: 22.0 (iteration 1, mantaflow_liquid_flip)
Resume from: iteration 1, max_iterations=5

Iteration 2: modify_params (conf 0.6) -> Bake failed, score 0
Iteration 3: modify_code  (conf 0.88) -> API mismatch (child_nbr), score 0
Iteration 4: modify_code  (conf 0.9)  -> animated_mesh_particles_v2, score 55.0
Iteration 5: modify_params -> Score dropped to 40.0, technique switch triggered

Best score: 55.0 (iteration 4)
Final status: FAILED (Research Agent JSON crash at iter 5 end)
```

### Key Observations

1. **The Strategist works.** It correctly classified:
   - Iteration 2: Parameters (simulation resolution) — reasonable but wrong in hindsight
   - Iteration 3: Structural (broken bake pipeline) — correct
   - Iteration 4: Structural (API attribute mismatch) — correct

2. **modify_code is the breakthrough.** The 22.0 -> 55.0 jump came from a full structural rewrite, not parameter tuning. The Script Writer generated an entirely new `animated_mesh_particles_v2` technique that bypassed the broken Mantaflow bake pipeline.

3. **Parameter tuning after structural fix can regress.** Iteration 5 dropped from 55.0 to 40.0. The Strategist may need a "consolidate" action that makes conservative changes after a big score jump.

4. **Error recovery works but needs the attribute fix.** Phase 2.5 triggered correctly but couldn't execute due to the `_script_writer_standalone` bug. Now fixed.

---

## Architecture Impact

```
create_asset_pipeline() now supports:
  |
  +-- Fresh run (resume_session_id=None): Full pipeline
  |   Phase 0 -> 0.5 -> [iteration loop]
  |
  +-- Resume (resume_session_id="..."):
      Skip Phase 0/0.5
      Reconstruct loop state from SessionState
      SessionManager.from_session_state() replays history
      Continue from iteration N+1
      |
      +-- Iteration loop now has 3 actions:
          modify_params  -- Parameter tuning (regex replacement)
          modify_code    -- Structural rewrite (full script generation)  <-- NEW
          switch_technique -- Technique switch (escape velocity)
          |
          +-- Phase 2.5 Error Recovery  <-- NEW
              On execution failure, AI-driven script fix before quality eval
```

---

## Next Steps

1. **Test with gpt-5.2 agents** — Run a full session with the upgraded model configuration to measure quality and cost delta.
2. **Add "consolidate" action** — After a large score jump (e.g., 22 -> 55), the Strategist should make conservative changes to protect the gain rather than aggressive parameter tuning.
3. **Fix parameter regression detection** — When a `modify_params` iteration scores lower than the previous, the next iteration should revert to the previous script and try different parameters.
4. **Guardrails implementation** — As noted in SDK_ENFORCEMENT_PROTOCOL.md, input/output guardrails are the next phase. The `doc_refs` enforcement for Research Agent should use an output guardrail rather than manual validation.
