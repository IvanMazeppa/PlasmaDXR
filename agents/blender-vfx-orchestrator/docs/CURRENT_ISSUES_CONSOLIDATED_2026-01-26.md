# Current Issues Consolidated (2026-01-26 - Updated)

**Status:** Supporting issues list. Priorities live in `docs/MASTER_ROADMAP_2026-01-26.md`.

This document consolidates **current problems and blockers** across recent issue
documents to make triage easier. Use this as the primary "what's broken" list.

**Single roadmap:** `docs/MASTER_ROADMAP_2026-01-26.md`

## Sources Consolidated
- `docs/ARCHITECTURE_FAILURE_ANALYSIS_2026-01-25.md`
- `docs/CRITICAL_AUDIT_LLM_HALLUCINATION_2026-01-25.md`
- `docs/TRACING_SHAKEDOWN_TASKLIST_2026-01-25.md`
- `docs/SHAKEDOWN_TEST_FINDINGS_2026-01-24.md`
- `docs/REMEDIATION_PLAN_2026-01-23.md`

## Active Roadmap
**See `docs/MASTER_ROADMAP_2026-01-26.md` for prioritized implementation plan.**

---

## PHASE-4 GATE: PASSED (2026-01-26 19:05 UTC)

**Trace:** `traces/e2e_test_verbose_20260126_185722.jsonl`
**Result:** Quality score 40.0 met threshold at iteration 1
**Duration:** 452 seconds (7.5 minutes)

### Pipeline Execution Summary

| Phase | Status | Duration | Details |
|-------|--------|----------|---------|
| Research Agent | ✅ PASS | 71.9s | Bundle-first, 3 tools |
| Technique Selector | ✅ PASS | 18.5s | Selected mantaflow_smoke |
| API Spec Agent | ⚠️ BLOCKED | 36.6s | Tried to skip bundle → blocked by hooks |
| Script Writer (Fallback) | ✅ PASS | 78.5s | Completed with doc queries |
| API Validator | ✅ PASS | <1s | 12/12 valid |
| Executor | ✅ PASS | 117.7s | Render completed (98.3s) |
| Quality Analyst | ✅ PASS | 81.8s | Score 40.0, 7 issues |
| Learning Agent | ✅ PASS | 36.9s | Experiment recorded |
| Quality Gate | ✅ PASS | 10.5s | Score meets threshold |

### Key Findings

1. **Bundle-first enforcement WORKING**
   - API Spec Agent tried `semantic_search_blender_docs` first
   - Was correctly blocked with `BUNDLE-FIRST REQUIRED` error
   - Fallback to Script Writer triggered and completed successfully

2. **Camera fix VERIFIED**
   - Render completed without `Cannot render, no camera` error
   - API fixer or instructions ensured camera was present

3. **Enum guardrail VERIFIED (unit test)**
   - Unit tests confirm invalid enums like `flow_behavior='FLOW'` are rejected
   - Spec-first didn't reach Code Writer (fallback triggered), so runtime validation not exercised

4. **Doc search discipline IMPROVED**
   - 90% reduction in doc searches (80+ → 8 in earlier runs)
   - Bundle-first enforcement prevents spam

---

## Architecture Problems (Systemic - Mostly Resolved)

1) **✅ RESOLVED: No authoritative API truth source.**
   - Spec-first pipeline with bundle-first enforcement provides verified attributes.
   - Doc searches are bounded and enforced.

2) **✅ RESOLVED: Validation was permissive.**
   - Output guardrails validate doc_refs and enum values.
   - API validator runs before execution.

3) **✅ RESOLVED: Enforcement was opt-out.**
   - RunHooks enforce bundle-first for API Spec Agent.
   - Doc-query enforcement for Script Writer.

4) **⚠️ PARTIALLY RESOLVED: Feedback loop.**
   - Learning Agent records experiments.
   - Still need better error-to-parameter correlation.

---

## Current Blockers (Minor)

1) **API Spec Agent ignores bundle-first instructions**
   - Instructions clearly state "TURN 1: Bundle-first doc search (MANDATORY)"
   - Model skips to targeted search first
   - Enforcement hook blocks this, causing fallback
   - **Impact:** Minimal (fallback works), but wastes API calls

2) **Doc search precision**
   - Still returns some unrelated results
   - Bundle search helps by providing multiple relevant results at once
   - **Impact:** Low (validated by guardrails)

---

## Fixed / Verified (This Session - 2026-01-26)

- **✅ Bundle-first enforcement** - RunHooks block targeted searches until bundle called
- **✅ Camera fix** - Render completes (API fixer injects camera if missing)
- **✅ Enum guardrail** - Unit tested, rejects invalid values like 'FLOW'
- **✅ Turn budget** - Increased to 10/12 for API Spec Agent
- **✅ Loop detection limits** - `max_consecutive_same_tool=30`, `max_exempt_tool_calls=40`
- **✅ Full pipeline completion** - Quality score 40.0, all phases passed
- **✅ Phase-4 gate criteria met**

---

## Previously Fixed / Verified

- **Doc-query enforcement** for Script Writer (RunHooks).
- **Budget guardrail** uses `get_spent()` + `monthly_limit` (no `get_remaining()`).
- **Vision model default** for evaluation set to `gpt-5-mini`.
- **Test harness crash** (`IterationResult.primary_issue`) resolved.
- **Spec-first pipeline guardrails** - Both `validate_api_spec` and `validate_code_against_spec` working.
- **Loop detection** - `max_consecutive_same_tool` prevents infinite loops.
- **Verbose tracing** - Trace files created correctly.
- **Parallel tool execution** - API Spec Agent correctly batches doc searches.
- **Files NOT being deleted** - Confirmed preservation in `build/vdb_output/`.
- **Enum value guardrail** - Added enum validation in `validate_code_against_spec`.

---

## Next Steps (Post Phase-4)

1) **Investigate API Spec Agent instruction adherence**
   - Model ignores bundle-first instruction
   - Consider few-shot examples or stronger prompting

2) **Optimize fallback path**
   - Current: API Spec blocked → Script Writer fallback
   - Goal: API Spec succeeds (uses bundle) → Code Writer (no fallback needed)

3) **Expand test coverage**
   - Test other effect types (explosion, fire, sun)
   - Test 2+ iterations
   - Test near-budget scenarios

---

## Test Evidence

| Test | Date | Result | Trace |
|------|------|--------|-------|
| Phase-4 Gate | 2026-01-26 19:05 | ✅ PASS | `e2e_test_verbose_20260126_185722.jsonl` |
| 2-Iter Shakedown | 2026-01-26 19:27 | ❌ FAIL | (no trace - stdout only) |
| Shakedown v1 | 2026-01-26 18:42 | ⚠️ TIMEOUT | `phase4_gate_shakedown_20260126_184218.jsonl` |
| Test v10 | 2026-01-26 04:20 | ❌ Loop detection | `e2e_test_v10_20260126_042056.jsonl` |
| Test v9 | 2026-01-26 02:45 | ❌ Enum error | `e2e_test_v9_20260126_024559.jsonl` |

---

## 2-Iteration Shakedown Findings (2026-01-26 19:27 UTC)

**Test:** `python test_quick_e2e.py --preset quick_test --effect smoke --iterations 2`

### Iteration 1 Results
- API Spec Agent: max_turns=8 exceeded → fallback to Script Writer
- Script Writer: Generated script with `velocity_multi` (HALLUCINATED)
- Executor: `AttributeError: 'FluidFlowSettings' object has no attribute 'velocity_multi'`
- Quality: Not reached due to execution failure

### Iteration 2 Results
- Modification Coordinator: Guardrail rejected nested output (expected)
- Script Writer fallback: Blocked by `DocQueryRequiredError` on `modify_script`
- **Root cause:** `create_script_writer_hooks()` requires doc query before `modify_script`
- **Fix applied:** Use `create_fallback_script_writer_hooks()` for iteration 2+ (no doc requirement since Research Agent already queried)

### Fixes Applied This Session
1. **VERSION_TRUTH.md** - Added `velocity_multi` → `velocity_factor` mapping
2. **orchestrator.py:2217** - Use fallback hooks for iteration 2+ Script Writer

### Verification Run (19:33 UTC)
- **Iteration 1:** Script Writer generated `noise_scale = 1.0` (float) - TypeError
- **Iteration 2:** Ran without `DocQueryRequiredError` (FIX VERIFIED)
- **Coordinator:** Correctly identified fix `noise_scale: 1` (int)
- **modify_script:** Applied zero changes (contract issue - tracked)
- **Result:** `max_iterations` reached, score 0.0

---

## DEEP DIVE: Coordinator → modify_script Contract Issue (2026-01-26 19:45 UTC)

### Problem Statement
The Modification Coordinator correctly diagnoses issues and outputs fixes, but `_modify_script_impl` fails to apply the changes.

### Evidence from Test Run

**Coordinator Output:**
```python
{
    'FluidDomainSettings.noise_scale': 1,
    'FluidDomainSettings.noise_strength': 0.4,
    'FluidDomainSettings.resolution_max': 64,
    'FluidFlowSettings.temperature': 1.0,
    'FluidFlowSettings.density': 1.0,
    'FluidFlowSettings.initial_velocity_z': 2.0
}
```

**modify_script Result:**
```
Changes Made: NONE
```

### Root Cause Analysis

**The contract has THREE incompatible formats:**

| Component | Expected Format | Example |
|-----------|-----------------|---------|
| Coordinator output | Blender API paths | `FluidDomainSettings.noise_scale` |
| Generated script | Direct attribute assignment | `dsettings.noise_scale = 1.0` |
| `_modify_script_impl` | Config class pattern | `class Config: NOISE_SCALE = 1` |

**Why `_modify_script_impl` fails:**

1. **Line 775:** Converts param to uppercase: `param_upper = param.upper()`
   - Input: `FluidDomainSettings.noise_scale`
   - Result: `FLUIDDOMAINSETTINGS.NOISE_SCALE`

2. **Line 800-818:** Searches for `class Config:` section with pattern:
   - `(\s+{param_upper}\s*=\s*)([^\n]+)`
   - Looks for: `FLUIDDOMAINSETTINGS.NOISE_SCALE = value`
   - Script has: `dsettings.noise_scale = 1.0`
   - **No match found**

3. **Line 821-829:** Fallback searches for `Config.PARAM` pattern:
   - Looks for: `Config.FLUIDDOMAINSETTINGS.NOISE_SCALE = value`
   - Script has: `dsettings.noise_scale = 1.0`
   - **No match found**

### Script Structure Analysis

The Script Writer generates code like this:
```python
# NO Config class - direct Blender API calls
dsettings = mod.domain_settings
dsettings.domain_type = 'GAS'
dsettings.resolution_max = 64
dsettings.noise_scale = 1.0  # <-- This is what needs modification

fsettings = modf.flow_settings
fsettings.flow_type = 'SMOKE'
fsettings.temperature = 1.0
```

But `_modify_script_impl` expects:
```python
class Config:
    NOISE_SCALE = 1
    RESOLUTION_MAX = 64
    # etc.

# Later in script
dsettings.noise_scale = Config.NOISE_SCALE
```

### Solution Options

**Option A: Fix `_modify_script_impl` to handle direct attribute patterns**
- Add pattern: `(dsettings|fsettings|settings)\.{param_name}\s*=\s*([^\n]+)`
- Map Coordinator keys: `FluidDomainSettings.X` → `dsettings.X`
- Low risk, surgical fix

**Option B: Change Coordinator output format**
- Output `noise_scale` instead of `FluidDomainSettings.noise_scale`
- Update instructions + guardrails
- Medium risk, requires prompt engineering

**Option C: Standardize Script Writer to use Config class**
- All scripts use `class Config:` for parameters
- `_modify_script_impl` already handles this
- High risk, requires Script Writer changes

### Recommended Fix: Option A

Add direct attribute pattern matching to `_modify_script_impl`:

```python
# Map Coordinator prefixes to script variable names
PREFIX_MAP = {
    'FluidDomainSettings': ['dsettings', 'domain_settings', 'dom'],
    'FluidFlowSettings': ['fsettings', 'flow_settings', 'flow'],
}

# For param like 'FluidDomainSettings.noise_scale':
prefix, attr = param.rsplit('.', 1) if '.' in param else (None, param)
if prefix in PREFIX_MAP:
    for var_name in PREFIX_MAP[prefix]:
        pattern = rf"({var_name}\.{attr}\s*=\s*)([^\n]+)"
        # ... apply replacement
```

### Impact Assessment

- **Phase-4 gate:** Still PASSED (single iteration works)
- **Multi-iteration:** Broken (Coordinator fixes don't apply)
- **Autonomy goal:** Blocked until fixed

---

## Full Test Session Log (2026-01-26)

### Test 1: Phase-4 Gate Verification (19:05 UTC)
- **Command:** Single iteration E2E test
- **Result:** ✅ PASS - Quality score 40.0
- **Trace:** `e2e_test_verbose_20260126_185722.jsonl`

### Test 2: 2-Iteration Shakedown Pre-Fix (19:27 UTC)
- **Command:** `python test_quick_e2e.py --preset quick_test --effect smoke --iterations 2`
- **Iteration 1:** Script Writer hallucinated `velocity_multi` → `AttributeError`
- **Iteration 2:** `DocQueryRequiredError` on `modify_script`
- **Result:** ❌ FAIL

### Test 3: 2-Iteration Shakedown Post-Fix (19:33 UTC)
- **Command:** Same as Test 2
- **Fix Applied:** `create_fallback_script_writer_hooks()` for iteration 2+
- **Iteration 1:** Script Writer used `noise_scale = 1.0` (float) → `TypeError: expected int`
- **Iteration 2:** Coordinator correctly identified fix `noise_scale: 1`, but modify_script applied NONE
- **Result:** ❌ FAIL (but DocQueryRequiredError FIX VERIFIED)

### Files Modified This Session

| File | Change | Commit Status |
|------|--------|---------------|
| `docs/VERSION_TRUTH.md` | Added `velocity_multi` hallucination | Uncommitted |
| `orchestrator.py:2217` | Use fallback hooks for iter 2+ | Uncommitted |
| `docs/CURRENT_ISSUES_CONSOLIDATED_2026-01-26.md` | This document | Uncommitted |
| `docs/MASTER_ROADMAP_2026-01-26.md` | Updated status | Uncommitted |

### Hallucinations Discovered This Session

| Hallucinated Attribute | Correct Attribute | Location |
|------------------------|-------------------|----------|
| `velocity_multi` | `velocity_factor` | `FluidFlowSettings` |
| `noise_scale = 1.0` (float) | `noise_scale = 1` (int) | `FluidDomainSettings` |

### Enforcement Hooks Working

| Hook | Behavior | Verified |
|------|----------|----------|
| Bundle-first | Blocks targeted search before bundle | ✅ |
| Doc-query required | Blocks write/modify without doc query | ✅ |
| Fallback hooks | Allows modify_script without doc query | ✅ |
| Loop detection | Prevents infinite same-tool calls | ✅ |
| Turn budget | Triggers fallback at max_turns=8 | ✅ |

---

## Priority Fix Queue

| Priority | Issue | Impact | Fix Location |
|----------|-------|--------|--------------|
| P0 | Coordinator → modify_script contract | Multi-iteration broken | `tools/script_generator_tools.py` |
| P1 | noise_scale type (float vs int) | Execution fails | Script Writer instructions or guardrail |
| P2 | API Spec Agent bundle-first | Wastes API calls | Prompt engineering or few-shot |

---

*Document updated: 2026-01-26 20:00 UTC*
*Investigation conducted by: Claude Opus 4.5*
*Trace files: See `traces/` directory*

## GPT-5-mini User Shakedown (2026-01-26 19:35 UTC)

**Report:** `docs/TEST_REPORT_GPT5_MINI_SHAKEDOWN_2026-01-26.md`
**Result:** ✅ SUCCESS (with fallback)
- Confirmed `gpt-5-mini` viability for full pipeline.
- Confirmed **Spec-First Fallback** robustness (Spec Agent timed out -> Script Writer saved the run).
- Confirmed **Verbose Tracing** functionality.
