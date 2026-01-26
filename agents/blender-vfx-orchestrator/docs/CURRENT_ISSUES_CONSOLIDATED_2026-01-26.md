# Current Issues Consolidated (2026-01-26)

This document consolidates **current problems and blockers** across recent issue
documents to make triage easier. Use this as the primary “what’s broken” list.

## Sources Consolidated
- `docs/ARCHITECTURE_FAILURE_ANALYSIS_2026-01-25.md`
- `docs/CRITICAL_AUDIT_LLM_HALLUCINATION_2026-01-25.md`
- `docs/TRACING_SHAKEDOWN_TASKLIST_2026-01-25.md`
- `docs/SHAKEDOWN_TEST_FINDINGS_2026-01-24.md`
- `docs/REMEDIATION_PLAN_2026-01-23.md`

---

## Architecture Problems (Systemic)
1) **No authoritative API truth source.**
   Doc search results are not attribute-level, so scripts still guess API names.
2) **Validation is permissive.**
   API validator allows unknown attributes (examples continue to reach Blender).
3) **Enforcement was opt-out.**
   Doc-query enforcement now exists, but doc search quality is still weak.
4) **Feedback loop is weak.**
   Execution errors do not force attribute-level corrections or KB updates.

---

## Latest Run Findings (2026-01-26 04:20) - Test v10

**Trace:** `traces/e2e_test_v10_20260126_042056.jsonl`
**Script:** `assets/blender_scripts/generated/test_smoke_e2e_v2.py`
**Model:** gpt-5-mini | **Iterations:** 2

### Phase Results (Iteration 1)

| Phase | Status | Details |
|-------|--------|---------|
| Research Agent | ✅ PASS | 3 tools, doc queries working |
| Technique Selector | ✅ PASS | Selected mantaflow_smoke |
| API Spec Agent | ❌ FAIL | Doc search loop detection triggered |
| Code Writer | ⚠️ FALLBACK | Spec-first failed; fell back to original Script Writer |
| API Validator | ✅ PASS | 11/11 valid |
| Executor | ❌ FAIL | No camera for render |

### Critical Findings

1. **Spec-first still unstable**  
   - API Spec Agent exceeded consecutive doc-search limit (21+ in a row).  
   - Spec-first pipeline aborted and fell back to original Script Writer.

2. **Render pipeline missing camera**  
   - Executor error: `Cannot render, no camera`  
   - Occurred in both iterations (pattern reuse included).

3. **Enum validation not exercised**  
   - Enum guardrail exists, but spec-first did not reach Code Writer.  
   - Needs a successful spec-first run to verify enum enforcement.

---

## Latest Run Findings (2026-01-26 02:45) - Test v9

**Trace:** `traces/e2e_test_v9_20260126_024559.jsonl` (226KB)
**Script:** `assets/blender_scripts/generated/test_v9_specfirst.py`
**Model:** gpt-5-mini | **Iterations:** 1

### Phase Results

| Phase | Status | Duration | Details |
|-------|--------|----------|---------|
| Research Agent | ✅ PASS | 51.7s | 3 tools, doc queries working |
| Technique Selector | ✅ PASS | 26.5s | Selected mantaflow_smoke |
| API Spec Agent | ✅ PASS | 223s | 14 attrs, 3 ops, 17 doc queries |
| Code Writer | ✅ PASS | 138s | 12 attrs, 2 ops verified |
| API Validator | ✅ PASS | <1s | 3/3 API calls valid |
| Executor | ❌ FAIL | 36.7s | Enum value error at runtime |

### Critical Finding: Enum Value Hallucination

**Error:**
```
TypeError: bpy_struct: item.attr = val: enum "FLOW" not found in ('INFLOW', 'OUTFLOW', 'GEOMETRY')
Script failed while setting fset.flow_behavior = 'FLOW'
```

**Root Cause:** API Spec Agent verified `flow_behavior` attribute exists (doc_ref valid), but hallucinated enum value `'FLOW'` instead of correct values `'INFLOW'`, `'OUTFLOW'`, `'GEOMETRY'`.

**Gap in Spec-First Pipeline:**
- ✅ Guardrail validates doc_refs exist
- ✅ Guardrail validates attributes are from verified spec
- ❌ **NO validation of enum VALUES** - LLM can still hallucinate values

### Positive Findings

1. **Loop detection not triggered** - API Spec Agent stayed under 20-call limit
2. **Tracing working** - Full verbose trace captured to file
3. **Parallel tool calls** - 14 doc searches executed in parallel (17.9s batch)
4. **Guardrails all passed** - Both `validate_api_spec` and `validate_code_against_spec`

---

## Previous Run Findings (2026-01-25 21:05)
Trace: `traces/e2e_test_verbose_water_20260125_210557.jsonl`

- **Doc queries executed** before `write_script` (enforcement working).
- **Doc grounding still weak**: API search returns unrelated docs (e.g., Material preview).
- **Runtime failure:** `FluidDomainSettings.bake_frame_start` does not exist → AttributeError.
- **Result:** Both iterations fail; no quality eval or learning recording reached.

---

## Current Blockers (Must Fix)

1) **Enum value validation verification (2026-01-26)**
   - Guardrail now checks enum values via `VALID_ENUM_VALUES` fallback map.
   - Needs a successful run to confirm it blocks invalid values like `'FLOW'`.

2) **Spec-first doc search throttling still failing**
   - API Spec Agent triggers loop detection (21+ consecutive doc calls).
   - Need tighter search plan or fewer per-turn queries.

3) **Render pipeline missing camera**
   - `bpy.ops.render.render` invoked with no active camera.
   - Either create/set camera or skip render for headless runs.

4) **`bake_frame_start` / cache frame API mismatch** (Partially fixed by spec-first)
   - Spec-first pipeline prevents attribute hallucination when API Spec Agent works correctly.
   - Still need fallback Script Writer to have this blocked.

5) **Doc search precision**
   - `search_blender_api_by_intent` returns unrelated APIs for fluid-domain intents.
   - API doc hits are mostly genindex snippets, not attribute-level sources.

6) **API validator too permissive**
   - Mark unknown attributes invalid (strict mode) before execution.

7) **Pattern library propagates outdated API**
   - Pattern application can reintroduce removed attributes.
   - Needs validation on pattern application step.

---

## High Priority (Next)

1) **Enum validation verification** (CRITICAL)
   - Run v10 to confirm enum values are rejected before execution
   - Expand `VALID_ENUM_VALUES` map as needed

2) **Spec-first doc search stabilization**
   - Reduce per-turn doc queries or use bundled search
   - Prevent loop detection while keeping doc coverage

3) **Render pipeline camera fix**
   - Ensure a camera exists before calling render
   - Or disable render in headless cache-only runs

4) **Learning agent baseline + experiment recording**
   - Baseline sync was fixed but not verified in a complete run.
   - Test v9 didn't reach learning phase (execution failed first).

5) **Quality Analyst verification**
   - Not reached in test v9 due to execution failure.
   - Need successful script execution to verify ML evaluation.

---

## Fixed / Verified
- **Doc-query enforcement** for Script Writer (RunHooks).
- **Budget guardrail** uses `get_spent()` + `monthly_limit` (no `get_remaining()`).
- **Vision model default** for evaluation set to `gpt-5-mini`.
- **Test harness crash** (`IterationResult.primary_issue`) resolved.
- **🆕 Spec-first pipeline guardrails** - Both `validate_api_spec` and `validate_code_against_spec` working (2026-01-26).
- **🆕 Loop detection** - `max_consecutive_same_tool=20` for API Spec Agent prevents infinite loops (2026-01-26).
- **🆕 Verbose tracing** - Trace files created correctly when using `test_e2e_orchestrator.py` or enabling `enable_verbose_tracing()` (2026-01-26).
- **🆕 Parallel tool execution** - API Spec Agent correctly batches 14 doc searches in single turn (~18s total) (2026-01-26).
- **🆕 Files NOT being deleted** - Confirmed 115+ directories preserved in `build/vdb_output/`, logs in `build/blender_cli_logs/` (2026-01-26).
- **🆕 Enum value guardrail** - Added enum validation in `validate_code_against_spec` (2026-01-26).

---

## Still Unverified (Needs a successful full run)
- Baseline sync + experiment recording
- Quality Analyst run using `gpt-5-mini`
- Budget guardrail behavior under near-exhaustion

---

## Suggested Next Test (Verify enum validation)

**Test v10 Requirements:**
1. Run 1-iteration test with gpt-5-mini
2. Confirm enum validation blocks invalid values before execution

**Success Criteria:**
- ✅ No enum errors in Blender execution
- ✅ Script executes to completion (creates VDB/render)
- ✅ Quality Analyst evaluates the render
- ✅ Learning Agent records experiment

**Command:**
```bash
VERBOSE_TRACING=1 VERBOSE_TRACE_FILE="traces/e2e_test_v10_$(date +%Y%m%d_%H%M%S).jsonl" \
python test_e2e_orchestrator.py
```

