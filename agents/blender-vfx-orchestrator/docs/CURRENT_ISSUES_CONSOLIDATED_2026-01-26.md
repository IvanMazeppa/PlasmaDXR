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

1) **🆕 Enum value hallucination (2026-01-26)**
   - API Spec Agent verifies attribute doc_refs exist but doesn't extract/verify enum values.
   - Example: `flow_behavior = 'FLOW'` hallucinated despite correct docs showing `('INFLOW', 'OUTFLOW', 'GEOMETRY')`.
   - **Fix options:**
     a) Enhance API Spec Agent instructions to extract enum values from doc results
     b) Add enum validation guardrail that checks values against known enums
     c) Add `VALID_ENUM_VALUES` constant map like `KNOWN_API_CHANGES`

2) **`bake_frame_start` / cache frame API mismatch** (Partially fixed by spec-first)
   - Spec-first pipeline prevents attribute hallucination when API Spec Agent works correctly.
   - Still need fallback Script Writer to have this blocked.

3) **Doc search precision**
   - `search_blender_api_by_intent` returns unrelated APIs for fluid-domain intents.
   - API doc hits are mostly genindex snippets, not attribute-level sources.

4) **API validator too permissive**
   - Mark unknown attributes invalid (strict mode) before execution.

5) **Pattern library propagates outdated API**
   - Pattern application can reintroduce removed attributes.
   - Needs validation on pattern application step.

---

## High Priority (Next)

1) **🆕 Enum value validation** (CRITICAL)
   - Add `VALID_ENUM_VALUES` map for common Blender enums
   - Validate API Spec Agent output includes only valid enum values
   - Example entry: `flow_behavior: ['INFLOW', 'OUTFLOW', 'GEOMETRY']`

2) **Learning agent baseline + experiment recording**
   - Baseline sync was fixed but not verified in a complete run.
   - Test v9 didn't reach learning phase (execution failed first).

3) **Quality Analyst verification**
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

---

## Still Unverified (Needs a successful full run)
- Baseline sync + experiment recording
- Quality Analyst run using `gpt-5-mini`
- Budget guardrail behavior under near-exhaustion

---

## Suggested Next Test (Once enum validation added)

**Test v10 Requirements:**
1. Add `VALID_ENUM_VALUES` validation to API Spec Agent or guardrail
2. Run 1-iteration test with gpt-5-mini

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

