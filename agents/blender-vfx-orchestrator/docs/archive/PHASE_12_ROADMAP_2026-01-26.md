# Phase 12 Completion Roadmap

**Date:** 2026-01-26
**Status:** IN PROGRESS
**Goal:** Achieve first successful end-to-end VFX asset generation with spec-first pipeline
**Dependencies:** SDK_ENFORCEMENT_PROTOCOL.md, VERSION_TRUTH.md, LLM_PRIMER_2026-01-26.md

---

## Executive Summary

Phase 12 (Spec-First Pipeline) is **partially complete**. The architecture is sound, but two independent blockers prevent a successful full run:

1. **Spec-first search instability** - Loop detection triggers, causing fallback to old Script Writer
2. **Missing render camera** - Executor fails even when script succeeds

This document provides a prioritized roadmap to complete Phase 12 and achieve the first successful end-to-end run.

---

## Current State Assessment

### Test Results Summary

| Test | Date | Spec-First | Code Writer | Executor | Failure Mode |
|------|------|------------|-------------|----------|--------------|
| v7 | 01-26 00:39 | ✅ PASS | ✅ PASS | ⚠️ "ERROR: None" | Unknown executor issue |
| v9 | 01-26 02:45 | ✅ PASS | ✅ PASS | ❌ FAIL | Enum hallucination (`'FLOW'`) |
| v10 | 01-26 04:20 | ❌ LOOP DETECT | ⚠️ FALLBACK | ❌ FAIL | No camera for render |

### Component Status

| Component | Status | Notes |
|-----------|--------|-------|
| API Spec Agent | ⚠️ UNSTABLE | Works sometimes (v9), hits loop detection other times (v10) |
| Code Writer Agent | ✅ WORKING | Passes guardrail when spec-first succeeds |
| Enum Validation | ✅ IMPLEMENTED | Added to guardrail, needs verification |
| API Validator | ✅ WORKING | 11/11 valid in v10 |
| Executor | ❌ BLOCKED | Missing camera prevents all renders |
| Quality Analyst | ⏸️ NOT REACHED | Blocked by Executor |
| Learning Agent | ⏸️ NOT REACHED | Blocked by Quality Analyst |

---

## Blockers Analysis

### Blocker 1: Missing Render Camera (P0 - CRITICAL)

**Impact:** Blocks ALL downstream phases (Quality Analyst, Learning Agent)
**Effort:** LOW (~30 minutes)
**First seen:** Test v10

**Error:**
```
RuntimeError: Cannot render, no camera
```

**Root Cause:** Generated scripts set up fluid simulation but don't create a camera. Blender requires an active camera for `bpy.ops.render.render()`.

**Fix Strategy:**
- Option A: Add camera creation to all generated scripts (template change)
- Option B: Add camera check/creation in Executor before render
- Option C: Add camera setup as post-script hook

**Recommended:** Option B (Executor-level fix) - Single point of change, doesn't rely on LLM.

**Implementation:**
```python
# In executor.py, before render call:
def ensure_camera_exists():
    """Ensure scene has an active camera for rendering."""
    scene = bpy.context.scene
    if not scene.camera:
        # Create camera
        cam_data = bpy.data.cameras.new("RenderCamera")
        cam_obj = bpy.data.objects.new("RenderCamera", cam_data)
        bpy.context.collection.objects.link(cam_obj)
        scene.camera = cam_obj

        # Position camera to view origin
        cam_obj.location = (7.0, -7.0, 5.0)
        cam_obj.rotation_euler = (1.1, 0, 0.8)

        # Point at origin (where fluid domain typically is)
        constraint = cam_obj.constraints.new('TRACK_TO')
        constraint.target = None  # Track to world origin
        constraint.track_axis = 'TRACK_NEGATIVE_Z'
        constraint.up_axis = 'UP_Y'
```

---

### Blocker 2: Spec-First Search Instability (P1 - CRITICAL)

**Impact:** Causes fallback to old Script Writer, bypassing spec-first benefits
**Effort:** MEDIUM (~2 hours)
**First seen:** Test v10

**Behavior:**
- v9: API Spec Agent used 17 doc queries, stayed under limit, PASSED
- v10: API Spec Agent exceeded 20 consecutive doc queries, triggered loop detection

**Root Cause:** API Spec Agent search behavior is non-deterministic. Sometimes batches efficiently, sometimes spams sequentially.

**Current Limits (from SDK_ENFORCEMENT_PROTOCOL.md):**
```python
max_consecutive_same_tool = 20
max_exempt_tool_calls = 30
max_turns = 6
hard_turn_limit = 8
```

**Fix Strategies:**

| Strategy | Pros | Cons | Recommended |
|----------|------|------|-------------|
| A. Increase limit to 30 | Quick fix | May allow spam | No |
| B. Bundled search first | Forces efficiency | Needs instruction change | **Yes** |
| C. Turn 4 hard output | Guarantees output | May truncate research | Partial |
| D. Reduce attributes per spec | Smaller scope | May miss needed attrs | No |

**Recommended:** Strategy B + C combined

**Implementation:**
1. Update API Spec Agent instructions to require `blender_doc_search_bundle` as first tool call
2. Add hard limit: "Turn 4 MUST output APISpec, no more searches allowed"
3. Keep `max_consecutive_same_tool = 20` as safety net

**Instruction Update:**
```markdown
## Search Strategy (MANDATORY)

Turn 1: Call `blender_doc_search_bundle` with effect_type to get core APIs
Turn 2-3: Use `semantic_search_blender_docs` for specific attributes not covered
Turn 4: OUTPUT APISpec - NO MORE SEARCHES ALLOWED

If you reach Turn 4 without outputting, you MUST output immediately with whatever
attributes you have verified. Incomplete specs are better than no specs.
```

---

### Blocker 3: Enum Validation Verification (P2 - HIGH)

**Impact:** Prevents enum hallucination like `'FLOW'`
**Effort:** LOW (verification only)
**First seen:** Test v9

**Status:** IMPLEMENTED but NOT VERIFIED

**Implementation (already done):**
```python
# In guardrails/api_spec_guardrails.py
VALID_ENUM_VALUES = {
    "flow_behavior": ["INFLOW", "OUTFLOW", "GEOMETRY"],
    "flow_type": ["SMOKE", "FIRE", "BOTH"],
    "domain_type": ["GAS", "LIQUID"],
    "cache_type": ["MODULAR", "ALL", "FINAL"],
    # ...
}
```

**Verification Plan:**
Once Blockers 1 & 2 are fixed, run test v11. If spec-first succeeds and script contains invalid enum, guardrail should:
1. Detect the invalid value
2. Reject the script
3. Force regeneration or fail with clear error

**Success Criteria:**
- Invalid enum values are caught BEFORE Blender execution
- Error message clearly identifies the invalid value and valid alternatives

---

## Deferred Issues (Post-Phase 12)

These issues are real but should not block Phase 12 completion:

| Issue | Priority | Reason to Defer |
|-------|----------|-----------------|
| Doc search precision | P3 | Spec-first mitigates noisy results |
| API validator strict mode | P3 | Spec-first handles attribute validation |
| Pattern library validation | P3 | Lower priority until pipeline stable |
| Budget guardrail edge cases | P4 | Not blocking, can test later |

---

## Implementation Roadmap

### Phase 12.1: Camera Fix (P0)

**Objective:** Ensure all renders have a camera
**Owner:** TBD
**Effort:** 30 minutes

**Tasks:**
1. [ ] Add `ensure_camera_exists()` function to executor.py
2. [ ] Call function before any render operation
3. [ ] Test with simple script that has no camera
4. [ ] Verify render completes successfully

**Files to Modify:**
- `specialized_agents/executor.py`

**Verification:**
```bash
# Run simple test that previously failed with "no camera"
python -c "
import bpy
bpy.ops.mesh.primitive_cube_add()
# No camera added
bpy.ops.render.render(write_still=True)
"
# Should now succeed after fix
```

---

### Phase 12.2: Spec-First Stabilization (P1)

**Objective:** API Spec Agent reliably outputs without hitting loop detection
**Owner:** TBD
**Effort:** 2 hours

**Tasks:**
1. [ ] Update API Spec Agent instructions with mandatory search strategy
2. [ ] Add Turn 4 hard output requirement
3. [ ] Test with 3 consecutive runs to verify stability
4. [ ] Document any edge cases

**Files to Modify:**
- `specialized_agents/api_spec_agent.py` (instructions)
- `hooks/enforcement_hooks.py` (optional: add Turn 4 enforcement)

**Verification:**
```bash
# Run 3 consecutive tests
for i in 1 2 3; do
  VERBOSE_TRACING=1 python test_e2e_orchestrator.py 2>&1 | grep -E "(API Spec Agent|PASS|FAIL|LOOP)"
done
# All 3 should show API Spec Agent PASS
```

---

### Phase 12.3: Enum Validation Verification (P2)

**Objective:** Confirm enum guardrail catches invalid values
**Owner:** TBD
**Effort:** 30 minutes (verification only)

**Tasks:**
1. [ ] Run test v11 after 12.1 and 12.2 complete
2. [ ] If enum error occurs, verify it's caught by guardrail (not Blender)
3. [ ] If no enum error, manually test guardrail with invalid value
4. [ ] Document results

**Files to Modify:**
- None (verification only)
- May need to expand `VALID_ENUM_VALUES` if new enums encountered

**Verification:**
```python
# Manual guardrail test
from guardrails.api_spec_guardrails import validate_code_against_spec

# Test with invalid enum
code = "fset.flow_behavior = 'FLOW'"  # Invalid
result = validate_code_against_spec(code, api_spec)
assert result.tripwire_triggered == True
assert "FLOW" in result.output_info["errors"][0]
```

---

### Phase 12.4: Full Pipeline Verification (P2)

**Objective:** Achieve first successful end-to-end run
**Owner:** TBD
**Effort:** 1 hour (testing)

**Tasks:**
1. [ ] Run full test with 1 iteration
2. [ ] Verify all phases complete:
   - [ ] Research Agent
   - [ ] Technique Selector
   - [ ] API Spec Agent (spec-first, not fallback)
   - [ ] Code Writer
   - [ ] API Validator
   - [ ] Executor (script runs, render completes)
   - [ ] Quality Analyst (evaluates render)
   - [ ] Learning Agent (records experiment)
3. [ ] Document trace file and results
4. [ ] Update CURRENT_ISSUES_CONSOLIDATED with success

**Success Criteria:**
```
Pipeline Status: ✅ COMPLETE
Quality Score: >= 35.0 (or any non-zero score)
Phases Reached: ALL (including Learning Agent)
Fallbacks Used: NONE
```

---

## Timeline

| Phase | Task | Estimated Duration | Dependencies |
|-------|------|-------------------|--------------|
| 12.1 | Camera Fix | 30 min | None |
| 12.2 | Spec-First Stabilization | 2 hours | None |
| 12.3 | Enum Verification | 30 min | 12.1, 12.2 |
| 12.4 | Full Pipeline Verification | 1 hour | 12.1, 12.2, 12.3 |
| **Total** | | **4 hours** | |

---

## Success Metrics

### Phase 12 Complete When:

1. **Camera Fix Verified**
   - [ ] Render succeeds without pre-existing camera

2. **Spec-First Stable**
   - [ ] 3 consecutive runs without loop detection
   - [ ] API Spec Agent outputs within Turn 4

3. **Enum Validation Verified**
   - [ ] Invalid enum values caught by guardrail
   - [ ] Clear error message with valid alternatives

4. **Full Pipeline Success**
   - [ ] All phases complete without fallback
   - [ ] Quality Analyst evaluates render
   - [ ] Learning Agent records experiment
   - [ ] Non-zero quality score achieved

---

## Document Compatibility Matrix

| Document | Relationship | Update Needed |
|----------|--------------|---------------|
| `VERSION_TRUTH.md` | Source of truth for APIs/enums | ✅ Updated (enum values) |
| `SDK_ENFORCEMENT_PROTOCOL.md` | Hook limits and patterns | May need Turn 4 enforcement |
| `LLM_PRIMER_2026-01-26.md` | Onboarding context | Update after Phase 12 complete |
| `CURRENT_ISSUES_CONSOLIDATED_2026-01-26.md` | Issue tracker | Update as blockers resolved |
| `ARCHITECTURE_OPTIMIZATION_PLAN_2026-01-22.md` | Phase tracking | Update Phase 12 status |
| `SCRIPT_WRITER_OVERHAUL_PROPOSAL_2026-01-25.md` | Spec-first design | Update with stability fixes |

---

## References

- Test v9 trace: `traces/e2e_test_v9_20260126_024559.jsonl`
- Test v10 trace: `traces/e2e_test_v10_20260126_042056.jsonl`
- Enum validation: `guardrails/api_spec_guardrails.py:VALID_ENUM_VALUES`
- Hook limits: `hooks/enforcement_hooks.py:create_api_spec_hooks()`

---

*This roadmap should be updated as each phase completes.*
