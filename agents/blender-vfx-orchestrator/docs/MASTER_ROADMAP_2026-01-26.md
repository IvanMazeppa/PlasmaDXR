# Master Roadmap - Spec-First to Autonomy

**Date:** 2026-01-26 (Updated)
**Purpose:** Single roadmap to stabilize the pipeline and reach autonomous operation
**Scope:** Replaces Phase-12 + Autonomy roadmaps (see `docs/archive/`)

---

## 0) Ground Truth (Non-Negotiables)

1. **Agents SDK is the source of truth**
   Use `docs/SDK_ENFORCEMENT_PROTOCOL.md` and official SDK docs.
2. **Spec-first is mandatory**
   No unverified attributes or ops.
3. **Training data is outdated**
   Always consult `docs/VERSION_TRUTH.md`.

---

## 1) Current State (Snapshot - 2026-01-26 19:05 UTC)

**Source of truth:** `docs/CURRENT_ISSUES_CONSOLIDATED_2026-01-26.md`

**VERIFIED WORKING (Phase-4 Gate PASSED):**
- Full pipeline completed successfully (452 seconds)
- Quality score 40.0 met threshold at iteration 1
- Bundle-first enforcement WORKING (blocks unauthorized targeted searches)
- Camera injection fix verified
- Enum guardrail verified (rejects invalid values like `flow_behavior='FLOW'`)
- Loop detection prevents infinite doc searches

**FALLBACK PATH USED:**
- API Spec Agent tried to skip bundle → was blocked → fell back to Script Writer
- Script Writer completed successfully with doc queries
- This is acceptable for Phase-4 gate

**Trace Evidence:**
- `traces/e2e_test_verbose_20260126_185722.jsonl` (successful run)
- `traces/phase4_gate_shakedown_20260126_184218.jsonl` (earlier partial run)

---

## 2) Phase 12 Completion (Stabilize Spec-First)

**Goal:** Achieve first successful end-to-end run with spec-first (no fallback).

### P0 — Camera Safety ✅ VERIFIED
- **What:** Inject camera if render is called without one.
- **Where:** `tools/blender_api_fixer.py` (camera fix), `tools/dynamic_instructions.py` (explicit camera setup).
- **Status:** Unit tested, injects camera correctly.

### P1 — Spec-First Search Stabilization ✅ VERIFIED (Partial)
- **What:** Bundle-first search plan enforced via RunHooks.
- **Where:** `hooks/enforcement_hooks.py` (`require_bundle_first=True`).
- **Status:**
  - Bundle-first enforcement WORKING (targeted searches blocked until bundle called)
  - Doc searches reduced from 80+ to 8 (90% reduction)
  - **ISSUE:** API Spec Agent still hits max_turns=8 before output

### P2 — Enum Guardrail ✅ VERIFIED
- **What:** Guardrail rejects invalid enum values.
- **Where:** `guardrails/api_spec_guardrails.py` (`VALID_ENUM_VALUES` fallback + enforcement).
- **Status:** Unit tested, rejects invalid enums like `flow_behavior='FLOW'`.

**Phase 12 Exit Criteria**
- [x] Bundle-first enforcement working
- [x] Camera fix verified
- [x] Enum guardrail verified
- [x] Full pipeline completes with passing quality (via fallback path)
- [ ] Spec-first completes without fallback (API Spec Agent needs model tuning)

---

## 3) Phase-4 Gate (Stability Gate)

Use `docs/PHASE_4_GATING_ROADMAP_2026-01-25.md` as criteria.

**Must pass:**
- [x] Modification contract enforced (flat config keys only).
- [x] Doc grounding reliable (real DocPath, not temp files).
- [x] Trace correlation (single trace + group_id).
- [x] Spec-first doc search discipline (bounded - 8 searches vs 80+).

**Status: PASSED (via fallback)**
- Pipeline completes with quality score meeting threshold
- Bundle-first enforcement working (blocks unauthorized searches)
- Fallback to Script Writer is acceptable for gate criteria

**Future Optimization:**
- Train API Spec Agent to call bundle first (currently ignored)
- This is a model behavior issue, not an enforcement issue

---

## 4) Baseline Autonomous Workflow

**Goal:** Unattended runs for small tasks with consistent outputs.

**Default loop:**
1. Research → Technique Selection
2. Spec-first script generation
3. Execute + render
4. Evaluate (Quality Analyst)
5. Learn + decide next action

**Metrics:**
- Script success rate
- Render success rate
- Spec-first usage rate (no fallback)
- Doc queries per spec (target: <10)
- Quality score delta per iteration

---

## 5) Autonomy Expansion

**Additions:**
- Session summarization + compaction
- Cross-session bootstrap (patterns + warnings)
- Regression tests (evals + trace grading)

**Risk controls:**
- Budget guardrail enforced
- Guardrail coverage across agents
- Pattern library validation gate

---

## 6) Workflow Optimization (Optional)

Only after stability:
- Planner-Executor-Verifier core
- Beam search (N=2-3, K=1)
- Bandit technique selection
- Multi-candidate script generation

---

## 7) Immediate Next Steps (Priority Order)

1. ✅ Verify API Spec Agent bundle-first enforcement (DONE - hooks enforce)
2. ✅ Verify camera fix (DONE - unit tested)
3. ✅ Verify enum guardrail (DONE - unit tested)
4. ✅ Increase API Spec Agent turn budget (max_turns: 10, hard_limit: 12)
5. ✅ Run E2E shakedown - PASSED (Score 40.0, fallback to Script Writer)

**Next Phase:**
6. Investigate API Spec Agent ignoring bundle-first instructions
7. Consider model prompt tuning or few-shot examples

---

## 8) Implementation Changes (This Session)

### hooks/enforcement_hooks.py
- Added `require_bundle_first` config option
- Added `bundle_tool` and `targeted_search_tools` config
- Added `_bundle_called` state tracking
- Updated `on_tool_start` to enforce bundle-first
- Updated `create_api_spec_hooks()` with bundle-first enforcement
- Increased limits: `max_consecutive_same_tool=30`, `max_exempt_tool_calls=40`

### Verification Tests Run
1. `hooks.enforcement_hooks` unit test - PASS
2. `blender_api_fixer.py` camera injection test - PASS
3. `api_spec_guardrails.py` enum validation test - PASS
4. E2E shakedown test - **PASS** (Score 40.0, pipeline completed in 452s)

### E2E Test Summary (2026-01-26 19:04)
- **Result:** PASSED (quality score 40.0 met threshold)
- **Trace:** `traces/e2e_test_verbose_20260126_185722.jsonl`
- **Bundle-first enforcement:** WORKING (API Spec Agent was blocked when skipping bundle)
- **Fallback:** Triggered successfully (Script Writer completed the job)
- **Render:** Successful (test_smoke_e2e.png, 98.3s)
- **Quality Analysis:** 7 issues identified, score meets threshold

---

## 9) References

- `docs/CURRENT_ISSUES_CONSOLIDATED_2026-01-26.md`
- `docs/PHASE_4_GATING_ROADMAP_2026-01-25.md`
- `docs/SCRIPT_WRITER_OVERHAUL_PROPOSAL_2026-01-25.md`
- `docs/SDK_ENFORCEMENT_PROTOCOL.md`
- `docs/VERSION_TRUTH.md`
- `traces/phase4_gate_shakedown_20260126_184218.jsonl`

---

*End of Roadmap*
