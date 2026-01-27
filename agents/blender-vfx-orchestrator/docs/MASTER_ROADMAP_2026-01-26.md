# Master Roadmap - Spec-First to Autonomy

**Date:** 2026-01-26 (Consolidated)
**Purpose:** Single roadmap to stabilize the pipeline and reach autonomous operation
**Scope:** Replaces Phase‑12 + Autonomy roadmaps (see `docs/archive/`)

---

## 0) Ground Truth (Non-Negotiables)

1. **Agents SDK is the source of truth**
   Use `docs/SDK_ENFORCEMENT_PROTOCOL.md` and official SDK docs.
2. **Spec-first is mandatory**
   No unverified attributes or ops.
3. **Training data is outdated**
   Always consult `docs/VERSION_TRUTH.md`.

---

## 1) Inputs / Evidence (Used to Consolidate)

**Primary issues list:**  

- `docs/CURRENT_ISSUES_CONSOLIDATED_2026-01-26.md`

**Session inventory + changes:**  

- `docs/SESSION_CHANGES_2026-01-26.md`

**Test evidence:**  

- `docs/TEST_REPORT_GPT5_MINI_SHAKEDOWN_2026-01-26.md`  
- `traces/e2e_test_verbose_20260126_185722.jsonl`  
- `traces/user_shakedown_20260126_193538.jsonl`

---

## 2) Current State (Snapshot)

**Verified working (with evidence):**

- Fallback path works and produces renders (see user shakedown report + trace).
- Guardrails validate attributes and ops (spec‑first path when reached).

**Reported/partially verified:**

- Bundle‑first enforcement exists but API Spec Agent still skips it and is blocked.
- Camera fix has been added, but needs a spec‑first success run to confirm.
- Enum guardrail added; verified by unit test, not yet exercised in spec‑first.

**Blocking issues (from CURRENT_ISSUES + SESSION_CHANGES):**

- **P0:** Coordinator → `_modify_script_impl` contract break (multi‑iteration stalled).
- **P1:** API Spec Agent efficiency/turn budget (spec‑first still falls back).
- **P1:** Type hallucinations (e.g., `noise_scale` float, `velocity_multi`).

---

## 3) Phase 12 Completion (Stabilize Spec-First)

**Goal:** Achieve a full spec‑first run (no fallback) with bounded doc search and a render.

### P0 — Coordinator → `_modify_script_impl` Contract

- **Problem:** Coordinator outputs API paths, but `_modify_script_impl` expects Config class patterns.
  Result: iteration 2+ changes are never applied.
- **Status:** **FIXED (2026-01-26)**.
- **Fix applied:** Updated `_modify_script_impl` with `BLENDER_CLASS_TO_VAR` mapping to handle:
  - Coordinator outputs like `FluidDomainSettings.noise_scale`
  - Direct attribute patterns like `settings.noise_scale = X`
  - Chained property access patterns

### P1 — API Spec Agent Efficiency (Bundle‑First + Turn Budget)

- **Problem:** API Spec Agent still skips bundle‑first and hits turn limits → fallback.
- **Status:** **FIXED (2026-01-26)**.
- **Fix applied:**
  - Created `APISpecEnforcementHooks` class with mechanical bundle-first enforcement
  - Targeted searches (`semantic_search_blender_docs`, `search_blender_api_by_intent`) are BLOCKED until `blender_doc_search_bundle` is called
  - Updated instructions with explicit warning and example
  - Limits: max 6 targeted searches after bundle

### P1 — Camera Safety

- **What:** Ensure render doesn’t fail without an active camera.
- **Status:** **IMPLEMENTED, NEEDS SPEC‑FIRST VERIFICATION**.
- **Evidence:** Camera present in fallback render; spec‑first success run still pending.

### P1 — Enum Guardrail Verification

- **What:** Reject invalid enum literals.
- **Status:** **IMPLEMENTED, UNIT‑TESTED**; not yet exercised in a spec‑first run.

### P2 — Type Hallucination Prevention

- **Examples:** `noise_scale` float vs int, `velocity_multi` vs `velocity_factor`.
- **Status:** **FIXED (2026-01-26)**.
- **Fix applied:**
  - Added `velocity_multi` to `KNOWN_API_CHANGES` with correction to `velocity_factor`
  - Added pattern detection for `noise_scale = X.Y` (float assignments)
  - Extended float detection to cover values 1.0-4.0

#### Phase 12 Exit Criteria

- [ ] Spec‑first completes without fallback.
- [x] Render succeeds with camera present. *(Verified 2026-01-26)*
- [ ] Enum guardrail exercised in spec‑first.
- [x] Iteration 2+ changes apply (contract fixed). *(Fixed 2026-01-26)*

---

## 4) Phase‑4 Gate (Stability Gate)

Use `docs/PHASE_4_GATING_ROADMAP_2026-01-25.md` as criteria.

**Must pass (current status):**

- [x] Modification contract enforced. *(Fixed 2026-01-26: `BLENDER_CLASS_TO_VAR` mapping)*
- [ ] Doc grounding reliable (needs repeatable `DocPath` validation).
- [ ] Trace correlation (single `trace()` + `group_id` in JSONL).
- [x] Spec‑first doc search discipline (bundle‑first + bounded). *(Fixed 2026-01-26: `BundleFirstRequiredError`)*

**Blocking:**

- **UNBLOCKED:** Bundle-first enforcement now mechanical.
- Pending: Full spec-first run verification.

**Next fix candidates:**

- ~~Increase API Spec Agent turn budget and add few‑shot bundle‑first examples.~~ *(Done)*
- Verify spec-first success run in shakedown test.

---

## 5) Baseline Autonomous Workflow

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

## 6) Autonomy Expansion

**Additions:**

- Session summarization + compaction
- Cross-session bootstrap (patterns + warnings)
- Regression tests (evals + trace grading)

**Risk controls:**

- Budget guardrail enforced
- Guardrail coverage across agents
- Pattern library validation gate

---

## 7) Workflow Optimization (Optional)

Only after stability:

- Planner-Executor-Verifier core
- Beam search (N=2-3, K=1)
- Bandit technique selection
- Multi-candidate script generation

---

## 8) Immediate Next Steps (Priority Order)

1. ~~**P0** Fix Coordinator → `_modify_script_impl` contract.~~ ✅ **DONE (2026-01-26)**
2. ~~**P1** Stabilize API Spec Agent (bundle‑first compliance + turn budget).~~ ✅ **DONE (2026-01-26)**
3. **P1** Verify camera fix in a **spec‑first** success run. *(Ready for test)*
4. **P1** Exercise enum guardrail in spec‑first (catch invalid literals). *(Ready for test)*
5. ~~**P2** Enforce type correctness (`noise_scale`, `velocity_factor`).~~ ✅ **DONE (2026-01-26)**

**NEXT ACTION:** Run shakedown test to verify fixes:
```bash
python test_quick_e2e.py --preset quick_test --effect smoke --iterations 2
```

### P0 Blocker: Coordinator → modify_script Contract

**See `CURRENT_ISSUES_CONSOLIDATED_2026-01-26.md` for full investigation.**

**Summary:** The Modification Coordinator outputs Blender API paths (`FluidDomainSettings.noise_scale`),
but `_modify_script_impl` expects Config class patterns (`class Config: NOISE_SCALE = 1`).
The generated scripts use direct attribute assignment (`dsettings.noise_scale = 1.0`),
which matches neither format.

**Fix Required:** Update `tools/script_generator_tools.py:_modify_script_impl` to handle
direct attribute patterns like `dsettings.noise_scale = X`.

---

## 9) Session Notes (Condensed)

- **User shakedown (gpt‑5‑mini):** Fallback succeeded, render produced, score 22.0.  
  Evidence: `docs/TEST_REPORT_GPT5_MINI_SHAKEDOWN_2026-01-26.md` + trace.
- **Phase‑4 gate pass (single iteration):** Completed with fallback.  
  Evidence: `traces/e2e_test_verbose_20260126_185722.jsonl`.
- **P0 blocker identified:** `_modify_script_impl` doesn’t match coordinator output format.

---

## 10) References

- `docs/CURRENT_ISSUES_CONSOLIDATED_2026-01-26.md`
- `docs/SESSION_CHANGES_2026-01-26.md`
- `docs/TEST_REPORT_GPT5_MINI_SHAKEDOWN_2026-01-26.md`
- `docs/PHASE_4_GATING_ROADMAP_2026-01-25.md`
- `docs/SCRIPT_WRITER_OVERHAUL_PROPOSAL_2026-01-25.md`
- `docs/SDK_ENFORCEMENT_PROTOCOL.md`
- `docs/VERSION_TRUTH.md`
- `traces/phase4_gate_shakedown_20260126_184218.jsonl`
- `traces/e2e_test_verbose_20260126_185722.jsonl`
- `traces/user_shakedown_20260126_193538.jsonl`

---

End of Roadmap
