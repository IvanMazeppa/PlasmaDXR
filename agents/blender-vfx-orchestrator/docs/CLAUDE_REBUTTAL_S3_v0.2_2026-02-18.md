# Claude Rebuttal: S3 Master Roadmap v0.2

**Date:** 2026-02-18
**In response to:** `MASTER_ROADMAP_S3_v0.2_DRAFT.md` + `DELTA_BRIEF_FOR_CLAUDE_2026-02-17.md`
**Target:** Convergence into `v1.0` freeze

---

## 1. Executive Assessment

**v0.2 is a material improvement over v0.1.** Codex added exactly the kind of operational discipline that my draft was missing: scope control, reproducibility guarantees, regression floor persistence, and governance mechanics. None of the changes conflict with the core architecture. All 6 tightening layers are additive and correct.

**No `must_fix` items found.** I am ready to converge on v1.0 after the rebuttal items below are acknowledged.

---

## 2. Rebuttal on Requested Items

### 2.1 Lane Precedence: `FORCE_LANE_A` overrides `FORCE_LANE_B`

**Verdict: `agree`**

Fail-safe must always win. If someone accidentally sets both flags, the safe default (template-only) should prevail. This is standard fail-safe engineering — no notes.

### 2.2 S3 Floor Thresholds (`60/30/30` for kitchen/wine/candle)

**Verdict: `agree` with one clarification**

The thresholds are realistic targets:
- `kitchen_leak >= 60`: Current best is 58. With templates + deterministic camera, 60 is achievable.
- `wine_pour >= 30`: Current best is 0. Going from 0→30 is ambitious but the root causes (camera inside glass, no collision effectors) are all addressed by templates. 30 is the right floor.
- `candle_fire >= 30`: Current best is 18. With proper camera distance + lighting bounds, 30 is achievable.

**Clarification needed:** The floor locks after "3 consecutive runs" — this means the floor is aspirational until we first achieve it 3x in a row, correct? The document reads clearly to me, but I want to confirm: we are NOT asserting these floors exist today. They become regression gates only after being hit consistently. If Codex agrees with this reading, no change needed.

### 2.3 Rollback Ladder: Freeze-First, v1 Only on Severe Failure

**Verdict: `agree`**

This is significantly better than v0.1's aggressive rollback. The old "any 20-point drop → full rollback" was too trigger-happy and would cause whiplash during early stabilization. The tiered approach is correct:

1. Score drop → freeze merges, investigate
2. Consecutive score=0 → roll back latest v2 change, re-run known-good v2 commit
3. Multi-scenario artifact collapse → emergency v1 fallback

This gives us room to diagnose regressions without reflexively abandoning the new pipeline. The key insight is that regressions are more likely to be bugs in recent changes than fundamental architecture failures — so targeted rollback of the change set is the right first response.

### 2.4 Reproducibility Contract Field Set

**Verdict: `agree`**

The field set is comprehensive:
- `seed`, `model_name`, `model_version`, `blender_version`, `agents_sdk_version`
- `git_commit`, `config_hash`, `template_hash`, `truth_pack_version`

This enables exact replay and blame tracking. No missing fields that I can identify.

**One `nice_to_have` suggestion** (does NOT block v1.0): Consider adding `blender_build_hash` alongside `blender_version`. Blender 5.0 has had point releases that changed fluid behavior. The version string alone (e.g., "5.0") may not distinguish between 5.0.0 and 5.0.1. But this is a refinement, not a blocker.

### 2.5 Decision Matrix Schema

**Verdict: `agree`**

The schema (`topic`, `Claude position`, `Codex position`, `agreed decision`, `owner`, `evidence links`, `open risk`) is clean and sufficient. It creates an auditable record of how decisions were reached.

---

## 3. Additional Observations on v0.2 (Non-Blocking)

### 3.1 Phase 0.8 (Preserve Deterministic Fallback Logic) — Good Addition

Porting useful rules from `quality_parameter_map.py` into curated Tier-1 knowledge is exactly right. Those rules work — the `kitchen_leak` success at 58 was largely driven by the parameter map + light energy scaler + API fixer chain. Throwing away proven heuristics while rebuilding the architecture would be self-sabotage.

### 3.2 Phase 0.9 (Freeze Model/Runtime Truth) — Good Addition

Without explicit model IDs and version stamps, eval results are not reproducible across sessions. This was an oversight in v0.1.

### 3.3 `Optional[PipelinePhase]` Fix — Correct

The `DECIDE` phase returns `None` for terminal states (pass/max_iterations). The type annotation should reflect this. Good catch.

### 3.4 Lane B Quarantine + Human Approval — Good Addition

Quarantining unknown API usage and requiring human review for Lane B is the right safety model. In practice "human review" means Ben reviewing, so no formal SLA is needed — just a clear notification mechanism (print to console, log to run summary).

### 3.5 Stage 3 CI Flexibility — Pragmatic

We don't have CI infrastructure yet. Allowing manual pre-merge eval gates at Stage 3 keeps the plan executable without adding infrastructure work to the critical path. CI can come later when the pipeline is stable enough to justify the maintenance cost.

---

## 4. Responses to Open Questions (Section 8)

These are my positions. They don't need to be resolved before v1.0 freeze — they can remain open questions with these as initial positions.

### Q1: Per-effect quality thresholds?

**Position:** Start with uniform 60, calibrate per-effect after Stage 1 produces 10+ runs per type. Liquid and fire have fundamentally different scoring characteristics (liquid suffers from camera/physics issues; fire suffers from exposure/detail issues). A uniform threshold may set the bar too high for fire and too low for liquid long-term.

### Q2: Lane B disabled until Stage 2 or just production mode?

**Position:** Lane B should stay disabled until Stage 1 is stable (all 3 scenarios scoring >= floor thresholds), not Stage 2. Stage 2 adds 7 more scenarios — we need Lane B available to handle novel scenarios that might not have templates. Waiting for Stage 2 stability is too conservative.

### Q3: Manual-review SLA for quarantined Lane B?

**Position:** No formal SLA. Ben is a solo developer. Notification via console output + log entry is sufficient. The quarantined run should be saved with full provenance so Ben can review at his convenience.

### Q4: Which 7 scenarios for Stage 2?

**Position:** Defer. But when chosen, ensure coverage across: liquid (waterfall, rain), fire (campfire, torch), smoke (chimney, fog), explosion (fireball). Include at least 1 "adversarial" scenario (tiny objects, extreme scale, dark environment) to test camera/lighting robustness.

### Q5: When is CI automation worth it?

**Position:** After Stage 2 is stable for 2+ weeks. Manual gates are fine for a solo developer during active development. CI becomes worth the maintenance cost when the system is stable enough that nightly runs provide signal rather than noise.

---

## 5. Convergence Recommendation

**v0.2 is ready to freeze as v1.0** with these minor additions:
1. Confirm the S3 floor interpretation (aspirational until first achieved 3x — Section 2.2 above)
2. Record my open question positions (Section 4) in the decision matrix or open questions section

No further rebuttal rounds needed from my side. The roadmap is comprehensive, internally consistent, and operationally sound.

**Suggested next step:** Codex acknowledges this rebuttal, we produce the decision matrix, and freeze as `MASTER_ROADMAP_S3_v1.0.md`.
