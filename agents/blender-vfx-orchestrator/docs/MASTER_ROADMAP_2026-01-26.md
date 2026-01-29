# Master Roadmap - Autonomy Stabilization (Single Source of Truth)

**Date:** 2026-01-27
**Purpose:** One authoritative plan for stabilizing the Blender VFX Orchestrator
**Scope:** Supersedes scattered issue lists and gating docs in this folder

---

## 0) Ground Truth (Non-Negotiables)

1. **Agents SDK is the source of truth.**
   Follow `docs/SDK_ENFORCEMENT_PROTOCOL.md` and official SDK docs.
2. **Spec-first is mandatory.**
   No unverified attributes or ops.
3. **Training data is outdated.**
   Always consult `docs/VERSION_TRUTH.md`.

---

## 1) Executive Assessment (Condensed)

- The system is **structurally sound** (pipeline, RunHooks, guardrails, tools exist).
- Autonomy is blocked by **deterministic enforcement and verification gaps**, not missing components.
- The **primary architectural failure** is reliance on LLM compliance in steps that must be mechanical.

---

## 2) Current Reality (Truth Snapshot)

**Verified (repeatable evidence in recent traces):**
- Fallback path can complete and render.
- Bundle-first enforcement blocks targeted searches until bundle call.
- Doc-query enforcement blocks Script Writer when doc query missing.

**Claimed but NOT yet consistently verified:**
- Spec-first completes end-to-end without fallback.
- Multi-iteration modification reliably applies changes.
- Enum guardrail exercised in a spec-first run.

**Conflicting claims (must re-verify):**
- Phase-4 gate marked “passed” in some docs, but gating criteria remain unverified elsewhere.

---

## 3) Phase 4 Gate (Must Pass BEFORE Phase 4)

If any item is red, **Phase 4 must not start**.

1) **Modification contract enforced**
   - Coordinator output applies to script changes in iteration 2+.
   - **Status:** NEEDS VERIFICATION (conflicting doc claims).

2) **Doc grounding reliable**
   - `doc_refs` resolve to real `DocPath`, not temp filenames.
   - **Status:** NEEDS VERIFICATION.

3) **Trace correlation working**
   - Single outer `trace()` per run; `group_id=session_id` in local JSONL.
   - **Status:** NEEDS VERIFICATION.

4) **Spec-first doc search discipline**
   - Bundle-first called early; bounded targeted searches; APISpec emitted on time.
   - **Status:** PARTIAL (enforcement exists, model still attempts to skip bundle).

---

## 4) Unified Issue Register (All Active Jobs)

### P0 — Pipeline Integrity (Blockers)
1. **Spec-first completion (no fallback)**
   - Ensure API Spec Agent reliably calls bundle-first and outputs APISpec without turn-limit fallback.
   - Evidence gap: spec-first success still unverified.

2. **Modification contract stability (iteration 2+)**
   - Validate that coordinator-driven changes are applied to generated scripts.
   - If still broken: expand `_modify_script_impl` matching for direct attribute assignments.

3. **Doc grounding fidelity**
   - Ensure vector store returns real `DocPath` and not temp filenames.
   - Add explicit DocPath validation in guardrails and tests.

4. **Trace correlation discipline**
   - Enforce single outer `trace()` with `group_id=session_id` and propagate to JSONL.

5. **API sequence safety (Fluid domain order)**
   - Confirm `fluid_type='DOMAIN'` set before `domain_settings` access.
   - Guard in validator and/or code writer instructions.

6. **Deterministic enforcement vs compliance**
   - Any step relying on “please do X” must be converted to a hard gate or tool-order enforcement.

7. **Disable handoff pipeline**
   - Ensure deprecated `create_asset()` and any handoff-based orchestration cannot run.

8. **Dynamic instructions active for all agents**
   - Verify dynamic instruction functions are enabled and failures are logged (no silent fallback).

9. **Normalize output schema usage**
   - Choose a single field set (e.g., `parameters_set`, `execution_time_seconds`) and enforce via schema tests.

10. **Route direct modify paths through enforcement**
   - Prevent bypass of RunHooks/guardrails by ensuring modifications go through enforced tool paths.

11. **Remove handoff prompt prefix on standalone agents**
   - Avoid instruction noise from handoff-oriented prompt wrappers.

### P1 — Spec Accuracy and Safety
12. **Enum guardrail verification in spec-first run**
   - Exercise invalid enum cases through the spec-first path, not only unit tests.

13. **Type correctness enforcement**
   - Validate integer-only values (e.g., `noise_scale`) and corrected attributes (e.g., `velocity_factor`).

14. **Pattern library safety**
   - Prevent patterns from reintroducing outdated APIs; require doc_refs for pattern application.

15. **Doc search precision**
   - Reduce unrelated results; improve bundle quality and ranking.

16. **Strict API validation (unknown == invalid)**
    - Flip validator default to invalid for unknown attributes and require explicit verification.

17. **API index / whitelist from Blender 5 docs**
    - Extract valid attributes from API docs for fast, deterministic validation.

18. **Attribute verification tool**
    - Tool or index lookup required before using any new attribute name.

19. **Doc-coverage guardrail**
    - Every attribute used in a script must map to a DocPath from the API store.

20. **Integrate full API Validator agent when needed**
    - Invoke heavy validator on iteration > 1 or when lightweight validation fails.

### P2 — Learning Loop Reliability
21. **Mandatory pre-iteration research**
   - Ensure it runs for iteration > 1 and directly influences decisions.

22. **Experiment and pattern recording**
   - Verify baseline recording, experiment outcome logging, and pattern outcome reporting.

23. **Session persistence**
   - Complete SQLiteSession integration so cross-agent context is stable across iterations.

### P3 — Observability and Ops
24. **RunHooks coverage across all agents**
   - Guardrails and hooks must cover every agent tool call.

25. **Documentation integrity**
   - Keep SDK version consistent across docs; avoid conflicting status claims.

26. **Structured research output**
    - Replace regex parsing of free text with Pydantic outputs (approach, params, alternatives).

27. **Instruction unification**
    - Align prompts and rules to a single canonical script-writing mode.

28. **Session compaction strategy**
    - Summarize every N iterations to control session growth (Phase 4 prerequisite).

### P1 — Autonomy Architecture (Addendum 2026-01-29)
29. **State-machine driven orchestration**
    - Formalize state transitions: PLAN → GENERATE → VALIDATE → EXECUTE → EVALUATE → DECIDE.
    - Ensure failure paths route to DIAGNOSE/FIX instead of ad-hoc retries.

30. **Artifact-first handoffs** ✅ IMPLEMENTED (2026-01-29)
    - Agents must write scripts/logs/evals to disk and pass file refs, not inline dumps.
    - Enforce this in Coordinator prompts and guardrails.

31. **Run manifest + scorecard**
    - Emit a per-run manifest (inputs, outputs, decisions, env).
    - Emit a quality scorecard (pass/fail, cache size, render count, critical issues).

32. **Bake/render gate thresholds** ✅ IMPLEMENTED (2026-01-29)
    - Reject runs with empty/too-small caches or missing renders.
    - Gate before marking PASS or proceeding to iteration > 1.
    - Implementation: `guardrails/artifact_gates.py` with Phase 2.7 integration in orchestrator.

33. **Critic loop for acceptance**
    - Add a deterministic critic step: only accept if gates pass and quality >= threshold.

34. **Learning registry (append-only)**
    - Persist run outcomes and fixes; promote reliable fixes into the API fixer.

---

## 5) Execution Plan (Next Actions)

1) **Run a 2-iteration spec-first shakedown**
   - Confirm no fallback, no doc gaps, and modifications apply.
2) **Verify Phase-4 gate items**
   - Record results and update gate status in this file.
3) **Fix any red gate item immediately**
   - Only after all gate items are green should Phase 4 begin.

---

## 6) Verification Checklist (Required Evidence)

- Spec-first completes without fallback.
- `doc_refs` contain valid `DocPath` entries.
- Coordinator modifications apply to script (non-empty `changes_made`).
- Trace correlation shows one `trace()` with correct `group_id`.
- API sequence safety verified in runtime logs.

---

## 8) Action Checklist (Working List)

- [ ] Run a 2-iteration spec-first shakedown and confirm no fallback.
- [ ] Verify `doc_refs` resolve to real `DocPath` entries.
- [ ] Confirm iteration 2+ modifications change the script (`changes_made` non-empty).
- [ ] Verify single outer `trace()` and `group_id=session_id` in JSONL.
- [ ] Validate fluid domain setup order before accessing `domain_settings`.
- [ ] Exercise enum guardrail in a spec-first run (not only unit tests).
- [ ] Enforce strict unknown-attribute validation (unknown == invalid).
- [ ] Require doc_refs for any pattern application to prevent legacy API drift.
- [ ] Complete SQLiteSession integration and confirm cross-agent context.
- [ ] Standardize output schema fields and add a schema drift test.
- [ ] Implement state-machine orchestration with explicit failure routing.
- [ ] Enforce artifact-first handoffs (file refs only).
- [ ] Add per-run manifest + quality scorecard outputs.
- [x] Gate on cache size + render count before PASS. ✅ DONE (2026-01-29)
- [ ] Add critic loop before acceptance.
- [ ] Persist learning registry for outcomes/fixes.

---

## 7) Document Authority

This file is the **only roadmap**. All other issue lists and gating docs are
superseded and should not be used for planning.
