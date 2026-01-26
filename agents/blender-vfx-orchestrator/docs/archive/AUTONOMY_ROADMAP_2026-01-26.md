# Autonomy Roadmap - Blender VFX Orchestrator

**Date:** 2026-01-26  
**Purpose:** Roadmap from “stable pipeline” → “autonomous workflow”

---

## 1) Non‑Negotiables

1. **Agents SDK is ground truth.**  
   Follow `docs/SDK_ENFORCEMENT_PROTOCOL.md`.
2. **Spec‑first pipeline is mandatory.**  
   No unverified attributes or ops.
3. **Training data is outdated.**  
   Use `docs/VERSION_TRUTH.md`.

---

## 2) Current State (Summary)

**What’s working:**
- Spec‑first guardrails validate attribute usage.
- Loop detection prevents infinite doc searches.
- Enum guardrail exists (needs verification).

**What’s still blocking:**
- API Spec Agent still trips loop detection (doc search volume).
- Render fails when no camera exists.
- Doc search precision is noisy (genindex/manual hits).

**Source of truth:**  
`docs/CURRENT_ISSUES_CONSOLIDATED_2026-01-26.md`

---

## 3) Phase 0 — Stabilize (Immediate)

**Goal:** Run a full spec‑first iteration without falling back.

**Must pass:**
- API Spec Agent completes without loop detection.
- Code Writer runs using spec‑first (no fallback).
- Render completes (camera present).
- Executor succeeds without AttributeError / enum errors.

**Actions:**
- Bundle‑first doc search plan for API Spec Agent.
- Camera safety fix (API fixer + instructions).
- Verify enum guardrail in spec‑first path.

---

## 4) Phase 1 — Phase‑4 Gate (Stability Gate)

Use `docs/PHASE_4_GATING_ROADMAP_2026-01-25.md`.

**Gate requirements:**
- Modification contract enforced (flat config keys).
- Doc grounding reliable (real DocPath, not temp files).
- Trace correlation (single trace + group_id).
- Spec‑first doc search discipline (bounded + Turn 4 output).

**Pass criteria:**
- 2‑iteration shakedown completes without MaxTurnsExceeded.
- Spec‑first pipeline runs both iterations without fallback.

---

## 5) Phase 2 — Baseline Autonomous Workflow

**Goal:** A reliable default workflow that can run unattended for small tasks.

**Baseline loop:**
1. Research → Technique selection
2. Spec‑first script generation
3. Execute + render
4. Evaluate (Quality Analyst)
5. Learn + decide next action

**Metrics to track:**
- Script success rate
- Render success rate
- Spec‑first usage rate (no fallback)
- Average doc queries per spec
- Quality score improvement per iteration

---

## 6) Phase 3 — Autonomy Expansion

**Additions:**
- Session summarization + compaction
- Cross‑session bootstrap (patterns + warnings)
- Automatic regression tests (evals + trace grading)

**Risk controls:**
- Budget guardrail enforced
- Guardrail coverage for all agents
- Pattern library validation gate

---

## 7) Phase 4 — Workflow Optimization

**Optional once stable:**
- Planner‑Executor‑Verifier core
- Beam search (N=2–3, K=1)
- Bandit selection for techniques
- Multi‑candidate script generation

---

## 8) Immediate Next Steps (Recommended Order)

1. Verify API Spec Agent no longer loops (bundle‑first plan).
2. Verify camera fix (render completes).
3. Verify enum guardrail in spec‑first.
4. Run 2‑iteration shakedown for Phase‑4 gate.

---

## 9) Reference Docs

- `docs/PHASE_4_GATING_ROADMAP_2026-01-25.md`
- `docs/SCRIPT_WRITER_OVERHAUL_PROPOSAL_2026-01-25.md`
- `docs/CURRENT_ISSUES_CONSOLIDATED_2026-01-26.md`
- `docs/VERSION_TRUTH.md`
- `docs/SDK_ENFORCEMENT_PROTOCOL.md`

---

*End of Roadmap*
