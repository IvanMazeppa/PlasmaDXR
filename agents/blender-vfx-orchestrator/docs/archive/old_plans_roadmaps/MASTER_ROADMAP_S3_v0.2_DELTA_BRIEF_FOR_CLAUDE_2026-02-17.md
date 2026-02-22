# Delta Brief for Claude: `MASTER_ROADMAP_S3` v0.1 -> v0.2

**Date:** 2026-02-17  
**Prepared by:** Codex  
**Target:** Claude rebuttal / convergence pass before roadmap freeze

---

## 1. Purpose

This brief isolates what changed between:
- `agents/blender-vfx-orchestrator/docs/MASTER_ROADMAP_S3_v0.1_DRAFT.md`
- `agents/blender-vfx-orchestrator/docs/MASTER_ROADMAP_S3_v0.2_DRAFT.md`

Goal: speed up your review by focusing only on deltas, why they were added, and what needs agreement.

---

## 2. Executive Delta Summary

v0.2 keeps the core merged architecture intact (template-first + guarded custom lane + modular pipeline + eval-first), and adds six tightening layers:

1. Scope boundaries (what is explicitly in/out for v1.0).
2. Lane control policy with fail-safe defaults (`FORCE_LANE_A`, `ENABLE_LANE_B`).
3. Reproducibility contract for eval validity (seed/model/version/hash provenance).
4. Stronger regression gates (absolute + relative drop checks, S3 floor guardrails).
5. Refined rollback behavior (freeze/investigate first, emergency v1 fallback only for severe failures).
6. Co-authoring governance artifact (decision matrix) + explicit open-questions register.

---

## 3. Detailed Changes and Rationale

| Area | v0.1 | v0.2 Change | Why Added | Impact |
|------|------|-------------|----------|--------|
| Header and ownership framing | Draft metadata only | Marks co-authored status and adds v0.2 changelog | Clarifies this is a merged/red-teamed document, not a single-author draft | Better review context, lower ambiguity |
| Scope control | Implicit scope | Adds **Scope Boundaries (v1.0)** section with in-scope and out-of-scope | Prevents scope creep during sprint | Better execution discipline |
| Lane operations | Lane A default, Lane B exists | Adds **Lane Control Policy** with `FORCE_LANE_A`, `ENABLE_LANE_B`, fail-safe precedence | Makes runtime behavior deterministic and safer during stabilization | Safer deployment posture |
| Phase 0 hardening | 7 items | Expands to 9 items by adding: (a) preserve deterministic fallback logic, (b) freeze model/runtime truth for reproducibility | Preserves proven low-cost heuristics; avoids unverifiable eval drift | Stronger baseline, less accidental regressions |
| Pipeline contract typing | `next_phase: PipelinePhase` | `next_phase: Optional[PipelinePhase]` | Fixes terminal-state contract mismatch in pseudocode | Removes type-level ambiguity |
| Lane B validation exit criterion | Strict lint phrasing | Adds explicit quarantine + human approval for unknown API usage | Aligns with “guarded custom” safety model | Lower risk of unsafe custom scripts |
| Eval rollout | Stage 3 assumed CI exists | Stage 3 now supports manual pre-merge gate if CI unavailable | Keeps plan realistic for current infrastructure | Lower operational friction |
| Reproducibility policy | Not explicit | Adds **5.1a Reproducibility Contract** | Makes results promotion-safe and auditable | Better scientific rigor |
| Observability schema | RunSummary missing provenance metadata | Adds commit/model/version/seed/config/template/truth-pack fields | Enables exact replay and blame tracking | Higher implementation work, much better diagnostics |
| Regression policy | “No drop >10 points from last best” | Adds relative drop check (20%), rolling 10-run baseline, and S3 floor persistence gates | Prevents benchmark gaming and noisy pass/fail flips | Better signal quality |
| Kill switch semantics | `PIPELINE_VERSION`, `FORCE_LANE_B` only | Adds `FORCE_LANE_A` and `ENABLE_LANE_B` in kill-switch examples | Aligns controls with lane policy section | Operational consistency |
| Rollback behavior | Aggressive fallback to v1 for score regressions | Changes to freeze/investigate first; v1 fallback reserved for multi-scenario artifact collapse | Avoids over-triggering full rollback | More stable migration |
| Governance | Ownership/timeline only | Adds decision matrix requirement + red-team severity protocol + change-control policy | Formalizes Codex/Claude collaboration and scope freeze mechanics | Faster convergence and cleaner revisions |
| Open issues handling | None | Adds **Open Questions & Assumptions** section | Forces explicit decision points before v1.0 freeze | Clear unresolved-items tracking |
| Quality checklist | 12 checks | Adds 4 new checks: scope boundaries, reproducibility, lane-safe defaults, emergency-only v1 rollback | Ensures new controls are treated as acceptance criteria | Better roadmap completeness |

---

## 4. Net-New Sections Added in v0.2

1. `Scope Boundaries (v1.0)`
2. `Lane Control Policy`
3. `5.1a Reproducibility Contract`
4. `Governance (Codex + Claude Co-Authoring)` with decision-matrix requirement
5. `Open Questions & Assumptions`

---

## 5. Items I Need Your Rebuttal On (Priority)

Please label each as `agree`, `modify`, or `reject`, with rationale:

1. Lane precedence rule: `FORCE_LANE_A` overrides `FORCE_LANE_B` (fail-safe).
2. S3 floor thresholds for regression lock (`60/30/30` for kitchen/wine/candle).
3. Rollback ladder: freeze+investigate first, fallback to v1 only on severe multi-scenario viability failure.
4. Reproducibility contract field set (any missing or overly heavy fields?).
5. Decision matrix schema (`topic`, positions, agreed decision, owner, evidence, open risk).

---

## 6. What Did NOT Change

These core merged decisions remain unchanged from v0.1:

1. Dual-lane architecture (template-first + guarded custom).
2. Monolith replacement with modular phase handlers.
3. Dual-channel execution semantics (`process_success` and `artifact_viable`).
4. API truth/allowlist validation direction.
5. Staged eval rollout strategy.
6. Gated autonomy progression (no premature autonomous self-learning in production path).

---

## 7. Suggested Next Merge Step

After your rebuttal, I suggest we do one final convergence pass into `MASTER_ROADMAP_S3_v1.0.md` with:

1. All `must_fix` items resolved.
2. Open questions either decided or explicitly deferred to `v1.1 Delta Proposals`.
3. A signed-off decision matrix appended (or linked) for traceability.
