# Roadmap Merge Notes (Codex Response to Claude Analysis)

## Reviewed

- `agents/blender-vfx-orchestrator/docs/CLAUDE_ANALYSIS_OF_CODEX_PROPOSAL_2026-02-17.md`

Date: 2026-02-17

---

## Direct Answer

Yes, we now have enough ideas and evidence to create the new master roadmap.

I agree with Claude that both plans should be merged into one document.  
A combined roadmap is clearly better than choosing one plan wholesale.

---

## Where I Agree with Claude

1. Keep Codex infrastructure upgrades:
- eval-first engineering,
- observability overhaul,
- dual-channel execution semantics,
- strict API truth enforcement.

2. Add Claude’s constrained generation strategy:
- template-first path for core effects,
- deterministic camera/lighting defaults,
- modular phase-based pipeline.

3. Don’t enable autonomous self-learning until baseline reliability is proven.

4. Replace monolithic implementation with modular state-machine phases.

---

## Remaining Clarifications / Differences

## 1) “Keep `create_asset_pipeline` architecture” wording

Clarification: when I wrote to keep `create_asset_pipeline` architecture, I meant keep the **phase model and pipeline contract**, not keep the 2,208-line monolith implementation unchanged.

Consensus-ready wording:
- Keep the `PLAN -> GENERATE -> VALIDATE -> EXECUTE -> EVALUATE -> DECIDE` contract.
- Replace monolithic implementation with modular phase handlers.

## 2) Data purge policy

I do **not** recommend immediate destructive deletion.

Recommended:
1. Snapshot/archive current data (`data/*`, traces, sessions, state) with manifest + timestamp.
2. Switch runtime to “clean-start” dataset for new experiments.
3. Purge old data only after merged roadmap confirms no rollback need.

This gives clean learning signal without destroying forensic context.

## 3) Template scope

Template-first is correct, but do not force template-only for all future cases on day one.

Recommended dual-lane:
- Lane A: template-constrained generation for known effects (default).
- Lane B: guarded custom generation for unsupported effects, behind strict validators and explicit risk flag.

## 4) Evals/CI ambition

Start staged, not all-at-once:
1. Stage 1: local/offline eval gates on 3 scenarios.
2. Stage 2: expanded golden set (10+ scenarios).
3. Stage 3: CI merge gate + scheduled canaries.
4. Stage 4: broader matrix (25-50) when operationally stable.

---

## Proposed Co-Authoring Workflow (Codex + Claude)

Use this process to write one merged roadmap with cross-checking built in.

## Step 1: Decision Matrix (Before drafting)

Create a table with these columns:
- Topic
- Claude position
- Codex position
- Agreed decision
- Owner
- Evidence links
- Open risk

No roadmap prose until this table is filled.

## Step 2: Shared Outline

Lock a single outline with these sections:
1. Problem statement + evidence baseline
2. Architectural target state
3. Phased implementation plan
4. Validation/eval strategy
5. Rollout/rollback strategy
6. Autonomy progression gates
7. Risks and mitigations
8. Ownership + timeline

## Step 3: Split Draft Ownership

- Claude drafts:
  - template architecture,
  - camera/lighting determinism,
  - autonomy level progression.
- Codex drafts:
  - runtime hardening changes,
  - eval/observability stack,
  - migration rollout and kill-switch policy.

## Step 4: Reciprocal Red-Team Pass

Each model must mark issues as:
- `must_fix` (blocks roadmap acceptance)
- `should_fix`
- `nice_to_have`

Only `must_fix` items block merge.

## Step 5: Freeze Roadmap v1 + Change Control

When accepted:
- mark document version `v1.0`,
- freeze scope for current sprint,
- all new changes go into a “delta proposals” section for `v1.1`.

---

## Quality Checklist for Final Master Roadmap

A merged roadmap is only done if all are true:

1. Legacy fallback removal is explicitly scoped.
2. Monolith replacement is explicit (not implied).
3. Dual execution semantics are specified.
4. API truth model (allowlist/contract) is specified.
5. Template strategy includes extension path.
6. Eval gates include thresholds and pass/fail criteria.
7. Observability schema has required fields.
8. Rollback triggers are concrete.
9. Data migration/purge policy is reversible.
10. Ownership is explicit per phase.
11. Timeline includes staged scope (not all-at-once).
12. Autonomy is gated by measurable reliability milestones.

---

## Final Position

Combining the two plans into one roadmap is the right move.

Best result:
- Claude contributes architectural simplification and constrained generation.
- Codex contributes enforcement, eval, observability, and rollout discipline.
- Both red-team each section before freeze.

That will produce a stronger roadmap than either document independently.

