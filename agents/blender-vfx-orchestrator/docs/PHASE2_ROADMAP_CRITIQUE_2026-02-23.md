# Phase 2 Roadmap Critique (Design Audit)

Date: 2026-02-23  
Author: Codex (roadmap design review)  
Scope: `PHASE2_ROADMAP.md` and associated research docs  
Non-goal: This is not a codebase implementation audit.

---

## 1) Why This Document Exists

The current Phase 2 roadmap is unusually ambitious and unusually well-researched. That is a strength. It is also the main source of risk: as plan size and coupling increase, execution clarity drops.

This document is a rigorous criticism of roadmap design quality, focusing on:

1. Where execution is likely to stall or fragment.
2. Which assumptions are fragile.
3. Which sections are over-specified vs under-specified.
4. How to tighten the roadmap without losing vision.

This critique assumes your status update is correct: work is at `2A-5` and Phase 2 execution has only recently started.

---

## 2) Evaluation Method

Reviewed primary roadmap and mission artifacts:

- `agents/blender-vfx-orchestrator/docs/PHASE2_ROADMAP.md`
- `agents/blender-vfx-orchestrator/docs/MISSION_STATEMENT_2026-02-22.md`
- `agents/blender-vfx-orchestrator/docs/research/REVISED_PHASE_2A_2B.md`
- `agents/blender-vfx-orchestrator/docs/research/REVISED_PHASE_2C_2D.md`
- `agents/blender-vfx-orchestrator/docs/research/REVISED_CROSS_CUTTING_SECTIONS.md`
- `agents/blender-vfx-orchestrator/docs/research/SDK_FEATURES_ANALYSIS.md`
- `agents/blender-vfx-orchestrator/docs/research/MONITORING_QUALITY_ARCHITECTURE.md`
- `agents/blender-vfx-orchestrator/docs/research/AUTONOMOUS_SYSTEMS_RESEARCH.md`

Audit criteria:

1. Internal consistency (definitions, numbering, phase intent).
2. Dependency realism (critical path vs soft dependencies).
3. Gate quality (definition-of-done and entry/exit criteria).
4. Control-plane quality (rollback, feature flags, escalation, budget controls).
5. Learning-system safety (KB quality, poisoning resistance, decay/reinforcement).
6. Delivery survivability (what can fail while still preserving progress).

---

## 3) Executive Assessment

### Bottom line

The roadmap is strategically strong but operationally overloaded.

The architecture direction is coherent:

- deterministic safety for known failure modes,
- agent creativity where value is highest,
- evidence-gated learning,
- staged autonomy progression.

The execution plan is where risk concentrates:

- too many cross-coupled systems advancing together,
- several gates that are conceptually good but not statistically strong enough,
- directional work that can leak into committed scope,
- governance rules that are broad but not sharp at decision time.

### What this means at your current stage (`2A-5`)

At `2A-5`, the roadmap should be judged by whether it creates a narrow, controlled path to reliability. Right now, it still allows too many concurrent paths. That increases schedule and quality risk even if engineering quality is high.

---

## 4) Critical Strengths to Preserve

These should not be diluted while tightening the plan:

1. Ranked design principles with explicit precedence (`P1` through `P7`) in `PHASE2_ROADMAP.md` and mission alignment in `MISSION_STATEMENT_2026-02-22.md`.
2. Pipeline-level architecture tying documentation, research, generation, evaluation, learning, and KB updates (`PHASE2_ROADMAP.md`, Architecture section).
3. Explicit decomposition gate between 2B and 2C (`PHASE2_ROADMAP.md`, Orchestrator Decomposition Gate).
4. Clear insistence on deterministic monitoring and bounded corrective behavior (`MONITORING_QUALITY_ARCHITECTURE.md`).
5. Concrete acknowledgment of budget as a first-class system constraint (`MISSION_STATEMENT_2026-02-22.md`, Budget; roadmap budget analysis).

These are not the problem. They are the anchor points.

---

## 5) Severity-Ranked Criticisms, Predicted Failure Modes, and Fixes

## C1 (Critical): Phase scope is too broad for the reliability-first principle

Roadmap signal:

- `P1` says reliability before capability.
- Phase 2 still carries a large number of new controls, learning mechanisms, autonomy structures, and directional items across 2A-2D.

Why this is a problem:

- A roadmap can be logically correct and still fail operationally when too many partially-dependent systems change in the same period.
- You have multiple high-interaction control loops: monitoring, modification, grading, memory, dynamic instructions, escalation, and budget degradation.
- Reliability work requires low state-space; this plan expands state-space early.

Predicted failure mode:

- Frequent regressions that are hard to attribute.
- Teams spend cycles diagnosing interactions rather than advancing capability.
- Success appears random because too many variables move between runs.

Proposed fix:

Create three explicit work classes in the roadmap and enforce them:

1. `Committed` (must ship for current phase gate).
2. `Experimental` (can be prototyped but cannot block phase completion).
3. `Directional` (design-only; no implementation in current gate window).

Document change:

- Add a required `Class` column to every item in 2B-2D.
- Add policy: only `Committed` items count toward gate readiness.

Decision test:

- If a feature misses schedule and the gate still holds, it was correctly classified.
- If a missed feature blocks all progress, classification was wrong.

---

## C2 (Critical): Gate criteria exist but are not statistically robust enough

Roadmap signal:

- E2E gate checks are mostly 5-run windows (`PHASE2_ROADMAP.md`, E2E Validation Gates).
- Autonomy progression thresholds include low-count transitions (10 runs for level movement in 2D-1).

Why this is a problem:

- Small windows are highly sensitive to prompt variance and random model behavior.
- Promotion/demotion decisions on low sample sizes create oscillation in governance itself.

Predicted failure mode:

- False confidence after lucky streaks.
- Overreaction to short-term dips.
- Gate pass/fail decisions that are not reproducible week-to-week.

Proposed fix:

Upgrade gate statistics:

1. Use minimum sample size and confidence bounds for gate decisions.
2. Separate `smoke gate` and `stability gate`.
3. Require rolling-window metrics (not single batch).

Concrete thresholds:

- Smoke gate: 5 runs, basic sanity only.
- Stability gate: 20 runs, required for phase promotion.
- Any level-based autonomy change: minimum 30 samples for that effect type.

Document change:

- Replace single 5-run phase progression checks with two-stage checks.
- Add explicit confidence requirement language (for example: do not promote if lower confidence bound is below target).

---

## C3 (Critical): Dependency graph is accurate but under-communicates critical path pressure

Roadmap signal:

- Large dependency graph in `PHASE2_ROADMAP.md`.
- Many “parallelism opportunities” listed.

Why this is a problem:

- A dependency graph shows what depends on what, but not what can kill schedule if delayed.
- “Parallelizable” items still compete for shared architecture surfaces and reviewer bandwidth.

Predicted failure mode:

- Team starts many threads in parallel.
- Merge debt and integration risk rise.
- Critical-path work slows while non-critical work progresses.

Proposed fix:

Add a critical-path overlay:

1. Label each item `Critical Path`, `Buffer`, or `Optional`.
2. Enforce WIP limit: max 2 active roadmap threads at once during 2B.
3. Add `blocker owner` and `latest safe start` fields.

Document change:

- Add a “Critical Path Table” immediately after Dependency Graph.
- Add WIP policy in Implementation Sequence section.

---

## C4 (High): Definition-of-done is inconsistent across items

Roadmap signal:

- Some items have detailed tests and rollback plans.
- Some (especially directional and cross-cutting transitions) have broad intent but no measurable completion proof.

Why this is a problem:

- Teams can declare “done” on implementation without “done” on impact.
- This allows feature accumulation without reliability movement.

Predicted failure mode:

- Progress appears high in checklists but mission metrics stay flat.
- Retrospectives become argument-heavy because completion semantics differ per item.

Proposed fix:

Standardize every roadmap item on one contract:

1. Objective (one sentence).
2. Primary metric (numerical).
3. Guardrail metric (what must not regress).
4. Exit test (automated if possible).
5. Rollback trigger.
6. Owner.

Document change:

- Add this as a mandatory template under Testing Strategy and apply it to all 2B+ entries.

---

## C5 (High): Directional 2D items are correctly marked, but still too execution-adjacent

Roadmap signal:

- 2D-3, 2D-4, 2D-5 are “directional — design after 2C”.
- They are still represented in an implementation-like structure with expected integrations.

Why this is a problem:

- Teams tend to begin coding against directional specs once fields/files are named.
- Directional items consume attention before prerequisite data exists.

Predicted failure mode:

- Premature implementation of autonomy features.
- Rework once 2C data contradicts initial assumptions.

Proposed fix:

Move directional items into a separate appendix:

1. Keep rationale and research links.
2. Remove file-level implementation references until design gate opens.
3. Add explicit unlock condition for each directional item.

Document change:

- Create “Phase 2D Candidate Backlog (Locked)” section.
- Keep only 2D-1 and 2D-2 in active roadmap body if those are truly near-term.

---

## C6 (High): Budget model is directionally useful but under-specifies worst-case envelopes

Roadmap signal:

- Budget Impact Analysis provides net savings and projected run cost in `PHASE2_ROADMAP.md`.
- Mission budget is strict ($20/month).

Why this is a problem:

- Average-case estimates are insufficient for protecting monthly budget.
- Orchestrator behavior during failure modes can be cost-amplifying.

Predicted failure mode:

- Good average cost but periodic blowups consume monthly budget early.
- Team optimizes expected value while violating hard budget constraint.

Proposed fix:

Require three-cost reporting:

1. Median run cost.
2. P90 run cost.
3. Worst-case capped cost (absolute).

And enforce hard stop rules:

- Per-run spend cap.
- Per-day spend cap.
- Auto-disable expensive evaluators above cap until manual override.

Document change:

- Expand Budget Impact Analysis with distribution assumptions.
- Add a cost SLO table to Rollback Decision Criteria.

---

## C7 (High): KB seeding and memory decay are strong ideas, but trust bootstrapping is risky

Roadmap signal:

- KB seeding after manual rewrite with `emerging` trust.
- Memory decay and evidence gating in 2B.

Why this is a problem:

- Seeded entries are better than empty KB, but they are still synthetic priors, not proven execution truths.
- If injected too early or too broadly, they can steer generation toward coherent but non-performing patterns.

Predicted failure mode:

- “Confidently wrong” guidance.
- Lower exploration diversity because seeded patterns dominate prompt context.

Proposed fix:

Split seeded knowledge into two lanes:

1. `reference_seed` (never directly prescriptive).
2. `execution_validated` (eligible for prescriptive injection).

Injection policy:

- Script Writer receives only execution-validated items above threshold.
- Research Agent may see reference seeds as exploratory hints.

Document change:

- Update KB Seeding Strategy with lane separation and injection policy.
- Add explicit anti-poisoning checks and quarantine path.

---

## C8 (High): Multi-grader plan can become a control-loop instability source

Roadmap signal:

- 3-tier evaluation in roadmap; strong motivation from research docs.
- QA diagnosis bridge and bounded modification strategies in monitoring architecture.

Why this is a problem:

- Multiple graders with different priors can produce conflicting optimization pressure.
- If weighting and short-circuit logic are not carefully bounded, the system can chase metric artifacts.

Predicted failure mode:

- Score improves while visual quality stagnates for user-relevant outcomes.
- Modification loop oscillates between grader preferences.

Proposed fix:

Add explicit evaluator governance:

1. Frozen weighting per phase window.
2. Conflict resolution rules (deterministic safety gates override all).
3. Drift detection between evaluator outputs.
4. “No action” outcome when confidence is low.

Document change:

- Expand 2B-3 section with evaluator governance and anti-oscillation policy.
- Add one regression test category: “grader conflict scenarios”.

---

## C9 (Medium-High): Feature-flag strategy defaults create high blast radius

Roadmap signal:

- Feature flags default to `True` for new behavior in Rollback Strategy.

Why this is a problem:

- Default-on is fine after stabilization, risky during early rollout.
- Multiple default-on flags increase combinatorial behavior changes from one deployment.

Predicted failure mode:

- Hard-to-diagnose regressions after merges because several new systems are simultaneously active.

Proposed fix:

Use phased defaults:

1. New features default `False` until smoke gate passes.
2. Then default `True` in staging profile only.
3. Then default `True` globally after stability gate.

Document change:

- Update Feature Flags policy with rollout states (`experimental`, `staging`, `global`).

---

## C10 (Medium): Decomposition gate effort appears underestimated

Roadmap signal:

- Decomposition gate estimates low risk and modest effort.
- Current monolith size and phase-touch points are large.

Why this is a problem:

- Mechanical refactor in orchestration-heavy systems often exposes implicit contracts.
- Existing hidden couplings can surface only after split.

Predicted failure mode:

- “Refactor-only” phase spills into behavior changes.
- Test suite passes but runtime edge cases regress.

Proposed fix:

Treat decomposition as an explicit migration program:

1. Snapshot behavioral contract before split.
2. Use compatibility wrappers during transition.
3. Phase split by one module at a time.
4. Require trace-diff checks on representative runs.

Document change:

- Replace single decomposition step with staged migration checklist.
- Add a rollback branch strategy specifically for decomposition.

---

## C11 (Medium): Internal doc drift is already visible

Roadmap signal:

- `REVISED_CROSS_CUTTING_SECTIONS.md` contains outdated 2C stage mapping (`Agent.clone`, old numbering for later 2C items).
- Main roadmap has updated item mapping.

Why this is a problem:

- Conflicting source documents fragment implementation.
- Contributors may implement against obsolete references.

Predicted failure mode:

- Wrong feature shipped under wrong phase ID.
- Integration discussions spend time reconciling labels instead of behavior.

Proposed fix:

Adopt “single source of phase truth” policy:

1. `PHASE2_ROADMAP.md` is canonical.
2. Research docs reference canonical IDs only.
3. Add a consistency lint script that validates phase identifiers across docs.

Document change:

- Add “Source of Truth and Sync Policy” section.

---

## C12 (Medium): The roadmap lacks a compact risk register with owner-level accountability

Roadmap signal:

- Risks are distributed throughout many sections, but no concise top-risk table with ownership and next review date.

Why this is a problem:

- In large plans, distributed risk prose is hard to operationalize.
- Teams need one place that says: what can break us this month, and who owns mitigation.

Predicted failure mode:

- Known risks re-discovered repeatedly.
- Risk response is reactive rather than scheduled.

Proposed fix:

Add a one-page Phase Risk Register:

1. Risk statement.
2. Leading indicator.
3. Trigger threshold.
4. Mitigation action.
5. Owner.
6. Review cadence.

Document change:

- Add register near Dependency Graph and update weekly.

---

## 6) Predicted Problem Patterns by Phase

## Phase 2A (now through 2A completion)

Likely issues:

1. “Done but not absorbed”: controls land but downstream behavior still unstable.
2. Premature confidence from isolated tests rather than multi-run behavior.
3. Rollout friction when many flags are active at once.

Key protection:

- Freeze new feature intake until 2A stability gate passes on rolling window.

## Phase 2B

Likely issues:

1. Context and iteration-state changes interact in non-obvious ways.
2. Multi-grader rollout obscures whether improvements come from generation or evaluation.
3. HITL insertion increases flow complexity and pause/resume edge cases.

Key protection:

- Enforce strict critical path and WIP cap.

## Phase 2C

Likely issues:

1. Technique selection, experiments, and distillation can multiply state complexity quickly.
2. Budget degradation logic may hide true quality potential if triggered aggressively.
3. Multi-physics extension increases API surface and failure branching.

Key protection:

- Only enter 2C after decomposition gate and 2B stability gate, not just functional pass.

## Phase 2D

Likely issues:

1. Autonomy progression can be gamed by narrow prompt distributions.
2. Cross-session transfer can overfit if not diversity-checked.
3. Directional items risk early implementation despite unknowns.

Key protection:

- Treat 2D as controlled pilot mode until robust sample sizes exist per effect type.

---

## 7) Tightening Plan: Concrete Structural Changes

## A) Add roadmap status classes

Add `Class` field for each item:

- `Committed`
- `Experimental`
- `Directional`

Rule:

- Phase gates only depend on `Committed`.

## B) Add gate packet template

Each phase gate must include:

1. Metric dashboard snapshot.
2. Run sample size and distribution.
3. Regression matrix.
4. Budget distribution.
5. Rollback readiness.

## C) Add critical path and WIP limits

For 2B and 2C:

- max two concurrent critical-path streams,
- max one experimental stream in parallel.

## D) Split “done” into two states

Use:

- `Implemented`
- `Operationally Validated`

This prevents checklist progress from being mistaken for reliability progress.

## E) Add cross-doc consistency policy

Mandatory:

1. Canonical phase ID list.
2. Pre-merge doc consistency check for phase references.
3. One owner for roadmap integrity.

---

## 8) Proposed Edit List for `PHASE2_ROADMAP.md`

High-value edits with low disruption:

1. Add a “Roadmap Governance” section after Design Principles:
   - item class policy,
   - done-state policy,
   - source-of-truth policy.
2. Replace single-phase gate checks with two-stage smoke/stability gates.
3. Add critical-path table and WIP limits next to Dependency Graph.
4. Expand Budget Impact Analysis with median/P90/worst-case cost.
5. Add one-page risk register with owners and weekly review cadence.
6. Move 2D directional items into a locked candidate backlog appendix.
7. Update Feature Flag policy from default-on to staged rollout defaults.

---

## 9) What Not to Change

Do not weaken these:

1. Principle ranking (`P1` through `P7`).
2. Blender truth-pack authority model.
3. Deterministic-first monitoring strategy with optional escalation.
4. Decomposition gate as mandatory boundary before 2C scale-up.
5. Evidence-gated learning philosophy.

These are your strategic advantages. Tightening should target execution mechanics, not architectural intent.

---

## 10) Final Judgment

The roadmap is not wrong. It is heavy.

The core challenge is not “what to build”; it is preventing a high-quality strategy from becoming an execution tangle.

If you apply the control-plane changes in this critique, you keep the roadmap’s ambition while sharply reducing the most likely failure mode: too many intelligent ideas landing at once without enough statistical gating and sequencing discipline.

---

## 11) Traceability Map (Criticism → Source Anchors)

This section maps each criticism to concrete source locations so the critique can be audited quickly.

### C1 (Scope too broad vs reliability-first)

- `PHASE2_ROADMAP.md`: Design Principles (P1), full 2A-2D scope, Implementation Sequence.
- `MISSION_STATEMENT_2026-02-22.md`: Principle P1 and Success Criteria.

### C2 (Gate statistics not robust enough)

- `PHASE2_ROADMAP.md`: E2E Validation Gates Between Phases (5-run progression checks), 2D-1 level criteria.
- `MISSION_STATEMENT_2026-02-22.md`: “Earn autonomy through evidence”.

### C3 (Dependency graph lacks critical-path pressure model)

- `PHASE2_ROADMAP.md`: Dependency Graph and Parallelism opportunities.
- `PHASE2_ROADMAP.md`: Implementation Sequence.

### C4 (DoD inconsistency)

- `PHASE2_ROADMAP.md`: Mixed detail levels across 2B/2C/2D items.
- `REVISED_PHASE_2A_2B.md` and `REVISED_PHASE_2C_2D.md`: uneven item test/exit specificity.

### C5 (Directional leakage risk)

- `PHASE2_ROADMAP.md`: 2D directional framing for 2D-3/2D-4/2D-5.
- `REVISED_PHASE_2C_2D.md`: directional status language and dependencies.

### C6 (Budget envelope under-specified)

- `MISSION_STATEMENT_2026-02-22.md`: hard monthly budget and target run cost.
- `PHASE2_ROADMAP.md`: Budget Impact Analysis and projected run cost.
- `MONITORING_QUALITY_ARCHITECTURE.md`: token-budget assumptions and per-agent allocation.

### C7 (KB seeding trust risk)

- `PHASE2_ROADMAP.md`: KB Seeding Strategy and trust-level treatment.
- `AUTONOMOUS_SYSTEMS_RESEARCH.md`: memory decay and poisoning concerns.
- `MISSION_STATEMENT_2026-02-22.md`: knowledge base quality risk in current-state section.

### C8 (Multi-grader control-loop instability risk)

- `PHASE2_ROADMAP.md`: 2B-3 Multi-Grader and conflict resolution section.
- `MONITORING_QUALITY_ARCHITECTURE.md`: quality feedback loop and bounded modification strategy.
- `AUTONOMOUS_SYSTEMS_RESEARCH.md`: multi-grader and feedback-loop prevention.

### C9 (Flag default blast radius)

- `PHASE2_ROADMAP.md`: Rollback Strategy → Feature Flags defaults.

### C10 (Decomposition effort underestimated)

- `PHASE2_ROADMAP.md`: Orchestrator Decomposition Gate and low-risk characterization.
- `REVISED_CROSS_CUTTING_SECTIONS.md`: decomposition gate rationale and success criteria.

### C11 (Internal document drift)

- `REVISED_CROSS_CUTTING_SECTIONS.md`: “Roadmap Items by Pipeline Stage” table with outdated 2C mappings.
- `PHASE2_ROADMAP.md`: updated 2C numbering and removal/relabeling.
- `REVISED_PHASE_2C_2D.md`: current 2C numbering.

### C12 (No compact risk register)

- `PHASE2_ROADMAP.md`: distributed risk and rollback text without a single owner-driven risk register.
- `REVISED_CROSS_CUTTING_SECTIONS.md`: broad governance guidance without consolidated risk ownership table.
