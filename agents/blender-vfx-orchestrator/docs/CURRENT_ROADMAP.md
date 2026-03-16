# Current Roadmap

Status: authoritative
Last verified: 2026-03-16
Purpose: canonical synthesis of the March 2026 architecture reviews and live repo state

## Scope

This roadmap is the planning document to use for active implementation.

It synthesizes:

- `docs/reviews/2026-03/20260314_DEEP_ANALYSIS_ACTIONABLE_ROADMAP.md`
- `docs/reviews/2026-03/20260314_DEEP_AUTONOMY_ANALYSIS_GPT_54_XHIGH.md`
- the current live code and current-state docs

The source reviews remain valuable, but this file is the canonical roadmap.

## Conflict Resolution Rules

When the source reviews disagree:

1. Prefer live code over either review.
2. Prefer the Opus review for immediate implementation-state facts.
3. Prefer the GPT-5.4 review for medium-term architecture sequencing and autonomy design.
4. If a review recommendation contradicts the current codebase reality, record the resolved conclusion here instead of carrying both versions forward.

## Confirmed Current State

These points are treated as true for planning unless new code evidence disproves them:

- `TechniqueContract` is implemented and materially improved technique binding.
- stale render reuse is no longer the main active blocker
- `RepairIntent` / `choose_repair_intent()` already exist
- `QualityIssue` / `structured_issues` already exist, but are not yet authoritative enough
- section patching exists, but structural repair is not yet proven end to end in live runs
- the dominant remaining problems are wiring, state authority, and repair-boundary quality

## Roadmap Overview

| Phase | Goal | Status | Main Source |
|------|------|--------|-------------|
| 1 | Fix state authority and repair-routing inputs | Complete | Opus |
| 2 | Make typed QA-to-repair signals authoritative | Active | Opus + GPT-5.4 |
| 3 | Prove structural repair in live runs | Next | Opus |
| 4 | Make learning/evaluation evidence trustworthy | Queued | GPT-5.4 + Opus |
| 5 | Gate autonomy with evidence and integrate hybrid HITL | Later | GPT-5.4 |
| 6 | Expand capability acquisition and raise the quality ceiling | Later | GPT-5.4 + Opus |

## Phase 1: Fix State Authority and Routing Inputs — COMPLETE (2026-03-15)

Goal:
- make the deterministic state machinery actually drive live repair routing

Status: **DONE** — all exit criteria met, 9 dedicated tests pass, full suite 447 pass.

What shipped:
- `orchestrator.py`: replaced `session.iterations.append(iter_result)` with `session.record_iteration(iter_result)` — activates `StuckDetectionState.update_from_iteration()`
- `orchestrator.py`: `choose_repair_intent()` now reads `session.stuck_state.plateau_count`, `.same_issue_count`, `.escape_level` directly (not LLM-derived values)
- `orchestrator.py`: Quality Gate no longer overwrites deterministic `escape_level` — logs disagreement instead
- `orchestrator.py`: both Quality Gate prompt paths inject deterministic stuck-state into context and show `Plateau:` count
- `session_manager.py`: `get_context_for_agents()` returns `escape_level` and `plateau_count` fields
- `tests/test_state_authority_wiring.py`: 9 tests covering record_iteration → stuck_state → repair_intent flow

Exit criteria verification:
- `StuckDetectionState.update_from_iteration()` fires every iteration ✓
- `plateau_count` and `same_issue_count` change on real runs ✓ (test_plateau_increments_on_small_delta, test_same_issue_count_increments)
- a plateau or repeated-issue case can change repair behavior without LLM guesswork ✓ (test_three_plateau_iterations_trigger_modify_code)

## Phase 2: Make Typed QA-to-Repair Signals Authoritative

Goal:
- stop routing from depending mainly on prose and keyword heuristics

Why this is second:
- once state inputs are correct, the next biggest ambiguity is defect classification

Work:
- require `structured_issues` when `passed=False`
- improve Quality Analyst prompting so every issue gets a `kind`, `repair_mode_hint`, `target`, and `confidence`
- prioritize `structured_issues` inside `choose_repair_intent()`, keeping keyword heuristics as fallback only
- harden `QualityIssue` from soft string fields toward validated categories
- keep code-grounded QA feedback, but make it a supporting signal rather than the only bridge

Job order:

0. **Prerequisite: Pipeline-level tracing.** Emit pipeline events (repair intent, truth pack fixes, execution results, phase transitions) to the JSONL trace file alongside SDK spans. Without this, Phase 2-4 exit criteria and proof obligations cannot be verified from trace data. The current trace only captures SDK-level spans (agent/tool/generation); all pipeline decisions are printed to stdout and lost.
1. Update the Quality Analyst prompt and runtime instructions so failed evaluations always emit `structured_issues`.
2. Add or tighten the quality output guardrail so `passed=False` without `structured_issues` is treated as invalid.
3. Harden `QualityIssue` validation so `kind` and `repair_mode_hint` are constrained to accepted categories.
4. Change `choose_repair_intent()` to prioritize `structured_issues` before keyword heuristics.
5. Build a small repair-mode classification benchmark from historical traces and use it to measure agreement with human labels.

Exit criteria:
- structural defects are routinely tagged as structural
- parameter-only issues are routinely tagged as parameter
- repair-mode classification matches human labels at a useful rate on a small regression set

Proof obligations:
- an Ireland Flag-style defect is classified as structural without relying on keyword luck
- the router can explain whether it used structured issues or heuristic fallback
- the benchmark gives a stable number you can track over time

## Phase 3: Prove Structural Repair in Live Runs

Goal:
- turn `modify_code` and section patching from infrastructure into operational reality

Why this is third:
- without real production-style proof, structural repair remains theoretical

Work:
- make `RepairIntent` the sole repair-mode authority
- allow Learning Agent parameter edits only when `repair_intent.mode == "modify_params"`
- stop `iterate` from collapsing into “tune parameters”
- build a small structural-repair regression set
- prove at least one live path of `modify_code -> patch_script_section -> execute -> evaluate`
- close the hot-path enforcement gaps where direct `_impl` calls bypass the intended guardrailed surface or equivalent validation boundary

Suggested first cases:
- wrong stripe order
- missing collider
- camera inside geometry
- wrong attachment topology
- missing key light

Job order:

1. Gate Learning Agent parameter application on `RepairIntent.mode`.
2. Demote Learning Agent `next_action` so it can advise but not override repair mode.
3. Decide whether to expand the action vocabulary (`iterate_params` / `iterate_code`) or keep the existing vocabulary and treat it as advisory only.
4. Build the structural-repair regression set.
5. Force at least one live `modify_code` proof on a real case.
6. After proof, harden the hot path so critical validation is attached to the actual execution/modification path, not only to tool wrappers that can be bypassed.

Exit criteria:
- at least one structural defect reaches `modify_code` without manual forcing
- section patching preserves script quality better than full rewrite on at least one real case
- structural failures no longer get trapped in parameter loops by default

Proof obligations:
- `modify_code` succeeds or fails honestly on at least one production-style case
- parameter-first fast paths no longer silently preempt structural repair
- the regression suite can catch a future relapse back into parameter-tuning ruts

## Phase 4: Make Evaluation and Learning Evidence Trustworthy

Goal:
- ensure learning is built on provenance-safe evidence rather than loosely correlated outcomes

Why this is fourth:
- once repair behavior is real, the next risk is teaching the system from bad evidence

Work:
- create a canonical per-iteration manifest or experiment ledger
- record script hash, run dir, artifact validity, evaluator profile, repair mode, and score deltas
- calibrate evaluator score bands by physics family
- fix stale runtime/version metadata in artifact outputs and authority docs
- require clean manifest evidence before promoting patterns or strategies

Job order:

1. Define the minimum canonical iteration manifest.
2. Record that manifest from the live loop and make sure it is persisted alongside scorecards/artifacts.
3. Fix stale version metadata in artifacts and authority docs so manifests are trustworthy.
4. Add a manifest integrity test: manifest, session state, trace summary, and scorecard must agree.
5. Run a cross-physics reliability baseline on the current stack.
6. Build the evaluator calibration ladder by physics family.
7. Require clean manifests before learning promotion or pattern extraction becomes trusted.

Exit criteria:
- manifest, scorecard, session state, and trace summary agree on core iteration facts
- evaluator score bands are at least directionally sane for known-good and known-bad samples
- learning promotion can be audited without depending on an LLM reconstruction

Proof obligations:
- a sample of promoted patterns can be traced back to multiple clean manifests
- the pass threshold and score interpretation are no longer purely intuitive
- docs and artifacts no longer disagree on SDK/runtime truth

## Phase 5: Gate Autonomy with Evidence and Integrate Hybrid HITL

Goal:
- make autonomy something the system earns, not something configured optimistically

Why this is fifth:
- autonomy gates only mean anything after routing and evidence quality are trustworthy

Work:
- define autonomy promotion criteria by effect family
- measure valid-artifact rate, repair success rate, pass rate, and average cost
- keep pipeline-level HITL for session-level decisions
- add native SDK `needs_approval` for high-impact tool actions once routing is stable
- update stale authority docs so human and model planning stop wasting time on version confusion

Job order:

1. Define an autonomy promotion scorecard per effect family.
2. Pick the first pilot family for promotion measurement.
3. Keep the existing pipeline HITL for session-level checkpoints.
4. Add native `needs_approval` only at high-impact tool boundaries after Phases 2-4 are stable.
5. Document the promotion rules and demotion/fallback rules in canonical docs.

Exit criteria:
- every effect family can be classified as guided, assisted, or autonomous with evidence
- hybrid HITL is in place for the right boundaries
- “autonomous” is no longer shorthand for “fewer checkpoints”

Proof obligations:
- autonomy level changes can be justified from measured evidence
- hybrid HITL protects the expensive/high-risk actions without reintroducing prompt-level policy sprawl

## Phase 6: Capability Acquisition and Quality Ceiling

Goal:
- move beyond reliable repetition into bounded capability expansion and better scene quality

Why this is last:
- these are high-value, but they should not compete with core loop integrity

Work:
- add a `sandbox_experiment` or equivalent bounded capability-incubation path
- separate production iteration from novelty acquisition
- promote successful sandbox discoveries into packs, truth-pack extensions, or anti-pattern knowledge
- reintroduce deterministic scene assembly and first-render correctness checks after reliability is proven
- continue reducing prompt-simulated policy where deterministic seams are now clear

Job order:

1. Define the separation between `production`, `recovery`, and `sandbox_experiment` run modes.
2. Build the smallest useful sandbox path: doc retrieval -> minimal script -> run -> artifact capture -> verdict.
3. Decide how successful sandbox outcomes become pack candidates, anti-patterns, or truth-pack extensions.
4. Reintroduce deterministic scene assembly and first-render correctness checks only after the loop is trustworthy.
5. Continue shrinking prompt-based policy and coordinator ambiguity where deterministic seams are now obvious.

Exit criteria:
- at least one weak or novel feature is promoted through the sandbox path
- scene-fit / contact / layout quality work is happening on a trustworthy iteration loop
- the system improves both breadth and quality without growing another brittle coordinator layer

Proof obligations:
- at least one capability is operationalized through the sandbox path rather than a full-scene guess
- scene assembly work lands after, not before, the evidence plane is reliable

## Cross-Cutting Workstreams

These do not replace the phase order above. They are supporting work that should be advanced alongside the relevant phase.

### A. Benchmarks and Regression Coverage

Add and maintain:

- state-authority regression tests
- repair-mode classification benchmark
- structural-repair regression suite
- cross-physics reliability baseline
- evaluator calibration corpus and score-band tracking
- manifest integrity test

### C. Pipeline-Level Tracing

The SDK `TracingProcessor` only captures agent/tool/generation spans. Pipeline-level decisions (repair intent, truth pack fixes, execution outcomes, phase transitions, stuck-state changes) must also be emitted to the trace JSONL so that:

- E2E test verification checks work against trace data rather than stdout parsing
- Phase exit criteria and proof obligations can be verified from persisted traces
- The manifest integrity test (Phase 4) has a single source of truth

### B. Documentation and Runtime Truth Freshness

Keep these aligned as changes land:

- `CURRENT_STATE.md`
- `CURRENT_ROADMAP.md`
- `VERSION_TRUTH.md`
- `AI_OPERATION_MANUAL.md`
- `SDK_ENFORCEMENT_PROTOCOL.md`
- artifact/runtime version stamps such as `ArtifactManager.ManifestArtifact`

Rule:
- when a roadmap phase materially changes runtime behavior, update the relevant authority docs in the same change rather than leaving the truth trapped in a review or worklog

## Deferred or Rejected for Now

These ideas are not discarded forever, but they are not part of the current canonical roadmap:

- full line-by-line merge of the source reviews
- new broad coordinator layers as a substitute for deterministic routing
- raising `max_iterations` as a fix for structural repair gaps
- multi-grader ensembles before evaluator calibration and repair proof
- heavy branching/session infrastructure before the simpler authoritative-loop fixes land

## Source Mapping

Use this as the quick answer for “which document do I trust for what?”

| Topic | Preferred Source |
|------|-------------------|
| split state authority | Opus |
| `record_iteration()` bypass | Opus |
| `escape_level` overwrite / routing input poisoning | Opus |
| typed QA issue enforcement | Both |
| section patching proof | Opus |
| experiment ledger / provenance-safe learning | GPT-5.4 |
| evaluator calibration ladder | Both |
| autonomy promotion contract | GPT-5.4 |
| capability incubation / sandbox mode | GPT-5.4 |
| long-horizon autonomy framing | GPT-5.4 |

## Working Rule

When a future review produces a better idea:

- do not treat the review itself as the new roadmap
- update this file
- update `CURRENT_STATE.md` if the active priorities changed
- keep the review as supporting evidence only
