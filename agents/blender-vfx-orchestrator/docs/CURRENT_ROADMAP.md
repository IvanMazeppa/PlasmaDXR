# Current Roadmap

Status: authoritative
Last verified: 2026-03-15
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

| Phase | Goal | Main Source |
|------|------|-------------|
| 1 | Fix state authority and repair-routing inputs | Opus |
| 2 | Make typed QA-to-repair signals authoritative | Opus + GPT-5.4 |
| 3 | Prove structural repair in live runs | Opus |
| 4 | Make learning/evaluation evidence trustworthy | GPT-5.4 + Opus |
| 5 | Gate autonomy with evidence and integrate hybrid HITL | GPT-5.4 |
| 6 | Expand capability acquisition and raise the quality ceiling | GPT-5.4 + Opus |

## Phase 1: Fix State Authority and Routing Inputs

Goal:
- make the deterministic state machinery actually drive live repair routing

Why this is first:
- every later phase depends on truthful `same_issue_count`, `plateau_count`, and `escape_level`

Work:
- route iteration completion through `SessionState.record_iteration()`
- stop manual `session.iterations.append(...)` in the orchestrator hot path
- stop the Quality Gate from overwriting deterministic `escape_level`
- feed deterministic `plateau_count`, `same_issue_count`, and `escape_level` into `choose_repair_intent()`
- expose the correct stuck-state values to any prompt context that still needs them

Exit criteria:
- `StuckDetectionState.update_from_iteration()` fires every iteration
- `plateau_count` and `same_issue_count` change on real runs
- a plateau or repeated-issue case can change repair behavior without LLM guesswork

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

Exit criteria:
- structural defects are routinely tagged as structural
- parameter-only issues are routinely tagged as parameter
- repair-mode classification matches human labels at a useful rate on a small regression set

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

Suggested first cases:
- wrong stripe order
- missing collider
- camera inside geometry
- wrong attachment topology
- missing key light

Exit criteria:
- at least one structural defect reaches `modify_code` without manual forcing
- section patching preserves script quality better than full rewrite on at least one real case
- structural failures no longer get trapped in parameter loops by default

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

Exit criteria:
- manifest, scorecard, session state, and trace summary agree on core iteration facts
- evaluator score bands are at least directionally sane for known-good and known-bad samples
- learning promotion can be audited without depending on an LLM reconstruction

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

Exit criteria:
- every effect family can be classified as guided, assisted, or autonomous with evidence
- hybrid HITL is in place for the right boundaries
- “autonomous” is no longer shorthand for “fewer checkpoints”

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

Exit criteria:
- at least one weak or novel feature is promoted through the sandbox path
- scene-fit / contact / layout quality work is happening on a trustworthy iteration loop
- the system improves both breadth and quality without growing another brittle coordinator layer

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
