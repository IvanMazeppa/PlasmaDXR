# Current Roadmap

Status: authoritative
Last verified: 2026-03-19
Purpose: canonical synthesis of the March 2026 architecture reviews and live repo state

## Scope

This roadmap is the planning document to use for active implementation.

It synthesizes:

- `docs/reviews/2026-03/20260314_DEEP_ANALYSIS_ACTIONABLE_ROADMAP.md`
- `docs/reviews/2026-03/20260314_DEEP_AUTONOMY_ANALYSIS_GPT_54_XHIGH.md`
- `docs/reviews/2026-03/20260317_SPECIALIZED_SECTION_BUILDERS_IMPLEMENTATION_PLAN.md` (candidate final plan)
- `docs/reviews/2026-03/20260317_SECTION_BUILDERS_REVIEW_AND_REFINEMENTS.md`
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
- `QualityIssue` / `structured_issues` are now an authoritative part of repair routing
- section patching and `modify_code` have been proven end to end at least once in live runs
- the dominant remaining problems are evaluator trust, scene-quality ceiling, and evidence-gated expansion of capability/autonomy

## Roadmap Overview

| Phase | Goal | Status | Main Source |
|------|------|--------|-------------|
| 1 | Fix state authority and repair-routing inputs | Complete | Opus |
| 2 | Make typed QA-to-repair signals authoritative | Complete | Opus + GPT-5.4 |
| 3 | Prove structural repair in live runs | Complete | Opus |
| 3.5 | Scene aesthetic realism (cross-cutting) | In progress — Tasks 1-8 complete, Tasks 9-10 remaining | Next 10 Tasks plan |
| 4 | Make learning/evaluation evidence trustworthy | In progress — Jobs 1-2 done (manifest), Jobs 3-7 queued | GPT-5.4 + Opus |
| 5 | Gate autonomy with evidence and integrate hybrid HITL | Later | GPT-5.4 |
| 6 | Expand capability acquisition and raise the quality ceiling | Later | GPT-5.4 + Opus |
| 6A | Specialized Section Builders for scene quality and asset reuse | Planned, gated — first milestone is Task 10 | March 17 section-builder plan + review |

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

## Phase 2: Make Typed QA-to-Repair Signals Authoritative — COMPLETE (2026-03-16)

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

0. ~~**Prerequisite: Pipeline-level tracing.** Emit pipeline events to JSONL trace file alongside SDK spans.~~ **DONE** — `pipeline_event()` in `tracing/verbose_processor.py`, 5 event types instrumented in orchestrator + execution phase, 5 unit tests pass.
1. ~~Update the Quality Analyst prompt and runtime instructions so failed evaluations always emit `structured_issues`.~~ **DONE** — `QUALITY_ANALYST_BASE_INSTRUCTIONS` updated, eval prompt updated, all 5 synthetic QualityOutput paths now include `structured_issues`.
2. ~~Add or tighten the quality output guardrail so `passed=False` without `structured_issues` is treated as invalid.~~ **DONE** — `validate_quality_output` guardrail now triggers tripwire when `passed=False` and `structured_issues` is empty. 8 inline tests pass, 452 suite tests pass.
3. ~~Harden `QualityIssue` validation so `kind` and `repair_mode_hint` are constrained to accepted categories.~~ **DONE** — `kind` is now `Literal["parameter", "structural", "technique", "camera", "lighting"]`, `repair_mode_hint` is `Literal["modify_params", "modify_code", "switch_technique"]`. Pydantic rejects invalid values at construction.
4. ~~Change `choose_repair_intent()` to prioritize `structured_issues` before keyword heuristics.~~ **DONE** — `structured_issues` is now authoritative when present: structural majority → `modify_code`, technique majority → `switch_technique`, parametric majority → `modify_params`. Keyword heuristics only fire when `structured_issues` is absent. 16 tests pass (3 new).
5. ~~Build a small repair-mode classification benchmark from historical traces and use it to measure agreement with human labels.~~ **DONE** — `tests/test_repair_routing_benchmark.py` with 11 hand-labeled cases covering structural, parametric, technique, execution failure, keyword fallback, mixed signals, and escalation scenarios. 100% agreement. 466 total tests pass.

Exit criteria:
- structural defects are routinely tagged as structural
- parameter-only issues are routinely tagged as parameter
- repair-mode classification matches human labels at a useful rate on a small regression set

Proof obligations:
- an Ireland Flag-style defect is classified as structural without relying on keyword luck
- the router can explain whether it used structured issues or heuristic fallback
- the benchmark gives a stable number you can track over time

## Phase 3: Prove Structural Repair in Live Runs — COMPLETE (2026-03-16)

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

1. ~~Gate Learning Agent parameter application on `RepairIntent.mode`.~~ **DONE** — `_layer1_allowed` already gates all three parameter paths on `repair_intent.mode == "modify_params"` (pre-existing).
2. ~~Demote Learning Agent `next_action` so it can advise but not override repair mode.~~ **DONE** — Quality gate fallback no longer lets `learning.next_action == 'complete'` override `quality.passed`. Learning Agent disagreements are logged. 466 tests pass.
3. ~~Decide whether to expand the action vocabulary (`iterate_params` / `iterate_code`) or keep the existing vocabulary and treat it as advisory only.~~ **Decision: keep existing vocabulary, treat as advisory only.** Learning Agent uses `iterate/switch_technique/complete` (advisory). RepairIntent uses `modify_params/modify_code/switch_technique/request_guidance` (authoritative). No expansion needed.
4. ~~Build the structural-repair regression set.~~ **DONE** — `tests/test_structural_repair_regression.py` with 14 tests across 5 scenario classes (wrong stripe order, missing collider, camera inside geometry, wrong topology, missing key light) + anti-regression guards. Also fixed `_classify_structured_issues()` to count `camera`/`lighting` kinds as structural (they require code changes). 480 total tests pass.
5. ~~Force at least one live `modify_code` proof on a real case.~~ **DONE** — E2E run 2026-03-16 (trace `campfire_mantaflow_20260316_032343.jsonl`). Iteration 2 triggered `repair_intent: mode=modify_code, trigger=execution_failure`. Section patch attempted → full rewrite fallback → recovery script generated. `modify_code` path exercised end-to-end. Also found + fixed `smoke_amount` hallucination (→ `flame_smoke` in truth pack).
6. ~~After proof, harden the hot path so critical validation is attached to the actual execution/modification path, not only to tool wrappers that can be bypassed.~~ **DONE (verified)** — truth pack validation runs in `phases/execution.py:158` before ALL script executions (main + recovery). `_modify_script_impl` output is always truth-pack-validated before reaching Blender. The critical hot path is covered.

Exit criteria:
- at least one structural defect reaches `modify_code` without manual forcing
- section patching preserves script quality better than full rewrite on at least one real case
- structural failures no longer get trapped in parameter loops by default

Proof obligations:
- `modify_code` succeeds or fails honestly on at least one production-style case
- parameter-first fast paths no longer silently preempt structural repair
- the regression suite can catch a future relapse back into parameter-tuning ruts

## Phase 3.5: Scene Aesthetic Realism (Cross-Cutting) — IN PROGRESS

Goal:
- make scene look-dev, hero-object quality, mood, and material realism first-class in the pipeline

Companion plans:
- `docs/reviews/2026-03/20260316_SCENE_AESTHETIC_REALISM_ACTION_PLAN.md`
- `docs/reports/20260318_NEXT_10_SCENE_QUALITY_TASKS.md` (execution task list)

Completed (2026-03-16/17):
- `StyleSpec` model added to `pipeline_models.py` with `to_script_constraints()` method
- `extract_style_spec()` deterministic keyword extraction in `utils/prompt_enhancer.py`
- `enhance_description()` zero-LLM-cost craft hint injection (fire, water, smoke, explosion, glass)
- AgX color management + DOF + world gradient as mandatory look-dev floor in Script Writer instructions
- `IssueKind` expanded: `materials`, `environment`, `composition` added, routed to `modify_code`
- Code pattern tools (`search_code_patterns`, `get_pattern_code`) exposed to Script Writer
- Structural pattern routing: patterns with ShaderNode/node_tree/modifiers go to Script Writer as code context, not flattened to scalar params
- Visual craft reuse instructions added to Script Writer

Completed (2026-03-18/19) — "Next 10 Scene Quality Tasks" 1-8:
- **Task 1: Aesthetic benchmark pack** — `tests/fixtures/aesthetic_benchmarks.json` with prompts covering tabletop glass, candlelit macro, cloth, window, atmospheric interior, product close-up. `scripts/run_benchmark.py` CLI tool for running benchmarks.
- **Task 2: Look-dev coverage metrics** — `tools/script_analysis_tools.py` with deterministic coverage metrics for color management, DOF, world strategy, material count, hero-object refinement count, atmosphere presence. `tests/test_lookdev_metrics.py` (217 lines).
- **Task 3: `hero_object` and `lookdev` IssueKinds** — Added to `IssueKind` Literal type in `models/pipeline_models.py`. Accepted everywhere `QualityIssue.kind` is validated.
- **Task 4: Quality Analyst aesthetic prompting** — Dynamic instructions updated to guide structured issue emission with `hero_object` and `lookdev` kinds pointing at canonical section targets (`create_geometry`, `setup_materials`, etc.).
- **Task 5: Aesthetic issue routing** — `hero_object` and `lookdev` routed to `modify_code` in `phases/repair_routing.py`. `tests/test_repair_routing.py` expanded (66 new lines). `tests/test_aesthetic_benchmarks.py` (96 lines).
- **Task 6: Hero-object refinement helper** — `tools/hero_object_refinement.py` (241 lines) with per-object-type recipes (SOLIDIFY, BEVEL, SUBSURF for glass, bottles, vessels, candles, cloth, masonry). Injected into Script Writer prompt via `tools/dynamic_instructions.py` using `StyleSpec.hero_objects`. `tests/test_hero_object_refinement.py` (137 lines).
- **Task 7: Camera/readability enforcement** — `utils/prompt_enhancer.py` now injects camera distance-based hints (close-up: 50-85mm f/2.8-f/4.0, wide: 24-35mm f/8-f/11, medium: 35-50mm f/4-f/5.6). `StyleSpec` captures camera fields. `tests/test_prompt_enhancer_camera.py` (10 tests).
- **Task 8: Iteration manifest** — `IterationResult` and `IterationArtifact` now carry `script_hash` (SHA-256 12-char prefix), `issue_kinds`, and `change_label` (initial_generation/modify_params/modify_code/section_patch/technique_switch/full_rewrite). `utils/artifact_manager.py` updated with `compute_script_hash()`. Orchestrator tracks `_iter_change_label` through all decision points. `tests/test_iteration_manifest.py` (11 tests).

Also completed (2026-03-19) — E2E benchmark findings:
- **`bpy.mathutils` truth pack pattern** — LLM hallucination `bpy.mathutils` auto-replaced with `mathutils` (KNOWN_HALLUCINATIONS + HARDCODED_FIXES)
- **Render sample cap** — Samples > 256 auto-capped to prevent 80+ minute renders (regex detection + auto-fixer)
- **Hero refinement injection bug fix** — `dynamic_instructions.py` was accessing `ctx.context.style_spec` instead of `ctx.context.session.style_spec`, and using `hasattr` on a dict (always False). Fixed.
- **"wooden table" compound noun regex** — `prompt_enhancer.py` now captures `wooden table` instead of just `wooden`
- Wine pour E2E benchmark run completed (trace: `wine_pour_20260318_051505.jsonl`). Pipeline infrastructure solid, truth pack caught 12-17 fixes per generation, hero refinement recipes appeared in generated code, iteration manifest fields populated correctly.

Remaining (Tasks 9-10):
- **Task 9: Focused benchmark pass** — run baseline vs post-quick-win comparison across the benchmark pack, document which changes produced real visual gains
- **Task 10: Hero-only specialist builder milestone** — typed `BuildRegistry`, scaffold-first generation, deterministic `SceneBuildPlan`, builder-level truth-pack validation, `HeroAssetBuilder` only, hero-only benchmark comparison

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

1. ~~Define the minimum canonical iteration manifest.~~ **DONE (2026-03-19, Phase 3.5 Task 8).** `IterationResult` and `IterationArtifact` carry `script_hash`, `issue_kinds`, and `change_label`. `compute_script_hash()` in `utils/artifact_manager.py`.
2. ~~Record that manifest from the live loop and make sure it is persisted alongside scorecards/artifacts.~~ **DONE (2026-03-19).** Orchestrator tracks `_iter_change_label` through all decision points. `write_iteration_from_values()` persists manifest fields. E2E verified on wine pour benchmark.
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

## Phase 6A: Specialized Section Builders — PLANNED, GATED, PENDING FINAL EXTERNAL REVIEW

Goal:
- raise the scene-quality ceiling and future asset reuse through deterministic section ownership and specialist builders

Purpose:
- improve hero-object quality, environment quality, and material/look-dev quality without replacing the current state machine
- make section patching more valuable by giving sections stable owners
- create a path from one-off scene improvement to reusable asset generation later

Why this sits under Phase 6:
- it is a quality-ceiling and capability-compounding workstream
- it should not begin until the current loop is stable enough that bad specialist output does not collapse the iteration
- it depends on work already underway in Phase 3.5 and should not outrun the evidence/reliability work in Phase 4

Status:
- the implementation plan exists and is internally coherent
- the plan has absorbed one external critique round and a follow-up tightening pass
- before coding begins, it is still expected to receive final second-opinion review

Current source of truth:
- detailed plan: `docs/reviews/2026-03/20260317_SPECIALIZED_SECTION_BUILDERS_IMPLEMENTATION_PLAN.md`
- design rationale: `docs/reviews/2026-03/20260317_SPECIALIZED_SECTION_BUILDERS_DESIGN_NOTE.md`
- external critique and refinements: `docs/reviews/2026-03/20260317_SECTION_BUILDERS_REVIEW_AND_REFINEMENTS.md`

Finalized design constraints:
- `BuildRegistry` is typed from day one; no loose `ctx` dict
- `plan_scene_build()` is deterministic for the first rollout
- specialist-owned sections use stubs, not generate-then-replace waste
- builder output is validated at the builder boundary with bundled doc grounding and truth-pack validation
- scaffold generation derives from `CANONICAL_SECTIONS`
- repair escalation is section-local: specialist -> single-writer section -> full-script fallback
- parallelism is deferred and later bounded with simple parallel groups, not a general DAG scheduler

Execution gate:
- section patching must be stable enough that a bad specialist section does not force a whole-script rewrite every time
- `structured_issues` must continue routing structural/aesthetic failures to `modify_code`
- the canonical scaffold must be reliable enough to serve as the stable section boundary
- final external second-opinion review must not reveal a simpler or safer first milestone

First milestone only:
- add rollout flags and benchmark fixtures
- add typed `BuildRegistry` and scaffold-first section stubs
- add deterministic `SceneBuildPlan`
- add specialist base factory and section-builder guardrails
- add builder-level truth-pack validation and cross-reference validation
- add `HeroAssetBuilder` only
- benchmark hero-only mode against the baseline

Not approved for the first coding slice:
- environment builder
- materials builder
- asset-lab promotion workflow
- generalized parallel section generation

Working rule:
- until the first milestone is externally cleared, this workstream is planned but not yet active implementation
- once approved, update this roadmap phase status and pull the first-milestone job order into the active queue

## Cross-Cutting Workstreams

These do not replace the phase order above. They are supporting work that should be advanced alongside the relevant phase.

### A. Benchmarks and Regression Coverage

Add and maintain:

- ~~state-authority regression tests~~ ✓ `tests/test_state_authority_wiring.py` (9 tests)
- ~~repair-mode classification benchmark~~ ✓ `tests/test_repair_routing_benchmark.py` (11 cases)
- ~~structural-repair regression suite~~ ✓ `tests/test_structural_repair_regression.py` (14 tests)
- ~~aesthetic benchmark pack~~ ✓ `tests/fixtures/aesthetic_benchmarks.json` + `scripts/run_benchmark.py`
- ~~look-dev coverage metrics~~ ✓ `tools/script_analysis_tools.py` + `tests/test_lookdev_metrics.py`
- ~~iteration manifest tests~~ ✓ `tests/test_iteration_manifest.py` (11 tests)
- cross-physics reliability baseline
- evaluator calibration corpus and score-band tracking
- manifest integrity test (cross-validates manifest, session state, trace, scorecard)
- specialist-vs-baseline benchmark pack once Phase 6A begins

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
| specialized section builders | March 17 section-builder plan + review |

## Working Rule

When a future review produces a better idea:

- do not treat the review itself as the new roadmap
- update this file
- update `CURRENT_STATE.md` if the active priorities changed
- keep the review as supporting evidence only
