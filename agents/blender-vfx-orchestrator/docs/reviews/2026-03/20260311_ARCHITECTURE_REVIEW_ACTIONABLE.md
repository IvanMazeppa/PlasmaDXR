# Actionable Architecture Review - Blender VFX Orchestrator

Date: 2026-03-11

This document supersedes the earlier review drafts by incorporating:
- the original repo-grounded architecture review,
- the first feedback cycle,
- the revised feedback in `docs/ARCHITECTURE_REVIEW_REVISED_FEEDBACK_20260311.md`,
- and the first successful Cell Fracture execution/render produced under `TechniqueContract`.

## 1. EXECUTIVE SUMMARY

The architecture review has crossed an important threshold: the first successful Cell Fracture run proves that the core thesis was correct. The main historical failure was not "the LLM is too weak," but "the system kept shouting guidance through prompts instead of binding behavior through contracts." `TechniqueContract`, contract adherence checks, and the capability-pack registry have now demonstrated that binding works. That changes the roadmap materially: the priority is no longer proving the contract architecture, but exploiting it fast enough to reduce the still-high execution failure rate, which remains the largest failure bucket. The most concrete path is a 3-wave plan: Wave 1 expands capability coverage, hardens context/research boundaries, and rolls GPT-5.4 out now for code-critical agents; Wave 2 makes scripts patchable and the pipeline easier to evolve; Wave 3 adds deterministic scene assembly and evidence-backed learning once execution reliability is no longer the primary bottleneck.

## 2. ROOT CAUSE ANALYSIS

1. Symptom: historical F1 Technique Monotony and F2 Script Writer Ignores Technique Research. Root cause: technique choice used to be advisory prose, so the writer could silently substitute its priors. Evidence: the original failure path was visible in `orchestrator.py:1325` and `orchestrator.py:1927`; the recent Cell Fracture success with `CONTRACT ADHERENCE: PASSED` is the first proof that a binding contract fixes this class of problem rather than just reducing it statistically.

2. Symptom: F3 Recovery Scripts Lose Quality, F10 headless/context-sensitive Blender failures, and much of F11 Correction Complexity. Root cause: the system still evolves scripts destructively through full rewrites and ad hoc correction layers instead of stable section-level patches against a last-known-good baseline. Evidence: execution recovery still rewrites the complete script in `phases/execution.py:274`, recovery still relaxes doc-query enforcement in `hooks/enforcement_hooks.py:737`, and structure-preserving patchability is not yet enforced in generated scripts.

3. Symptom: F6 Research Agent Exhausts Turns, F7 Context Bloat, and ongoing cost inefficiency. Root cause: the pipeline still relies on open-ended agent conversations and oversized downstream inputs where bounded retrieval/synthesis and per-phase context filtering should exist. Evidence: the research flow is still effectively conversational, `call_model_input_filter` is still the biggest unused leverage point, and the feedback cycle confirms that context boundary issues are degrading both cost and output quality.

4. Symptom: F8 Orchestrator Monolith and high change risk. Root cause: stable phase boundaries have existed mostly on paper rather than in modules, so the orchestrator still accumulates control logic that should live next to the contract it enforces. Evidence: the prior review already identified this in the 2,980-line control loop, and the revised feedback correctly argues that extraction should happen continuously as each contract stabilizes, not as a late cleanup step.

5. Symptom: F4 Scene Geometry Does Not Fit Together and the remaining low-quality ceiling after execution succeeds. Root cause: the system still lacks a deterministic spatial assembly layer, but this is now a secondary bottleneck because spatial quality only matters on runs that already execute and render. Evidence: floating/contact-gap failures remain real, but current trace analysis still puts execution failure at roughly 45% of runs and low visual quality at roughly 35%, so geometry solving is important but not Wave 1 work.

6. Symptom: capability-pack success on Cell Fracture does not yet generalize to broad user coverage. Root cause: the current pack set is still too small, and the project has not yet defined a concrete minimum viable capability surface. Evidence: the current registry has four packs (`cell_fracture_rigid_body`, `mantaflow_fire`, `mantaflow_liquid`, `simple_rigid_body`), but cloth, particles, geometry nodes, and combination effects still lack a scoped plan.

## 3. ARCHITECTURAL INTERVENTIONS

1. What changes: treat `TechniqueContract` as the permanent binding layer and explicitly stop revisiting API-call-level `APISpec` generation. Keep the contract focused on technique, required capabilities, object inventory, semantic relationships, and quality goals, while leaving code structure inside named sections creative. Why it fixes the root cause: it preserves the proven part of the new architecture and avoids regressing to the over-constrained, brittle spec design that previously failed. Estimated complexity: Low-Medium, because the core mechanism already exists.

2. What changes: expand the capability-pack registry to an MVP set of seven packs: the current four plus `particles_core`, `cloth_softbody`, and `geometry_nodes_environment`; treat combination effects as pack composition, not standalone packs, in v1. Why it fixes the root cause: this gives Phase 2 a concrete definition of done and should cover most common request classes without exploding the pack surface. Estimated complexity: Medium.

3. What changes: roll GPT-5.4 out now for code-critical agents: Script Writer / Code Realizer, research, and modification strategy. Keep non-critical or arithmetic-heavy stages on cheaper models or deterministic code paths. Why it fixes the root cause: the contract architecture is now present, so better instruction following and coding ability will amplify a real boundary instead of trying to compensate for a missing one. Estimated complexity: Medium.

4. What changes: harden context and research boundaries immediately. Add `call_model_input_filter`, turn research into a three-stage plan -> bounded retrieval -> synthesis flow, use `tool_use_behavior="stop_on_first_tool"` on helper flows, and keep large historical narrative out of generation/modification. Why it fixes the root cause: this is the highest-leverage direct fix for F6/F7 and should reduce both cost and output degradation quickly. Estimated complexity: Medium.

5. What changes: standardize generated scripts around named functions such as `setup_scene`, `create_geometry`, `setup_materials`, `setup_physics`, `setup_lighting`, `setup_camera`, and `bake_and_render`, then use section-level patching with explicit branch budgets. Adopt the policy: max 2 repair branches per iteration, max 4 per session, then escalate to regeneration, technique switch, or HITL. Why it fixes the root cause: it makes recovery concrete, bounded, and cost-aware instead of destructive and open-ended. Estimated complexity: Medium.

6. What changes: extract orchestrator modules continuously as each contract stabilizes, and simplify the agent roster around what is computable. Technique selection should become deterministic when capability evidence is sufficient; Quality Gate should be mostly deterministic; Modification Strategy should become deterministic-plus-LLM hybrid for genuinely novel changes. Why it fixes the root cause: it reduces monolith risk without waiting for a future rewrite and keeps LLM judgment only where it adds value. Estimated complexity: Medium.

7. What changes: postpone deterministic scene assembly until execution failure is no longer the dominant bottleneck, but define its v1 scope now: tabletop/contact scenes first, then container liquids, then room interiors. Why it fixes the root cause: it keeps the roadmap honest about ROI while still preserving a concrete future target for F4 and the remaining quality ceiling. Estimated complexity: Medium.

8. What changes: calibrate the evaluator against known-good historical assets and wire learning to contracts, packs, and section-level patterns rather than prompt text. Why it fixes the root cause: it ensures future optimization is measured against a stable target and prevents the system from rebuilding prompt sprawl in a new form. Estimated complexity: Low-Medium.

## 4. PROPOSED ROADMAP

### Wave 1 - Exploit the Contract Architecture Now ✅ (2026-03-12)

Content: keep `TechniqueContract` as the binding core, expand the registry from 4 packs to the 7-pack MVP, harden research into bounded retrieval/synthesis, add `call_model_input_filter`, and roll GPT-5.4 out for code-critical agents immediately. Dependencies: `TechniqueContract`, adherence checking, and the current registry are already in place. Exit criteria: Cell Fracture remains stable, MVP pack registry exists, research no longer burns turns conversationally, and execution failure drops materially from the current ~45% baseline.

**Completed items:**
- [x] 7-pack MVP registry: `cell_fracture_rigid_body`, `mantaflow_fire`, `mantaflow_liquid`, `simple_rigid_body`, `particles_core`, `cloth_softbody`, `geometry_nodes_environment` — 33 tests pass
- [x] `call_model_input_filter` implemented: turn-based context filtering per agent, handles reasoning model item groups (GPT-5.4/o3 safe) — 29 tests pass
- [x] GPT-5.4 rollout for code-critical agents via `codex_upgrade` preset: Script Writer, Research, Modification Coordinator, Quality Analyst
- [x] Research guardrail updated: accepts manual physics doc refs as sufficient grounding (cloth/particles/geometry_nodes have sparse API coverage in vector store)
- [x] Truth pack type resolution: added `cloth`, `geometry_nodes`, `particle_system` entries to `TECHNIQUE_TYPES`, 15+ aliases, substring alias matching for LLM-generated technique names
- [x] `EffectType` enum expanded: CLOTH, FLAG, FABRIC, PARTICLES, SPARKS, SNOW, DUST, ENVIRONMENT, PROCEDURAL
- [x] Artifact gates updated: skip Mantaflow cache check for cloth/particle/geometry_nodes techniques
- [x] Keyword routing fixed: ordered list (not dict) prevents substring collisions (e.g., "rain" in "terrain")
- [x] Contract adherence checking proven across glass shatter (Cell Fracture) and cloth simulation

**Remaining (deferred to future work):**
- [ ] Research bounded retrieval/synthesis (plan→retrieve→synthesize flow) — functional but still conversational
- [ ] Execution failure rate measurement post-Wave-1

### Wave 2 - Make Recovery Cheap, Precise, and Maintainable 🔄 (In Progress)

Content: enforce named script sections, implement section-level patching with `AdvancedSQLiteSession` branch caps, extract stabilized modules out of the orchestrator as they mature, and calibrate the evaluator on known-good assets such as the wine-pour baseline. Dependencies: Wave 1 must stabilize the active contract, pack, and context boundaries first. Exit criteria: new runs are patchable by section name, repair follows the 2-branch-per-iteration policy, evaluator drift is understood, and the monolith is shrinking continuously rather than growing.

**Implementation plan:** `docs/superpowers/plans/2026-03-12-script-section-patching.md`

**Wave 2A — Section Patching ✅ (2026-03-12, 6 commits):**
- [x] AST-based section parser (`utils/script_sections.py`) — 16 unit tests
- [x] `patch_script_section` tool — 5 unit tests
- [x] Section naming guardrail (warning-only) — 4 tests, section detection in `write_script`
- [x] Patch budget tracking in `SessionState` (`sections_found` in `ScriptOutput`)
- [x] Rewired `modify_code` to use section patching with budget (2/iter, 4/session)
- [x] Canonical section structure guidance in Script Writer generation prompt

**Wave 2A.1 — Stale Render Reuse Fix ✅ (2026-03-15, P0):**
- [x] `_discover_render()` replaced with `_discover_render_current_run()` — scoped to `run_dir` only, never scans shared per-asset dir
- [x] `_apply_render_discovery()` rewritten — never promotes `success=False` to `success=True`
- [x] `partial_render_path` field on `ExecutionOutput` for diagnostic-only render tracking
- [x] 12 unit tests covering core bug scenario (stale render in shared dir not used)

**Wave 2A.2 — RepairIntent Deterministic Routing ✅ (2026-03-15):**
- [x] `QualityIssue` model with typed kind/repair_mode_hint
- [x] `RepairIntent` model for deterministic routing decisions
- [x] `phases/repair_routing.py` — `choose_repair_intent()` priority chain: execution_failure → structural keywords → plateau → same_issue → escape_level → default
- [x] QA prompt updated for `structured_issues` output
- [x] Orchestrator modification cascade rewired: RepairIntent gates modify_params vs modify_code vs switch_technique
- [x] 13 unit tests including Ireland Flag scenario (structural issues → modify_code by iter 2)

**Wave 2B — HITL Framework ✅ (2026-03-15):**
- [x] `utils/hitl_handler.py` — `HITLHandler` with 5 checkpoint types (stall, budget, critical, escalation, quality_plateau)
- [x] 5 autonomy levels: Guided (0) → Supervised (1) → Semi-autonomous (2) → Autonomous (3) → Full autonomous (4)
- [x] Interactive mode (CLI input() prompt) and non-interactive mode (saves checkpoint to session, sets PAUSED)
- [x] Pending checkpoint resume on session reload
- [x] Config integration: `AgentConfigManager.use_hitl()`, `get_hitl_autonomy_level()`, `is_hitl_interactive()`
- [x] Wired into orchestrator: Phase 3.95 (post-eval), escape L4 (escalation), session resume (pending)
- [x] 26 unit tests: checkpoint models, autonomy gating, budget/stall/critical triggers, interactive prompt mocking, config integration

**Wave 2 — Truth Pack Hardening ✅ (cumulative):**
- [x] `Fac` → `Factor` context-aware auto-fixer (mix nodes fixed, ColorRamp preserved)
- [x] `use_dissolve` → `use_dissolve_smoke` substring bug fixed (removed from HARDCODED_FIXES, regex handles it)
- [x] 25+ truth pack patterns total with auto-fix coverage

**Wave 2 — SDK Upgrade ✅ (2026-03-13):**
- [x] Upgraded from v0.10.5 to v0.12.0
- [x] `needs_approval` on `@function_tool` confirmed working (hybrid HITL architecture validated)
- [x] VERSION_TRUTH.md updated with corrected HITL information

**Wave 2 — Remaining Items:**
| Item | Status | Priority |
|------|--------|----------|
| Evaluator calibration | NOT STARTED | P2 |
| Orchestrator extraction | NOT STARTED | P3 (monolith stable, not growing) |
| Bounded research retrieval | NOT STARTED | P2 (reclassified from Wave 1, now Wave 2C) |
| Mantaflow re-baseline | IN PROGRESS | P2 (campfire E2E test running) |
| Native SDK `needs_approval` on high-impact tools | NOT STARTED | P2 (switch_technique, increase_budget) |

**Test suite:** 438 tests pass, 0 failures (as of 2026-03-15).

### Wave 3 - Raise the Quality Ceiling After Reliability Improves

Content: add deterministic scene assembly for tabletop/contact and container scenes first, then room interiors; move learning from prompt accumulation to evidence-backed deltas against contracts and packs; consider addon automation only if pack count or deployment demands it. Dependencies: Wave 2 should reduce execution failure to below 20% so geometry quality becomes a high-ROI target. Exit criteria: contact-gap failures are meaningfully reduced on targeted scene families, learning is writing to structured artifacts, and the architecture supports wrapped-to-taught graduation with evidence rather than opinion.

## 5. SDK FEATURES TO LEVERAGE

1. `call_model_input_filter`: ✅ Implemented (Wave 1). Turn-based per-agent context filtering in `utils/context_filter.py`.

2. `AdvancedSQLiteSession`: Deferred. Basic `SQLiteSession` satisfies `Session` protocol in v0.12.0 — use for token tracking. Branch budgets implemented via `SessionState.patch_budget` instead.

3. `tool_input_guardrails` and `tool_output_guardrails`: ✅ Implemented. Truth pack input guardrail on `execute_blender_script`, critical failure output guardrail on `evaluate_render`/`analyze_with_vision`, script length output guardrail on `generate_script`, section naming guardrail (warning-only) on `write_script`.

4. `tool_use_behavior="stop_on_first_tool"`: Not yet applied. Planned for Wave 2C bounded research.

5. `parallel_tool_calls`: use it inside the bounded retrieval stage, not in open-ended agent loops.

6. `is_enabled` and GPT-5.4 `allowed_tools`: gate packs and tools by addon availability, runtime state, and current phase.

7. GPT-5.4 rollout now: ✅ Implemented (Wave 1). `codex_upgrade` preset assigns gpt-5.4 to Script Writer, Research, Modification Coordinator, Quality Analyst.

8. `needs_approval`: ✅ Confirmed available on `@function_tool` in SDK v0.12.0 (not MCP-only). Hybrid HITL architecture: pipeline-level `HITLHandler` for session checkpoints + native `needs_approval` for tool-level approvals (switch_technique, increase_budget). Pipeline-level HITL implemented; native SDK approvals planned.

9. Retry policies (v0.12.0): `ModelSettings` retry configuration available. Not yet applied — planned for `codex_upgrade` preset.

## 6. WHAT TO STOP DOING

1. ~~Stop treating the first contract success as proof that the rest of the system can stay unchanged.~~ ✅ Contracts proven across 4 physics types (Cell Fracture, Mantaflow, cloth, destruction).

2. ~~Stop postponing GPT-5.4 for code-critical agents now that the binding contract exists.~~ ✅ Rolled out via `codex_upgrade` preset.

3. ~~Stop growing the capability-pack registry without defining an MVP target and completion criteria.~~ ✅ 7-pack MVP defined and implemented.

4. Stop using open-ended research conversations when the desired flow is already known and bounded. *(Wave 2C)*

5. ~~Stop recovering from execution failure by rewriting entire scripts.~~ ✅ Section patching with patch budget (2/iter, 4/session). RepairIntent routes to modify_code instead of full rewrite.

6. ~~Stop letting modification strategy remain fully free-form when many common corrections are computable.~~ ✅ RepairIntent deterministic routing replaces free-form Modification Coordinator as first pass.

7. Stop deferring orchestrator extraction to a future cleanup phase. *(P3, monolith stable)*

8. ~~Stop spending Wave 1 effort on deterministic spatial solving while execution failure is still the dominant bottleneck.~~ ✅ Correctly deferred to Wave 3.

9. Stop storing operational truth in overlapping prompts, fixers, guardrails, and validators once the contract/pack path exists. *(Ongoing — truth pack consolidation helps)*

10. Stop assuming combination effects need dedicated packs before pack composition has been tried. *(Not yet relevant — 0% multi-physics prompts)*

## 7. OPEN QUESTIONS

1. What exact request distribution should validate the claim that the 7-pack MVP covers roughly 80% of user prompts? The pack set is concrete, but the coverage assumption still needs measurement.

2. When should pack composition become a first-class planning feature instead of an implicit technique-selection behavior?

3. What diversity metric should be tracked to detect when named function enforcement starts making scripts too structurally similar?

4. At what deployment scale or pack count should addon installation move from manual documentation to automated setup?

5. What exact Wave 1 execution-failure target should gate Wave 3 work: below 20%, below 15%, or a fixed improvement relative to baseline?
