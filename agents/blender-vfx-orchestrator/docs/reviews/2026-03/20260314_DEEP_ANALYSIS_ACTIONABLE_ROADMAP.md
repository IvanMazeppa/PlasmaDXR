# Deep Analysis and Actionable Roadmap

**Date:** 2026-03-14
**Reviewer:** Claude Opus 4.6 (deep analysis mode)
**Scope:** Full `agents/blender-vfx-orchestrator/` codebase and documentation corpus
**Method:** Code-grounded, mission-anchored, wave-aware analysis with actionable implementation roadmap
**Purpose:** Comprehensive assessment of the multiagent system's current state, synthesized into an ordered roadmap that targets the Mission Statement's success criteria

**Codebase examined:**

| Component | Files | Approximate LOC |
|-----------|-------|----------------|
| Orchestrator core | `orchestrator.py` | 3,355 |
| Pipeline models | `models/pipeline_models.py`, `models/shared_context.py` | 1,579 |
| Session management | `session_manager.py` | 527 |
| Specialized agents | `script_writer.py`, `quality_analyst.py`, `learning_agent.py` | 652 |
| Core tools | `script_generator_tools.py`, `truth_pack.py` | 2,743 |
| Guardrails | 8 files in `guardrails/` | ~2,600 |
| Hooks | 2 files in `hooks/` | ~970 |
| Phases | `execution.py`, `research.py`, `repair_routing.py` | ~959 |
| HITL | `utils/hitl_handler.py` | 335 |

**Documents examined:** Mission Statement, Current State, 4 March 2026 architecture reviews (GPT-5.4, Gemini, Comprehensive, Modify-Code Response), Deep Autonomy Analysis, Wave 2 Status Report, 3 postmortems, VERSION_TRUTH, AI_OPERATION_MANUAL, AGENTS.md, docs README.

---

## 1. Executive Summary

The Blender VFX Orchestrator has made meaningful architectural progress since Wave 1. It is no longer primarily blocked by technique monotony or Blender API hallucinations. `TechniqueContract` binding, truth-pack validation with 25+ auto-fix patterns, cross-physics generalization (cloth, rigid body, Mantaflow), stale-render invalidation, section patching infrastructure, GPT-5.4 adoption, and `call_model_input_filter` have moved the system from "can it ever work?" to "can it adapt correctly, learn honestly, and earn autonomy through evidence?"

The answer is still "not yet," but the reasons are now sharper. This document identifies 10 concrete findings, ordered by impact, and presents a 6-phase implementation roadmap with dependencies, exit criteria, and pseudocode sketches.

**The three most critical findings are:**

1. **Split state authority** -- `SessionState.record_iteration()` is bypassed in the hot path. `StuckDetectionState.update_from_iteration()` is never called. `plateau_count` is permanently 0. The deterministic stuck-state logic that the architecture invested in building is dead code in production.

2. **Repair routing receives wrong inputs** -- `choose_repair_intent()` exists and is well-designed, but it consumes `plateau_count=0` (always) and `escape_level` from LLM coordinator output (not from deterministic computation). The routing function cannot make correct structural-vs-parameter decisions when its inputs are wrong.

3. **The evaluation-to-repair boundary is still prose-heavy** -- `QualityIssue` with typed `kind` and `repair_mode_hint` exists in the model definition, but the Quality Analyst prompt and the downstream routing still depend heavily on free-text `primary_issue`, `issues`, and `suggestions`. Structured issue typing is available but underutilized.

**The bottom line:** The system has the right pieces but they are not connected. Fixing the wiring -- not adding new capabilities -- is the highest-leverage work available.

---

## 2. Architecture Assessment

### What the System Actually Is

The live system is **deterministic orchestration with bounded LLM specialists**. It is not an agent swarm, not an LLM-only loop, and not a template-driven generator.

```
User Prompt
    |
    v
[Research Phase] ---- Research Agent + Docs Expert (parallel)
    |
    v
[Technique Selection] ---- Technique Coordinator (LLM)
    |
    v
[Technique Contract] ---- Deterministic binding from capability packs
    |
    v
[Truth Pack Build] ---- Blender headless introspection (deterministic)
    |
    v
+--- ITERATION LOOP -------------------------------------------+
|                                                               |
|  [Pre-Iteration Research] ---- Early warning, escape actions  |
|       |                                                       |
|  [Script Generation/Modification]                             |
|       |-- Iter 1: Script Writer (LLM creative generation)     |
|       |-- Iter 2+: RepairIntent -> modify_params / modify_code|
|       |            / switch_technique / request_guidance       |
|       |                                                       |
|  [API Validation] ---- Truth pack + hallucination scanner     |
|       |                                                       |
|  [Execution] ---- Deterministic Blender subprocess            |
|       |                                                       |
|  [Artifact Gates] ---- Cache, render, VDB checks              |
|       |                                                       |
|  [Quality Evaluation] ---- Vision + ML metrics (LLM)          |
|       |                                                       |
|  [Code-Grounded Feedback] ---- QA diagnosis bridge            |
|       |                                                       |
|  [HITL Checkpoint] ---- Autonomy-gated                        |
|       |                                                       |
|  [Learning + Quality Gate] ---- Parallel LLM calls            |
|       |                                                       |
|  [Decision] ---- Pass / Iterate / Switch / Escalate           |
+---------------------------------------------------------------+
    |
    v
Final Result
```

### Active Runtime Topology

| Agent/Coordinator | Type | Hot Path? |
|-------------------|------|-----------|
| Research Agent | Standalone agent | Yes |
| Documentation Expert | Standalone agent | Yes |
| Script Writer | Standalone agent | Yes |
| Quality Analyst | Standalone agent | Yes |
| Learning Agent | Standalone agent | Yes |
| Technique Selector | Coordinator | Yes |
| Modification Strategist | Coordinator | Yes (iter 2+) |
| Quality Gate Judge | Coordinator | Yes |
| Executor Agent | Historical | No (removed from hot path) |
| API Spec Agent | Historical | No |
| Code Writer Agent | Historical | No |

### LLM vs Deterministic Ownership

| Responsibility | Owner | Assessment |
|----------------|-------|------------|
| Research synthesis | LLM | Correct |
| Technique selection | LLM coordinator | Partially reducible to policy |
| Technique binding | Deterministic (TechniqueContract) | Strong |
| Truth-pack build | Deterministic (Blender introspection) | Strong |
| Script generation | LLM (Script Writer) | Correct -- core creative value |
| API validation/correction | Deterministic (truth pack) | Strong |
| Execution | Deterministic (subprocess) | Correct |
| Artifact gating | Deterministic | Correct |
| Visual quality judgment | LLM (Quality Analyst) | Necessary but uncalibrated |
| Repair routing | Deterministic (choose_repair_intent) | Correct design, wrong inputs |
| Learning/logging | Mixed (Learning Agent) | Overloaded; should split |
| Quality gate decision | LLM coordinator | Partially reducible to policy |

---

## 3. Mission Alignment Scorecard

Assessed against Mission Statement Section 14: Success Criteria.

| # | Criterion | Status | Evidence | Gap |
|---|-----------|--------|----------|-----|
| 1 | Every run produces a render | MOSTLY | 2/3 E2E tests produced renders on iter 1. Stale render reuse fixed. But execution failures on iter 2+ still lose iterations. | Execution recovery needs hardening. |
| 2 | Known effects reliably score >= 60 | NO | Best scores: Mantaflow 58, Cloth 41, Destruction 12. No effect family has crossed 60 reliably. | Quality ceiling work blocked by repair routing and evaluator calibration. |
| 3 | Novel prompts produce reasonable first attempts | YES | Cloth and rigid body both selected correct physics on first attempt without physics-specific code. | Technique monotony solved for known families. |
| 4 | System improves over time | EMERGING | Cloth improved 34 to 41 in 2 iterations. Destruction stalled at 12 due to routing bugs. Mechanism exists but is fragile. | Improvement blocked by split state and parameter-first bias. |
| 5 | Failed runs produce useful diagnostics | YES | Trace logs, error parsing, QA diagnosis bridge, code-grounded feedback all produce actionable information. | QA bridge output underutilized by routing. |
| 6 | User can intervene at any point | PARTIAL | HITLHandler exists with 5 checkpoint types and autonomy levels. Not yet fully exercised in production. | Hybrid HITL (pipeline + native SDK approvals) not integrated. |
| 7 | Budget respected ($0.50/run) | YES | Cloth: ~$0.25, Destruction: ~$0.40. Well within target. | No gap. |

**Overall:** 3/7 criteria met, 2/7 partially met, 2/7 not met. The two unmet criteria (reliable scores >= 60, system improves over time) are blocked primarily by the state authority and repair routing issues, not by missing capabilities.

---

## 4. Critical Findings

### Finding 1: Split State Authority (CRITICAL)

**Root cause:** The orchestrator builds `IterationResult` manually and appends it with `session.iterations.append(iter_result)` instead of calling `session.record_iteration(result)`. As a result, `StuckDetectionState.update_from_iteration()` -- which computes `plateau_count`, `same_issue_count`, and `escape_level` deterministically -- is never invoked.

**Evidence (orchestrator.py ~2683):**

```python
iter_result = IterationResult(
    iteration=iteration,
    script=script_mod,
    execution=blender_exec,
    quality=quality_metrics,
    passed=quality.passed,
    score=quality.overall_score,
)
session.iterations.append(iter_result)
```

**Meanwhile, the intended path exists (shared_context.py):**

```python
def record_iteration(self, result: IterationResult) -> EscapeLevel:
    self.iterations.append(result)
    self.current_iteration = result.iteration
    ...
    escape_level = self.stuck_state.update_from_iteration(
        score=result.score,
        primary_issue=primary_issue,
        technique_used=technique
    )
```

**Consequence:** `plateau_count` is permanently 0. `same_issue_count` in `StuckDetectionState` is permanently 0. The deterministic escape level computation is dead code. The system operates with two disconnected stuck-state systems: `SessionManager.issue_tracker` (updated) and `SessionState.stuck_state` (not updated). Repair routing consumes `plateau_count=0`, making plateau-triggered `modify_code` impossible.

**Why this matters after the current roadmap:** Every downstream feature -- repair routing, learning validity, autonomy promotion, evaluator calibration -- depends on trustworthy iteration state. Without fixing this, the system cannot distinguish a plateau from normal iteration.

---

### Finding 2: Repair Routing Receives Wrong Inputs (CRITICAL)

**Root cause:** `choose_repair_intent()` in `phases/repair_routing.py` is correctly designed with clear priority ordering, but it receives poisoned inputs:

- `plateau_count`: Always 0 (Finding 1).
- `escape_level`: Set from LLM Quality Gate output, not from deterministic `StuckDetectionState`.
- `same_issue_count`: From `SessionManager.issue_tracker.consecutive_same_issue` (the only working counter).

**Evidence (orchestrator.py ~1490):**

```python
repair_intent = _choose_ri(
    quality=quality,
    code_grounded_feedback=code_grounded_feedback,
    plateau_count=getattr(session.stuck_state, "plateau_count", 0),
    same_issue_count=getattr(session_mgr.issue_tracker, "consecutive_same_issue", 0),
    iteration=iteration,
    escape_level=int(_escape_val),
)
```

**Evidence (orchestrator.py ~2948, escape level override):**

```python
session.stuck_state.escape_level = EscapeLevel(gate_decision.escape_level)
```

**Evidence (session_manager.py, missing escape_level in context):**

`get_context_for_agents()` does not include `escape_level`. The Quality Gate prompt uses `ctx.get('escape_level', 0)`, so the gate always sees 0 and returns 0.

**Consequence:** Repair routing rule #4 (`plateau >= 2 -> modify_code`) can never fire. Rule #6 (`escape_level >= 2 -> switch_technique`) depends on LLM output that itself receives 0. The Ireland Flag postmortem demonstrated this: three iterations with structural issues, all routed to `modify_params`.

---

### Finding 3: Evaluation-to-Repair Boundary Still Prose-Heavy (HIGH)

**Root cause:** The `QualityIssue` model with typed fields (`kind`, `repair_mode_hint`, `target`, `confidence`) exists in `pipeline_models.py`, but the Quality Analyst prompt and downstream routing still depend on free-text `primary_issue`, `issues`, and `suggestions`.

**Evidence (pipeline_models.py):**

```python
class QualityIssue(BaseModel):
    summary: str
    kind: Literal["parameter", "structural", "technique", "camera", "lighting"]
    repair_mode_hint: Literal["modify_params", "modify_code", "switch_technique"]
    target: str | None = None
    confidence: float = 0.0
```

**Evidence (repair_routing.py keyword heuristics):**

```python
STRUCTURAL_KEYWORDS = [
    "wrong order", "missing", "geometry", "shape", "topology", ...
]
PARAMETRIC_KEYWORDS = [
    "too dark", "density", "intensity", "resolution", ...
]
```

The routing function has to infer repair class from prose because `structured_issues` is not reliably populated or prioritized.

**Consequence:** Correct visual diagnosis keeps being translated into the wrong class of intervention. The QA can say "shader tweaks alone will not make this production-ready" while the routing sends `modify_params`.

---

### Finding 4: Learning Agent Overloaded (HIGH)

**Root cause:** One agent currently owns 22 tools spanning experiment tracking, KB querying, pattern extraction, modification advice, physics observation, code pattern storage, and doc search. Its hot-path role mixes logging with actuation.

**Evidence (learning_agent.py):** 22 tools listed in agent initialization.

**Evidence (LearningOutput):**

```python
class LearningOutput(BaseModel):
    next_action: str       # 'iterate', 'switch_technique', 'complete'
    parameter_modifications: Dict[str, Any]
```

`next_action = "iterate"` collapses into `modify_params` because `parameter_modifications` is emphasized in the prompt and treated as immediately actionable.

**Consequence:** The Learning Agent implicitly decides repair mode by returning parameter modifications. This competes with `RepairIntent` and the Modification Coordinator. Three different components can independently influence whether the system does parameter tweaks vs structural changes.

---

### Finding 5: Enforcement Gaps via Direct `_impl` Calls (HIGH)

**Root cause:** The execution phase and modification paths call implementation functions directly, bypassing tool wrappers and their attached guardrails.

**Evidence (phases/execution.py ~176):**

```python
exec_json_str = await _execute_blender_script_impl(
    script_path=script.script_path,
    script_args={"bake": "1"},
    output_dir=exec_output_dir,
    timeout_seconds=600,
)
```

This bypasses the `execute_blender_script` tool and its `truth_pack_input_guardrail`. Truth pack validation is done separately in the phase, so the behavior is similar but not enforced through the guardrail pipeline.

**Evidence (orchestrator.py, modification path):**

```python
result_raw = _modify_script_impl(
    script_path=script_path,
    modifications=modifications,
    output_name=output_name,
)
```

This bypasses `modify_script` tool and its guardrails.

**Consequence:** The project has a layered enforcement model, but several layers are not on the hot path. New enforcement logic added to tool guardrails will not apply to production execution.

---

### Finding 6: No Evaluator Calibration (HIGH)

**Root cause:** Quality scores range from 12 to 58 across physics families, but there are no baseline score bands to distinguish "almost there" from "fundamentally broken."

**Evidence:** Best scores by family: Mantaflow 58 (historical, pre-Wave-1), Cloth 41, Destruction 12. No calibration corpus exists. The evaluator has not been measured against known-good or known-bad renders.

**Consequence:** The system cannot distinguish model weakness from evaluator weakness from workflow weakness. Learning signals built on uncalibrated scores may promote bad patterns or demote good ones. The 60/100 pass threshold may be unrealistic for some physics families and too lenient for others.

---

### Finding 7: No Canonical Experiment Ledger (MEDIUM-HIGH)

**Root cause:** The system records outcomes through the Learning Agent and experiment tracker, but there is no provenance-safe per-iteration manifest that captures script hash, valid artifact provenance, evaluator version, repair mode, truth-pack version, and actual structural deltas.

**Evidence:** `IterationResult` captures script, execution, quality, and score, but not script hash, run directory provenance, evaluator profile, truth-pack build metadata, or technique-contract adherence result. `ArtifactManager.ManifestArtifact` still stamps `sdk_version = "0.7.0"`.

**Consequence:** Learning promotion cannot distinguish whether a score improvement came from a real code change, an evaluator fluctuation, a truth-pack correction, or an artifact provenance error. Evidence-gating is only as trustworthy as the evidence it gates.

---

### Finding 8: Autonomy Not Benchmark-Gated (MEDIUM)

**Root cause:** Autonomy levels (0-4) exist conceptually and in `HITLHandler` configuration, but there is no benchmark-gated promotion framework by effect family.

**Evidence:** `HITLHandler` uses a numeric `autonomy_level` to gate checkpoints, but the level is set by configuration, not earned by measured evidence. No per-family reliability thresholds, evaluator confidence metrics, or promotion criteria exist.

**Consequence:** "Autonomous" risks meaning "fewer checkpoints turned on" instead of "proven reliable in a measured envelope." The Mission Statement's principle P7 ("Earn Autonomy Through Evidence") is not yet enforced.

---

### Finding 9: Novel Capability Acquisition Weak (MEDIUM)

**Root cause:** Capability packs and technique contracts solve known-technique binding. For unknown or weakly trained Blender features, the system still relies on doc retrieval plus Script Writer obedience. There is no sandbox experimentation loop.

**Evidence:** The Mission Statement Section 8 describes sandbox mode. The architecture supports it conceptually. No implementation exists.

**Consequence:** The system may plateau as a better-known-technique orchestrator without becoming a true capability-expanding agent. The jump from Autonomy Level 3 to Level 4 ("discovers new techniques") requires capability acquisition.

---

### Finding 10: Documentation Drift (MEDIUM)

**Root cause:** Documents that declare themselves authoritative are partially stale about the actual runtime.

**Evidence:**
- `VERSION_TRUTH.md`: SDK v0.12.0 (correct).
- `AI_OPERATION_MANUAL.md`: SDK v0.10.5 (stale).
- `ArtifactManager.ManifestArtifact`: `sdk_version = "0.7.0"` (very stale).
- `RUNTIME_TRUTH_AND_DOC_GROUNDING_2026-02-13.md`: Claims `openai-agents==0.8.3` (stale).
- Several docs still describe spec-first as active; live code marks it deprecated.

**Consequence:** Architecture decisions made from stale docs will be wrong. Every future review wastes time on document reconciliation.

---

## 5. What Already Works

Credit where it is deserved. The system has made real progress.

### 5.1 Truth Pack -- The Most Valuable Infrastructure

The truth pack system (`tools/truth_pack.py`, ~1,037 LOC) is the strongest piece of the architecture:

- Runs Blender headless with `bl_rna.properties` introspection to enumerate every valid attribute, type, range, and default.
- 25+ hallucination patterns with deterministic auto-fixers.
- Catches `resolution_divisions`, `use_adaptive_time_steps`, `ShaderNodeMixRGB`, `Fac` vs `Factor`, `steps_per_second`, `Color 1` vs `Color1`, and more.
- Converts what would be Blender crashes into auto-fixed scripts.
- Every E2E test has truth pack catches. Without it, cross-physics generalization would be impossible.

### 5.2 TechniqueContract Solved Technique Monotony

Before Wave 1, "everything becomes Mantaflow" was the #1 failure mode. `TechniqueContract` binding with capability packs solved this:

- Correct physics selection for cloth, rigid body, destruction, Mantaflow.
- Contract adherence checking passes on every E2E test.
- 7 packs with 90% single-pack coverage of a 30-prompt validation set.

### 5.3 Cross-Physics Generalization

The system now handles 3+ physics families without physics-specific hardcoding:

| Physics Type | E2E Tested | Best Score | Execution Success |
|-------------|-----------|------------|-------------------|
| Mantaflow Gas | Historical | 58/100 | ~60% |
| Mantaflow Liquid | Historical | 70/100 (manual) | ~60% |
| Cloth | Yes (Wave 2) | 41/100 | 100% (2/2 iters) |
| Rigid Body | Yes (Wave 2) | 12/100 | 33% (1/3 iters) |

### 5.4 Cost Control

`call_model_input_filter` + GPT-5.4 routing + deterministic validation keeps runs under $0.50:
- Cloth (2 iterations): ~$0.25
- Destruction (3 iterations): ~$0.40
- Well within the Mission Statement's $0.50/run target.

### 5.5 Test Suite

413 unit tests covering models, tools, guardrails, pipeline logic, escape velocity, context filtering, and section patching. Changes do not regress.

### 5.6 Stale Render Fix

Iteration-scoped render discovery is implemented. `_discover_render_current_run()` only searches the current execution's `run_dir`. Failed executions keep `success=False` and store partial renders for diagnostics without promoting them to scoring.

### 5.7 Section Patching Infrastructure

AST-based section parser, `patch_script_section` tool, section naming guardrail, patch budget tracking (2/iter, 4/session), and `modify_code` wiring are all in place. Production proof is pending but infrastructure is solid.

### 5.8 HITL Handler

`HITLHandler` in `utils/hitl_handler.py` (~335 LOC) implements 5 checkpoint types, autonomy-level gating, interactive and non-interactive modes, and decision capture. Integrated at 3 points in the orchestrator.

### 5.9 Repair Routing Design

`choose_repair_intent()` in `phases/repair_routing.py` (~184 LOC) is well-designed with clear priority ordering, deterministic keyword heuristics, and support for `QualityIssue` structured issues. The design is correct; the inputs are wrong.

---

## 6. Actionable Roadmap

### Phase 1: Fix the Wiring (NOW -- Week 1-2)

**Goal:** Make the existing infrastructure work by connecting the pieces that are already built.

**Dependencies:** None. This is the foundation for everything else.

**Exit criteria:**

- `SessionState.record_iteration()` is the sole iteration-recording path.
- `StuckDetectionState.update_from_iteration()` fires every iteration.
- `plateau_count` and `same_issue_count` reflect actual run history.
- `choose_repair_intent()` consumes deterministic stuck-state values.
- A structural defect case reaches `modify_code` by iteration 2.

#### 1.1 Route Iteration Completion Through `record_iteration()`

**What changes:** Replace the manual `session.iterations.append(iter_result)` in the orchestrator with a call to `session.record_iteration(iter_result)`.

**Why:** This single change activates `StuckDetectionState.update_from_iteration()`, which computes `plateau_count`, `same_issue_count`, and `escape_level` deterministically.

**Implementation sketch:**

```python
# BEFORE (orchestrator.py ~2683):
session.iterations.append(iter_result)

# AFTER:
escape_level = session.record_iteration(iter_result)
```

**Complexity:** Low (~10 lines changed).

**Verification:** Unit test that runs 3 iterations with same `primary_issue` and verifies `same_issue_count == 3` and `escape_level >= SWITCH_TECHNIQUE`.

#### 1.2 Feed Deterministic Stuck-State to Repair Routing

**What changes:** Replace the `plateau_count` and `escape_level` inputs to `choose_repair_intent()` with values from the now-updated `SessionState.stuck_state`.

**Implementation sketch:**

```python
# BEFORE:
repair_intent = _choose_ri(
    ...
    plateau_count=getattr(session.stuck_state, "plateau_count", 0),
    escape_level=int(_escape_val),  # from LLM gate output
)

# AFTER:
repair_intent = _choose_ri(
    ...
    plateau_count=session.stuck_state.plateau_count,
    same_issue_count=session.stuck_state.same_issue_count,
    escape_level=int(session.stuck_state.escape_level),
)
```

**Complexity:** Low (~5 lines changed).

**Verification:** Integration test with plateau scenario (scores 40, 41, 40) where `repair_intent.mode == "modify_code"` by iteration 3.

#### 1.3 Add Escape Level to Agent Context

**What changes:** Add `escape_level` and `plateau_count` to `SessionManager.get_context_for_agents()` so the Quality Gate receives actual state.

**Implementation sketch:**

```python
# session_manager.py, get_context_for_agents():
context["escape_level"] = session.stuck_state.escape_level.value
context["plateau_count"] = session.stuck_state.plateau_count
context["same_issue_count"] = session.stuck_state.same_issue_count
```

**Complexity:** Low (~5 lines).

**Verification:** Quality Gate prompt context includes non-zero `escape_level` after 2+ plateau iterations.

#### 1.4 Stop Quality Gate from Overwriting Deterministic Escape Level

**What changes:** Remove or downgrade the line that overwrites `session.stuck_state.escape_level` from Quality Gate coordinator output.

**Implementation sketch:**

```python
# BEFORE (orchestrator.py ~2948):
session.stuck_state.escape_level = EscapeLevel(gate_decision.escape_level)

# AFTER:
# Quality Gate escape_level is advisory only; deterministic stuck-state is authoritative
if gate_decision.escape_level > session.stuck_state.escape_level.value:
    logger.info(f"Quality Gate suggests higher escape level "
                f"({gate_decision.escape_level} vs {session.stuck_state.escape_level}), "
                f"noting but not overriding deterministic computation")
```

**Complexity:** Low (~5 lines).

**Verification:** After 3 iterations with same issue, `escape_level` matches `StuckDetectionState` logic, not Quality Gate output.

---

### Phase 2: Strengthen the Evaluation-to-Repair Boundary (NOW -- Week 2-3)

**Goal:** Make the Quality Analyst produce typed repair signals that the routing can use without keyword heuristics.

**Dependencies:** Phase 1 (repair routing must work before typing its inputs matters).

**Exit criteria:**

- `QualityOutput.structured_issues` is reliably populated with `kind` and `repair_mode_hint`.
- `choose_repair_intent()` prioritizes structured issues over keyword heuristics.
- The Ireland Flag structural defect is classified as `kind="structural"` by the Quality Analyst.

#### 2.1 Enforce `structured_issues` in Quality Analyst Prompt

**What changes:** Update the Quality Analyst's dynamic instructions to require populating `structured_issues` with typed `QualityIssue` entries for every identified problem.

**Implementation sketch for prompt guidance:**

```
For every issue you identify, you MUST add a structured_issues entry with:
- kind: "parameter" | "structural" | "technique" | "camera" | "lighting"
- repair_mode_hint: "modify_params" | "modify_code" | "switch_technique"
- target: the specific element or section affected (e.g. "setup_materials", "light_energy")
- confidence: 0.0-1.0

A "structural" issue means the scene is architecturally wrong (wrong geometry, missing objects,
wrong topology, wrong physics setup). A "parameter" issue means the scene structure is correct
but numeric values need adjustment (light energy, density, resolution).
```

**Complexity:** Low-Medium (~20 lines of prompt, ~10 lines of guardrail update).

#### 2.2 Update Quality Output Guardrail

**What changes:** Add validation that `structured_issues` is non-empty when `passed=False`.

**Implementation sketch:**

```python
# guardrails/quality_guardrails.py
if not output.passed and len(output.structured_issues) == 0:
    return OutputGuardrailResult(
        output_info="Quality Analyst must provide structured_issues when passed=False",
        tripwire_triggered=True,
    )
```

**Complexity:** Low (~10 lines).

#### 2.3 Prioritize Structured Issues in Repair Routing

**What changes:** `choose_repair_intent()` checks `structured_issues` first, falls through to keyword heuristics only when structured issues are empty.

**Implementation sketch:**

```python
def choose_repair_intent(...) -> RepairIntent:
    # Priority 1: Execution failure
    if _is_execution_failure(quality):
        return RepairIntent(mode="modify_code", trigger="execution_failure")

    # Priority 2: Structured issues (authoritative when present)
    if quality.structured_issues:
        structural = [i for i in quality.structured_issues if i.kind == "structural"]
        parametric = [i for i in quality.structured_issues if i.kind == "parameter"]
        if len(structural) > len(parametric):
            return RepairIntent(
                mode="modify_code",
                trigger="structural_issue",
                target_sections=[i.target for i in structural if i.target],
                confidence=max(i.confidence for i in structural),
            )

    # Priority 3: Keyword heuristics (fallback)
    ...
```

**Complexity:** Low (~15 lines).

**Verification:** Replay Ireland Flag quality output through routing and confirm `modify_code` is selected.

---

### Phase 3: Demote Learning Agent from Repair Authority (NEXT -- Week 3-4)

**Goal:** Prevent the Learning Agent from implicitly deciding repair mode through `parameter_modifications`.

**Dependencies:** Phase 1 and Phase 2 (repair routing must be authoritative before demoting competing signals).

**Exit criteria:**

- `RepairIntent` is the sole source of truth for repair mode.
- Learning Agent `parameter_modifications` are applied only when `repair_intent.mode == "modify_params"`.
- Learning Agent `next_action` does not override `RepairIntent`.

#### 3.1 Gate Parameter Application on RepairIntent

**What changes:** The orchestrator applies Learning Agent `parameter_modifications` only when `repair_intent.mode == "modify_params"`. When `repair_intent.mode == "modify_code"`, parameter modifications are logged but not applied.

**Implementation sketch:**

```python
if repair_intent.mode == "modify_params":
    # Apply Learning Agent parameter suggestions
    if learning_output.parameter_modifications:
        modifications.update(learning_output.parameter_modifications)
elif repair_intent.mode == "modify_code":
    # Log but do not apply parameter suggestions
    if learning_output.parameter_modifications:
        logger.info(f"RepairIntent is modify_code; ignoring {len(learning_output.parameter_modifications)} "
                     f"parameter suggestions from Learning Agent")
```

**Complexity:** Low (~15 lines).

#### 3.2 Split `next_action` Vocabulary

**What changes:** Expand `LearningOutput.next_action` from `iterate`/`switch_technique`/`complete` to include `iterate_params`/`iterate_code`, making the Learning Agent's recommendation explicit. But make it advisory, not authoritative.

**Implementation sketch:**

```python
class LearningOutput(BaseModel):
    next_action: Literal["iterate_params", "iterate_code", "switch_technique", "complete"]
    parameter_modifications: Dict[str, Any] = {}
    code_change_suggestions: List[str] = []
```

**Complexity:** Low-Medium (~20 lines of model + prompt changes).

---

### Phase 4: Prove Structural Repair End-to-End (NEXT -- Week 4-5)

**Goal:** Demonstrate that `modify_code -> patch_script_section -> execute -> evaluate` works in a live run.

**Dependencies:** Phases 1-3 (routing must correctly select `modify_code`).

**Exit criteria:**

- At least one production-style E2E test where `modify_code` fires.
- Section patching preserves script quality better than full rewrite.
- Patch budget is honored.
- The pipeline improves or fails honestly.

#### 4.1 Create Structural-Repair Regression Suite

Build 5 cases where the expected fix is structural:

| Case | Expected Repair | Physics |
|------|----------------|---------|
| Ireland Flag (wrong stripe order) | modify_code (setup_materials) | Cloth |
| Missing collision effector | modify_code (setup_physics) | Liquid |
| Camera inside geometry | modify_code (setup_camera) | Any |
| Wrong attachment topology | modify_code (setup_physics) | Cloth |
| Missing key light | modify_code (setup_lighting) | Any |

**Complexity:** Medium (~200 lines of test cases).

#### 4.2 Run Live `modify_code` Proof

Execute the Ireland Flag case with Phases 1-3 fixes applied. Verify:

1. Iteration 1: Script generated, quality evaluates, structural issue identified.
2. Iteration 2: `RepairIntent.mode == "modify_code"`, section patching targets `setup_materials`.
3. Iteration 2 render: stripe order corrected (or honest failure with diagnostic).

**Complexity:** Low (running existing infrastructure).

---

### Phase 5: Calibrate Evaluation and Build Experiment Ledger (LATER -- Week 5-7)

**Goal:** Make quality scores trustworthy enough for learning promotion and autonomy gating.

**Dependencies:** Phase 4 (structural repair must work before calibrating evaluation against it).

**Exit criteria:**

- Score bands established per physics family (what "good" and "bad" look like).
- Per-iteration manifest captures script hash, artifact provenance, evaluator profile, repair mode.
- Learning promotion only occurs with provenance-safe evidence.

#### 5.1 Evaluator Calibration Ladder

Build a small corpus per physics family:

| Family | Known-Good | Known-Bad | Adversarial |
|--------|-----------|----------|-------------|
| Mantaflow Gas | 3 renders | 3 renders | 2 renders |
| Mantaflow Liquid | 3 renders | 3 renders | 2 renders |
| Cloth | 3 renders | 3 renders | 2 renders |
| Rigid Body | 3 renders | 3 renders | 2 renders |

Run through evaluator. Establish expected score ranges. Adjust if bands overlap or are unrealistic.

**Complexity:** Medium (requires curating renders + running evaluator).

#### 5.2 Canonical Iteration Manifest

**What changes:** Extend `IterationResult` or create a sibling `IterationManifest` that captures:

```python
class IterationManifest(BaseModel):
    iteration: int
    script_hash: str
    run_dir: str
    render_path: str | None
    artifact_valid: bool
    evaluator_profile: str
    truth_pack_version: str
    technique_contract_adherence: bool
    repair_mode: str
    primary_issue: str | None
    score: float
    delta_from_previous: float | None
```

**Complexity:** Medium (~50 lines of model + ~30 lines of recording logic).

#### 5.3 Learning Promotion with Provenance

**What changes:** Learning promotion requires:

- `artifact_valid == True`
- `evaluator_profile` matches calibrated profile
- `script_hash` is different from previous iteration (actual change occurred)
- At least N=3 clean manifests supporting the pattern

**Complexity:** Medium (~40 lines).

---

### Phase 6: Autonomy Promotion and Capability Acquisition (LATER -- Week 7-10)

**Goal:** Make autonomy evidence-based and add a pathway for the system to learn genuinely new capabilities.

**Dependencies:** Phase 5 (calibrated evaluation and provenance-safe learning).

**Exit criteria:**

- Per-family autonomy scoreboard with promotion thresholds.
- At least one capability incubated through sandbox mode without full-scene first attempt.
- Hybrid HITL (pipeline + native SDK approvals) integrated.

#### 6.1 Autonomy Promotion Contract

Define per-family promotion gates:

| Gate | Threshold | Measured By |
|------|-----------|-------------|
| Valid artifact rate | >= 80% | Iteration manifests |
| Repair success rate | >= 50% | `modify_code` cases that improve score |
| Evaluator calibration | Score bands established | Calibration ladder |
| Cost envelope | <= $0.50/run average | Budget tracker |
| Pass rate (score >= 60) | >= 50% for Assisted, >= 70% for Autonomous | Session history |

```python
class AutonomyPromotion(BaseModel):
    effect_family: str
    current_level: int
    valid_artifact_rate: float
    repair_success_rate: float
    evaluator_calibrated: bool
    avg_cost_per_run: float
    pass_rate: float
    qualifies_for_level: int
```

**Complexity:** Medium (~80 lines).

#### 6.2 Sandbox Experiment Mode

**What changes:** Add an explicit `sandbox_experiment` run mode:

1. Retrieve docs for a novel Blender feature.
2. Generate a minimal operator/addon micro-test (< 100 lines).
3. Run in sandbox Blender instance.
4. Capture artifacts and result.
5. Decide: promote to pack candidate, record as anti-pattern, or inconclusive.

**Complexity:** Medium-High (~200 lines).

#### 6.3 Hybrid HITL Integration

**What changes:** Combine pipeline-level `HITLHandler` with native SDK `needs_approval=True` on high-impact tools:

- `switch_technique`: `needs_approval=True` (at autonomy level < 2).
- `patch_script_section` (full rewrite): `needs_approval=True` (at autonomy level < 3).
- Budget increase: `needs_approval=True` (always).

**Complexity:** Low-Medium (~30 lines per tool).

---

## 7. Benchmarks and Experiments to Add

### 7.1 State-Authority Regression Test

Create a deterministic test where 3 iterations have the same `primary_issue`.

**Success criteria:**
- `session.stuck_state.same_issue_count == 3`
- `session.stuck_state.escape_level >= SWITCH_TECHNIQUE`
- `session.stuck_state.plateau_count` reflects actual score trajectory

### 7.2 Repair-Mode Classification Benchmark

Build a labeled dataset from historical traces with human labels for `modify_params`, `modify_code`, `switch_technique`, and `request_guidance`.

**Success criteria:**
- `choose_repair_intent()` agrees with human labels >= 80% of the time.

### 7.3 Cross-Physics Reliability Baseline

Run the 30-prompt validation set through 1 iteration each with the current stack.

**Measure:** Execution validity, artifact validity, evaluator invocation, cost.

**Success criteria:** A trustworthy baseline number exists.

### 7.4 Evaluator Calibration Ladder

Maintain a fixed corpus of good, bad, and misleading renders per physics family.

**Success criteria:** Score bands are sane and non-overlapping for "clearly good" vs "clearly bad."

### 7.5 Section-Patching Production Proof

Force at least one `modify_code` path on a real structural defect.

**Success criteria:** Patch budget honored, script quality preserved, pipeline improves or fails honestly.

### 7.6 Manifest Integrity Test

For each completed iteration, verify that persisted manifest, session state, trace summary, and scorecard agree on render path, score, repair mode, and primary issue.

**Success criteria:** Zero disagreements in a 10-iteration test run.

---

## 8. What to Stop Doing

| # | Stop This | Rationale |
|---|-----------|-----------|
| 1 | Bypassing `SessionState.record_iteration()` | This is the root cause of split state authority. |
| 2 | Letting Quality Gate overwrite deterministic escape level | The Quality Gate should advise, not override. |
| 3 | Applying parameter modifications before repair routing decides | Parameter-first bias preempts structural repair. |
| 4 | Treating `iterate` as synonymous with "tune parameters" | The action vocabulary must distinguish parameter from structural changes. |
| 5 | Attaching guardrails to code paths the hot path bypasses | Either route through the guardrailed path or move the guardrail to where execution actually happens. |
| 6 | Increasing `max_iterations` as a fix for missing structural repair | A larger budget hides the routing bug by making the parameter rut longer. |
| 7 | Treating evaluator scores as reliable learning signal before calibration | Uncalibrated scores may promote bad patterns or demote good ones. |
| 8 | Keeping the Learning Agent on every hot-path responsibility | Split into deterministic experiment logger + periodic distiller. |
| 9 | Maintaining stale version claims in authority docs | `AI_OPERATION_MANUAL` saying v0.10.5 while `VERSION_TRUTH` says v0.12.0 wastes review time. |
| 10 | Calling section patching "done" before a live `modify_code` proof | Infrastructure without production evidence is theoretical. |

---

## 9. Open Questions

1. **Where should structural-vs-parameter classification live?**
   Options: (a) Quality Analyst output only, (b) QA diagnosis bridge, (c) deterministic `RepairIntent` that consumes typed QA signals. Recommendation: (c), with (a) as the authoritative input.

2. **Should one `modify_code` attempt be the default for structural issues?**
   Or should the system require section hints and a confidence threshold first? Recommendation: require `confidence >= 0.5` and at least one `target_section` before `modify_code`; fall back to full rewrite if section hints are absent.

3. **How much of the Learning Agent should remain online?**
   Recommendation: Keep experiment recording and KB querying online. Move pattern distillation, negative-knowledge synthesis, and modification advisory to periodic offline processing.

4. **Which effect family should be the first autonomy-promotion pilot?**
   Recommendation: Cloth (simplest physics, best recent E2E evidence, most iterations available).

5. **What is the minimum per-iteration manifest?**
   Recommendation: `iteration`, `script_hash`, `run_dir`, `render_path`, `artifact_valid`, `evaluator_profile`, `repair_mode`, `score`. Extend later.

6. **How should `production`, `recovery`, and `sandbox_experiment` run modes differ?**
   Recommendation: Different patch budgets, different learning consequences, different HITL thresholds. Same pipeline loop, different configuration profiles.

---

## 10. Appendix

### A. File Map (Active Hot Path)

```
orchestrator.py                           # Main pipeline loop (3,355 LOC)
phases/
  research.py                             # Research, technique selection, contract, truth pack (382 LOC)
  execution.py                            # Execution, recovery, artifact gates (393 LOC)
  repair_routing.py                       # Deterministic repair intent (184 LOC)
models/
  pipeline_models.py                      # Agent outputs, coordinator decisions, RepairIntent (608 LOC)
  shared_context.py                       # SessionState, SharedContext, StuckDetectionState (971 LOC)
session_manager.py                        # Experiment state tracking (527 LOC)
specialized_agents/
  script_writer.py                        # Creative script generation (200 LOC)
  quality_analyst.py                      # Quality evaluation (208 LOC)
  learning_agent.py                       # Experiment tracking + learning (244 LOC)
tools/
  script_generator_tools.py               # Script write/modify/validate (1,706 LOC)
  truth_pack.py                           # Blender API introspection + validation (1,037 LOC)
  dynamic_instructions.py                 # Runtime instruction injection
  qa_diagnosis_bridge.py                  # Code-grounded feedback
  asset_evaluator_tools.py                # ML quality metrics
  semantic_docs_tools.py                  # Vector store search
  blender_executor_tools.py               # Blender process management
  experiment_tracker_tools.py             # Knowledge base operations
guardrails/
  script_guardrails.py                    # Script Writer input/output (375 LOC)
  research_guardrails.py                  # Research doc grounding (191 LOC)
  artifact_gates.py                       # Post-execution deterministic checks (399 LOC)
  quality_guardrails.py                   # Budget + quality output (375 LOC)
  tool_guardrails.py                      # Truth pack, critical failure, script length (275 LOC)
  coordinator_guardrails.py               # Decision validation (415 LOC)
hooks/
  enforcement_hooks.py                    # Loop detection, doc-query, turn budget (575 LOC)
  diagnostic_hooks.py                     # Observability (395 LOC)
utils/
  hitl_handler.py                         # HITL checkpoints (335 LOC)
```

### B. State Flow Diagram

```
                    +-------------------+
                    |   SessionState    |  <-- Persistent; saved to JSON
                    |  .iterations[]    |      Holds: iterations, best_score,
                    |  .stuck_state     |      stuck_state, technique_contract
                    |  .best_score      |
                    +--------+----------+
                             |
                   record_iteration()  <-- CURRENTLY BYPASSED
                             |
                    +--------v----------+
                    | StuckDetectionState|  <-- DEAD CODE (never updated)
                    |  .plateau_count   |      Should compute: plateau_count,
                    |  .same_issue_count|      same_issue_count, escape_level
                    |  .escape_level    |
                    +-------------------+

                    +-------------------+
                    |  SessionManager   |  <-- Derived; rebuilt from iterations
                    |  .issue_tracker   |      Holds: consecutive_same_issue,
                    |  .best_score      |      last_primary_issue, baselines
                    +--------+----------+
                             |
                     record_result()   <-- WORKING (updates issue_tracker)
                             |
                    +--------v----------+
                    |   IssueTracker    |  <-- Only working stuck-state source
                    | .consecutive_     |
                    |  same_issue       |
                    +-------------------+

                    +-------------------+
                    |  Quality Gate     |  <-- LLM coordinator
                    |  .escape_level    |      OVERWRITES deterministic
                    +-------------------+      stuck_state (wrong)
```

**After Phase 1 fix:**

```
                    +-------------------+
                    |   SessionState    |
                    |  .iterations[]    |
                    |  .stuck_state ----+---> StuckDetectionState
                    |  .best_score      |     .plateau_count (LIVE)
                    +--------+----------+     .same_issue_count (LIVE)
                             |                .escape_level (AUTHORITATIVE)
                   record_iteration()
                             |
                    +--------v----------+
                    | choose_repair_    |  <-- Consumes deterministic state
                    |   intent()        |
                    +-------------------+
                             |
                    +--------v----------+
                    |    RepairIntent   |  <-- Sole repair-mode authority
                    | .mode             |
                    | .trigger          |
                    | .target_sections  |
                    +-------------------+
```

### C. Document Freshness Matrix

| Document | Last Updated | SDK Version Claimed | Stale? |
|----------|-------------|--------------------|----|
| `VERSION_TRUTH.md` | 2026-03-14 | v0.12.0 | No |
| `CURRENT_STATE.md` | 2026-03-14 | N/A | No |
| `MISSION_STATEMENT_2026-02-22.md` | 2026-02-22 | v0.9.0+ | Slightly (implementation status) |
| `AI_OPERATION_MANUAL.md` | 2026-02-13 | v0.10.5 | Yes (SDK version) |
| `SDK_ENFORCEMENT_PROTOCOL.md` | ~2026-01 | v0.10.5 | Yes (SDK version) |
| `RUNTIME_TRUTH_AND_DOC_GROUNDING_2026-02-13.md` | 2026-02-13 | v0.8.3 | Very stale |
| `DOCS_AUTHORITY_AND_FRESHNESS_2026-02-13.md` | 2026-02-13 | N/A | Routes to stale doc |
| `ArtifactManager` (code) | ~2026-01 | v0.7.0 | Very stale |

### D. Roadmap Summary Timeline

```
Week 1-2:  Phase 1 -- Fix the Wiring (state authority, repair routing inputs)
Week 2-3:  Phase 2 -- Strengthen Eval-to-Repair Boundary (typed QA issues)
Week 3-4:  Phase 3 -- Demote Learning Agent from Repair Authority
Week 4-5:  Phase 4 -- Prove Structural Repair End-to-End
Week 5-7:  Phase 5 -- Calibrate Evaluation and Build Experiment Ledger
Week 7-10: Phase 6 -- Autonomy Promotion and Capability Acquisition
```

| Phase | Effort | Risk | Payoff |
|-------|--------|------|--------|
| 1. Fix the Wiring | Low | Low | CRITICAL -- unblocks everything |
| 2. Typed Eval Boundary | Low-Medium | Low | HIGH -- correct repair routing |
| 3. Demote Learning Agent | Low | Low | HIGH -- eliminates competing repair signals |
| 4. Structural Repair Proof | Medium | Medium | HIGH -- validates architecture |
| 5. Evaluator Calibration | Medium | Low | HIGH -- trustworthy learning |
| 6. Autonomy + Capability | Medium-High | Medium | STRATEGIC -- mission completion |

---

## References

### Internal Documents

- `docs/MISSION_STATEMENT_2026-02-22.md` -- Canonical product vision
- `docs/CURRENT_STATE.md` -- Current high-level state (authoritative)
- `docs/VERSION_TRUTH.md` -- Model/SDK/Blender ground truth
- `docs/reviews/2026-03/20260312_WAVE2_STATUS_AND_PATH_FORWARD.md` -- Wave 2 implementation status
- `docs/reviews/2026-03/20260313_COMPREHENSIVE_MULTIAGENT_SYSTEM_ANALYSIS.md` -- Comprehensive system analysis
- `docs/reviews/2026-03/20260314_ARCHITECTURE_REVIEW_MODIFY_CODE_RESPONSE.md` -- Modify-code crisis response
- `docs/reviews/2026-03/20260314_DEEP_AUTONOMY_ANALYSIS_GPT_54_XHIGH.md` -- Deep autonomy analysis
- `docs/reviews/2026-03/20260314_DEEP_ANALYSIS_MULTIAGENT_SYSTEM_GEMINI.md` -- Gemini review
- `docs/postmortems/2026-03/20260314_WAVE2A_POSTMORTEM_AND_MODIFY_CODE_CRISIS.md` -- Ireland Flag postmortem

### External Research

- Anthropic, "Building Effective Agents" -- Simple composable patterns over complex agent stacks
- OpenAI Agents SDK HITL docs -- `needs_approval`, `RunState` for durable pause/resume
- OpenAI Agents SDK guardrails docs -- Tool guardrails apply only to function-tool pipeline
- Google DeepMind, MONA -- Reward hacking risks in multi-step optimization
- Blender 5.0 Manual (cloth, collision, physics) -- Structural vs parametric workflow distinction
