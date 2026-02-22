# Phase 2+ Roadmap: From Reliability to Full Autonomy

**Version:** 1.0
**Date:** 2026-02-22
**Authors:** Integration Architect (Claude Opus 4.6), synthesizing research from SDK Specialist, Codebase Analyst, Autonomy Researcher, and Monitoring Architect
**Status:** Approved plan — implementation starting

---

## Executive Summary

Phase 1 (Reliability) shipped 1,574 lines across 11 files: truth pack, QA feedback bridge, KB wipe, session compaction, and escape velocity testing. The pipeline runs end-to-end without crashing. Scripts still have quality issues, but the infrastructure is solid.

Phase 2+ takes the system from "produces evaluable renders" to "reliably produces quality renders that improve over time." It is organized into four sub-phases:

| Phase | Focus | Estimated Effort | Timeline |
|-------|-------|-----------------|----------|
| **2A: Quick Wins** | Cost savings, deterministic safety nets, monitoring | ~600 lines new/modified | 1-2 weeks |
| **2B: Core Architecture** | Stateless iterations, context management, multi-grader eval | ~1,200 lines | 2-4 weeks |
| **2C: Advanced Capabilities** | Multi-physics, experimentation, technique diversity | ~1,500 lines | 4-8 weeks |
| **2D: Full Autonomy** | Self-improvement, autonomy tracking, novel prompt handling | ~1,000 lines | 8+ weeks |

**Total estimated new/modified code: ~4,300 lines.**

**Expected outcomes by end of Phase 2B:**
- Every run produces a render (reliability target maintained)
- Known effects (fire, liquid, smoke) score >= 60 in 50%+ of runs
- Cost per 3-iteration run: $0.15-0.35 (down from $0.30-0.70)
- No parameter oscillation across iterations
- Context stays within budget per agent per iteration

---

## Foundation: What Phase 1 Delivered

| Deliverable | File | Impact |
|------------|------|--------|
| Truth Pack (bl_rna introspection) | `tools/truth_pack.py` (743 LOC) | Eliminates #1 failure mode (hallucinated attributes) at $0 |
| Truth Pack Validator | `tools/truth_pack_validator.py` (187 LOC) | Validate-fix-write cycle for scripts |
| QA Feedback Bridge | `tools/qa_diagnosis_bridge.py` (254 LOC) | Pairs visual critique with script code |
| KB Wipe Script | `scripts/wipe_kb.py` (99 LOC) | Clean slate, 18 items backed up |
| Escape Velocity Tests | `tests/test_escape_velocity.py` (144 LOC) | 16 tests, L4→PAUSED verified |
| Session Compaction | `orchestrator.py` modification | Auto-triggers at ~25K tokens |
| Evidence Gating | Multiple files | min_confidence 50, min_success_rate 0.8, decay |

**Live E2E test result** (trace `63eb4ef1`): Pipeline completed without crash. 21KB fire mantaflow script generated. Execution failed on `use_nodes` deprecation and `collection.get(int)` bugs — both now fixed by truth pack patterns #13 and #15.

---

## Design Principles (Ranked — Higher Overrides Lower)

These principles resolve conflicts between competing approaches. Every work item references which principles it serves.

| Rank | Principle | Shorthand |
|------|-----------|-----------|
| **P1** | Reliability before capability | Don't add features that destabilize |
| **P2** | LLM creativity is the core value | Don't replace creative agents with templates |
| **P3** | Blender is the source of truth for its own API | Runtime introspection > training data |
| **P4** | Compute what you can, generate what you must | Deterministic where possible, LLM where valuable |
| **P5** | Context is precious — every token must earn its place | Dense output, artifact-based sharing |
| **P6** | Every run produces learning signal | Even failures have value if captured |
| **P7** | Earn autonomy through evidence | Trust levels based on outcomes, not calendar time |

---

## Phase 2A: Quick Wins

**Goal:** Immediate cost savings, deterministic safety nets, monitoring infrastructure.
**Risk:** Low — all changes are additive or replace redundant code.
**Principles served:** P1 (reliability), P4 (compute what you can), P5 (context precious).

### 2A-1: Conditional Tool Enabling (`is_enabled`)

**What:** Hide expensive tools from agents when budget is low, instead of letting agents try to use them and hitting guardrail errors.

**Why it matters:** Currently all 20+ tools are visible to every agent regardless of budget state. The Learning Agent must choose from 20+ tools per turn — causing tool selection confusion and wasted turns. Vision evaluation tools remain visible even when budget is exhausted, leading to guardrail-blocked calls that waste turns.

**Research support:**
- SDK Analysis §4: "Trivial to implement, immediate budget savings, cleaner than guardrail-based budget checks"
- Codebase Analysis §7.9: "Learning Agent has 20+ tools — use is_enabled to contextually enable only relevant tools"
- Codebase Analysis §7.10: "No graceful budget degradation — is_enabled enables tiered availability"

**Implementation:**

| File | Change | Lines |
|------|--------|-------|
| `specialized_agents/quality_analyst.py` | Add `is_enabled=budget_allows_vision` to vision evaluation tool | ~5 |
| `specialized_agents/docs_expert.py` | Add `is_enabled=budget_allows_docs` to doc search tools | ~5 |
| `orchestrator.py` (coordinator creation) | Wrap expensive agent-as-tool calls with `is_enabled` callbacks | ~10 |
| `tools/asset_evaluator_tools.py` | Move budget check from guardrail to `is_enabled` | ~10 |
| New: `utils/tool_visibility.py` | Callback functions for budget/phase/escape-level gating | ~40 |

**Callback examples:**
```python
def budget_allows_vision(ctx: RunContextWrapper, agent) -> bool:
    return ctx.context.budget_tracker.can_afford_evaluation()

def budget_allows_docs(ctx: RunContextWrapper, agent) -> bool:
    return ctx.context.budget_tracker.get_remaining() > 2.0

def learning_tool_for_iteration(ctx: RunContextWrapper, agent) -> bool:
    # Hide mine_docs_for_patterns before iteration 2 (no data yet)
    return ctx.context.session.current_iteration > 1
```

**Estimated effort:** ~70 lines | **Dependencies:** None | **Risk:** Trivial

---

### 2A-2: Deterministic Agent Control (`tool_use_behavior`)

**What:** Eliminate unnecessary LLM post-processing calls for agents whose tool output IS the final answer.

**Why it matters:** The Executor agent calls `execute_blender_script`, then the LLM processes the result to produce a "response" — pure waste. Same for the API Validator and Learning Agent's recording operations. Each unnecessary LLM call costs ~$0.01-0.02. With 15-46 LLM calls per run, eliminating even 4-5 unnecessary calls saves $0.05-0.10/run.

**Research support:**
- SDK Analysis §5: "Saves ~$0.05-0.10/run from a single-line change"
- Codebase Analysis §6.5: "Executor should be deterministic — call execute_blender_script once and stop"
- Autonomy Research §3.1: "Agents that should become DETERMINISTIC functions: Executor, API Validator, Budget tracker"

**Implementation:**

| File | Change | Lines |
|------|--------|-------|
| `specialized_agents/executor.py` | Add `tool_use_behavior="stop_on_first_tool"` | 1 |
| `specialized_agents/api_validator.py` | Add `tool_use_behavior="stop_on_first_tool"` | 1 |
| `specialized_agents/learning_agent.py` | Add `tool_use_behavior="stop_on_first_tool"` for recording-only mode | 1 |
| `orchestrator.py` (quality gate coordinator) | Add `StopAtTools(stop_at_tool_names=["make_quality_decision"])` | 3 |

**Estimated effort:** ~6 lines | **Dependencies:** None | **Risk:** Trivial
**Monthly savings:** ~$2-5 at current run volume

---

### 2A-3: Tool Guardrails for Truth Pack Enforcement

**What:** Move truth pack validation from an orchestrator pipeline step INTO the tool itself using `ToolInputGuardrail`. Every time any agent calls `execute_blender_script`, the script is automatically validated against the truth pack first. If validation fails, the tool is skipped and the agent receives an error message.

**Why it matters:** Currently truth pack validation runs as a separate Phase 1.5 orchestrated by Python code. An agent could theoretically bypass it (e.g., during error recovery in Phase 2.5-2.7). Tool guardrails make validation **impossible to bypass** — it's part of the tool definition.

**Research support:**
- SDK Analysis §6: "Deterministic validation as a tool wrapper is the cheapest, most reliable way to enforce the truth pack. Replaces an entire pipeline step with a zero-cost guardrail."
- Codebase Analysis §7.8: "Error recovery creates new scripts without full context — recovery scripts may re-introduce hallucinated attributes"
- Codebase Analysis §3.3: "Four overlapping validation layers — truth pack should subsume others over time"

**Implementation:**

| File | Change | Lines |
|------|--------|-------|
| New: `guardrails/tool_guardrails.py` | Truth pack input guardrail, execution output guardrail, script length output guardrail | ~100 |
| `tools/blender_executor_tools.py` | Wrap `execute_blender_script` with truth pack input guardrail | ~5 |
| `tools/asset_evaluator_tools.py` | Wrap `evaluate_render` with output guardrail for critical failures | ~5 |
| `tools/script_generator_tools.py` | Wrap `generate_script` with output guardrail for script length check | ~5 |

**Guardrail logic (input):**
```python
@tool_input_guardrail
def validate_script_against_truth_pack(data):
    script_path = json.loads(data.context.tool_arguments or "{}").get("script_path")
    if script_path:
        errors = validate_against_truth_pack(script_path, data.context.context.truth_pack)
        if errors:
            fix_report = auto_fix_errors(script_path, errors)
            return ToolGuardrailFunctionOutput.reject_content(
                f"Script had {len(errors)} hallucinated attributes. Auto-fixed: {fix_report}."
            )
    return ToolGuardrailFunctionOutput.allow()
```

**Estimated effort:** ~115 lines | **Dependencies:** Truth pack (Phase 1, done) | **Risk:** Low

---

### 2A-4: PipelineMonitor (Deterministic Monitoring Layer)

**What:** A Python class (NOT an LLM agent) that runs at orchestrator checkpoints between pipeline phases. Detects parameter oscillation, stuck loops, wrong feedback cascades, budget overruns, and script quality issues.

**Why it matters:** Ben explicitly wants a monitoring layer (Interview Q10, Q16). The system's known failure modes — oscillating parameters, quality analyst giving wrong advice, stuck loops — are all detectable deterministically at $0 cost. This is the monitoring agent Ben described, implemented as a deterministic class rather than an expensive LLM.

**Research support:**
- Autonomy Research §2.4 (MASC): "A single faulty step can propagate across agents and disrupt the trajectory" — deterministic monitoring catches this at $0
- Monitoring Architecture §1.1: "95% deterministic monitor, 5% LLM escalation"
- Monitoring Architecture §1.3: Complete PipelineMonitor class design with oscillation detection, cascade detection, budget tracking
- Mission Statement §9: "A monitoring agent/layer that watches pipeline progress"

**Implementation:**

| File | Change | Lines |
|------|--------|-------|
| New: `tools/pipeline_monitor.py` | `PipelineMonitor` class with `check_after_generation()`, `check_after_evaluation()`, `get_status_report()` | ~250 |
| `orchestrator.py` (iteration loop) | Add monitor checkpoint calls after generation and after evaluation | ~40 |
| `orchestrator.py` (iteration loop) | Wire monitor alerts into modification pipeline (parameter clamping, cascade revert) | ~30 |

**Signals monitored:**

| Signal | Threshold | Action |
|--------|-----------|--------|
| Parameter oscillation | Same param changes direction 2x in 3 iterations | Clamp to bounded midpoint |
| Stuck loop (same error) | 3x same `primary_issue` | Force technique switch (escape L2) |
| Score plateau | `abs(score_delta) < 3.0` for 3 iterations | Escalate escape velocity |
| Wrong feedback cascade | Score drops >5 after applying QA suggestion | Revert params, warn |
| Budget overrun | >$0.50 per run | Disable expensive tools |
| Script quality | <500 lines | Inject "scene too basic" warning |
| Critical render issue | BLACK_SCREEN, WHITE_SCREEN, ZERO_LIGHTS | Skip normal iteration, targeted fix |

**Integration with escape velocity:** Monitor feeds `StuckDetectionState` with additional signals. Oscillation counts as "same issue" for escape level escalation. Parameter clamping is additive — escape velocity doesn't do this.

**Estimated effort:** ~320 lines | **Dependencies:** None | **Risk:** Low

---

### 2A-5: Parameter Bounds and Damped Convergence

**What:** Per-effect-type parameter bounds that prevent overcorrection. All parameter modifications are clamped to safe ranges and limited to a maximum step size per iteration.

**Why it matters:** Light energy oscillation (50 → 2500 → 10) is a known top failure mode (Interview Q16, Mission Statement §9). The QA says "too dark" → energy jumps to 2500 → QA says "overexposed" → energy drops to 10 → repeat forever.

**Research support:**
- Monitoring Architecture §3.3: Complete bounded modification design with damped convergence
- Autonomy Research §5.3: "Parameter clamping: prevent the script writer from making extreme changes"
- Mission Statement §12: "Parameter oscillation — overcorrection between iterations (50 → 2500 → 10)"

**Implementation:**

| File | Change | Lines |
|------|--------|-------|
| New: `tools/parameter_bounds.py` | `ParameterBound` dataclass, `PARAMETER_BOUNDS` dict per effect type, `clamp()` and `damped_change()` methods | ~120 |
| `orchestrator.py` (modification phase) | Apply bounds before passing parameter changes to Script Writer | ~30 |

**Example bounds:**
```python
PARAMETER_BOUNDS = {
    "fire": {
        "energy": ParameterBound("energy", 20.0, 200.0, step_size=50.0),
        "density": ParameterBound("density", 1.0, 15.0, step_size=3.0),
        "blackbody_intensity": ParameterBound("blackbody_intensity", 0.5, 10.0, step_size=2.0),
    },
    "liquid": {
        "energy": ParameterBound("energy", 30.0, 300.0, step_size=80.0),
        "viscosity_base": ParameterBound("viscosity_base", 0.0, 5.0, step_size=1.0),
        "resolution_max": ParameterBound("resolution_max", 64, 256, step_size=32),
    },
}
```

**Estimated effort:** ~150 lines | **Dependencies:** PipelineMonitor (2A-4) for runtime bounds refinement | **Risk:** Low

---

### 2A-6: Deprecate Spec-First Pipeline

**What:** Remove the LLM-powered API Spec Agent from the pipeline. The Truth Pack provides the same data deterministically at $0.

**Why it matters:** The API Spec Agent runs in Phase 0.5 (parallel with technique selection), costing ~2 LLM calls per run. The Truth Pack in Phase 0.6 provides the same data from Blender introspection. Running both is redundant.

**Research support:**
- Codebase Analysis §7.4: "Spec-First pipeline costs LLM calls; Truth Pack costs $0"
- Codebase Analysis §10: "Deprecate Spec-First pipeline — saves ~2 LLM calls per run"

**Implementation:**

| File | Change | Lines |
|------|--------|-------|
| `orchestrator.py` (Phase 0.5) | Remove `_run_parallel_technique_and_spec()` parallel branch for API Spec. Technique selection runs alone. | ~30 |
| `orchestrator.py` (Phase 1) | Remove `_run_spec_first_pipeline()` code path. Script Writer gets truth pack directly via dynamic instructions. | ~50 |
| `specialized_agents/api_spec_agent.py` | Add deprecation notice at top. Keep file for reference but don't import. | ~5 |

**What's preserved:** `models/api_spec.py` Pydantic models stay — `truth_pack_to_api_spec()` populates them from truth pack data, maintaining backward compatibility with the Code Writer guardrail.

**Estimated effort:** ~85 lines changed | **Dependencies:** Truth Pack (Phase 1, done) | **Risk:** Low-Medium (must verify Code Writer still gets good context)
**Savings:** ~$0.04-0.10/run (2 LLM calls eliminated)

---

### 2A-7: Remove Dead Executor Agent

**What:** Phase 2 (Execution) calls `_execute_blender_script_impl()` directly — deterministic, no LLM. The Executor agent (170 LOC) is created in `initialize()` but never used in the pipeline path. Remove it.

**Research support:**
- Codebase Analysis §7.3: "Dead code. Executor agent is created but never used in the pipeline path."

**Implementation:**

| File | Change | Lines |
|------|--------|-------|
| `orchestrator.py` (initialize) | Remove Executor agent creation | ~20 |
| `specialized_agents/executor.py` | Add deprecation notice or delete | ~5 |

**Estimated effort:** ~25 lines | **Dependencies:** None | **Risk:** Trivial

---

### 2A-8: Function Tool Timeouts

**What:** Add `timeout` parameter to Blender execution tools to prevent hung pipelines.

**Why it matters:** Blender execution can hang on complex simulations (high resolution, many frames). A 5-minute timeout prevents infinite waits.

**Research support:**
- SDK Analysis §B1: "Blender execution can hang on complex simulations. A 5-minute timeout prevents infinite waits."

**Implementation:**

| File | Change | Lines |
|------|--------|-------|
| `tools/blender_executor_tools.py` | Add `timeout=300.0` to `execute_blender_script` function_tool | 1 |
| `tools/blender_executor_tools.py` | Add `failure_error_function=blender_error_handler` for actionable error messages | ~20 |

**Estimated effort:** ~21 lines | **Dependencies:** None | **Risk:** Trivial

---

### Phase 2A Summary

| Item | Lines | Cost Savings | Reliability Impact |
|------|-------|-------------|-------------------|
| 2A-1: is_enabled | ~70 | Prevents budget waste | Medium |
| 2A-2: tool_use_behavior | ~6 | ~$0.05-0.10/run | Low |
| 2A-3: Tool guardrails | ~115 | Replaces pipeline step | High |
| 2A-4: PipelineMonitor | ~320 | $0 (deterministic) | **Critical** |
| 2A-5: Parameter bounds | ~150 | Prevents wasted iterations | High |
| 2A-6: Deprecate Spec-First | ~85 | ~$0.04-0.10/run | Medium |
| 2A-7: Remove Executor agent | ~25 | Cleaner codebase | Low |
| 2A-8: Tool timeouts | ~21 | Prevents hung pipelines | Medium |
| **Total** | **~792** | **~$0.10-0.20/run** | |

**Phase 2A success criteria:**
1. No parameter oscillation across a 5-iteration run (monitor catches and clamps)
2. Budget per 3-iteration run drops below $0.30 (from ~$0.50)
3. All scripts validated by tool guardrails — zero hallucinated attributes reach Blender
4. PipelineMonitor produces status artifacts for every iteration
5. Dead code removed (Executor agent, Spec-First pipeline)

---

## Phase 2B: Core Architecture

**Goal:** Stateless iterations, precision context management, multi-grader evaluation, HITL framework.
**Risk:** Medium — changes iteration loop structure and context management.
**Principles served:** P1 (reliability), P5 (context precious), P6 (learning signal), P7 (evidence).

### 2B-1: Ralph-Style Stateless Iterations

**What:** Each iteration gets a fresh `Runner.run()` invocation with clean context. Memory persists through a structured state file written between iterations. No accumulated conversation history.

**Why it matters:** This is the **single highest-impact pattern** identified across all four research reports. The orchestrator's #1 reliability problem after hallucinations is context degradation across iterations. By iteration 3-4, context is polluted with verbose evaluation output, failed script fragments, and stale research. The Ralph pattern eliminates this entirely.

**Research support:**
- Autonomy Research §2.1 (Ralph): "CRITICAL impact. Expected: eliminates context degradation, reduces token cost per iteration by 60-80%"
- Autonomy Research §6.1: "Statelessness actually becomes a strength — each fresh context prevents the model from compounding errors"
- Monitoring Architecture §2.1: "Session compaction was completely disabled until Phase 1 fixed it" — compaction is a band-aid; stateless iterations are the cure

**Implementation:**

| File | Change | Lines |
|------|--------|-------|
| New: `utils/iteration_state.py` | `IterationState` dataclass + `write_state()` / `read_state()` — structured JSON state between iterations | ~120 |
| `orchestrator.py` (iteration loop) | Restructure: each iteration is a new `Runner.run()` call. Between iterations, write state file, read state file into next iteration's prompt. | ~200 |
| `orchestrator.py` (prompt construction) | Build iteration prompt from state file only: current script path, score history (last 3), critical issues, QA diagnosis, monitor alerts, parameter bounds | ~80 |

**State file structure:**
```json
{
  "session_id": "session_20260222_fire_001",
  "iteration": 3,
  "effect_type": "fire",
  "technique": "mantaflow_gas",
  "current_script_path": "output/fire_001/iter3/script.py",
  "current_score": 45,
  "score_history": [0, 28, 45],
  "critical_issues": ["LIGHTING_TOO_DIM"],
  "escape_level": 1,
  "learnings": ["Light energy 500 too low — need 2000+"],
  "qa_diagnosis": {
    "symptoms": ["render too dark"],
    "root_causes": ["area_light energy=500 at line 342"],
    "suggested_fixes": ["increase light energy to 2000-3000"]
  },
  "monitor_alerts": ["No oscillation detected"],
  "parameter_bounds": {"energy": [20.0, 200.0]},
  "truth_pack_types": ["FluidDomainSettings", "FluidFlowSettings"]
}
```

**Compatibility with SDK:** The SDK's `call_model_input_filter` (2B-2) provides an alternative path — clearing history between iterations within a single session. The Ralph approach is cleaner (separate Runner.run() calls) but both can coexist: Ralph for cross-iteration freshness, input filter for intra-iteration trimming.

**Estimated effort:** ~400 lines | **Dependencies:** 2A-4 (PipelineMonitor for monitor_alerts), 2A-5 (parameter bounds) | **Risk:** Medium (changes iteration loop structure — must preserve all existing failure routing)

---

### 2B-2: Context Trimming (`call_model_input_filter`)

**What:** Per-agent input filters that trim conversation history before each LLM call. Each agent type gets only the context it needs — truth pack, current iteration data, relevant feedback.

**Why it matters:** Even within a single iteration, agents accumulate tool call outputs and intermediate reasoning in their conversation history. The Script Writer doesn't need to see the Learning Agent's experiment recording. The Quality Analyst shouldn't see its own previous evaluations (prevents self-reinforcing bias).

**Research support:**
- SDK Analysis §2: "Context bloat is a known top-3 problem. This is the cheapest, most direct fix."
- Monitoring Architecture §2.2: Complete per-agent trimming strategy with token budgets
- Codebase Analysis §7.5: "Dynamic instructions inject full content every call — truth pack is static per session, shouldn't be reformatted every iteration"
- Anthropic research: "Token usage alone explains 80% of variance in performance"

**Implementation:**

| File | Change | Lines |
|------|--------|-------|
| New: `utils/context_filters.py` | Per-agent `call_model_input_filter` functions | ~80 |
| `orchestrator.py` (agent creation) | Add filters to `RunConfig` in `_run_agent()` | ~15 |

**Per-agent strategy:**

| Agent | Keep | Drop | Token Budget |
|-------|------|------|-------------|
| ScriptWriter | System + truth pack + current prompt | All previous iterations | ~8K |
| QualityAnalyst | System + current render/script path | Previous evaluations (prevents bias) | ~4K |
| LearningAgent | System + last 2 iteration summaries | Old research, old scripts | ~6K |
| ModificationStrategist | System + current prompt | Previous modification history | ~4K |
| QualityGateJudge | System + current prompt | Everything else | ~2K |

**Estimated effort:** ~95 lines | **Dependencies:** 2B-1 (Ralph iterations reduce baseline context) | **Risk:** Low

---

### 2B-3: Multi-Grader Quality Evaluation

**What:** Three-tier evaluation pipeline that runs deterministic checks ($0) before ML metrics ($0 local) before LLM vision ($0.01-0.05). Short-circuits on critical failures — no point spending $0.05 on vision if the render is black.

**Why it matters:** Currently the full quality evaluation (ML metrics + vision) runs on every render, even obviously failed ones. Deterministic pre-checks catch critical failures for free. The combined scoring is more robust than any single grader.

**Research support:**
- Autonomy Research §5.1: "Multi-grader evaluation with early termination saves budget on obviously failed renders"
- Monitoring Architecture §3.2: Complete 3-tier design with combined scoring weights
- Mission Statement §4: "Quality evaluation combining ML metrics + vision-based critique"

**Implementation:**

| File | Change | Lines |
|------|--------|-------|
| New: `tools/deterministic_quality_checks.py` | Tier 1: render exists? correct size? not blank (histogram check)? lights in script? camera bounds check? | ~180 |
| `orchestrator.py` (Phase 3) | Insert Tier 1 before ML/vision. Short-circuit: if critical issue found, skip expensive tiers. | ~50 |
| `tools/asset_evaluator_tools.py` | Modify scoring to weight: Tier 1 (pass/fail gates) + ML (60%) + Vision (40%) | ~40 |

**Short-circuit logic:**
```
Tier 1 critical issue → score=0, skip Tiers 2+3 (save $0.05)
Tier 2 score > 80, no issues → skip Tier 3 (save $0.05)
Tier 3 always runs on iteration 1 (baseline assessment)
```

**Estimated effort:** ~270 lines | **Dependencies:** None | **Risk:** Low

---

### 2B-4: AdvancedSQLiteSession (Token Tracking + Branching)

**What:** Replace `SQLiteSession` + `OpenAIResponsesCompactionSession` with `AdvancedSQLiteSession` for built-in token tracking, conversation branching for technique experiments, and keyword search.

**Why it matters:** Token tracking replaces the brittle character-count heuristic (Codebase Analysis §7.7). Branching enables trying different techniques from the same starting point without losing state — critical for escape velocity L2+ technique switches.

**Research support:**
- SDK Analysis §1: "Branching is a game-changer for technique experimentation. Token tracking replaces custom code."
- Codebase Analysis §6.1: "Replace manual sum(len(str(item))) heuristic with SDK-native token counting"
- Codebase Analysis §7.7: "Compaction trigger uses character-count heuristic that doesn't account for actual token usage"

**Implementation:**

| File | Change | Lines |
|------|--------|-------|
| `orchestrator.py:367-401` | Replace `get_or_create_sdk_session()` with `AdvancedSQLiteSession` | ~30 |
| `orchestrator.py` (technique switch) | Before switching: `branch_id = await session.create_branch_from_turn(technique_selection_turn)` | ~40 |
| `orchestrator.py` (budget tracking) | Replace custom budget tracking with `session.store_run_usage(result)` + `session.get_turn_usage()` | ~50 |
| `session_manager.py` | Add `branches: Dict[str, BranchInfo]` to SessionState | ~20 |

**Estimated effort:** ~140 lines | **Dependencies:** None (can run in parallel with 2B-1) | **Risk:** Medium (must verify compaction wrapping still works)

---

### 2B-5: HITL Framework (`needs_approval`)

**What:** Native SDK approval gates that pause the pipeline, serialize state, and resume after human review. Implements the 5 HITL checkpoints from the Mission Statement.

**Why it matters:** Ben explicitly wants this (Interview Q16, Mission Statement §9). Currently Level 4 escape velocity sets `session.status = PAUSED` and breaks the loop — no structured pause/resume flow. Budget exhaustion auto-stops with no human override option.

**Research support:**
- SDK Analysis §3: "Native pipeline pauses with state serialization"
- Mission Statement §9: "5 HITL checkpoints: prompt approval, stall detection, budget warning, critical issues, escalation"

**Implementation:**

| File | Change | Lines |
|------|--------|-------|
| `tools/blender_executor_tools.py` | Add conditional `needs_approval` (based on iteration count or budget) | ~10 |
| `tools/asset_evaluator_tools.py` | Add conditional `needs_approval` (based on budget) | ~10 |
| `orchestrator.py:_run_agent()` | Handle `result.interruptions` — serialize state, notify user | ~40 |
| New: `utils/hitl_handler.py` | Approval logic: serialize state to file, poll for approval, resume | ~100 |
| `session_manager.py` | Add `pending_approvals: List[ApprovalRequest]` to SessionState | ~15 |

**HITL triggers:**

| Checkpoint | Trigger | Implementation |
|-----------|---------|----------------|
| Stall detection | 3+ iterations, no score improvement | `needs_approval` on evaluate_render (conditional) |
| Budget warning | >80% of per-run budget spent | `needs_approval` on execute_blender_script (conditional) |
| Critical issue | BLACK_SCREEN etc. detected by monitor | Pipeline pause with diagnostic report |
| Escalation | Escape velocity L4 | Pipeline pause with full state dump |

**Estimated effort:** ~175 lines | **Dependencies:** 2A-4 (PipelineMonitor for critical issue detection) | **Risk:** Medium-High (state serialization edge cases)

---

### 2B-6: Ebbinghaus Memory Decay

**What:** Knowledge base entries decay exponentially if not reinforced by successful outcomes. Entries below retention threshold are archived (not deleted).

**Why it matters:** The KB was just wiped (Phase 1). As it accumulates new entries, decay prevents the same bloat problem from recurring. Without decay, stale patterns from early (low-quality) runs will pollute the KB within weeks.

**Research support:**
- Autonomy Research §1.3 (SAGE): "Exponential decay with reinforcement. Each successful use resets decay clock."
- Mission Statement §7: "Memory decay — knowledge entries that are not reinforced should naturally lose weight"

**Implementation:**

| File | Change | Lines |
|------|--------|-------|
| `utils/code_pattern_memory.py` | Add `compute_retention(entry, now)` — exponential decay function | ~30 |
| `tools/code_pattern_tools.py` | Filter patterns by retention score during `search_patterns()` | ~15 |
| `tools/dynamic_instructions.py` | Filter KB entries by retention during injection | ~15 |
| `tools/experiment_tracker_tools.py` | Add `reinforce_entry(entry_id)` — resets decay clock on successful use | ~20 |

**Decay function:**
```python
def compute_retention(entry, now):
    days_since = (now - entry.last_reinforced).days
    strength = entry.success_count * entry.avg_quality_score / 100
    return math.exp(-days_since / max(strength, 0.1))
```

**Thresholds:** retention < 0.3 → excluded from queries. retention < 0.1 → archived.

**Estimated effort:** ~80 lines | **Dependencies:** KB wipe (Phase 1, done) | **Risk:** Low

---

### 2B-7: Effect-Type-Scoped Evidence Gating

**What:** A pattern's trust level is tracked per effect type, not globally. A pattern that works for fire but fails for liquid gets separate trust scores.

**Why it matters:** Currently evidence gating is global — a `density=5.0` pattern that works for fire gets "Trusted" status and may be injected into a liquid script where it's wrong.

**Research support:**
- Autonomy Research §1.2: "Add effect-type scoping to evidence gating. A pattern's trust level should be tracked per effect type."

**Implementation:**

| File | Change | Lines |
|------|--------|-------|
| `tools/experiment_tracker_tools.py` | Add `effect_type` field to KB entries | ~20 |
| `tools/knowledge_distillation_tools.py` | Filter distilled patterns by effect type | ~10 |
| `tools/dynamic_instructions.py` | Filter KB injection by current effect type | ~10 |

**Estimated effort:** ~40 lines | **Dependencies:** 2B-6 (memory decay) | **Risk:** Low

---

### 2B-8: Artifact-Based Sharing Formalization

**What:** Mandate that every agent returns a structured summary (< 500 tokens) with a file path to the full report. The orchestrator never receives full reports inline.

**Why it matters:** Research output (~3000 tokens), quality evaluation (~2000 tokens), and Blender stdout (~1000-5000 tokens) all bloat context. Artifact-based sharing reduces this to ~200 tokens per handoff.

**Research support:**
- Autonomy Research §2.2 (Anthropic): "Agents store full results in files, pass only lightweight summaries. Context grows by 1 line instead of 50 paragraphs."
- Autonomy Research §3.2: "Expected 75% token reduction across all sources"
- Monitoring Architecture §2.3: Complete artifact catalog and `read_artifact` tool design

**Implementation:**

| File | Change | Lines |
|------|--------|-------|
| `orchestrator.py` (all phase prompts) | Replace inline data with artifact paths + 1-line summaries | ~80 |
| New tool: `read_artifact` in `tools/artifact_tools.py` | Agents pull detailed context on-demand | ~30 |
| Artifact outputs: `monitor_iter{N}.md`, `params_iter{N}.json`, `diagnosis_iter{N}.md`, `iteration_summary.md` | Written by orchestrator/monitor | ~40 |

**Estimated effort:** ~150 lines | **Dependencies:** 2A-4 (PipelineMonitor for monitor artifacts) | **Risk:** Low

---

### Phase 2B Summary

| Item | Lines | Impact |
|------|-------|--------|
| 2B-1: Ralph stateless iterations | ~400 | **Critical** — eliminates context degradation |
| 2B-2: Context trimming | ~95 | High — per-agent precision |
| 2B-3: Multi-grader evaluation | ~270 | High — saves $0.05/failed render |
| 2B-4: AdvancedSQLiteSession | ~140 | Medium — branching + token tracking |
| 2B-5: HITL framework | ~175 | Medium — user-requested |
| 2B-6: Memory decay | ~80 | Medium — prevents KB re-poisoning |
| 2B-7: Effect-type evidence gating | ~40 | Medium — prevents cross-effect contamination |
| 2B-8: Artifact-based sharing | ~150 | High — 75% token reduction per handoff |
| **Total** | **~1,350** | |

**Phase 2B success criteria:**
1. Known effects (fire, liquid, smoke) score >= 60 in 50%+ of runs
2. Context per agent stays within token budget (8K script writer, 4K QA, etc.)
3. 5-iteration runs show no quality cliff at iteration 3+
4. AdvancedSQLiteSession token tracking matches actual API usage within 10%
5. HITL pauses at escape level 4 with full state serialization
6. KB entries show decay over time, no stale patterns accumulate

---

## Phase 2C: Advanced Capabilities

**Goal:** Multi-physics support, technique diversity, experimentation mode.
**Risk:** Medium — extends architecture to new physics types.
**Principles served:** P2 (LLM creativity), P3 (Blender is truth), P6 (learning signal).

### 2C-1: UCB1 Technique Selector

**What:** Replace LLM-only technique selection with UCB1 (Upper Confidence Bound) as the default algorithm, with LLM override for novel prompts.

**Why it matters:** Technique selection is a classic multi-armed bandit problem. UCB1 provides optimal exploration/exploitation balance — always trying untried techniques first, then balancing between the best-performing technique and underexplored alternatives.

**Research support:**
- Autonomy Research §4.2 (IBM/AAAI 2026): "UCB1 as default, LLM override when research agent identifies specific reason to deviate"
- Mission Statement §8: "The system should actively explore Blender's capabilities"

**Implementation:**

| File | Change | Lines |
|------|--------|-------|
| New: `utils/technique_selector.py` | `select_technique_ucb1()` — UCB1 algorithm with technique statistics | ~80 |
| `orchestrator.py` (Phase 0.5) | Use UCB1 as pre-filter, then Technique Selection Coordinator for final decision with research context | ~30 |
| `tools/experiment_tracker_tools.py` | Add `get_technique_statistics(effect_type)` — per-technique run count, avg score, last used | ~40 |

**Estimated effort:** ~150 lines | **Dependencies:** 2B-7 (effect-type scoping) | **Risk:** Low

---

### 2C-2: Multi-Physics Truth Pack Extension

**What:** Extend truth pack `TECHNIQUE_TYPES` mapping to cover rigid body, particle systems, cloth, soft body, and geometry nodes.

**Why it matters:** Currently `TECHNIQUE_TYPES` covers mantaflow_gas, mantaflow_liquid, rigid_body, particle_system, plus `_common`. Missing: cloth, geometry_nodes, soft_body (Codebase Analysis §8.1). The truth pack approach is architecture-neutral — adding new physics types is a data addition, not a code change.

**Research support:**
- Codebase Analysis §8.1: "TECHNIQUE_TYPES mapping covers mantaflow_gas, mantaflow_liquid, rigid_body, particle_system, plus _common. Missing: cloth, geometry_nodes, soft_body."
- Mission Statement §15: "P1: Rigid body, particle systems. P2: Geometry nodes, cloth/soft body."

**Implementation:**

| File | Change | Lines |
|------|--------|-------|
| `tools/truth_pack.py` | Add `TECHNIQUE_TYPES` entries for `cloth`, `soft_body`, `geometry_nodes` with their respective `bpy.types` | ~30 |
| `tools/truth_pack.py` | Add `SETTINGS_MAP` entries for new types (cloth: `ClothSettings`, `ClothCollisionSettings`; geometry nodes: `GeometryNodeGroup`, etc.) | ~40 |
| `tools/parameter_bounds.py` | Add bounds for new physics types | ~60 |

**Estimated effort:** ~130 lines | **Dependencies:** Truth pack (Phase 1, done) | **Risk:** Low

---

### 2C-3: Micro-Experiment Sandbox Mode

**What:** Lightweight experiment runner that tests techniques in isolation with minimal scripts (50-100 lines, 10-30 second execution). Findings feed into the knowledge base.

**Why it matters:** When the system encounters a novel prompt or is stuck (escape level 2+), it should be able to test a technique quickly before committing to a full 500+ line scene script. This builds practical Blender experience through hands-on experimentation.

**Research support:**
- Autonomy Research §4.1: "Micro-experiments: 50-100 line scripts, 10-30 seconds, $0 cost"
- Mission Statement §8: "Sandbox mode — run small experimental scripts, introspect capabilities, record findings"
- Interview: "Even a small sandbox where agents explore different ideas practically"

**Implementation:**

| File | Change | Lines |
|------|--------|-------|
| New: `tools/micro_experiments.py` | `generate_micro_experiment()`, `run_micro_experiment()`, `ExperimentResult` dataclass | ~200 |
| `orchestrator.py` (escape velocity L2+) | Before full technique switch, run micro-experiment to validate technique works | ~40 |
| `orchestrator.py` (novel prompt) | If no matching technique in KB, run experiments on candidate techniques | ~40 |

**Triggers:**
1. Novel prompt that doesn't match known techniques
2. Escape velocity level 2+ (stuck — test alternative before full switch)
3. New truth pack version detected (Blender update)
4. User-requested exploration
5. Between production runs (idle time)

**Estimated effort:** ~280 lines | **Dependencies:** 2C-1 (UCB1 technique selector), 2C-2 (multi-physics truth pack) | **Risk:** Medium

---

### 2C-4: Agent Specialization (`Agent.clone()`)

**What:** Create effect-type-specific agent variants using `Agent.clone()`. Base agent has common tools and settings; clones have specialized instructions with known patterns, parameter ranges, and technique-specific context.

**Research support:**
- SDK Analysis §8: "Useful for specialization — dynamic instructions already solve this but clone is cleaner"

**Implementation:**

| File | Change | Lines |
|------|--------|-------|
| `specialized_agents/script_writer.py` | Define base agent, add `get_writer_for_effect(effect_type)` factory using `clone()` | ~60 |
| `orchestrator.py` | Use `get_writer_for_effect()` instead of generic Script Writer | ~10 |

**Estimated effort:** ~70 lines | **Dependencies:** 2C-2 (multi-physics types defined) | **Risk:** Low

---

### 2C-5: Streaming for Monitoring (`run_streamed()`)

**What:** Use `Runner.run_streamed()` for the Script Writer (longest-running agent) to provide real-time progress visibility and enable mid-stream hallucination detection.

**Research support:**
- SDK Analysis §7: "Real-time progress display, early hallucination detection, tool call monitoring"

**Implementation:**

| File | Change | Lines |
|------|--------|-------|
| `orchestrator.py:_run_agent()` | Add `_run_agent_streamed()` variant | ~40 |
| New: `utils/stream_monitor.py` | Event consumer: log progress, detect hallucination patterns in generated text | ~80 |
| `orchestrator.py` (Phase 1) | Use streamed variant for Script Writer | ~10 |

**Estimated effort:** ~130 lines | **Dependencies:** None | **Risk:** Low-Medium

---

### 2C-6: Graceful Budget Degradation

**What:** Instead of binary budget check (full evaluation OR stop), implement tiered evaluation: full vision ($$$) → ML-only ($0) → deterministic-only ($0) based on remaining budget.

**Research support:**
- Codebase Analysis §7.10: "No intermediate modes. Could do lightweight evaluation when budget is low."

**Implementation:**

| File | Change | Lines |
|------|--------|-------|
| `tools/asset_evaluator_tools.py` | Add `evaluate_render_lightweight()` — ML metrics only, no vision | ~40 |
| `orchestrator.py` (Phase 3) | Budget-based evaluation tier selection | ~30 |
| `tools/deterministic_quality_checks.py` | Add `evaluate_render_deterministic()` — histogram + script checks only | ~30 |

**Estimated effort:** ~100 lines | **Dependencies:** 2B-3 (multi-grader evaluation) | **Risk:** Low

---

### 2C-7: Enhanced QA Diagnosis Bridge

**What:** Extend the QA bridge with automatic parameter extraction (no keyword needed), truth-pack-enhanced ranges on every parameter, and delta feedback from the monitor.

**Research support:**
- Monitoring Architecture §3.6: "Automatic parameter extraction, truth-pack-enhanced ranges, delta feedback"
- Autonomy Research §5.2: "Deterministic script analysis — $0 cost"

**Implementation:**

| File | Change | Lines |
|------|--------|-------|
| `tools/qa_diagnosis_bridge.py` | Add `extract_all_modifiable_params()` — complete parameter inventory without issue matching | ~60 |
| `tools/qa_diagnosis_bridge.py` | Add truth-pack-enhanced ranges: `Line 245: resolution_max = 64 [range: 1-10000]` | ~30 |
| `tools/qa_diagnosis_bridge.py` | Add monitor delta: `Changed 3x in last 3 iterations. OSCILLATING. Bound: [35, 65]` | ~30 |

**Estimated effort:** ~120 lines | **Dependencies:** 2A-4 (PipelineMonitor), 2A-5 (parameter bounds) | **Risk:** Low

---

### 2C-8: Knowledge Distillation from Successful Scripts

**What:** After a run scores >= 60, extract key code sections (lighting setup, material definitions, camera placement, physics config) as named patterns with metadata.

**Research support:**
- Autonomy Research §1.4: "After a run scores >= 60, extract key code sections as named patterns"
- Mission Statement §7: "What to learn: script patterns/code snippets from successful runs"

**Implementation:**

| File | Change | Lines |
|------|--------|-------|
| `tools/knowledge_distillation_tools.py` | Add `extract_code_patterns_from_script()` — parse script into functional sections | ~100 |
| `orchestrator.py` (after quality pass) | Trigger distillation on passing scripts | ~20 |
| `tools/code_pattern_tools.py` | Enhanced `store_pattern()` with section type, effect type, quality score | ~30 |

**Estimated effort:** ~150 lines | **Dependencies:** 2B-6 (memory decay), 2B-7 (effect-type gating) | **Risk:** Low

---

### Phase 2C Summary

| Item | Lines | Impact |
|------|-------|--------|
| 2C-1: UCB1 technique selector | ~150 | High — optimal exploration/exploitation |
| 2C-2: Multi-physics truth pack | ~130 | High — enables rigid body, particles, cloth |
| 2C-3: Micro-experiment sandbox | ~280 | High — practical technique validation |
| 2C-4: Agent.clone() specialization | ~70 | Medium — cleaner than dynamic instructions |
| 2C-5: run_streamed() monitoring | ~130 | Medium — UX and hallucination detection |
| 2C-6: Graceful budget degradation | ~100 | Medium — extends budget runway |
| 2C-7: Enhanced QA bridge | ~120 | Medium — better diagnosis quality |
| 2C-8: Knowledge distillation | ~150 | Medium — learns from successes |
| **Total** | **~1,130** | |

**Phase 2C success criteria:**
1. Novel prompts (rigid body, particles) produce evaluable renders on first attempt
2. UCB1 selects different techniques for different effect types
3. Micro-experiments validate techniques in <30 seconds before full runs
4. KB accumulates effect-type-scoped patterns from successful runs
5. Budget-exhausted runs still produce deterministic quality scores

---

## Phase 2D: Full Autonomy

**Goal:** Self-improvement, autonomy progression tracking, novel prompt handling.
**Risk:** High — these are experimental capabilities.
**Principles served:** P2 (LLM creativity), P6 (learning signal), P7 (earn autonomy).

### 2D-1: Autonomy Progression Tracking

**What:** Implement the 5-level autonomy system from the Mission Statement. Track pass rates per effect type. Automatically relax HITL checkpoints as reliability improves.

**Research support:**
- Mission Statement §10: "Autonomy is earned by evidence, not time. Level 0-4 progression."

**Implementation:**

| File | Change | Lines |
|------|--------|-------|
| New: `utils/autonomy_tracker.py` | `AutonomyLevel` enum, per-effect-type tracking, level evaluation | ~120 |
| `orchestrator.py` | Check autonomy level at HITL checkpoints; skip if level >= threshold | ~30 |
| `session_manager.py` | Add autonomy metrics to session persistence | ~20 |

**Level criteria (from Mission Statement):**

| Level | Requirement | HITL Changes |
|-------|-------------|-------------|
| 0: Guided | Default | All checkpoints active |
| 1: Assisted | 10+ runs, some passes | Skip prompt approval for known types |
| 2: Semi-Autonomous | >50% pass rate, 3+ effect types | Only budget + critical issue checks |
| 3: Autonomous (Known) | >70% pass rate, 50+ runs for this type | Human sees only final result |
| 4: Autonomous (Novel) | Stable L3 + external review | Self-directed exploration |

**Estimated effort:** ~170 lines | **Dependencies:** 2B-5 (HITL framework) | **Risk:** Medium

---

### 2D-2: Cross-Session Learning Transfer

**What:** Patterns learned in one session automatically enhance future sessions. The dynamic instructions system already supports this — this work item makes it systematic with quality gates.

**Implementation:**

| File | Change | Lines |
|------|--------|-------|
| `tools/dynamic_instructions.py` | Load top-N trusted patterns (by retention score) for current effect type into agent instructions | ~40 |
| `tools/experiment_tracker_tools.py` | Add cross-session analytics: "which patterns are most effective across sessions?" | ~60 |

**Estimated effort:** ~100 lines | **Dependencies:** 2B-6 (memory decay), 2B-7 (effect-type gating), 2C-8 (distillation) | **Risk:** Low

---

### 2D-3: Prompt Versioning for Script Writer

**What:** Track which system prompt variants (instructions, truth pack format, research injection style) produce better quality scores. Auto-promote the best-performing variant.

**Research support:**
- Autonomy Research §1.1 (OpenAI cookbook): "Prompt versioning system with rollback"

**Implementation:**

| File | Change | Lines |
|------|--------|-------|
| New: `utils/prompt_versioning.py` | Version tracking, A/B scoring, auto-promotion | ~150 |
| `tools/dynamic_instructions.py` | Use versioned instruction templates | ~30 |

**Estimated effort:** ~180 lines | **Dependencies:** 2B-3 (multi-grader eval for consistent scoring) | **Risk:** Medium

---

### 2D-4: Adaptive Replanning (Magentic-One Style)

**What:** Replace the fixed state machine with an adaptive pipeline that can jump back to research/technique selection when evaluation reveals the technique is fundamentally wrong.

**Research support:**
- Autonomy Research §2.3 (Magentic-One): "When the plan is revised, all agents clear their contexts and reset states"

**Implementation:**

| File | Change | Lines |
|------|--------|-------|
| New: `utils/progress_ledger.py` | 5-question check: task complete? looping? progress? who next? what instruction? | ~80 |
| `orchestrator.py` (iteration loop) | Replace fixed phase sequence with ledger-driven routing | ~120 |

**Estimated effort:** ~200 lines | **Dependencies:** 2B-1 (Ralph iterations), 2A-4 (PipelineMonitor) | **Risk:** High

---

### 2D-5: LLM-Based Anomaly Detection (Full MASC)

**What:** Train a lightweight anomaly detector on normal pipeline trajectories. When anomaly score exceeds threshold, a correction agent intervenes.

**Research support:**
- Autonomy Research §2.4 (MASC): "77.84% AUC-ROC on step-level error detection"

**Implementation:** Deferred. The deterministic PipelineMonitor (2A-4) handles 95% of cases. Full MASC only needed if deterministic monitoring proves insufficient.

**Estimated effort:** ~300 lines | **Dependencies:** All Phase 2A-C | **Risk:** High

---

### Phase 2D Summary

| Item | Lines | Impact |
|------|-------|--------|
| 2D-1: Autonomy progression | ~170 | Medium — earned independence |
| 2D-2: Cross-session learning | ~100 | Medium — cumulative improvement |
| 2D-3: Prompt versioning | ~180 | Medium — instruction optimization |
| 2D-4: Adaptive replanning | ~200 | High — flexible pipeline |
| 2D-5: Full MASC (deferred) | ~300 | Medium — only if needed |
| **Total** | **~950** | |

**Phase 2D success criteria:**
1. Autonomy level progression demonstrated: at least one effect type reaches Level 2
2. System demonstrably improves over time (same prompt scores higher after 50 runs vs 5)
3. Prompt versioning shows measurable quality improvement
4. Novel prompts produce reasonable first attempts with correct physics system

---

## Dependency Graph

```
Phase 1 (DONE)
│
├── 2A-1: is_enabled ──────────────────────────────────────────────┐
├── 2A-2: tool_use_behavior ───────────────────────────────────────┤
├── 2A-3: Tool guardrails ─────────────────────────────────────────┤
├── 2A-4: PipelineMonitor ─────────┬───────────────────────────────┤
├── 2A-5: Parameter bounds ────────┤                               │
├── 2A-6: Deprecate Spec-First ────┤                               │
├── 2A-7: Remove Executor agent ───┤                               │
├── 2A-8: Tool timeouts ───────────┘                               │
│                                                                  │
│   ┌──── 2A-4 + 2A-5 ────┐                                       │
│   │                      │                                       │
│   ▼                      ▼                                       │
├── 2B-1: Ralph iterations ──────────────────┐                     │
├── 2B-2: Context trimming ──────────────────┤ (depends on 2B-1)   │
├── 2B-3: Multi-grader eval ─────────────────┤                     │
├── 2B-4: AdvancedSQLiteSession ─────────────┤ (parallel w/ 2B-1)  │
├── 2B-5: HITL framework ───────────────────┤ (depends on 2A-4)   │
├── 2B-6: Memory decay ─────────────────────┤                     │
├── 2B-7: Effect-type evidence ──────────────┤ (depends on 2B-6)   │
├── 2B-8: Artifact-based sharing ────────────┘ (depends on 2A-4)   │
│                                                                  │
│   ┌──── 2B-6 + 2B-7 ────┐                                       │
│   │                      │                                       │
│   ▼                      ▼                                       │
├── 2C-1: UCB1 technique ──────────┐                               │
├── 2C-2: Multi-physics truth pack ┤                               │
├── 2C-3: Micro-experiments ───────┤ (depends on 2C-1 + 2C-2)     │
├── 2C-4: Agent.clone() ──────────┤ (depends on 2C-2)             │
├── 2C-5: run_streamed() ─────────┤                               │
├── 2C-6: Budget degradation ──────┤ (depends on 2B-3)            │
├── 2C-7: Enhanced QA bridge ──────┤ (depends on 2A-4 + 2A-5)     │
├── 2C-8: Knowledge distillation ──┘ (depends on 2B-6 + 2B-7)     │
│                                                                  │
│   ┌──── All Phase 2A-C ─┐                                       │
│   ▼                      │                                       │
├── 2D-1: Autonomy tracking ───────┤ (depends on 2B-5)            │
├── 2D-2: Cross-session learning ──┤ (depends on 2C-8)            │
├── 2D-3: Prompt versioning ───────┤ (depends on 2B-3)            │
├── 2D-4: Adaptive replanning ─────┤ (depends on 2B-1 + 2A-4)    │
└── 2D-5: Full MASC (deferred) ────┘ (depends on all above)       │
```

**Parallelism opportunities within phases:**
- 2A: Items 1-3 can be implemented in parallel. Item 4 (monitor) can run in parallel with 1-3. Items 5-8 can run in parallel.
- 2B: Items 1 and 4 can run in parallel. Items 6 and 7 are sequential. Item 8 depends on 4.
- 2C: Items 1 and 2 can run in parallel. Items 5 and 6 can run in parallel.

---

## Conflict Resolutions

Where different research agents recommended different approaches for the same problem, these decisions were made:

### 1. Context Management: Ralph vs AdvancedSQLiteSession Branching

**Conflict:** Ralph pattern (Autonomy Research) says fresh context per iteration. AdvancedSQLiteSession (SDK Analysis) enables branching within a conversation.

**Resolution:** Use BOTH — they're complementary. Ralph for iteration-level freshness (2B-1): each iteration is a new `Runner.run()` with clean context built from a state file. AdvancedSQLiteSession for session-level tracking (2B-4): token counting, technique branching, persistence. The session wraps the iterations; iterations don't accumulate within the session.

### 2. Monitoring: RunHooks vs PipelineMonitor vs Full MASC

**Conflict:** EnforcementHooks (existing) work at intra-agent level. PipelineMonitor (Monitoring Architecture) works at inter-iteration level. MASC (Autonomy Research) provides ML-based anomaly detection.

**Resolution:** Three layers, each at the right scope:
- **RunHooks (existing):** Intra-agent loop detection, turn budgets, tool counting. No changes needed.
- **PipelineMonitor (2A-4):** Inter-iteration monitoring — oscillation, cascades, budget. Deterministic, $0.
- **Full MASC (2D-5):** Only if deterministic monitoring proves insufficient. Deferred.

The Monitoring Architecture correctly identified that RunHooks can't see across Runner.run() invocations.

### 3. Spec-First Pipeline: Deprecate or Keep as Safety Net?

**Conflict:** Codebase Analysis recommends deprecation. The existing Code Writer guardrail depends on APISpec models.

**Resolution:** Deprecate the LLM-powered Spec-First pipeline (2A-6) but preserve the Pydantic models. The `truth_pack_to_api_spec()` function populates APISpec from truth pack data, maintaining backward compatibility. The Code Writer guardrail continues to work. Net effect: same validation, $0 cost.

### 4. Orchestrator Monolith: Refactor Now or Later?

**Conflict:** Codebase Analysis §7.1 recommends extracting phases into modules. But Ralph iterations (2B-1) will change the iteration loop significantly.

**Resolution:** Defer orchestrator decomposition to Phase 2C or later. Reason: Ralph iterations (2B-1) will restructure the iteration loop, and decomposing before that change would mean decomposing twice. Do the Ralph change first, then decompose the stabilized structure.

### 5. Quality Evaluation: Who Decides the Score?

**Conflict:** Multiple research reports suggest different scoring approaches.

**Resolution:** The 3-tier approach from the Monitoring Architecture (2B-3) is the best synthesis:
- Tier 1 (deterministic): Hard gates. Critical issues = auto-fail regardless of other scores. $0.
- Tier 2 (ML metrics): 60% weight. Objective, reproducible, local. $0.
- Tier 3 (LLM vision): 40% weight. Subjective but valuable. $0.01-0.05.
- Short-circuit: critical issue in Tier 1 → skip Tiers 2-3.

This matches the OpenAI self-evolving agents cookbook pattern (4 complementary graders with early termination).

---

## Seven Pillars Coverage

Mapping every work item to the seven pillars from the task description:

### SDK Integration
- 2A-1: `is_enabled` | 2A-2: `tool_use_behavior` | 2A-3: Tool guardrails | 2A-8: Tool timeouts
- 2B-2: `call_model_input_filter` | 2B-4: `AdvancedSQLiteSession` | 2B-5: `needs_approval`
- 2C-4: `Agent.clone()` | 2C-5: `run_streamed()`

### Self-Learning
- 2B-6: Ebbinghaus memory decay | 2B-7: Effect-type evidence gating
- 2C-8: Knowledge distillation | 2D-2: Cross-session learning | 2D-3: Prompt versioning

### Monitoring & Observability
- 2A-4: PipelineMonitor | 2A-5: Parameter bounds
- 2C-5: run_streamed() | 2C-7: Enhanced QA bridge | 2D-5: Full MASC

### Context Management
- 2B-1: Ralph stateless iterations | 2B-2: Context trimming | 2B-8: Artifact-based sharing
- 2A-6: Deprecate Spec-First (reduces context from spec output)

### Multi-Physics Support
- 2C-1: UCB1 technique selector | 2C-2: Multi-physics truth pack
- 2C-3: Micro-experiments | 2C-4: Agent specialization

### Autonomy Progression
- 2D-1: Autonomy tracking (L0-L4) | 2D-4: Adaptive replanning
- 2B-5: HITL framework (the foundation for relaxing checkpoints)

### Experimentation Mode
- 2C-3: Micro-experiment sandbox | 2C-1: UCB1 exploration/exploitation
- 2D-2: Cross-session learning (experiments feed into future runs)

---

## Budget Impact Analysis

**Current cost per 3-iteration run:** ~$0.30-0.70 (15-46 LLM calls)

| Phase | Savings | New Costs | Net Impact |
|-------|---------|-----------|------------|
| **2A** | -$0.10-0.20/run (stop_on_first_tool, deprecate Spec-First, is_enabled) | $0 (all deterministic) | **-$0.10-0.20** |
| **2B** | -$0.05-0.15/run (Ralph reduces context tokens, multi-grader skips vision on failures) | $0 (all deterministic or SDK-native) | **-$0.05-0.15** |
| **2C** | -$0.02-0.05/run (budget degradation, agent specialization) | +$0.01-0.02/run (micro-experiments use Blender time, not API) | **-$0.01-0.03** |
| **2D** | -$0.01-0.03/run (prompt versioning finds cheaper prompts) | +$0.01-0.02/run (autonomy tracking, MASC) | **~$0** |

**Projected cost per 3-iteration run after Phase 2B:** $0.15-0.35
**Monthly run capacity at $20 budget:** 57-133 runs (up from 29-67)

---

## Implementation Sequence (Recommended)

```
Week 1: Phase 2A Quick Wins
├── Day 1-2: 2A-1 (is_enabled) + 2A-2 (tool_use_behavior) + 2A-8 (timeouts)
│            [All trivial, independent, ship together]
├── Day 2-3: 2A-3 (tool guardrails)
│            [Depends on truth pack, moderate effort]
├── Day 3-4: 2A-6 (deprecate Spec-First) + 2A-7 (remove Executor)
│            [Independent cleanup]
└── Day 4-7: 2A-4 (PipelineMonitor) + 2A-5 (parameter bounds)
             [Most complex 2A items, depend on nothing]

Week 2-3: Phase 2B Core Architecture
├── Week 2: 2B-1 (Ralph iterations) + 2B-4 (AdvancedSQLiteSession) [parallel]
│           2B-6 (memory decay) [parallel, independent]
├── Week 3: 2B-2 (context trimming) [depends on 2B-1]
│           2B-3 (multi-grader eval) [independent]
│           2B-7 (effect-type evidence) [depends on 2B-6]
│           2B-8 (artifact sharing) [depends on 2A-4]
└── Week 3-4: 2B-5 (HITL framework) [depends on 2A-4]
              [E2E validation of Phase 2A+2B together]

Week 4-8: Phase 2C Advanced Capabilities
├── 2C-1 + 2C-2 [parallel: UCB1 + multi-physics truth pack]
├── 2C-3 (micro-experiments) [depends on 2C-1 + 2C-2]
├── 2C-4 + 2C-5 + 2C-6 [parallel: clone, streaming, budget degradation]
├── 2C-7 (enhanced QA bridge) [depends on 2A-4]
└── 2C-8 (knowledge distillation) [depends on 2B-6 + 2B-7]

Week 8+: Phase 2D Full Autonomy
├── 2D-1 (autonomy tracking) [depends on 2B-5]
├── 2D-2 + 2D-3 [parallel: cross-session learning + prompt versioning]
├── 2D-4 (adaptive replanning) [depends on 2B-1]
└── 2D-5 (full MASC) [deferred, only if needed]
```

---

## End Goals (From Mission Statement §14)

| Goal | Phase That Delivers It | How We Measure |
|------|----------------------|----------------|
| Every run produces a render | Phase 1 (done) + 2A (monitoring catches remaining failures) | 95%+ completion rate across 20 runs |
| Known effects score >= 60 | Phase 2B (Ralph + multi-grader + context management) | 50%+ pass rate for fire, liquid, smoke |
| Novel prompts produce reasonable first attempts | Phase 2C (multi-physics + UCB1 + micro-experiments) | First attempt scores >= 30 for untried effect types |
| System demonstrably improves over time | Phase 2C-2D (distillation + cross-session learning + prompt versioning) | Same prompt scores higher after 50 runs vs 5 |
| Failed runs produce useful diagnostics | Phase 2A (PipelineMonitor) + 2B (multi-grader) | Every failed run produces structured diagnostic artifact |
| Budget respected | Phase 2A (is_enabled + stop_on_first_tool) + 2C (degradation) | Cost per 3-iteration run < $0.35 |

---

## Appendix A: New File Inventory

| File | Phase | Lines (est) | Purpose |
|------|-------|-------------|---------|
| `utils/tool_visibility.py` | 2A | ~40 | is_enabled callback functions |
| `guardrails/tool_guardrails.py` | 2A | ~100 | Truth pack + execution + script length guardrails |
| `tools/pipeline_monitor.py` | 2A | ~250 | Deterministic PipelineMonitor class |
| `tools/parameter_bounds.py` | 2A | ~120 | Per-effect-type bounds + damped convergence |
| `utils/iteration_state.py` | 2B | ~120 | Ralph-style stateless iteration state |
| `utils/context_filters.py` | 2B | ~80 | Per-agent call_model_input_filter functions |
| `tools/deterministic_quality_checks.py` | 2B | ~180 | Tier 1 quality checks (histogram, lights, camera) |
| `utils/hitl_handler.py` | 2B | ~100 | HITL approval/resume logic |
| `tools/artifact_tools.py` | 2B | ~30 | read_artifact tool for agents |
| `utils/technique_selector.py` | 2C | ~80 | UCB1 algorithm |
| `tools/micro_experiments.py` | 2C | ~200 | Sandbox experiment runner |
| `utils/stream_monitor.py` | 2C | ~80 | Streaming event consumer |
| `utils/autonomy_tracker.py` | 2D | ~120 | L0-L4 autonomy progression |
| `utils/prompt_versioning.py` | 2D | ~150 | Prompt version tracking + A/B scoring |
| `utils/progress_ledger.py` | 2D | ~80 | 5-question adaptive routing |

**Total new files: 15 | Total new lines: ~1,730**

## Appendix B: Modified File Inventory

| File | Phases | Total Changes (est) |
|------|--------|-------------------|
| `orchestrator.py` | 2A, 2B, 2C, 2D | ~700 lines across all phases |
| `session_manager.py` | 2B, 2D | ~55 lines |
| `tools/asset_evaluator_tools.py` | 2A, 2B, 2C | ~95 lines |
| `tools/dynamic_instructions.py` | 2B, 2D | ~65 lines |
| `tools/experiment_tracker_tools.py` | 2B, 2C | ~80 lines |
| `tools/blender_executor_tools.py` | 2A | ~36 lines |
| `specialized_agents/quality_analyst.py` | 2A | ~5 lines |
| `specialized_agents/docs_expert.py` | 2A | ~5 lines |
| `specialized_agents/executor.py` | 2A | ~5 lines (deprecate) |
| `specialized_agents/api_validator.py` | 2A | ~1 line |
| `specialized_agents/learning_agent.py` | 2A | ~1 line |
| `specialized_agents/script_writer.py` | 2C | ~60 lines |
| `tools/code_pattern_tools.py` | 2B | ~15 lines |
| `tools/knowledge_distillation_tools.py` | 2B, 2C | ~40 lines |
| `tools/truth_pack.py` | 2C | ~70 lines |
| `tools/qa_diagnosis_bridge.py` | 2C | ~120 lines |
| `tools/script_generator_tools.py` | 2A | ~5 lines |

**Total modified lines: ~1,358**

## Appendix C: Source Cross-References

Every recommendation traces back to at least one research document:

| Pattern/Feature | SDK Analysis | Codebase Analysis | Autonomy Research | Monitoring Architecture |
|----------------|:---:|:---:|:---:|:---:|
| is_enabled | §4 | §7.9, §7.10 | | |
| tool_use_behavior | §5 | §6.5 | §3.1 | |
| Tool guardrails | §6 | §3.3, §7.8 | | |
| call_model_input_filter | §2 | §7.5 | §3.2 | §2.2 |
| AdvancedSQLiteSession | §1 | §6.1, §7.7 | | |
| needs_approval | §3 | §6.7 | | |
| Agent.clone() | §8 | | | |
| run_streamed() | §7 | §6.6 | | |
| Function tool timeouts | §B1 | | | |
| PipelineMonitor | | §7.6 | §2.4 (MASC) | §1.2, §1.3 |
| Parameter bounds | | | §5.3 | §3.3 |
| Ralph iterations | | | §2.1 | §2.1 |
| Multi-grader eval | | | §5.1 | §3.2 |
| UCB1 technique | | | §4.2 | |
| Memory decay | | | §1.3 (SAGE) | |
| Effect-type gating | | | §1.2 | |
| Artifact sharing | | | §2.2 (Anthropic) | §2.3 |
| Micro-experiments | | | §4.1 | |
| Autonomy tracking | | | | |
| Prompt versioning | | | §1.1 (OpenAI) | |
| Adaptive replanning | | | §2.3 (Magentic-One) | |
| Deprecate Spec-First | | §7.4 | | |
| Remove Executor | | §7.3 | §3.1 | |
| Budget degradation | | §7.10 | | |
| Knowledge distillation | | | §1.4 | |
