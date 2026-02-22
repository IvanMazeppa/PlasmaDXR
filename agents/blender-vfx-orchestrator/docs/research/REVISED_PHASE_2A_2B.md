## Phase 2A: Quick Wins

**Goal:** Enable technique discovery, immediate cost savings, deterministic safety nets, monitoring infrastructure.
**Risk:** Low — all changes are additive or replace redundant code.
**Principles served:** P1 (reliability), P3 (Blender is truth), P4 (compute what you can), P5 (context precious).

### 2A-0: Documentation Pipeline (LLM-Optimized Manual)

**What:** Process the physics section of the Blender manual (119 pages) through an LLM rewrite pipeline, upload to the vector store, and validate that the research agent can discover new techniques.

**Why it matters:** This is the **highest-ROI item in the entire roadmap**. Without it, the self-learning pipeline has nothing to learn from. The research agent defaults to LLM training data — which contains 3-4 well-known Mantaflow techniques. Every downstream item that depends on technique diversity (UCB1, micro-experiments, multi-physics, cross-session learning) is building on sand without this.

**The complete Research → Learning pipeline requires BOTH sources:**

```
MANUAL (LLM-optimized)          API REFERENCE (truth pack)
        │                                │
        ▼                                ▼
   "HOW/WHY/WHEN"                  "WHAT EXISTS"
   Techniques, workflows,          Attribute names, types,
   parameter relationships         valid ranges, defaults
        │                                │
        └──────────┬─────────────────────┘
                   ▼
           RESEARCH AGENT → TECHNIQUE SELECTION → SCRIPT GENERATION
```

The top-left box — "MANUAL (LLM-optimized)" — is what this work item delivers.

**Evidence: Manual rewrite experiment (already run)**

| Page | Original Words | Rewritten Words | Compression | Cost |
|------|---------------|----------------|-------------|------|
| Domain Settings | 4,041 | 898 | 78% smaller | $0.003 |
| Flow | 2,387 | 919 | 61% smaller | $0.002 |
| Gas (index) | 37 | 281 | Expanded (sparse) | $0.001 |
| Noise | 756 | 451 | 40% smaller | $0.001 |
| Cache | 1,751 | 723 | 59% smaller | $0.002 |
| **Total** | **8,972** | **3,272** | **64% avg** | **$0.008** |

The rewritten output produces dense, structured content with explicit TECHNIQUE and GOTCHAS sections — exactly what the research agent needs.

**Prerequisite fix (already done):** Three bugs in `semantic_docs_tools.py` were found and fixed during the current session:
- API results were filling a shared quota before manual queries ran — manual content was completely blocked
- Manual queries weren't explicitly routed to the manual vector store
- No technique discovery queries existed — the system never asked "what alternative methods exist for X?"

**Research support:**
- Critique §1: "This is the highest ROI work item in the entire Phase 2 plan"
- Mission Statement §4: "The manual should be optimized for LLM consumption"
- Mission Statement §12: "Technique monotony — System defaults to Mantaflow for everything"

**Implementation:**

| File | Change | Lines |
|------|--------|-------|
| `scripts/experiment_manual_rewrite.py` | Already exists — extend with `--all-physics` flag for batch processing + `--upload` flag to push to vector store | ~75 |
| New: `scripts/upload_rewritten_manual.py` | Upload rewritten pages to OpenAI vector store, verify upload, report stats | ~120 |
| `tools/semantic_docs_tools.py` | Update vector store IDs to include rewritten manual store (already partially done via doc search fix) | ~15 |
| New: `tests/test_technique_discovery.py` | Validate research agent discovers techniques it previously couldn't | ~30 |

**Workflow:**
1. Run `python scripts/experiment_manual_rewrite.py --all-physics --output rewritten_manual/` (~$0.19)
2. Review a sample of output for quality
3. Run `python scripts/upload_rewritten_manual.py --input rewritten_manual/`
4. Update `semantic_docs_tools.py` with new vector store ID
5. Run validation test: ask "what alternative techniques exist for creating fire effects?" → must return beyond basic mantaflow_gas

**Tests:**
1. Batch processing: run `--all-physics` on 5 pages → assert all produce non-empty output with TECHNIQUE section
2. Upload validation: upload 5 pages → assert vector store query returns results from rewritten content
3. Technique discovery: query "alternative fire techniques" → assert results include at least 2 techniques beyond mantaflow_gas
4. No regression: existing API reference queries still return relevant results (doc search fix preserved)
5. Cost check: processing 119 pages costs < $0.30

**Rollback:** Revert vector store ID to original manual store in `semantic_docs_tools.py`. Rewritten content can be deleted from the vector store via API.

**Estimated effort:** ~240 lines new code + $0.19 processing cost | **Dependencies:** None | **Risk:** Low

---

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
| `specialized_agents/quality_analyst.py` | Add `is_enabled=budget_allows_vision` to vision evaluation tool | ~8 |
| `specialized_agents/docs_expert.py` | Add `is_enabled=budget_allows_docs` to doc search tools | ~8 |
| `orchestrator.py` (coordinator creation) | Wrap expensive agent-as-tool calls with `is_enabled` callbacks | ~15 |
| `tools/asset_evaluator_tools.py` | Move budget check from guardrail to `is_enabled` | ~15 |
| New: `utils/tool_visibility.py` | Callback functions for budget/phase/escape-level gating | ~60 |

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

**Tests:**
1. Budget exhausted: set budget to $0 → assert vision tool not in agent.tools list
2. Budget available: set budget to $10 → assert vision tool IS in agent.tools list
3. Iteration gating: iteration=1 → assert mine_docs_for_patterns is hidden; iteration=3 → assert visible
4. Transition: exhaust budget mid-run → assert next agent call has tool hidden

**Rollback:** Set `ENABLE_CONDITIONAL_TOOLS = False` in `config/agent_config.py`. When False, all `is_enabled` callbacks return True.

**Estimated effort:** ~106 lines | **Dependencies:** None | **Risk:** Trivial

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
| `orchestrator.py` (quality gate coordinator) | Add `StopAtTools(stop_at_tool_names=["make_quality_decision"])` | 5 |

**Tests:**
1. Executor: call with valid script → assert only 1 LLM call made (tool call, no follow-up)
2. API Validator: call with script → assert only 1 LLM call
3. Quality Gate: call with score data → assert LLM processes eval tools then stops at make_quality_decision
4. Cost comparison: run 3 iterations with vs without → assert measurable cost reduction

**Rollback:** Remove `tool_use_behavior` parameter from agent definitions. Agents revert to default `run_llm_again` behavior.

**Estimated effort:** ~8 lines | **Dependencies:** None | **Risk:** Trivial
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
| New: `guardrails/tool_guardrails.py` | Truth pack input guardrail, execution output guardrail, script length output guardrail | ~150 |
| `tools/blender_executor_tools.py` | Wrap `execute_blender_script` with truth pack input guardrail | ~8 |
| `tools/asset_evaluator_tools.py` | Wrap `evaluate_render` with output guardrail for critical failures | ~8 |
| `tools/script_generator_tools.py` | Wrap `generate_script` with output guardrail for script length check | ~8 |

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

**Tests:**
1. Hallucinated attribute: script with `resolution_divisions` → assert guardrail catches and auto-fixes to `resolution_max`
2. Clean script: script with all valid attributes → assert guardrail allows execution
3. Error recovery path: trigger Phase 2.5 recovery → assert recovery script is also validated by guardrail
4. Script length warning: generate 200-line script → assert output guardrail warns "scene too basic"
5. Critical failure output: render produces BLACK_SCREEN → assert output guardrail rejects with targeted message

**Rollback:** Set `ENABLE_TOOL_GUARDRAILS = False` in `config/agent_config.py`. When False, tools are created without guardrail wrappers, reverting to pipeline-level validation.

**Estimated effort:** ~174 lines | **Dependencies:** Truth pack (Phase 1, done) | **Risk:** Low

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
| New: `tools/pipeline_monitor.py` | `PipelineMonitor` class with `check_after_generation()`, `check_after_evaluation()`, `get_status_report()` | ~375 |
| `orchestrator.py` (iteration loop) | Add monitor checkpoint calls after generation and after evaluation | ~60 |
| `orchestrator.py` (iteration loop) | Wire monitor alerts into modification pipeline (parameter clamping, cascade revert) | ~45 |

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

**Tests:**
1. Oscillation detection: feed values [50, 2500, 10] → assert `is_oscillating() == True`
2. No false positive: feed values [50, 100, 150] → assert `is_oscillating() == False`
3. Cascade detection: score drops 8 points after QA suggestion → assert WARNING alert with revert action
4. Budget alert: $0.60 spent on $0.50 budget → assert CRITICAL alert
5. Integration: mock a 3-iteration loop with synthetic data → verify monitor produces status artifact at each checkpoint

**Rollback:** Set `ENABLE_PIPELINE_MONITOR = False` in `config/agent_config.py`. When False, monitor checkpoint calls in orchestrator are skipped.

**Estimated effort:** ~480 lines | **Dependencies:** None | **Risk:** Low

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
| New: `tools/parameter_bounds.py` | `ParameterBound` dataclass, `PARAMETER_BOUNDS` dict per effect type, `clamp()` and `damped_change()` methods | ~180 |
| `orchestrator.py` (modification phase) | Apply bounds before passing parameter changes to Script Writer | ~45 |

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

**Tests:**
1. Clamp: propose energy=5000 with max=200 → assert clamped to 200
2. Damped change: current=50, target=200, step_size=50 → assert result=100 (moved by step_size, not full jump)
3. Within bounds: propose energy=150 within [20, 200] → assert unchanged
4. Effect type routing: fire bounds applied for fire, NOT applied for liquid
5. Integration with monitor: oscillating parameter gets tighter bounds from monitor + static bounds → most restrictive wins

**Rollback:** Set `ENABLE_PARAMETER_BOUNDS = False` in `config/agent_config.py`. When False, bounds enforcement is skipped.

**Estimated effort:** ~225 lines | **Dependencies:** PipelineMonitor (2A-4) for runtime bounds refinement | **Risk:** Low

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
| `orchestrator.py` (Phase 0.5) | Remove `_run_parallel_technique_and_spec()` parallel branch for API Spec. Technique selection runs alone. | ~45 |
| `orchestrator.py` (Phase 1) | Remove `_run_spec_first_pipeline()` code path. Script Writer gets truth pack directly via dynamic instructions. | ~75 |
| `specialized_agents/api_spec_agent.py` | Add deprecation notice at top. Keep file for reference but don't import. | ~8 |

**What's preserved:** `models/api_spec.py` Pydantic models stay — `truth_pack_to_api_spec()` populates them from truth pack data, maintaining backward compatibility with the Code Writer guardrail.

**Tests:**
1. Script Writer context: remove Spec-First → assert Script Writer still receives truth pack data via dynamic instructions
2. Code Writer guardrail: assert `truth_pack_to_api_spec()` produces valid APISpec from truth pack data
3. Technique selection: assert technique selection still runs (was previously parallel with API Spec)

**Rollback:** Set `ENABLE_SPEC_FIRST_PIPELINE = True` in `config/agent_config.py`. When True, restores parallel API Spec + Technique Selection.

**Estimated effort:** ~128 lines changed | **Dependencies:** Truth Pack (Phase 1, done) | **Risk:** Low-Medium (must verify Code Writer still gets good context)
**Savings:** ~$0.04-0.10/run (2 LLM calls eliminated)

---

### 2A-7: Remove Dead Executor Agent

**What:** Phase 2 (Execution) calls `_execute_blender_script_impl()` directly — deterministic, no LLM. The Executor agent (170 LOC) is created in `initialize()` but never used in the pipeline path. Remove it.

**Research support:**
- Codebase Analysis §7.3: "Dead code. Executor agent is created but never used in the pipeline path."

**Implementation:**

| File | Change | Lines |
|------|--------|-------|
| `orchestrator.py` (initialize) | Remove Executor agent creation | ~30 |
| `specialized_agents/executor.py` | Add deprecation notice or delete | ~8 |

**Tests:**
1. E2E pipeline: run full pipeline → assert execution still works via `_execute_blender_script_impl()` direct call
2. No import: assert `orchestrator.py` does not import from `specialized_agents/executor.py`

**Rollback:** Not needed — this is dead code removal. If Executor is needed later, the file is in git history.

**Estimated effort:** ~38 lines | **Dependencies:** None | **Risk:** Trivial

---

### 2A-8: Function Tool Timeouts

**What:** Add `timeout` parameter to Blender execution tools to prevent hung pipelines.

**Why it matters:** Blender execution can hang on complex simulations (high resolution, many frames). A 5-minute timeout prevents infinite waits.

**Research support:**
- SDK Analysis §B1: "Blender execution can hang on complex simulations. A 5-minute timeout prevents infinite waits."

**Implementation:**

| File | Change | Lines |
|------|--------|-------|
| `tools/blender_executor_tools.py` | Add `timeout=300.0` to `execute_blender_script` function_tool | 2 |
| `tools/blender_executor_tools.py` | Add `failure_error_function=blender_error_handler` for actionable error messages | ~30 |

**Tests:**
1. Timeout fires: mock Blender subprocess that hangs → assert timeout error returned to agent within 300s
2. Normal execution: run valid script → assert completes well under timeout
3. Error handler: simulate MemoryError → assert agent receives actionable message "Reduce resolution_max or particle count"

**Rollback:** Remove `timeout` parameter — reverts to no timeout (original behavior).

**Estimated effort:** ~32 lines | **Dependencies:** None | **Risk:** Trivial

---

### Phase 2A Summary

| Item | Lines | Cost Savings | Reliability Impact | Tests |
|------|-------|-------------|-------------------|-------|
| **2A-0: Documentation Pipeline** | **~240** | **Enables technique diversity** | **CRITICAL** | **5** |
| 2A-1: is_enabled | ~106 | Prevents budget waste | Medium | 4 |
| 2A-2: tool_use_behavior | ~8 | ~$0.05-0.10/run | Low | 4 |
| 2A-3: Tool guardrails | ~174 | Replaces pipeline step | High | 5 |
| 2A-4: PipelineMonitor | ~480 | $0 (deterministic) | **Critical** | 5 |
| 2A-5: Parameter bounds | ~225 | Prevents wasted iterations | High | 5 |
| 2A-6: Deprecate Spec-First | ~128 | ~$0.04-0.10/run | Medium | 3 |
| 2A-7: Remove Executor agent | ~38 | Cleaner codebase | Low | 2 |
| 2A-8: Tool timeouts | ~32 | Prevents hung pipelines | Medium | 3 |
| **Total** | **~1,431** | **~$0.10-0.20/run** | | **36** |

**Note:** Line estimates include a 1.5x multiplier based on Phase 1 experience (estimated 1,200 lines, shipped 1,574).

**Phase 2A success criteria:**
1. Research agent discovers >= 3 new techniques per effect type from rewritten manual (2A-0)
2. No parameter oscillation across a 5-iteration run (monitor catches and clamps)
3. Budget per 3-iteration run drops below $0.30 (from ~$0.50)
4. All scripts validated by tool guardrails — zero hallucinated attributes reach Blender
5. PipelineMonitor produces status artifacts for every iteration
6. Dead code removed (Executor agent, Spec-First pipeline)

---

## Phase 2B: Core Architecture

**Goal:** Stateless iterations, precision context management, multi-grader evaluation, HITL framework.
**Risk:** Medium — changes iteration loop structure and context management.
**Principles served:** P1 (reliability), P5 (context precious), P6 (learning signal), P7 (evidence).

**Schedule note:** 2B-1 (Ralph-Style Stateless Iterations) gets its own dedicated sprint in Week 2 with isolated E2E testing. All other 2B items are Week 3+. Ralph is the highest-risk change in Phase 2 and must not be bundled with AdvancedSQLiteSession or memory decay.

### 2B-1: Ralph-Style Stateless Iterations

**What:** Each iteration gets a fresh `Runner.run()` invocation with clean context. Memory persists through a structured state file written between iterations. No accumulated conversation history.

**Why it matters:** This is the **single highest-impact pattern** identified across all four research reports. The orchestrator's #1 reliability problem after hallucinations is context degradation across iterations. By iteration 3-4, context is polluted with verbose evaluation output, failed script fragments, and stale research. The Ralph pattern eliminates this entirely.

**Research support:**
- Autonomy Research §2.1 (Ralph): "CRITICAL impact. Expected: eliminates context degradation, reduces token cost per iteration by 60-80%"
- Autonomy Research §6.1: "Statelessness actually becomes a strength — each fresh context prevents the model from compounding errors"
- Monitoring Architecture §2.1: "Session compaction was completely disabled until Phase 1 fixed it" — compaction is a band-aid; stateless iterations are the cure

**Implementation strategy — INCREMENTAL:**

1. **Week 2, Day 1-2:** Write `IterationState` dataclass + serialization
2. **Week 2, Day 2-3:** Convert ONLY iteration 2 to stateless. Iteration 1 stays as-is (baseline for A/B comparison).
3. **Week 2, Day 3-4:** Run E2E — compare iteration 2's behavior between old (accumulated context) and new (fresh context).
4. **Week 2, Day 4-5:** If quality holds or improves, convert all remaining iterations.
5. **Week 3:** Only THEN proceed to other 2B items that build on stable Ralph foundation.

| File | Change | Lines |
|------|--------|-------|
| New: `utils/iteration_state.py` | `IterationState` dataclass + `write_state()` / `read_state()` — structured JSON state between iterations | ~180 |
| `orchestrator.py` (iteration loop) | Restructure: each iteration is a new `Runner.run()` call. Between iterations, write state file, read state file into next iteration's prompt. | ~300 |
| `orchestrator.py` (prompt construction) | Build iteration prompt from state file only: current script path, score history (last 3), critical issues, QA diagnosis, monitor alerts, parameter bounds | ~120 |

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

**Tests:**
1. State roundtrip: write state → read state → assert all fields preserved exactly
2. Fresh context: iteration 2 starts with only state file content, NOT iteration 1 conversation → assert context size < 5K tokens
3. A/B comparison: run same prompt twice — once with accumulated context (old), once with Ralph (new) → assert Ralph score >= old score
4. Failure routing: trigger execution error in iteration 2 → assert error recovery still works across Runner.run() boundary
5. SharedContext rebuild: assert truth_pack, session state, stuck_state all survive across Runner.run() boundaries

**Rollback:** Set `ENABLE_RALPH_ITERATIONS = False` in `config/agent_config.py`. When False, iteration loop reverts to accumulated context (original behavior).

**Estimated effort:** ~600 lines | **Dependencies:** 2A-4 (PipelineMonitor for monitor_alerts), 2A-5 (parameter bounds) | **Risk:** Medium (changes iteration loop structure — must preserve all existing failure routing)

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
| New: `utils/context_filters.py` | Per-agent `call_model_input_filter` functions | ~120 |
| `orchestrator.py` (agent creation) | Add filters to `RunConfig` in `_run_agent()` | ~23 |

**Per-agent strategy:**

| Agent | Keep | Drop | Token Budget |
|-------|------|------|-------------|
| ScriptWriter | System + truth pack + current prompt | All previous iterations | ~8K |
| QualityAnalyst | System + current render/script path | Previous evaluations (prevents bias) | ~4K |
| LearningAgent | System + last 2 iteration summaries | Old research, old scripts | ~6K |
| ModificationStrategist | System + current prompt | Previous modification history | ~4K |
| QualityGateJudge | System + current prompt | Everything else | ~2K |

**Tests:**
1. ScriptWriter filter: inject 10 messages → assert filter keeps only system + last message
2. QualityAnalyst isolation: inject previous QA results → assert filter strips them (prevents self-reinforcing bias)
3. Token budget: assert each agent's filtered input is within its token budget
4. No data loss: assert current iteration's critical data (score, issues, script path) survives filtering

**Rollback:** Set `ENABLE_CONTEXT_FILTERS = False` in `config/agent_config.py`. When False, agents receive unfiltered context (original behavior).

**Estimated effort:** ~143 lines | **Dependencies:** 2B-1 (Ralph iterations reduce baseline context) | **Risk:** Low

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
| New: `tools/deterministic_quality_checks.py` | Tier 1: render exists? correct size? not blank (histogram check)? lights in script? camera bounds check? | ~270 |
| `orchestrator.py` (Phase 3) | Insert Tier 1 before ML/vision. Short-circuit: if critical issue found, skip expensive tiers. | ~75 |
| `tools/asset_evaluator_tools.py` | Modify scoring to weight: Tier 1 (pass/fail gates) + ML (60%) + Vision (40%) | ~60 |

**Short-circuit logic:**
```
Tier 1 critical issue → score=0, skip Tiers 2+3 (save $0.05)
Tier 2 score > 80, no issues → skip Tier 3 (save $0.05)
Tier 3 always runs on iteration 1 (baseline assessment)
```

**Tests:**
1. Black screen: blank render image → assert Tier 1 catches, Tiers 2+3 skipped, score=0
2. Good render: quality render → assert all 3 tiers run, combined score reflects weighted average
3. High ML score: Tier 2 score=85 → assert Tier 3 skipped (except on iteration 1)
4. Cost tracking: run 5 iterations with 2 critical failures → assert 2 vision calls skipped
5. Scoring weights: ML=70, Vision=50 → assert combined = 70*0.6 + 50*0.4 = 62.0

**Rollback:** Set `ENABLE_MULTI_GRADER = False` in `config/agent_config.py`. When False, evaluation reverts to single-grader (vision+ML always runs).

**Estimated effort:** ~405 lines | **Dependencies:** None | **Risk:** Low

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
| `orchestrator.py:367-401` | Replace `get_or_create_sdk_session()` with `AdvancedSQLiteSession` | ~45 |
| `orchestrator.py` (technique switch) | Before switching: `branch_id = await session.create_branch_from_turn(technique_selection_turn)` | ~60 |
| `orchestrator.py` (budget tracking) | Replace custom budget tracking with `session.store_run_usage(result)` + `session.get_turn_usage()` | ~75 |
| `session_manager.py` | Add `branches: Dict[str, BranchInfo]` to SessionState | ~30 |

**Tests:**
1. Session swap: replace SQLiteSession with AdvancedSQLiteSession → assert pipeline still runs end-to-end
2. Token tracking: run 3 iterations → assert `get_session_usage()` returns non-zero total_tokens matching actual API usage within 15%
3. Branching: create branch at turn 2 → assert original conversation preserved, branch starts from turn 2
4. Compaction: verify compaction wrapping still works with AdvancedSQLiteSession as underlying store

**Rollback:** Set `ENABLE_ADVANCED_SESSION = False` in `config/agent_config.py`. When False, reverts to SQLiteSession + OpenAIResponsesCompactionSession.

**Estimated effort:** ~210 lines | **Dependencies:** None (can run in parallel with 2B-1) | **Risk:** Medium (must verify compaction wrapping still works)

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
| `tools/blender_executor_tools.py` | Add conditional `needs_approval` (based on iteration count or budget) | ~15 |
| `tools/asset_evaluator_tools.py` | Add conditional `needs_approval` (based on budget) | ~15 |
| `orchestrator.py:_run_agent()` | Handle `result.interruptions` — serialize state, notify user | ~60 |
| New: `utils/hitl_handler.py` | Approval logic: serialize state to file, poll for approval, resume | ~150 |
| `session_manager.py` | Add `pending_approvals: List[ApprovalRequest]` to SessionState | ~23 |

**HITL triggers:**

| Checkpoint | Trigger | Implementation |
|-----------|---------|----------------|
| Stall detection | 3+ iterations, no score improvement | `needs_approval` on evaluate_render (conditional) |
| Budget warning | >80% of per-run budget spent | `needs_approval` on execute_blender_script (conditional) |
| Critical issue | BLACK_SCREEN etc. detected by monitor | Pipeline pause with diagnostic report |
| Escalation | Escape velocity L4 | Pipeline pause with full state dump |

**Tests:**
1. Approval gate fires: set budget to 80% spent → assert `needs_approval` triggers on next execution
2. State serialization: trigger approval → assert state file written with all pipeline state
3. Resume from state: serialize → deserialize → assert pipeline continues from exact same point
4. Timeout: trigger approval, don't respond for 5 min → assert pipeline pauses gracefully (no crash)
5. Always_approve: approve with `always_approve=True` → assert subsequent calls skip approval

**Rollback:** Set `ENABLE_HITL = False` in `config/agent_config.py`. When False, all `needs_approval` conditions return False.

**Estimated effort:** ~263 lines | **Dependencies:** 2A-4 (PipelineMonitor for critical issue detection) | **Risk:** Medium-High (state serialization edge cases)

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
| `utils/code_pattern_memory.py` | Add `compute_retention(entry, now)` — exponential decay function | ~45 |
| `tools/code_pattern_tools.py` | Filter patterns by retention score during `search_patterns()` | ~23 |
| `tools/dynamic_instructions.py` | Filter KB entries by retention during injection | ~23 |
| `tools/experiment_tracker_tools.py` | Add `reinforce_entry(entry_id)` — resets decay clock on successful use | ~30 |

**Decay function:**
```python
def compute_retention(entry, now):
    days_since = (now - entry.last_reinforced).days
    strength = entry.success_count * entry.avg_quality_score / 100
    return math.exp(-days_since / max(strength, 0.1))
```

**Thresholds:** retention < 0.3 → excluded from queries. retention < 0.1 → archived.

**Tests:**
1. Fresh entry: created today, retention = 1.0 → assert included in queries
2. Stale entry: last reinforced 60 days ago, low strength → assert retention < 0.3, excluded from queries
3. Reinforcement: use pattern successfully → assert `last_reinforced` resets, retention jumps back to ~1.0
4. Archive threshold: retention < 0.1 → assert entry moved to archive, not deleted

**Rollback:** Set `ENABLE_MEMORY_DECAY = False` in `config/agent_config.py`. When False, all entries have retention=1.0 (no decay).

**Estimated effort:** ~121 lines | **Dependencies:** KB wipe (Phase 1, done) | **Risk:** Low

---

### 2B-7: Effect-Type-Scoped Evidence Gating

**What:** A pattern's trust level is tracked per effect type, not globally. A pattern that works for fire but fails for liquid gets separate trust scores.

**Why it matters:** Currently evidence gating is global — a `density=5.0` pattern that works for fire gets "Trusted" status and may be injected into a liquid script where it's wrong.

**Research support:**
- Autonomy Research §1.2: "Add effect-type scoping to evidence gating. A pattern's trust level should be tracked per effect type."

**Implementation:**

| File | Change | Lines |
|------|--------|-------|
| `tools/experiment_tracker_tools.py` | Add `effect_type` field to KB entries | ~30 |
| `tools/knowledge_distillation_tools.py` | Filter distilled patterns by effect type | ~15 |
| `tools/dynamic_instructions.py` | Filter KB injection by current effect type | ~15 |

**Tests:**
1. Scoped storage: store pattern for "fire" → assert pattern has effect_type="fire"
2. Scoped retrieval: query patterns for "liquid" → assert fire-only patterns NOT returned
3. Cross-type: pattern used successfully for both fire and liquid → assert separate trust scores per type
4. Dynamic instructions: current effect_type="fire" → assert only fire-scoped patterns injected

**Rollback:** Set `ENABLE_EFFECT_SCOPING = False` in `config/agent_config.py`. When False, all patterns treated as global (original behavior).

**Estimated effort:** ~60 lines | **Dependencies:** 2B-6 (memory decay) | **Risk:** Low

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
| `orchestrator.py` (all phase prompts) | Replace inline data with artifact paths + 1-line summaries | ~120 |
| New tool: `read_artifact` in `tools/artifact_tools.py` | Agents pull detailed context on-demand | ~45 |
| Artifact outputs: `monitor_iter{N}.md`, `params_iter{N}.json`, `diagnosis_iter{N}.md`, `iteration_summary.md` | Written by orchestrator/monitor | ~60 |

**Tests:**
1. Token reduction: compare prompt size with inline data vs artifact paths → assert >50% reduction
2. Artifact readability: write artifact → read with `read_artifact` tool → assert content matches
3. Agent access: agent calls `read_artifact` → assert full report content returned (capped at 4K chars)
4. Missing artifact: agent calls `read_artifact` on non-existent path → assert graceful error message

**Rollback:** Set `ENABLE_ARTIFACT_SHARING = False` in `config/agent_config.py`. When False, prompts include inline data (original behavior).

**Estimated effort:** ~225 lines | **Dependencies:** 2A-4 (PipelineMonitor for monitor artifacts) | **Risk:** Low

---

### Phase 2B Summary

| Item | Lines | Impact | Tests | Sprint |
|------|-------|--------|-------|--------|
| **2B-1: Ralph stateless iterations** | **~600** | **Critical — eliminates context degradation** | **5** | **Week 2 (ISOLATED)** |
| 2B-2: Context trimming | ~143 | High — per-agent precision | 4 | Week 3 |
| 2B-3: Multi-grader evaluation | ~405 | High — saves $0.05/failed render | 5 | Week 3 |
| 2B-4: AdvancedSQLiteSession | ~210 | Medium — branching + token tracking | 4 | Week 3 (parallel w/ 2B-1 testing) |
| 2B-5: HITL framework | ~263 | Medium — user-requested | 5 | Week 4 |
| 2B-6: Memory decay | ~121 | Medium — prevents KB re-poisoning | 4 | Week 3 |
| 2B-7: Effect-type evidence gating | ~60 | Medium — prevents cross-effect contamination | 4 | Week 3 (after 2B-6) |
| 2B-8: Artifact-based sharing | ~225 | High — 75% token reduction per handoff | 4 | Week 3 |
| **Total** | **~2,027** | | **35** | |

**Note:** Line estimates include a 1.5x multiplier based on Phase 1 experience.

**Phase 2B schedule (revised for Ralph isolation):**
```
Week 2: 2B-1 (Ralph iterations) ALONE
├── Day 1-2: IterationState dataclass + serialization
├── Day 2-3: Convert iteration 2 only to stateless
├── Day 3-4: E2E A/B comparison: old vs new iteration 2
├── Day 4-5: If quality holds, convert all iterations
└── E2E validation: 3-iteration fire run with full monitoring

Week 3: All remaining 2B items
├── 2B-4 (AdvancedSQLiteSession) — can start parallel with Week 2
├── 2B-6 (memory decay) — independent
├── 2B-2 (context trimming) — depends on stable 2B-1
├── 2B-3 (multi-grader eval) — independent
├── 2B-7 (effect-type evidence) — depends on 2B-6
└── 2B-8 (artifact sharing) — depends on 2A-4

Week 4: 2B-5 (HITL framework)
├── Depends on 2A-4 (PipelineMonitor)
└── E2E validation of Phase 2A+2B together
```

**Phase 2B success criteria:**
1. Known effects (fire, liquid, smoke) score >= 60 in 50%+ of runs
2. Context per agent stays within token budget (8K script writer, 4K QA, etc.)
3. 5-iteration runs show no quality cliff at iteration 3+ (Ralph eliminates context degradation)
4. AdvancedSQLiteSession token tracking matches actual API usage within 15%
5. HITL pauses at escape level 4 with full state serialization and successful resume
6. KB entries show decay over time, no stale patterns accumulate
7. All 35 Phase 2B tests pass
