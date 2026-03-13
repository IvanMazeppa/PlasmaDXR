# Architecture Review Response - Wave 2 Reassessment

Date: 2026-03-13

This document updates the prior architecture reviews using:
- `docs/20260312_WAVE2_STATUS_AND_PATH_FORWARD.md`
- the live repo state after the GPT-5.4 rollout
- the current `openai-agents==0.12.0` runtime surface
- the implemented section-patching system and current troubleshooting state

## 1. EXECUTIVE SUMMARY

Wave 1 is no longer hypothetical. `TechniqueContract`, the 7-pack registry, GPT-5.4 rollout, truth-pack hardening, and `call_model_input_filter` are implemented, and the first Cell Fracture run proves the contract architecture works. That changes the architecture question: the main blocker is no longer "can the system bind technique?" but "can the iteration loop preserve the value of patching and repeated improvement?" Right now the answer is not yet, because stale render reuse can turn a failed execution into a fake successful evaluation and block escalation to `modify_code` or `switch_technique`. The SDK upgrade to 0.12.0 also changes the HITL recommendation: native tool approvals and resumable `RunState` exist in the installed runtime, so the roadmap should move from a fully custom HITL assumption to a hybrid model. The updated roadmap should focus Wave 2 on state-integrity, patch-path proof, and measurement; Wave 3 remains gated on execution failure dropping below 25% across multiple physics types.

## 2. ROOT CAUSE ANALYSIS

1. Symptom: iteration 2+ can evaluate an old image and repeat the same score after a new script crashes. Root cause: the execution/evaluation boundary currently treats "a render exists somewhere on disk" as equivalent to "this iteration produced a valid render." Evidence: `_discover_render()` scans sibling directories and subdirectories in `phases/execution.py:50-78`, and `_apply_render_discovery()` can promote a failed run to `success=True` when any render is found in `phases/execution.py:111-133`. This is now the dominant implementation blocker because it corrupts the feedback loop itself.

2. Symptom: section patching is implemented and budgeted, but it has not yet proven itself end-to-end in production runs. Root cause: the `modify_code` path depends on correct failure propagation, and the stale-render path can prevent the pipeline from ever reaching structural escalation. Evidence: section-patching is wired into the `modify_code` branch in `orchestrator.py:1742-1814`, patch budgets are already enforced there, but the Wave 2 status report shows the real destruction run never reached that path because repeated evaluation of a stale render masked the need to escalate.

3. Symptom: the current Wave 2 HITL recommendation assumes native Python approvals are unavailable, even after the SDK upgrade. Root cause: the roadmap is now partially based on stale SDK assumptions and stale runtime-truth docs. Evidence: `requirements.txt` now pins `openai-agents==0.12.0`, but `docs/AI_OPERATION_MANUAL.md:12` still says `0.10.5` and `docs/RUNTIME_TRUTH_AND_DOC_GROUNDING_2026-02-13.md:9` still says `0.8.3`; meanwhile the installed 0.12.0 runtime exposes `needs_approval` on `function_tool` and `Agent.as_tool` in `venv/.../agents/tool.py:271-278` and `venv/.../agents/agent.py:487-517`, plus `RunResult.to_state()` in `venv/.../agents/result.py:310-332` and `RunState.approve()/reject()` in `venv/.../agents/run_state.py:137-279`.

4. Symptom: research still burns turns and costs more than it should, even though output quality is acceptable when it completes. Root cause: the system still uses conversational retrieval where it already knows the desired control flow: plan -> bounded retrieval -> synthesis. Evidence: the Research Agent remains open-ended in both prompt shape and execution model, and the Wave 2 status report explicitly reclassifies bounded retrieval from "Wave 1 blocker" to "Wave 2 optimization," which is correct but still leaves a real efficiency gap.

5. Symptom: quality scores are still below the pass threshold, but the architecture is no longer obviously wrong. Root cause: the remaining quality ceiling is currently underdetermined because the system lacks clean current-stack baselines and true multi-iteration data on failed runs. Evidence: cloth improved from 34 to 41 with only parameter updates, destruction stalled at 12 because the same stale image was scored repeatedly, and Mantaflow has not yet been re-baselined after the Wave 1 + Wave 2 changes.

6. Symptom: the project still talks about `AdvancedSQLiteSession` as if it were required for section patching. Root cause: the roadmap had not yet caught up with the simpler implementation that is already working. Evidence: the Wave 2 status report says section patching, budget counters, and orchestrator integration are done without `AdvancedSQLiteSession`; branching is now optional for later A/B exploration, not a dependency for the current repair budget.

## 3. ARCHITECTURAL INTERVENTIONS

1. What changes: make render validity iteration-scoped instead of path-scoped. Add a `partial_render_path` field to `ExecutionOutput`, only accept renders from the current `run_dir` or a current-iteration manifest, and never promote `success=False` to `success=True` just because an image exists on disk. Why it fixes the root cause: it restores feedback-loop integrity and lets the Quality Gate see real failures instead of stale artifacts. Estimated complexity: Low-Medium.

```python
# models/pipeline_models.py
class ExecutionOutput(BaseModel):
    success: bool
    render_path: str | None = None
    partial_render_path: str | None = None
    run_dir: str | None = None
    error_message: str | None = None


# phases/execution.py
def _apply_render_discovery(execution: ExecutionOutput) -> None:
    current_render = _discover_render_current_run(execution.run_dir)
    if execution.success:
        execution.render_path = execution.render_path or current_render
        return

    # Never treat a failed iteration as successful because an older render exists.
    execution.partial_render_path = current_render
    execution.render_path = None
```

2. What changes: treat section patching as a Wave 2 proof obligation, not a completed checkbox. After stale-render invalidation is fixed, force at least one end-to-end `modify_code` run in cloth, destruction, or Mantaflow and verify patch budgets, section targeting, and recovery telemetry in production traces. Why it fixes the root cause: the architecture should not promote section patching to solved status until the real escalation path has exercised it. Estimated complexity: Low.

3. What changes: adopt a hybrid HITL architecture. Keep `utils/hitl_handler.py` for semantic checkpoints such as post-eval stall, quality plateau, and escape-level escalation; add native 0.12.0 tool approvals for tool-local decisions such as high-cost regeneration, technique switch, or budget-increasing actions. Why it fixes the root cause: the pipeline-level handler remains the right abstraction for session decisions, while native approvals eliminate custom pause/resume plumbing for local tool-level approval points. Estimated complexity: Medium.

```python
from agents import Agent, Runner, function_tool


@function_tool(needs_approval=True)
async def switch_technique(technique: str, reason: str) -> str:
    return f"Approved technique switch to {technique}: {reason}"


result = await Runner.run(orchestrator_agent, "try the next technique")
if result.interruptions:
    state = result.to_state()
    for item in result.interruptions:
        state.approve(item)
    result = await Runner.run(orchestrator_agent, state)
```

4. What changes: demote `AdvancedSQLiteSession` from Wave 2 dependency to optional later infrastructure. Keep the existing Python-level 2-per-iteration / 4-per-session patch budget for now, and reserve `AdvancedSQLiteSession` for later technique A/B testing, branch search, or persistent what-if exploration. Why it fixes the root cause: it prevents over-engineering and keeps attention on the blocker that is actually breaking iteration today. Estimated complexity: Low.

5. What changes: implement bounded research as a staged flow instead of an open conversation. Create a retriever step that stops after the required bundle lookup, run independent retrievals in parallel where useful, and then do a compact synthesis step that produces `ResearchOutput` once. Why it fixes the root cause: it cuts turn waste, reduces cost, and makes research behavior more deterministic without changing the rest of the architecture. Estimated complexity: Medium.

```python
retriever = Agent(
    name="ResearchRetriever",
    tools=[blender_doc_search_bundle, list_capability_packs],
    tool_use_behavior={"stop_at_tool_names": ["blender_doc_search_bundle"]},
)

# Orchestrator:
# 1. planning call
# 2. parallel retrieval calls
# 3. single synthesis call -> ResearchOutput
```

6. What changes: add a current-stack re-baseline loop before making more Wave 3 decisions. Re-run Mantaflow on the GPT-5.4 + truth-pack + context-filter + section-guidance stack, measure cross-physics execution failure over 20+ runs, and calibrate evaluator score ranges using known-good assets. Why it fixes the root cause: it converts today’s quality discussion from speculation into controlled measurement and gives the project a real gate for Wave 3. Estimated complexity: Low-Medium.

7. What changes: refresh runtime-truth documentation immediately after the SDK/model upgrade. Update the authoritative docs to reflect `openai-agents==0.12.0`, GPT-5.4 rollout, native approvals, and the actual current orchestration shape. Why it fixes the root cause: the architecture process is now being slowed down by stale truth docs, not just stale code. Estimated complexity: Low.

## 4. PROPOSED ROADMAP

1. Wave 2A - Restore Iteration Integrity. Content: fix stale render reuse, add explicit current-iteration render validity, surface partial renders only as diagnostics, and rerun at least one destruction case until `modify_code` or `switch_technique` is reached for real. Dependencies: none beyond the current stack. Exit criteria: failed executions never reuse old renders for scoring, and section patching has been exercised end-to-end on a real pipeline run.

2. Wave 2B - Update Human Control and Measurement. Content: layer native `needs_approval`/`RunState` approvals onto selected tools while retaining pipeline HITL checkpoints, refresh runtime-truth docs, and run the current-stack baselines: Mantaflow re-baseline, cross-physics execution benchmark, and evaluator calibration. Dependencies: Wave 2A, because baselines and approval routing are not trustworthy while stale render reuse exists. Exit criteria: HITL is hybrid rather than custom-only, current-stack score ranges exist, and cross-physics execution failure is measured instead of estimated.

3. Wave 2C - Reduce Cost and Friction. Content: implement bounded research retrieval, use `StopAtTools`/`stop_on_first_tool` and parallel retrieval where appropriate, and keep orchestrator extraction incremental as stable boundaries emerge. Dependencies: Wave 2A is enough to start this work, but Wave 2B gives better measurement. Exit criteria: research no longer exhausts turns conversationally, tool surfaces are tighter, and the monolith does not regrow while fixes land.

4. Wave 3 - Raise the Quality Ceiling Only After Reliability Is Proven. Content: deterministic scene assembly for tabletop/contact scenes first, then container liquids and room interiors; move learning to structured deltas against contracts, packs, and section-level patterns; consider pack composition as a first-class planner concern only if measured prompt distribution justifies it. Dependencies: cross-physics execution failure below 25% across 20+ runs and calibrated evaluator baselines. Exit criteria: geometry quality work is being applied to trustworthy iterations rather than masked failures.

Continuous workstream: keep `TechniqueContract`, pack coverage, and truth-pack coverage improving, but do not let those tracks distract from iteration-state integrity. They are now multiplicative improvements, not the main blocker.

## 5. SDK FEATURES TO LEVERAGE

1. `needs_approval` + `RunResult.to_state()` + `RunState.approve()/reject()`: this is the main roadmap change introduced by the 0.12.0 upgrade. Use it for tool-local approvals; do not keep assuming Python-native approvals are unavailable.

2. `call_model_input_filter`: keep it as a permanent part of the architecture. It is no longer a planned optimization; it is part of the baseline system.

3. `tool_use_behavior="stop_on_first_tool"` and `StopAtTools`: these are the right SDK primitives for bounded research helpers and single-purpose retrievers.

4. `tool_input_guardrails` and `tool_output_guardrails`: use them to harden section naming, capability-pack availability, and current-iteration artifact validity before the pipeline reaches recovery or evaluation.

5. `parallel_tool_calls`: useful inside bounded retrieval stages, not as a cure for open-ended conversations.

6. `defer_loading` on function tools: now worth considering for rarely used or expensive tool surfaces so the active tool vocabulary stays smaller in code-critical phases.

7. `AdvancedSQLiteSession`: still available in 0.12.0, but no longer a near-term requirement. Defer it until there is a real need for branch history beyond the existing repair-budget counters.

8. `run_streamed`: useful for visibility and long-run monitoring, but it still sits below render validity, recovery proof, and baseline measurement in priority.

## 6. WHAT TO STOP DOING

1. Stop treating any render found on disk as proof the current iteration succeeded.

2. Stop promoting failed executions to `success=True` during fallback render discovery.

3. Stop calling section patching "done" until `modify_code` has succeeded on a real recovery path.

4. Stop planning HITL as if the Python SDK still lacks native approvals and resumable run state.

5. Stop treating `AdvancedSQLiteSession` as mandatory for the current patch budget design.

6. Stop running open-ended research conversations for pack-selection and doc-bundle tasks that already have a bounded shape.

7. Stop making Wave 3 decisions from invalid or stale iteration data.

8. Stop leaving runtime-truth documents pinned to old SDK and model versions after major upgrades.

9. Stop assuming quality stagnation is purely creative-model weakness before the evaluator is recalibrated on the current stack.

10. Stop spending architecture attention on spatial solvers before the iteration loop is trustworthy.

## 7. OPEN QUESTIONS

1. Which tool actions should use native 0.12.0 approvals first: technique switch, budget increase, full regeneration, or `patch_script_section` after repeated failures?

2. What exact artifact should certify a render as "current-iteration valid": `run_dir`, timestamp, script hash, iteration id, or a small manifest file written by the executor?

3. Does section patching need stronger section metadata than function names alone, or is the current canonical section convention sufficient once production traces exist?

4. What is the smallest cross-physics benchmark that still gives a trustworthy execution-failure rate: 20 runs, 30 runs, or the full 30-prompt validation set?

5. At what measured rate of multi-physics prompts should pack composition move from "later" to "planned Wave 3 work"?
