# Autonomy‑Critical Changes (2026-01-23)

**Purpose:** This document lays out the **specific changes** required for the system to *actually* perform the tasks end‑to‑end (not just appear to). The emphasis is on deterministic enforcement, not hopeful prompting.

---

## Implementation Status (Updated 2026-01-23)

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 0 | ✅ COMPLETE | All 3 quick wins implemented |
| Phase 1 | ✅ COMPLETE | Items 1-5, 8-9 fully implemented; 6-7 deferred |
| Phase 1.5 | 🔄 PARTIAL | Learning Agent enhanced but not mandatory pre-generation |
| Phase 2 | ⏳ PENDING | Pattern outcome reporting in place; reuse pending |
| Phase 3 | ⏳ PENDING | Research output still free-text |
| Phase 4 | ⏳ PENDING | Optional advanced features |

---

## Executive Summary (Cold Truth)
Right now the loop can run, but several **core autonomy mechanisms are either missing or non‑binding**. The system relies on LLM compliance for actions that must be deterministic. This prevents real self‑improvement and makes “emergent behavior” unreliable. The changes below make the learning loop *mechanically true* instead of “suggested”.

**Note:** Dynamic instructions are now implemented (see `docs/DYNAMIC_INSTRUCTIONS_GUIDE_2026-01-23.md`). That is a **major enabler**, but it does not replace deterministic enforcement in the orchestrator.

---

## Phase 0 — Quick Wins (Low risk, high leverage) ✅ COMPLETE

### QW‑1) Add budget checks before expensive phases ✅
**Why:** Prevents burning budget after it's already exhausted.
**Impact:** Immediate cost safety.
**Complexity:** Low.

**Implementation (2026-01-23):**
- Added `BUDGET_EXHAUSTED` status to `SessionStatus` enum in `models/shared_context.py`
- Added budget check before Phase 3 (Quality Evaluation) in `orchestrator.py`
- Pipeline exits gracefully with last available score when budget exhausted

### QW‑2) Log dynamic‑instruction fallbacks ✅
**Why:** Silent KB failures kill learning; logging makes it visible.
**Impact:** Immediate observability.
**Complexity:** Low.

**Implementation (2026-01-23):**
- Added `logging` module to `tools/dynamic_instructions.py`
- All context extraction failures now logged via `logger.warning()`
- Includes error details and which fallback was used

### QW‑3) Reset stuck state on technique switch ✅
**Why:** New techniques shouldn't inherit old failure streaks.
**Impact:** Immediate exploration benefit.
**Complexity:** Low.

**Implementation (2026-01-23):**
- Added `IssueTracker.reset()` method in `session_manager.py`
- Added `SessionManager.reset_for_technique_switch()` method
- Orchestrator calls reset when technique switch detected

---

## Phase 1 — Make Autonomy Real (Deterministic loop control) ✅ COMPLETE

### 1) Mandatory Pre‑Iteration Research (Not Optional) ✅

### Problem
The pipeline does not call `pre_iteration_research()` before script modification. This means early warning signals are ignored, so the system repeats known failures before escalating.

### Change
Insert a **required** pre‑iteration research step for `iteration > 1`, **before** Phase 1 modification.

### Why it matters
This is the difference between a reactive loop and a *self‑correcting* loop. The system should proactively look for alternatives before wasting another iteration.

**Implementation (2026-01-23):**
- Added `pre_iteration_research_direct()` in `tools/proactive_research_tools.py` (callable without `@function_tool` wrapper)
- Added Phase 0.9 in `orchestrator.py` that runs before Phase 1 when `iteration > 1`
- Checks `warning_level` and handles `escape_action` (query_kb, force_technique_switch, increase_exploration, request_human_help)
- Records baseline score before each iteration via `record_baseline()`

### Example (conceptual)
```
if iteration > 1:
    research = pre_iteration_research(
        current_issue=quality.primary_issue,
        current_approach=previous_script.technique_used,
        iteration_history=iteration_summary,
        effect_type=request.effect_type.value,
    )
    if research["warning_level"] != "none":
        # Execute escape_action immediately
```

---

### 2) Escape Level Must Drive Decisions ✅

### Problem
The quality gate produces an `escape_level`, but the **SessionManager context never includes it**, so downstream logic sees `escape_level=0` forever.

### Change
Persist the escape level into SessionManager (or a single shared stuck‑state) **immediately after** QualityGateJudge.

### Why it matters
Without a live escape level, "stuck detection" is a mirage. The system can't prove that it learns from repeated failures.

**Implementation (2026-01-23):**
- Added `stuck_state` field to `SharedContext` in `models/shared_context.py`
- Orchestrator syncs escape level immediately after QualityGateJudge:
  ```python
  session.stuck_state.escape_level = EscapeLevel(gate_decision.escape_level)
  context.stuck_state.escape_level = EscapeLevel(gate_decision.escape_level)
  ```
- Both `SessionState` and `SharedContext` now share live escape level

### Example (conceptual)
```
if gate_decision:
    session_mgr.sync_escape_level(gate_decision.escape_level)
```

---

### 3) Pattern Extraction Must Be Enforced by the Orchestrator ✅

### Problem
The Learning Agent is instructed to extract patterns, but the pipeline does **nothing** with `learning.pattern_extracted` or `learning.pattern_id`.

### Change
After Learning Agent returns, the orchestrator must:
1. Record extracted patterns in session state.
2. Report pattern outcomes after execution.

### Why it matters
Self‑improvement requires *persistent knowledge*. If success doesn't create a reusable pattern, the system is not learning.

**Implementation (2026-01-23):**
- Added `extracted_patterns: List[Dict]` field to `SessionState` in `models/shared_context.py`
- Added `last_applied_pattern_id: Optional[str]` field to `SharedContext`
- Orchestrator records extracted patterns after Learning Agent:
  ```python
  if learning.pattern_extracted and learning.pattern_id:
      session.extracted_patterns.append({...})
      context.last_applied_pattern_id = learning.pattern_id
  ```
- Added `_report_pattern_outcome_impl()` in `tools/code_pattern_tools.py` for direct orchestrator calls
- Orchestrator reports pattern outcomes after quality evaluation

### Example (conceptual)
```
if learning.pattern_extracted and learning.pattern_id:
    session.extracted_patterns.append(learning.pattern_id)

# after evaluation
if script.pattern_id:
    report_pattern_outcome(script.pattern_id, success, improvement)
```

---

### 4) Technique Switching Must Reset Stuck State ✅

### Problem
When switching techniques, the system keeps the old issue streaks. This means a new technique inherits stale "stuck" penalties.

### Change
Reset the issue tracker when a new technique is selected.

### Why it matters
Exploration is impossible if every new idea is penalized by old failures. This is one of the biggest blockers to emergent behavior.

**Implementation (2026-01-23):**
- Added `IssueTracker.reset()` method in `session_manager.py`
- Added `SessionManager.reset_for_technique_switch()` method
- Resets: `issue_counts`, `issue_first_seen`, `consecutive_same_issue`, `last_primary_issue`
- Orchestrator calls `session_mgr.reset_for_technique_switch()` when technique switch detected
- New technique added to `techniques_tried` list to prevent re-trying

### Example (conceptual)
```
if switched_technique:
    session_mgr.reset_issue_tracker()
```

---

### 5) Dynamic Instructions Must Never Fail Silently ✅

### Problem
Dynamic instructions swallow exceptions and fall back without logging. That makes KB‑injection failures invisible.

### Change
Add lightweight logging when context extraction fails, and fallback to a safe default.

### Why it matters
If KB‑injection fails, the system stops learning *silently*. Silent failure kills self‑improvement.

**Implementation (2026-01-23):**
- Added `import logging` and `logger = logging.getLogger(__name__)` to `tools/dynamic_instructions.py`
- All 4 dynamic instruction functions now log warnings on context extraction failure:
  - `dynamic_script_writer_instructions()`
  - `dynamic_quality_analyst_instructions()`
  - `dynamic_learning_instructions()`
  - `dynamic_docs_expert_instructions()`
- Log includes error details and fallback value used

### Example (conceptual)
```
try:
    effect_type = ctx.context.session.request.effect_type.value
except Exception as e:
    logger.warning("Dynamic instructions context failed", exc_info=e)
    effect_type = "general"
```

---

### 6) Do Not Bypass Enforcement When Modifying Scripts ⏳ DEFERRED

### Problem
Direct `_modify_script_impl` bypasses RunHooks, guardrails, and tool tracing.

### Change
Route *all* parameter modifications through a tool wrapper (function tool) so hooks and validation apply.

### Why it matters
If enforcement can be bypassed, the system can't be trusted to behave consistently.

**Status (2026-01-23):** Deferred to Phase 2. Current implementation still uses direct script generation through Script Writer agent, which is traced via SDK. Full enforcement through tool wrappers requires additional refactoring.

### Example (conceptual)
```
# Replace direct call with a function_tool that enforces hooks
apply_modifications_tool(modifications=learning.parameter_modifications)
```

---

### 7) Research Output Should Be Structured (Not Free‑Text) ⏳ DEFERRED

### Problem
Research output is free‑text and parsed with heuristics. This is fragile and prevents deterministic reuse.

### Change
Make Research Agent return a structured output schema:
- `recommended_approach`
- `key_parameters`
- `warnings`
- `alternatives`

### Why it matters
You can't build a self‑improving system on free‑text heuristics.

**Status (2026-01-23):** Deferred to Phase 3. Requires defining Pydantic `AgentOutputSchema` and updating Research Agent. Current free-text output works but is fragile.

---

### 8) Budget Enforcement Must Be Iteration‑Aware ✅

### Problem
Budget is checked once at the start, not before expensive steps.

### Change
Check budget before **each evaluation** or other costly phase.

### Why it matters
The system may waste expensive calls after the budget is already exhausted.

**Implementation (2026-01-23):**
- Added budget check before Phase 3 (Quality Evaluation) in each iteration
- Uses `self._budget_tracker.can_afford_evaluation()`
- On exhaustion: sets `session.status = SessionStatus.BUDGET_EXHAUSTED` and exits loop gracefully
- Logs warning: `"[Pipeline] BUDGET EXHAUSTED - using last available score"`

---

### 9) Turn Limits for Sub‑Agents Must Be Guaranteed ✅

### Problem
`agent.as_tool()` does not accept `max_turns`. Any sub‑agent can loop longer than intended.

### Change
Replace `as_tool()` with `@function_tool` wrappers that call `Runner.run(..., max_turns=...)`.

### Why it matters
Turn limits enforce bounded reasoning and prevent runaway loops.

**Implementation (2026-01-23):**
- **Discovery:** SDK v0.7.0 supports `agent.as_tool(max_turns=X)` natively!
- Refactored `create_agent_tool_wrappers()` to use native SDK pattern:
  ```python
  research_agent.as_tool(
      tool_name="research_approach",
      tool_description="Research best approach for effect type",
      max_turns=4,
  )
  ```
- Turn limits enforced: Research=4, Script=6, Executor=3, Quality=3, Learning=5
- No custom `@function_tool` wrappers needed

Reference: SDK tools docs: https://github.com/openai/openai-agents-python/blob/main/docs/tools.md

---

### 10) "Self‑Improvement" Must Be Mechanical, Not Instructional ✅

### Problem
Many learning behaviors are instructions only (e.g., "record pattern if improvement >= 5"). The pipeline doesn't enforce them.

### Change
Move all "self‑learning" requirements into the orchestrator:
- enforce pattern extraction on improvement
- enforce pattern outcome reporting
- enforce pre‑iteration research

### Why it matters
An autonomous system must not depend on LLM compliance for core logic.

**Implementation (2026-01-23):**
All three requirements now enforced by orchestrator, not LLM instructions:
1. **Pattern extraction:** Orchestrator checks `learning.pattern_extracted` and records to `session.extracted_patterns`
2. **Pattern outcome reporting:** Orchestrator calls `_report_pattern_outcome_impl()` after quality evaluation
3. **Pre-iteration research:** Orchestrator calls `pre_iteration_research_direct()` before Phase 1 when `iteration > 1`

Learning behaviors are now **deterministic pipeline steps**, not optional LLM responses.

---

## Phase 1.5 — Learning Agent as Exploration Controller (Doc‑Grounded)

### Problem
The Learning Agent is mostly post‑hoc and optional. It doesn’t *drive* exploration, and Blender 5.0.1 specifics are not guaranteed to propagate. This creates training‑data drift.

### Change
Make the Learning Agent a **mandatory pre‑generation step** that outputs doc‑grounded proposals and micro‑experiments. Every proposal must cite **Blender 5** documentation; anything else is rejected or routed to DocsExpert.

### Why it matters
This is the only way to reliably learn new Blender 5.0.1 behavior without drifting back to prior training data. You force the system to test, measure, and store evidence.

### Required outputs (structured)
- `proposals[]`: technique, params, expected_effect, `doc_refs[]`
- `micro_experiments[]`: minimal script + success_criteria + `doc_refs[]`
- `anti_patterns[]`: “avoid this” + evidence (errors/metrics)

### Enforcement rules
- Reject any proposal with empty `doc_refs`.
- If `doc_refs` do not map to Blender 5 docs, force a DocsExpert query.
- If a proposal uses a new API, run **at least one** micro‑experiment first.

### Example (conceptual)
```
learning = run_learning_agent(...)
assert learning.proposals and all(p.doc_refs for p in learning.proposals)

if learning.uses_new_api:
    run_micro_experiment(learning.micro_experiments[0])
```

---

## Phase 2 — Self‑Improvement Plumbing (Persistence + reuse)

### 11) Report pattern outcomes consistently
**Why:** The pattern library needs negative feedback to evolve.  
**Impact:** Prevents repeating bad “learned” fixes.  
**Complexity:** Medium.

### 12) Use extracted patterns in later iterations
**Why:** Self‑improvement requires reuse of successful patterns.  
**Impact:** Converts learning into better outcomes.  
**Complexity:** Medium.

---

## Phase 3 — Structured Inputs/Outputs (Reduce ambiguity)

### 13) Structured research output (Pydantic schema)
**Why:** Eliminates regex heuristics and ambiguity.  
**Impact:** Reliable decision‑making.  
**Complexity:** Medium.

### 14) Align prompt budgets with `max_turns`
**Why:** Prompt budgets without enforcement create drift.  
**Impact:** More predictable agent behavior.  
**Complexity:** Low.

---

## Phase 4 — Advanced Autonomy (Optional, higher cost)

### 15) Session summarization + compaction
**Why:** Prevents context bloat while preserving learning.  
**Impact:** Long‑run reliability.  
**Complexity:** Medium.

### 16) Cross‑session learning bootstrap
**Why:** Don’t restart from zero every run.  
**Impact:** Accelerates skill growth.  
**Complexity:** Medium‑High.

---

---

## Implementation Map (Files & Hotspots)
This is the exact place each phase likely touches.

| Item | Status | Primary Files | Likely Functions/Sections |
|------|--------|---------------|---------------------------|
| QW‑1 Budget checks | ✅ | `orchestrator.py`, `models/shared_context.py` | Before Phase 3 evaluation; `BUDGET_EXHAUSTED` enum |
| QW‑2 Dynamic fallback logs | ✅ | `tools/dynamic_instructions.py` | `dynamic_*_instructions()` with `logger.warning()` |
| QW‑3 Reset stuck state | ✅ | `orchestrator.py`, `session_manager.py` | `reset_for_technique_switch()`; `IssueTracker.reset()` |
| 1 Pre‑iteration research | ✅ | `orchestrator.py`, `tools/proactive_research_tools.py` | Phase 0.9; `pre_iteration_research_direct()` |
| 2 Escape level sync | ✅ | `orchestrator.py`, `models/shared_context.py` | Post‑QualityGateJudge; `stuck_state` field |
| 3 Pattern extraction enforcement | ✅ | `orchestrator.py`, `tools/code_pattern_tools.py` | `extracted_patterns` list; `_report_pattern_outcome_impl()` |
| 4 Reset stuck state | ✅ | `session_manager.py` | `IssueTracker.reset()` helper |
| 5 Dynamic instruction logging | ✅ | `tools/dynamic_instructions.py` | `logger.warning()` on fallback |
| 6 Enforce tool wrapper for modifications | ⏳ | `orchestrator.py`, `tools/script_generator_tools.py` | Deferred to Phase 2 |
| 7 Structured research output | ⏳ | `specialized_agents/research_agent.py`, `orchestrator.py` | Deferred to Phase 3 |
| 8 Iteration‑aware budget check | ✅ | `orchestrator.py` | Guard before Phase 3 evaluation |
| 9 Sub‑agent max_turns | ✅ | `orchestrator.py` | `as_tool(max_turns=X)` native SDK |
| 10 Mechanical self‑learning | ✅ | `orchestrator.py` | Enforce pattern extraction/outcome reporting |
| LA‑1 Learning Agent as controller | ⏳ | `orchestrator.py`, `specialized_agents/learning_agent.py` | Deferred to Phase 1.5 |
| 11 Pattern outcome reporting | ✅ | `orchestrator.py`, `tools/code_pattern_tools.py` | `_report_pattern_outcome_impl()` |
| 12 Reuse extracted patterns | ⏳ | `orchestrator.py` | Use `search_code_patterns`/`apply_pattern_to_script` |
| 13 Research schema | ⏳ | `models/` (new), `orchestrator.py` | Pydantic output model |
| 14 Prompt budgets | ✅ | `orchestrator.py` | `as_tool(max_turns=X)` aligns with prompt |
| 15 Session compaction | ⏳ | `orchestrator.py`, `session_manager.py` | Summarize + rotate SDK sessions |
| 16 Cross‑session bootstrap | ⏳ | `session_persistence.py`, `tools/experiment_tracker_tools.py` | Load past patterns/learnings |

---

## Rough Effort Estimates (Engineering Hours)
These are conservative ranges assuming one developer familiar with the code.

| Phase | Item Count | Status | Estimated Effort |
|-------|------------|--------|------------------|
| Phase 0 | 3 items | ✅ COMPLETE | 2–4 hours |
| Phase 1 | 10 items | ✅ 8/10 COMPLETE | 10–18 hours |
| Phase 2 | 2 items | ⏳ PENDING | 4–6 hours |
| Phase 3 | 2 items | ⏳ PENDING | 4–8 hours |
| Phase 4 | 2 items | ⏳ PENDING | 6–12 hours |

**Completed (Phase 0 + Phase 1):** ~12–22 hours equivalent
**Remaining (Phase 2-4):** ~14–26 hours

---

## Is Loop‑Based Iteration Optimal?
Short answer: **not by itself**. A single linear loop is a useful baseline, but it is usually **sub‑optimal** for systems that must demonstrate *self‑improvement* and *emergent behavior*. It tends to:
- Over‑exploit local tweaks instead of exploring new techniques.
- Repeat failure patterns because there is no branching or competition.
- Hide uncertainty (one path gives no comparison set).

The loop should be upgraded into a **search or population** process if autonomy is the goal.

---

## Alternative Workflow Families
These patterns are better aligned with autonomy and learning. They are **examples**, not prescriptions.

### A) Population‑Based Search (Evolutionary / Swarm)
Maintain **N parallel candidates** (scripts or parameter sets). Each iteration:
1. Mutate / crossover candidates.
2. Evaluate in parallel.
3. Keep top‑K and discard the rest.

Why it helps: diversity + competition drive exploration and emergent behavior.  
References: [EvoFlow](https://arxiv.org/pdf/2502.07373), [SwarmAgentic](https://yaoz720.github.io/SwarmAgentic/).

### B) Tree Search (Branching Iterations)
Branch into multiple candidate modifications and score them (MCTS‑style):
1. Propose multiple modifications.
2. Evaluate or predict outcomes.
3. Expand best branches.

Why it helps: prevents premature convergence and encourages novel paths.  
Reference: [MC‑NEST](https://arxiv.org/html/2503.19309v1).

### C) Planner–Executor–Verifier (Hierarchical Control)
Use a **planner** to propose strategy, **executor** to run, and **verifier** to score. The planner is updated based on verifier outcomes.

Why it helps: separates goals from execution; improves stability.  
Reference: [AgentFlow](https://agentflow.stanford.edu/).

### D) Multi‑Armed Bandit (Exploration vs Exploitation)
Treat each technique as an arm; update weights based on reward (quality score).

Why it helps: formalizes exploration vs exploitation with low overhead.

---

## Suggested Next‑Step Workflow (Cost‑Constrained)
If you want autonomy without exploding compute:
1. **Beam search**: Generate 2–3 candidate modifications each iteration; evaluate top‑K.
2. **Bandit technique selection**: Use UCB/Thompson over techniques to avoid lock‑in.
3. **Small population**: Maintain 3–5 concurrent scripts; prune worst each iteration.

This moves you beyond “single‑path looping” while keeping cost in check.

---

## Plugin Snippets (agent-orchestration:multi-agent-optimize)
**Purpose:** Head‑start templates for integrating the plugin. These are **pseudo‑code** placeholders; replace with the actual plugin API once installed.

### Template A — Optimizer Wrapper (beam + selection)
```
# PSEUDO-CODE: replace with actual plugin API
from <plugin_package> import MultiAgentOptimize

optimizer = MultiAgentOptimize(
    model="claude-opus-4.5",
    beam_width=3,
    keep_top_k=1,
    candidate_generator=generate_candidate,   # Script Writer + params
    evaluator=evaluate_candidate,             # Executor + Quality Analyst
    selector=select_top_k,                     # Deterministic ranking
)

best = optimizer.run(
    seed=baseline_candidate,
    max_rounds=5,
)
```

### Template B — Hook‑style integration
```
# PSEUDO-CODE: replace with actual plugin hooks/config keys
plugin: agent-orchestration:multi-agent-optimize
model: claude-opus-4.5
beam_width: 3
keep_top_k: 1
hooks:
  generate_candidates: generate_candidate
  evaluate_candidate: evaluate_candidate
  select_top_k: select_top_k
```

### Template C — Candidate generator stub
```
def generate_candidate(seed: Candidate, variant_id: int) -> Candidate:
    # 1) Build prompt (include research + technique guidance)
    # 2) Call Script Writer (possibly with different seed/technique)
    # 3) Return Candidate(script_path, params, technique, ...)
    return candidate
```

---

## Bottom Line
If these phases are not implemented, the system will **look** active but won't reliably self‑improve.
If they are implemented, you get a loop that can *provably* learn, adapt, and escalate — the minimum required for real autonomy.

---

## Implementation Notes (2026-01-23)

### Key SDK Discovery
During implementation, discovered that **OpenAI Agents SDK v0.6.9+** now supports `agent.as_tool(max_turns=X)` natively. This eliminated the need for custom `@function_tool` wrappers as originally planned in item 9.

### Pattern for Direct Callable Functions
Several `@function_tool` decorated functions needed to be called directly by the orchestrator (not through agent context). Solution: create `*_direct()` or `*_impl()` versions without the decorator:

```python
# For agents (has @function_tool decorator)
@function_tool
def pre_iteration_research(wrapper: RunContextWrapper[SharedContext], ...) -> str:
    return pre_iteration_research_direct(...)

# For orchestrator (no decorator)
def pre_iteration_research_direct(current_issue: str, ...) -> str:
    # Actual implementation
```

### Files Modified
- `orchestrator.py` - Major autonomy enforcement changes
- `models/shared_context.py` - New fields and enum values
- `session_manager.py` - New reset methods
- `tools/dynamic_instructions.py` - Added logging
- `tools/proactive_research_tools.py` - Added direct callable
- `tools/code_pattern_tools.py` - Added `_impl` function

### Tests Passing
- `test_orchestrator_tools.py` - All passed
- `test_self_learning_tools.py` - 4/4 passed
- Import verification - All files compile

