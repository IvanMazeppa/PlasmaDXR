# Autonomy‑Critical Changes (2026-01-23)

**Purpose:** This document lays out the **specific changes** required for the system to *actually* perform the tasks end‑to‑end (not just appear to). The emphasis is on deterministic enforcement, not hopeful prompting. **No code changes yet** — this is the plan.

---

## Executive Summary (Cold Truth)
Right now the loop can run, but several **core autonomy mechanisms are either missing or non‑binding**. The system relies on LLM compliance for actions that must be deterministic. This prevents real self‑improvement and makes “emergent behavior” unreliable. The changes below make the learning loop *mechanically true* instead of “suggested”.

**Note:** Dynamic instructions are now implemented (see `docs/DYNAMIC_INSTRUCTIONS_GUIDE_2026-01-23.md`). That is a **major enabler**, but it does not replace deterministic enforcement in the orchestrator.

---

## Phase 0 — Quick Wins (Low risk, high leverage)

### QW‑1) Add budget checks before expensive phases
**Why:** Prevents burning budget after it’s already exhausted.  
**Impact:** Immediate cost safety.  
**Complexity:** Low.

### QW‑2) Log dynamic‑instruction fallbacks
**Why:** Silent KB failures kill learning; logging makes it visible.  
**Impact:** Immediate observability.  
**Complexity:** Low.

### QW‑3) Reset stuck state on technique switch
**Why:** New techniques shouldn’t inherit old failure streaks.  
**Impact:** Immediate exploration benefit.  
**Complexity:** Low.

---

## Phase 1 — Make Autonomy Real (Deterministic loop control)

### 1) Mandatory Pre‑Iteration Research (Not Optional)

### Problem
The pipeline does not call `pre_iteration_research()` before script modification. This means early warning signals are ignored, so the system repeats known failures before escalating.

### Change
Insert a **required** pre‑iteration research step for `iteration > 1`, **before** Phase 1 modification.

### Why it matters
This is the difference between a reactive loop and a *self‑correcting* loop. The system should proactively look for alternatives before wasting another iteration.

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

### 2) Escape Level Must Drive Decisions

### Problem
The quality gate produces an `escape_level`, but the **SessionManager context never includes it**, so downstream logic sees `escape_level=0` forever.

### Change
Persist the escape level into SessionManager (or a single shared stuck‑state) **immediately after** QualityGateJudge.

### Why it matters
Without a live escape level, “stuck detection” is a mirage. The system can’t prove that it learns from repeated failures.

### Example (conceptual)
```
if gate_decision:
    session_mgr.sync_escape_level(gate_decision.escape_level)
```

---

### 3) Pattern Extraction Must Be Enforced by the Orchestrator

### Problem
The Learning Agent is instructed to extract patterns, but the pipeline does **nothing** with `learning.pattern_extracted` or `learning.pattern_id`.

### Change
After Learning Agent returns, the orchestrator must:
1. Record extracted patterns in session state.
2. Report pattern outcomes after execution.

### Why it matters
Self‑improvement requires *persistent knowledge*. If success doesn’t create a reusable pattern, the system is not learning.

### Example (conceptual)
```
if learning.pattern_extracted and learning.pattern_id:
    session.extracted_patterns.append(learning.pattern_id)

# after evaluation
if script.pattern_id:
    report_pattern_outcome(script.pattern_id, success, improvement)
```

---

### 4) Technique Switching Must Reset Stuck State

### Problem
When switching techniques, the system keeps the old issue streaks. This means a new technique inherits stale “stuck” penalties.

### Change
Reset the issue tracker when a new technique is selected.

### Why it matters
Exploration is impossible if every new idea is penalized by old failures. This is one of the biggest blockers to emergent behavior.

### Example (conceptual)
```
if switched_technique:
    session_mgr.reset_issue_tracker()
```

---

### 5) Dynamic Instructions Must Never Fail Silently

### Problem
Dynamic instructions swallow exceptions and fall back without logging. That makes KB‑injection failures invisible.

### Change
Add lightweight logging when context extraction fails, and fallback to a safe default.

### Why it matters
If KB‑injection fails, the system stops learning *silently*. Silent failure kills self‑improvement.

### Example (conceptual)
```
try:
    effect_type = ctx.context.session.request.effect_type.value
except Exception as e:
    logger.warning("Dynamic instructions context failed", exc_info=e)
    effect_type = "general"
```

---

### 6) Do Not Bypass Enforcement When Modifying Scripts

### Problem
Direct `_modify_script_impl` bypasses RunHooks, guardrails, and tool tracing.

### Change
Route *all* parameter modifications through a tool wrapper (function tool) so hooks and validation apply.

### Why it matters
If enforcement can be bypassed, the system can’t be trusted to behave consistently.

### Example (conceptual)
```
# Replace direct call with a function_tool that enforces hooks
apply_modifications_tool(modifications=learning.parameter_modifications)
```

---

### 7) Research Output Should Be Structured (Not Free‑Text)

### Problem
Research output is free‑text and parsed with heuristics. This is fragile and prevents deterministic reuse.

### Change
Make Research Agent return a structured output schema:
- `recommended_approach`
- `key_parameters`
- `warnings`
- `alternatives`

### Why it matters
You can’t build a self‑improving system on free‑text heuristics.

---

### 8) Budget Enforcement Must Be Iteration‑Aware

### Problem
Budget is checked once at the start, not before expensive steps.

### Change
Check budget before **each evaluation** or other costly phase.

### Why it matters
The system may waste expensive calls after the budget is already exhausted.

---

### 9) Turn Limits for Sub‑Agents Must Be Guaranteed

### Problem
`agent.as_tool()` does not accept `max_turns`. Any sub‑agent can loop longer than intended.

### Change
Replace `as_tool()` with `@function_tool` wrappers that call `Runner.run(..., max_turns=...)`.

### Why it matters
Turn limits enforce bounded reasoning and prevent runaway loops.

Reference: SDK tools docs: https://github.com/openai/openai-agents-python/blob/main/docs/tools.md

---

### 10) “Self‑Improvement” Must Be Mechanical, Not Instructional

### Problem
Many learning behaviors are instructions only (e.g., “record pattern if improvement >= 5”). The pipeline doesn’t enforce them.

### Change
Move all “self‑learning” requirements into the orchestrator:
- enforce pattern extraction on improvement
- enforce pattern outcome reporting
- enforce pre‑iteration research

### Why it matters
An autonomous system must not depend on LLM compliance for core logic.

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

## Bottom Line
If these phases are not implemented, the system will **look** active but won’t reliably self‑improve.  
If they are implemented, you get a loop that can *provably* learn, adapt, and escalate — the minimum required for real autonomy.

