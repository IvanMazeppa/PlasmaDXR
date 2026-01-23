# Remediation Plan (2026-01-23)

Purpose: actionable, ordered fixes to stabilize the Agents SDK workflow and unlock self-learning behavior without changing architecture.

## Priority 0 (Stop-the-bleed)
1) **Disable handoff pipeline**
   - Problem: guardrails do not fire for mid-chain agents in handoffs.
   - Action: hard-disable `create_asset()` and stop initializing handoff agents.

2) **Re-enable dynamic instructions in pipeline**
   - Problem: self-learning rules never injected (dynamic instructions disabled).
   - Action: set `instructions=dynamic_*_instructions` for pipeline agents.
   - If extra static text is needed, wrap the function and append static text.

3) **Normalize output schema usage**
   - Problem: `key_parameters` vs `parameters_set`, `execution_time` vs `execution_time_seconds`.
   - Action: pick one field name per model and update pipeline usage. Add a tiny schema test to prevent drift.

## Priority 1 (Enforcement)
4) **Tighten doc-query enforcement**
   - Problem: `generate_script` and `recommend_technique` allow script creation without doc search.
   - Action: extend RunHooks to require a doc query before any script generation tool call.

5) **Remove handoff prompt prefix on standalone agents**
   - Problem: `prompt_with_handoff_instructions()` adds noise to agents without handoffs.
   - Action: use plain instructions for standalone agents.

6) **Route direct `_modify_script_impl` through enforcement**
   - Problem: bypasses RunHooks/guardrails/tracing.
   - Action: wrap in `@function_tool` and call through runner hooks, or route via Script Writer in a “apply explicit params” mode.

## Priority 2 (Quality + Cost Control)
7) **Integrate full API Validator agent when needed**
   - Problem: lightweight validator only catches known deltas.
   - Action: call API Validator agent on iteration > 1 or when lightweight validation fails.

8) **Session compaction strategy**
   - Problem: SQLiteSession grows indefinitely; costs rise.
   - Action: add a summarization checkpoint every N iterations and start a fresh session with the summary.

## Priority 3 (Reliability + Structure)
9) **Structured research output**
   - Problem: research extraction uses regex on free text.
   - Action: give Research Agent a Pydantic output schema (recommended_approach, key_params, alternatives).

10) **Instruction unification**
   - Problem: multiple instruction sources contradict each other.
   - Action: choose one canonical script-writing mode (template vs write-code) and align all prompts and RunHooks.

## Minimal Acceptance Criteria
- Pipeline uses dynamic instructions.
- Handoff pipeline cannot be invoked.
- Script parameters and execution time are recorded consistently.
- Doc query required before any script generation tool call.

## Testing Checklist (Post-fix)
- One smoke run with max_iterations=1 and check:
  - Dynamic instructions present in trace prompt.
  - `parameters_set` (or chosen field) populated in `SessionState`.
  - No handoff agent initialization in logs.
- One run with iteration > 2 and verify:
  - API Validator invoked when needed.
  - No tool loops triggered incorrectly.

