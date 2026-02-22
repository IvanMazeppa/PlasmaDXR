# Blender VFX Orchestrator SDK Workflow Audit (2026-01-23)

Scope: `orchestrator.py`, `tools/dynamic_instructions.py`, `specialized_agents/api_validator.py`, plus the docs cited by the request. Focus is Agents SDK usage, workflow integrity, and instruction reliability.

## Executive Summary
The code-based pipeline and coordinator pattern are the correct backbone. The largest reliability risks are (1) the still-active handoff pipeline, (2) self-learning dynamic instructions disabled in the main pipeline, (3) schema drift in structured outputs, and (4) conflicting instruction sets that encourage incompatible behavior. These issues are fixable without architectural changes.

## Findings

### Critical
1) Handoff pipeline still active (guardrail gaps)

Evidence:
```
# orchestrator.py (handoff agents still created)
return_to_orchestrator = handoff(...)
...
self._orchestrator = Agent(..., handoffs=handoffs_list, ...)

async def create_asset(self, request: AssetRequest) -> SessionState:
    """DEPRECATED: Use create_asset_pipeline() instead."""
```

Impact:
- Guardrails run only for first/last agent in a handoff chain. Mid-chain agents lose enforcement.
- The deprecated pathway is still callable, so production can silently bypass guardrails.

Recommendation:
- Hard-disable `create_asset()` (raise or remove).
- Remove handoff agent initialization after deprecation window.
- Update documentation and tests to target pipeline-only behavior.

2) Dynamic instructions disabled in the main pipeline

Evidence:
```
# orchestrator.py (standalone agents used in pipeline)
base_script_writer_standalone = create_script_writer(use_dynamic_instructions=False)
base_quality_standalone = create_quality_analyst(use_dynamic_instructions=False)
base_learning_standalone = create_learning_agent(use_dynamic_instructions=False)
```

Impact:
- Validated knowledge-base rules are never injected during pipeline runs.
- Self-learning strategies exist in tooling but are not reflected in prompts.

Recommendation:
- Re-enable dynamic instructions for pipeline agents.
- If extra static instructions are required, wrap the instruction function and append static text (do not disable dynamic behavior).

3) Structured output schema drift (data loss)

Evidence:
```
# orchestrator.py (ScriptOutput)
script = ScriptOutput(
    ...,
    key_parameters=learning.parameter_modifications,  # field mismatch
)

# orchestrator.py (ExecutionOutput)
execution_time_seconds=execution.execution_time if hasattr(execution, 'execution_time') else None,
```

Impact:
- Parameter tracking and timing are dropped or inconsistent.
- Learning records and diagnostics lose fidelity.

Recommendation:
- Normalize fields across models and pipeline usage.
- Add a small schema-compat test (one sample pipeline run, assert fields present).

### High
4) Doc-query enforcement can be bypassed

Evidence:
```
# Script Writer standalone instructions (pipeline)
1. Call recommend_technique ONCE
2. Call generate_script ONCE
3. Call validate_script ONCE
...

# RunHooks enforcement only checks write_script/modify_script
require_doc_query_before=[write_script, modify_script]
```

Impact:
- Scripts can be generated without any authoritative doc search.
- The enforcement intended to prevent API drift can be bypassed.

Recommendation:
- Require doc query before `generate_script` and `recommend_technique`, or
- Enforce a "doc-tool call" sentinel in the prompt before any script generation tool.

5) Handoff prompt prefix used on non-handoff agents

Evidence:
```
instructions=prompt_with_handoff_instructions(...)
```

Impact:
- Adds noise and handoff instructions to agents that have no handoffs.
- Increases token spend and risk of invalid tool calls.

Recommendation:
- Apply `prompt_with_handoff_instructions()` only to agents that actually use `handoffs=[...]`.

### Medium
6) Instruction contradictions (Blender 5.0 and tool usage)

Evidence:
```
# tools/dynamic_instructions.py
"CRITICAL: BLENDER 5.0 ONLY - NO BACKWARDS COMPATIBILITY"
"FORBIDDEN: hasattr() for version detection"
...
"GENERAL RULE: Always use hasattr() guards"
```

Impact:
- The model gets conflicting directives and produces inconsistent code.

Recommendation:
- Consolidate to one rule set. Put all version-compat logic in API Validator (or remove it entirely).

7) Direct `_modify_script_impl` bypasses enforcement/tracing

Evidence:
```
modify_result_json = _modify_script_impl(...)
```

Impact:
- No RunHooks, no guardrails, no tool tracing.
- Changes can be applied without doc checks or structured validation.

Recommendation:
- Wrap `_modify_script_impl` in a `@function_tool` and enforce via hooks, or
- Route this path through the Script Writer with a "apply explicit params" mode.

8) API Validator full agent not integrated

Evidence:
```
# orchestrator.py uses validate_code_api() (lightweight)
api_validation = await validate_code_api(script_content)
```

Impact:
- Only known patterns are corrected.
- Unknown Blender 5.0 changes still slip through.

Recommendation:
- Use the full API Validator agent as a fallback when validation fails or on iteration > 1.

9) SDK Session growth risk

Impact:
- Long sessions accumulate context across agents and iterations.
- Token usage grows and older info may dominate.

Recommendation:
- Add periodic summarization (Learning Agent) and prune context by starting a new SDK session after N iterations.

### Low
10) Research output is unstructured

Impact:
- Alternative approaches extracted via regex are brittle.

Recommendation:
- Give Research Agent a structured output schema for recommended approach and alternatives.

## SDK Features Not Fully Leveraged
- `AgentOutputSchema(strict_json_schema=True)` for coordinators to harden decision outputs.
- Wrapper function tools around agent tools when you need `max_turns` (since `as_tool()` cannot set it).
- Use `trace(..., group_id=session_id)` to group all phases under a session.
- Convert key prompts to structured outputs instead of regex extraction (Research Agent).

## Suggested Next Steps (No Code Changes Yet)
- Align instruction sources (one canonical instruction set).
- Re-enable dynamic instructions in the pipeline.
- Normalize output schema usage (no `key_parameters` vs `parameters_set` drift).
- Sunset handoff pipeline.
- Add a small schema regression test to prevent drift.

