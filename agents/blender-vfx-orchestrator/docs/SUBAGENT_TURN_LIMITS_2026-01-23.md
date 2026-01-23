# Enforcing `max_turns` for Sub‑Agents (Agents‑as‑Tools)

This note addresses the `agent.as_tool()` limitation and lays out SDK‑compliant ways to enforce turn limits for sub‑agents.

References:
- SDK tools docs (agents‑as‑tools + customization): https://github.com/openai/openai-agents-python/blob/main/docs/tools.md
- SDK README (agent loop + `max_turns`): https://github.com/openai/openai-agents-python/tree/main
- Local analysis: `docs/MULTI_AGENT_OPTIMIZATION_ANALYSIS_2026-01-23.md`

## Key SDK Constraint
`agent.as_tool()` is a **convenience wrapper** and does **not** accept `max_turns`.  
The SDK explicitly recommends using a custom tool that calls `Runner.run()` directly when you need to set `max_turns`.

## SDK‑Recommended Pattern (Turn‑Limited Wrapper)
Use a `@function_tool` wrapper that calls `Runner.run()` with explicit `max_turns`:
```
@function_tool
async def run_quality_agent(render_path: str) -> str:
    result = await Runner.run(
        quality_agent,
        f"Evaluate {render_path}",
        max_turns=4,  # Enforced here
        run_config=...,  # optional
    )
    return str(result.final_output)
```

This wrapper becomes the “tool” used by the coordinator instead of `quality_agent.as_tool(...)`.

## Why This Works
From the SDK docs:
- `Runner.run()` accepts `max_turns` and bounds the agent loop.
- `agent.as_tool()` does not expose `max_turns`.
- The recommended workaround is to call `Runner.run()` inside a tool implementation.

## Additional Enforcement Options (Complementary)
These don’t replace `max_turns`, but can add guardrails:
- **RunHooks**: enforce max tool calls or stop conditions within a run.
- **Output Guardrails**: validate outputs and fail fast when malformed.
- **Tool guardrails**: validate inputs/outputs for the wrapper tool.

Note: RunHooks still need a `Runner.run()` call to attach; they can’t be attached through `as_tool()`.

## Practical Guidance for This Project
1) Use function tool wrappers for every sub‑agent invoked by a coordinator.
2) Centralize turn limits in one place (e.g., `create_agent_tool_wrappers()`).
3) Keep `max_turns` aligned with agent prompt budgets.
4) Use `run_config` or hooks to capture telemetry and enforce doc‑query rules.

## What to Avoid
- Using `agent.as_tool()` for critical sub‑agents that must be bounded.
- Assuming `max_turns` can be set on `as_tool()`.
- Relying solely on prompt text (“max 3 turns”) without an actual `max_turns` limit.

## Suggested Turn‑Limit Defaults
Use short budgets for tool‑style agents:
| Agent | Suggested `max_turns` | Rationale |
|-------|------------------------|-----------|
| Research | 4 | Search + summarize |
| Script Writer | 6 | Generate + validate |
| Executor | 3 | Execute + parse |
| Quality Analyst | 4 | Evaluate + summarize |
| Learning | 3 | Record + suggest |

These are already consistent with the SDK’s guidance and your internal analysis.

