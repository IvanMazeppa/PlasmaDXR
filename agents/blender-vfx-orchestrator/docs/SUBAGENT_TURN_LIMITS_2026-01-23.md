# Enforcing `max_turns` for Sub‑Agents (Agents‑as‑Tools)

This note addresses the `agent.as_tool()` limitation and lays out SDK‑compliant ways to enforce turn limits for sub‑agents.

References:
- SDK tools docs (agents‑as‑tools + customization): https://github.com/openai/openai-agents-python/blob/main/docs/tools.md
- SDK README (agent loop + `max_turns`): https://github.com/openai/openai-agents-python/tree/main
- Local analysis: `docs/MULTI_AGENT_OPTIMIZATION_ANALYSIS_2026-01-23.md`

---

## ✅ UPDATE (2026-01-23): SDK v0.7.0 Supports `max_turns` Natively

**Important Discovery:** As of SDK v0.6.9+ (now v0.7.0), `agent.as_tool()` accepts `max_turns` directly:

```python
research_agent.as_tool(
    tool_name="research_approach",
    tool_description="Research best approach for effect type",
    max_turns=4,  # NOW SUPPORTED NATIVELY
)
```

This eliminates the need for custom `@function_tool` wrappers in most cases. The orchestrator has been updated to use this native pattern.

---

## Original Approach (Still Valid for Custom Logic)

The original `@function_tool` wrapper approach is still valid when you need custom logic beyond what `as_tool()` provides:

### SDK‑Recommended Pattern (Turn‑Limited Wrapper)
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

This wrapper becomes the "tool" used by the coordinator instead of `quality_agent.as_tool(...)`.

### Why Custom Wrappers Are Still Useful
- Adding pre/post processing logic
- Custom context injection not available through `as_tool()`
- Complex error handling or retry logic
- Telemetry/logging not covered by RunHooks

## Additional Enforcement Options (Complementary)
These don't replace `max_turns`, but can add guardrails:
- **RunHooks**: enforce max tool calls or stop conditions within a run.
- **Output Guardrails**: validate outputs and fail fast when malformed.
- **Tool guardrails**: validate inputs/outputs for the wrapper tool.

Note: RunHooks still need a `Runner.run()` call to attach; they can't be attached through `as_tool()`.

## Current Implementation (2026-01-23)

The orchestrator now uses native `as_tool(max_turns=X)` in `create_agent_tool_wrappers()`:

| Agent | `max_turns` | Rationale |
|-------|-------------|-----------|
| Research | 4 | Search + summarize |
| Script Writer | 6 | Generate + validate |
| Executor | 3 | Execute + parse |
| Quality Analyst | 3 | Evaluate + summarize |
| Learning | 5 | Record + analyze + suggest |

## What to Avoid
- Relying solely on prompt text ("max 3 turns") without an actual `max_turns` limit.
- Using very high turn limits (>10) for simple tasks.
- Forgetting to align turn limits with agent prompt budgets.
