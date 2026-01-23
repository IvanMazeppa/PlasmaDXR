# Dynamic Instructions Guide (Agents SDK) — Blender VFX Orchestrator

This guide summarizes official SDK behavior and shows how to enable dynamic instructions safely in this project.

## Official SDK Behavior (Summary)
From the OpenAI Agents SDK docs:
- `Agent.instructions` can be a **string** or a **callable**.
- The callable must accept **exactly two parameters**: `(context, agent)`.
- The callable can be **sync or async** and must return a string.
- If a callable has a different signature, the SDK raises a `TypeError`.

Sources:
- `docs/agents.md` (dynamic instructions example)
- `docs/context.md` (instructions can be a function)
- API reference (`ref/agent`) (signature enforcement)

## Minimal Example (Official Pattern)
```
from agents import Agent, RunContextWrapper

def dynamic_instructions(context: RunContextWrapper[UserContext], agent: Agent[UserContext]) -> str:
    return f"The user's name is {context.context.name}. Help them with their questions."

agent = Agent(
    name="Triage agent",
    instructions=dynamic_instructions,
)
```

## Project-Specific Implementation

You already have dynamic instruction generators in:
`tools/dynamic_instructions.py`

Key functions:
- `dynamic_script_writer_instructions(ctx, agent)`
- `dynamic_quality_analyst_instructions(ctx, agent)`
- `dynamic_learning_agent_instructions(ctx, agent)`

### How to enable (conceptually)
Replace static string instructions with function references for the pipeline agents:
- Script Writer → `dynamic_script_writer_instructions`
- Quality Analyst → `dynamic_quality_analyst_instructions`
- Learning Agent → `dynamic_learning_agent_instructions`

### If you must append static text
Do not disable dynamic instructions. Instead wrap:
```
def script_writer_instructions_with_overrides(ctx, agent) -> str:
    base = dynamic_script_writer_instructions(ctx, agent)
    return base + "\n\n## Extra Rules\n- Keep tool calls <= 6\n"
```

Use the wrapper as `instructions=script_writer_instructions_with_overrides`.

## Common Failure Modes (and Fixes)

1) **TypeError about arguments**
   - Cause: instruction function doesn’t accept exactly 2 args.
   - Fix: ensure signature is `(context, agent)` only.

2) **Context is None**
   - Cause: no session/context attached.
   - Fix: use safe fallback inside the function (already in your module).

3) **Dynamic rules not showing**
   - Cause: agent constructed with `use_dynamic_instructions=False` or overwritten by static strings.
   - Fix: pass the function directly to `instructions`.

4) **Performance issues**
   - Cause: heavy I/O inside the instruction function.
   - Fix: keep instructions fast; cache and summarize knowledge outside the function.

## Integration Notes for This Project

- Your dynamic functions call `_query_knowledge_base_impl` (in-process).
- Avoid MCP calls or heavy network calls in instruction functions.
- Keep the dynamic text small; large instructions can dominate context.
- If you use SDK sessions, dynamic instructions can read `ctx.context.session`.

## Troubleshooting Checklist
- Verify the agent’s `instructions` is a callable, not a string.
- Confirm the callable returns a string (not dict or list).
- Check for exceptions inside the instruction function (add minimal try/except).
- Confirm you pass `session=...` to `Runner.run()` so context is populated.

## Optional: MCP Prompt Server Pattern (SDK Example)
The SDK docs also show that you can fetch prompts via MCP and set them as instructions in user code (outside the agent run):
```
prompt_result = await server.get_prompt("generate_code_review_instructions", {...})
instructions = prompt_result.messages[0].content.text
agent = Agent(name="Code Reviewer", instructions=instructions, mcp_servers=[server])
```
Use this only if you already have a reliable in-process MCP server available to the orchestration layer.

