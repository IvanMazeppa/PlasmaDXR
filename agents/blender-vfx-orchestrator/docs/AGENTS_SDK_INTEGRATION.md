# OpenAI Agents SDK Integration Guide

## CRITICAL: Always Consult Official Documentation

**Before making ANY changes to the multi-agent system, consult the official documentation:**

- **Main Repository:** https://github.com/openai/openai-agents-python
- **Documentation:** https://github.com/openai/openai-agents-python/tree/main/docs
- **Key Docs:**
  - [Multi-Agent Patterns](https://github.com/openai/openai-agents-python/blob/main/docs/multi_agent.md)
  - [Tools Reference](https://github.com/openai/openai-agents-python/blob/main/docs/tools.md)
  - [Handoffs](https://github.com/openai/openai-agents-python/blob/main/docs/handoffs.md)

The SDK is rapidly evolving. At time of writing, we're on **v0.6.8**. Check for updates regularly.

---

## Architecture Overview

The blender-vfx-orchestrator uses the OpenAI Agents SDK to coordinate multiple specialized agents:

```
┌─────────────────────────────────────────────────────────────────┐
│                     VFX ORCHESTRATOR                            │
│              (Central coordination agent)                       │
└─────────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
          ▼                   ▼                   ▼
    ┌──────────┐       ┌──────────┐       ┌──────────┐
    │ Script   │       │ Quality  │       │ Blender  │
    │ Writer   │       │ Analyst  │       │ Docs     │
    └──────────┘       └──────────┘       └──────────┘
```

**All agents must be proper subagents under the orchestrator umbrella.**

---

## Lesson Learned: MCP vs Agents SDK Tools

### The Problem We Solved

We had tools that tried to call external MCP servers from within agent context:

```python
# ❌ WRONG - This doesn't work inside Agents SDK context
@function_tool
async def find_reference_images(effect_type: str) -> str:
    # Tried to spawn MCP subprocess
    server = await get_mcp_server("asset-evaluator")  # FAILS
    return await server.call_tool(...)
```

**Result:** Silent failures with empty error messages in traces.

### Why It Failed

1. **MCP servers are external processes** - They spawn subprocesses that communicate via stdio
2. **Agents SDK runs in-process** - Tools must execute within the same Python process
3. **Context isolation** - Agent execution context can't spawn arbitrary subprocesses

### The Solution: Direct In-Process Implementations

```python
# ✅ CORRECT - Tools run entirely in-process

# Internal implementation (plain async function)
async def _find_reference_images_impl(effect_type: str, limit: int = 5) -> str:
    """Direct filesystem search - no external dependencies."""
    results = []
    for img_path in REFERENCE_DIR.rglob("*"):
        if effect_type in str(img_path).lower():
            results.append({"path": str(img_path), ...})
    return json.dumps({"results": results})

# Tool wrapper (exposed to agents)
@function_tool
async def find_reference_images(effect_type: str, limit: int = 5) -> str:
    """Find reference images for a specific effect type."""
    return await _find_reference_images_impl(effect_type, limit)
```

---

## The Two-Layer Tool Pattern

### Why Two Layers?

The `@function_tool` decorator wraps functions in a `FunctionTool` object that **is not directly callable**. If Tool A needs to call Tool B internally, calling the decorated function fails.

### Pattern: `_impl` Functions + `@function_tool` Wrappers

```python
# Layer 1: Internal implementations (plain async functions)
async def _analyze_with_vision_impl(render_path: str, ...) -> str:
    """Actual logic - can be called by other internal functions."""
    client = OpenAI()
    response = client.responses.create(model="gpt-5.2", ...)
    return json.dumps(result)

async def _evaluate_render_impl(render_path: str, ...) -> str:
    """Calls _analyze_with_vision_impl internally."""
    quality_result = await _analyze_with_vision_impl(render_path, ...)
    return json.dumps({"score": ...})

# Layer 2: Tool wrappers (exposed to agents)
@function_tool
async def analyze_with_vision(render_path: str, ...) -> str:
    """Tool wrapper for agents."""
    return await _analyze_with_vision_impl(render_path, ...)

@function_tool
async def evaluate_render(render_path: str, ...) -> str:
    """Tool wrapper for agents."""
    return await _evaluate_render_impl(render_path, ...)
```

### Benefits

1. **Internal calls work** - `_evaluate_render_impl` can call `_analyze_with_vision_impl`
2. **Agent calls work** - Agents use the `@function_tool` wrappers
3. **No code duplication** - Logic lives in `_impl` functions
4. **Testable** - Can test `_impl` functions directly without mock contexts

---

## Key SDK Concepts

### 1. Agents as Tools (`agent.as_tool()`)

Convert an agent into a tool that another agent can invoke:

```python
quality_analyst = Agent(name="Quality Analyst", ...)
orchestrator = Agent(
    name="Orchestrator",
    tools=[quality_analyst.as_tool()]  # Agent becomes a tool
)
```

### 2. Handoffs

Transfer control between agents:

```python
from agents import handoff

orchestrator = Agent(
    name="Orchestrator",
    handoffs=[
        handoff(target=script_writer, description="Write Blender scripts"),
        handoff(target=quality_analyst, description="Evaluate render quality"),
    ]
)
```

### 3. Function Tools

Expose Python functions as tools:

```python
@function_tool
async def my_tool(param: str) -> str:
    """Tool description (becomes the tool's docstring)."""
    return f"Result: {param}"
```

---

## Common Pitfalls

### ❌ Don't: Call MCP servers from agent tools
```python
# This will fail silently
server = await get_mcp_server("some-server")
```

### ❌ Don't: Call `@function_tool` functions directly from other tools
```python
# FunctionTool is not callable
result = await some_other_tool(args)  # TypeError
```

### ❌ Don't: Use blocking I/O in async tools
```python
# Use async alternatives
result = requests.get(url)  # Bad - use httpx
```

### ✅ Do: Keep all tool logic in-process
### ✅ Do: Use the two-layer pattern for tools that call other tools
### ✅ Do: Check SDK docs before adding new agent patterns

---

## Testing Tools

Test tools directly by calling `_impl` functions:

```python
async def test_find_reference_images():
    result = await _find_reference_images_impl("explosion", limit=3)
    data = json.loads(result)
    assert data["count"] > 0
```

Or via the FunctionTool API:

```python
from dataclasses import dataclass

@dataclass
class MockToolContext:
    context: Any = None

async def call_tool(tool, args: dict) -> str:
    ctx = MockToolContext()
    return await tool.on_invoke_tool(ctx, json.dumps(args))

result = await call_tool(find_reference_images, {"effect_type": "explosion"})
```

---

## File Reference

| File | Purpose |
|------|---------|
| `orchestrator.py` | Main VFX orchestrator agent |
| `specialized_agents/` | Subagent definitions |
| `tools/asset_evaluator_tools.py` | Quality evaluation tools (in-process) |
| `tools/blender_tools.py` | Blender script execution tools |

---

## Updating the SDK

```bash
# Check current version
pip show openai-agents

# Update to latest
pip install --upgrade openai-agents

# Pin in requirements.txt
echo "openai-agents>=0.6.8" >> requirements.txt
```

**Always test after SDK updates** - the API surface may change.

---

## Summary

1. **All agents must be subagents** under a central orchestrator
2. **Tools must run in-process** - no MCP subprocess calls
3. **Use the two-layer pattern** for tools that call other tools
4. **Consult official docs** before any architectural changes
5. **Test after SDK updates** - the API evolves rapidly

**When in doubt, read the docs:** https://github.com/openai/openai-agents-python/tree/main/docs
