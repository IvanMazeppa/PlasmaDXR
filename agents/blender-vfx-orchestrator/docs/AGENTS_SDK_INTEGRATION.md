# OpenAI Agents SDK Integration Guide

**Last Updated:** 2026-01-26
**SDK Version:** 0.7.0

---

## Implementation Status (Updated 2026-01-23)

| Feature | Status | Notes |
|---------|--------|-------|
| Agents-as-tools pattern | ✅ COMPLETE | All sub-agents exposed via `as_tool()` |
| Native `max_turns` enforcement | ✅ COMPLETE | Using SDK v0.7.0 `as_tool(max_turns=X)` |
| Two-layer tool pattern | ✅ COMPLETE | `_impl` + `@function_tool` wrapper |
| RunHooks enforcement | ✅ IN USE | Hooks wired for Research, Script Writer, QA |
| Input/Output guardrails | ⚠️ PARTIAL | Script Writer + QA covered; extend to all agents |
| SQLiteSession persistence | ⏳ PENDING | Helper exists, not fully integrated |
| Handoffs deprecated | ✅ COMPLETE | Using code-based pipeline |

---

## SDK v0.7.0 Changes (2026-01-23)

**Breaking Changes:**
- Nested handoffs now disabled by default → **No impact** (we use agents-as-tools, not handoffs)
- Default `reasoning_effort` for GPT-5.1/5.2 changed from "low" to "none" → May need explicit setting if using reasoning models

**New Features:**
- `MCPServerManager` for simplified MCP server lifecycle management
- `session_input_callback` now optional (auto-appends to session history)
- Additional WebSocket customization options

---

## CRITICAL: Always Consult Official Documentation

**Before making ANY changes to the multi-agent system, consult the official documentation:**

- **Main Repository:** https://github.com/openai/openai-agents-python
- **Documentation:** https://github.com/openai/openai-agents-python/tree/main/docs
- **Key Docs:**
  - [Multi-Agent Patterns](https://github.com/openai/openai-agents-python/blob/main/docs/multi_agent.md)
  - [Tools Reference](https://github.com/openai/openai-agents-python/blob/main/docs/tools.md)
  - [Handoffs](https://github.com/openai/openai-agents-python/blob/main/docs/handoffs.md)
  - [Guardrails](https://github.com/openai/openai-agents-python/blob/main/docs/guardrails.md)
  - [Sessions](https://github.com/openai/openai-agents-python/blob/main/docs/sessions/index.md) ⭐ NEW
  - [Tracing](https://github.com/openai/openai-agents-python/blob/main/docs/tracing.md)

The SDK is rapidly evolving. At time of writing, we're on **v0.7.0**. Check for updates regularly.

---

## Architecture Overview

The blender-vfx-orchestrator uses the OpenAI Agents SDK with **code-based pipeline orchestration** and **agents-as-tools** pattern:

```
┌─────────────────────────────────────────────────────────────────┐
│                 create_asset_pipeline() (Python)                │
│                 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━                 │
│  - Deterministic state machine (Python controls flow)           │
│  - Calls agents via Runner.run() with RunHooks                  │
│  - Uses Coordinators for intelligent decisions                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Coordinator Agents (Decision Layer)                            │
│  ┌────────────────┐ ┌──────────────────┐ ┌────────────────┐    │
│  │TechniqueSelector│ │ModificationStrat.│ │QualityGateJudge│    │
│  └────────────────┘ └──────────────────┘ └────────────────┘    │
│                                                                 │
│  Specialized Agents (Execution Layer)                           │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐        │
│  │Research│ │Script  │ │Executor│ │Quality │ │Learning│        │
│  │Agent   │ │Writer  │ │        │ │Analyst │ │Agent   │        │
│  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘        │
│                                                                 │
│  Enforcement Layer                                              │
│  ┌────────────────────┐ ┌─────────────────────┐                │
│  │ RunHooks           │ │ Guardrails          │                │
│  │ (Tool-level)       │ │ (Agent-level)       │                │
│  └────────────────────┘ └─────────────────────┘                │
│                                                                 │
│  Persistence Layer                                              │
│  ┌─────────────────────────────────────────────┐               │
│  │ SDK Session (SQLiteSession)                 │               │
│  │ Shared conversation context across agents   │               │
│  └─────────────────────────────────────────────┘               │
└─────────────────────────────────────────────────────────────────┘
```

**Key Pattern:** Use `agents-as-tools` (NOT handoffs) for centralized control.

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

### 1. Agents as Tools (`agent.as_tool()`) ⭐ RECOMMENDED

Convert an agent into a tool that another agent can invoke. **This is the correct pattern for centralized control.**

```python
quality_analyst = Agent(name="Quality Analyst", ...)
orchestrator = Agent(
    name="Orchestrator",
    tools=[
        quality_analyst.as_tool(
            tool_name="evaluate_quality",
            tool_description="Evaluate render quality using vision + ML metrics",
        )
    ]
)
```

**Key Difference from Handoffs:**
- `as_tool()`: Agent called as utility, control returns to caller ✅
- Handoffs: New agent takes over conversation completely ❌

**Project-Specific Execution Order (Learning First)**
- The Learning Agent runs **before Script Writer** on every iteration.
- Its proposals **must include Blender 5 doc references**; missing refs trigger Docs Expert or rejection.
- New API usage requires **at least one** micro-experiment before full integration.

### 2. Handoffs (DEPRECATED for this project)

> **Note:** Handoffs are deprecated in favor of agents-as-tools. The handoff-based `create_asset()` method should NOT be used.

Transfer control between agents (for reference only):

```python
from agents import handoff

# ❌ NOT RECOMMENDED - use as_tool() instead
orchestrator = Agent(
    name="Orchestrator",
    handoffs=[
        handoff(target=script_writer, description="Write Blender scripts"),
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

### 4. Input/Output Guardrails ⭐ NEW

Validate agent inputs and outputs at the agent level:

```python
from agents import Agent, input_guardrail, output_guardrail, GuardrailFunctionOutput

@input_guardrail
async def require_research_context(ctx, agent, input):
    """Block if no research findings in prompt."""
    if not has_research_indicators(input):
        return GuardrailFunctionOutput(
            tripwire_triggered=True,
            output_info={"reason": "No research context found"}
        )
    return GuardrailFunctionOutput(
        tripwire_triggered=False,
        output_info={"status": "passed"}  # REQUIRED even for success
    )

@output_guardrail
async def validate_quality_output(ctx, agent, output):
    """Validate QualityOutput has required fields."""
    if output.overall_score < 0 or output.overall_score > 100:
        return GuardrailFunctionOutput(
            tripwire_triggered=True,
            output_info={"reason": "Score out of range 0-100"}
        )
    return GuardrailFunctionOutput(
        tripwire_triggered=False,
        output_info={"status": "passed"}
    )

agent = Agent(
    name="Quality Analyst",
    input_guardrails=[check_budget_before_quality],
    output_guardrails=[validate_quality_output],
)
```

**Critical Notes (SDK):**
- `GuardrailFunctionOutput` must include `output_info` even when passing.
- Guardrails raise tripwire exceptions when triggered.
- The decorator returns `InputGuardrail`/`OutputGuardrail` objects; access `.guardrail_function` for testing.

### 5. RunHooks (Lifecycle Callbacks) ⭐ NEW

Intercept tool calls for enforcement:

```python
from agents import RunHooks

class EnforcementHooks(RunHooks):
    async def on_tool_start(self, context, agent, tool):
        # Block infinite loops
        if self._tool_counts[tool.name] > self.config.max_same_tool_calls:
            raise LoopDetectedError(f"Tool {tool.name} called too many times")

        # Require research before scripting
        if tool.name in ["write_script", "modify_script"]:
            if not self._doc_query_made:
                raise DocQueryRequiredError("Research required before scripting")

result = await Runner.run(agent, prompt, hooks=EnforcementHooks())
```

**Factory Functions in `hooks/enforcement_hooks.py`:**
```python
create_research_hooks()        # max_same_tool=3, max_turns=8
create_script_writer_hooks()   # require_doc_query_before=[write_script, modify_script]
create_quality_analyst_hooks() # max_same_tool=3, max_turns=6
create_learning_agent_hooks()  # max_same_tool=3, max_turns=8
```

### 5.1 ModelSettings: tool_choice + parallel_tool_calls
Use `ModelSettings.tool_choice` to force a specific tool when required:

```python
from agents import ModelSettings

agent = Agent(
    name="Doc-First Agent",
    instructions="Always query docs before code.",
    tools=[blender_doc_search_bundle, write_script],
    model_settings=ModelSettings(tool_choice="blender_doc_search_bundle")
)
```

Use `parallel_tool_calls=False` to restrict to one tool call per turn when you need deterministic ordering.

### 6. SQLiteSession (Conversation Persistence) ⭐ NEW

Enable agents to remember previous conversation context:

```python
from agents import Agent, Runner, SQLiteSession

# Create persistent session tied to VFX session ID
session = SQLiteSession("session_20260123_explosion_001", "sessions/sdk/vfx_conversations.db")

# All agents share the same session for context awareness
result = await Runner.run(research_agent, "Research explosion effects", session=session)
result = await Runner.run(script_writer, "Generate script based on research", session=session)
# Script Writer can see what Research Agent found!
```

**Helper Function in `orchestrator.py`:**
```python
def get_or_create_sdk_session(session_id: str) -> SQLiteSession:
    SDK_SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    db_path = SDK_SESSIONS_DIR / "vfx_conversations.db"
    return SQLiteSession(session_id, str(db_path))
```

**Key Benefit:** Agents automatically maintain context across pipeline phases without manual conversation history management.

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

## SDK Compliance Findings (2026-01-23)

These items are **actionable gaps** between current usage and the SDK's documented behaviors.

### 1. Guardrails coverage in handoff flows
- **SDK behavior:** Input guardrails run **only for the first agent**, and output guardrails run **only for the last agent** in a handoff chain.
- **Impact:** If you keep the deprecated handoff pipeline, sub-agents will **not** get their guardrails.
- **Recommendation:** Prefer the code-based pipeline (standalone `Runner.run()` per agent). If a handoff path remains, explicitly re-run sub-agents as standalone runs when guardrail enforcement is required.
  - **Source:** Agents SDK `docs/guardrails.md`

### 2. Tool guardrails apply only to function tools
- **SDK behavior:** Tool guardrails run only for `@function_tool` tools.
- **Impact:** Guardrails **will not** apply to `agent.as_tool()` or hosted tools.
- **Recommendation:** Use RunHooks for cross-tool enforcement, and add tool guardrails only where applicable.

### 3. `agent.as_tool()` now supports `max_turns` ✅ RESOLVED (SDK v0.6.9+)
- **SDK behavior (Updated):** As of SDK v0.6.9+, `agent.as_tool()` now accepts `max_turns` directly:

```python
research_agent.as_tool(
    tool_name="research_approach",
    tool_description="Research best approach for effect type",
    max_turns=4,  # NOW SUPPORTED NATIVELY
)
```

- **Implementation (2026-01-23):** Orchestrator now uses native `as_tool(max_turns=X)`:
  - Research: 4 turns
  - Script Writer: 6 turns
  - Executor: 3 turns
  - Quality Analyst: 3 turns
  - Learning: 5 turns
- **Legacy note:** The custom wrapper approach (below) is still valid for additional custom logic:

```python
@function_tool
async def run_quality_agent(render_path: str) -> str:
    result = await Runner.run(
        quality_agent,
        f"Evaluate {render_path}",
        max_turns=4
    )
    return result.final_output
```

### 4. Handoff prompt injection should be scoped
- **SDK behavior:** The handoff prompt prefix is recommended for **agents that actually hand off**.
- **Impact:** Using `prompt_with_handoff_instructions()` on non-handoff agents adds noise.
- **Recommendation:** Use the prefix only on agents with `handoffs=[...]`.

### 5. Runner.run uses `hooks=...`
- **SDK behavior:** The `Runner.run(...)` signature includes `hooks` for RunHooks enforcement.
- **Recommendation:** Keep all enforcement hooks wired via `hooks=...` (not `run_hooks=`) to match SDK API.

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
echo "openai-agents>=0.7.0" >> requirements.txt
```

**Always test after SDK updates** - the API surface may change.

---

---

## Defense-in-Depth Pattern

The system uses TWO validation layers for robust agent behavior:

```
Agent Input → Input Guardrails → Agent Reasoning → Tool Call
                                                       ↓
                                              RunHooks.on_tool_start()
                                                       ↓
                                                 Tool Execution
                                                       ↓
                                              RunHooks.on_tool_end()
                                                       ↓
Agent Output ← Output Guardrails ← Agent Response ←────┘
```

| Layer | Location | Purpose | Example |
|-------|----------|---------|---------|
| **RunHooks** | Tool level | Block problematic tool calls | Prevent script write without research |
| **Guardrails** | Agent level | Validate I/O structure | Ensure valid TechniqueDecision output |

---

## Summary

1. **Use agents-as-tools** (NOT handoffs) for centralized control
2. **Tools must run in-process** - no MCP subprocess calls
3. **Use the two-layer pattern** for tools that call other tools
4. **Add guardrails** for agent input/output validation
5. **Add RunHooks** for tool-level enforcement
6. **Consult official docs** before any architectural changes
7. **Test after SDK updates** - the API evolves rapidly

**When in doubt, read the docs:** https://github.com/openai/openai-agents-python/tree/main/docs

---

## File Reference (Updated)

| File | Purpose |
|------|---------|
| `orchestrator.py` | Main VFX orchestrator with pipeline and Coordinators |
| `specialized_agents/` | Subagent definitions (7 agents) |
| `specialized_agents/api_validator.py` | Blender 5.0 API validation |
| `tools/asset_evaluator_tools.py` | Quality evaluation tools (in-process) |
| `tools/blender_tools.py` | Blender script execution tools |
| `hooks/enforcement_hooks.py` | RunHooks for loop/doc enforcement |
| `guardrails/` | Input/output guardrails (9 guardrails) |
