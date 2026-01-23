# SDK Documentation Enforcement Protocol

**Purpose:** Ensure OpenAI Agents SDK documentation is ALWAYS consulted before making changes.

**Current SDK Version:** v0.6.9+ (2026-01-23)

---

## The Problem

The OpenAI Agents SDK is bleeding-edge technology not in LLM training data. Without explicit enforcement, patterns from training data (older or different frameworks) override SDK-specific approaches.

## The Solution: Mandatory Documentation Queries

### Before ANY Agent/Orchestrator Change

**Step 1: Query context7 for SDK patterns**
```
Query: /openai/openai-agents-python
Topic: [specific feature you're implementing]
```

**Step 2: Search OpenAI Developer Docs**
```
mcp__openaiDeveloperDocs__search_openai_docs
Query: "agents SDK [feature]"
```

**Step 3: Fetch specific doc if needed**
```
mcp__openaiDeveloperDocs__fetch_openai_doc
```

### SDK Documentation URLs (Bookmark These)

| Feature | URL |
|---------|-----|
| Multi-Agent Patterns | https://github.com/openai/openai-agents-python/blob/main/docs/multi_agent.md |
| Handoffs | https://github.com/openai/openai-agents-python/blob/main/docs/handoffs.md |
| Tools | https://github.com/openai/openai-agents-python/blob/main/docs/tools.md |
| Guardrails | https://github.com/openai/openai-agents-python/blob/main/docs/guardrails.md |
| Sessions | https://github.com/openai/openai-agents-python/blob/main/docs/sessions/index.md |
| Tracing | https://github.com/openai/openai-agents-python/blob/main/docs/tracing.md |
| Models | https://github.com/openai/openai-agents-python/blob/main/docs/models.md |

### context7 Library ID
```
/openai/openai-agents-python
```

---

## Key SDK Patterns Reference

### 1. Tracing (ALWAYS USE)

```python
from agents import Agent, Runner, trace

with trace("Workflow Name", group_id=session_id):
    result = await Runner.run(agent, prompt, max_turns=10)

# View traces: https://platform.openai.com/traces
```

### 2. Handoffs (Agent Transfer)

```python
from agents import Agent

specialist = Agent(
    name="Specialist",
    instructions="You handle specific tasks...",
    handoff_description="Use for X tasks",  # REQUIRED for triage
)

coordinator = Agent(
    name="Coordinator",
    handoffs=[specialist],  # List of agents that can be handed to
)
```

### 3. Agents as Tools (Utility Agents)

```python
# Agent called as a tool - returns control to caller
coordinator = Agent(
    tools=[
        helper_agent.as_tool(
            tool_name="helper_function",
            tool_description="What the helper does",
        ),
    ],
)
```

**Key Difference:**
- Handoffs: New agent takes over conversation
- as_tool: Agent called as utility, control returns to caller

### 4. Input/Output Guardrails

```python
from agents import Agent, input_guardrail, GuardrailFunctionOutput

@input_guardrail
async def validate_input(ctx, agent, input):
    # Validation logic
    return GuardrailFunctionOutput(
        tripwire_triggered=is_invalid,
        output_info={"reason": "why invalid"}
    )

agent = Agent(
    input_guardrails=[validate_input],
)
```

### 5. RunHooks (Lifecycle Callbacks)

```python
from agents import RunHooks

class MyHooks(RunHooks):
    async def on_agent_start(self, context, agent):
        print(f"Agent {agent.name} starting")

    async def on_tool_start(self, context, agent, tool):
        print(f"Tool {tool.name} called")

    async def on_tool_end(self, context, agent, tool, result):
        print(f"Tool {tool.name} returned: {result}")

    async def on_handoff(self, context, from_agent, to_agent):
        print(f"Handoff: {from_agent.name} -> {to_agent.name}")

result = await Runner.run(agent, prompt, run_hooks=MyHooks())
```

### 6. Structured Output

```python
from pydantic import BaseModel
from agents import Agent, AgentOutputSchema

class MyOutput(BaseModel):
    field1: str
    field2: int

agent = Agent(
    output_type=AgentOutputSchema(MyOutput, strict_json_schema=False),
)

result = await Runner.run(agent, prompt)
output = result.final_output_as(MyOutput)  # Type-safe!
```

### 7. SQLiteSession (Conversation Persistence)

```python
from agents import Agent, Runner, SQLiteSession

# Create agent
agent = Agent(name="Assistant", instructions="...")

# Create persistent session
session = SQLiteSession("conversation_123", "path/to/db.sqlite")

# All Runner.run() calls with same session share conversation history
result = await Runner.run(agent, "First message", session=session)
result = await Runner.run(agent, "Second message", session=session)  # Remembers first message
```

**Key Benefit:** Agents automatically remember previous context without manual history management.

---

## Enforcement Checklist

Before modifying orchestrator code, verify:

- [ ] Queried context7 for relevant SDK patterns
- [ ] Checked SDK docs for the feature being implemented
- [ ] Pattern matches SDK documentation exactly
- [ ] Tracing is enabled for the Runner.run() call
- [ ] No patterns from training data override SDK docs

---

## Tools Available for SDK Lookup

### MCP Tools
```
mcp__plugin_context7_context7__resolve-library-id
mcp__plugin_context7_context7__query-docs
mcp__openaiDeveloperDocs__search_openai_docs
mcp__openaiDeveloperDocs__fetch_openai_doc
mcp__openaiDeveloperDocs__list_openai_docs
```

### Usage Example
```python
# First resolve the library
resolve-library-id: "openai agents sdk python"
# Returns: /openai/openai-agents-python

# Then query
query-docs:
  libraryId: "/openai/openai-agents-python"
  query: "how to implement RunHooks lifecycle callbacks"
```

---

## Red Flags: Signs SDK Docs Weren't Consulted

1. Using `asyncio.create_task()` instead of SDK patterns
2. Manual conversation history management
3. Custom handoff logic instead of `handoffs=[]`
4. Missing `trace()` wrapper
5. No structured output for type safety
6. Hardcoded max_turns without understanding what a "turn" is

---

*This protocol is non-negotiable. SDK docs are the source of truth.*
