# Multi-Agent Architecture Optimization Plan

**Date:** 2026-01-22
**Author:** Ben + Claude
**Status:** Active Implementation

---

## Executive Summary

This document captures the comprehensive analysis of the Blender VFX Orchestrator's multi-agent architecture and provides a detailed optimization plan based on OpenAI Agents SDK best practices.

**Key Finding:** The orchestrator is using **handoffs** when it should be using **agents-as-tools**. This fundamental pattern mismatch causes most of the reliability issues.

---

## Current State Assessment

### What's Working Well

| Component | Status | Notes |
|-----------|--------|-------|
| 5 Specialized Agents | ✅ Good | Research, Script Writer, Executor, Quality Analyst, Learning |
| SessionManager | ✅ Excellent | Deterministic Python-side state tracking |
| Self-Learning Strategies | ✅ Good | 5 strategies implemented (Vector store, Distillation, Proactive, Patterns, Escape) |
| Quality Evaluation | ✅ Good | Vision + LPIPS/CLIP/TOPIQ metrics |
| Structured Outputs | ✅ Good | Pydantic models for type-safe data passing |

### What's Not Working

| Problem | Severity | Impact |
|---------|----------|--------|
| Blender 5.0 API Errors | 🔴 Critical | Scripts fail immediately; iteration loop never completes |
| Research Loop Infinity | 🔴 Critical | Agents call search tools 15+ times; MaxTurnsExceeded |
| SDK Docs Not Used | 🟡 High | Training data patterns override SDK-specific approaches |
| Workflow Fragility | 🟡 High | Handoff mode unpredictable; pipeline mode loses intelligence |

---

## Root Cause Analysis

### Problem 1: Blender 5.0 API Errors

**Symptoms:**
- Scripts fail with `AttributeError` for non-existent properties
- Socket names changed between versions (`"Smoke"` → `"Grid"`)
- Properties removed or renamed in FluidDomainSettings

**Root Cause:**
The LLM (gpt-5.2) generates code from training data (Blender 2.8x-4.x) instead of querying Blender 5.0 documentation first.

**Why It Happens:**
No **enforcement mechanism** exists. Documentation tools are available but not REQUIRED before code generation. The instruction "ALWAYS query docs first" is not mechanically enforced.

**Example Failures:**
```python
# FAILS: Old API pattern from training data
domain.modifier.effector_weights.wind = 1.0  # Removed in 5.0

# FAILS: Old socket names
principled_volume.inputs["Smoke"].default_value  # Now called "Grid"

# CORRECT: Blender 5.0 API
domain.effector_weights.wind = 1.0
principled_volume.inputs["Grid"].default_value
```

### Problem 2: Research Loop Infinity

**Symptoms:**
- Agent calls `search_blender_api_by_intent` 15+ times
- Never advances to Executor phase
- Hits `MaxTurnsExceeded` exception (25 turns)

**Root Cause:**
The Research Agent and Script Writer have documentation search tools but no explicit stopping condition. They keep searching for "more context" instead of producing output.

**Why It Happens:**
Missing **RunHooks** to detect and stop repeated tool calls. No "research done, produce output" guardrail.

**Trace Evidence:**
```
Turn 1:  search_blender_api_by_intent("mantaflow domain setup")
Turn 2:  search_blender_api_by_intent("smoke simulation parameters")
Turn 3:  semantic_search_blender_docs("fire effect Blender 5")
Turn 4:  search_blender_api_by_intent("FluidDomainSettings properties")
Turn 5:  search_blender_api_by_intent("FluidFlowSettings")
...
Turn 25: MaxTurnsExceeded
```

### Problem 3: SDK Documentation Not Being Used

**Symptoms:**
- SDK features underutilized (Tracing, RunHooks, Guardrails)
- Patterns from training data override SDK docs
- No mandatory consultation before changes

**Root Cause:**
The OpenAI Agents SDK is bleeding-edge technology not in LLM training data. Without explicit enforcement, older/different framework patterns override SDK-specific approaches.

**Critical SDK Features Being Underutilized:**

| Feature | What It Does | Why We Need It |
|---------|--------------|----------------|
| **Tracing** | Built-in execution monitoring | Debug agent behavior, view in dashboard |
| **RunHooks** | Lifecycle callbacks | Detect loops, enforce behavior |
| **Guardrails** | Input/output validation | Stop invalid requests early |
| **Handoffs** | Agent-to-agent transfer | Proper multi-agent coordination |
| **Agents as Tools** | Call agent as function | Utility agents that return control |
| **Structured Output** | Type-safe responses | Reliable data passing between agents |
| **Sessions** | Conversation persistence | Memory across runs |

### Problem 4: Workflow Logic Fragility

**Symptoms:**
- `create_asset()` (handoff-based): Unpredictable, loops, agents don't return
- `create_asset_pipeline()` (Python-controlled): More stable but less intelligent

**Root Cause:**
Using **handoffs** when **agents-as-tools** pattern is more appropriate.

**The SDK Pattern Distinction:**

| Pattern | Control Flow | Best For |
|---------|--------------|----------|
| **Handoffs** | Decentralized - new agent takes over conversation | Multi-turn dialogues, triage routing |
| **Agents-as-Tools** | Centralized - coordinator calls sub-agents and retains control | Workflows, pipelines |

**Your VFX pipeline is fundamentally a WORKFLOW**, not a dialogue. The coordinator should retain control and call specialists as tools.

---

## SDK Best Practices Reference

### 1. Tracing (ALWAYS USE)

```python
from agents import Agent, Runner, trace

with trace("Workflow Name", group_id=session_id):
    result = await Runner.run(agent, prompt, max_turns=10)

# View traces: https://platform.openai.com/traces
```

### 2. Handoffs (Agent Transfer - Decentralized)

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

### 3. Agents as Tools (Centralized Control) ⭐ RECOMMENDED

```python
# Agent called as a tool - returns control to caller
orchestrator = Agent(
    name="Orchestrator",
    tools=[
        specialist_agent.as_tool(
            tool_name="do_specialist_task",
            tool_description="What the specialist does and when to use it",
        ),
    ],
)
```

**Key Difference:**
- Handoffs: New agent takes over conversation completely
- as_tool: Agent called as utility, control returns to caller

### 4. RunHooks (Lifecycle Callbacks) ⭐ CRITICAL FOR ENFORCEMENT

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

### 5. Input/Output Guardrails

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

### 7. Sessions (Conversation Persistence)

```python
from agents import Agent, Runner, SQLiteSession

session = SQLiteSession("conversation_123", "history.db")

# First turn
result = await Runner.run(agent, "Question 1", session=session)

# Second turn - agent remembers previous context
result = await Runner.run(agent, "Follow-up", session=session)
```

---

## Proposed Architecture: Hybrid Orchestrator

### Current vs Proposed

```
CURRENT (Problematic):
┌─────────────────────────────────────┐
│     Orchestrator (handoffs)         │
│            ↓ ↑                      │
│  Script ←→ Executor ←→ Quality      │  ← Agents hand control back and forth
│            ↓ ↑                      │    (unpredictable, loops)
│         Learning                    │
└─────────────────────────────────────┘

PROPOSED (Reliable):
┌─────────────────────────────────────────────────────────────────────┐
│                 PipelineOrchestrator (Python)                       │
│                 ━━━━━━━━━━━━━━━━━━━━━━━━━━━━                        │
│  - Deterministic state machine (no LLM)                             │
│  - Calls agents-as-tools in sequence                                │
│  - Handles quality gate logic                                       │
│  - Manages escape velocity transitions                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐                                                    │
│  │ Coordinator │ ← Reasoning agent for DECISIONS only               │
│  │   Agent     │   (which technique? should we switch?)             │
│  │  (gpt-5.2)  │   Uses agents-as-tools for sub-tasks               │
│  └─────────────┘                                                    │
│         │                                                           │
│         ├── research_agent.as_tool("research_approach")             │
│         ├── docs_expert.as_tool("search_blender_docs")              │
│         ├── script_writer.as_tool("generate_script")                │
│         ├── executor.as_tool("execute_script")                      │
│         ├── quality_analyst.as_tool("evaluate_render")              │
│         └── learning_agent.as_tool("record_experiment")             │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

                         ↓ Uses RunHooks for enforcement ↓

┌─────────────────────────────────────────────────────────────────────┐
│                        EnforcementHooks                             │
├─────────────────────────────────────────────────────────────────────┤
│  LoopDetectionHook     - Max 3 calls to same tool                   │
│  DocQueryRequiredHook  - Block generate_script if no prior doc query│
│  TurnBudgetHook        - Hard limit on turns per agent              │
│  OutputValidationHook  - Verify structured output completeness      │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Architecture Decisions

1. **Keep `create_asset_pipeline()`, deprecate `create_asset()`**
   - Python-controlled pipeline is the right approach
   - Handoff-based method fights against SDK design intent

2. **SessionManager + SDK Sessions = Best of Both**
   - SessionManager: tracks params, scores, issues (deterministic Python)
   - SQLiteSession: tracks conversation history (SDK's persistence)

3. **Mechanical Enforcement Over Instructions**
   - LLMs don't reliably follow "ALWAYS do X" instructions
   - RunHooks: Block tools if prerequisites not met
   - Guardrails: Validate inputs/outputs
   - Turn budgets: Hard limits on agent reasoning

4. **API Validation as Separate Agent**
   - Don't embed Blender 5.0 validation in Script Writer instructions
   - Create dedicated validator agent called via `as_tool()`

---

## Implementation Plan

### Phase 1: RunHooks for Loop Detection and Enforcement ⭐ PRIORITY: CRITICAL

**Effort:** 2 hours
**Impact:** Immediately stops infinite research loops (Problem 2) and enforces doc queries (Problem 3)

**Implementation:**
- Create `hooks/enforcement_hooks.py`
- Add `EnforcementHooks` class with:
  - Loop detection (max 3 calls to same tool)
  - Doc query requirement before code generation
  - Turn budget enforcement
- Integrate with `create_asset_pipeline()`

### Phase 2: Convert to Agents-as-Tools Pattern

**Effort:** 4 hours
**Impact:** Coordinator retains control, predictable workflow (Problem 4)

**Implementation:**
- Create `_coordinator` agent that uses `as_tool()` for all sub-agents
- Remove handoff-based orchestration
- Update `create_asset_pipeline()` to use coordinator for decisions

### Phase 3: Add Input/Output Guardrails

**Effort:** 2 hours
**Impact:** Validation layer for all agent inputs/outputs

**Implementation:**
- Create `guardrails/script_guardrails.py`
- Add `require_research_context` input guardrail
- Add `validate_script_output` output guardrail
- Apply to Script Writer agent

### Phase 4: Implement SDK Sessions for Persistence

**Effort:** 2 hours
**Impact:** Conversation memory across runs

**Implementation:**
- Add `SQLiteSession` to all `Runner.run()` calls
- Create session database at `sessions/vfx_conversations.db`
- Integrate with existing SessionManager

### Phase 5: Add Turn Budget Per Agent

**Effort:** 1 hour
**Impact:** Prevents runaway agents

**Implementation:**
- Create `TurnBudgetMonitor` utility
- Use `agent.clone()` to inject turn budget instructions
- Apply to all standalone agents

### Phase 6: Create Blender API Validation Agent ⭐ PRIORITY: HIGH

**Effort:** 3 hours
**Impact:** Fixes Blender 5.0 API errors at source (Problem 1)

**Implementation:**
- Create `specialized_agents/api_validator.py`
- Add `validate_api_call` function tool
- Integrate with Script Writer via `as_tool()`
- Require API validation before any `bpy.types` or `bpy.ops` usage

### Phase 7: Enable Tracing Everywhere ⭐ PRIORITY: HIGH

**Effort:** 1 hour
**Impact:** Full visibility into agent behavior

**Implementation:**
- Wrap every `Runner.run()` in `trace()`
- Use nested traces for pipeline phases
- Add metadata for filtering (effect type, iteration, score)

---

## Implementation Priority Matrix

| Phase | Task | Effort | Impact | Priority |
|-------|------|--------|--------|----------|
| 1 | RunHooks for loop detection | 2h | Stops infinite loops | 🔴 CRITICAL |
| 6 | API Validator agent | 3h | Fixes Blender 5.0 errors | 🔴 CRITICAL |
| 2 | Agents-as-tools pattern | 4h | Coordinator retains control | 🟡 HIGH |
| 3 | Input/Output guardrails | 2h | Validation layer | 🟡 HIGH |
| 7 | Tracing everywhere | 1h | Visibility | 🟡 HIGH |
| 4 | SDK Sessions | 2h | Persistence | 🟢 MEDIUM |
| 5 | Turn budget per agent | 1h | Runaway prevention | 🟢 MEDIUM |

**Total Effort:** ~15 hours

---

## Quick Wins

### Immediate (< 30 min)

1. **Enable tracing** - Add `with trace()` to existing `create_asset_pipeline()`
2. **Reduce max_turns** - Change from 25 to 10 for most agents
3. **Add stderr logging** - Print tool calls as they happen

### Short-term (2-4 hours)

1. **Implement EnforcementHooks** (Phase 1)
2. **Add to all Runner.run() calls**
3. **Test with simple VFX generation**

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Research loops hitting MaxTurns | ~50% | < 5% |
| Blender API errors on first run | ~70% | < 10% |
| Scripts passing validation | ~30% | > 80% |
| Iterations to quality threshold | 5-10 | 3-5 |
| Cost per asset | $2-5 | $0.50-1 |

---

## References

- [OpenAI Agents SDK Documentation](https://github.com/openai/openai-agents-python/tree/main/docs)
- [Multi-Agent Patterns](https://github.com/openai/openai-agents-python/blob/main/docs/multi_agent.md)
- [Tools Reference](https://github.com/openai/openai-agents-python/blob/main/docs/tools.md)
- [Handoffs](https://github.com/openai/openai-agents-python/blob/main/docs/handoffs.md)
- [Guardrails](https://github.com/openai/openai-agents-python/blob/main/docs/guardrails.md)
- [Tracing](https://github.com/openai/openai-agents-python/blob/main/docs/tracing.md)
- [Sessions](https://github.com/openai/openai-agents-python/blob/main/docs/sessions.md)

---

*This document should be updated as implementation progresses.*
