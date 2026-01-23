# Dynamic Instructions Guide — Blender VFX Orchestrator

**Version:** 3.4.0
**Last Updated:** 2026-01-23
**Status:** Fully Implemented

This guide provides comprehensive documentation on how dynamic instructions work in this project, including architecture, implementation details, edge cases, and guidance for AI agents.

---

## Table of Contents

1. [What Are Dynamic Instructions?](#what-are-dynamic-instructions)
2. [Architecture Overview](#architecture-overview)
3. [How It Works (Step by Step)](#how-it-works-step-by-step)
4. [The Three-Layer Architecture](#the-three-layer-architecture)
5. [Knowledge Base Integration](#knowledge-base-integration)
6. [Edge Cases and Failure Modes](#edge-cases-and-failure-modes)
7. [AI Agent Guidelines](#ai-agent-guidelines)
8. [Implementation Reference](#implementation-reference)
9. [Troubleshooting](#troubleshooting)

---

## What Are Dynamic Instructions?

Dynamic instructions are **functions that generate agent instructions at runtime** instead of static strings defined at agent creation time.

### Why This Matters

**Static Instructions (Old Pattern):**
```python
agent = Agent(
    name="Script Writer",
    instructions="You are a script writer. Always use X technique..."  # Fixed forever
)
```

**Dynamic Instructions (Current Pattern):**
```python
agent = Agent(
    name="Script Writer",
    instructions=dynamic_script_writer_standalone_instructions,  # Function!
)
```

The function is called **at the start of each agent run**, allowing instructions to change based on:
- Knowledge base learnings (validated physics rules)
- Current session context (effect type, iteration count)
- Accumulated experiment data

### SDK Specification

From the OpenAI Agents SDK:
- `Agent.instructions` accepts a **string** OR a **callable**
- Callable signature: `(ctx: RunContextWrapper[T], agent: Agent[T]) -> str`
- Can be sync or async
- Must return a string
- Called once at the START of each `Runner.run()` invocation

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    orchestrator.py                               │
│                                                                  │
│  self._script_agent_standalone = Agent(                         │
│      instructions=dynamic_script_writer_standalone_instructions │
│  )                                                               │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          │ At Runner.run() start, SDK calls:
                          │ instructions(ctx, agent)
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│              tools/dynamic_instructions.py                       │
│                                                                  │
│  def dynamic_script_writer_standalone_instructions(ctx, agent):  │
│      base = dynamic_script_writer_instructions(ctx, agent)       │
│      return base + _SCRIPT_WRITER_STANDALONE_EXTRAS              │
│                         │                                        │
│                         ▼                                        │
│  def dynamic_script_writer_instructions(ctx, agent):             │
│      base = SCRIPT_WRITER_BASE_INSTRUCTIONS                      │
│      learnings = query_validated_learnings(effect_type, ...)     │
│      return base + format_learnings_as_instructions(learnings)   │
│                         │                                        │
│                         ▼                                        │
│  query_validated_learnings() → _query_knowledge_base_impl()      │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│            tools/experiment_tracker_tools.py                     │
│                                                                  │
│  _query_knowledge_base_impl(query) → JSON results                │
│  Returns validated learnings with success_rate >= threshold      │
└─────────────────────────────────────────────────────────────────┘
```

---

## How It Works (Step by Step)

### 1. Agent Creation (orchestrator.py initialize())

```python
from tools.dynamic_instructions import dynamic_script_writer_standalone_instructions

self._script_agent_standalone = Agent[SharedContext](
    name="Script Writer",
    instructions=dynamic_script_writer_standalone_instructions,  # Store function reference
    ...
)
```

At this point, the function is **not called** — it's just stored.

### 2. Pipeline Execution (create_asset_pipeline)

```python
script_result = await Runner.run(
    self._script_agent_standalone,
    prompt,
    context=shared_context,  # SharedContext with session info
    session=sdk_session,     # SDK session for conversation history
)
```

### 3. SDK Calls Instruction Function

**Inside the SDK**, before the agent starts processing:

```python
# SDK internal logic (simplified)
if callable(agent.instructions):
    instructions_text = agent.instructions(ctx, agent)
else:
    instructions_text = agent.instructions
```

### 4. Instruction Function Executes

```python
def dynamic_script_writer_standalone_instructions(ctx, agent) -> str:
    # Step 4a: Call base dynamic function
    base = dynamic_script_writer_instructions(ctx, agent)

    # Step 4b: Append pipeline-specific rules
    return base + _SCRIPT_WRITER_STANDALONE_EXTRAS
```

### 5. Base Function Queries Knowledge Base

```python
def dynamic_script_writer_instructions(ctx, agent) -> str:
    base = SCRIPT_WRITER_BASE_INSTRUCTIONS  # Static base text

    # Extract effect type from context (with safe fallbacks)
    effect_type = None
    try:
        if hasattr(ctx, 'context') and ctx.context:
            if hasattr(ctx.context, 'session') and ctx.context.session:
                effect_type = ctx.context.session.request.effect_type.value
    except Exception:
        pass  # Fallback to None

    # Query knowledge base for validated learnings
    learnings = []
    if effect_type:
        learnings.extend(query_validated_learnings(effect_type, "physics", 0.7))
        learnings.extend(query_validated_learnings(effect_type, "visual", 0.7))

    # Format and append learnings
    if learnings:
        base += format_learnings_as_instructions(learnings)
    else:
        base += "\n\n## Physics Rules\nNo validated rules yet..."

    return base
```

### 6. Agent Receives Complete Instructions

The agent now has instructions that include:
1. **Base instructions** (always present)
2. **KB-injected learnings** (if available and validated)
3. **Pipeline-specific rules** (efficiency, output format)

---

## The Three-Layer Architecture

### Layer 1: Base Instructions (Static Constants)

Location: `tools/dynamic_instructions.py`

```python
SCRIPT_WRITER_BASE_INSTRUCTIONS = """## ROLE
YOU ARE THE CODE GENERATOR...

## TURN BUDGET: MAX 5 TURNS
...

## KNOWN BLENDER 5.0 API CHANGES
...
"""
```

These are **always present** and define the core behavior.

### Layer 2: KB-Injected Learnings (Dynamic)

Injected at runtime based on:
- Effect type (sun, explosion, fire, etc.)
- Validation threshold (success_rate >= 0.7 for most agents)

Example output:
```
## Validated Rules for sun (learned from experiments)
- [sun] Emission strength 15-20 works best for corona (success: 85%)
- [sun] Blackbody intensity 5.0 prevents clipping (success: 78%)
```

### Layer 3: Pipeline-Specific Rules (Static Extras)

Location: `tools/dynamic_instructions.py` (constants)

```python
_SCRIPT_WRITER_STANDALONE_EXTRAS = """

## EFFICIENCY REQUIREMENT - CRITICAL
You have LIMITED turns (max 10). Be efficient:
...

## Output Requirements
After generating and validating the script, return a structured ScriptOutput...
"""
```

These are **always appended** by the standalone wrapper functions.

---

## Knowledge Base Integration

### How Learnings Get Into the KB

1. **Quality Analyst** observes physics anomalies → `observe_physics_anomaly()`
2. **Learning Agent** correlates observations → `correlate_observation()`
3. **Learning Agent** records experiment results → `record_experiment_result()`
4. Over multiple experiments, learnings accumulate with success rates

### How Learnings Are Retrieved

```python
def query_validated_learnings(effect_type, category, min_success_rate):
    kb_query = _get_knowledge_base()  # In-process function
    result = kb_query(f"{category} rules for {effect_type}")

    # Filter by success rate
    validated = [l for l in results if l.get("success_rate") >= min_success_rate]
    return validated
```

### Validation Thresholds

| Agent | Threshold | Rationale |
|-------|-----------|-----------|
| Script Writer | 0.7 (70%) | Only inject rules that usually work |
| Quality Analyst | 0.5 (50%) | Show both expected behaviors AND known problems |
| Learning Agent | 0.0 (all) | Needs full context to decide what to validate |

---

## Edge Cases and Failure Modes

### 1. Context is None or Missing Session

**Symptom:** No KB learnings injected, falls back to generic instructions.

**Why it happens:**
- `Runner.run()` called without `context=` parameter
- SharedContext doesn't have session attached
- Session doesn't have request attached

**Current handling:**
```python
try:
    effect_type = ctx.context.session.request.effect_type.value
except Exception:
    pass  # effect_type remains None → no effect-specific learnings
```

**Impact:** Agent works but without KB-injected rules. Not a failure, just reduced capability.

### 2. Knowledge Base Empty

**Symptom:** Instructions say "No validated rules yet for this effect type."

**Why it happens:**
- New effect type with no experiments
- Fresh database
- All learnings below threshold

**Impact:** Agent uses Blender defaults. Learning Agent will build knowledge from experiments.

### 3. Heavy I/O in Instruction Function

**Symptom:** Slow agent startup, potential timeouts.

**Why it's a risk:**
- Instruction function runs synchronously at agent start
- Network calls or database queries add latency

**Current mitigation:**
- `_query_knowledge_base_impl` is in-process (no network)
- No MCP calls inside instruction functions
- Results are small (filtered to validated learnings only)

### 4. Exception in Instruction Function

**Symptom:** SDK may raise error or use empty instructions.

**Why it's a risk:**
- Any unhandled exception breaks agent initialization

**Current mitigation:**
- All context access wrapped in try/except
- Fallback to base instructions on any error
- No exceptions propagate from instruction functions

### 5. Circular Import

**Symptom:** ImportError at module load time.

**Why it's a risk:**
- `dynamic_instructions.py` imports from `experiment_tracker_tools.py`
- If tracker imports from dynamic_instructions, circular dependency

**Current mitigation:**
- Lazy import of `_query_knowledge_base_impl` via `_get_knowledge_base()`
- Only imported when function is first called, not at module load

---

## AI Agent Guidelines

### What AI Agents Should Know

#### 1. Your Instructions May Change Between Runs

If you're the Script Writer and you run twice:
- Run 1: No KB rules for "sun" → generic instructions
- Run 2: After Learning Agent recorded success → "Emission strength 15-20 works best"

**Don't assume your instructions are static.**

#### 2. KB Rules Are Validated, Not Hardcoded

When you see:
```
## Validated Rules for sun (learned from experiments)
- Blackbody intensity 5.0 prevents clipping (success: 78%)
```

This means:
- The rule was tested in real experiments
- It worked 78% of the time
- You should follow it, but it's not absolute

**If the rule doesn't work for your specific case, the Learning Agent will update it.**

#### 3. No Rules ≠ Do Nothing

If you see:
```
## Physics Rules
No validated rules yet for this effect type.
```

This means:
- Use Blender defaults
- Observe outcomes
- The Learning Agent will build knowledge from your experiments

**Don't invent rules. Let the system learn.**

#### 4. Context Matters for Instructions

Your instructions are generated based on:
- **effect_type** from the session request
- **Knowledge base** state at that moment

If you're generating a "sun" effect, you get sun-specific rules.
If you're generating an "explosion" effect, you get explosion-specific rules.

#### 5. Efficiency Rules Are Non-Negotiable

The pipeline-specific rules (turn limits, output format) are always appended:
```
## EFFICIENCY REQUIREMENT - CRITICAL
You have LIMITED turns (max 10). Be efficient...
```

**These are not suggestions. They're constraints.**

### What AI Agents Should NOT Do

1. **Don't hardcode physics rules** — Let them come from the KB
2. **Don't ignore turn limits** — They're there to prevent runaway agents
3. **Don't assume context exists** — Always handle missing data gracefully
4. **Don't make heavy calls in instructions** — Keep instruction functions fast

---

## Implementation Reference

### Files Involved

| File | Purpose |
|------|---------|
| `tools/dynamic_instructions.py` | All instruction functions and constants |
| `orchestrator.py` | Agent creation with `instructions=function` |
| `tools/experiment_tracker_tools.py` | KB query implementation |

### Function Reference

#### Standalone Wrapper Functions (Use These)

```python
# For Script Writer standalone agent
dynamic_script_writer_standalone_instructions(ctx, agent) -> str

# For Quality Analyst standalone agent
dynamic_quality_analyst_standalone_instructions(ctx, agent) -> str

# For Learning Agent standalone agent
dynamic_learning_agent_standalone_instructions(ctx, agent) -> str
```

#### Base Dynamic Functions (Called by Wrappers)

```python
# Base Script Writer (KB injection only)
dynamic_script_writer_instructions(ctx, agent) -> str

# Base Quality Analyst (KB injection only)
dynamic_quality_analyst_instructions(ctx, agent) -> str

# Base Learning Agent (KB injection only)
dynamic_learning_agent_instructions(ctx, agent) -> str
```

#### Static Fallbacks (For Testing)

```python
# Returns base instructions without KB query
get_script_writer_instructions_static() -> str
get_quality_analyst_instructions_static() -> str
get_learning_agent_instructions_static() -> str
```

### Adding a New Dynamic Instruction Function

If you need to add dynamic instructions for a new agent:

```python
# 1. Define base instructions constant
NEW_AGENT_BASE_INSTRUCTIONS = """## ROLE
...
"""

# 2. Create base dynamic function
def dynamic_new_agent_instructions(ctx, agent) -> str:
    base = NEW_AGENT_BASE_INSTRUCTIONS

    # Extract context safely
    effect_type = None
    try:
        if hasattr(ctx, 'context') and ctx.context:
            # ... extract what you need
            pass
    except Exception:
        pass

    # Query KB and append learnings
    if effect_type:
        learnings = query_validated_learnings(effect_type, "category", 0.7)
        if learnings:
            base += format_learnings_as_instructions(learnings)

    return base

# 3. Define extras for standalone version
_NEW_AGENT_STANDALONE_EXTRAS = """
## Pipeline-Specific Rules
...
"""

# 4. Create standalone wrapper
def dynamic_new_agent_standalone_instructions(ctx, agent) -> str:
    base = dynamic_new_agent_instructions(ctx, agent)
    return base + _NEW_AGENT_STANDALONE_EXTRAS
```

---

## Troubleshooting

### Problem: Agent doesn't receive KB learnings

**Check:**
1. Is `Runner.run()` called with `context=shared_context`?
2. Does `shared_context.session` exist?
3. Does `shared_context.session.request` have `effect_type`?
4. Are there any learnings in the KB for that effect type?
5. Do learnings meet the success_rate threshold?

**Debug:**
```python
# In instruction function, add logging:
import sys
print(f"[DEBUG] ctx.context = {ctx.context}", file=sys.stderr)
print(f"[DEBUG] effect_type = {effect_type}", file=sys.stderr)
print(f"[DEBUG] learnings = {learnings}", file=sys.stderr)
```

### Problem: Instructions are empty or malformed

**Check:**
1. Does the instruction function raise any exceptions?
2. Is the return value a string?
3. Are there encoding issues in the base instructions?

**Debug:**
```python
# Test the function directly:
result = dynamic_script_writer_standalone_instructions(mock_ctx, mock_agent)
print(type(result))  # Should be <class 'str'>
print(len(result))   # Should be > 0
```

### Problem: Agent startup is slow

**Check:**
1. Is `_query_knowledge_base_impl` making network calls?
2. Are there heavy computations in the instruction function?
3. Is the KB query returning too many results?

**Fix:**
- Keep instruction functions fast (<100ms)
- Use in-process KB queries only
- Limit results with appropriate filters

### Problem: TypeError about function signature

**Check:**
1. Does the function accept exactly 2 parameters?
2. Are the parameter names `ctx` and `agent`?
3. Is the function returning a string?

**Correct signature:**
```python
def my_instructions(ctx: RunContextWrapper[SharedContext], agent: Agent[SharedContext]) -> str:
    return "..."
```

---

## Changelog

### v3.5.0 (2026-01-23) - Fallback Logging Implemented ✅

- **QW-2 Complete:** Added `logging` module to `tools/dynamic_instructions.py`
- All context extraction failures now logged via `logger.warning()`
- Includes error details and which fallback value was used
- Silent KB failures are now observable for debugging

### v3.4.0 (2026-01-23) - Dynamic Instructions Fully Implemented

- Created standalone wrapper functions for all three agents
- Orchestrator now uses function references instead of string concatenation
- Added comprehensive documentation

### v3.3.0 (2026-01-23) - Contradictions Resolved

- Fixed conflicting Blender 5.0 API guidance
- Added issue→parameter mapping to Learning Agent
- Removed prompt_with_handoff_instructions from standalone agents

---

*Documentation by Claude Code — 2026-01-23*
