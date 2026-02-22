# Self-Learning Architecture for Blender VFX Orchestrator

**Date:** 2026-01-20
**Status:** Implementation in Progress
**Author:** Claude (assisted by Ben)

---

## Executive Summary

This document captures findings from analyzing the OpenAI Agents SDK documentation and proposes an architecture for true self-learning behavior in the VFX orchestrator. The key insight is that **hardcoded instructions prevent learning** - instead, we should use dynamic instructions that inject validated knowledge from experimentation.

---

## Problem Statement

### Observed Issue
After a 45-minute E2E test (80 tool calls, 5 iterations):
- All renders were identical (static white sphere)
- VDB files all ~170kb with no variation
- Animation showed no movement between frames

### Root Cause
Hardcoded physics instructions in `orchestrator.py:500-507`:
```python
## Domain Physics Context
- sun/star/nebula effects: SPACE environment → set gravity=0, buoyancy=0 (beta=0)
```

This ALWAYS applied for sun effects, removing ALL driving forces from Mantaflow:
- `gravity=0` → no external force
- `alpha=0` → no thermal buoyancy
- `beta=0` → no density buoyancy

With no forces, the fluid has nothing to make it evolve = static VDB.

### The Anti-Pattern
```
Current: User sees issue → Claude hardcodes fix → Agent applies blindly → Breaks something else
```

### The Desired Pattern
```
Desired: Agent observes issue → Agent researches cause → Agent experiments → Agent records learning → Agent applies fix intelligently
```

---

## OpenAI Agents SDK Analysis

### Features Currently Used
| Feature | Usage |
|---------|-------|
| `Agent`, `Runner`, `handoff`, `trace` | Core orchestration |
| `function_tool` decorators | Tool wrappers |
| Structured outputs (`BaseModel`) | Type-safe data passing |
| `RunContextWrapper[SharedContext]` | Typed context |
| Code-based orchestration | `create_asset_pipeline()` |

### Features NOT Used (High Impact for Self-Learning)

#### 1. Dynamic Instructions
**What it does:** Instructions can be a function that generates instructions at runtime based on context.

**SDK Pattern:**
```python
from agents import Agent, RunContextWrapper

def dynamic_instructions(
    context: RunContextWrapper[UserContext],
    agent: Agent[UserContext]
) -> str:
    # Access runtime context to customize instructions
    return f"The user's name is {context.context.name}. Help them."

agent = Agent[UserContext](
    name="Dynamic Agent",
    instructions=dynamic_instructions,  # Function, not string!
)
```

**How it helps self-learning:**
- Inject learnings from knowledge base at runtime
- Only inject validated rules (success_rate > 70%)
- Rules emerge from experimentation, not hardcoding

#### 2. Output Guardrails
**What it does:** Validate agent outputs before they proceed, with tripwire capability.

**SDK Pattern:**
```python
from agents import output_guardrail, GuardrailFunctionOutput

@output_guardrail
async def physics_guardrail(
    ctx: RunContextWrapper,
    agent: Agent,
    output: MessageOutput
) -> GuardrailFunctionOutput:
    violations = check_physics_sanity(output)
    return GuardrailFunctionOutput(
        output_info={"violations": violations},
        tripwire_triggered=len(violations) > 2
    )
```

**How it helps self-learning:**
- Detect physics violations before they break renders
- Log violations for pattern analysis
- Block only severe issues, warn on minor ones

#### 3. Lifecycle Hooks
**What it does:** Execute code before/after tool calls and handoffs.

**SDK Pattern:**
```python
class LearningHooks(AgentHooksBase):
    async def on_tool_end(self, ctx, agent, tool, result):
        # Auto-record tool outcomes
        if tool.name == "modify_script":
            record_modification(ctx, tool.arguments, result)
```

**How it helps self-learning:**
- Automatically correlate parameter changes with outcomes
- Build causal model without explicit agent action
- Reduce turn budget spent on manual recording

#### 4. ToolContext
**What it does:** Access tool metadata during execution.

**SDK Pattern:**
```python
from agents.tool_context import ToolContext

@function_tool
def my_tool(ctx: ToolContext[MyContext], arg: str) -> str:
    print(f"Tool: {ctx.tool_name}, Args: {ctx.tool_arguments}")
    return result
```

**How it helps self-learning:**
- Track which exact parameters were passed to tools
- Enable replay and analysis of tool sequences

---

## Proposed Architecture: Dynamic Instructions + KB Integration

### Overview
```
┌─────────────────────────────────────────────────────────────────┐
│                    VFX Orchestrator                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    │
│  │   Research   │────▶│Script Writer │────▶│   Executor   │    │
│  │    Agent     │     │  (dynamic)   │     │              │    │
│  └──────────────┘     └──────────────┘     └──────────────┘    │
│         │                    │                    │             │
│         │                    ▼                    │             │
│         │            ┌──────────────┐            │             │
│         │            │   Physics    │◀───────────┘             │
│         │            │  Guardrail   │                          │
│         │            └──────────────┘                          │
│         │                    │                                  │
│         ▼                    ▼                                  │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    │
│  │  Knowledge   │◀────│   Quality    │────▶│   Learning   │    │
│  │    Base      │     │   Analyst    │     │    Agent     │    │
│  └──────────────┘     └──────────────┘     └──────────────┘    │
│         │                                         │             │
│         └─────────────────────────────────────────┘             │
│                    (learnings fed back)                         │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

#### 1. Dynamic Instructions for Script Writer
```python
def dynamic_script_writer_instructions(
    ctx: RunContextWrapper[SharedContext],
    agent: Agent[SharedContext]
) -> str:
    """Generate instructions based on accumulated knowledge."""
    base = SCRIPT_WRITER_BASE_INSTRUCTIONS

    # Get effect type from context
    effect_type = ctx.context.session.request.effect_type.value

    # Query KB for VALIDATED learnings only
    kb_learnings = query_knowledge_base_sync(
        f"physics rules for {effect_type} with success_rate > 0.7"
    )

    if kb_learnings:
        base += "\n\n## Validated Physics Rules (learned from experiments)\n"
        for learning in kb_learnings:
            base += f"- {learning['rule']} (success: {learning['success_rate']:.0%})\n"
    else:
        base += "\n\n## Physics Rules\n"
        base += "- No validated rules yet. Use standard Blender defaults.\n"
        base += "- Observe outcomes and the Learning Agent will build knowledge.\n"

    return base
```

#### 2. Physics Observation Tool
```python
@function_tool
async def observe_physics_anomaly(
    effect_type: str,
    observation: str,
    expected_behavior: str,
    actual_behavior: str,
    suspected_parameters: str
) -> str:
    """
    Record a physics anomaly for the Learning Agent to analyze.

    The Quality Analyst uses this when render doesn't match expectations.
    """
    return _record_physics_observation_impl(
        effect_type=effect_type,
        observation=observation,
        expected=expected_behavior,
        actual=actual_behavior,
        suspected_params=json.loads(suspected_parameters)
    )
```

#### 3. Knowledge Extraction Pipeline
```
Iteration N:
  1. Script Writer generates script (with current KB rules)
  2. Executor runs script
  3. Quality Analyst evaluates render
  4. If anomaly detected: observe_physics_anomaly()
  5. Learning Agent correlates: {params} → {outcome}
  6. If score_delta >= 5: extract_successful_pattern()

After M iterations:
  - Patterns with success_rate > 70% get injected into dynamic instructions
  - Patterns with success_rate < 30% get flagged as "avoid"
```

---

## Implementation Plan

### Phase 1: Remove Hardcoding (COMPLETED)
- [x] Document the problem (this file)
- [x] Remove hardcoded physics rules from orchestrator.py
- [x] Remove hardcoded physics rules from script_writer.py

### Phase 2: Dynamic Instructions (COMPLETED)
- [x] Create `dynamic_instructions.py` module
- [x] Implement `dynamic_script_writer_instructions()`
- [x] Implement `dynamic_quality_analyst_instructions()`
- [x] Implement `dynamic_learning_agent_instructions()`
- [x] Add KB query function with success_rate filtering
- [x] Wire up to all three agents (Script Writer, Quality Analyst, Learning Agent)

### Phase 3: Physics Observation (COMPLETED)
- [x] Create `physics_observation_tools.py` module with:
  - `observe_physics_anomaly()` - Record unexpected behavior
  - `get_pending_observations()` - Get observations awaiting analysis
  - `correlate_observation()` - Link observations to causal parameters
  - `get_physics_patterns()` - Get established patterns
- [x] Add to Quality Analyst's toolset
- [x] Update Quality Analyst instructions to observe (not assume)

### Phase 4: Learning Integration (COMPLETED)
- [x] Update Learning Agent to process physics observations
- [x] Add `get_pending_observations` and `correlate_observation` tools
- [x] Success_rate tracking already exists in experiment tracker
- [x] Pattern validation threshold (70%) implemented in dynamic_instructions.py

### Phase 5: Guardrails (Future)
- [ ] Implement physics sanity guardrail
- [ ] Add tripwire for severe violations
- [ ] Integrate with orchestrator

---

## Files Modified/Created

### New Files
- `tools/dynamic_instructions.py` - Dynamic instruction generators with KB integration
- `tools/physics_observation_tools.py` - Physics anomaly observation and correlation
- `docs/SELF_LEARNING_ARCHITECTURE.md` - This document

### Modified Files
- `specialized_agents/script_writer.py` - Now uses dynamic instructions, removed `apply_space_physics_fix`
- `specialized_agents/quality_analyst.py` - Now uses dynamic instructions, added physics observation tools
- `specialized_agents/learning_agent.py` - Now uses dynamic instructions, processes physics observations
- `orchestrator.py` - Removed hardcoded physics rules from standalone agents

### Key Removals
- Removed `apply_space_physics_fix` tool from Script Writer
- Removed hardcoded "Domain Physics Context" sections from all agents
- Removed effect-type-specific physics assumptions

### Phase 5.5: Bug Fixes (2026-01-21)

After E2E testing revealed iterations showing no meaningful differences:

**Bug 1: `modify_script` regex too broad**
- Problem: Regex pattern `rf"(\b{param}\s*=\s*)([^\n,\)]+)"` matched ANY `param = value` anywhere in file
- Result: Changes applied to `parse_args()` instead of `Config` class
- Fix: New regex now finds Config class boundaries first, then modifies ONLY within that section

**Bug 2: Hardcoded SPACE PHYSICS in `generate_script`**
- Problem: Lines 452-458 still had `if is_space_effect: domain_params["beta"] = 0.0`
- Result: ALL sun/nebula effects got beta=0.0 regardless of learning
- Fix: Removed hardcoded override, added comment explaining self-learning approach

**Bug 3: `apply_space_physics_fix` tool still existed**
- Problem: Tool definition remained at lines 933-1087 despite documentation saying it was removed
- Result: Agents could still call it, bypassing self-learning
- Fix: Removed entire function and @function_tool wrapper

---

## Alternative Approaches Evaluated

| Approach | Feasibility | Verdict |
|----------|-------------|---------|
| **OpenAI Evals API** | Medium | Good for offline trace analysis, but won't fix runtime learning |
| **Fine-tuning on SDK docs** | Low | SDK evolves rapidly; fine-tuning becomes stale |
| **RAG over SDK docs** | Medium | Could add SDK docs to existing vector store |
| **Pipeline Optimizer Agent** | High | **Best complement** to dynamic instructions |

---

## Success Metrics

1. **No identical renders** - VDB files should vary in size across frames
2. **Score progression** - Quality should improve across iterations
3. **Knowledge accumulation** - KB should grow with validated patterns
4. **Reduced hardcoding** - System prompts should be generic, not effect-specific

---

## References

- [OpenAI Agents SDK Documentation](https://github.com/openai/openai-agents-python/tree/main/docs)
- [Agents SDK Examples](https://github.com/openai/openai-agents-python/tree/main/examples)
- [Dynamic Instructions Pattern](https://openai.github.io/openai-agents-python/agents/#dynamic-instructions)
- [Guardrails Documentation](https://openai.github.io/openai-agents-python/guardrails/)
- [Lifecycle Hooks](https://openai.github.io/openai-agents-python/agents/#lifecycle-events-hooks)
