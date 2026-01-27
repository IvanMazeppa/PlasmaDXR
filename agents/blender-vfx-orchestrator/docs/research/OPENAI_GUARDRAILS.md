# OpenAI Agents SDK Guardrails

**Source:** [OpenAI Agents SDK - Guardrails](https://github.com/openai/openai-agents-python/blob/main/docs/guardrails.md)
**Relevance:** Official patterns for input/output/tool validation

---

## Overview

Guardrails enable validation of:
- **User input** before agent processes it
- **Agent output** before returning to user
- **Tool calls** before and after execution

They run checks using fast/cheap models to prevent expensive model execution when issues are detected.

---

## Guardrail Types

| Type | When It Runs | Purpose |
|------|--------------|---------|
| **Input guardrail** | Before agent starts | Validate user request |
| **Output guardrail** | After agent finishes | Validate agent response |
| **Tool input guardrail** | Before tool executes | Validate tool arguments |
| **Tool output guardrail** | After tool executes | Validate tool result |

---

## Input Guardrails

### Execution Flow

1. Receive same input passed to agent
2. Run guardrail function → `GuardrailFunctionOutput`
3. Check if `.tripwire_triggered` is true
4. If triggered: raise `InputGuardrailTripwireTriggered`

### Execution Modes

| Mode | Behavior | Use Case |
|------|----------|----------|
| **Parallel** (default) | Guardrail runs concurrently with agent | Best latency |
| **Blocking** | Guardrail completes before agent starts | Cost optimization, prevent side effects |

### Example: Block Invalid Requests

```python
from pydantic import BaseModel
from agents import (
    Agent, GuardrailFunctionOutput, InputGuardrailTripwireTriggered,
    RunContextWrapper, Runner, TResponseInputItem, input_guardrail,
)

class ValidationOutput(BaseModel):
    is_valid_vfx_request: bool
    reasoning: str

guardrail_agent = Agent(
    name="VFX Request Validator",
    instructions="""Check if this is a valid VFX asset request.
    Valid requests specify an effect type (explosion, fire, smoke, etc.)
    Invalid: vague requests, non-VFX tasks, inappropriate content""",
    output_type=ValidationOutput,
    model="gpt-5-mini",  # Fast, cheap model for validation
)

@input_guardrail
async def vfx_request_guardrail(
    ctx: RunContextWrapper[None],
    agent: Agent,
    input: str | list[TResponseInputItem]
) -> GuardrailFunctionOutput:
    result = await Runner.run(guardrail_agent, input, context=ctx.context)
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=not result.final_output.is_valid_vfx_request,
    )

# Attach to orchestrator
orchestrator = Agent(
    name="VFX Orchestrator",
    instructions="Create VFX assets...",
    input_guardrails=[vfx_request_guardrail],
)
```

---

## Output Guardrails

### Execution Flow

1. Receive output produced by agent
2. Run guardrail function → `GuardrailFunctionOutput`
3. Check tripwire status
4. If triggered: raise `OutputGuardrailTripwireTriggered`

**Key constraint:** Output guardrails only run when agent is the **last** agent in a chain.

### Example: Validate Script Quality

```python
from agents import output_guardrail, GuardrailFunctionOutput

class ScriptValidationOutput(BaseModel):
    has_unverified_apis: bool
    unverified_calls: List[str]
    reasoning: str

script_validator_agent = Agent(
    name="Script Validator",
    instructions="""Analyze this Blender Python script.
    Check for unverified API calls - attributes that may not exist in Blender 5.0.
    Flag any suspicious patterns like:
    - resolution_divisions (should be resolution_max)
    - use_adaptive_time_steps (should be use_adaptive_timesteps)
    - velocity_multi (should be velocity_factor)""",
    output_type=ScriptValidationOutput,
    model="gpt-5-mini",
)

@output_guardrail
async def script_quality_guardrail(
    ctx: RunContextWrapper,
    agent: Agent,
    output: ScriptOutput
) -> GuardrailFunctionOutput:
    result = await Runner.run(
        script_validator_agent,
        output.script_content,
        context=ctx.context
    )
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=result.final_output.has_unverified_apis,
    )

script_writer = Agent(
    name="ScriptWriter",
    instructions="Generate Blender Python scripts...",
    output_guardrails=[script_quality_guardrail],
    output_type=ScriptOutput,
)
```

---

## Tool Guardrails

### Types

| Type | When | Can Do |
|------|------|--------|
| **Input guardrail** | Before tool executes | Skip call, replace output, trigger tripwire |
| **Output guardrail** | After tool executes | Replace output, trigger tripwire |

**Limitation:** Only apply to `function_tool` creations. Hosted tools (WebSearch, etc.) don't use this pipeline.

### Example: Block Unverified API Calls

```python
import json
from agents import (
    function_tool, tool_input_guardrail, tool_output_guardrail,
    ToolGuardrailFunctionOutput,
)

# Known hallucinated APIs to block
BLOCKED_APIS = [
    "resolution_divisions",
    "use_adaptive_time_steps",
    "velocity_multi",
    "time_scale",
    ".absolute_density",
]

@tool_input_guardrail
def block_hallucinated_apis(data) -> ToolGuardrailFunctionOutput:
    """Block tool calls that contain hallucinated API references."""
    args = json.loads(data.context.tool_arguments or "{}")
    args_str = json.dumps(args)

    for blocked_api in BLOCKED_APIS:
        if blocked_api in args_str:
            return ToolGuardrailFunctionOutput.reject_content(
                f"BLOCKED: '{blocked_api}' is a hallucinated API. "
                f"Use semantic_search_blender_docs to find the correct attribute."
            )

    return ToolGuardrailFunctionOutput.allow()

@tool_output_guardrail
def validate_script_output(data) -> ToolGuardrailFunctionOutput:
    """Validate generated script doesn't contain hallucinated APIs."""
    output = str(data.output or "")

    for blocked_api in BLOCKED_APIS:
        if blocked_api in output:
            return ToolGuardrailFunctionOutput.reject_content(
                f"REJECTED: Generated script contains hallucinated API '{blocked_api}'. "
                f"Regenerate with verified APIs only."
            )

    return ToolGuardrailFunctionOutput.allow()

@function_tool(
    tool_input_guardrails=[block_hallucinated_apis],
    tool_output_guardrails=[validate_script_output],
)
def generate_blender_script(
    effect_type: str,
    parameters: dict
) -> str:
    """Generate a Blender Python script for the specified effect."""
    # ... implementation
```

---

## Tripwires

When a guardrail detects a problem, it signals via **tripwire**.

### Behavior

1. Tripwire triggers
2. System immediately raises exception:
   - `InputGuardrailTripwireTriggered`
   - `OutputGuardrailTripwireTriggered`
3. Agent execution halts

### Handling Tripwires

```python
from agents import InputGuardrailTripwireTriggered, OutputGuardrailTripwireTriggered

async def safe_run(agent: Agent, input: str) -> Result:
    try:
        result = await Runner.run(agent, input)
        return Result(success=True, output=result.final_output)

    except InputGuardrailTripwireTriggered as e:
        return Result(
            success=False,
            error=f"Invalid input: {e.guardrail_result.output_info}"
        )

    except OutputGuardrailTripwireTriggered as e:
        return Result(
            success=False,
            error=f"Invalid output: {e.guardrail_result.output_info}"
        )
```

---

## Application to blender-vfx-orchestrator

### Current State

You already have `EnforcementHooks` which implement similar patterns via RunHooks. The SDK guardrails provide a more structured approach.

### Recommended Additions

#### 1. Input Guardrail on Orchestrator

```python
# Validate VFX requests before starting expensive pipeline
orchestrator = Agent(
    name="VFXOrchestrator",
    input_guardrails=[
        vfx_request_validator,  # Is this a valid VFX request?
        budget_check_guardrail,  # Do we have budget?
    ],
)
```

#### 2. Output Guardrail on ScriptWriter

```python
# Validate script before passing to Executor
script_writer = Agent(
    name="ScriptWriter",
    output_guardrails=[
        api_verification_guardrail,  # All APIs verified?
        syntax_check_guardrail,       # Valid Python?
    ],
)
```

#### 3. Tool Guardrails on Script Generation

```python
@function_tool(
    tool_input_guardrails=[block_hallucinated_apis],
    tool_output_guardrails=[validate_script_output],
)
async def modify_script(...) -> str:
    """Modify script with API validation."""
```

### Integration with Existing Hooks

Your `EnforcementHooks` handle:
- Loop detection
- Doc query requirements
- Turn budgets

SDK guardrails handle:
- Content validation
- API verification
- Tripwire-based rejection

**They complement each other.** Use both:
- `EnforcementHooks` via `Runner.run(..., hooks=...)`
- SDK guardrails via agent definition

---

## Key Design Principles

1. **Guardrails colocate with agents** - Different agents have different guardrails
2. **Input guardrails** only execute if agent is **first** in chain
3. **Output guardrails** only execute if agent is **last** in chain
4. **Tool guardrails** run on every tool call regardless of position
5. **Use fast/cheap models** for guardrail validation to minimize cost/latency
