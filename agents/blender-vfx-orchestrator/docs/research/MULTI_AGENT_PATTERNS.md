# Multi-Agent Orchestration Patterns

**Source:** [OpenAI Agents SDK - Multi Agent](https://openai.github.io/openai-agents-python/multi_agent/)
**Relevance:** Official patterns for coordinating multiple agents

---

## Two Primary Orchestration Approaches

### 1. LLM-Driven Orchestration

The LLM autonomously decides which tools/agents to invoke.

```python
coordinator = Agent(
    name="Coordinator",
    instructions="You coordinate VFX asset creation...",
    tools=[
        research_agent.as_tool(...),
        script_writer.as_tool(...),
        executor.as_tool(...),
    ],
)

# LLM decides order and when to call each
result = await Runner.run(coordinator, user_request)
```

**Pros:**
- Flexible, handles unexpected situations
- Can reason about task decomposition

**Cons:**
- Less predictable timing/cost
- May make suboptimal routing decisions
- Harder to debug

### 2. Code-Based Orchestration (RECOMMENDED)

Deterministic control flow with LLM decision points.

```python
async def create_asset_pipeline(request: AssetRequest) -> AssetResult:
    """Code controls flow, LLMs handle decisions."""

    # Phase 0: Research (LLM decides approach)
    research = await Runner.run(research_agent, request.description)

    # Phase 1: Generate script (LLM generates code)
    script = await Runner.run(script_writer, research.output)

    # Phase 2: Execute (deterministic)
    render_result = await execute_blender(script.path)

    # Phase 3: Evaluate (LLM interprets quality)
    quality = await Runner.run(quality_analyst, render_result.image)

    # Phase 4: Decide next step (code logic)
    if quality.score >= threshold:
        return AssetResult(success=True, ...)
    else:
        return await iterate(script, quality.feedback)
```

**Pros:**
- Predictable speed, cost, performance
- Easy to debug and trace
- Clear control flow

**Cons:**
- Less flexible for unexpected situations
- Requires upfront design

---

## Orchestration Tactics

### 1. Structured Outputs for Routing

Use Pydantic models to force structured decisions:

```python
class RoutingDecision(BaseModel):
    """Coordinator's routing decision."""
    next_agent: Literal["research", "script_writer", "executor", "quality"]
    reasoning: str
    confidence: float

coordinator = Agent(
    name="Coordinator",
    output_type=RoutingDecision,
    instructions="Decide which agent should handle this task...",
)
```

### 2. Sequential Chaining

Transform one agent's output into another's input:

```python
# Research → ScriptWriter → Executor → QualityAnalyst
async def sequential_chain(request: str) -> QualityMetrics:
    # Each step feeds into the next
    research_output = await Runner.run(research_agent, request)

    script_prompt = f"Based on this research: {research_output}\n\nGenerate script for: {request}"
    script_output = await Runner.run(script_writer, script_prompt)

    # ... continue chain
```

### 3. Feedback Loops

Run agents iteratively until quality threshold met:

```python
async def feedback_loop(
    script: str,
    threshold: float = 60.0,
    max_iterations: int = 5
) -> str:
    """Iterate until quality passes or max iterations reached."""

    for i in range(max_iterations):
        # Execute and evaluate
        render = await execute_blender(script)
        quality = await Runner.run(quality_analyst, render.image)

        if quality.score >= threshold:
            return script  # Success!

        # Modify based on feedback
        modify_prompt = f"Script has issues: {quality.feedback}\n\nModify to fix."
        script = await Runner.run(script_writer, modify_prompt)

    raise QualityThresholdNotMet(f"Failed after {max_iterations} iterations")
```

### 4. Parallel Execution

Run independent tasks concurrently:

```python
import asyncio

async def parallel_research(request: str) -> Dict[str, Any]:
    """Research multiple aspects in parallel."""

    tasks = [
        Runner.run(docs_expert, f"Find Blender docs for: {request}"),
        Runner.run(pattern_agent, f"Find code patterns for: {request}"),
        Runner.run(reference_agent, f"Find reference images for: {request}"),
    ]

    results = await asyncio.gather(*tasks)

    return {
        "docs": results[0].final_output,
        "patterns": results[1].final_output,
        "references": results[2].final_output,
    }
```

---

## Agents as Tools Pattern

Convert agents to tools for hierarchical composition:

```python
from agents import Agent, function_tool

# Specialized agent
script_writer = Agent(
    name="ScriptWriter",
    instructions="Generate Blender Python scripts...",
    tools=[semantic_search, validate_api],
)

# Wrap as tool for coordinator
@function_tool
async def write_script(
    effect_type: str,
    parameters: Dict[str, Any],
    api_spec: Optional[str] = None
) -> str:
    """Generate a Blender script for the specified effect."""
    prompt = f"Generate {effect_type} script with params: {parameters}"
    if api_spec:
        prompt += f"\n\nUse ONLY these verified APIs:\n{api_spec}"

    result = await Runner.run(script_writer, prompt)
    return result.final_output

# Coordinator uses script_writer as a tool
coordinator = Agent(
    name="Coordinator",
    tools=[write_script, execute_script, evaluate_quality],
)
```

### Advantages

1. **Encapsulation**: Coordinator doesn't need to know ScriptWriter's internal tools
2. **Reusability**: Same agent can be used standalone or as tool
3. **Control returns**: After tool call, control returns to coordinator

---

## Handoffs (Use Sparingly)

Transfer control completely to another agent:

```python
from agents import Agent, handoff

research_agent = Agent(name="Research", ...)
script_agent = Agent(name="ScriptWriter", ...)

# Research hands off to ScriptWriter when done
research_agent = Agent(
    name="Research",
    handoffs=[script_agent],  # Can hand off to ScriptWriter
    instructions="Research the approach. When ready, hand off to ScriptWriter.",
)
```

**Warning:** Handoffs transfer control completely. Use agents-as-tools for more control.

---

## Your Current Architecture Analysis

```
Your Pipeline (Code-Based - CORRECT):

create_asset_pipeline()  ─────────────────────────────────────────┐
│                                                                  │
├── Phase 0.5: TechniqueSelector (LLM decision)                   │
├── Phase 1: ScriptWriter (LLM generation)                        │
├── Phase 1.5: APIValidator (LLM + rules validation)              │
├── Phase 2: Executor (deterministic)                             │
├── Phase 3: QualityAnalyst (LLM evaluation)                      │
├── Phase 4: Loop decision (code logic)                           │
│   ├── if quality >= threshold: return success                   │
│   └── else: goto Phase 1 with feedback                          │
└── Phase 5: LearningAgent (LLM pattern extraction)               │
```

**Assessment:** Your architecture follows the recommended code-based orchestration pattern. The LLM handles decisions at specific points, but code controls the flow.

---

## Recommended Improvements

### 1. Add Structured Routing Outputs

```python
class ModificationDecision(BaseModel):
    """What kind of modification to make."""
    action: Literal["modify_params", "regenerate", "try_different_technique"]
    target_params: Optional[List[str]]
    reasoning: str

modification_strategist = Agent(
    name="ModificationStrategist",
    output_type=ModificationDecision,
    instructions="Decide how to improve the script based on quality feedback...",
)
```

### 2. Add Parallel Research Phase

```python
async def enhanced_research(effect_type: str) -> ResearchOutput:
    """Parallel research from multiple sources."""

    docs_task = Runner.run(docs_expert, f"Blender 5.0 {effect_type} attributes")
    patterns_task = Runner.run(pattern_agent, f"Working {effect_type} code patterns")
    refs_task = Runner.run(reference_agent, f"Reference images for {effect_type}")

    docs, patterns, refs = await asyncio.gather(docs_task, patterns_task, refs_task)

    return ResearchOutput(
        verified_apis=docs.final_output,
        code_patterns=patterns.final_output,
        reference_images=refs.final_output,
    )
```

### 3. Add Quality-Gated Checkpoints

```python
async def quality_checkpoint(
    stage: str,
    output: Any,
    min_confidence: float = 0.7
) -> bool:
    """Verify output meets quality bar before proceeding."""

    result = await Runner.run(
        quality_gate_agent,
        f"Evaluate {stage} output: {output}. Minimum confidence: {min_confidence}"
    )

    if result.final_output.confidence < min_confidence:
        raise CheckpointFailed(f"{stage} failed quality gate: {result.final_output.reason}")

    return True
```
