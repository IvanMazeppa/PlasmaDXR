# Blender Librarian Enhancement Recommendations

**Date:** 2026-01-06
**Status:** Analysis Complete
**Based on:** OpenAI Agents SDK documentation + current implementation review

---

## Executive Summary

The blender-librarian agent system is already well-architected with multi-agent handoffs, MCP connection pooling, budget tracking, and streaming for timeout resilience. However, the OpenAI Agents SDK offers several powerful features that are **not currently utilized**. This document outlines 12 enhancement opportunities prioritized by impact and implementation complexity.

---

## Current Architecture Strengths

| Feature | Implementation | Status |
|---------|---------------|--------|
| Multi-agent handoffs | doc_expert, vision_expert | ✅ Implemented |
| MCP connection pooling | MCPConnectionPool singleton | ✅ Implemented |
| Streaming mode | Runner.run_streamed() | ✅ Implemented |
| Budget tracking | Dataclass + JSON persistence | ✅ Implemented |
| Dynamic reasoning effort | Query complexity estimation | ✅ Implemented |
| Playbook system | FREE known fixes | ✅ Implemented |
| Session learning | SQLite cross-session persistence | ✅ Implemented |

---

## Enhancement Opportunities

### Priority 1: High Impact, Medium Effort

#### 1. Structured Output Types (Pydantic Models)

**Current:** Returns JSON strings that callers must parse manually.

**Enhancement:** Use Pydantic models for type-safe, validated outputs.

**Benefits:**
- Automatic validation of agent outputs
- IDE autocomplete for downstream code
- Eliminates JSON parse errors
- Self-documenting response schemas

**Implementation:**

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class ModificationAdvice(BaseModel):
    """Structured output for get_modification_advice tool."""
    answer: str = Field(description="Summary of documentation findings")
    modifications: dict[str, float] = Field(description="Parameter changes to apply")
    rationale: str = Field(description="Why these changes should help")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in recommendation")
    citations: List[str] = Field(default_factory=list, description="Documentation paths used")

class RenderDiagnosis(BaseModel):
    """Structured output for diagnose_render_issue tool."""
    diagnosis: str = Field(description="Primary issue description")
    primary_issue: str = Field(description="Issue category name")
    severity: str = Field(pattern="^(critical|high|medium|low)$")
    secondary_issues: List[str] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)

# Usage in agent definition
doc_expert = Agent(
    name="Documentation Expert",
    instructions=DOC_EXPERT_INSTRUCTIONS,
    model="gpt-5.2",
    output_type=ModificationAdvice,  # <-- Structured output
    tools=[validate_parameter_range, get_parameter_defaults],
    mcp_servers=[mcp_server]
)
```

**Files to modify:**
- `librarian_agents/doc_expert.py` - Add output_type
- `librarian_agents/vision_expert.py` - Add output_type
- `librarian_agents/librarian_orchestrator.py` - Handle typed outputs
- `server.py` - Return Pydantic models from tools

---

#### 2. Guardrails for Input/Output Validation

**Current:** No validation of agent inputs or outputs beyond manual checks.

**Enhancement:** Use SDK guardrails for automatic validation with tripwires.

**Benefits:**
- Catch malformed inputs before API calls (save budget)
- Validate outputs match expected schema
- Tripwire pattern for early failure detection
- Audit trail for validation failures

**Implementation:**

```python
from agents import Agent, InputGuardrail, OutputGuardrail, GuardrailFunctionOutput
from pydantic import BaseModel

class QueryValidation(BaseModel):
    """Validation result for input queries."""
    is_valid: bool
    issues: List[str] = Field(default_factory=list)
    sanitized_query: Optional[str] = None

async def validate_query_guardrail(ctx, agent, input_data: str) -> GuardrailFunctionOutput:
    """Input guardrail to validate and sanitize queries."""
    issues = []

    # Check for empty or too-short queries
    if not input_data or len(input_data.strip()) < 10:
        issues.append("Query too short - provide more context")

    # Check for path injection attempts
    if ".." in input_data or "/etc/" in input_data:
        issues.append("Potentially unsafe path patterns detected")

    # Check for budget-expensive patterns without explicit request
    if "compare all" in input_data.lower() or "analyze everything" in input_data.lower():
        issues.append("Broad queries are expensive - be more specific")

    if issues:
        return GuardrailFunctionOutput(
            output_info=QueryValidation(is_valid=False, issues=issues),
            tripwire_triggered=len(issues) > 1  # Tripwire on multiple issues
        )

    return GuardrailFunctionOutput(
        output_info=QueryValidation(is_valid=True, sanitized_query=input_data.strip())
    )

async def validate_modification_output(ctx, agent, output: ModificationAdvice) -> GuardrailFunctionOutput:
    """Output guardrail to validate parameter modifications."""
    issues = []

    # Validate parameter ranges
    for param, value in output.modifications.items():
        if param in BLENDER_PARAMETER_RANGES:
            range_info = BLENDER_PARAMETER_RANGES[param]
            if not (range_info["min"] <= value <= range_info["max"]):
                issues.append(f"{param}={value} outside valid range [{range_info['min']}, {range_info['max']}]")

    # Validate confidence is reasonable
    if output.confidence > 0.9 and not output.citations:
        issues.append("High confidence without citations is suspicious")

    if issues:
        return GuardrailFunctionOutput(
            output_info={"valid": False, "issues": issues},
            tripwire_triggered=any("outside valid range" in i for i in issues)
        )

    return GuardrailFunctionOutput(output_info={"valid": True})

# Apply guardrails to agent
doc_expert = Agent(
    name="Documentation Expert",
    instructions=DOC_EXPERT_INSTRUCTIONS,
    model="gpt-5.2",
    output_type=ModificationAdvice,
    input_guardrails=[InputGuardrail(guardrail_function=validate_query_guardrail)],
    output_guardrails=[OutputGuardrail(guardrail_function=validate_modification_output)],
    tools=[...],
    mcp_servers=[mcp_server]
)
```

**Files to modify:**
- `librarian_agents/guardrails.py` (NEW) - Define guardrail functions
- `librarian_agents/doc_expert.py` - Add guardrails
- `librarian_agents/vision_expert.py` - Add guardrails
- `librarian_agents/librarian_orchestrator.py` - Handle tripwires

---

#### 3. Agent Lifecycle Hooks (AgentHooks)

**Current:** No observability into agent execution lifecycle.

**Enhancement:** Use AgentHooks for logging, metrics, and debugging.

**Benefits:**
- Trace tool calls and their results
- Log handoff decisions
- Measure latency per agent/tool
- Debug complex multi-agent workflows
- Feed data to experiment-tracker

**Implementation:**

```python
from agents import AgentHooks, RunContextWrapper, Tool, Agent
from typing import Any
import time

class LibrarianAgentHooks(AgentHooks):
    """Lifecycle hooks for observability and metrics."""

    def __init__(self, session_id: str = ""):
        self.session_id = session_id
        self.tool_timings: dict[str, list[float]] = {}
        self.handoff_chain: list[str] = []

    async def on_start(self, context: RunContextWrapper, agent: Agent) -> None:
        """Called when agent starts processing."""
        print(f"[{self.session_id}] Agent '{agent.name}' started")
        context.context["agent_start_time"] = time.time()

    async def on_end(self, context: RunContextWrapper, agent: Agent, output: Any) -> None:
        """Called when agent completes."""
        elapsed = time.time() - context.context.get("agent_start_time", time.time())
        print(f"[{self.session_id}] Agent '{agent.name}' completed in {elapsed:.2f}s")

        # Record to experiment-tracker if available
        if hasattr(context.context, "experiment_session"):
            await context.context["experiment_session"].record_agent_completion(
                agent_name=agent.name,
                elapsed_seconds=elapsed,
                output_preview=str(output)[:200]
            )

    async def on_tool_start(self, context: RunContextWrapper, agent: Agent, tool: Tool) -> None:
        """Called before a tool is invoked."""
        context.context[f"tool_{tool.name}_start"] = time.time()
        print(f"[{self.session_id}] Tool '{tool.name}' starting...")

    async def on_tool_end(self, context: RunContextWrapper, agent: Agent, tool: Tool, result: str) -> None:
        """Called after a tool completes."""
        start_time = context.context.get(f"tool_{tool.name}_start", time.time())
        elapsed = time.time() - start_time

        if tool.name not in self.tool_timings:
            self.tool_timings[tool.name] = []
        self.tool_timings[tool.name].append(elapsed)

        print(f"[{self.session_id}] Tool '{tool.name}' completed in {elapsed:.2f}s")

    async def on_handoff(self, context: RunContextWrapper, agent: Agent, target: Agent) -> None:
        """Called when handing off to another agent."""
        self.handoff_chain.append(f"{agent.name} -> {target.name}")
        print(f"[{self.session_id}] Handoff: {agent.name} -> {target.name}")

    def get_metrics_summary(self) -> dict:
        """Get summary of collected metrics."""
        return {
            "tool_timings": {
                name: {
                    "count": len(times),
                    "avg_ms": sum(times) / len(times) * 1000,
                    "max_ms": max(times) * 1000
                }
                for name, times in self.tool_timings.items()
            },
            "handoff_chain": self.handoff_chain
        }

# Usage
hooks = LibrarianAgentHooks(session_id="sess_12345")
orchestrator = Agent(
    name="Librarian Orchestrator",
    instructions=ORCHESTRATOR_INSTRUCTIONS,
    model="gpt-5.2",
    hooks=hooks,
    handoffs=[doc_expert_handoff, vision_expert_handoff]
)

# After run, get metrics
result = await Runner.run(orchestrator, query)
print(hooks.get_metrics_summary())
```

**Files to modify:**
- `librarian_agents/hooks.py` (NEW) - Define hook classes
- `librarian_agents/librarian_orchestrator.py` - Add hooks to agents
- `server.py` - Expose metrics via new MCP tool

---

### Priority 2: Medium Impact, Low Effort

#### 4. Handoff Input Filtering

**Current:** Full context passed to every agent on handoff.

**Enhancement:** Filter context to only what each agent needs.

**Benefits:**
- Reduce token usage (cost savings)
- Faster agent responses
- Cleaner agent context
- Security (don't leak sensitive data)

**Implementation:**

```python
from agents import handoff

def filter_for_doc_expert(input_data: dict) -> dict:
    """Filter context for documentation expert - only needs query and effect type."""
    return {
        "query": input_data.get("query", ""),
        "effect_type": input_data.get("effect_type", "general"),
        "current_params": input_data.get("current_params", {}),
        # Exclude: render_path, reference_path, session history, etc.
    }

def filter_for_vision_expert(input_data: dict) -> dict:
    """Filter context for vision expert - needs image paths and effect type."""
    return {
        "render_path": input_data.get("render_path", ""),
        "reference_path": input_data.get("reference_path", ""),
        "effect_type": input_data.get("effect_type", "general"),
        "known_issues": input_data.get("known_issues", []),
        # Exclude: query text, session history, budget info, etc.
    }

# Create handoffs with input filtering
doc_expert_handoff = handoff(
    agent=doc_expert,
    input_filter=filter_for_doc_expert
)

vision_expert_handoff = handoff(
    agent=vision_expert,
    input_filter=filter_for_vision_expert
)
```

**Files to modify:**
- `librarian_agents/librarian_orchestrator.py` - Add input_filter to handoffs

---

#### 5. Handoff Callbacks (on_handoff)

**Current:** No pre-processing before handoffs.

**Enhancement:** Use on_handoff for logging, context preparation, budget checks.

**Benefits:**
- Pre-validate before expensive agent calls
- Inject runtime context
- Log handoff decisions for debugging
- Check budget before vision calls

**Implementation:**

```python
from agents import handoff, RunContextWrapper

async def on_handoff_to_vision(ctx: RunContextWrapper) -> None:
    """Callback before handing off to vision expert."""
    # Check budget before expensive vision call
    budget = ctx.context.get("budget_tracker")
    if budget and budget.vision_remaining < 0.50:
        raise ValueError(f"Insufficient vision budget: ${budget.vision_remaining:.2f} remaining")

    # Log the handoff
    print(f"Handing off to vision expert for: {ctx.context.get('render_path', 'unknown')}")

    # Pre-warm image loading (optional optimization)
    render_path = ctx.context.get("render_path")
    if render_path:
        ctx.context["preloaded_image"] = await preload_image_async(render_path)

async def on_handoff_to_doc(ctx: RunContextWrapper) -> None:
    """Callback before handing off to documentation expert."""
    # Check if MCP connection is healthy
    pool = get_connection_pool()
    if not pool.is_connected:
        print("MCP connection lost, reconnecting...")
        await pool.get_server()

    # Log the handoff
    print(f"Handing off to doc expert for: {ctx.context.get('query', 'unknown')[:50]}...")

vision_expert_handoff = handoff(
    agent=vision_expert,
    on_handoff=on_handoff_to_vision,
    input_filter=filter_for_vision_expert
)

doc_expert_handoff = handoff(
    agent=doc_expert,
    on_handoff=on_handoff_to_doc,
    input_filter=filter_for_doc_expert
)
```

**Files to modify:**
- `librarian_agents/librarian_orchestrator.py` - Add on_handoff callbacks

---

#### 6. Tool Name Override for Handoffs

**Current:** Handoff tools use default names like "transfer_to_Documentation Expert".

**Enhancement:** Use semantic tool names that help the orchestrator decide.

**Benefits:**
- Clearer tool semantics for LLM
- Better handoff decisions
- Self-documenting agent API

**Implementation:**

```python
doc_expert_handoff = handoff(
    agent=doc_expert,
    tool_name_override="search_blender_documentation",  # Clear, action-oriented name
    tool_description_override="Search official Blender 5.0 documentation and Python API reference. Use for questions about Mantaflow, fluid simulation, VDB export, and bpy.* APIs.",
    input_filter=filter_for_doc_expert,
    on_handoff=on_handoff_to_doc
)

vision_expert_handoff = handoff(
    agent=vision_expert,
    tool_name_override="analyze_render_quality",  # Clear, action-oriented name
    tool_description_override="Analyze a render screenshot for visual quality issues. Use when you have an image path and need to diagnose rendering problems like wrong colors, missing features, or artifacts.",
    input_filter=filter_for_vision_expert,
    on_handoff=on_handoff_to_vision
)
```

**Files to modify:**
- `librarian_agents/librarian_orchestrator.py` - Add tool_name_override

---

### Priority 3: Medium Impact, Medium Effort

#### 7. Agents as Tools Pattern

**Current:** Only handoffs (full control transfer) are used.

**Enhancement:** Use agents as tools for sub-queries without full handoff.

**Benefits:**
- Orchestrator maintains control
- Can call multiple sub-agents in sequence
- Better for composite queries
- More predictable execution flow

**Use Case:** "Search for limb darkening documentation AND analyze the render" - orchestrator calls both as tools, synthesizes results.

**Implementation:**

```python
from agents import Agent, function_tool

# Wrap doc_expert as a callable tool
@function_tool
async def query_documentation(query: str, effect_type: str = "general") -> str:
    """
    Query Blender documentation for a specific topic.

    Args:
        query: The documentation question
        effect_type: Type of effect for context (sun, explosion, fire, nebula)

    Returns:
        JSON with answer, modifications, rationale, and citations
    """
    doc_agent = await create_doc_expert_pooled()
    result = await Runner.run(
        doc_agent,
        f"Effect type: {effect_type}\nQuery: {query}"
    )
    return result.final_output

@function_tool
async def analyze_image(render_path: str, reference_path: str = "", effect_type: str = "general") -> str:
    """
    Analyze a render image for quality issues.

    Args:
        render_path: Path to the render to analyze
        reference_path: Optional reference image for comparison
        effect_type: Type of effect (sun, explosion, fire, nebula)

    Returns:
        JSON with diagnosis, primary_issue, severity, and recommendations
    """
    vision_agent = create_vision_expert()
    prompt = f"Analyze this {effect_type} render: {render_path}"
    if reference_path:
        prompt += f"\nCompare against reference: {reference_path}"

    result = await Runner.run(vision_agent, prompt)
    return result.final_output

# Orchestrator with agents as tools (not handoffs)
orchestrator = Agent(
    name="Librarian Orchestrator",
    instructions="""You are the Blender Librarian orchestrator.

    You have two specialized tools:
    - query_documentation: For searching Blender docs
    - analyze_image: For analyzing render quality

    You can call both tools and synthesize their results.
    For complex queries, call multiple tools as needed.""",
    model="gpt-5.2",
    tools=[query_documentation, analyze_image, get_budget_status, add_to_playbook]
)
```

**When to use handoffs vs agents-as-tools:**
- **Handoffs:** When the sub-agent needs full autonomy (complex multi-step research)
- **Agents-as-tools:** When orchestrator needs to combine results from multiple agents

**Files to modify:**
- `librarian_agents/tool_wrappers.py` (NEW) - Wrap agents as tools
- `librarian_agents/librarian_orchestrator.py` - Add tool-wrapped agents

---

#### 8. LiteLLM Multi-Model Support

**Current:** Hardcoded to GPT-5.2 only.

**Enhancement:** Use LiteLLM for model flexibility.

**Benefits:**
- Fall back to cheaper models when budget is low
- Use Claude for some tasks (better at code)
- Use local models for privacy-sensitive data
- A/B test different models

**Implementation:**

```python
from agents import Agent, ModelSettings
from agents.extensions.models.litellm_model import LiteLLMModel

# Create different model configurations
MODELS = {
    "premium": LiteLLMModel(model="gpt-5.2"),
    "standard": LiteLLMModel(model="gpt-4o"),
    "budget": LiteLLMModel(model="gpt-4o-mini"),
    "claude": LiteLLMModel(model="claude-sonnet-4-20250514"),
    "local": LiteLLMModel(model="ollama/llama3.2:latest"),
}

def select_model_for_budget(budget_remaining: float, task_type: str) -> LiteLLMModel:
    """Select model based on budget and task type."""
    if budget_remaining < 1.0:
        return MODELS["budget"]
    elif task_type == "vision":
        return MODELS["premium"]  # Vision needs GPT-5.2
    elif task_type == "documentation":
        return MODELS["standard"]  # Docs work well with 4o
    else:
        return MODELS["premium"]

# Dynamic model selection in orchestrator
class AdaptiveOrchestrator:
    async def run(self, query: str, context: dict) -> str:
        budget = context.get("budget_tracker")
        task_type = estimate_task_type(query)

        model = select_model_for_budget(
            budget.total_remaining if budget else 10.0,
            task_type
        )

        # Create agent with selected model
        agent = Agent(
            name="Adaptive Librarian",
            instructions=ORCHESTRATOR_INSTRUCTIONS,
            model=model,
            tools=[...]
        )

        result = await Runner.run(agent, query)
        return result.final_output
```

**Files to modify:**
- `librarian_agents/model_selection.py` (NEW) - Model selection logic
- `librarian_agents/librarian_orchestrator.py` - Use adaptive model selection
- `requirements.txt` - Add litellm dependency

---

#### 9. Built-in Tracing Integration

**Current:** Custom logging, no structured tracing.

**Enhancement:** Use SDK's built-in tracing with external platform export.

**Benefits:**
- Structured trace data
- Integration with Langfuse, Logfire, Braintrust, etc.
- Waterfall visualization of agent execution
- Token usage tracking per agent

**Implementation:**

```python
from agents import set_tracing_export_api_key, set_trace_processors
from agents.tracing import TracingProcessor
import os

# Option 1: Use Langfuse (open-source)
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk_..."
os.environ["LANGFUSE_SECRET_KEY"] = "sk_..."
os.environ["LANGFUSE_HOST"] = "https://cloud.langfuse.com"

# Enable tracing export
set_tracing_export_api_key(os.environ["LANGFUSE_SECRET_KEY"])

# Option 2: Custom trace processor for local logging
class LocalTraceProcessor(TracingProcessor):
    def __init__(self, log_file: str):
        self.log_file = log_file

    def process_trace(self, trace):
        with open(self.log_file, "a") as f:
            f.write(json.dumps({
                "trace_id": trace.trace_id,
                "agent": trace.agent_name,
                "duration_ms": trace.duration_ms,
                "tokens_used": trace.tokens,
                "tools_called": trace.tool_calls
            }) + "\n")

set_trace_processors([LocalTraceProcessor("traces.jsonl")])

# Traces are automatically collected for all Runner.run() calls
```

**Files to modify:**
- `server.py` - Configure tracing on startup
- `.env.example` - Add tracing configuration options

---

### Priority 4: Lower Impact, Variable Effort

#### 10. Hosted MCP Tools (vs Stdio)

**Current:** Uses MCPServerStdio which spawns subprocess per connection.

**Enhancement:** Consider MCPServerStreamableHTTP for production.

**Trade-offs:**
- Stdio: Simpler setup, works well with connection pooling
- HTTP: Better for distributed systems, easier health checks

**Recommendation:** Keep Stdio with connection pooling for now. Consider HTTP if deploying as microservice.

---

#### 11. Agent Cloning for Variants

**Current:** Create new agent instances manually.

**Enhancement:** Use agent.clone() for creating variants.

**Use Case:** Create effect-specific variants of doc_expert.

```python
base_doc_expert = Agent(
    name="Documentation Expert",
    instructions=DOC_EXPERT_INSTRUCTIONS,
    model="gpt-5.2",
    tools=[validate_parameter_range, get_parameter_defaults],
    mcp_servers=[mcp_server]
)

# Clone with effect-specific instructions
sun_doc_expert = base_doc_expert.clone(
    name="Sun Documentation Expert",
    instructions=DOC_EXPERT_INSTRUCTIONS + "\n\nFocus on solar/stellar simulation parameters: limb darkening, granulation, prominences, coronal effects, blackbody emission."
)

explosion_doc_expert = base_doc_expert.clone(
    name="Explosion Documentation Expert",
    instructions=DOC_EXPERT_INSTRUCTIONS + "\n\nFocus on pyro simulation parameters: turbulence, vorticity, smoke density, flame temperature, burning rate."
)
```

---

#### 12. Tool Use Behavior Customization

**Current:** Default tool use behavior.

**Enhancement:** Customize when/how agents use tools.

```python
from agents import Agent, ModelSettings

agent = Agent(
    name="Documentation Expert",
    instructions=DOC_EXPERT_INSTRUCTIONS,
    model="gpt-5.2",
    model_settings=ModelSettings(
        tool_choice="required"  # Force tool use (no freeform responses)
        # Or: "auto" (default), "none" (disable tools), {"type": "function", "function": {"name": "specific_tool"}}
    ),
    tools=[...]
)
```

---

## Implementation Roadmap

### Phase 1: Quick Wins (1-2 days)
1. ✅ Handoff input filtering (Priority 4)
2. ✅ Handoff callbacks (Priority 5)
3. ✅ Tool name override (Priority 6)

### Phase 2: Structured Outputs (2-3 days)
4. ✅ Define Pydantic models (Priority 1)
5. ✅ Update agents to use output_type
6. ✅ Update server.py to handle typed outputs

### Phase 3: Guardrails & Observability (3-4 days)
7. ✅ Implement input/output guardrails (Priority 2)
8. ✅ Implement AgentHooks (Priority 3)
9. ✅ Add tracing integration (Priority 9)

### Phase 4: Advanced Patterns (4-5 days)
10. ✅ Agents as tools pattern (Priority 7)
11. ✅ LiteLLM multi-model support (Priority 8)

---

## Metrics to Track

After implementing enhancements, track:

| Metric | Current | Target |
|--------|---------|--------|
| Average response time | ~15s | <10s |
| Budget utilization | Unknown | <80% |
| Guardrail tripwires/day | N/A | <5 |
| Successful handoffs | Unknown | >95% |
| Token usage per query | Unknown | -20% |
| Output validation errors | Unknown | <1% |

---

## Conclusion

The blender-librarian agent system has a solid foundation. The highest-impact enhancements are:

1. **Structured Output Types** - Eliminates JSON parsing errors, enables type checking
2. **Guardrails** - Prevents wasted API calls, catches errors early
3. **AgentHooks** - Essential for debugging and optimization

These three enhancements alone would significantly improve reliability, debuggability, and cost efficiency.

---

**Document Version:** 1.0
**Created by:** Claude Code analysis
**Review Status:** Ready for implementation
