# Architecture Proposal: Autonomous Blender 5 VFX Asset Generator

**Author:** Claude Opus 4.6 (synthesized from 3-agent research team)
**Date:** 2026-02-18
**Status:** Proposal for review
**Scope:** If I were building this system from scratch, here's exactly how I'd do it.

---

## Table of Contents

1. [Philosophy](#1-philosophy)
2. [The Core Problem and My Solution](#2-the-core-problem-and-my-solution)
3. [Agent Architecture](#3-agent-architecture)
4. [The Anti-Hallucination System](#4-the-anti-hallucination-system)
5. [Pipeline Design](#5-pipeline-design)
6. [SDK Patterns I'd Use](#6-sdk-patterns-id-use)
7. [Self-Learning System](#7-self-learning-system)
8. [Quality Evaluation](#8-quality-evaluation)
9. [State Management](#9-state-management)
10. [Human-in-the-Loop](#10-human-in-the-loop)
11. [Budget Management](#11-budget-management)
12. [What I'd Build First](#12-what-id-build-first)
13. [What I'd Explicitly NOT Build](#13-what-id-explicitly-not-build)
14. [Key Risks and Mitigations](#14-key-risks-and-mitigations)
15. [Research Sources](#15-research-sources)

---

## 1. Philosophy

### The One Sentence Version

**Runtime truth over memorized knowledge, always.**

### The Expanded Version

The fundamental tension in this system is: LLMs are powerful code generators but they hallucinate API details from their training data. Blender's API changes across versions, and the LLM's training data is a mix of Blender 2.x through 4.x. The system needs the LLM's creativity and reasoning while rejecting its unreliable memory.

My approach: **Use the LLM for what it's good at (reasoning, planning, creative problem-solving) and use runtime introspection for what it's bad at (remembering exact API details).**

This means:

1. **The LLM never writes final Blender code from memory.** It writes *intent-based code* — code that expresses what it wants to achieve — and a validation layer resolves the actual API calls against a live Blender process.

2. **Every API attribute is verified before execution.** Not by the LLM reading docs and hoping. By actually querying `dir(bpy.types.FluidDomainSettings)` in a running Blender instance.

3. **Deterministic systems handle deterministic problems.** Camera placement, lighting bounds, API validation — these are not creative decisions. They have correct answers. Don't waste LLM inference on them.

4. **The LLM handles creative, adaptive decisions.** Technique selection, scene composition, debugging novel errors, exploring new approaches — these are where LLM reasoning is irreplaceable.

### Design Principles (Ranked)

| Priority | Principle | Implication |
|----------|-----------|-------------|
| P0 | **Runtime truth** | Never trust the LLM's memory of an API. Always verify. |
| P1 | **Fail loud, fix fast** | Errors surface immediately with full context. No silent swallowing. |
| P2 | **Deterministic > Prompted > Guardrailed** | If you can solve it with code, don't prompt it. If you must prompt it, guardrail it. |
| P3 | **Earn autonomy** | The system starts conservative and unlocks capability through demonstrated reliability. |
| P4 | **Every run teaches** | Failures produce learning signal. Successes produce validated patterns. Nothing is wasted. |
| P5 | **Budget is a first-class constraint** | Cost awareness isn't bolted on — it's baked into every decision point. |

---

## 2. The Core Problem and My Solution

### The Problem

You want an LLM-powered system that generates Blender 5 Python scripts for arbitrary VFX effects. The LLM needs to:

- Understand Blender's physics systems (Mantaflow, rigid body, particles, cloth, geometry nodes)
- Generate working Python scripts using bpy
- Iterate on quality through multiple render-evaluate-improve cycles
- Handle novel prompts it hasn't seen before
- Never use deprecated or hallucinated API attributes

The fatal flaw in the naive approach: **the LLM will confidently generate `obj.fluid.resolution_divisions = 128` when the correct Blender 5 attribute is `resolution_max`**. And it will do this consistently because it appears hundreds of times in its training data.

### My Solution: The "Introspection Sandbox" Pattern

Instead of trying to prevent the LLM from hallucinating (which is fighting nature), I'd build a system where hallucinations are **caught and corrected automatically** before they reach Blender:

```
LLM generates script (may contain hallucinated attributes)
        │
        ▼
┌─────────────────────────────┐
│   INTROSPECTION SANDBOX     │
│                             │
│  1. Parse script AST        │
│  2. Extract all bpy attr    │
│     accesses                │
│  3. For each attribute:     │
│     - Query live Blender    │
│       process for valid     │
│       attributes            │
│     - If invalid, find      │
│       closest valid match   │
│     - Auto-correct OR       │
│       return error with     │
│       valid alternatives    │
│  4. Return validated script │
│     OR correction report    │
└─────────────────────────────┘
        │
        ▼
  Execute validated script in Blender
```

The key insight: **You don't need the LLM to know the right attribute name. You need the LLM to know the right concept, and a deterministic system to resolve the concept to the correct API call.**

This is fundamentally different from:
- **Prompting the LLM to use correct attributes** (unreliable — training data wins)
- **RAG over documentation** (better, but the LLM can still ignore the context)
- **Post-execution error fixing** (too late — the script already failed)

The Introspection Sandbox catches problems **before execution**, using **runtime truth** from a live Blender process, and either auto-corrects or provides the LLM with verified alternatives.

---

## 3. Agent Architecture

### Topology: Python-Orchestrated with Specialist Agents

I would NOT use a pure agent-to-agent handoff architecture. Here's why:

- **Handoffs lose context.** When agent A hands off to agent B, the full conversation history transfers, but the *intent* and *state* are implicit. Python code can track state explicitly.
- **Agents-as-tools is more controllable.** The orchestrator calls agents as tools, gets structured results back, and makes the next decision in Python code.
- **Deterministic routing beats LLM routing.** The pipeline has clear phases. You don't need an LLM to decide "should I validate now?" — you always validate after generation.

### The Agent Map

```
┌──────────────────────────────────────────────────────────┐
│                    PYTHON ORCHESTRATOR                     │
│  (Not an agent — pure Python state machine)               │
│                                                           │
│  Responsibilities:                                        │
│  - Pipeline phase management                              │
│  - State persistence                                      │
│  - Budget tracking                                        │
│  - Deterministic decisions (which phase next)             │
│  - Error routing                                          │
│  - Iteration control (when to stop, when to escalate)    │
└───────┬──────────┬──────────┬──────────┬─────────────────┘
        │          │          │          │
   ┌────▼───┐ ┌───▼────┐ ┌──▼───┐ ┌───▼─────┐
   │PLANNER │ │ CODER  │ │CRITIC│ │EXPLORER │
   │(Agent) │ │(Agent) │ │(Agent│ │ (Agent) │
   └────────┘ └────────┘ └──────┘ └─────────┘
```

#### Agent 1: Planner (gpt-5.2 or o3)

**Job:** Given a user prompt, decompose it into a concrete Blender scene plan.

**Input:** User description + effect type + context from previous iterations (if any)

**Output:** Structured scene plan:
```json
{
  "physics_systems": ["mantaflow_gas", "particle_system"],
  "scene_elements": [
    {"name": "candle", "type": "mesh", "role": "emitter_source"},
    {"name": "flame", "type": "mantaflow_gas", "subtype": "fire", "role": "primary_effect"},
    {"name": "table", "type": "mesh", "role": "environment"}
  ],
  "camera": {"framing": "medium_shot", "subject": "candle", "angle": "slightly_above"},
  "lighting": {"style": "warm_interior", "key_light": "area"},
  "duration_frames": 120,
  "technique_notes": "Use fire+smoke combo with buoyancy. Candle wick as flow source."
}
```

**Why a separate Planner:** The planning step is where creative reasoning happens — choosing which physics systems to combine, what the scene should look like, how to frame the shot. This is the LLM's strength. Separating it from code generation means the coder receives a clear spec rather than trying to plan and code simultaneously.

**Model choice:** gpt-5.2 or o3 for reasoning. This is the highest-stakes decision point — a bad plan cascades into bad code. Worth the extra cost.

#### Agent 2: Coder (gpt-5.2)

**Job:** Given a scene plan, generate a Blender Python script.

**Input:** Scene plan from Planner + API Reference Context (verified attributes)

**Output:** Complete Blender Python script

**Critical constraint:** The Coder's instructions include:
> "You are generating code for Blender 5.0. You DO NOT know the correct attribute names from memory. Before using any bpy attribute, check the API Reference Context provided. If an attribute you want to use is not in the reference, use a placeholder comment `# VERIFY: attribute_name on Type` and the validation system will resolve it."

This instruction is crucial. It tells the model to **distrust its own memory** and lean on the provided context. Combined with the Introspection Sandbox (which catches anything that slips through), this gives us two layers of defense.

**Model choice:** gpt-5.2. Code generation quality matters. gpt-5-mini would be too prone to shortcuts.

#### Agent 3: Critic (gpt-5-mini or gpt-5.2)

**Job:** Given render output + quality metrics, diagnose what's wrong and prescribe specific fixes.

**Input:** Rendered image(s) + quality scores + scene plan + script

**Output:** Structured diagnosis:
```json
{
  "overall_assessment": "Flame is present but too dim, smoke dissipates too quickly",
  "issues": [
    {
      "severity": "major",
      "category": "physics_parameter",
      "description": "Smoke dissolve rate too high — smoke vanishes within 10 frames",
      "suggested_fix": "Reduce dissolve_speed (currently ~0.05, try 0.01-0.02)",
      "target": "domain.dissolve_speed"
    },
    {
      "severity": "minor",
      "category": "lighting",
      "description": "Key light overpowering flame emission",
      "suggested_fix": "Reduce key light energy by 50%"
    }
  ],
  "strategy": "modify_parameters",
  "confidence": 0.7
}
```

**Why a separate Critic:** Evaluation and diagnosis is a fundamentally different cognitive task than code generation. Mixing them in one agent leads to the agent rationalizing its own work rather than objectively assessing it. The Critic has no attachment to the code — it only sees the result.

**Model choice:** gpt-5-mini for cost efficiency (this runs every iteration), or gpt-5.2 if using vision to analyze renders directly.

#### Agent 4: Explorer (gpt-5.2 or o3)

**Job:** Research and experiment with Blender capabilities when the system encounters something novel.

**Input:** A specific question or challenge ("How do I make cloth interact with rigid body physics in Blender 5?")

**Output:** Verified findings with working code snippets

**Tools available to Explorer:**
- `introspect_blender_type(type_name)` — returns all valid attributes of a bpy type
- `run_diagnostic_script(script)` — runs a small Python snippet in Blender and returns output
- `search_blender_docs(query)` — semantic search over Blender 5 documentation
- `search_web(query)` — web search for Blender tutorials and references

**This is the agent that embodies Design Principle P2 from the mission statement** — "Exploration Is a Feature, Not a Bug." When the system encounters a novel prompt or an unfamiliar physics system, the Explorer investigates programmatically, just as a human developer would. It runs `dir()` calls, tests small snippets, reads docs, and builds verified knowledge.

**Model choice:** o3 for complex reasoning about unfamiliar APIs, gpt-5.2 for straightforward research.

### What's NOT an Agent

These are deterministic Python functions, not agents:

| Function | Why Not an Agent |
|----------|-----------------|
| **API Validator** | Deterministic AST parsing + introspection lookup. No reasoning needed. |
| **Camera Placer** | Geometric calculation based on scene bounds. Math, not creativity. |
| **Light Bounds Enforcer** | Per-effect-type clamping. Lookup table, not reasoning. |
| **Script Formatter** | Code formatting, import management. Pure transformation. |
| **Budget Tracker** | Arithmetic. |
| **State Manager** | Read/write to disk. |
| **Render Executor** | Shell command execution. |

**This is a key architectural choice.** Every time you make something an agent when it could be a function, you're paying for LLM inference, adding latency, and introducing non-determinism. Agents should only be used for tasks that require *reasoning*.

---

## 4. The Anti-Hallucination System

This is the core innovation. It has three layers:

### Layer 1: API Reference Injection (Prevention)

Before the Coder agent generates code, we inject verified API context into its prompt:

```python
async def get_api_context_for_plan(scene_plan: dict) -> str:
    """Query live Blender for all relevant API attributes based on the scene plan."""
    context_parts = []

    # Determine which bpy types are relevant
    types_needed = extract_types_from_plan(scene_plan)
    # e.g., ["FluidDomainSettings", "FluidFlowSettings", "PointLight", ...]

    for type_name in types_needed:
        # Actually run this in a live Blender process
        attrs = await introspect_blender_type(type_name)
        context_parts.append(f"## bpy.types.{type_name}\nValid attributes: {attrs}")

    return "\n\n".join(context_parts)
```

The Coder's dynamic instructions include this context:

```python
def coder_instructions(ctx: RunContextWrapper[OrchestratorContext]) -> str:
    return f"""You are a Blender 5.0 Python script generator.

CRITICAL: Do NOT use attribute names from memory. Use ONLY the attributes listed
in the API Reference below. If you need an attribute not listed, add a
# VERIFY: comment and the validation system will resolve it.

## API Reference (verified against Blender 5.0 runtime)
{ctx.context.api_reference}

## Scene Plan
{ctx.context.current_plan}
"""
```

**Why this works:** The LLM is good at following instructions when given explicit, authoritative context. By injecting the actual valid attributes directly into the prompt, the LLM uses them instead of its training data in most cases.

**Why this alone isn't enough:** The LLM can still hallucinate attributes not covered by the reference, or ignore the reference for attributes that are deeply embedded in its training data.

### Layer 2: Introspection Sandbox (Detection + Auto-Correction)

After the Coder generates a script, before execution:

```python
import ast
import re

async def validate_and_fix_script(script: str, blender_process) -> ValidationResult:
    """Parse the script, find all bpy attribute accesses, validate each one."""

    tree = ast.parse(script)
    issues = []
    fixes = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            # Extract the full attribute chain, e.g., "domain.resolution_max"
            chain = extract_attribute_chain(node)
            bpy_type = infer_bpy_type(chain, tree)

            if bpy_type:
                attr_name = node.attr
                # Ask live Blender: does this attribute exist?
                valid = await blender_process.check_attribute(bpy_type, attr_name)

                if not valid:
                    # Get all valid attributes and find closest match
                    valid_attrs = await blender_process.get_attributes(bpy_type)
                    closest = find_closest_match(attr_name, valid_attrs)

                    if closest and similarity(attr_name, closest) > 0.7:
                        # Auto-fix: close enough to be confident
                        fixes[attr_name] = closest
                        issues.append(AutoFixed(attr_name, closest, bpy_type))
                    else:
                        # Can't auto-fix: report to the Coder with valid alternatives
                        issues.append(NeedsReview(attr_name, bpy_type, valid_attrs))

    if fixes:
        fixed_script = apply_fixes(script, fixes)
    else:
        fixed_script = script

    return ValidationResult(
        script=fixed_script,
        issues=issues,
        auto_fixed=len([i for i in issues if isinstance(i, AutoFixed)]),
        needs_review=len([i for i in issues if isinstance(i, NeedsReview)]),
    )
```

**The Introspection Sandbox runs real queries against a live Blender process.** It's not guessing. It's not using training data. It's running `hasattr(bpy.types.FluidDomainSettings, 'resolution_max')` and getting a ground truth answer.

**Auto-correction strategy:**

| Scenario | Action |
|----------|--------|
| Attribute doesn't exist but close match found (>0.7 similarity) | Auto-fix silently |
| Attribute doesn't exist, no close match | Return error + full list of valid attributes to Coder |
| Type doesn't exist | Return error, trigger Explorer agent |
| Script has syntax errors | Return error to Coder immediately |

### Layer 3: Execution Feedback Loop (Learning)

When a validated script still fails at execution time (runtime errors, not caught by static analysis):

```python
async def execute_with_learning(script: str, blender_process) -> ExecutionResult:
    result = await blender_process.execute(script)

    if result.error:
        # Parse the error to extract the specific failure
        error_info = parse_blender_error(result.stderr)

        # If it's an attribute error, update our knowledge
        if error_info.type == "AttributeError":
            # Introspect the correct attribute
            valid_attrs = await blender_process.get_attributes(error_info.object_type)

            # Store this mapping for future reference
            await knowledge_store.record_correction(
                wrong=error_info.attribute_name,
                correct=find_closest_match(error_info.attribute_name, valid_attrs),
                type=error_info.object_type,
                blender_version="5.0",
            )

    return result
```

### Why Three Layers?

| Layer | Catches | Cost | Latency |
|-------|---------|------|---------|
| API Reference Injection | ~80% of hallucinations | Low (one introspection query per type) | ~2s |
| Introspection Sandbox | ~95% of remaining | Low (AST parse + attribute checks) | ~3s |
| Execution Feedback | The last ~5% (runtime-only errors) | Higher (full script execution) | ~30s+ |

The layers are ordered by cost and specificity. Most problems are caught cheaply at Layer 1. The few that slip through are caught at Layer 2. Only genuinely novel runtime issues reach Layer 3.

### The Persistent Blender Process

A critical implementation detail: **keep a long-running Blender process for introspection queries.**

```python
class BlenderIntrospector:
    """Maintains a persistent Blender process for API queries."""

    def __init__(self):
        self.process = None
        self._cache = {}  # Cache introspection results

    async def start(self):
        self.process = await asyncio.create_subprocess_exec(
            "blender", "--background", "--python-console",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    async def get_attributes(self, type_name: str) -> list[str]:
        """Get all valid attributes of a bpy type."""
        if type_name in self._cache:
            return self._cache[type_name]

        script = f"import bpy; print([a for a in dir(bpy.types.{type_name}) if not a.startswith('_')])"
        result = await self._execute_snippet(script)
        attrs = parse_attribute_list(result)
        self._cache[type_name] = attrs
        return attrs

    async def check_attribute(self, type_name: str, attr_name: str) -> bool:
        attrs = await self.get_attributes(type_name)
        return attr_name in attrs
```

This avoids launching a new Blender process for every introspection query. The cache means we query each type at most once per session. Total overhead: ~2-5 seconds for the initial queries, near-zero for subsequent ones.

---

## 5. Pipeline Design

### The State Machine

```
                    ┌──────────┐
                    │   PLAN   │ ◄── User prompt + context
                    └────┬─────┘
                         │
                         ▼
                  ┌──────────────┐
         ┌───────│   GENERATE   │ ◄── Plan + API Reference
         │       └──────┬───────┘
         │              │
         │              ▼
         │       ┌──────────────┐
         │       │   VALIDATE   │ ◄── Introspection Sandbox
         │       └──────┬───────┘
         │              │
         │         pass │ fail → back to GENERATE with corrections
         │              │
         │              ▼
         │       ┌──────────────┐
         │       │   EXECUTE    │ ◄── Run in Blender
         │       └──────┬───────┘
         │              │
         │         pass │ fail → DIAGNOSE → fix → EXECUTE (max 3)
         │              │
         │              ▼
         │       ┌──────────────┐
         │       │   EVALUATE   │ ◄── Render + Quality metrics
         │       └──────┬───────┘
         │              │
         │         pass │ fail
         │              │     │
         │              │     ▼
         │              │  ┌──────────────┐
         │              │  │   CRITIQUE   │ ◄── What went wrong?
         │              │  └──────┬───────┘
         │              │         │
         │              │         ▼
         │              │  ┌──────────────┐
         │              │  │   IMPROVE    │ ◄── Apply fixes or escalate
         │              │  └──────┬───────┘
         │              │         │
         │              │    ┌────┴────┐
         │              │    │         │
         │              │ modify   new approach
         │              │    │         │
         │              │    ▼         ▼
         │              │ GENERATE  PLAN (re-plan with new technique)
         │              │
         │              ▼
         │       ┌──────────────┐
         │       │   COMPLETE   │ ──► Return result + learning signal
         │       └──────────────┘
         │
         └── budget exceeded / max iterations → COMPLETE (with failure info)
```

### Phase Details

#### PLAN Phase
- **Actor:** Planner Agent (gpt-5.2/o3)
- **Input:** User prompt, effect type, optional previous attempt context
- **Output:** Structured scene plan
- **Deterministic checks:** Validate plan has all required fields, effect type is recognized

#### GENERATE Phase
- **Actor:** Coder Agent (gpt-5.2)
- **Input:** Scene plan + verified API reference context
- **Output:** Blender Python script
- **Key:** API reference injected via dynamic instructions

#### VALIDATE Phase
- **Actor:** Introspection Sandbox (deterministic)
- **Input:** Generated script
- **Output:** Validated script (auto-corrected) OR correction report
- **On failure:** Return to GENERATE with specific corrections needed

#### EXECUTE Phase
- **Actor:** Blender subprocess (deterministic)
- **Input:** Validated script
- **Output:** Execution result (success/failure + logs)
- **On failure:** Parse error, attempt auto-fix (max 3 retries), then escalate to Coder

#### EVALUATE Phase
- **Actor:** Quality evaluation pipeline (deterministic metrics + optional vision)
- **Input:** Rendered images
- **Output:** Quality scores + issue list
- **On pass:** Go to COMPLETE
- **On fail:** Go to CRITIQUE

#### CRITIQUE Phase
- **Actor:** Critic Agent (gpt-5-mini/gpt-5.2)
- **Input:** Render + scores + plan + script
- **Output:** Diagnosis + suggested fixes
- **Decision:** Modify existing script (minor issues) or re-plan (fundamental issues)

#### IMPROVE Phase
- **Actor:** Python orchestrator (deterministic routing)
- **Routes to:** GENERATE (modify) or PLAN (re-approach)
- **Escalation:** If same issue persists 2+ iterations, force technique switch

### Iteration Control

```python
MAX_ITERATIONS = 5
MAX_SAME_ISSUE = 2
MAX_EXECUTION_RETRIES = 3

async def should_continue(state: SessionState) -> IterationDecision:
    if state.iteration >= MAX_ITERATIONS:
        return IterationDecision.STOP_MAX_ITERATIONS

    if state.budget_remaining < state.estimated_iteration_cost:
        return IterationDecision.STOP_BUDGET

    if state.quality_score >= state.quality_threshold:
        return IterationDecision.STOP_SUCCESS

    if state.consecutive_same_issue >= MAX_SAME_ISSUE:
        return IterationDecision.ESCALATE_NEW_TECHNIQUE

    if state.quality_improving:
        return IterationDecision.CONTINUE_MODIFY
    else:
        return IterationDecision.ESCALATE_NEW_TECHNIQUE
```

---

## 6. SDK Patterns I'd Use

### Pattern 1: Python Orchestrator + Agents-as-Tools

**This is the primary pattern.** The orchestrator is Python code, not an agent. It calls agents as tools when it needs reasoning.

```python
from agents import Agent, Runner, function_tool, RunContextWrapper

# Context passed through entire run
class PipelineContext:
    session_state: SessionState
    api_reference: str
    current_plan: dict | None
    budget: BudgetTracker

# The Planner agent
planner = Agent(
    name="Planner",
    model="gpt-5.2",
    instructions=planner_dynamic_instructions,  # callable
    output_type=ScenePlan,  # structured output
)

# The Coder agent
coder = Agent(
    name="Coder",
    model="gpt-5.2",
    instructions=coder_dynamic_instructions,  # includes API reference
    output_type=BlenderScript,
)

# The Critic agent
critic = Agent(
    name="Critic",
    model="gpt-5-mini",
    instructions=critic_dynamic_instructions,
    output_type=Diagnosis,
)

# The Explorer agent — this one has tools
explorer = Agent(
    name="Explorer",
    model="o3",
    instructions=explorer_instructions,
    tools=[
        introspect_blender_type,
        run_diagnostic_script,
        search_blender_docs,
    ],
)
```

The orchestrator calls them:

```python
async def run_plan_phase(ctx: PipelineContext, prompt: str) -> ScenePlan:
    result = await Runner.run(
        planner,
        input=f"Create a scene plan for: {prompt}",
        context=ctx,
    )
    return result.final_output  # Typed as ScenePlan
```

### Pattern 2: Dynamic Instructions for Runtime Context

Every agent gets instructions that are *functions*, not static strings:

```python
def coder_dynamic_instructions(ctx: RunContextWrapper[PipelineContext]) -> str:
    base = "You generate Blender 5.0 Python scripts."

    # Inject verified API reference
    api_ref = f"\n\n## Verified API Reference\n{ctx.context.api_reference}"

    # Inject scene plan
    plan = f"\n\n## Scene Plan\n{json.dumps(ctx.context.current_plan, indent=2)}"

    # Inject learnings from past runs
    learnings = ""
    if ctx.context.session_state.learnings:
        learnings = "\n\n## Lessons from Previous Iterations\n"
        for l in ctx.context.session_state.learnings:
            learnings += f"- {l}\n"

    return base + api_ref + plan + learnings
```

This is the mechanism for self-improvement: validated learnings from past iterations flow directly into agent instructions.

### Pattern 3: Output Types for Structured Responses

Every agent returns structured output, not free text:

```python
from pydantic import BaseModel

class ScenePlan(BaseModel):
    physics_systems: list[str]
    scene_elements: list[SceneElement]
    camera: CameraSpec
    lighting: LightingSpec
    duration_frames: int
    technique_notes: str

class Diagnosis(BaseModel):
    overall_assessment: str
    issues: list[Issue]
    strategy: Literal["modify_parameters", "modify_structure", "new_technique", "human_help"]
    confidence: float
```

Structured output eliminates parsing ambiguity and makes the orchestrator's routing logic clean.

### Pattern 4: Guardrails for Safety

```python
from agents import InputGuardrail, GuardrailFunctionOutput, Agent

async def budget_guardrail(ctx, agent, input_data) -> GuardrailFunctionOutput:
    """Prevent agent execution if budget is exhausted."""
    if ctx.context.budget.remaining < ctx.context.budget.estimated_cost(agent.name):
        return GuardrailFunctionOutput(
            output_info="Budget exhausted",
            tripwire_triggered=True,
        )
    return GuardrailFunctionOutput(output_info="Budget OK", tripwire_triggered=False)

async def loop_guardrail(ctx, agent, input_data) -> GuardrailFunctionOutput:
    """Detect if we're in a loop (same error 3+ times)."""
    if ctx.context.session_state.is_looping():
        return GuardrailFunctionOutput(
            output_info="Loop detected — escalating",
            tripwire_triggered=True,
        )
    return GuardrailFunctionOutput(output_info="No loop", tripwire_triggered=False)

coder = Agent(
    name="Coder",
    model="gpt-5.2",
    instructions=coder_dynamic_instructions,
    output_type=BlenderScript,
    input_guardrails=[budget_guardrail, loop_guardrail],
)
```

### Pattern 5: RunHooks for Observability

```python
from agents import RunHooks, Agent, Tool

class PipelineHooks(RunHooks):
    async def on_agent_start(self, context, agent):
        logger.info(f"Agent {agent.name} starting")
        context.context.budget.record_start(agent.name)

    async def on_agent_end(self, context, agent, output):
        logger.info(f"Agent {agent.name} finished")
        context.context.budget.record_end(agent.name)

    async def on_tool_start(self, context, agent, tool):
        logger.info(f"Tool {tool.name} called by {agent.name}")

    async def on_tool_end(self, context, agent, tool, result):
        logger.info(f"Tool {tool.name} returned to {agent.name}")

# Applied to all agent runs
result = await Runner.run(agent, prompt, context=ctx, hooks=PipelineHooks())
```

### Pattern 6: Model Routing

Different agents use different models based on the cognitive demand:

| Agent | Model | Reasoning |
|-------|-------|-----------|
| Planner | gpt-5.2 / o3 | Highest-stakes creative decisions |
| Coder | gpt-5.2 | Code quality matters |
| Critic | gpt-5-mini | Runs every iteration, mostly structured analysis |
| Explorer | o3 | Complex reasoning about unfamiliar APIs |
| Guardrails | gpt-5-mini | Simple yes/no checks |

Use o3 for tasks that require multi-step reasoning. Use gpt-5-mini for tasks that are mostly pattern matching. Use gpt-5.2 as the default for generation tasks.

For Codex CLI (gpt-5.3): reserve this for the Explorer agent on genuinely novel research tasks where maximum capability matters.

---

## 7. Self-Learning System

### Philosophy: Validated Knowledge Only

The system only promotes learnings to "trusted knowledge" when they meet evidence thresholds:

```
Observation → Hypothesis → Test → Validated Knowledge
```

Not:

```
Observation → Assumed Knowledge  (WRONG — leads to bad dynamic instructions)
```

### Three Tiers of Knowledge

#### Tier 1: Session Memory (Ephemeral)

Lives only within the current session. Examples:
- "This specific script had a syntax error on line 42"
- "The user wants the camera angle from above"
- "Current best quality score is 45"

**Storage:** In-memory `SessionState` object.

#### Tier 2: Validated Patterns (Persistent, Evidence-Gated)

Promoted from session observations after multiple confirmations. Examples:
- "For fire effects, `dissolve_speed` between 0.01-0.03 produces best results (confirmed 5/7 runs)"
- "Camera distance of 3x scene_bounds_radius works for single-object effects (confirmed 8/10 runs)"

**Storage:** JSON/SQLite knowledge base.

**Promotion criteria:**
```python
def should_promote(observation: Observation) -> bool:
    # Minimum 3 observations
    if observation.count < 3:
        return False
    # Success rate > 70%
    if observation.success_rate < 0.7:
        return False
    # Consistent across different prompts (not overfitted to one)
    if observation.unique_prompts < 2:
        return False
    return True
```

#### Tier 3: API Truth (Ground Truth, Always Trusted)

Results of runtime introspection. Examples:
- "`bpy.types.FluidDomainSettings` has attribute `resolution_max` (not `resolution_divisions`)"
- "`ShaderNodeMix` is the correct node type (not `ShaderNodeMixRGB`)"

**Storage:** Cached introspection results, refreshed on Blender version change.

### How Learning Flows Into Agent Behavior

Validated patterns are injected into agent instructions via dynamic instructions:

```python
def coder_dynamic_instructions(ctx: RunContextWrapper[PipelineContext]) -> str:
    instructions = BASE_CODER_INSTRUCTIONS

    # Inject validated patterns relevant to this effect type
    patterns = knowledge_base.get_patterns(
        effect_type=ctx.context.effect_type,
        min_confidence=0.7,
    )

    if patterns:
        instructions += "\n\n## Validated Patterns (use these)\n"
        for p in patterns:
            instructions += f"- {p.description} (confidence: {p.confidence:.0%})\n"

    # Inject known API corrections
    corrections = knowledge_base.get_corrections()
    if corrections:
        instructions += "\n\n## Known API Corrections\n"
        for c in corrections:
            instructions += f"- Use `{c.correct}` not `{c.wrong}` on {c.type}\n"

    return instructions
```

### What NOT to Learn

Explicit anti-learning rules:
- **Don't learn from single observations.** One success doesn't make a pattern.
- **Don't learn prompt-specific tricks.** "For the candle prompt, use X" is overfitting.
- **Don't learn from interrupted runs.** If the user stopped early, the result isn't informative.
- **Don't learn subjective quality.** Quality scores are the ground truth, not the Critic's opinion.

---

## 8. Quality Evaluation

### Multi-Signal Approach

Quality evaluation uses multiple independent signals, not a single metric:

```python
class QualityEvaluation(BaseModel):
    # Render-level checks (deterministic, fast, cheap)
    is_black_screen: bool          # All pixels below threshold
    is_white_screen: bool          # All pixels above threshold
    has_content: bool              # Non-trivial pixel variance
    has_expected_resolution: bool  # Matches requested resolution

    # Metric-based scores (deterministic, moderate cost)
    brightness_score: float        # 0-100, penalizes extremes
    contrast_score: float          # 0-100, measures dynamic range
    spatial_complexity: float      # 0-100, measures detail level

    # ML-based scores (expensive, high quality)
    topiq_score: float | None      # No-reference image quality
    clip_alignment: float | None   # Text-image alignment with prompt

    # Composite
    overall_score: float           # Weighted combination
    critical_failures: list[str]   # Auto-fail conditions
```

### Evaluation Tiers (Cost-Conscious)

| Tier | When | Cost | What |
|------|------|------|------|
| **Quick Check** | Every execution | Free | Black screen, white screen, resolution, file exists |
| **Metric Check** | Every iteration | Cheap | Brightness, contrast, spatial complexity (numpy only) |
| **ML Check** | Iterations 1, 3, 5 | Expensive | TOPIQ, CLIP alignment (skip on early iterations) |
| **Vision Check** | Final iteration only | Most expensive | GPT vision for subjective quality (only when ML score is borderline) |

This tiered approach means early iterations get fast, cheap feedback, and expensive evaluation is reserved for when it matters most.

### Critical Failure Detection (Deterministic, Free)

These are non-negotiable auto-fails that don't require any ML:

```python
def check_critical_failures(image_path: str) -> list[str]:
    img = load_image(image_path)
    failures = []

    mean_brightness = img.mean()
    if mean_brightness < 5:  # Out of 255
        failures.append("BLACK_SCREEN")
    if mean_brightness > 250:
        failures.append("WHITE_SCREEN")

    pixel_variance = img.std()
    if pixel_variance < 2:
        failures.append("NO_CONTENT")  # Solid color

    # Check if render actually completed (not a partial frame)
    if img.shape[0] < expected_height or img.shape[1] < expected_width:
        failures.append("INCOMPLETE_RENDER")

    return failures
```

---

## 9. State Management

### Session State

One session = one user request, potentially spanning multiple iterations:

```python
class SessionState(BaseModel):
    session_id: str
    created_at: datetime

    # Request
    user_prompt: str
    effect_type: str
    quality_threshold: float
    max_iterations: int

    # Current state
    current_phase: Phase
    current_iteration: int
    current_plan: ScenePlan | None
    current_script_path: str | None

    # History
    iterations: list[IterationRecord]

    # Learning
    observations: list[Observation]
    corrections_applied: list[APICorrection]

    # Budget
    tokens_used: int
    estimated_cost: float

    # Quality tracking
    best_score: float
    best_render_path: str | None
    score_history: list[float]

    # Escalation
    consecutive_same_issue: int
    techniques_tried: list[str]
```

### Persistence

State is persisted to disk after every phase transition:

```python
async def transition_phase(state: SessionState, new_phase: Phase):
    state.current_phase = new_phase
    await save_state(state)  # Atomic write to JSON file
```

This means if the process crashes, you can resume from the last completed phase. No work is lost.

### What State is NOT

- **State is not conversation history.** Agent conversations are ephemeral. Each agent call starts fresh with context injected via dynamic instructions.
- **State is not the knowledge base.** Session state is session-specific. The knowledge base is cross-session.
- **State is not the script.** Scripts are files on disk. State references them by path.

---

## 10. Human-in-the-Loop

### Progressive Autonomy

The system starts with HITL at every decision point and gradually reduces it as confidence grows:

```python
class AutonomyLevel(Enum):
    GUIDED = 0        # Confirm every decision
    ASSISTED = 1      # Confirm technique selection, auto-approve parameters
    SEMI_AUTO = 2     # Auto technique selection, confirm only on novel approaches
    AUTONOMOUS = 3    # Full auto, notify on completion
```

### Intervention Points

| Point | Guided | Assisted | Semi-Auto | Autonomous |
|-------|--------|----------|-----------|------------|
| Plan approval | Ask | Ask | Auto | Auto |
| Technique selection | Ask | Ask | Auto (known), Ask (novel) | Auto |
| Script approval | Ask | Auto | Auto | Auto |
| Quality judgment | Ask | Auto | Auto | Auto |
| Escalation decision | Ask | Ask | Ask | Auto |
| Budget warning | Always | Always | Always | Always |

**Budget warnings are always human-visible**, regardless of autonomy level.

### How It Works with the SDK

The Agents SDK doesn't have built-in HITL. I'd implement it as a gate in the Python orchestrator:

```python
async def maybe_ask_human(decision: str, context: str, autonomy: AutonomyLevel) -> bool:
    """Returns True if approved (either by human or auto-approved by autonomy level)."""
    if should_auto_approve(decision, autonomy):
        logger.info(f"Auto-approved: {decision}")
        return True

    # Block and wait for human input
    print(f"\n--- APPROVAL NEEDED ---")
    print(f"Decision: {decision}")
    print(f"Context: {context}")
    response = input("Approve? (y/n/modify): ")

    if response == 'y':
        return True
    elif response == 'modify':
        # Let human modify the plan/script
        modifications = input("Enter modifications: ")
        return modifications  # Orchestrator handles this
    else:
        return False
```

---

## 11. Budget Management

### Cost Model

| Operation | Estimated Cost | Per |
|-----------|---------------|-----|
| Planner call (gpt-5.2) | $0.05 | call |
| Coder call (gpt-5.2) | $0.08 | call |
| Critic call (gpt-5-mini) | $0.01 | call |
| Explorer call (o3) | $0.15 | call |
| CLIP evaluation | $0.02 | image |
| TOPIQ evaluation | Free | local |
| Vision evaluation (gpt-5.2) | $0.05 | image |
| Blender execution | Free | local |
| Introspection query | Free | local |

### Budget-Aware Decisions

```python
class BudgetTracker:
    def __init__(self, monthly_limit: float = 20.0):
        self.monthly_limit = monthly_limit
        self.spent = 0.0

    def can_afford(self, operation: str) -> bool:
        cost = COST_TABLE[operation]
        return (self.spent + cost) <= self.monthly_limit

    def should_use_cheap_eval(self) -> bool:
        """Use cheap evaluation when budget is getting tight."""
        return self.remaining_fraction < 0.3

    def should_reduce_iterations(self) -> bool:
        """Reduce max iterations when budget is very tight."""
        return self.remaining_fraction < 0.15
```

### Model Routing by Budget

When budget is tight, downgrade models:

```python
def select_model(agent_name: str, budget: BudgetTracker) -> str:
    if budget.remaining_fraction > 0.5:
        return MODEL_TABLE_PREMIUM[agent_name]  # gpt-5.2, o3
    elif budget.remaining_fraction > 0.2:
        return MODEL_TABLE_STANDARD[agent_name]  # gpt-5.2, gpt-5-mini
    else:
        return MODEL_TABLE_ECONOMY[agent_name]   # gpt-5-mini for everything
```

---

## 12. What I'd Build First

### Phase 1: Foundation (Week 1-2)

Build the minimal system that can generate and validate a single script:

1. **BlenderIntrospector** — persistent Blender process for API queries
2. **Introspection Sandbox** — AST parsing + attribute validation
3. **Coder Agent** — basic script generation with API reference injection
4. **Execution pipeline** — run validated scripts in Blender
5. **Session state** — basic state persistence

**Success criterion:** Generate a valid Blender script for "fire on a plane" that passes the Introspection Sandbox and executes without AttributeError.

### Phase 2: Iteration Loop (Week 3-4)

Add the quality loop:

6. **Quality evaluation** — basic metrics (no ML yet, just deterministic checks)
7. **Critic Agent** — diagnose issues from renders
8. **Planner Agent** — structured scene planning
9. **Iteration control** — orchestrator loop with escalation

**Success criterion:** The system iterates 3 times on "candle flame" and each iteration improves the quality score.

### Phase 3: Learning (Week 5-6)

Add self-improvement:

10. **Knowledge base** — store validated patterns
11. **Dynamic instruction injection** — feed learnings into agents
12. **Observation tracking** — record what works per effect type
13. **Pattern promotion** — evidence-gated knowledge validation

**Success criterion:** After 10 runs of fire effects, the system's first attempt quality improves measurably.

### Phase 4: Exploration (Week 7-8)

Add the Explorer and novel prompt handling:

14. **Explorer Agent** — research unfamiliar physics systems
15. **Technique switching** — escalation to new approaches
16. **Novel prompt handling** — decompose unknown prompts into known systems

**Success criterion:** The system produces a reasonable first attempt for "a brick wall collapsing" (rigid body physics, not previously encountered).

### Phase 5: Polish (Week 9+)

17. **Deterministic camera placement** — geometric calculation, not LLM
18. **Deterministic lighting bounds** — per-effect-type clamping
19. **ML evaluation** — TOPIQ, CLIP alignment
20. **Vision evaluation** — GPT-5.2 vision for subjective quality
21. **Progressive autonomy** — earned trust system

---

## 13. What I'd Explicitly NOT Build

| Feature | Why Not |
|---------|---------|
| **Template/parameter schema system** | The mission statement explicitly says this isn't the core capability. Start with pure code generation. Templates can be added later as an acceleration layer. |
| **Handoff-based agent architecture** | Handoffs lose orchestrator control. Agents-as-tools + Python orchestrator is more reliable and debuggable. |
| **Vector store for Blender docs** | Runtime introspection is more reliable and doesn't depend on doc coverage. Use it as a secondary source only. |
| **Complex escape velocity system** | Simple rule: if same issue 2x, switch technique. If no progress 3x, ask human. Don't over-engineer escalation. |
| **MCP server wrapper** | Build the core system first. MCP is an integration concern, not an architecture concern. |
| **Custom tracing system** | Use the SDK's built-in tracing. Don't reinvent it. |
| **Agent-based API validation** | This is a deterministic problem. AST parsing + introspection lookup. Don't waste LLM inference on it. |

---

## 14. Key Risks and Mitigations

### Risk 1: Blender Introspection Is Insufficient

**Risk:** `dir()` and `hasattr()` might not catch all API issues (e.g., attributes that exist but behave differently in v5).

**Mitigation:** Layer the introspection with doc search as a secondary source. For behavioral changes (same attribute, different semantics), build a known-changes registry that's manually curated.

### Risk 2: LLM Ignores Injected API Reference

**Risk:** Despite being told to use the API reference, the LLM generates code from training data anyway.

**Mitigation:** The Introspection Sandbox catches this mechanically. Also, use strong instruction framing: "The API reference below is the ONLY source of truth. Any attribute not listed here DOES NOT EXIST."

### Risk 3: Quality Evaluation Doesn't Correlate with Actual Quality

**Risk:** Deterministic metrics (brightness, contrast) don't capture whether the effect actually looks good.

**Mitigation:** Use CLIP alignment as the primary quality signal once budget allows. CLIP measures whether the image matches the text description, which is closer to "does this look like what was requested?" Start with deterministic metrics for fast iteration, add ML metrics as the system matures.

### Risk 4: Self-Learning Produces Bad Knowledge

**Risk:** The system learns something that's actually wrong and it degrades future runs.

**Mitigation:** Evidence-gating prevents premature learning. Knowledge must be confirmed across multiple runs and different prompts before promotion. Also, learnings have a confidence score that decays if new evidence contradicts them.

### Risk 5: Budget Exhaustion Mid-Session

**Risk:** An expensive session burns through the monthly budget, leaving nothing for future runs.

**Mitigation:** Per-session budget limits (fraction of monthly budget). Budget-aware model routing. Aggressive early stopping when progress stalls. Always reserve 10% for emergency use.

### Risk 6: Blender Process Management

**Risk:** Blender crashes, hangs, or produces corrupt output in headless mode.

**Mitigation:** Process timeouts (kill after N seconds). Output validation (check file exists, check file size > 0, check image is decodable). Retry with clean process on crash. Log all Blender stdout/stderr for debugging.

---

## 15. Research Sources

This proposal was synthesized from research across three parallel tracks:

### OpenAI Agents SDK Research
- OpenAI Agents SDK documentation (v0.9.0): agents.md, tools.md, multi_agent.md, guardrails.md, handoffs.md, tracing.md, context.md, streaming.md, config.md
- GitHub: https://github.com/openai/openai-agents-python
- Context7 MCP documentation queries

### Multi-Agent Architecture Research
- Microsoft AutoGen and multi-agent orchestration patterns
- LangGraph agent architectures (supervisor, hierarchical, flat)
- SWE-agent and Devin architecture analysis
- RAG-grounded code generation literature
- Agentic coding best practices (2024-2026)
- Budget-aware LLM system design

### Blender 5 Automation Research
- Blender 5.0 Python API documentation
- Blender RNA introspection system
- Blender headless automation patterns
- AI+Blender integration projects (BlenderGPT, SceneX, NodeWeaver)
- OpenVDB/NanoVDB export workflows
- Blender Python script validation techniques

---

## Appendix A: Quick Reference — Agent Responsibilities

| Agent | Input | Output | Model | When Called |
|-------|-------|--------|-------|------------|
| **Planner** | User prompt + context | ScenePlan | gpt-5.2/o3 | Start + re-plan |
| **Coder** | ScenePlan + API ref | Blender script | gpt-5.2 | Generate + modify |
| **Critic** | Render + scores | Diagnosis | gpt-5-mini | After evaluation |
| **Explorer** | Research question | Findings + code | o3 | Novel situations |

## Appendix B: Quick Reference — Deterministic Components

| Component | Input | Output | Cost |
|-----------|-------|--------|------|
| **Introspection Sandbox** | Script | Validated script | Free |
| **Camera Placer** | Scene bounds | Camera transform | Free |
| **Light Bounds** | Effect type + params | Clamped params | Free |
| **Quality Checks** | Render image | Scores + failures | Free |
| **Budget Tracker** | Operation name | Allow/deny | Free |
| **State Manager** | Phase transition | Persisted state | Free |

## Appendix C: The Key Insight Summarized

```
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  LLM = Creative reasoning about WHAT to build           │
│  Runtime Introspection = Ground truth about HOW to      │
│                          express it in Blender's API    │
│  Deterministic Systems = Known solutions to known       │
│                          problems (camera, lighting,    │
│                          bounds)                        │
│                                                         │
│  Don't use LLMs for deterministic tasks.                │
│  Don't use deterministic systems for creative tasks.    │
│  Don't trust LLM memory for API details.                │
│                                                         │
└─────────────────────────────────────────────────────────┘
```
