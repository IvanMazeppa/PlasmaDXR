# Architecture Proposal: Blender VFX Orchestrator

**Author:** Claude Opus 4.6
**Date:** 2026-02-18 (Revised — based on deep SDK research + multi-agent literature review)
**Context:** Fresh-start architecture for an autonomous, multi-agent VFX asset generator built on the OpenAI Agents SDK v0.9.0, targeting Blender 5.0 headless. Written without inspecting the existing codebase — purely from web research, SDK documentation, and the mission statement.

---

## Executive Summary

After researching the OpenAI Agents SDK documentation thoroughly (all 10 core areas: Agent, Runner, tools, handoffs, tracing, guardrails, dynamic instructions, HITL, streaming, hooks, context management), reviewing the multi-agent code generation literature (AgentCoder, MapCoder, hybrid cascade), self-improving agent architectures (GEPA, Ralph, Mem0), and real-world Agents SDK deployments — here is how I would build this system from scratch.

**The core philosophy:** An LLM is a creative problem-solver, not a reliable engineer. Every architectural decision should maximize the LLM's creative contribution while minimizing the surface area where it can fail silently. Deterministic systems handle everything that *can* be deterministic. The LLM handles everything that *must* be creative.

---

## Table of Contents

1. [Philosophy and Principles](#1-philosophy-and-principles)
2. [Agent Architecture](#2-agent-architecture)
3. [The Pipeline](#3-the-pipeline)
4. [State Management](#4-state-management)
5. [Deterministic Safety Layer](#5-deterministic-safety-layer)
6. [Self-Learning System](#6-self-learning-system)
7. [Model Selection Strategy](#7-model-selection-strategy)
8. [Error Recovery](#8-error-recovery)
9. [Human-in-the-Loop](#9-human-in-the-loop)
10. [Budget Management](#10-budget-management)
11. [What I Would NOT Build](#11-what-i-would-not-build)
12. [Implementation Order](#12-implementation-order)
13. [SDK Features Reference](#appendix-a-sdk-features-used)

---

## 1. Philosophy and Principles

### 1.1 The Two-Layer Architecture

Every decision in this system falls into one of two categories:

| Layer | Examples | Who decides |
|-------|----------|-------------|
| **Creative** | What Blender technique to use, how to compose a scene, what parameters to try, how to interpret quality feedback | LLM agents |
| **Mechanical** | Camera distance, light energy bounds, API attribute names, file I/O, bake commands, render settings | Deterministic Python code |

**Principle:** If a decision can be made correctly by a Python function 100% of the time, it MUST NOT be delegated to an LLM. The LLM's job is to decide *what* to render. Python's job is to make sure it renders correctly.

### 1.2 Fewer Agents, Clearer Boundaries

The multi-agent literature consistently shows that **more agents = more failure surface**. A 2025 paper on hybrid cascade approaches (arXiv:2505.18286) demonstrated that routing simple tasks to a single agent and only escalating failures to multi-agent achieves 12% better accuracy than multi-agent alone, at lower cost.

My proposal uses **3 agents total** for the core pipeline, plus 1 optional research agent. Each agent has one job, one model, and one output type. There is no ambiguity about who does what.

### 1.3 Context Resets Are a Feature

The Ralph pattern (an autonomous coding agent loop, github.com/snarktank/ralph) demonstrates that **fresh context on each iteration outperforms continuous context**. Performance degrades at large context windows. The agent doesn't need to remember the full history of what it tried — it needs a concise summary of what worked, what didn't, and what to try next.

Each iteration of the pipeline starts with a fresh `Runner.run()` call. The iteration history is compressed into a structured briefing, not carried as raw conversation history.

### 1.4 Validated by Literature

The AgentCoder framework (arXiv:2312.13010) validates the three-agent pattern: Programmer/TestDesigner/TestExecutor. My ScriptWriter/Evaluator/Strategist maps directly to this proven architecture. AgentCoder achieved 79.9% pass@1 with GPT-3.5 and 91.5% with GPT-4 by separating generation from evaluation from feedback.

---

## 2. Agent Architecture

### 2.1 The Three Core Agents

```
                    +-----------------------+
                    |   Pipeline Controller |   (Python, NOT an agent)
                    |   State machine logic |
                    +-----------+-----------+
                                |
              +-----------------+-----------------+
              |                 |                 |
    +---------v------+  +------v--------+  +-----v-----------+
    |  Script Writer |  |   Evaluator   |  |   Strategist    |
    |  (gpt-5.2)     |  |   (gpt-5-mini)|  |   (o3)          |
    |                |  |               |  |                 |
    | Writes Blender |  | Interprets    |  | Decides what to |
    | Python scripts |  | render output |  | change and why  |
    | from specs     |  | quality score |  |                 |
    +----------------+  +---------------+  +-----------------+
```

#### Agent 1: Script Writer

- **Model:** `gpt-5.2` (best code generation)
- **Input:** A structured spec (effect type, technique, scene description, constraints, and optionally a previous script + feedback)
- **Output:** A complete, self-contained Blender Python script
- **Tools:** `search_blender_docs` (function_tool), `search_code_patterns` (function_tool), `search_technique_knowledge` (function_tool)
- **Output type:** Pydantic model with `script: str`, `technique_used: str`, `assumptions: list[str]`
- **SDK features used:** Dynamic instructions, function_tool, output_type

**Key design:** The Writer NEVER sees raw error logs or quality scores. It receives a curated briefing from the Strategist. This prevents the writer from getting confused by noisy diagnostic output.

```python
from agents import Agent, function_tool, RunContextWrapper
from pydantic import BaseModel

class WriterOutput(BaseModel):
    script: str
    technique_used: str
    assumptions: list[str]

@function_tool
async def search_blender_docs(ctx: RunContextWrapper[PipelineContext], query: str) -> str:
    """Search Blender 5.0 documentation for API reference."""
    return await _search_docs_impl(query)

@function_tool
async def search_code_patterns(ctx: RunContextWrapper[PipelineContext], effect_type: str) -> str:
    """Retrieve known working code patterns for this effect type."""
    return ctx.context.knowledge.get_patterns(effect_type)

def writer_instructions(ctx: RunContextWrapper[PipelineContext], agent: Agent) -> str:
    """Dynamic instructions inject learned knowledge at runtime."""
    base = """You are a Blender Python script writer. Generate complete, self-contained
    scripts for volumetric VFX effects. Your scripts will be post-processed by a
    deterministic fixer that handles camera placement, light energy, and API corrections.
    Focus on creative scene composition and physics setup."""

    # Inject technique-specific knowledge
    patterns = ctx.context.knowledge.get_successful_patterns(
        ctx.context.request.effect_type, limit=3
    )
    if patterns:
        base += "\n\n## Known Working Patterns\n"
        for p in patterns:
            base += f"\n### {p.name} (success rate: {p.success_rate:.0%})\n```python\n{p.code}\n```\n"

    failures = ctx.context.knowledge.get_common_failures(
        ctx.context.request.effect_type, limit=3
    )
    if failures:
        base += "\n\n## Known Pitfalls (AVOID THESE)\n"
        for f in failures:
            base += f"- {f.error_signature}: {f.description}\n"

    return base

script_writer = Agent[PipelineContext](
    name="Script Writer",
    instructions=writer_instructions,  # Dynamic — injects knowledge
    model="gpt-5.2",
    model_settings=ModelSettings(temperature=0.4),
    output_type=WriterOutput,
    tools=[search_blender_docs, search_code_patterns, search_technique_knowledge],
)
```

#### Agent 2: Evaluator

- **Model:** `gpt-5-mini` (vision-capable, cost-efficient)
- **Input:** Rendered image(s) + effect type + quality criteria
- **Output:** Structured quality assessment
- **Tools:** None — pure vision analysis
- **Output type:** Pydantic model with `overall_score: int`, `issues: list[Issue]`, `strengths: list[str]`, `is_acceptable: bool`

**Key design:** The Evaluator is **stateless**. It sees one render at a time. It doesn't know what iteration this is or what was tried before. This prevents anchoring bias.

```python
class Issue(BaseModel):
    type: str
    severity: Literal["critical", "major", "minor"]
    description: str

class EvalOutput(BaseModel):
    overall_score: int = Field(ge=0, le=100)
    issues: list[Issue]
    strengths: list[str]
    is_acceptable: bool
    critical_failure: str | None = None  # "BLACK_SCREEN", "NO_SUBJECT", etc.

evaluator = Agent[PipelineContext](
    name="Evaluator",
    instructions="""You are a VFX quality evaluator. Analyze the rendered image and score
    it on a 0-100 scale. Focus on: subject visibility, lighting quality, physical plausibility,
    visual appeal. A score of 60+ means acceptable quality. Report all issues with severity.""",
    model="gpt-5-mini",
    output_type=EvalOutput,
    tools=[],  # NO TOOLS. Pure evaluation.
)
```

No tools means no side effects, no budget surprises, no runaway tool loops. The evaluator looks and reports. That's it.

#### Agent 3: Strategist

- **Model:** `o3` (reasoning model, best for multi-step planning)
- **Input:** Current script, evaluator's assessment, iteration history summary, available techniques, knowledge base entries
- **Output:** Structured modification plan
- **Tools:** `search_knowledge_base` (function_tool), `list_available_techniques` (function_tool)
- **Output type:** Pydantic model with `action: Literal["modify", "rewrite", "escalate"]`, `writer_briefing: str`, `new_technique: str | None`, `reasoning: str`

**Key design:** The Strategist is the ONLY agent that sees iteration history. It compresses its analysis into a clean briefing for the Writer. It decides when to modify vs. rewrite vs. give up.

```python
class StrategyOutput(BaseModel):
    action: Literal["modify", "rewrite", "escalate"]
    writer_briefing: str  # Curated instructions for the Script Writer
    new_technique: str | None = None
    reasoning: str
    parameter_overrides: dict[str, Any] = {}

strategist = Agent[PipelineContext](
    name="Strategist",
    instructions="""You are a VFX strategy agent. Given the quality evaluation and iteration
    history, decide the best next action:
    - "modify": Adjust specific aspects of the current script
    - "rewrite": Start fresh with a different technique (use when stuck for 2+ iterations)
    - "escalate": Request human help (use after 2 failed rewrites)

    Your writer_briefing is the ONLY thing the Script Writer will see. Make it clear,
    specific, and actionable. Do NOT include raw error logs or score histories.""",
    model="o3",
    model_settings=ModelSettings(
        reasoning=Reasoning(effort="high"),  # o3 reasoning
    ),
    output_type=StrategyOutput,
    tools=[search_knowledge_base, list_available_techniques],
)
```

### 2.2 Optional: Research Agent

Called as a tool by the Strategist when it needs more information about an unfamiliar effect type.

```python
class ResearchOutput(BaseModel):
    recommended_technique: str
    relevant_apis: list[str]
    code_snippets: list[str]
    confidence: float

researcher = Agent[PipelineContext](
    name="Researcher",
    instructions="""Research Blender techniques for the given effect type. Search the
    documentation, find relevant APIs, and provide code snippets that the Script Writer
    can use as reference.""",
    model="gpt-5-mini",
    output_type=ResearchOutput,
    tools=[search_blender_docs, run_diagnostic_script],
)

# Used as a tool on the Strategist:
strategist = Agent[PipelineContext](
    ...,
    tools=[
        search_knowledge_base,
        list_available_techniques,
        researcher.as_tool(
            tool_name="research_technique",
            tool_description="Research Blender techniques for an unfamiliar effect type. "
                           "Use when the knowledge base has no entries for this effect.",
        ),
    ],
)
```

This is the SDK's `agent.as_tool()` pattern — the Strategist retains control, the Researcher is a tool it can choose to invoke. The key SDK behavior: the Researcher receives *generated input* (not conversation history), runs independently, and returns its output to the Strategist as a tool result.

### 2.3 Why Not More Agents?

| Current Pattern | My Proposal | Rationale |
|----------------|-------------|-----------|
| Separate TechniqueSelector agent | Strategist handles this | Technique selection IS strategy |
| Separate ModificationStrategist agent | Strategist handles this | Modification planning IS strategy |
| Separate QualityGateJudge agent | Evaluator handles this | Quality gating IS evaluation |
| Executor as an agent | Python function | Execution is `subprocess.run()`, not reasoning |
| LearningAgent | Python functions | Pattern extraction is data processing |
| DocsExpert as an agent | `@function_tool` on Writer | Doc search is a tool, not a conversation |
| API Validator as an agent | Deterministic fixer function | Validation is a lookup table |

**Key insight from the literature:** An "agent" should only exist when you need *creative reasoning* about *ambiguous inputs*. If the task is "run this script in Blender" — that's `subprocess.run()`. If it's "extract patterns from data" — that's `ast.parse()`. If it's "search docs" — that's a `@function_tool`. Agents are expensive ($0.01-0.05/call) and fallible. Use them only for judgment calls.

---

## 3. The Pipeline

### 3.1 State Machine

```
START
  |
  v
[RESEARCH?] ──no──> [GENERATE] ──> [FIX] ──> [EXECUTE] ──> [EVALUATE]
  |                      ^                        |              |
  yes                    |                   fail |         pass |
  |                      |                        v              v
  v                      |                   [DIAGNOSE]       [DONE]
Research                 |                        |
  |                      |                        v
  v                      +──── [STRATEGIZE] <─────+
technique found                     |
                               escalate?
                                    |
                                    v
                               [HUMAN HELP]
```

### 3.2 Phase Detail

#### Phase 0: RESEARCH (conditional)

- **Trigger:** Effect type has no known techniques in the knowledge base
- **Actor:** Research Agent (called as tool by Strategist)
- **Output:** Technique recommendation + relevant API documentation
- **Skip condition:** Effect type matches a known technique with >50% historical success rate

#### Phase 1: GENERATE

- **Actor:** Script Writer agent
- **Input via Runner.run():** A prompt containing:
  - Effect description (from user)
  - Selected technique (from Strategist or default)
  - Blender API reference snippets (injected via dynamic instructions)
  - Code patterns from knowledge base (injected via dynamic instructions)
  - Modification instructions (if iteration > 0, from Strategist's `writer_briefing`)
- **Output:** `WriterOutput` (structured, typed via Pydantic)

#### Phase 2: FIX (deterministic, NOT an agent)

This is the most critical phase and it's **pure Python, no LLM**.

```python
def fix_script(script: str, effect_type: str) -> tuple[str, list[str]]:
    """Apply all deterministic fixes. Returns (fixed_script, fixes_applied)."""
    fixes = []

    # 1. API attribute corrections (lookup table)
    script, api_fixes = fix_blender_api_attributes(script)
    fixes.extend(api_fixes)

    # 2. Camera placement (geometry calculation, injected into script)
    script, cam_fixes = inject_camera_validation(script, effect_type)
    fixes.extend(cam_fixes)

    # 3. Lighting bounds (arithmetic clamping)
    script, light_fixes = clamp_light_energy(script, effect_type)
    fixes.extend(light_fixes)

    # 4. Physics effector injection (rule-based)
    script, phys_fixes = inject_required_effectors(script, effect_type)
    fixes.extend(phys_fixes)

    # 5. Render settings enforcement
    script, render_fixes = enforce_render_settings(script)
    fixes.extend(render_fixes)

    # 6. Material injection (rule-based per effect type)
    script, mat_fixes = inject_materials(script, effect_type)
    fixes.extend(mat_fixes)

    return script, fixes
```

**This is the lesson from the S1.5 benchmarks:** Camera inside the glass. Light energy oscillating wildly. API attributes wrong. None of these are creative problems. They're all solvable by deterministic code that runs AFTER the LLM generates its script and BEFORE Blender executes it.

The fix layer is a **compiler pass**, not an agent.

#### Phase 3: EXECUTE (deterministic, NOT an agent)

```python
async def execute_script(script_path: str, timeout: int = 300) -> ExecutionResult:
    """Run script in headless Blender. Pure subprocess management."""
    proc = await asyncio.create_subprocess_exec(
        "blender", "-b", "--python", script_path,
        "--python-exit-code", "1",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )
    except asyncio.TimeoutError:
        proc.kill()
        return ExecutionResult(exit_code=-1, error="Timeout", ...)

    return ExecutionResult(
        exit_code=proc.returncode,
        stdout=stdout.decode(),
        stderr=stderr.decode(),
        render_path=find_render_output(stdout.decode()),
    )
```

No LLM needed. This is a subprocess call with timeout management.

#### Phase 4: EVALUATE

- **Actor:** Evaluator agent (vision model)
- **Input via Runner.run():** Rendered image + effect type + quality criteria
- **Output:** `EvalOutput` (structured)
- **Additional:** ML metrics (LPIPS, CLIP) computed by Python functions, provided as context alongside the image

```python
# Vision input format for the Evaluator
eval_input = [
    {"role": "user", "content": [
        {"type": "input_text", "text": build_eval_prompt(request, ml_metrics)},
        {"type": "input_image", "image_url": f"file://{exec_result.render_path}"},
    ]}
]
result = await Runner.run(evaluator, eval_input, context=ctx)
verdict: EvalOutput = result.final_output
```

#### Phase 5: STRATEGIZE (if quality < threshold)

- **Actor:** Strategist agent
- **Input:** Current script, evaluator's report, iteration summary, knowledge base
- **Output:** `StrategyOutput` — including the `writer_briefing` that the Writer will receive next iteration

### 3.3 The Complete Pipeline

```python
async def run_pipeline(request: AssetRequest) -> PipelineResult:
    """The complete asset generation pipeline. ~80 lines of control flow."""
    ctx = PipelineContext(request=request, session_id=generate_session_id())
    hooks = PipelineHooks()

    with trace("asset_pipeline", group_id=ctx.session_id):

        # Determine starting technique
        if ctx.knowledge.has_technique(request.effect_type):
            technique = ctx.knowledge.best_technique(request.effect_type)
        else:
            # Use Strategist to research (it has the Researcher as a tool)
            research_result = await Runner.run(
                strategist,
                f"Research the best technique for: {request.description}",
                context=ctx, hooks=hooks,
            )
            technique = research_result.final_output.new_technique

        writer_briefing = build_initial_briefing(request, technique)

        for iteration in range(request.max_iterations or 5):
            ctx.current_iteration = iteration

            # 1. GENERATE
            writer_result = await Runner.run(
                script_writer, writer_briefing, context=ctx, hooks=hooks,
            )
            script = writer_result.final_output.script

            # 2. FIX (deterministic)
            script, fixes = fix_script(script, request.effect_type)
            ctx.log_fixes(fixes)

            # 3. EXECUTE
            exec_result = await execute_script(save_script(script))

            if exec_result.exit_code != 0:
                # Execution failed — diagnose and retry
                ctx.log_execution_failure(exec_result)
                if iteration < request.max_iterations - 1:
                    strategy = await Runner.run(
                        strategist,
                        build_diagnosis_prompt(exec_result, ctx),
                        context=ctx, hooks=hooks,
                    )
                    if strategy.final_output.action == "escalate":
                        return PipelineResult(success=False, needs_human=True, ...)
                    writer_briefing = strategy.final_output.writer_briefing
                continue

            # 4. EVALUATE
            ml_metrics = compute_ml_metrics(exec_result.render_path)
            eval_result = await Runner.run(
                evaluator,
                build_eval_input(exec_result, request, ml_metrics),
                context=ctx, hooks=hooks,
            )
            verdict = eval_result.final_output
            ctx.log_evaluation(verdict)

            if verdict.is_acceptable:
                await record_success(ctx, iteration)
                return PipelineResult(
                    success=True, score=verdict.overall_score,
                    render_path=exec_result.render_path,
                    iterations=iteration + 1,
                )

            # 5. STRATEGIZE
            if iteration < request.max_iterations - 1:
                strategy = await Runner.run(
                    strategist,
                    build_strategy_prompt(ctx, verdict),
                    context=ctx, hooks=hooks,
                )
                if strategy.final_output.action == "escalate":
                    return PipelineResult(success=False, needs_human=True, ...)
                writer_briefing = strategy.final_output.writer_briefing

        return PipelineResult(
            success=False, best_score=ctx.best_score,
            iterations=request.max_iterations,
        )
```

**Note:** The pipeline controller is plain Python. It's a state machine with a `for` loop. LLMs don't decide "what phase comes next" — the pipeline always follows the same sequence. LLMs decide *what to generate* and *how to improve*.

---

## 4. State Management

### 4.1 Context Object (SDK `RunContextWrapper`)

The SDK's context mechanism is perfect for shared state. A single dataclass flows through all agents, tools, hooks, and guardrails:

```python
@dataclass
class PipelineContext:
    # Request
    request: AssetRequest
    session_id: str

    # Current state
    current_technique: str | None = None
    current_script: str | None = None
    current_iteration: int = 0

    # History (compressed summaries, NOT raw conversation)
    iteration_summaries: list[IterationSummary] = field(default_factory=list)

    # Knowledge (loaded at startup, updated after runs)
    knowledge: KnowledgeBase = field(default_factory=KnowledgeBase)

    # Budget tracking
    budget: BudgetTracker = field(default_factory=BudgetTracker)

    # Best result so far
    best_score: float = 0.0
    best_render_path: str | None = None
```

**Critical design choice:** `iteration_summaries` is a list of compressed summaries (~200 tokens each), NOT raw conversation history. Each contains: what was tried, the score, the main issue, and the strategist's recommendation. This prevents context window bloat across iterations.

**SDK note:** The context object is **mutable** and shared. Tools, guardrails, and hooks can all read and modify it. It is NOT sent to the LLM — it's purely for Python-side coordination.

### 4.2 Why Not SQLiteSession?

The SDK provides `SQLiteSession` for automatic conversation history management. I'd skip it for this use case because:

1. **Cross-iteration context needs compression** — Raw conversation from iteration 1 is noise for iteration 5. The Strategist produces a clean briefing; raw history doesn't help.
2. **State is structured, not conversational** — We need "best score", "current technique", "fixes applied" — not "what the LLM said 3 turns ago."
3. **JSON files are more debuggable** — You can `cat session.json` and see exactly what state the system is in. You can't easily inspect SQLite session tables.

### 4.3 Session Persistence (Cross-Session Resume)

Simple JSON file per session:

```python
@dataclass
class SessionState:
    session_id: str
    request: AssetRequest
    iterations: list[IterationRecord]
    best_score: float
    best_script_path: str | None
    best_render_path: str | None
    status: Literal["running", "paused", "completed", "failed"]
    technique_used: str
    created_at: str
    updated_at: str

def save_session(ctx: PipelineContext):
    state = SessionState.from_context(ctx)
    path = f"sessions/{ctx.session_id}/state.json"
    Path(path).write_text(state.to_json())

def load_session(session_id: str) -> PipelineContext:
    path = f"sessions/{session_id}/state.json"
    state = SessionState.from_json(Path(path).read_text())
    return state.to_context()
```

Save after every iteration. Resume is trivial: load JSON, reconstruct `PipelineContext`, continue.

---

## 5. Deterministic Safety Layer

This is the layer that would have prevented every S1.5 failure.

### 5.1 API Attribute Fixer

A pure lookup table. No LLM.

```python
BLENDER_5_RENAMES = {
    "resolution_divisions": "resolution_max",
    "use_adaptive_time_steps": "use_adaptive_timesteps",
    "use_dissolve": "use_dissolve_smoke",
    "timesteps_per_frame": "timesteps_max",
    "timesteps_maximum": "timesteps_max",
    "ShaderNodeMixRGB": "ShaderNodeMix",
    "ShaderNodeSeparateRGB": "ShaderNodeSeparateColor",
    "BLENDER_EEVEE_NEXT": "BLENDER_EEVEE",
    "reaction_speed": "burning_rate",
    "Subsurface Color": "",  # Removed in Blender 5
}

def fix_blender_api_attributes(script: str) -> tuple[str, list[str]]:
    fixes = []
    for wrong, correct in BLENDER_5_RENAMES.items():
        if wrong in script:
            if correct:  # Rename
                script = script.replace(wrong, correct)
                fixes.append(f"Renamed '{wrong}' -> '{correct}'")
            else:  # Removed attribute
                # Comment out the line
                for line in script.split('\n'):
                    if wrong in line:
                        script = script.replace(line, f"# REMOVED IN BLENDER 5: {line}")
                        fixes.append(f"Commented out removed attribute '{wrong}'")
    return script, fixes
```

### 5.2 Camera Placement (The #1 Fix)

Camera placement is trigonometry, not art direction. This validation code gets **injected into every generated script** as a post-processing step:

```python
CAMERA_VALIDATION_SNIPPET = '''
# === CAMERA VALIDATION (auto-injected) ===
import math
import mathutils

def _validate_camera():
    """Ensure camera can see the scene. Repositions if inside or too close."""
    camera = None
    for obj in bpy.data.objects:
        if obj.type == 'CAMERA':
            camera = obj
            break
    if not camera:
        return

    # Compute world-space bounding box of all mesh objects
    min_corner = mathutils.Vector((float('inf'),) * 3)
    max_corner = mathutils.Vector((float('-inf'),) * 3)
    depsgraph = bpy.context.evaluated_depsgraph_get()
    has_geometry = False

    for obj in bpy.data.objects:
        if obj.type not in ('MESH', 'CURVE', 'SURFACE', 'META'):
            continue
        has_geometry = True
        eval_obj = obj.evaluated_get(depsgraph)
        for corner in eval_obj.bound_box:
            world_corner = eval_obj.matrix_world @ mathutils.Vector(corner)
            for i in range(3):
                min_corner[i] = min(min_corner[i], world_corner[i])
                max_corner[i] = max(max_corner[i], world_corner[i])

    if not has_geometry:
        return

    center = (min_corner + max_corner) / 2
    diagonal = (max_corner - min_corner).length
    radius = diagonal / 2

    # Minimum distance from FOV geometry
    fov = camera.data.angle  # radians
    min_distance = (radius * 1.5) / math.tan(fov / 2) if fov > 0.01 else radius * 4
    min_distance = max(min_distance, 0.5)

    # Check current camera position
    cam_to_center = (camera.location - center).length

    if cam_to_center < min_distance:
        # Camera too close or inside scene — reposition
        direction = (camera.location - center)
        if direction.length < 0.01:
            direction = mathutils.Vector((0.5, -1.0, 0.4))
        direction.normalize()
        camera.location = center + direction * min_distance
        print(f"[CAMERA FIX] Repositioned from distance {cam_to_center:.2f} to {min_distance:.2f}")

    # Ensure camera points at scene center
    direction = center - camera.location
    rot = direction.to_track_quat('-Z', 'Y').to_euler()
    camera.rotation_euler = rot

_validate_camera()
# === END CAMERA VALIDATION ===
'''

def inject_camera_validation(script: str, effect_type: str) -> tuple[str, list[str]]:
    """Inject camera validation before the render call."""
    # Find the render call and inject before it
    render_markers = ['bpy.ops.render.render', 'scene.render']
    for marker in render_markers:
        if marker in script:
            script = script.replace(marker, CAMERA_VALIDATION_SNIPPET + '\n' + marker, 1)
            return script, ["Injected camera validation"]
    # If no render call found, append at end
    script += '\n' + CAMERA_VALIDATION_SNIPPET
    return script, ["Appended camera validation (no render call found)"]
```

### 5.3 Light Energy Bounds

Per-effect-type clamping via regex on the generated script:

```python
import re

LIGHT_BOUNDS = {
    "fire":      {"key": (50, 500),    "fill": (10, 100),   "world": (0.0, 0.1)},
    "explosion": {"key": (200, 2000),  "fill": (50, 500),   "world": (0.0, 0.05)},
    "smoke":     {"key": (100, 1000),  "fill": (20, 200),   "world": (0.0, 0.2)},
    "liquid":    {"key": (100, 1500),  "fill": (30, 300),   "world": (0.0, 0.3)},
    "candle":    {"key": (5, 50),      "fill": (1, 10),     "world": (0.0, 0.02)},
    "default":   {"key": (50, 1000),   "fill": (10, 200),   "world": (0.0, 0.2)},
}

def clamp_light_energy(script: str, effect_type: str) -> tuple[str, list[str]]:
    bounds = LIGHT_BOUNDS.get(effect_type, LIGHT_BOUNDS["default"])
    fixes = []

    # Find light.energy assignments and clamp
    pattern = r'(\.energy\s*=\s*)(\d+\.?\d*)'
    for match in re.finditer(pattern, script):
        value = float(match.group(2))
        lo, hi = bounds["key"]  # Use key light bounds as default
        if value < lo or value > hi:
            clamped = max(lo, min(hi, value))
            script = script.replace(match.group(0), f'{match.group(1)}{clamped}')
            fixes.append(f"Clamped light energy {value} -> {clamped} (bounds: {lo}-{hi})")

    return script, fixes
```

### 5.4 Collision Effector Injection

For liquid simulations, automatically ensure collision objects exist:

```python
def inject_required_effectors(script: str, effect_type: str) -> tuple[str, list[str]]:
    if effect_type not in ("liquid", "water", "pour"):
        return script, []

    if "collision" not in script.lower() and "'COLLISION'" not in script:
        injection = '''
# === COLLISION INJECTION (auto-injected) ===
for obj in bpy.data.objects:
    if obj.type == 'MESH' and not any(
        m.type == 'FLUID' for m in obj.modifiers
    ):
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.modifier_add(type='COLLISION')
        print(f"[COLLISION FIX] Added collision to {obj.name}")
# === END COLLISION INJECTION ===
'''
        # Insert before bake/render
        for marker in ['bpy.ops.fluid.bake', 'bpy.ops.render']:
            if marker in script:
                script = script.replace(marker, injection + '\n' + marker, 1)
                return script, ["Injected collision effectors for liquid containment"]

        script += '\n' + injection
        return script, ["Appended collision effectors"]

    return script, []
```

### 5.5 The Key Insight

All of these fixers share a property: **they are correct by construction**. They don't guess. They compute. They use geometry, lookup tables, and rule-based logic.

The LLM's job is to generate a script that captures the *creative intent*. The deterministic layer's job is to make sure the result is *physically valid*.

---

## 6. Self-Learning System

### 6.1 What to Learn

The system should learn:

1. **Technique effectiveness:** "For fire effects, `mantaflow_gas` with `heat_source` scores 72 average over 12 runs"
2. **Parameter correlations:** "For candle flames, `burning_rate` between 2.0-4.0 with `resolution_max` 96 produces best results"
3. **Failure patterns:** "Camera inside mesh detected 3 times for liquid pours with container geometry"
4. **Code patterns:** Working snippets for materials, physics setup, scene composition

It should NOT learn: raw scripts (too large), LLM conversation logs (noise), transient errors (Blender crashes).

### 6.2 Knowledge Store: SQLite, Not a Vector Store

At the scale of <1000 knowledge entries, SQLite with FTS5 is simpler, faster, cheaper, and more debuggable than a vector store. No external service dependency, no API cost, no embedding cost.

```sql
CREATE TABLE techniques (
    id INTEGER PRIMARY KEY,
    effect_type TEXT NOT NULL,
    technique_name TEXT NOT NULL,
    avg_score REAL DEFAULT 0,
    run_count INTEGER DEFAULT 0,
    success_count INTEGER DEFAULT 0,
    last_used TEXT,
    notes TEXT
);

CREATE TABLE code_patterns (
    id INTEGER PRIMARY KEY,
    effect_type TEXT NOT NULL,
    pattern_name TEXT NOT NULL,
    code TEXT NOT NULL,
    success_rate REAL DEFAULT 0,
    use_count INTEGER DEFAULT 0,
    description TEXT
);

CREATE TABLE failure_patterns (
    id INTEGER PRIMARY KEY,
    error_signature TEXT NOT NULL,
    effect_type TEXT,
    occurrence_count INTEGER DEFAULT 1,
    fix_applied TEXT,
    last_seen TEXT
);

CREATE TABLE iteration_records (
    id INTEGER PRIMARY KEY,
    session_id TEXT NOT NULL,
    iteration INTEGER NOT NULL,
    effect_type TEXT NOT NULL,
    technique TEXT NOT NULL,
    score REAL,
    issues TEXT,        -- JSON array
    fixes_applied TEXT, -- JSON array
    strategy_action TEXT,
    created_at TEXT
);
```

Add a vector store when you have 10,000+ entries and need semantic similarity that keyword search can't handle. That's months away.

### 6.3 Dynamic Instructions (SDK Feature)

The SDK's dynamic instructions feature lets you inject learned knowledge at runtime:

```python
def writer_instructions(ctx: RunContextWrapper[PipelineContext], agent: Agent) -> str:
    base = "You are a Blender Python script writer..."

    # Inject technique-specific knowledge from the SQLite knowledge base
    effect_type = ctx.context.request.effect_type
    patterns = ctx.context.knowledge.get_successful_patterns(effect_type, limit=3)
    if patterns:
        base += "\n\n## Known Working Patterns\n"
        for p in patterns:
            base += f"\n### {p.name} (success rate: {p.success_rate:.0%})\n```python\n{p.code}\n```\n"

    return base
```

This is how learning translates to behavior: successful runs update the knowledge base, and the dynamic instructions surface that knowledge to the Writer on future runs. The OpenAI Self-Evolving Agents cookbook validates this approach — their GEPA optimization evolves system prompts based on evaluation outcomes.

### 6.4 Escape Velocity (Simplified)

Two levels instead of five:

| Level | Trigger | Action |
|-------|---------|--------|
| **MODIFY** | Default | Strategist adjusts parameters/approach in current script |
| **REWRITE** | Same issue 2 iterations OR no score improvement for 2 iterations | Strategist selects a different technique; Writer generates from scratch |

If REWRITE fails twice, escalate to human. No levels 3, 4, or 5. If the system can't solve it with two different techniques, it genuinely needs human guidance.

---

## 7. Model Selection Strategy

### 7.1 Model Assignments

| Role | Model | Why | Approx Cost/Call |
|------|-------|-----|-----------------|
| Script Writer | `gpt-5.2` | Best code generation; scripts are complex Python | ~$0.02 |
| Evaluator | `gpt-5-mini` | Vision analysis doesn't need the full model | ~$0.01 |
| Strategist | `o3` | Multi-step reasoning benefits from chain-of-thought | ~$0.03 |
| Researcher | `gpt-5-mini` | Doc search and synthesis | ~$0.01 |

### 7.2 Cost Per Asset

Per iteration: Writer ($0.02) + Evaluator ($0.01) + Strategist ($0.03) = **~$0.06**

At 5 iterations per asset and $20/month budget: **~66 assets/month**. This is viable.

### 7.3 SDK's `ModelSettings` for Reasoning Models

The SDK has explicit support for reasoning models like `o3`:

```python
from agents import ModelSettings, Reasoning

strategist = Agent(
    model="o3",
    model_settings=ModelSettings(
        reasoning=Reasoning(effort="high"),  # Controls reasoning depth
    ),
)
```

This is a significant advantage — the Strategist gets deep reasoning for "what went wrong and what to try" without adding more agents.

---

## 8. Error Recovery

### 8.1 Error Classification

```python
class ErrorType(Enum):
    BLENDER_CRASH = "blender_crash"       # Segfault, OOM — retry
    TIMEOUT = "timeout"                    # Bake/render timeout — retry with lower res
    PYTHON_SYNTAX = "python_syntax"        # Bad script — send to Strategist
    ATTRIBUTE_ERROR = "attribute_error"    # Wrong API — try deterministic fix first
    PHYSICS_ERROR = "physics_error"        # Simulation failed — send to Strategist
    INFRASTRUCTURE = "infrastructure"      # Disk full, API key — escalate immediately
```

### 8.2 Recovery Strategy

```python
async def handle_execution_failure(exec_result, ctx) -> RecoveryAction:
    error_type = classify_error(exec_result.stderr)

    match error_type:
        case ErrorType.BLENDER_CRASH | ErrorType.TIMEOUT:
            if ctx.retry_count < 2:
                return RecoveryAction.RETRY  # Simple retry
            return RecoveryAction.STRATEGIZE  # Ask Strategist after 2 retries

        case ErrorType.ATTRIBUTE_ERROR:
            # Try deterministic fix first
            fixed, fixes = fix_blender_api_attributes(ctx.current_script)
            if fixes:
                return RecoveryAction.RETRY_WITH_FIXES
            return RecoveryAction.STRATEGIZE  # Unknown attribute — ask Strategist

        case ErrorType.PYTHON_SYNTAX | ErrorType.PHYSICS_ERROR:
            return RecoveryAction.STRATEGIZE  # Needs creative reasoning

        case ErrorType.INFRASTRUCTURE:
            return RecoveryAction.ESCALATE_TO_HUMAN
```

### 8.3 Circuit Breaker

```python
def check_circuit_breaker(ctx: PipelineContext) -> bool:
    """Returns True if the pipeline should halt."""
    if len(ctx.iteration_summaries) < 3:
        return False

    # Same error 3 times consecutively
    errors = [s.primary_error for s in ctx.iteration_summaries[-3:] if s.primary_error]
    if len(errors) == 3 and len(set(errors)) == 1:
        return True

    # No score improvement over 3 iterations (plateau)
    scores = [s.score for s in ctx.iteration_summaries[-3:] if s.score is not None]
    if len(scores) == 3 and max(scores) - min(scores) < 5:
        return True

    return False
```

---

## 9. Human-in-the-Loop

### 9.1 When to Involve the Human

| Trigger | Action |
|---------|--------|
| Novel effect type (no known technique) | Ask for guidance before research |
| Circuit breaker tripped | Report failure + show last render |
| Score plateau (3 iterations, <5 point change) | Show renders, ask if acceptable |
| Budget > 80% consumed | Warn and ask to continue |
| Strategist action == "escalate" | Full diagnostic report |

### 9.2 Implementation

The SDK doesn't have a native "pause and wait for human input" mechanism. The documented pattern is: save state, raise exception, resume with new `Runner.run()`.

```python
class HumanInputRequired(Exception):
    def __init__(self, question: str, options: list[str] | None = None):
        self.question = question
        self.options = options

async def request_human_input(ctx, question, options=None):
    ctx.status = "paused"
    save_session(ctx)
    raise HumanInputRequired(question, options)

# Resume:
async def resume_with_input(session_id: str, answer: str):
    ctx = load_session(session_id)
    ctx.human_answer = answer
    ctx.status = "running"
    return await run_pipeline(ctx)
```

### 9.3 Transparency Report

Every pipeline run produces a structured report. The human can see exactly what happened, why, and what it cost:

```python
@dataclass
class PipelineReport:
    session_id: str
    request: AssetRequest
    total_iterations: int
    best_score: float
    best_render_path: str | None
    technique_used: str
    fixes_applied: list[str]
    strategist_reasoning: list[str]  # Summary of each strategic decision
    total_cost_estimate: float
    duration_seconds: float
    knowledge_entries_created: int
```

---

## 10. Budget Management

### 10.1 Pre-flight Estimation

```python
def estimate_cost(request: AssetRequest) -> CostEstimate:
    per_iteration = 0.06
    max_cost = per_iteration * (request.max_iterations or 5)
    remaining = get_remaining_budget()
    return CostEstimate(
        per_iteration=per_iteration,
        max_total=max_cost,
        remaining_budget=remaining,
        can_afford=max_cost <= remaining,
    )
```

### 10.2 Token Tracking via RunHooks

The SDK's `RunHooks` provide `on_llm_end` callbacks where you can track usage:

```python
from agents import RunHooks

class BudgetHooks(RunHooks[PipelineContext]):
    async def on_llm_end(self, context, agent):
        # context.usage tracks token counts
        cost = estimate_token_cost(
            input_tokens=context.usage.input_tokens,
            output_tokens=context.usage.output_tokens,
            model=agent.model,
        )
        context.context.budget.add_cost(cost)

        if context.context.budget.exceeded_warning_threshold():
            context.context.budget_warning = True
```

### 10.3 Input Guardrail for Budget

```python
from agents import input_guardrail, GuardrailFunctionOutput

@input_guardrail
async def budget_guardrail(ctx, agent, input) -> GuardrailFunctionOutput:
    if not ctx.context.budget.can_afford_call():
        return GuardrailFunctionOutput(
            output_info="Budget exhausted",
            tripwire_triggered=True,
        )
    return GuardrailFunctionOutput(output_info="OK", tripwire_triggered=False)

# Apply to the most expensive agent
script_writer = Agent(
    ...,
    input_guardrails=[budget_guardrail],
)
```

**SDK note:** Input guardrails run in parallel with the agent's first LLM call. If the guardrail trips, the response is discarded and `InputGuardrailTripwireTriggered` is raised. Catch it in the pipeline controller.

---

## 11. What I Would NOT Build

### 11.1 No Agent for Execution

`subprocess.run()` doesn't need creative reasoning. An executor "agent" adds cost and failure surface for zero value.

### 11.2 No Agent for API Validation

A lookup table is more reliable than any LLM at catching `resolution_divisions` → `resolution_max`. LLMs hallucinate API names. Lookup tables don't.

### 11.3 No Agent for Learning/Pattern Extraction

Pattern extraction is `ast.parse()` and database inserts. Not creative reasoning.

### 11.4 No Vector Store (Initially)

SQLite + FTS5 handles <1000 entries trivially. Add a vector store when you need semantic similarity at scale.

### 11.5 No Handoffs

The SDK's handoff pattern transfers control from one agent to another — the second agent takes over the conversation. This is perfect for customer support routing. It's wrong for a pipeline where Python needs to control flow between phases. Use `agent.as_tool()` instead.

From the SDK docs: "Agents as tools: The inner agent's output is used as a tool output, so the orchestrator processes it. Use when you want a central agent to orchestrate."

### 11.6 No Streaming (Initially)

The pipeline is batch-oriented. `Runner.run()` returns a complete result. Streaming (`Runner.run_streamed()`) adds complexity for real-time UIs that don't exist yet.

### 11.7 No MCP Servers

All tools are in-process Python functions. The SDK's MCP support (`MCPServerStdio`, `MCPServerSse`) is for connecting to external services. The tools here (doc search, knowledge base queries, diagnostics) are local function calls.

---

## 12. Implementation Order

### Phase 1: Minimal Loop (Days 1-3)

1. Define `WriterOutput`, `EvalOutput` Pydantic models
2. Create Script Writer agent with static instructions
3. Create Evaluator agent (vision model)
4. Write `fix_script()` with API attribute fixer only
5. Write `execute_script()` subprocess wrapper
6. Wire up pipeline: prompt → write → fix → execute → evaluate
7. **Test:** Generate a fire effect, get a score

### Phase 2: Iteration (Days 4-6)

1. Create Strategist agent with `StrategyOutput`
2. Add iteration loop with Strategist driving improvements
3. Add `IterationSummary` compression
4. Implement circuit breaker
5. **Test:** Fire effect improves over 3 iterations

### Phase 3: Deterministic Fixes (Days 7-9)

1. Camera validation injection
2. Light energy clamping
3. Collision effector injection
4. Material injection
5. Render settings enforcement
6. **Test:** All three S1.5 benchmark scenarios score >30 on first iteration

### Phase 4: Learning (Days 10-12)

1. SQLite knowledge base (schema + queries)
2. Dynamic instructions on Writer (inject patterns)
3. After-run learning: update technique stats, extract patterns
4. **Test:** Second run of same effect type produces better results

### Phase 5: Polish (Days 13-15)

1. Session persistence (JSON save/resume)
2. Budget tracking (RunHooks)
3. Human input request/response flow
4. Pipeline reports
5. Tracing (`with trace(...)`)
6. Research Agent on Strategist (for novel effects)
7. **Test:** Full end-to-end with pause/resume

---

## Appendix A: SDK Features Used

| SDK Feature | Where Used | Why |
|-------------|-----------|-----|
| `Agent` (dataclass) | Writer, Evaluator, Strategist, Researcher | Core building block |
| `Runner.run()` | Pipeline controller | Executes each agent |
| `output_type` (Pydantic) | All agents | Structured, typed outputs — no parsing |
| `@function_tool` | Doc search, knowledge queries, diagnostics | Expose Python functions to agents |
| `agent.as_tool()` | Researcher (on Strategist) | Nested agent execution, caller retains control |
| Dynamic instructions (callable) | Writer, Strategist | Inject learned knowledge at runtime |
| `RunContextWrapper[T]` | All agents/tools/hooks | Shared pipeline state (mutable, not sent to LLM) |
| `RunHooks` | Budget tracking, logging | Lifecycle callbacks across ALL agents |
| `@input_guardrail` | Writer | Budget check before expensive calls |
| `ModelSettings.reasoning` | Strategist (o3) | Enable chain-of-thought reasoning |
| `Tracing` (`with trace(...)`) | Pipeline controller | End-to-end observability |
| `RunConfig` | Pipeline controller | Global model settings, tracing config |
| `tool_use_behavior` | Not needed (default `"run_llm_again"` is correct) | Agents process tool results naturally |

### SDK Features Deliberately Not Used

| Feature | Why Not |
|---------|---------|
| Handoffs | Python controls flow; no agent-to-agent transfers needed |
| `SQLiteSession` | Custom JSON state is simpler and more debuggable |
| `MCPServer*` | All tools are in-process |
| `HostedMCPTool` | No external MCP servers needed |
| `Runner.run_streamed()` | Batch-oriented pipeline; no real-time UI |
| `@output_guardrail` | Input guardrails + structured outputs sufficient |
| `tool_input_guardrail` / `tool_output_guardrail` | Deterministic fixers handle safety |
| `AgentHooks` | `RunHooks` (global) is sufficient for budget/logging |
| `Prompt` object | Only for OpenAI Responses API prompts; not needed |

## Appendix B: What This Architecture Prevents

| S1.5 Failure | Prevention Mechanism |
|-------------|---------------------|
| Camera inside glass (wine_pour) | `inject_camera_validation()` — detects camera inside mesh bounds, repositions |
| Camera too close (candle) | Minimum distance computed from scene bounding sphere + FOV |
| Light energy oscillation (candle) | `LIGHT_BOUNDS` per-effect-type clamping |
| No collision effectors (wine_pour) | `inject_required_effectors()` for liquid effects |
| API attribute errors | `BLENDER_5_RENAMES` lookup table |
| Infinite loops on same error | Circuit breaker (3 identical errors = halt) |
| Budget exhaustion | `BudgetHooks` + `budget_guardrail` |
| LLM deciding pipeline flow | Pipeline is a Python `for` loop, not agent handoffs |

## Appendix C: File Structure

```
blender-vfx-orchestrator/
  pipeline.py              # ~80 lines — the state machine controller
  agents/
    writer.py              # Script Writer agent + WriterOutput model
    evaluator.py           # Evaluator agent + EvalOutput model
    strategist.py          # Strategist agent + StrategyOutput model
    researcher.py          # Research Agent (optional, used as tool)
  fixers/
    __init__.py            # fix_script() entry point
    api_attributes.py      # Blender 5 API rename table
    camera.py              # Camera placement validation + injection
    lighting.py            # Light energy bounds per effect type
    collision.py           # Effector injection for liquids
    materials.py           # Material snippet injection
    render_settings.py     # Render config enforcement
  execution/
    blender_runner.py      # execute_script() — subprocess management
  knowledge/
    store.py               # SQLite knowledge base (FTS5)
    learning.py            # After-run pattern extraction
  models/
    context.py             # PipelineContext dataclass
    request.py             # AssetRequest model
    results.py             # PipelineResult, ExecutionResult, etc.
    iteration.py           # IterationSummary, IterationRecord
  sessions/
    manager.py             # JSON save/load/resume
  hooks/
    budget.py              # BudgetHooks (RunHooks subclass)
    logging.py             # LoggingHooks (RunHooks subclass)
  guardrails/
    budget.py              # @input_guardrail for budget check
  tools/
    doc_search.py          # @function_tool — Blender doc search
    knowledge_query.py     # @function_tool — knowledge base queries
    diagnostic.py          # @function_tool — run diagnostic Blender scripts
```

## Appendix D: Research Sources

**Multi-Agent Code Generation:**
- AgentCoder (arXiv:2312.13010) — Programmer/TestDesigner/TestExecutor pattern
- Hybrid Cascade (arXiv:2505.18286) — Single-agent first, escalate on failure
- EvoMAS (arXiv:2602.06511) — Evolutionary multi-agent system generation

**Self-Improving Agents:**
- OpenAI Self-Evolving Agents Cookbook — GEPA optimization, evaluation-driven prompt evolution
- Ralph (github.com/snarktank/ralph) — Autonomous loop with context resets and persistent memory
- Mem0 (arXiv:2504.19413) — Production-ready long-term memory for agents

**OpenAI Agents SDK:**
- Official docs (github.com/openai/openai-agents-python/tree/main/docs)
- context7 MCP documentation queries
- Community patterns (community.openai.com)

**Blender Automation:**
- Blender 5.0 Python API — depsgraph evaluation for reliable bounding boxes
- `bpy_extras.camera_utils` — camera_view_bounds_2d for frame validation
- Headless execution patterns — `--python-exit-code` for CI/CD integration

---

*This document represents my honest assessment of how I'd build this system from scratch, based solely on web research, SDK documentation, and the mission statement. It deliberately avoids inspecting the existing codebase to provide an unbiased perspective. The core thesis: maximize LLM creativity, minimize LLM mechanical work, control flow with Python, and earn complexity through demonstrated reliability.*
