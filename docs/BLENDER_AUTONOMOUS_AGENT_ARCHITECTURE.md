# Blender Autonomous Agent Architecture

**Date:** 2026-01-06
**Status:** Design Document
**Goal:** Replace fragmented MCP servers with unified Agents SDK orchestration

---

## The Problem with Current Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      CURRENT SYSTEM                              │
│                                                                  │
│   Claude Code (manual orchestration)                            │
│        │                                                         │
│        ├──► script-generator MCP ──► generates script           │
│        │                                                         │
│        ├──► blender-executor MCP ──► runs Blender               │
│        │                                                         │
│        ├──► asset-evaluator MCP ──► evaluates render            │
│        │                                                         │
│        ├──► experiment-tracker MCP ──► logs results             │
│        │                                                         │
│        ├──► iteration-controller MCP ──► suggests changes       │
│        │                                                         │
│        └──► blender-librarian MCP ──► searches docs             │
│                                                                  │
│   Problems:                                                      │
│   - Claude Code must manually orchestrate 6+ MCP servers        │
│   - No shared state between servers                             │
│   - Each iteration requires multiple round-trips                │
│   - Learning is fragmented across systems                       │
│   - No autonomous iteration capability                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## The Solution: Unified Autonomous Agent

```
┌─────────────────────────────────────────────────────────────────┐
│                      NEW SYSTEM                                  │
│                                                                  │
│   Claude Code                                                    │
│        │                                                         │
│        └──► blender-autonomous MCP (single entry point)         │
│                    │                                             │
│                    ▼                                             │
│        ┌─────────────────────────────────────────┐              │
│        │   OpenAI Agents SDK Orchestrator        │              │
│        │   (GPT-5.2 with reasoning)              │              │
│        │                                         │              │
│        │   Shared Context:                       │              │
│        │   - Current script state                │              │
│        │   - Render history                      │              │
│        │   - Accumulated learnings               │              │
│        │   - Budget tracking                     │              │
│        └─────────────────────────────────────────┘              │
│                    │                                             │
│        ┌──────────┼──────────┬──────────┬──────────┐           │
│        ▼          ▼          ▼          ▼          ▼           │
│   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ │
│   │ Script  │ │Executor │ │ Quality │ │Learning │ │  Docs   │ │
│   │ Writer  │ │  Agent  │ │ Analyst │ │  Agent  │ │ Expert  │ │
│   └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘ │
│                                                                  │
│   Benefits:                                                      │
│   - Single MCP call triggers full autonomous workflow           │
│   - Shared context across all agents                            │
│   - Tight feedback loops (iterate without Claude Code)          │
│   - Self-improving through accumulated learnings                │
│   - Structured handoffs with type-safe outputs                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Agent Specifications

### 1. Master Orchestrator Agent

**Role:** Central coordinator that plans and executes asset creation workflows.

**Capabilities:**
- Receives high-level requests ("create a sun with prominences")
- Breaks down into sub-tasks
- Coordinates specialist agents via handoffs OR agents-as-tools
- Maintains shared context across iterations
- Decides when to stop iterating (quality threshold met)

```python
from agents import Agent, RunContextWrapper, handoff
from pydantic import BaseModel, Field
from typing import List, Optional
import json

class AssetRequest(BaseModel):
    """Input schema for asset creation requests."""
    asset_name: str = Field(description="Name for the asset")
    effect_type: str = Field(description="Type: sun, explosion, fire, nebula, smoke")
    description: str = Field(description="Detailed description of desired effect")
    reference_path: Optional[str] = Field(default=None, description="Reference image path")
    quality_threshold: float = Field(default=70.0, ge=0, le=100)
    max_iterations: int = Field(default=5, ge=1, le=20)

class IterationResult(BaseModel):
    """Result of a single iteration."""
    iteration: int
    script_path: str
    render_path: str
    quality_score: float
    issues: List[str]
    improvements_made: List[str]
    should_continue: bool

class AssetResult(BaseModel):
    """Final result of asset creation."""
    success: bool
    asset_name: str
    final_score: float
    iterations_used: int
    script_path: str
    render_path: str
    vdb_path: Optional[str]
    learnings: List[str]

ORCHESTRATOR_INSTRUCTIONS = """You are the Blender Autonomous Asset Creator.

Your mission: Create high-quality VFX assets through intelligent iteration.

## WORKFLOW

1. **Understand the Request**
   - Parse the asset request
   - Identify key visual characteristics needed
   - Check if we have relevant learnings from past attempts

2. **Generate Initial Script**
   - Hand off to Script Writer with detailed requirements
   - Script Writer will consult documentation as needed

3. **Execute and Render**
   - Hand off to Executor Agent to run Blender
   - Capture any errors for learning

4. **Evaluate Quality**
   - Hand off to Quality Analyst with render + requirements
   - Get structured quality assessment

5. **Iterate or Complete**
   - If quality_score >= threshold: SUCCESS
   - If iterations >= max: STOP (return best)
   - Otherwise: Analyze issues, plan improvements, goto step 2

6. **Record Learnings**
   - Hand off to Learning Agent with full session data
   - Accumulate knowledge for future attempts

## DECISION MAKING

- Trust specialist agents for their domains
- Aggregate their outputs to make workflow decisions
- Be aggressive about iteration - quality matters
- Record EVERYTHING for learning

## CONTEXT MANAGEMENT

You have access to shared context containing:
- current_script: The current Blender Python script
- render_history: List of (path, score, issues) for each iteration
- accumulated_learnings: Knowledge from past sessions
- budget_remaining: API budget status

Update context after each agent interaction."""

class SharedContext(BaseModel):
    """Shared state across all agents in a workflow."""
    asset_request: AssetRequest
    current_script: str = ""
    current_script_path: str = ""
    render_history: List[dict] = Field(default_factory=list)
    accumulated_learnings: List[str] = Field(default_factory=list)
    budget_remaining: float = 20.0
    iteration: int = 0
    best_score: float = 0.0
    best_render_path: str = ""
```

---

### 2. Script Writer Agent

**Role:** Generates and modifies Blender Python scripts.

**Key Innovation:** Has DIRECT MCP connection to blender-manual, so it can research while writing.

```python
from agents import Agent, function_tool, MCPServerStdio
from pathlib import Path

SCRIPT_WRITER_INSTRUCTIONS = """You are a Blender 5.0 Python script specialist.

## YOUR CAPABILITIES

1. **Write Blender Python scripts** for volumetric effects (Mantaflow, smoke, fire)
2. **Search documentation** via MCP tools when unsure about API
3. **Validate parameters** against known Blender API ranges
4. **Modify existing scripts** based on quality feedback

## SCRIPT REQUIREMENTS

All scripts MUST:
- Use Blender 5.0 API (not deprecated 4.x patterns)
- Include proper domain setup for Mantaflow
- Configure VDB export with correct paths
- Handle baking with appropriate resolution
- Include render setup (camera, lighting, output)

## PARAMETER VALIDATION

Before outputting any script, validate ALL parameters:
- turbulence: 0.0-1.0
- vorticity: 0.0-1.0
- flame_max_temp: 0.0-100000.0
- flame_smoke: 0.0-8.0
- burning_rate: 0.01-4.0
- domain_resolution: 32-512

## OUTPUT FORMAT

Always return a ScriptOutput with:
- script_content: Complete Python script
- script_path: Where to save it
- parameters_used: Dict of key parameters
- documentation_consulted: List of doc paths used
- confidence: 0.0-1.0

## WHEN MODIFYING SCRIPTS

Given quality feedback like "too dark", "wrong color", "lacks structure":
1. Search documentation for relevant parameters
2. Identify specific changes needed
3. Make MINIMAL changes (max 3 parameters per iteration)
4. Explain rationale for each change"""

class ScriptOutput(BaseModel):
    """Structured output from Script Writer."""
    script_content: str = Field(description="Complete Blender Python script")
    script_path: str = Field(description="Path where script should be saved")
    parameters_used: dict = Field(description="Key parameters and their values")
    documentation_consulted: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(description="Why these parameters were chosen")

@function_tool
def save_script(script_content: str, script_path: str) -> str:
    """Save a Blender Python script to disk."""
    path = Path(script_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(script_content)
    return json.dumps({"success": True, "path": str(path)})

@function_tool
def load_script(script_path: str) -> str:
    """Load an existing script for modification."""
    path = Path(script_path)
    if not path.exists():
        return json.dumps({"error": f"Script not found: {script_path}"})
    return json.dumps({"content": path.read_text()})

@function_tool
def validate_blender_params(params: str) -> str:
    """Validate parameters against Blender 5.0 API ranges."""
    RANGES = {
        "turbulence": (0.0, 1.0),
        "vorticity": (0.0, 1.0),
        "flame_max_temp": (0.0, 100000.0),
        "flame_smoke": (0.0, 8.0),
        "burning_rate": (0.01, 4.0),
        "smoke_density": (0.0, 1.0),
        "domain_resolution": (32, 512),
    }

    try:
        param_dict = json.loads(params)
    except:
        return json.dumps({"error": "Invalid JSON"})

    issues = []
    validated = {}
    for key, value in param_dict.items():
        if key in RANGES:
            min_val, max_val = RANGES[key]
            if value < min_val or value > max_val:
                issues.append(f"{key}={value} outside range [{min_val}, {max_val}]")
                validated[key] = max(min_val, min(max_val, value))
            else:
                validated[key] = value
        else:
            validated[key] = value

    return json.dumps({
        "valid": len(issues) == 0,
        "issues": issues,
        "validated_params": validated
    })

async def create_script_writer_agent(blender_manual_mcp: MCPServerStdio) -> Agent:
    """Create Script Writer with MCP documentation access."""
    return Agent(
        name="Script Writer",
        instructions=SCRIPT_WRITER_INSTRUCTIONS,
        model="gpt-5.2",
        output_type=ScriptOutput,
        tools=[save_script, load_script, validate_blender_params],
        mcp_servers=[blender_manual_mcp],  # Direct doc access!
    )
```

---

### 3. Executor Agent

**Role:** Runs Blender CLI, captures output, manages files.

**Key Innovation:** Wraps blender-executor functionality directly as agent tools.

```python
import subprocess
import asyncio
from pathlib import Path

EXECUTOR_INSTRUCTIONS = """You are the Blender execution specialist.

## YOUR ROLE

Execute Blender Python scripts and capture results.

## WORKFLOW

1. Receive script path from orchestrator
2. Run Blender in background mode
3. Capture stdout/stderr
4. Locate output files (renders, VDBs)
5. Return structured ExecutionResult

## ERROR HANDLING

Common errors and responses:
- "Python script error": Return error details for Script Writer
- "Out of memory": Suggest reducing resolution
- "File not found": Check paths and report
- "Bake failed": Often needs domain adjustment

## OUTPUT

Return ExecutionResult with:
- success: bool
- stdout/stderr: Full output
- render_paths: List of rendered images
- vdb_paths: List of VDB files
- duration_seconds: How long it took
- error_details: If failed, what went wrong"""

class ExecutionResult(BaseModel):
    """Structured output from Executor."""
    success: bool
    script_path: str
    stdout: str = ""
    stderr: str = ""
    render_paths: List[str] = Field(default_factory=list)
    vdb_paths: List[str] = Field(default_factory=list)
    duration_seconds: float = 0.0
    error_details: Optional[str] = None

@function_tool
async def run_blender_script(
    script_path: str,
    output_dir: str = "",
    timeout_seconds: int = 600
) -> str:
    """
    Execute a Blender Python script via CLI.

    Args:
        script_path: Path to the .py script
        output_dir: Override output directory
        timeout_seconds: Max execution time (default 10 minutes)

    Returns:
        JSON with execution results
    """
    script = Path(script_path)
    if not script.exists():
        return json.dumps({"success": False, "error": f"Script not found: {script_path}"})

    # Determine output directory
    if not output_dir:
        output_dir = f"build/vdb_output/{script.stem}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Build Blender command
    blender_path = "/usr/bin/blender"  # Adjust for system
    cmd = [
        blender_path,
        "--background",
        "--python", str(script),
        "--",
        "--output-dir", output_dir
    ]

    start_time = asyncio.get_event_loop().time()

    try:
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        stdout, stderr = await asyncio.wait_for(
            process.communicate(),
            timeout=timeout_seconds
        )

        duration = asyncio.get_event_loop().time() - start_time

        # Find output files
        output_path = Path(output_dir)
        render_paths = list(output_path.glob("*.png")) + list(output_path.glob("*.jpg"))
        vdb_paths = list(output_path.glob("*.vdb")) + list(output_path.glob("*.nvdb"))

        return json.dumps({
            "success": process.returncode == 0,
            "script_path": script_path,
            "stdout": stdout.decode()[-5000:],  # Last 5KB
            "stderr": stderr.decode()[-2000:],  # Last 2KB
            "render_paths": [str(p) for p in render_paths],
            "vdb_paths": [str(p) for p in vdb_paths],
            "duration_seconds": duration,
            "return_code": process.returncode
        })

    except asyncio.TimeoutError:
        return json.dumps({
            "success": False,
            "error": f"Execution timed out after {timeout_seconds}s"
        })
    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        })

@function_tool
def list_output_files(output_dir: str) -> str:
    """List all files in an output directory."""
    path = Path(output_dir)
    if not path.exists():
        return json.dumps({"error": f"Directory not found: {output_dir}"})

    files = {
        "renders": [str(f) for f in path.glob("*.png")] + [str(f) for f in path.glob("*.jpg")],
        "vdbs": [str(f) for f in path.glob("*.vdb")] + [str(f) for f in path.glob("*.nvdb")],
        "logs": [str(f) for f in path.glob("*.log")],
        "other": [str(f) for f in path.iterdir() if f.suffix not in [".png", ".jpg", ".vdb", ".nvdb", ".log"]]
    }
    return json.dumps(files)

def create_executor_agent() -> Agent:
    """Create Executor Agent."""
    return Agent(
        name="Executor",
        instructions=EXECUTOR_INSTRUCTIONS,
        model="gpt-4o-mini",  # Cheap model - just running commands
        output_type=ExecutionResult,
        tools=[run_blender_script, list_output_files]
    )
```

---

### 4. Quality Analyst Agent

**Role:** Evaluates render quality using vision + metrics.

**Key Innovation:** Combines GPT-5.2 vision with programmatic metrics.

```python
QUALITY_ANALYST_INSTRUCTIONS = """You are a VFX render quality analyst.

## YOUR ROLE

Evaluate render quality against requirements and provide structured feedback.

## ANALYSIS APPROACH

1. **Load the render** using load_image tool
2. **Evaluate against criteria** for the effect type
3. **Compare to reference** if provided
4. **Calculate quality score** (0-100)
5. **Identify specific issues** with actionable feedback

## EFFECT-SPECIFIC CRITERIA

**Sun:**
- Limb darkening (edges darker than center)
- Granulation (surface texture)
- Color temperature (~5778K yellow-white)
- Prominences (if requested)
- Corona visibility

**Explosion:**
- Brightness (hot core)
- Color gradient (hot→cool)
- Smoke presence
- Mushroom/billow structure
- Dynamic range

**Fire:**
- Flame colors (orange-yellow-white)
- Turbulence (flickering)
- Smoke integration
- Realistic falloff

## SCORING

- 90-100: Production ready
- 70-89: Good, minor issues
- 50-69: Acceptable, needs improvement
- 30-49: Poor, significant issues
- 0-29: Failed, fundamental problems

## OUTPUT

Return QualityAssessment with:
- overall_score: 0-100
- criterion_scores: Dict of per-criterion scores
- issues: List of specific problems found
- recommendations: Actionable improvement suggestions
- comparison_notes: If reference provided, specific differences"""

class QualityAssessment(BaseModel):
    """Structured output from Quality Analyst."""
    overall_score: float = Field(ge=0, le=100)
    criterion_scores: dict = Field(description="Per-criterion scores")
    issues: List[str] = Field(description="Specific problems found")
    recommendations: List[str] = Field(description="Actionable improvements")
    comparison_notes: Optional[str] = Field(default=None)
    confidence: float = Field(ge=0.0, le=1.0)

@function_tool
def compute_image_metrics(image_path: str) -> str:
    """
    Compute programmatic image quality metrics.

    Returns brightness, contrast, color stats, edge density.
    """
    from PIL import Image
    import numpy as np

    path = Path(image_path)
    if not path.exists():
        return json.dumps({"error": f"Image not found: {image_path}"})

    try:
        img = Image.open(path).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0

        # Brightness stats
        luminance = 0.299 * arr[:,:,0] + 0.587 * arr[:,:,1] + 0.114 * arr[:,:,2]

        # Color stats
        r_mean, g_mean, b_mean = arr[:,:,0].mean(), arr[:,:,1].mean(), arr[:,:,2].mean()
        warm_ratio = (r_mean + 0.5 * g_mean) / (b_mean + 0.001)

        # Edge detection (structure)
        from scipy import ndimage
        edges = ndimage.sobel(luminance)
        edge_density = (edges > 0.1).mean()

        return json.dumps({
            "brightness_mean": float(luminance.mean()),
            "brightness_std": float(luminance.std()),
            "brightness_max": float(luminance.max()),
            "dynamic_range": float(luminance.max() - luminance.min()),
            "warm_ratio": float(warm_ratio),
            "color_rgb_means": [float(r_mean), float(g_mean), float(b_mean)],
            "edge_density": float(edge_density),
            "resolution": list(img.size)
        })
    except Exception as e:
        return json.dumps({"error": str(e)})

@function_tool
def compare_images_lpips(image1_path: str, image2_path: str) -> str:
    """
    Compute LPIPS perceptual similarity between two images.

    Lower score = more similar (0 = identical, 1 = very different)
    """
    # Lazy load to avoid startup delay
    import torch
    import lpips
    from PIL import Image
    from torchvision import transforms

    try:
        loss_fn = lpips.LPIPS(net='alex')

        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        img1 = transform(Image.open(image1_path).convert("RGB")).unsqueeze(0)
        img2 = transform(Image.open(image2_path).convert("RGB")).unsqueeze(0)

        with torch.no_grad():
            distance = loss_fn(img1, img2).item()

        return json.dumps({
            "lpips_distance": distance,
            "similarity_percent": (1 - distance) * 100,
            "interpretation": "similar" if distance < 0.3 else "different" if distance > 0.5 else "moderate"
        })
    except Exception as e:
        return json.dumps({"error": str(e)})

def create_quality_analyst_agent() -> Agent:
    """Create Quality Analyst with vision capabilities."""
    return Agent(
        name="Quality Analyst",
        instructions=QUALITY_ANALYST_INSTRUCTIONS,
        model="gpt-5.2",  # Needs vision
        output_type=QualityAssessment,
        tools=[
            load_image_as_base64,  # From vision_expert.py
            compute_image_metrics,
            compare_images_lpips,
            get_quality_criteria  # From vision_expert.py
        ]
    )
```

---

### 5. Learning Agent

**Role:** Accumulates knowledge, identifies patterns, improves system.

**Key Innovation:** Self-improving prompts based on accumulated learnings.

```python
LEARNING_AGENT_INSTRUCTIONS = """You are the learning and knowledge accumulation specialist.

## YOUR ROLE

1. Record outcomes of each iteration (what worked, what didn't)
2. Identify patterns across sessions
3. Build reusable knowledge for future attempts
4. Suggest prompt improvements for other agents

## KNOWLEDGE TYPES

**Parameter Correlations:**
- "Increasing flame_max_temp above 3000K causes color to shift orange→white"
- "turbulence > 0.7 creates unstable simulations"

**Issue→Fix Mappings:**
- "too_dark" → increase emission_intensity, blackbody_intensity
- "lacks_structure" → increase turbulence, vorticity
- "wrong_color" → adjust temperature, check color ramp

**Effect-Specific Learnings:**
- Sun: "Limb darkening requires specific shader setup, not just parameters"
- Explosion: "Mushroom shape needs upward velocity + turbulence balance"

## OUTPUT

Return LearningUpdate with:
- new_learnings: Knowledge gained this session
- pattern_updates: Updates to existing patterns
- prompt_suggestions: Improvements for other agents
- success_rate_update: How well the system is performing"""

class LearningUpdate(BaseModel):
    """Structured output from Learning Agent."""
    new_learnings: List[str] = Field(description="New knowledge gained")
    pattern_updates: dict = Field(description="Updates to existing patterns")
    prompt_suggestions: dict = Field(description="Suggested prompt improvements per agent")
    success_indicators: List[str] = Field(description="What contributed to success")
    failure_indicators: List[str] = Field(description="What contributed to failure")

@function_tool
def load_knowledge_base(effect_type: str = "") -> str:
    """Load accumulated knowledge, optionally filtered by effect type."""
    kb_path = Path("data/knowledge_base.json")
    if not kb_path.exists():
        return json.dumps({"learnings": [], "patterns": {}})

    kb = json.loads(kb_path.read_text())

    if effect_type:
        # Filter to relevant learnings
        kb["learnings"] = [l for l in kb.get("learnings", []) if effect_type in l.get("tags", [])]

    return json.dumps(kb)

@function_tool
def save_learning(
    learning: str,
    category: str,
    effect_types: str,
    confidence: float
) -> str:
    """
    Save a new learning to the knowledge base.

    Args:
        learning: The knowledge to save
        category: parameter_correlation, issue_fix, effect_specific, general
        effect_types: Comma-separated list of relevant effect types
        confidence: How confident we are in this learning (0-1)
    """
    kb_path = Path("data/knowledge_base.json")
    kb_path.parent.mkdir(parents=True, exist_ok=True)

    if kb_path.exists():
        kb = json.loads(kb_path.read_text())
    else:
        kb = {"learnings": [], "patterns": {}}

    kb["learnings"].append({
        "content": learning,
        "category": category,
        "tags": [t.strip() for t in effect_types.split(",")],
        "confidence": confidence,
        "timestamp": datetime.now().isoformat()
    })

    kb_path.write_text(json.dumps(kb, indent=2))
    return json.dumps({"success": True, "total_learnings": len(kb["learnings"])})

@function_tool
def get_relevant_learnings(query: str, effect_type: str = "", limit: int = 10) -> str:
    """
    Search knowledge base for relevant learnings.

    Uses simple keyword matching. Could be upgraded to embeddings.
    """
    kb_path = Path("data/knowledge_base.json")
    if not kb_path.exists():
        return json.dumps({"learnings": []})

    kb = json.loads(kb_path.read_text())

    query_terms = query.lower().split()
    scored = []

    for learning in kb.get("learnings", []):
        content = learning["content"].lower()
        score = sum(1 for term in query_terms if term in content)

        if effect_type and effect_type in learning.get("tags", []):
            score += 2  # Boost for matching effect type

        if score > 0:
            scored.append((score, learning))

    scored.sort(key=lambda x: x[0], reverse=True)
    return json.dumps({"learnings": [l for _, l in scored[:limit]]})

def create_learning_agent() -> Agent:
    """Create Learning Agent."""
    return Agent(
        name="Learning Agent",
        instructions=LEARNING_AGENT_INSTRUCTIONS,
        model="gpt-4o",  # Good reasoning, cheaper than 5.2
        output_type=LearningUpdate,
        tools=[load_knowledge_base, save_learning, get_relevant_learnings]
    )
```

---

## Complete System Assembly

```python
from agents import Agent, Runner, handoff, RunContextWrapper
from agents.mcp import MCPServerStdio
import asyncio

class BlenderAutonomousSystem:
    """
    Complete autonomous VFX asset creation system.

    Single entry point that orchestrates all specialist agents.
    """

    def __init__(self):
        self.mcp_server: Optional[MCPServerStdio] = None
        self.orchestrator: Optional[Agent] = None
        self.initialized = False

    async def initialize(self):
        """Initialize all agents and MCP connections."""

        # 1. Connect to blender-manual MCP
        self.mcp_server = MCPServerStdio(
            name="blender-manual",
            params={
                "command": "python",
                "args": ["agents/blender-manual/blender_server.py"],
            },
            cache_tools_list=True
        )
        await self.mcp_server.connect()

        # 2. Create specialist agents
        script_writer = await create_script_writer_agent(self.mcp_server)
        executor = create_executor_agent()
        quality_analyst = create_quality_analyst_agent()
        learning_agent = create_learning_agent()

        # 3. Create handoffs with filtering and callbacks
        script_writer_handoff = handoff(
            agent=script_writer,
            tool_name_override="write_blender_script",
            tool_description_override="Generate or modify a Blender Python script for VFX",
            input_filter=lambda ctx: {
                "effect_type": ctx.get("effect_type"),
                "description": ctx.get("description"),
                "current_script": ctx.get("current_script", ""),
                "issues_to_fix": ctx.get("issues", []),
                "learnings": ctx.get("relevant_learnings", [])
            }
        )

        executor_handoff = handoff(
            agent=executor,
            tool_name_override="execute_blender",
            tool_description_override="Run a Blender script and capture output",
            input_filter=lambda ctx: {
                "script_path": ctx.get("script_path"),
                "output_dir": ctx.get("output_dir", "")
            }
        )

        quality_handoff = handoff(
            agent=quality_analyst,
            tool_name_override="evaluate_render",
            tool_description_override="Evaluate render quality with vision and metrics",
            input_filter=lambda ctx: {
                "render_path": ctx.get("render_path"),
                "reference_path": ctx.get("reference_path"),
                "effect_type": ctx.get("effect_type"),
                "requirements": ctx.get("description")
            }
        )

        learning_handoff = handoff(
            agent=learning_agent,
            tool_name_override="record_and_learn",
            tool_description_override="Record iteration results and extract learnings",
            input_filter=lambda ctx: {
                "iteration_history": ctx.get("iteration_history", []),
                "final_success": ctx.get("success", False),
                "effect_type": ctx.get("effect_type")
            }
        )

        # 4. Create orchestrator with all handoffs
        self.orchestrator = Agent(
            name="Blender Autonomous Orchestrator",
            instructions=ORCHESTRATOR_INSTRUCTIONS,
            model="gpt-5.2",
            model_settings=ModelSettings(
                reasoning=Reasoning(effort="high"),  # Complex orchestration
            ),
            output_type=AssetResult,
            handoffs=[
                script_writer_handoff,
                executor_handoff,
                quality_handoff,
                learning_handoff
            ],
            tools=[
                load_knowledge_base,
                get_relevant_learnings
            ]
        )

        self.initialized = True

    async def create_asset(
        self,
        asset_name: str,
        effect_type: str,
        description: str,
        reference_path: Optional[str] = None,
        quality_threshold: float = 70.0,
        max_iterations: int = 5
    ) -> AssetResult:
        """
        Create a VFX asset through autonomous iteration.

        This is the main entry point. The orchestrator handles everything.
        """
        if not self.initialized:
            await self.initialize()

        request = AssetRequest(
            asset_name=asset_name,
            effect_type=effect_type,
            description=description,
            reference_path=reference_path,
            quality_threshold=quality_threshold,
            max_iterations=max_iterations
        )

        # Load relevant learnings for context
        learnings_result = await Runner.run(
            self.orchestrator,
            f"Get relevant learnings for {effect_type}: {description}"
        )

        # Run the full autonomous workflow
        result = await Runner.run(
            self.orchestrator,
            f"""Create asset with these requirements:

            Asset Name: {asset_name}
            Effect Type: {effect_type}
            Description: {description}
            Reference: {reference_path or 'None provided'}
            Quality Threshold: {quality_threshold}
            Max Iterations: {max_iterations}

            Relevant past learnings:
            {learnings_result.final_output}

            Begin the create→execute→evaluate→iterate loop."""
        )

        return result.final_output

    async def close(self):
        """Clean up resources."""
        if self.mcp_server:
            await self.mcp_server.cleanup()


# =============================================================================
# MCP SERVER EXPOSURE
# =============================================================================

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("blender-autonomous")
_system: Optional[BlenderAutonomousSystem] = None

@mcp.tool()
async def create_vfx_asset(
    asset_name: str,
    effect_type: str,
    description: str,
    reference_path: str = "",
    quality_threshold: float = 70.0,
    max_iterations: int = 5
) -> str:
    """
    Create a VFX asset through autonomous iteration.

    The system will:
    1. Generate a Blender script
    2. Execute it to create renders/VDBs
    3. Evaluate quality
    4. Iterate until threshold met or max iterations
    5. Return the best result with learnings

    Args:
        asset_name: Name for the asset (used for file paths)
        effect_type: sun, explosion, fire, nebula, smoke
        description: Detailed description of desired effect
        reference_path: Optional reference image for comparison
        quality_threshold: Minimum quality score (0-100, default 70)
        max_iterations: Maximum improvement attempts (default 5)

    Returns:
        JSON with success status, paths, score, and learnings
    """
    global _system
    if _system is None:
        _system = BlenderAutonomousSystem()

    result = await _system.create_asset(
        asset_name=asset_name,
        effect_type=effect_type,
        description=description,
        reference_path=reference_path or None,
        quality_threshold=quality_threshold,
        max_iterations=max_iterations
    )

    return result.model_dump_json(indent=2)

@mcp.tool()
async def get_system_learnings(effect_type: str = "") -> str:
    """
    Get accumulated learnings from past asset creation sessions.

    Args:
        effect_type: Optional filter by effect type

    Returns:
        JSON with learnings, patterns, and statistics
    """
    return load_knowledge_base(effect_type)

if __name__ == "__main__":
    mcp.run()
```

---

## Key Advantages Over Current System

| Aspect | Current | Autonomous |
|--------|---------|------------|
| **Orchestration** | Claude Code manually calls 6 MCPs | Single call triggers full workflow |
| **Context Sharing** | None between servers | Shared state across all agents |
| **Iteration** | Manual, requires user intervention | Autonomous until threshold/max |
| **Learning** | Fragmented across experiment-tracker | Unified knowledge base |
| **Latency** | Many round-trips | Tight internal loops |
| **Self-Improvement** | None | Accumulated learnings improve prompts |
| **Error Handling** | Manual per-server | Coordinated retry/fallback |
| **Cost Control** | Separate budget tracking | Unified budget with smart model selection |

---

## Migration Path

### Phase 1: Core Agents (Week 1)
1. Implement Script Writer with MCP
2. Implement Executor Agent
3. Basic orchestrator without handoffs (agents-as-tools)

### Phase 2: Quality Loop (Week 2)
4. Implement Quality Analyst
5. Add iteration logic to orchestrator
6. Implement handoffs with filtering

### Phase 3: Learning (Week 3)
7. Implement Learning Agent
8. Knowledge base persistence
9. Inject learnings into prompts

### Phase 4: Production (Week 4)
10. MCP server exposure
11. Budget/guardrails integration
12. Testing and optimization

---

## Self-Improvement Mechanisms

### 1. Prompt Evolution
```python
# Learning Agent can suggest prompt improvements
class PromptImprovement(BaseModel):
    agent_name: str
    current_instruction_snippet: str
    suggested_replacement: str
    rationale: str
    confidence: float

# After N successful iterations, review and update prompts
async def evolve_prompts(learning_agent: Agent, session_history: List[dict]):
    result = await Runner.run(
        learning_agent,
        f"""Review these {len(session_history)} sessions and suggest prompt improvements:

        {json.dumps(session_history, indent=2)}

        Focus on:
        - Instructions that led to repeated failures
        - Missing guidance that caused confusion
        - Successful patterns that should be reinforced
        """
    )

    # Apply improvements (with human review)
    for improvement in result.final_output.prompt_suggestions:
        print(f"Suggested improvement for {improvement.agent_name}:")
        print(f"  Current: {improvement.current_instruction_snippet}")
        print(f"  Suggested: {improvement.suggested_replacement}")
        print(f"  Rationale: {improvement.rationale}")
```

### 2. Knowledge Injection
```python
# Before each session, inject relevant learnings into prompts
async def prepare_session(orchestrator: Agent, effect_type: str, description: str):
    # Get relevant learnings
    learnings = await get_relevant_learnings(
        query=description,
        effect_type=effect_type,
        limit=5
    )

    # Inject into orchestrator context
    session_context = f"""
    RELEVANT LEARNINGS FROM PAST SESSIONS:
    {json.dumps(learnings, indent=2)}

    Apply these learnings to avoid past mistakes and replicate past successes.
    """

    return orchestrator.clone(
        instructions=ORCHESTRATOR_INSTRUCTIONS + "\n\n" + session_context
    )
```

### 3. Parameter Pattern Mining
```python
# Learning Agent mines patterns from successful sessions
@function_tool
def mine_parameter_patterns(effect_type: str) -> str:
    """
    Analyze successful sessions to find optimal parameter ranges.

    Returns patterns like:
    - "For sun effects, flame_max_temp in [5000, 6000] works best"
    - "explosion turbulence sweet spot is 0.6-0.8"
    """
    # Load successful sessions for this effect type
    sessions = load_sessions(effect_type=effect_type, min_score=80)

    # Extract parameter distributions
    param_values = defaultdict(list)
    for session in sessions:
        for param, value in session.final_params.items():
            param_values[param].append(value)

    # Compute statistics
    patterns = {}
    for param, values in param_values.items():
        if len(values) >= 3:
            patterns[param] = {
                "optimal_range": [np.percentile(values, 25), np.percentile(values, 75)],
                "mean": np.mean(values),
                "successful_samples": len(values)
            }

    return json.dumps(patterns)
```

---

## Conclusion

This architecture replaces 6 separate MCP servers with a unified autonomous system that:

1. **Handles complete workflows** with a single call
2. **Shares context** across all specialist agents
3. **Iterates autonomously** until quality threshold
4. **Learns and improves** from every session
5. **Self-corrects** through accumulated knowledge

The OpenAI Agents SDK provides all the primitives needed:
- Handoffs for specialist delegation
- Structured outputs for type-safe communication
- MCP integration for documentation access
- Guardrails for validation
- Hooks for observability

Ready to build?
