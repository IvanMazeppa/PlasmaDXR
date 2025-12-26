# Self-Improving NanoVDB Asset Generation Pipeline

**Goal:** Create a self-improving system that can research techniques, generate Blender scripts, execute them via CLI, evaluate results against reference images, and iterate until quality thresholds are met.

**Target Assets:** NanoVDB volumetric assets from Blender physics simulations:
- Pyro effects (explosions, fire, smoke)
- Liquid simulations (splashes, waves, droplets)
- Atmospheric phenomena (clouds, fog, storms)
- Celestial objects (nebulae, stars, accretion disks)

**Inspiration:** EmberGen-style high-quality volumetric assets for real-time RT rendering.

---

## Architecture Overview

### Generate-Evaluate-Improve Loop

```
┌─────────────────────────────────────────────────────────────────┐
│                    ITERATION CONTROLLER                         │
│              (Coordinates the improvement loop)                 │
└─────────────┬───────────────────────────────────────┬───────────┘
              │                                       │
              ▼                                       ▼
┌─────────────────────────┐             ┌─────────────────────────┐
│   1. RESEARCH PHASE     │             │   4. LEARNING PHASE     │
│ ─────────────────────── │             │ ─────────────────────── │
│ • WebSearch (tutorials) │             │ • Error pattern DB      │
│ • blender-manual MCP    │◄────────────│ • Working script lib    │
│ • context7 (bpy docs)   │   feedback  │ • Technique memory      │
└───────────┬─────────────┘             └─────────────────────────┘
            │                                       ▲
            ▼                                       │
┌─────────────────────────┐             ┌───────────┴─────────────┐
│   2. GENERATE PHASE     │             │   3. EVALUATE PHASE     │
│ ─────────────────────── │             │ ─────────────────────── │
│ • Script generation     │             │ • Render VDB asset      │
│ • CLI execution         │────────────►│ • Compare to reference  │
│ • Error capture/parse   │             │ • LPIPS similarity      │
└─────────────────────────┘             │ • Claude vision assess  │
                                        └─────────────────────────┘
```

---

## Phase 1: Research

### Purpose
Gather techniques and code patterns before generating scripts. This prevents the AI from relying on stale training data.

### Available Tools

| Tool | Source | Purpose |
|------|--------|---------|
| `WebSearch` | Claude Code | Find tutorials, techniques, EmberGen references |
| `blender-manual` MCP | 12 tools | Official Blender docs, API reference |
| `context7` MCP | Library docs | bpy Python API documentation |

### Research Workflow

```
User: "Create an explosion VDB like EmberGen"
  ↓
1. WebSearch: "blender mantaflow explosion tutorial 2024"
2. WebSearch: "embergen style explosion blender volumetrics"
3. WebSearch: "blender 5 pyro simulation openvdb export"
4. blender-manual: search_tutorials("volumetrics", "explosion")
5. blender-manual: search_vdb_workflow("export openvdb mantaflow")
6. context7: get-library-docs("/blender/blender") topic="fluid simulation"
  ↓
Output: Technique summary with verified code patterns
```

### Research Output Format

```json
{
  "asset_type": "explosion",
  "techniques_found": [
    {
      "source": "BlenderGuru tutorial",
      "method": "Mantaflow GAS domain with quick smoke",
      "key_settings": {
        "resolution_max": 128,
        "domain_type": "GAS",
        "use_adaptive_domain": true
      }
    }
  ],
  "api_calls_verified": [
    "bpy.ops.fluid.bake_all()",
    "domain_settings.cache_data_format = 'OPENVDB'"
  ],
  "reference_images": [
    "https://example.com/embergen_explosion.jpg"
  ]
}
```

---

## Phase 2: Generate

### Purpose
Create Blender Python scripts based on researched techniques, then execute via CLI.

### Existing Infrastructure

| Component | Location | Status |
|-----------|----------|--------|
| CLI Runner | `assets/blender_scripts/GPT-5.2/run_blender_cli.sh` | ✅ Working |
| Template Scripts | `assets/blender_scripts/GPT-5.2/*.py` | ✅ Working |
| Agent Prompt | `agents/blender-scripting/AGENT_PROMPT.md` | ✅ Use for generation |

### Script Generation Guidelines

From `AGENT_PROMPT.md`:
1. **MCP-First Verification** - Query blender-manual before writing any bpy code
2. **Context-Aware** - Set active object, mode, selection before operators
3. **Error Handling** - Check objects exist, catch exceptions
4. **Logging** - Print progress for debugging
5. **Parameterized** - Accept CLI arguments, don't hardcode values

### Execution Flow

```bash
# Execute with full logging
./assets/blender_scripts/GPT-5.2/run_blender_cli.sh \
  generated_script.py -- \
  --output_dir "build/vdb_output/explosion_v1" \
  --resolution 128 \
  --frame_end 60 \
  --bake 1 \
  --render_still 1

# Outputs:
# - build/blender_cli_logs/<timestamp>/stdout_stderr.txt
# - build/blender_cli_logs/<timestamp>/blender.log
# - build/vdb_output/explosion_v1/*.vdb
# - build/vdb_output/explosion_v1/render_*.png
```

### New Component: `blender-executor` MCP Server

```python
# agents/blender-executor/server.py

@mcp.tool()
async def execute_blender_script(
    script_path: str,
    args: dict,
    timeout_seconds: int = 600
) -> ExecutionResult:
    """
    Execute Blender script via CLI with full capture.

    Returns:
        ExecutionResult:
            success: bool
            exit_code: int
            stdout: str
            stderr: str
            output_dir: str
            vdb_files: list[str]
            render_files: list[str]
            errors_parsed: list[BlenderError]
            duration_seconds: float
    """

@mcp.tool()
async def parse_blender_errors(stderr: str) -> list[BlenderError]:
    """
    Parse Blender/Python errors into structured format.

    Returns list of:
        BlenderError:
            type: "PYTHON" | "BLENDER" | "CONTEXT"
            message: str
            file: str
            line: int
            suggested_fix: str
    """

@mcp.tool()
async def list_run_outputs(run_dir: str) -> RunOutputs:
    """List all outputs from a Blender CLI run."""
```

---

## Phase 3: Evaluate

### Purpose
Render the generated VDB and compare to reference images. This is the critical feedback mechanism.

### Evaluation Methods

#### 1. LPIPS Perceptual Similarity (Existing)
- **Tool:** `dxr-image-quality-analyst` (repurposed)
- **What:** Compares rendered VDB to reference image
- **Output:** Score 0-1 (>0.85 = visually similar)
- **Accuracy:** ~92% correlation with human judgment

#### 2. CLIP Semantic Similarity (New)
- **What:** "Does this render look like an explosion?"
- **How:** Compare embedding of render vs text description
- **Output:** Score 0-1 (>0.7 = semantically matches)
- **Advantage:** Works without reference images

#### 3. Claude Vision Analysis (Existing)
- **What:** Qualitative assessment via multimodal
- **How:** Show render + reference, ask for comparison
- **Output:** Improvement suggestions in natural language

### New Component: `asset-evaluator` MCP Server

```python
# agents/asset-evaluator/server.py

@mcp.tool()
async def render_vdb_preview(
    vdb_path: str,
    output_dir: str,
    angles: list[tuple] = [(0, 0, 0), (0, 90, 0), (90, 0, 0)],
    resolution: tuple = (512, 512)
) -> list[str]:
    """
    Render VDB from multiple angles for evaluation.
    Uses Blender Cycles/EEVEE with volume shader.

    Returns: List of render file paths
    """

@mcp.tool()
async def compare_to_reference(
    renders: list[str],
    references: list[str],
    method: str = "lpips"  # "lpips" | "clip" | "both"
) -> ComparisonResult:
    """
    Compare renders to reference images.

    Returns:
        ComparisonResult:
            lpips_score: float  # 0-1, higher = more similar
            clip_score: float   # 0-1, semantic similarity
            best_match_angle: int
            per_angle_scores: list[float]
    """

@mcp.tool()
async def semantic_match(
    render_path: str,
    description: str
) -> float:
    """
    Check if render matches semantic description.
    "Does this look like a nebula?" → 0.87

    Uses CLIP embeddings for comparison.
    """

@mcp.tool()
async def vision_assess(
    render_path: str,
    reference_path: str,
    target_description: str
) -> VisionAssessment:
    """
    Get Claude vision analysis of render vs reference.

    Returns:
        VisionAssessment:
            quality_rating: float  # 1-10
            matches_reference: bool
            differences: list[str]
            improvement_suggestions: list[str]
            specific_fixes: list[dict]  # {area, problem, fix}
    """
```

### Evaluation Workflow

```python
async def evaluate_generated_asset(
    vdb_path: str,
    reference_images: list[str],
    target_description: str
) -> EvaluationResult:

    # 1. Render VDB from multiple angles
    renders = await render_vdb_preview(
        vdb_path=vdb_path,
        angles=[(0,0,0), (0,90,0), (0,180,0), (90,0,0)]
    )

    # 2. Quantitative comparison (LPIPS + CLIP)
    comparison = await compare_to_reference(
        renders=renders,
        references=reference_images,
        method="both"
    )

    # 3. Qualitative assessment (Claude Vision)
    assessment = await vision_assess(
        render_path=renders[comparison.best_match_angle],
        reference_path=reference_images[0],
        target_description=target_description
    )

    # 4. Combine into actionable feedback
    return EvaluationResult(
        passed=comparison.lpips_score >= 0.80,
        lpips_score=comparison.lpips_score,
        clip_score=comparison.clip_score,
        vision_rating=assessment.quality_rating,
        suggestions=assessment.improvement_suggestions,
        specific_fixes=assessment.specific_fixes
    )
```

---

## Phase 4: Iterate

### Purpose
Feed evaluation results back into script generation. Learn from errors. Converge on quality.

### Iteration Controller Logic

```python
MAX_ITERATIONS = 5
QUALITY_THRESHOLD = 0.80

async def create_asset_with_iteration(
    target: str,                    # "explosion like EmberGen"
    reference_images: list[str],    # Reference image URLs/paths
    asset_type: str                 # "pyro" | "liquid" | "atmospheric" | "celestial"
) -> FinalAsset:

    feedback_history = []
    best_result = None
    best_score = 0.0

    # Initial research
    techniques = await research_phase(target, asset_type)

    for iteration in range(MAX_ITERATIONS):
        print(f"[Iteration {iteration + 1}/{MAX_ITERATIONS}]")

        # Generate script with accumulated knowledge
        script = await generate_script(
            target=target,
            asset_type=asset_type,
            techniques=techniques,
            previous_feedback=feedback_history,
            iteration=iteration
        )

        # Execute
        result = await execute_blender_script(script)

        if not result.success:
            # Parse and learn from error
            error_analysis = await analyze_execution_error(result)
            techniques.add_fix(error_analysis)
            feedback_history.append({
                "type": "error",
                "iteration": iteration,
                "error": error_analysis.message,
                "fix_applied": error_analysis.suggested_fix
            })
            continue

        # Evaluate
        evaluation = await evaluate_generated_asset(
            vdb_path=result.vdb_files[0],
            reference_images=reference_images,
            target_description=target
        )

        # Track best result
        if evaluation.lpips_score > best_score:
            best_score = evaluation.lpips_score
            best_result = result

        # Check if passed threshold
        if evaluation.passed:
            print(f"[SUCCESS] Quality threshold met: {evaluation.lpips_score:.2f}")
            return FinalAsset(
                vdb_path=result.vdb_files[0],
                renders=result.render_files,
                quality_score=evaluation.lpips_score,
                iterations=iteration + 1
            )

        # Collect feedback for next iteration
        feedback_history.append({
            "type": "evaluation",
            "iteration": iteration,
            "score": evaluation.lpips_score,
            "suggestions": evaluation.suggestions,
            "specific_fixes": evaluation.specific_fixes
        })

        # Update techniques based on feedback
        techniques.incorporate_feedback(evaluation)

    # Return best attempt after max iterations
    print(f"[MAX ITERATIONS] Best score: {best_score:.2f}")
    return FinalAsset(
        vdb_path=best_result.vdb_files[0],
        renders=best_result.render_files,
        quality_score=best_score,
        iterations=MAX_ITERATIONS,
        converged=False
    )
```

### Learning System

```python
class TechniqueMemory:
    """Persistent storage of learned patterns."""

    def __init__(self, db_path: str):
        self.db = sqlite3.connect(db_path)
        self._init_tables()

    def store_successful_pattern(
        self,
        asset_type: str,
        script_template: str,
        settings: dict,
        quality_score: float
    ):
        """Store a pattern that worked well."""

    def retrieve_patterns(
        self,
        asset_type: str,
        min_quality: float = 0.75
    ) -> list[Pattern]:
        """Get patterns that worked for this asset type."""

    def store_error_fix(
        self,
        error_type: str,
        error_message: str,
        fix_applied: str,
        success: bool
    ):
        """Learn from errors and their fixes."""

    def get_fix_for_error(
        self,
        error_message: str
    ) -> Optional[str]:
        """Retrieve known fix for similar error."""
```

---

## Asset Type Specifications

### Pyro Effects (Explosions, Fire, Smoke)

**Blender Approach:** Mantaflow GAS domain

**Key Settings:**
```python
domain_settings.domain_type = 'GAS'
domain_settings.use_noise = True  # For detail
domain_settings.noise_scale = 2
domain_settings.vorticity = 0.5
flow_settings.flow_type = 'FIRE'  # or 'SMOKE', 'BOTH'
```

**VDB Grids to Export:**
- `density` - Smoke/dust density
- `flame` - Fire intensity
- `temperature` - Heat distribution
- `velocity` - Motion vectors (for motion blur)

**Reference Sources:**
- EmberGen explosion presets
- Houdini pyro tutorials
- Real explosion footage

### Liquid Simulations

**Blender Approach:** Mantaflow LIQUID domain

**Key Settings:**
```python
domain_settings.domain_type = 'LIQUID'
domain_settings.use_spray_particles = True
domain_settings.use_foam_particles = True
domain_settings.use_bubble_particles = True
```

**VDB Grids to Export:**
- `density` - Liquid surface
- `velocity` - Flow direction

**Reference Sources:**
- FlipFluids plugin examples
- Houdini FLIP tutorials
- Slow-motion water footage

### Atmospheric Phenomena

**Blender Approach:** Geometry Nodes + Volume Shader OR Mantaflow

**Methods:**
1. **Geometry Nodes:** Procedural cloud generation
2. **Mantaflow:** Simulated cloud dynamics
3. **Volume Shader:** Noise-based procedural

**VDB Grids:**
- `density` - Cloud density
- `temperature` (optional) - For storm clouds

**Reference Sources:**
- NASA cloud imagery
- Weather photography
- Blender cloud tutorials

### Celestial Objects

**Blender Approach:** Mantaflow + Emission shaders

**Types:**
- Nebulae (emission + absorption)
- Stars (high emission, low density)
- Accretion disks (velocity-based distortion)

**Key Settings:**
```python
# For nebulae
material.volume.emission_strength = 5.0
material.volume.emission_color = (0.8, 0.2, 0.5)  # Pink/magenta
material.volume.density = 0.1  # Wispy
```

**VDB Grids:**
- `density` - Gas distribution
- `temperature` - Color temperature mapping
- `emission` - Self-illumination intensity

**Reference Sources:**
- Hubble/JWST imagery
- NASA nebula catalogs
- Space art references

---

## Implementation Roadmap

### Week 1: Foundation

| Day | Task | Hours |
|-----|------|-------|
| 1 | Create `blender-executor` MCP server skeleton | 2 |
| 1 | Implement `execute_blender_script` tool | 2 |
| 2 | Implement `parse_blender_errors` tool | 2 |
| 2 | Test with existing GPT-5.2 scripts | 1 |
| 3 | Create `asset-evaluator` MCP server skeleton | 2 |
| 3 | Implement `render_vdb_preview` tool | 3 |
| 4 | Implement `compare_to_reference` (LPIPS) | 2 |
| 4 | Test evaluation with manual VDB | 1 |
| 5 | Wire up basic iteration loop (manual) | 3 |

### Week 2: Intelligence

| Day | Task | Hours |
|-----|------|-------|
| 1 | Add CLIP semantic similarity | 3 |
| 2 | Implement `vision_assess` tool | 2 |
| 2 | Create research workflow with WebSearch | 2 |
| 3 | Build TechniqueMemory database | 3 |
| 4 | Implement feedback incorporation | 3 |
| 5 | End-to-end test: "Create explosion" | 4 |

### Week 3: Refinement

| Day | Task | Hours |
|-----|------|-------|
| 1-2 | Create pyro effect templates | 4 |
| 3 | Create liquid simulation templates | 3 |
| 4 | Create atmospheric templates | 3 |
| 5 | Create celestial templates | 3 |
| 5 | Documentation and testing | 2 |

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| First-try success rate | 30% | Scripts run without error on first attempt |
| Iteration convergence | 80% | Assets reach quality threshold within 5 iterations |
| Quality score average | >0.75 LPIPS | Perceptual similarity to references |
| Error recovery rate | 70% | Known errors auto-fixed by learning system |
| Asset creation time | <30 min | From prompt to usable VDB |

---

## File Locations

```
PlasmaDXR/
├── agents/
│   ├── blender-executor/        # NEW: Script execution
│   │   ├── server.py
│   │   └── requirements.txt
│   ├── asset-evaluator/         # NEW: VDB evaluation
│   │   ├── server.py
│   │   └── requirements.txt
│   ├── blender-manual/          # KEEP: Documentation lookup
│   └── blender-scripting/       # KEEP: Agent prompt
│
├── assets/
│   └── blender_scripts/
│       └── GPT-5.2/             # Template scripts
│           ├── run_blender_cli.sh
│           ├── templates/       # NEW: Reusable templates
│           │   ├── pyro_base.py
│           │   ├── liquid_base.py
│           │   ├── atmospheric_base.py
│           │   └── celestial_base.py
│           └── generated/       # NEW: AI-generated scripts
│
├── build/
│   ├── blender_cli_logs/        # Execution logs
│   ├── vdb_output/              # Generated VDB files
│   └── evaluation/              # Renders and comparisons
│
└── data/
    └── technique_memory.db      # NEW: Learning database
```

---

## Dependencies

### Python Packages

```
# agents/blender-executor/requirements.txt
mcp>=1.0.0
python-dotenv>=1.0.0

# agents/asset-evaluator/requirements.txt
mcp>=1.0.0
python-dotenv>=1.0.0
torch>=2.0.0
lpips>=0.1.4
transformers>=4.35.0  # For CLIP
Pillow>=10.0.0
numpy>=1.24.0
```

### External Tools

- Blender 5.0+ (with Python 3.11+)
- nanovdb_convert (VDB → NanoVDB)

---

**Last Updated:** 2025-12-24
**Status:** Design complete, implementation pending
**Owner:** Ben + Claude Code
