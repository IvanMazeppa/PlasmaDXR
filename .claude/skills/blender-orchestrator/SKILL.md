---
name: blender-orchestrator
description: Autonomous VFX asset generator using Blender + ML evaluation. Creates NanoVDB volumetric assets (explosions, fire, nebulae, sun effects) through iterative improvement with quality-gated feedback loops.
---

# Blender VFX Orchestrator

Autonomous orchestrator for generating high-quality NanoVDB volumetric assets using Blender and ML-based quality evaluation. Creates VFX assets through iterative improvement until quality thresholds are met.

## When to Use This Skill

Invoke this skill when you need to:
- **Generate VFX assets** (explosions, fire, smoke, nebulae, sun/star effects)
- **Create NanoVDB volumetric data** for use in PlasmaDX-Clean renderer
- **Iterate on asset quality** using ML evaluation (VFX diagnostics, ground truth comparison)
- **Use reference images** to guide asset creation toward a target visual
- **Resume interrupted sessions** to continue asset generation

## Core Capabilities

### 1. Autonomous Asset Generation
- Generates Blender Python scripts using `script-generator` MCP
- Executes simulations using `blender-executor` MCP
- Evaluates quality using `asset-evaluator` MCP (VFX diagnostics, ground truth)
- Iterates until quality threshold met or max iterations reached

### 2. Quality-Gated Workflow
- **VFX Quality Score**: 0-100 composite score (brightness, color, structure, coverage)
- **Ground Truth Comparison**: Distribution-based comparison against real reference footage
- **Temporal Consistency**: Animation smoothness analysis
- **Automatic Diagnosis**: Identifies issues and suggests parameter fixes

### 3. Session Management
- **Persistent Sessions**: Progress saved to `build/orchestrator_state/`
- **Resume Capability**: Continue interrupted sessions from last checkpoint
- **Trust Score**: 0-1 score based on past success, affects autonomy level

## Available MCP Tools

### `get_status`
Get current orchestrator status:
- Trust score (0.0 - 1.0)
- Autonomy level (supervised, guided, autonomous, trusted)
- Token/cost limits
- Active sessions

### `list_sessions`
List VFX asset generation sessions with:
- Session ID
- Asset name
- Status (in_progress, completed, failed, paused)
- Best score achieved
- Iteration count

### `create_asset`
Start a new VFX asset generation:
- **asset_name**: Name for the asset (e.g., "explosion_v1")
- **effect_type**: Type (pyro, explosion, fire, smoke, nebula, sun)
- **description**: What to create
- **reference_path**: Optional reference image for evaluation
- **semantic_query**: Optional text for CLIP evaluation
- **resolution**: Blender simulation resolution (default 96)
- **frame_end**: Animation end frame (default 50)
- **technique_name**: Optional specific technique to use

### `resume_session`
Resume a paused/incomplete session:
- Loads session state (parameters, scores, iteration count)
- Continues iteration loop from checkpoint
- Uses same quality thresholds

## Underlying MCP Agents

The orchestrator coordinates these specialized agents:

**1. script-generator**
- Generates Blender Python scripts from descriptions
- Modifies scripts based on evaluation feedback
- Validates parameters against Blender 5.0 API ranges

**2. blender-executor**
- Executes Blender scripts via CLI
- Captures VDB output and renders
- Parses errors with suggested fixes

**3. asset-evaluator**
- **evaluate_vfx_quality**: Standalone VFX quality (no reference needed)
- **evaluate_ground_truth**: Compare against real footage distributions
- **analyze_temporal_quality**: Animation consistency
- **extract_vfx_diagnostics**: Detailed feature extraction

**4. experiment-tracker**
- Records experiments with outcomes
- Builds knowledge base of what works
- Suggests fixes based on past success

**5. iteration-controller**
- Coordinates the full pipeline
- Diagnoses issues and suggests next steps
- Manages state persistence

## Quality Thresholds

| Metric | Threshold | Description |
|--------|-----------|-------------|
| VFX Quality Score | >= 60 | Composite quality (brightness, color, structure) |
| Temporal Consistency | >= 0.7 | Animation smoothness (1.0 = no flicker) |
| Ground Truth Similarity | >= 0.65 | Distribution match to real footage |

## Example Usage

### Creating an Explosion Asset

**User Request**: "Create a mushroom cloud explosion with orange/red fire"

**Orchestrator Response**:
```
Starting asset generation: mushroom_explosion_v1

**Configuration:**
- Effect Type: pyro (explosion)
- Description: Rising mushroom cloud with orange/red fire
- Resolution: 96
- Frame Range: 1-50

**Iteration 1:**
- Generated script using "rising_mushroom" technique
- Executing Blender simulation...
- Evaluating quality...
  - VFX Score: 45/100 (TOO DARK, LOW STRUCTURE)
  - Issues: Brightness mean 42, needs >100

**Iteration 2:**
- Adjusting parameters: +flame_max_temp, +turbulence
- Executing...
- VFX Score: 68/100 (PASSED)

**Results:**
- Best Score: 68/100
- Total Iterations: 2
- VDB Output: build/vdb_output/mushroom_explosion_v1/
- Render: build/vdb_output/mushroom_explosion_v1/render_0025.png

Quality gate PASSED. Asset ready for use.
```

### Using Reference Images

**User Request**: "Create a sun surface effect that looks like this NASA footage"

**Orchestrator Response**:
```
Starting asset generation: sun_surface_v1

**Reference:** assets/reference_images/star/Eruptions_20241008/frame_00639.jpg

**Iteration 1:**
- VFX Score: 52/100
- Ground Truth: 0.45 (target: 0.65)
- Issues: Color too cool, lacks granulation

**Iteration 2:**
- Increased temperature, adjusted color profile
- Ground Truth: 0.58
- Still needs more surface texture

**Iteration 3:**
- Added noise for granulation effect
- Ground Truth: 0.67 (PASSED)
- VFX Score: 71/100

**Results:**
- Ground Truth Similarity: 0.67 (matches reference)
- 3 iterations to pass
- Asset ready for production use
```

### Resuming Sessions

**User Request**: "Continue the sun_surface_v1 session"

**Orchestrator Response**:
```
Resuming session: sun_surface_v1

**Loaded State:**
- Last Iteration: 3
- Best Score: 67
- Current Issue: Prominence shapes too regular

**Iteration 4:**
- Adjusted prominence turbulence
- VFX Score: 74/100
- Prominence quality improved

Session continued. Ready for next iteration or completion.
```

## Architecture Details

### Trust & Autonomy System
The orchestrator uses a graduated autonomy model:

| Level | Trust Score | Behavior |
|-------|-------------|----------|
| Supervised | 0.0 - 0.3 | Requires approval for each action |
| Guided | 0.3 - 0.6 | Can execute, reports decisions |
| Autonomous | 0.6 - 0.8 | Full autonomy within guardrails |
| Trusted | 0.8 - 1.0 | Extended limits, minimal oversight |

### Guardrails
- **Token Limits**: Per-session (100K), per-iteration (20K), absolute max (150K)
- **Cost Limits**: Per-session ($5), per-day ($20), per-week ($75)
- **Quality Gates**: Must pass thresholds before completion

### Session Persistence
Sessions are saved to `build/orchestrator_state/<session_id>.json` with:
- Current iteration and best score
- Parameters and script versions
- Evaluation history
- Next action to take

## Best Practices

1. **Start with Description**: Detailed descriptions produce better initial scripts
2. **Use References**: Ground truth comparison dramatically improves realism
3. **Iterate Patiently**: Complex effects may need 5+ iterations
4. **Review Diagnostics**: VFX quality issues are specific and actionable
5. **Resume Don't Restart**: Use `resume_session` to continue from checkpoints
6. **Trust the Scores**: ML evaluation correlates well with human perception

## Output Locations

| Asset | Path |
|-------|------|
| VDB Files | `build/vdb_output/<asset_name>/` |
| Renders | `build/vdb_output/<asset_name>/render_*.png` |
| Scripts | `assets/blender_scripts/generated/<asset_name>.py` |
| Sessions | `build/orchestrator_state/<session_id>.json` |

## Integration with PlasmaDX-Clean

Generated NanoVDB assets can be loaded into PlasmaDX-Clean renderer:
1. VDB files exported to `build/vdb_output/`
2. NanoVDB system loads and renders volumetric data
3. Integrates with existing RT lighting and particle systems

---

**Remember**: The orchestrator works autonomously but respects quality gates. It will iterate until the asset meets thresholds or max iterations are reached. For best results, provide detailed descriptions and reference images when available.
