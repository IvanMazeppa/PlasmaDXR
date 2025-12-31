# Blender VFX Asset Pipeline - Status Report

**Last Updated:** 2025-12-31
**Pipeline Version:** 0.2.0 (Experimental)
**Blender Version:** 5.0.1

---

## Executive Summary

The Blender VFX pipeline is an experimental multi-agent system designed to autonomously generate NanoVDB volumetric assets for PlasmaDX's real-time ray tracing renderer. The system combines Blender 5.0's Mantaflow fluid simulation with ML-based quality evaluation to iteratively refine pyrotechnic and celestial effects.

**Current Status:** Functional but requires significant refinement. Successfully generates VDB sequences but ML evaluation metrics (LPIPS, CLIP) don't correlate well with volumetric VFX quality. Lacks proper orchestration and knowledge persistence.

---

## 1. Purpose and Objectives

### Primary Goal
Autonomously produce high-quality NanoVDB volumetric assets for the PlasmaDX ray tracing engine, including:
- Stellar phenomena (sun surfaces, supernovae, stellar flares)
- Pyrotechnic effects (explosions, fire, smoke)
- Nebulae and gas clouds
- Liquid simulations (water, lava)

### Design Philosophy
1. **Autonomous Research** - Agents should consult documentation, search for techniques, and learn from experiments
2. **Iterative Refinement** - Generate, evaluate, improve in automated loops
3. **Knowledge Persistence** - Store successful techniques and avoid repeating failures
4. **Human-in-the-Loop Optional** - Run unattended overnight or with supervision

### End-to-End Vision
```
[Research] → [Generate Script] → [Execute Blender] → [Evaluate Quality] → [Iterate/Approve] → [Export to PlasmaDX]
```

---

## 2. System Architecture

### 2.1 Agent Overview

The pipeline consists of 6 specialized MCP servers:

| Agent | Purpose | Status |
|-------|---------|--------|
| **script-generator** | Creates Blender Python scripts from descriptions | Operational |
| **blender-executor** | Runs Blender CLI with timeout/error handling | Operational |
| **asset-evaluator** | LPIPS/CLIP quality scoring, VLM diagnostics | Partially Working |
| **iteration-controller** | Orchestrates generate→evaluate→improve loops | Basic |
| **experiment-tracker** | Stores learnings, warns about past failures | Minimal |
| **context7** | External documentation lookup | Operational |

### 2.2 Data Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        ORCHESTRATOR (Missing)                        │
│   Should: Plan tasks, delegate to agents, manage state, decide next  │
└─────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────────┐
        │                           │                               │
        ▼                           ▼                               ▼
┌───────────────┐          ┌───────────────┐              ┌───────────────┐
│script-generator│          │blender-executor│              │asset-evaluator │
│               │          │               │              │               │
│ - Templates   │          │ - CLI wrapper │              │ - LPIPS       │
│ - Techniques  │   ───▶   │ - Error parse │     ───▶     │ - CLIP        │
│ - Validation  │          │ - Output list │              │ - Temporal    │
└───────────────┘          └───────────────┘              └───────────────┘
        │                           │                               │
        │                           │                               │
        ▼                           ▼                               ▼
┌───────────────┐          ┌───────────────┐              ┌───────────────┐
│iteration-ctrl │          │experiment-    │              │ context7      │
│               │          │  tracker      │              │               │
│ - Loop logic  │   ◀───   │ - Learnings   │     ◀───     │ - Blender docs│
│ - Thresholds  │          │ - Warnings    │              │ - API refs    │
└───────────────┘          └───────────────┘              └───────────────┘
```

### 2.3 File Locations

```
agents/
├── script-generator/      # Script generation MCP server
├── blender-executor/      # Blender CLI execution
├── asset-evaluator/       # ML quality evaluation
├── iteration-controller/  # Loop orchestration
└── experiment-tracker/    # Knowledge persistence

assets/
├── blender_scripts/
│   ├── GPT-5.2/          # Template scripts
│   └── generated/        # Auto-generated scripts

build/
└── vdb_output/           # Baked VDB sequences and renders
    ├── sun_v6_proper_test/
    ├── sun_v11_material_test/
    └── [experiment_name]/
```

---

## 3. Agent Deep Dive

### 3.1 script-generator

**Purpose:** Generate Blender Python scripts from natural language descriptions.

**Tools:**
- `generate_script` - Create new script from effect description
- `modify_script` - Adjust parameters in existing script
- `list_templates` - Available base templates
- `list_techniques` - Technique catalog for variety
- `validate_parameters` - Check against Blender API ranges
- `get_parameter_ranges` - All documented fluid sim parameters

**Technique Catalog:**
The agent maintains a catalog of distinct pyro techniques to ensure variety:
- `rising_mushroom` - Classic rising mushroom cloud
- `ground_burst` - Outward spreading explosion
- `fuel_burst` - High flame, rapid burnout
- `rolling_fire` - Turbulent rolling flames
- `smoke_pillar` - Dense vertical smoke column

**Current Issues:**
1. Templates were designed for Blender 4.x, need updating for 5.0.1 API changes
2. Cache frame range often not set (defaults to 250 frames)
3. Material node graphs hardcoded rather than parameterized
4. No validation of generated node connections

### 3.2 blender-executor

**Purpose:** Execute Blender scripts via CLI with proper error handling.

**Tools:**
- `execute_blender_script` - Run script with timeout
- `parse_blender_errors` - Structure error output
- `list_run_outputs` - Find VDB/render outputs
- `get_latest_run` - Most recent execution
- `list_available_scripts` - Scripts in project

**Configuration:**
```python
BLENDER_EXECUTABLE = "/home/maz3ppa/.local/share/Steam/steamapps/common/Blender/blender"
DEFAULT_TIMEOUT = 600  # 10 minutes
```

**Current Issues:**
1. **Timeout doesn't kill Blender** - Only kills the executor process, Blender continues running
2. **WSL memory thrashing** - vmmemWSL can spike to 80%+ CPU, causing false timeouts
3. **No progress reporting** - Can't tell if bake is 10% or 90% complete
4. **Error parsing incomplete** - Many Blender 5.0 errors not recognized

### 3.3 asset-evaluator

**Purpose:** Evaluate render quality using ML metrics.

**Tools:**
- `compare_lpips` - Perceptual similarity (lower is better, threshold: 0.35)
- `compare_clip` - Semantic similarity (higher is better, threshold: 0.60)
- `evaluate_render` - Combined LPIPS + CLIP evaluation
- `enhanced_evaluate` - Adds aesthetic scoring + VLM diagnostics
- `multi_prompt_clip_analysis` - Graduated quality prompts
- `analyze_temporal_quality` - Detect flickering across frames
- `get_alternative_approaches` - Suggest changes when stuck

**Critical Problem: Metrics Don't Work for VFX**

The LPIPS and CLIP metrics were designed for natural images and don't evaluate volumetric VFX well:

| Issue | Impact |
|-------|--------|
| **Reference mismatch** | Comparing volumetric sun to photo of real sun - fundamentally different |
| **CLIP plateau** | All "sun-like" images score similarly (0.65-0.70), no gradient signal |
| **LPIPS inversions** | Better-looking renders sometimes score worse |
| **No structure awareness** | Can't detect visible turbulence vs flat sphere |

**Example from Sun iterations:**
```
v7 (RED sphere): LPIPS 0.744, CLIP 0.659 - Visible structure, wrong color
v11 (ORANGE sphere): LPIPS 0.738, CLIP 0.671 - Better color, best so far
v8 (PINK sphere): LPIPS 0.759, CLIP 0.660 - Worse overall

All far above 0.35 threshold despite v11 being visually acceptable!
```

### 3.4 iteration-controller

**Purpose:** Orchestrate generate → execute → evaluate → improve loops.

**Tools:**
- `create_asset` - Full pipeline with max iterations
- `run_iteration` - Single iteration step
- `get_history` - Iteration history for asset
- `list_sessions` - All asset sessions
- `compare_iterations` - Score comparison

**Current Issues:**
1. **No intelligent decision making** - Just runs fixed number of iterations
2. **Doesn't analyze evaluation results** - Passes scores but doesn't interpret
3. **No early stopping** - Continues even if quality plateaus
4. **Missing orchestration** - Doesn't coordinate between agents

### 3.5 experiment-tracker

**Purpose:** Persist learnings across sessions.

**Tools:**
- `start_experiment_session` - Begin tracking
- `record_baseline` - Store starting state
- `record_experiment_result` - Log what happened
- `get_warnings_before_change` - Check past failures
- `suggest_experiments` - Recommendations based on history
- `query_knowledge_base` - Search learnings
- `add_manual_learning` - Store insight manually

**Current Issues:**
1. **Barely used** - Other agents don't query it
2. **No structured categories** - Learnings not organized by effect type
3. **Missing integration** - script-generator doesn't check before generating
4. **No automatic learning** - Must manually record insights

---

## 4. Current Problems

### 4.1 Critical Issues

#### P1: ML Evaluation Metrics Don't Work for Volumetric VFX
**Problem:** LPIPS and CLIP were designed for natural images, not procedurally generated volumetric effects.

**Evidence:**
- Sun renders at LPIPS 0.73-0.76 despite visible quality differences
- CLIP scores plateau at 0.65-0.67 regardless of color accuracy
- No correlation between metric scores and human visual assessment

**Impact:** Cannot automate quality gates, requires human review for every iteration.

**Potential Solutions:**
1. **Custom VFX quality model** - Train on volumetric effects with human ratings
2. **Feature-based evaluation** - Check specific properties (color histogram, edge density, temporal stability)
3. **VLM with structured prompts** - GPT-4V/Gemini with specific quality criteria
4. **Hybrid approach** - Basic metrics + human spot-checks

#### P2: No Orchestrator Agent
**Problem:** Claude Code acts as ad-hoc orchestrator, but there's no dedicated agent managing the pipeline.

**Impact:**
- No persistent state across sessions
- Manual intervention required between steps
- No parallel execution of independent tasks
- Lost context when conversation hits token limit

**Needed Capabilities:**
- Task planning and decomposition
- Agent delegation and coordination
- State machine for pipeline stages
- Automatic retry and error recovery
- Progress reporting and logging

#### P3: Blender 5.0.1 API Compatibility
**Problem:** Templates and techniques assume Blender 4.x API.

**Known Issues:**
- `openvdb_cache_compress_type` → `cache_compress_type`
- Some fluid modifier attributes renamed
- Material node behavior changes
- Volume Info temperature mapping different

**Solution:** Audit all templates against Blender 5.0.1 Python API documentation.

### 4.2 Major Issues

#### P4: Cache Frame Range Not Controlled
**Problem:** Scripts set `scene.frame_end` but not domain `cache_frame_start/end`.

**Impact:** Every bake runs 250 frames even for quick tests.

**Fix:** Add to all scripts:
```python
domain.modifiers['Fluid'].domain_settings.cache_frame_start = 1
domain.modifiers['Fluid'].domain_settings.cache_frame_end = frame_count
```

#### P5: Timeout Doesn't Stop Blender
**Problem:** MCP executor timeout kills Python process but Blender continues.

**Impact:** Zombie Blender processes consume resources, subsequent runs may conflict.

**Fix:** Use process groups or explicit kill:
```python
import os
import signal
os.killpg(os.getpgid(process.pid), signal.SIGTERM)
```

#### P6: No Progress Reporting
**Problem:** Can't tell bake progress during execution.

**Impact:** Unknown if 10% or 90% complete, difficult to estimate timeouts.

**Possible Solutions:**
1. Parse Blender stdout for frame numbers
2. Use Blender's Python API to report progress
3. Monitor VDB file creation timestamps

### 4.3 Minor Issues

- **Template duplication** - Similar templates with minor variations
- **No asset versioning** - Hard to track which iteration produced which VDB
- **Missing cleanup** - Old test outputs accumulate
- **Error messages unhelpful** - Generic "bake failed" without details

---

## 5. Lessons Learned (Sun Surface Iterations)

### What Works

1. **Color Ramp > Blackbody for Color Control**
   - Volume Info Temperature → Blackbody produces inconsistent colors
   - Flame → Color Ramp gives direct artistic control
   - Can define exact orange→yellow→white gradient

2. **Low Emission Values**
   - Blackbody Intensity 8.0 causes white overexposure
   - 0.5-1.0 base emission with 1.5-2.0 flame multiplier works
   - Filmic + High Contrast view transform helps

3. **Volume Absorption Adds Depth**
   - Pure emission looks flat
   - Adding orange-tinted absorption creates realistic depth
   - Density scaling 4.0-5.0 for visible structure

4. **Mantaflow Generates Good Data**
   - The fluid simulation itself works well
   - Problem is material/shader interpretation
   - VDB contains proper Density, Flame, Temperature channels

### What Doesn't Work

1. **Temperature → Blackbody Direct Mapping**
   - Volume Info Temperature is normalized 0-1
   - Remapping to Kelvin produces unexpected results
   - Red instead of orange-yellow even at 5000-6000K

2. **Automated LPIPS/CLIP Evaluation**
   - Metrics designed for photos, not VFX
   - No correlation with visual quality
   - Cannot automate quality decisions

3. **Long Bake + Quick Material Iteration**
   - 250 frame bakes take 15-20 minutes
   - Material changes only need 3-5 test frames
   - Should separate bake from material testing

---

## 6. Improvement Roadmap

### Phase 1: Stabilization (Immediate)

1. **Fix cache frame range** - All scripts set explicit cache range
2. **Update for Blender 5.0.1** - Audit templates against current API
3. **Proper process termination** - Kill Blender on timeout
4. **Basic progress reporting** - Parse stdout for frame numbers

### Phase 2: Better Evaluation (Short-term)

1. **Feature-based VFX metrics**
   - Color histogram analysis (is it orange? yellow?)
   - Edge/gradient detection (visible structure?)
   - Temporal difference analysis (stable? flickering?)

2. **VLM structured evaluation**
   - Use GPT-4V or Gemini with specific prompts
   - "Rate the solar corona visibility 1-5"
   - "Is the color temperature appropriate for a star?"

3. **A/B comparison interface**
   - Side-by-side render comparison
   - Human preference recording
   - Build training dataset for custom model

### Phase 3: Orchestration (Medium-term)

1. **Dedicated orchestrator agent**
   - State machine for pipeline stages
   - Task queue with priorities
   - Automatic retry with backoff
   - Progress persistence

2. **Parallel execution**
   - Multiple material tests simultaneously
   - Background baking while evaluating
   - Distributed across machines (future)

3. **Smart iteration decisions**
   - Analyze evaluation results
   - Decide next parameter changes
   - Early stopping when plateaued
   - Alternative approach suggestions

### Phase 4: Knowledge System (Long-term)

1. **Structured knowledge base**
   - Per-effect-type learnings
   - Parameter sensitivity data
   - Failure mode catalog

2. **Automatic learning extraction**
   - Detect successful patterns
   - Record failure signatures
   - Build recommendation engine

3. **Cross-session continuity**
   - Resume experiments after restart
   - Share learnings across projects
   - Version control for techniques

---

## 7. Recommended Next Steps

### Immediate (This Week)

1. **Run v12 test** - Complete current sun iteration
2. **Fix cache frame range** - Update sun_surface_v6.py template
3. **Save v11 learnings** - Record Color Ramp technique in experiment-tracker

### Short-term (Next Sprint)

1. **Implement feature-based evaluation**
   - Color histogram for sun (should peak in orange-yellow)
   - Structure detection (should have visible turbulence)

2. **Create orchestrator skeleton**
   - Define state machine states
   - Implement basic task queue
   - Add session persistence

### Medium-term (Next Month)

1. **VLM integration** - Add GPT-4V evaluation with structured prompts
2. **Parallel material testing** - Multiple variations simultaneously
3. **Template overhaul** - Full Blender 5.0.1 compatibility

---

## 8. Technical Reference

### Blender 5.0.1 Volume Info Node

The Volume Info node provides these outputs from Mantaflow simulations:
- **Density** - Smoke/fluid density (0-1)
- **Flame** - Fire intensity (0-1)
- **Temperature** - Normalized temperature (0-1 maps to 0-1000K internally)

### Working Sun Material Node Graph (v11)

```
[Volume Info] ─── Flame ──► [Color Ramp] ─── Color ──► [Emission] ──┐
      │                          │                          │      │
      │                          │           ┌─── Flame ────┘      │
      │                          │           │                     │
      │                    (red→orange       [Math: ×1.5] ─► Strength
      │                     →yellow→                               │
      │                      white)                                │
      │                                                           │
      └─── Density ──► [Math: ×4.0] ──► [Vol Absorption] ────────┴──► [Add Shader] ──► [Output: Volume]
```

### Quality Thresholds (Current - Need Revision)

| Metric | Threshold | Reality |
|--------|-----------|---------|
| LPIPS | < 0.35 | VFX scores 0.7-0.8, unusable |
| CLIP | > 0.60 | Plateaus at 0.65-0.67 |
| Temporal | > 0.70 | Works reasonably |

---

## Appendix A: Agent MCP Tool Reference

### script-generator
```
generate_script(effect_type, description, output_name, resolution, frame_start, frame_end, template_name, technique_name)
modify_script(script_path, modifications, output_name)
list_templates(script_type)
list_techniques(effect_type)
validate_parameters(params)
get_parameter_ranges()
```

### blender-executor
```
execute_blender_script(script_path, script_args, output_dir, blend_file, timeout_seconds, use_ui)
parse_blender_errors(stderr, stdout)
list_run_outputs(run_dir)
get_latest_run()
list_available_scripts(directory)
```

### asset-evaluator
```
compare_lpips(image1_path, image2_path, generate_heatmap)
compare_clip(image_path, query, query_is_image)
evaluate_render(render_path, reference_path, semantic_query, lpips_threshold, clip_threshold)
enhanced_evaluate(render_path, semantic_query, effect_type, reference_path, ...)
multi_prompt_clip_analysis(image_path, base_description, effect_type)
analyze_temporal_quality(frame_directory, frame_pattern, sample_rate)
get_alternative_approaches(current_approach, scores_history)
```

### iteration-controller
```
create_asset(asset_name, description, effect_type, reference_path, semantic_query, max_iterations, ...)
run_iteration(asset_name, iteration, effect_type, description, ...)
get_history(asset_name)
list_sessions(status_filter)
compare_iterations(asset_name)
```

### experiment-tracker
```
start_experiment_session(asset_name, effect_type, description, ...)
record_baseline(params, scores, render_path, script_path)
record_experiment_result(hypothesis, issue_addressed, result_params, result_scores, ...)
get_warnings_before_change(parameter, change_type)
suggest_experiments(issue, current_params, current_scores)
query_knowledge_base(query)
get_parameter_knowledge(parameter)
add_manual_learning(parameter, rule, warning, context)
```

---

## Appendix B: Sun Surface Iteration Log

| Version | Approach | Result | LPIPS | CLIP | Notes |
|---------|----------|--------|-------|------|-------|
| v6 | Volume Info + Blackbody | White/pale | - | - | Volume Info not reading correctly |
| v7 | Lower emission (0.5) | RED sphere | 0.744 | 0.659 | First visible structure |
| v8 | Higher temp (4000-6000K) | PINK sphere | 0.759 | 0.660 | Wrong color |
| v9 | Fixed 5800K + scatter | WHITE | - | - | Too bright again |
| v10 | v7 emission + higher temp | RED | 0.747 | 0.662 | Still red |
| v11 | **Color Ramp approach** | **ORANGE** | 0.738 | 0.671 | **Best result** |
| v12 | Refined ramp + brightness | TBD | TBD | TBD | Created, not run |

---

*Document generated from experimental pipeline development session, December 2025.*
