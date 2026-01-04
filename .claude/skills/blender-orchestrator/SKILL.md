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

**6. blender-manual**
- Searches Blender documentation
- Finds tutorials and techniques
- Provides API reference

## Quality Thresholds

| Metric | Threshold | Description |
|--------|-----------|-------------|
| VFX Quality Score | >= 60 | Composite quality (brightness, color, structure) |
| Temporal Consistency | >= 0.7 | Animation smoothness (1.0 = no flicker) |
| Ground Truth Similarity | >= 0.65 | Distribution match to real footage |

---

## CRITICAL: Autonomous Workflow Instructions

**IMPORTANT:** When this skill is invoked, follow these stages in order. Execute MCP tools directly - do not describe what you would do, actually call the tools.

---

## Iteration Tracking (MANDATORY)

At the START of each iteration, you MUST:

1. Increment iteration counter: `current_iteration += 1`
2. Log: `"=== ITERATION {current_iteration} of 10 ==="`
3. Check circuit breakers (see below) BEFORE proceeding

**Track these variables throughout the session:**

| Variable | Initial Value | Description |
|----------|---------------|-------------|
| `current_iteration` | 0 | Increments each loop |
| `max_iterations` | 10 | Hard limit |
| `best_score` | 0 | Highest VFX score achieved |
| `best_iteration` | 0 | Which iteration achieved best_score |
| `best_render_path` | "" | Path to best render |
| `best_script_path` | "" | Path to best script |
| `iterations_without_improvement` | 0 | Resets when score improves |
| `consecutive_blender_failures` | 0 | Resets on successful execution |
| `consecutive_low_scores` | 0 | Count of scores < 40 |
| `session_start_time` | now() | Timestamp when session began |

---

## Circuit Breakers (HARD STOPS)

**Before EVERY iteration, check these conditions. If ANY trigger, STOP immediately:**

| Breaker | Condition | Action |
|---------|-----------|--------|
| MAX_ITERATIONS | `current_iteration >= 10` | STOP, report best result |
| MAX_WALL_TIME | `elapsed > 30 minutes` | STOP, save state for resume |
| NO_IMPROVEMENT | `iterations_without_improvement >= 3` | STOP, local optimum reached |
| QUALITY_FLOOR | `consecutive_low_scores >= 2` (score < 40) | PAUSE, request human review |
| BLENDER_FAILURES | `consecutive_blender_failures >= 2` | STOP, script has fundamental issue |

**When a circuit breaker triggers:**

1. Log: `"CIRCUIT BREAKER: {breaker_name} triggered"`
2. Save session state:
   ```
   Call: mcp__iteration-controller__save_iteration_state(
       session_id=<session_id>,
       asset_name=<asset_name>,
       effect_type=<effect_type>,
       current_iteration=<current_iteration>,
       best_score=<best_score>,
       best_iteration=<best_iteration>,
       parameters_current=<params_json>,
       issues_current=<issues_json>,
       next_action="circuit_breaker_stop",
       status="stopped"
   )
   ```
3. Report final status with best score and output paths
4. **DO NOT continue iterating**

---

## Knowledge Base Consultation (MANDATORY)

**Before EVERY parameter modification, you MUST consult the knowledge base:**

### Step 1: Check for Warnings

```
Call: mcp__experiment-tracker__get_warnings_before_change(
    parameter=<parameter_being_changed>,
    change_type="increase" or "decrease"
)
```

**If warnings returned with severity "critical":**
- Apply the suggested mitigation, OR
- Skip that parameter change entirely
- Log: `"Skipped {param} due to warning: {reason}"`

### Step 2: Get Suggestions for Current Issues

```
Call: mcp__experiment-tracker__suggest_experiments(
    issue=<primary_issue_from_evaluation>,
    current_params=<current_params_json>,
    current_scores=<current_scores_json>
)
```

**If suggestion confidence > 0.6:**
- USE the knowledge base suggestion instead of heuristic fix
- Log: `"Using KB suggestion: {suggestion} (confidence: {confidence})"`

**If no high-confidence suggestions:**
- Fall back to heuristic fixes from diagnosis
- Consider research integration (see below)

### Step 3: Record Learning After Each Iteration

```
Call: mcp__experiment-tracker__record_experiment_result(
    hypothesis=<what we tried>,
    issue_addressed=<what problem we targeted>,
    result_params=<new_params_json>,
    result_scores=<new_scores_json>,
    result_render=<render_path>,
    result_script=<script_path>,
    success=<true/false based on score improvement>,
    observed_effects='["<effect1>", "<effect2>"]',
    learnings='["<learning1>", "<learning2>"]',
    warnings='["<warning1>"]'
)
```

**This is NOT optional. Skipping knowledge consultation wastes learned experience.**

---

## Research Integration (When Stuck)

**If `iterations_without_improvement >= 2` AND no high-confidence suggestions from knowledge base:**

### Step 1: Research in Blender Documentation

```
Call: mcp__blender-manual__search_tutorials(
    topic=<effect_type>,
    technique=<current_technique>
)
```

### Step 2: Search for Alternative Approaches

```
Call: mcp__blender-manual__search_vdb_workflow(
    query="<effect_type> <primary_issue> solution"
)
```

### Step 3: Consider Technique Switch

```
Call: mcp__script-generator__list_techniques(effect_type=<effect_type>)
```

If a promising DIFFERENT technique is found:
- Log: `"Switching technique from {old} to {new} due to stagnation"`
- Restart from Stage 2 (Generate Script) with new technique
- Reset `iterations_without_improvement` to 0

---

## Quality Evaluation Decision Tree

**Follow this EXACT sequence when evaluating quality:**

### Step 1: Always Run VFX Diagnostics First (no reference needed)

```
Call: mcp__asset-evaluator__evaluate_vfx_quality(
    image_path="build/vdb_output/<asset_name>/render_0025.png",
    effect_type=<effect_type>
)
```

### Step 2: Apply Decision Based on VFX Score

**If VFX score < 40:** REJECT immediately
- Do NOT run LPIPS/CLIP (waste of time)
- Increment `consecutive_low_scores`
- Diagnose issues and iterate

**If VFX score >= 40 but < 60:** Check secondary metrics
- Reset `consecutive_low_scores` to 0
- If `reference_path` provided: Run LPIPS
- If `semantic_query` provided: Run CLIP
- Use decision matrix below

**If VFX score >= 60:** Quality gate passed
- Reset `consecutive_low_scores` to 0
- Still run LPIPS/CLIP for completeness
- Note any warnings but ACCEPT

### Decision Matrix

| VFX | LPIPS | CLIP | Decision |
|-----|-------|------|----------|
| >=60 | <0.35 | >0.60 | ACCEPT |
| >=60 | >=0.35 | * | ACCEPT (note: style differs from reference) |
| >=60 | * | <=0.60 | ACCEPT (note: semantic drift) |
| 40-59 | <0.35 | >0.60 | ITERATE (close, minor fixes needed) |
| 40-59 | * | * | ITERATE (needs work) |
| <40 | * | * | REJECT (fundamental issues) |

---

## Stage-by-Stage Workflow

### Stage 1: Initialize Session

1. Parse the user's request for:
   - `asset_name`: Name for output files
   - `effect_type`: pyro, explosion, fire, smoke, nebula, or sun
   - `description`: What the effect should look like
   - `reference_path`: (optional) Path to reference image
   - `semantic_query`: (optional) Text description for CLIP

2. Initialize tracking variables (see Iteration Tracking above)

3. Check if resuming an existing session:
   ```
   Call: mcp__iteration-controller__load_iteration_state(session_id=<asset_name>)
   ```
   If found, restore variables and skip to appropriate stage.

4. Preload relevant knowledge (MANDATORY - Phase 4.3):
   ```
   Call: mcp__experiment-tracker__query_knowledge_base(query=<effect_type>)
   ```

   **Knowledge preloading retrieves:**
   - Known parameter ranges and optimal values for this effect type
   - Accumulated rules from past experiments (e.g., "Always adjust domain_location_z when scaling")
   - Warnings specific to parameters commonly used for this effect
   - Success rates for different parameter combinations

   **Store the preloaded knowledge for reference during iteration:**
   - Parameter-specific warnings will be checked automatically by modify_script()
   - Critical warnings (severity="critical") will be flagged before any modification
   - Rules and optimal values guide parameter selection

   **Effect type → Parameters preloaded:**
   - `pyro/explosion`: flame_smoke, vorticity, burning_rate, temperature, domain_scale
   - `fire`: flame_max_temp, flame_smoke, burning_rate, vorticity
   - `smoke`: vorticity, dissolve_speed, domain_scale
   - `soft_body`: step_min, step_max, damping, goal_spring, friction
   - `cloth`: quality, mass, air_damping, collision_quality
   - `nebula/sun`: noise_scale, noise_strength, vorticity, temperature

5. Announce the session start with configuration summary

### Stage 2: Generate Script

1. **Check circuit breakers** (see above)

2. First, explore available techniques:
   ```
   Call: mcp__script-generator__list_techniques(effect_type=<effect_type>)
   ```

3. Generate the initial script:
   ```
   Call: mcp__script-generator__generate_script(
       effect_type=<effect_type>,
       description=<description>,
       output_name=<asset_name>,
       resolution=96,
       frame_end=50
   )
   ```

4. Extract the script path from the result

### Stage 2.5: Validate Script (MANDATORY - Phase 2)

**This validation step prevents wasted Blender execution time by catching errors early.**

1. Run comprehensive script validation:
   ```
   Call: mcp__script-generator__validate_script(
       script_path=<generated_script_path>,
       strict=false
   )
   ```

2. **If validation fails (valid=false):**
   - Log: `"Script validation FAILED: {error_count} errors"`
   - Review the `issues` array for specific problems
   - Common fixes:
     - Syntax error → Regenerate script with corrected template
     - Parameter out of range → Modify with clamped values
     - Missing required patterns → Add domain/flow setup
   - Return to Stage 2 (Generate Script) with adjustments
   - **DO NOT proceed to execution with an invalid script**

3. **If validation passes with warnings:**
   - Log: `"Script validation PASSED ({warning_count} warnings)"`
   - Review warnings but proceed to execution
   - Warnings may indicate:
     - Absolute Windows paths (cross-platform issue)
     - No explicit output path (will use defaults)
     - Dangerous patterns (os.system, eval) - review carefully

4. **If validation passes with no issues:**
   - Log: `"Script validation PASSED (clean)"`
   - Proceed to Stage 3 (Execute Blender)

5. Record validation metadata for learning:
   - `detected_effect_type`: What the validator detected (volumetric/mesh)
   - `detected_simulation_pattern`: bake_export or live_render
   - `extracted_params`: Parameters found in script

### Stage 3: Execute Blender

1. **Check circuit breakers** (see above)

2. Log: `"=== ITERATION {current_iteration} of 10 ==="`

3. Run the Blender simulation:
   ```
   Call: mcp__blender-executor__execute_blender_script(
       script_path=<script_path>,
       script_args={"--bake": "1", "--resolution": "96"}
   )
   ```

4. If execution fails:
   - Increment `consecutive_blender_failures`
   - Parse the error:
     ```
     Call: mcp__blender-executor__parse_blender_errors(stderr=<error_output>)
     ```
   - Apply suggested fixes and retry (max 3 attempts)
   - If still failing after 3 attempts, check BLENDER_FAILURES circuit breaker

5. If execution succeeds:
   - Reset `consecutive_blender_failures` to 0
   - Locate output files in `build/vdb_output/<asset_name>/`

### Stage 4: Evaluate Quality

1. Find the middle frame render (e.g., `render_0025.png`)

2. Follow the **Quality Evaluation Decision Tree** above

3. Update tracking variables:
   ```python
   if new_score > best_score:
       best_score = new_score
       best_iteration = current_iteration
       best_render_path = render_path
       best_script_path = script_path
       iterations_without_improvement = 0
   else:
       iterations_without_improvement += 1
   ```

4. Extract issues from evaluation result

### Stage 5: Decide Next Action

**If quality passed (score >= 60 AND no critical issues):**
- Log: `"Quality gate PASSED at iteration {current_iteration}"`
- Report success with final score and output paths
- Record final learning to knowledge base
- Session complete

**If quality failed AND circuit breakers not triggered:**

1. Increment `current_iteration`

2. **Consult knowledge base** (MANDATORY - see above)

3. Diagnose issues:
   ```
   Call: mcp__iteration-controller__diagnose_vfx_issues(
       quality_json=<evaluation_result_json>
   )
   ```

4. Apply fixes from knowledge base OR diagnosis

5. Modify the script (AUTOMATIC WARNING CHECKS - Phase 4.1):
   ```
   Call: mcp__script-generator__modify_script(
       script_path=<current_script>,
       modifications={
           "turbulence": <new_value>,
           "temperature": <new_value>,
           ...
       }
   )
   ```

   **The result will include warning checks for each parameter:**
   - `warnings`: List of warnings from knowledge base
   - `mitigations`: Suggested mitigations for critical warnings
   - `has_critical_warnings`: True if any parameter has critical issues

   **If `has_critical_warnings` is True:**
   - Review the warnings carefully before proceeding
   - Apply suggested mitigations
   - Consider alternative parameter values
   - Critical patterns include: "MUST", "NEVER", "ALWAYS", "explosion", "crash", "fail"

   **Severity levels:**
   - `critical`: Should not proceed without mitigation
   - `high`: Significant risk, proceed with caution
   - `medium`: Caution advised
   - `low`: Informational only

6. **Return to Stage 3** (Execute Blender)

**If max iterations reached OR circuit breaker triggered:**
- Report best score achieved across all iterations
- Provide the best iteration's output paths (`best_render_path`, `best_script_path`)
- Suggest manual refinements based on remaining issues
- Save state for potential resume

### Stage 6: Record Learning

After EVERY iteration (success or failure), record what was learned:

```
Call: mcp__experiment-tracker__record_experiment_result(
    hypothesis="<what we tried>",
    issue_addressed="<what problem we targeted>",
    result_params=<params_json>,
    result_scores=<scores_json>,
    result_render=<render_path>,
    result_script=<script_path>,
    success=<true if score improved>,
    observed_effects='["<effect1>", "<effect2>"]',
    learnings='["<what we learned>"]',
    warnings='["<any gotchas discovered>"]'
)
```

This builds a knowledge base for future sessions.

---

## Common Issues and Fixes

| Issue | Cause | Fix |
|-------|-------|-----|
| TOO DARK | Low flame temperature | Increase `flame_max_temp` by 500-1000K |
| NO STRUCTURE | Low turbulence | Increase `turbulence` to 0.5-0.8 |
| WRONG COLOR | Temperature mismatch | Adjust `flame_max_temp` for target color |
| CLIPPING | Domain too small | Increase `domain_scale` AND reposition emitter |
| SPARSE | Low density | Increase `density_multiplier` |
| FLAT | No depth variation | Increase `noise_scale` and `vorticity` |

---

## Trust & Autonomy System

The orchestrator uses a graduated autonomy model:

| Level | Trust Score | Behavior |
|-------|-------------|----------|
| Supervised | 0.0 - 0.3 | Requires approval for each action |
| Guided | 0.3 - 0.6 | Can execute, reports decisions |
| Autonomous | 0.6 - 0.8 | Full autonomy within guardrails |
| Trusted | 0.8 - 1.0 | Extended limits, minimal oversight |

---

## Output Locations

| Asset | Path |
|-------|------|
| VDB Files | `build/vdb_output/<asset_name>/` |
| Renders | `build/vdb_output/<asset_name>/render_*.png` |
| Scripts | `assets/blender_scripts/generated/<asset_name>.py` |
| Sessions | `build/orchestrator_state/<session_id>.json` |

---

## Best Practices

1. **Start with Description**: Detailed descriptions produce better initial scripts
2. **Use References**: Ground truth comparison dramatically improves realism
3. **Iterate Patiently**: Complex effects may need 5+ iterations
4. **Review Diagnostics**: VFX quality issues are specific and actionable
5. **Resume Don't Restart**: Use `resume_session` to continue from checkpoints
6. **Trust the Scores**: ML evaluation correlates well with human perception
7. **Consult Knowledge Base**: Past learnings accelerate current sessions
8. **Research When Stuck**: Blender manual has solutions to common problems

---

## Parallel Evaluation (Advanced)

For complex effects, spawn parallel Task subagents:

```
Use Task tool to spawn:
1. VFX quality evaluation subagent
2. Temporal consistency subagent (if animated)
3. Ground truth comparison subagent (if reference provided)
```

Combine results and use worst score for quality gate.

---

**Remember**: The orchestrator works autonomously but respects circuit breakers and quality gates. It will iterate until the asset meets thresholds or a circuit breaker triggers. For best results, provide detailed descriptions and reference images when available.
