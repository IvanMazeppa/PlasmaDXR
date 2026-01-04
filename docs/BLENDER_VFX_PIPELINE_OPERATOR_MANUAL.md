# Blender VFX Pipeline Operator Manual

**For AI Assistants Operating the Multi-Agent System**

This manual provides instructions for AI models to effectively operate the Blender VFX asset generation pipeline. Follow these guidelines to achieve optimal results.

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Quick Start](#quick-start)
3. [MCP Tools Reference](#mcp-tools-reference)
4. [Complete Workflow](#complete-workflow)
5. [Quality Evaluation Guide](#quality-evaluation-guide)
6. [Knowledge Base Integration](#knowledge-base-integration)
7. [Technique Selection](#technique-selection)
8. [Circuit Breakers](#circuit-breakers)
9. [Troubleshooting](#troubleshooting)
10. [Best Practices](#best-practices)

---

## System Overview

### Architecture

```
User Request → Claude Code Session → MCP Servers → Blender → Evaluation → Iteration
                                         ↓
                    ┌────────────────────┴────────────────────┐
                    │                                          │
              script-generator                          blender-executor
              (generate, modify, validate)              (run simulations)
                    │                                          │
                    └──────────────┬───────────────────────────┘
                                   ↓
                            asset-evaluator
                            (quality metrics)
                                   ↓
                    ┌──────────────┴──────────────┐
                    │                              │
             iteration-controller          experiment-tracker
             (state management)            (learning/knowledge)
```

### Key Principles

1. **You execute the loop** - MCP tools return results; you call the next tool
2. **Always validate before executing** - Scripts must pass validation before Blender runs
3. **Consult knowledge before modifying** - Warning checks are automatic but heed critical warnings
4. **Track iterations explicitly** - Maintain counters and best scores in your context
5. **Respect circuit breakers** - Stop when limits are reached

---

## Quick Start

### Minimum Viable Workflow

```
1. Generate script     → mcp__script-generator__generate_script()
2. Validate script     → mcp__script-generator__validate_script()
3. Execute Blender     → mcp__blender-executor__execute_blender_script()
4. Evaluate quality    → mcp__asset-evaluator__evaluate_vfx_quality()
5. If score < 60       → mcp__script-generator__modify_script() → Go to step 2
6. If score >= 60      → Success! Report results
```

### Example Session Start

```
User: "Create a rising mushroom cloud explosion"

Your response:
"I'll create a mushroom cloud explosion. Let me start by generating the initial script."

Then call:
mcp__script-generator__generate_script(
    effect_type="pyro",
    description="A rising mushroom cloud explosion with bright orange fire and billowing smoke",
    output_name="mushroom_cloud_v1",
    resolution=96,
    frame_end=50
)
```

---

## MCP Tools Reference

### script-generator

| Tool | Purpose | When to Use |
|------|---------|-------------|
| `generate_script` | Create new Blender script | Start of session |
| `validate_script` | Check script before execution | After generate/modify, BEFORE execute |
| `modify_script` | Adjust parameters | After failed evaluation |
| `list_techniques` | Show available techniques | When exploring options |
| `recommend_technique` | UCB1-based selection | When unsure which technique |
| `record_technique_outcome` | Record success/failure | After session completes |
| `get_technique_stats` | View technique performance | When analyzing history |
| `validate_parameters` | Check parameter ranges | Before setting values |
| `get_parameter_ranges` | Get valid ranges | When unsure of limits |

### blender-executor

| Tool | Purpose | When to Use |
|------|---------|-------------|
| `execute_blender_script` | Run simulation | After validation passes |
| `parse_blender_errors` | Analyze failures | When execution fails |
| `list_run_outputs` | Find output files | After successful execution |
| `get_latest_run` | Get most recent run info | When locating outputs |
| `list_available_scripts` | Browse existing scripts | When reusing templates |

### asset-evaluator

| Tool | Purpose | When to Use |
|------|---------|-------------|
| `evaluate_vfx_quality` | PRIMARY: Score 0-100 | Every iteration |
| `extract_vfx_diagnostics` | Detailed metrics | When debugging quality |
| `compare_vfx_iterations` | A/B comparison | When comparing versions |
| `evaluate_ground_truth` | Compare to reference dataset | For sun/star effects |
| `compare_lpips` | Perceptual similarity | When reference image available |
| `compare_clip` | Semantic similarity | When semantic query available |
| `analyze_temporal_quality` | Animation smoothness | For animated sequences |

### iteration-controller

| Tool | Purpose | When to Use |
|------|---------|-------------|
| `diagnose_vfx_issues` | Interpret quality results | After evaluation |
| `get_next_iteration_params` | Suggest parameter changes | Before modify_script |
| `save_iteration_state` | Persist progress | After each iteration |
| `load_iteration_state` | Resume session | When continuing work |
| `list_orchestration_sessions` | View all sessions | When browsing history |

### experiment-tracker

| Tool | Purpose | When to Use |
|------|---------|-------------|
| `query_knowledge_base` | Search accumulated knowledge | Session start, when stuck |
| `get_warnings_before_change` | Check parameter risks | Before modifications |
| `suggest_experiments` | Get fix suggestions | When quality stuck |
| `record_experiment_result` | Log iteration outcome | After every iteration |
| `add_manual_learning` | Add discovered knowledge | When learning something new |
| `get_parameter_knowledge` | Deep dive on parameter | When troubleshooting |

### blender-manual

| Tool | Purpose | When to Use |
|------|---------|-------------|
| `search_tutorials` | Find guides | When learning technique |
| `search_vdb_workflow` | VDB-specific docs | For volumetric issues |
| `search_python_api` | API documentation | When writing scripts |
| `search_semantic` | Concept search | When unsure of terminology |

---

## Complete Workflow

### Stage 1: Initialize Session

```python
# 1. Parse user request
asset_name = "mushroom_cloud"
effect_type = "pyro"  # pyro, explosion, fire, smoke, nebula, sun, soft_body, cloth
description = "Rising mushroom cloud with orange fire"
reference_path = None  # Optional
semantic_query = "a dramatic mushroom cloud explosion"  # Optional

# 2. Initialize tracking (maintain in your context)
current_iteration = 0
best_score = 0
best_iteration = 0
best_render_path = None
best_script_path = None
iterations_without_improvement = 0
consecutive_blender_failures = 0

# 3. Check for existing session
Call: mcp__iteration-controller__load_iteration_state(session_id=asset_name)
# If found, restore variables and skip to appropriate stage

# 4. Preload knowledge
Call: mcp__experiment-tracker__query_knowledge_base(query=effect_type)
# Note relevant warnings and rules for this effect type

# 5. Announce session start
"Starting VFX generation session for {asset_name}
Effect type: {effect_type}
Description: {description}
Max iterations: 10"
```

### Stage 2: Generate Script

```python
# 1. Check circuit breakers first
if current_iteration >= 10:
    STOP("Max iterations reached")
if iterations_without_improvement >= 3:
    CONSIDER("Research or try different technique")

# 2. Get technique recommendation (optional but recommended)
Call: mcp__script-generator__recommend_technique(
    effect_type=effect_type,
    description=description
)
# Use recommended technique_name in generate_script

# 3. Generate the script
Call: mcp__script-generator__generate_script(
    effect_type=effect_type,
    description=description,
    output_name=asset_name,
    resolution=96,
    frame_end=50,
    technique_name=<recommended_technique>  # Optional
)

# 4. Extract script_path from result
script_path = result["script_path"]
```

### Stage 2.5: Validate Script (MANDATORY)

```python
# NEVER skip this step
Call: mcp__script-generator__validate_script(
    script_path=script_path,
    strict=False
)

# Check result
if not result["valid"]:
    # Log errors
    for issue in result["issues"]:
        if issue["level"] == "error":
            LOG_ERROR(issue["message"])

    # Return to Stage 2 with fixes
    # Common fixes:
    # - Syntax error → Regenerate with different template
    # - Parameter out of range → Will be auto-clamped
    # - Missing patterns → Add domain/flow setup
    RETURN_TO_STAGE_2()

# If valid (even with warnings), proceed
LOG(f"Validation passed ({result['warning_count']} warnings)")
```

### Stage 3: Execute Blender

```python
# 1. Check circuit breakers
if consecutive_blender_failures >= 3:
    STOP("Too many Blender failures")

# 2. Log iteration
LOG(f"=== ITERATION {current_iteration + 1} of 10 ===")

# 3. Execute
Call: mcp__blender-executor__execute_blender_script(
    script_path=script_path,
    script_args={"--bake": "1", "--resolution": "96"},
    timeout_seconds=600
)

# 4. Handle result
if result["success"]:
    consecutive_blender_failures = 0
    # Find outputs in build/vdb_output/{asset_name}/
    render_path = find_middle_frame_render(result)
else:
    consecutive_blender_failures += 1

    # Parse error
    Call: mcp__blender-executor__parse_blender_errors(
        stderr=result["stderr"]
    )

    # Apply suggested fix and retry
    RETURN_TO_STAGE_2()
```

### Stage 4: Evaluate Quality

```python
# 1. Run VFX quality evaluation (PRIMARY)
Call: mcp__asset-evaluator__evaluate_vfx_quality(
    image_path=render_path,
    effect_type=effect_type
)

# 2. Extract score and issues
score = result["composite_score"]  # 0-100
issues = result["issues"]  # List of problems
passed = result["passed"]  # True if score >= 60

# 3. Update tracking
current_iteration += 1
if score > best_score:
    best_score = score
    best_iteration = current_iteration
    best_render_path = render_path
    best_script_path = script_path
    iterations_without_improvement = 0
else:
    iterations_without_improvement += 1

# 4. Optional: Run additional evaluations
if reference_path:
    Call: mcp__asset-evaluator__compare_lpips(render_path, reference_path)

if semantic_query:
    Call: mcp__asset-evaluator__compare_clip(render_path, semantic_query)
```

### Stage 5: Decide Next Action

```python
# QUALITY PASSED
if passed and score >= 60:
    LOG(f"Quality gate PASSED at iteration {current_iteration}")
    LOG(f"Final score: {score}")
    LOG(f"Output: {best_render_path}")

    # Record success
    Call: mcp__experiment-tracker__record_experiment_result(...)
    Call: mcp__script-generator__record_technique_outcome(
        technique_name=technique_used,
        effect_type=effect_type,
        success=True,
        final_score=score,
        iterations=current_iteration
    )

    SESSION_COMPLETE()

# QUALITY FAILED - Check circuit breakers
if current_iteration >= 10:
    STOP("Max iterations reached")
    REPORT_BEST_RESULT()

if iterations_without_improvement >= 3:
    # Try different approach
    Call: mcp__experiment-tracker__suggest_experiments(issue=primary_issue)
    # Or try different technique
    Call: mcp__script-generator__list_techniques(effect_type=effect_type)

# ITERATE
# 1. Diagnose issues
Call: mcp__iteration-controller__diagnose_vfx_issues(
    quality_json=json.dumps(result)
)

# 2. Get parameter suggestions
Call: mcp__iteration-controller__get_next_iteration_params(
    current_params=current_params_json,
    quality_json=json.dumps(result),
    iteration=current_iteration
)

# 3. Modify script (warnings checked automatically)
Call: mcp__script-generator__modify_script(
    script_path=script_path,
    modifications=suggested_modifications
)

# 4. Check warnings in result
if result["has_critical_warnings"]:
    LOG_WARNING("Critical warnings detected:")
    for warning in result["warnings"]:
        LOG_WARNING(f"  - {warning}")
    # Apply mitigations or adjust parameters

# 5. Return to Stage 2.5 (Validate)
RETURN_TO_STAGE_2_5()
```

### Stage 6: Record Learning

```python
# After EVERY iteration (success or failure)
Call: mcp__experiment-tracker__record_experiment_result(
    hypothesis="Increased turbulence to add more detail",
    issue_addressed="Flat, boring smoke appearance",
    result_params=json.dumps(current_params),
    result_scores=json.dumps({"vfx": score, "lpips": lpips_score}),
    result_render=render_path,
    result_script=script_path,
    success=(score > previous_score),
    observed_effects='["More turbulent smoke", "Better dynamic range"]',
    learnings='["Turbulence 0.8 works well for mushroom clouds"]',
    warnings='["High turbulence can cause flickering"]'
)
```

---

## Quality Evaluation Guide

### VFX Quality Score Interpretation

| Score | Quality | Action |
|-------|---------|--------|
| 80-100 | Excellent | Accept immediately |
| 60-79 | Good | Accept (minor improvements possible) |
| 40-59 | Fair | Iterate (significant issues) |
| 20-39 | Poor | Major rework needed |
| 0-19 | Very Poor | Fundamental problems |

### Common Issues and Fixes

| Issue | Meaning | Fix |
|-------|---------|-----|
| TOO DARK | Insufficient brightness | Increase `flame_max_temp`, `emission_strength` |
| TOO BRIGHT | Overexposed | Decrease temperature, add smoke |
| LOW COVERAGE | Effect too small | Increase `domain_scale`, adjust emitter size |
| NO STRUCTURE | Flat/blobby | Increase `vorticity`, add noise |
| WRONG COLOR | Color mismatch | Adjust `flame_smoke` ratio, temperature |
| LOW DYNAMIC RANGE | No contrast | Increase temperature range, add variation |

### Decision Matrix

| VFX Score | LPIPS | CLIP | Decision |
|-----------|-------|------|----------|
| >= 60 | < 0.35 | > 0.60 | **ACCEPT** |
| >= 60 | >= 0.35 | * | Accept (note style difference) |
| >= 60 | * | <= 0.60 | Accept (note semantic drift) |
| 40-59 | < 0.35 | > 0.60 | Iterate (close) |
| 40-59 | * | * | Iterate (needs work) |
| < 40 | * | * | **REJECT** (fundamental issues) |

---

## Knowledge Base Integration

### Automatic Warning Checks

When you call `modify_script()`, warnings are checked automatically:

```python
result = modify_script(script_path, modifications)

# Check the result
if result["has_critical_warnings"]:
    # Critical warnings include patterns: MUST, NEVER, ALWAYS, explosion, crash
    for warning in result["warnings"]:
        LOG_WARNING(warning)

    # Apply mitigations
    for mitigation in result["mitigations"]:
        APPLY_MITIGATION(mitigation)
```

### Severity Levels

| Severity | Meaning | Action |
|----------|---------|--------|
| critical | Must not proceed without mitigation | Stop and fix |
| high | Significant risk | Proceed with caution |
| medium | Caution advised | Note and continue |
| low | Informational | Continue |
| none | No warnings | Continue |

### Querying Knowledge Manually

```python
# Search for relevant knowledge
Call: mcp__experiment-tracker__query_knowledge_base(
    query="domain_scale clipping"
)

# Get parameter-specific info
Call: mcp__experiment-tracker__get_parameter_knowledge(
    parameter="domain_scale"
)

# Get warnings before a specific change
Call: mcp__experiment-tracker__get_warnings_before_change(
    parameter="step_min",
    change_type="decrease"
)
```

### Adding New Knowledge

When you discover something important:

```python
Call: mcp__experiment-tracker__add_manual_learning(
    parameter="domain_scale",
    rule="When increasing domain_scale beyond 10, also increase domain_location_z proportionally",
    warning="Failure to adjust position causes effect clipping at edges",
    context="pyro simulations"
)
```

---

## Technique Selection

### UCB1 Algorithm

The system uses Upper Confidence Bound (UCB1) for technique selection:

```
UCB Score = Average Reward + sqrt(2 * ln(total_trials) / technique_trials)
```

- **Untried techniques** get priority (infinite score)
- **Successful techniques** build higher average reward
- **Exploration bonus** decreases as technique is used more

### Using Technique Selection

```python
# 1. Get recommendation
Call: mcp__script-generator__recommend_technique(
    effect_type="pyro",
    description="rising mushroom cloud with orange flames"
)

# Result includes:
# - technique_name: Recommended technique
# - confidence: 0-1 confidence score
# - selection_reason: Why this was chosen
# - alternatives: Other options with stats

# 2. Use in generation
Call: mcp__script-generator__generate_script(
    ...,
    technique_name=result["technique_name"]
)

# 3. Record outcome after session
Call: mcp__script-generator__record_technique_outcome(
    technique_name="rising_mushroom",
    effect_type="pyro",
    success=True,
    final_score=75.0,
    iterations=3
)
```

### Available Techniques (Pyro)

| Technique | Best For | Characteristics |
|-----------|----------|-----------------|
| rising_mushroom | Nuclear explosions | Tall, rising column with cap |
| ground_hugger | Fuel explosions | Low, spreading flames |
| aerial_burst | Airbursts | Spherical expansion |
| directional_jet | Rocket exhaust | Focused stream |
| rolling_fireball | Fireballs | Tumbling, chaotic |
| smoldering_embers | Aftermath | Slow, glowing particles |

---

## Circuit Breakers

### Automatic Limits

| Circuit Breaker | Limit | What Happens |
|-----------------|-------|--------------|
| MAX_ITERATIONS | 10 | Session ends, reports best result |
| MAX_WALL_TIME | 30 min | Session ends, saves state |
| NO_IMPROVEMENT | 3 iterations | Triggers research/technique change |
| BLENDER_FAILURES | 3 consecutive | Session ends, reports error |
| LOW_SCORES | 3 consecutive < 30 | Triggers fundamental rethink |

### Checking Circuit Breakers

Before every iteration:

```python
# Check iteration limit
if current_iteration >= 10:
    STOP("Max iterations reached")
    REPORT_BEST()

# Check improvement stall
if iterations_without_improvement >= 3:
    # Try research
    Call: mcp__blender-manual__search_tutorials(topic=effect_type)
    # Or try different technique
    Call: mcp__script-generator__list_techniques(effect_type=effect_type)

# Check consecutive failures
if consecutive_blender_failures >= 3:
    STOP("Too many execution failures")

# Check consecutive low scores
if consecutive_low_scores >= 3:
    # Fundamental rethink needed
    Call: mcp__experiment-tracker__suggest_experiments(issue="consistently low scores")
```

---

## Troubleshooting

### Script Validation Fails

| Error | Cause | Fix |
|-------|-------|-----|
| SyntaxError | Python syntax issue | Regenerate with different template |
| Parameter out of range | Invalid value | Use `validate_parameters()` first |
| Missing domain setup | No fluid domain | Use volumetric template |
| Dangerous pattern | Security risk | Remove eval/exec/subprocess |

### Blender Execution Fails

| Error | Cause | Fix |
|-------|-------|-----|
| "Context is incorrect" | Wrong active object | Ensure domain is selected |
| "Operator poll failed" | Prerequisites missing | Check bpy.context setup |
| Out of memory | Resolution too high | Reduce resolution to 64-96 |
| Timeout | Simulation too slow | Reduce frame_end, resolution |

### Quality Always Low

| Symptom | Possible Cause | Fix |
|---------|----------------|-----|
| Score stuck at 30-40 | Wrong effect type | Check evaluator matches effect |
| Always "TOO DARK" | Emission too low | Increase temperature, emission |
| Always "NO STRUCTURE" | Missing turbulence | Add vorticity, noise |
| LPIPS always high | Style mismatch | Focus on VFX score instead |

### Knowledge Base Not Helping

| Issue | Solution |
|-------|----------|
| No warnings returned | Knowledge base may be empty - add manual learnings |
| Conflicting rules | Check `resolve_conflicts()` or request human review |
| Wrong suggestions | Query with more specific context |

---

## Best Practices

### DO

1. **Always validate before executing** - Catches 90% of errors
2. **Log every iteration** - Helps debugging and learning
3. **Record experiment results** - Builds knowledge base
4. **Check warnings in modify_script results** - Prevents known mistakes
5. **Use recommend_technique** - Benefits from past experience
6. **Save state periodically** - Enables session resume
7. **Report best result even on failure** - User may accept "good enough"

### DON'T

1. **Skip validation** - Wastes Blender execution time
2. **Ignore critical warnings** - Leads to known failures
3. **Exceed circuit breaker limits** - Wastes resources
4. **Guess parameter ranges** - Use `get_parameter_ranges()`
5. **Forget to record outcomes** - Knowledge base won't learn
6. **Run identical parameters twice** - Track what you've tried

### Communication Style

```
GOOD:
"Iteration 3: Score improved from 45 to 58 by increasing vorticity.
Still below threshold (60). Issues: LOW_COVERAGE.
Applying fix: Increasing domain_scale from 8 to 12."

BAD:
"Running another iteration..."
```

### Error Reporting

```
GOOD:
"Blender execution failed: FluidDomainSettings has no attribute 'openvdb_cache_compress_type'
This is a Blender 5.0 API change. Regenerating script with updated API."

BAD:
"Something went wrong. Trying again."
```

---

## Effect Type Reference

### Volumetric Effects

| Type | Key Parameters | Typical Score Range |
|------|----------------|---------------------|
| pyro | flame_smoke, vorticity, burning_rate | 50-85 |
| explosion | vorticity, domain_scale, frame_end | 55-90 |
| fire | flame_max_temp, burning_rate | 45-80 |
| smoke | dissolve_speed, vorticity | 50-75 |
| nebula | noise_scale, noise_strength | 40-70 |
| sun | temperature, emission_strength | 45-75 |

### Mesh Physics Effects

| Type | Key Parameters | Notes |
|------|----------------|-------|
| soft_body | step_min=20, damping=2.0 | Use live render pattern |
| cloth | quality, mass, air_damping | Requires collision setup |
| rigid_body | friction, bounciness | Needs physics world |

### Custom Effects

For effects not matching standard categories:
- Use `effect_type="custom"`
- Rely on CLIP semantic evaluation
- Provide detailed `semantic_query`

---

## Session Templates

### Standard VFX Session

```
1. Parse request → Extract asset_name, effect_type, description
2. Load knowledge → query_knowledge_base(effect_type)
3. Generate → generate_script(...)
4. Validate → validate_script(...)
5. Execute → execute_blender_script(...)
6. Evaluate → evaluate_vfx_quality(...)
7. If passed → Report success
8. If failed → diagnose_vfx_issues → modify_script → Go to 4
9. Record → record_experiment_result(...)
```

### Research-Heavy Session

```
1. Research first → search_tutorials(topic)
2. List techniques → list_techniques(effect_type)
3. Recommend → recommend_technique(...)
4. Standard workflow...
5. If stuck → search_vdb_workflow(specific_issue)
6. Continue...
```

### Resume Session

```
1. Load state → load_iteration_state(session_id)
2. Verify files exist (script_path, render_path)
3. Skip to appropriate stage based on saved state
4. Continue workflow...
```

---

## Version Information

- **Manual Version:** 1.0.0
- **Pipeline Version:** Phase 4 Complete
- **Last Updated:** 2026-01-04
- **Compatible Phases:** 0, 0.5, 1, 2, 2.5, 3, 4

---

## Related Documents

- `docs/MULTI_AGENT_IMPROVEMENT_PLAN_V3.md` - Implementation status
- `.claude/skills/blender-orchestrator/SKILL.md` - Skill definition
- `docs/BLENDER_VFX_PIPELINE_ANALYSIS_REPORT.md` - System analysis
- `agents/*/README.md` - Individual agent documentation
