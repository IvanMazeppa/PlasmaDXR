# VFX Asset Creation Pipeline - Operator Manual v2.0

**Version:** 2.0 (GPT-5.2 Tool Calling Integration)
**Date:** 2025-01-05
**Status:** Production Ready

---

## Executive Summary

This manual documents the correct operation of the PlasmaDX VFX asset creation pipeline. **Version 2.0 introduces mandatory Blender documentation lookup via GPT-5.2 tool calling**, eliminating reliance on outdated LLM training data.

### Critical Change from v1.x

| Aspect | v1.x (Broken) | v2.0 (Correct) |
|--------|---------------|----------------|
| Documentation | LLM training data (stale) | Live blender-manual MCP queries |
| API Reference | Guessed from memory | Real-time bpy.types/bpy.ops lookup |
| Parameter Ranges | Often wrong | Validated against Blender 5.0 docs |
| Orchestration | Often bypassed | Mandatory orchestrator involvement |

---

## Table of Contents

1. [Pipeline Architecture](#1-pipeline-architecture)
2. [MCP Server Reference](#2-mcp-server-reference)
3. [Correct Asset Creation Workflow](#3-correct-asset-creation-workflow)
4. [Best Practices](#4-best-practices)
5. [Troubleshooting](#5-troubleshooting)
6. [Quick Reference Commands](#6-quick-reference-commands)

---

## 1. Pipeline Architecture

### 1.1 System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CLAUDE CODE (Orchestrator)                          │
│                                                                             │
│  Coordinates all agents, makes strategic decisions, tracks quality gates   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MCP SERVER LAYER                                  │
├──────────────────┬──────────────────┬──────────────────┬───────────────────┤
│ blender-librarian│ script-generator │ blender-executor │ asset-evaluator   │
│ (GPT-5.2 + Docs) │ (Script Creation)│ (Blender CLI)    │ (Quality Metrics) │
├──────────────────┼──────────────────┼──────────────────┼───────────────────┤
│ experiment-      │ iteration-       │ blender-manual   │ blender-          │
│ tracker          │ controller       │ (Documentation)  │ orchestrator      │
└──────────────────┴──────────────────┴──────────────────┴───────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           EXECUTION LAYER                                   │
│                                                                             │
│  Blender 5.0 CLI  →  VDB Export  →  Render  →  Evaluation  →  Iteration   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Data Flow for Asset Creation

```
User Request
     │
     ▼
┌─────────────────┐
│ blender-        │ ──► GPT-5.2 searches blender-manual (12 tools)
│ librarian       │ ──► Returns validated parameter modifications
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ script-         │ ──► Generates Blender Python script
│ generator       │ ──► Uses technique catalog for variety
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ blender-        │ ──► Executes script via Blender CLI
│ executor        │ ──► Captures VDB output + renders
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ asset-          │ ──► Evaluates quality (VFX metrics, ground truth)
│ evaluator       │ ──► Returns issues + suggestions
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ experiment-     │ ──► Records results for learning
│ tracker         │ ──► Suggests next experiments
└────────┬────────┘
         │
         ▼
    Loop until quality threshold met
```

---

## 2. MCP Server Reference

### 2.1 blender-librarian (NEW in v2.0)

**Purpose:** Intelligent documentation lookup using GPT-5.2 with tool calling.

**Why This Exists:** Previous versions relied on Claude's training data for Blender knowledge, which was:
- Outdated (pre-Blender 5.0)
- Incomplete (missing many API details)
- Often wrong (parameter ranges, deprecated APIs)

**Tools:**

| Tool | Purpose | When to Use |
|------|---------|-------------|
| `diagnose_render_issue` | Vision-based render analysis | When render has visual problems |
| `get_modification_advice` | Get parameter changes for issues | After identifying problems |
| `get_budget_status` | Check GPT-5.2 API budget | Before expensive operations |
| `add_to_playbook` | Save learned fixes | When a fix works well |

**Internal Operation (search_docs_intelligent):**
1. Receives query about Blender issue
2. GPT-5.2 autonomously calls blender-manual tools (up to 5 iterations)
3. Searches: semantic, VDB workflow, Python API, nodes, modifiers
4. Synthesizes findings into actionable parameter modifications
5. Returns JSON with answer, modifications, confidence, doc_paths_used

**Cost:** ~$0.04 per query (12 tool calls typical)

### 2.2 blender-manual (Documentation Server)

**Purpose:** Provides 12 specialized search tools over Blender 5.0 documentation.

**Tools:**

| Tool | Description | Best For |
|------|-------------|----------|
| `search_manual` | General keyword search | Broad queries |
| `search_semantic` | AI embedding search | Natural language questions |
| `search_vdb_workflow` | VDB/OpenVDB specific | Export, caching, volumes |
| `search_python_api` | bpy.* documentation | Scripting questions |
| `search_bpy_operators` | bpy.ops.* reference | Finding operators |
| `search_bpy_types` | bpy.types.* reference | Object properties |
| `search_nodes` | Shader/geometry nodes | Material setup |
| `search_modifiers` | Modifier documentation | Mesh/volume modifiers |
| `search_tutorials` | Learning resources | How-to guides |
| `browse_hierarchy` | Navigate doc structure | Discovery |
| `read_page` | Full page content | Deep reading |
| `list_api_modules` | API module listing | API exploration |

**Critical:** These tools search a **local index of 4,227 pages** from Blender 5.0 documentation. This is authoritative and current.

### 2.3 script-generator

**Purpose:** Generate and modify Blender Python scripts.

**Tools:**

| Tool | Purpose |
|------|---------|
| `generate_script` | Create new script from description |
| `modify_script` | Adjust parameters in existing script |
| `list_templates` | Show available base templates |
| `list_techniques` | Show technique catalog |
| `recommend_technique` | UCB1-based technique selection |
| `record_technique_outcome` | Update technique performance stats |
| `validate_script` | Pre-execution validation |
| `validate_parameters` | Check parameter ranges |
| `get_parameter_ranges` | Show valid Blender parameter ranges |

**Technique Catalog:** Ensures variety by tracking which techniques have been tried:
- `rising_mushroom` - Classic mushroom cloud
- `ground_burst` - Explosive ground impact
- `aerial_detonation` - Mid-air explosion
- `fuel_air_explosion` - Thermobaric effect
- etc.

### 2.4 blender-executor

**Purpose:** Execute Blender scripts via CLI.

**Tools:**

| Tool | Purpose |
|------|---------|
| `execute_blender_script` | Run script with arguments |
| `parse_blender_errors` | Structured error analysis |
| `list_run_outputs` | Show outputs from a run |
| `get_latest_run` | Get most recent execution |
| `list_available_scripts` | Show scripts in project |

**Output:** VDB files, rendered images, logs in `build/vdb_output/<asset_name>/`

### 2.5 asset-evaluator

**Purpose:** Evaluate render quality using multiple metrics.

**Key Tools:**

| Tool | Purpose | When to Use |
|------|---------|-------------|
| `evaluate_vfx_quality` | Primary VFX scoring (0-100) | Every iteration |
| `evaluate_ground_truth` | Compare to reference dataset | For sun/star effects |
| `analyze_texture_procedural` | Detect procedural artifacts | When texture looks synthetic |
| `analyze_prominence_shapes` | Detect cat-ear artifacts | For solar prominences |
| `compare_vfx_iterations` | A/B comparison | Comparing versions |
| `extract_vfx_diagnostics` | Raw diagnostic data | Debugging |

**Quality Dimensions:**
- Brightness (dynamic range, not too dark/bright)
- Color (warm ratio, appropriate for effect type)
- Coverage (fills frame appropriately)
- Structure (has detail, not flat/blobby)

### 2.6 experiment-tracker

**Purpose:** Learn from iterations to improve future runs.

**Tools:**

| Tool | Purpose |
|------|---------|
| `start_experiment_session` | Begin tracking new asset |
| `record_baseline` | Save starting state |
| `record_experiment_result` | Log what was tried and outcome |
| `get_warnings_before_change` | Check for known gotchas |
| `suggest_experiments` | Get ranked suggestions |
| `query_knowledge_base` | Search accumulated learnings |
| `add_manual_learning` | Add human knowledge |

**Knowledge Base:** Persists learnings across sessions. Example:
```json
{
  "parameter": "domain_scale",
  "rule": "Always adjust emitter position when scaling domain",
  "warning": "Scaling domain without moving emitter causes clipping"
}
```

### 2.7 iteration-controller

**Purpose:** Orchestrate the full create-evaluate-improve loop.

**Tools:**

| Tool | Purpose |
|------|---------|
| `create_asset` | Full automated pipeline |
| `run_iteration` | Single iteration step |
| `diagnose_vfx_issues` | Interpret evaluation results |
| `get_next_iteration_params` | Calculate parameter adjustments |
| `save_iteration_state` | Checkpoint for resumption |
| `load_iteration_state` | Resume from checkpoint |
| `list_orchestration_sessions` | Show all sessions |

### 2.8 blender-orchestrator

**Purpose:** High-level session management with trust scoring.

**Tools:**

| Tool | Purpose |
|------|---------|
| `get_status` | Current trust score and limits |
| `create_asset` | Start new asset session |
| `resume_session` | Continue paused session |
| `list_sessions` | Show all sessions |

**Trust Score:** 0.0-1.0, increases with successful iterations, decreases with failures.

---

## 3. Correct Asset Creation Workflow

### 3.1 The Golden Path (ALWAYS Follow This)

```
Step 1: ALWAYS start with blender-librarian
        ↓
Step 2: Use script-generator with validated parameters
        ↓
Step 3: Execute with blender-executor
        ↓
Step 4: Evaluate with asset-evaluator
        ↓
Step 5: Record results with experiment-tracker
        ↓
Step 6: If not passing, get fixes from blender-librarian
        ↓
Step 7: Loop until quality threshold met
```

### 3.2 Detailed Workflow

#### Step 1: Research Phase (MANDATORY)

**DO NOT SKIP THIS STEP.** This is where v1.x failed.

```python
# Use blender-librarian to search documentation
mcp__blender-librarian__get_modification_advice(
    effect_type="sun",
    issues='["needs limb darkening", "edges too sharp"]',
    current_params='{"temperature": 5778}',
    evaluator_scores='{}'
)
```

This will:
1. Call GPT-5.2 with your query
2. GPT-5.2 searches blender-manual (semantic, API, tutorials)
3. Returns validated parameter modifications with doc references

#### Step 2: Script Generation

```python
# Generate script using validated parameters
mcp__script-generator__generate_script(
    effect_type="pyro",
    description="Rising mushroom cloud with orange flames",
    output_name="explosion_v1",
    resolution=96,
    frame_end=50,
    technique_name="rising_mushroom"  # From technique catalog
)

# ALWAYS validate before execution
mcp__script-generator__validate_script(
    script_path="assets/blender_scripts/generated/explosion_v1.py"
)
```

#### Step 3: Execution

```python
mcp__blender-executor__execute_blender_script(
    script_path="assets/blender_scripts/generated/explosion_v1.py",
    script_args={"--bake": "1", "--resolution": "96"},
    output_dir="build/vdb_output/explosion_v1"
)
```

#### Step 4: Evaluation

```python
# Primary evaluation
mcp__asset-evaluator__evaluate_vfx_quality(
    image_path="build/vdb_output/explosion_v1/render_0050.png",
    effect_type="explosion"
)

# For sun/star effects, also use ground truth
mcp__asset-evaluator__evaluate_ground_truth(
    image_path="build/vdb_output/sun_v1/render_0060.png",
    effect_type="sun",
    pass_threshold=0.65
)
```

#### Step 5: Record Results

```python
mcp__experiment-tracker__record_experiment_result(
    hypothesis="Increasing flame_smoke improves density",
    issue_addressed="render too transparent",
    result_params='{"flame_smoke": 2.0}',
    result_scores='{"composite_score": 72}',
    result_render="build/vdb_output/explosion_v1/render_0050.png",
    result_script="assets/blender_scripts/generated/explosion_v1.py",
    success=True,
    observed_effects='["denser smoke", "slightly darker"]',
    learnings='["flame_smoke > 1.5 gives good density"]',
    warnings='["too high causes performance drop"]'
)
```

#### Step 6: Iterate

If quality < threshold, go back to Step 1 with the new issues.

### 3.3 Anti-Patterns (What NOT to Do)

#### WRONG: Guessing Blender Parameters

```python
# BAD - Using hardcoded values from stale knowledge
mcp__script-generator__modify_script(
    script_path="...",
    modifications={"burning_rate": 5.0}  # Wrong! Max is 4.0
)
```

#### CORRECT: Validate Parameters First

```python
# GOOD - Check valid ranges
mcp__script-generator__validate_parameters(
    params={"burning_rate": 5.0}
)
# Returns: {"valid": false, "warnings": ["burning_rate clamped to 4.0"]}

# GOOD - Or look up in documentation
mcp__blender-manual__search_bpy_types(typename="FluidFlowSettings", limit=5)
```

#### WRONG: Skipping Documentation Lookup

```python
# BAD - Going straight to script generation without research
mcp__script-generator__generate_script(
    effect_type="sun",
    description="realistic sun with limb darkening"
    # No prior research on HOW to achieve limb darkening!
)
```

#### CORRECT: Research First

```python
# GOOD - Ask blender-librarian first
mcp__blender-librarian__get_modification_advice(
    effect_type="sun",
    issues='["needs limb darkening"]',
    current_params='{}'
)
# Returns documentation-backed approach using Layer Weight node, etc.
```

#### WRONG: Not Recording Experiments

```python
# BAD - Running iterations without tracking
for i in range(10):
    modify_script(...)
    execute(...)
    evaluate(...)
    # No record of what was tried!
```

#### CORRECT: Track Everything

```python
# GOOD - Use experiment-tracker
mcp__experiment-tracker__start_experiment_session(
    asset_name="sun_v1",
    effect_type="sun",
    description="Realistic sun with prominences"
)

# After each iteration:
mcp__experiment-tracker__record_experiment_result(...)
```

---

## 4. Best Practices

### 4.1 Documentation-First Development

**Rule:** Never modify Blender parameters without documentation backing.

**Process:**
1. Identify the issue (e.g., "render too dark")
2. Query blender-librarian for solutions
3. Verify suggested parameters against blender-manual
4. Apply changes with confidence

### 4.2 Technique Variety

**Problem:** Trying the same technique repeatedly leads to local optima.

**Solution:** Use the technique catalog with UCB1 selection:

```python
# Get recommended technique (explores untried options)
mcp__script-generator__recommend_technique(
    effect_type="pyro",
    description="bright explosion with shockwave"
)

# After completion, record outcome for learning
mcp__script-generator__record_technique_outcome(
    technique_name="aerial_detonation",
    effect_type="pyro",
    success=True,
    final_score=78.5,
    iterations=3
)
```

### 4.3 Quality Gate Thresholds

| Metric | Pass Threshold | Notes |
|--------|----------------|-------|
| VFX Quality Score | ≥60 | Primary metric |
| Ground Truth (sun) | ≥0.65 | Compare to NASA footage |
| Texture Procedural | <50 | Lower = more natural |
| No Critical Issues | 0 | Must fix all critical |

### 4.4 Iteration Limits

- **Max iterations per asset:** 10
- **If stuck after 5 iterations:** Try different technique
- **If stuck after 10 iterations:** Escalate to human review

### 4.5 Budget Management

GPT-5.2 costs ~$0.04 per blender-librarian query. Monitor with:

```python
mcp__blender-librarian__get_budget_status()
```

Monthly budget: $10 (configurable in `.env`)

### 4.6 Checkpointing for Long Sessions

Save state regularly to survive context limits:

```python
mcp__iteration-controller__save_iteration_state(
    session_id="sun_prominences_v2",
    asset_name="sun_prominences",
    effect_type="sun",
    current_iteration=5,
    best_score=72.5,
    best_iteration=4,
    parameters_current='{"flame_max_temp": 3000}',
    issues_current='["needs more structure"]',
    next_action="adjust_params"
)
```

Resume in new session:

```python
mcp__iteration-controller__load_iteration_state(session_id="sun_prominences_v2")
```

---

## 5. Troubleshooting

### 5.1 "Parameter out of range" Errors

**Cause:** Using values outside Blender's valid ranges.

**Fix:**
```python
# Check valid ranges
mcp__script-generator__get_parameter_ranges()

# Or validate specific params
mcp__script-generator__validate_parameters(
    params={"burning_rate": 5.0, "flame_smoke": 3.5}
)
```

### 5.2 Stale Knowledge Problems

**Symptoms:**
- Suggested API doesn't exist
- Parameters have wrong effect
- Deprecated features referenced

**Fix:** Always route through blender-librarian which queries live documentation:
```python
mcp__blender-librarian__get_modification_advice(
    effect_type="...",
    issues='[...]'
)
```

### 5.3 Evaluation Scores Plateau

**Symptoms:** Score stuck at 50-60 despite iterations.

**Fix:**
1. Check for procedural texture artifacts:
   ```python
   mcp__asset-evaluator__analyze_texture_procedural(render_path="...")
   ```

2. Try fundamentally different technique:
   ```python
   mcp__script-generator__recommend_technique(
       effect_type="pyro",
       description="...",
       prefer_untried=True  # Force exploration
   )
   ```

3. Get alternative approaches:
   ```python
   mcp__asset-evaluator__get_alternative_approaches(
       current_approach="sphere_emitter",
       scores_history='[{"iteration": 1, "score": 52}, ...]'
   )
   ```

### 5.4 Blender Execution Failures

**Symptoms:** Script crashes or produces no output.

**Fix:**
1. Parse errors for structured diagnosis:
   ```python
   mcp__blender-executor__parse_blender_errors(
       stderr="...",
       stdout="..."
   )
   ```

2. Common issues:
   - `context is incorrect` → Need to set active object/mode
   - `AttributeError` → API changed in Blender 5.0, check docs
   - `KeyError` → Object/collection doesn't exist

### 5.5 Vision API Budget Exceeded

**Symptoms:** `get_modification_advice` returns budget error.

**Fix:**
1. Check status: `mcp__blender-librarian__get_budget_status()`
2. Use playbook (free): Previously learned fixes are cached
3. Use blender-manual directly (free): No GPT-5.2 cost

---

## 6. Quick Reference Commands

### Start New Asset

```python
# 1. Research
advice = mcp__blender-librarian__get_modification_advice(
    effect_type="explosion",
    issues='["needs brighter flames", "lacks smoke"]'
)

# 2. Generate
script = mcp__script-generator__generate_script(
    effect_type="pyro",
    description="bright mushroom cloud explosion",
    output_name="explosion_v1"
)

# 3. Validate
validation = mcp__script-generator__validate_script(script_path=script.path)

# 4. Execute
result = mcp__blender-executor__execute_blender_script(
    script_path=script.path,
    output_dir="build/vdb_output/explosion_v1"
)

# 5. Evaluate
quality = mcp__asset-evaluator__evaluate_vfx_quality(
    image_path="build/vdb_output/explosion_v1/render_0050.png",
    effect_type="explosion"
)
```

### Resume Session

```python
state = mcp__iteration-controller__load_iteration_state(session_id="...")
# Continue from state.next_action
```

### Check System Status

```python
mcp__blender-librarian__get_budget_status()
mcp__blender-orchestrator__get_status()
mcp__iteration-controller__list_orchestration_sessions()
mcp__experiment-tracker__get_experiment_statistics()
```

### Search Documentation

```python
# Natural language
mcp__blender-manual__search_semantic(query="how to create limb darkening")

# API reference
mcp__blender-manual__search_bpy_types(typename="FluidDomainSettings")

# VDB specific
mcp__blender-manual__search_vdb_workflow(query="export openvdb cache")
```

---

## Appendix A: MCP Server Locations

| Server | Path | Port |
|--------|------|------|
| blender-librarian | `agents/blender-librarian/` | stdio |
| blender-manual | `agents/blender-manual/` | stdio |
| script-generator | `agents/script-generator/` | stdio |
| blender-executor | `agents/blender-executor/` | stdio |
| asset-evaluator | `agents/asset-evaluator/` | stdio |
| experiment-tracker | `agents/experiment-tracker/` | stdio |
| iteration-controller | `agents/iteration-controller/` | stdio |
| blender-orchestrator | `agents/blender-orchestrator/` | stdio |

---

## Appendix B: Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2024-12 | Initial pipeline (broken doc lookup) |
| 2.0 | 2025-01-05 | GPT-5.2 tool calling for mandatory doc lookup |

---

## Appendix C: Related Documentation

- `docs/MULTI_AGENT_IMPROVEMENT_PLAN_V3.md` - Architecture design
- `docs/BLENDER_LIBRARIAN_AGENT_DESIGN_GPT52.md` - GPT-5.2 integration design
- `plans/feat-gpt52-tool-calling-agents-sdk-v4.md` - Implementation plan
- `docs/openai_api_documentation/` - OpenAI Responses API reference

---

**Document Maintainer:** Claude Code
**Last Updated:** 2025-01-05
