# Blender VFX Orchestrator Specification

**Version:** 1.0
**Last Updated:** 2026-01-03
**Status:** Phases 0, 0.5, 1 Complete | Phases 2-5 Planned

---

## Executive Summary

The Blender VFX Orchestrator is an autonomous multi-agent system that generates high-quality NanoVDB volumetric assets (explosions, fire, nebulae, sun effects) through iterative improvement with ML-based quality evaluation. It uses Claude Code as the agent runtime with 6 specialized MCP servers coordinating the workflow.

**Key Capabilities:**
- Autonomous asset generation from text descriptions
- ML-powered quality evaluation (VFX diagnostics, ground truth comparison)
- Iterative improvement until quality thresholds are met
- Knowledge accumulation across sessions
- Session persistence and resumption

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLAUDE CODE                                     │
│                         (Agent Runtime + Skill)                              │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │                    blender-orchestrator SKILL                          │  │
│  │  - Iteration tracking (current_iteration, best_score, etc.)           │  │
│  │  - Circuit breakers (MAX_ITERATIONS, NO_IMPROVEMENT, etc.)            │  │
│  │  - Quality decision tree (VFX → LPIPS → CLIP)                         │  │
│  │  - Knowledge base consultation (mandatory)                             │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MCP SERVER LAYER                                   │
├─────────────────┬─────────────────┬─────────────────┬───────────────────────┤
│ blender-        │ script-         │ blender-        │ asset-                │
│ orchestrator    │ generator       │ executor        │ evaluator             │
│ (coordinator)   │ (script gen)    │ (Blender CLI)   │ (ML quality)          │
├─────────────────┼─────────────────┼─────────────────┼───────────────────────┤
│ experiment-     │ iteration-      │ blender-        │                       │
│ tracker         │ controller      │ manual          │                       │
│ (knowledge)     │ (state mgmt)    │ (documentation) │                       │
└─────────────────┴─────────────────┴─────────────────┴───────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           OUTPUT LAYER                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│  VDB Files: build/vdb_output/<asset_name>/                                  │
│  Renders:   build/vdb_output/<asset_name>/render_*.png                      │
│  Scripts:   assets/blender_scripts/generated/<asset_name>.py                │
│  Sessions:  build/orchestrator_state/<session_id>.json                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Workflow Overview

### High-Level Flow

```
User Request → Initialize Session → Generate Script → Execute Blender
                                                           ↓
                    ┌──────────────────────────────────────┘
                    ▼
              Evaluate Quality ──────→ Quality Passed? ──YES──→ Report Success
                    │                        │
                    │                       NO
                    │                        │
                    ▼                        ▼
              Circuit Breaker? ──YES──→ Stop & Report Best
                    │
                   NO
                    │
                    ▼
              Consult Knowledge Base → Diagnose Issues → Modify Script
                    │
                    └──────────────────→ Execute Blender (loop)
```

### Detailed Stage-by-Stage

#### Stage 1: Initialize Session
1. Parse user request for: `asset_name`, `effect_type`, `description`, `reference_path`, `semantic_query`
2. Initialize tracking variables:
   - `current_iteration = 0`
   - `max_iterations = 10`
   - `best_score = 0`
   - `iterations_without_improvement = 0`
3. Check for existing session to resume
4. Preload relevant knowledge from experiment-tracker

#### Stage 2: Generate Script
1. Query available techniques: `mcp__script-generator__list_techniques()`
2. Generate Blender Python script: `mcp__script-generator__generate_script()`
3. Validate parameters against Blender 5.0 API ranges
4. Store script path for execution

#### Stage 3: Execute Blender
1. **Check circuit breakers** (see Circuit Breakers section)
2. Increment `current_iteration`
3. Execute simulation: `mcp__blender-executor__execute_blender_script()`
4. On failure: parse errors, apply fixes, retry (max 3 attempts)
5. On success: locate output files in `build/vdb_output/<asset_name>/`

#### Stage 4: Evaluate Quality
1. **Primary:** VFX Quality Score (no reference needed)
   ```
   mcp__asset-evaluator__evaluate_vfx_quality(image_path, effect_type)
   ```
   - Returns 0-100 composite score
   - Checks: brightness, color, structure, coverage

2. **Secondary (if VFX ≥ 40):**
   - LPIPS perceptual similarity (if reference provided)
   - CLIP semantic match (if semantic query provided)

3. **Decision Matrix:**

   | VFX Score | LPIPS | CLIP | Decision |
   |-----------|-------|------|----------|
   | ≥60 | <0.35 | >0.60 | ACCEPT |
   | ≥60 | any | any | ACCEPT (with notes) |
   | 40-59 | <0.35 | >0.60 | ITERATE (close) |
   | 40-59 | any | any | ITERATE |
   | <40 | any | any | REJECT |

#### Stage 5: Decide Next Action
- **Quality Passed:** Report success, record learning, end session
- **Quality Failed + No Circuit Breaker:** Consult knowledge base → diagnose → modify script → loop
- **Circuit Breaker Triggered:** Stop, report best result, save state

#### Stage 6: Record Learning
After every iteration, record to knowledge base:
```
mcp__experiment-tracker__record_experiment_result(
    hypothesis, issue_addressed, result_params, result_scores,
    success, observed_effects, learnings, warnings
)
```

---

## Circuit Breakers (Hard Stops)

| Breaker | Condition | Action |
|---------|-----------|--------|
| MAX_ITERATIONS | `current_iteration >= 10` | STOP, report best result |
| MAX_WALL_TIME | `elapsed > 30 minutes` | STOP, save state for resume |
| NO_IMPROVEMENT | `iterations_without_improvement >= 3` | STOP, local optimum reached |
| QUALITY_FLOOR | `consecutive_low_scores >= 2` (score < 40) | PAUSE, request human review |
| BLENDER_FAILURES | `consecutive_blender_failures >= 2` | STOP, script has fundamental issue |

---

## MCP Agent Specifications

### 1. blender-orchestrator (Coordinator)

**Purpose:** Top-level coordinator exposing workflow to Claude Code

**Tools:**
| Tool | Purpose |
|------|---------|
| `get_status` | Check trust score, autonomy level, active sessions |
| `list_sessions` | List all VFX generation sessions |
| `create_asset` | Start new asset generation |
| `resume_session` | Resume interrupted session |

**Key Files:**
- `orchestrator.py` - Main workflow logic
- `mcp_server.py` - FastMCP server exposing tools
- `health_check.py` - MCP server health verification
- `workflow_tracer.py` - Session trace logging
- `circuit_breakers.py` - Hard stop enforcement

---

### 2. script-generator

**Purpose:** Generate and modify Blender Python scripts

**Tools:**
| Tool | Purpose |
|------|---------|
| `list_templates` | Available template scripts |
| `get_template` | Get template content |
| `analyze_script` | Understand script structure |
| `generate_script` | Create new script from description |
| `modify_script` | Apply parameter changes |
| `list_techniques` | Available pyro techniques |
| `validate_parameters` | Check against Blender 5.0 API |
| `get_parameter_ranges` | Valid parameter ranges |

**Technique Catalog:**
- `rising_mushroom` - Classic nuclear mushroom cloud
- `ground_burst` - Outward horizontal expansion
- `aerial_burst` - Spherical mid-air explosion
- `directed_jet` - Shaped charge directional
- `rolling_fireball` - Tumbling fire mass
- `deflagration` - Slow-burning fuel-air

---

### 3. blender-executor

**Purpose:** Execute Blender scripts via CLI

**Tools:**
| Tool | Purpose |
|------|---------|
| `execute_blender_script` | Run script with args |
| `parse_blender_errors` | Structure errors with fixes |
| `list_run_outputs` | Find VDB/render outputs |
| `get_latest_run` | Most recent execution info |
| `list_available_scripts` | Scripts in project |

**Configuration:**
- Blender executable: `/home/maz3ppa/apps/blender-5.0.1-linux-x64/blender`
- Default timeout: 600 seconds (10 minutes)
- Output: `build/vdb_output/<asset_name>/`

---

### 4. asset-evaluator

**Purpose:** ML-powered quality evaluation

**Primary Tools:**
| Tool | Purpose |
|------|---------|
| `evaluate_vfx_quality` | Standalone VFX score (0-100) |
| `evaluate_ground_truth` | Compare to reference footage distribution |
| `compare_lpips` | Perceptual similarity (~92% human correlation) |
| `compare_clip` | Semantic similarity |
| `analyze_temporal_quality` | Animation consistency |

**Advanced Tools:**
| Tool | Purpose |
|------|---------|
| `extract_vfx_diagnostics` | Detailed feature extraction |
| `compare_vfx_iterations` | Compare two renders |
| `evaluate_solar_features` | Sun-specific (granulation, limb darkening) |
| `analyze_prominence_shapes` | Detect synthetic artifacts |
| `predict_real_or_synthetic` | ML discriminator |

**Quality Thresholds:**
| Metric | Threshold | Interpretation |
|--------|-----------|----------------|
| VFX Quality | ≥60 | Pass |
| LPIPS | <0.35 | Perceptually similar |
| CLIP | >0.60 | Semantically matching |
| Temporal | ≥0.70 | Smooth animation |

---

### 5. experiment-tracker

**Purpose:** Knowledge accumulation and learning

**Tools:**
| Tool | Purpose |
|------|---------|
| `start_experiment_session` | Begin tracking session |
| `record_baseline` | Save initial state |
| `record_experiment_result` | Log iteration outcome |
| `get_warnings_before_change` | Check known pitfalls |
| `suggest_experiments` | Recommend fixes |
| `query_knowledge_base` | Search accumulated knowledge |
| `get_parameter_knowledge` | Rules for specific parameter |
| `add_manual_learning` | Human-provided insights |
| `get_experiment_statistics` | Overall success rates |

**Knowledge Categories:**
- Parameter effects (e.g., "domain_scale increase causes clipping")
- Issue→Fix mappings (e.g., "TOO DARK → increase flame_max_temp")
- Technique performance (success rates per effect type)
- Warnings and gotchas

---

### 6. iteration-controller

**Purpose:** State management and diagnosis

**Tools:**
| Tool | Purpose |
|------|---------|
| `diagnose_vfx_issues` | Parse evaluation, suggest fixes |
| `get_next_iteration_params` | Calculate parameter changes |
| `save_iteration_state` | Persist session checkpoint |
| `load_iteration_state` | Resume from checkpoint |
| `list_orchestration_sessions` | All saved sessions |
| `create_asset` | Full pipeline (generate→execute→evaluate→iterate) |
| `run_iteration` | Single iteration step |
| `get_history` | Session iteration history |
| `compare_iterations` | Score comparison analysis |

---

### 7. blender-manual

**Purpose:** Blender documentation search

**Tools:**
| Tool | Purpose |
|------|---------|
| `search_manual` | General documentation search |
| `search_tutorials` | Learning resources |
| `search_vdb_workflow` | VDB/OpenVDB specific |
| `search_python_api` | bpy.ops, bpy.types |
| `search_nodes` | Shader/geometry nodes |
| `search_modifiers` | Modifier documentation |
| `browse_hierarchy` | Navigate doc structure |
| `read_page` | Full page content |
| `search_semantic` | AI-powered similarity search |

---

## Quality Evaluation Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    EVALUATION PIPELINE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Step 1: VFX Quality (ALWAYS, no reference needed)              │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ evaluate_vfx_quality(image_path, effect_type)           │    │
│  │ → Returns: composite_score (0-100), issues[], passed    │    │
│  └─────────────────────────────────────────────────────────┘    │
│           │                                                      │
│           ▼                                                      │
│  ┌────────────────────┐                                         │
│  │ Score < 40?        │──YES──→ REJECT (skip secondary)         │
│  └────────────────────┘                                         │
│           │ NO                                                   │
│           ▼                                                      │
│  Step 2: Secondary Metrics (if reference/query provided)        │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ compare_lpips(render, reference)  → 0-1 (lower=similar) │    │
│  │ compare_clip(render, query)       → 0-1 (higher=match)  │    │
│  └─────────────────────────────────────────────────────────┘    │
│           │                                                      │
│           ▼                                                      │
│  Step 3: Decision Matrix                                         │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ VFX ≥60 AND (LPIPS <0.35 OR CLIP >0.60) → ACCEPT        │    │
│  │ VFX 40-59                                → ITERATE       │    │
│  │ VFX <40                                  → REJECT        │    │
│  └─────────────────────────────────────────────────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Knowledge Base Integration

### Mandatory Consultation (Before Parameter Changes)

```
1. Check Warnings:
   mcp__experiment-tracker__get_warnings_before_change(
       parameter="domain_scale",
       change_type="increase"
   )
   → Returns warnings with severity, suggested mitigations

2. Get Suggestions:
   mcp__experiment-tracker__suggest_experiments(
       issue="clipping at top edge",
       current_params={...},
       current_scores={...}
   )
   → Returns ranked suggestions with confidence scores

3. Apply:
   - If confidence > 0.6: USE knowledge base suggestion
   - If confidence ≤ 0.6: Fall back to heuristic fix
```

### Learning After Each Iteration

```
mcp__experiment-tracker__record_experiment_result(
    hypothesis="Increasing domain_scale will fix clipping",
    issue_addressed="clipping at top edge",
    result_params='{"domain_scale": 3.0, ...}',
    result_scores='{"vfx_score": 65, ...}',
    success=true,
    observed_effects='["Clipping fixed", "Density reduced"]',
    learnings='["domain_scale 3.0 eliminates clipping for mushroom clouds"]',
    warnings='["May need to increase density to compensate"]'
)
```

---

## Trust & Autonomy System

| Level | Trust Score | Behavior |
|-------|-------------|----------|
| Supervised | 0.0 - 0.3 | Requires approval for each action |
| Guided | 0.3 - 0.6 | Can execute, reports decisions |
| Autonomous | 0.6 - 0.8 | Full autonomy within guardrails |
| Trusted | 0.8 - 1.0 | Extended limits, minimal oversight |

Trust score increases with successful sessions, decreases with failures.

---

## Common Issues and Fixes

| Issue | Diagnosis | Fix |
|-------|-----------|-----|
| TOO DARK | Low brightness score | Increase `flame_max_temp` by 500-1000K |
| NO STRUCTURE | Low edge density | Increase `turbulence` to 0.5-0.8 |
| WRONG COLOR | Color warmth mismatch | Adjust `flame_max_temp` for target |
| CLIPPING | Content outside domain | Increase `domain_scale` AND reposition emitter |
| SPARSE | Low coverage score | Increase `density_multiplier` |
| FLAT | No depth variation | Increase `noise_scale` and `vorticity` |
| FLICKERING | Temporal instability | Reduce `turbulence`, increase temporal smoothing |
| BLENDER SEGFAULT DURING BAKE | Headless Blender crash (exit 139) early in setup | Avoid touching unstable RNA props in headless runs (e.g., `openvdb_cache_compress_type`); avoid `bpy.ops.fluid.free_all()`; simplify ops order |
| CACHE OUTPUT WAY TOO LONG | Mantaflow caches default to ~250 frames even when `scene.frame_end` is small | **Always set** `domain_settings.cache_frame_start/end` (and offset=0) to match `frame_start/end` |
| RENDER DOESN’T MATCH BAKED CACHE | Render-only pass shows empty/black volume; evaluation says “brightness 0” | In render-only runs set `cache_type='REPLAY'` and ensure `cache_directory` points at baked cache folder; validate output is non-empty before evaluation |
| GROUND-TRUTH TOOLS ERROR (JSON) | `Object of type bool is not JSON serializable` | Ensure evaluation outputs contain pure Python types (convert numpy scalars); add a “tool result serialization” layer or restart MCP server after code updates |

---

## Known Failure Modes (Observed During Sun/Star Asset Runs)

These are real issues encountered while generating sun-like volumetrics for NASA ground-truth comparison. They should be treated as **guardrails** in the orchestrator loop.

### 1) Blender headless segfault (exit 139) during bake
- **Symptom**: Blender writes `/tmp/blender.crash.txt` and exits 139 shortly after scene/domain setup.
- **Most likely trigger**: touching unstable/incorrect RNA enum properties in some builds (we observed RNA warnings around `openvdb_cache_compress_type` preceding crashes).
- **Mitigation**:
  - Prefer “leave defaults alone” for compression/precision props unless verified at runtime.
  - Avoid `bpy.ops.fluid.free_all()` in headless runs unless strictly needed.
  - If an enum exists, only set it via safe helper that checks supported enum keys.

### 2) Mantaflow cache frame range mismatch (250 frames baked unexpectedly)
- **Symptom**: even with `--frame_end 20`, the bake produces `fluid_data_0001..0250`.
- **Root cause**: Mantaflow uses **cache frame range** (domain settings) independent of scene frame range.
- **Mitigation**:
  - Always set:
    - `domain_settings.cache_frame_start = frame_start`
    - `domain_settings.cache_frame_end = frame_end`
    - `domain_settings.cache_frame_offset = 0`
  - Add a post-bake validation step:
    - expected frame count vs actual `.vdb` count.

### 3) Render-only pass not loading caches
- **Symptom**: render output exists but represents an empty/black volume; evaluation becomes nonsense.
- **Root cause**: render-only scripts often rebuild the scene but do not correctly switch to `cache_type='REPLAY'` and/or point to the correct `cache_directory`.
- **Mitigation**:
  - When `--bake 0`, set `cache_type='REPLAY'`.
  - Add a “non-empty render” gate before evaluation:
    - if coverage/brightness indicate near-empty frame, rerender with corrected cache settings instead of iterating.

### 4) Evaluation tool serialization failures
- **Symptom**: ground-truth tools return `"Object of type bool is not JSON serializable"`.
- **Root cause**: numpy scalar booleans (e.g., `np.bool_`) leaking into result dicts without conversion.
- **Mitigation**:
  - Centralize a `convert_np()` serializer in asset-evaluator before `json.dumps`.
  - Operational: MCP servers are long-lived; after patching server code, a restart may be required for changes to take effect.

## Planned Improvements (Phases 2-5)

### Phase 2: Intelligent Technique Selection
**Status:** Planned

| Task | Purpose |
|------|---------|
| Technique performance table | Track success rates per technique |
| Exploration/exploitation | 80% best techniques, 20% exploration |
| `recommend_technique` tool | Ranked suggestions with confidence |

**Expected Impact:** Reduce average iterations by selecting proven techniques first.

---

### Phase 3: Knowledge Base Improvements
**Status:** Planned

| Task | Purpose |
|------|---------|
| Automatic rule extraction | Generate rules from experiment history |
| Cross-session preloading | Load relevant knowledge at session start |
| Confidence decay | Reduce confidence of old, untested rules |

**Expected Impact:** Better suggestions over time as knowledge accumulates.

---

### Phase 4: External Research Integration
**Status:** Planned

| Task | Purpose |
|------|---------|
| `research_vfx_technique` tool | Search Blender docs + web |
| Technique import | Add discovered techniques to catalog |
| Reference image organization | Tagged collection for evaluation |

**Expected Impact:** Escape local optima by discovering new approaches.

---

### Phase 5: Session Resumption Reliability
**Status:** Planned

| Task | Purpose |
|------|---------|
| State schema versioning | Migration support for format changes |
| File existence validation | Verify renders/scripts exist on resume |
| Multi-checkpoint rollback | Keep last 3 checkpoints per session |

**Expected Impact:** >90% session resume success rate.

---

## Usage Examples

### Basic Asset Generation

```
User: Create a bright orange mushroom cloud explosion

System executes:
1. script-generator: Generate explosion script with rising_mushroom technique
2. blender-executor: Run Blender simulation
3. asset-evaluator: Evaluate VFX quality → Score: 45 (needs improvement)
4. experiment-tracker: Get suggestions for "TOO DARK" issue
5. script-generator: Modify script (increase flame_max_temp)
6. blender-executor: Re-run simulation
7. asset-evaluator: Evaluate → Score: 68 (PASS)
8. experiment-tracker: Record successful learning

Output:
- VDB: build/vdb_output/mushroom_cloud_v1/
- Render: build/vdb_output/mushroom_cloud_v1/render_0025.png
- Script: assets/blender_scripts/generated/mushroom_cloud_v1.py
```

### With Reference Image

```
User: Create a sun surface effect matching assets/reference_images/star/frame_00639.jpg

System executes:
1. script-generator: Generate sun script
2. blender-executor: Run simulation
3. asset-evaluator:
   - evaluate_vfx_quality → 52
   - compare_lpips(render, reference) → 0.42
   - evaluate_ground_truth → 58%
4. Iterate with knowledge base guidance until:
   - VFX ≥ 60 AND LPIPS < 0.35 AND ground_truth ≥ 65%
```

### Resume Interrupted Session

```
User: Resume the sun_prominences session

System executes:
1. iteration-controller: load_iteration_state("sun_prominences")
   → Restores: iteration=3, best_score=55, parameters, issues
2. Continue from Stage 3 (Execute Blender)
```

---

## File Locations

| Category | Path |
|----------|------|
| MCP Servers | `agents/<server-name>/` |
| Skill Definition | `.claude/skills/blender-orchestrator/SKILL.md` |
| Blender Scripts | `assets/blender_scripts/generated/` |
| Script Templates | `assets/blender_scripts/GPT-5.2/` |
| VDB Output | `build/vdb_output/<asset_name>/` |
| Session State | `build/orchestrator_state/` |
| Knowledge Base | `build/experiment_tracker/` |
| Reference Images | `assets/reference_images/` |
| Documentation | `docs/` |

---

## Appendix: Tracking Variables

| Variable | Type | Initial | Description |
|----------|------|---------|-------------|
| `current_iteration` | int | 0 | Increments each loop |
| `max_iterations` | int | 10 | Hard limit |
| `best_score` | float | 0 | Highest VFX score |
| `best_iteration` | int | 0 | Which iteration achieved best |
| `best_render_path` | str | "" | Path to best render |
| `best_script_path` | str | "" | Path to best script |
| `iterations_without_improvement` | int | 0 | Resets on improvement |
| `consecutive_blender_failures` | int | 0 | Resets on success |
| `consecutive_low_scores` | int | 0 | Scores < 40 |
| `session_start_time` | datetime | now() | For wall time limit |

---

**Document maintained by:** Claude Code sessions
**Related documents:**
- `MULTI_AGENT_IMPROVEMENT_PLAN_V2.md` - Improvement roadmap
- `MULTI_AGENT_PIPELINE_ANALYSIS.md` - System analysis
- `.claude/skills/blender-orchestrator/SKILL.md` - Skill definition
