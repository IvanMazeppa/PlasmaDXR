# Blender VFX Asset Generation Pipeline

## System Overview

This is a **self-improving, ML-evaluated NanoVDB asset generation pipeline** for creating volumetric VFX assets (explosions, fire, smoke, nebulae) that can be imported into PlasmaDX-Clean's real-time renderer.

### Core Loop

```
┌─────────────────────────────────────────────────────────────────┐
│                    ITERATIVE IMPROVEMENT LOOP                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Script     │───>│   Blender    │───>│   Asset      │       │
│  │  Generator   │    │   Executor   │    │  Evaluator   │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         ^                                       │                │
│         │                                       │                │
│         │         ┌──────────────┐              │                │
│         └─────────│  Experiment  │<─────────────┘                │
│                   │   Tracker    │                               │
│                   └──────────────┘                               │
│                                                                  │
│  Orchestrated by: Iteration Controller                          │
│  Knowledge Base: SQLite (learns from each iteration)            │
└─────────────────────────────────────────────────────────────────┘
```

---

## MCP Servers (Agents)

### 1. Script Generator (`agents/script-generator/`)
**Status:** ✅ OPERATIONAL

**Purpose:** Generates Blender Python scripts for VFX simulations.

**Tools:**
- `list_templates` - List available template scripts (pyro, liquid, mesh)
- `get_template` - Get full content of a template
- `analyze_script` - Analyze script structure and parameters
- `generate_script` - Generate new script from description
- `modify_script` - Modify existing script based on feedback

**Integration Gap:** Does NOT currently consult Blender Manual MCP for API validation.

---

### 2. Blender Executor (`agents/blender-executor/`)
**Status:** ✅ OPERATIONAL

**Purpose:** Executes Blender scripts via CLI with full output capture.

**Tools:**
- `execute_blender_script` - Run script with args, capture output
- `parse_blender_errors` - Parse errors into structured format with fixes
- `list_run_outputs` - List VDBs, renders, logs from a run
- `get_latest_run` - Get most recent execution info
- `list_available_scripts` - List scripts in project

**Blender Path:** `/home/maz3ppa/apps/blender-5.0.1-linux-x64/blender`

---

### 3. Asset Evaluator (`agents/asset-evaluator/`)
**Status:** ✅ OPERATIONAL

**Purpose:** ML-based quality evaluation of rendered frames.

**Tools:**
- `compare_lpips` - Perceptual similarity (lower = more similar, ~92% human correlation)
- `compare_clip` - Semantic similarity to text description
- `evaluate_render` - Combined scoring with pass/fail thresholds
- `enhanced_evaluate` - Multi-modal with LAION Aesthetics + ImageReward + VLM diagnostics
- `multi_prompt_clip_analysis` - Fine-grained quality gradient (avoids CLIP plateau)
- `analyze_temporal_quality` - Detect flickering across animation frames
- `get_alternative_approaches` - Suggest alternatives when stuck in local optima
- `find_reference_images` - Search reference images by keyword
- `list_recent_renders` - List recent renders

**Quality Thresholds:**
- LPIPS: < 0.35 (fair), < 0.2 (good), < 0.1 (excellent)
- CLIP: > 0.60 (fair), > 0.70 (good), > 0.85 (excellent)

---

### 4. Iteration Controller (`agents/iteration-controller/`)
**Status:** ✅ OPERATIONAL

**Purpose:** Orchestrates the full self-improving loop.

**Tools:**
- `create_asset` - Full pipeline: generate → execute → evaluate → improve → repeat
- `run_iteration` - Execute one cycle
- `get_history` - View iteration history for an asset
- `list_sessions` - List all sessions (in_progress, passed, failed)
- `compare_iterations` - Compare quality across attempts

**Parameters:**
- `max_iterations`: default 5
- `lpips_threshold`: default 0.35
- `clip_threshold`: default 0.60

---

### 5. Experiment Tracker (`agents/experiment-tracker/`)
**Status:** ✅ OPERATIONAL (just fixed)

**Purpose:** Tracks parameter changes and learns from outcomes.

**Tools:**
- `start_experiment_session` - Begin tracking for an asset
- `record_baseline` - Record current state before changes
- `record_experiment_result` - Record outcome with learnings
- `get_warnings_before_change` - Check knowledge base before modifying params
- `suggest_experiments` - Get ranked suggestions based on past successes
- `query_knowledge_base` - Search for relevant knowledge
- `get_parameter_knowledge` - Get rules/warnings for specific parameter
- `add_manual_learning` - Add rule to knowledge base
- `add_human_feedback` - Rate experiments (1-5)
- `get_experiment_statistics` - Overall success rates
- `get_session_report` - Session summary
- `end_experiment_session` - Close session

**Knowledge Base:** SQLite at `agents/experiment-tracker/experiments.db`

**Learned Rules (current):**
1. `domain_scale`: Increase domain size but keep location at z=0 (symmetric expansion)
2. `emitter_location_z`: Lower emitter for rising effects to prevent temporal clipping

---

### 6. Blender Manual (`agents/blender-manual/`)
**Status:** ✅ OPERATIONAL but UNDERUTILIZED

**Purpose:** Blender 5.0 documentation and Python API reference.

**Tools:**
- `search_manual` - General search across manual
- `search_tutorials` - Find learning resources
- `browse_hierarchy` - Browse manual structure
- `search_vdb_workflow` - VDB/OpenVDB specific searches
- `search_python_api` - bpy.ops, bpy.types, bpy.data
- `search_nodes` - Shader/compositor/geometry nodes
- `search_modifiers` - Modifier documentation
- `read_page` - Get full page content
- `list_api_modules` - List available API modules
- `search_bpy_operators` - Search operators by category
- `search_bpy_types` - Search types by name
- `search_semantic` - AI embedding-based semantic search

**Key Value:** Contains exact parameter ranges (e.g., `burning_rate: float in [0.01, 4], default 0.75`) that could validate generated scripts.

---

## Other Supporting Agents

### DXR-Related (PlasmaDX Renderer)
- `dxr-image-quality-analyst` - Screenshot comparison, LPIPS, visual quality assessment
- `dxr-shadow-engineer` - Shadow technique research and shader generation
- `dxr-volumetric-pyro-specialist` - Pyrotechnic effect design
- `gaussian-analyzer` - 3D Gaussian particle analysis
- `material-system-engineer` - Particle material system design

### Debugging/Analysis
- `pix-debug` - PIX GPU capture analysis, buffer validation
- `log-analysis-rag` - Log ingestion and diagnostic queries
- `path-and-probe` - Probe grid optimization for indirect lighting

### Orchestration
- `mission-control` - Strategic orchestrator for multi-agent coordination
- `rendering-council` - DXR rendering decisions

---

## Current Gaps & Unfinished Areas

### 1. **Blender Manual Integration** ❌ NOT INTEGRATED
**Problem:** Script Generator creates scripts using training data alone. Does NOT validate:
- Parameter ranges (e.g., `burning_rate` must be in [0.01, 4])
- API existence in Blender 5.0 (breaking changes from 4.x)
- Deprecated attributes (e.g., `Material.use_nodes` removed in Blender 6.0)

**Solution:** Script Generator should query `blender-manual` before generating:
```python
# Before setting settings.burning_rate = 1.5
# Query: mcp__blender-manual__search_python_api("burning_rate FluidDomainSettings")
# Validate: 1.5 is within [0.01, 4] ✓
```

### 2. **Error Recovery Loop** ⚠️ PARTIAL
**Problem:** When Blender execution fails (e.g., API error), the iteration controller doesn't automatically:
- Parse the error via `parse_blender_errors`
- Query the manual for correct API
- Generate a fix

**Current:** Requires manual intervention.

### 3. **Reference Image Library** ⚠️ MINIMAL
**Problem:** `find_reference_images` searches `assets/reference_images/` but this directory is sparse.

**Solution:** Curate reference images for:
- Explosions (grenade, nuclear, fireball)
- Fire/smoke (campfire, wildfire, exhaust)
- Celestial (nebulae, supernovae, stellar flares)

### 4. **Temporal Coherence Evaluation** ⚠️ PARTIAL
**Problem:** `analyze_temporal_quality` exists but isn't integrated into `create_asset` loop.

**Solution:** After spatial quality passes, run temporal check across animation frames.

### 5. **VLM Diagnostics Integration** ⚠️ EXPERIMENTAL
**Problem:** `enhanced_evaluate` uses Moondream VLM for diagnostics but:
- Moondream model loading is slow
- VLM responses need mapping to Blender parameters

**Solution:** Pre-load VLM or use lighter alternatives.

### 6. **Mission Control Integration** ❌ NOT INTEGRATED
**Problem:** `mission-control` MCP exists for multi-agent orchestration but isn't used in the Blender pipeline.

**Solution:** Mission control could coordinate:
- Script Generator + Blender Manual (validation)
- Experiment Tracker + Asset Evaluator (learning)
- Multiple asset creation in parallel

### 7. **Automated Bake Parameter Optimization** ❌ NOT IMPLEMENTED
**Problem:** No automated search for optimal Mantaflow parameters.

**Solution:** Implement grid search or Bayesian optimization over:
- `resolution_max` (speed vs quality)
- `flame_vorticity`, `burning_rate`, etc.

---

## Recommended Improvements

### Priority 1: Blender Manual Validation
```
Script Generator --query--> Blender Manual MCP
                          - Validate API exists
                          - Check parameter ranges
                          - Get deprecation warnings
                <--response-- Valid/Invalid + correct values
```

### Priority 2: Error-Driven Learning
```
Blender Executor (fails) --> Experiment Tracker
                            - Record error
                            - Link to parameter that caused it
                            - Add to knowledge base for future prevention
```

### Priority 3: Mission Control Orchestration
```
User Request --> Mission Control
                - Dispatch to Script Generator (with manual validation)
                - Monitor Blender Executor
                - Route failures to Experiment Tracker
                - Aggregate results
```

---

## File Locations

| Component | Path |
|-----------|------|
| MCP Servers | `/home/maz3ppa/projects/PlasmaDXR/agents/` |
| Generated Scripts | `assets/blender_scripts/generated/` |
| VDB Output | `build/vdb_output/<asset_name>/` |
| Renders | `build/vdb_output/<asset_name>/render_*.png` |
| Blender Logs | `build/blender_cli_logs/<timestamp>/` |
| Experiment DB | `agents/experiment-tracker/experiments.db` |
| Reference Images | `assets/reference_images/` |
| Templates | `assets/blender_scripts/GPT-5.2/` |

---

## Quick Start

### Create an Explosion Asset
```bash
# Using iteration-controller MCP tool:
mcp__iteration-controller__create_asset(
    asset_name="dramatic_explosion",
    description="A powerful fiery explosion with bright orange flames",
    effect_type="pyro",
    semantic_query="a bright explosion with flames and smoke",
    max_iterations=5
)
```

### Manual Single Iteration
```bash
# 1. Generate script
mcp__script-generator__generate_script(
    effect_type="pyro",
    description="Mushroom cloud explosion",
    output_name="mushroom_cloud"
)

# 2. Execute
mcp__blender-executor__execute_blender_script(
    script_path="assets/blender_scripts/generated/mushroom_cloud.py",
    script_args={"--bake": "1", "--render": "1"}
)

# 3. Evaluate
mcp__asset-evaluator__evaluate_render(
    render_path="build/vdb_output/mushroom_cloud/render_0025.png",
    semantic_query="a dramatic mushroom cloud explosion"
)
```

---

## Version History

- **2025-12-27:** Experiment Tracker fixed (FastMCP migration), learned 2 rules
- **2025-12-26:** Initial pipeline operational
- **2025-12-25:** Asset Evaluator enhanced with temporal analysis

---

*Document created for multi-agent orchestration assessment*
