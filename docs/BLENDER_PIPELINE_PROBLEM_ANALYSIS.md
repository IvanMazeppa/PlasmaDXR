# Blender VFX Pipeline: Problem Analysis Document

**Date:** 2026-01-02
**Status:** BLOCKED - Pipeline non-functional
**Purpose:** Comprehensive analysis for multi-agent orchestrator consultation

---

## 1. Executive Summary

The Blender VFX Asset Generation Pipeline is a multi-agent system designed to autonomously create volumetric VFX assets (explosions, fire, smoke, stellar phenomena) using Blender's Mantaflow simulation. The pipeline is currently **non-functional** due to a critical blocking issue where the inner Claude agent does not execute MCP tools, causing an infinite loop and burning ~$7 per failed session.

**Key Problems Identified:**
1. **P0 (CRITICAL BLOCKER):** Inner Claude agent returns text descriptions instead of executing MCP tools
2. **P1 (RESOLVED):** ML evaluation metrics (LPIPS/CLIP) don't work for VFX - already fixed with VFX-specific metrics
3. **P2 (MEDIUM):** Agents don't share knowledge across iterations - leads to repeated failures
4. **P3 (LOW):** Evaluation system was blind to structural/morphological issues - already fixed with DINOv2

---

## 2. Complete Pipeline Architecture

### 2.1 System Layers

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LAYER 1: USER INTERFACE                           │
│                                                                             │
│   Claude Code (Interactive Session)                                        │
│   └── User invokes: /blender-orchestrator create_asset                     │
│       or calls mcp__blender-orchestrator__create_asset()                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │ MCP Tool Call (stdio transport)
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        LAYER 2: ORCHESTRATOR MCP SERVER                     │
│                                                                             │
│   File: agents/blender-orchestrator/mcp_server.py                          │
│   Transport: FastMCP over stdio                                            │
│                                                                             │
│   Exposed Tools:                                                            │
│   ├── create_asset(asset_name, effect_type, description, ...)              │
│   ├── resume_session(session_id)                                           │
│   ├── list_sessions(status_filter, limit)                                  │
│   └── get_status()                                                         │
│                                                                             │
│   When create_asset() is called:                                           │
│   1. Instantiates BlenderOrchestratorAgent                                 │
│   2. Calls agent.start() → Creates ClaudeSDKClient                         │
│   3. Calls agent.create_asset() → Runs workflow state machine              │
│   4. Calls agent.stop() → Cleans up                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │ Creates via Claude Agent SDK 0.1.18
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     LAYER 3: ORCHESTRATOR AGENT                             │
│                                                                             │
│   File: agents/blender-orchestrator/orchestrator.py                        │
│   Class: BlenderOrchestratorAgent                                          │
│                                                                             │
│   Components:                                                               │
│   ├── ClaudeSDKClient          - Communicates with inner Claude agent      │
│   ├── WorkflowStateMachine     - Manages stage transitions                 │
│   ├── AutonomyController       - Trust score & permission management       │
│   ├── TokenGuardrails          - Cost/token limit enforcement              │
│   └── SessionManager           - State persistence for resumability        │
│                                                                             │
│   Workflow Stages:                                                          │
│   SESSION_START → GENERATE_SCRIPT → EXECUTE_BLENDER → EVALUATE_QUALITY     │
│        ↑              │                                      │              │
│        │              ▼                                      ▼              │
│        │         ERROR_RECOVERY ←───────────────── DECIDE_NEXT_ACTION       │
│        │              │                                      │              │
│        └──────────────┴──────────────────────────────────────┘              │
│                                      │                                      │
│                                      ▼                                      │
│                               SESSION_END                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │ Sends prompts via _query_agent()
                                      │ SDK spawns bundled Claude Code CLI
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      LAYER 4: INNER CLAUDE AGENT                            │
│                                                                             │
│   Created by: ClaudeSDKClient (Claude Agent SDK)                           │
│   Type: Bundled Claude Code CLI instance                                   │
│                                                                             │
│   Configuration (from ClaudeAgentOptions):                                 │
│   ├── system_prompt: Describes VFX workflow, autonomy level, tools         │
│   ├── mcp_servers: Dict of 6 MCP server configurations                     │
│   ├── allowed_tools: List of 43 permitted MCP tool names                   │
│   ├── max_budget_usd: $5.00 per session                                    │
│   ├── max_thinking_tokens: 8000                                            │
│   ├── permission_mode: 'acceptEdits'                                       │
│   └── hooks: Pre/post tool use hooks for guardrails                        │
│                                                                             │
│   ⚠️ PROBLEM LOCATION: This agent receives prompts like:                   │
│      "Use script-generator.generate_script() to create..."                 │
│                                                                             │
│   But responds with TEXT like:                                              │
│      "I'll use script-generator.generate_script()..."                      │
│                                                                             │
│   Instead of ACTUALLY CALLING the MCP tools.                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │ SHOULD call these (but doesn't)
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       LAYER 5: PIPELINE MCP SERVERS                         │
│                                                                             │
│   6 specialized MCP servers (each runs as subprocess via stdio):           │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ script-generator (agents/script-generator/)                         │  │
│   │                                                                     │  │
│   │ Purpose: Generate and modify Blender Python scripts                 │  │
│   │ Tools:                                                              │  │
│   │   • generate_script(effect_type, description, output_name, ...)     │  │
│   │   • modify_script(script_path, modifications, ...)                  │  │
│   │   • list_techniques(effect_type) - Catalog of VFX approaches        │  │
│   │   • validate_parameters(params) - Check Blender API ranges          │  │
│   │   • get_parameter_ranges() - All valid parameter bounds             │  │
│   │   • list_templates() - Available script templates                   │  │
│   │   • analyze_script(script_path) - Parse existing script             │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ blender-executor (agents/blender-executor/)                         │  │
│   │                                                                     │  │
│   │ Purpose: Execute Blender scripts and capture outputs                │  │
│   │ Tools:                                                              │  │
│   │   • execute_blender_script(script_path, script_args, output_dir)    │  │
│   │   • parse_blender_errors(stderr, stdout) - Error categorization     │  │
│   │   • list_run_outputs(run_dir) - VDB files, renders, logs            │  │
│   │   • get_latest_run() - Most recent execution info                   │  │
│   │   • list_available_scripts(directory) - Find scripts                │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ asset-evaluator (agents/asset-evaluator/)                           │  │
│   │                                                                     │  │
│   │ Purpose: Evaluate VFX quality using ML and heuristic metrics        │  │
│   │ Tools:                                                              │  │
│   │   • evaluate_vfx_quality(image_path, effect_type) ⭐ PRIMARY        │  │
│   │   • compare_vfx_iterations(image_a, image_b, effect_type)           │  │
│   │   • extract_vfx_diagnostics(image_path) - Feature extraction        │  │
│   │   • analyze_temporal_quality(frame_directory, ...) - Animation      │  │
│   │   • evaluate_ground_truth(image_path, effect_type) - Real footage   │  │
│   │   • evaluate_structural_quality(render, reference) - DINOv2         │  │
│   │   • analyze_texture_procedural(render_path) - Detect artifacts      │  │
│   │   • compare_lpips(image1, image2) - Perceptual similarity           │  │
│   │   • enhanced_evaluate(...) - Multi-modal comprehensive eval         │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ iteration-controller (agents/iteration-controller/)                 │  │
│   │                                                                     │  │
│   │ Purpose: Manage iteration loop logic and state persistence          │  │
│   │ Tools:                                                              │  │
│   │   • diagnose_vfx_issues(quality_json) - Parse scores to issues      │  │
│   │   • get_next_iteration_params(current, quality, iteration)          │  │
│   │   • save_iteration_state(session_id, ...) - Checkpoint state        │  │
│   │   • load_iteration_state(session_id) - Resume from checkpoint       │  │
│   │   • list_orchestration_sessions() - All saved sessions              │  │
│   │   • run_iteration(...) - Execute single iteration (returns TEXT!)   │  │
│   │   • create_asset(...) - Full pipeline (returns TEXT!)               │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ experiment-tracker (agents/experiment-tracker/)                     │  │
│   │                                                                     │  │
│   │ Purpose: Knowledge base for learning from past experiments          │  │
│   │ Tools:                                                              │  │
│   │   • start_experiment_session(asset_name, effect_type, desc)         │  │
│   │   • record_baseline(params, scores, render_path, script_path)       │  │
│   │   • record_experiment_result(hypothesis, result, success, ...)      │  │
│   │   • get_warnings_before_change(parameter, change_type)              │  │
│   │   • suggest_experiments(issue, current_params, current_scores)      │  │
│   │   • query_knowledge_base(query) - Search past learnings             │  │
│   │   • get_parameter_knowledge(parameter) - Rules for param            │  │
│   │   • add_manual_learning(parameter, rule, warning, context)          │  │
│   │   • get_experiment_statistics() - Success rates, counts             │  │
│   │   • get_session_report(session_id) - Full session history           │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ blender-manual (agents/blender-manual/)                             │  │
│   │                                                                     │  │
│   │ Purpose: Blender documentation search for API help                  │  │
│   │ Tools:                                                              │  │
│   │   • search_manual(query) - General documentation search             │  │
│   │   • search_python_api(operation) - bpy.ops, bpy.types, etc.         │  │
│   │   • search_nodes(node_type, category) - Shader/geometry nodes       │  │
│   │   • search_modifiers(modifier_name) - Mesh/volume modifiers         │  │
│   │   • search_vdb_workflow(query) - VDB/OpenVDB documentation          │  │
│   │   • search_tutorials(topic, technique) - Learning resources         │  │
│   │   • read_page(path, max_length) - Full page content                 │  │
│   │   • browse_hierarchy(path) - Navigate doc structure                 │  │
│   │   • search_semantic(query) - AI-powered semantic search             │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        LAYER 6: EXTERNAL SYSTEMS                            │
│                                                                             │
│   ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐    │
│   │  Blender 5.0.1  │  │  PyTorch/LPIPS  │  │  Reference Footage      │    │
│   │                 │  │  CLIP/DINOv2    │  │  840 frames NASA solar  │    │
│   │  • Mantaflow    │  │                 │  │  in assets/reference_   │    │
│   │  • Cycles       │  │  ML Models for  │  │  images/star/           │    │
│   │  • NanoVDB      │  │  evaluation     │  │                         │    │
│   └─────────────────┘  └─────────────────┘  └─────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Agent Role Descriptions

#### 2.2.1 Orchestrator Agent (BlenderOrchestratorAgent)

**Role:** Central coordinator that manages the entire asset generation workflow.

**Responsibilities:**
- Creates and manages workflow state machine
- Sends prompts to inner Claude agent for each stage
- Extracts results from agent responses (script paths, scores, etc.)
- Makes decisions about iteration, technique changes, or termination
- Enforces token/cost guardrails
- Manages trust-based autonomy levels
- Persists session state for resumability

**Key Methods:**
| Method | Purpose |
|--------|---------|
| `create_asset()` | Start new generation session |
| `_run_workflow()` | Main loop executing stages |
| `_execute_stage()` | Dispatch to stage handlers |
| `_query_agent()` | Send prompt to inner agent |
| `_extract_script_path()` | Parse response for paths |

**Current State:** BLOCKED - Cannot extract results because inner agent doesn't execute tools.

---

#### 2.2.2 Script Generator Agent

**Role:** Creates Blender Python scripts for Mantaflow simulations.

**Responsibilities:**
- Generate initial scripts from effect type + description
- Modify existing scripts based on feedback
- Validate parameters against Blender API ranges
- Maintain technique catalog for variety
- Consult Blender documentation when needed

**Input → Output:**
```
Input:  effect_type="pyro", description="mushroom cloud explosion"
Output: assets/blender_scripts/generated/mushroom_cloud_v1.py
```

**Dependencies:**
- Templates in `assets/blender_scripts/GPT-5.2/`
- Blender Manual MCP for API reference
- Experiment Tracker for known issues (currently not used)

**Current State:** HEALTHY - Tools work when called directly, but orchestrator never calls them.

---

#### 2.2.3 Blender Executor Agent

**Role:** Executes Blender Python scripts via CLI and captures outputs.

**Responsibilities:**
- Run Blender in background mode with Python scripts
- Capture stdout/stderr for error analysis
- Parse Blender-specific errors (context issues, API changes)
- Manage output directories for VDB files and renders
- Track execution history

**Input → Output:**
```
Input:  script_path="assets/blender_scripts/generated/mushroom_cloud_v1.py"
Output: {
  "vdb_files": ["build/vdb_output/mushroom_v1/density_0001.vdb", ...],
  "render_files": ["build/vdb_output/mushroom_v1/render_0001.png", ...],
  "success": true,
  "duration_seconds": 120
}
```

**Dependencies:**
- Blender 5.0.1 installation
- Blender Manual MCP for error resolution

**Current State:** HEALTHY - Works when called directly.

---

#### 2.2.4 Asset Evaluator Agent

**Role:** Evaluates VFX quality using ML models and heuristic analysis.

**Responsibilities:**
- Score renders on VFX-specific metrics (0-100 scale)
- Compare iterations to detect improvement/regression
- Analyze temporal consistency across animation frames
- Detect procedural texture artifacts
- Compare against ground truth reference footage
- Provide actionable diagnostic feedback

**Evaluation Methods:**
| Method | What It Measures | Status |
|--------|------------------|--------|
| `evaluate_vfx_quality` | Brightness, color, structure, coverage | ✅ Working |
| `evaluate_ground_truth` | Distribution match to real footage | ✅ Working |
| `evaluate_structural_quality` | DINOv2 patch-level similarity | ✅ Working |
| `analyze_texture_procedural` | Feature size CV, wavelet analysis | ✅ Working |
| `compare_lpips` | Perceptual similarity (limited use) | ⚠️ Not ideal for VFX |

**Input → Output:**
```
Input:  image_path="build/vdb_output/explosion_v1/render_0060.png", effect_type="explosion"
Output: {
  "composite_score": 72,
  "passed": true,
  "dimension_scores": {"brightness": 85, "color": 68, "structure": 75, "coverage": 60},
  "issues": ["Color slightly too cool for explosion"],
  "critical_issues": []
}
```

**Current State:** HEALTHY - Sophisticated evaluation system ready, but never called by orchestrator.

---

#### 2.2.5 Iteration Controller Agent

**Role:** Manages iteration logic and parameter adjustments.

**Responsibilities:**
- Diagnose issues from evaluation scores
- Suggest parameter changes based on issues
- Save/load iteration state for resumability
- Track iteration history and best scores

**Critical Issue:** The `run_iteration()` and `create_asset()` tools return TEXT INSTRUCTIONS instead of actually executing:

```python
async def run_iteration(...):
    result = IterationResult(
        recommendations=[
            "1. Use script-generator to create/modify script...",  # TEXT!
            "2. Use blender-executor to run the script...",        # TEXT!
        ]
    )
    return json.dumps(asdict(result))  # Returns instructions, not execution
```

**Current State:** DESIGN FLAW - Returns instructions instead of executing.

---

#### 2.2.6 Experiment Tracker Agent

**Role:** Knowledge base for learning from past experiments.

**Responsibilities:**
- Record experiment hypotheses and outcomes
- Accumulate rules about parameter behavior
- Suggest experiments based on past success/failure
- Warn about known problematic changes
- Provide session history for analysis

**Knowledge Schema:**
```python
{
  "parameter": "domain_scale",
  "rules": [
    "Always adjust emitter position when scaling domain",
    "Values > 5.0 cause clipping at top edge"
  ],
  "warnings": [
    "Scaling domain without adjusting position causes effect to leave frame"
  ],
  "success_rate": 0.65,
  "experiments_count": 12
}
```

**Current State:** UNDERUTILIZED - Other agents don't query it before making changes.

---

#### 2.2.7 Blender Manual Agent

**Role:** Documentation search for Blender API help.

**Responsibilities:**
- Search Blender 5.0 documentation
- Find Python API examples
- Look up node types and modifiers
- Provide VDB workflow guidance

**Current State:** HEALTHY - Available but rarely used in current workflow.

---

## 3. Detailed Problem Analysis

### 3.1 Problem #1: Inner Agent Not Executing MCP Tools (P0 - CRITICAL BLOCKER)

#### 3.1.1 Symptom

The orchestrator enters an infinite loop:
```
04:00:07 - Workflow transition: generate_script -> error_recovery (outcome=failure)
04:00:29 - Workflow transition: error_recovery -> generate_script (outcome=success)
04:00:45 - Workflow transition: generate_script -> error_recovery (outcome=failure)
... repeats until $5 budget exhausted
```

Each iteration takes ~15 seconds - far too fast for actual Blender script generation + execution.

#### 3.1.2 Root Cause Chain

```
1. Orchestrator calls _stage_generate_script()
   │
   ▼
2. _stage_generate_script() builds prompt:
   "Use script-generator.generate_script() to create the Blender Python script..."
   │
   ▼
3. _query_agent(prompt) sends to inner Claude agent
   │
   ▼
4. Inner Claude agent receives prompt BUT:
   ⚠️ Does NOT call mcp__script-generator__generate_script()
   ⚠️ Instead returns TEXT: "I'll use script-generator.generate_script()..."
   │
   ▼
5. _query_agent() returns the text response
   │
   ▼
6. _extract_script_path(response) searches for path patterns:
   - "Script path: /path/to/script.py"
   - "Generated: /path/to/script.py"
   - "saved to /path/to/script.py"
   │
   ▼
7. No path found → returns None
   │
   ▼
8. _stage_generate_script() returns StageResult(outcome=FAILURE)
   │
   ▼
9. WorkflowStateMachine transitions to ERROR_RECOVERY
   │
   ▼
10. ERROR_RECOVERY increments retry counter, transitions back to GENERATE_SCRIPT
    │
    ▼
11. LOOP REPEATS until max retries or budget exhausted
```

#### 3.1.3 Hypotheses for Why Inner Agent Doesn't Execute Tools

**Hypothesis A: MCP Server Configuration Format**

Current format in `orchestrator.py:223-241`:
```python
mcp_config[name] = {
    "type": "stdio",
    "command": "bash",
    "args": ["-c", f"cd {cwd} && {server_config.get('command', './run_server.sh')}"],
    "cwd": str(self.project_root),
    "env": {"PROJECT_ROOT": str(self.project_root)}
}
```

SDK documentation shows simpler format:
```python
mcp_config[name] = {
    "type": "stdio",
    "command": "python",
    "args": ["-m", "server"]
}
```

The bash wrapper (`bash -c "cd ... && ./run_server.sh"`) may be causing startup issues.

**Hypothesis B: MCP Servers Not Starting**

The servers may fail to start due to:
- Missing virtual environments
- Import errors
- Port conflicts
- Timeout before ready

Evidence: No MCP tool call logs appear in orchestrator output.

**Hypothesis C: SDK Client Misconfiguration**

The `ClaudeAgentOptions` may be missing required fields:
- `tool_choice` parameter to force tool use
- Incorrect `permission_mode`
- Missing async context handling

**Hypothesis D: Prompt Engineering**

The prompts say "Use X tool to do Y" but the inner agent may:
- Not recognize the tool names
- Treat it as a description rather than instruction
- Lack context about available tools

#### 3.1.4 Effect on Rest of System

| Component | Effect |
|-----------|--------|
| **script-generator** | Never called - scripts never created |
| **blender-executor** | Never called - no simulations run |
| **asset-evaluator** | Never called - no quality assessment |
| **experiment-tracker** | Session started but no experiments recorded |
| **iteration-controller** | State never saved - no resumability |
| **User** | $7 burned, no assets produced |

**Cascade Effect:**
```
Inner agent fails to call tools
    → No script generated
        → No Blender execution
            → No renders produced
                → No evaluation possible
                    → No learning recorded
                        → Next iteration has no context
                            → Same failure repeats
```

---

### 3.2 Problem #2: ML Metrics Don't Work for VFX (P1 - RESOLVED)

#### 3.2.1 Original Problem

LPIPS and CLIP were designed for photorealistic images, not procedural VFX:

| Metric | Expected Behavior | Actual Behavior for VFX |
|--------|-------------------|-------------------------|
| LPIPS | 0.0-0.35 = similar | 0.7-0.8 for ALL VFX (no discrimination) |
| CLIP | Semantic match 0-1 | Plateaus at 0.65-0.67 regardless of quality |

Evidence from sun surface iterations:
```
v7:  LPIPS 0.745, CLIP 0.665
v8:  LPIPS 0.759, CLIP 0.667
v9:  LPIPS 0.738, CLIP 0.662
v10: LPIPS 0.752, CLIP 0.668
v11: LPIPS 0.741, CLIP 0.664
```

No gradient signal - can't tell which is better.

#### 3.2.2 Resolution

Asset-evaluator now has VFX-specific metrics:

1. **evaluate_vfx_quality()** - 0-100 composite score with dimension breakdown
2. **evaluate_ground_truth()** - Distribution match to 840 real solar frames
3. **evaluate_structural_quality()** - DINOv2 patch-level structural similarity
4. **analyze_texture_procedural()** - Feature size CV to detect procedural artifacts

These tools ARE implemented and working. They just need to be called (blocked by P0).

---

### 3.3 Problem #3: Agents Don't Share Knowledge (P2 - MEDIUM)

#### 3.3.1 The Problem

When generating a new script, the orchestrator prompts:
```
"Generate initial Blender script for: explosion effect..."
```

It does NOT first query:
```
"Check experiment-tracker for known issues with explosion effects"
```

This means every iteration starts from scratch without learning from:
- Previous parameter failures
- Successful technique patterns
- Known API quirks

#### 3.3.2 Example Failure Loop

```
Iteration 1:
  script-generator creates explosion with domain_scale=5.0
  Blender runs, effect clips at top
  asset-evaluator: "Clipping at top edge" (score 45)
  experiment-tracker records: domain_scale=5.0 causes clipping

Iteration 2:
  script-generator creates explosion with domain_scale=5.0  ← SAME!
  (Never queried experiment-tracker)
  Same failure repeats

Iteration 3:
  ... same failure ...
```

#### 3.3.3 Effect on System

- Wasted iterations repeating known failures
- No accumulated learning across sessions
- experiment-tracker fills with redundant failure records
- Token budget spent on predictable failures

---

### 3.4 Problem #4: iteration-controller Returns Text, Not Execution (P2 - DESIGN FLAW)

#### 3.4.1 The Problem

The `run_iteration()` tool was designed to return TEXT INSTRUCTIONS:

```python
async def run_iteration(...):
    result = IterationResult(
        status="in_progress",
        iteration=iteration,
        recommendations=[
            "1. Use script-generator.generate_script() to create a Blender script",
            "2. Use blender-executor.execute_blender_script() to run it",
            "3. Use asset-evaluator.evaluate_vfx_quality() to assess quality",
        ]
    )
    return json.dumps(asdict(result))
```

This was intended for a different architecture where Claude Code would interpret the instructions. But in the current design, the orchestrator expects actual execution.

#### 3.4.2 Effect on System

- Confusing responsibility boundaries
- Caller must interpret and execute recommendations
- No single-call automation possible

---

## 4. Theory of Primary Problem

### 4.1 Root Cause Theory

**The inner Claude agent spawned by the SDK is configured correctly but is NOT being instructed to use autonomous tool execution mode.**

Evidence:
1. The prompt says "Use script-generator.generate_script()" but the agent may interpret this as a description, not a command
2. The SDK may require explicit `tool_choice` configuration
3. The bash wrapper for MCP servers may cause startup failures before tools are available

### 4.2 Verification Steps

1. **Add verbose logging** to `_query_agent()`:
   ```python
   logger.info(f"=== QUERY ===\n{prompt}")
   # ... query ...
   logger.info(f"=== RESPONSE ===\n{full_response}")
   logger.info(f"Tool calls observed: {tool_calls}")
   ```

2. **Test MCP servers independently**:
   ```bash
   cd agents/script-generator && source venv/bin/activate
   python -c "from server import generate_script; import asyncio; print(asyncio.run(generate_script(...)))"
   ```

3. **Simplify MCP config** to direct Python execution:
   ```python
   mcp_config[name] = {
       "type": "stdio",
       "command": str(cwd / "venv/bin/python"),
       "args": ["server.py"]
   }
   ```

4. **Check SDK documentation** for `tool_choice` or `auto_execute_tools` options

### 4.3 Proposed Fix Priority

| Order | Fix | Effort | Confidence |
|-------|-----|--------|------------|
| 1 | Add verbose logging | 15 min | Will reveal actual behavior |
| 2 | Simplify MCP config format | 30 min | High - matches SDK examples |
| 3 | Add tool_choice if available | 15 min | Medium - may not be exposed |
| 4 | Test with single tool | 30 min | Validates entire chain |
| 5 | Consider in-process MCP servers | 2 hr | Eliminates subprocess issues |

---

## 5. Files Reference

| File | Purpose | Lines |
|------|---------|-------|
| `agents/blender-orchestrator/orchestrator.py` | Main agent, workflow, SDK integration | 1242 |
| `agents/blender-orchestrator/mcp_server.py` | FastMCP exposure to Claude Code | 309 |
| `agents/blender-orchestrator/config.yaml` | Autonomy, guardrails, MCP config | ~100 |
| `agents/blender-orchestrator/workflow.py` | WorkflowStateMachine, stages | ~300 |
| `agents/blender-orchestrator/autonomy.py` | Trust scores, permission matrix | ~200 |
| `agents/blender-orchestrator/guardrails.py` | Token/cost limits | ~150 |
| `agents/blender-orchestrator/state.py` | Session persistence | ~250 |
| `agents/script-generator/server.py` | Script generation MCP server | ~400 |
| `agents/blender-executor/server.py` | Blender CLI execution | ~300 |
| `agents/asset-evaluator/server.py` | VFX quality evaluation | ~2000 |
| `agents/iteration-controller/server.py` | Iteration logic | ~500 |
| `agents/experiment-tracker/server.py` | Knowledge base | ~600 |

---

## 6. Questions for Multi-Agent Orchestrator Analysis

1. **SDK Configuration:** What is the correct `ClaudeAgentOptions` configuration for autonomous tool execution with external MCP servers?

2. **MCP Server Format:** Is the bash wrapper (`bash -c "cd ... && ./run_server.sh"`) valid for SDK 0.1.18, or should we use direct Python execution?

3. **Tool Execution Verification:** How can we verify that the inner Claude agent is actually calling MCP tools vs. just generating text about them?

4. **In-Process Migration:** Should we migrate all 6 MCP servers to in-process SDK MCP servers for reliability?

5. **Prompt Engineering:** What prompt patterns ensure the inner agent executes tools rather than describing them?

6. **Architecture Simplification:** Is there a simpler architecture that avoids the orchestrator → inner agent → MCP servers chain?

---

## 7. Appendix: Key Code Snippets

### 7.1 MCP Server Configuration (orchestrator.py:223-241)

```python
def _create_mcp_config(self) -> Dict[str, Dict[str, Any]]:
    """Create MCP server configuration for Blender pipeline."""
    servers = self.config.get("mcp_servers", {})
    mcp_config = {}

    for name, server_config in servers.items():
        if not server_config.get("enabled", True):
            continue

        cwd = self.project_root / server_config.get("cwd", f"agents/{name}")

        mcp_config[name] = {
            "type": "stdio",
            "command": "bash",
            "args": ["-c", f"cd {cwd} && {server_config.get('command', './run_server.sh')}"],
            "cwd": str(self.project_root),
            "env": {"PROJECT_ROOT": str(self.project_root)}
        }

    return mcp_config
```

### 7.2 Query Agent Method (orchestrator.py:1080-1106)

```python
async def _query_agent(self, prompt: str) -> str:
    """Send query to Claude agent and get response."""
    if not self.client:
        raise RuntimeError("Agent not started")

    logger.debug(f"Query: {prompt[:100]}...")

    await self.client.query(prompt)

    full_response = ""
    async for message in self.client.receive_response():
        if hasattr(message, 'uuid') and message.uuid:
            self._last_checkpoint_id = message.uuid

        message_text = str(message) if not isinstance(message, str) else message
        full_response += message_text

    logger.debug(f"Response: {full_response[:200]}...")

    return full_response
```

### 7.3 Generate Script Stage (orchestrator.py:689-730)

```python
async def _stage_generate_script(self) -> StageResult:
    """Generate Blender Python script."""
    logger.info(f"Stage: GENERATE_SCRIPT (iteration {self.workflow.iteration + 1})")

    if self.workflow.iteration == 0:
        prompt = f"""Generate initial Blender script for:

Asset: {self.session.request.asset_name}
Effect Type: {self.session.request.effect_type}
Description: {self.session.request.description}
Resolution: {self.session.request.resolution}
Frames: 1-{self.session.request.frame_end}

Actions:
1. Use script-generator.generate_script() to create the Blender Python script
2. Validate parameters using validate_parameters()
3. Report the script path and key parameters
"""
    # ... prompt for subsequent iterations ...

    response = await self._query_agent(prompt)
    script_path = self._extract_script_path(response)

    return StageResult(
        stage=WorkflowStage.GENERATE_SCRIPT,
        outcome=WorkflowOutcome.SUCCESS if script_path else WorkflowOutcome.FAILURE,
        data={"script_path": script_path} if script_path else None,
        error="Failed to extract script path from response" if not script_path else None
    )
```

---

**End of Document**
