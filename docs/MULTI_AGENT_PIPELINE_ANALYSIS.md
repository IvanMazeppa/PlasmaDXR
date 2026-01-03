# Multi-Agent Blender VFX Pipeline: Deep Dive Analysis

**Document Purpose:** Comprehensive analysis of the autonomous VFX asset generation system for future reference and improvement planning.

**Last Updated:** 2026-01-03
**System Version:** blender-orchestrator v0.2.0

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Architecture](#system-architecture)
3. [MCP Server Components](#mcp-server-components)
4. [Workflow State Machine](#workflow-state-machine)
5. [Trust & Autonomy System](#trust--autonomy-system)
6. [Technique Catalog](#technique-catalog)
7. [Knowledge Base & Learning](#knowledge-base--learning)
8. [Configuration Reference](#configuration-reference)
9. [Current Limitations](#current-limitations)
10. [Improvement Recommendations](#improvement-recommendations)

---

## Executive Summary

The PlasmaDX multi-agent Blender VFX pipeline is an autonomous system for generating NanoVDB volumetric assets (explosions, fire, nebulae, sun effects) through iterative improvement with ML-based quality evaluation. The system coordinates 6 specialized MCP (Model Context Protocol) servers orchestrated by the `blender-orchestrator` agent.

### Key Capabilities

- **Autonomous Asset Generation**: Generates Blender Python scripts, executes simulations, evaluates quality, and iterates until thresholds are met
- **Quality-Gated Workflow**: VFX quality scores (0-100), ground truth comparison, temporal consistency analysis
- **Adaptive Autonomy**: Trust-based system that earns more autonomy with good performance
- **Knowledge Persistence**: Experiment tracking with SQLite database for learning from outcomes
- **Session Management**: Persistent sessions with resume capability across context limits

### System Flow

```
User Request
    ↓
blender-orchestrator (coordinator)
    ↓
┌─────────────────────────────────────────────────────────────┐
│  script-generator → blender-executor → asset-evaluator     │
│         ↑                                    ↓              │
│         └──── iteration-controller ←─────────┘              │
│                       ↓                                     │
│              experiment-tracker (knowledge base)            │
└─────────────────────────────────────────────────────────────┘
    ↓
NanoVDB Assets + Quality Report
```

---

## System Architecture

### Overview

The pipeline uses a **hub-and-spoke architecture** where `blender-orchestrator` acts as the central coordinator, dispatching tasks to specialized MCP servers and managing the overall workflow state.

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                      BLENDER-ORCHESTRATOR                           │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Workflow State Machine                                      │   │
│  │  ├─ SESSION_START                                            │   │
│  │  ├─ GENERATE_SCRIPT                                          │   │
│  │  ├─ EXECUTE_BLENDER                                          │   │
│  │  ├─ EVALUATE_QUALITY                                         │   │
│  │  ├─ DECIDE_NEXT_ACTION                                       │   │
│  │  ├─ RECORD_LEARNING                                          │   │
│  │  └─ SESSION_END                                              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                 │
│  │ Trust Score │  │ Guardrails  │  │   Session   │                 │
│  │   Manager   │  │   Monitor   │  │   Persist   │                 │
│  └─────────────┘  └─────────────┘  └─────────────┘                 │
└─────────────────────────────────────────────────────────────────────┘
           │                │                │
           ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   script-    │  │   blender-   │  │    asset-    │
│  generator   │  │   executor   │  │   evaluator  │
├──────────────┤  ├──────────────┤  ├──────────────┤
│ • Templates  │  │ • CLI exec   │  │ • VFX qual   │
│ • Techniques │  │ • VDB output │  │ • Ground     │
│ • Params     │  │ • Error parse│  │   truth      │
│ • Validation │  │ • Timeouts   │  │ • Temporal   │
└──────────────┘  └──────────────┘  └──────────────┘
           │                │                │
           ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  iteration-  │  │  experiment- │  │   blender-   │
│  controller  │  │   tracker    │  │    manual    │
├──────────────┤  ├──────────────┤  ├──────────────┤
│ • Diagnosis  │  │ • Knowledge  │  │ • API docs   │
│ • Next params│  │ • Experiments│  │ • Tutorials  │
│ • State save │  │ • Warnings   │  │ • Validation │
└──────────────┘  └──────────────┘  └──────────────┘
```

### Technology Stack

| Component | Technology |
|-----------|------------|
| Orchestrator Runtime | Python 3.11+ |
| Agent Framework | Claude Agent SDK 0.1.17+ |
| MCP Protocol | FastMCP (mcp.server.fastmcp) |
| Knowledge Store | SQLite 3 |
| VFX Engine | Blender 5.0 (Python API) |
| ML Evaluation | PyTorch, LPIPS, CLIP |
| Output Format | NanoVDB (.vdb) |

---

## MCP Server Components

### 1. script-generator

**Location:** `agents/script-generator/`
**Purpose:** Generate and modify Blender Python scripts for VFX simulations

**Key Tools:**
- `list_templates()` - List available template scripts
- `list_techniques(effect_type)` - List techniques from catalog
- `generate_script(effect_type, description, output_name, ...)` - Create new script
- `modify_script(script_path, modifications)` - Adjust parameters
- `validate_parameters(params)` - Validate against Blender 5.0 API ranges
- `get_parameter_ranges()` - Get documented parameter limits

**Technique Catalog:** Contains 10+ categorically different pyro techniques:
- `rising_mushroom` - Classic mushroom cloud
- `ground_hugger` - Low, spreading fire
- `aerial_burst` - Mid-air detonation
- `slow_smolder` - Smoldering debris
- `plasma_jet` - Directed plasma stream
- `volcanic_plume` - Volcanic eruption
- `flashover` - Rapid fire spread
- `stellar_flare` - Solar flare effect
- `main_sequence_star` - Star surface
- `solar_prominence` - Solar prominence loops

### 2. blender-executor

**Location:** `agents/blender-executor/`
**Purpose:** Execute Blender scripts via CLI with output capture

**Key Tools:**
- `execute_blender_script(script_path, script_args, output_dir, ...)` - Run simulation
- `parse_blender_errors(stderr, stdout)` - Parse errors with suggested fixes
- `list_run_outputs(run_dir)` - List VDB/render outputs
- `get_latest_run()` - Get most recent execution info
- `list_available_scripts(directory)` - List available scripts

**Execution Model:**
- Background mode (no UI) by default
- 15-minute timeout for simulations
- Captures stdout/stderr for error analysis
- Output to `build/vdb_output/<asset_name>/`

### 3. asset-evaluator

**Location:** `agents/asset-evaluator/`
**Purpose:** ML-powered quality evaluation of rendered assets

**Primary Tools:**
- `evaluate_vfx_quality(image_path, effect_type)` - Standalone VFX quality (0-100)
- `evaluate_ground_truth(image_path, effect_type)` - Compare to real footage distributions
- `analyze_temporal_quality(frame_directory, ...)` - Animation consistency
- `extract_vfx_diagnostics(image_path)` - Detailed feature extraction

**Quality Dimensions:**
1. **Brightness** - Mean, max, min, dynamic range
2. **Color Presence** - Warm ratio, orange/yellow detection
3. **Coverage** - Non-black ratio, bright pixel ratio
4. **Structure** - Edge density, variance, local contrast
5. **Histogram** - Percentiles, entropy

**Score Interpretation:**
| Score | Quality |
|-------|---------|
| >= 80 | Excellent - ready to use |
| >= 60 | Good - minor improvements possible |
| >= 40 | Fair - significant issues |
| < 40 | Poor - major rework needed |

### 4. experiment-tracker

**Location:** `agents/experiment-tracker/`
**Purpose:** Track experiments and build knowledge base

**Key Tools:**
- `start_experiment_session(asset_name, effect_type, description, ...)` - Begin tracking
- `record_baseline(params, scores, render_path, script_path)` - Capture baseline
- `record_experiment_result(hypothesis, issue_addressed, success, learnings, ...)` - Record outcome
- `get_warnings_before_change(parameter, change_type)` - Get warnings before modification
- `suggest_experiments(issue, current_params, current_scores)` - Get informed suggestions
- `query_knowledge_base(query)` - Search accumulated knowledge
- `get_parameter_knowledge(parameter)` - Get rules/warnings for parameter
- `add_manual_learning(parameter, rule, warning, context)` - Add knowledge manually

**Database Schema:**
- `experiments` table - Individual experiments with outcomes
- `sessions` table - Session metadata and final status
- `knowledge` table - Accumulated parameter rules and warnings
- `causal_relationships` table - Parameter → effect mappings

### 5. iteration-controller

**Location:** `agents/iteration-controller/`
**Purpose:** Coordinate iteration pipeline and state persistence

**Key Tools:**
- `create_asset(asset_name, description, effect_type, ...)` - Full autonomous pipeline
- `run_iteration(asset_name, iteration, effect_type, ...)` - Single iteration
- `diagnose_vfx_issues(quality_json)` - Diagnose problems from evaluation
- `get_next_iteration_params(current_params, quality_json, iteration)` - Get improved params
- `save_iteration_state(session_id, ...)` - Persist state for resume
- `load_iteration_state(session_id)` - Load saved state
- `list_orchestration_sessions()` - List all sessions
- `get_history(asset_name)` - Get iteration history
- `compare_iterations(asset_name)` - Compare quality across iterations

### 6. blender-manual

**Location:** `agents/blender-manual/`
**Purpose:** Query Blender documentation and API reference

**Key Tools:**
- `search_manual(query, limit)` - General documentation search
- `search_python_api(operation)` - Python API docs (bpy.ops, bpy.types)
- `search_bpy_operators(category, operation)` - Find operators
- `search_bpy_types(typename)` - Find types
- `browse_hierarchy(path)` - Browse doc structure
- `read_page(path)` - Read full page content
- `search_vdb_workflow(query)` - VDB-specific docs
- `search_semantic(query)` - AI-powered semantic search

---

## Workflow State Machine

### States

```
┌─────────────────┐
│  SESSION_START  │ ◄── User request received
└────────┬────────┘
         │ Parse request, initialize session
         ▼
┌─────────────────┐
│ GENERATE_SCRIPT │ ◄── script-generator MCP
└────────┬────────┘
         │ Blender Python script created
         ▼
┌─────────────────┐
│ EXECUTE_BLENDER │ ◄── blender-executor MCP
└────────┬────────┘
         │ VDB + renders produced
         ▼
┌─────────────────┐
│EVALUATE_QUALITY │ ◄── asset-evaluator MCP
└────────┬────────┘
         │ VFX score, issues identified
         ▼
┌─────────────────┐
│DECIDE_NEXT_ACT. │ ──┬── Quality passed? → SESSION_END
└────────┬────────┘   │
         │            └── Max iterations? → SESSION_END
         │ Quality failed, more iterations available
         ▼
┌─────────────────┐
│ RECORD_LEARNING │ ◄── experiment-tracker MCP
└────────┬────────┘
         │ Knowledge base updated
         │
         └──────────► GENERATE_SCRIPT (loop)
```

### State Transitions

| From State | To State | Condition |
|------------|----------|-----------|
| SESSION_START | GENERATE_SCRIPT | Request parsed successfully |
| GENERATE_SCRIPT | EXECUTE_BLENDER | Script generated |
| GENERATE_SCRIPT | SESSION_END | Script generation failed 3x |
| EXECUTE_BLENDER | EVALUATE_QUALITY | Blender execution succeeded |
| EXECUTE_BLENDER | GENERATE_SCRIPT | Execution failed, retry with fix |
| EXECUTE_BLENDER | SESSION_END | Execution failed 3x |
| EVALUATE_QUALITY | DECIDE_NEXT_ACTION | Evaluation complete |
| DECIDE_NEXT_ACTION | SESSION_END | Quality passed (score >= threshold) |
| DECIDE_NEXT_ACTION | SESSION_END | Max iterations reached |
| DECIDE_NEXT_ACTION | RECORD_LEARNING | Need more iterations |
| RECORD_LEARNING | GENERATE_SCRIPT | Learning recorded, modify script |

---

## Trust & Autonomy System

### Trust Score Economy

The orchestrator maintains a trust score (0.0 - 1.0) that affects how autonomously it operates.

**Positive Adjustments:**
| Event | Adjustment |
|-------|------------|
| Quality improved | +0.05 |
| Quality threshold met | +0.10 |
| Asset completed | +0.15 |
| Human approval | +0.02 |

**Negative Adjustments:**
| Event | Adjustment |
|-------|------------|
| Quality degraded | -0.05 |
| Execution error | -0.10 |
| Technique change | -0.03 |
| Token limit exceeded | -0.15 |
| Human override | -0.10 |
| Critical failure | -0.20 |

### Autonomy Levels

| Level | Trust Range | Behavior |
|-------|-------------|----------|
| SUPERVISED | 0.0 - 0.3 | Ask before every major step |
| GUIDED | 0.3 - 0.6 | Ask before technique changes and session end |
| AUTONOMOUS | 0.6 - 0.8 | Ask only for session end and error recovery |
| TRUSTED | 0.8 - 1.0 | Fully autonomous, only notify on completion |

**Current Default:** `override: "autonomous"` in config (bypasses trust score for testing)

### Approval Requirements by Level

| Action | SUPERVISED | GUIDED | AUTONOMOUS | TRUSTED |
|--------|------------|--------|------------|---------|
| Start session | Ask | Auto | Auto | Auto |
| Generate script | Ask | Auto | Auto | Auto |
| Execute Blender | Ask | Auto | Auto | Auto |
| Technique change | Ask | Ask | Auto | Auto |
| Retry on error | Ask | Ask | Ask | Auto |
| End session | Ask | Ask | Ask | Notify |

---

## Technique Catalog

### How Techniques Work

Each technique in `technique_catalog.py` defines a complete parameter profile for a specific visual effect:

```python
"rising_mushroom": {
    "description": "Classic mushroom cloud explosion with rising fireball",
    "domain_params": {
        "domain_scale": (3.0, 4.0, 6.0),
        "domain_position": (0, 0, 1.5),
        "resolution": 96,
        "cache_type": "OPENVDB"
    },
    "noise_params": {
        "noise_strength": 0.8,
        "noise_scale": 3.0,
        "turbulence_strength": 0.7
    },
    "emission_dynamics": {
        "flame_max_temp": 3500,
        "flame_smoke": 2.5,
        "burning_rate": 1.2,
        "fuel_amount": 3.0
    },
    "flow_params": {
        "initial_velocity": (0, 0, 8.0),
        "temperature": 4.0,
        "density": 2.0
    },
    "effectors": {...},
    "camera": {...}
}
```

### Technique Selection Algorithm

The current technique selection in `get_technique_by_keywords()`:

1. **Keyword Matching:** Count matching keywords in technique metadata
2. **Strong Match:** 2+ keyword matches → return that technique
3. **Weak Match:** 1 keyword match → 50% chance to use, 50% random
4. **No Match:** Random selection from catalog

**Problem:** This approach is simplistic and doesn't consider:
- Past success rates
- Effect type compatibility
- Quality score history
- User preferences

### Available Techniques

| Technique | Description | Best For |
|-----------|-------------|----------|
| rising_mushroom | Classic mushroom cloud | Nuclear-style explosions |
| ground_hugger | Low spreading fire | Ground-level blasts |
| aerial_burst | Mid-air detonation | Airborne explosions |
| slow_smolder | Smoldering debris | Aftermath scenes |
| plasma_jet | Directed plasma stream | Sci-fi effects |
| volcanic_plume | Volcanic eruption | Natural disasters |
| flashover | Rapid fire spread | Indoor fire scenes |
| stellar_flare | Solar flare effect | Space scenes |
| main_sequence_star | Star surface | Star rendering |
| solar_prominence | Solar prominence loops | Sun close-ups |

---

## Knowledge Base & Learning

### How Learning Works

1. **Experiment Recording:**
   - Before making changes, record baseline (params, scores, render)
   - After changes, record result with hypothesis and outcome
   - System extracts causal relationships (parameter → effect)

2. **Knowledge Accumulation:**
   - Successful experiments → positive rules
   - Failed experiments → warnings
   - Repeated patterns → confidence scores

3. **Knowledge Application:**
   - Before making changes, query `get_warnings_before_change()`
   - When stuck, call `suggest_experiments()` for informed suggestions
   - Query `get_parameter_knowledge()` for specific parameter insights

### Knowledge Schema

```sql
-- Experiments table
CREATE TABLE experiments (
    id TEXT PRIMARY KEY,
    session_id TEXT,
    hypothesis TEXT,
    issue_addressed TEXT,
    baseline_params JSON,
    result_params JSON,
    baseline_scores JSON,
    result_scores JSON,
    success BOOLEAN,
    partial_success BOOLEAN,
    learnings JSON,
    warnings JSON,
    human_feedback_rating INTEGER,
    human_feedback_notes TEXT,
    created_at TIMESTAMP
);

-- Knowledge rules
CREATE TABLE knowledge (
    id INTEGER PRIMARY KEY,
    parameter TEXT,
    rule TEXT,
    warning TEXT,
    context TEXT,
    confidence REAL,
    source_experiments JSON,
    created_at TIMESTAMP
);

-- Causal relationships
CREATE TABLE causal_relationships (
    id INTEGER PRIMARY KEY,
    parameter TEXT,
    change_type TEXT,  -- increase, decrease, modify
    effect TEXT,
    effect_type TEXT,  -- positive, negative, neutral
    confidence REAL,
    occurrences INTEGER
);
```

### Example Knowledge Entry

```json
{
  "parameter": "domain_scale",
  "rule": "When increasing domain_scale, also adjust domain_position.z proportionally",
  "warning": "Increasing domain_scale without adjusting position causes clipping",
  "context": "pyro simulations",
  "confidence": 0.85,
  "source_experiments": ["exp_001", "exp_003", "exp_007"]
}
```

---

## Configuration Reference

### Key Configuration Sections

**Autonomy Settings:**
```yaml
autonomy:
  initial_trust_score: 0.2
  override: "autonomous"  # Bypasses trust for testing
  levels:
    supervised: {min_trust: 0.0, max_trust: 0.3}
    guided: {min_trust: 0.3, max_trust: 0.6}
    autonomous: {min_trust: 0.6, max_trust: 0.8}
    trusted: {min_trust: 0.8, max_trust: 1.0}
```

**Guardrails:**
```yaml
guardrails:
  token_limits:
    per_session: 100000
    per_iteration: 20000
    per_tool_call: 5000
    absolute_max: 150000
  cost_limits:
    per_session_usd: 5.00
    per_day_usd: 20.00
    per_week_usd: 75.00
```

**Quality Thresholds:**
```yaml
quality:
  vfx_pass_threshold: 70
  vfx_good_threshold: 80
  vfx_excellent_threshold: 90
  plateau_detection:
    threshold: 5      # Score change < 5 = plateau
    iterations: 3     # Plateau if stable for 3 iterations
```

**Workflow Limits:**
```yaml
workflow:
  max_iterations: 10
  max_retries: 3
  max_technique_changes: 2
  timeouts:
    blender_execution: 900  # 15 min
    evaluation: 120         # 2 min
    mcp_tool_call: 60       # 1 min
```

---

## Current Limitations

### 1. Orchestrator Effectiveness Issues

**Problem:** The blender-orchestrator agent may not be executing MCP tools effectively.

**Symptoms:**
- Tools described but not called
- Workflow stages mentioned but not executed
- State machine not properly transitioning

**Root Cause Analysis:**
- SKILL.md documents workflow but orchestrator.py implementation may not follow it
- MCP tool calls require proper async handling
- State persistence may be incomplete

### 2. Limited Technique Exploration

**Problem:** System doesn't actively explore new techniques or learn which work best.

**Current Behavior:**
- Keyword matching for technique selection
- Random fallback when no keywords match
- No success rate tracking per technique

**What's Missing:**
- Technique performance history
- Effect type → technique success mapping
- Exploration vs exploitation balance

### 3. Research Capability Gaps

**Problem:** System doesn't research known effective techniques from external sources.

**What's Available:**
- `blender-manual` MCP for documentation
- Technique catalog with static definitions

**What's Missing:**
- Academic paper search (SIGGRAPH, etc.)
- Community resource integration (BlenderArtists, etc.)
- Real-time technique discovery

### 4. Knowledge Base Underutilization

**Problem:** Experiment tracker exists but may not be queried before decisions.

**Current State:**
- SQLite database with experiment history
- Warning and suggestion tools available
- Causal relationship tracking

**What's Missing:**
- Automatic warning checks before parameter changes
- Suggestion integration into iteration loop
- Cross-session knowledge application

### 5. Session Resumption Reliability

**Problem:** Resuming interrupted sessions may not work consistently.

**Current Implementation:**
- State saved to JSON files in `build/orchestrator_state/`
- `load_iteration_state()` tool available

**Potential Issues:**
- State format compatibility across versions
- Incomplete state capture
- Missing render/VDB file references

---

## Improvement Recommendations

### Priority 1: Orchestrator Reliability

**Goal:** Ensure orchestrator actually executes MCP tools, not just describes them.

**Recommendations:**

1. **Add Execution Verification**
   ```python
   async def execute_with_verification(self, tool_name, params):
       result = await self.call_mcp_tool(tool_name, params)
       if result is None or 'error' in result:
           self.log_execution_failure(tool_name, result)
           return self.handle_tool_failure(tool_name, params)
       return result
   ```

2. **Implement Workflow Tracing**
   - Log every state transition with timestamp
   - Capture tool call inputs/outputs
   - Generate session trace for debugging

3. **Add Health Checks**
   - Verify MCP servers are running before workflow
   - Test each tool with minimal input
   - Report server health in status

### Priority 2: Intelligent Technique Selection

**Goal:** Learn which techniques work best for different effect types.

**Recommendations:**

1. **Track Technique Performance**
   ```python
   # In experiment-tracker
   CREATE TABLE technique_performance (
       technique_name TEXT,
       effect_type TEXT,
       success_count INTEGER,
       failure_count INTEGER,
       avg_score REAL,
       avg_iterations REAL,
       last_used TIMESTAMP
   );
   ```

2. **Implement Exploration Strategy**
   ```python
   def select_technique(self, effect_type, keywords):
       # 80% exploitation (use known good techniques)
       # 20% exploration (try new/less-used techniques)
       if random.random() < 0.8:
           return self.get_best_technique(effect_type)
       else:
           return self.get_unexplored_technique(effect_type)
   ```

3. **Add Technique Recommendation Tool**
   ```python
   @mcp.tool()
   def recommend_technique(effect_type: str, description: str) -> str:
       """
       Recommend best technique based on:
       - Effect type compatibility
       - Historical success rate
       - Description keyword matching
       - Exploration bonus for underused techniques
       """
   ```

### Priority 3: External Research Integration

**Goal:** Discover and incorporate proven VFX techniques from external sources.

**Recommendations:**

1. **Add Web Research Tool**
   ```python
   @mcp.tool()
   def research_vfx_technique(query: str, sources: list) -> str:
       """
       Search for VFX techniques across:
       - Blender documentation
       - SIGGRAPH papers
       - BlenderArtists community
       - YouTube tutorials (metadata)
       """
   ```

2. **Implement Technique Import**
   - Parse discovered techniques into catalog format
   - Validate parameters against Blender API
   - Add as "experimental" techniques for testing

3. **Create Reference Image Database**
   - Collect high-quality reference images
   - Tag by effect type, style, complexity
   - Use for ground truth evaluation

### Priority 4: Knowledge Base Integration

**Goal:** Automatically apply accumulated knowledge during iterations.

**Recommendations:**

1. **Mandatory Warning Checks**
   ```python
   async def modify_parameters(self, current_params, modifications):
       for param, new_value in modifications.items():
           change_type = "increase" if new_value > current_params.get(param, 0) else "decrease"
           warnings = await self.call_mcp_tool(
               "experiment-tracker",
               "get_warnings_before_change",
               {"parameter": param, "change_type": change_type}
           )
           if warnings:
               self.log_warning(f"Warning for {param}: {warnings}")
               # Adjust modification or ask for approval based on severity
   ```

2. **Suggestion-Driven Iteration**
   ```python
   async def get_next_modifications(self, issues, current_params, current_scores):
       # First, get suggestions from knowledge base
       suggestions = await self.call_mcp_tool(
           "experiment-tracker",
           "suggest_experiments",
           {
               "issue": issues[0],  # Primary issue
               "current_params": json.dumps(current_params),
               "current_scores": json.dumps(current_scores)
           }
       )

       if suggestions and suggestions[0]['confidence'] > 0.6:
           return suggestions[0]['parameter_changes']
       else:
           # Fall back to heuristic-based modifications
           return self.heuristic_modifications(issues)
   ```

3. **Cross-Session Learning**
   - Load relevant knowledge at session start
   - Pre-populate warnings for effect type
   - Use best techniques from similar past sessions

### Priority 5: Enhanced Autonomy Pipeline

**Goal:** Increase pipeline autonomy while maintaining quality.

**Recommendations:**

1. **Parallel Evaluation**
   ```python
   async def evaluate_comprehensive(self, render_path, effect_type):
       # Run all evaluations in parallel
       tasks = [
           self.call_mcp_tool("asset-evaluator", "evaluate_vfx_quality", {...}),
           self.call_mcp_tool("asset-evaluator", "evaluate_ground_truth", {...}),
           self.call_mcp_tool("asset-evaluator", "analyze_temporal_quality", {...})
       ]
       results = await asyncio.gather(*tasks)
       return self.combine_evaluations(results)
   ```

2. **Predictive Iteration**
   - Use ML to predict required iterations based on initial evaluation
   - Adjust resolution/quality based on predicted difficulty
   - Early exit for simple effects, more resources for complex ones

3. **Multi-Technique Trials**
   ```python
   async def generate_with_trials(self, effect_type, description, num_trials=3):
       # Generate multiple scripts with different techniques in parallel
       techniques = self.select_diverse_techniques(effect_type, num_trials)

       # Execute all in parallel (if resources allow)
       results = await asyncio.gather(*[
           self.execute_pipeline(effect_type, description, technique)
           for technique in techniques
       ])

       # Return best result
       return max(results, key=lambda r: r['score'])
   ```

4. **Automatic Quality Threshold Adjustment**
   - Track user satisfaction per effect type
   - Adjust thresholds based on feedback
   - Different standards for different use cases

---

## Appendix A: Quick Reference Commands

### Starting an Asset Generation

```
User: Create a mushroom cloud explosion with orange fire

Orchestrator Actions:
1. list_techniques("pyro") → Get available techniques
2. generate_script(effect_type="pyro", description="mushroom cloud...", output_name="explosion_v1")
3. execute_blender_script(script_path="assets/blender_scripts/generated/explosion_v1.py")
4. evaluate_vfx_quality(image_path="build/vdb_output/explosion_v1/render_0025.png", effect_type="explosion")
5. [If score < 70] diagnose_vfx_issues(quality_json=...) → Get modifications
6. modify_script(script_path=..., modifications=...)
7. [Repeat 3-6 until score >= 70 or max iterations]
8. record_experiment_result(...) → Save learnings
```

### Resuming a Session

```
User: Continue the explosion_v1 session

Orchestrator Actions:
1. load_iteration_state("explosion_v1")
2. [Resume from saved state]
3. [Continue iteration loop]
```

### Checking System Status

```
User: What's the status of the orchestrator?

Orchestrator Response:
- Trust Score: 0.65 (AUTONOMOUS level)
- Active Sessions: 2
- Token Usage: 45,000 / 100,000 (45%)
- Knowledge Base: 127 experiments, 34 rules
```

---

## Appendix B: File Locations

| Component | Location |
|-----------|----------|
| Orchestrator | `agents/blender-orchestrator/orchestrator.py` |
| Config | `agents/blender-orchestrator/config.yaml` |
| Skill Definition | `.claude/skills/blender-orchestrator/SKILL.md` |
| Script Generator | `agents/script-generator/server.py` |
| Technique Catalog | `agents/script-generator/technique_catalog.py` |
| Blender Executor | `agents/blender-executor/server.py` |
| Asset Evaluator | `agents/asset-evaluator/server.py` |
| Experiment Tracker | `agents/experiment-tracker/server.py` |
| Iteration Controller | `agents/iteration-controller/server.py` |
| Blender Manual | `agents/blender-manual/server.py` |
| Generated Scripts | `assets/blender_scripts/generated/` |
| VDB Output | `build/vdb_output/` |
| Session State | `build/orchestrator_state/` |
| Experiments DB | `agents/experiment-tracker/experiments.db` |

---

## Document History

| Date | Version | Changes |
|------|---------|---------|
| 2026-01-03 | 1.0 | Initial comprehensive analysis |

---

**Note:** This document is intended for use in future Claude Code sessions to provide context about the multi-agent pipeline. It should be updated as the system evolves.
