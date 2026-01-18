# Blender VFX Orchestrator - Agent Specification v2.0

**Version:** 2.0.0
**Date:** 2026-01-18
**Model:** GPT-5.2 with Reasoning (all agents)
**SDK:** OpenAI Agents SDK

---

## Executive Summary

The Blender VFX Orchestrator is an autonomous multi-agent system that generates high-quality volumetric VFX assets through iterative improvement. It coordinates 6 specialized agents using the OpenAI Agents SDK handoff mechanism, with 5 self-learning strategies for continuous improvement.

### Key Capabilities

| Capability | Description |
|------------|-------------|
| **Autonomous Iteration** | Generates and refines scripts until quality thresholds met |
| **Multi-Agent Coordination** | 6 specialized agents with handoff-based delegation |
| **Self-Learning** | 5 strategies for knowledge accumulation and reuse |
| **Stuck Detection** | Escape velocity mechanism with 5 escalation levels |
| **Session Persistence** | Recoverable sessions across context limits |
| **Budget Enforcement** | $20/month limit with per-category tracking |

---

## System Architecture

### Agent Hierarchy

```
BlenderVFXOrchestrator (GPT-5.2, coordinator)
├── ScriptWriter (GPT-5.2)     - Blender Python script generation
├── Executor (GPT-5.2)         - Script execution, error handling
├── QualityAnalyst (GPT-5.2)   - ML-powered quality evaluation
├── LearningAgent (GPT-5.2)    - Experiment tracking, knowledge base
└── DocsExpert (GPT-5.2)       - Blender documentation search
```

### Data Flow

```
Request → Orchestrator
              │
              ├──[handoff]→ ScriptWriter ──→ Script
              │                               │
              ├──[handoff]→ Executor ─────────┤
              │                               │
              │                          Render Output
              │                               │
              ├──[handoff]→ QualityAnalyst ───┤
              │                               │
              │                          Quality Metrics
              │                               │
              ├──[if failed]→ LearningAgent ──┤
              │                               │
              ├──[if stuck]→ DocsExpert ──────┘
              │
              └──→ Session Complete / Max Iterations
```

---

## Agent Specifications

### 1. Orchestrator (Main Coordinator)

**Model:** `gpt-5.2` with `reasoning.effort="medium"`
**Role:** Coordinates all specialized agents, manages iteration loop, enforces quality gates

#### Direct Tools (19 total)

| Category | Tools | Purpose |
|----------|-------|---------|
| **Proactive Research** | `pre_iteration_research`, `evaluate_escape_velocity`, `search_alternative_approaches` | Strategy 3: Early warning detection |
| **Semantic Search** | `semantic_search_blender_docs`, `find_alternative_approaches`, `search_blender_api_by_intent` | Strategy 1: Vector store search |
| **Code Patterns** | `record_code_pattern`, `search_code_patterns`, `get_pattern_code`, `report_pattern_outcome`, `get_pattern_library_stats`, `list_patterns_by_effect` | Strategy 4: Pattern memory |
| **Knowledge Distillation** | `extract_successful_pattern`, `apply_pattern_to_script`, `analyze_script_for_patterns`, `compare_scripts` | Strategy 2: Auto-extraction |

#### Handoffs (Delegations)

| Handoff | Target Agent | When Used |
|---------|--------------|-----------|
| `delegate_to_script_writer` | ScriptWriter | Generate/modify Blender scripts |
| `delegate_to_executor` | Executor | Execute scripts, handle errors |
| `delegate_to_quality_analyst` | QualityAnalyst | Evaluate render quality |
| `delegate_to_learning_agent` | LearningAgent | Get fix suggestions, record outcomes |
| `delegate_to_docs_expert` | DocsExpert | Search Blender documentation |

---

### 2. Script Writer

**Model:** `gpt-5.2` with `reasoning.effort="high"`, `temperature=0.3`
**Role:** Generate and modify Blender Python scripts for VFX effects

#### Tools

| Tool | Source | Purpose |
|------|--------|---------|
| `generate_script` | script-generator MCP | Generate new scripts from templates |
| `modify_script` | script-generator MCP | Modify existing scripts |
| `validate_script` | script-generator MCP | Validate script syntax/parameters |
| `list_techniques` | script-generator MCP | List available techniques |
| `recommend_technique` | script-generator MCP | UCB1-based technique selection |
| `get_parameter_ranges` | script-generator MCP | Get valid parameter ranges |

#### Technique Library

| Technique | Effect Types | Description |
|-----------|--------------|-------------|
| `basic_pyro` | pyro, fire, smoke | Standard Mantaflow fluid simulation |
| `explosion_burst` | explosion | Outward burst with turbulence |
| `rising_smoke` | smoke | Buoyancy-driven smoke column |
| `fire_flickering` | fire | Animated fire with turbulence |
| `nebula_volumetric` | nebula | Sparse volumetric for space scenes |
| `sun_corona` | sun, star | High-temperature emission |

---

### 3. Executor

**Model:** `gpt-5.2` with `reasoning.effort="medium"`
**Role:** Execute Blender scripts and capture outputs

#### Tools

| Tool | Source | Purpose |
|------|--------|---------|
| `execute_blender_script` | blender-executor MCP | Run script in Blender |
| `parse_blender_errors` | blender-executor MCP | Analyze error output |
| `list_run_outputs` | blender-executor MCP | List generated files |
| `get_latest_run` | blender-executor MCP | Get most recent run info |

#### Output Structure

```
runs/
└── {session_id}/
    └── iteration_{n}/
        ├── script.py           # Executed script
        ├── stdout.log          # Blender output
        ├── stderr.log          # Error output (if any)
        ├── render_0001.png     # Rendered frame
        └── vdb/                # NanoVDB files (if generated)
            ├── density_0001.vdb
            └── flame_0001.vdb
```

---

### 4. Quality Analyst

**Model:** `gpt-5.2` with `reasoning.effort="high"`
**Role:** Evaluate render quality using vision and ML metrics

#### Tools

| Tool | Source | Purpose |
|------|--------|---------|
| `evaluate_render_v2` | asset-evaluator MCP | Consolidated quality evaluation |
| `compare_renders_v2` | asset-evaluator MCP | A/B comparison |
| `diagnose_issues_v2` | asset-evaluator MCP | Issue identification |
| `get_reference_stats_v2` | asset-evaluator MCP | Reference statistics |

#### Quality Metrics

| Metric | Range | Weight | Description |
|--------|-------|--------|-------------|
| `overall_score` | 0-100 | - | Combined weighted score |
| `lpips_score` | 0-1 | 25% | Perceptual similarity (lower=better) |
| `siglip_score` | 0-1 | 25% | Semantic similarity |
| `topiq_score` | 0-100 | 25% | Aesthetic quality |
| `structural_score` | 0-1 | 25% | DINOv2 structural similarity |

#### Quality Thresholds

| Condition | Action |
|-----------|--------|
| `overall_score >= 60` AND no critical issues | **PASS** |
| `overall_score >= 60` BUT critical issues | **FAIL** - address issues |
| `overall_score < 60` | **FAIL** - iterate |

#### Critical Issues (Auto-Fail)

- `ZERO_LIGHTS_ACTIVE` - No lighting in scene
- `BLACK_SCREEN` - Completely dark render
- `WHITE_SCREEN` - Completely overexposed
- `CLIPPING_ARTIFACTS` - Volume clipping at boundaries

---

### 5. Learning Agent

**Model:** `gpt-5.2` with `reasoning.effort="high"`
**Role:** Maintain experiment knowledge, suggest fixes, record outcomes

#### Tools

| Tool | Source | Purpose |
|------|--------|---------|
| `start_experiment_session` | experiment-tracker MCP | Begin tracking |
| `record_baseline` | experiment-tracker MCP | Record starting point |
| `record_experiment_result` | experiment-tracker MCP | Log iteration outcome |
| `get_warnings_before_change` | experiment-tracker MCP | Check for known issues |
| `suggest_experiments` | experiment-tracker MCP | Get fix recommendations |
| `query_knowledge_base` | experiment-tracker MCP | Search past experiments |
| `get_parameter_knowledge` | experiment-tracker MCP | Parameter-specific knowledge |
| `add_manual_learning` | experiment-tracker MCP | Add custom learnings |

#### Knowledge Base Schema

```json
{
  "experiment_id": "exp_20260118_001",
  "effect_type": "pyro",
  "issue_addressed": "smoke_too_thin",
  "hypothesis": "Increasing flame_smoke will add density",
  "parameter_changes": {
    "flame_smoke": {"from": 1.0, "to": 2.5}
  },
  "outcome": {
    "success": true,
    "score_improvement": 12.5,
    "new_issues": []
  },
  "learnings": [
    "flame_smoke above 2.0 significantly increases density",
    "May need to reduce dissolve_speed to compensate"
  ],
  "warnings": [
    "Values above 4.0 cause over-saturation"
  ]
}
```

---

### 6. Docs Expert

**Model:** `gpt-5.2` with `reasoning.effort="medium"`
**Role:** Search Blender 5.0 documentation for solutions and alternatives

#### Tools (14 total)

| Category | Tools |
|----------|-------|
| **Keyword Search** | `search_manual`, `search_tutorials`, `search_vdb_workflow`, `search_python_api`, `search_nodes`, `search_modifiers` |
| **Semantic Search** | `search_semantic` |
| **API Reference** | `list_api_modules`, `search_bpy_operators`, `search_bpy_types` |
| **Navigation** | `browse_hierarchy`, `read_page` |
| **Validation** | `validate_parameter_range`, `get_parameter_defaults` |

---

## Self-Learning Strategies

### Strategy 1: Vector Store for Blender Documentation

**Status:** Implemented
**Vector Store ID:** `vs_696acc41b74c8191a8d6f614c0223923`
**Documents:** 4,225 files indexed

Uses OpenAI vector stores for semantic search over Blender documentation. Unlike keyword search, finds conceptually related content (e.g., searching "turbulence" also finds "vorticity", "noise_strength").

**Tools:**
- `semantic_search_blender_docs(query, max_results, include_code_examples)`
- `find_alternative_approaches(current_approach, issue, effect_type, exclude_techniques)`
- `search_blender_api_by_intent(intent, domain)`

---

### Strategy 2: Knowledge Distillation Loop

**Status:** Implemented

Automatically extracts successful code patterns from script modifications. Analyzes diffs to identify what changed and generalizes into reusable templates.

**Tools:**
- `extract_successful_pattern(original_path, modified_path, issue, improvement, effect_type, experiment_id)`
- `apply_pattern_to_script(script_path, pattern_id, output_path, parameter_overrides)`
- `analyze_script_for_patterns(script_path, effect_type)`
- `compare_scripts(script_a_path, script_b_path)`

**Trigger:** Called after any iteration with >= 5 point improvement

---

### Strategy 3: Proactive Documentation Mining

**Status:** Implemented

Pre-iteration research phase that detects early warning signs and proactively searches for alternatives before getting stuck.

**Tools:**
- `pre_iteration_research(current_issue, current_approach, iteration_history, effect_type)`
- `search_alternative_approaches(issue, effect_type, current_techniques)`

**Warning Levels:**
| Level | Trigger | Action |
|-------|---------|--------|
| `none` | Normal progress | Proceed |
| `early` | Same issue 2x | Check knowledge base |
| `stuck` | Same issue 3x | Switch technique or mine docs |

---

### Strategy 4: Code Pattern Memory

**Status:** Implemented
**Storage:** `data/code_patterns/*.json`

Stores actual working Python code snippets (not just parameter metadata) for semantic retrieval. Tracks confidence based on success rate and usage count.

**Tools:**
- `record_code_pattern(issue, code_snippet, effect_type, improvement, experiment_id)`
- `search_code_patterns(issue, effect_type, min_confidence, max_results)`
- `get_pattern_code(pattern_id)`
- `report_pattern_outcome(pattern_id, success, improvement)`
- `get_pattern_library_stats()`
- `list_patterns_by_effect(effect_type, min_confidence)`

**Pattern Confidence Formula:**
```
confidence = (success_rate * 0.6) + (usage_factor * 0.3) + (improvement_factor * 0.1)
where:
  usage_factor = min(usage_count / 10, 1.0)
  improvement_factor = min(average_improvement / 20, 1.0)
```

---

### Strategy 5: Escape Velocity Mechanism

**Status:** Implemented

Escalating exploration strategy when iterations plateau. Tracks stuck indicators and progressively escalates to more aggressive exploration.

**Escape Levels:**

| Level | Name | Trigger | Action |
|-------|------|---------|--------|
| 0 | `NORMAL` | Default | Standard modification |
| 1 | `KNOWLEDGE_CHECK` | Plateau 2x | Query knowledge base |
| 2 | `SWITCH_TECHNIQUE` | Same issue 2x | Generate new script with different technique |
| 3 | `MINE_DOCS` | Same issue 3x | Semantic search for novel approaches |
| 4 | `REQUEST_GUIDANCE` | No progress 4x | Report stuck, request human input |

**State Tracking:**
```python
class StuckDetectionState:
    same_issue_count: int       # Consecutive iterations with same primary issue
    plateau_count: int          # Consecutive iterations with <3 point change
    escape_level: EscapeLevel   # Current escalation level
    techniques_tried: List[str] # Techniques attempted this session
    techniques_failed: List[str] # Techniques that didn't make progress
```

**Step-Down Capability:** If significant progress resumes (score +5 or issue changes), escape level can step down after 2 consecutive progress iterations.

---

## Pydantic Models

### Core Models

| Model | Purpose | Key Fields |
|-------|---------|------------|
| `AssetRequest` | User request | asset_name, description, effect_type, quality_threshold, max_iterations |
| `SessionState` | Persistent state | session_id, status, iterations, best_score, stuck_state |
| `IterationResult` | Single iteration | script, execution, quality, passed, score, improvement |
| `SharedContext` | Agent coordination | session, current_script, current_execution, current_quality |

### Quality Models

| Model | Purpose | Key Fields |
|-------|---------|------------|
| `QualityMetrics` | Evaluation results | overall_score, passed, lpips_score, issues, primary_issue, suggestions |
| `ScriptModification` | Script changes | script_path, modifications, technique_name, validation_passed |
| `BlenderExecution` | Run results | success, run_dir, render_path, vdb_files, errors |

### Learning Models

| Model | Purpose | Key Fields |
|-------|---------|------------|
| `ExperimentOutcome` | Experiment result | hypothesis, issue_addressed, success, score_improvement, learnings |
| `CodePattern` | Stored pattern | pattern_id, code_snippet, issue_category, average_improvement, confidence |
| `StuckDetectionState` | Escape tracking | escape_level, same_issue_count, techniques_tried |

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | (required) | OpenAI API key for GPT-5.2 |
| `ORCHESTRATOR_MODEL` | `gpt-5.2` | Main orchestrator model |
| `SCRIPT_WRITER_MODEL` | `gpt-5.2` | Script writer model |
| `EXECUTOR_MODEL` | `gpt-5.2` | Executor model |
| `QUALITY_ANALYST_MODEL` | `gpt-5.2` | Quality analyst model |
| `LEARNING_AGENT_MODEL` | `gpt-5.2` | Learning agent model |
| `DOC_EXPERT_MODEL` | `gpt-5.2` | Docs expert model |
| `MONTHLY_BUDGET_USD` | `20` | Monthly budget limit |
| `STATE_DIR` | `build/orchestrator_state` | Session persistence directory |
| `BLENDER_DOCS_VECTOR_STORE_ID` | (required) | OpenAI vector store ID |

### Budget Allocation

| Category | Allocation | Purpose |
|----------|------------|---------|
| Vision/Evaluation | $10 | QualityAnalyst GPT-4o vision calls |
| Documentation Search | $8 | DocsExpert semantic search |
| Emergency Buffer | $2 | Overhead and retries |

---

## MCP Server Dependencies

| Server | Port | Purpose |
|--------|------|---------|
| `script-generator` | 8101 | Script generation and validation |
| `blender-executor` | 8102 | Blender execution |
| `asset-evaluator` | 8103 | ML quality evaluation |
| `experiment-tracker` | 8104 | Knowledge base and learning |
| `blender-manual` | (embedded) | Documentation search (function_tools) |

---

## Session Lifecycle

### States

```
PENDING → IN_PROGRESS → PASSED
                     ↘ FAILED
                     ↘ MAX_ITERATIONS
                     ↘ PAUSED (recoverable)
                     ↘ CANCELLED
```

### Persistence

Sessions are saved to JSON after each iteration:
```
{state_dir}/sessions/
└── {session_id}.json
```

Recovery is automatic when calling `resume_session(session_id)`.

---

## Performance Characteristics

### Typical Session

| Metric | Value |
|--------|-------|
| Iterations to pass | 2-5 |
| Time per iteration | 30-90 seconds |
| API cost per iteration | $0.15-0.50 |
| Success rate | ~85% within max_iterations |

### Bottlenecks

| Operation | Time | Notes |
|-----------|------|-------|
| Blender simulation | 10-60s | Depends on resolution |
| Quality evaluation | 5-15s | Vision API + ML metrics |
| Script generation | 2-5s | GPT-5.2 reasoning |
| Documentation search | 1-3s | Vector store query |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-15 | Initial release with 5 agents |
| 1.1.0 | 2026-01-16 | Added Strategy 3, 5 (proactive research, escape velocity) |
| 2.0.0 | 2026-01-17 | Added Strategy 1, 2, 4 (vector store, distillation, patterns) |
| 2.0.1 | 2026-01-18 | Upgraded all agents to GPT-5.2 |

---

*End of Specification*
