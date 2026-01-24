# Blender VFX Orchestrator - Agent Specification v3.0

**Version:** 3.0.0
**Date:** 2026-01-23
**Model:** GPT-5.2 with Reasoning (all agents)
**SDK:** OpenAI Agents SDK v0.6.9+

---

## Executive Summary

The Blender VFX Orchestrator is an autonomous multi-agent system that generates high-quality volumetric VFX assets through iterative improvement. It uses a **code-based pipeline** with **agents-as-tools** pattern and **3 Coordinator agents** for intelligent decisions, coordinating 7 specialized agents with 5 self-learning strategies for continuous improvement.

> **Architecture Note:** This system uses **agents-as-tools** pattern (NOT handoffs). The deprecated handoff-based `create_asset()` method should NOT be used. Use `create_vfx_asset()` which calls `create_asset_pipeline()`.

### Key Capabilities

| Capability | Description |
|------------|-------------|
| **Autonomous Iteration** | Generates and refines scripts until quality thresholds met |
| **Multi-Agent Coordination** | 7 specialized agents + 3 Coordinators via agents-as-tools |
| **Defense-in-Depth Validation** | RunHooks (tool-level) + Guardrails (agent-level) |
| **Conversation Persistence** | SDK Sessions enable agents to share context across phases |
| **Self-Learning** | 5 strategies for knowledge accumulation and reuse |
| **Stuck Detection** | Escape velocity mechanism with 5 escalation levels |
| **Session Persistence** | Recoverable sessions across context limits |
| **Budget Enforcement** | $20/month limit with per-category tracking |
| **API Validation** | Automatic Blender 5.0 API correction (Phase 1.5) |

---

## System Architecture

### Agent Hierarchy

```
create_asset_pipeline() (Python-controlled)
│
├── Coordinator Agents (Decision-Making)
│   ├── TechniqueSelector (GPT-5.2)      - Initial technique selection (Phase 0.5)
│   ├── ModificationStrategist (GPT-5.2) - Modification strategy (Phase 1.1)
│   └── QualityGateJudge (GPT-5.2)       - Quality gate decisions (Phase 5)
│
├── Specialized Agents (Execution)
│   ├── ResearchAgent (GPT-5.2)     - Documentation & approach research
│   ├── ScriptWriter (GPT-5.2)      - Blender Python script generation
│   ├── APIValidator (GPT-5.2)      - Blender 5.0 API validation
│   ├── Executor (GPT-5.2)          - Script execution, error handling
│   ├── QualityAnalyst (GPT-5.2)    - ML-powered quality evaluation
│   ├── LearningAgent (GPT-5.2)     - Experiment tracking, knowledge base
│   └── DocsExpert (GPT-5.2)        - Blender documentation search
│
├── Enforcement Layer
│   ├── RunHooks          - Tool-level enforcement (loop detection, doc requirements)
│   └── Guardrails        - Agent-level validation (input/output validation)
│
└── Persistence Layer
    └── SDK Sessions      - Conversation context shared across all agents (SQLiteSession)
```

### Data Flow (Code-Based Pipeline)

```
Request → create_asset_pipeline()
              │
              ├── [Phase 0] Research Agent ───────────────────→ ResearchOutput (Phase 3: Pydantic)
              │
              ├── [Phase 0.5] TechniqueSelector Coordinator ──→ TechniqueDecision
              │
              ├── [Iteration Loop]
              │   ├── [Phase 1.1] ModificationStrategist ─────→ ModificationDecision
              │   ├── [Phase 1] Script Writer ────────────────→ Script
              │   ├── [Phase 1.5] API Validator (inline) ─────→ Corrected Script
              │   ├── [Phase 2] Executor ─────────────────────→ Render Output
              │   ├── [Phase 3] Quality Analyst ──────────────→ Quality Metrics
              │   ├── [Phase 4] Learning Agent ───────────────→ Experiment Record
              │   └── [Phase 5] QualityGateJudge Coordinator ─→ QualityDecision
              │       ├── passed=True ──→ Complete
              │       └── passed=False ─→ Continue Loop
              │
              └── Session Complete / Max Iterations
```

**Key Architectural Decisions:**
- Python controls the pipeline sequence (deterministic)
- Coordinators make intelligent decisions at specific points (LLM reasoning)
- Defense-in-depth: RunHooks + Guardrails validate all agent interactions

---

## Agent Specifications

### Coordinator Agents (Decision-Making Layer)

#### 1. TechniqueSelector Coordinator

**Model:** `gpt-5.2` with `reasoning.effort="medium"`
**Role:** Select initial technique based on research findings (Phase 0.5)
**Output Type:** `TechniqueDecision`
**Guardrails:** `validate_technique_decision` (output)

| Field | Type | Description |
|-------|------|-------------|
| `selected_technique` | str | Technique to use (e.g., "mantaflow_fire") |
| `reasoning` | str | Why this technique was selected |
| `key_parameters` | dict | Important parameters to set |
| `alternative_techniques` | list | Fallback techniques if first fails |

#### 2. ModificationStrategist Coordinator

**Model:** `gpt-5.2` with `reasoning.effort="medium"`
**Role:** Decide modification strategy for iteration 2+ (Phase 1.1)
**Output Type:** `ModificationDecision`
**Guardrails:** `validate_modification_decision` (output)

| Field | Type | Description |
|-------|------|-------------|
| `action` | str | One of: `modify_params`, `switch_technique`, `continue` |
| `parameter_changes` | dict | Parameters to change (if action=modify_params) |
| `new_technique` | str | Technique to switch to (if action=switch_technique) |
| `reasoning` | str | Explanation of the decision |
| `confidence` | float | 0.0-1.0 confidence in this strategy |

#### 3. QualityGateJudge Coordinator

**Model:** `gpt-5.2` with `reasoning.effort="medium"`
**Role:** Interpret quality results and decide next action (Phase 5)
**Output Type:** `QualityDecision`
**Guardrails:** `validate_quality_decision` (output)

| Field | Type | Description |
|-------|------|-------------|
| `passed` | bool | Whether quality gate passed |
| `should_continue` | bool | Whether to continue iterating |
| `next_action` | str | One of: `complete`, `iterate`, `switch_technique`, `request_guidance` |
| `escape_level` | int | Escalation level (0-4) |
| `reasoning` | str | Explanation of the decision |

---

### Specialized Agents (Execution Layer)

#### 1. Pipeline Orchestrator (Python-Controlled)

**Implementation:** `create_asset_pipeline()` function
**Role:** Coordinates all agents via deterministic Python code with Runner.run()

**RunHooks Applied:**
| Agent | Hook Factory | Purpose |
|-------|--------------|---------|
| Research | `create_research_hooks()` | Max 3 same-tool, 8 turns |
| Script Writer | `create_script_writer_hooks()` | Requires doc query first |
| Quality Analyst | `create_quality_analyst_hooks()` | Max 3 same-tool, 6 turns |
| Learning Agent | `create_learning_agent_hooks()` | Max 3 same-tool, 8 turns |

**Direct Tools Available to Coordinators (19 total):**

| Category | Tools | Purpose |
|----------|-------|---------|
| **Proactive Research** | `pre_iteration_research`, `evaluate_escape_velocity`, `search_alternative_approaches` | Strategy 3: Early warning detection |
| **Semantic Search** | `semantic_search_blender_docs`, `find_alternative_approaches`, `search_blender_api_by_intent` | Strategy 1: Vector store search |
| **Code Patterns** | `record_code_pattern`, `search_code_patterns`, `get_pattern_code`, `report_pattern_outcome`, `get_pattern_library_stats`, `list_patterns_by_effect` | Strategy 4: Pattern memory |
| **Knowledge Distillation** | `extract_successful_pattern`, `apply_pattern_to_script`, `analyze_script_for_patterns`, `compare_scripts` | Strategy 2: Auto-extraction |

**Agents-as-Tools (for Coordinators):**

| Tool Name | Target Agent | When Used |
|-----------|--------------|-----------|
| `research_approach` | ResearchAgent | Research best approach for effect type |
| `write_script` | ScriptWriter | Generate/modify Blender scripts |
| `execute_script` | Executor | Execute scripts, handle errors |
| `evaluate_quality` | QualityAnalyst | Evaluate render quality |
| `record_learning` | LearningAgent | Doc-grounded proposals + record outcomes |
| `search_docs` | DocsExpert | Search Blender documentation |

---

#### 1. Research Agent (Phase 3: Structured Output)

**Model:** `gpt-5.2` with `reasoning.effort="medium"`
**Role:** Research best approach for VFX effect BEFORE script generation (Phase 0)
**Output Type:** `ResearchOutput` (Pydantic schema)
**Guardrails:** `validate_research_output` (output)
**RunHooks:** `create_research_hooks()` - max 4 same-tool calls, 4 turns

##### Turn Budget
| Target | Hard Limit | Usage |
|--------|------------|-------|
| 4 turns | 4 turns | Aligned with prompt budget (Phase 3) |

##### Tools

| Tool | Source | Purpose |
|------|--------|---------|
| `semantic_search_blender_docs` | semantic-docs | Search Blender 5.0 documentation |
| `find_alternative_approaches` | semantic-docs | Find alternative techniques |
| `search_blender_api_by_intent` | semantic-docs | Find APIs by description |
| `search_code_patterns` | code-patterns | Find proven code patterns |
| `list_patterns_by_effect` | code-patterns | List patterns by effect type |

##### Output Schema (ResearchOutput)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `recommended_approach` | str | Yes | Best approach from documentation |
| `key_parameters` | Dict[str, Any] | No | Recommended parameter values |
| `api_modules` | List[str] | No | Blender API modules to use |
| `code_patterns` | List[{pattern_id, issue, code_snippet}] | No | Proven patterns from library |
| `warnings` | List[str] | No | Potential pitfalls to avoid |
| `alternative_approaches` | List[str] | No | Backup approaches if primary fails |
| `doc_refs` | List[str] | Yes | Blender 5.0 doc references (Phase 3) |

---

#### 2. Script Writer

**Model:** `gpt-5.2` with `reasoning.effort="high"`, `temperature=0.3`
**Role:** Generate and modify Blender Python scripts for VFX effects
**Guardrails:** `require_research_context`, `validate_effect_type` (input), `validate_script_output` (output)
**RunHooks:** `create_script_writer_hooks()` - requires doc query before script generation

##### Tools

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

#### 3. API Validator (NEW in v3.0)

**Model:** `gpt-5.2` with `reasoning.effort="medium"`
**Role:** Validate Blender 5.0 API calls before execution (Phase 1.5)
**Location:** `specialized_agents/api_validator.py`

##### Known API Corrections

| Deprecated API | Blender 5.0 Replacement |
|----------------|------------------------|
| `inputs["Smoke"]` | `inputs["Grid"]` |
| `inputs["Smoke Color"]` | `inputs["Grid Color"]` |
| `modifier.effector_weights` | `effector_weights` (direct) |
| `flow_type` | `flow_behavior` |

##### Tools

| Tool | Purpose |
|------|---------|
| `extract_blender_api_calls` | Parse code for bpy.* API calls |
| `check_known_api_changes` | Fast check against known breaking changes |
| `validate_api_call_against_docs` | Verify against vector store (comprehensive) |
| `format_validation_report` | Generate structured validation report |

##### Integration

```python
# Phase 1.5: Between Script Generation and Execution
if script.script_path:
    api_validation = await validate_code_api(script_content)
    if not api_validation.is_valid:
        corrected_content = apply_known_corrections(script_content)
        Path(corrected_path).write_text(corrected_content)
```

---

#### 4. Executor

**Model:** `gpt-5.2` with `reasoning.effort="medium"`
**Role:** Execute Blender scripts and capture outputs

##### Tools

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

#### 5. Quality Analyst

**Model:** `gpt-5.2` with `reasoning.effort="high"`
**Role:** Evaluate render quality using vision and ML metrics
**Guardrails:** `check_budget_before_quality` (input), `validate_quality_output` (output)
**RunHooks:** `create_quality_analyst_hooks()` - max 3 same-tool calls, 6 turns

##### Tools

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

#### 6. Learning Agent

**Model:** `gpt-5.2` with `reasoning.effort="high"`
**Role:** **Mandatory pre-generation exploration controller**; propose doc-grounded ideas, define micro-experiments, record outcomes
**RunHooks:** `create_learning_agent_hooks()` - max 3 same-tool calls, 8 turns

**Execution Order:** Runs **before Script Writer** on every iteration. Proposals **must** include Blender 5 doc references; missing refs trigger Docs Expert or rejection.

##### Tools

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

#### Doc-Grounded Proposals (Required)

- `proposals[]`: technique, params, expected_effect, `doc_refs[]`
- `micro_experiments[]`: minimal script + success_criteria + `doc_refs[]`
- `anti_patterns[]`: “avoid this” + evidence (errors/metrics)

**Enforcement:** Reject any proposal with empty `doc_refs`. If a proposal uses a new API, run **at least one** micro-experiment first.

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

#### 7. Docs Expert

**Model:** `gpt-5.2` with `reasoning.effort="medium"`
**Role:** Search Blender 5.0 documentation for solutions and alternatives

##### Tools (14 total)

| Category | Tools |
|----------|-------|
| **Keyword Search** | `search_manual`, `search_tutorials`, `search_vdb_workflow`, `search_python_api`, `search_nodes`, `search_modifiers` |
| **Semantic Search** | `search_semantic` |
| **API Reference** | `list_api_modules`, `search_bpy_operators`, `search_bpy_types` |
| **Navigation** | `browse_hierarchy`, `read_page` |
| **Validation** | `validate_parameter_range`, `get_parameter_defaults` |

---

## Enforcement Layer

### RunHooks (Tool-Level Enforcement)

**Location:** `hooks/enforcement_hooks.py`

RunHooks intercept tool calls to enforce requirements before execution:

| Hook | Purpose | Trigger |
|------|---------|---------|
| `LoopDetectedError` | Stop infinite loops | Same tool called >3x |
| `DocQueryRequiredError` | Require research first | `write_script`/`modify_script` without prior doc search |
| `TurnBudgetExceededError` | Limit agent reasoning | Agent exceeds turn limit |

**Factory Functions:**

```python
create_research_hooks()        # max_same_tool=3, max_turns=8
create_script_writer_hooks()   # require_doc_query_before=[write_script, modify_script]
create_quality_analyst_hooks() # max_same_tool=3, max_turns=6
create_learning_agent_hooks()  # max_same_tool=3, max_turns=8
```

### Guardrails (Agent-Level Validation)

**Location:** `guardrails/`

Guardrails validate agent inputs and outputs at the agent level:

#### Script Writer Guardrails

| Guardrail | Type | Purpose |
|-----------|------|---------|
| `require_research_context` | Input | Blocks if no research findings in prompt |
| `validate_effect_type` | Input | Ensures valid effect type specified |
| `validate_script_output` | Output | Validates ScriptOutput has script_path, technique_used |

#### Quality Analyst Guardrails

| Guardrail | Type | Purpose |
|-----------|------|---------|
| `check_budget_before_quality` | Input | Blocks if budget exhausted (vision API is expensive) |
| `validate_quality_output` | Output | Validates score range 0-100, passed boolean, critical issue consistency |

#### Coordinator Guardrails

| Guardrail | Type | Purpose |
|-----------|------|---------|
| `validate_technique_decision` | Output | Validates selected_technique, reasoning present |
| `validate_modification_decision` | Output | Validates action, parameter_changes when modify_params |
| `validate_quality_decision` | Output | Validates passed/next_action consistency |

### Defense-in-Depth Pattern

```
Agent Input → Input Guardrails → Agent Reasoning → Tool Call
                                                       ↓
                                              RunHooks.on_tool_start()
                                                       ↓
                                                 Tool Execution
                                                       ↓
                                              RunHooks.on_tool_end()
                                                       ↓
Agent Output ← Output Guardrails ← Agent Response ←────┘
```

**Key Benefit:** Two validation layers catch issues at different points:
- RunHooks: Block tools that violate requirements (e.g., script without research)
- Guardrails: Validate agent I/O structure and consistency (e.g., invalid decision types)

---

## Persistence Layer

### SDK Sessions (Conversation Context)

**Location:** `sessions/sdk/vfx_conversations.db`

SDK Sessions provide automatic conversation persistence across all agents in a pipeline run:

```python
from agents import SQLiteSession

# Create session at pipeline start
sdk_session = get_or_create_sdk_session(session_id)

# All Runner.run() calls share the session
result = await Runner.run(agent, prompt, session=sdk_session, ...)
```

**How Context Flows:**

| Phase | Agent | Can See Previous |
|-------|-------|------------------|
| 0 | Research Agent | (starts fresh) |
| 0.5 | TechniqueSelector | Research findings |
| 1 | Script Writer | Research + technique decision |
| 2 | Executor | Script generated |
| 3 | Quality Analyst | Execution results |
| 4 | Learning Agent | Quality evaluation |
| 5 | QualityGateJudge | Full iteration context |

**Key Benefits:**
- Agents automatically reference previous phase outputs
- No manual conversation threading required
- Session persists to SQLite for durability
- Each VFX asset gets its own conversation thread

**Helper Function:**

```python
SDK_SESSIONS_DIR = Path(__file__).parent / "sessions" / "sdk"

def get_or_create_sdk_session(session_id: str) -> SQLiteSession:
    SDK_SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    db_path = SDK_SESSIONS_DIR / "vfx_conversations.db"
    return SQLiteSession(session_id, str(db_path))
```

---

## Self-Learning Strategies

### Strategy 1: Vector Store for Blender Documentation

**Status:** Implemented (v2 - Two-Store Architecture)

**Vector Stores:**
- **Manual Store:** `vs_6975104199c08191acb1495c86d581ce` - Conceptual docs (physics, tutorials)
- **API Store:** `vs_697512bf81c481919ae3b7a8ffb8223a` - Python API reference (bpy.types, bpy.ops)

**Documents:** ~17,430 chunks indexed (section-aware chunking)

Uses OpenAI vector stores for semantic search over Blender documentation. Unlike keyword search, finds conceptually related content (e.g., searching "turbulence" also finds "vorticity", "noise_strength").

**Architecture:**
- Intent-based routing auto-detects whether to search Manual or API store
- Each chunk has standardized headers: `DocType`, `DocPath`, `DocVersion`, `ChunkId`
- Returns stable `doc_refs` for agent citations and guardrail validation

**Tools:**
- `semantic_search_blender_docs(query, max_results, include_code_examples, intent)`
- `blender_doc_search_bundle(effect_type, description, intent, domain, max_results)`
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
| `BLENDER_MANUAL_VECTOR_STORE_ID` | `vs_6975104199c08191acb1495c86d581ce` | Manual docs vector store |
| `BLENDER_API_VECTOR_STORE_ID` | `vs_697512bf81c481919ae3b7a8ffb8223a` | API docs vector store |

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

## Audit Findings (2026-01-23)

This section captures **spec-to-implementation gaps** and high-impact risks identified in the current codebase.

### Critical / High

1. **Mixed orchestration models still active**
   - Spec says **agents-as-tools** only, but `orchestrator.py` still builds a **handoff-based** orchestrator for `create_asset()` (deprecated but callable).
   - **Impact:** Guardrails are only guaranteed on the first/last agent in a run; in a handoff chain, mid-agent guardrails may not fire.
   - **Recommendation:** Hard-disable `create_asset()` in production or remove the handoff pipeline entirely.

2. **Self-learning dynamic instructions are disabled in the pipeline**
   - All standalone agents are created with `use_dynamic_instructions=False`, so knowledge-base rules never get injected during the main pipeline.
   - **Impact:** The "self-learning" strategies exist in tooling but are not applied to the core runs.
   - **Recommendation:** Wrap dynamic instruction functions to append static text instead of disabling them.

3. **Output schema mismatches cause silent data loss**
   - `ScriptOutput` defines `parameters_set`, but the pipeline writes/reads `key_parameters`.
   - `ExecutionOutput` defines `execution_time_seconds`, but the pipeline checks `execution.execution_time`.
   - **Impact:** Parameter tracking and timing data are dropped or incorrect, which degrades learning and diagnostics.

### Medium

4. **Non-handoff agents still receive handoff prompt injection**
   - Standalone agents are wrapped with `prompt_with_handoff_instructions(...)` even when they have no handoffs.
   - **Impact:** Extra tokens + potential confusion in agent behavior.
   - **Recommendation:** Only add handoff prompts to agents that can actually handoff.

5. **Doc-query enforcement can be satisfied by non-doc tools**
   - RunHooks treat `search_code_patterns` as a "doc query", which can allow code generation without authoritative API validation.
   - **Recommendation:** Require at least one **documentation** tool (e.g., `semantic_search_blender_docs` or `search_blender_api_by_intent`) before `write_script`.

### Low / Informational

6. **Turn budget enforcement is not wired**
   - `TurnBudgetExceededError` is defined but `check_turn_budget()` is never invoked.
   - **Impact:** Turn budget is effectively advisory only.

7. **Session growth risk**
   - A single `SQLiteSession` is shared across all agents and iterations with no summarization.
   - **Impact:** Context can grow large and increase token cost over long runs.

### Evidence Snippets

```python
# orchestrator.py (standalone agents)
base_script_writer_standalone = create_script_writer(use_dynamic_instructions=False)
base_quality_standalone = create_quality_analyst(use_dynamic_instructions=False)
base_learning_standalone = create_learning_agent(use_dynamic_instructions=False)
```

```python
# orchestrator.py (schema mismatch)
script = ScriptOutput(
    script_path=...,
    technique_used=...,
    key_parameters=learning.parameter_modifications,  # field does not exist on ScriptOutput
)
```

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-15 | Initial release with 5 agents |
| 1.1.0 | 2026-01-16 | Added Strategy 3, 5 (proactive research, escape velocity) |
| 2.0.0 | 2026-01-17 | Added Strategy 1, 2, 4 (vector store, distillation, patterns) |
| 2.0.1 | 2026-01-18 | Upgraded all agents to GPT-5.2 |
| 3.0.0 | 2026-01-23 | **Major Architecture Update:** |
|       |            | - Added 3 Coordinator agents (agents-as-tools pattern) |
|       |            | - Added API Validator agent for Blender 5.0 API correction |
|       |            | - Deprecated handoff-based `create_asset()` method |
|       |            | - Added RunHooks for loop detection and doc requirements |
|       |            | - Added 9 Input/Output Guardrails for agent validation |
|       |            | - Full tracing with metadata for visibility |
| 3.1.0 | 2026-01-23 | Added SDK Sessions for conversation persistence across agents |

---

*End of Specification*
