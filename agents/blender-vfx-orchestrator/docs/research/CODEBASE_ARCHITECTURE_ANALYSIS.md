# Codebase Architecture Analysis: blender-vfx-orchestrator

**Date:** 2026-02-22
**Scope:** Complete architecture analysis for Phase 2+ planning
**Codebase:** ~43,500 LOC across 60+ Python files

---

## 1. Agent Hierarchy

### 1.1 Standalone Agents (LLM-powered, called via `Runner.run()`)

| Agent | Model | Reasoning | Max Turns | Structured Output | Dynamic Instructions |
|-------|-------|-----------|-----------|-------------------|---------------------|
| Research Agent | gpt-5.2 | high | 4 | `ResearchOutput` | No |
| Script Writer | gpt-5.2 | high | 10 | `ScriptOutput` | Yes (`dynamic_script_writer_instructions`) |
| Executor | gpt-5.2 | medium | 3 | `ExecutionOutput` | No |
| Quality Analyst | gpt-5.2 | high | 6 | `QualityOutput` | Yes (`dynamic_quality_analyst_instructions`) |
| Learning Agent | gpt-5.2 | high | 8 | `LearningOutput` | Yes (`dynamic_learning_agent_instructions`) |
| Docs Expert | gpt-5.2 | medium | 3 | `ResearchOutput` | No |

### 1.2 Coordinator Agents (Decision-making, agents-as-tools pattern)

| Coordinator | Model | Sub-agents (as tools) | Structured Output | Purpose |
|-------------|-------|-----------------------|-------------------|---------|
| Technique Selector | gpt-5.2 (high) | research_approach, docs_search | `TechniqueDecision` | Phase 0.5: Choose initial technique |
| Modification Strategist | gpt-5.2 (high) | — | `ModificationDecision` | Phase 1.1: Decide modify vs switch |
| Quality Gate Judge | gpt-5.2 (high) | — | `QualityDecision` | Phase 5: Pass/fail + next action |

### 1.3 Spec-First Pipeline Agents (Hallucination prevention)

| Agent | Model | Output Guardrail | Purpose |
|-------|-------|------------------|---------|
| API Spec Agent | gpt-5.2 (high, parallel_tool_calls=False) | `validate_api_spec` | Generate verified API spec with doc_refs |
| Code Writer Agent | gpt-5.2 (high) | `validate_code_writer_output` | Generate script constrained by API spec |

**Note:** The Spec-First pipeline is partially superseded by the Truth Pack (Phase 1), but both run. The Spec-First pipeline is LLM-based ($$$), while the Truth Pack is deterministic ($0).

### 1.4 Agent Creation (orchestrator.py:1840-2040)

All agents are created in `BlenderVFXOrchestrator.initialize()`. Each standalone agent gets:
- A `ModelSettings` configured via `_get_model_settings()`
- Tools from the function_tool registry
- Optional `RunHooks` from `hooks/enforcement_hooks.py`
- Optional output guardrails from `guardrails/api_spec_guardrails.py`
- Dynamic instructions (functions, not strings) from `tools/dynamic_instructions.py`

---

## 2. Pipeline State Machine

### 2.1 Complete Phase Flow

```
Phase 0:   Research (parallel: Research Agent + Docs Expert)
Phase 0.5: Technique Selection + API Spec (parallel)
Phase 0.6: Build Truth Pack (deterministic, ~2s Blender subprocess)
           ┌──────── ITERATION LOOP ────────┐
Phase 0.9: │ Pre-Iteration Research (iter>1) │
Phase 0.95:│ Self-Learning Reuse             │
Phase 1:   │ Script Generation               │
Phase 1.0.1│ Pattern Application             │
Phase 1.1: │ Modification Strategy (iter>1)  │
Phase 1.5: │ API Validation (truth pack)     │
Phase 2:   │ Execution (DETERMINISTIC)       │
Phase 2.5: │ Diagnose (on failure)           │
Phase 2.6: │ Fix Plan (on failure)           │
Phase 2.7: │ Artifact Gates                  │
Phase 2.8: │ Diagnose artifact gate fail     │
Phase 2.9: │ Fix artifact gate fail          │
Phase 3:   │ Quality Evaluation (LLM judge)  │
Phase 3.5: │ QA Diagnosis Bridge             │
Phase 4+5: │ Learning + Quality Gate (parallel or sequential) │
Phase 4.5: │ Pattern Extraction              │
Phase 4.6: │ Pattern Outcome Reporting       │
           │ Session Compaction              │
           └────────────────────────────────┘
           Pipeline Complete → Save Session
```

### 2.2 Failure Routing

| Phase | Failure | Route |
|-------|---------|-------|
| Phase 1 (Script) | Generation fails | Deterministic fallback → continue |
| Phase 1.5 (Validation) | Hallucinated attrs | Auto-fix via truth pack → continue |
| Phase 2 (Execution) | Script crash | Phase 2.5→2.6→2.7 (Diagnose→Fix→Recovery) |
| Phase 2.7 (Artifact Gates) | Missing renders/cache | Phase 2.8→2.9 → continue to next iter |
| Phase 3 (Quality) | Loop detected | Score=0, continue |
| Phase 5 (Gate) | Escape Level 4 | PAUSED → break |
| Any | Budget exhausted | Break with best result |
| Any | Max iterations | MAX_ITERATIONS status |

### 2.3 Parallel Execution Points

Three `asyncio.gather` fan-out/fan-in patterns:

1. **Phase 0:** `_run_parallel_preflight()` — Research Agent + Docs Expert in parallel
2. **Phase 0.5:** `_run_parallel_technique_and_spec()` — Technique Selection + API Spec in parallel
3. **Phase 4+5:** `_run_parallel_learning_and_gate()` — Learning Agent + Quality Gate in parallel

### 2.4 LLM vs Deterministic Phases

| Phase | Type | Cost |
|-------|------|------|
| 0 Research | LLM | $$$ |
| 0.5 Technique Selection | LLM (Coordinator) | $$ |
| 0.5 API Spec | LLM (Agent) | $$ |
| 0.6 Truth Pack | **Deterministic** (Blender subprocess) | $0 |
| 0.9 Pre-Iteration Research | LLM | $$ |
| 0.95 Self-Learning Reuse | **Deterministic** (DB query) | $0 |
| 1 Script Generation | LLM (Script Writer/Code Writer) | $$$ |
| 1.0.1 Pattern Application | **Deterministic** (regex/AST) | $0 |
| 1.1 Modification Strategy | LLM (Coordinator) | $$ |
| 1.5 API Validation | **Deterministic** (truth pack) | $0 |
| 2 Execution | **Deterministic** (Blender subprocess) | $0 |
| 2.5-2.7 Error Recovery | LLM (Script Writer) | $$ |
| 2.7 Artifact Gates | **Deterministic** (file checks) | $0 |
| 3 Quality Evaluation | LLM (Quality Analyst + vision) | $$$ |
| 3.5 QA Bridge | **Deterministic** (script analysis) | $0 |
| 4 Learning | LLM (Learning Agent) | $$ |
| 4.5-4.6 Pattern Tracking | **Deterministic** | $0 |
| 5 Quality Gate | LLM (Coordinator) | $ |

**Cost breakdown per iteration:** ~6 LLM calls (Script Writer, Quality Analyst, Learning, Quality Gate, + optional Modification Strategist, Error Recovery). Quality Analyst with vision is the most expensive single call.

---

## 3. Tool Inventory

### 3.1 Tool → Agent Mapping

| Tool File | LOC | Used By | Type |
|-----------|-----|---------|------|
| `tools/script_generator_tools.py` | 1,584 | Script Writer, Code Writer | LLM-callable (`@function_tool`) |
| `tools/blender_executor_tools.py` | 926 | Orchestrator (direct), Executor Agent | Deterministic subprocess |
| `tools/semantic_docs_tools.py` | 1,146 | Docs Expert, API Spec Agent, Research Agent | LLM-callable, OpenAI vector store |
| `tools/asset_evaluator_tools.py` | 921 | Quality Analyst | LLM-callable, vision API |
| `tools/experiment_tracker_tools.py` | 1,034 | Learning Agent | LLM-callable, SQLite |
| `tools/dynamic_instructions.py` | 1,113 | Script Writer, Quality Analyst, Learning Agent | Dynamic instruction generators |
| `tools/truth_pack.py` | 743 | Orchestrator (direct) | Deterministic, Blender subprocess |
| `tools/truth_pack_validator.py` | 187 | Orchestrator, agents | Validate-fix cycle |
| `tools/blender_api_fixer.py` | 1,962 | Script Writer (legacy) | 57-rule regex fixer |
| `tools/code_pattern_tools.py` | 492 | Learning Agent | LLM-callable, JSON storage |
| `tools/knowledge_distillation_tools.py` | 814 | Learning Agent | LLM-callable |
| `tools/proactive_research_tools.py` | 676 | Orchestrator (Phase 0.9) | LLM-callable |
| `tools/qa_diagnosis_bridge.py` | 254 | Orchestrator (Phase 3.5) | Deterministic |
| `tools/script_analysis_tools.py` | 450 | QA Bridge, Learning Agent | Deterministic analysis |
| `shared/blender_docs_tools.py` | 1,675 | Docs Expert, API Spec Agent | OpenAI vector store queries |
| `tools/trace_visualizer.py` | 1,232 | Diagnostic hooks | Trace analysis |

### 3.2 Two-Layer Pattern

All tools follow the SDK-mandated two-layer pattern:

```python
# Layer 1: _impl function (callable by other Python code)
async def _execute_blender_script_impl(script_path: str, ...) -> ExecutionResult:
    ...

# Layer 2: @function_tool wrapper (callable by agents)
@function_tool
async def execute_blender_script(script_path: str, ...) -> str:
    result = await _execute_blender_script_impl(script_path, ...)
    return json.dumps(asdict(result))
```

This enables the orchestrator to call `_impl` functions directly (deterministic execution in Phase 2) while agents can call the `@function_tool` wrappers.

### 3.3 Redundant Validation Layers

Four overlapping API validation mechanisms exist:

| Layer | File | Type | Cost | Coverage |
|-------|------|------|------|----------|
| 1. Truth Pack | `tools/truth_pack.py` | Deterministic (bl_rna introspection) | $0 | Complete for introspected types |
| 2. API Spec Guardrail | `guardrails/api_spec_guardrails.py` | Deterministic (hardcoded dict) | $0 | ~30 known deprecations |
| 3. API Validator | `specialized_agents/api_validator.py` | Deterministic (KNOWN_API_CHANGES) | $0 | ~20 known changes |
| 4. Blender API Fixer | `tools/blender_api_fixer.py` | Deterministic (57 regex rules) | $0 | Broad but brittle |

**Recommendation:** Truth Pack (Layer 1) should subsume Layers 2-4 over time. Currently all four run, creating redundancy but also a safety net.

---

## 4. Data Models (models/)

### 4.1 Core State Models (`models/shared_context.py`, 926 LOC)

| Model | Purpose | Key Fields |
|-------|---------|------------|
| `SharedContext` | Central agent coordination | `session`, `truth_pack`, `api_spec`, `stuck_state`, `last_applied_pattern_id`, `last_gate_failure` |
| `SessionState` | Persistent session (saved to disk) | `iterations[]`, `best_score`, `stuck_state`, `techniques_tried`, `alternative_approaches`, SICA utility tracking |
| `AssetRequest` | User request params | `asset_name`, `description`, `effect_type`, `quality_threshold`, `max_iterations`, `resolution`, `frame_end` |
| `StuckDetectionState` | Escape velocity L0-L4 | `escape_level`, `consecutive_same_issue`, `techniques_tried`, `step_down_counter` |
| `IterationResult` | Per-iteration snapshot | `script`, `execution`, `quality`, `passed`, `score` |
| `ScriptModification` | Script change record | `script_path`, `modifications`, `technique_name`, `validation_passed` |
| `BlenderExecution` | Execution record | `success`, `run_dir`, `render_path`, `vdb_files`, `errors` |
| `QualityMetrics` | Quality scores | `overall_score`, `passed`, `issues`, `primary_issue` |

### 4.2 Structured Output Models (orchestrator.py:1-260)

| Model | Agent | Key Fields |
|-------|-------|------------|
| `ResearchOutput` | Research Agent | `recommended_approach`, `key_parameters`, `alternative_approaches`, `warnings` |
| `ScriptOutput` | Script Writer | `script_path`, `technique_used`, `parameters_set`, `validation_passed` |
| `ExecutionOutput` | Executor | `success`, `render_path`, `error_message`, `run_dir` |
| `QualityOutput` | Quality Analyst | `overall_score`, `passed`, `primary_issue`, `issues`, `suggestions`, `vision_assessment` |
| `LearningOutput` | Learning Agent | `experiment_recorded`, `pattern_extracted`, `pattern_id`, `next_action`, `parameter_modifications` |
| `TechniqueDecision` | Technique Selector | `selected_technique`, `reasoning`, `confidence` |
| `ModificationDecision` | Modification Strategist | `action` (modify_params/modify_code/switch_technique/continue), `parameters_to_change`, `reasoning` |
| `QualityDecision` | Quality Gate Judge | `passed`, `next_action`, `escape_level`, `reasoning` |

### 4.3 API Spec Models (`models/api_spec.py`, 466 LOC)

| Model | Purpose |
|-------|---------|
| `APISpec` | Verified API specification with `doc_ref` requirements |
| `APIAttribute` | Single attribute with name, type, range, default, doc_ref |
| `APIOperation` | API operation (create, modify, etc.) |
| `VerifiedScriptOutput` | Script output constrained by API spec |

**Note:** APISpec has field validators that reject specs without proper documentation references (enforced at Pydantic level).

---

## 5. Support Systems

### 5.1 Enforcement Hooks (`hooks/enforcement_hooks.py`, 894 LOC)

`EnforcementHooks(RunHooks)` provides lifecycle callbacks:

| Hook | Purpose |
|------|---------|
| `on_tool_start` | Loop detection (same tool 3x), doc query requirements |
| Turn budget | Configurable per-agent max turns |
| Communication flow | Tracks Coordinator → modify_script path |

Factory functions create specialized hooks per agent:
- `create_research_hooks()` — 4-turn budget
- `create_script_writer_hooks()` — 10-turn budget, doc query required
- `create_quality_analyst_hooks()` — 6-turn budget
- `create_learning_hooks()` — 8-turn budget

### 5.2 Session Manager (`session_manager.py`, 527 LOC)

Deterministic (no LLM) experiment state tracking:
- `ExperimentState` — baselines, iteration history
- `IssueTracker` — consecutive same-issue counting
- Parameter-score correlation tracking
- `get_context_for_agents()` — supplies session context to agent prompts

### 5.3 Config System (`config/agent_config.py`, 309 LOC)

YAML-based presets with `AgentSettings` dataclass:
- Handles GPT-5 parameter compatibility (temperature only with reasoning_effort=none)
- Model selection, reasoning effort, temperature, verbosity
- Preset system for different quality/speed tradeoffs

### 5.4 Budget Tracker (`utils/`)

$20/month limit with per-category allocation:
- Vision/Evaluation: $10
- Documentation Search: $8
- Emergency Buffer: $2
- `can_afford_evaluation()` check before Phase 3

### 5.5 Artifact Manager

Artifact-first handoff pattern between pipeline phases:
- Writes structured artifacts to disk (scripts, quality reports, scorecards, diagnoses)
- Passes file references (not inline content) between agents
- `get_iteration_summary(max_iterations=3)` for context-bounded history

### 5.6 Session Persistence

Save/load `SessionState` to disk:
- Resume paused/incomplete sessions via `resume_session()`
- `resume_session_id` parameter in `create_asset_pipeline()` reconstructs loop state

---

## 6. Phase 2 SDK Feature Integration Points

### 6.1 `AdvancedSQLiteSession` (Branching + Token Tracking)

**Current state:** Uses `OpenAIResponsesCompactionSession` with auto-compaction lambda at `orchestrator.py:375-381`.

**Integration points:**
- `orchestrator.py:370-385` — Replace `OpenAIResponsesCompactionSession` with `AdvancedSQLiteSession`
- Branching: Create branch per technique experiment (Phase 0.5 technique switch)
- Token tracking: Replace manual `sum(len(str(item)))` heuristic with SDK-native token counting
- `session_manager.py` — May overlap with AdvancedSQLiteSession's built-in tracking; evaluate redundancy

### 6.2 `call_model_input_filter` (Context Trimming)

**Current state:** No input filtering. Dynamic instructions inject full truth pack + knowledge base results into every prompt.

**Integration points:**
- `orchestrator.py:1840-2040` (agent creation) — Add `call_model_input_filter` to each agent
- `tools/dynamic_instructions.py` — Currently generates full instruction text; input filter could trim stale/irrelevant sections
- Priority targets: Script Writer (receives largest prompts with truth pack + KB patterns), Learning Agent (receives full iteration history)

### 6.3 `ToolInputGuardrail` / `ToolOutputGuardrail` (Tool Guardrails)

**Current state:** `guardrails/api_spec_guardrails.py` implements output guardrails on API Spec Agent and Code Writer Agent. `tools/script_generator_tools.py` imports `tool_input_guardrail` and `tool_output_guardrail` from the SDK.

**Integration points:**
- `tools/script_generator_tools.py:33-38` — Already imports guardrail types; add input guardrail to `generate_script` and `modify_script` to reject hallucinated attributes before script generation
- `tools/blender_executor_tools.py` — Add input guardrail to `execute_blender_script` to validate script path exists and is under size limit
- `specialized_agents/script_writer.py` — Add output guardrail to verify ScriptOutput contains valid technique name
- Truth pack validation could move from orchestrator-level (post-generation) to tool-level guardrail (pre-execution)

### 6.4 `is_enabled` (Conditional Tool Enabling)

**Current state:** All tools always available to all agents regardless of budget or phase.

**Integration points:**
- `tools/asset_evaluator_tools.py` — Disable vision-based evaluation tools when budget is exhausted (`budget_tracker.can_afford_evaluation()`)
- `tools/semantic_docs_tools.py` — Disable OpenAI vector store tools when doc search budget depleted
- Phase-aware enabling: Disable `switch_technique` tools before iteration 2 (no meaningful data to decide on)
- Escape-level-aware: Enable `mine_docs` tools only at escape level 3+

### 6.5 `tool_use_behavior: stop_on_first_tool` (Deterministic Agent Control)

**Current state:** Agents can call multiple tools per turn, sometimes making redundant calls.

**Integration points:**
- `specialized_agents/executor.py` — Executor should be deterministic: call `execute_blender_script` once and stop. Currently has 3-turn budget as a workaround.
- `specialized_agents/docs_expert.py` — Should do one doc search and return results, not loop
- Coordinator agents — Should make one decision and return structured output

### 6.6 `run_streamed` (Streaming)

**Current state:** All agents run with `Runner.run()` (non-streaming). No real-time visibility into agent work.

**Integration points:**
- `orchestrator.py` `_run_agent()` method (~line 800) — Central dispatch point; switching to `Runner.run_streamed()` here affects all agents
- Phase 3 (Quality Evaluation) — Streaming would show real-time evaluation progress
- Phase 1 (Script Generation) — Streaming would show script being written

### 6.7 `needs_approval` (Human-in-the-Loop)

**Current state:** Level 4 escape velocity sets `session.status = PAUSED` and breaks the loop. No structured HITL flow.

**Integration points:**
- `orchestrator.py:4190-4201` — Level 4 handler currently just breaks. Could use `needs_approval` for structured pause/resume with state serialization
- Budget exhaustion path (`orchestrator.py:3749-3760`) — Could pause for human approval instead of auto-stopping
- Technique switch decisions — Human could approve/reject technique changes at escape level 2+

---

## 7. Architecture Weaknesses

### 7.1 Orchestrator Monolith (4,544 LOC)

`orchestrator.py` contains the entire pipeline, all agent creation, all phase logic, all failure routing, and all helper methods in a single file. The `create_asset_pipeline()` method alone is ~2,300 lines.

**Impact:** Difficult to test individual phases, hard to modify one phase without reading the entire file.

**Recommendation:** Extract phases into separate modules (e.g., `phases/research.py`, `phases/execution.py`, `phases/quality.py`). The pipeline orchestrator becomes a thin coordinator that calls phase modules.

### 7.2 Redundant Validation Layers

Four overlapping API validation systems (truth pack, API spec guardrails, API validator, blender API fixer) with no clear precedence or deduplication.

**Impact:** Maintenance burden — a new Blender 5.x change requires updates in 4 places. Possible conflicting fixes.

**Recommendation:** Truth pack should be the single source of truth. Other layers can exist as safety nets but should defer to truth pack results when available.

### 7.3 Execution Is Now Deterministic But Executor Agent Still Exists

Phase 2 calls `_execute_blender_script_impl()` directly (deterministic), bypassing the Executor agent entirely. But the Executor agent is still created in `initialize()` and consumes resources.

**Impact:** Dead code. The `specialized_agents/executor.py` (170 LOC) and executor hooks are created but never used in the pipeline path.

**Recommendation:** Remove the Executor agent. If error diagnosis needs an LLM, that's a separate "Diagnosis Agent" concern, not an executor concern.

### 7.4 Spec-First Pipeline Overlap with Truth Pack

The Spec-First pipeline (API Spec Agent → Code Writer Agent) and the Truth Pack both solve the same problem: preventing API hallucinations. The Spec-First pipeline costs LLM calls; the Truth Pack costs $0.

**Impact:** Redundant LLM spend. The API Spec Agent runs in Phase 0.5 (parallel with technique selection) even though the Truth Pack in Phase 0.6 provides the same data deterministically.

**Recommendation:** Consider deprecating the Spec-First pipeline in favor of Truth Pack + direct Code Writer with truth pack injection. Alternatively, use the Truth Pack to validate the API Spec Agent's output (cheaper than regenerating specs).

### 7.5 Dynamic Instructions Inject Full Content Every Call

`tools/dynamic_instructions.py` queries the KB and builds full instruction text every time an agent is invoked. This includes truth pack formatting, KB pattern results, and session context — all injected into system prompt.

**Impact:** Token waste. The truth pack is static per session; it shouldn't be reformatted every iteration. KB results may be identical across iterations.

**Recommendation:** Cache formatted truth pack at Phase 0.6 (already partially done — `format_truth_pack_for_prompt()` result could be stored). Use `call_model_input_filter` to trim dynamic instructions that haven't changed.

### 7.6 No Monitoring of Agent-to-Agent Communication Quality

The pipeline logs `[Pipeline]` messages to stderr but has no structured monitoring of:
- Whether agents actually use the truth pack injected into their instructions
- Whether the Quality Analyst's feedback is actionable (or vague)
- Whether the Learning Agent's `parameter_modifications` match actual script identifiers

**Impact:** Silent quality degradation. Agents may ignore injected context, and the pipeline has no way to detect this.

**Recommendation:** Phase 2 monitoring agent/layer that audits agent outputs against expectations (e.g., "Script Writer output references truth pack attributes" → yes/no).

### 7.7 Session Compaction Heuristic Is Brittle

The compaction trigger uses `sum(len(str(item)) for item in (history or [])) > 100_000` — a character-count heuristic that doesn't account for actual token usage.

**Impact:** May trigger too early (wasting useful context) or too late (hitting token limits). The `str(item)` conversion is also expensive for large history objects.

**Recommendation:** Use SDK-native token counting when available (AdvancedSQLiteSession). Alternatively, count based on message count rather than character length.

### 7.8 Error Recovery Creates New Scripts Without Full Context

Phase 2.5-2.7 (error recovery) generates fix scripts using the Script Writer, but the fix prompt doesn't always include the truth pack or the full original script context.

**Impact:** Recovery scripts may re-introduce the same hallucinated attributes that the truth pack was designed to prevent.

**Recommendation:** Ensure truth pack is injected into all error recovery prompts. The `_run_spec_first_modification()` path does this; verify the standard modification path does too.

### 7.9 Learning Agent Has Too Many Tools (20+)

The Learning Agent has access to 20+ tools spanning experiment tracking, knowledge distillation, code patterns, physics observation, script analysis, and documentation mining.

**Impact:** Tool selection confusion. The LLM must choose from 20+ tools per turn, leading to suboptimal tool usage and wasted turns.

**Recommendation:** Use `is_enabled` to contextually enable only relevant tools. E.g., at iteration 1, disable `mine_docs_for_patterns` (no data yet). At iteration 3+, enable escape-velocity tools.

### 7.10 No Graceful Budget Degradation

Budget exhaustion is binary: either `can_afford_evaluation()` returns true (full quality evaluation) or false (break immediately with last score).

**Impact:** No intermediate modes. Could do lightweight evaluation (no vision, just structural checks) when budget is low.

**Recommendation:** Tiered evaluation: full vision ($$$) → structural-only ($) → deterministic-only ($0) based on remaining budget.

---

## 8. Phase 1 Implementation Review

### 8.1 Truth Pack (`tools/truth_pack.py`, 743 LOC)

**Strengths:**
- Complete `bl_rna.properties` introspection via Blender subprocess
- 15+ known hallucination patterns with regex matching
- `auto_fix_script()` applies deterministic corrections
- `truth_pack_to_api_spec()` provides backward compatibility with existing APISpec infrastructure
- Cache to `data/truth_packs/{blender_version}_{technique}.json`

**Gaps:**
- `TECHNIQUE_TYPES` mapping covers mantaflow_gas, mantaflow_liquid, rigid_body, particle_system, plus `_common`. Missing: cloth, geometry_nodes, soft_body.
- `SETTINGS_MAP` and `VARIABLE_PATTERNS` are hardcoded — new variable naming conventions in scripts won't be caught
- No validation of material/shader node attributes (only physics simulation types)

### 8.2 QA Diagnosis Bridge (`tools/qa_diagnosis_bridge.py`, 254 LOC)

**Strengths:**
- Maps 12 QA issue keywords to script parameter categories
- Uses existing `analyze_script_structure()` for code parsing
- Pairs "too dark" with actual `light.energy = 10` line references

**Gaps:**
- Keyword matching is string-based (not semantic). "Insufficient illumination" won't match "dark"
- Only 12 issue categories; new issue types require manual additions

### 8.3 Session Compaction

**Strengths:**
- Replaces disabled `lambda _: False` with actual trigger
- SharedContext and SessionState survive compaction (Python-managed)

**Gaps:**
- Character-count heuristic (see weakness 7.7)

### 8.4 Escape Velocity

**Strengths:**
- Level 4 → PAUSED verified working (16 tests)
- `get_untried_techniques()` properly excludes tried techniques
- Step-down mechanism for recovery after progress

**Gaps:**
- Technique switch re-runs full Research Agent ($$$). Could use cached research from Phase 0 filtered by untried techniques.

---

## 9. Key File Reference

### Top 20 Files by LOC

| # | File | LOC | Role |
|---|------|-----|------|
| 1 | `orchestrator.py` | 4,544 | Main pipeline orchestration |
| 2 | `tools/blender_api_fixer.py` | 1,962 | Legacy 57-rule regex fixer |
| 3 | `shared/blender_docs_tools.py` | 1,675 | OpenAI vector store doc search |
| 4 | `tools/script_generator_tools.py` | 1,584 | Script generation/modification |
| 5 | `tools/trace_visualizer.py` | 1,232 | Trace analysis |
| 6 | `tools/semantic_docs_tools.py` | 1,146 | Semantic doc search tools |
| 7 | `tools/dynamic_instructions.py` | 1,113 | Runtime instruction injection |
| 8 | `tools/experiment_tracker_tools.py` | 1,034 | Knowledge base operations |
| 9 | `tools/blender_executor_tools.py` | 926 | Blender subprocess management |
| 10 | `models/shared_context.py` | 926 | All data models |
| 11 | `tools/asset_evaluator_tools.py` | 921 | ML quality metrics |
| 12 | `specialized_agents/api_validator.py` | 907 | API validation |
| 13 | `hooks/enforcement_hooks.py` | 894 | RunHooks implementation |
| 14 | `tools/knowledge_distillation_tools.py` | 814 | Pattern extraction |
| 15 | `tools/truth_pack.py` | 743 | Phase 1: Deterministic validation |
| 16 | `tools/proactive_research_tools.py` | 676 | Pre-iteration research |
| 17 | `guardrails/api_spec_guardrails.py` | 618 | Spec-first guardrails |
| 18 | `session_manager.py` | 527 | Deterministic state tracking |
| 19 | `tools/code_pattern_tools.py` | 492 | Code snippet storage |
| 20 | `models/api_spec.py` | 466 | API spec Pydantic models |

### Total: ~43,500 LOC across 60+ Python files

---

## 10. Summary for Phase 2 Planning

### Highest-Impact SDK Features (ordered by expected ROI)

1. **`is_enabled` (Conditional Tools)** — Immediate cost savings. Disable expensive tools when budget is low. Reduce Learning Agent tool confusion from 20+ to ~8 contextually relevant tools.

2. **`tool_use_behavior: stop_on_first_tool`** — Eliminate multi-tool loops in Docs Expert, Executor (if revived), and Coordinator agents. Reduces turn count and cost.

3. **`call_model_input_filter`** — Trim stale context from agent prompts. Cache truth pack formatting. Prevent dynamic instructions from injecting unchanged content.

4. **`AdvancedSQLiteSession`** — Native token tracking replaces brittle character-count heuristic. Branching enables parallel technique experiments.

5. **`needs_approval` (HITL)** — Structured pause/resume at escape level 4, budget exhaustion, and technique switches. Currently uses ad-hoc `session.status = PAUSED`.

6. **`run_streamed` (Streaming)** — Real-time visibility. Lower priority than cost/quality improvements but important for debugging and user experience.

### Highest-Impact Architecture Changes

1. **Deprecate Spec-First pipeline** — Replace with Truth Pack + direct Script Writer. Saves ~2 LLM calls per run.
2. **Remove Executor Agent** — Execution is already deterministic. Remove dead code.
3. **Extract pipeline phases** — Break orchestrator.py monolith into phase modules.
4. **Consolidate validation layers** — Truth Pack as single source, others as safety nets.
5. **Tiered quality evaluation** — Full vision → structural → deterministic based on budget.
