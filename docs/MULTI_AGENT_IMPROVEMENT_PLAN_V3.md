# Multi-Agent Pipeline Improvement Plan v3

**Condensed from v2** - Full details in `MULTI_AGENT_IMPROVEMENT_PLAN_V2.md`
**Goal:** Address all identified issues with formal specifications
**Updated:** 2026-01-04

---

## Quick Status

| Phase | Status | Key Outcome |
|-------|--------|-------------|
| Phase 0: Foundation | ✅ COMPLETE | Created tools.py, orchestrator.py, circuit_breakers.py |
| Phase 0.5: Skill Enhancement | ✅ COMPLETE | SKILL.md has loop control, circuit breakers, knowledge integration |
| Phase 1: Orchestrator Reliability | ✅ COMPLETE | 4 modules: tool_executor, workflow_tracer, health_check, state_machine |
| Phase 2: Pre-Execution Validation | ✅ COMPLETE | validator.py, validate_script MCP tool, VALIDATE_SCRIPT state |
| Phase 2.5: Simulation Extensibility | ✅ COMPLETE | effect_registry.py, execution_patterns.py, mesh template support |
| Phase 3: Technique Selection | ✅ COMPLETE | technique_selector.py with UCB1, recommend_technique MCP tool |
| Phase 4: Knowledge Base | ✅ COMPLETE | knowledge_client.py, mandatory warnings, conflict resolution |
| Phase 5: Evaluation Reliability | 🔄 PARTIAL | Task 5.4 JSON fix done, others pending |
| Phase 6: Session Resumption | ⏳ PENDING | State versioning, file validation |
| Phase 7: External Research | ✅ DESIGNED | Blender Librarian MCP server with GPT-5.2 vision + local doc retrieval |

---

## Architecture Decision: Skills + MCP (No Agent SDK)

**Decision:** Use Claude Code Skills + MCP servers instead of Claude Agent SDK.

**Rationale:** Agent SDK requires separate API billing. Max subscription only covers Claude.ai web. Claude Code already provides agent runtime with tool execution loops.

```
SKILL.md (Workflow) → Claude Code Session → MCP Servers → State Layer
                                              ↓
                          script-generator | blender-executor | asset-evaluator
                                              ↓
                          iteration-controller | experiment-tracker | blender-manual
```

**Key Point:** `create_asset()` returns immediately with a plan - Claude Code executes the iteration loop by calling tools sequentially.

---

## Issue Summary

| # | Issue | Status |
|---|-------|--------|
| 1 | Orchestrator describes tools but may not execute | Original - mitigated by tool_executor.py |
| 2 | Technique selection too random | ✅ FIXED Phase 3 |
| 3 | Knowledge base not consulted | ✅ FIXED Phase 4 |
| 4 | External research missing | ✅ DESIGNED Phase 7 |
| 5 | Session resumption unreliable | Pending Phase 6 |
| 6 | Missing core files | ✅ FIXED Phase 0.1 |
| 7 | Dual state persistence | ✅ FIXED Phase 0.3 |
| 8 | claude_agent_sdk import | ✅ FIXED Phase 0.2 |
| 9 | No circuit breakers | ✅ FIXED Phase 0.4 |
| 10 | No pre-execution validation | ✅ FIXED Phase 2 |
| 11 | LPIPS timeout risk | Pending Phase 5 |
| 12 | No formal state machine | ✅ FIXED Phase 1.4 |
| 13 | No convergence criteria | ✅ FIXED Phase 0.4/0.5 |
| 14 | Volumetric-only effect types | ✅ FIXED Phase 2.5 |
| 15 | Mesh physics evaluation fails | Pending Phase 5.3 |
| 16 | JSON numpy serialization | ✅ FIXED Task 5.4 |

---

## Completed Phases (Condensed)

### Phase 0: Foundation ✅

**Files Created:**
- `agents/blender-orchestrator/tools.py` - MCP tool wrappers
- `agents/blender-orchestrator/orchestrator.py` - Main orchestration
- `agents/blender-orchestrator/circuit_breakers.py` - Convergence criteria

**Key Changes:**
- Removed claude_agent_sdk dependency
- iteration-controller is single source of truth for state
- Circuit breakers: MAX_ITERATIONS=10, MAX_WALL_TIME=30min, NO_IMPROVEMENT=3

### Phase 0.5: Skill Enhancement ✅

**File Modified:** `.claude/skills/blender-orchestrator/SKILL.md`

**Additions:**
- Iteration tracking (counter, best_score, best_iteration)
- Circuit breaker checks before every iteration
- Mandatory knowledge base consultation before parameter changes
- Research integration when stuck (2+ iterations without improvement)
- Quality decision tree with VFX/LPIPS/CLIP hierarchy

### Phase 1: Orchestrator Reliability ✅

**Files Created (1,920 lines total):**

| File | Lines | Key Classes |
|------|-------|-------------|
| `tool_executor.py` | 442 | ToolExecutionVerifier, ToolStatus enum, ToolSchema |
| `workflow_tracer.py` | 595 | WorkflowTracer, TraceEventType enum |
| `health_check.py` | 437 | HealthChecker, HealthState enum, ServerConfig |
| `state_machine.py` | 446 | WorkflowState (12 states), TRANSITIONS dict |

**States:** SESSION_START, GENERATE_SCRIPT, VALIDATE_SCRIPT, EXECUTE_BLENDER, EVALUATE_QUALITY, DECIDE_NEXT_ACTION, RECORD_LEARNING, CHECK_CONVERGENCE, SESSION_END, ERROR_RECOVERY, AWAITING_APPROVAL, PAUSED

### Phase 2: Pre-Execution Validation ✅

**Files Created/Modified:**
- `agents/script-generator/validator.py` - BlenderScriptValidator class
- `agents/script-generator/server.py` - Added validate_script, validate_script_inline tools
- `agents/blender-orchestrator/workflow.py` - Added VALIDATE_SCRIPT state
- `.claude/skills/blender-orchestrator/SKILL.md` - Added Stage 2.5 documentation

**Validation Checks:**
- Python syntax (AST)
- Blender 5.0 parameter ranges
- Required patterns (domain, flow for volumetric)
- Dangerous patterns (eval, exec, os.system)

**Workflow:** GENERATE_SCRIPT → VALIDATE_SCRIPT → EXECUTE_BLENDER (loops back on validation failure)

### Task 5.4: JSON Serialization ✅

**File:** `agents/asset-evaluator/server.py`

Added `convert_numpy_types()` and `safe_json_dumps()` utilities. Applied to extract_vfx_diagnostics, evaluate_vfx_quality, compare_vfx_iterations.

### Phase 2.5: Simulation Type Extensibility ✅

**Files Created:**
- `agents/script-generator/effect_registry.py` - Effect type definitions with categories (volumetric, mesh, custom)
- `agents/blender-executor/execution_patterns.py` - Live render pattern for mesh physics
- `agents/experiment-tracker/knowledge_base/soft_body_stability.json` - Soft body stability settings

**Key Changes:**
- SOFT_BODY_TEMPLATE, CLOTH_TEMPLATE, RIGID_BODY_TEMPLATE added to server.py
- generate_script() modified to detect mesh physics types and use appropriate templates
- Mesh templates use `view_layer.update()` pattern for headless physics simulation
- Recommended soft body settings: step_min=20, step_max=100, damping=2.0

### Phase 3: Intelligent Technique Selection ✅

**Files Created:**
- `agents/script-generator/technique_catalog.py` - Pyro techniques with keywords (rising_mushroom, ground_hugger, aerial_burst, etc.)
- `agents/script-generator/technique_selector.py` - UCB1 algorithm for exploration/exploitation balance

**MCP Tools Added:**
- `recommend_technique(effect_type, description)` - Returns technique recommendation with confidence
- `record_technique_outcome(technique_name, effect_type, success, final_score, iterations)` - Records performance
- `get_technique_stats(effect_type)` - Returns technique performance statistics

**Key Features:**
- UCB1 score balances average reward with exploration bonus
- Untried techniques get priority (infinite UCB score)
- Keyword filtering narrows candidates before UCB1 selection
- Performance persisted across sessions in `technique_performance.json`

### Phase 4: Knowledge Base Integration ✅

**Files Created:**
- `agents/script-generator/knowledge_client.py` - Knowledge base client for mandatory warning checks

**Key Classes:**
- `WarningCheck` - Dataclass containing warnings, severity, mitigations, recommendations
- `ConflictResolution` - Dataclass for resolving conflicting knowledge with weighted scoring
- `KnowledgeClient` - Client for accessing experiment-tracker knowledge base

**Integration Points:**
- `modify_script()` now automatically calls `check_before_modify()` for each parameter
- Returns `warnings`, `mitigations`, `has_critical_warnings` in ScriptModification result
- SKILL.md Stage 1 enhanced with knowledge preloading details
- SKILL.md Stage 5 documents automatic warning checks

**Severity Levels:**
- `critical`: Should not proceed without mitigation (patterns: MUST, NEVER, ALWAYS)
- `high`: Significant risk, proceed with caution
- `medium`: Caution advised
- `low`: Informational only

**Conflict Resolution Weights:**
- RECENCY: 2× (recent experiments more valuable)
- SUCCESS_RATE: 1× (higher rate wins)
- HUMAN_FEEDBACK: 3× (human review highly weighted)
- CONTEXT_MATCH: 5× (exact context match highest priority)

---

## Pending Phases (Full Detail)

### Phase 5: Evaluation Reliability

#### Task 5.1: LPIPS Lazy Loading

```python
class LazyLPIPS:
    _model = None
    @classmethod
    def get_model(cls):
        if cls._model is None:
            cls._model = lpips.LPIPS(net='alex')  # 528MB, only on first call
        return cls._model
```

#### Task 5.2: Quality Decision Tree

```
1. Always run VFX diagnostics first (no reference needed)
2. If VFX < 40: REJECT immediately
3. If VFX >= 60: Quality gate passed
4. Check LPIPS if reference provided (< 0.35 = PASS)
5. Check CLIP if query provided (> 0.60 = PASS)
```

#### Task 5.3: Effect Type Evaluation Registry

**File:** `agents/asset-evaluator/effect_evaluators.py` (new)

**Problem:** Evaluator penalizes clean mesh geometry (expects volumetric noise).

```python
EVALUATORS = {
    "volumetric": {  # Expects: noise, turbulence, density variance
        "metrics": ["edge_density", "noise_frequency", "dynamic_range"]
    },
    "mesh": {  # Expects: smooth surfaces, clear silhouettes
        "metrics": ["surface_smoothness", "silhouette_clarity", "shadow_definition"]
    },
    "custom": {  # CLIP only
        "metrics": ["clip_score"]
    }
}
```

---

### Phase 6: Session Resumption

#### Task 6.1: State Schema Versioning

```python
STATE_SCHEMA_VERSION = "1.0.0"

def migrate_state(state: dict) -> dict:
    # Handle pre-versioning states
    # Support last 2 schema versions
```

#### Task 6.2: File Existence Validation

Check that script_path and render_path exist before resuming.

---

### Phase 7: External Research ✅ DESIGNED

**Status:** Final plan complete, ready for implementation

**Implementation:** `plans/feat-blender-librarian-FINAL.md`

**Architecture (corrected Jan 2026 - Gemini 3 Pro's "corrections" were WRONG):**
- **Model:** GPT-5.2 (released December 11, 2025 - current SOTA)
- **API:** Responses API (recommended modern API, NOT legacy Chat Completions)
- **Doc Retrieval:** Local `blender-manual` MCP tool (NOT OpenAI Vector Store)
- **Images:** Resized to 512×512 before vision calls

**Note:** Gemini 3 Pro incorrectly claimed GPT-5.2 doesn't exist. Web research confirmed
GPT-5.2 was released December 11, 2025 with variants: gpt-5.2 (Thinking), gpt-5.2-pro,
gpt-5.2-chat-latest (Instant). The Responses API (`client.responses.create()`) is
OpenAI's recommended API since August 2025.

**Components:**
1. **Playbook System** (FREE) - Known fixes for common issues
2. **Ground Truth Cache** (FREE) - Cache reference image analysis
3. **Local Doc Retrieval** (FREE) - Uses existing `blender-manual` MCP server
4. **GPT-5.2 Vision** (PAID) - Only on escalation when stuck
5. **GPT-5.2 Synthesis** (PAID) - Combine docs + issues → safe modifications

**Budget:** $20/month split: $14 docs, $6 vision

**MCP Tools:**
- `diagnose_render_issue(render_path, reference_path, effect_type, current_issues)`
- `get_modification_advice(issues, effect_type, current_params)`
- `get_budget_status()`
- `add_to_playbook(effect_type, symptom, fix, confidence)`

**Escalation Trigger:** 2+ iterations without improvement, or repeated same issue category

**User Setup Required:**
1. OpenAI API key in `~/.plasmadxr_secrets`
2. Python venv with `openai`, `tenacity`, `pillow`, `mcp`
3. MCP server registration in Claude Code settings

**Files to Create:**
- `agents/blender-librarian/server.py` - Main MCP server
- `agents/blender-librarian/budget_tracker.json` - Cost tracking
- `agents/blender-librarian/playbooks/solar_playbook.json` - Known fixes

---

## Files Summary

### Created (Complete)
| File | Phase |
|------|-------|
| `agents/blender-orchestrator/tools.py` | 0.1 |
| `agents/blender-orchestrator/orchestrator.py` | 0.1 |
| `agents/blender-orchestrator/circuit_breakers.py` | 0.4 |
| `agents/blender-orchestrator/tool_executor.py` | 1.1 |
| `agents/blender-orchestrator/workflow_tracer.py` | 1.2 |
| `agents/blender-orchestrator/health_check.py` | 1.3 |
| `agents/blender-orchestrator/state_machine.py` | 1.4 |
| `agents/script-generator/validator.py` | 2.1 |
| `agents/script-generator/effect_registry.py` | 2.5.1 |
| `agents/blender-executor/execution_patterns.py` | 2.5.2 |
| `agents/script-generator/technique_catalog.py` | 3.1 |
| `agents/script-generator/technique_selector.py` | 3.2 |
| `agents/experiment-tracker/knowledge_base/soft_body_stability.json` | 2.5.4 |
| `agents/script-generator/knowledge_client.py` | 4.1 |

### Created (Pending)
| File | Phase |
|------|-------|
| `agents/asset-evaluator/effect_evaluators.py` | 5.3 |
| `agents/iteration-controller/state_schema.py` | 6.1 |
| `agents/blender-librarian/server.py` | 7.1 |
| `agents/blender-librarian/playbooks/solar_playbook.json` | 7.2 |

### Modified
| File | Changes |
|------|---------|
| `.claude/skills/blender-orchestrator/SKILL.md` | Loop control, circuit breakers, knowledge, decision tree, Stage 2.5 |
| `agents/blender-orchestrator/server.py` | Removed claude_agent_sdk |
| `agents/blender-orchestrator/workflow.py` | VALIDATE_SCRIPT, CHECK_CONVERGENCE states |
| `agents/script-generator/server.py` | validate_script tools, mesh templates, recommend_technique MCP tool |
| `agents/asset-evaluator/server.py` | JSON serialization fix, Phase 5.1 pending |
| `agents/experiment-tracker/tracker.py` | Auto-load knowledge from JSON files at init |

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Core files exist | ✅ | 2/2 |
| SKILL.md complete | ✅ | All enhancements |
| Phase 1 modules | ✅ | 4/4 |
| Phase 2 validation | ✅ | Working |
| Scripts validated pre-execution | ✅ 100% | 100% |
| Technique selection informed | ✅ 100% | 100% |
| Knowledge base consulted | ✅ 100% | 100% |
| Session resume success | Unknown | >90% |
| Average iterations to pass | ~3-5 | <3 |

---

## Related Documents

- `docs/MULTI_AGENT_IMPROVEMENT_PLAN_V2.md` - Full details, code examples
- `docs/MULTI_AGENT_PIPELINE_ANALYSIS.md` - System documentation
- `.claude/skills/blender-orchestrator/SKILL.md` - Skill definition
- `docs/GEMINI_FEEDBACK_ANALYSIS_AND_PLAN_AMENDMENTS.md` - Gemini 3 Pro feedback
- `plans/feat-blender-librarian-FINAL.md` - **Phase 7 implementation plan**
- `docs/GPT52_BLENDER_LIBRARIAN_MERGED_DESIGN.md` - Full architectural design
- `docs/FEEDBACK_AND_CORRECTIONS_LIBRARIAN_AGENT.md` - Gemini corrections to original plan
