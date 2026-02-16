# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## !!!!! CRITICAL: READ FIRST - AI TRAINING DATA WARNING !!!!!

**YOUR TRAINING DATA IS OUTDATED. DO NOT TRUST YOUR MEMORY.**

Before making ANY changes, read: `agents/blender-vfx-orchestrator/VERSION_TRUTH.md`

### Models (2026-01)
| USE | DO NOT USE |
|-----|------------|
| `gpt-5.2`, `gpt-5-mini` | `gpt-4o`, `gpt-4-mini`, `gpt-4` |
| `o3`, `o4-mini` | `gpt-3.5-turbo` |

### Blender 5.0 - VERIFY EVERY ATTRIBUTE
| WRONG (hallucinated) | CORRECT |
|----------------------|---------|
| `resolution_divisions` | `resolution_max` |
| `use_adaptive_time_steps` | `use_adaptive_timesteps` |
| `use_dissolve` | `use_dissolve_smoke` |
| `absolute_density` | `density` + `use_absolute` |
| `timesteps_per_frame` | `timesteps_max` |
| `timesteps_maximum` | `timesteps_max` |

**Before using ANY bpy attribute:** Call `semantic_search_blender_docs()` to verify.

### Agents SDK v0.9.0
- Use `from agents import ...` (NOT `from openai.agents`)
- Use `await Runner.run(agent, prompt)` (NOT `agent.run_sync()`)
- Use `agent.as_tool()` (NOT `transfer_to_agent()` handoffs)

**Run enforcement:** `python version_enforcement.py --strict .`

---

## Project Overview

The user is named Ben, a novice programmer with C++, Java, and Python experience. He has high-functioning autism with a strong passion for AI/ML/LLMs, leveraging these tools to create experimental systems.

### Collaboration Preferences

- **Be corrective when wrong** - Correct misunderstandings immediately but kindly, explain the "why"
- **Validate effort** - Acknowledge reasonable approaches even if technically incorrect
- **Show what's salvageable** - Emphasize reusable work when fixing issues
- **Break down complex problems** - Manageable steps when Ben is uncertain
- **Test ideas immediately** - Working code examples over descriptions

### Feedback Philosophy: Brutal Honesty

**CRITICAL:** Brutal honesty is strongly preferred over sugar-coating.

Good: "ZERO LIGHTS ACTIVE - this is catastrophic, cannot assess visual quality"
Bad: "Lighting could use some refinement to improve visual quality"

Direct, specific language accelerates debugging and saves development time.

---

## Current Focus: Blender VFX Orchestrator

The primary focus of this project is **blender-vfx-orchestrator** - an autonomous multi-agent system built on the **OpenAI Agents SDK** that generates high-quality volumetric VFX assets through iterative improvement.

**Location:** `agents/blender-vfx-orchestrator/`

### Ground Truth for LLMs (Read First)

- **Roadmap:** `agents/blender-vfx-orchestrator/docs/MASTER_ROADMAP_2026-01-26.md`
- **Ops Manual:** `agents/blender-vfx-orchestrator/docs/AI_OPERATION_MANUAL.md`
- **Architecture Addendum:** `agents/blender-vfx-orchestrator/docs/MULTI_AGENT_ARCHITECTURE_ADDENDUM_2026-01-29.md`
- **SDK Enforcement:** `agents/blender-vfx-orchestrator/docs/SDK_ENFORCEMENT_PROTOCOL.md`
- **Version Truth:** `agents/blender-vfx-orchestrator/docs/VERSION_TRUTH.md`
- **Recent Fixes:** `agents/blender-vfx-orchestrator/docs/WORKLOG_2026-01-29_API_FIXER_RENDER_BAKE.md`

**Important:** Use Blender 5 MCP docs for all API validation. Do not rely on memory. Code can be wrong; validate against docs + runtime behavior.

### What It Does

- Autonomously generates Blender Python scripts for VFX effects (explosions, fire, smoke, nebulae, solar effects)
- Iterates on quality using ML-powered evaluation until thresholds are met
- Uses **code-based pipeline orchestration** with **3 Coordinator agents** for intelligent decisions
- Implements 5 self-learning strategies for continuous improvement
- Exports NanoVDB volumetric files for use in the DXR renderer

### Required Operational Flow (State Machine)

```
PLAN → GENERATE → VALIDATE → EXECUTE → EVALUATE → DECIDE
```

Failure routing:
- VALIDATE fail → back to GENERATE with fixes
- EXECUTE fail → DIAGNOSE → FIX → EXECUTE
- EVALUATE fail → IMPROVE → GENERATE

### Recent Stability Fixes (2026-01-29)

- Render loop limiter now catches common patterns and prevents full-frame renders.
- `use_plane_init = True` is auto-injected for **all** liquid flows (prevents empty bakes).
- Headless paths use env overrides (`BLENDER_OUTPUT_DIR`, `BLENDER_CACHE_DIR`).

### Headless Gotchas (Blender 5)

- `bpy.ops.fluid.bake_all()` requires an **active domain object** in **OBJECT** mode.
- `animation=True` does not emit still frames in headless mode.
- `cache_type = 'REPLAY'` does not produce full bake data; use `ALL`.
- `bpy.path.abspath('//')` is not reliable headless; prefer env-based paths.

---

## OpenAI Agents SDK Architecture

**SDK Version:** v0.9.0
**Documentation:** https://github.com/openai/openai-agents-python/tree/main/docs

### Agent Hierarchy

```
create_asset_pipeline() (Python-controlled)
│
├── COORDINATOR AGENTS (Decision Points)
│   ├── TechniqueSelector      - Phase 0.5: Selects initial technique
│   ├── ModificationStrategist - Phase 1.1: Decides modification strategy
│   └── QualityGateJudge       - Phase 5: Interprets quality results
│
├── SPECIALIZED AGENTS (Execution)
│   ├── ResearchAgent    - Researches best approach for effect type
│   ├── ScriptWriter     - Generates/modifies Blender Python scripts
│   ├── Executor         - Runs scripts in Blender, parses errors
│   ├── QualityAnalyst   - ML-powered quality evaluation (vision + metrics)
│   ├── LearningAgent    - Experiment tracking, knowledge base queries
│   └── DocsExpert       - Blender documentation search (semantic + keyword)
│
└── API Validator        - Validates Blender 5.0 API calls before execution
```

### SDK Patterns Used

**Agents as Tools** (Primary Pattern) - Coordinators call sub-agents as tools:
```python
coordinator = Agent(
    tools=[
        research_agent.as_tool(
            tool_name="research_approach",
            tool_description="Research best approach for effect type",
        ),
        # Sub-agents called as tools, control returns to coordinator
    ],
)
```

**RunHooks** - Lifecycle callbacks for enforcement:
```python
from agents import RunHooks

class EnforcementHooks(RunHooks):
    async def on_tool_start(self, context, agent, tool):
        # Loop detection, doc query requirements
        pass

result = await Runner.run(agent, prompt, hooks=EnforcementHooks())
```

**Handoffs** (Deprecated) - Transfer control between agents:
```python
# NOTE: Handoff pattern is deprecated - use agents-as-tools instead
# The create_asset() method using handoffs is deprecated
```

**Function Tools** - Expose Python functions to agents:
```python
from agents import function_tool

@function_tool
async def my_tool(param: str) -> str:
    """Tool description becomes the docstring."""
    return result
```

### Critical SDK Lesson: MCP vs In-Process Tools

**Tools must run in-process.** The SDK cannot call external MCP servers from within agent context.

**Wrong:**
```python
@function_tool
async def my_tool() -> str:
    server = await get_mcp_server("some-server")  # FAILS
    return await server.call_tool(...)
```

**Correct - Two-Layer Pattern:**
```python
# Layer 1: Internal implementation
async def _my_tool_impl(param: str) -> str:
    """Actual logic - callable by other functions."""
    return result

# Layer 2: Tool wrapper for agents
@function_tool
async def my_tool(param: str) -> str:
    return await _my_tool_impl(param)
```

---

## Self-Learning Strategies (All Implemented)

| Strategy | Purpose |
|----------|---------|
| **1. Vector Store** | Semantic search over Blender 5.0 documentation |
| **2. Knowledge Distillation** | Auto-extract patterns from successful experiments |
| **3. Proactive Mining** | Research alternatives BEFORE getting stuck |
| **4. Code Pattern Memory** | Store/retrieve working Python code snippets |
| **5. Escape Velocity** | Escalating exploration when iterations plateau |

### Escape Velocity Levels

| Level | Trigger | Action |
|-------|---------|--------|
| 0 | Normal | Standard modification |
| 1 | Plateau 2x | Query knowledge base |
| 2 | Same issue 2x | Generate NEW script with different technique |
| 3 | Same issue 3x | Semantic search for novel approaches |
| 4 | No progress 4x | Request human guidance |

**Critical:** At Level 2+, do NOT modify the current script. Switch techniques entirely.

---

## Key Files and Directories

### Orchestrator Core
| File | Purpose |
|------|---------|
| `orchestrator.py` | Main coordinator agent with pipeline logic |
| `session_manager.py` | Tracks experiment state without LLM intervention |
| `server.py` | MCP server wrapper (if running as MCP) |

### Specialized Agents (`specialized_agents/`)
| File | Purpose |
|------|---------|
| `script_writer.py` | Blender Python script generation |
| `executor.py` | Script execution in Blender |
| `quality_analyst.py` | ML quality evaluation (vision + metrics) |
| `learning_agent.py` | Experiment tracking, pattern extraction |
| `docs_expert.py` | Blender documentation search |
| `api_validator.py` | Blender 5.0 API validation (Phase 1.5) |

### Tools (`tools/`)
| File | Purpose |
|------|---------|
| `semantic_docs_tools.py` | Vector store search (Strategy 1) |
| `knowledge_distillation_tools.py` | Pattern extraction (Strategy 2) |
| `proactive_research_tools.py` | Early warning detection (Strategy 3) |
| `code_pattern_tools.py` | Code snippet storage (Strategy 4) |
| `dynamic_instructions.py` | Runtime instruction injection |
| `asset_evaluator_tools.py` | ML quality metrics (LPIPS, CLIP, TOPIQ) |
| `script_generator_tools.py` | Script generation/modification |
| `blender_executor_tools.py` | Blender process management |
| `experiment_tracker_tools.py` | Knowledge base operations |

### Models (`models/`)
| Model | Purpose |
|-------|---------|
| `SharedContext` | Agent coordination context |
| `SessionState` | Persistent session state |
| `AssetRequest` | User request parameters |
| `QualityMetrics` | Evaluation results |

---

## Build and Run Commands

### Setup
```bash
cd agents/blender-vfx-orchestrator
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### Environment Variables
```bash
export OPENAI_API_KEY="sk-..."
export BLENDER_DOCS_VECTOR_STORE_ID="vs_..."  # For semantic search
export AGENTS_DEBUG=1  # Enable verbose SDK logging
```

### Running Tests
```bash
# E2E orchestrator test
python test_e2e_orchestrator.py

# Quick E2E test
python test_quick_e2e.py

# Specific component tests
python test_docs_expert.py
python test_self_learning_tools.py
python test_orchestrator_tools.py
```

### Running the Orchestrator

**Programmatic:**
```python
from orchestrator import create_vfx_asset

result = await create_vfx_asset(
    asset_name="explosion_001",
    description="Large fiery explosion with rising smoke",
    effect_type="explosion",
    quality_threshold=60.0,
    max_iterations=5
)
```

**Resume Session:**
```python
from orchestrator import resume_vfx_session

result = await resume_vfx_session("session_20260118_explosion_001")
```

---

## Quality Gates

**PASS** requires ALL of:
- `overall_score >= 60`
- No critical issues (ZERO_LIGHTS, BLACK_SCREEN, CLIPPING)
- If reference provided: acceptable similarity

**Critical Issues (Auto-Fail):**
- `ZERO_LIGHTS_ACTIVE` - No lighting in scene
- `BLACK_SCREEN` - Completely dark render
- `WHITE_SCREEN` - Completely overexposed
- `CLIPPING_ARTIFACTS` - Volume clipping at boundaries

---

## Common Pitfalls

### 1. Modifying at High Escape Level

**Wrong:**
```python
if escape_level >= 2:
    modify_script(script_path, {"flame_smoke": 3.5})  # BAD
```

**Correct:**
```python
if escape_level >= 2:
    generate_script(effect_type, technique=untried_techniques[0])
    stuck_state.reset_for_new_technique()
```

### 2. Forgetting Pattern Feedback

**Always report pattern outcomes:**
```python
apply_pattern_to_script(script, pattern_id)
result = evaluate_quality()
report_pattern_outcome(pattern_id, success=result.improvement > 0)  # REQUIRED
```

### 3. Skipping Pre-Iteration Research

**Always check before iteration 2+:**
```python
if iteration > 0:
    research = pre_iteration_research(...)
    if research["warning_level"] != "none":
        handle_escape_action(research["escape_action"])
```

---

## Key Documentation

| Document | Purpose |
|----------|---------|
| `docs/ARCHITECTURE_OPTIMIZATION_PLAN_2026-01-22.md` | **Phase 1-7 implementation status** |
| `docs/AGENT_SPECIFICATION.md` | Complete agent specs, tools, schemas |
| `docs/AI_OPERATION_MANUAL.md` | Operational guide for AI agents |
| `docs/AGENTS_SDK_INTEGRATION.md` | SDK patterns and lessons learned |
| `docs/SELF_LEARNING_ARCHITECTURE.md` | Learning system implementation |
| `docs/BLEND_PATCHING_PROPOSAL.md` | Future: .blend file patching proposal |

---

## Budget Management

**Monthly limit:** $20

| Category | Allocation |
|----------|------------|
| Vision/Evaluation | $10 |
| Documentation Search | $8 |
| Emergency Buffer | $2 |

Check budget before starting:
```python
orchestrator = await get_orchestrator()
budget = orchestrator.get_budget_status()
if not budget["can_afford_evaluation"]:
    print(f"Budget exhausted: ${budget['total_spent']:.2f}")
```

---

## SDK Documentation Reference

**Always consult before making changes:**
- https://github.com/openai/openai-agents-python/tree/main/docs
- [Multi-Agent Patterns](https://github.com/openai/openai-agents-python/blob/main/docs/multi_agent.md)
- [Tools Reference](https://github.com/openai/openai-agents-python/blob/main/docs/tools.md)
- [RunHooks](https://github.com/openai/openai-agents-python/blob/main/docs/run_hooks.md)
- [Guardrails](https://github.com/openai/openai-agents-python/blob/main/docs/guardrails.md) - **Next Phase**
- [Tracing](https://github.com/openai/openai-agents-python/blob/main/docs/tracing.md)

**Use context7** when you need code generation, setup steps, or library/API documentation.

---

## PlasmaDXR (DXR Renderer)

The DXR volumetric particle renderer that consumes VFX assets is in the main `src/` directory. This is secondary to the orchestrator focus but still functional.

### Quick Build Reference
```bash
# Build
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64
MSBuild.exe build/PlasmaDXR.sln /p:Configuration=Debug /p:Platform=x64

# Run
./build/bin/Debug/PlasmaDXR.exe --config=configs/user/default.json
```

### Key DXR Concepts
- **DXR 1.1 RayQuery API** - Inline ray tracing from any shader stage
- **3D Gaussian Splatting** - Volumetric ellipsoids (not 2D splats)
- **NVIDIA RTXDI** - Weighted reservoir sampling for lighting

**CRITICAL:** Stale .dxil files are the #1 cause of mysterious visual bugs. Rebuild or manually recompile if shader changes produce unexpected results.
