# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

### What It Does

- Autonomously generates Blender Python scripts for VFX effects (explosions, fire, smoke, nebulae, solar effects)
- Iterates on quality using ML-powered evaluation until thresholds are met
- Coordinates 5 specialized agents via SDK handoffs
- Implements 5 self-learning strategies for continuous improvement
- Exports NanoVDB volumetric files for use in the DXR renderer

---

## OpenAI Agents SDK Architecture

**SDK Version:** v0.6.8+
**Documentation:** https://github.com/openai/openai-agents-python/tree/main/docs

### Agent Hierarchy

```
BlenderVFXOrchestrator (coordinator)
├── ScriptWriter     - Generates/modifies Blender Python scripts
├── Executor         - Runs scripts in Blender, parses errors
├── QualityAnalyst   - ML-powered quality evaluation (vision + metrics)
├── LearningAgent    - Experiment tracking, knowledge base queries
└── DocsExpert       - Blender documentation search (semantic + keyword)
```

### SDK Patterns Used

**Handoffs** - Transfer control between agents:
```python
from agents import Agent, handoff

orchestrator = Agent(
    name="Orchestrator",
    handoffs=[script_writer, quality_analyst, learning_agent],
)
```

**Agents as Tools** - Use agents as callable tools:
```python
orchestrator = Agent(
    tools=[quality_analyst.as_tool(tool_name="evaluate_quality", ...)],
)
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
| `docs/AGENT_SPECIFICATION.md` | Complete agent specs, tools, schemas |
| `docs/AI_OPERATION_MANUAL.md` | Operational guide for AI agents |
| `docs/AGENTS_SDK_INTEGRATION.md` | SDK patterns and lessons learned |
| `docs/SELF_LEARNING_ARCHITECTURE.md` | Learning system implementation |
| `docs/WORKFLOW_ANALYSIS_2026-01-21.md` | Latest workflow issues and fixes |

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
- [Handoffs](https://github.com/openai/openai-agents-python/blob/main/docs/handoffs.md)

**Use context7** when you need code generation, setup steps, or library/API documentation.

---

## PlasmaDX-Clean (DXR Renderer)

The DXR volumetric particle renderer that consumes VFX assets is in the main `src/` directory. This is secondary to the orchestrator focus but still functional.

### Quick Build Reference
```bash
# Build
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64
MSBuild.exe build/PlasmaDX-Clean.sln /p:Configuration=Debug /p:Platform=x64

# Run
./build/bin/Debug/PlasmaDX-Clean.exe --config=configs/user/default.json
```

### Key DXR Concepts
- **DXR 1.1 RayQuery API** - Inline ray tracing from any shader stage
- **3D Gaussian Splatting** - Volumetric ellipsoids (not 2D splats)
- **NVIDIA RTXDI** - Weighted reservoir sampling for lighting

**CRITICAL:** Stale .dxil files are the #1 cause of mysterious visual bugs. Rebuild or manually recompile if shader changes produce unexpected results.
