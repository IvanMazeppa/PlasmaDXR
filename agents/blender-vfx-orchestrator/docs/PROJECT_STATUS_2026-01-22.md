# Blender VFX Orchestrator: Project Status & Vision

**Date:** 2026-01-22
**Author:** Ben (with Claude)
**Status:** Active Development - Critical Issues Identified

---

## Executive Summary

The Blender VFX Orchestrator is an autonomous multi-agent system built on the **OpenAI Agents SDK** that generates high-quality volumetric VFX assets through iterative improvement. The system coordinates 5 specialized agents to research, write, execute, evaluate, and learn from each iteration.

**The Vision:** An intelligent, self-improving agent ecosystem that can operate Blender skillfully, requiring only a natural language prompt to create complex 3D VFX from scratch.

---

## Roadmap (Updated 2026-01-25)

### Phase 4 Decision Gate (Do NOT start until these pass)
1) **Modification contract enforced** (flat Config keys only)  
2) **Doc grounding reliable** (`doc_refs` resolve to real `DocPath`)  
3) **Trace correlation working** (`group_id=session_id` for SDK trace + local JSONL)

### Next Best Path (Autonomy‑First)
1) **Planner–Executor–Verifier core**  
   - Use coordinators as Planner, Quality Analyst as Verifier.
2) **Beam search as exploration layer (N=2–3, K=1)**  
   - Low‑cost branching; not the end state.
3) **Bandit technique selection**  
   - Prevent technique lock‑in.
4) **Small population (3–5 scripts)**  
   - Add only after beam search shows consistent gains.
5) **Phase 4 (session compaction + cross‑session bootstrap)**  
   - Start only after the gate above passes.

---

## Current Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    BlenderVFXOrchestrator                       │
│                     (Coordinator Agent)                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   Research   │  │Script Writer │  │      Executor        │  │
│  │    Agent     │──│    Agent     │──│       Agent          │  │
│  │              │  │              │  │                      │  │
│  │ • Semantic   │  │ • Code Gen   │  │ • Blender Process    │  │
│  │   Search     │  │ • API Lookup │  │ • Error Parsing      │  │
│  │ • Patterns   │  │ • Validation │  │ • VDB Export         │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐                            │
│  │   Quality    │  │   Learning   │                            │
│  │   Analyst    │  │    Agent     │                            │
│  │              │  │              │                            │
│  │ • Vision     │  │ • Knowledge  │                            │
│  │ • LPIPS/CLIP │  │   Base       │                            │
│  │ • TOPIQ      │  │ • Patterns   │                            │
│  └──────────────┘  └──────────────┘                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Problem 1: Blender 5.0 Python API Errors

### Symptoms
- Scripts fail with `AttributeError` for properties that don't exist in Blender 5.0
- Socket names changed between versions (e.g., `"Smoke"` → `"Grid"`)
- Properties removed or renamed in FluidDomainSettings

### Root Cause
The LLM (gpt-5.2) generates code from its training data, which includes older Blender versions (2.8x-4.x). When it writes Mantaflow scripts, it uses outdated API patterns instead of strictly following Blender 5.0 documentation.

### Example Failures
```python
# FAILS: Old API pattern from training data
domain.modifier.effector_weights.wind = 1.0  # Removed in 5.0

# FAILS: Old socket names
principled_volume.inputs["Smoke"].default_value  # Now called "Grid"

# CORRECT: Blender 5.0 API
domain.effector_weights.wind = 1.0
principled_volume.inputs["Grid"].default_value
```

### Impact
- Scripts fail on first execution
- Iteration loop never completes
- System stuck in error → modify → error cycle

### Required Solution
**Strict documentation-first code generation:**
1. ALWAYS query Blender 5.0 vector store before writing code
2. Never use API patterns from memory/training data
3. Validate every `bpy.types` and `bpy.ops` call against docs

---

## Problem 2: Agent Gets Stuck in Research Loops

### Symptoms
- Agent calls `search_blender_api_by_intent` 15+ times
- Never advances to Executor phase
- Hits `MaxTurnsExceeded` exception (25 turns)

### Root Cause
The Research Agent and Script Writer have documentation search tools but no explicit stopping condition. They keep searching for "more context" instead of producing output.

### Trace Evidence
```
Turn 1:  search_blender_api_by_intent("mantaflow domain setup")
Turn 2:  search_blender_api_by_intent("smoke simulation parameters")
Turn 3:  semantic_search_blender_docs("fire effect Blender 5")
Turn 4:  search_blender_api_by_intent("FluidDomainSettings properties")
Turn 5:  search_blender_api_by_intent("FluidFlowSettings")
...
Turn 25: MaxTurnsExceeded
```

### SDK Solution Available
The OpenAI Agents SDK provides `RunHooks` for lifecycle callbacks:

```python
from agents import RunHooks, Tool

class LoopDetectionHooks(RunHooks):
    def __init__(self, max_same_tool: int = 3):
        self.tool_counts = {}
        self.max_same_tool = max_same_tool

    async def on_tool_start(self, context, agent, tool: Tool):
        tool_name = tool.name
        self.tool_counts[tool_name] = self.tool_counts.get(tool_name, 0) + 1

        if self.tool_counts[tool_name] > self.max_same_tool:
            # Force the agent to stop researching and produce output
            raise LoopDetectedError(f"Tool {tool_name} called {self.tool_counts[tool_name]} times")
```

---

## Problem 3: SDK Documentation Not Being Used as Primary Source

### The Core Issue
The OpenAI Agents SDK documentation contains critical patterns, examples, and best practices that should guide ALL development decisions. However, these docs are often overlooked because:

1. **Context loss** - SDK docs aren't persisted across conversation boundaries
2. **Training data bias** - LLM may prefer patterns from training data over SDK docs
3. **No enforcement mechanism** - Nothing mandates SDK doc consultation before changes

### Critical SDK Features Being Underutilized

| Feature | What It Does | Why We Need It |
|---------|--------------|----------------|
| **Tracing** | Built-in execution monitoring | Debug agent behavior, view in dashboard |
| **RunHooks** | Lifecycle callbacks | Detect loops, enforce behavior |
| **Guardrails** | Input/output validation | Stop invalid requests early |
| **Handoffs** | Agent-to-agent transfer | Proper multi-agent coordination |
| **Agents as Tools** | Call agent as function | Utility agents that return control |
| **Structured Output** | Type-safe responses | Reliable data passing between agents |

### SDK Tracing (Available Now!)

```python
from agents import Agent, Runner, trace

# Wrap execution for dashboard visibility
with trace("VFX Generation", group_id=session_id):
    result = await Runner.run(orchestrator, prompt)

# View at: https://platform.openai.com/traces
```

### Required Solution
**Mandatory SDK consultation protocol:**
1. Before ANY orchestrator change, query SDK docs via context7
2. Implement SDK patterns exactly as documented
3. Use tracing for ALL runs to enable debugging
4. Add RunHooks for loop detection and behavior enforcement

---

## Problem 4: Workflow Logic Fragility

### Current State
Two orchestration methods exist:

| Method | Control | Status |
|--------|---------|--------|
| `create_asset()` | LLM-driven with handoffs | Unpredictable, loops |
| `create_asset_pipeline()` | Python-controlled sequence | More stable, but less intelligent |

### The Dilemma
- **Handoff-based orchestration** is what we WANT (intelligent, adaptive)
- **Pipeline orchestration** is what WORKS (deterministic, predictable)

The goal is to make handoff-based orchestration work reliably by:
1. Using SDK guardrails to prevent invalid behavior
2. Using RunHooks to detect and stop loops
3. Explicit instructions that leave no room for interpretation

---

## The Vision: Autonomous 3D VFX Generation

### What We're Building

An AI system that, given only a text prompt, can:

1. **Research** - Query documentation, find patterns, understand best practices
2. **Plan** - Decide on technique, parameters, approach
3. **Code** - Generate valid Blender Python scripts
4. **Execute** - Run scripts in Blender, handle errors
5. **Evaluate** - Use vision + ML metrics to assess quality
6. **Learn** - Store successful patterns, avoid failures
7. **Iterate** - Improve until quality threshold met

### Example Interaction

```
User: "Create a realistic sun with corona, prominences, and surface granulation"

System:
├── Research: Queries Blender 5.0 docs for volumetric emission, noise patterns
├── Plan: Selects dual-domain approach (core + corona as separate volumes)
├── Code: Generates 200-line Blender Python script
├── Execute: Runs in Blender, exports VDB + PNG preview
├── Evaluate:
│   ├── Vision: "Corona visible but lacks detail"
│   ├── LPIPS: 0.42 (needs improvement)
│   └── Score: 58/100 (below threshold)
├── Learn: Records "corona needs higher noise scale"
├── Iterate: Modifies script, re-runs
│   ├── Vision: "Good corona definition, surface detail present"
│   ├── LPIPS: 0.71 (acceptable)
│   └── Score: 72/100 (PASS)
└── Output: sun_001.vdb, sun_001_preview.png
```

### Why This Matters

This isn't just about generating VFX. It's a proof-of-concept for:

- **Autonomous creative tools** that understand domain knowledge
- **Self-improving systems** that learn from their mistakes
- **Multi-agent orchestration** at the bleeding edge of AI capabilities

### Is It Ambitious? Yes.

But the foundation is already working:
- Agents communicate via SDK handoffs
- Quality evaluation provides real feedback
- Knowledge base stores learnings
- Vector store enables semantic documentation search

The remaining work is **reliability engineering** - making the system robust enough to run autonomously without human intervention at each step.

---

## Immediate Action Items

### 1. Enforce SDK Documentation as Primary Source
- [ ] Create mandatory SDK query step before ANY code changes
- [ ] Document all SDK patterns currently in use
- [ ] Add SDK doc links to CLAUDE.md for persistence

### 2. Enable Tracing for All Runs
```python
# Add to all Runner.run() calls
with trace(f"VFX: {asset_name}", group_id=session_id):
    result = await Runner.run(agent, prompt, max_turns=N)
```

### 3. Implement Loop Detection Hooks
```python
hooks = LoopDetectionHooks(max_same_tool=3)
result = await Runner.run(agent, prompt, run_hooks=hooks)
```

### 4. Fix Script Writer API Compliance
- [ ] Add strict "NEVER use API from memory" instruction
- [ ] Require doc search BEFORE any bpy.types/bpy.ops usage
- [ ] Validate generated code against Blender 5.0 API

### 5. Add Integration Tests
- [ ] Test: Research → Script → Execute completes
- [ ] Test: API error triggers correction
- [ ] Test: Quality loop iterates correctly

---

## Conclusion

The Blender VFX Orchestrator is at an inflection point. The core architecture is sound, the agents can communicate, and the learning systems are in place.

**The path forward is clear:**
1. Use the SDK as it was designed to be used
2. Enable tracing for visibility
3. Add guardrails and hooks for reliability
4. Trust the docs, not the training data

This system CAN work. The breakthroughs we've seen prove it. Now we need to make it **reliable**.

---

*"The best way to predict the future is to invent it." - Alan Kay*

*Let's invent autonomous 3D creation.*
