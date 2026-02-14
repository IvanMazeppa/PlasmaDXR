# Blender VFX Orchestrator - Project State & Problem Analysis

**Date:** 2026-02-06
**Status:** Historical analysis snapshot; some issues listed here are now mitigated.
**Current runtime truth:** `docs/RUNTIME_TRUTH_AND_DOC_GROUNDING_2026-02-13.md`
**Purpose:** Comprehensive assessment for deep analysis session
**Author:** Claude Opus 4.6 analysis of codebase, traces, and test results

---

## 1. What This Project Is

An autonomous multi-agent system built on the OpenAI Agents SDK (snapshot at analysis time referenced v0.7.0) that generates high-quality volumetric VFX assets (fire, smoke, explosions, liquid effects) by:

1. Taking a natural language description of a desired VFX effect
2. Researching the best Blender API approach via vector store search
3. Generating a Blender Python script
4. Executing it headlessly in Blender 5.0
5. Evaluating the output with ML-powered quality metrics
6. Iterating until a quality threshold is met

**Location:** `agents/blender-vfx-orchestrator/`
**Orchestrator:** `orchestrator.py` (4,661 lines)
**Agent count:** 10+ specialized agents + 3 coordinator agents

---

## 2. What It's Supposed to Do vs. What Actually Happens

### Target Behavior
```
User: "Create a fuel ignition with flame front propagation"
→ Agent researches Mantaflow fire, generates 500+ line script
→ Script bakes simulation, renders frames
→ Quality analyst evaluates, suggests improvements
→ Agent iterates 2-3 times until score >= 60
→ Output: High-quality volumetric fire renders + NanoVDB files
```

### Actual Behavior (as of 2026-02-04)
```
User: "Create a fuel ignition with flame front propagation"
→ Agent researches Mantaflow fire ✓
→ Enters Spec-First pipeline validation loop
→ Code Writer writes script → validate_script → rewrite → validate → rewrite...
→ Pipeline exhausts turns in validation loop
→ Never reaches execution
→ OR: Falls back to simplified script that doesn't match the prompt
→ Simplified script produces generic smoke, not the requested fire effect
```

---

## 3. Critical Problems (Ranked by Impact)

### Problem 1: Spec-First Pipeline Catch-22 (BLOCKING)

**Files:** `guardrails/api_spec_guardrails.py` (37KB), `specialized_agents/api_spec_agent.py`, `specialized_agents/code_writer_agent.py`

The Spec-First pipeline was designed to prevent Blender API hallucinations by requiring every attribute to be verified against documentation before use. In practice, it creates a deadlock:

- **API Spec Agent** only verifies `FluidDomainSettings` / `FluidFlowSettings` attributes
- **Complexity guardrail** (`validate_code_against_spec`) requires the script to also include materials, render setup, camera setup, and save operations
- These are NOT FluidDomainSettings attributes, so they can never be "verified" by the API Spec
- Result: Script repeatedly fails validation, gets rewritten simpler each time, eventually produces trivial output

**Evidence from trace** (`fuel_ignition_20260203_234101.jsonl`):
- 192 trace entries, pipeline ends at Code Writer phase
- Multiple `write_script → validate_script → write_script` cycles
- Final entry: `validate_code_against_spec` guardrail at trace end
- Never reaches Executor

### Problem 2: Generated Scripts Crash Blender (Cycles Volume Shader Bug)

**Root cause identified:** `ShaderNodeMixRGB` nodes connected to `ShaderNodeVolumePrincipled` inputs cause Blender 5.0 Cycles to crash during `ccl::ShaderGraph::optimize_volume_output → ccl::MixNode::is_linear_operation` (EXCEPTION_ACCESS_VIOLATION).

**Impact:** Even when scripts bypass the Spec-First guardrail, complex volume materials with color mixing crash Cycles during render. The agent has no way to detect or prevent this pattern.

**The fix is simple:** Connect `ShaderNodeBlackbody` directly to `Principled Volume` inputs without going through `MixRGB`. But the agent doesn't know this pattern exists.

**Files affected:** Any generated script using complex volume shader chains.

### Problem 3: Oversimplification During Iteration

When the agent encounters errors (API, validation, or runtime), its recovery strategy is to simplify the script. This creates a race to the bottom:

- Iteration 1: Complex, detailed script (matches prompt) → fails validation or crashes
- Iteration 2: Simplified script (removes volume materials, camera, lighting) → passes validation
- Iteration 3: Trivial script (basic smoke puff) → technically "works" but doesn't match the prompt at all

The result you observed - loading the .blend and finding "no pyro simulation" - is the end state of this simplification cascade.

### Problem 4: Enforcement Hooks Fighting Agent Creativity

**File:** `hooks/enforcement_hooks.py` (33KB)

The enforcement system has accumulated layers of protection:
- Loop detection (max same-tool calls)
- Doc query requirements (must search docs before writing)
- Turn budgets (hard limits on reasoning)
- Bundle-first enforcement (must call bundle search before targeted search)

These were individually reasonable but in aggregate they create a straitjacket. The agent spends most of its turn budget complying with enforcement rather than solving the actual problem.

### Problem 5: Blender 5.0 API Hallucination (Partially Mitigated)

LLM training data contains Blender 2.8-4.x patterns. Despite the vector store search, the agent still occasionally uses:
- `resolution_divisions` instead of `resolution_max`
- `BLENDER_EEVEE_NEXT` instead of `BLENDER_EEVEE`
- `Action.fcurves` (changed in Blender 5.0)
- `use_dissolve` instead of `use_dissolve_smoke`

The doc search tool now works reliably, which actually makes many of the guardrails redundant. The agent CAN find the right attributes when it searches - the problem is the enforcement overhead around making it search.

### Problem 6: No Crash Pattern Detection

The orchestrator has no mechanism to:
1. Detect known crash-causing patterns in generated scripts (like MixRGB→Volume)
2. Learn from previous crashes and avoid the same patterns
3. Test shader compilation before full bake+render (which takes 20+ minutes)

### Problem 7: Excessive Architectural Complexity

The system has accumulated significant accidental complexity:
- 4,661-line orchestrator
- 37KB API spec guardrails
- 33KB enforcement hooks
- 37KB API validator
- 10+ specialized agents, 3 coordinators
- Multiple overlapping validation layers
- 35 open roadmap items across 4 priority levels

Much of this complexity was built to compensate for LLM unreliability that may no longer be necessary with better doc search.

---

## 4. Architecture Overview

```
orchestrator.py (4,661 lines) - Main pipeline control
├── hooks/
│   ├── enforcement_hooks.py (33KB) - Loop detection, doc query enforcement
│   └── diagnostic_hooks.py (19KB) - Tracing and diagnostics
├── guardrails/
│   ├── api_spec_guardrails.py (37KB) - Spec-First validation (PROBLEMATIC)
│   ├── artifact_gates.py (18KB) - Bake/render output validation
│   ├── coordinator_guardrails.py (20KB) - Decision validation
│   ├── script_guardrails.py (16KB) - Script structure validation
│   ├── quality_guardrails.py (13KB) - Quality metric validation
│   └── research_guardrails.py (2KB) - Research output validation
├── specialized_agents/
│   ├── api_spec_agent.py (10KB) - API attribute verification
│   ├── api_validator.py (37KB) - Full API validation
│   ├── code_writer_agent.py (17KB) - Blender script generation
│   ├── script_writer.py (8KB) - Script generation (original)
│   ├── docs_expert.py (9KB) - Documentation search
│   ├── executor.py (5KB) - Blender execution
│   ├── quality_analyst.py (6KB) - Quality evaluation
│   └── learning_agent.py (8KB) - Pattern learning
├── tools/ (12+ tool modules)
├── models/ (Pydantic schemas)
└── docs/ (10+ documentation files)
```

**Total codebase:** ~250KB of Python across 30+ files

---

## 5. What Works

1. **Vector store doc search** - Reliably finds correct Blender 5.0 attributes
2. **Blender headless execution** - Script execution and artifact collection work
3. **ML quality evaluation** - LPIPS, CLIP, TOPIQ metrics functional
4. **Session persistence** - SQLiteSession tracks state across iterations
5. **Tracing** - JSONL traces capture full agent interaction history
6. **Artifact gates** - Cache size and render count validation (recently added)
7. **API fixer** - Auto-corrects known bad patterns (node links, attributes)

---

## 6. Proposed Simplification Strategy

Based on the analysis in the previous session, the recommended approach:

### Remove
- **API Spec Agent** (`api_spec_agent.py`) - Catch-22 creator
- **API Spec Guardrails** (`api_spec_guardrails.py`, 37KB) - Source of validation deadlock
- **Code Writer Agent complexity guardrail** - Fights against script completeness
- **Bundle-first enforcement** - Unnecessary with reliable doc search

### Keep
- **Doc query requirement** - Agent must search docs before writing (simple check)
- **Deprecated attribute blacklist** - Quick regex check for known-bad patterns
- **Artifact gates** - Validate bake/render outputs exist and are non-trivial
- **Loop detection** - Prevent infinite research loops (but raise limits)
- **API fixer** - Post-generation auto-correction of common mistakes

### Add
- **Crash pattern detector** - Check for MixRGB→Volume and similar known crashes
- **Shader compilation pre-check** - Validate materials before full bake
- **Prompt fidelity check** - Verify generated script addresses key requirements from the description
- **Learning from crashes** - Record and avoid patterns that caused previous crashes

---

## 7. Key Files for Analysis

| File | Lines | Purpose | Health |
|------|-------|---------|--------|
| `orchestrator.py` | 4,661 | Main pipeline | Overly complex, needs simplification |
| `guardrails/api_spec_guardrails.py` | ~900 | Spec-First validation | **REMOVE** - creates deadlock |
| `hooks/enforcement_hooks.py` | ~800 | SDK enforcement | Needs loosening |
| `specialized_agents/code_writer_agent.py` | ~450 | Script generation | Instructions too restrictive |
| `specialized_agents/api_validator.py` | ~900 | API validation | Useful but heavy |
| `guardrails/artifact_gates.py` | ~450 | Output validation | Good, recently fixed |
| `specialized_agents/script_writer.py` | ~200 | Original script writer | Simpler, more reliable |
| `tools/semantic_docs_tools.py` | - | Vector store search | Works well |

---

## 8. Recent Test Results

| Test | Date | Outcome | Root Cause |
|------|------|---------|------------|
| `fuel_ignition` (run 1) | 2026-02-03 23:22 | Pipeline stalled in Code Writer | Spec-First catch-22 |
| `fuel_ignition` (run 2) | 2026-02-03 23:41 | Script generated, crashed in Cycles | MixRGB→Volume crash |
| `oil_pour_resume` | 2026-02-03 21:07 | Partial success | Liquid domain issues |
| `oil_pour_3iter` | 2026-02-02 02:15 | 3 iterations, mediocre quality | Oversimplification |
| `p0_verification` | 2026-02-02 00:58 | Mixed results | Multiple issues |

---

## 9. Questions for Deep Analysis

1. **Can the Spec-First pipeline be removed entirely?** What would break? What's the minimal replacement?

2. **How should crash patterns be detected?** Static analysis of shader node graphs? Runtime AST inspection? Pattern matching against known-bad configurations?

3. **What's the minimum viable enforcement?** If doc search works reliably, how much guardrail infrastructure is actually needed?

4. **How should the agent learn from failures?** The learning system exists but doesn't currently prevent known crash patterns. How should crash knowledge propagate?

5. **Should the orchestrator be split?** 4,661 lines suggests it's doing too many things. What's the right decomposition?

6. **How do we preserve script complexity?** The simplification cascade is the core quality problem. How do we fix failures without dumbing down the script?

7. **What would a v2 architecture look like?** If we could start fresh with lessons learned, what would be simpler?

---

## 10. Environment

- **Blender:** 5.0 (Linux headless via WSL2)
- **Python:** 3.12+
- **OpenAI SDK:** Snapshot used v0.7.0; current runtime pin is `openai-agents==0.8.3` (see runtime truth doc).
- **Models:** gpt-5.2, gpt-5-mini, o4-mini
- **Budget:** $20/month
- **Host:** WSL2 Ubuntu 24.04 on Windows
