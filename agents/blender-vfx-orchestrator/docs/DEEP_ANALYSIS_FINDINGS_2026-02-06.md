# Deep Analysis Findings - Synthesized Report

**Date:** 2026-02-06
**Method:** 4 parallel analysis agents (pipeline flow, guardrail audit, agent instructions, trace forensics)
**Scope:** Full codebase + 5 trace files + all guardrails + all specialized agents

---

## Verdict: The System Is Fighting Itself

The orchestrator has accumulated layers of protection that now actively prevent it from doing its job. The evidence is unambiguous:

- **0 out of 5 recent traces achieved a passing quality score**
- **2 of 5 traces never reached Blender execution** (stuck in validation loops)
- **1 trace retried the same failing script 4 times** without modification
- **1 trace ran for 60 minutes** producing renders scoring 12-20 (threshold: 60)
- The API Spec Agent consumes **10-60% of total pipeline runtime** producing specs that actively block script generation

---

## The 3 Root Causes (Everything Else Is Downstream)

### Root Cause 1: validate_code_against_spec Creates an Impossible Constraint

**Location:** `guardrails/api_spec_guardrails.py` lines 424-844

Three validation layers that contradict each other:
- **Layer 1 (Spec Compliance):** Only allows `dset.*`/`fset.*` attributes from the APISpec
- **Layer 2 (Deprecated Blacklist):** Rejects known bad Blender 4.x patterns
- **Layer 3 (Complexity):** Requires 200+ lines with lighting, camera, materials, render, .blend save

The APISpec only covers FluidDomainSettings/FluidFlowSettings. Materials, camera, lighting, render settings are NOT fluid attributes and cannot be in the spec. The agent is told "only use what's in the spec" AND "you must include things that can't be in the spec."

**Trace evidence:** `validate_code_against_spec` guardrail triggered 3 times in fuel_ignition_234101, causing 4 Code Writer respawns consuming 741 seconds (56% of runtime). The pipeline never reached quality evaluation.

**Fix:** Remove Layers 1 and 3. Keep Layer 2 (deprecated blacklist) - it's the only one catching real bugs.

### Root Cause 2: No Mechanism to Fix Scripts Without Simplifying Them

When the SDK retries an agent after a guardrail rejection, the agent sees "your output was rejected because X" and overcorrects by removing code. There's no way to say "add this missing thing" or "change this specific line." Each retry produces a simpler script until creative intent is lost.

**Trace evidence:** oil_pour_3iter went through 3 iterations, quality scores stuck at 12-20. The overexposure problem was never fixed despite clear quality feedback saying "severely overexposed, blown-out white."

**Fix:** Replace guardrail-on-retry with a post-generation correction pass. Instead of rejecting and retrying, auto-fix known issues (like the API fixer already does for attributes).

### Root Cause 3: Contradictory Instructions Across Agents

10 specific contradictions identified across agent instructions, guardrails, and enforcement hooks:

| # | Contradiction | Files |
|---|--------------|-------|
| 1 | API Spec covers only fluid attrs; Code Writer must produce full scene | api_spec_agent.py vs code_writer_agent.py |
| 2 | Fire attrs (flame_smoke, fuel_amount) not whitelisted; need exact doc_ref match | api_spec_guardrails.py KNOWN_GOOD_ATTRIBUTES |
| 3 | dynamic_instructions says `timesteps_per_frame`; api_validator says `timesteps_maximum`; correct is `timesteps_max` | 3 files disagree |
| 4 | Learning Agent hardcoded param names vs script's actual identifiers | dynamic_instructions.py |
| 5 | Docs Expert vorticity range 0-1; Script Writer range 0-4 | docs_expert.py vs script_writer.py |
| 6 | Fire validation warnings on liquid simulations | All traces |
| 7 | validate_script returns "valid:true" for scripts with IndentationError | p0_verification trace |
| 8 | Deprecated check substring-matches entire script (can flag comments/strings) | api_spec_guardrails.py |
| 9 | Script Writer instructions ~15,000 tokens, consuming massive context | dynamic_instructions.py |
| 10 | Session compaction erases guardrail rejection context, causing repeated mistakes | orchestrator.py line 4301 |

---

## What Should Be Removed

### Dead Code (~400 lines)
- Deprecated handoff pipeline in `initialize()` (lines 1957-2097): 5 unnecessary agents + handoff wiring
- `create_asset()` method (lines 2282-2354)
- `_build_generation_prompt()` and `_build_resumption_prompt()` (lines 4367-4438)
- `_parse_result()` (lines 4441-4472)
- `ORCHESTRATOR_INSTRUCTIONS` (lines 412-499)
- `create_coordinator_agent()` (lines 644-697)
- `create_agent_tool_wrappers()` (lines 569-618)

### Guardrails to Remove
| Guardrail | File | Reason |
|-----------|------|--------|
| `validate_code_against_spec` Layer 1 (spec compliance) | api_spec_guardrails.py | Creates deadlock |
| `validate_code_against_spec` Layer 3 (complexity) | api_spec_guardrails.py | Redundant with artifact gates |
| `validate_api_spec` | api_spec_guardrails.py | Over-strict doc_ref URL matching |
| `validate_effect_type` | script_guardrails.py | Redundant with AssetRequest |
| `safe_ops` allowlist | api_spec_guardrails.py | Invert to blocklist |

### Agent to Remove
| Agent | File | Reason |
|-------|------|--------|
| API Spec Agent | specialized_agents/api_spec_agent.py | Source of catch-22; consumes 10-60% of runtime |

### Hooks to Simplify
| Hook | Change |
|------|--------|
| Bundle-first enforcement | Remove (doc search works reliably) |
| Code Writer turn budget | Raise from 6-8 to 12-15 |
| Learning Agent tool limits | Raise from 2 to 4 same-tool calls |

---

## What Should Be Kept

| Component | Reason |
|-----------|--------|
| `DEPRECATED_ATTRIBUTES` blacklist | Catches real LLM hallucinations of Blender 4.x APIs |
| Artifact gates (`artifact_gates.py`) | Best guardrails in the system - deterministic, no LLM |
| Coordinator guardrails (technique, modification, quality) | Lightweight structured output validation |
| Budget guardrail (`check_budget_before_quality`) | Essential cost control |
| Quality output validation (`validate_quality_output`) | Catches rubber-stamped passes |
| Research output validation (`validate_research_output`) | Lightweight, useful |
| Script output validation (`validate_script_output`) | Hallucination scan + path validation |
| Loop detection hooks | Prevent infinite research loops |
| Doc query requirement (for Script Writer only) | Ensures doc search before writing |
| API fixer (Phase 1.5) | Auto-corrects known bad patterns without rejection |
| 3 Coordinator agents | Genuinely influence pipeline decisions |
| Vector store doc search | Works reliably - the foundation everything else should build on |

---

## What Should Be Added

### 1. Crash Pattern Detector
Static analysis of generated scripts for known crash-causing patterns:
- `ShaderNodeMixRGB` → `ShaderNodeVolumePrincipled` connections (Cycles crash)
- Invalid shader socket names for Blender 5.0
- Missing domain object activation before `bpy.ops.fluid.bake_all()`

### 2. Post-Generation Auto-Fixer (Expand API Fixer)
Instead of rejecting scripts via guardrails, auto-fix known issues:
- Replace deprecated attributes automatically
- Fix shader node connections
- Add missing `use_plane_init = True` for liquids
- Replace `BLENDER_EEVEE_NEXT` with `BLENDER_EEVEE`

### 3. Prompt Fidelity Check
Verify the generated script addresses key elements from the user's description:
- Extract keywords from prompt (fire, liquid, smoke, etc.)
- Check script contains corresponding setup (GAS domain for fire, LIQUID for liquid)
- Flag if script is missing major requested elements

### 4. Actionable Quality Feedback
Convert quality analyst output into specific code changes:
- "Overexposed" → reduce light energy, add tone mapping
- "Too dark" → increase emission strength, add area lights
- "No simulation visible" → check bake succeeded, check domain visibility

### 5. Unified Attribute Truth Source
Single file with correct Blender 5.0 attribute names, types, and ranges. Eliminate the 3-way contradiction between dynamic_instructions, api_validator, and docs_expert.

---

## Proposed Simplified Architecture

```
create_asset_pipeline() (Python state machine)
│
├── PHASE 0: Research
│   ├── Research Agent (vector store search)
│   └── Docs Expert (API verification)
│
├── PHASE 0.5: Technique Selection
│   └── Technique Coordinator
│
├── PHASE 1: Script Generation
│   └── Script Writer (with doc query requirement)
│       Post-processing: Auto-fixer + Crash pattern check + Deprecated blacklist
│
├── PHASE 1.5: API Validation (static, auto-fix)
│
├── PHASE 2: Execution
│   └── Executor Agent
│
├── PHASE 2.7: Artifact Gates (deterministic)
│
├── PHASE 3: Quality Evaluation
│   └── Quality Analyst (vision + metrics)
│
├── PHASE 4: Learning + Decision
│   ├── Learning Agent (record, extract patterns)
│   └── Quality Gate Coordinator (pass/iterate/switch)
│
└── PHASE 5: Iteration (if needed)
    └── Modification Coordinator → back to PHASE 1

REMOVED:
- API Spec Agent (catch-22 source)
- Code Writer Agent (replaced by Script Writer + auto-fixer)
- validate_code_against_spec guardrail (deadlock source)
- validate_api_spec guardrail (over-strict)
- Bundle-first enforcement
- ~400 lines of dead handoff code
```

**Net reduction:** ~2 agents removed, ~1,000 lines of guardrail code removed, ~400 lines of dead code removed. The system goes from "7 validation layers before execution" to "3 layers (auto-fix + deprecated check + artifact gates)" with zero rejection-based guardrails on script content.

---

## Execution Priority

| Priority | Action | Impact | Effort |
|----------|--------|--------|--------|
| P0 | Remove `validate_code_against_spec` Layers 1+3 | Unblocks pipeline immediately | Small (delete code) |
| P0 | Remove API Spec Agent from pipeline flow | Eliminates 10-60% runtime waste | Medium (rewire orchestrator) |
| P0 | Fix `timesteps_per_frame`/`timesteps_maximum`/`timesteps_max` contradiction | Prevents silent API errors | Small (3 file edits) |
| P1 | Delete dead handoff code (~400 lines) | Reduces complexity, removes 5 unused agents | Small (delete code) |
| P1 | Add crash pattern detector (MixRGB→Volume) | Prevents Blender crashes | Medium (new static analysis) |
| P1 | Expand API fixer to auto-correct more patterns | Replaces rejection with correction | Medium (extend existing) |
| P2 | Trim Script Writer instructions from ~15K tokens | Frees context for reasoning | Medium (rewrite instructions) |
| P2 | Unify attribute truth source | Eliminates contradictions | Medium (consolidate 3 files) |
| P2 | Add actionable quality feedback mapping | Fixes quality plateau problem | Medium (new mapping logic) |
| P3 | Add prompt fidelity check | Prevents simplification cascade | Medium (new analysis) |
