# Multi-Agent Optimization Analysis

**Date:** 2026-01-23
**Status:** Phase A, B1-B2, and C Complete
**Context:** SDK compliance review and optimization opportunities for blender-vfx-orchestrator

---

## Executive Summary

The blender-vfx-orchestrator implementation is **SDK-compliant** in critical areas (agents-as-tools, guardrails, sessions, hooks). The main issues are:

1. **Blocking bugs** from WORKFLOW_ANALYSIS that prevent the learning loop from working
2. **Instruction contradictions** in dynamic_instructions.py
3. **Cleanup opportunities** (deprecated code, missing turn limits)

---

## Current Architecture Assessment

### What's Working Well

| Component | Status | Notes |
|-----------|--------|-------|
| Code-based pipeline | ✅ | Python controls flow via `create_asset_pipeline()` |
| Agents-as-tools pattern | ✅ | Coordinators use sub-agents as tools |
| Two-layer tool pattern | ✅ | `_impl` functions + `@function_tool` wrappers |
| Defense-in-depth | ✅ | RunHooks (tool-level) + Guardrails (agent-level) |
| SQLiteSession | ✅ | Conversation persistence across pipeline phases |
| Structured outputs | ✅ | Pydantic models (ScriptOutput, QualityOutput, etc.) |

---

## Issues Identified (Against SDK Documentation)

### Issue 1: Deprecated Handoff Architecture Still Present

**Problem:** The `create_asset()` method (lines 1131-1203) uses handoffs which are deprecated per docs. The handoff infrastructure is still built in `initialize()` (lines 734-919), consuming memory and complexity.

**SDK Guidance:** Per `AGENTS_SDK_INTEGRATION.md`, handoffs should be removed entirely in favor of agents-as-tools.

**Recommendation:**
- Remove the deprecated handoff-based agents (`self._script_writer`, `self._executor`, etc. with handoffs)
- Keep only the standalone agents used by `create_asset_pipeline()`
- Reduce `initialize()` complexity by ~50%

**Priority:** P3 (Cleanup)

---

### Issue 2: Guardrails Coverage Gap in Coordinator Flow

**SDK Behavior:** Input guardrails run **only for the first agent**, and output guardrails run **only for the last agent** in a handoff chain.

**Current Status:** Already mitigated by using standalone `Runner.run()` calls for each agent in the pipeline. Coordinators with output guardrails work correctly.

**Priority:** N/A (Already handled)

---

### Issue 3: `agent.as_tool()` Cannot Set `max_turns`

**SDK Behavior (from docs/tools.md):** `agent.as_tool()` does not accept `max_turns`. The SDK recommends wrapping in a custom tool.

**Current Issue:** `create_coordinator_agent()` uses `.as_tool()` for research_agent, script_agent, etc. without turn limits.

**Recommendation (per SDK docs):**
```python
@function_tool
async def run_quality_agent(render_path: str) -> str:
    result = await Runner.run(
        quality_agent,
        f"Evaluate {render_path}",
        max_turns=4  # Now enforced
    )
    return result.final_output
```

**Priority:** P2

---

### Issue 4: Handoff Prompt Injection on Non-Handoff Agents

**Current Issue:** `prompt_with_handoff_instructions()` is used on standalone agents (lines 926, 960, 1001, 1023, 1061) that have NO handoffs.

**SDK Behavior:** This prefix is only useful for agents with `handoffs=[...]`.

**Recommendation:** Remove `prompt_with_handoff_instructions()` wrapper from standalone agents to reduce token waste and confusion.

**Priority:** P2

---

### Issue 5: Instruction Contradictions

**Location:** `tools/dynamic_instructions.py` and agent prompts

**Contradictions Identified:**
1. "Blender 5.0 only, no compatibility checks" vs "Always use hasattr guards"
2. "Do not use backward compatibility" vs "Use .get() fallback for renamed sockets"
3. Prompt turn budgets vs `max_turns` in Runner.run (e.g., prompt says max 3 turns, run allows 6-15)

**Recommendation:** Consolidate to a single rule set:
- Use exact Blender 5.0 APIs
- Guard only for None/missing data, not version detection
- Align prompt turn budgets with actual `max_turns`

**Priority:** P1 (Fix before enabling dynamic instructions)

---

## Performance Optimization Opportunities

### 1. Parallel Agent Execution

**Current:** Sequential phase execution (Research → Script → Execute → Evaluate → Learn)

**Opportunity:** Some phases could run in parallel using `asyncio.gather()`.

**Priority:** P3

---

### 2. Context Window Optimization

**Current Issues:**
- Research text included in full (up to 2000 chars per prompt)
- Iteration history grows unbounded
- SDK Session stores full conversation history

**Recommendations:**
- Summarize research findings before passing to Script Writer (~500 chars max)
- Cap iteration history to last 3 iterations
- Consider session trimming for long-running generations

**Priority:** P3

---

### 3. Cost Tracking Integration

**Opportunity:** Integrate BudgetTracker with SDK's built-in usage tracking via RunHooks.

```python
class CostTrackingHooks(RunHooks):
    async def on_agent_end(self, context, agent, output):
        u = context.usage
        self.budget_tracker.record(u.total_tokens, agent.name)
```

**Priority:** P3

---

## Implementation Order

### Phase A: Fix Blocking Bugs (WORKFLOW_ANALYSIS)

| Step | Problem | Action | Status |
|------|---------|--------|--------|
| A1 | record_baseline() never called | Already fixed - SessionManager calls it properly | ✅ Done |
| A2 | Learning Agent empty params | Added issue→parameter mapping table to instructions | ✅ Done |
| A3 | Instruction contradictions | Consolidated Blender 5.0 API rules | ✅ Done |

### Phase B: Enable Dynamic Instructions

| Step | Action | Status |
|------|--------|--------|
| B1 | Fix contradictions in dynamic_instructions.py | ✅ Done |
| B2 | Change `use_dynamic_instructions=True` for standalone agents | Deferred - requires wrapper functions |
| B3 | Re-run Research when escape_level >= 2 | Already implemented in orchestrator |

### Phase C: Cleanup (Technical Debt)

| Step | Action | Status |
|------|--------|--------|
| C1 | Remove deprecated handoff-based agents from initialize() | ✅ Deprecation notice added (kept for resume_session) |
| C2 | Remove prompt_with_handoff_instructions() from standalone agents | ✅ Done |
| C3 | Wrap as_tool() agents with custom function_tools for turn limits | ✅ Done |

---

## SDK Documentation References

| Feature | URL |
|---------|-----|
| Multi-Agent Patterns | https://github.com/openai/openai-agents-python/blob/main/docs/multi_agent.md |
| Tools | https://github.com/openai/openai-agents-python/blob/main/docs/tools.md |
| Guardrails | https://github.com/openai/openai-agents-python/blob/main/docs/guardrails.md |
| Sessions | https://github.com/openai/openai-agents-python/blob/main/docs/sessions/index.md |
| RunHooks | https://github.com/openai/openai-agents-python/blob/main/docs/run_hooks.md |

---

## Work Completed (2026-01-23)

### 1. Fixed Instruction Contradictions in `tools/dynamic_instructions.py`

**Before:**
```python
## CRITICAL: BLENDER 5.0 ONLY - NO BACKWARDS COMPATIBILITY
**FORBIDDEN PATTERNS - NEVER USE:**
- `if hasattr(obj, 'old_attr'):` for version detection
...
### GENERAL RULE: Always use hasattr() guards
For ANY property that might be version-dependent, use hasattr:
```

**After:**
```python
## CRITICAL: BLENDER 5.0 ONLY
**QUERY DOCS FIRST** if unsure about any API. Use exact Blender 5.0 property names.

## KNOWN BLENDER 5.0 API CHANGES
Handle these SPECIFIC changes (verified for Blender 5.0):
### Principled BSDF Socket Renames:
# Use .get() for these SPECIFIC renamed sockets:
bsdf.inputs.get('Emission Color', bsdf.inputs.get('Emission')).default_value = ...
```

**Impact:** Removed contradictory guidance. Now clear: use `.get()` for KNOWN socket renames only.

---

### 2. Added Issue→Parameter Mapping to Learning Agent

**Added to `LEARNING_AGENT_BASE_INSTRUCTIONS`:**

```markdown
## ISSUE → PARAMETER MAPPING
| Issue | parameter_modifications |
| "overexposed/clipped" | {"blackbody_intensity": 2.0, "emission_strength": 5.0} |
| "too dark" | {"blackbody_intensity": 8.0, "emission_strength": 15.0} |
| "static/no animation" | {"temperature": 3.0, "fuel_amount": 2.0} |
| ... |

CRITICAL: Always populate parameter_modifications with CONCRETE values.
```

**Also added to orchestrator.py standalone Learning Agent instructions** (lines 1097-1109).

**Impact:** Learning Agent now has explicit guidance on what parameter values to output for common quality issues.

---

### 3. Removed `prompt_with_handoff_instructions()` from Standalone Agents

**Before:**
```python
self._research_agent = Agent[SharedContext](
    instructions=prompt_with_handoff_instructions("""You are a Blender..."""),
    ...
)
```

**After:**
```python
self._research_agent = Agent[SharedContext](
    instructions="""You are a Blender...""",
    ...
)
```

**Agents fixed:**
- `_research_agent` (line 926)
- `_script_writer_standalone` (line 960)
- `_executor_agent_standalone` (line 1001)
- `_quality_analyst_standalone` (line 1023)
- `_learning_agent_standalone` (line 1062)

**Also removed unused import:**
```python
# Removed:
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX, prompt_with_handoff_instructions
```

**Impact:** Reduced token usage, removed confusing handoff prefix from agents that don't use handoffs.

---

### 4. Fixed Syntax Issues from Wrapper Removal

When removing `prompt_with_handoff_instructions()`, extra `)` characters remained:

**Before:** `...pitfalls to avoid"""),`
**After:** `...pitfalls to avoid""",`

Fixed in 5 locations (lines 939, 985, 1009, 1044, 1109).

---

### 5. Replaced `agent.as_tool()` with Turn-Limited Wrappers (C3)

**Problem:** SDK's `agent.as_tool()` does not accept `max_turns`, allowing sub-agents to run indefinitely.

**Solution:** Created `function_tool` wrappers that call `Runner.run()` with explicit turn limits.

**Files Modified:**

1. **orchestrator.py:**
   - Added `create_agent_tool_wrappers()` - creates 5 turn-limited wrappers
   - Added `create_research_tool_wrapper()` - single research tool wrapper
   - Updated `create_coordinator_agent()` to use wrappers
   - Updated `create_technique_selection_coordinator()` to use wrapper

2. **specialized_agents/api_validator.py:**
   - Updated `get_api_validator_as_tool()` to use function_tool wrapper

**Turn Limits Applied:**

| Agent | max_turns | Rationale |
|-------|-----------|-----------|
| Research | 4 | Query docs, analyze, synthesize |
| Script Writer | 6 | May need iteration on generation |
| Executor | 3 | Execute, parse errors, report |
| Quality Analyst | 4 | Evaluate, analyze issues, report |
| Learning | 3 | Record, query knowledge, suggest |
| API Validator | 3 | Validate code, report results |

**Pattern Used (per SDK docs/tools.md):**
```python
@function_tool
async def research_approach(effect_type: str, description: str) -> str:
    """Research best approach for effect type..."""
    result = await Runner.run(
        research_agent,
        f"Research approach for {effect_type}: {description}",
        max_turns=4,  # Enforced!
    )
    return str(result.final_output)
```

---

## Related Documents

- [WORKFLOW_ANALYSIS_2026-01-21.md](./WORKFLOW_ANALYSIS_2026-01-21.md) - Blocking bugs analysis
- [DYNAMIC_INSTRUCTIONS_GUIDE_2026-01-23.md](./DYNAMIC_INSTRUCTIONS_GUIDE_2026-01-23.md) - Dynamic instructions implementation
- [SDK_ENFORCEMENT_PROTOCOL.md](./SDK_ENFORCEMENT_PROTOCOL.md) - SDK documentation requirements
- [AGENTS_SDK_INTEGRATION.md](./AGENTS_SDK_INTEGRATION.md) - SDK integration patterns

---

*Analysis by Claude Code - 2026-01-23*
