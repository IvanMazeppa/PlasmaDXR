# Tracing Issue Analysis: Separate Trace Rows Instead of Nested Spans

**Date:** 2026-01-23
**Status:** ✅ RESOLVED (2026-01-23)
**Severity:** Medium (affects observability, not functionality)
**Freshness Note:** Historical tracing incident report. For current runtime truth, see `docs/RUNTIME_TRUTH_AND_DOC_GROUNDING_2026-02-13.md`.

## Resolution Summary

**Fix Applied:** Removed all 10 inner `with trace()` blocks from `orchestrator.py`, keeping only the 3 outer traces. Added `group_id=session.session_id` to outer traces for filtering.

**Changes Made:**
- Removed inner traces at lines: 1260-1267, 1330-1337, 1441-1449, 1553-1561, 1691-1700, 1833-1841, 1890-1898, 2027-2037, 2091-2100, 2157-2166
- Added `group_id=session.session_id` to `VFX Pipeline` trace
- Added `group_id=session.session_id` to `VFX Resume` trace

**Expected Result:** Single trace row per pipeline with Flow column showing agent sequence.

---

## Problem Statement

The OpenAI Platform Traces dashboard shows each pipeline phase as a **separate top-level trace** instead of **nested spans within a single trace**.

### Expected Behavior (CORRECT - seen in older runs)

| Trace Name | Flow Column |
|------------|-------------|
| VFX Pipeline: test_sun_api_validator | Research Agent ► Script Writer ► Executor ► Quality Analyst ► Learning Agent |
| VFX Pipeline: test_pipeline | Research Agent ► Script Writer ► Executor ► Quality Analyst ► Learning Agent |

- **ONE row** per pipeline execution
- **Flow column** shows agent sequence with arrows (►)
- All phases nested as spans within the trace

### Actual Behavior (BROKEN - seen in recent runs)

| Trace Name | Flow Column |
|------------|-------------|
| Phase 3: Quality Analyst (iter 1) | N/A |
| Phase 2: Executor (iter 1) | N/A |
| Phase 1: Script Writer (iter 1) | N/A |
| Phase 0.5: Technique Selection | N/A |
| Phase 0: Research | N/A |
| VFX Pipeline: explosion_v4 | N/A |

- **MULTIPLE rows** per pipeline execution (one per phase)
- **Flow column** shows "N/A"
- Phases appear as separate top-level traces, not nested spans

---

## Visual Evidence

Screenshot: `images/Screenshot 2026-01-23 053917.png`

The screenshot clearly shows:
- Top 5-7 entries: Each phase is a separate row with "N/A" flow
- Bottom entries: Single row per pipeline with full agent flow sequence

---

## Technical Analysis

### Current Code Structure (orchestrator.py)

The current implementation uses **nested `with trace()` blocks**:

```python
# Line ~1243: OUTER trace
with trace(f"VFX Pipeline: {request.asset_name}"):

    # Line ~1257: INNER trace for Phase 0
    with trace(
        "Phase 0: Research",
        metadata={
            "phase": "research",
            "effect_type": request.effect_type.value,
            "iteration": "0",
        }
    ):
        research_result = await Runner.run(...)

    # Line ~1327: INNER trace for Phase 0.5
    with trace(
        "Phase 0.5: Technique Selection",
        metadata={...}
    ):
        technique_result = await Runner.run(...)

    # ... more inner traces for Phase 1, 2, 3, 4, 5
```

**Total trace() calls in orchestrator.py:** 13
- 3 outer traces (VFX Pipeline, VFX Resume, VFX Asset)
- 10 inner traces (Phase 0, 0.5, 1, 1.1, 2, 3, 4, 5, technique switch, etc.)

### SDK Documentation Reference

**Source:** https://github.com/openai/openai-agents-python/blob/main/docs/tracing.md

#### What the SDK Says About Nested Traces

> "Multiple calls to `run()` can be grouped into a single trace by wrapping the entire code block in a `trace()` context manager."

> "When calls to `Runner.run` are wrapped in a `with trace()` statement, the individual runs become part of the overall trace rather than creating separate traces for each run."

#### SDK Example (Correct Pattern)

```python
from agents import Agent, Runner, trace

async def main():
    agent = Agent(name="Joke generator", instructions="Tell funny jokes.")

    with trace("Joke workflow"):  # ONE outer trace
        first_result = await Runner.run(agent, "Tell me a joke")
        second_result = await Runner.run(agent, f"Rate this joke: {first_result.final_output}")
```

**Key Point:** The SDK example shows ONE `trace()` wrapper with multiple `Runner.run()` calls inside - NO nested `trace()` calls.

---

## What Changed?

### Timeline of Changes

| Date | Change | Relevant Files |
|------|--------|----------------|
| ~2026-01-22 | Added Phase 7 (Tracing everywhere) | orchestrator.py |
| ~2026-01-22 | Added inner `with trace()` for each phase | orchestrator.py |
| ~2026-01-22 | Added metadata to inner traces | orchestrator.py |

### The Problematic Commit

The "Tracing everywhere" implementation (Phase 7) added inner `trace()` calls for each phase to provide per-phase metadata. However, this appears to conflict with how the SDK handles nested traces.

### Warning Message Observed

During test runs, this warning appears:
```
Trace already exists. Creating a new trace, but this is probably a mistake.
```

This warning appears for EACH inner `with trace()` call, suggesting the SDK is detecting the nesting and warning about it.

---

## Hypotheses

### Hypothesis 1: Nested traces create separate top-level traces

The SDK may interpret nested `trace()` calls as requests for separate traces rather than nested spans. The warning "Creating a new trace, but this is probably a mistake" supports this.

### Hypothesis 2: Missing group_id linking

The inner traces may need a `group_id` parameter to link them to the outer trace. Without this, they may be treated as independent traces.

```python
# Potentially needed:
with trace("Phase 0: Research", group_id=outer_trace_id):
    ...
```

### Hypothesis 3: trace() vs span behavior

The `trace()` function may always create a new trace, not a span. There may be a separate `span()` function or pattern for nested operations.

### Hypothesis 4: SDK version change

The behavior may have changed between SDK versions. Current version: v0.6.9.

---

## Attempted Solutions

### Solution 1: Keep inner traces (current state)

**Result:** ❌ Creates separate top-level traces

### Solution 2: Remove inner traces (not yet implemented)

**Proposed change:** Remove all inner `with trace()` blocks, keeping only the outer one.

**Challenge:** The inner traces span multiple lines with metadata dictionaries. Removing them requires careful refactoring to:
1. Remove the `with trace(...)` block opener
2. Dedent the code inside
3. Remove the closing of the block

**Code locations to modify:** Lines 1257, 1327, 1438, 1550, 1688, 1830, 1887, 2010, 2074, 2140

---

## Questions for Investigation

1. **Does the SDK support nested traces?** Or only a single trace with automatic span creation?

2. **What is the correct pattern for per-phase metadata?** If inner traces aren't the answer, how do we attach metadata to specific Runner.run() calls?

3. **Is there a `span()` function** for creating nested operations within a trace?

4. **What changed between working runs (test_pipeline) and broken runs (explosion_v4)?** Were there SDK updates or code changes?

5. **Does `group_id` help?** Can inner traces be linked to outer traces via group_id?

---

## SDK Documentation to Consult

1. **Tracing docs:** https://github.com/openai/openai-agents-python/blob/main/docs/tracing.md
2. **Context7 query:** `/openai/openai-agents-python` - "how to create nested spans within a trace"
3. **OpenAI Developer Docs:** Search for "agents SDK tracing spans nested"

---

## Recommended Next Steps

1. **Query SDK docs** via context7 for nested span patterns
2. **Check SDK source code** for `trace()` vs `span()` implementations
3. **Test removal of inner traces** to see if Runner.run() automatically creates spans
4. **Compare git diff** between working runs and broken runs
5. **Check SDK changelog** for v0.6.9 tracing changes

---

## Appendix: All trace() Calls in orchestrator.py

```
Line 1180: with trace(f"VFX Asset: {request.asset_name}"):      # OUTER (deprecated method)
Line 1243: with trace(f"VFX Pipeline: {request.asset_name}"):  # OUTER (main pipeline)
Line 1257: with trace("Phase 0: Research", metadata={...}):    # INNER
Line 1327: with trace("Phase 0.5: Technique Selection", ...):  # INNER
Line 1438: with trace("Phase 1: Script Writer (iter...)", ...): # INNER
Line 1550: with trace("Phase 1.1: Modification...", ...):      # INNER
Line 1688: with trace("Phase 1: Script Writer (iter...)", ...): # INNER (modification)
Line 1830: with trace("Phase 2: Executor (iter...)", ...):     # INNER
Line 1887: with trace("Phase 3: Quality Analyst...", ...):     # INNER
Line 2010: with trace("Phase 4: Learning Agent...", ...):      # INNER
Line 2074: with trace("Phase 5: Quality Gate...", ...):        # INNER
Line 2140: with trace("Technique Switch Research...", ...):    # INNER
Line 2220: with trace(f"VFX Resume: {session.session_id}"):    # OUTER (resume method)
```

---

*This document should be updated as investigation progresses.*
