# Parallel Workflow Audit: Blender VFX Orchestrator

**Date:** 2026-01-29
**Trace Analyzed:** `shakedown_2iter_20260129_195522.jsonl` (10.5 min, 632s)
**Status:** Actionable recommendations

---

## Executive Summary

The current orchestrator workflow is **strictly sequential**, running agents one after another with almost no parallelism. Analysis of a 2-iteration trace reveals:

| Metric | Value |
|--------|-------|
| Total runtime | 632s (10.5 min) |
| Script Writer time | 331s (52% of total) |
| Doc search calls | 22 total (14 by Script Writer alone) |
| Wasted gap time | 13s |
| Parallelization potential | **25-93s savings** |

**Key Finding:** Script Writer spending 331s with 14 doc searches is the #1 bottleneck. The Spec-First pipeline should eliminate this, but wasn't used in this trace.

---

## Current Workflow Timeline

```
TIME(s)  PHASE                    AGENT                   DURATION
────────────────────────────────────────────────────────────────────
0        Phase 0: Research        Research Agent          84s
84       Phase 0.5: Technique     Technique Selector      25s
109      [GAP]                    -                       13s
122      Phase 7: API Spec        API Spec Agent          93s
216      Phase 1: Script Gen      Script Writer           124s
340      Phase 1.1: Mod Strategy  Modification Strat.     16s
356      Phase 1: Script Modify   Script Writer           208s
564      Phase 2: Execute         Executor                68s
632      END
```

### Bottleneck Analysis

| Agent | Calls | Total Time | % of Runtime | Doc Searches |
|-------|-------|------------|--------------|--------------|
| Script Writer | 2 | 331s | 52% | 14 |
| API Spec Agent | 1 | 93s | 15% | 8 |
| Research Agent | 1 | 84s | 13% | 2 |
| Executor | 1 | 68s | 11% | 0 |
| Technique Selector | 1 | 25s | 4% | 0 |
| Modification Strat. | 1 | 16s | 3% | 0 |

---

## Parallelization Opportunities

### Tier 1: High Impact (Implement First)

#### 1. Technique Selector || API Spec Agent Prep

**Current:** Sequential (Technique 25s → API Spec 93s)
**Proposed:** Run in parallel

```
BEFORE: Research → Technique (25s) → API Spec (93s) → Script
AFTER:  Research → [Technique || API Spec] → Script
                   ^^^^^^^^^^^^^^^^^^^^
                   Run in parallel (max 93s instead of 118s)
```

**Savings:** ~25s (4% of total)

**Implementation:**
```python
async def _run_parallel_phase_0_5(self, request, context, ...):
    """Run Technique Selection and API Spec in parallel."""
    technique_task = self._run_agent(self._technique_coordinator, ...)
    api_spec_task = self._run_spec_first_phase_1a(...)  # Just API Spec, not Code Writer

    technique_result, api_spec_result = await asyncio.gather(
        technique_task, api_spec_task, return_exceptions=True
    )
    return technique_result, api_spec_result
```

**Dependencies:**
- API Spec Agent needs effect_type (from request) ✓
- API Spec Agent needs technique hint (can use default or research output)
- Code Writer still waits for both to complete

---

#### 2. Learning Agent || Quality Gate Coordinator

**Current:** Sequential (Learning → Quality Gate)
**Proposed:** Run in parallel

```
BEFORE: Quality Analyst → Learning (Xs) → Quality Gate (Ys)
AFTER:  Quality Analyst → [Learning || Quality Gate]
                          ^^^^^^^^^^^^^^^^^^^^^^^
                          Both only need QualityOutput
```

**Savings:** min(Learning, QualityGate) ≈ 10-20s

**Implementation:**
```python
async def _run_parallel_post_eval(self, quality_output, context, ...):
    """Run Learning Agent and Quality Gate in parallel."""
    learning_task = self._run_agent(self._learning_agent_standalone, ...)
    gate_task = self._run_agent(self._quality_gate_coordinator, ...)

    learning_result, gate_result = await asyncio.gather(
        learning_task, gate_task, return_exceptions=True
    )
    return learning_result, gate_result
```

**Dependencies:**
- Both need QualityOutput ✓
- Gate decision determines next action (loop vs complete)
- Learning patterns inform next iteration (but not blocking)

---

### Tier 2: Medium Impact

#### 3. Pre-warm Pattern Search During Script Generation

**Current:** Pattern search happens reactively in iteration 2+
**Proposed:** Pre-fetch patterns in background during Script Writer run

```python
# Start pattern search early
pattern_task = asyncio.create_task(
    search_patterns_direct(effect_type=effect_type, issue="")
)

# Run Script Writer
script_result = await self._run_agent(self._script_agent_standalone, ...)

# Patterns ready for next phase if needed
patterns = await pattern_task
```

**Savings:** Variable (reduces iteration 2+ latency)

---

#### 4. Executor Progress via Streaming

**Current:** Executor blocks until Blender completes (68s black box)
**Proposed:** Use `Runner.run_streamed()` to show Blender progress

```python
async def _run_executor_streamed(self, script_path, context, ...):
    """Run Executor with streaming for real-time Blender output."""
    result = Runner.run_streamed(
        self._executor_agent_standalone,
        f"Execute script: {script_path}",
        context=context,
    )

    async for event in result.stream_events():
        if event.type == "run_item_stream_event":
            if event.item.type == "tool_call_output_item":
                # Show Blender stdout/stderr in real-time
                print(f"[Blender] {event.item.output[:100]}", file=sys.stderr)

    return await result.final_result()
```

**Savings:** 0s (UX improvement, not time savings)
**Value:** User sees Blender progress instead of waiting in dark

---

### Tier 3: Lower Impact / More Complex

#### 5. Speculative Execution: Start Quality Analyst During Execution

**Current:** Wait for Executor to complete before Quality Analyst
**Proposed:** Start Quality Analyst warmup while Executor runs

```
BEFORE: Executor (68s) → Quality Analyst (Xs)
AFTER:  Executor (68s)
            ↓ [at 50s mark, artifact gate passes]
            └─→ Quality Analyst starts warmup
```

**Complexity:** High (requires artifact gate to trigger early)
**Savings:** Variable (5-15s if artifacts ready early)

---

#### 6. Parallel API Validation

**Current:** API Validator runs after Script Writer
**Proposed:** Run during Script Writer's final turn

**Complexity:** Requires Script Writer to emit script_path before completing
**Savings:** ~5-10s

---

## Context Bloat Analysis

### Problem: 78k Token Trace

The trace file is 78k tokens, indicating massive context accumulation:

| Issue | Evidence | Impact |
|-------|----------|--------|
| Doc search spam | 14 doc searches in Script Writer | 14x tool call/response cycles |
| Full research in prompts | `research_text[:1500]` inline | 1500 chars per prompt |
| Verbose tool outputs | `blender_doc_search_bundle` returns full JSON | Large payloads |
| No compaction | SDK session doesn't truncate | Unbounded growth |

### Solutions

1. **Artifact References Instead of Inline Content**
   ```python
   # BAD: Inline dump
   prompt = f"Research: {research_text[:1500]}"

   # GOOD: File reference
   prompt = f"Research artifact: {research_artifact_path}"
   ```

2. **Aggressive Session Compaction**
   ```python
   from agents.memory import OpenAIResponsesCompactionSession

   sdk_session = OpenAIResponsesCompactionSession(
       model="gpt-5.2",
       max_context_tokens=16000,  # Force compaction earlier
   )
   ```

3. **Streaming to Reduce Buffering**
   - Stream tool outputs instead of buffering full responses
   - Use `run_item_stream_event` for progress without accumulation

---

## Streaming Integration Plan

Based on SDK docs: https://github.com/openai/openai-agents-python/blob/main/docs/streaming.md

### Phase 1: Executor Streaming (UX)

```python
async def _run_executor_streamed(self, ...):
    result = Runner.run_streamed(self._executor_agent_standalone, prompt, ...)

    async for event in result.stream_events():
        if event.type == "run_item_stream_event":
            item = event.item
            if item.type == "tool_call_item":
                print(f"[Executor] Calling: {item.call.name}", file=sys.stderr)
            elif item.type == "tool_call_output_item":
                # Parse Blender output for progress
                output = item.output[:200]
                if "Frame" in output:
                    print(f"[Blender] {output}", file=sys.stderr)

    return await result.final_result()
```

### Phase 2: Quality Analyst Streaming (Debug)

```python
async def _run_quality_streamed(self, ...):
    result = Runner.run_streamed(self._quality_agent_standalone, prompt, ...)

    async for event in result.stream_events():
        if event.type == "raw_response_event":
            if hasattr(event.data, 'delta'):
                # Show quality analysis as it's generated
                print(event.data.delta, end="", flush=True)
```

### Phase 3: Script Writer Streaming (Critical Path)

Since Script Writer is 52% of runtime, streaming could:
- Show doc search progress
- Early-exit on loop detection
- Yield partial script for validation

---

## Recommended Implementation Order

### Sprint 1: Quick Wins (This Week)

1. **Technique || API Spec parallel preflight**
   - Extend existing `_run_parallel_preflight` pattern
   - Expected savings: 25s

2. **Learning || Quality Gate parallel**
   - Simple fan-out/fan-in
   - Expected savings: 10-20s

### Sprint 2: Streaming (Next Week)

3. **Executor streaming**
   - UX improvement
   - Foundation for streaming pattern

4. **Session compaction tuning**
   - Reduce context bloat
   - Enable longer runs

### Sprint 3: Advanced (Future)

5. **Pattern pre-warming**
6. **Speculative Quality Analyst**
7. **Script Writer streaming with early-exit**

---

## Projected Impact

| Optimization | Savings | Cumulative |
|--------------|---------|------------|
| Current baseline | - | 632s |
| Technique || API Spec | 25s | 607s |
| Learning || Quality Gate | 15s | 592s |
| Pre-warm patterns | 10s | 582s |
| Session compaction | N/A | (enables longer runs) |
| **Total projected** | **50s** | **582s (8% faster)** |

**Note:** The biggest win (331s Script Writer) requires Spec-First pipeline to be fully operational. If working correctly, it should reduce Script Writer from 331s to ~120s.

---

## Appendix: Tool Call Distribution

From trace analysis:

```
Research Agent:
  - blender_doc_search_bundle: 1x
  - list_patterns_by_effect: 1x

API Spec Agent:
  - semantic_search_blender_docs: 8x   ← High, but expected

Script Writer:
  - semantic_search_blender_docs: 14x  ← EXCESSIVE (should be 0 with Spec-First)
  - search_blender_api_by_intent: 1x
  - write_script: 1x
  - validate_script: 1x

Executor:
  - execute_blender_script: 1x
  - list_run_outputs: 1x
```

The Script Writer's 14 doc searches indicate it's hallucinating APIs and validating them reactively. The Spec-First pipeline should prevent this by providing pre-verified APIs.

---

## References

- [SDK Streaming Docs](https://github.com/openai/openai-agents-python/blob/main/docs/streaming.md)
- [Parallel Preflight Patch](./ORCHESTRATOR_PATCH_RUNCONFIG_PARALLEL_PREFLIGHT_2026-01-29.md)
- [Architecture Optimization Plan](./ARCHITECTURE_OPTIMIZATION_PLAN_2026-01-22.md)
