# Tracing Fixes & Solution Ideas (2026-01-23)

This note responds to `TRACING_ISSUE_ANALYSIS.md` and aligns with the SDK guidance in `SDK_ENFORCEMENT_PROTOCOL.md` and the official tracing docs.

## SDK Facts (from official docs)
- The SDK expects **one outer `trace()`** to group multiple `Runner.run()` calls into a single trace.
- Nested `trace()` calls are treated as **new traces**, not spans.
- The current trace is tracked via a **context variable**, so wrapping the whole workflow is the intended pattern.

References:
- https://github.com/openai/openai-agents-python/blob/main/docs/tracing.md
- `docs/SDK_ENFORCEMENT_PROTOCOL.md` (Tracing section + required doc lookup)

## Most Likely Root Cause
The inner `with trace(...)` blocks (Phase 0, 0.5, 1, 2, 3, 4, 5, etc.) are creating **new top-level traces** instead of nested spans. This matches the SDK warning you saw:
```
Trace already exists. Creating a new trace, but this is probably a mistake.
```

This behavior is consistent with the SDK documentation: `trace()` creates a **trace**, not a span. Spans are created automatically for agent runs inside an active trace.

## Recommended Fix (Primary)
**Remove inner `trace()` blocks** and keep only the outer trace that wraps the entire pipeline:

```
from agents import trace

with trace(f"VFX Pipeline: {request.asset_name}", group_id=session_id):
    research_result = await Runner.run(...)
    technique_result = await Runner.run(...)
    script_result = await Runner.run(...)
    exec_result = await Runner.run(...)
    eval_result = await Runner.run(...)
    learn_result = await Runner.run(...)
```

Expected outcome:
- One row per pipeline execution.
- Flow column populated with the agent sequence.
- Individual `Runner.run()` calls become spans within that trace.

## If You Need Per‑Phase Metadata
The SDK docs do not show a separate `span()` API, so the safe approach is:
1) Keep **one outer trace**.
2) Add per‑phase metadata via **logs** or **structured outputs**.
3) If you need filtering at the trace level, use `group_id=session_id` on the outer trace and filter on that.

Avoid nested `trace()` for phase metadata. It will fragment traces.

## Secondary Option (Not Ideal, But Useful)
If you insist on keeping per‑phase traces, use a **shared `group_id`** for every trace in the same session. This won’t give a single flow row, but you can filter/group related traces in the dashboard.

```
with trace("Phase 2: Executor", group_id=session_id):
    exec_result = await Runner.run(...)
```

This is still multiple traces, but at least grouped by session.

## Additional Checks
- Ensure tracing is **not disabled** anywhere (the SDK supports `set_tracing_disabled(True)`).
- Make sure the outer `trace()` wraps **all** `Runner.run()` calls (no early returns inside/outside the context).
- If you use concurrency, ensure all awaited tasks are created within the active trace context (contextvars do not propagate to tasks created outside).

## Evidence Mapping to Your Code
From `TRACING_ISSUE_ANALYSIS.md`:
- Outer trace exists (good).
- Inner traces were added for each phase (problem).
- Warning confirms the nested trace path is treated as a mistake.

## Proposed Minimal Change Sequence
1) Remove all inner `trace()` blocks in `create_asset_pipeline()`.
2) Keep only the outer `trace(f"VFX Pipeline: {request.asset_name}")`.
3) Add `group_id=session_id` to the outer trace (optional but recommended for filtering).
4) Run a single test pipeline and verify the Flow column and single-row trace.

## References
- `docs/tracing.md` (OpenAI Agents SDK)
- `docs/SDK_ENFORCEMENT_PROTOCOL.md` (Tracing rule + doc requirements)
