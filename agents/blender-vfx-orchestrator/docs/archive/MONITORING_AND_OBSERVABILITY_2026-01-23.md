# Monitoring & Observability Enhancements (Agents SDK) — 2026-01-23

This document lists Agents SDK features and patterns you can use to improve monitoring and runtime observability beyond what is currently in place.

## 1) Tracing (Core)
SDK tracing is already in use, but the most reliable pattern is:
- **One outer `trace()`** wrapping the whole pipeline.
- Use `group_id=session_id` for filtering.

Reference: https://github.com/openai/openai-agents-python/blob/main/docs/tracing.md

Why it helps:
- Single row per pipeline execution.
- Flow column shows agent sequence.
- Each `Runner.run()` call becomes a span.

## 2) Structured Outputs Everywhere
You already use `AgentOutputSchema` for some agents. Extend it so:
- Research Agent returns structured fields (approach, alternatives, params).
- Coordinator decisions use `strict_json_schema=True` where safe.

Reference: https://github.com/openai/openai-agents-python/blob/main/docs/agents.md

Why it helps:
- Logs become machine-parseable.
- Iteration history becomes consistent.

## 3) RunHooks for Instrumentation
RunHooks can act as a **runtime telemetry hook**:
- on_agent_start / on_tool_start: log timing and metadata.
- on_tool_end: record outputs and latency.
- on_handoff: record transitions.

Reference: https://github.com/openai/openai-agents-python/blob/main/docs/run_hooks.md

Why it helps:
- Centralized telemetry without polluting agent prompts.
- Consistent instrumentation across all agents.

## 4) Sessions for Context Audit
SQLiteSession gives persistent conversation context. Add a “session snapshot” log:
- after each iteration, log summary + context size
- optionally rotate sessions to control growth

Reference: https://github.com/openai/openai-agents-python/blob/main/docs/sessions/index.md

Why it helps:
- Debugs “why did the agent decide X?”
- Prevents unbounded context growth.

## 5) Guardrails for Output Validation
Guardrails already exist. Add **diagnostic logging** when a guardrail trips:
- include `guardrail_result.output_info`
- log which rule failed

Reference: https://github.com/openai/openai-agents-python/blob/main/docs/guardrails.md

Why it helps:
- Rapid diagnosis of invalid outputs.

## 6) Visualization (Topology)
Use `draw_graph()` to generate a static topology diagram:
- show agent + tool + MCP relationships
- include in docs for onboarding

Reference: https://github.com/openai/openai-agents-python/blob/main/docs/visualization.md

Why it helps:
- Quick sanity check for orchestration architecture.

## 7) Explicit Trace Metadata
The SDK supports trace metadata; consistently include:
- `phase`
- `iteration`
- `effect_type`
- `technique`

Reference: https://github.com/openai/openai-agents-python/blob/main/docs/tracing.md

Why it helps:
- Filter traces by phase / issue.

## Quick Wins (No Code Refactor)
1) Remove nested `trace()` blocks (keep one outer).
2) Use `group_id=session_id` on the outer trace.
3) Add RunHooks timing logs for each tool call.
4) Structured output for Research Agent.

