# Orchestrator Patch: RunConfig + Parallel Preflight (2026-01-29)

This document describes the concrete patch applied to the Blender VFX Orchestrator to integrate additional OpenAI Agents SDK features and reduce rigid linearity.

## What Was Implemented

### 1) RunConfig Support (Best‑Effort)
A safe wrapper was added that **only uses RunConfig if the SDK supports it**.

**Why:** The Agents SDK signature can change; this avoids breaking older versions.

**Behavior:**
- If `Runner.run` supports `run_config`, the orchestrator passes a RunConfig with tracing metadata.
- If not supported, it silently falls back to the standard `Runner.run` call.

**Code areas:**
- `orchestrator.py`: `_supports_run_config`, `_build_run_config`, `_run_agent`

---

### 2) Parallel Preflight (Research + DocsExpert)
A fan‑out/fan‑in preflight step runs **Research Agent** and **Docs Expert** concurrently.

**Why:** Docs search and research do not depend on each other; parallelism reduces latency and adds robustness.

**Behavior:**
- If enabled, both agents run in parallel and their outputs are merged.
- If disabled or no DocsExpert, it falls back to sequential research only.

**Toggle:**
- Env var: `VFX_PARALLEL_PREFLIGHT=1` (default enabled)

**Code areas:**
- `orchestrator.py`: `_run_parallel_preflight`

---

### 3) RunConfig Propagation to Key Phases
The wrapper `_run_agent` is now used in these phases:
- Research
- Technique selection
- Script generation (new + modification + fallback)
- Spec‑first pipeline (API spec + code writer)
- Execution + recovery
- Quality evaluation
- Learning
- Quality gate
- Technique switch research

This ensures consistent tracing metadata and centralized configuration.

---

## How to Operate / Test

### Quick Enable/Disable Parallel Preflight
```
export VFX_PARALLEL_PREFLIGHT=1   # enable (default)
export VFX_PARALLEL_PREFLIGHT=0   # disable
```

### Expected Observations
- Research output now includes a **Docs Expert Notes** section when available.
- Traces (if enabled in your SDK config) will show phase metadata.
- If the SDK lacks RunConfig support, behavior matches previous runs.

### Validation Checklist
- Research completes even if Docs Expert fails (non‑fatal).
- Script generation still proceeds normally.
- No errors about unexpected `run_config` parameters.

---

## Files Changed
- `agents/blender-vfx-orchestrator/orchestrator.py`

---

## Known Limits
- Streaming was not added yet.
- Tool guardrails and advanced gating via `RunResult` remain planned (not part of this patch).
- Durable workflows (Temporal/Restate) not included.

---

## Next Steps (Optional)
- Add `Runner.run_streamed()` for executor phase.
- Add tool guardrails for executor and API fixer.
- Use `RunResult.new_items` for deterministic gating.

