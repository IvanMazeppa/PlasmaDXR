# Tracing Findings and Changes Report

**Date:** 2026-01-24
**Author:** Claude (Opus 4.5)
**Session Focus:** Local trace infrastructure for AI-parseable workflow analysis

---

## Executive Summary

Implemented local tracing infrastructure to diagnose why the VFX orchestrator produces repetitive results. The traces revealed a **critical communication breakdown** between the Modification Coordinator and `_modify_script_impl` - the Coordinator outputs human-readable fix instructions that cannot be parsed by the regex-based parameter modifier.

---

## Problem Statement

### Original Symptoms
1. Generated `.blend` files showed identical object names, shapes, and modifiers across sessions
2. Same techniques repeated despite "different" iterations
3. Coordinator correctly diagnosed issues but fixes never applied
4. Infinite loops of the same error → same diagnosis → no change

### User's Question
> "Do you think the hardcoded data we use is causing problems with the agent, or is it some kind of breakdown in communication between subagents?"

**Answer: Communication breakdown, not hardcoded data.**

---

## Tracing Infrastructure Implemented

### 1. CommunicationFlowTracker (`hooks/diagnostic_hooks.py`)

Records the full Coordinator → modify_script data flow for offline analysis.

```python
class CommunicationFlowTracker:
    def record_coordinator_output(coordinator, decision, parameters, reasoning)
    def record_modify_result(changes_made, success, error)
    def analyze() -> Dict  # Returns breakdown statistics
```

**Output Location:** `traces/communication_flow.jsonl`

**Data Captured:**
- Coordinator name and decision type
- Full parameter dictionary from Coordinator
- Reasoning excerpt (first 200 chars)
- Whether modify_script applied changes
- Success/failure status

### 2. DiagnosticHooks Integration (`orchestrator.py`)

Added `enable_diagnostics` parameter to `create_vfx_asset()`:

```python
session = await create_vfx_asset(
    asset_name="my_asset",
    description="...",
    effect_type="fire",
    enable_diagnostics=True,  # NEW
    diagnostic_log="traces/my_trace.jsonl",  # NEW
)
```

### 3. Pre-Execution API Fixer (`tools/blender_api_fixer.py`)

Auto-fixes known Blender 5.0 API issues BEFORE script execution, breaking the infinite loop.

**Patterns Fixed (22 total):**
| Pattern | Replacement | Description |
|---------|-------------|-------------|
| `ColorRamp.elements.clear()` | while loop removal | Removed in Blender 5.0 |
| `.adaptive_domain =` | `.use_adaptive_domain =` | Renamed property |
| `flow_behavior = 'GAS'` | `flow_behavior = 'INFLOW'` | Invalid enum |
| `flow_type = 'GAS'` | `flow_type = 'FIRE'` | Invalid enum |
| `.use_caching =` | (line deleted) | Property removed |
| `obj.select = True` | `obj.select_set(True)` | Deprecated method |
| `scene.update()` | `depsgraph.update()` | API change |
| And 15 more... | | |

**Integration:** Called automatically in `blender_executor_tools.py` before every Blender execution.

---

## Trace Analysis Results

### Test Run: `diagnostic_160058_fire` (2 iterations)

**Flow Statistics:**
```
Total flow events: 8
Successful flows: 0
Breakdown flows: 4
Success rate: 0%
```

### Root Cause Identified

The Modification Coordinator outputs **human-readable instructions** instead of **machine-parseable Config class parameters**.

**Example 1 - Verbose Instructions:**
```json
{
  "coordinator": "ModificationCoordinator",
  "parameters": {
    "domain.adaptive_property_fix": "replace settings.adaptive_domain = True with settings.use_adaptive_domain = True (or remove the line to use fixed bounds)",
    "domain.resolution_max": 64
  }
}
```

**Problem:** `"domain.adaptive_property_fix"` is a descriptive key with natural language value. `_modify_script_impl` searches for `class Config:` with `DOMAIN.ADAPTIVE_PROPERTY_FIX = ...` which doesn't exist.

**Example 2 - Nested Objects:**
```json
{
  "parameters": {
    "script_fixes": {
      "replace_settings_domain_resolution_assignment": "remove 'settings.domain_resolution = (64, 64, 64)' and instead set...",
      "remove_deprecated_world_use_nodes": "remove or comment out 'world.use_nodes = True'..."
    },
    "concrete_numeric_values": {
      "resolution_scalar": 64,
      "frames_end": 25
    }
  }
}
```

**Problem:** `_modify_script_impl` expects flat `{"PARAM_NAME": value}` structure, not nested objects with instructions.

**Example 3 - Wrong Parameter Names:**
```json
{
  "parameters": {
    "domain_resolution": 64,
    "frame_end": 5
  }
}
```

**Problem:** Script uses `RESOLUTION` and `FRAME_END` (uppercase), not `domain_resolution` and `frame_end`.

### Communication Flow Breakdown

```
┌─────────────────────────────────────────────────────────────┐
│ Modification Coordinator                                     │
│                                                             │
│ Input: "Execution timed out after 600 seconds"              │
│                                                             │
│ Output: {                                                   │
│   "action": "modify_params",                                │
│   "parameter_changes": {                                    │
│     "domain_resolution": 64,        ← Wrong name format     │
│     "script_fixes": {...},          ← Nested instructions   │
│     "adaptive_property_fix": "..."  ← Natural language      │
│   }                                                         │
│ }                                                           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ _modify_script_impl()                                       │
│                                                             │
│ Searches for:                                               │
│   class Config:                                             │
│       DOMAIN_RESOLUTION = ...    ← Not found                │
│       SCRIPT_FIXES = ...         ← Not found                │
│       ADAPTIVE_PROPERTY_FIX = ...← Not found                │
│                                                             │
│ Result: changes_made = []                                   │
│         success = False (file created but unchanged)        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ Next Iteration                                              │
│                                                             │
│ Same script → Same error → Same diagnosis → Loop forever    │
└─────────────────────────────────────────────────────────────┘
```

---

## Files Changed

### New Files

| File | Purpose |
|------|---------|
| `tools/blender_api_fixer.py` | Pre-execution API pattern fixer (22 patterns) |
| `traces/communication_flow.jsonl` | Flow tracker output (AI-parseable) |
| `docs/WORKFLOW_AUDIT_ROOT_CAUSE_2026-01-24.md` | Root cause analysis |
| `docs/TRACING_FINDINGS_AND_CHANGES_2026-01-24.md` | This document |

### Modified Files

| File | Changes |
|------|---------|
| `orchestrator.py` | Added `_flow_tracker`, `_diagnostic_hooks`, breakdown detection logging, `enable_diagnostics` parameter |
| `hooks/diagnostic_hooks.py` | Added `CommunicationFlowTracker` class, `analyze_coordinator_flow()` method |
| `hooks/__init__.py` | Export `CommunicationFlowTracker` |
| `tools/blender_executor_tools.py` | Integrated API fixer before execution |
| `test_diagnostic_analysis.py` | Updated to use `enable_diagnostics`, read flow tracker output |

---

## Recommended Next Steps

### Option A: Fix Coordinator Output Format (Recommended)

Update the Modification Coordinator's system prompt to output ONLY flat parameter dictionaries with exact Config class variable names:

```python
# Current (broken)
{"domain_resolution": 64, "script_fixes": {...}}

# Required (working)
{"RESOLUTION": 64, "FRAME_END": 50, "TURBULENCE": 3.5}
```

**Pros:** Clean fix, respects separation of concerns
**Cons:** Requires Coordinator prompt engineering

### Option B: Enhance _modify_script_impl

Add parameter name mapping and support for nested structures:

```python
PARAM_MAP = {
    "domain_resolution": "RESOLUTION",
    "frame_end": "FRAME_END",
    # ...
}
```

**Pros:** Tolerates varied Coordinator output
**Cons:** Grows complexity, doesn't fix root cause

### Option C: Bypass modify_script for API Fixes

When Coordinator detects an API error (not a parameter issue), regenerate the script instead of modifying:

```python
if mod_decision.fix_type == "api_pattern":
    # Don't modify - regenerate with constraints
    script = await generate_script_with_constraints(
        avoid_patterns=["ColorRamp.elements.clear()"],
        required_patterns=[mod_decision.fix_code]
    )
```

**Pros:** API fixes handled properly
**Cons:** More expensive (regeneration vs modification)

---

## SDK Trace Alignment Addendum (2026-01-24)

This addendum treats the Agents SDK docs as the source of truth for tracing
behavior and explains gaps between local JSONL tracing and SDK-native tracing.

### Findings (SDK-Referenced)
1) **SDK tracing is not the primary spine.** The local JSONL flow tracker is
   useful, but the SDK expects `trace()` + `Runner.run()` to be the canonical
   record (see `docs/tracing.md` in the SDK).
2) **Doc-query detection likely undercounts.** Any hook that only flags
   `semantic_search_blender_docs` will miss `blender_doc_search_bundle`,
   causing false "no doc query" records.
3) **Modification output contract is not enforced by guardrails.**
   The SDK supports output guardrails and structured outputs to enforce
   machine-parseable parameter changes (see `docs/guardrails.md` and
   `docs/tools.md` in the SDK).
4) **No correlation key between local JSONL and SDK traces.** Without a shared
   `group_id`, it is hard to match local flow events to SDK trace spans.

### Recommended Changes (SDK-First)
- Wrap every pipeline run with:
  ```python
  from agents import trace

  with trace("VFX Pipeline", group_id=session_id):
      result = await Runner.run(agent, prompt, max_turns=...)
  ```
- Update doc-query detection to include `blender_doc_search_bundle`.
- Add an output guardrail to enforce **flat, exact Config keys** for
  `parameter_changes` from the Modification Coordinator.
- Add `group_id` (session_id) into each JSONL flow record so it can be
  correlated to SDK trace spans.

### Changes Applied In This Document
- Added SDK-aligned findings and recommendations (this addendum).
- Documented required `trace()` + `group_id` usage per SDK docs.
- Documented doc-query detection update for the new bundled tool.

---

## Verification Commands

```bash
# View communication flow breakdowns
cat traces/communication_flow.jsonl | jq 'select(.success == false)'

# Count breakdown rate
cat traces/communication_flow.jsonl | jq -s '[.[] | select(.event == "flow_complete")] | {total: length, failures: [.[] | select(.success == false)] | length}'

# Test API fixer on a script
cd agents/blender-vfx-orchestrator
source venv/bin/activate
python tools/blender_api_fixer.py /path/to/script.py --check-only

# Run diagnostic test
python test_diagnostic_analysis.py --effect fire --iterations 2
```

---

## Appendix: Full Trace Sample

From `traces/communication_flow.jsonl`:

```json
{
  "event": "flow_complete",
  "timestamp": "2026-01-24T16:15:28.926188",
  "coordinator": "ModificationCoordinator",
  "decision": "modify_params",
  "parameters": {
    "domain_resolution": 64,
    "domain_padding_margin": 0.2,
    "frame_start": 1,
    "frame_end": 5,
    "render_resolution_percent": 30,
    "sampling_substeps": 0,
    "bake_simulation": false,
    "export_manta_script": false,
    "use_gpu_for_cycles": false,
    "skip_final_render": true
  },
  "reasoning": "The failure is an execution timeout (script likely spending most time baking and/or enumerating GPU devices). This is not a fundamental technique problem, so we should first modify parameters to make",
  "modify_result": {
    "changes_made": [],
    "success": false,
    "error": ""
  },
  "success": false
}
```

**Analysis:** Coordinator correctly identified timeout issue and suggested reducing resolution/frames. But parameter names (`domain_resolution`, `frame_end`) don't match Config class variables, so zero changes were applied.

---

## Related Documentation

- `docs/PATTERN_REPETITION_ANALYSIS_2026-01-24.md` - Initial pattern analysis
- `docs/WORKFLOW_AUDIT_ROOT_CAUSE_2026-01-24.md` - Detailed root cause
- `docs/VECTOR_STORE_REVAMP_SPEC_2026-01-24.md` - Vector store improvements (in progress)
- `docs/VECTOR_STORE_IMPLEMENTATION_PLAN_2026-01-24.md` - Implementation plan (in progress)
