# Workflow Audit: Root Cause Analysis

**Date:** 2026-01-24
**Status:** ROOT CAUSE IDENTIFIED
**Severity:** Critical - Core pipeline ineffective for API fixes

---

## Executive Summary

The **communication breakdown** between Coordinators and the modification system is the primary issue, NOT hardcoded data. The Modification Coordinator correctly identifies problems and proposes fixes, but `_modify_script_impl` can only modify **Config class parameters**, not arbitrary Python code patterns.

---

## Root Cause: Parameter Modifier vs Code Fixer Mismatch

### The Problem

```
Coordinator Output: "Replace ColorRamp.elements.clear() with while loop"
                           ↓
                    _modify_script_impl(modifications={...})
                           ↓
                    ONLY looks for: class Config:
                                      PARAM = value
                           ↓
                    ❌ CANNOT fix arbitrary Python code patterns
```

### Evidence from orchestrator.py (lines 1773-1794)

```python
# Coordinator provides fix
if mod_decision.action == 'modify_params' and mod_decision.parameter_changes:
    modify_result_json = _modify_script_impl(
        script_path=previous_script.script_path,
        modifications=mod_decision.parameter_changes,  # e.g., {"ColorRamp": "use while loop"}
        output_name=output_name
    )
```

### What _modify_script_impl Actually Does (lines 774-796)

```python
# ONLY modifies Config class parameters
config_class_pattern = r"(class Config:.*?)"
param_pattern = rf"(\s+{param_upper}\s*=\s*)([^\n]+)"  # Matches: TURBULENCE = 3.5

# Cannot fix:
# - ColorRamp.elements.clear()  <- arbitrary method call
# - bpy.ops.fluid.bake_all()    <- operator call
# - Any code outside Config class
```

### Why This Causes Repetition

1. **Iteration 1:** Script fails with `ColorRamp.elements.clear()` error
2. **Coordinator correctly diagnoses:** "Use while loop instead"
3. **`_modify_script_impl` receives:** `{"clear_method": "while_loop"}` or similar
4. **Regex finds nothing:** No `CLEAR_METHOD = ...` in Config class
5. **Zero changes made:** Script unchanged
6. **Iteration 2:** Same script, same error, same diagnosis
7. **Repeat forever:** Until max_iterations hit

---

## Traces ARE Being Saved

Traces go to: **https://platform.openai.com/traces**

The orchestrator uses `trace()` correctly:

```python
# orchestrator.py:1244
with trace(f"VFX Asset: {request.asset_name}"):

# orchestrator.py:1307
with trace(f"VFX Pipeline: {request.asset_name}", group_id=session.session_id):

# orchestrator.py:2495
with trace(f"VFX Resume: {session.session_id}", group_id=session.session_id):
```

**Action:** Check your OpenAI dashboard at platform.openai.com/traces to see all historical runs.

---

## Fix Options

### Option A: Regenerate Script (Recommended)

When the Coordinator detects an **API fix** (not a parameter tweak), bypass modification and regenerate:

```python
# In orchestrator.py, after Coordinator decision
if mod_decision.fix_type == "api_pattern":
    # Don't modify - regenerate with correct pattern
    script = await generate_script_with_constraints(
        technique=current_technique,
        avoid_patterns=["ColorRamp.elements.clear()"],
        required_patterns=[mod_decision.fix_code]
    )
```

### Option B: Code Pattern Replacement

Extend `_modify_script_impl` to handle code pattern replacements:

```python
# New parameter type
modifications = {
    # Parameter changes (existing)
    "param:TURBULENCE": 3.5,

    # Code pattern replacements (new)
    "replace:ColorRamp.elements.clear()":
        "while len(ramp.color_ramp.elements) > 1:\n    ramp.color_ramp.elements.remove(ramp.color_ramp.elements[0])"
}
```

### Option C: Validation Gate Before Execution

Add a pre-execution validation that catches known-bad patterns:

```python
KNOWN_BAD_PATTERNS = {
    "ColorRamp.elements.clear()": "# Remove all but one element:\nwhile len(ramp.color_ramp.elements) > 1:\n    ramp.color_ramp.elements.remove(ramp.color_ramp.elements[0])",
    # Add more as discovered
}

def validate_and_fix_script(script_path):
    content = Path(script_path).read_text()
    for bad, fix in KNOWN_BAD_PATTERNS.items():
        if bad in content:
            content = content.replace(bad, fix)
    Path(script_path).write_text(content)
```

---

## Communication Flow Audit

### Current Flow (Broken)

```
Coordinator Decision
       │
       ├── action: "modify_params"
       ├── parameter_changes: {"fix_clear": "while loop"}
       │
       ▼
_modify_script_impl(modifications={"fix_clear": "while loop"})
       │
       ├── Searches for: class Config:
       │                   FIX_CLEAR = ...
       ├── Finds: NOTHING
       │
       ▼
Zero changes made → Same script → Same error → Loop forever
```

### Required Flow (Fixed)

```
Coordinator Decision
       │
       ├── action: "modify_params" | "fix_api_pattern" | "regenerate"
       ├── parameter_changes: {...}  (for params)
       ├── code_replacements: {...}  (for API fixes)
       │
       ▼
IF action == "fix_api_pattern":
    apply_code_replacements()
ELIF action == "modify_params":
    _modify_script_impl(modifications=parameter_changes)
ELIF action == "regenerate":
    generate_new_script(avoid=current_technique)
```

---

## SDK Tracing for Local Analysis

To export traces locally (not just to OpenAI dashboard):

```python
from agents.tracing import add_trace_processor, TracingProcessor
import json
from pathlib import Path

class LocalTraceExporter(TracingProcessor):
    def __init__(self, output_dir: str = "traces"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.current_trace = []

    def on_trace_start(self, trace):
        self.current_trace = []

    def on_span_start(self, span):
        self.current_trace.append({
            "event": "span_start",
            "name": span.name,
            "timestamp": span.start_time
        })

    def on_span_end(self, span):
        self.current_trace.append({
            "event": "span_end",
            "name": span.name,
            "duration_ms": span.duration_ms,
            "output": str(span.output)[:500] if span.output else None
        })

    def on_trace_end(self, trace):
        trace_file = self.output_dir / f"trace_{trace.trace_id}.json"
        trace_file.write_text(json.dumps(self.current_trace, indent=2))
        print(f"[Trace] Saved to {trace_file}")

# Add to orchestrator.py startup
add_trace_processor(LocalTraceExporter("./traces"))
```

---

## Recommended Next Steps

1. **Immediate:** Check platform.openai.com/traces for historical runs
2. **Short-term:** Implement Option C (validation gate) as quick fix
3. **Medium-term:** Implement Option A (regenerate on API fix)
4. **Long-term:** Refactor Coordinator outputs to distinguish param changes from code fixes

---

## Files to Modify

| File | Change |
|------|--------|
| `orchestrator.py` | Add `fix_type` handling after Coordinator decision |
| `tools/script_generator_tools.py` | Add code pattern replacement support |
| `specialized_agents/modification_coordinator.py` | Output `fix_type` field |
| `models/coordinator_outputs.py` | Add `fix_type` to `ModificationDecision` |

---

## Verification Queries

Check OpenAI dashboard for these patterns:
- Same tool call repeated 3+ times with identical parameters
- Coordinator outputs that include "replace" or "fix" language
- modify_script calls with zero `changes_made`

---

## Appendix: Related Documentation

- `docs/PATTERN_REPETITION_ANALYSIS_2026-01-24.md` - Pattern repetition issues
- `docs/WORKFLOW_AUDIT_2026-01-23.md` - Previous audit findings
- `docs/TRACING_SOLUTIONS_2026-01-23.md` - Tracing configuration

