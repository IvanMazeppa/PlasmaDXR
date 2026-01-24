# Pattern Repetition Analysis & Fixes

**Date:** 2026-01-24
**Status:** Investigation + Partial Fix
**Severity:** High - Affects asset quality and variety

---

## Executive Summary

Multiple issues were identified causing repetitive, near-identical VFX asset generation despite different prompts and iterations. This document details the root causes, implemented fixes, and ongoing investigation strategies.

---

## Issue 1: Output Files Landing in /tmp (FIXED)

### Symptom
Generated renders and .blend files were being saved to `/tmp/` instead of organized asset folders like `build/vdb_output/{asset_name}/`.

### Root Cause
The `tools/dynamic_instructions.py` file contained hardcoded `/tmp/` paths in the code examples that agents follow:

```python
# BEFORE (problematic):
scene.render.filepath = "/tmp/sun_render.png"  # Hardcoded /tmp
cache_dir = "/tmp/mantaflow_cache"  # Hardcoded /tmp
```

When the LLM generates Blender scripts, it follows these examples closely, resulting in all outputs going to `/tmp/` regardless of the asset name.

### Fix Applied
Updated `tools/dynamic_instructions.py` to use proper asset folder structure:

```python
# AFTER (fixed):
ASSET_NAME = "explosion_v1"  # From request
OUTPUT_DIR = f"/home/maz3ppa/projects/PlasmaDXR/build/vdb_output/{ASSET_NAME}"
CACHE_DIR = f"{OUTPUT_DIR}/cache"
RENDER_PATH = f"{OUTPUT_DIR}/{ASSET_NAME}.png"
BLEND_PATH = f"{OUTPUT_DIR}/{ASSET_NAME}.blend"
```

### Expected Behavior After Fix
- Each asset gets its own folder: `build/vdb_output/{asset_name}/`
- Subfolder for Mantaflow cache: `build/vdb_output/{asset_name}/cache/`
- Render and .blend in asset root: `build/vdb_output/{asset_name}/{asset_name}.png`

---

## Issue 2: record_baseline Bug (FIXED)

### Symptom
Error on iteration 2+: "No baseline recorded for this session"

### Root Cause
The `previous_score` variable was being updated **before** `record_experiment_baseline()` was called:

```python
# BEFORE (buggy order):
previous_score = quality.overall_score  # Line 2130 - updates to CURRENT score
# ... later ...
baseline_scores = json.dumps({"overall": previous_score})  # Uses CURRENT, not PREVIOUS
record_experiment_baseline(...)  # Records wrong baseline
```

The baseline should represent the state **before** the current experiment, but it was recording the **current** state.

### Fix Applied
Capture baseline snapshot **before** quality evaluation updates the variables:

```python
# AFTER (correct order):
# Capture BEFORE quality evaluation
baseline_score_snapshot = previous_score
baseline_params_snapshot = dict(previous_params)
baseline_render_snapshot = session.final_render_path

# ... quality evaluation runs, updates previous_score ...

# Use snapshots for baseline
baseline_scores_json = json.dumps({"overall": baseline_score_snapshot})
record_experiment_baseline(
    params=baseline_params_snapshot,
    scores=baseline_scores_json,
    render_path=baseline_render_snapshot or execution.render_path or "",
    ...
)
```

### Files Modified
- `orchestrator.py`: Lines ~2080-2210

---

## Issue 3: Pattern Repetition in Generated Assets (UNDER INVESTIGATION)

### Symptom
Generated .blend files show alarming structural similarity across different sessions:
- Identical object names (e.g., "FireDomain", "FlowEmitter")
- Same shapes and modifiers
- Same technique approach (always Mantaflow)
- Only small parameter variations

This occurs even when:
- Different AI systems are used (Claude, GPT agents)
- Different prompts are given
- Different sessions are run

### Potential Root Causes

#### A. Soft Template Effect (Confirmed)
The `dynamic_instructions.py` contains detailed code patterns that serve as implicit templates:

```python
## MANTAFLOW CLI USAGE (CRITICAL - READ FIRST)
### Complete Mantaflow + Cycles Setup Pattern:
def setup_mantaflow_scene():
    # ... 40+ lines of specific implementation ...
```

Even though explicit templates are disabled, these code examples in the instructions act as "soft templates" that LLMs follow closely.

**Evidence:** Scripts across sessions share structural patterns matching the examples in dynamic_instructions.py.

#### B. Limited Technique Variety (Confirmed)
All fire/explosion techniques in `technique_catalog.py` are Mantaflow-based:
- `rising_mushroom` - Mantaflow with high turbulence
- `ground_hugger` - Mantaflow with low buoyancy
- `aerial_burst` - Mantaflow with burst emission
- etc.

There are **no alternative approaches** like:
- Shader-based procedural fire
- Geometry Nodes particle fire
- Volume shader fire (no simulation)

**Evidence:** Every generated fire asset uses Fluid modifier with domain_settings, regardless of the "technique" selected.

#### C. Training Data Reversion (Suspected)
LLMs may revert to patterns from their training data when generating Blender Python code. Fire VFX tutorials commonly use:
- Mantaflow quick smoke/fire setup
- Standard object names ("Smoke Domain", "Flow Emitter")
- Similar parameter ranges

**Evidence:** Similarity persists across different AI models (Claude, GPT), suggesting a common training source.

#### D. Legacy Code Paths (Possible)
Old template-based code may still be influencing generation through:
- Cached imports
- Stale technique recommendations
- Deprecated function calls that haven't been fully removed

**Status:** Requires investigation with diagnostic hooks.

### Diagnostic Tools Created

#### DiagnosticHooks (`hooks/diagnostic_hooks.py`)
New SDK RunHooks implementation that tracks:

```python
from hooks import DiagnosticHooks

hooks = DiagnosticHooks(
    log_file="agent_trace.jsonl",
    verbose=True,
    track_patterns=True
)

result = await Runner.run(agent, prompt, hooks=hooks)
hooks.print_summary()
```

**Captures:**
| Pattern Type | Detection Method | Threshold |
|-------------|------------------|-----------|
| Repeated Prompts | MD5 fingerprint | 2+ occurrences |
| Repeated Tool Calls | Tool name + args hash | 3+ occurrences |
| Repeated Outputs | Output string hash | 2+ occurrences |

**Output:**
```
⚠️ REPEATED PROMPTS (3):
  [4x] Generate a Blender Python script for fire VFX...
  [2x] Evaluate the render quality...

⚠️ REPEATED TOOL CALLS (2):
  [5x] write_script
  [3x] semantic_search_blender_docs

⚠️ REPEATED OUTPUTS (1):
  [2x] ScriptOutput: def setup_mantaflow_scene()...
```

#### Diagnostic Test Script (`test_diagnostic_analysis.py`)
```bash
python test_diagnostic_analysis.py --effect fire --iterations 2
```

Runs asset generation with diagnostic hooks and produces analysis report.

### Planned Fixes

#### Short-term: Reduce Soft Template Influence
1. **Minimize code examples** in dynamic_instructions.py
2. **Add variety prompts** encouraging different approaches
3. **Randomize instruction ordering** to reduce pattern lock-in

#### Medium-term: Add Alternative Techniques
1. **Shader-based fire** - Procedural noise + emission shader
2. **Geometry Nodes fire** - Point cloud with velocity-based emission
3. **Hybrid approaches** - Combine simulation with procedural elements

#### Long-term: Training Data Mitigation
1. **Explicit anti-pattern instructions** - "Do NOT use standard tutorial approaches"
2. **Reference image guidance** - Force deviation based on unique reference
3. **Technique rotation** - Never use same technique twice in session

---

## Issue 4: Model Configuration (FIXED)

### Symptom
Incorrect model names and parameter incompatibility errors.

### Root Cause (from OpenAI Platform docs via context7)
GPT-5 model family has specific parameter restrictions:
- `gpt-5.2` / `gpt-5.1`: Support `temperature`/`top_p`/`logprobs` **ONLY** with `reasoning_effort=none`
- `gpt-5` / `gpt-5-mini` / `gpt-5-nano`: **NO** temperature support at all

### Fix Applied
Updated `config/presets.yaml` and `config/agent_config.py`:

| Preset | Model | Parameters |
|--------|-------|------------|
| `quick_test` | gpt-5-mini | reasoning=low, text.verbosity=medium |
| `development` | gpt-5-mini | reasoning=medium, text.verbosity=medium |
| `production` | gpt-5.2 | reasoning=medium, text.verbosity=high |
| `budget_saver` | gpt-5-nano | reasoning=low, text.verbosity=low |
| `debug` | gpt-5-mini | reasoning=low, text.verbosity=high |
| `reasoning` | gpt-5.2 | reasoning=high, text.verbosity=medium |
| `creative` | gpt-5.2 | reasoning=none, **temperature=0.7** |

### Alternative Parameters for gpt-5-mini/nano
Instead of `temperature`, use:
- `text: { verbosity: "low" | "medium" | "high" }` - Controls output verbosity
- `max_output_tokens` - Controls output length

---

## Files Modified

| File | Changes |
|------|---------|
| `orchestrator.py` | Fixed record_baseline timing bug |
| `tools/dynamic_instructions.py` | Changed /tmp to asset folder structure |
| `config/presets.yaml` | Updated models to gpt-5.1-mini/nano + gpt-5.2 |
| `hooks/__init__.py` | Export diagnostic hooks |
| `hooks/diagnostic_hooks.py` | NEW - Pattern detection hooks |
| `test_diagnostic_analysis.py` | NEW - Diagnostic test script |

---

## Verification Steps

### 1. Output Path Fix
```bash
# Run quick test
python test_quick_e2e.py --preset quick_test --iterations 1 --effect fire

# Check output location (should NOT be /tmp)
ls -la build/vdb_output/
```

### 2. record_baseline Fix
```bash
# Run 2+ iterations
python test_quick_e2e.py --preset quick_test --iterations 3 --effect fire

# Check logs for "Baseline recorded: score=X.X" messages
# Should NOT see "No baseline recorded" errors
```

### 3. Pattern Analysis
```bash
# Run diagnostic analysis
python test_diagnostic_analysis.py --effect fire --iterations 2

# Review trace file
cat diagnostic_trace_*.jsonl | jq '.type' | sort | uniq -c
```

---

## Open Questions

1. **Why does similarity persist across different AI models?**
   - Hypothesis: Common training data from Blender tutorials
   - Test: Run with explicit "avoid tutorial patterns" prompt

2. **Are there hidden code paths triggering templates?**
   - Hypothesis: Legacy imports or cached modules
   - Test: Full module reload + diagnostic trace

3. **Would reference images reduce repetition?**
   - Hypothesis: Unique reference forces deviation
   - Test: Run with NASA SDO reference vs. no reference

---

## Related Documentation

- `docs/ARCHITECTURE_OPTIMIZATION_PLAN_2026-01-22.md` - Phase implementation status
- `docs/SELF_LEARNING_ARCHITECTURE.md` - Learning system that may amplify patterns
- `docs/AGENTS_SDK_INTEGRATION.md` - SDK patterns and hooks usage

---

## Revision History

| Date | Author | Changes |
|------|--------|---------|
| 2026-01-24 | Claude/Ben | Initial documentation of issues and fixes |
