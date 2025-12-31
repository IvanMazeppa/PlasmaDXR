# Multi-Agent VFX Pipeline Optimization Analysis

**Date:** 2025-12-31
**Version:** 1.0
**Pipeline Version:** 0.2.0 (Experimental)
**Analyst:** Claude Code (Opus 4.5)

---

## Executive Summary

Analysis of the Blender 5 autonomous VDB asset generation pipeline revealed **3 critical problems** that prevent the iteration loop from converging:

1. **ML Metrics Don't Work for VFX (P1)** - LPIPS and CLIP provide no meaningful gradient signal for volumetric effects
2. **No Intelligent Orchestrator (P1)** - iteration-controller returns instructions but doesn't execute them
3. **Agents Don't Share Knowledge (P2)** - experiment-tracker has good tools but other agents never query them

---

## Pipeline Architecture

### Current Components

| MCP Server | Purpose | Status |
|------------|---------|--------|
| script-generator | Creates Blender Python scripts from descriptions | Working |
| blender-executor | Runs Blender CLI, captures output | Working |
| asset-evaluator | LPIPS/CLIP quality assessment | **Broken for VFX** |
| iteration-controller | Orchestrates iteration loop | **Doesn't execute** |
| experiment-tracker | Tracks experiments, learns patterns | **Unused** |
| context7 | External documentation lookup | Working |

### Data Flow

```
Description → script-generator → Blender Script
                                      ↓
                              blender-executor
                                      ↓
                            VDB Files + Renders
                                      ↓
                              asset-evaluator
                                      ↓
                         LPIPS/CLIP Scores (BROKEN)
                                      ↓
                          iteration-controller
                                      ↓
                         "Instructions" (not execution)
```

---

## Critical Problem #1: ML Metrics Don't Work for VFX

### Evidence

From sun surface iteration log (v7-v11):

| Version | LPIPS | CLIP | Visible Quality |
|---------|-------|------|-----------------|
| v7 | 0.738 | 0.671 | Dim, lacks corona |
| v8 | 0.759 | 0.659 | Better color |
| v9 | 0.752 | 0.663 | Good structure |
| v10 | 0.745 | 0.667 | Best so far |
| v11 | 0.753 | 0.661 | Added granulation |

**Problem:** LPIPS scores range 0.738-0.759 (essentially noise) despite visible quality differences. CLIP plateaus at 0.659-0.671 regardless of improvements.

### Root Cause

**LPIPS** (Learned Perceptual Image Patch Similarity):
- Trained on photorealistic image pairs from BAPPS dataset
- Measures "perceptual distance" - lower = more similar
- Designed for: photo editing, super-resolution, style transfer
- **Fails for VFX because:** No reference image exists, volumetric effects don't have "correct" appearance

**CLIP** (Contrastive Language-Image Pre-training):
- Trained on 400M image-text pairs from internet
- Measures semantic alignment between image and text
- Designed for: image classification, zero-shot recognition
- **Fails for VFX because:** Plateaus once basic concept matches ("this is a sun"), provides no quality gradient

### Threshold Mismatch

Current thresholds (asset-evaluator):
- LPIPS < 0.35 (pass) - **VFX consistently scores 0.7-0.8**
- CLIP > 0.60 (pass) - **VFX plateaus at 0.65-0.67**

The thresholds assume photorealistic comparison. VFX will never pass LPIPS < 0.35.

---

## Critical Problem #2: Orchestrator Doesn't Execute

### Current Behavior

```python
# iteration-controller/server.py:238-262
async def run_iteration(...):
    result = IterationResult(
        recommendations=[
            "1. Use script-generator to create/modify script...",
            "2. Use blender-executor to run the script...",
            "3. Use asset-evaluator evaluate_render..."
        ]
    )
    return json.dumps(asdict(result))  # Returns TEXT, not execution
```

The iteration-controller returns **instructions as strings** rather than calling the other MCP tools. This forces Claude Code to manually orchestrate each step.

### Consequences

1. **No automation** - Human must copy-paste tool calls
2. **Context limits** - Claude Code runs out of context mid-iteration
3. **Lost state** - When context resets, iteration progress is lost
4. **Inconsistent execution** - Different sessions may execute differently

---

## Critical Problem #3: Agents Don't Share Knowledge

### Unused Capabilities

experiment-tracker provides excellent tools:
- `get_warnings_before_change()` - Warns about known failure patterns
- `suggest_experiments()` - Recommends parameters based on past successes
- `query_knowledge_base()` - Retrieves learned patterns

**But:** script-generator never queries these before generating. Each iteration starts from scratch.

### Example Flow (Broken)

```
script-generator.generate_script("sun surface")
  → Generates with default parameters
  → Blender fails (flame_max_temp too high)
  → experiment-tracker records failure
  → NEXT ITERATION:
  → script-generator.generate_script("sun surface")
  → SAME default parameters (didn't query tracker!)
  → Same failure
```

---

## Proposed Solutions

### Solution 1: VFX-Specific Quality Evaluation

Replace LPIPS/CLIP with feature-based diagnostics:

```python
def extract_vfx_diagnostics(image_path: str) -> dict:
    """Extract VFX-specific quality signals."""
    img = Image.open(image_path)
    arr = np.array(img)

    return {
        "brightness": {
            "mean": float(np.mean(arr)),
            "max": float(np.max(arr)),
            "dynamic_range": float(np.max(arr) - np.min(arr))
        },
        "color_presence": {
            "warm_ratio": float(np.sum(arr[..., 0]) / max(np.sum(arr[..., 2]), 1)),
            "has_orange": float(np.mean(arr[..., 0] > arr[..., 2]))
        },
        "coverage": {
            "non_black_pixels": float(np.sum(np.any(arr > 10, axis=-1)) / arr[..., 0].size),
            "bright_pixels": float(np.sum(np.any(arr > 200, axis=-1)) / arr[..., 0].size)
        },
        "structure": {
            "edge_density": _compute_edge_density(arr),
            "has_gradients": _detect_gradients(arr)
        }
    }
```

### Solution 2: Intelligent Orchestrator Agent

Create a proper orchestrator with:
- State machine (GENERATE → EXECUTE → EVALUATE → DECIDE → loop)
- Decision engine (diagnose issues → suggest fixes)
- Session persistence (survives context limits)
- MCP client capabilities (calls other agents)

### Solution 3: Agent Communication Protocol

Require agents to query experiment-tracker before acting:

```python
async def generate_script_with_knowledge(...):
    # FIRST: Query knowledge base
    warnings = await experiment_tracker.get_warnings_before_change(...)
    past = await experiment_tracker.query_knowledge_base(effect_type)

    # THEN: Generate with context
    return generate_script(
        avoid_patterns=warnings.get("failed_approaches", []),
        successful_patterns=past.get("successful_params", {})
    )
```

---

## Implementation Priority

| Fix | Impact | Effort | Priority |
|-----|--------|--------|----------|
| VFX diagnostics in asset-evaluator | HIGH | LOW | **P1** |
| Iteration decision engine | HIGH | MEDIUM | **P1** |
| Session state persistence | HIGH | LOW | **P1** |
| Iteration comparison tool | MEDIUM | LOW | **P2** |
| experiment-tracker integration | MEDIUM | MEDIUM | **P2** |
| Full orchestrator agent | HIGH | HIGH | **P3** |

---

## Recommended Implementation Order

### Phase 1: Quick Wins (This Week)
1. Add `extract_vfx_diagnostics()` to asset-evaluator
2. Add `compute_vfx_quality_score()` composite scoring
3. Add session state persistence tool

### Phase 2: Architecture (Next Week)
1. Implement iteration decision engine
2. Add iteration comparison tool (compare A vs B)
3. Integrate experiment-tracker queries into script-generator

### Phase 3: Long-term
1. Design and implement proper orchestrator agent
2. Add MCP client capabilities for agent-to-agent calls
3. Implement learning loop (successes inform future generations)

---

## Files Modified/Created

This analysis recommends changes to:

- `agents/asset-evaluator/server.py` - Add VFX diagnostics
- `agents/iteration-controller/server.py` - Add decision engine
- `agents/script-generator/server.py` - Query experiment-tracker
- NEW: `agents/session-state/server.py` - State persistence
- NEW: `agents/orchestrator/server.py` - Proper orchestration (Phase 3)

---

## Conclusion

The pipeline architecture is fundamentally sound. The MCP server pattern, Blender execution, and knowledge tracking infrastructure all work. The critical gap is that **LPIPS/CLIP don't provide useful signal for VFX**, causing iterations to wander without direction.

The fix is **architectural, not incremental**: Replace perceptual similarity metrics with VFX-specific feature extraction that provides actual gradient signal for convergence.

---

**Document Location:** `docs/MULTI_AGENT_PIPELINE_OPTIMIZATION_ANALYSIS.md`
**Related:** `docs/BLENDER_VFX_PIPELINE_STATUS.md`
