# ML Vision System Limitations and Ground Truth Proposal

**Date:** 2025-12-31
**Context:** Sun surface rendering pipeline evaluation
**Author:** Claude Code session with Ben

---

## Executive Summary

During end-to-end testing of the sun surface rendering pipeline, a critical limitation was discovered: **synthetic renders scored higher (94/100) than real solar footage (84/100)**. This reveals fundamental issues with threshold-based VFX evaluation metrics and presents an opportunity for a ground-truth ML/NN approach.

---

## The Problem: Threshold-Based Scoring vs Reality

### What Happened

| Source | Score | warm_ratio | edge_density | brightness |
|--------|-------|------------|--------------|------------|
| **Real Sun Footage** (frame_00420) | 84 | **18.88** | 0.0387 | 56 |
| **v8 Procedural Render** | 94 | 2.66 | 0.0268 | 126 |

The real sun footage:
- Is **7× more orange** (warm_ratio 18.88 vs 2.66)
- Has **44% more visible structure** (edge_density 0.0387 vs 0.0268)
- Yet scores **10 points lower**

### Why This Happens

The current `evaluate_vfx_quality` system uses **threshold-based scoring**:

```python
# Simplified logic from current system
if warm_ratio > 1.5:
    color_score = 25  # Full points
else:
    color_score = 0   # Fail

if brightness > 100:
    brightness_score = 25  # Full points
else:
    brightness_score = 15  # Partial (penalized)
```

**Problems:**

1. **Binary thresholds** - Once you pass 1.5 warm_ratio, there's no reward for being more realistic (18.88 vs 2.66 both get 25 points)

2. **Arbitrary targets** - The brightness target of 100 is arbitrary. Real solar footage is often intentionally darker to preserve HDR detail in prominences and filaments. The metric punishes professional photography choices.

3. **No ground truth comparison** - The system doesn't compare against real footage, it compares against hardcoded thresholds designed to catch obvious VFX failures.

4. **Designed for iteration, not realism** - The system answers "is this obviously broken?" not "does this look like a real sun?"

---

## Current System Value: Iteration Feedback

Despite its limitations, the current system provides **significant value** during the creative iteration loop:

### What It Does Well

| Capability | Example |
|------------|---------|
| **Catch catastrophic failures** | "GRAYSCALE! warm_ratio = 1.0" caught color pipeline bugs in v2, v4, v6 |
| **Track improvement/regression** | Score progression: v1(68) → v3(75) → v7(85) → v8(94) |
| **Identify specific issues** | "NO STRUCTURE: edge density 0.006" directed texture improvements |
| **Fast feedback loop** | ~2 second evaluation vs minutes for full ML inference |
| **Actionable recommendations** | "Increase turbulence/vorticity" guided parameter changes |

### Iteration History Showing System Value

```
v1 (fluid sim):     Score 65  → "NO STRUCTURE, TOO DARK"
v3 (procedural):    Score 75  → "Color back, needs structure"
v6 (scalar bug):    Score 75  → "GRAYSCALE! warm_ratio 1.0" ← CAUGHT BUG
v7 (fixed):         Score 85  → "warm_ratio 2.12 ✓, needs structure"
v8 (crackle):       Score 94  → "PASSED" (but not realistic!)
```

The system successfully guided us from broken renders to acceptable VFX. It just can't tell us how close we are to photorealism.

---

## Proposed Solution: Dual-System Architecture

### System 1: Threshold-Based (Current)
**Purpose:** Fast iteration feedback, catch failures, track progress
**Speed:** ~2 seconds
**Use case:** Every render during development

### System 2: Ground Truth ML/NN
**Purpose:** Measure realism against actual solar footage
**Speed:** ~30-60 seconds (acceptable for milestone checks)
**Use case:** Before finalizing assets, quality gates

### Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    RENDER ITERATION LOOP                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Generate Render ──► System 1 (Threshold) ──► Score < 80?     │
│         │                    │                      │           │
│         │                    │                      ▼           │
│         │                    │              Fix Issues          │
│         │                    │              (color, structure)  │
│         │                    │                      │           │
│         │                    ▼                      │           │
│         │              Score ≥ 80?                  │           │
│         │                    │                      │           │
│         │                    ▼                      │           │
│         │         System 2 (Ground Truth ML)        │           │
│         │                    │                      │           │
│         │                    ▼                      │           │
│         │         Similarity < 0.7? ───────────────┘           │
│         │                    │                                  │
│         │                    ▼                                  │
│         │         Similarity ≥ 0.7?                             │
│         │                    │                                  │
│         │                    ▼                                  │
│         │              ✅ ASSET READY                           │
│         │                                                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Ground Truth Dataset: Solar Activity Footage

### Available Reference Data

**Source:** `Eruptions_20241008_Activity_2048p30.mp4`
**Duration:** 14 hours of solar rotation (139.87 seconds timelapse)
**Resolution:** 2048×2048
**Frame rate:** 30fps original, extracted every 5th frame

**Extracted Frames:**
```
assets/reference_images/star/Eruptions_20241008_Activity_2048p30/
├── frame_00001.jpg  (1.6MB, 2048×2048)
├── frame_00002.jpg
├── ...
└── frame_00840.jpg

Total: 840 frames, 1.3GB
```

### Why This Dataset Is Valuable

1. **Real solar phenomena captured:**
   - Eruptions and prominences
   - Filaments appearing/disappearing
   - Granulation texture
   - Limb darkening
   - 14-hour rotation showing surface evolution

2. **High dynamic range:** Professional solar imaging preserves detail in prominences (intentionally darker overall)

3. **Consistent conditions:** Same telescope, same day, consistent processing

4. **Temporal coverage:** 840 frames capture varied activity states - quiet regions, active regions, eruptions

---

## ML/NN Ground Truth System Requirements

### Input
- Rendered image (1024×1024 or 2048×2048)
- Effect type ("sun", "prominence", "flare")

### Ground Truth Comparison
- Compare against appropriate reference frames from the 840-frame dataset
- Use perceptual similarity (not pixel-perfect matching)
- Account for acceptable variations (rotation angle, specific feature positions)

### Desired Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Perceptual Similarity** | LPIPS or learned metric against reference set | < 0.3 |
| **Color Distribution Match** | Histogram/distribution comparison | > 0.8 |
| **Texture Realism** | Granulation pattern similarity | > 0.7 |
| **Feature Presence** | Detects expected solar features | Pass/Fail |
| **Overall Realism Score** | Weighted combination | > 0.75 |

### Key Differences from Current System

| Current System | Ground Truth ML System |
|----------------|------------------------|
| Fixed thresholds | Learned from real data |
| Binary pass/fail per metric | Continuous similarity scores |
| No reference comparison | Compares against 840 real frames |
| Rewards "not obviously broken" | Rewards "looks like real sun" |
| Same score for 2.66 and 18.88 warm_ratio | Would correctly rank 18.88 as more realistic |

---

## Implementation Suggestions for ML/NN Plugin

### Option A: Fine-tuned CLIP
- Fine-tune CLIP on solar imagery dataset
- Create embeddings for all 840 reference frames
- Compare render embeddings against reference cluster
- **Pro:** Leverages existing architecture
- **Con:** May not capture fine texture details

### Option B: Custom CNN Discriminator
- Train discriminator: "real solar footage" vs "synthetic render"
- Output: probability that image is real
- **Pro:** Directly learns what makes footage look real
- **Con:** Requires training infrastructure

### Option C: Perceptual Feature Matching
- Extract features from pretrained network (VGG, ResNet)
- Compare feature distributions against reference set
- Similar to LPIPS but with domain-specific tuning
- **Pro:** No training required, interpretable
- **Con:** May miss domain-specific features

### Option D: Hybrid Approach (Recommended)
- Use CLIP for semantic similarity ("is this a sun?")
- Use LPIPS for perceptual similarity ("does texture match?")
- Use custom color histogram matching ("does color distribution match?")
- Weighted combination with learned weights
- **Pro:** Combines strengths of multiple approaches
- **Con:** More complex integration

---

## Proposed API for Ground Truth System

```python
# Evaluate single render against ground truth
result = evaluate_solar_realism(
    render_path="build/vdb_output/sun_v8/render_0060.png",
    effect_type="sun_surface",  # or "prominence", "flare", "full_disk"
    reference_set="Eruptions_20241008_Activity_2048p30",
    metrics=["perceptual", "color", "texture", "features"]
)

# Result structure
{
    "overall_realism": 0.72,
    "passed": True,  # if overall_realism >= 0.7
    "metrics": {
        "perceptual_similarity": 0.68,
        "color_distribution_match": 0.85,
        "texture_similarity": 0.65,
        "feature_detection": {
            "granulation": True,
            "limb_darkening": True,
            "prominences": False  # not present in this render
        }
    },
    "closest_reference_frames": [
        "frame_00234.jpg",  # similarity: 0.74
        "frame_00567.jpg",  # similarity: 0.71
        "frame_00123.jpg"   # similarity: 0.69
    ],
    "recommendations": [
        "Color is close but render is less saturated than reference",
        "Granulation texture is smoother than real footage",
        "Consider adding more contrast in cell boundaries"
    ]
}
```

---

## Summary

### Current State
- Threshold-based system: Good for iteration, blind to realism
- Ground truth available: 840 frames of real solar footage (14 hours)
- Gap identified: Synthetic scores 94, real scores 84

### Proposed Solution
1. **Keep threshold system** for fast iteration feedback
2. **Add ground truth ML system** for realism measurement
3. **Dual-system workflow:** Iterate with System 1, validate with System 2

### Next Steps
1. Integrate ML/NN computer vision plugin
2. Create reference frame embeddings/features
3. Define realism scoring based on reference comparison
4. Establish quality gate thresholds
5. Test on existing renders (v1-v8, prominences v1-v2)

---

## Implementation Status (Updated 2025-12-31)

### Ground Truth Evaluation System: IMPLEMENTED ✅

The proposed dual-system architecture has been implemented:

**New MCP Tools (in asset-evaluator):**

1. `evaluate_ground_truth(image_path, effect_type, pass_threshold)`
   - PRIMARY evaluation tool for realism
   - Compares against 840 real solar reference frames
   - Returns 0-100 score with detailed analysis

2. `compare_to_reference_distribution(image_path, effect_type)`
   - Shows per-feature similarity to reference distributions
   - Useful for debugging why a render doesn't match

3. `evaluate_solar_features(image_path)`
   - Domain-specific metrics: granulation, limb darkening, prominences, corona
   - Color temperature estimation

4. `get_reference_statistics(effect_type, sample_size)`
   - Shows what "real" looks like (percentile distributions)
   - Cached after first computation

5. `extract_image_features(image_path)`
   - Low-level diagnostic: brightness, color, structure, radial profile

### Validation Results

| Image | OLD Score | NEW Score | warm_ratio | brightness |
|-------|-----------|-----------|------------|------------|
| Real Sun (frame_420) | 95 | 85.1 | 18.88 | 56.4 |
| Synthetic v7 | 90 | **34.0** | 2.12 | 131.6 |

The new system correctly:
- Ranks real footage higher than synthetic
- Identifies brightness mismatch (synthetic too bright)
- Identifies color warmth mismatch (synthetic not orange enough)
- Provides actionable recommendations

### Files Created

- `agents/asset-evaluator/ground_truth_evaluation.py` (766 lines)
  - Feature extraction from images
  - Reference statistics computation with caching
  - Distribution-based comparison (percentile matching)
  - Solar-specific feature detection
  - Bhattacharyya coefficient for histogram similarity

- `agents/asset-evaluator/server.py` (updated)
  - Added 5 new MCP tool wrappers for ground truth evaluation

---

## Appendix: Reference Frame Statistics

From analysis of `frame_00420.jpg` (mid-sequence):

```
Resolution:      2048×2048
File size:       ~1.5MB (JPEG qscale=2)
warm_ratio:      18.88 (extremely orange)
edge_density:    0.0387 (visible granulation)
brightness:      56 (intentionally dark for HDR detail)
dynamic_range:   255 (full range)
coverage:        73.6% (sun fills most of frame)
```

This represents the "ground truth" target that our renders should approach.
