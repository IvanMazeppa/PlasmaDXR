# ML Vision Evaluation Challenges in Self-Improving Asset Pipeline

## Executive Summary

The PlasmaDXR Self-Improving NanoVDB Asset Pipeline uses ML-based image evaluation (LPIPS + CLIP) to guide iterative improvement of procedurally generated volumetric effects. While the pipeline successfully generates and renders effects, **the ML evaluation stage is the current bottleneck**, preventing effective autonomous iteration.

This document outlines the specific challenges for review by ML specialists.

---

## Current Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Script Generator│────▶│ Blender Executor │────▶│ Asset Evaluator │
│ (MCP Server)    │     │ (MCP Server)     │     │ (MCP Server)    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
        │                       │                        │
        │                       │                        ▼
        │                       │               ┌─────────────────┐
        │                       │               │ LPIPS + CLIP    │
        │                       │               │ Evaluation      │
        │                       │               └────────┬────────┘
        │                       │                        │
        ▼                       ▼                        ▼
   .py script             .vdb + .png              Pass/Fail + Score
                           renders
```

### Current Evaluation Tools

| Tool | Model | Purpose | Output |
|------|-------|---------|--------|
| **LPIPS** | AlexNet/VGG | Perceptual similarity to reference | 0.0 (identical) to 1.0+ (different) |
| **CLIP** | ViT-B/32 | Semantic match to text description | 0.0 (unrelated) to 1.0 (perfect match) |

### Current Thresholds

- LPIPS: < 0.35 to pass (lower = more similar)
- CLIP: > 0.60 to pass (higher = better semantic match)

---

## Challenge 1: Domain Mismatch (Rendered vs. Real)

### Problem

LPIPS was trained on natural images. When comparing:
- **Rendered volumetric simulation** (Blender Cycles, transparent background)
- **Real explosion photograph** (camera capture, complex background)

The perceptual distance is inherently high (~0.84) even when the render looks good to humans.

### Observed Results

| Comparison | LPIPS Score | Human Assessment |
|------------|-------------|------------------|
| Render vs. Real Photo | 0.836 | "Looks like a fireball" |
| Render vs. Render (same sim) | ~0.05 | "Nearly identical" |
| Bad Render vs. Real Photo | 0.85 | "Doesn't look like explosion" |

The **discrimination range is too narrow** (0.83-0.85) to guide meaningful iteration.

### Questions for ML Specialists

1. Are there perceptual similarity models trained on synthetic/rendered images?
2. Should we fine-tune LPIPS on rendered volumetric datasets?
3. Is there a domain adaptation technique to bridge rendered↔real comparison?

---

## Challenge 2: CLIP Plateau Effect

### Problem

CLIP scores plateau quickly and don't provide gradient for fine-tuning:

| Iteration | CLIP Score | Visual Difference |
|-----------|------------|-------------------|
| 1 (faint smoke) | 0.58 | Barely visible |
| 2 (visible fire) | 0.62 | Clear improvement |
| 3 (bright flames) | 0.63 | Significant improvement |
| 4 (dramatic explosion) | 0.65 | Major improvement |

The score range (0.58 → 0.65) doesn't reflect the magnitude of visual improvement.

### Root Cause

CLIP's text-image alignment saturates once "explosion" features are present. It answers "is this an explosion?" but not "how good of an explosion is this?"

### Questions for ML Specialists

1. Can we use CLIP with more specific/graduated prompts? (e.g., "faint smoke" vs "dramatic fireball")
2. Are there aesthetic scoring models (LAION-Aesthetics, NIMA) suitable for VFX?
3. Could we train a small quality classifier on ranked examples?

---

## Challenge 3: No Actionable Feedback

### Problem

Current evaluation returns:
```json
{
  "passed": true,
  "lpips_score": 0.836,
  "clip_score": 0.650,
  "recommendations": ["Quality acceptable"]
}
```

This tells us nothing about **what to change** in the next iteration:
- Is the fire too dim or too bright?
- Is there enough smoke? Too much?
- Is the color palette correct?
- Is the shape/silhouette explosion-like?

### Desired Output

```json
{
  "passed": true,
  "scores": { "lpips": 0.836, "clip": 0.650 },
  "diagnostics": {
    "color_temperature": "too_cool",  // Needs more orange/yellow
    "density": "good",
    "shape": "needs_more_mushroom_cloud",
    "brightness": "slightly_dim",
    "smoke_ratio": "too_little_smoke"
  },
  "suggested_parameters": {
    "flame_max_temp": "increase 20%",
    "flame_smoke": "increase 50%",
    "vorticity": "increase for mushroom shape"
  }
}
```

### Questions for ML Specialists

1. Can vision-language models (GPT-4V, LLaVA, Gemini) provide structured VFX feedback?
2. Are there attribute classifiers for fire/smoke properties?
3. Could we use segmentation to analyze regions (fire core vs. smoke plume)?

---

## Challenge 4: Single-Path Optimization Trap

### Problem

The current iteration loop refines one script incrementally:

```
Script v1 → Eval → Tweak parameters → Script v2 → Eval → Tweak → ...
```

This gets stuck in **local optima**. If the initial approach is wrong (e.g., wrong emitter shape, wrong domain size), parameter tweaking won't fix it.

### Desired Behavior

```
                    ┌─── Approach A (sphere emitter) ───┐
                    │                                    │
Initial Concept ────┼─── Approach B (plane emitter)  ───┼─── Best Result
                    │                                    │
                    └─── Approach C (animated inflow) ──┘
```

Explore multiple fundamentally different approaches, then refine the best one.

### Questions for ML Specialists

1. How can evaluation guide exploration vs. exploitation?
2. Can we cluster "types" of results to encourage diversity?
3. Are there novelty-seeking metrics (like novelty search in evolutionary algorithms)?

---

## Challenge 5: Temporal Quality Not Evaluated

### Problem

Current evaluation looks at single frames. Animations have temporal properties:

- **Flickering/strobing** (bad)
- **Smooth motion** (good)
- **Temporal coherence** (fire should evolve naturally)
- **Proper timing** (explosion should peak, then dissipate)

A render might look great at frame 35 but flicker badly in animation.

### Questions for ML Specialists

1. Are there video quality metrics suitable for VFX (VMAF, temporal SSIM)?
2. Can optical flow analysis detect unnatural motion?
3. Should we evaluate short video clips instead of frames?

---

## Challenge 6: Reference Image Dependency

### Problem

LPIPS requires a reference image. Options:

| Reference Type | Problem |
|----------------|---------|
| Real photo | Domain mismatch (rendered vs. real) |
| Previous best render | Only measures relative improvement |
| "Gold standard" render | Requires manual creation |
| No reference | Can only use CLIP (no perceptual grounding) |

### Current Workaround

Using CLIP-only evaluation with semantic queries. This works for pass/fail but doesn't guide iteration.

### Questions for ML Specialists

1. Are there no-reference image quality metrics suitable for VFX?
2. Can we build a "reference-free" perceptual quality model?
3. Could we use GAN discriminators trained on good VFX as quality proxies?

---

## Proposed Evaluation Pipeline (Ideal)

```
┌──────────────────────────────────────────────────────────────────┐
│                    Multi-Modal Evaluation                        │
├──────────────┬──────────────┬──────────────┬────────────────────┤
│  Perceptual  │   Semantic   │  Aesthetic   │    Diagnostic      │
│    (LPIPS)   │    (CLIP)    │   (NIMA?)    │    (VLM/GPT-4V)    │
├──────────────┼──────────────┼──────────────┼────────────────────┤
│ Similarity   │ "Is this an  │ "How good    │ "What's wrong?     │
│ to reference │  explosion?" │  does it     │  What should       │
│              │              │  look?"      │  change?"          │
└──────────────┴──────────────┴──────────────┴────────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │  Unified Score + Action Plan  │
              │  - Overall quality: 7.2/10    │
              │  - Next action: increase      │
              │    flame_smoke by 30%         │
              └───────────────────────────────┘
```

---

## Technical Constraints

1. **Latency**: Evaluation should complete in <30 seconds per image
2. **Cost**: Prefer local models over API calls (GPT-4V costs $0.01+ per image)
3. **Consistency**: Same image should get same score (deterministic)
4. **GPU**: RTX 4060 Ti (8GB VRAM) available for inference

---

## Existing Assets for Training/Calibration

- 3 curated reference images (real explosions, watermark-free)
- ~10 rendered iterations at various quality levels
- Blender simulation parameters for each render
- Human quality rankings (informal, could be formalized)

---

## Summary of Key Questions

1. **Domain gap**: How to compare rendered vs. real images fairly?
2. **Gradient signal**: How to get fine-grained quality scores, not just pass/fail?
3. **Actionable feedback**: How to get diagnostic info that maps to simulation parameters?
4. **Exploration**: How to encourage trying different approaches, not just parameter tweaking?
5. **Temporal**: How to evaluate animation quality, not just single frames?
6. **Reference-free**: Can we evaluate quality without ground truth references?

---

## Contact

This document prepared for review by ML specialists to improve the PlasmaDXR self-improving asset pipeline.

**Project**: PlasmaDXR-Clean (DirectX 12 Volumetric Renderer)
**Component**: Self-Improving NanoVDB Asset Generation Pipeline
**Date**: 2025-12-25
