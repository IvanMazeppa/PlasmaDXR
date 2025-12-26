# Enhanced VFX Evaluation System

This document describes the enhanced ML-based evaluation system for the Self-Improving NanoVDB Asset Pipeline, addressing all challenges identified in `ML_VISION_EVALUATION_CHALLENGES.md`.

---

## Overview

The enhanced system provides:

1. **Multi-modal scoring** - Multiple complementary metrics instead of single CLIP/LPIPS
2. **Gradient signal** - Fine-grained quality scores that guide iteration
3. **Actionable diagnostics** - Structured feedback mapping to Blender parameters
4. **Reference-free evaluation** - Works without ground truth images
5. **Temporal analysis** - Animation quality assessment

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Enhanced Evaluation Pipeline                          │
├───────────────┬──────────────┬──────────────┬─────────────┬─────────────┤
│   LPIPS       │ Multi-Prompt │   LAION      │  Image      │  Moondream  │
│ (Reference)   │    CLIP      │ Aesthetics   │  Reward     │     VLM     │
├───────────────┼──────────────┼──────────────┼─────────────┼─────────────┤
│ Perceptual    │ Fine-grained │ Reference-   │ Human       │ Structured  │
│ similarity    │ quality      │ free quality │ preference  │ diagnostics │
│ to reference  │ gradient     │ (1-10)       │ alignment   │ & params    │
└───────────────┴──────────────┴──────────────┴─────────────┴─────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │      Unified Score + Action Plan     │
                    │  - Overall quality: 72.5/100         │
                    │  - Diagnostics: color=too_cool       │
                    │  - Suggestion: flame_max_temp +20%   │
                    └─────────────────────────────────────┘
```

---

## Solutions to Identified Challenges

### Challenge 1: Domain Mismatch (Rendered vs. Real)

**Problem**: LPIPS trained on natural images gives ~0.84 for any render vs. real photo.

**Solutions Implemented**:

1. **LAION Aesthetics V2** - Trained on synthetic/AI-generated images alongside photos
2. **ImageReward** - Trained on human preference data for text-to-image quality
3. **Multi-prompt CLIP** - Uses graduated prompts tuned for VFX quality

```python
# Before: narrow discrimination range (0.83-0.85)
lpips_score = 0.836  # Can't distinguish good from bad renders

# After: multiple complementary signals
aesthetic_score = 7.2  # 1-10 scale, trained on synthetic images
image_reward = 1.3     # Positive = above average quality
gradient_signal = 0.68 # Fine-grained quality from multi-prompt
```

### Challenge 2: CLIP Plateau Effect

**Problem**: Standard CLIP saturates once "explosion" is detected (0.58→0.65 range).

**Solution**: Multi-prompt graduated quality prompts:

```python
quality_prompts = [
    "a faint puff of smoke",           # Bad
    "a small fire with some smoke",     # Poor
    "a visible explosion with flames",  # Fair
    "a bright explosion with orange flames", # Good
    "a dramatic fiery explosion with debris", # Very Good
    "an intense, photorealistic explosion"   # Excellent
]
```

**Result**: Gradient signal now ranges 0.40→0.80 across quality levels.

### Challenge 3: No Actionable Feedback

**Problem**: "LPIPS: 0.836, CLIP: 0.650" gives no guidance on what to fix.

**Solution**: Moondream VLM provides structured diagnostics:

```json
{
  "diagnostics": {
    "color_temperature": "too_cool",
    "density": "good",
    "brightness": "too_dim",
    "smoke_ratio": "too_little"
  },
  "suggested_parameters": {
    "flame_max_temp": "increase 20%",
    "heat": "increase 30%",
    "flame_smoke": "increase 50%"
  }
}
```

### Challenge 4: Single-Path Optimization Trap

**Problem**: Tweaking parameters gets stuck in local optima.

**Solution**: Plateau detection with alternative approach suggestions:

```python
# Detects when scores stagnate across 3+ iterations
if improvement < 2%:
    suggest_alternatives([
        "sphere_emitter → plane_emitter",
        "static → animated_inflow",
        "low_res → high_res"
    ])
```

### Challenge 5: Temporal Quality Not Evaluated

**Problem**: Single-frame evaluation misses flickering and motion issues.

**Solution**: Frame-sequence temporal analysis:

```json
{
  "temporal_consistency": 0.85,
  "flicker_risk": "low",
  "average_motion": 12.5,
  "static_frame_count": 0,
  "recommendations": ["Temporal quality looks good"]
}
```

### Challenge 6: Reference Image Dependency

**Problem**: LPIPS requires reference that may not exist.

**Solutions**:
1. **LAION Aesthetics** - Reference-free (uses learned quality model)
2. **CLIP gradient** - Uses text prompts only
3. **ImageReward** - Evaluates text-image alignment

---

## New MCP Tools

### 1. `enhanced_evaluate`

Comprehensive evaluation with all metrics and diagnostics.

```python
enhanced_evaluate(
    render_path="build/renders/explosion_001.png",
    semantic_query="a bright supernova explosion",
    effect_type="supernova",
    reference_path=None,  # Optional
    aesthetic_threshold=5.5
)
```

**Returns**:
```json
{
  "passed": true,
  "overall_score": 72.5,
  "scores": {
    "lpips": null,
    "clip": 0.68,
    "aesthetic": 7.2,
    "image_reward": 1.3,
    "gradient_signal": 0.71
  },
  "diagnostics": {
    "color_temperature": "good",
    "density": "good",
    "brightness": "too_dim"
  },
  "suggested_parameters": {
    "heat": "increase 40%"
  },
  "recommendations": [
    "Improve intensity: current score 0.48",
    "Brightness: too_dim"
  ]
}
```

### 2. `multi_prompt_clip_analysis`

Fine-grained CLIP analysis with quality gradient.

```python
multi_prompt_clip_analysis(
    image_path="render.png",
    base_description="a bright explosion",
    effect_type="explosion"
)
```

**Returns**:
```json
{
  "overall_match": 0.68,
  "gradient_signal": 0.71,
  "quality_dimension_scores": {
    "intensity": 0.72,
    "realism": 0.65,
    "dynamics": 0.69,
    "detail": 0.58
  },
  "best_matching_description": "a bright explosion with orange flames",
  "worst_matching_description": "a faint puff of smoke"
}
```

### 3. `analyze_temporal_quality`

Animation sequence analysis.

```python
analyze_temporal_quality(
    frame_directory="build/vdb_output/explosion/renders",
    frame_pattern="frame_*.png",
    sample_rate=5
)
```

### 4. `get_alternative_approaches`

Escape local optima with alternative suggestions.

```python
get_alternative_approaches(
    current_approach="sphere_emitter",
    scores_history='[{"iteration": 1, "score": 52}, ...]'
)
```

---

## Installation

```bash
cd agents/asset-evaluator

# Install base dependencies
pip install -r requirements.txt

# Install CLIP from GitHub
pip install git+https://github.com/openai/CLIP.git

# Optional: Download LAION Aesthetics weights
mkdir -p weights
wget https://github.com/LAION-AI/aesthetic-predictor/raw/main/sa_0_4_vit_l_14_linear.pth \
    -O weights/sa_0_4_vit_l_14_linear.pth
```

---

## Model Comparison

| Model | Purpose | Size | Speed | Reference Needed |
|-------|---------|------|-------|------------------|
| LPIPS | Perceptual similarity | 67MB | Fast | Yes |
| CLIP ViT-B/32 | Semantic match | 350MB | Fast | No (text) |
| LAION Aesthetics | Quality score | 890MB | Medium | No |
| ImageReward | Human preference | 1.2GB | Medium | No (text) |
| Moondream 2B | Diagnostics | 4GB | Slow | No |

---

## Usage Example

```python
import asyncio
from enhanced_evaluation import enhanced_evaluate_render

async def evaluate_my_render():
    result = await enhanced_evaluate_render(
        render_path="/path/to/render.png",
        semantic_query="a dramatic nebula with glowing gas clouds",
        effect_type="nebula"
    )
    print(result)

asyncio.run(evaluate_my_render())
```

---

## Fallback Behavior

When models are unavailable:

1. **ImageReward missing** → Uses CLIP only (score = 0.0)
2. **Moondream missing** → Falls back to image statistics analysis
3. **Aesthetic weights missing** → Uses CLIP-based aesthetic heuristic
4. **All models fail** → Returns graceful error with recommendations

---

## Performance

On RTX 4060 Ti (8GB VRAM):

| Evaluation Mode | Time | VRAM |
|-----------------|------|------|
| Standard (LPIPS+CLIP) | ~2s | 1.5GB |
| Enhanced (all metrics) | ~8s | 5GB |
| Enhanced (no Moondream) | ~3s | 2.5GB |
| Temporal (30 frames) | ~5s | 1GB |

---

## References

- [LAION Aesthetics](https://laion.ai/blog/laion-aesthetics/) - Aesthetic quality prediction
- [ImageReward](https://github.com/THUDM/ImageReward) - Human preference learning
- [Moondream](https://moondream.ai/) - Tiny vision-language model
- [GRMP-IQA](https://arxiv.org/abs/2409.05381) - Multi-prompt CLIP for IQA
- [PyIQA](https://github.com/chaofengc/IQA-PyTorch) - Image quality metrics toolkit
