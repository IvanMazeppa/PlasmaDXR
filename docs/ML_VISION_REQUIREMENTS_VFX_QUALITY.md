# ML Vision Requirements for VFX Quality Assessment

**Date:** 2025-12-31
**Target:** NN/ML Plugin Agent
**Priority:** P1 - Critical Pipeline Blocker
**Author:** Claude Code Analysis

---

## Executive Summary

The current Blender VFX asset generation pipeline uses **LPIPS** and **CLIP** for quality assessment. These metrics fundamentally fail for volumetric VFX content, causing the iteration loop to never converge. This document details the problems and researches ML/AI technologies that could provide effective quality signal for VFX evaluation.

**Goal:** Replace or augment LPIPS/CLIP with metrics that provide meaningful gradient signal for volumetric effects (fire, explosions, nebulae, stars, smoke).

---

## Problem Statement

### Current Metrics Fail for VFX

| Metric | Designed For | Works For VFX? | Evidence |
|--------|--------------|----------------|----------|
| **LPIPS** | Photorealistic image similarity | **NO** | Scores 0.73-0.76 for all sun renders regardless of visible quality differences |
| **CLIP** | Semantic image-text alignment | **NO** | Plateaus at 0.65-0.67 once basic concept matches ("this is a sun") |

### Observed Failure Pattern

From sun surface iteration log (v7-v11):

```
v7:  LPIPS=0.738, CLIP=0.671  (dim, lacks corona)
v8:  LPIPS=0.759, CLIP=0.659  (better color)
v9:  LPIPS=0.752, CLIP=0.663  (good structure)
v10: LPIPS=0.745, CLIP=0.667  (best so far)
v11: LPIPS=0.753, CLIP=0.661  (added granulation)
```

**Problem:** 5 iterations with visible quality improvements show no meaningful score change. The metrics provide noise, not signal.

### Root Causes

1. **LPIPS** (Learned Perceptual Image Patch Similarity)
   - Trained on BAPPS dataset (photorealistic image pairs)
   - Measures "perceptual distance" - lower = more similar to reference
   - **Requires a reference image** - VFX has no "correct" reference
   - Designed for: photo editing, super-resolution, style transfer
   - Not designed for: evaluating standalone generated content

2. **CLIP** (Contrastive Language-Image Pre-training)
   - Trained on 400M web image-text pairs
   - Measures semantic alignment (does image match description?)
   - **Saturates at concept level** - "sun" = "sun" regardless of quality
   - No gradient signal for quality within same concept

### Current Threshold Mismatch

```python
# asset-evaluator thresholds:
LPIPS < 0.35  (pass)  # VFX scores 0.7-0.8 - IMPOSSIBLE to pass
CLIP > 0.60   (pass)  # VFX plateaus at 0.65-0.67 - barely passes
```

---

## Research: ML/AI Technologies for VFX Quality Assessment

### 1. No-Reference Image Quality Assessment (NR-IQA)

**What it is:** Deep learning models that predict image quality WITHOUT a reference image.

**Recent Advances (2024-2025):**
- [Transfer Learning-Based NR-IQA](https://pmc.ncbi.nlm.nih.gov/articles/PMC11888849/) - Uses pre-trained features for quality prediction
- [Transformer-Based Methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC11457998/) - Swin Transformer + Global Self-Attention for long-range dependencies
- [Dual-Branch Networks](https://francis-press.com/papers/18374) - HSV content features + structural gradient features

**Pros:**
- No reference image needed
- Can score individual images
- Well-researched problem space

**Cons:**
- Most trained on natural image distortions (blur, noise, compression)
- May not understand VFX-specific quality (volumetric depth, color temperature)
- Unclear if they provide actionable diagnostics

**Recommendation:** Worth investigating as baseline, but likely needs fine-tuning for VFX domain.

---

### 2. LAION Aesthetic Predictor

**What it is:** A lightweight model (CLIP + MLP) that predicts "how much people like an image" on a 1-10 scale.

**Sources:**
- [LAION-Aesthetics Blog](https://laion.ai/blog/laion-aesthetics/)
- [GitHub: aesthetic-predictor](https://github.com/LAION-AI/aesthetic-predictor)
- [Improved version](https://github.com/christophschuhmann/improved-aesthetic-predictor)

**How it works:**
1. Extract CLIP ViT-L/14 embeddings from image
2. Pass through small MLP head
3. Output: aesthetic score 1-10

**Training Data:** 5000+ image-rating pairs from SAC (Simulacra Aesthetic Captions) dataset, expanded to larger subsets.

**Thresholds (from LAION):**
- Score > 5.0: Good aesthetic quality
- Score > 6.0: High aesthetic quality
- Score > 6.5: Excellent aesthetic quality

**Pros:**
- Reference-free (scores standalone images)
- Fast inference (just CLIP embedding + small MLP)
- Already understands aesthetic appeal
- Available on PyPI: `simple-aesthetics-predictor`

**Cons:**
- Trained on general images, not VFX specifically
- Aesthetic quality ≠ VFX quality (an ugly but accurate explosion may score low)
- May not correlate with VFX-specific goals

**Recommendation:** **HIGH PRIORITY** - Easy to integrate, provides reference-free baseline. Use as one dimension of composite score.

---

### 3. ImageReward (Human Preference for Text-to-Image)

**What it is:** A reward model trained on 137k expert comparisons to predict human preference for text-to-image generation.

**Sources:**
- [NeurIPS 2023 Paper](https://arxiv.org/abs/2304.05977)
- [GitHub: ImageReward](https://github.com/zai-org/ImageReward)

**Performance:**
- Outperforms CLIP by 38.6%
- Outperforms Aesthetic Predictor by 39.6%
- Outperforms BLIP by 31.6%

**What it evaluates:**
1. Text-image alignment (does image match prompt?)
2. Fidelity (is the image well-formed?)
3. Aesthetics (does it look good?)

**Pros:**
- Specifically designed for generated images
- Considers multiple quality dimensions
- Strong correlation with human preference

**Cons:**
- Designed for diffusion model outputs (Stable Diffusion, DALL-E)
- May not transfer directly to Blender-rendered VFX
- Heavier than pure CLIP (but not enormous)

**Recommendation:** **HIGH PRIORITY** - Best available metric for generated content quality. Test on VFX renders to validate correlation.

**Note:** Team released **VisionReward** (Dec 2024) - next-gen multi-dimensional reward model for visual generation.

---

### 4. Vision Language Models (VLMs) for Structured Assessment

**What it is:** Large multimodal models (GPT-4V, Claude 3.5, Moondream) that can analyze images and provide textual feedback.

**Sources:**
- [IQAGPT: CT Image Quality Assessment with VLMs](https://pmc.ncbi.nlm.nih.gov/articles/PMC11300764/)
- [VLM Benchmarks](https://www.clarifai.com/blog/best-vision-language-models-vlms-for-image-classification-performance-benchmarks)

**Approach:**
1. Pass image to VLM with structured prompt
2. Ask for quality assessment on specific dimensions
3. Parse structured response (JSON) with scores and diagnostics

**Example Prompt:**
```
Analyze this volumetric fire/explosion render for quality on these dimensions:
1. Color Temperature: Does it show hot-to-cool gradient? (1-10)
2. Volumetric Depth: Does it have 3D structure? (1-10)
3. Brightness Distribution: Appropriate contrast? (1-10)
4. Edge Detail: Sharp vs fuzzy? (1-10)
5. Coverage: Does effect fill frame appropriately? (1-10)

Return JSON with scores and specific issues found.
```

**VLM Options:**

| Model | Size | Local? | Cost | Quality |
|-------|------|--------|------|---------|
| GPT-4V | Cloud | No | $$$$ | Best |
| Claude 3.5 Sonnet | Cloud | No | $$$ | Excellent |
| Qwen2-VL-72B | 72B | Possible | GPU | Very Good |
| **Moondream 2B** | 2B | **Yes** | Free | Good |
| **Moondream 0.5B** | 0.5B | **Yes** | Free | Usable |

**Moondream Details:**
- [GitHub](https://github.com/vikhyat/moondream)
- [Hugging Face](https://huggingface.co/vikhyatk/moondream2)
- Moondream 0.5B: Only 479 MiB, runs on CPU
- Moondream 3 Preview: 9B total, 2B active (mixture of experts)

**Pros:**
- Provides actionable textual feedback ("lacks volumetric depth in center")
- Can score on VFX-specific dimensions
- Easy to customize via prompt engineering
- Local options available (Moondream)

**Cons:**
- Slower than pure embedding models
- Requires prompt engineering to get consistent output
- Local models less capable than cloud APIs

**Recommendation:** **MEDIUM-HIGH PRIORITY** - Moondream 2B for local inference provides structured diagnostics that pure metrics cannot. Use for detailed issue identification.

---

### 5. AI-Generated Content Quality Assessment (AIGC-QA)

**What it is:** Research field specifically focused on evaluating AI-generated images.

**Sources:**
- [Awesome Evaluation of Visual Generation](https://github.com/ziqihuangg/Awesome-Evaluation-of-Visual-Generation)
- [NTIRE 2024 AIGC Quality Challenge](https://arxiv.org/html/2404.16687v2)
- [SF-IQA: Quality and Similarity for AIGC](https://openaccess.thecvf.com/content/CVPR2024W/NTIRE/papers/Yu_SF-IQA_Quality_and_Similarity_Integration_for_AI_Generated_Image_Quality_CVPRW_2024_paper.pdf)

**Evaluation Dimensions (from NTIRE 2024):**
1. **Aesthetic Score** - Visual appeal
2. **Technical Score** - Image quality (sharpness, noise)
3. **Text-Image Consistency** - Prompt alignment
4. **Fluency** - Artifact-free generation
5. **Temporal Consistency** - For video/animation

**Key Insight:** AIGC research separates quality into multiple dimensions rather than single score.

**Pros:**
- Designed for generated content (not just natural images)
- Multi-dimensional evaluation matches VFX needs
- Active research area with new methods

**Cons:**
- Most focused on diffusion model outputs
- May need adaptation for Blender-rendered VFX
- Heavier computational requirements

**Recommendation:** **MEDIUM PRIORITY** - Review [Awesome-Evaluation-of-Visual-Generation](https://github.com/ziqihuangg/Awesome-Evaluation-of-Visual-Generation) repo for latest methods.

---

### 6. Feature-Based Diagnostics (Non-ML Baseline)

**What it is:** Traditional computer vision feature extraction for quick diagnostics.

**Approach:**
```python
def extract_vfx_diagnostics(image_path: str) -> dict:
    img = Image.open(image_path)
    arr = np.array(img)

    return {
        "brightness": {
            "mean": float(np.mean(arr)),
            "max": float(np.max(arr)),
            "dynamic_range": float(np.max(arr) - np.min(arr))
        },
        "color": {
            "warm_ratio": float(np.sum(arr[..., 0]) / max(np.sum(arr[..., 2]), 1)),
            "has_gradient": _detect_color_gradient(arr)
        },
        "structure": {
            "edge_density": float(cv2.Canny(arr, 50, 150).mean() / 255),
            "variance": float(np.var(arr))
        },
        "coverage": {
            "non_black": float(np.sum(arr > 10) / arr.size),
            "bright_pixels": float(np.sum(arr > 200) / arr.size)
        }
    }
```

**Pros:**
- Instant inference (no model loading)
- Completely deterministic
- Provides concrete, interpretable diagnostics
- Can detect specific issues (too dark, no structure, wrong color)

**Cons:**
- No semantic understanding
- Requires manual threshold tuning
- Cannot assess "realism" or "aesthetic appeal"

**Recommendation:** **HIGH PRIORITY** - Implement as immediate baseline. Use alongside ML metrics.

---

## Proposed Hybrid Architecture

Based on research, recommend a **multi-signal composite score**:

```python
class VFXQualityAssessor:
    def __init__(self):
        self.aesthetic_model = load_aesthetic_predictor()
        self.image_reward = load_image_reward()
        self.vlm = Moondream("moondream2")

    def assess(self, image_path: str, prompt: str, effect_type: str) -> dict:
        # 1. Fast feature diagnostics (instant)
        diagnostics = extract_vfx_diagnostics(image_path)

        # 2. Aesthetic score (fast, reference-free)
        aesthetic_score = self.aesthetic_model(image_path)  # 1-10

        # 3. ImageReward (human preference)
        reward_score = self.image_reward(image_path, prompt)  # unbounded

        # 4. VLM structured assessment (slower, detailed)
        vlm_assessment = self.vlm.assess_vfx(
            image_path,
            effect_type,
            dimensions=["color_temp", "volumetric_depth", "structure"]
        )

        # 5. Composite score
        composite = self._compute_composite(
            diagnostics, aesthetic_score, reward_score, vlm_assessment
        )

        return {
            "composite_score": composite,  # 0-100
            "diagnostics": diagnostics,
            "aesthetic": aesthetic_score,
            "image_reward": reward_score,
            "vlm_assessment": vlm_assessment,
            "issues": self._identify_issues(diagnostics, vlm_assessment),
            "recommendations": self._suggest_fixes(...)
        }
```

### Processing Pipeline

```
Image + Prompt
      ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 1: Feature Diagnostics (instant, 0ms)                 │
│   → brightness, color ratio, edge density, coverage         │
│   → Immediate pass/fail on obvious issues                   │
└─────────────────────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 2: Aesthetic + ImageReward (fast, ~100ms)             │
│   → LAION aesthetic score (1-10)                            │
│   → ImageReward human preference score                      │
│   → Reference-free quality baselines                        │
└─────────────────────────────────────────────────────────────┘
      ↓
┌─────────────────────────────────────────────────────────────┐
│ Layer 3: VLM Structured Assessment (slower, ~1-2s)          │
│   → Moondream 2B local inference                            │
│   → VFX-specific quality dimensions                         │
│   → Textual diagnostics + actionable feedback               │
└─────────────────────────────────────────────────────────────┘
      ↓
    Composite Score + Issue List + Recommendations
```

---

## Recommended Implementation Phases

### Phase 1: Immediate (This Week)
1. **Add feature-based diagnostics** to asset-evaluator
2. **Integrate LAION Aesthetic Predictor** (`pip install simple-aesthetics-predictor`)
3. Create composite scoring function

### Phase 2: Short-term (Next Week)
1. **Integrate ImageReward** (`pip install image-reward`)
2. **Integrate Moondream 2B** for structured assessment
3. Build VLM prompt templates for VFX dimensions

### Phase 3: Validation (Following Week)
1. Run all metrics on existing sun surface iterations
2. Verify metrics correlate with visible quality differences
3. Tune composite weights based on empirical results

---

## Dependencies to Add

```bash
# Phase 1
pip install opencv-python numpy pillow

# Phase 2
pip install simple-aesthetics-predictor  # LAION
pip install image-reward                  # ImageReward

# Phase 3
pip install transformers accelerate       # Moondream
pip install torch                         # If not present
```

**Model Sizes:**
- LAION Aesthetic: ~1.5GB (CLIP ViT-L/14 + MLP)
- ImageReward: ~2GB
- Moondream 2B: ~4GB
- Total: ~7.5GB (can share CLIP backbone)

---

## Success Criteria

The new metrics should:

1. **Distinguish visible quality differences** - v7 (dim) should score lower than v10 (best)
2. **Provide gradient signal** - Small improvements should produce small score increases
3. **Be actionable** - Output should say "too dark" or "lacks structure", not just a number
4. **Work without reference** - No "ground truth" image required
5. **Run locally** - No cloud API dependency for core evaluation

---

## Questions for NN/ML Agent

1. Should we fine-tune LAION Aesthetic on VFX-specific data?
2. Is ImageReward's training on diffusion outputs transferable to Blender renders?
3. Can Moondream 2B be quantized further for faster inference?
4. Should we collect human ratings on our VFX renders to create training data?
5. Are there VFX-specific quality assessment models in the film/games industry?

---

## Sources

- [LAION-Aesthetics Blog](https://laion.ai/blog/laion-aesthetics/)
- [LAION Aesthetic Predictor GitHub](https://github.com/LAION-AI/aesthetic-predictor)
- [ImageReward NeurIPS 2023 Paper](https://arxiv.org/abs/2304.05977)
- [ImageReward GitHub](https://github.com/zai-org/ImageReward)
- [Moondream VLM](https://github.com/vikhyat/moondream)
- [NTIRE 2024 AIGC Quality Challenge](https://arxiv.org/html/2404.16687v2)
- [Awesome Evaluation of Visual Generation](https://github.com/ziqihuangg/Awesome-Evaluation-of-Visual-Generation)
- [Transfer Learning NR-IQA](https://pmc.ncbi.nlm.nih.gov/articles/PMC11888849/)
- [IQAGPT: VLM for Image Quality](https://pmc.ncbi.nlm.nih.gov/articles/PMC11300764/)

---

**Document Location:** `docs/ML_VISION_REQUIREMENTS_VFX_QUALITY.md`
**Related:** `docs/MULTI_AGENT_PIPELINE_OPTIMIZATION_ANALYSIS.md`, `docs/BLENDER_VFX_PIPELINE_STATUS.md`
