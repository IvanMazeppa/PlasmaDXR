# Evaluation System Improvement Proposal

**Date:** 2025-12-31
**Purpose:** Document research findings and implementation plan for structural/morphological evaluation to replace aggregate-statistics-based system

---

## Executive Summary

The current evaluation system is **blind to structural and morphological issues** because it uses aggregate statistics (mean brightness, edge_density, warm_ratio). Two images can have identical statistics while looking completely different.

**Example:**
- Synthetic render with "popcorn" texture and "cat ear" ribbon prominences scores 46.5/100
- System's complaint: "color too cool"
- What humans see: wrong texture, wrong prominence shape, no dark features, no limb darkening

**Solution:** Replace aggregate statistics with structural/morphological analysis using:
1. DINOv2 self-supervised features (patch-level structural similarity)
2. Multi-scale wavelet analysis (detect procedural vs organic texture)
3. Real/Synthetic discriminator with Grad-CAM (localized "fake" detection)
4. Domain-specific shape classifiers (prominence loop vs ribbon)

---

## Research Findings

### 1. DINOv2 Self-Supervised Features (RECOMMENDED PRIMARY)

**Source:** [DINOv2 vs CLIP comparison](https://medium.com/aimonks/clip-vs-dinov2-in-image-similarity-6fa5aa7ed8c6)

**Key findings:**
- DINOv2 achieves **64% accuracy vs CLIP's 28%** on challenging similarity tasks
- Learns **structural understanding** without captions/labels
- Patch-level features detect WHERE problems are, not just that they exist
- Demonstrates semantic part correspondence across different objects

**Why this matters for solar VFX:**
- DINOv2's patch features would immediately highlight that synthetic render has uniform texture while reference has varied multi-scale features
- Can detect that prominences are "ribbon-shaped" vs "loop-shaped" because it understands structure

**Technical details:**
- Model: `facebook/dinov2-base` (86M params) or `facebook/dinov2-large` (300M params)
- Output: 14x14 grid of 768-dim patch features + 1 CLS token
- Comparison: Cosine similarity at patch level → spatial heatmap of differences

**Source:** [Meta's DINOv2 Blog](https://ai.meta.com/blog/dino-v2-computer-vision-self-supervised-learning/)

---

### 2. DISTS (Deep Image Structure and Texture Similarity)

**Source:** [Paper: Image Quality Assessment: Unifying Structure and Texture Similarity](https://arxiv.org/abs/2004.07728)

**Key insight:** Traditional IQA methods are overly sensitive to texture resampling. DISTS explicitly separates:
- **Structure similarity:** Correlations of feature maps (spatial arrangement)
- **Texture similarity:** Correlations of spatial averages (statistical properties)

**Why this matters:**
- The "popcorn" texture problem is exactly texture resampling - procedural noise at wrong scale
- DISTS would score this lower because structure is wrong even if texture statistics match

**Implementation:** Available in `piq` library: `pip install piq`

```python
from piq import DISTS
dists = DISTS()
score = dists(render_tensor, reference_tensor)  # Lower = more similar
```

---

### 3. Multi-Scale Wavelet Analysis

**Source:** [Wavelet-optimized whitening for solar images](https://www.aanda.org/articles/aa/full_html/2023/02/aa45345-22/aa45345-22.html)

**Key insight:** This paper is specifically about **solar corona image enhancement** using à trous wavelet decomposition.

**Why wavelets detect procedural noise:**
- Procedural textures have energy concentrated at specific scales (low entropy)
- Natural images have broad-spectrum scale distribution (high entropy)
- Single-octave Perlin noise → spike at one scale
- Real sun → energy distributed across 5+ scales

**Algorithm:**
```python
import pywt

# 5-level wavelet decomposition
coeffs = pywt.wavedec2(image, 'db4', level=5)

# Energy at each scale
scale_energy = [np.mean(c**2) for c in coeffs[1:]]

# Entropy of distribution
energy_dist = scale_energy / sum(scale_energy)
scale_entropy = -sum(e * log2(e) for e in energy_dist)

# Low entropy = procedural, High entropy = natural
is_procedural = scale_entropy < 1.5
```

---

### 4. GAN Discriminator + Grad-CAM

**Source:** [AI-Generated Image Detection: An Empirical Study](https://arxiv.org/html/2511.02791)

**Key findings:**
- Fine-tuned GAN discriminators achieve **91% recall for synthetic artifacts**
- Grad-CAM heatmaps show WHERE the "fake" signal comes from
- Best approach: EfficientNetV2 with transfer learning (94.7% accuracy, AUC 0.98)

**Source:** [Deepfake Detection with Grad-CAM](https://medium.com/@seyma.gulsen/deepfake-detection-using-cnn-ensembles-and-grad-cam-e99aaaa638ce)

**Training data for solar discriminator:**
- Positive (real): 840 frames from `Eruptions_20241008_Activity_2048p30`
- Negative (synthetic): Generated Blender renders + augmentation

**Why this matters:**
- Would immediately flag the synthetic render as "94% likely synthetic"
- Grad-CAM would highlight the popcorn texture and ribbon prominences as "fake" regions

---

### 5. Dual-Branch Shape-Texture Networks

**Source:** [Deep Shape-Texture Statistics for Blind IQA](https://dl.acm.org/doi/10.1145/3694977)

**Key insight:** Deep features are texture-biased and lack shape-bias. The Shape-Texture Adaptive Fusion (STAF) module merges both.

**Why this matters:**
- "Ribbon vs loop prominences" is a SHAPE problem, not texture
- Current system can't detect this because it only measures texture statistics
- Need explicit shape analysis for prominence classification

---

### 6. Foundation Model Approaches (Future)

**Source:** [Foundation Models Boost Low-Level Perceptual Similarity](https://pdxscholar.library.pdx.edu/cgi/viewcontent.cgi?article=1382&context=compsci_fac)

**Key finding:** DINOv1-ViT-B is top performer across datasets, particularly for geometric distortions.

**Source:** [Awesome Image Quality Assessment](https://github.com/chaofengc/Awesome-Image-Quality-Assessment) - comprehensive collection of IQA papers.

---

## Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    NEW EVALUATION PIPELINE                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  INPUT: Render Image + Reference Image(s) + Reference Stats     │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  LAYER 1: DINOv2 Structural Similarity                   │   │
│  │  - Extract 14x14 patch features from both images         │   │
│  │  - Cosine similarity at each patch → spatial heatmap     │   │
│  │  - Identify worst-matching regions                       │   │
│  │  - Global structural similarity score                    │   │
│  └──────────────────────────────────────────────────────────┘   │
│                            ↓                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  LAYER 2: Multi-Scale Wavelet Analysis                   │   │
│  │  - À trous wavelet decomposition (5 scales)              │   │
│  │  - Compute energy at each scale                          │   │
│  │  - Calculate scale entropy                               │   │
│  │  - Flag if entropy < 1.5 (procedural noise)              │   │
│  └──────────────────────────────────────────────────────────┘   │
│                            ↓                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  LAYER 3: Real/Synthetic Discriminator                   │   │
│  │  - EfficientNetV2 fine-tuned on 840 solar frames         │   │
│  │  - Output: P(real) probability                           │   │
│  │  - Grad-CAM: localized "fake" regions                    │   │
│  └──────────────────────────────────────────────────────────┘   │
│                            ↓                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  LAYER 4: Domain-Specific Detectors (Solar)              │   │
│  │  - Prominence shape classifier (loop/ribbon/spray/blob)  │   │
│  │  - Dark feature detector (filaments, sunspots)           │   │
│  │  - Granulation quality scorer                            │   │
│  │  - Limb darkening gradient analyzer                      │   │
│  └──────────────────────────────────────────────────────────┘   │
│                            ↓                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  OUTPUT: Structured Diagnosis                            │   │
│  │                                                          │   │
│  │  CRITICAL: TEXTURE_PATTERN = procedural_uniform          │   │
│  │  CRITICAL: PROMINENCE_SHAPE = flat_ribbon                │   │
│  │  HIGH: MISSING_DARK_FEATURES = 0% coverage               │   │
│  │  MODERATE: COLOR_TOO_COOL                                │   │
│  │                                                          │   │
│  │  + Spatial heatmaps showing WHERE problems are           │   │
│  │  + Blender parameter suggestions for each issue          │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation Plan

### Phase 1: DINOv2 Structural Evaluation (Priority: HIGHEST)

**File:** `agents/asset-evaluator/dino_structural_eval.py`

**Dependencies:**
```bash
pip install transformers torch torchvision
```

**Key functions:**
- `extract_patch_features(image_path)` → 14x14x768 tensor
- `compute_structural_similarity(render, reference)` → similarity map + score
- `identify_problem_regions(similarity_map)` → list of (region, severity)
- `generate_similarity_heatmap(similarity_map)` → visualization

**Integration:** New MCP tool `evaluate_structural_quality`

**Expected impact:** Will immediately detect that synthetic render has uniform texture pattern while reference has varied multi-scale features.

---

### Phase 2: Multi-Scale Wavelet Analysis (Priority: HIGH)

**File:** `agents/asset-evaluator/wavelet_scale_analysis.py`

**Dependencies:**
```bash
pip install PyWavelets
```

**Key functions:**
- `analyze_scale_distribution(image_path)` → scale energies + entropy
- `detect_procedural_texture(image_path)` → bool + confidence
- `compare_scale_distributions(render, reference)` → similarity

**Expected impact:** Will flag "popcorn" texture as single-scale procedural noise.

---

### Phase 3: Solar Discriminator Training (Priority: MEDIUM)

**File:** `agents/asset-evaluator/solar_discriminator.py`

**Dependencies:**
```bash
pip install timm pytorch-grad-cam
```

**Training data:**
- Positive: 840 frames from `assets/reference_images/star/Eruptions_20241008_Activity_2048p30/`
- Negative: Synthetic renders from `build/vdb_output/` + augmentation

**Key functions:**
- `train(real_dir, synthetic_dir, epochs)` → saves model checkpoint
- `predict_with_explanation(image_path)` → p_real + gradcam heatmap
- `identify_fake_regions(gradcam)` → list of regions

**Expected impact:** 90%+ accuracy distinguishing real from synthetic, with localized explanations.

---

### Phase 4: Prominence Shape Classifier (Priority: MEDIUM)

**File:** `agents/asset-evaluator/prominence_shape_classifier.py`

**Key functions:**
- `segment_prominences(image_path)` → list of prominence masks
- `classify_shape(prominence_mask)` → "loop" | "ribbon" | "spray" | "blob"
- `compute_thickness_variation(prominence_mask)` → 0-1 score
- `compute_volumetric_appearance(prominence_mask)` → 0-1 score

**Expected impact:** Will detect "cat ear" ribbon prominences vs volumetric loops.

---

## Expected Output Comparison

### Current System (Aggregate Statistics)

```json
{
  "overall_score": 46.5,
  "passed": false,
  "primary_issues": [
    "color_too_cool: Not warm enough (render: 2.60, reference: 19.16)"
  ],
  "suggested_fixes": [
    {"parameter": "flame_color", "change": "shift toward orange"}
  ]
}
```

### New System (Structural + Morphological)

```json
{
  "overall_score": 22.5,
  "passed": false,
  "primary_issues": [
    {
      "severity": "CRITICAL",
      "category": "TEXTURE_PATTERN",
      "description": "Uniform procedural noise detected (popcorn texture)",
      "scale_entropy": 1.2,
      "expected": "> 2.0 for natural imagery",
      "regions": ["entire_disk"],
      "fix": "Use multi-octave FBM noise with varying scales"
    },
    {
      "severity": "CRITICAL",
      "category": "PROMINENCE_SHAPE",
      "description": "Prominences classified as flat_ribbon (87% confidence)",
      "expected": "volumetric_loop",
      "regions": ["prominence_left", "prominence_right"],
      "fix": "Add curl/vorticity to prominence flow, increase domain depth"
    },
    {
      "severity": "CRITICAL",
      "category": "LOOKS_SYNTHETIC",
      "description": "Discriminator 94% confident this is synthetic",
      "gradcam_regions": ["disk_surface", "prominences"],
      "fix": "See heatmap for specific problem areas"
    },
    {
      "severity": "HIGH",
      "category": "MISSING_DARK_FEATURES",
      "description": "0% dark feature coverage",
      "expected": "15-30%",
      "fix": "Add absorbing material for dark filaments"
    },
    {
      "severity": "MODERATE",
      "category": "COLOR_TOO_COOL",
      "description": "warm_ratio 2.60 vs reference 19.16",
      "fix": "Increase flame temperature"
    }
  ],
  "structural_similarity": {
    "global_score": 0.34,
    "worst_regions": [
      {"region": "disk_center", "similarity": 0.21},
      {"region": "prominence_left", "similarity": 0.28}
    ],
    "heatmap_path": "evaluation_outputs/similarity_heatmap.png"
  }
}
```

---

## References

1. [CLIP vs DINOv2 in Image Similarity](https://medium.com/aimonks/clip-vs-dinov2-in-image-similarity-6fa5aa7ed8c6)
2. [DINOv2: State-of-the-art CV models with self-supervised learning](https://ai.meta.com/blog/dino-v2-computer-vision-self-supervised-learning/)
3. [DISTS: Unifying Structure and Texture Similarity](https://arxiv.org/abs/2004.07728)
4. [Wavelet-optimized whitening for solar images](https://www.aanda.org/articles/aa/full_html/2023/02/aa45345-22/aa45345-22.html)
5. [AI-Generated Image Detection: An Empirical Study](https://arxiv.org/html/2511.02791)
6. [Deepfake Detection with Grad-CAM](https://medium.com/@seyma.gulsen/deepfake-detection-using-cnn-ensembles-and-grad-cam-e99aaaa638ce)
7. [Deep Shape-Texture Statistics for Blind IQA](https://dl.acm.org/doi/10.1145/3694977)
8. [Awesome Image Quality Assessment](https://github.com/chaofengc/Awesome-Image-Quality-Assessment)
9. [Foundation Models Boost Low-Level Perceptual Similarity](https://pdxscholar.library.pdx.edu/cgi/viewcontent.cgi?article=1382&context=compsci_fac)

---

## Quick Start (After Implementation)

```python
# Evaluate a render with the new structural system
result = await evaluate_structural_quality(
    image_path="build/vdb_output/sun_prominences_v2/render_0060.png",
    reference_path="assets/reference_images/star/Eruptions_20241008_Activity_2048p30/frame_00639.jpg",
    effect_type="sun"
)

# Result will include:
# - Structural similarity heatmap (shows WHERE differences are)
# - Scale entropy analysis (detects procedural noise)
# - Real/Synthetic probability with Grad-CAM
# - Prominence shape classification
# - Prioritized, actionable fixes
```

---

**Document Author:** Claude Code session
**Last Updated:** 2025-12-31
