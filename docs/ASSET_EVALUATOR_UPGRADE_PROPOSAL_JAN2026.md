# Asset-Evaluator Design Analysis & Upgrade Proposal

**Date:** January 2026
**Status:** Proposal
**Author:** Claude Code Analysis

---

## Executive Summary

This document analyzes the current asset-evaluator implementation and proposes upgrades based on state-of-the-art ML tools available as of January 2026. The analysis also addresses architectural concerns about tool proliferation (currently 32 tools) and recommends consolidation.

---

## Part 1: Current Implementation Analysis

### Strengths

The asset-evaluator implements several sophisticated techniques:

| Component | Description | Effectiveness |
|-----------|-------------|---------------|
| **Feature Size Uniformity** | CV metric for procedural detection | EXCELLENT (20× discrimination) |
| **Multi-prompt CLIP** | Gradient signal for fine-grained quality | Good |
| **DINOv2 structural** | Patch-level structural similarity | Good (64% vs CLIP 28%) |
| **Wavelet scale entropy** | Procedural texture detection | Good |
| **EfficientNetV2-S discriminator** | Real vs synthetic with Grad-CAM | Good |
| **Ground truth evaluation** | Distribution comparison to NASA frames | Good |

### Key Metric Discovery

The **feature size coefficient of variation (CV)** is the MOST DISCRIMINATING metric discovered:

```
Synthetic "popcorn" render: CV = 1.83 (uniform feature sizes)
Real solar footage:         CV = 36.97 (varied sizes)
                            ─────────────────────────────
                            20× difference!
```

This metric alone provides better discrimination than many ML models.

---

## Part 2: State-of-the-Art Research (January 2026)

### New Tools Available

| Tool | Release | Description | Recommendation |
|------|---------|-------------|----------------|
| **SigLIP 2** | Feb 2025 | Improved vision-language encoder | Replace CLIP |
| **TOPIQ** | IEEE TIP 2024 | Semantic-guided quality assessment | Add via pyiqa |
| **QualiCLIP** | Jan 2025 | CLIP-based no-reference IQA | Add via pyiqa |
| **MACLIP** | Dec 2025 | Multi-scale CLIP features | Replace ImageReward |
| **DMM** | Dec 2025 | Degradation model matching | Consider for future |
| **Q-Align** | 2024 | VLM-based quality scoring | Consider for future |

### pyiqa Library

The `pyiqa` library (IQA-PyTorch) has become the standard toolbox, providing:
- 40+ IQA metrics in one package
- Consistent API across all metrics
- GPU acceleration
- Active maintenance (latest: Dec 2025)

---

## Part 3: Tool Proliferation Problem

### Current State: 32 Tools

The asset-evaluator currently exposes 32 MCP tools, including highly specific detectors:

```
- analyze_prominence_shapes
- detect_cat_ear_artifacts
- compare_prominence_quality
- evaluate_solar_features
- analyze_feature_size_distribution
- analyze_texture_procedural
- compare_texture_quality
- predict_real_or_synthetic
- evaluate_structural_quality
- evaluate_structural_against_dataset
- save_structural_heatmap
- evaluate_ground_truth
- compare_to_reference_distribution
- evaluate_with_suggestions
- get_fix_suggestions
- analyze_prominence
- extract_vfx_diagnostics
- evaluate_vfx_quality
- compare_vfx_iterations
- enhanced_evaluate
- multi_prompt_clip_analysis
- analyze_temporal_quality
- get_alternative_approaches
- compare_lpips
- compare_clip
- evaluate_render
- find_reference_images
- list_recent_renders
- train_discriminator
- compare_real_synthetic_batch
- get_reference_statistics
- extract_image_features
```

### Problems with This Approach

1. **Cognitive Overhead**: Agents must choose from 32 tools - which one for "my render looks wrong"?
2. **Overlapping Functionality**: Multiple tools analyze similar aspects (e.g., 5+ texture/procedural tools)
3. **Maintenance Burden**: Each tool has its own error handling, documentation, tests
4. **Brittle Specificity**: `detect_cat_ear_artifacts` only detects one specific artifact type
5. **Poor Generalization**: New artifact types require new tools
6. **Context Window Waste**: Tool descriptions consume tokens

### Why Specific Detectors Are Problematic

The `detect_cat_ear_artifacts` tool exemplifies the problem:

```python
# This tool ONLY detects triangular prominence artifacts
# If a new artifact type appears (e.g., "ribbon artifacts", "blob prominences")
# we need to write a NEW specific detector

# Current approach: N artifact types = N tools
# Better approach: 1 general diagnostic system
```

---

## Part 4: Recommended Architecture - Consolidated Evaluation

### Principle: Few Tools, Many Capabilities

Instead of 32 specific tools, consolidate into **5-7 general-purpose tools** with configurable behavior.

### Proposed Tool Structure

```
BEFORE (32 tools):
├── analyze_prominence_shapes
├── detect_cat_ear_artifacts
├── compare_prominence_quality
├── evaluate_solar_features
├── analyze_feature_size_distribution
├── analyze_texture_procedural
├── ... (26 more)

AFTER (6 tools):
├── evaluate_render          # Primary evaluation endpoint
├── compare_renders          # A/B comparison
├── diagnose_issues          # VLM-powered diagnosis (replaces specific detectors)
├── get_reference_stats      # Reference dataset statistics
├── list_renders             # File discovery
└── train_model              # Model training (rare use)
```

### Tool 1: `evaluate_render` (Unified Evaluation)

```python
@mcp_tool
async def evaluate_render(
    render_path: str,
    reference_path: str = None,
    effect_type: str = "auto",  # auto-detect or: sun, explosion, nebula, fire
    profile: str = "standard",   # quick, standard, comprehensive
    include_diagnostics: bool = True,
    include_suggestions: bool = True
) -> EvaluationResult:
    """
    Unified render evaluation - replaces 15+ specific evaluation tools.

    Profiles:
    - quick: LPIPS + QualiCLIP only (~2 seconds)
    - standard: + TOPIQ, feature_cv, structural (~10 seconds)
    - comprehensive: + all metrics, VLM diagnosis (~30 seconds)

    Returns:
        EvaluationResult with:
        - overall_score: 0-100
        - metric_scores: {lpips, clip, topiq, structural, texture, ...}
        - diagnostics: VLM-identified issues (if include_diagnostics)
        - suggestions: Parameter changes to try (if include_suggestions)
        - pass/fail: Against thresholds
    """
```

### Tool 2: `compare_renders` (Unified Comparison)

```python
@mcp_tool
async def compare_renders(
    render_a: str,
    render_b: str,
    reference_path: str = None,
    comparison_type: str = "quality"  # quality, iteration, temporal
) -> ComparisonResult:
    """
    Compare two renders - replaces compare_vfx_iterations, compare_texture_quality, etc.

    comparison_type:
    - quality: Which is better overall?
    - iteration: Did changes improve the render?
    - temporal: Consistency across animation frames
    """
```

### Tool 3: `diagnose_issues` (VLM-Powered, Replaces Specific Detectors)

```python
@mcp_tool
async def diagnose_issues(
    render_path: str,
    reference_path: str = None,
    effect_type: str = "auto",
    known_issues: list[str] = None  # Optional hints: ["too dark", "wrong color"]
) -> DiagnosisResult:
    """
    VLM-powered issue diagnosis - replaces ALL specific artifact detectors.

    Instead of hardcoded "cat ear" detection, the VLM analyzes the image
    and identifies ANY visual issues, including ones we haven't seen before.

    Returns:
        DiagnosisResult with:
        - issues: List of identified problems with severity
        - locations: Where in the image (if applicable)
        - likely_causes: What Blender parameters might cause this
        - suggested_fixes: Specific parameter changes
    """
```

**Why VLM > Specific Detectors:**

| Aspect | Specific Detectors | VLM Diagnosis |
|--------|-------------------|---------------|
| New artifact types | Need new code | Works automatically |
| Maintenance | High (N detectors) | Low (1 system) |
| Accuracy | High for known types | Good for all types |
| Generalization | None | Excellent |
| Context needed | Minimal | More tokens |

### Tool 4: `get_reference_stats` (Consolidated)

```python
@mcp_tool
async def get_reference_stats(
    effect_type: str,
    stat_type: str = "all"  # all, distribution, features, samples
) -> ReferenceStats:
    """
    Get reference dataset statistics - consolidates multiple reference tools.
    """
```

### Tool 5: `list_renders` (Unchanged)

```python
@mcp_tool
async def list_renders(
    pattern: str = None,
    limit: int = 20
) -> list[RenderInfo]:
    """List available renders with metadata."""
```

### Tool 6: `train_model` (Rare Use)

```python
@mcp_tool
async def train_model(
    model_type: str,  # discriminator, quality_predictor
    dataset_config: dict
) -> TrainingResult:
    """Train/fine-tune evaluation models. Rarely used."""
```

---

## Part 5: VLM-Based Diagnosis System

### Replace Hardcoded Detectors with Prompted VLM

```python
class VLMDiagnostics:
    """Use Moondream or similar VLM for general issue detection."""

    DIAGNOSIS_PROMPT = """
    Analyze this VFX render of a {effect_type}.

    Compare to the reference image (if provided) and identify:
    1. Visual artifacts or quality issues
    2. Physical inaccuracies (for {effect_type})
    3. Rendering problems (noise, banding, aliasing)

    For each issue found, provide:
    - Description of the problem
    - Severity (critical/major/minor)
    - Location in image (if localized)
    - Likely Blender parameter causing it
    - Suggested fix

    Known issue types for {effect_type}:
    {known_issues_for_effect_type}

    Be specific and actionable. If the render looks good, say so.
    """

    KNOWN_ISSUES = {
        "sun": [
            "Cat-ear triangular prominences (uniform procedural noise)",
            "Missing limb darkening (edges same brightness as center)",
            "Popcorn texture (uniform feature sizes)",
            "Missing granulation (no cellular convection pattern)",
            "Wrong color temperature (should be ~5778K)",
            "Ribbon artifacts (too-regular prominence shapes)"
        ],
        "explosion": [
            "Mushroom cap clipping (domain too small)",
            "Missing smoke trails",
            "Uniform density (no turbulence)",
            "Wrong color gradient (should be hot core, cool edges)"
        ],
        # ... other effect types
    }

    async def diagnose(self, render_path: str, reference_path: str = None,
                       effect_type: str = "sun") -> DiagnosisResult:
        """
        General-purpose diagnosis that can identify ANY issue,
        including novel artifact types we haven't explicitly coded for.
        """
        prompt = self.DIAGNOSIS_PROMPT.format(
            effect_type=effect_type,
            known_issues_for_effect_type="\n".join(self.KNOWN_ISSUES.get(effect_type, []))
        )

        # VLM analyzes image with prompt
        response = await self.vlm.analyze(render_path, reference_path, prompt)

        return self._parse_diagnosis(response)
```

### Benefits of VLM Approach

1. **Zero new code for new artifacts**: VLM can identify "ribbon prominences" without a `detect_ribbon_artifacts` tool
2. **Contextual understanding**: VLM knows what a sun SHOULD look like
3. **Natural language output**: Explanations agents can use directly
4. **Continuous improvement**: Better VLMs = better diagnosis (no code changes)

### Hybrid: VLM + Statistical Validation

For critical metrics where we KNOW the discriminating feature (like feature size CV), combine VLM diagnosis with statistical validation:

```python
async def diagnose_with_validation(self, render_path: str, ...) -> DiagnosisResult:
    # VLM diagnosis (general)
    vlm_issues = await self.vlm_diagnose(render_path, ...)

    # Statistical validation (specific, high-confidence)
    feature_cv = self.analyze_feature_size_cv(render_path)

    if feature_cv < 10:  # Strong procedural signal
        # Ensure VLM caught it, or add if missed
        if not any("procedural" in issue.lower() for issue in vlm_issues):
            vlm_issues.append({
                "issue": "Procedural texture detected",
                "severity": "major",
                "evidence": f"Feature size CV = {feature_cv:.1f} (natural should be >20)",
                "source": "statistical_validation"
            })

    return vlm_issues
```

---

## Part 6: Migration Plan

### Phase 1: Add Unified Tools (Non-Breaking)

```python
# Add new consolidated tools alongside existing ones
@mcp_tool
async def evaluate_render_v2(...):  # New unified tool
    ...

# Existing tools remain functional
@mcp_tool
async def detect_cat_ear_artifacts(...):  # Still works
    # Internally calls evaluate_render_v2 with specific config
    result = await evaluate_render_v2(image_path, profile="comprehensive")
    return result.diagnostics.filter(type="cat_ear")
```

### Phase 2: Deprecation Warnings

```python
@mcp_tool
async def detect_cat_ear_artifacts(...):
    warnings.warn(
        "detect_cat_ear_artifacts is deprecated. "
        "Use diagnose_issues() instead.",
        DeprecationWarning
    )
    return await diagnose_issues(image_path, effect_type="sun")
```

### Phase 3: Remove Deprecated Tools

After agents are updated to use new tools, remove the 26 deprecated tools.

### Timeline

| Phase | Duration | Tools |
|-------|----------|-------|
| Phase 1 | 1-2 weeks | 32 existing + 6 new = 38 |
| Phase 2 | 2-4 weeks | 32 deprecated + 6 active |
| Phase 3 | After validation | 6 tools |

---

## Part 7: Model Upgrades

### Upgrade 1: Replace CLIP with SigLIP 2

```python
# requirements.txt
transformers>=4.40.0

# Implementation
from transformers import AutoModel, AutoProcessor

class SigLIPEvaluator:
    def __init__(self):
        self.model = AutoModel.from_pretrained("google/siglip2-base-patch16-224")
        self.processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")
```

**Benefits:** Better zero-shot, improved text-image alignment, more efficient.

### Upgrade 2: Add pyiqa Metrics

```python
# requirements.txt
pyiqa>=0.1.12

# Implementation
import pyiqa

class PyIQAMetrics:
    def __init__(self, device="cuda"):
        self.topiq_nr = pyiqa.create_metric('topiq_nr', device=device)
        self.topiq_fr = pyiqa.create_metric('topiq_fr', device=device)
        self.qualiclip = pyiqa.create_metric('qualiclip', device=device)
        self.maclip = pyiqa.create_metric('maclip', device=device)  # Replaces ImageReward
```

### Upgrade 3: Resolve ImageReward Conflict

**Problem:** ImageReward requires `timm==0.6.13`, Moondream needs `timm>=0.9.0`

**Solution:** Remove ImageReward, use MACLIP from pyiqa instead.

```python
# BEFORE (conflict)
# image-reward>=1.0.0  # timm==0.6.13

# AFTER (no conflict)
# Use pyiqa's MACLIP (timm>=0.9.0 compatible)
self.maclip = pyiqa.create_metric('maclip', device=device)
```

---

## Part 8: Updated Requirements

```
# Asset Evaluator MCP Server - UPGRADED January 2026

# Core
mcp[cli]>=1.0.0
python-dotenv>=1.0.0

# Image processing
Pillow>=10.0.0
numpy>=1.24.0

# PyTorch
torch>=2.0.0
torchvision>=0.15.0

# Perceptual similarity (keep)
lpips>=0.1.4

# =============================================================================
# NEW: State-of-the-art IQA (January 2026)
# =============================================================================

# PyIQA - TOPIQ, QualiCLIP, MACLIP, and 40+ more metrics
pyiqa>=0.1.12

# SigLIP 2 - Replaces CLIP (February 2025)
transformers>=4.40.0

# =============================================================================
# KEEP: VLM for diagnostics
# =============================================================================
timm>=0.9.0
einops>=0.7.0

# CLIP compatibility (SigLIP preferred for new code)
ftfy>=6.1.0
regex>=2023.0.0

# =============================================================================
# REMOVED: ImageReward (timm version conflict)
# Replacement: pyiqa's MACLIP
# =============================================================================
```

---

## Part 9: Implementation Priority

| Priority | Task | Effort | Impact |
|----------|------|--------|--------|
| **1** | Create unified `evaluate_render` tool | Medium | High |
| **2** | Implement VLM `diagnose_issues` (replaces 10+ specific detectors) | Medium | High |
| **3** | Add pyiqa integration (TOPIQ, QualiCLIP, MACLIP) | Low | High |
| **4** | Upgrade CLIP to SigLIP 2 | Medium | Medium |
| **5** | Create unified `compare_renders` tool | Low | Medium |
| **6** | Deprecate and remove old tools | Low | Maintenance |

---

## Appendix A: Tool Consolidation Map

| Old Tool(s) | New Tool | Notes |
|-------------|----------|-------|
| `evaluate_render`, `evaluate_vfx_quality`, `evaluate_ground_truth`, `evaluate_with_suggestions`, `enhanced_evaluate` | `evaluate_render` | Profile-based |
| `compare_lpips`, `compare_clip`, `compare_vfx_iterations`, `compare_texture_quality` | `compare_renders` | Type-based |
| `detect_cat_ear_artifacts`, `analyze_prominence_shapes`, `compare_prominence_quality`, `evaluate_solar_features`, `analyze_texture_procedural`, `predict_real_or_synthetic` | `diagnose_issues` | VLM-powered |
| `get_reference_statistics`, `compare_to_reference_distribution`, `extract_image_features` | `get_reference_stats` | Consolidated |
| `list_recent_renders`, `find_reference_images` | `list_renders` | Unified |
| `train_discriminator` | `train_model` | Generalized |

---

## Appendix B: Example Usage After Consolidation

### Before (Multiple Tool Calls)

```python
# Agent trying to diagnose a bad render currently needs to:
cat_ears = await detect_cat_ear_artifacts(render_path)
prominence = await analyze_prominence_shapes(render_path)
texture = await analyze_texture_procedural(render_path)
solar = await evaluate_solar_features(render_path)
quality = await evaluate_vfx_quality(render_path, "sun")
# ... potentially more calls

# Then synthesize results manually
```

### After (Single Tool Call)

```python
# One call does everything
result = await evaluate_render(
    render_path=render_path,
    reference_path=reference_path,
    effect_type="sun",
    profile="comprehensive",
    include_diagnostics=True,
    include_suggestions=True
)

# Result contains:
# - overall_score: 72
# - metrics: {lpips: 0.23, topiq: 0.81, structural: 0.65, ...}
# - diagnostics: [
#     {"issue": "Cat-ear triangular prominences", "severity": "major", ...},
#     {"issue": "Missing limb darkening", "severity": "minor", ...}
#   ]
# - suggestions: [
#     {"parameter": "turbulence", "change": "increase to 0.8", ...}
#   ]
```

---

## Conclusion

The asset-evaluator has strong foundations (especially the feature size CV metric), but suffers from tool proliferation. By consolidating 32 tools into 6 and using VLM-powered diagnosis instead of hardcoded artifact detectors, we achieve:

1. **Simpler API**: 6 tools vs 32
2. **Better generalization**: VLM can identify novel artifacts
3. **Lower maintenance**: One diagnostic system vs N detectors
4. **Modern models**: SigLIP 2, TOPIQ, pyiqa integration
5. **Resolved conflicts**: ImageReward removed, MACLIP used instead

The key insight is that **specific artifact detectors are an anti-pattern** - they require new code for each new artifact type, while VLM-based diagnosis handles unknown artifacts automatically.
