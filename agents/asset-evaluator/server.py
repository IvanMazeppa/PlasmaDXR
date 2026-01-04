#!/usr/bin/env python3
"""
Asset Evaluator MCP Server

Evaluates generated NanoVDB assets and renders against reference images
using ML-based perceptual metrics (LPIPS, CLIP).

Part of the Self-Improving NanoVDB Asset Generation Pipeline.

Tools:
    - compare_lpips: Perceptual similarity (lower = more similar)
    - compare_clip: Semantic similarity (higher = more similar)
    - evaluate_render: Combined quality scoring with pass/fail
    - find_reference_images: Search for reference images by keyword

Usage:
    python server.py  # Run as MCP server (stdio transport)
"""

import asyncio
import json
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple
import hashlib

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
import numpy as np

# Load environment
load_dotenv()

# Project paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = Path(os.getenv(
    "PROJECT_ROOT",
    SCRIPT_DIR.parent.parent
))

# Reference images directory
REFERENCE_DIR = PROJECT_ROOT / "assets/reference_images"
RENDER_OUTPUT_DIR = PROJECT_ROOT / "build/renders"

# ============================================================================
# JSON Serialization Utilities (Task 5.4: Handle numpy types)
# ============================================================================

def convert_numpy_types(obj):
    """
    Recursively convert numpy types to Python native types for JSON serialization.

    Handles: np.bool_, np.integer, np.floating, np.ndarray, nested dicts/lists.
    """
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_numpy_types(v) for v in obj]
    return obj


def safe_json_dumps(obj, **kwargs):
    """
    JSON dumps with automatic numpy type conversion.

    Use this instead of json.dumps() when serializing evaluation results
    that may contain numpy types.
    """
    return json.dumps(convert_numpy_types(obj), **kwargs)


# Create FastMCP server
mcp = FastMCP("asset-evaluator")

# Lazy-loaded ML models (heavy imports)
_lpips_model = None
_clip_model = None
_clip_preprocess = None


# =============================================================================
# Lazy Loading for Heavy Dependencies
# =============================================================================

def get_lpips_model():
    """Lazy load LPIPS model (528MB weights)."""
    global _lpips_model
    if _lpips_model is None:
        try:
            import torch
            import lpips
            _lpips_model = lpips.LPIPS(net='alex')
            if torch.cuda.is_available():
                _lpips_model = _lpips_model.cuda()
        except ImportError as e:
            raise RuntimeError(f"LPIPS not installed: {e}. Run: pip install lpips torch torchvision")
    return _lpips_model


def get_clip_model():
    """Lazy load CLIP model."""
    global _clip_model, _clip_preprocess
    if _clip_model is None:
        try:
            import torch
            import clip
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _clip_model, _clip_preprocess = clip.load("ViT-B/32", device=device)
        except ImportError as e:
            raise RuntimeError(f"CLIP not installed: {e}. Run: pip install git+https://github.com/openai/CLIP.git")
    return _clip_model, _clip_preprocess


def load_image_for_lpips(image_path: str):
    """Load and preprocess image for LPIPS comparison."""
    import torch
    from PIL import Image
    import torchvision.transforms as transforms

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    img = Image.open(image_path).convert('RGB')
    return transform(img).unsqueeze(0)


def load_image_for_clip(image_path: str, preprocess):
    """Load and preprocess image for CLIP."""
    from PIL import Image
    img = Image.open(image_path).convert('RGB')
    return preprocess(img).unsqueeze(0)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class LPIPSResult:
    """Result of LPIPS comparison."""
    similarity_score: float  # 0-1, lower = more similar
    perceptual_match: str    # "excellent", "good", "fair", "poor"
    image1_path: str
    image2_path: str
    interpretation: str


@dataclass
class CLIPResult:
    """Result of CLIP comparison."""
    similarity_score: float  # 0-1, higher = more similar
    semantic_match: str      # "excellent", "good", "fair", "poor"
    query: str               # text query or image path
    image_path: str
    interpretation: str


@dataclass
class EvaluationResult:
    """Combined evaluation result."""
    passed: bool
    overall_score: float     # 0-100
    lpips_score: Optional[float]
    clip_score: Optional[float]
    details: str
    recommendations: List[str]
    render_path: str
    reference_path: Optional[str]


# =============================================================================
# Scoring Helpers
# =============================================================================

def interpret_lpips(score: float) -> Tuple[str, str]:
    """Interpret LPIPS score (lower = better)."""
    if score < 0.1:
        return "excellent", "Nearly identical perceptually"
    elif score < 0.2:
        return "good", "Very similar with minor differences"
    elif score < 0.35:
        return "fair", "Noticeable differences but similar structure"
    elif score < 0.5:
        return "poor", "Significant perceptual differences"
    else:
        return "very_poor", "Images are perceptually very different"


def interpret_clip(score: float) -> Tuple[str, str]:
    """Interpret CLIP score (higher = better)."""
    if score > 0.85:
        return "excellent", "Strong semantic match"
    elif score > 0.70:
        return "good", "Good semantic similarity"
    elif score > 0.55:
        return "fair", "Moderate semantic similarity"
    elif score > 0.40:
        return "poor", "Weak semantic match"
    else:
        return "very_poor", "Little to no semantic similarity"


# =============================================================================
# VFX-Specific Diagnostics (Reference-Free Quality Assessment)
# =============================================================================

def _compute_edge_density(arr: np.ndarray) -> float:
    """Compute edge density using Sobel-like gradients (no OpenCV dependency)."""
    # Simple gradient magnitude using numpy
    gray = np.mean(arr, axis=-1) if len(arr.shape) == 3 else arr

    # Sobel-like kernels via numpy gradient
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))

    # Gradient magnitude
    edges = np.sqrt(gx**2 + gy**2)

    # Normalize to 0-1
    return float(np.mean(edges) / 255.0) if edges.max() > 0 else 0.0


def _detect_color_gradient(arr: np.ndarray) -> bool:
    """Detect if image has meaningful color gradients (hot-to-cool)."""
    if len(arr.shape) != 3 or arr.shape[2] < 3:
        return False

    # Check if there's variation in the red-to-blue ratio across the image
    r_channel = arr[:, :, 0].astype(float)
    b_channel = arr[:, :, 2].astype(float)

    # Compute warmth ratio per row (for radial effects)
    warmth = np.sum(r_channel, axis=1) / (np.sum(b_channel, axis=1) + 1)

    # Check if warmth varies significantly (gradient exists)
    warmth_std = np.std(warmth)
    return warmth_std > 0.1


def _compute_histogram_spread(arr: np.ndarray) -> dict:
    """Compute histogram statistics for quality assessment."""
    gray = np.mean(arr, axis=-1) if len(arr.shape) == 3 else arr

    # Simple histogram using numpy
    hist, _ = np.histogram(gray.flatten(), bins=256, range=(0, 256))
    hist = hist.astype(float) / hist.sum()

    # Find percentiles
    cumsum = np.cumsum(hist)
    p5 = np.searchsorted(cumsum, 0.05)
    p95 = np.searchsorted(cumsum, 0.95)

    return {
        "percentile_5": int(p5),
        "percentile_95": int(p95),
        "dynamic_range_90": int(p95 - p5),
        "histogram_entropy": float(-np.sum(hist[hist > 0] * np.log2(hist[hist > 0])))
    }


def extract_vfx_diagnostics_impl(image_path: str) -> dict:
    """
    Extract VFX-specific quality signals from an image.

    This function provides actionable diagnostics that LPIPS/CLIP cannot:
    - Brightness levels (is it too dark/bright?)
    - Color presence (does it have fire colors? cool colors?)
    - Coverage (does the effect fill the frame appropriately?)
    - Structure (does it have detail or is it flat?)

    Returns:
        Dictionary with diagnostic categories and measurements
    """
    from PIL import Image

    img = Image.open(image_path).convert('RGB')
    arr = np.array(img, dtype=np.float32)

    # Basic brightness analysis
    brightness = {
        "mean": float(np.mean(arr)),
        "max": float(np.max(arr)),
        "min": float(np.min(arr)),
        "dynamic_range": float(np.max(arr) - np.min(arr)),
        "std": float(np.std(arr))
    }

    # Color analysis (critical for VFX)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    color_presence = {
        # Fire/explosion colors (red > blue)
        "warm_ratio": float(np.sum(r) / max(np.sum(b), 1)),
        "has_orange": float(np.mean((r > 180) & (g > 100) & (g < 200) & (b < 100))),
        "has_yellow": float(np.mean((r > 200) & (g > 180) & (b < 150))),
        "has_red": float(np.mean((r > 180) & (g < 100) & (b < 100))),
        "has_blue": float(np.mean((b > 150) & (r < 100))),
        # Color gradient detection
        "has_gradient": _detect_color_gradient(arr)
    }

    # Coverage analysis
    arr_int = arr.astype(np.uint8)
    coverage = {
        "non_black_ratio": float(np.mean(np.any(arr_int > 15, axis=-1))),
        "bright_pixel_ratio": float(np.mean(np.any(arr_int > 200, axis=-1))),
        "very_bright_ratio": float(np.mean(np.any(arr_int > 240, axis=-1))),
        "dark_pixel_ratio": float(np.mean(np.all(arr_int < 30, axis=-1)))
    }

    # Structure analysis
    structure = {
        "edge_density": _compute_edge_density(arr_int),
        "variance": float(np.var(arr)),
        "local_contrast": float(np.std(arr[:, :, 0]) + np.std(arr[:, :, 1]) + np.std(arr[:, :, 2])) / 3
    }

    # Histogram analysis
    histogram = _compute_histogram_spread(arr_int)

    return {
        "brightness": brightness,
        "color_presence": color_presence,
        "coverage": coverage,
        "structure": structure,
        "histogram": histogram,
        "image_size": {"width": img.width, "height": img.height}
    }


def compute_vfx_quality_score(diagnostics: dict, target_effect: str) -> dict:
    """
    Compute a composite VFX quality score (0-100) based on diagnostics.

    Unlike LPIPS (lower=better, requires reference), this score:
    - Higher = better
    - No reference image required
    - Provides per-dimension breakdown
    - Returns actionable issues

    Args:
        diagnostics: Output from extract_vfx_diagnostics_impl()
        target_effect: Effect type (explosion, fire, sun, supernova, nebula, smoke)

    Returns:
        Dictionary with composite score, dimension scores, and identified issues
    """
    scores = {}
    issues = []
    max_score = 0
    total_score = 0

    brightness = diagnostics.get("brightness", {})
    color = diagnostics.get("color_presence", {})
    coverage = diagnostics.get("coverage", {})
    structure = diagnostics.get("structure", {})

    # === Brightness Score (25 points) ===
    max_score += 25
    mean_brightness = brightness.get("mean", 0)
    dynamic_range = brightness.get("dynamic_range", 0)

    # Effect-specific brightness expectations
    brightness_targets = {
        "explosion": (80, 180),  # Should be bright
        "fire": (60, 160),
        "sun": (100, 220),  # Very bright core
        "supernova": (90, 200),
        "nebula": (40, 120),  # Can be dimmer
        "smoke": (30, 100)  # Darker
    }
    target_min, target_max = brightness_targets.get(target_effect, (50, 150))

    if mean_brightness < target_min * 0.5:
        issues.append(f"TOO DARK: mean brightness {mean_brightness:.0f} << target {target_min}")
        scores["brightness"] = 5
    elif mean_brightness < target_min:
        issues.append(f"Somewhat dark: mean brightness {mean_brightness:.0f} < target {target_min}")
        scores["brightness"] = 15
    elif mean_brightness > target_max * 1.3:
        issues.append(f"Overexposed: mean brightness {mean_brightness:.0f} >> target {target_max}")
        scores["brightness"] = 10
    else:
        scores["brightness"] = 25

    total_score += scores["brightness"]

    # === Dynamic Range Score (15 points) ===
    max_score += 15
    if dynamic_range < 50:
        issues.append(f"LOW CONTRAST: dynamic range {dynamic_range:.0f} (needs > 100)")
        scores["dynamic_range"] = 3
    elif dynamic_range < 100:
        issues.append(f"Could use more contrast: dynamic range {dynamic_range:.0f}")
        scores["dynamic_range"] = 10
    else:
        scores["dynamic_range"] = 15

    total_score += scores["dynamic_range"]

    # === Color Appropriateness Score (25 points) ===
    max_score += 25
    warm_effects = {"explosion", "fire", "sun", "supernova"}
    cool_effects = {"nebula"}

    warm_ratio = color.get("warm_ratio", 1)
    has_orange = color.get("has_orange", 0)
    has_gradient = color.get("has_gradient", False)

    if target_effect in warm_effects:
        if warm_ratio < 1.0:
            issues.append(f"WRONG COLOR: warm_ratio {warm_ratio:.2f} (should be > 1.5 for {target_effect})")
            scores["color"] = 5
        elif warm_ratio < 1.5:
            issues.append(f"Needs more warm color: warm_ratio {warm_ratio:.2f}")
            scores["color"] = 15
        elif has_orange < 0.01 and target_effect in {"explosion", "fire"}:
            issues.append("Lacks orange/fire tones")
            scores["color"] = 18
        else:
            scores["color"] = 25
    elif target_effect in cool_effects:
        if warm_ratio > 1.5:
            issues.append(f"Too warm for nebula: warm_ratio {warm_ratio:.2f}")
            scores["color"] = 10
        else:
            scores["color"] = 25
    else:
        scores["color"] = 20  # Neutral for unknown types

    # Bonus for color gradient
    if has_gradient:
        scores["color"] = min(25, scores["color"] + 3)

    total_score += scores["color"]

    # === Coverage Score (15 points) ===
    max_score += 15
    non_black = coverage.get("non_black_ratio", 0)
    bright_ratio = coverage.get("bright_pixel_ratio", 0)

    # Effect-specific coverage expectations
    coverage_targets = {
        "explosion": (0.15, 0.5),  # 15-50% of frame
        "fire": (0.1, 0.4),
        "sun": (0.2, 0.6),  # Larger
        "supernova": (0.2, 0.7),
        "nebula": (0.3, 0.8),  # Can fill more
        "smoke": (0.2, 0.6)
    }
    target_coverage = coverage_targets.get(target_effect, (0.15, 0.5))

    if non_black < target_coverage[0] * 0.5:
        issues.append(f"TOO SMALL: only {non_black*100:.1f}% coverage (target: {target_coverage[0]*100:.0f}-{target_coverage[1]*100:.0f}%)")
        scores["coverage"] = 3
    elif non_black < target_coverage[0]:
        issues.append(f"Effect is small: {non_black*100:.1f}% coverage")
        scores["coverage"] = 10
    elif non_black > target_coverage[1] * 1.3:
        issues.append(f"Effect fills too much frame: {non_black*100:.1f}%")
        scores["coverage"] = 10
    else:
        scores["coverage"] = 15

    total_score += scores["coverage"]

    # === Structure Score (20 points) ===
    max_score += 20
    edge_density = structure.get("edge_density", 0)
    variance = structure.get("variance", 0)

    if edge_density < 0.02:
        issues.append(f"NO STRUCTURE: edge density {edge_density:.4f} (needs > 0.05)")
        scores["structure"] = 3
    elif edge_density < 0.05:
        issues.append(f"Lacks detail: edge density {edge_density:.4f}")
        scores["structure"] = 12
    else:
        scores["structure"] = 20

    # Bonus for high variance (indicates texture)
    if variance > 2000:
        scores["structure"] = min(20, scores["structure"] + 2)

    total_score += scores["structure"]

    # === Compute Final Score ===
    composite_score = (total_score / max_score) * 100

    # Determine pass/fail
    passed = composite_score >= 60 and len([i for i in issues if "TOO" in i or "NO " in i or "WRONG" in i]) == 0

    return {
        "composite_score": round(composite_score, 1),
        "passed": passed,
        "dimension_scores": scores,
        "max_possible": max_score,
        "issues": issues,
        "issue_count": len(issues),
        "critical_issues": [i for i in issues if "TOO" in i or "NO " in i or "WRONG" in i],
        "target_effect": target_effect
    }


# =============================================================================
# MCP Tools
# =============================================================================

@mcp.tool()
async def compare_lpips(
    image1_path: str,
    image2_path: str,
    generate_heatmap: bool = False
) -> str:
    """
    Compare two images using LPIPS perceptual similarity.

    LPIPS (Learned Perceptual Image Patch Similarity) correlates ~92% with
    human perceptual judgments. Lower scores = more similar.

    Args:
        image1_path: Path to first image (render or reference)
        image2_path: Path to second image (render or reference)
        generate_heatmap: Generate visual difference heatmap (slower)

    Returns:
        JSON with LPIPSResult containing similarity score and interpretation

    Score interpretation:
        < 0.1:  Excellent - nearly identical
        < 0.2:  Good - very similar
        < 0.35: Fair - noticeable differences
        < 0.5:  Poor - significant differences
        >= 0.5: Very poor - very different

    Example:
        compare_lpips(
            "build/renders/nebula_001.png",
            "assets/reference_images/nebula/hubble_crab.jpg"
        )
    """
    import torch

    # Validate paths
    img1 = Path(image1_path)
    img2 = Path(image2_path)

    if not img1.is_absolute():
        img1 = PROJECT_ROOT / image1_path
    if not img2.is_absolute():
        img2 = PROJECT_ROOT / image2_path

    if not img1.exists():
        return json.dumps({"error": f"Image not found: {img1}"})
    if not img2.exists():
        return json.dumps({"error": f"Image not found: {img2}"})

    try:
        model = get_lpips_model()

        # Load images
        tensor1 = load_image_for_lpips(str(img1))
        tensor2 = load_image_for_lpips(str(img2))

        if torch.cuda.is_available():
            tensor1 = tensor1.cuda()
            tensor2 = tensor2.cuda()

        # Compute LPIPS
        with torch.no_grad():
            distance = model(tensor1, tensor2)

        score = float(distance.item())
        match_quality, interpretation = interpret_lpips(score)

        result = LPIPSResult(
            similarity_score=round(score, 4),
            perceptual_match=match_quality,
            image1_path=str(img1),
            image2_path=str(img2),
            interpretation=interpretation
        )

        return json.dumps(asdict(result), indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def compare_clip(
    image_path: str,
    query: str,
    query_is_image: bool = False
) -> str:
    """
    Compare image to text description or another image using CLIP.

    CLIP measures semantic similarity - does the image match the concept?
    Higher scores = better match.

    Args:
        image_path: Path to image to evaluate
        query: Text description (e.g., "a glowing nebula with stars") or image path
        query_is_image: If True, query is an image path for image-to-image comparison

    Returns:
        JSON with CLIPResult containing similarity score and interpretation

    Score interpretation:
        > 0.85: Excellent - strong semantic match
        > 0.70: Good - good semantic similarity
        > 0.55: Fair - moderate similarity
        > 0.40: Poor - weak match
        <= 0.40: Very poor - little similarity

    Example (text query):
        compare_clip(
            "build/renders/supernova_001.png",
            "a bright stellar explosion with expanding shockwave"
        )

    Example (image comparison):
        compare_clip(
            "build/renders/nebula_001.png",
            "assets/reference_images/nebula/reference.jpg",
            query_is_image=True
        )
    """
    import torch

    # Validate image path
    img_path = Path(image_path)
    if not img_path.is_absolute():
        img_path = PROJECT_ROOT / image_path

    if not img_path.exists():
        return json.dumps({"error": f"Image not found: {img_path}"})

    try:
        model, preprocess = get_clip_model()
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load target image
        image_tensor = load_image_for_clip(str(img_path), preprocess).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_tensor)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            if query_is_image:
                # Image-to-image comparison
                query_path = Path(query)
                if not query_path.is_absolute():
                    query_path = PROJECT_ROOT / query

                if not query_path.exists():
                    return json.dumps({"error": f"Query image not found: {query_path}"})

                query_tensor = load_image_for_clip(str(query_path), preprocess).to(device)
                query_features = model.encode_image(query_tensor)
            else:
                # Text-to-image comparison
                import clip
                text_tokens = clip.tokenize([query]).to(device)
                query_features = model.encode_text(text_tokens)

            query_features /= query_features.norm(dim=-1, keepdim=True)

            # Cosine similarity
            similarity = (image_features @ query_features.T).item()

        # CLIP similarity is typically in [-1, 1], normalize to [0, 1]
        score = (similarity + 1) / 2
        match_quality, interpretation = interpret_clip(score)

        result = CLIPResult(
            similarity_score=round(score, 4),
            semantic_match=match_quality,
            query=query,
            image_path=str(img_path),
            interpretation=interpretation
        )

        return json.dumps(asdict(result), indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def evaluate_render(
    render_path: str,
    reference_path: Optional[str] = None,
    semantic_query: Optional[str] = None,
    lpips_threshold: float = 0.35,
    clip_threshold: float = 0.60,
    require_both: bool = False
) -> str:
    """
    Comprehensive evaluation of a render against quality thresholds.

    Can use reference image (LPIPS), semantic query (CLIP), or both.
    Returns pass/fail with detailed scoring and recommendations.

    Args:
        render_path: Path to rendered image to evaluate
        reference_path: Optional reference image for LPIPS comparison
        semantic_query: Optional text description for CLIP comparison
        lpips_threshold: Max LPIPS score to pass (default 0.35 = fair)
        clip_threshold: Min CLIP score to pass (default 0.60 = fair+)
        require_both: If True, must pass both LPIPS and CLIP (default: pass either)

    Returns:
        JSON with EvaluationResult containing pass/fail, scores, recommendations

    Example:
        evaluate_render(
            "build/renders/hydrogen_cloud_001.png",
            reference_path="assets/reference_images/nebula/emission_nebula.jpg",
            semantic_query="a glowing hydrogen gas cloud with red emission",
            lpips_threshold=0.30,
            clip_threshold=0.65
        )
    """
    render = Path(render_path)
    if not render.is_absolute():
        render = PROJECT_ROOT / render_path

    if not render.exists():
        return json.dumps({"error": f"Render not found: {render}"})

    lpips_score = None
    clip_score = None
    lpips_passed = True
    clip_passed = True
    recommendations = []

    # LPIPS evaluation
    if reference_path:
        lpips_result = await compare_lpips(str(render), reference_path)
        lpips_data = json.loads(lpips_result)

        if "error" in lpips_data:
            recommendations.append(f"LPIPS failed: {lpips_data['error']}")
        else:
            lpips_score = lpips_data["similarity_score"]
            lpips_passed = lpips_score <= lpips_threshold

            if not lpips_passed:
                recommendations.append(
                    f"LPIPS {lpips_score:.3f} exceeds threshold {lpips_threshold}. "
                    f"Render differs significantly from reference."
                )
                if lpips_score > 0.5:
                    recommendations.append("Consider: major structural changes needed")
                elif lpips_score > 0.35:
                    recommendations.append("Consider: adjust density, lighting, or color balance")

    # CLIP evaluation
    if semantic_query:
        clip_result = await compare_clip(str(render), semantic_query)
        clip_data = json.loads(clip_result)

        if "error" in clip_data:
            recommendations.append(f"CLIP failed: {clip_data['error']}")
        else:
            clip_score = clip_data["similarity_score"]
            clip_passed = clip_score >= clip_threshold

            if not clip_passed:
                recommendations.append(
                    f"CLIP {clip_score:.3f} below threshold {clip_threshold}. "
                    f"Render doesn't match semantic description well."
                )
                if clip_score < 0.40:
                    recommendations.append("Consider: fundamental changes to match description")
                elif clip_score < 0.55:
                    recommendations.append("Consider: enhance key visual features in description")

    # Determine pass/fail
    if require_both:
        passed = lpips_passed and clip_passed
    else:
        # Pass if either metric passes (or wasn't tested)
        if reference_path and semantic_query:
            passed = lpips_passed or clip_passed
        elif reference_path:
            passed = lpips_passed
        elif semantic_query:
            passed = clip_passed
        else:
            passed = True  # No metrics to test
            recommendations.append("No reference or query provided - cannot evaluate")

    # Calculate overall score (0-100)
    scores = []
    if lpips_score is not None:
        # Invert LPIPS (lower is better) and scale to 0-100
        scores.append(max(0, (1 - lpips_score / 0.5)) * 100)
    if clip_score is not None:
        scores.append(clip_score * 100)

    overall_score = sum(scores) / len(scores) if scores else 0

    # Build details string
    details_parts = []
    if lpips_score is not None:
        status = "PASS" if lpips_passed else "FAIL"
        details_parts.append(f"LPIPS: {lpips_score:.3f} ({status}, threshold: {lpips_threshold})")
    if clip_score is not None:
        status = "PASS" if clip_passed else "FAIL"
        details_parts.append(f"CLIP: {clip_score:.3f} ({status}, threshold: {clip_threshold})")

    result = EvaluationResult(
        passed=passed,
        overall_score=round(overall_score, 1),
        lpips_score=lpips_score,
        clip_score=clip_score,
        details=" | ".join(details_parts),
        recommendations=recommendations if not passed else ["Quality acceptable"],
        render_path=str(render),
        reference_path=reference_path
    )

    return json.dumps(asdict(result), indent=2)


@mcp.tool()
async def find_reference_images(
    keyword: str,
    limit: int = 10
) -> str:
    """
    Search for reference images in the project by keyword.

    Searches in assets/reference_images/ directory.

    Args:
        keyword: Search keyword (e.g., "nebula", "explosion", "star")
        limit: Maximum number of results (default 10)

    Returns:
        JSON array of matching image paths with metadata

    Example:
        find_reference_images("supernova")
    """
    results = []
    keyword_lower = keyword.lower()

    # Ensure reference directory exists
    if not REFERENCE_DIR.exists():
        REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
        return json.dumps({
            "results": [],
            "message": f"Reference directory created at {REFERENCE_DIR}. Add reference images here."
        })

    # Search for matching images
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.exr'}

    for img_path in REFERENCE_DIR.rglob("*"):
        if img_path.suffix.lower() in image_extensions:
            # Check if keyword in path or filename
            if keyword_lower in str(img_path).lower():
                # Get file info
                stat = img_path.stat()
                results.append({
                    "path": str(img_path.relative_to(PROJECT_ROOT)),
                    "filename": img_path.name,
                    "category": img_path.parent.name if img_path.parent != REFERENCE_DIR else "uncategorized",
                    "size_kb": round(stat.st_size / 1024, 1),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })

                if len(results) >= limit:
                    break

    # Sort by modification time (newest first)
    results.sort(key=lambda x: x["modified"], reverse=True)

    return json.dumps({
        "keyword": keyword,
        "count": len(results),
        "results": results
    }, indent=2)


@mcp.tool()
async def list_recent_renders(
    limit: int = 20,
    pattern: Optional[str] = None
) -> str:
    """
    List recent renders from the build output directory.

    Args:
        limit: Maximum number of results (default 20)
        pattern: Optional filename pattern to filter (e.g., "nebula", "explosion")

    Returns:
        JSON array of recent render paths with metadata
    """
    results = []
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.exr'}

    # Search in render output and blender CLI logs
    search_dirs = [
        RENDER_OUTPUT_DIR,
        PROJECT_ROOT / "build/blender_cli_logs",
        PROJECT_ROOT / "build/vdb_output"
    ]

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue

        for img_path in search_dir.rglob("*"):
            if img_path.suffix.lower() in image_extensions:
                # Apply pattern filter if specified
                if pattern and pattern.lower() not in str(img_path).lower():
                    continue

                stat = img_path.stat()
                results.append({
                    "path": str(img_path),
                    "filename": img_path.name,
                    "directory": str(img_path.parent.relative_to(PROJECT_ROOT)),
                    "size_kb": round(stat.st_size / 1024, 1),
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                })

    # Sort by modification time (newest first)
    results.sort(key=lambda x: x["modified"], reverse=True)
    results = results[:limit]

    return json.dumps({
        "count": len(results),
        "pattern": pattern,
        "results": results
    }, indent=2)


# =============================================================================
# Enhanced Evaluation Tools
# =============================================================================

@mcp.tool()
async def extract_vfx_diagnostics(
    image_path: str
) -> str:
    """
    Extract VFX-specific quality diagnostics from an image (NO REFERENCE NEEDED).

    Unlike LPIPS/CLIP which require reference images or plateau on VFX content,
    this provides actionable diagnostics:
    - Brightness: Is it too dark/bright for the effect type?
    - Color: Does it have appropriate colors (warm for fire, cool for nebula)?
    - Coverage: Does the effect fill the frame appropriately?
    - Structure: Does it have detail or is it flat/blobby?

    Args:
        image_path: Path to rendered image to analyze

    Returns:
        JSON with detailed diagnostics:
        - brightness: mean, max, min, dynamic_range, std
        - color_presence: warm_ratio, has_orange, has_yellow, has_gradient
        - coverage: non_black_ratio, bright_pixel_ratio
        - structure: edge_density, variance, local_contrast
        - histogram: percentiles, entropy

    Example:
        extract_vfx_diagnostics("build/renders/explosion_001.png")
    """
    img_path = Path(image_path)
    if not img_path.is_absolute():
        img_path = PROJECT_ROOT / image_path

    if not img_path.exists():
        return json.dumps({"error": f"Image not found: {img_path}"})

    try:
        diagnostics = extract_vfx_diagnostics_impl(str(img_path))
        return safe_json_dumps({
            "success": True,
            "image_path": str(img_path),
            "diagnostics": diagnostics
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def evaluate_vfx_quality(
    image_path: str,
    effect_type: str = "explosion"
) -> str:
    """
    Compute VFX quality score (0-100) with actionable issues (NO REFERENCE NEEDED).

    This is the PRIMARY evaluation tool for VFX. Unlike LPIPS which:
    - Requires a reference image
    - Scores 0.7-0.8 for all VFX regardless of quality
    - Provides no actionable feedback

    This tool:
    - Works standalone (no reference image)
    - Scores 0-100 (higher = better)
    - Returns specific issues: "TOO DARK", "WRONG COLOR", "NO STRUCTURE"
    - Suggests what to fix

    Args:
        image_path: Path to rendered image to evaluate
        effect_type: Effect category - one of:
            explosion, fire, sun, supernova, nebula, smoke

    Returns:
        JSON with:
        - composite_score: 0-100 overall score
        - passed: True if score >= 60 and no critical issues
        - dimension_scores: Per-category breakdown
        - issues: List of problems found
        - critical_issues: Severe problems that need fixing

    Example:
        evaluate_vfx_quality("build/renders/sun_v10.png", "sun")

    Score interpretation:
        >= 80: Excellent - ready to use
        >= 60: Good - minor improvements possible
        >= 40: Fair - significant issues to address
        < 40: Poor - major rework needed
    """
    img_path = Path(image_path)
    if not img_path.is_absolute():
        img_path = PROJECT_ROOT / image_path

    if not img_path.exists():
        return json.dumps({"error": f"Image not found: {img_path}"})

    valid_effects = {"explosion", "fire", "sun", "supernova", "nebula", "smoke"}
    if effect_type not in valid_effects:
        return json.dumps({
            "error": f"Unknown effect_type: {effect_type}",
            "valid_types": list(valid_effects)
        })

    try:
        # Extract diagnostics
        diagnostics = extract_vfx_diagnostics_impl(str(img_path))

        # Compute quality score
        quality = compute_vfx_quality_score(diagnostics, effect_type)

        # Generate recommendations based on issues
        recommendations = []
        for issue in quality.get("issues", []):
            if "TOO DARK" in issue:
                recommendations.append("Increase flame_max_temp or emission_multiplier")
            elif "TOO SMALL" in issue:
                recommendations.append("Increase domain size or flow radius")
            elif "WRONG COLOR" in issue or "warm" in issue.lower():
                recommendations.append("Increase burning_rate or adjust flame_smoke ratio")
            elif "NO STRUCTURE" in issue:
                recommendations.append("Increase turbulence/vorticity or add more noise")
            elif "CONTRAST" in issue:
                recommendations.append("Increase temperature range or density variation")

        return safe_json_dumps({
            "success": True,
            "image_path": str(img_path),
            "effect_type": effect_type,
            "quality": quality,
            "diagnostics_summary": {
                "mean_brightness": round(diagnostics["brightness"]["mean"], 1),
                "dynamic_range": round(diagnostics["brightness"]["dynamic_range"], 1),
                "warm_ratio": round(diagnostics["color_presence"]["warm_ratio"], 2),
                "coverage_percent": round(diagnostics["coverage"]["non_black_ratio"] * 100, 1),
                "edge_density": round(diagnostics["structure"]["edge_density"], 4)
            },
            "recommendations": recommendations if not quality["passed"] else ["Quality acceptable - ready to use"]
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def compare_vfx_iterations(
    image_a_path: str,
    image_b_path: str,
    effect_type: str = "explosion"
) -> str:
    """
    Compare two VFX iterations and determine which is BETTER.

    Unlike LPIPS which compares to a reference, this compares two
    candidate renders and tells you which one is higher quality.

    Args:
        image_a_path: Path to first render (e.g., iteration v9)
        image_b_path: Path to second render (e.g., iteration v10)
        effect_type: Effect category (explosion, fire, sun, etc.)

    Returns:
        JSON with:
        - winner: "A" or "B"
        - score_a, score_b: Individual quality scores
        - improvements: What got better in the winner
        - regressions: What got worse
        - recommendation: What to do next

    Example:
        compare_vfx_iterations(
            "build/renders/sun_v9.png",
            "build/renders/sun_v10.png",
            "sun"
        )
    """
    path_a = Path(image_a_path)
    path_b = Path(image_b_path)

    if not path_a.is_absolute():
        path_a = PROJECT_ROOT / image_a_path
    if not path_b.is_absolute():
        path_b = PROJECT_ROOT / image_b_path

    if not path_a.exists():
        return json.dumps({"error": f"Image A not found: {path_a}"})
    if not path_b.exists():
        return json.dumps({"error": f"Image B not found: {path_b}"})

    try:
        # Analyze both images
        diag_a = extract_vfx_diagnostics_impl(str(path_a))
        diag_b = extract_vfx_diagnostics_impl(str(path_b))

        quality_a = compute_vfx_quality_score(diag_a, effect_type)
        quality_b = compute_vfx_quality_score(diag_b, effect_type)

        score_a = quality_a["composite_score"]
        score_b = quality_b["composite_score"]

        # Determine winner
        winner = "B" if score_b > score_a else "A"
        score_diff = abs(score_b - score_a)

        # Identify what changed
        improvements = []
        regressions = []

        dim_a = quality_a["dimension_scores"]
        dim_b = quality_b["dimension_scores"]

        for dim in dim_a:
            diff = dim_b.get(dim, 0) - dim_a.get(dim, 0)
            if diff > 2:
                improvements.append(f"{dim}: +{diff:.0f} pts")
            elif diff < -2:
                regressions.append(f"{dim}: {diff:.0f} pts")

        # Recommendation
        if score_diff < 3:
            recommendation = "Marginal difference - try a different parameter"
        elif max(score_a, score_b) >= 75:
            recommendation = f"Keep {winner} - quality is good"
        elif max(score_a, score_b) >= 60:
            recommendation = f"Keep {winner} - address remaining issues"
        else:
            recommendation = "Both need significant improvement"

        return safe_json_dumps({
            "winner": winner,
            "score_a": score_a,
            "score_b": score_b,
            "score_difference": round(score_diff, 1),
            "improvements": improvements,
            "regressions": regressions,
            "issues_a": quality_a.get("critical_issues", []),
            "issues_b": quality_b.get("critical_issues", []),
            "recommendation": recommendation
        }, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def enhanced_evaluate(
    render_path: str,
    semantic_query: str,
    effect_type: str = "explosion",
    reference_path: Optional[str] = None,
    lpips_threshold: float = 0.35,
    clip_threshold: float = 0.60,
    aesthetic_threshold: float = 5.5
) -> str:
    """
    Comprehensive VFX evaluation with multi-modal metrics and actionable diagnostics.

    Addresses all ML evaluation challenges:
    - Domain mismatch: Uses aesthetic scoring + ImageReward
    - CLIP plateau: Multi-prompt gradient signal for fine-grained quality
    - No actionable feedback: VLM diagnostics with Blender parameter suggestions
    - Reference dependency: Aesthetic scoring works without reference

    Combines:
    - LPIPS: Perceptual similarity (if reference provided)
    - Multi-prompt CLIP: Fine-grained quality gradient (not just pass/fail)
    - LAION Aesthetics: Reference-free aesthetic quality (1-10 scale)
    - ImageReward: Human preference alignment
    - Moondream VLM: Structured diagnostics with parameter suggestions

    Args:
        render_path: Path to rendered image to evaluate
        semantic_query: Text description (e.g., "a bright supernova explosion")
        effect_type: Effect category (explosion, pyro, nebula, supernova)
        reference_path: Optional reference image for LPIPS
        lpips_threshold: Max LPIPS to pass (default 0.35)
        clip_threshold: Min CLIP to pass (default 0.60)
        aesthetic_threshold: Min aesthetic score to pass (default 5.5, scale 1-10)

    Returns:
        JSON with comprehensive evaluation including:
        - Multi-metric scores (LPIPS, CLIP, aesthetic, gradient_signal)
        - Structured diagnostics (color_temperature, density, brightness, etc.)
        - Suggested Blender parameters (flame_max_temp, vorticity, etc.)
        - Specific recommendations for improvement

    Example:
        enhanced_evaluate(
            "build/renders/supernova_001.png",
            "a bright stellar explosion with expanding shockwave",
            effect_type="supernova"
        )
    """
    try:
        from enhanced_evaluation import enhanced_evaluate_render
        return await enhanced_evaluate_render(
            render_path=render_path,
            semantic_query=semantic_query,
            effect_type=effect_type,
            reference_path=reference_path,
            lpips_threshold=lpips_threshold,
            clip_threshold=clip_threshold,
            aesthetic_threshold=aesthetic_threshold
        )
    except ImportError as e:
        return json.dumps({
            "error": f"Enhanced evaluation module not available: {e}",
            "fallback": "Use standard evaluate_render tool instead"
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def analyze_temporal_quality(
    frame_directory: str,
    frame_pattern: str = "*.png",
    sample_rate: int = 5
) -> str:
    """
    Analyze temporal consistency across animation frames.

    Detects flickering, motion smoothness, and temporal artifacts
    that single-frame evaluation misses.

    Args:
        frame_directory: Directory containing animation frames
        frame_pattern: Glob pattern for frames (default "*.png")
        sample_rate: Analyze every Nth frame (default 5)

    Returns:
        JSON with temporal quality metrics:
        - temporal_consistency: 0-1 score (higher = smoother)
        - flicker_risk: low/medium/high
        - average_motion: Frame-to-frame change magnitude
        - static_frame_count: Frames with little motion
        - recommendations: Specific improvement suggestions

    Example:
        analyze_temporal_quality(
            "build/vdb_output/explosion/renders",
            "frame_*.png",
            sample_rate=3
        )
    """
    try:
        from enhanced_evaluation import analyze_temporal_quality as analyze_temp

        # Get frame paths
        frame_dir = Path(frame_directory)
        if not frame_dir.is_absolute():
            frame_dir = PROJECT_ROOT / frame_directory

        if not frame_dir.exists():
            return json.dumps({"error": f"Directory not found: {frame_dir}"})

        frame_paths = sorted([str(p) for p in frame_dir.glob(frame_pattern)])

        if len(frame_paths) < 2:
            return json.dumps({"error": f"Need at least 2 frames, found {len(frame_paths)}"})

        result = analyze_temp(frame_paths, sample_rate)
        return json.dumps(result, indent=2)

    except ImportError as e:
        return json.dumps({"error": f"Enhanced evaluation module not available: {e}"})
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def get_alternative_approaches(
    current_approach: str,
    scores_history: str
) -> str:
    """
    Suggest alternative approaches when stuck in local optima.

    Detects score plateaus and suggests fundamentally different
    approaches to escape local optima.

    Args:
        current_approach: Description of current approach (e.g., "sphere_emitter")
        scores_history: JSON array of previous iteration scores
            e.g., '[{"iteration": 1, "overall_score": 45}, ...]'

    Returns:
        JSON with alternative approach suggestions:
        - reason: Why suggesting change
        - suggestion: What to try
        - options: Specific alternatives
        - expected_benefit: What improvement to expect

    Example:
        get_alternative_approaches(
            "sphere_emitter with high turbulence",
            '[{"iteration": 1, "score": 52}, {"iteration": 2, "score": 53}, {"iteration": 3, "score": 52.5}]'
        )
    """
    try:
        from enhanced_evaluation import suggest_alternative_approaches

        history = json.loads(scores_history)
        alternatives = suggest_alternative_approaches(current_approach, history)

        return json.dumps({
            "current_approach": current_approach,
            "iterations_analyzed": len(history),
            "alternatives": alternatives
        }, indent=2)

    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid scores_history JSON format"})
    except ImportError as e:
        return json.dumps({"error": f"Enhanced evaluation module not available: {e}"})
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
async def multi_prompt_clip_analysis(
    image_path: str,
    base_description: str,
    effect_type: str = "explosion"
) -> str:
    """
    Fine-grained CLIP analysis using graduated quality prompts.

    Unlike standard CLIP which plateaus at "is this an explosion?",
    this uses multiple prompts from "faint wisp" to "dramatic fireball"
    to provide gradient signal for iteration.

    Args:
        image_path: Path to image to analyze
        base_description: Base semantic description
        effect_type: Effect type (explosion, pyro, nebula, supernova)

    Returns:
        JSON with:
        - overall_match: Standard CLIP score
        - gradient_signal: Composite score with more range
        - quality_dimension_scores: Per-dimension scores (intensity, realism, etc.)
        - best_matching_description: Which quality prompt matches best
        - worst_matching_description: Which matches worst

    Example:
        multi_prompt_clip_analysis(
            "build/renders/explosion_001.png",
            "a bright explosion with flames",
            "explosion"
        )
    """
    try:
        from enhanced_evaluation import evaluate_multi_prompt_clip
        from dataclasses import asdict

        img_path = Path(image_path)
        if not img_path.is_absolute():
            img_path = PROJECT_ROOT / image_path

        if not img_path.exists():
            return json.dumps({"error": f"Image not found: {img_path}"})

        result = evaluate_multi_prompt_clip(str(img_path), base_description, effect_type)
        return json.dumps(asdict(result), indent=2)

    except ImportError as e:
        return json.dumps({"error": f"Enhanced evaluation module not available: {e}"})
    except Exception as e:
        return json.dumps({"error": str(e)})


# =============================================================================
# Ground Truth Evaluation Tools
# =============================================================================

@mcp.tool()
def evaluate_ground_truth(
    image_path: str,
    effect_type: str = "sun",
    pass_threshold: float = 0.65
) -> str:
    """
    Evaluate render against REAL reference footage distribution.

    This is the PRIMARY evaluation tool for realism. Unlike LPIPS/CLIP which
    plateau on VFX content, this compares your render's feature distributions
    against 840 frames of actual solar footage.

    The key insight: The old system rewards "not obviously broken" (threshold-based).
    This system rewards "looks like real footage" (distribution-based).

    Why this matters:
    - Real sun footage scored 84/100 on old system
    - Synthetic renders scored 94/100 on old system
    - This was BACKWARDS - the old system penalized real footage!
    - This new system correctly ranks real footage higher

    Args:
        image_path: Path to rendered image to evaluate
        effect_type: Effect category - "sun", "star" (more types coming)
        pass_threshold: Minimum similarity score to pass (0-1, default 0.65)

    Returns:
        JSON with:
        - overall_score: 0-100 (higher = more like real footage)
        - passed: True if meets threshold
        - distribution_comparison: How each feature matches reference
        - solar_specific: Domain metrics (granulation, limb darkening, etc.)
        - recommendations: What to improve

    Example:
        evaluate_ground_truth("build/renders/sun_v10.png", "sun", 0.7)
    """
    try:
        from ground_truth_evaluation import evaluate_against_ground_truth
        import numpy as np

        img_path = Path(image_path)
        if not img_path.is_absolute():
            img_path = PROJECT_ROOT / image_path

        if not img_path.exists():
            return json.dumps({"error": f"Image not found: {img_path}"})

        result = evaluate_against_ground_truth(str(img_path), effect_type, pass_threshold)
        # Convert numpy scalar/array types into JSON-serializable Python types.
        def convert_np(obj):
            if isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            if isinstance(obj, (np.integer, int)):
                return int(obj)
            if isinstance(obj, (np.floating, float)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: convert_np(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_np(v) for v in obj]
            return obj

        return json.dumps(convert_np(result), indent=2)

    except ImportError as e:
        return json.dumps({
            "error": f"Ground truth evaluation module not available: {e}",
            "hint": "Ensure ground_truth_evaluation.py is in the same directory"
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def compare_to_reference_distribution(
    image_path: str,
    effect_type: str = "sun"
) -> str:
    """
    Compare render features directly against reference dataset distributions.

    Lower-level tool than evaluate_ground_truth - shows exactly how each
    feature matches against the reference statistical distribution.

    Useful for debugging WHY a render doesn't match reference footage.

    Args:
        image_path: Path to rendered image
        effect_type: Effect category ("sun", "star")

    Returns:
        JSON with per-feature similarity scores and analysis:
        - brightness_distribution_match: 0-1
        - warm_ratio_match: 0-1 (critical for sun)
        - structure_similarity: 0-1
        - color_distribution_match: 0-1
        - analysis: Per-dimension explanation
        - recommendations: What to fix

    Example:
        compare_to_reference_distribution("build/renders/sun_v8.png", "sun")
    """
    try:
        from ground_truth_evaluation import compare_to_ground_truth
        from dataclasses import asdict

        img_path = Path(image_path)
        if not img_path.is_absolute():
            img_path = PROJECT_ROOT / image_path

        if not img_path.exists():
            return json.dumps({"error": f"Image not found: {img_path}"})

        result = compare_to_ground_truth(str(img_path), effect_type)
        return json.dumps(asdict(result), indent=2)

    except ImportError as e:
        return json.dumps({"error": f"Ground truth module not available: {e}"})
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def evaluate_solar_features(image_path: str) -> str:
    """
    Evaluate sun/star-specific visual features that make a render realistic.

    Detects domain-specific features that generic VFX metrics miss:

    1. GRANULATION: Cellular convection pattern on sun's surface
       - Real sun has visible "rice grain" texture from convection cells
       - Score 0-1 based on high-frequency texture presence

    2. LIMB DARKENING: Edges appear darker than center
       - Real optical effect from viewing angle through atmosphere
       - Sun's edge is ~40% darker than center
       - Score based on actual brightness falloff

    3. PROMINENCES: Eruptions and loops extending from surface
       - Bright features visible beyond the disk edge
       - Scored by coverage of bright pixels outside disk

    4. CORONA: Faint outer atmosphere glow
       - Very faint glow extending well beyond prominences
       - Boolean detection

    5. COLOR TEMPERATURE: Estimated from RGB ratios
       - Real sun is ~5778K
       - Returns estimated Kelvin value

    Args:
        image_path: Path to sun/star render

    Returns:
        JSON with:
        - has_granulation: bool + granulation_score: 0-1
        - has_limb_darkening: bool + limb_darkening_score: 0-1
        - has_prominences: bool + prominence_score: 0-1
        - corona_visible: bool
        - color_temperature_kelvin: float

    Example:
        evaluate_solar_features("build/renders/sun_surface_v10.png")
    """
    try:
        from ground_truth_evaluation import evaluate_solar_specific
        from dataclasses import asdict
        import numpy as np

        img_path = Path(image_path)
        if not img_path.is_absolute():
            img_path = PROJECT_ROOT / image_path

        if not img_path.exists():
            return json.dumps({"error": f"Image not found: {img_path}"})

        result = evaluate_solar_specific(str(img_path))
        # Convert numpy scalar/array types into JSON-serializable Python types.
        def convert_np(obj):
            if isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            if isinstance(obj, (np.integer, int)):
                return int(obj)
            if isinstance(obj, (np.floating, float)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: convert_np(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_np(v) for v in obj]
            return obj

        return json.dumps(convert_np(asdict(result)), indent=2)

    except ImportError as e:
        return json.dumps({"error": f"Ground truth module not available: {e}"})
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def get_reference_statistics(effect_type: str = "sun", sample_size: int = 100) -> str:
    """
    Get computed statistics from the reference image dataset.

    Shows what "real" looks like for a given effect type by computing
    aggregate statistics across the reference dataset.

    This data is cached after first computation to avoid reprocessing
    840 images every time.

    Args:
        effect_type: Effect category ("sun", "star")
        sample_size: Number of reference images to sample (default 100)

    Returns:
        JSON with statistical distributions for:
        - brightness: mean, std, percentiles (p5, p25, median, p75, p95)
        - warm_ratio: color warmth distribution
        - edge_density: structure/texture distribution
        - color histograms: RGB distribution
        - radial_profile: brightness from center to edge

    Example:
        get_reference_statistics("sun", 100)

    Note: First call may take 30-60 seconds to process reference images.
          Subsequent calls use cached statistics.
    """
    try:
        from ground_truth_evaluation import compute_reference_statistics

        result = compute_reference_statistics(effect_type, sample_size)
        return json.dumps(result, indent=2)

    except ImportError as e:
        return json.dumps({"error": f"Ground truth module not available: {e}"})
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def extract_image_features(image_path: str) -> str:
    """
    Extract comprehensive features from a single image.

    Lower-level diagnostic tool that shows exactly what features are
    extracted from an image before comparison. Useful for debugging.

    Args:
        image_path: Path to image to analyze

    Returns:
        JSON with:
        - brightness: mean, std, histogram
        - color: warm_ratio, RGB means, orange/yellow presence, histogram
        - structure: edge_density, edge_histogram, variance
        - coverage: fraction of non-black pixels
        - radial_profile: brightness at 10 radial distances
        - image_size: width, height

    Example:
        extract_image_features("build/renders/sun_v10.png")
    """
    try:
        from ground_truth_evaluation import extract_image_features as extract_features

        img_path = Path(image_path)
        if not img_path.is_absolute():
            img_path = PROJECT_ROOT / image_path

        if not img_path.exists():
            return json.dumps({"error": f"Image not found: {img_path}"})

        result = extract_features(str(img_path))
        return json.dumps(result, indent=2)

    except ImportError as e:
        return json.dumps({"error": f"Ground truth module not available: {e}"})
    except Exception as e:
        return json.dumps({"error": str(e)})


# =============================================================================
# Connected Evaluation Tools (Evaluation + Experiment-Tracker)
# =============================================================================

@mcp.tool()
def evaluate_with_suggestions(
    image_path: str,
    effect_type: str = "sun",
    pass_threshold: float = 0.65
) -> str:
    """
    Evaluate render AND get learned fix suggestions from experiment-tracker.

    This is the RECOMMENDED evaluation tool for the iteration loop.
    It combines:
    1. Ground truth evaluation (diagnoses WHAT and WHERE)
    2. Experiment-tracker integration (suggests WHAT TO DO based on past success)

    Unlike generic recommendations, suggestions are based on accumulated
    knowledge from past experiments - they're LEARNED, not hardcoded.

    Args:
        image_path: Path to rendered image to evaluate
        effect_type: Effect category - "sun", "star", "prominence"
        pass_threshold: Minimum similarity score to pass (0-1)

    Returns:
        JSON with:
        - overall_score: 0-100
        - passed: True if meets threshold
        - issues: Diagnosed problems mapped to experiment-tracker categories
        - suggested_fixes: Parameter changes with confidence scores
        - primary_issue: Most important problem to fix
        - recommended_action: What to try next
        - suggested_hypothesis: For recording in experiment-tracker

    Example:
        evaluate_with_suggestions(
            "build/vdb_output/sun_prominences_v2/render_0060.png",
            "sun",
            0.65
        )
    """
    try:
        from evaluation_tracker_bridge import create_connected_evaluation
        from dataclasses import asdict

        img_path = Path(image_path)
        if not img_path.is_absolute():
            img_path = PROJECT_ROOT / image_path

        if not img_path.exists():
            return json.dumps({"error": f"Image not found: {img_path}"})

        result = create_connected_evaluation(str(img_path), effect_type, pass_threshold)

        # Convert dataclasses to dicts for JSON
        output = {
            "overall_score": result.overall_score,
            "passed": result.passed,
            "primary_issue": result.primary_issue,
            "recommended_action": result.recommended_action,
            "issues": [asdict(i) for i in result.issues],
            "suggested_fixes": [asdict(f) for f in result.suggested_fixes],
            "experiment_tracking": {
                "should_record_baseline": result.should_record_baseline,
                "suggested_hypothesis": result.suggested_hypothesis
            }
        }
        return json.dumps(output, indent=2)

    except ImportError as e:
        return json.dumps({
            "error": f"Evaluation-tracker bridge not available: {e}",
            "hint": "Ensure evaluation_tracker_bridge.py is in the same directory"
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def analyze_prominence(
    render_path: str,
    reference_path: str
) -> str:
    """
    Specialized analysis for solar prominence renders.

    Compares a prominence render against real solar prominence footage,
    focusing on prominence-specific issues:

    - Prominence COVERAGE: How much of the corona zone shows prominences
    - Prominence COLOR: Should be redder/cooler than the disk
    - Prominence SHAPE: Loops, arcs, eruptions vs blobs

    Use this when evaluating renders specifically targeting prominences,
    as the general sun evaluation may miss prominence-specific issues.

    Args:
        render_path: Path to prominence render
        reference_path: Path to real solar footage showing prominences

    Returns:
        JSON with:
        - disk_analysis: Brightness/color of main solar disk
        - prominence_analysis: Coverage, color warmth of prominences
        - issues: Specific prominence problems
        - suggestions: Parameter changes to improve prominences

    Example:
        analyze_prominence(
            "build/vdb_output/sun_prominences_v2/render_0060.png",
            "assets/reference_images/star/Eruptions_20241008_Activity_2048p30/frame_00639.jpg"
        )
    """
    try:
        from evaluation_tracker_bridge import analyze_prominence_quality

        render = Path(render_path)
        reference = Path(reference_path)

        if not render.is_absolute():
            render = PROJECT_ROOT / render_path
        if not reference.is_absolute():
            reference = PROJECT_ROOT / reference_path

        if not render.exists():
            return json.dumps({"error": f"Render not found: {render}"})
        if not reference.exists():
            return json.dumps({"error": f"Reference not found: {reference}"})

        result = analyze_prominence_quality(str(render), str(reference))
        return json.dumps(result, indent=2)

    except ImportError as e:
        return json.dumps({"error": f"Prominence analysis not available: {e}"})
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def get_fix_suggestions(
    issues_json: str
) -> str:
    """
    Get fix suggestions from experiment-tracker for diagnosed issues.

    Takes issue diagnoses (from evaluate_ground_truth or spatial_diagnostics)
    and queries experiment-tracker for known fixes.

    This enables learning: as more experiments are recorded with their
    outcomes, the suggestions improve over time.

    Args:
        issues_json: JSON array of issue objects with format:
            [{"category": "color_too_cool", "severity": "critical", ...}, ...]

    Returns:
        JSON with suggested fixes including:
        - parameter: Which Blender parameter to change
        - change: What to do (increase, decrease, etc.)
        - confidence: How confident based on past success
        - rationale: Why this might help
        - warnings: Known gotchas

    Example:
        get_fix_suggestions('[{"category": "color_too_cool", "severity": "critical"}]')
    """
    try:
        from evaluation_tracker_bridge import (
            get_suggestions_from_tracker,
            IssueDiagnosis
        )
        from dataclasses import asdict

        issues_data = json.loads(issues_json)

        # Convert to IssueDiagnosis objects
        issues = []
        for i in issues_data:
            issues.append(IssueDiagnosis(
                category=i.get("category", "unknown"),
                severity=i.get("severity", "moderate"),
                description=i.get("description", ""),
                region=i.get("region"),
                render_value=i.get("render_value", 0),
                reference_value=i.get("reference_value", 0),
                deviation_percent=i.get("deviation_percent", 0)
            ))

        suggestions = get_suggestions_from_tracker(issues)
        return json.dumps([asdict(s) for s in suggestions], indent=2)

    except Exception as e:
        return json.dumps({"error": str(e)})


# =============================================================================
# DINOv2 Structural Evaluation (NEW - Phase 1 of Evaluation System Improvement)
# =============================================================================

# Lazy-loaded DINOv2 evaluator
_dino_evaluator = None


def get_dino_evaluator():
    """Lazy load DINOv2 evaluator."""
    global _dino_evaluator
    if _dino_evaluator is None:
        try:
            from dino_structural_eval import DinoStructuralEvaluator
            _dino_evaluator = DinoStructuralEvaluator()
        except ImportError as e:
            raise RuntimeError(
                f"DINOv2 dependencies not installed: {e}\n"
                "Run: pip install transformers torch torchvision"
            )
    return _dino_evaluator


@mcp.tool()
def evaluate_structural_quality(
    render_path: str,
    reference_path: str,
    effect_type: str = "sun",
    include_heatmap: bool = True
) -> str:
    """
    Evaluate structural similarity using DINOv2 self-supervised features.

    This tool addresses the critical flaw where aggregate statistics miss
    morphological issues. DINOv2 detects:

    - TEXTURE PATTERNS: Procedural vs organic (catches "popcorn" texture)
    - SHAPE DIFFERENCES: Ribbon vs loop prominences
    - STRUCTURAL COHERENCE: Random noise vs coherent features
    - LOCALIZED PROBLEMS: WHERE differences are, not just that they exist

    Unlike aggregate statistics (edge_density, warm_ratio) which can be
    identical for visually different images, DINOv2 understands structure.

    Args:
        render_path: Path to rendered image to evaluate
        reference_path: Path to reference image (ground truth)
        effect_type: Effect category (sun, star, nebula, explosion)
        include_heatmap: Include similarity heatmap in result

    Returns:
        JSON with:
        - global_structural_similarity: 0-1 (higher = more similar)
        - structural_quality: excellent/good/fair/poor/very_poor
        - problem_regions: List of regions with issues, sorted by severity
        - recommendations: Actionable fixes based on problem regions
        - similarity_heatmap: 14x14 grid showing WHERE differences are

    Example:
        evaluate_structural_quality(
            "build/vdb_output/sun_prominences_v2/render_0060.png",
            "assets/reference_images/star/.../frame_00639.jpg",
            "sun"
        )

    Research basis:
        - DINOv2 achieves 64% accuracy vs CLIP's 28% on structural similarity
        - Patch-level features enable spatial localization
        - See: docs/EVALUATION_SYSTEM_IMPROVEMENT_PROPOSAL.md
    """
    try:
        from dino_structural_eval import DinoStructuralEvaluator, asdict

        evaluator = DinoStructuralEvaluator(effect_type=effect_type)
        result = evaluator.evaluate(
            render_path,
            reference_path,
            include_heatmap=include_heatmap
        )

        # Convert to dict for JSON serialization
        result_dict = asdict(result)

        return json.dumps(result_dict, indent=2)

    except Exception as e:
        import traceback
        return json.dumps({
            "error": str(e),
            "traceback": traceback.format_exc(),
            "hint": "Ensure transformers and torch are installed: pip install transformers torch torchvision"
        }, indent=2)


@mcp.tool()
def evaluate_structural_against_dataset(
    render_path: str,
    reference_dir: str,
    sample_size: int = 10
) -> str:
    """
    Evaluate render against multiple reference images from a dataset.

    Useful when you have a dataset of reference images (like the 840 NASA
    solar frames). Computes similarity to a sample and finds closest matches.

    Args:
        render_path: Path to rendered image
        reference_dir: Directory containing reference images
        sample_size: Number of reference images to sample (evenly distributed)

    Returns:
        JSON with:
        - mean_similarity: Average structural similarity
        - max_similarity: Best match score
        - min_similarity: Worst match score
        - closest_matches: Top 5 most similar references
        - farthest_matches: 3 least similar references

    Example:
        evaluate_structural_against_dataset(
            "build/vdb_output/sun_v10/render_0060.png",
            "assets/reference_images/star/Eruptions_20241008_Activity_2048p30/",
            sample_size=20
        )
    """
    try:
        from dino_structural_eval import DinoStructuralEvaluator

        evaluator = DinoStructuralEvaluator()
        result = evaluator.evaluate_against_dataset(
            render_path,
            reference_dir,
            sample_size=sample_size
        )

        return json.dumps(result, indent=2)

    except Exception as e:
        import traceback
        return json.dumps({
            "error": str(e),
            "traceback": traceback.format_exc(),
            "hint": "Ensure transformers and torch are installed: pip install transformers torch torchvision"
        }, indent=2)


@mcp.tool()
def save_structural_heatmap(
    render_path: str,
    reference_path: str,
    output_path: str
) -> str:
    """
    Generate and save a visual structural similarity heatmap.

    Creates a side-by-side visualization showing:
    - Render image
    - Reference image
    - 14x14 heatmap (green = similar, red = different)

    This is essential for understanding WHERE structural problems are
    in the render, not just that they exist.

    Args:
        render_path: Path to render
        reference_path: Path to reference
        output_path: Where to save the heatmap image

    Returns:
        Path to saved heatmap image, or error message

    Example:
        save_structural_heatmap(
            "build/vdb_output/sun_v10/render_0060.png",
            "assets/reference_images/star/.../frame_00639.jpg",
            "evaluation_outputs/structural_heatmap.png"
        )
    """
    try:
        from dino_structural_eval import save_similarity_heatmap

        # Ensure output directory exists
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        result_path = save_similarity_heatmap(
            render_path,
            reference_path,
            output_path
        )

        return json.dumps({
            "success": True,
            "heatmap_path": result_path,
            "render_path": render_path,
            "reference_path": reference_path
        }, indent=2)

    except Exception as e:
        import traceback
        return json.dumps({
            "error": str(e),
            "traceback": traceback.format_exc()
        }, indent=2)


# =============================================================================
# Phase 2: Multi-Scale Wavelet/Texture Analysis Tools
# =============================================================================

@mcp.tool()
def analyze_texture_procedural(
    render_path: str,
    reference_path: str = ""
) -> str:
    """
    Comprehensive procedural texture detection using multiple methods.

    This tool detects whether a render has PROCEDURAL (synthetic/generated)
    or NATURAL (organic/realistic) texture patterns. Addresses the critical
    flaw where aggregate statistics miss texture issues like "popcorn" noise.

    Combines 4 analysis methods:
    1. Feature Size Distribution (60% weight - MOST DISCRIMINATING)
       - Procedural textures have UNIFORM feature sizes (low CV)
       - Natural textures have VARIED sizes (high CV)
       - Tested: Render CV=1.83 vs Reference CV=36.97 (20x difference!)

    2. Wavelet Scale Entropy (25% weight)
       - Procedural noise concentrates energy at specific scales
       - Natural images distribute energy across many scales

    3. Texture Uniformity (10% weight)
       - Procedural has consistent local variance
       - Natural has varying local variance

    4. Repetitive Pattern Detection (5% weight)
       - Procedural often has periodic autocorrelation peaks

    Args:
        render_path: Path to rendered image to analyze
        reference_path: Optional reference image for comparison

    Returns:
        JSON with:
        - procedural_score: 0-100 (higher = more procedural)
        - texture_quality: procedural/mixed/natural
        - texture_verdict: Human-readable assessment
        - feature_sizes: Detailed feature size metrics
        - wavelet: Scale entropy and dominant scale
        - signals_detected: Which procedural signals were found
        - recommendations: Actionable fixes

    Example:
        analyze_texture_procedural(
            "build/vdb_output/sun_prominences_v2/render_0060.png",
            "assets/reference_images/star/.../frame_00639.jpg"
        )

    Research basis:
        - Feature size CV is empirically the best discriminator
        - See: docs/EVALUATION_SYSTEM_IMPROVEMENT_PROPOSAL.md
    """
    try:
        from wavelet_scale_analysis import comprehensive_texture_analysis
        import numpy as np

        # Convert numpy types for JSON serialization
        def convert_np(obj):
            if isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            if isinstance(obj, (np.integer, int)):
                return int(obj)
            if isinstance(obj, (np.floating, float)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: convert_np(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_np(v) for v in obj]
            return obj

        result = comprehensive_texture_analysis(
            render_path,
            reference_path if reference_path else None
        )

        return json.dumps(convert_np(result), indent=2)

    except Exception as e:
        import traceback
        return json.dumps({
            "error": str(e),
            "traceback": traceback.format_exc(),
            "hint": "Ensure PyWavelets and opencv-python-headless are installed"
        }, indent=2)


@mcp.tool()
def analyze_feature_size_distribution(
    image_path: str,
    content_threshold: int = 30
) -> str:
    """
    Analyze feature size distribution to detect procedural textures.

    This is the SINGLE MOST DISCRIMINATING metric discovered for detecting
    procedural vs natural textures:

    - Procedural textures: CV = 1.83 (uniform feature sizes)
    - Natural textures: CV = 36.97 (varied feature sizes - 20x higher!)

    Uses connected component analysis to find features and measures
    their size distribution. Low coefficient of variation (CV) indicates
    uniform/procedural texture.

    Args:
        image_path: Path to image to analyze
        content_threshold: Brightness threshold for content mask (default 30)

    Returns:
        JSON with:
        - size_cv: Coefficient of variation (key metric)
        - is_uniform_size: True if CV < 10 (procedural signature)
        - feature_count: Number of features detected
        - uniformity_type: highly_uniform/uniform/mixed/varied
        - interpretation: Human-readable assessment
        - size_mean, size_std, size_min, size_max: Statistics

    Example:
        analyze_feature_size_distribution(
            "build/vdb_output/sun_prominences_v2/render_0060.png"
        )

    Interpretation:
        - CV < 3: HIGHLY UNIFORM - strong procedural signature
        - CV < 10: UNIFORM - likely procedural
        - CV < 20: MIXED - some variation
        - CV > 20: VARIED - natural multi-scale structure
    """
    try:
        from wavelet_scale_analysis import analyze_feature_sizes
        import numpy as np

        def convert_np(obj):
            if isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            if isinstance(obj, (np.integer, int)):
                return int(obj)
            if isinstance(obj, (np.floating, float)):
                return float(obj)
            if isinstance(obj, dict):
                return {k: convert_np(v) for k, v in obj.items()}
            return obj

        result = analyze_feature_sizes(image_path, content_threshold)

        return json.dumps(convert_np(result), indent=2)

    except Exception as e:
        import traceback
        return json.dumps({
            "error": str(e),
            "traceback": traceback.format_exc(),
            "hint": "Ensure opencv-python-headless is installed"
        }, indent=2)


@mcp.tool()
def compare_texture_quality(
    render_path: str,
    reference_path: str
) -> str:
    """
    Compare texture quality between render and reference.

    Computes procedural scores for both images and determines which
    is more natural/realistic. Useful for A/B testing renders against
    real footage.

    Args:
        render_path: Path to rendered image
        reference_path: Path to reference image (ideally real footage)

    Returns:
        JSON with:
        - render_score: Procedural score for render (0-100)
        - reference_score: Procedural score for reference (0-100)
        - score_difference: How much more procedural render is
        - discrimination_quality: strong/good/weak/failed
        - render_verdict: Assessment of render
        - reference_verdict: Assessment of reference
        - which_is_better: render/reference/similar
        - recommendations: What to fix in render

    Example:
        compare_texture_quality(
            "build/vdb_output/sun_v10/render_0060.png",
            "assets/reference_images/star/.../frame_00639.jpg"
        )
    """
    try:
        from wavelet_scale_analysis import comprehensive_texture_analysis
        import numpy as np

        def convert_np(obj):
            if isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            if isinstance(obj, (np.integer, int)):
                return int(obj)
            if isinstance(obj, (np.floating, float)):
                return float(obj)
            if isinstance(obj, dict):
                return {k: convert_np(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_np(v) for v in obj]
            return obj

        render_result = convert_np(comprehensive_texture_analysis(render_path))
        ref_result = convert_np(comprehensive_texture_analysis(reference_path))

        r_score = render_result["combined_score"]["procedural_score"]
        ref_score = ref_result["combined_score"]["procedural_score"]
        diff = r_score - ref_score

        # Determine discrimination quality
        if diff > 30:
            discrimination = "strong"
        elif diff > 15:
            discrimination = "good"
        elif diff > 0:
            discrimination = "weak"
        else:
            discrimination = "failed"

        # Determine which is better (lower procedural score = more natural)
        if diff > 10:
            which_better = "reference"
        elif diff < -10:
            which_better = "render"
        else:
            which_better = "similar"

        return json.dumps({
            "render_score": r_score,
            "reference_score": ref_score,
            "score_difference": round(diff, 1),
            "discrimination_quality": discrimination,
            "render_verdict": render_result["combined_score"]["texture_verdict"],
            "reference_verdict": ref_result["combined_score"]["texture_verdict"],
            "which_is_better": which_better,
            "render_signals": render_result["combined_score"]["signals_detected"],
            "reference_signals": ref_result["combined_score"]["signals_detected"],
            "render_feature_cv": render_result["feature_sizes"].get("size_cv"),
            "reference_feature_cv": ref_result["feature_sizes"].get("size_cv"),
            "recommendations": render_result["recommendations"]
        }, indent=2)

    except Exception as e:
        import traceback
        return json.dumps({
            "error": str(e),
            "traceback": traceback.format_exc()
        }, indent=2)


# =============================================================================
# Phase 3: Solar Discriminator Tools
# =============================================================================
# EfficientNetV2-based binary classifier: real solar footage vs synthetic renders
# Trained on 840 real frames + 750 synthetic renders (100% val accuracy)
# Provides Grad-CAM visualization of "fake" regions

@mcp.tool()
def predict_real_or_synthetic(
    image_path: str,
    include_gradcam: bool = True,
    gradcam_output: str = ""
) -> str:
    """
    Predict whether an image is real solar footage or a synthetic render.

    Uses a trained EfficientNetV2 discriminator that achieved 100% validation
    accuracy on held-out data. Also provides Grad-CAM heatmap showing WHERE
    the "fake" signal comes from.

    Args:
        image_path: Path to image to analyze
        include_gradcam: Whether to compute Grad-CAM regions (default True)
        gradcam_output: Optional path to save Grad-CAM visualization

    Returns:
        JSON with:
        - verdict: REAL, SYNTHETIC, LIKELY_REAL, LIKELY_SYNTHETIC, or UNCERTAIN
        - real_probability: 0-100%
        - synthetic_probability: 0-100%
        - confidence: How confident the prediction is
        - gradcam_regions: Where the discriminator focused attention
        - interpretation: Human-readable explanation

    Example:
        predict_real_or_synthetic(
            "build/vdb_output/sun_prominences_v2/render_0060.png",
            gradcam_output="evaluation_outputs/gradcam.png"
        )

    Note: Model must be trained first. Default path: models/solar_discriminator.pth
    """
    try:
        from solar_discriminator import predict_with_explanation
        import os

        # Use default gradcam path if not specified but requested
        if include_gradcam and not gradcam_output:
            os.makedirs("evaluation_outputs", exist_ok=True)
            base = os.path.splitext(os.path.basename(image_path))[0]
            gradcam_output = f"evaluation_outputs/{base}_gradcam.png"

        result = predict_with_explanation(
            image_path,
            model_path="models/solar_discriminator.pth",
            gradcam_output=gradcam_output if include_gradcam else None
        )

        return json.dumps({
            "image_path": result.image_path,
            "verdict": result.verdict,
            "is_real": result.is_real,
            "real_probability": round(result.real_probability * 100, 2),
            "synthetic_probability": round(result.synthetic_probability * 100, 2),
            "confidence": round(result.confidence * 100, 2),
            "gradcam_regions": result.gradcam_regions[:5],  # Top 5 regions
            "gradcam_heatmap_path": result.gradcam_heatmap_path,
            "interpretation": (
                f"Image is {result.verdict} with {result.confidence*100:.1f}% confidence. "
                f"{'Primary synthetic signature detected in ' + result.gradcam_regions[0]['region'] + ' region.' if result.gradcam_regions and not result.is_real else ''}"
                if result.confidence > 0.65 else
                "Classification uncertain - image may have ambiguous characteristics."
            )
        }, indent=2)

    except FileNotFoundError:
        return json.dumps({
            "error": "Model not found",
            "hint": "Train the discriminator first: python solar_discriminator.py train --epochs 10",
            "model_path": "models/solar_discriminator.pth"
        }, indent=2)
    except Exception as e:
        import traceback
        return json.dumps({
            "error": str(e),
            "traceback": traceback.format_exc()
        }, indent=2)


@mcp.tool()
def train_discriminator(
    epochs: int = 10,
    batch_size: int = 4,
    real_dir: str = "assets/reference_images/star/Eruptions_20241008_Activity_2048p30",
    synthetic_dir: str = "build/vdb_output"
) -> str:
    """
    Train the real/synthetic solar discriminator.

    Uses EfficientNetV2-S with transfer learning. Automatically handles
    class imbalance through weighted sampling and augmentation.

    Args:
        epochs: Number of training epochs (default 10)
        batch_size: Batch size (default 4, reduce if OOM)
        real_dir: Directory with real solar frames (JPG)
        synthetic_dir: Directory with synthetic renders (PNG)

    Returns:
        JSON with training results:
        - final_accuracy: Validation accuracy
        - best_epoch: Which epoch had best accuracy
        - model_path: Where model was saved
        - training_history: Per-epoch metrics

    Note: Uses GPU if available. Training takes ~5-10 minutes on RTX 4060 Ti.
    """
    try:
        from solar_discriminator import train_solar_discriminator

        result = train_solar_discriminator(
            real_dir=real_dir,
            synthetic_dirs=[synthetic_dir],
            epochs=epochs,
            batch_size=batch_size,
            output_path="models/solar_discriminator.pth"
        )

        return json.dumps({
            "success": True,
            "final_accuracy": round(result.final_accuracy * 100, 2),
            "best_accuracy": round(result.best_accuracy * 100, 2),
            "best_epoch": result.best_epoch,
            "model_path": result.model_path,
            "epochs_trained": result.epochs_trained,
            "training_samples": result.train_samples,
            "validation_samples": result.val_samples,
            "history": result.history
        }, indent=2)

    except Exception as e:
        import traceback
        return json.dumps({
            "error": str(e),
            "traceback": traceback.format_exc(),
            "hint": "Ensure PyTorch and timm are installed, and training data exists"
        }, indent=2)


@mcp.tool()
def compare_real_synthetic_batch(
    image_paths: str,
    reference_path: str = ""
) -> str:
    """
    Batch analysis of multiple images for real/synthetic classification.

    Useful for evaluating an entire animation or comparing iterations.

    Args:
        image_paths: Comma-separated list of image paths OR glob pattern
        reference_path: Optional reference image for comparison

    Returns:
        JSON with:
        - results: Per-image classification
        - summary: Aggregate statistics
        - recommendations: Overall suggestions

    Example:
        compare_real_synthetic_batch(
            "build/vdb_output/sun_v10/render_*.png",
            "assets/reference_images/star/.../frame_00639.jpg"
        )
    """
    try:
        from solar_discriminator import SolarDiscriminator
        import glob

        # Parse image paths
        paths = []
        for p in image_paths.split(","):
            p = p.strip()
            if "*" in p or "?" in p:
                paths.extend(glob.glob(p))
            else:
                paths.append(p)

        if not paths:
            return json.dumps({"error": "No valid image paths found"})

        discriminator = SolarDiscriminator(model_path="models/solar_discriminator.pth")

        results = []
        synthetic_count = 0
        real_count = 0

        for path in paths[:20]:  # Limit to 20 images
            try:
                result = discriminator.predict(path, include_gradcam=False)
                results.append({
                    "path": path,
                    "verdict": result.verdict,
                    "confidence": round(result.confidence * 100, 2)
                })
                if "SYNTHETIC" in result.verdict:
                    synthetic_count += 1
                elif "REAL" in result.verdict:
                    real_count += 1
            except Exception as e:
                results.append({"path": path, "error": str(e)})

        # Analyze reference if provided
        ref_result = None
        if reference_path:
            try:
                ref = discriminator.predict(reference_path, include_gradcam=False)
                ref_result = {
                    "path": reference_path,
                    "verdict": ref.verdict,
                    "confidence": round(ref.confidence * 100, 2)
                }
            except Exception as e:
                ref_result = {"path": reference_path, "error": str(e)}

        return json.dumps({
            "total_analyzed": len(results),
            "synthetic_count": synthetic_count,
            "real_count": real_count,
            "uncertain_count": len(results) - synthetic_count - real_count,
            "results": results,
            "reference": ref_result,
            "recommendation": (
                "All images appear synthetic - consider structural improvements"
                if synthetic_count == len(results) else
                f"{synthetic_count}/{len(results)} images classified as synthetic"
            )
        }, indent=2)

    except FileNotFoundError:
        return json.dumps({
            "error": "Model not found",
            "hint": "Train the discriminator first"
        }, indent=2)
    except Exception as e:
        import traceback
        return json.dumps({
            "error": str(e),
            "traceback": traceback.format_exc()
        }, indent=2)


# =============================================================================
# Phase 4: Prominence Shape Classifier Tools
# =============================================================================

@mcp.tool()
def analyze_prominence_shapes(
    image_path: str,
    save_visualization: bool = True,
    output_path: str = ""
) -> str:
    """
    Analyze prominence morphology to detect synthetic artifacts.

    Detects:
    - "Cat ear" triangular protrusions
    - Ribbon artifacts (too regular, uniform width)
    - Missing filamentary structure
    - Unnatural symmetry

    Real prominences have high internal texture variance (>2.0) from filamentary structure.
    Synthetic artifacts have low variance (<1.5) from uniform procedural noise.

    Args:
        image_path: Path to solar render or reference image
        save_visualization: Save annotated image showing detected regions
        output_path: Custom output path for visualization (optional)

    Returns:
        JSON with:
        - overall_score: 0-100 shape quality score
        - artifact_count: Number of detected artifacts
        - natural_count: Number of natural-looking prominences
        - prominences: Detailed per-prominence analysis
        - issues: List of detected morphological problems
        - summary: Human-readable assessment

    Example:
        analyze_prominence_shapes("build/vdb_output/sun_prominences_v2/render_0060.png")
    """
    try:
        from prominence_shape_classifier import analyze_image

        # Determine output path
        if save_visualization:
            if not output_path:
                path = Path(image_path)
                output_path = str(PROJECT_ROOT / "evaluation_outputs" / f"{path.stem}_prominence_analysis.png")

        result = analyze_image(image_path, output_path if save_visualization else None)

        # Add visualization path to result
        if save_visualization and output_path:
            result["visualization_path"] = output_path

        return json.dumps(result, indent=2, default=float)

    except Exception as e:
        import traceback
        return json.dumps({
            "error": str(e),
            "traceback": traceback.format_exc()
        }, indent=2)


@mcp.tool()
def compare_prominence_quality(
    synthetic_path: str,
    reference_path: str
) -> str:
    """
    Compare prominence quality between synthetic render and real reference.

    Provides side-by-side comparison highlighting the morphological differences
    between synthetic and real solar prominences.

    Args:
        synthetic_path: Path to synthetic render
        reference_path: Path to real solar reference image

    Returns:
        JSON with:
        - synthetic: Full analysis of synthetic render
        - reference: Full analysis of reference image
        - comparison: Key differences and severity rating
        - recommendations: What to improve in the synthetic render

    Example:
        compare_prominence_quality(
            "build/vdb_output/sun_prominences_v2/render_0060.png",
            "assets/reference_images/star/.../frame_00639.jpg"
        )
    """
    try:
        from prominence_shape_classifier import analyze_image

        # Analyze both images
        syn_result = analyze_image(
            synthetic_path,
            str(PROJECT_ROOT / "evaluation_outputs" / "prominence_comparison_synthetic.png")
        )
        ref_result = analyze_image(
            reference_path,
            str(PROJECT_ROOT / "evaluation_outputs" / "prominence_comparison_reference.png")
        )

        # Build comparison
        score_diff = ref_result.get("overall_score", 0) - syn_result.get("overall_score", 0)
        artifact_diff = syn_result.get("artifact_count", 0) - ref_result.get("artifact_count", 0)

        # Determine severity
        if score_diff > 60:
            severity = "CRITICAL"
            color = "red"
        elif score_diff > 30:
            severity = "HIGH"
            color = "orange"
        elif score_diff > 10:
            severity = "MEDIUM"
            color = "yellow"
        else:
            severity = "LOW"
            color = "green"

        # Extract key issues from synthetic
        key_issues = []
        for issue in syn_result.get("issues", [])[:5]:
            if "CAT_EAR" in issue:
                key_issues.append("Cat ear triangular artifacts detected")
            elif "UNIFORM_TEXTURE" in issue:
                key_issues.append("Missing filamentary structure (uniform procedural texture)")
            elif "HIGH_SYMMETRY" in issue:
                key_issues.append("Unnatural bilateral symmetry")
            elif "ANGULAR" in issue:
                key_issues.append("Sharp angular transitions (not smooth curves)")

        # Remove duplicates
        key_issues = list(dict.fromkeys(key_issues))

        # Generate recommendations
        recommendations = []
        if "Cat ear" in str(key_issues):
            recommendations.append("Reduce flame_max_temp or adjust smoke_color to soften triangular shapes")
        if "filamentary" in str(key_issues):
            recommendations.append("Increase turbulence/vorticity to add internal structure")
            recommendations.append("Add procedural displacement to prominence geometry")
        if "symmetry" in str(key_issues):
            recommendations.append("Add asymmetric noise or randomization to prominence positions")
        if "angular" in str(key_issues):
            recommendations.append("Smooth geometry or add noise to edge contours")

        comparison = {
            "synthetic": {
                "score": syn_result.get("overall_score", 0),
                "artifacts": syn_result.get("artifact_count", 0),
                "natural": syn_result.get("natural_count", 0),
                "visualization": str(PROJECT_ROOT / "evaluation_outputs" / "prominence_comparison_synthetic.png")
            },
            "reference": {
                "score": ref_result.get("overall_score", 0),
                "artifacts": ref_result.get("artifact_count", 0),
                "natural": ref_result.get("natural_count", 0),
                "visualization": str(PROJECT_ROOT / "evaluation_outputs" / "prominence_comparison_reference.png")
            },
            "comparison": {
                "score_difference": score_diff,
                "artifact_difference": artifact_diff,
                "severity": severity,
                "key_issues": key_issues
            },
            "recommendations": recommendations if recommendations else ["Prominence shapes appear acceptable"]
        }

        return json.dumps(comparison, indent=2)

    except Exception as e:
        import traceback
        return json.dumps({
            "error": str(e),
            "traceback": traceback.format_exc()
        }, indent=2)


@mcp.tool()
def detect_cat_ear_artifacts(
    image_path: str,
    sensitivity: str = "medium"
) -> str:
    """
    Specifically detect "cat ear" triangular prominence artifacts.

    Cat ears are symmetric triangular protrusions that are a hallmark of
    poorly-configured procedural prominence generation. They occur when:
    - Flame/smoke rises uniformly without turbulence
    - Domain boundaries create regular peaks
    - Insufficient noise/randomization in the simulation

    Args:
        image_path: Path to solar render
        sensitivity: Detection sensitivity ("low", "medium", "high")

    Returns:
        JSON with:
        - cat_ear_count: Number of cat ear patterns detected
        - locations: Bounding boxes and angles of each cat ear
        - confidence_scores: Per-detection confidence
        - severity: Overall severity rating
        - fixes: Specific parameter changes to try

    Example:
        detect_cat_ear_artifacts("build/vdb_output/sun_prominences_v2/render_0060.png")
    """
    try:
        from prominence_shape_classifier import ProminenceShapeClassifier
        import cv2

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return json.dumps({"error": f"Could not load image: {image_path}"})
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Analyze
        classifier = ProminenceShapeClassifier()
        result = classifier.analyze_prominences(image)

        if result["status"] != "analyzed":
            return json.dumps(result, indent=2)

        # Filter for cat ear patterns
        cat_ears = [p for p in result["prominences"] if p["classification"] == "cat_ear"]

        # Adjust thresholds based on sensitivity
        if sensitivity == "high":
            threshold = 0.5
        elif sensitivity == "low":
            threshold = 0.8
        else:
            threshold = 0.65

        filtered_ears = [e for e in cat_ears if e["cat_ear_confidence"] >= threshold]

        # Determine severity
        if len(filtered_ears) >= 5:
            severity = "CRITICAL"
            recommendation = "Major prominence overhaul needed"
        elif len(filtered_ears) >= 3:
            severity = "HIGH"
            recommendation = "Significant cat ear artifacts - adjust parameters"
        elif len(filtered_ears) >= 1:
            severity = "MEDIUM"
            recommendation = "Some cat ear artifacts detected"
        else:
            severity = "NONE"
            recommendation = "No cat ear artifacts detected at this sensitivity"

        # Compile locations
        locations = [{
            "bbox": ear["bbox"],
            "angle_from_center": round(ear["angle_from_center"], 1),
            "confidence": round(ear["cat_ear_confidence"], 2),
            "symmetry": round(ear["symmetry"], 2)
        } for ear in filtered_ears]

        # Generate specific fixes
        fixes = []
        if len(filtered_ears) > 0:
            fixes = [
                "Increase vorticity (try 0.5-0.8) to break symmetric updrafts",
                "Add turbulence noise to flame/smoke source",
                "Reduce flame_max_temp slightly to soften peaks",
                "Expand domain height to avoid boundary clipping",
                "Add wind force with slight randomization"
            ]

        return json.dumps({
            "cat_ear_count": len(filtered_ears),
            "sensitivity": sensitivity,
            "threshold_used": threshold,
            "locations": locations,
            "severity": severity,
            "recommendation": recommendation,
            "fixes": fixes if fixes else ["No fixes needed"]
        }, indent=2)

    except Exception as e:
        import traceback
        return json.dumps({
            "error": str(e),
            "traceback": traceback.format_exc()
        }, indent=2)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    mcp.run()
