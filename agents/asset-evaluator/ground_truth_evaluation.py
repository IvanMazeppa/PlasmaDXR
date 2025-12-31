#!/usr/bin/env python3
"""
Ground Truth VFX Evaluation Module

Addresses the critical flaw where synthetic renders score higher than real footage
by comparing against actual reference datasets instead of arbitrary thresholds.

Key insight: The threshold-based system answers "is this obviously broken?"
This system answers "does this look like real footage?"

Reference Dataset: 840 frames of real solar activity footage
- Source: Eruptions_20241008_Activity_2048p30
- 14 hours of solar rotation captured
- Features: prominences, granulation, filaments, eruptions
"""

import json
import hashlib
from dataclasses import dataclass, asdict, field
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import numpy as np

# Lazy-loaded caches
_reference_stats_cache: Dict[str, Dict] = {}
_reference_embeddings_cache: Dict[str, np.ndarray] = {}


# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
REFERENCE_DIRS = {
    "sun": PROJECT_ROOT / "assets/reference_images/star/Eruptions_20241008_Activity_2048p30",
    "star": PROJECT_ROOT / "assets/reference_images/star/Eruptions_20241008_Activity_2048p30",
    # Future: add more effect types
    # "explosion": PROJECT_ROOT / "assets/reference_images/explosion",
    # "nebula": PROJECT_ROOT / "assets/reference_images/nebula",
}

# Stats cache location
STATS_CACHE_DIR = Path(__file__).parent / "reference_stats_cache"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class DistributionStats:
    """Statistical distribution of an image feature."""
    mean: float
    std: float
    percentile_5: float
    percentile_25: float
    median: float
    percentile_75: float
    percentile_95: float
    histogram: List[float] = field(default_factory=list)


@dataclass
class GroundTruthComparison:
    """Result of comparing render to ground truth reference set."""
    # Core similarity scores (0-1, higher = more similar to real footage)
    overall_similarity: float
    color_distribution_match: float
    brightness_distribution_match: float
    structure_similarity: float
    warm_ratio_match: float

    # Which reference frames are most similar
    closest_reference_frames: List[Tuple[str, float]]  # (filename, similarity)

    # Per-dimension analysis
    analysis: Dict[str, str]  # e.g., {"brightness": "darker than average reference"}

    # Actionable recommendations
    recommendations: List[str]

    # Pass/fail with threshold
    passed: bool
    pass_threshold: float


@dataclass
class SolarSpecificMetrics:
    """Metrics specific to sun/star evaluation."""
    has_granulation: bool       # Visible cell-like surface texture
    granulation_score: float    # 0-1 quality of granulation
    has_limb_darkening: bool    # Edges darker than center
    limb_darkening_score: float
    has_prominences: bool       # Visible eruptions/loops
    prominence_score: float
    corona_visible: bool        # Glowing outer atmosphere
    color_temperature_kelvin: float  # Estimated from color


# =============================================================================
# Feature Extraction
# =============================================================================

def extract_image_features(image_path: str) -> Dict[str, Any]:
    """
    Extract comprehensive features from an image for comparison.

    Returns features that can be compared against reference distribution.
    """
    from PIL import Image

    img = Image.open(image_path).convert('RGB')
    arr = np.array(img, dtype=np.float32)

    # Color channel analysis
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # Brightness
    brightness = np.mean(arr)
    brightness_hist, _ = np.histogram(arr.mean(axis=-1), bins=64, range=(0, 255))
    brightness_hist = brightness_hist / brightness_hist.sum()

    # Color distribution
    warm_ratio = np.sum(r) / max(np.sum(b), 1)

    # Color histogram (simplified: 16 bins per channel)
    r_hist, _ = np.histogram(r, bins=16, range=(0, 255))
    g_hist, _ = np.histogram(g, bins=16, range=(0, 255))
    b_hist, _ = np.histogram(b, bins=16, range=(0, 255))

    color_hist = np.concatenate([r_hist, g_hist, b_hist]).astype(float)
    color_hist = color_hist / color_hist.sum()

    # Edge/structure analysis (gradient magnitude)
    gray = arr.mean(axis=-1)
    gx = np.abs(np.diff(gray, axis=1, prepend=gray[:, :1]))
    gy = np.abs(np.diff(gray, axis=0, prepend=gray[:1, :]))
    edges = np.sqrt(gx**2 + gy**2)
    edge_density = np.mean(edges) / 255.0

    # Edge histogram
    edge_hist, _ = np.histogram(edges.flatten(), bins=32, range=(0, 100))
    edge_hist = edge_hist / edge_hist.sum()

    # Radial analysis (for limb darkening detection in sun images)
    h, w = arr.shape[:2]
    cy, cx = h // 2, w // 2
    y_coords, x_coords = np.ogrid[:h, :w]
    r_dist = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
    r_max = min(cx, cy)

    # Radial brightness profile (10 rings)
    radial_profile = []
    for i in range(10):
        r_inner = i * r_max / 10
        r_outer = (i + 1) * r_max / 10
        mask = (r_dist >= r_inner) & (r_dist < r_outer)
        if mask.sum() > 0:
            radial_profile.append(float(np.mean(gray[mask])))
        else:
            radial_profile.append(0.0)

    # Detect coverage (non-black ratio)
    non_black_mask = np.any(arr > 15, axis=-1)
    coverage = np.mean(non_black_mask)

    # Orange/yellow presence (key for sun)
    has_orange = np.mean((r > 180) & (g > 100) & (g < 200) & (b < 100))
    has_yellow = np.mean((r > 200) & (g > 180) & (b < 150))

    return {
        "brightness": {
            "mean": float(brightness),
            "std": float(np.std(arr)),
            "histogram": brightness_hist.tolist()
        },
        "color": {
            "warm_ratio": float(warm_ratio),
            "histogram": color_hist.tolist(),
            "r_mean": float(np.mean(r)),
            "g_mean": float(np.mean(g)),
            "b_mean": float(np.mean(b)),
            "has_orange": float(has_orange),
            "has_yellow": float(has_yellow)
        },
        "structure": {
            "edge_density": float(edge_density),
            "edge_histogram": edge_hist.tolist(),
            "variance": float(np.var(arr))
        },
        "coverage": float(coverage),
        "radial_profile": radial_profile,
        "image_size": {"width": w, "height": h}
    }


# =============================================================================
# Reference Statistics (Computed Once, Cached)
# =============================================================================

def compute_reference_statistics(effect_type: str, sample_size: int = 100) -> Dict[str, Any]:
    """
    Compute statistical distributions from reference image set.

    This captures what "real" looks like for a given effect type.
    Results are cached to avoid recomputation.
    """
    cache_key = f"{effect_type}_{sample_size}"
    if cache_key in _reference_stats_cache:
        return _reference_stats_cache[cache_key]

    # Check file cache
    STATS_CACHE_DIR.mkdir(exist_ok=True)
    cache_file = STATS_CACHE_DIR / f"{cache_key}_stats.json"
    if cache_file.exists():
        with open(cache_file) as f:
            stats = json.load(f)
            _reference_stats_cache[cache_key] = stats
            return stats

    ref_dir = REFERENCE_DIRS.get(effect_type)
    if ref_dir is None or not ref_dir.exists():
        return {"error": f"No reference images for effect type: {effect_type}"}

    # Get reference image paths
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    ref_paths = [p for p in ref_dir.iterdir() if p.suffix.lower() in image_extensions]

    if len(ref_paths) == 0:
        return {"error": f"No images found in {ref_dir}"}

    # Sample evenly across the dataset
    step = max(1, len(ref_paths) // sample_size)
    sampled_paths = ref_paths[::step][:sample_size]

    # Extract features from all sampled images
    all_features = []
    for path in sampled_paths:
        try:
            features = extract_image_features(str(path))
            features["filename"] = path.name
            all_features.append(features)
        except Exception as e:
            print(f"Failed to process {path}: {e}")
            continue

    if len(all_features) == 0:
        return {"error": "Failed to extract features from any reference images"}

    # Compute aggregate statistics
    stats = _aggregate_statistics(all_features)
    stats["sample_count"] = len(all_features)
    stats["total_available"] = len(ref_paths)
    stats["effect_type"] = effect_type

    # Cache to file
    with open(cache_file, 'w') as f:
        json.dump(stats, f, indent=2)

    _reference_stats_cache[cache_key] = stats
    return stats


def _aggregate_statistics(features_list: List[Dict]) -> Dict[str, Any]:
    """Compute aggregate statistics from a list of feature dicts."""

    # Collect arrays for each metric
    brightness_means = [f["brightness"]["mean"] for f in features_list]
    brightness_stds = [f["brightness"]["std"] for f in features_list]
    warm_ratios = [f["color"]["warm_ratio"] for f in features_list]
    edge_densities = [f["structure"]["edge_density"] for f in features_list]
    coverages = [f["coverage"] for f in features_list]
    r_means = [f["color"]["r_mean"] for f in features_list]
    g_means = [f["color"]["g_mean"] for f in features_list]
    b_means = [f["color"]["b_mean"] for f in features_list]
    has_oranges = [f["color"]["has_orange"] for f in features_list]
    radial_profiles = [f["radial_profile"] for f in features_list]

    # Aggregate color histograms
    color_hists = np.array([f["color"]["histogram"] for f in features_list])
    mean_color_hist = np.mean(color_hists, axis=0).tolist()

    # Aggregate edge histograms
    edge_hists = np.array([f["structure"]["edge_histogram"] for f in features_list])
    mean_edge_hist = np.mean(edge_hists, axis=0).tolist()

    # Aggregate radial profiles
    radial_profiles_arr = np.array(radial_profiles)
    mean_radial = np.mean(radial_profiles_arr, axis=0).tolist()
    std_radial = np.std(radial_profiles_arr, axis=0).tolist()

    def compute_percentiles(arr):
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "p5": float(np.percentile(arr, 5)),
            "p25": float(np.percentile(arr, 25)),
            "median": float(np.median(arr)),
            "p75": float(np.percentile(arr, 75)),
            "p95": float(np.percentile(arr, 95)),
            "max": float(np.max(arr))
        }

    return {
        "brightness": compute_percentiles(brightness_means),
        "brightness_std": compute_percentiles(brightness_stds),
        "warm_ratio": compute_percentiles(warm_ratios),
        "edge_density": compute_percentiles(edge_densities),
        "coverage": compute_percentiles(coverages),
        "color": {
            "r_mean": compute_percentiles(r_means),
            "g_mean": compute_percentiles(g_means),
            "b_mean": compute_percentiles(b_means),
            "has_orange": compute_percentiles(has_oranges),
            "mean_histogram": mean_color_hist
        },
        "structure": {
            "mean_edge_histogram": mean_edge_hist
        },
        "radial_profile": {
            "mean": mean_radial,
            "std": std_radial
        }
    }


# =============================================================================
# Ground Truth Comparison
# =============================================================================

def compare_to_ground_truth(
    image_path: str,
    effect_type: str = "sun",
    pass_threshold: float = 0.7
) -> GroundTruthComparison:
    """
    Compare a render against ground truth reference statistics.

    This is the PRIMARY evaluation for realism. It compares your render's
    feature distributions against real footage distributions.

    Args:
        image_path: Path to render to evaluate
        effect_type: Type of effect (sun, explosion, nebula, etc.)
        pass_threshold: Minimum overall similarity to pass (0-1)

    Returns:
        GroundTruthComparison with detailed similarity analysis
    """
    # Extract features from the render
    render_features = extract_image_features(image_path)

    # Get reference statistics
    ref_stats = compute_reference_statistics(effect_type)
    if "error" in ref_stats:
        return GroundTruthComparison(
            overall_similarity=0.0,
            color_distribution_match=0.0,
            brightness_distribution_match=0.0,
            structure_similarity=0.0,
            warm_ratio_match=0.0,
            closest_reference_frames=[],
            analysis={"error": ref_stats["error"]},
            recommendations=["Reference statistics not available"],
            passed=False,
            pass_threshold=pass_threshold
        )

    # Compare each dimension
    similarities = {}
    analysis = {}
    recommendations = []

    # 1. Brightness distribution match
    brightness_sim = _compare_to_distribution(
        render_features["brightness"]["mean"],
        ref_stats["brightness"]
    )
    similarities["brightness"] = brightness_sim

    if brightness_sim < 0.5:
        render_b = render_features["brightness"]["mean"]
        ref_b = ref_stats["brightness"]["median"]
        if render_b > ref_b * 1.3:
            analysis["brightness"] = f"Much brighter than reference (render: {render_b:.0f}, reference median: {ref_b:.0f})"
            recommendations.append("Reduce emission/brightness - real footage preserves HDR detail")
        elif render_b < ref_b * 0.7:
            analysis["brightness"] = f"Darker than reference (render: {render_b:.0f}, reference median: {ref_b:.0f})"
            recommendations.append("Increase emission or flame temperature")
        else:
            analysis["brightness"] = "Brightness outside typical range"
    else:
        analysis["brightness"] = "Brightness matches reference range"

    # 2. Warm ratio match (critical for sun)
    warm_sim = _compare_to_distribution(
        render_features["color"]["warm_ratio"],
        ref_stats["warm_ratio"]
    )
    similarities["warm_ratio"] = warm_sim

    render_warm = render_features["color"]["warm_ratio"]
    ref_warm = ref_stats["warm_ratio"]["median"]
    if warm_sim < 0.5:
        if render_warm < ref_warm * 0.5:
            analysis["warm_ratio"] = f"Not warm enough (render: {render_warm:.2f}, reference: {ref_warm:.2f})"
            recommendations.append(f"Increase warm colors - real sun has warm_ratio ~{ref_warm:.1f}")
        elif render_warm > ref_warm * 2:
            analysis["warm_ratio"] = f"Too saturated (render: {render_warm:.2f}, reference: {ref_warm:.2f})"
            recommendations.append("Reduce red saturation - may be over-processed")
        else:
            analysis["warm_ratio"] = "Color temperature outside typical range"
    else:
        analysis["warm_ratio"] = f"Color warmth matches reference (within {ref_warm:.1f} typical)"

    # 3. Structure/edge density match
    edge_sim = _compare_to_distribution(
        render_features["structure"]["edge_density"],
        ref_stats["edge_density"]
    )
    similarities["structure"] = edge_sim

    render_edge = render_features["structure"]["edge_density"]
    ref_edge = ref_stats["edge_density"]["median"]
    if edge_sim < 0.5:
        if render_edge < ref_edge * 0.5:
            analysis["structure"] = f"Lacks surface detail (render: {render_edge:.4f}, reference: {ref_edge:.4f})"
            recommendations.append("Add more turbulence/noise for visible granulation")
        elif render_edge > ref_edge * 2:
            analysis["structure"] = f"Over-textured (render: {render_edge:.4f}, reference: {ref_edge:.4f})"
            recommendations.append("Reduce noise/turbulence")
        else:
            analysis["structure"] = "Structure density atypical"
    else:
        analysis["structure"] = "Surface structure matches reference"

    # 4. Color histogram match (overall color distribution)
    color_sim = _histogram_similarity(
        render_features["color"]["histogram"],
        ref_stats["color"]["mean_histogram"]
    )
    similarities["color_distribution"] = color_sim

    if color_sim < 0.6:
        analysis["color_distribution"] = "Overall color distribution differs from reference"
        recommendations.append("Color balance differs from real footage")
    else:
        analysis["color_distribution"] = "Color distribution similar to reference"

    # 5. Coverage match
    coverage_sim = _compare_to_distribution(
        render_features["coverage"],
        ref_stats["coverage"]
    )
    similarities["coverage"] = coverage_sim

    # Compute overall similarity (weighted average)
    weights = {
        "brightness": 0.15,
        "warm_ratio": 0.25,  # Critical for sun
        "structure": 0.25,
        "color_distribution": 0.25,
        "coverage": 0.10
    }

    overall = sum(similarities[k] * weights[k] for k in weights)

    # Find closest reference frames (if we have individual frame data cached)
    closest_frames = []  # Would need per-frame feature storage for this

    passed = overall >= pass_threshold

    if passed:
        recommendations = ["Quality matches reference footage"] + recommendations[:2]

    return GroundTruthComparison(
        overall_similarity=round(overall, 4),
        color_distribution_match=round(similarities["color_distribution"], 4),
        brightness_distribution_match=round(similarities["brightness"], 4),
        structure_similarity=round(similarities["structure"], 4),
        warm_ratio_match=round(similarities["warm_ratio"], 4),
        closest_reference_frames=closest_frames,
        analysis=analysis,
        recommendations=recommendations,
        passed=passed,
        pass_threshold=pass_threshold
    )


def _compare_to_distribution(value: float, dist: Dict[str, float]) -> float:
    """
    Compare a single value to a reference distribution.

    Returns similarity score 0-1 based on where value falls in distribution.
    1.0 = within p25-p75 (typical range)
    0.5 = within p5-p95 (acceptable range)
    0.0 = beyond p5/p95 (outside reference range)
    """
    p5 = dist.get("p5", dist.get("min", 0))
    p25 = dist.get("p25", p5)
    p75 = dist.get("p75", dist.get("max", 255))
    p95 = dist.get("p95", p75)
    median = dist.get("median", (p25 + p75) / 2)

    if p25 <= value <= p75:
        # Within typical range
        return 1.0
    elif p5 <= value <= p95:
        # Within acceptable range but atypical
        if value < p25:
            return 0.5 + 0.5 * (value - p5) / (p25 - p5)
        else:
            return 0.5 + 0.5 * (p95 - value) / (p95 - p75)
    else:
        # Outside reference range
        if value < p5:
            return max(0, 0.5 * (1 - (p5 - value) / max(p5, 1)))
        else:
            return max(0, 0.5 * (1 - (value - p95) / max(p95, 1)))


def _histogram_similarity(hist1: List[float], hist2: List[float]) -> float:
    """
    Compute histogram similarity using Bhattacharyya coefficient.

    Returns 0-1 where 1 = identical distributions.
    """
    h1 = np.array(hist1)
    h2 = np.array(hist2)

    # Normalize
    h1 = h1 / (h1.sum() + 1e-10)
    h2 = h2 / (h2.sum() + 1e-10)

    # Bhattacharyya coefficient
    bc = np.sum(np.sqrt(h1 * h2))
    return float(bc)


# =============================================================================
# Solar-Specific Metrics
# =============================================================================

def evaluate_solar_specific(image_path: str) -> SolarSpecificMetrics:
    """
    Evaluate sun/star-specific features that make a render look realistic.

    These are domain-specific metrics that generic VFX evaluation misses:
    - Granulation: The cellular convection pattern visible on sun's surface
    - Limb darkening: Edges appear darker than center (real optical effect)
    - Prominences: Eruptions and loops extending from surface
    - Corona: Faint outer atmosphere glow
    """
    from PIL import Image

    img = Image.open(image_path).convert('RGB')
    arr = np.array(img, dtype=np.float32)

    h, w = arr.shape[:2]
    cy, cx = h // 2, w // 2

    gray = arr.mean(axis=-1)

    # Detect sun disk (bright circular region)
    threshold = np.percentile(gray, 70)
    disk_mask = gray > threshold

    # 1. Granulation detection
    # Look for small-scale cellular texture within the disk
    # Real granulation has ~1000km cells, appears as fine texture

    # High-pass filter to detect small-scale structure
    from scipy import ndimage
    if disk_mask.sum() > 0:
        disk_region = gray * disk_mask
        smooth = ndimage.uniform_filter(disk_region, size=20)
        high_freq = np.abs(disk_region - smooth)
        granulation_score = float(np.mean(high_freq[disk_mask]) / 50)  # Normalize
        granulation_score = min(1.0, granulation_score)
        has_granulation = granulation_score > 0.3
    else:
        has_granulation = False
        granulation_score = 0.0

    # 2. Limb darkening detection
    # Brightness should decrease from center to edge
    y_coords, x_coords = np.ogrid[:h, :w]
    r_dist = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)

    # Compare center brightness to edge brightness
    r_max = min(cx, cy)
    center_mask = r_dist < r_max * 0.3
    edge_mask = (r_dist > r_max * 0.7) & (r_dist < r_max * 1.0) & disk_mask

    if center_mask.sum() > 0 and edge_mask.sum() > 0:
        center_brightness = np.mean(gray[center_mask])
        edge_brightness = np.mean(gray[edge_mask])

        # Real sun has ~40% limb darkening
        darkening_ratio = 1 - (edge_brightness / max(center_brightness, 1))
        limb_darkening_score = min(1.0, darkening_ratio / 0.4)  # Normalize to expected 40%
        has_limb_darkening = darkening_ratio > 0.1
    else:
        has_limb_darkening = False
        limb_darkening_score = 0.0

    # 3. Prominence detection
    # Look for bright extensions beyond the disk edge
    outside_disk = ~disk_mask & (r_dist < r_max * 1.5)
    if outside_disk.sum() > 0:
        outside_brightness = np.mean(gray[outside_disk])
        prominence_ratio = outside_brightness / max(np.mean(gray[disk_mask]) if disk_mask.sum() > 0 else 1, 1)

        # Check for localized bright spots outside disk
        prominence_threshold = np.percentile(gray[disk_mask], 50) if disk_mask.sum() > 0 else 128
        prominence_pixels = outside_disk & (gray > prominence_threshold * 0.3)
        prominence_coverage = prominence_pixels.sum() / max(outside_disk.sum(), 1)

        has_prominences = prominence_coverage > 0.01  # At least 1% of outer region
        prominence_score = min(1.0, prominence_coverage * 10)
    else:
        has_prominences = False
        prominence_score = 0.0

    # 4. Corona detection
    # Faint glow extending beyond prominences
    far_outside = (r_dist > r_max * 1.3) & (r_dist < r_max * 2.0)
    if far_outside.sum() > 0:
        corona_brightness = np.mean(gray[far_outside])
        corona_visible = corona_brightness > 10  # Some glow present
    else:
        corona_visible = False

    # 5. Estimate color temperature from RGB ratio
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    if disk_mask.sum() > 0:
        r_avg = np.mean(r[disk_mask])
        g_avg = np.mean(g[disk_mask])
        b_avg = np.mean(b[disk_mask])

        # Approximate color temperature (simplified Wien's law approximation)
        # Real sun is ~5778K
        if r_avg > b_avg:
            ratio = b_avg / max(r_avg, 1)
            # Map ratio to temperature (rough approximation)
            # ratio ~0.3 = 3000K, ratio ~0.5 = 5000K, ratio ~0.8 = 8000K
            color_temp_kelvin = 3000 + (ratio / 0.5) * 4000
        else:
            color_temp_kelvin = 8000 + (b_avg / max(r_avg, 1) - 1) * 3000
        color_temp_kelvin = float(max(2500, min(15000, color_temp_kelvin)))
    else:
        color_temp_kelvin = 5778.0  # Default to solar temperature

    return SolarSpecificMetrics(
        has_granulation=has_granulation,
        granulation_score=round(granulation_score, 3),
        has_limb_darkening=has_limb_darkening,
        limb_darkening_score=round(limb_darkening_score, 3),
        has_prominences=has_prominences,
        prominence_score=round(prominence_score, 3),
        corona_visible=corona_visible,
        color_temperature_kelvin=round(color_temp_kelvin, 0)
    )


# =============================================================================
# Combined Ground Truth Evaluation
# =============================================================================

def evaluate_against_ground_truth(
    image_path: str,
    effect_type: str = "sun",
    pass_threshold: float = 0.65
) -> Dict[str, Any]:
    """
    Complete ground truth evaluation combining distribution matching
    with effect-specific feature detection.

    This is the recommended evaluation function for production use.

    Args:
        image_path: Path to render to evaluate
        effect_type: Type of effect ("sun", "star", etc.)
        pass_threshold: Minimum similarity score to pass (0-1)

    Returns:
        Comprehensive evaluation results with:
        - overall_score: 0-100 (higher = more realistic)
        - passed: True if meets threshold
        - distribution_comparison: How features match reference
        - effect_specific: Domain-specific quality metrics
        - recommendations: What to improve
    """
    results = {
        "image_path": image_path,
        "effect_type": effect_type,
        "pass_threshold": pass_threshold
    }

    # 1. Ground truth distribution comparison
    gt_comparison = compare_to_ground_truth(image_path, effect_type, pass_threshold)
    results["distribution_comparison"] = asdict(gt_comparison)

    # 2. Effect-specific metrics (if applicable)
    if effect_type in ["sun", "star"]:
        solar_metrics = evaluate_solar_specific(image_path)
        results["solar_specific"] = asdict(solar_metrics)

        # Incorporate solar-specific scores into overall
        solar_bonus = (
            solar_metrics.granulation_score * 0.15 +
            solar_metrics.limb_darkening_score * 0.15 +
            solar_metrics.prominence_score * 0.1
        )
    else:
        solar_bonus = 0.0

    # 3. Compute overall score (0-100)
    base_score = gt_comparison.overall_similarity * 100
    overall_score = min(100, base_score + solar_bonus * 10)

    results["overall_score"] = round(overall_score, 1)
    results["passed"] = overall_score >= (pass_threshold * 100)

    # 4. Aggregate recommendations
    all_recommendations = list(gt_comparison.recommendations)

    if effect_type in ["sun", "star"]:
        solar = results.get("solar_specific", {})
        if not solar.get("has_granulation", True):
            all_recommendations.append("Add surface granulation texture (small-scale cellular pattern)")
        if not solar.get("has_limb_darkening", True):
            all_recommendations.append("Add limb darkening (edges should be ~40% darker than center)")

        color_temp = solar.get("color_temperature_kelvin", 5778)
        if color_temp < 4500:
            all_recommendations.append(f"Color temperature too low ({color_temp:.0f}K) - sun is ~5778K")
        elif color_temp > 7000:
            all_recommendations.append(f"Color temperature too high ({color_temp:.0f}K) - sun is ~5778K")

    results["recommendations"] = all_recommendations[:5]  # Top 5

    return results


# =============================================================================
# CLI Interface
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ground_truth_evaluation.py <image_path> [effect_type]")
        print("\nExample: python ground_truth_evaluation.py render.png sun")
        sys.exit(1)

    image_path = sys.argv[1]
    effect_type = sys.argv[2] if len(sys.argv) > 2 else "sun"

    print(f"Evaluating {image_path} as {effect_type}...")

    result = evaluate_against_ground_truth(image_path, effect_type)
    print(json.dumps(result, indent=2))
