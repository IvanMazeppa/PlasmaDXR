#!/usr/bin/env python3
"""
Spatial Diagnostics Module

Provides LOCALIZED feedback about WHERE a render differs from reference,
not just aggregate scores. This helps artists understand the spatial
distribution of problems.

Philosophy: Diagnose thoroughly, but don't prescribe Blender fixes.
That's the experiment-tracker's job (learning from past successes).
"""

import json
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import numpy as np


@dataclass
class SpatialDiagnostic:
    """Localized analysis of a specific image region."""
    region: str  # "center", "edge", "top", "bottom", etc.
    issue: str   # What's wrong
    severity: str  # "critical", "moderate", "minor"
    render_value: float
    reference_value: float
    deviation_percent: float


@dataclass
class SpatialAnalysis:
    """Complete spatial breakdown of render vs reference."""
    diagnostics: List[SpatialDiagnostic]
    problem_regions: List[str]  # Regions with issues
    healthy_regions: List[str]  # Regions that match well
    summary: str  # Human-readable summary
    heatmap_data: Optional[Dict[str, Any]] = None  # For visualization


def analyze_spatial_distribution(
    image_path: str,
    reference_stats: Dict[str, Any],
    effect_type: str = "sun"
) -> SpatialAnalysis:
    """
    Analyze WHERE in the image problems occur.

    Instead of just "warm_ratio too low", tells you:
    "The center is 7200K (too blue), edges are 4500K (acceptable).
     The warmth issue is concentrated in the disk core."

    This helps iteration because you know WHERE to focus changes.
    """
    from PIL import Image

    img = Image.open(image_path).convert('RGB')
    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    cy, cx = h // 2, w // 2

    diagnostics = []
    problem_regions = []
    healthy_regions = []

    # Define regions for analysis
    regions = _define_regions(h, w, effect_type)

    # Analyze each region
    for region_name, mask in regions.items():
        if mask.sum() == 0:
            continue

        region_analysis = _analyze_region(arr, mask, region_name, reference_stats)

        for diag in region_analysis:
            diagnostics.append(diag)
            if diag.severity in ["critical", "moderate"]:
                if region_name not in problem_regions:
                    problem_regions.append(region_name)
            else:
                if region_name not in healthy_regions and region_name not in problem_regions:
                    healthy_regions.append(region_name)

    # Generate summary
    summary = _generate_spatial_summary(diagnostics, problem_regions, healthy_regions)

    # Generate heatmap data for visualization
    heatmap = _compute_deviation_heatmap(arr, reference_stats)

    return SpatialAnalysis(
        diagnostics=diagnostics,
        problem_regions=problem_regions,
        healthy_regions=healthy_regions,
        summary=summary,
        heatmap_data=heatmap
    )


def _define_regions(h: int, w: int, effect_type: str) -> Dict[str, np.ndarray]:
    """Define analysis regions based on effect type."""
    cy, cx = h // 2, w // 2
    y_coords, x_coords = np.ogrid[:h, :w]
    r_dist = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
    r_max = min(cx, cy)

    if effect_type in ["sun", "star"]:
        # Radial regions for circular objects
        return {
            "core": r_dist < r_max * 0.3,
            "mid_disk": (r_dist >= r_max * 0.3) & (r_dist < r_max * 0.7),
            "limb": (r_dist >= r_max * 0.7) & (r_dist < r_max * 1.0),
            "corona": (r_dist >= r_max * 1.0) & (r_dist < r_max * 1.5),
            "background": r_dist >= r_max * 1.5
        }
    else:
        # Quadrant regions for explosions/general effects
        return {
            "center": r_dist < r_max * 0.3,
            "top": (y_coords < h * 0.3) & (r_dist < r_max * 1.5),
            "bottom": (y_coords > h * 0.7) & (r_dist < r_max * 1.5),
            "left": (x_coords < w * 0.3) & (r_dist < r_max * 1.5),
            "right": (x_coords > w * 0.7) & (r_dist < r_max * 1.5),
        }


def _analyze_region(
    arr: np.ndarray,
    mask: np.ndarray,
    region_name: str,
    ref_stats: Dict[str, Any]
) -> List[SpatialDiagnostic]:
    """Analyze a single region against reference statistics."""
    diagnostics = []

    if mask.sum() == 0:
        return diagnostics

    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # Regional brightness
    region_brightness = float(np.mean(arr[mask]))
    ref_brightness = ref_stats.get("brightness", {}).get("median", 60)

    brightness_dev = abs(region_brightness - ref_brightness) / max(ref_brightness, 1) * 100

    if brightness_dev > 100:
        severity = "critical"
        if region_brightness > ref_brightness:
            issue = f"Severely overexposed ({region_brightness:.0f} vs ref {ref_brightness:.0f})"
        else:
            issue = f"Severely underexposed ({region_brightness:.0f} vs ref {ref_brightness:.0f})"
        diagnostics.append(SpatialDiagnostic(
            region=region_name,
            issue=issue,
            severity=severity,
            render_value=region_brightness,
            reference_value=ref_brightness,
            deviation_percent=brightness_dev
        ))
    elif brightness_dev > 50:
        severity = "moderate"
        direction = "brighter" if region_brightness > ref_brightness else "darker"
        issue = f"Noticeably {direction} than reference"
        diagnostics.append(SpatialDiagnostic(
            region=region_name,
            issue=issue,
            severity=severity,
            render_value=region_brightness,
            reference_value=ref_brightness,
            deviation_percent=brightness_dev
        ))

    # Regional color temperature (warm_ratio)
    region_r = np.mean(r[mask])
    region_b = np.mean(b[mask])
    region_warm = region_r / max(region_b, 1)

    ref_warm = ref_stats.get("warm_ratio", {}).get("median", 15.0)
    warm_dev = abs(region_warm - ref_warm) / max(ref_warm, 1) * 100

    if warm_dev > 80:
        severity = "critical"
        if region_warm < ref_warm:
            # Estimate color temperature from warm ratio
            if region_warm < 2:
                est_temp = "blue-white (~7000K+)"
            elif region_warm < 5:
                est_temp = "white (~6000K)"
            else:
                est_temp = "yellow-white (~5500K)"
            issue = f"Too cool/blue - appears {est_temp}, should be orange (~3500K)"
        else:
            issue = f"Over-saturated red (warm_ratio {region_warm:.1f} vs ref {ref_warm:.1f})"
        diagnostics.append(SpatialDiagnostic(
            region=region_name,
            issue=issue,
            severity=severity,
            render_value=region_warm,
            reference_value=ref_warm,
            deviation_percent=warm_dev
        ))
    elif warm_dev > 50:
        severity = "moderate"
        direction = "cooler (more blue)" if region_warm < ref_warm else "warmer (more red)"
        issue = f"Color is {direction} than reference"
        diagnostics.append(SpatialDiagnostic(
            region=region_name,
            issue=issue,
            severity=severity,
            render_value=region_warm,
            reference_value=ref_warm,
            deviation_percent=warm_dev
        ))

    return diagnostics


def _generate_spatial_summary(
    diagnostics: List[SpatialDiagnostic],
    problem_regions: List[str],
    healthy_regions: List[str]
) -> str:
    """Generate human-readable summary of spatial analysis."""
    if not diagnostics:
        return "No significant spatial deviations detected."

    critical = [d for d in diagnostics if d.severity == "critical"]
    moderate = [d for d in diagnostics if d.severity == "moderate"]

    parts = []

    if critical:
        regions_str = ", ".join(set(d.region for d in critical))
        parts.append(f"CRITICAL issues in: {regions_str}")

        # Group by issue type
        brightness_issues = [d for d in critical if "exposed" in d.issue.lower()]
        color_issues = [d for d in critical if "cool" in d.issue.lower() or "warm" in d.issue.lower()]

        if brightness_issues:
            parts.append(f"  - Brightness: {brightness_issues[0].issue}")
        if color_issues:
            parts.append(f"  - Color: {color_issues[0].issue}")

    if moderate:
        regions_str = ", ".join(set(d.region for d in moderate))
        parts.append(f"Moderate issues in: {regions_str}")

    if healthy_regions:
        parts.append(f"Good match in: {', '.join(healthy_regions)}")

    return "\n".join(parts)


def _compute_deviation_heatmap(
    arr: np.ndarray,
    ref_stats: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Compute per-pixel deviation from reference for visualization.

    Returns data that can be used to generate a heatmap showing
    where the render differs most from reference.
    """
    ref_brightness = ref_stats.get("brightness", {}).get("median", 60)
    ref_warm = ref_stats.get("warm_ratio", {}).get("median", 15.0)

    # Compute local brightness deviation
    local_brightness = np.mean(arr, axis=-1)
    brightness_dev = np.abs(local_brightness - ref_brightness) / max(ref_brightness, 1)

    # Compute local warm ratio deviation
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    local_warm = r / np.maximum(b, 1)
    warm_dev = np.abs(local_warm - ref_warm) / max(ref_warm, 1)

    # Combined deviation (could weight these)
    combined_dev = 0.5 * brightness_dev + 0.5 * np.minimum(warm_dev, 2.0) / 2.0

    return {
        "shape": list(arr.shape[:2]),
        "brightness_deviation_mean": float(np.mean(brightness_dev)),
        "warm_ratio_deviation_mean": float(np.mean(warm_dev)),
        "combined_deviation_mean": float(np.mean(combined_dev)),
        "max_deviation_location": _find_max_deviation_location(combined_dev),
        # For actual heatmap generation, would save the array
        # "heatmap_path": could save to file if needed
    }


def _find_max_deviation_location(dev_map: np.ndarray) -> Dict[str, Any]:
    """Find where the maximum deviation occurs."""
    max_idx = np.unravel_index(np.argmax(dev_map), dev_map.shape)
    h, w = dev_map.shape

    # Describe location in human terms
    y, x = max_idx
    y_pos = "top" if y < h * 0.33 else ("bottom" if y > h * 0.67 else "middle")
    x_pos = "left" if x < w * 0.33 else ("right" if x > w * 0.67 else "center")

    return {
        "pixel": [int(max_idx[1]), int(max_idx[0])],  # x, y
        "description": f"{y_pos}-{x_pos}",
        "deviation_value": float(dev_map[max_idx])
    }


# =============================================================================
# Integration with experiment-tracker
# =============================================================================

def format_for_experiment_tracker(
    spatial_analysis: SpatialAnalysis,
    overall_score: float
) -> Dict[str, Any]:
    """
    Format spatial diagnostics for experiment-tracker's knowledge base.

    The experiment-tracker can then learn:
    "When core region is too blue, increasing flame_color_temperature helped"

    This keeps the evaluation DIAGNOSTIC and lets experiment-tracker
    handle the PRESCRIPTION based on accumulated knowledge.
    """
    return {
        "diagnostics": [asdict(d) for d in spatial_analysis.diagnostics],
        "problem_regions": spatial_analysis.problem_regions,
        "overall_score": overall_score,
        "primary_issues": [
            d.issue for d in spatial_analysis.diagnostics
            if d.severity == "critical"
        ][:3],
        "suggested_focus_regions": spatial_analysis.problem_regions[:2],
        # This is the key insight for experiment-tracker:
        "issue_categories": _categorize_issues(spatial_analysis.diagnostics)
    }


def _categorize_issues(diagnostics: List[SpatialDiagnostic]) -> List[str]:
    """Categorize issues for experiment-tracker lookup."""
    categories = set()
    for d in diagnostics:
        if "exposed" in d.issue.lower():
            if "over" in d.issue.lower():
                categories.add("brightness_too_high")
            else:
                categories.add("brightness_too_low")
        if "cool" in d.issue.lower() or "blue" in d.issue.lower():
            categories.add("color_too_cool")
        if "warm" in d.issue.lower() or "red" in d.issue.lower():
            if "over" in d.issue.lower():
                categories.add("color_too_warm")
    return list(categories)


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python spatial_diagnostics.py <image_path>")
        sys.exit(1)

    from ground_truth_evaluation import compute_reference_statistics

    image_path = sys.argv[1]
    ref_stats = compute_reference_statistics("sun", 100)

    result = analyze_spatial_distribution(image_path, ref_stats, "sun")

    print("=" * 60)
    print("SPATIAL ANALYSIS")
    print("=" * 60)
    print(result.summary)
    print()
    print("Problem regions:", result.problem_regions)
    print("Healthy regions:", result.healthy_regions)
    print()
    print("Detailed diagnostics:")
    for d in result.diagnostics:
        print(f"  [{d.severity.upper()}] {d.region}: {d.issue}")
