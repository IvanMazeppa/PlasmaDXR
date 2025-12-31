#!/usr/bin/env python3
"""
Evaluation-Tracker Bridge

Connects evaluation diagnostics to experiment-tracker's knowledge base,
enabling LEARNED prescriptions instead of hardcoded Blender tips.

The flow:
1. Evaluation diagnoses issues (WHERE and WHAT)
2. This bridge maps issues to experiment-tracker categories
3. Experiment-tracker suggests fixes based on past successes
4. Script-generator uses those suggestions

This keeps evaluation DIAGNOSTIC and makes prescriptions LEARNABLE.
"""

import json
import sys
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

# Add experiment-tracker to path for direct import
EXPERIMENT_TRACKER_PATH = Path(__file__).parent.parent / "experiment-tracker"
sys.path.insert(0, str(EXPERIMENT_TRACKER_PATH))


@dataclass
class IssueDiagnosis:
    """A diagnosed issue mapped to experiment-tracker categories."""
    category: str           # experiment-tracker issue category
    severity: str           # critical, moderate, minor
    description: str        # Human-readable description
    region: Optional[str]   # Where in image (for spatial issues)
    render_value: float     # What the render shows
    reference_value: float  # What reference shows
    deviation_percent: float


@dataclass
class SuggestedFix:
    """A suggested fix from experiment-tracker knowledge."""
    parameter: str          # Blender parameter to change
    change: str             # What to do (increase, decrease, set to X)
    confidence: float       # How confident based on past success
    rationale: str          # Why this might help
    past_success_rate: Optional[float]  # Success rate from knowledge base
    warnings: List[str]     # Known gotchas


@dataclass
class ConnectedEvaluation:
    """Complete evaluation with connected suggestions."""
    # Evaluation results
    overall_score: float
    passed: bool

    # Diagnosed issues
    issues: List[IssueDiagnosis]

    # Suggested fixes from knowledge base
    suggested_fixes: List[SuggestedFix]

    # Summary for iteration
    primary_issue: str
    recommended_action: str

    # Experiment tracking
    should_record_baseline: bool
    suggested_hypothesis: str


# =============================================================================
# Issue Category Mapping
# =============================================================================

# Map evaluation diagnostics to experiment-tracker categories
ISSUE_CATEGORY_MAP = {
    # Brightness issues
    "brightness_too_high": {
        "tracker_category": "intensity",
        "related_params": ["blackbody_intensity", "emission_strength", "flame_max_temp"],
        "suggested_direction": "decrease"
    },
    "brightness_too_low": {
        "tracker_category": "intensity",
        "related_params": ["blackbody_intensity", "emission_strength", "flame_max_temp"],
        "suggested_direction": "increase"
    },

    # Color temperature issues
    "color_too_cool": {
        "tracker_category": "color_temperature",
        "related_params": ["flame_color_temperature", "blackbody_tint", "color_ramp_position"],
        "suggested_direction": "decrease_temp_for_orange"
    },
    "color_too_warm": {
        "tracker_category": "color_temperature",
        "related_params": ["flame_color_temperature", "blackbody_tint"],
        "suggested_direction": "increase_temp"
    },

    # Structure issues
    "lacks_granulation": {
        "tracker_category": "texture",
        "related_params": ["noise_scale", "turbulence", "vorticity"],
        "suggested_direction": "increase"
    },
    "lacks_limb_darkening": {
        "tracker_category": "shading",
        "related_params": ["volume_absorption", "density_falloff", "emission_falloff"],
        "suggested_direction": "add_radial_falloff"
    },

    # Prominence issues (NEW - for solar prominences)
    "prominence_missing": {
        "tracker_category": "prominence",
        "related_params": ["effector_strength", "outflow_velocity", "turbulence"],
        "suggested_direction": "increase"
    },
    "prominence_wrong_shape": {
        "tracker_category": "prominence_shape",
        "related_params": ["effector_direction", "gravity_strength", "curl_strength"],
        "suggested_direction": "adjust"
    },
    "prominence_wrong_color": {
        "tracker_category": "prominence_color",
        "related_params": ["emission_color", "temperature_gradient", "blackbody_offset"],
        "suggested_direction": "shift_to_red"
    },

    # Coverage issues
    "low_coverage": {
        "tracker_category": "framing",
        "related_params": ["domain_scale", "camera_distance", "particle_scale"],
        "suggested_direction": "adjust_framing"
    }
}

# Solar-specific issue patterns for experiment-tracker
SOLAR_ISSUE_PATTERNS = {
    "color_temperature": [
        {
            "param": "flame_color",
            "change": "shift toward orange (decrease temperature)",
            "confidence": 0.85,
            "note": "Real sun appears orange due to Rayleigh scattering, not actual 5778K blackbody"
        },
        {
            "param": "blackbody_intensity",
            "change": "reduce to avoid washing out to white",
            "confidence": 0.75
        },
        {
            "param": "emission_color",
            "change": "add orange/red tint",
            "confidence": 0.7
        }
    ],
    "prominence": [
        {
            "param": "effector_strength",
            "change": "increase for more dramatic loops",
            "confidence": 0.8,
            "note": "Prominences need strong outward force against gravity"
        },
        {
            "param": "magnetic_curl",
            "change": "increase for looping behavior",
            "confidence": 0.7
        }
    ],
    "prominence_color": [
        {
            "param": "prominence_temperature",
            "change": "set lower than disk (~8000K vs 5778K)",
            "confidence": 0.85,
            "note": "Prominences are cooler plasma, appear redder"
        }
    ],
    "limb_darkening": [
        {
            "param": "volume_absorption",
            "change": "increase at edges via radial gradient",
            "confidence": 0.8,
            "note": "Real limb darkening is ~40% at edge vs center"
        },
        {
            "param": "emission_falloff",
            "change": "add radial falloff function",
            "confidence": 0.75
        }
    ]
}


# =============================================================================
# Core Functions
# =============================================================================

def diagnose_issues_for_tracker(evaluation_result: Dict[str, Any]) -> List[IssueDiagnosis]:
    """
    Convert evaluation results into experiment-tracker issue categories.

    This is the KEY bridge function - it maps diagnostic output to
    actionable categories that experiment-tracker can look up.
    """
    issues = []

    # Extract from ground truth comparison
    dist_comp = evaluation_result.get("distribution_comparison", {})

    # Brightness issues
    brightness_match = dist_comp.get("brightness_distribution_match", 1.0)
    if brightness_match < 0.5:
        analysis = dist_comp.get("analysis", {}).get("brightness", "")
        if "brighter" in analysis.lower() or "overexposed" in analysis.lower():
            issues.append(IssueDiagnosis(
                category="brightness_too_high",
                severity="critical" if brightness_match < 0.2 else "moderate",
                description=analysis,
                region=None,
                render_value=brightness_match,
                reference_value=1.0,
                deviation_percent=(1.0 - brightness_match) * 100
            ))
        else:
            issues.append(IssueDiagnosis(
                category="brightness_too_low",
                severity="critical" if brightness_match < 0.2 else "moderate",
                description=analysis,
                region=None,
                render_value=brightness_match,
                reference_value=1.0,
                deviation_percent=(1.0 - brightness_match) * 100
            ))

    # Color/warm ratio issues
    warm_match = dist_comp.get("warm_ratio_match", 1.0)
    if warm_match < 0.5:
        analysis = dist_comp.get("analysis", {}).get("warm_ratio", "")
        if "not warm enough" in analysis.lower() or "cool" in analysis.lower():
            issues.append(IssueDiagnosis(
                category="color_too_cool",
                severity="critical" if warm_match < 0.2 else "moderate",
                description=analysis,
                region=None,
                render_value=warm_match,
                reference_value=1.0,
                deviation_percent=(1.0 - warm_match) * 100
            ))

    # Solar-specific issues
    solar = evaluation_result.get("solar_specific", {})

    if not solar.get("has_granulation", True):
        issues.append(IssueDiagnosis(
            category="lacks_granulation",
            severity="moderate",
            description="Missing surface granulation texture",
            region="disk",
            render_value=solar.get("granulation_score", 0),
            reference_value=0.5,
            deviation_percent=100
        ))

    if not solar.get("has_limb_darkening", True):
        issues.append(IssueDiagnosis(
            category="lacks_limb_darkening",
            severity="moderate",
            description="Missing limb darkening effect (edges should be 40% darker)",
            region="limb",
            render_value=solar.get("limb_darkening_score", 0),
            reference_value=0.4,
            deviation_percent=100
        ))

    # Prominence-specific issues (for prominence renders)
    if not solar.get("has_prominences", True):
        if "prominence" in evaluation_result.get("image_path", "").lower():
            issues.append(IssueDiagnosis(
                category="prominence_missing",
                severity="critical",
                description="No visible prominences despite being a prominence render",
                region="corona",
                render_value=solar.get("prominence_score", 0),
                reference_value=0.5,
                deviation_percent=100
            ))

    # Color temperature estimation
    color_temp = solar.get("color_temperature_kelvin", 5778)
    if color_temp > 6500:  # Too blue
        issues.append(IssueDiagnosis(
            category="color_too_cool",
            severity="critical" if color_temp > 8000 else "moderate",
            description=f"Color temperature {color_temp:.0f}K is too high (appears blue-white, should be orange ~3500K as seen from Earth)",
            region="disk",
            render_value=color_temp,
            reference_value=3500,  # Apparent temperature, not actual
            deviation_percent=abs(color_temp - 3500) / 3500 * 100
        ))

    return issues


def get_suggestions_from_tracker(issues: List[IssueDiagnosis]) -> List[SuggestedFix]:
    """
    Query experiment-tracker for suggestions based on diagnosed issues.

    First tries the knowledge base, then falls back to known patterns.
    """
    suggestions = []

    try:
        # Try to import and query experiment-tracker
        from tracker import ExperimentTracker
        tracker = ExperimentTracker()

        for issue in issues:
            # Query knowledge base
            category = issue.category
            mapping = ISSUE_CATEGORY_MAP.get(category, {})
            tracker_category = mapping.get("tracker_category", category)

            # Get suggestions from tracker
            tracker_suggestions = tracker.suggest_experiments(
                issue=f"{tracker_category}: {issue.description}",
                current_params={},
                current_scores={"overall": issue.render_value}
            )

            for ts in tracker_suggestions[:2]:  # Top 2 per issue
                # Get warnings for this parameter
                warnings = tracker.get_warnings_for_change(
                    parameter=ts.parameter,
                    change_type=ts.change
                )

                suggestions.append(SuggestedFix(
                    parameter=ts.parameter,
                    change=ts.change,
                    confidence=ts.confidence,
                    rationale=f"Based on experiment-tracker knowledge for '{tracker_category}'",
                    past_success_rate=None,  # Would need to query DB
                    warnings=warnings
                ))

    except Exception as e:
        # Fall back to built-in patterns
        for issue in issues:
            category = issue.category

            # Check solar-specific patterns first
            tracker_category = ISSUE_CATEGORY_MAP.get(category, {}).get("tracker_category", category)
            patterns = SOLAR_ISSUE_PATTERNS.get(tracker_category, [])

            for pattern in patterns[:2]:
                suggestions.append(SuggestedFix(
                    parameter=pattern["param"],
                    change=pattern["change"],
                    confidence=pattern.get("confidence", 0.5),
                    rationale=pattern.get("note", f"Common fix for {category}"),
                    past_success_rate=None,
                    warnings=[]
                ))

    # Deduplicate by parameter
    seen_params = set()
    unique_suggestions = []
    for s in suggestions:
        if s.parameter not in seen_params:
            seen_params.add(s.parameter)
            unique_suggestions.append(s)

    return unique_suggestions[:5]  # Top 5


def create_connected_evaluation(
    image_path: str,
    effect_type: str = "sun",
    pass_threshold: float = 0.65
) -> ConnectedEvaluation:
    """
    Run evaluation AND connect to experiment-tracker for suggestions.

    This is the main function that bridges evaluation to learned prescriptions.
    """
    from ground_truth_evaluation import evaluate_against_ground_truth

    # 1. Run ground truth evaluation
    eval_result = evaluate_against_ground_truth(image_path, effect_type, pass_threshold)

    # 2. Diagnose issues for tracker
    issues = diagnose_issues_for_tracker(eval_result)

    # 3. Get suggestions from tracker/knowledge base
    suggestions = get_suggestions_from_tracker(issues)

    # 4. Determine primary issue and recommended action
    if issues:
        critical_issues = [i for i in issues if i.severity == "critical"]
        primary = critical_issues[0] if critical_issues else issues[0]
        primary_issue = f"{primary.category}: {primary.description}"

        if suggestions:
            top_fix = suggestions[0]
            recommended_action = f"Try: {top_fix.parameter} - {top_fix.change} (confidence: {top_fix.confidence:.0%})"
        else:
            recommended_action = "No known fixes in knowledge base - record this experiment"
    else:
        primary_issue = "No significant issues detected"
        recommended_action = "Quality acceptable - consider finalizing asset"

    # 5. Generate hypothesis for experiment recording
    if issues and suggestions:
        hypothesis = f"Changing {suggestions[0].parameter} will improve {issues[0].category}"
    else:
        hypothesis = "Current parameters may be acceptable"

    return ConnectedEvaluation(
        overall_score=eval_result["overall_score"],
        passed=eval_result["passed"],
        issues=issues,
        suggested_fixes=suggestions,
        primary_issue=primary_issue,
        recommended_action=recommended_action,
        should_record_baseline=not eval_result["passed"],
        suggested_hypothesis=hypothesis
    )


# =============================================================================
# Prominence-Specific Analysis
# =============================================================================

def analyze_prominence_quality(
    render_path: str,
    reference_path: str
) -> Dict[str, Any]:
    """
    Specialized analysis for solar prominence renders.

    Prominences need special evaluation because they're:
    - Loops extending from the solar limb
    - Cooler than the disk (redder)
    - Dynamic (arcing, erupting)
    - Visible against dark space
    """
    from PIL import Image
    import numpy as np

    render_img = Image.open(render_path).convert('RGB')
    reference_img = Image.open(reference_path).convert('RGB')

    # Resize reference to match render if needed
    if render_img.size != reference_img.size:
        reference_img = reference_img.resize(render_img.size, Image.LANCZOS)

    render = np.array(render_img, dtype=np.float32)
    reference = np.array(reference_img, dtype=np.float32)

    h, w = render.shape[:2]
    cy, cx = h // 2, w // 2
    y_coords, x_coords = np.ogrid[:h, :w]
    r_dist = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
    r_max = min(cx, cy) * 0.45  # Sun disk radius (rough estimate)

    # Define regions
    disk_mask = r_dist < r_max
    prominence_zone = (r_dist >= r_max) & (r_dist < r_max * 2.0)

    analysis = {
        "disk_analysis": {},
        "prominence_analysis": {},
        "comparison": {},
        "issues": [],
        "suggestions": []
    }

    # Analyze disk
    if disk_mask.sum() > 0:
        render_disk_brightness = np.mean(render[disk_mask])
        ref_disk_brightness = np.mean(reference[disk_mask])

        analysis["disk_analysis"] = {
            "render_brightness": float(render_disk_brightness),
            "reference_brightness": float(ref_disk_brightness),
            "deviation": float(abs(render_disk_brightness - ref_disk_brightness) / max(ref_disk_brightness, 1) * 100)
        }

    # Analyze prominence zone
    if prominence_zone.sum() > 0:
        # Find bright pixels in prominence zone (actual prominences)
        render_prom = render[prominence_zone]
        ref_prom = reference[prominence_zone]

        render_brightness_threshold = np.percentile(render_prom.mean(axis=-1), 70)
        ref_brightness_threshold = np.percentile(ref_prom.mean(axis=-1), 70)

        # Count "prominence pixels" (bright enough to be visible)
        render_prom_mask = render_prom.mean(axis=-1) > render_brightness_threshold
        ref_prom_mask = ref_prom.mean(axis=-1) > ref_brightness_threshold

        render_prom_coverage = render_prom_mask.sum() / len(render_prom_mask)
        ref_prom_coverage = ref_prom_mask.sum() / len(ref_prom_mask)

        # Color analysis of prominences
        if render_prom_mask.sum() > 0:
            render_prom_color = render_prom[render_prom_mask].mean(axis=0)
            render_prom_warm = render_prom_color[0] / max(render_prom_color[2], 1)
        else:
            render_prom_color = [0, 0, 0]
            render_prom_warm = 1.0

        if ref_prom_mask.sum() > 0:
            ref_prom_color = ref_prom[ref_prom_mask].mean(axis=0)
            ref_prom_warm = ref_prom_color[0] / max(ref_prom_color[2], 1)
        else:
            ref_prom_color = [0, 0, 0]
            ref_prom_warm = 1.0

        analysis["prominence_analysis"] = {
            "render_coverage": float(render_prom_coverage),
            "reference_coverage": float(ref_prom_coverage),
            "render_warm_ratio": float(render_prom_warm),
            "reference_warm_ratio": float(ref_prom_warm),
            "coverage_ratio": float(render_prom_coverage / max(ref_prom_coverage, 0.01))
        }

        # Generate issues
        if render_prom_coverage < ref_prom_coverage * 0.3:
            analysis["issues"].append({
                "type": "prominence_missing",
                "severity": "critical",
                "description": f"Prominences cover {render_prom_coverage*100:.1f}% vs reference {ref_prom_coverage*100:.1f}%"
            })
            analysis["suggestions"].append({
                "parameter": "effector_strength",
                "change": "increase significantly (2-3x)",
                "confidence": 0.8
            })

        if render_prom_warm < ref_prom_warm * 0.5:
            analysis["issues"].append({
                "type": "prominence_too_blue",
                "severity": "critical",
                "description": f"Prominence warm_ratio {render_prom_warm:.2f} vs reference {ref_prom_warm:.2f}"
            })
            analysis["suggestions"].append({
                "parameter": "prominence_temperature",
                "change": "decrease (prominences should be cooler/redder than disk)",
                "confidence": 0.85
            })

    return analysis


# =============================================================================
# CLI and Testing
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python evaluation_tracker_bridge.py <image_path> [reference_path]")
        sys.exit(1)

    image_path = sys.argv[1]
    reference_path = sys.argv[2] if len(sys.argv) > 2 else None

    print("=" * 70)
    print("CONNECTED EVALUATION (Evaluation + Experiment-Tracker)")
    print("=" * 70)

    result = create_connected_evaluation(image_path, "sun", 0.65)

    print(f"\nOverall Score: {result.overall_score:.1f}/100")
    print(f"Passed: {result.passed}")
    print(f"\nPrimary Issue: {result.primary_issue}")
    print(f"Recommended Action: {result.recommended_action}")

    if result.issues:
        print(f"\nDiagnosed Issues ({len(result.issues)}):")
        for issue in result.issues:
            print(f"  [{issue.severity.upper()}] {issue.category}: {issue.description}")

    if result.suggested_fixes:
        print(f"\nSuggested Fixes ({len(result.suggested_fixes)}):")
        for fix in result.suggested_fixes:
            print(f"  • {fix.parameter}: {fix.change}")
            print(f"    Confidence: {fix.confidence:.0%}, Rationale: {fix.rationale}")
            if fix.warnings:
                print(f"    ⚠️ Warnings: {', '.join(fix.warnings)}")

    print(f"\nExperiment Tracking:")
    print(f"  Should record baseline: {result.should_record_baseline}")
    print(f"  Suggested hypothesis: {result.suggested_hypothesis}")

    # If reference provided, do prominence-specific analysis
    if reference_path and "prominence" in image_path.lower():
        print("\n" + "=" * 70)
        print("PROMINENCE-SPECIFIC ANALYSIS")
        print("=" * 70)
        prom_analysis = analyze_prominence_quality(image_path, reference_path)

        if prom_analysis["issues"]:
            print("\nProminence Issues:")
            for issue in prom_analysis["issues"]:
                print(f"  [{issue['severity'].upper()}] {issue['type']}: {issue['description']}")

        if prom_analysis["suggestions"]:
            print("\nProminence-Specific Suggestions:")
            for sug in prom_analysis["suggestions"]:
                print(f"  • {sug['parameter']}: {sug['change']} (confidence: {sug['confidence']:.0%})")
