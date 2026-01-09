"""
Function tool wrappers for asset-evaluator MCP server.

Exposes asset-evaluator capabilities to OpenAI Agents SDK agents.
Uses the consolidated v2 API (January 2026) for cleaner integration:
- evaluate_render_v2: Unified evaluation with profiles
- compare_renders_v2: Compare two iterations
- diagnose_issues_v2: VLM-powered issue detection
- list_renders_v2: Find available renders
- get_reference_stats_v2: Reference dataset statistics
- analyze_temporal_quality: Animation frame consistency

These wrappers translate MCP tool calls into @function_tool decorated
functions that agents can call directly.
"""

from __future__ import annotations

import json
from typing import Optional

from agents import function_tool

from utils.mcp_connection_pool import get_mcp_server


# =============================================================================
# UNIFIED EVALUATION (v2 API)
# =============================================================================

@function_tool
async def evaluate_render(
    render_path: str,
    reference_path: str = "",
    effect_type: str = "auto",
    profile: str = "standard",
    include_diagnostics: bool = True,
    include_suggestions: bool = True
) -> str:
    """
    Unified render evaluation with configurable depth profiles.

    This is the PRIMARY evaluation tool for the orchestrator.
    Consolidates 15+ legacy evaluation tools into a single interface.

    Profiles:
    - quick: LPIPS + SigLIP only (~2 seconds) - use for rapid iteration
    - standard: + TOPIQ, feature_cv (~10 seconds) - default for quality gates
    - comprehensive: + DINOv2, VLM diagnosis (~30 seconds) - deep analysis

    Args:
        render_path: Path to rendered image to evaluate
        reference_path: Optional reference image for comparison metrics
        effect_type: Effect category (auto, sun, explosion, nebula, fire, smoke)
        profile: Evaluation depth (quick, standard, comprehensive)
        include_diagnostics: Include VLM-based issue detection (comprehensive only)
        include_suggestions: Include parameter change suggestions

    Returns:
        JSON with:
        - overall_score: 0-100 (quality score)
        - passed: True if score >= 60
        - metric_scores: {lpips, siglip, topiq, qualiclip, structural_dino, feature_cv}
        - diagnostics: VLM-identified issues (if comprehensive + include_diagnostics)
        - suggestions: Parameter changes to try
        - profile_used: Which profile was run
        - evaluation_time_seconds: How long it took

    Example:
        evaluate_render(
            render_path="build/vdb_output/explosion_v1/render_0030.png",
            reference_path="assets/reference_images/explosion/ref_001.jpg",
            effect_type="explosion",
            profile="standard"
        )
    """
    server = await get_mcp_server("asset-evaluator")

    result = await server.call_tool(
        "evaluate_render_v2",
        {
            "render_path": render_path,
            "reference_path": reference_path,
            "effect_type": effect_type,
            "profile": profile,
            "include_diagnostics": include_diagnostics,
            "include_suggestions": include_suggestions,
        }
    )

    if hasattr(result, 'content') and result.content:
        return result.content[0].text if result.content else "{}"
    return str(result)


@function_tool
async def compare_renders(
    render_a: str,
    render_b: str,
    reference_path: str = "",
    comparison_type: str = "quality"
) -> str:
    """
    Compare two renders to determine which is better or if changes improved quality.

    Use this for:
    - A/B testing between iterations (did our changes help?)
    - Quality comparison between techniques
    - Finding the best render from a batch

    Args:
        render_a: Path to first render (e.g., iteration v9)
        render_b: Path to second render (e.g., iteration v10)
        reference_path: Optional reference image for context
        comparison_type: Type of comparison:
            - quality: Which render is objectively better?
            - iteration: Did changes from A to B improve quality?

    Returns:
        JSON with:
        - winner: "A", "B", or "similar"
        - score_a: Quality score for render A (0-100)
        - score_b: Quality score for render B (0-100)
        - improvements: What got better in the winner
        - regressions: What got worse
        - recommendation: What to do next

    Example:
        compare_renders(
            render_a="build/vdb_output/explosion_v9/render_0030.png",
            render_b="build/vdb_output/explosion_v10/render_0030.png",
            comparison_type="iteration"
        )
    """
    server = await get_mcp_server("asset-evaluator")

    result = await server.call_tool(
        "compare_renders_v2",
        {
            "render_a": render_a,
            "render_b": render_b,
            "reference_path": reference_path,
            "comparison_type": comparison_type,
        }
    )

    if hasattr(result, 'content') and result.content:
        return result.content[0].text if result.content else "{}"
    return str(result)


@function_tool
async def diagnose_issues(
    render_path: str,
    reference_path: str = "",
    effect_type: str = "auto",
    known_issues: str = ""
) -> str:
    """
    VLM-powered issue diagnosis using Moondream 2.

    Unlike hardcoded detectors, this uses vision-language models to:
    - Identify ANY visual issues (not just predefined types)
    - Compare against reference images
    - Provide actionable descriptions of problems
    - Detect novel artifacts we haven't seen before

    This tool replaces legacy artifact detectors:
    - detect_cat_ear_artifacts
    - analyze_prominence_shapes
    - compare_prominence_quality
    - analyze_texture_procedural
    - evaluate_solar_features

    Args:
        render_path: Path to render to diagnose
        reference_path: Optional reference image for comparison
        effect_type: Effect category (auto, sun, explosion, etc.)
        known_issues: Optional comma-separated hints (e.g., "too dark, wrong color")

    Returns:
        JSON with:
        - issues: List of identified problems with severity
        - primary_issue: Most critical issue to address first
        - overall_assessment: Summary of quality
        - vlm_used: Whether VLM was used

    Example:
        diagnose_issues(
            render_path="build/vdb_output/sun_v3/render_0060.png",
            reference_path="assets/reference_images/star/sun_ref.jpg",
            effect_type="sun"
        )
    """
    server = await get_mcp_server("asset-evaluator")

    result = await server.call_tool(
        "diagnose_issues_v2",
        {
            "render_path": render_path,
            "reference_path": reference_path,
            "effect_type": effect_type,
            "known_issues": known_issues,
        }
    )

    if hasattr(result, 'content') and result.content:
        return result.content[0].text if result.content else "{}"
    return str(result)


# =============================================================================
# REFERENCE DATA
# =============================================================================

@function_tool
async def get_reference_stats(
    effect_type: str,
    sample_size: int = 100
) -> str:
    """
    Get statistical distribution from reference image dataset.

    Useful for understanding what "good" looks like for an effect type.
    Statistics are computed from real footage (NASA solar, VFX reference, etc.).

    Args:
        effect_type: Effect category (sun, star, explosion, nebula)
        sample_size: Number of reference images to sample (default 100)

    Returns:
        JSON with:
        - effect_type: The effect type analyzed
        - sample_count: Number of images sampled
        - brightness_mean, brightness_std: Brightness statistics
        - warm_ratio_mean, warm_ratio_std: Color warmth statistics
        - edge_density_mean, edge_density_std: Texture statistics
        - feature_cv_mean, feature_cv_std: Feature size distribution

    Example:
        get_reference_stats("sun", sample_size=50)
    """
    server = await get_mcp_server("asset-evaluator")

    result = await server.call_tool(
        "get_reference_stats_v2",
        {
            "effect_type": effect_type,
            "sample_size": sample_size,
        }
    )

    if hasattr(result, 'content') and result.content:
        return result.content[0].text if result.content else "{}"
    return str(result)


@function_tool
async def list_renders(
    pattern: str = "",
    limit: int = 20
) -> str:
    """
    List available renders in output directories.

    Searches multiple output locations:
    - build/renders
    - build/vdb_output
    - evaluation_outputs

    Args:
        pattern: Optional filename pattern to filter (e.g., "nebula", "sun")
        limit: Maximum number of results (default 20)

    Returns:
        JSON array of render info:
        - path: Full path to render
        - filename: Just the filename
        - size_bytes: File size
        - modified_time: When it was last modified

    Example:
        list_renders(pattern="explosion", limit=10)
    """
    server = await get_mcp_server("asset-evaluator")

    result = await server.call_tool(
        "list_renders_v2",
        {
            "pattern": pattern,
            "limit": limit,
        }
    )

    if hasattr(result, 'content') and result.content:
        return result.content[0].text if result.content else "[]"
    return str(result)


# =============================================================================
# TEMPORAL ANALYSIS
# =============================================================================

@function_tool
async def analyze_temporal_quality(
    frame_directory: str,
    frame_pattern: str = "*.png",
    sample_rate: int = 5
) -> str:
    """
    Analyze temporal consistency across animation frames.

    Detects issues that single-frame evaluation misses:
    - Flickering between frames
    - Motion smoothness (jitter)
    - Temporal artifacts (popping, snapping)
    - Static frames (no motion)

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
            frame_directory="build/vdb_output/explosion_v1/renders",
            frame_pattern="frame_*.png",
            sample_rate=3
        )
    """
    server = await get_mcp_server("asset-evaluator")

    result = await server.call_tool(
        "analyze_temporal_quality",
        {
            "frame_directory": frame_directory,
            "frame_pattern": frame_pattern,
            "sample_rate": sample_rate,
        }
    )

    if hasattr(result, 'content') and result.content:
        return result.content[0].text if result.content else "{}"
    return str(result)
