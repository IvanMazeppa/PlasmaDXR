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
- analyze_with_vision: GPT-5.2 native vision analysis (NEW)

These wrappers translate MCP tool calls into @function_tool decorated
functions that agents can call directly.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Optional

from agents import function_tool
from openai import OpenAI

from utils.mcp_connection_pool import get_mcp_server


# =============================================================================
# GPT-5.2 NATIVE VISION ANALYSIS (Primary method)
# =============================================================================

@function_tool
async def analyze_with_vision(
    render_path: str,
    analysis_type: str = "quality",
    reference_path: str = "",
    effect_type: str = "auto",
    custom_prompt: str = ""
) -> str:
    """
    Analyze a render using GPT-5.2's native vision capabilities.

    This is the PRIMARY quality analysis tool - uses GPT-5.2 vision to directly
    "see" the render and provide intelligent, context-aware quality assessment.
    More intelligent than ML metrics for nuanced visual issues.

    Analysis types:
    - quality: Overall visual quality assessment with detailed breakdown
    - issues: Focus on identifying specific problems and artifacts
    - comparison: Compare render to reference image (requires reference_path)
    - realism: Assess how realistic/believable the effect looks

    Args:
        render_path: Path to rendered image to analyze
        analysis_type: Type of analysis (quality, issues, comparison, realism)
        reference_path: Optional reference image for comparison
        effect_type: Effect category hint (auto, explosion, fire, smoke, nebula, sun)
        custom_prompt: Optional custom analysis prompt

    Returns:
        JSON with:
        - overall_assessment: Text summary of quality
        - score: 0-100 quality score
        - issues: List of identified problems
        - strengths: What looks good
        - suggestions: Specific improvement recommendations
        - comparison_notes: If reference provided, how render compares

    Example:
        analyze_with_vision(
            render_path="build/vdb_output/explosion_v1/render_0012.png",
            analysis_type="quality",
            effect_type="explosion"
        )
    """
    import sys

    # Load render image as base64
    render_file = Path(render_path)
    if not render_file.exists():
        # Try relative to project root
        project_root = Path(os.getenv("PROJECT_ROOT", "/home/maz3ppa/projects/PlasmaDXR"))
        render_file = project_root / render_path

    if not render_file.exists():
        return json.dumps({
            "error": f"Render not found: {render_path}",
            "score": 0,
            "issues": ["Render file does not exist"]
        })

    # Read and encode image
    with open(render_file, "rb") as f:
        render_base64 = base64.b64encode(f.read()).decode("utf-8")

    # Determine image type
    suffix = render_file.suffix.lower()
    media_type = "image/png" if suffix == ".png" else "image/jpeg"

    # Build analysis prompt based on type
    if custom_prompt:
        prompt = custom_prompt
    elif analysis_type == "quality":
        prompt = f"""Analyze this VFX render for overall quality. Effect type: {effect_type if effect_type != 'auto' else 'volumetric effect'}.

Evaluate these aspects:
1. Visual Impact: Does it look impressive and believable?
2. Color & Lighting: Are colors natural? Is lighting convincing?
3. Detail & Structure: Is there good detail? Any flat/blobby areas?
4. Composition: Does the effect fill the frame appropriately?
5. Artifacts: Any visible problems (clipping, banding, noise)?

Provide:
- overall_assessment: 2-3 sentence summary
- score: 0-100 (60+ is passing, 80+ is excellent)
- strengths: List what looks good
- issues: List problems with severity (critical/high/medium/low)
- suggestions: Specific parameter changes to improve quality

Be BRUTALLY HONEST. If it looks bad, say so clearly."""

    elif analysis_type == "issues":
        prompt = f"""Identify ALL visual problems in this VFX render. Effect type: {effect_type if effect_type != 'auto' else 'volumetric effect'}.

Look for:
- Color issues (unrealistic colors, wrong tint, oversaturation)
- Structural issues (clipping, flat areas, missing detail)
- Lighting issues (too dark, too bright, unnatural)
- Artifacts (banding, noise, aliasing, temporal issues)
- Composition issues (poor framing, cutoff effects)
- Material issues (wrong opacity, missing emission, bad scattering)

For EACH issue found:
- category: What type of issue
- severity: critical/high/medium/low
- location: Where in the image
- description: What's wrong
- suggested_fix: How to fix it

Be EXHAUSTIVE - miss nothing. Better to flag questionable areas than miss problems."""

    elif analysis_type == "realism":
        prompt = f"""Assess how realistic this {effect_type if effect_type != 'auto' else 'volumetric effect'} render looks.

Compare to real-world expectations:
- Would this pass as real footage or clearly CGI?
- What gives away that it's synthetic?
- What aspects are convincingly realistic?

Score from 0-100 where:
- 0-30: Obviously fake
- 30-50: Clearly CGI but decent
- 50-70: Good quality, some tells
- 70-85: Very convincing
- 85-100: Photorealistic

Be honest about realism level."""

    else:  # comparison
        prompt = f"""Compare this render to the reference image for a {effect_type if effect_type != 'auto' else 'volumetric effect'}.

Analyze:
1. How well does the render match the reference style?
2. What key differences exist?
3. Is the render better or worse than reference?
4. What should change to get closer to reference?

Provide specific, actionable comparison notes."""

    # Build input for Responses API (not Chat Completions)
    # Uses input_text and input_image types per OpenAI Responses API spec
    content = [
        {"type": "input_text", "text": f"[RENDER TO ANALYZE]\n{prompt}"},
        {"type": "input_image", "image_url": f"data:{media_type};base64,{render_base64}"},
    ]

    # Add reference image if provided
    if reference_path:
        ref_file = Path(reference_path)
        if not ref_file.exists():
            project_root = Path(os.getenv("PROJECT_ROOT", "/home/maz3ppa/projects/PlasmaDXR"))
            ref_file = project_root / reference_path

        if ref_file.exists():
            with open(ref_file, "rb") as f:
                ref_base64 = base64.b64encode(f.read()).decode("utf-8")
            ref_suffix = ref_file.suffix.lower()
            ref_media = "image/png" if ref_suffix == ".png" else "image/jpeg"

            content.append({"type": "input_text", "text": "\n[REFERENCE IMAGE FOR COMPARISON]"})
            content.append({"type": "input_image", "image_url": f"data:{ref_media};base64,{ref_base64}"})

    # Add format instruction
    content.append({
        "type": "input_text",
        "text": "\n\nRespond with valid JSON only. Include: overall_assessment (string), score (int 0-100), issues (array of objects with category/severity/description), strengths (array of strings), suggestions (array of strings)."
    })

    # Call OpenAI Responses API with GPT-5.2 (improved multimodality)
    # Using Responses API instead of Chat Completions for consistency with Agents SDK
    try:
        client = OpenAI()  # Uses OPENAI_API_KEY from env

        response = client.responses.create(
            model="gpt-5.2",  # GPT-5.2 has improved multimodality/vision
            input=[{"role": "user", "content": content}],
        )

        result_text = response.output_text

        # Try to parse as JSON, wrap if needed
        try:
            result = json.loads(result_text)
        except json.JSONDecodeError:
            # Wrap plain text response
            result = {
                "overall_assessment": result_text,
                "score": 50,
                "issues": [],
                "strengths": [],
                "suggestions": [],
                "raw_response": True
            }

        result["vision_model"] = "gpt-5.2"
        result["analysis_type"] = analysis_type
        return json.dumps(result, indent=2)

    except Exception as e:
        return json.dumps({
            "error": str(e),
            "score": 0,
            "issues": [{"category": "error", "severity": "critical", "description": str(e)}]
        })


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
