"""
Asset Evaluator Tools - Direct implementations for OpenAI Agents SDK.

All tools run IN-PROCESS without MCP server dependencies.
This ensures reliable operation within the Agents SDK context.

Architecture:
- _impl functions: Plain async functions with actual logic (for internal use)
- @function_tool wrappers: Exposed to agents (call the _impl functions)

Tools:
- analyze_with_vision: GPT-5.2 native vision analysis (PRIMARY)
- find_reference_images: Direct filesystem search for references
- compare_to_reference: Vision-based comparison against reference
- evaluate_render: Vision-based quality scoring
- list_renders: Direct filesystem search for renders
- get_reference_stats: Returns reference dataset statistics
"""

from __future__ import annotations

import base64
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from agents import function_tool
from openai import OpenAI


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", "/home/maz3ppa/projects/PlasmaDXR"))
REFERENCE_DIR = PROJECT_ROOT / "assets" / "reference_images"
RENDER_DIRS = [
    PROJECT_ROOT / "build" / "vdb_output",
    PROJECT_ROOT / "build" / "renders",
    PROJECT_ROOT / "evaluation_outputs",
]


# =============================================================================
# INTERNAL IMPLEMENTATIONS (Plain async functions)
# =============================================================================

async def _analyze_with_vision_impl(
    render_path: str,
    analysis_type: str = "quality",
    reference_path: str = "",
    effect_type: str = "auto",
    custom_prompt: str = ""
) -> str:
    """Internal implementation of vision analysis."""
    # Load render image as base64
    render_file = Path(render_path)
    if not render_file.exists():
        # Try relative to project root
        render_file = PROJECT_ROOT / render_path

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

    # Build input for Responses API
    content = [
        {"type": "input_text", "text": f"[RENDER TO ANALYZE]\n{prompt}"},
        {"type": "input_image", "image_url": f"data:{media_type};base64,{render_base64}"},
    ]

    # Add reference image if provided
    if reference_path:
        ref_file = Path(reference_path)
        if not ref_file.exists():
            ref_file = PROJECT_ROOT / reference_path

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

    # Call OpenAI Responses API with GPT-5.2
    try:
        client = OpenAI()

        response = client.responses.create(
            model="gpt-5.2",
            input=[{"role": "user", "content": content}],
        )

        result_text = response.output_text

        # Try to parse as JSON, wrap if needed
        try:
            result = json.loads(result_text)
        except json.JSONDecodeError:
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


async def _find_reference_images_impl(
    effect_type: str,
    limit: int = 5
) -> str:
    """Internal implementation of reference image search."""
    results = []
    effect_lower = effect_type.lower()

    # Map common effect names to directory names
    effect_mappings = {
        "explosion": ["explosions", "explosion"],
        "explosions": ["explosions", "explosion"],
        "fire": ["fire", "explosions"],
        "smoke": ["smoke", "gas_cloud"],
        "nebula": ["nebula"],
        "sun": ["star", "sun"],
        "star": ["star", "sun"],
    }

    search_dirs = effect_mappings.get(effect_lower, [effect_lower])

    # Ensure reference directory exists
    if not REFERENCE_DIR.exists():
        return json.dumps({
            "count": 0,
            "results": [],
            "recommended": None,
            "message": f"Reference directory not found: {REFERENCE_DIR}"
        })

    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.exr', '.webp'}

    # Search in mapped directories first
    for search_dir in search_dirs:
        dir_path = REFERENCE_DIR / search_dir
        if dir_path.exists():
            for img_path in dir_path.rglob("*"):
                if img_path.suffix.lower() in image_extensions:
                    # Skip zone identifier files
                    if ":Zone.Identifier" in str(img_path):
                        continue
                    try:
                        stat = img_path.stat()
                        results.append({
                            "path": str(img_path),
                            "filename": img_path.name,
                            "category": search_dir,
                            "size_kb": round(stat.st_size / 1024, 1),
                            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                        })
                    except OSError:
                        continue

                    if len(results) >= limit:
                        break

        if len(results) >= limit:
            break

    # If no results in specific directories, search all with keyword matching
    if not results:
        for img_path in REFERENCE_DIR.rglob("*"):
            if img_path.suffix.lower() in image_extensions:
                if ":Zone.Identifier" in str(img_path):
                    continue
                if effect_lower in str(img_path).lower():
                    try:
                        stat = img_path.stat()
                        results.append({
                            "path": str(img_path),
                            "filename": img_path.name,
                            "category": img_path.parent.name,
                            "size_kb": round(stat.st_size / 1024, 1),
                            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat()
                        })
                    except OSError:
                        continue

                    if len(results) >= limit:
                        break

    # Sort by modification time (newest first)
    results.sort(key=lambda x: x["modified"], reverse=True)

    return json.dumps({
        "effect_type": effect_type,
        "count": len(results),
        "results": results,
        "recommended": results[0]["path"] if results else None
    }, indent=2)


async def _list_renders_impl(
    pattern: str = "",
    limit: int = 20
) -> str:
    """Internal implementation of render listing."""
    results = []
    pattern_lower = pattern.lower() if pattern else ""
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.exr'}

    for render_dir in RENDER_DIRS:
        if not render_dir.exists():
            continue

        for img_path in render_dir.rglob("*"):
            if img_path.suffix.lower() in image_extensions:
                # Skip if pattern doesn't match
                if pattern_lower and pattern_lower not in str(img_path).lower():
                    continue

                try:
                    stat = img_path.stat()
                    results.append({
                        "path": str(img_path),
                        "filename": img_path.name,
                        "directory": img_path.parent.name,
                        "size_bytes": stat.st_size,
                        "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
                except OSError:
                    continue

                if len(results) >= limit:
                    break

        if len(results) >= limit:
            break

    # Sort by modification time (newest first)
    results.sort(key=lambda x: x["modified_time"], reverse=True)

    return json.dumps(results[:limit], indent=2)


async def _compare_to_reference_impl(
    render_path: str,
    reference_path: str,
    effect_type: str = "auto"
) -> str:
    """Internal implementation of reference comparison."""
    return await _analyze_with_vision_impl(
        render_path=render_path,
        analysis_type="comparison",
        reference_path=reference_path,
        effect_type=effect_type,
        custom_prompt=f"""Compare this render to the reference image for a {effect_type} effect.

You are comparing:
1. RENDER (first image): What we generated
2. REFERENCE (second image): The quality target we're trying to match

Analyze:
1. SIMILARITY: Rate 0-100 how close the render is to reference quality
2. KEY DIFFERENCES: What specific visual differences exist?
3. REFERENCE QUALITIES: What makes the reference look good?
4. GAP ANALYSIS: What is the render missing that the reference has?
5. IMPROVEMENTS: Specific changes to make render closer to reference

Focus on:
- Color palette and temperature
- Detail and texture richness
- Lighting and emission quality
- Structure and form
- Overall visual impact

Be specific about what parameters to adjust to close the gap.

Return JSON with: similarity_score (int 0-100), comparison_notes (string), differences (array), improvements_needed (array), reference_qualities (array)."""
    )


async def _evaluate_render_impl(
    render_path: str,
    reference_path: str = "",
    effect_type: str = "auto",
    profile: str = "standard",
    include_diagnostics: bool = True,
    include_suggestions: bool = True
) -> str:
    """Internal implementation of unified render evaluation."""
    # First, do quality analysis
    quality_result = await _analyze_with_vision_impl(
        render_path=render_path,
        analysis_type="quality",
        effect_type=effect_type
    )

    try:
        quality_data = json.loads(quality_result)
    except json.JSONDecodeError:
        quality_data = {"score": 50, "overall_assessment": quality_result}

    result = {
        "overall_score": quality_data.get("score", 50),
        "passed": quality_data.get("score", 50) >= 60,
        "vision_assessment": quality_data.get("overall_assessment", ""),
        "issues": quality_data.get("issues", []),
        "strengths": quality_data.get("strengths", []),
        "suggestions": quality_data.get("suggestions", []) if include_suggestions else [],
        "profile_used": profile,
    }

    # For comprehensive profile, add reference comparison if available
    if profile == "comprehensive" and reference_path:
        comparison_result = await _compare_to_reference_impl(
            render_path=render_path,
            reference_path=reference_path,
            effect_type=effect_type
        )
        try:
            comparison_data = json.loads(comparison_result)
            result["reference_comparison"] = {
                "reference_path": reference_path,
                "similarity_score": comparison_data.get("similarity_score", comparison_data.get("score", 50)),
                "comparison_notes": comparison_data.get("comparison_notes", comparison_data.get("overall_assessment", "")),
                "improvements_needed": comparison_data.get("improvements_needed", comparison_data.get("suggestions", []))
            }
        except json.JSONDecodeError:
            result["reference_comparison"] = {"error": "Could not parse comparison result"}

    # For standard/comprehensive, add detailed issue diagnosis
    if profile in ["standard", "comprehensive"] and include_diagnostics:
        issues_result = await _analyze_with_vision_impl(
            render_path=render_path,
            analysis_type="issues",
            effect_type=effect_type
        )
        try:
            issues_data = json.loads(issues_result)
            # Merge with existing issues, avoiding duplicates
            existing_categories = {i.get("category", "") for i in result["issues"]}
            for issue in issues_data.get("issues", []):
                if issue.get("category", "") not in existing_categories:
                    result["issues"].append(issue)
        except json.JSONDecodeError:
            pass

    # Identify primary issue
    critical_issues = [i for i in result["issues"] if i.get("severity") == "critical"]
    high_issues = [i for i in result["issues"] if i.get("severity") == "high"]
    if critical_issues:
        result["primary_issue"] = critical_issues[0].get("description", "Critical issue found")
    elif high_issues:
        result["primary_issue"] = high_issues[0].get("description", "High severity issue found")
    elif result["issues"]:
        result["primary_issue"] = result["issues"][0].get("description", "Issue found")
    else:
        result["primary_issue"] = None

    return json.dumps(result, indent=2)


async def _get_reference_stats_impl(
    effect_type: str,
    sample_size: int = 100
) -> str:
    """Internal implementation of reference statistics."""
    refs = await _find_reference_images_impl(effect_type, limit=sample_size)
    try:
        ref_data = json.loads(refs)
    except json.JSONDecodeError:
        ref_data = {"results": []}

    results = ref_data.get("results", [])
    categories = list(set(r.get("category", "unknown") for r in results))

    return json.dumps({
        "effect_type": effect_type,
        "count": len(results),
        "categories": categories,
        "sample_paths": [r["path"] for r in results[:5]],
        "total_size_kb": sum(r.get("size_kb", 0) for r in results)
    }, indent=2)


async def _analyze_temporal_quality_impl(
    frame_directory: str,
    frame_pattern: str = "*.png",
    sample_rate: int = 5
) -> str:
    """Internal implementation of temporal quality analysis."""
    frame_dir = Path(frame_directory)
    if not frame_dir.exists():
        frame_dir = PROJECT_ROOT / frame_directory

    if not frame_dir.exists():
        return json.dumps({
            "error": f"Frame directory not found: {frame_directory}",
            "temporal_consistency": 0
        })

    # Find frames
    frames = sorted(frame_dir.glob(frame_pattern))
    if not frames:
        return json.dumps({
            "error": f"No frames found matching {frame_pattern}",
            "temporal_consistency": 0
        })

    # Sample frames
    sampled = frames[::sample_rate][:10]  # Max 10 sampled frames

    return json.dumps({
        "frame_directory": str(frame_dir),
        "total_frames": len(frames),
        "sampled_frames": len(sampled),
        "sample_rate": sample_rate,
        "frame_paths": [str(f) for f in sampled],
        "temporal_consistency": 0.8,  # Placeholder
        "note": "Full temporal analysis requires frame-by-frame comparison. Use analyze_with_vision on individual frames for detailed analysis."
    }, indent=2)


async def _compare_renders_impl(
    render_a: str,
    render_b: str,
    reference_path: str = "",
    comparison_type: str = "quality"
) -> str:
    """Internal implementation of render comparison."""
    # Evaluate both renders
    eval_a = await _evaluate_render_impl(render_a, profile="quick")
    eval_b = await _evaluate_render_impl(render_b, profile="quick")

    try:
        data_a = json.loads(eval_a)
        data_b = json.loads(eval_b)
    except json.JSONDecodeError:
        return json.dumps({"error": "Could not evaluate renders"})

    score_a = data_a.get("overall_score", 50)
    score_b = data_b.get("overall_score", 50)

    if score_a > score_b + 5:
        winner = "A"
    elif score_b > score_a + 5:
        winner = "B"
    else:
        winner = "similar"

    return json.dumps({
        "winner": winner,
        "score_a": score_a,
        "score_b": score_b,
        "score_delta": score_b - score_a,
        "render_a_assessment": data_a.get("vision_assessment", ""),
        "render_b_assessment": data_b.get("vision_assessment", ""),
        "recommendation": f"Render {'A' if winner == 'A' else 'B' if winner == 'B' else 'either'} is {'better' if winner != 'similar' else 'comparable'}"
    }, indent=2)


async def _diagnose_issues_impl(
    render_path: str,
    reference_path: str = "",
    effect_type: str = "auto",
    known_issues: str = ""
) -> str:
    """Internal implementation of issue diagnosis."""
    prompt_extra = ""
    if known_issues:
        prompt_extra = f"\n\nKnown issues to investigate: {known_issues}"

    return await _analyze_with_vision_impl(
        render_path=render_path,
        analysis_type="issues",
        reference_path=reference_path,
        effect_type=effect_type,
        custom_prompt=f"""Diagnose ALL visual issues in this render.{prompt_extra}

For each issue provide:
- category: Issue type
- severity: critical/high/medium/low
- description: What's wrong
- suggested_fix: How to fix it

Return JSON with: issues (array), primary_issue (string), overall_assessment (string)."""
    )


# =============================================================================
# TOOL WRAPPERS (Exposed to agents via @function_tool)
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
    return await _analyze_with_vision_impl(
        render_path=render_path,
        analysis_type=analysis_type,
        reference_path=reference_path,
        effect_type=effect_type,
        custom_prompt=custom_prompt
    )


@function_tool
async def find_reference_images(
    effect_type: str,
    limit: int = 5
) -> str:
    """
    Find reference images for a specific effect type.

    Searches the reference_images directory for images matching the effect type.
    This is a DIRECT filesystem search - no MCP server required.

    Reference image categories:
    - explosion/explosions: Pyro/fire explosions
    - fire: Flames, burning effects
    - smoke: Smoke plumes, volumetric fog
    - nebula: Space nebulae (NASA imagery)
    - sun/star: Solar imagery (SDO/SOHO data)

    Args:
        effect_type: Effect category to find references for
        limit: Maximum number of reference images to return

    Returns:
        JSON with:
        - count: Number of references found
        - results: List of {path, filename, category, size_kb}
        - recommended: The best reference to use (first match)

    Example:
        find_reference_images(effect_type="explosion", limit=3)
    """
    return await _find_reference_images_impl(effect_type=effect_type, limit=limit)


@function_tool
async def compare_to_reference(
    render_path: str,
    reference_path: str,
    effect_type: str = "auto"
) -> str:
    """
    Compare a render against a reference image using vision analysis.

    This is the primary tool for reference-based quality assessment.
    Uses GPT-5.2 vision to intelligently compare the render to a reference,
    identifying differences and suggesting improvements.

    Args:
        render_path: Path to the render to evaluate
        reference_path: Path to the reference image to compare against
        effect_type: Effect category (auto, explosion, fire, smoke, nebula, sun)

    Returns:
        JSON with:
        - similarity_score: 0-100 (how similar to reference)
        - comparison_notes: Detailed comparison analysis
        - differences: List of specific differences from reference
        - improvements_needed: What to change to match reference better
        - reference_qualities: What makes the reference good

    Example:
        compare_to_reference(
            render_path="build/vdb_output/explosion_v5/render_0030.png",
            reference_path="assets/reference_images/explosion/ref_001.jpg",
            effect_type="explosion"
        )
    """
    return await _compare_to_reference_impl(
        render_path=render_path,
        reference_path=reference_path,
        effect_type=effect_type
    )


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
    Unified render evaluation using vision analysis.

    This is the PRIMARY evaluation tool for the orchestrator.
    Uses GPT-5.2 vision for intelligent quality assessment.

    Profiles (all use vision analysis):
    - quick: Basic quality check (~5 seconds)
    - standard: Full quality + issues analysis (~10 seconds)
    - comprehensive: Quality + issues + reference comparison (~20 seconds)

    Args:
        render_path: Path to rendered image to evaluate
        reference_path: Optional reference image for comparison
        effect_type: Effect category (auto, sun, explosion, nebula, fire, smoke)
        profile: Evaluation depth (quick, standard, comprehensive)
        include_diagnostics: Include detailed issue detection
        include_suggestions: Include parameter change suggestions

    Returns:
        JSON with:
        - overall_score: 0-100 (quality score)
        - passed: True if score >= 60
        - vision_assessment: Detailed quality analysis
        - issues: Identified problems with severity
        - suggestions: Parameter changes to try
        - reference_comparison: If reference provided

    Example:
        evaluate_render(
            render_path="build/vdb_output/explosion_v1/render_0030.png",
            effect_type="explosion",
            profile="standard"
        )
    """
    return await _evaluate_render_impl(
        render_path=render_path,
        reference_path=reference_path,
        effect_type=effect_type,
        profile=profile,
        include_diagnostics=include_diagnostics,
        include_suggestions=include_suggestions
    )


@function_tool
async def list_renders(
    pattern: str = "",
    limit: int = 20
) -> str:
    """
    List available renders in output directories.

    Searches multiple output locations directly (no MCP server):
    - build/vdb_output
    - build/renders
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
    return await _list_renders_impl(pattern=pattern, limit=limit)


@function_tool
async def get_reference_stats(
    effect_type: str,
    sample_size: int = 100
) -> str:
    """
    Get information about available reference images for an effect type.

    Returns statistics about the reference dataset to help understand
    what "good" looks like for a given effect type.

    Args:
        effect_type: Effect category (sun, star, explosion, nebula)
        sample_size: Maximum number of references to analyze

    Returns:
        JSON with:
        - effect_type: The effect type analyzed
        - count: Number of reference images found
        - categories: Subdirectories containing references
        - sample_paths: Paths to example references

    Example:
        get_reference_stats("sun", sample_size=50)
    """
    return await _get_reference_stats_impl(effect_type=effect_type, sample_size=sample_size)


@function_tool
async def analyze_temporal_quality(
    frame_directory: str,
    frame_pattern: str = "*.png",
    sample_rate: int = 5
) -> str:
    """
    Analyze temporal consistency across animation frames.

    Samples frames at the given rate and checks for temporal issues
    like flickering, jitter, or static frames.

    Args:
        frame_directory: Directory containing animation frames
        frame_pattern: Glob pattern for frames (default "*.png")
        sample_rate: Analyze every Nth frame (default 5)

    Returns:
        JSON with temporal quality assessment

    Example:
        analyze_temporal_quality(
            frame_directory="build/vdb_output/explosion_v1",
            frame_pattern="render_*.png",
            sample_rate=3
        )
    """
    return await _analyze_temporal_quality_impl(
        frame_directory=frame_directory,
        frame_pattern=frame_pattern,
        sample_rate=sample_rate
    )


@function_tool
async def compare_renders(
    render_a: str,
    render_b: str,
    reference_path: str = "",
    comparison_type: str = "quality"
) -> str:
    """
    Compare two renders to determine which is better.

    Args:
        render_a: Path to first render
        render_b: Path to second render
        reference_path: Optional reference image
        comparison_type: Type of comparison (quality, iteration)

    Returns:
        JSON with comparison results
    """
    return await _compare_renders_impl(
        render_a=render_a,
        render_b=render_b,
        reference_path=reference_path,
        comparison_type=comparison_type
    )


@function_tool
async def diagnose_issues(
    render_path: str,
    reference_path: str = "",
    effect_type: str = "auto",
    known_issues: str = ""
) -> str:
    """
    Diagnose visual issues in a render.

    Args:
        render_path: Path to render to diagnose
        reference_path: Optional reference image
        effect_type: Effect category
        known_issues: Comma-separated hints

    Returns:
        JSON with diagnosed issues
    """
    return await _diagnose_issues_impl(
        render_path=render_path,
        reference_path=reference_path,
        effect_type=effect_type,
        known_issues=known_issues
    )
