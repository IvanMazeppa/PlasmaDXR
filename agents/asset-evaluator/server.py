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
# Main
# =============================================================================

if __name__ == "__main__":
    mcp.run()
