#!/usr/bin/env python3
"""
DINOv2 Structural Similarity Evaluation

Addresses the critical flaw where aggregate statistics miss morphological issues.
Uses DINOv2's self-supervised patch features to detect:
- Texture pattern differences (procedural vs organic)
- Shape differences (ribbon vs loop prominences)
- Structural coherence issues
- Localized problem regions

Key insight: DINOv2 learns semantic structure without labels.
It can tell "ribbon-shaped" from "loop-shaped" because it understands structure,
not just pixel statistics.

Research basis:
- DINOv2 achieves 64% accuracy vs CLIP's 28% on structural similarity
- Patch-level features enable spatial localization of differences
- See: docs/EVALUATION_SYSTEM_IMPROVEMENT_PROPOSAL.md

Usage:
    evaluator = DinoStructuralEvaluator()
    result = evaluator.evaluate(render_path, reference_path)
    print(result["problem_regions"])  # Shows WHERE differences are
"""

import json
import os
import sys
import numpy as np
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import warnings

# =============================================================================
# MCP Protocol Safety - Suppress all stdout pollution
# =============================================================================
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TQDM_DISABLE"] = "1"  # Global tqdm disable

import contextlib

@contextlib.contextmanager
def suppress_stdout_to_stderr():
    """
    Redirect stdout to stderr at the OS file descriptor level.
    Uses os.dup2() to catch ALL stdout output including C-level writes.
    """
    original_stdout_fd = sys.stdout.fileno()
    saved_stdout_fd = os.dup(original_stdout_fd)

    try:
        os.dup2(sys.stderr.fileno(), original_stdout_fd)
        sys.stdout = sys.stderr
        yield
    finally:
        os.dup2(saved_stdout_fd, original_stdout_fd)
        os.close(saved_stdout_fd)
        sys.stdout = sys.__stdout__

# Lazy-loaded models
_dino_model = None
_dino_processor = None


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class PatchSimilarity:
    """Similarity score for a single image patch."""
    row: int
    col: int
    similarity: float
    region_name: str  # e.g., "top_left", "center", "prominence_area"


@dataclass
class StructuralEvaluationResult:
    """Complete structural evaluation result."""
    # Global scores
    global_structural_similarity: float  # 0-1, higher = more similar
    mean_patch_similarity: float
    min_patch_similarity: float
    std_patch_similarity: float

    # Problem regions (sorted by severity)
    problem_regions: List[Dict[str, Any]]

    # Interpretation
    structural_quality: str  # "excellent", "good", "fair", "poor", "very_poor"
    interpretation: str

    # Recommendations
    recommendations: List[str]

    # Raw data for visualization
    similarity_heatmap: Optional[List[List[float]]] = None

    # Paths
    render_path: str = ""
    reference_path: str = ""


# =============================================================================
# Model Loading (Lazy)
# =============================================================================

def get_dino_model():
    """
    Lazy load DINOv2 model and processor.

    Uses facebook/dinov2-base (86M params) for good balance of
    quality and speed. For higher quality, use dinov2-large (300M params).
    """
    global _dino_model, _dino_processor

    if _dino_model is None:
        try:
            import torch
            from transformers import AutoModel, AutoImageProcessor

            print("Loading DINOv2 model (first time may take a minute)...", file=sys.stderr)

            # Use base model for speed, large for quality
            model_name = "facebook/dinov2-base"

            # Wrap from_pretrained to suppress any progress bars
            with suppress_stdout_to_stderr():
                _dino_processor = AutoImageProcessor.from_pretrained(model_name)
                _dino_model = AutoModel.from_pretrained(model_name)

                # Move to GPU if available
                if torch.cuda.is_available():
                    _dino_model = _dino_model.cuda()
                else:
                    pass  # CPU mode

            if torch.cuda.is_available():
                print("DINOv2 loaded on GPU", file=sys.stderr)
            else:
                print("DINOv2 loaded on CPU (slower)", file=sys.stderr)

            _dino_model.eval()

        except ImportError as e:
            raise RuntimeError(
                f"DINOv2 dependencies not installed: {e}\n"
                "Run: pip install transformers torch torchvision"
            )

    return _dino_model, _dino_processor


# =============================================================================
# Feature Extraction
# =============================================================================

def extract_patch_features(image_path: str) -> np.ndarray:
    """
    Extract DINOv2 patch features from an image.

    DINOv2 divides the image into a 14x14 grid of patches (for 224x224 input)
    and produces a 768-dimensional feature vector for each patch.

    Args:
        image_path: Path to image file

    Returns:
        numpy array of shape (14, 14, 768) containing patch features
    """
    import torch
    from PIL import Image

    model, processor = get_dino_model()

    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt")

    # Move to same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Extract features
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # Get patch tokens (excluding CLS token at index 0)
    # Shape: [1, 197, 768] -> [1, 196, 768] -> [14, 14, 768]
    patch_features = outputs.last_hidden_state[:, 1:, :]  # Remove CLS

    # Reshape to spatial grid
    # DINOv2-base uses 14x14 patches for 224x224 input
    num_patches = patch_features.shape[1]
    grid_size = int(np.sqrt(num_patches))

    patch_features = patch_features.reshape(grid_size, grid_size, -1)

    return patch_features.cpu().numpy()


def extract_cls_feature(image_path: str) -> np.ndarray:
    """
    Extract DINOv2 CLS token (global image representation).

    The CLS token captures the overall semantic content of the image.
    """
    import torch
    from PIL import Image

    model, processor = get_dino_model()

    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors="pt")

    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # CLS token is at index 0
    cls_feature = outputs.last_hidden_state[:, 0, :]

    return cls_feature.cpu().numpy().squeeze()


# =============================================================================
# Similarity Computation
# =============================================================================

def compute_patch_similarity_map(
    render_features: np.ndarray,
    reference_features: np.ndarray
) -> np.ndarray:
    """
    Compute cosine similarity between corresponding patches.

    Args:
        render_features: (H, W, D) patch features from render
        reference_features: (H, W, D) patch features from reference

    Returns:
        (H, W) similarity map where each value is cosine similarity (0-1)
    """
    # Normalize features
    render_norm = render_features / (np.linalg.norm(render_features, axis=-1, keepdims=True) + 1e-8)
    ref_norm = reference_features / (np.linalg.norm(reference_features, axis=-1, keepdims=True) + 1e-8)

    # Cosine similarity at each patch
    similarity = np.sum(render_norm * ref_norm, axis=-1)

    # Clip to [0, 1] (cosine similarity can be negative)
    similarity = np.clip((similarity + 1) / 2, 0, 1)

    return similarity


def compute_global_similarity(
    render_features: np.ndarray,
    reference_features: np.ndarray
) -> float:
    """
    Compute global structural similarity using CLS tokens.
    """
    # Normalize
    render_norm = render_features / (np.linalg.norm(render_features) + 1e-8)
    ref_norm = reference_features / (np.linalg.norm(reference_features) + 1e-8)

    # Cosine similarity
    similarity = np.dot(render_norm, ref_norm)

    # Scale to [0, 1]
    return float((similarity + 1) / 2)


# =============================================================================
# Region Analysis
# =============================================================================

def get_region_name(row: int, col: int, grid_size: int = 14) -> str:
    """
    Map patch coordinates to human-readable region names.

    Divides image into 9 regions (3x3 grid) plus special solar regions.
    """
    # Normalize to 0-1
    r = row / grid_size
    c = col / grid_size

    # Distance from center (for solar disk detection)
    center_dist = np.sqrt((r - 0.5)**2 + (c - 0.5)**2)

    # Solar-specific regions
    if center_dist < 0.25:
        return "disk_center"
    elif center_dist < 0.4:
        if r < 0.5:
            return "disk_upper" if c > 0.3 and c < 0.7 else ("disk_upper_left" if c < 0.5 else "disk_upper_right")
        else:
            return "disk_lower" if c > 0.3 and c < 0.7 else ("disk_lower_left" if c < 0.5 else "disk_lower_right")
    elif center_dist < 0.55:
        # Limb region (where prominences attach)
        if c < 0.3:
            return "limb_left"
        elif c > 0.7:
            return "limb_right"
        elif r < 0.3:
            return "limb_top"
        else:
            return "limb_bottom"
    else:
        # Corona/prominence region
        if c < 0.3:
            return "prominence_left"
        elif c > 0.7:
            return "prominence_right"
        elif r < 0.3:
            return "corona_top"
        else:
            return "corona_bottom"


def identify_problem_regions(
    similarity_map: np.ndarray,
    threshold: float = 0.7,
    min_severity: float = 0.3
) -> List[Dict[str, Any]]:
    """
    Identify regions with low structural similarity.

    Args:
        similarity_map: (H, W) patch similarity values
        threshold: Patches below this are considered problems
        min_severity: Minimum severity to report (1 - similarity)

    Returns:
        List of problem regions sorted by severity
    """
    problems = []
    grid_size = similarity_map.shape[0]

    for row in range(grid_size):
        for col in range(grid_size):
            sim = similarity_map[row, col]
            severity = 1 - sim

            if sim < threshold and severity >= min_severity:
                region = get_region_name(row, col, grid_size)
                problems.append({
                    "row": row,
                    "col": col,
                    "region": region,
                    "similarity": float(sim),
                    "severity": float(severity),
                    "severity_level": "critical" if severity > 0.5 else "high" if severity > 0.3 else "moderate"
                })

    # Sort by severity (worst first)
    problems.sort(key=lambda x: x["severity"], reverse=True)

    # Aggregate by region
    region_problems = {}
    for p in problems:
        region = p["region"]
        if region not in region_problems:
            region_problems[region] = {
                "region": region,
                "patch_count": 0,
                "total_severity": 0,
                "min_similarity": 1.0,
                "patches": []
            }
        region_problems[region]["patch_count"] += 1
        region_problems[region]["total_severity"] += p["severity"]
        region_problems[region]["min_similarity"] = min(
            region_problems[region]["min_similarity"],
            p["similarity"]
        )
        region_problems[region]["patches"].append(p)

    # Convert to list and compute average severity
    aggregated = []
    for region, data in region_problems.items():
        data["avg_severity"] = data["total_severity"] / data["patch_count"]
        data["severity_level"] = (
            "critical" if data["avg_severity"] > 0.5 else
            "high" if data["avg_severity"] > 0.3 else
            "moderate"
        )
        # Remove individual patches to reduce output size
        del data["patches"]
        aggregated.append(data)

    # Sort by average severity
    aggregated.sort(key=lambda x: x["avg_severity"], reverse=True)

    return aggregated


# =============================================================================
# Interpretation
# =============================================================================

def interpret_similarity(score: float) -> Tuple[str, str]:
    """
    Interpret structural similarity score.

    Args:
        score: Global similarity (0-1)

    Returns:
        (quality_level, interpretation)
    """
    if score >= 0.85:
        return "excellent", "Structures match very well - similar morphology"
    elif score >= 0.70:
        return "good", "Good structural similarity with minor differences"
    elif score >= 0.55:
        return "fair", "Noticeable structural differences"
    elif score >= 0.40:
        return "poor", "Significant structural/morphological differences"
    else:
        return "very_poor", "Structures are very different - major morphological issues"


def generate_recommendations(
    problem_regions: List[Dict],
    global_similarity: float,
    effect_type: str = "sun"
) -> List[str]:
    """
    Generate actionable recommendations based on problem regions.
    """
    recommendations = []

    # Group problems by region type
    disk_problems = [p for p in problem_regions if "disk" in p["region"]]
    limb_problems = [p for p in problem_regions if "limb" in p["region"]]
    prominence_problems = [p for p in problem_regions if "prominence" in p["region"] or "corona" in p["region"]]

    # Disk issues (texture, granulation)
    if disk_problems:
        avg_disk_severity = np.mean([p["avg_severity"] for p in disk_problems])
        if avg_disk_severity > 0.4:
            recommendations.append(
                "CRITICAL: Disk surface structure differs significantly from reference. "
                "Check texture/noise settings - may be procedural single-scale noise instead of organic multi-scale."
            )
        elif avg_disk_severity > 0.2:
            recommendations.append(
                "Disk surface texture could be improved. Consider adding multi-octave noise for granulation."
            )

    # Limb issues (limb darkening, edge definition)
    if limb_problems:
        avg_limb_severity = np.mean([p["avg_severity"] for p in limb_problems])
        if avg_limb_severity > 0.3:
            recommendations.append(
                "Limb region differs from reference. Check limb darkening settings - "
                "edges should be ~40% darker than center."
            )

    # Prominence issues (shape, volumetric appearance)
    if prominence_problems:
        avg_prom_severity = np.mean([p["avg_severity"] for p in prominence_problems])
        if avg_prom_severity > 0.4:
            recommendations.append(
                "CRITICAL: Prominence/corona structure differs significantly. "
                "Prominences may be flat ribbons instead of volumetric loops. "
                "Add curl/vorticity to flow, increase domain depth."
            )
        elif avg_prom_severity > 0.2:
            recommendations.append(
                "Prominence shapes could be improved. Consider adding more turbulence and depth variation."
            )

    # Global recommendation
    if global_similarity < 0.5:
        recommendations.insert(0,
            "Overall structural similarity is low. Multiple fundamental issues need addressing."
        )

    if not recommendations:
        recommendations.append("Structural similarity is acceptable. Focus on other quality aspects.")

    return recommendations


# =============================================================================
# Main Evaluation Class
# =============================================================================

class DinoStructuralEvaluator:
    """
    DINOv2-based structural similarity evaluator.

    Detects morphological issues that aggregate statistics miss:
    - Texture patterns (procedural vs organic)
    - Shape differences (ribbon vs loop prominences)
    - Structural coherence
    - Localized problem regions

    Example:
        evaluator = DinoStructuralEvaluator()
        result = evaluator.evaluate(
            "build/vdb_output/sun_prominences_v2/render_0060.png",
            "assets/reference_images/star/.../frame_00639.jpg"
        )
        print(f"Structural similarity: {result.global_structural_similarity}")
        for region in result.problem_regions:
            print(f"  Problem: {region['region']} (severity: {region['avg_severity']:.2f})")
    """

    def __init__(self, effect_type: str = "sun"):
        """
        Initialize evaluator.

        Args:
            effect_type: Type of effect being evaluated (affects region naming)
        """
        self.effect_type = effect_type
        self._model_loaded = False

    def _ensure_model_loaded(self):
        """Load model on first use."""
        if not self._model_loaded:
            get_dino_model()
            self._model_loaded = True

    def evaluate(
        self,
        render_path: str,
        reference_path: str,
        include_heatmap: bool = True
    ) -> StructuralEvaluationResult:
        """
        Evaluate structural similarity between render and reference.

        Args:
            render_path: Path to rendered image
            reference_path: Path to reference image
            include_heatmap: Include similarity heatmap in result

        Returns:
            StructuralEvaluationResult with detailed analysis
        """
        self._ensure_model_loaded()

        # Extract features
        render_patches = extract_patch_features(render_path)
        ref_patches = extract_patch_features(reference_path)

        render_cls = extract_cls_feature(render_path)
        ref_cls = extract_cls_feature(reference_path)

        # Compute similarities
        similarity_map = compute_patch_similarity_map(render_patches, ref_patches)
        global_sim = compute_global_similarity(render_cls, ref_cls)

        # Analyze
        problem_regions = identify_problem_regions(similarity_map)
        quality, interpretation = interpret_similarity(global_sim)
        recommendations = generate_recommendations(
            problem_regions, global_sim, self.effect_type
        )

        # Build result
        result = StructuralEvaluationResult(
            global_structural_similarity=round(global_sim, 4),
            mean_patch_similarity=round(float(np.mean(similarity_map)), 4),
            min_patch_similarity=round(float(np.min(similarity_map)), 4),
            std_patch_similarity=round(float(np.std(similarity_map)), 4),
            problem_regions=problem_regions,
            structural_quality=quality,
            interpretation=interpretation,
            recommendations=recommendations,
            similarity_heatmap=similarity_map.tolist() if include_heatmap else None,
            render_path=render_path,
            reference_path=reference_path
        )

        return result

    def evaluate_against_dataset(
        self,
        render_path: str,
        reference_dir: str,
        sample_size: int = 10
    ) -> Dict[str, Any]:
        """
        Evaluate render against multiple reference images.

        Useful when you have a dataset of reference images (like the 840 solar frames).
        Computes similarity to a sample and finds closest matches.

        Args:
            render_path: Path to rendered image
            reference_dir: Directory containing reference images
            sample_size: Number of reference images to sample

        Returns:
            Dictionary with aggregate statistics and closest matches
        """
        self._ensure_model_loaded()

        ref_dir = Path(reference_dir)
        ref_images = list(ref_dir.glob("*.jpg")) + list(ref_dir.glob("*.png"))

        if len(ref_images) == 0:
            return {"error": f"No images found in {reference_dir}"}

        # Sample evenly
        step = max(1, len(ref_images) // sample_size)
        sampled = ref_images[::step][:sample_size]

        # Extract render features once
        render_cls = extract_cls_feature(render_path)

        # Compare to each reference
        similarities = []
        for ref_path in sampled:
            ref_cls = extract_cls_feature(str(ref_path))
            sim = compute_global_similarity(render_cls, ref_cls)
            similarities.append({
                "path": str(ref_path.name),
                "similarity": round(sim, 4)
            })

        # Sort by similarity
        similarities.sort(key=lambda x: x["similarity"], reverse=True)

        sim_values = [s["similarity"] for s in similarities]

        return {
            "render_path": render_path,
            "reference_dir": reference_dir,
            "samples_compared": len(similarities),
            "mean_similarity": round(float(np.mean(sim_values)), 4),
            "max_similarity": round(float(np.max(sim_values)), 4),
            "min_similarity": round(float(np.min(sim_values)), 4),
            "std_similarity": round(float(np.std(sim_values)), 4),
            "closest_matches": similarities[:5],
            "farthest_matches": similarities[-3:]
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def evaluate_structural_similarity(
    render_path: str,
    reference_path: str,
    effect_type: str = "sun"
) -> Dict[str, Any]:
    """
    Convenience function for quick structural evaluation.

    Returns dict suitable for JSON serialization.
    """
    evaluator = DinoStructuralEvaluator(effect_type=effect_type)
    result = evaluator.evaluate(render_path, reference_path, include_heatmap=False)
    return asdict(result)


def save_similarity_heatmap(
    render_path: str,
    reference_path: str,
    output_path: str
) -> str:
    """
    Generate and save a visual similarity heatmap.

    Args:
        render_path: Path to render
        reference_path: Path to reference
        output_path: Where to save the heatmap image

    Returns:
        Path to saved heatmap
    """
    try:
        import matplotlib.pyplot as plt
        from PIL import Image
    except ImportError:
        return "matplotlib not installed - cannot generate heatmap"

    evaluator = DinoStructuralEvaluator()
    result = evaluator.evaluate(render_path, reference_path, include_heatmap=True)

    if result.similarity_heatmap is None:
        return "No heatmap generated"

    heatmap = np.array(result.similarity_heatmap)

    # Create figure with render, reference, and heatmap
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Render
    render_img = Image.open(render_path)
    axes[0].imshow(render_img)
    axes[0].set_title("Render")
    axes[0].axis('off')

    # Reference
    ref_img = Image.open(reference_path)
    axes[1].imshow(ref_img)
    axes[1].set_title("Reference")
    axes[1].axis('off')

    # Heatmap
    im = axes[2].imshow(heatmap, cmap='RdYlGn', vmin=0, vmax=1)
    axes[2].set_title(f"Structural Similarity\n(Global: {result.global_structural_similarity:.3f})")
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python dino_structural_eval.py <render_path> <reference_path> [--heatmap output.png]")
        print("\nExample:")
        print("  python dino_structural_eval.py render.png reference.jpg")
        print("  python dino_structural_eval.py render.png reference.jpg --heatmap similarity.png")
        sys.exit(1)

    render_path = sys.argv[1]
    reference_path = sys.argv[2]

    # Check for heatmap flag
    heatmap_path = None
    if "--heatmap" in sys.argv:
        idx = sys.argv.index("--heatmap")
        if idx + 1 < len(sys.argv):
            heatmap_path = sys.argv[idx + 1]

    print(f"Evaluating structural similarity...")
    print(f"  Render: {render_path}")
    print(f"  Reference: {reference_path}")

    evaluator = DinoStructuralEvaluator()
    result = evaluator.evaluate(render_path, reference_path, include_heatmap=heatmap_path is not None)

    print(f"\n=== Results ===")
    print(f"Global Structural Similarity: {result.global_structural_similarity:.4f}")
    print(f"Quality: {result.structural_quality}")
    print(f"Interpretation: {result.interpretation}")

    if result.problem_regions:
        print(f"\nProblem Regions ({len(result.problem_regions)}):")
        for region in result.problem_regions[:5]:
            print(f"  - {region['region']}: severity={region['avg_severity']:.2f} ({region['severity_level']})")

    print(f"\nRecommendations:")
    for rec in result.recommendations:
        print(f"  - {rec}")

    if heatmap_path:
        saved = save_similarity_heatmap(render_path, reference_path, heatmap_path)
        print(f"\nHeatmap saved to: {saved}")
