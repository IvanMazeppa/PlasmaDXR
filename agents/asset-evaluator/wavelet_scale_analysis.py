#!/usr/bin/env python3
"""
Multi-Scale Wavelet Analysis for Procedural Texture Detection

Detects whether an image has procedural (single-scale) or organic (multi-scale) texture.
This is critical because aggregate statistics (edge_density) can be identical for both.

Key insight:
- Procedural textures (Perlin noise, etc.) have energy concentrated at specific scales
- Natural images have energy distributed across many scales
- Scale entropy < 1.5 = procedural, > 2.0 = natural

Research basis:
- Wavelet-optimized whitening for solar images (A&A 2023)
- See: docs/EVALUATION_SYSTEM_IMPROVEMENT_PROPOSAL.md

Usage:
    analyzer = WaveletScaleAnalyzer()
    result = analyzer.analyze(image_path)
    print(f"Scale entropy: {result.scale_entropy}")
    print(f"Is procedural: {result.is_procedural}")
"""

import json
import numpy as np
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class WaveletAnalysisResult:
    """Result of wavelet scale analysis."""
    # Scale distribution
    scale_energies: List[float]  # Energy at each decomposition level
    scale_entropy: float  # Entropy of energy distribution (higher = more natural)
    dominant_scale: int  # Which scale has most energy (0 = finest)

    # Interpretation
    is_procedural: bool  # True if texture appears procedural
    procedural_confidence: float  # 0-1 confidence in procedural detection
    texture_quality: str  # "natural", "mixed", "procedural"

    # Detailed analysis
    energy_concentration: float  # How concentrated energy is (0-1, higher = more procedural)
    scale_spread: float  # Standard deviation of scale energies

    # Comparison to reference (if provided)
    reference_entropy: Optional[float] = None
    entropy_difference: Optional[float] = None

    # Recommendations
    interpretation: str = ""
    recommendations: List[str] = field(default_factory=list)

    # Metadata
    image_path: str = ""
    num_levels: int = 5


@dataclass
class ScaleComparisonResult:
    """Result of comparing scale distributions between two images."""
    render_entropy: float
    reference_entropy: float
    entropy_difference: float  # reference - render (positive = render has less entropy)

    render_is_procedural: bool
    reference_is_procedural: bool

    scale_correlation: float  # How similar the scale distributions are

    interpretation: str
    recommendations: List[str]


# =============================================================================
# Wavelet Decomposition
# =============================================================================

def wavelet_decompose(image: np.ndarray, wavelet: str = 'db4', levels: int = 5) -> List[np.ndarray]:
    """
    Perform multi-level 2D wavelet decomposition.

    Args:
        image: 2D grayscale image
        wavelet: Wavelet type (default: Daubechies-4)
        levels: Number of decomposition levels

    Returns:
        List of detail coefficient arrays at each level
    """
    import pywt

    # Ensure image is 2D
    if len(image.shape) == 3:
        image = np.mean(image, axis=-1)

    # Normalize to 0-1
    image = image.astype(np.float64)
    if image.max() > 1:
        image = image / 255.0

    # Perform wavelet decomposition
    coeffs = pywt.wavedec2(image, wavelet, level=levels)

    # Extract detail coefficients at each level
    # coeffs[0] is approximation, coeffs[1:] are (cH, cV, cD) tuples
    detail_energies = []

    for level_coeffs in coeffs[1:]:
        # level_coeffs is (cH, cV, cD) - horizontal, vertical, diagonal details
        cH, cV, cD = level_coeffs

        # Combine all detail directions
        level_energy = np.sqrt(cH**2 + cV**2 + cD**2)
        detail_energies.append(level_energy)

    return detail_energies


def compute_scale_energies(detail_coeffs: List[np.ndarray]) -> List[float]:
    """
    Compute energy at each scale level.

    Energy is the mean squared magnitude of wavelet coefficients.
    """
    energies = []
    for coeffs in detail_coeffs:
        energy = np.mean(coeffs ** 2)
        energies.append(float(energy))
    return energies


def compute_scale_entropy(energies: List[float]) -> float:
    """
    Compute entropy of scale energy distribution.

    Higher entropy = energy spread across more scales = more natural.
    Lower entropy = energy concentrated at few scales = more procedural.

    Args:
        energies: Energy values at each scale

    Returns:
        Entropy value (typically 0-3, with 1.5 as threshold)
    """
    # Normalize to probability distribution
    total = sum(energies) + 1e-10
    probs = [e / total for e in energies]

    # Compute entropy
    entropy = 0.0
    for p in probs:
        if p > 1e-10:
            entropy -= p * np.log2(p)

    return entropy


# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_scale_distribution(image_path: str, levels: int = 5) -> Dict[str, Any]:
    """
    Analyze the scale distribution of an image using wavelets.

    Args:
        image_path: Path to image file
        levels: Number of wavelet decomposition levels

    Returns:
        Dictionary with scale energies, entropy, and interpretation
    """
    from PIL import Image

    # Load image
    img = Image.open(image_path).convert('L')  # Grayscale
    arr = np.array(img, dtype=np.float64)

    # Decompose
    detail_coeffs = wavelet_decompose(arr, levels=levels)

    # Compute energies
    energies = compute_scale_energies(detail_coeffs)

    # Compute entropy
    entropy = compute_scale_entropy(energies)

    # Find dominant scale
    dominant_scale = np.argmax(energies)

    # Compute energy concentration (Gini-like coefficient)
    sorted_energies = sorted(energies, reverse=True)
    total = sum(sorted_energies) + 1e-10
    cumsum = np.cumsum(sorted_energies)
    # How much energy is in top 2 scales?
    top2_ratio = cumsum[1] / total if len(cumsum) > 1 else 1.0

    return {
        "scale_energies": energies,
        "scale_entropy": entropy,
        "dominant_scale": int(dominant_scale),
        "energy_concentration": float(top2_ratio),
        "scale_spread": float(np.std(energies)),
        "num_levels": levels
    }


def detect_procedural_texture(
    image_path: str,
    entropy_threshold: float = 1.5,
    concentration_threshold: float = 0.7
) -> Dict[str, Any]:
    """
    Detect whether an image has procedural (single-scale) texture.

    Procedural textures like Perlin noise have:
    - Low scale entropy (energy concentrated at specific frequencies)
    - High energy concentration in 1-2 scales

    Natural textures have:
    - High scale entropy (energy spread across scales)
    - Low energy concentration

    Args:
        image_path: Path to image
        entropy_threshold: Below this = procedural (default 1.5)
        concentration_threshold: Above this = procedural (default 0.7)

    Returns:
        Detection result with confidence
    """
    analysis = analyze_scale_distribution(image_path)

    entropy = analysis["scale_entropy"]
    concentration = analysis["energy_concentration"]

    # Determine if procedural
    entropy_says_procedural = entropy < entropy_threshold
    concentration_says_procedural = concentration > concentration_threshold

    # Combine signals
    if entropy_says_procedural and concentration_says_procedural:
        is_procedural = True
        confidence = 0.9
        quality = "procedural"
    elif entropy_says_procedural or concentration_says_procedural:
        is_procedural = True
        confidence = 0.6
        quality = "mixed"
    else:
        is_procedural = False
        confidence = 0.8
        quality = "natural"

    # Adjust confidence based on how far from threshold
    if is_procedural:
        entropy_margin = (entropy_threshold - entropy) / entropy_threshold
        confidence = min(0.95, confidence + entropy_margin * 0.2)
    else:
        entropy_margin = (entropy - entropy_threshold) / entropy_threshold
        confidence = min(0.95, confidence + entropy_margin * 0.2)

    return {
        **analysis,
        "is_procedural": is_procedural,
        "procedural_confidence": round(confidence, 3),
        "texture_quality": quality,
        "entropy_threshold": entropy_threshold,
        "concentration_threshold": concentration_threshold
    }


def compare_scale_distributions(
    render_path: str,
    reference_path: str
) -> ScaleComparisonResult:
    """
    Compare scale distributions between render and reference.

    Useful for detecting if render has different texture characteristics
    than reference (e.g., procedural vs natural).
    """
    render_analysis = analyze_scale_distribution(render_path)
    ref_analysis = analyze_scale_distribution(reference_path)

    render_entropy = render_analysis["scale_entropy"]
    ref_entropy = ref_analysis["scale_entropy"]

    # Entropy difference (positive = render has less entropy = more procedural)
    entropy_diff = ref_entropy - render_entropy

    # Scale correlation (how similar are the energy distributions)
    render_energies = np.array(render_analysis["scale_energies"])
    ref_energies = np.array(ref_analysis["scale_energies"])

    # Normalize for correlation
    render_norm = render_energies / (np.sum(render_energies) + 1e-10)
    ref_norm = ref_energies / (np.sum(ref_energies) + 1e-10)

    correlation = float(np.corrcoef(render_norm, ref_norm)[0, 1])

    # Determine procedural status
    render_procedural = render_entropy < 1.5
    ref_procedural = ref_entropy < 1.5

    # Interpretation
    if render_procedural and not ref_procedural:
        interpretation = (
            f"TEXTURE MISMATCH: Render has procedural texture (entropy={render_entropy:.2f}) "
            f"while reference is natural (entropy={ref_entropy:.2f}). "
            "This is the 'popcorn texture' problem."
        )
        recommendations = [
            "Use multi-octave noise (FBM) instead of single-frequency noise",
            "Add turbulence and scale variation to break uniformity",
            "Increase noise complexity with multiple overlapping frequencies"
        ]
    elif entropy_diff > 0.5:
        interpretation = (
            f"Render texture is simpler than reference. "
            f"Render entropy: {render_entropy:.2f}, Reference: {ref_entropy:.2f}"
        )
        recommendations = [
            "Add more detail scales to the texture",
            "Increase procedural noise octaves"
        ]
    elif correlation < 0.5:
        interpretation = (
            f"Scale distributions differ significantly (correlation={correlation:.2f}). "
            "The render may have different feature sizes than reference."
        )
        recommendations = [
            "Adjust noise frequency to match reference feature sizes",
            "Check that simulation resolution matches target detail level"
        ]
    else:
        interpretation = (
            f"Scale distributions are reasonably similar. "
            f"Entropy: render={render_entropy:.2f}, reference={ref_entropy:.2f}"
        )
        recommendations = []

    return ScaleComparisonResult(
        render_entropy=round(render_entropy, 4),
        reference_entropy=round(ref_entropy, 4),
        entropy_difference=round(entropy_diff, 4),
        render_is_procedural=render_procedural,
        reference_is_procedural=ref_procedural,
        scale_correlation=round(correlation, 4),
        interpretation=interpretation,
        recommendations=recommendations
    )


# =============================================================================
# Main Analyzer Class
# =============================================================================

class WaveletScaleAnalyzer:
    """
    Multi-scale wavelet analyzer for texture quality assessment.

    Detects procedural textures by analyzing the distribution of energy
    across wavelet decomposition scales.

    Key insight: Natural images have energy spread across many scales,
    while procedural noise concentrates energy at specific scales.

    Example:
        analyzer = WaveletScaleAnalyzer()
        result = analyzer.analyze("render.png")
        if result.is_procedural:
            print(f"Procedural texture detected (entropy={result.scale_entropy})")
    """

    def __init__(
        self,
        levels: int = 5,
        entropy_threshold: float = 1.5,
        concentration_threshold: float = 0.7
    ):
        """
        Initialize analyzer.

        Args:
            levels: Number of wavelet decomposition levels
            entropy_threshold: Below this entropy = procedural
            concentration_threshold: Above this concentration = procedural
        """
        self.levels = levels
        self.entropy_threshold = entropy_threshold
        self.concentration_threshold = concentration_threshold

    def analyze(
        self,
        image_path: str,
        reference_path: Optional[str] = None
    ) -> WaveletAnalysisResult:
        """
        Analyze texture quality using wavelet decomposition.

        Args:
            image_path: Path to image to analyze
            reference_path: Optional reference for comparison

        Returns:
            WaveletAnalysisResult with full analysis
        """
        # Get basic analysis
        detection = detect_procedural_texture(
            image_path,
            self.entropy_threshold,
            self.concentration_threshold
        )

        # Build recommendations based on findings
        recommendations = []

        if detection["is_procedural"]:
            if detection["procedural_confidence"] > 0.7:
                recommendations.append(
                    "CRITICAL: Texture appears procedural/uniform. "
                    "Use multi-octave FBM noise instead of single-frequency noise."
                )
            recommendations.append(
                f"Scale entropy ({detection['scale_entropy']:.2f}) is below natural threshold (1.5). "
                "Add more frequency variation."
            )
            recommendations.append(
                "Consider adding turbulence, curl noise, or Worley noise for organic appearance."
            )

        # Compare to reference if provided
        ref_entropy = None
        entropy_diff = None

        if reference_path:
            comparison = compare_scale_distributions(image_path, reference_path)
            ref_entropy = comparison.reference_entropy
            entropy_diff = comparison.entropy_difference
            recommendations.extend(comparison.recommendations)

        # Build interpretation
        quality = detection["texture_quality"]
        entropy = detection["scale_entropy"]

        if quality == "procedural":
            interpretation = (
                f"Texture is PROCEDURAL (entropy={entropy:.2f}). "
                "Energy is concentrated at specific scales, creating uniform/repetitive appearance. "
                "This is the 'popcorn texture' problem."
            )
        elif quality == "mixed":
            interpretation = (
                f"Texture has MIXED characteristics (entropy={entropy:.2f}). "
                "Some procedural elements present but also natural variation."
            )
        else:
            interpretation = (
                f"Texture appears NATURAL (entropy={entropy:.2f}). "
                "Energy is well-distributed across scales."
            )

        return WaveletAnalysisResult(
            scale_energies=detection["scale_energies"],
            scale_entropy=round(detection["scale_entropy"], 4),
            dominant_scale=detection["dominant_scale"],
            is_procedural=detection["is_procedural"],
            procedural_confidence=detection["procedural_confidence"],
            texture_quality=quality,
            energy_concentration=round(detection["energy_concentration"], 4),
            scale_spread=round(detection["scale_spread"], 6),
            reference_entropy=ref_entropy,
            entropy_difference=entropy_diff,
            interpretation=interpretation,
            recommendations=recommendations,
            image_path=image_path,
            num_levels=self.levels
        )

    def compare(
        self,
        render_path: str,
        reference_path: str
    ) -> ScaleComparisonResult:
        """
        Compare scale distributions between render and reference.
        """
        return compare_scale_distributions(render_path, reference_path)


# =============================================================================
# Visualization
# =============================================================================

def visualize_scale_analysis(
    image_path: str,
    output_path: str,
    reference_path: Optional[str] = None
) -> str:
    """
    Generate visualization of scale analysis.

    Creates a figure showing:
    - Original image
    - Scale energy distribution bar chart
    - Wavelet coefficients at each scale
    """
    try:
        import matplotlib.pyplot as plt
        from PIL import Image
    except ImportError:
        return "matplotlib not installed"

    # Analyze
    analyzer = WaveletScaleAnalyzer()
    result = analyzer.analyze(image_path, reference_path)

    # Load image
    img = Image.open(image_path)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(img)
    axes[0].set_title(f"Image\nEntropy: {result.scale_entropy:.3f}")
    axes[0].axis('off')

    # Scale energy distribution
    scales = list(range(1, len(result.scale_energies) + 1))
    colors = ['red' if i == result.dominant_scale else 'steelblue'
              for i in range(len(scales))]

    axes[1].bar(scales, result.scale_energies, color=colors)
    axes[1].set_xlabel('Scale Level (1=finest)')
    axes[1].set_ylabel('Energy')
    axes[1].set_title(f'Scale Energy Distribution\n{result.texture_quality.upper()}')

    # Add entropy annotation
    axes[1].axhline(y=np.mean(result.scale_energies), color='green',
                    linestyle='--', label='Mean')
    axes[1].legend()

    # Entropy comparison (if reference provided)
    if result.reference_entropy is not None:
        entropies = [result.scale_entropy, result.reference_entropy]
        labels = ['Render', 'Reference']
        colors = ['orange' if result.is_procedural else 'green', 'green']

        axes[2].bar(labels, entropies, color=colors)
        axes[2].axhline(y=1.5, color='red', linestyle='--', label='Procedural threshold')
        axes[2].set_ylabel('Scale Entropy')
        axes[2].set_title('Entropy Comparison')
        axes[2].legend()
    else:
        # Show threshold comparison
        axes[2].bar(['Image', 'Natural\nThreshold'],
                    [result.scale_entropy, 1.5],
                    color=['red' if result.is_procedural else 'green', 'gray'])
        axes[2].set_ylabel('Scale Entropy')
        axes[2].set_title(f'Procedural: {result.is_procedural}\nConfidence: {result.procedural_confidence:.0%}')

    plt.tight_layout()

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python wavelet_scale_analysis.py <image_path> [reference_path] [--visualize output.png]")
        print("\nExample:")
        print("  python wavelet_scale_analysis.py render.png")
        print("  python wavelet_scale_analysis.py render.png reference.jpg")
        print("  python wavelet_scale_analysis.py render.png reference.jpg --visualize analysis.png")
        sys.exit(1)

    image_path = sys.argv[1]
    reference_path = None
    visualize_path = None

    # Parse args
    args = sys.argv[2:]
    if "--visualize" in args:
        viz_idx = args.index("--visualize")
        if viz_idx + 1 < len(args):
            visualize_path = args[viz_idx + 1]
        args = args[:viz_idx] + args[viz_idx+2:]

    if args:
        reference_path = args[0]

    print(f"Analyzing scale distribution...")
    print(f"  Image: {image_path}")
    if reference_path:
        print(f"  Reference: {reference_path}")

    analyzer = WaveletScaleAnalyzer()
    result = analyzer.analyze(image_path, reference_path)

    print(f"\n=== Wavelet Scale Analysis Results ===")
    print(f"Scale Entropy: {result.scale_entropy:.4f}")
    print(f"Texture Quality: {result.texture_quality.upper()}")
    print(f"Is Procedural: {result.is_procedural} (confidence: {result.procedural_confidence:.0%})")
    print(f"Dominant Scale: {result.dominant_scale + 1} (of {result.num_levels})")
    print(f"Energy Concentration: {result.energy_concentration:.2%}")

    print(f"\nScale Energies:")
    for i, energy in enumerate(result.scale_energies):
        marker = "←" if i == result.dominant_scale else ""
        print(f"  Level {i+1}: {energy:.6f} {marker}")

    if result.reference_entropy is not None:
        print(f"\nComparison to Reference:")
        print(f"  Render entropy: {result.scale_entropy:.4f}")
        print(f"  Reference entropy: {result.reference_entropy:.4f}")
        print(f"  Difference: {result.entropy_difference:.4f}")

    print(f"\nInterpretation: {result.interpretation}")

    if result.recommendations:
        print(f"\nRecommendations:")
        for rec in result.recommendations:
            print(f"  - {rec}")

    if visualize_path:
        saved = visualize_scale_analysis(image_path, visualize_path, reference_path)
        print(f"\nVisualization saved to: {saved}")
