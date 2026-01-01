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
# Enhanced Texture Analysis (Uniformity Detection)
# =============================================================================

def compute_local_variance_map(image: np.ndarray, window_size: int = 16) -> np.ndarray:
    """
    Compute local variance across the image.

    Procedural textures have uniform variance across the image.
    Natural textures have varying variance (some areas smooth, some detailed).
    """
    from scipy.ndimage import uniform_filter

    if len(image.shape) == 3:
        image = np.mean(image, axis=-1)

    image = image.astype(np.float64)

    # Local mean
    local_mean = uniform_filter(image, size=window_size)

    # Local variance
    local_var = uniform_filter((image - local_mean)**2, size=window_size)

    return local_var


def analyze_texture_uniformity(image_path: str, content_threshold: int = 30) -> Dict[str, Any]:
    """
    Analyze texture uniformity to detect procedural patterns.

    Key insight: Procedural textures have UNIFORM local statistics across the image.
    Natural textures have VARYING local statistics (some areas smooth, some detailed).

    Args:
        image_path: Path to image
        content_threshold: Brightness threshold for content mask

    Returns:
        Dictionary with uniformity metrics
    """
    from PIL import Image
    from scipy.ndimage import uniform_filter

    img = Image.open(image_path)
    arr = np.array(img, dtype=np.float64)

    if len(arr.shape) == 3:
        gray = np.mean(arr, axis=-1)
    else:
        gray = arr

    # Create content mask (exclude black background)
    content_mask = gray > content_threshold

    if np.sum(content_mask) < 100:
        return {"error": "Not enough content pixels"}

    # Extract content pixels
    content = gray[content_mask]

    # Compute local variance map
    local_var = compute_local_variance_map(gray, window_size=16)
    content_local_var = local_var[content_mask]

    # Key metric: How uniform is the local variance?
    # Low CV = uniform texture (procedural)
    # High CV = varying texture (natural)
    var_mean = np.mean(content_local_var)
    var_std = np.std(content_local_var)
    variance_cv = var_std / (var_mean + 1e-10)  # Coefficient of variation

    # Compute local contrast variation
    # Procedural: similar contrast everywhere
    # Natural: varying contrast (some smooth, some detailed)
    window = 32
    local_contrast = uniform_filter(np.abs(np.gradient(gray)[0]), size=window)
    content_contrast = local_contrast[content_mask]

    contrast_mean = np.mean(content_contrast)
    contrast_std = np.std(content_contrast)
    contrast_cv = contrast_std / (contrast_mean + 1e-10)

    # Determine if procedural
    # Low CV (<0.8) suggests uniform/procedural texture
    uniformity_score = 1.0 - min(variance_cv, 1.0)
    is_uniform = variance_cv < 0.8

    # Interpretation
    if variance_cv < 0.5:
        texture_type = "highly_uniform"
        interpretation = "Texture is HIGHLY UNIFORM - strong procedural/noise pattern detected"
    elif variance_cv < 0.8:
        texture_type = "uniform"
        interpretation = "Texture is UNIFORM - likely procedural noise"
    elif variance_cv < 1.2:
        texture_type = "mixed"
        interpretation = "Texture has MIXED uniformity"
    else:
        texture_type = "varied"
        interpretation = "Texture is VARIED - appears natural/organic"

    return {
        "variance_cv": round(variance_cv, 4),
        "contrast_cv": round(contrast_cv, 4),
        "uniformity_score": round(uniformity_score, 4),
        "is_uniform": is_uniform,
        "texture_type": texture_type,
        "interpretation": interpretation,
        "variance_mean": round(var_mean, 2),
        "variance_std": round(var_std, 2),
        "content_pixels": int(np.sum(content_mask))
    }


def detect_repetitive_pattern(image_path: str, content_threshold: int = 30) -> Dict[str, Any]:
    """
    Detect repetitive patterns using autocorrelation.

    Procedural noise often has periodic autocorrelation peaks.
    Natural textures have smoother, non-periodic autocorrelation.
    """
    from PIL import Image
    from scipy import signal

    img = Image.open(image_path)
    arr = np.array(img, dtype=np.float64)

    if len(arr.shape) == 3:
        gray = np.mean(arr, axis=-1)
    else:
        gray = arr

    # Focus on center region (avoid edge effects)
    h, w = gray.shape
    center_region = gray[h//4:3*h//4, w//4:3*w//4]

    # Compute 2D autocorrelation via FFT
    fft = np.fft.fft2(center_region - np.mean(center_region))
    power = np.abs(fft) ** 2
    autocorr = np.fft.ifft2(power).real
    autocorr = np.fft.fftshift(autocorr)

    # Normalize
    autocorr = autocorr / autocorr.max()

    # Analyze peaks (excluding center)
    center_y, center_x = autocorr.shape[0] // 2, autocorr.shape[1] // 2

    # Mask out center region
    mask_radius = min(center_y, center_x) // 10
    y, x = np.ogrid[:autocorr.shape[0], :autocorr.shape[1]]
    center_mask = ((y - center_y)**2 + (x - center_x)**2) > mask_radius**2

    # Find secondary peaks
    masked_autocorr = autocorr * center_mask
    secondary_max = np.max(masked_autocorr)

    # High secondary peaks indicate repetitive pattern
    has_repetition = secondary_max > 0.3
    repetition_strength = min(secondary_max, 1.0)

    if secondary_max > 0.5:
        interpretation = "STRONG repetitive pattern detected - highly procedural"
    elif secondary_max > 0.3:
        interpretation = "Moderate repetitive pattern - some procedural elements"
    elif secondary_max > 0.15:
        interpretation = "Weak repetitive pattern"
    else:
        interpretation = "No significant repetitive pattern - appears organic"

    return {
        "has_repetition": has_repetition,
        "repetition_strength": round(repetition_strength, 4),
        "secondary_peak_max": round(secondary_max, 4),
        "interpretation": interpretation
    }


def analyze_feature_sizes(image_path: str, content_threshold: int = 30) -> Dict[str, Any]:
    """
    Analyze the distribution of feature sizes using connected components.

    This is the MOST DISCRIMINATING metric discovered:
    - Procedural textures have UNIFORM feature sizes (low CV)
    - Natural textures have VARIED feature sizes (high CV)

    Test results:
    - Synthetic "popcorn" render: CV = 1.83 (uniform sizes)
    - Real solar footage: CV = 36.97 (varied sizes - 20x higher!)

    Args:
        image_path: Path to image
        content_threshold: Brightness threshold for binary mask

    Returns:
        Dictionary with feature size metrics
    """
    import cv2

    img = cv2.imread(image_path)
    if img is None:
        return {"error": f"Could not load image: {image_path}"}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold to get features
    _, thresh = cv2.threshold(gray, content_threshold, 255, cv2.THRESH_BINARY)

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Get areas (filter tiny noise)
    areas = np.array([cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 10])

    if len(areas) < 3:
        return {
            "error": "Not enough features detected",
            "feature_count": len(areas),
            "is_uniform_size": None,
            "size_cv": None
        }

    # Key statistics
    mean_area = float(np.mean(areas))
    std_area = float(np.std(areas))
    cv_area = std_area / (mean_area + 1e-10)  # Coefficient of variation

    min_area = float(np.min(areas))
    max_area = float(np.max(areas))
    range_ratio = max_area / (min_area + 1e-10)

    p25, p50, p75 = [float(x) for x in np.percentile(areas, [25, 50, 75])]
    iqr = p75 - p25

    # Interpretation
    # Based on empirical testing:
    # - Render CV ~1.83 (uniform)
    # - Reference CV ~36.97 (highly varied)
    # Threshold at CV=5 separates procedural from natural

    if cv_area < 3:
        uniformity = "highly_uniform"
        interpretation = "Features are HIGHLY UNIFORM in size - strong procedural signature"
        is_uniform = True
    elif cv_area < 10:
        uniformity = "uniform"
        interpretation = "Features are UNIFORM in size - likely procedural"
        is_uniform = True
    elif cv_area < 20:
        uniformity = "mixed"
        interpretation = "Features have MIXED size distribution"
        is_uniform = False
    else:
        uniformity = "varied"
        interpretation = "Features are HIGHLY VARIED in size - natural multi-scale structure"
        is_uniform = False

    return {
        "feature_count": len(areas),
        "size_cv": round(cv_area, 4),
        "size_mean": round(mean_area, 2),
        "size_std": round(std_area, 2),
        "size_min": round(min_area, 2),
        "size_max": round(max_area, 2),
        "size_range_ratio": round(range_ratio, 2),
        "size_p25": round(p25, 2),
        "size_p50": round(p50, 2),
        "size_p75": round(p75, 2),
        "size_iqr": round(iqr, 2),
        "uniformity_type": uniformity,
        "is_uniform_size": is_uniform,
        "interpretation": interpretation
    }


def comprehensive_texture_analysis(
    image_path: str,
    reference_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Comprehensive texture analysis combining multiple methods.

    Combines:
    1. Wavelet scale entropy (procedural = concentrated at one scale)
    2. Texture uniformity (procedural = uniform local variance)
    3. Repetitive pattern detection (procedural = periodic autocorrelation)
    4. Feature size distribution (procedural = uniform sizes) <- MOST DISCRIMINATING

    Returns actionable assessment of texture quality with procedural score.
    """
    results = {
        "image_path": image_path
    }

    # 1. Wavelet scale analysis
    analyzer = WaveletScaleAnalyzer()
    wavelet_result = analyzer.analyze(image_path, reference_path)
    results["wavelet"] = {
        "scale_entropy": wavelet_result.scale_entropy,
        "is_procedural": wavelet_result.is_procedural,
        "procedural_confidence": wavelet_result.procedural_confidence,
        "dominant_scale": wavelet_result.dominant_scale
    }

    # 2. Texture uniformity analysis
    uniformity = analyze_texture_uniformity(image_path)
    results["uniformity"] = uniformity

    # 3. Repetitive pattern detection
    repetition = detect_repetitive_pattern(image_path)
    results["repetition"] = repetition

    # 4. Feature size distribution (MOST DISCRIMINATING)
    feature_sizes = analyze_feature_sizes(image_path)
    results["feature_sizes"] = feature_sizes

    # =================================================================
    # COMBINED PROCEDURAL SCORE (0-100, higher = more procedural)
    # =================================================================
    # Feature size CV is the DOMINANT discriminator based on empirical testing:
    # - Synthetic render: CV = 1.83 (uniform sizes)
    # - Real solar footage: CV = 36.97 (varied sizes - 20x higher!)
    # Other metrics (wavelet, uniformity, repetition) can give false signals

    procedural_score = 0.0
    weights_used = 0.0

    # Feature size uniformity (60% weight - DOMINANT METRIC)
    # This is empirically the best discriminator
    if feature_sizes.get("size_cv") is not None:
        size_cv = feature_sizes["size_cv"]
        # CV < 3 = very procedural (score 60), CV > 20 = natural (score 0)
        # Sigmoid-like scoring for smooth transition
        if size_cv < 3:
            procedural_score += 60
        elif size_cv < 5:
            procedural_score += 55 - (size_cv - 3) * 10  # 55 to 35
        elif size_cv < 10:
            procedural_score += 35 - (size_cv - 5) * 5  # 35 to 10
        elif size_cv < 20:
            procedural_score += 10 - (size_cv - 10) * 1  # 10 to 0
        weights_used += 60

    # Wavelet scale entropy (25% weight)
    # Entropy < 1.0 = procedural, > 2.0 = natural
    entropy = wavelet_result.scale_entropy
    if entropy < 1.0:
        procedural_score += 25
    elif entropy < 1.5:
        procedural_score += 20 * (1 - (entropy - 1.0) / 0.5)
    elif entropy < 2.0:
        procedural_score += 10 * (1 - (entropy - 1.5) / 0.5)
    weights_used += 25

    # Texture uniformity (10% weight - REDUCED, unreliable with black backgrounds)
    # Can give false signals due to background contrast effects
    if uniformity.get("variance_cv") is not None:
        if uniformity.get("is_uniform", False):
            procedural_score += 10
        weights_used += 10

    # Repetitive pattern (5% weight - REDUCED, can false-positive on real granulation)
    if repetition.get("has_repetition", False):
        strength = repetition.get("repetition_strength", 0)
        procedural_score += 5 * min(strength, 1.0)
    weights_used += 5

    # Normalize to 100 if weights < 100
    if weights_used > 0:
        procedural_score = (procedural_score / weights_used) * 100

    # Determine texture quality based on score
    if procedural_score >= 60:
        texture_quality = "procedural"
        verdict = "PROCEDURAL - Texture appears synthetic/generated"
    elif procedural_score >= 35:
        texture_quality = "mixed"
        verdict = "MIXED - Some procedural characteristics detected"
    else:
        texture_quality = "natural"
        verdict = "NATURAL - Texture appears organic/realistic"

    # Collect specific signals
    signals = []
    if feature_sizes.get("is_uniform_size", False):
        signals.append("uniform_feature_sizes")
    if uniformity.get("is_uniform", False):
        signals.append("uniform_local_variance")
    if repetition.get("has_repetition", False):
        signals.append("repetitive_pattern")
    if wavelet_result.is_procedural:
        signals.append("concentrated_scale_energy")

    results["combined_score"] = {
        "procedural_score": round(procedural_score, 1),
        "texture_quality": texture_quality,
        "texture_verdict": verdict,
        "signals_detected": signals,
        "signal_count": len(signals),
        "primary_indicator": "feature_size_uniformity" if feature_sizes.get("size_cv") is not None else "wavelet_entropy"
    }

    # Generate recommendations
    recommendations = []

    if feature_sizes.get("is_uniform_size", False):
        recommendations.append(
            "CRITICAL - UNIFORM FEATURE SIZES: All features are similar size. "
            "Use multi-scale noise (FBM) to create varied feature sizes like real solar granulation."
        )

    if "concentrated_scale_energy" in signals:
        recommendations.append(
            "SCALE CONCENTRATION: Energy concentrated at single scale. "
            "Add detail at multiple scales - large active regions AND fine granulation."
        )

    if "uniform_local_variance" in signals:
        recommendations.append(
            "TEXTURE UNIFORMITY: Local variance too consistent. "
            "Add dark filaments, bright faculae, and varied surface features."
        )

    if "repetitive_pattern" in signals:
        recommendations.append(
            "REPETITIVE PATTERN: Periodic texture detected. "
            "Use non-tiling noise or break up regularity with large-scale variation."
        )

    if not recommendations:
        recommendations.append("Texture appears natural with good variation across scales.")

    results["recommendations"] = recommendations

    return results


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
