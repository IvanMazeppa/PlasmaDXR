"""
ML-powered visual comparison tool using LPIPS (Learned Perceptual Image Patch Similarity)

Provides multi-level visual analysis:
1. Traditional CV metrics (SSIM, MSE, PSNR) - fast baseline
2. LPIPS perceptual similarity (pre-trained) - human-aligned
3. Difference heatmap generation
4. Aggregated similarity score
5. Human-readable interpretation
"""

from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import os

import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
import cv2


class MLVisualComparison:
    """ML-powered visual comparison using LPIPS and traditional CV metrics"""

    def __init__(self):
        """Initialize (lazy loading - LPIPS loaded on first use)"""
        self.lpips_model = None
        self.device = 'cpu'

    def _ensure_lpips_loaded(self):
        """Lazy load LPIPS model only when needed"""
        if self.lpips_model is None:
            # Import torch/lpips only when needed (avoids slow startup)
            import torch
            import lpips

            # Load LPIPS model (VGG backbone - best accuracy)
            # This downloads pre-trained weights on first use (~528MB)
            self.lpips_model = lpips.LPIPS(net='vgg', version='0.1')
            self.lpips_model.eval()
            self.lpips_model.to(self.device)

    def load_image(self, image_path: Path) -> np.ndarray:
        """
        Load image from path and convert to numpy array

        Args:
            image_path: Path to image file

        Returns:
            RGB image as numpy array (H, W, 3) with values in [0, 1]
        """
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img).astype(np.float32) / 255.0
        return img_array

    def prepare_tensor(self, img_array: np.ndarray):
        """
        Convert numpy array to PyTorch tensor for LPIPS

        Args:
            img_array: RGB image as numpy array (H, W, 3) in [0, 1]

        Returns:
            Tensor (1, 3, H, W) in [-1, 1] range
        """
        # Lazy load torch
        import torch

        # Convert to tensor and rearrange dimensions
        tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)

        # Normalize to [-1, 1] range (LPIPS expects this)
        tensor = tensor * 2.0 - 1.0

        return tensor.to(self.device)

    def traditional_metrics(
        self,
        img_before: np.ndarray,
        img_after: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute traditional computer vision similarity metrics

        Args:
            img_before: First image (H, W, 3) in [0, 1]
            img_after: Second image (H, W, 3) in [0, 1]

        Returns:
            Dictionary with SSIM, MSE, PSNR, histogram correlation
        """
        # SSIM (Structural Similarity Index) - closer to 1.0 = more similar
        ssim_score = ssim(
            img_before,
            img_after,
            data_range=1.0,
            channel_axis=2,  # RGB channels
            win_size=11
        )

        # MSE (Mean Squared Error) - lower = more similar
        mse_score = mse(img_before, img_after)

        # PSNR (Peak Signal-to-Noise Ratio) - higher = more similar
        psnr_score = psnr(img_before, img_after, data_range=1.0)

        # Histogram correlation (color distribution similarity)
        hist_corr = self.compute_histogram_correlation(img_before, img_after)

        return {
            "ssim": float(ssim_score),
            "mse": float(mse_score),
            "psnr": float(psnr_score),
            "histogram_correlation": float(hist_corr)
        }

    def compute_histogram_correlation(
        self,
        img_before: np.ndarray,
        img_after: np.ndarray
    ) -> float:
        """
        Compute histogram correlation between two images

        Args:
            img_before: First image (H, W, 3) in [0, 1]
            img_after: Second image (H, W, 3) in [0, 1]

        Returns:
            Correlation coefficient in [0, 1]
        """
        # Convert to 8-bit for OpenCV
        img1_8bit = (img_before * 255).astype(np.uint8)
        img2_8bit = (img_after * 255).astype(np.uint8)

        # Compute histograms for each channel
        correlations = []
        for channel in range(3):
            hist1 = cv2.calcHist([img1_8bit], [channel], None, [256], [0, 256])
            hist2 = cv2.calcHist([img2_8bit], [channel], None, [256], [0, 256])

            # Normalize histograms
            hist1 = cv2.normalize(hist1, hist1).flatten()
            hist2 = cv2.normalize(hist2, hist2).flatten()

            # Compute correlation
            corr = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            correlations.append(corr)

        # Average correlation across channels
        return float(np.mean(correlations))

    def lpips_similarity(
        self,
        img_before: np.ndarray,
        img_after: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute LPIPS perceptual similarity (pre-trained on ImageNet)

        LPIPS correlates well with human judgment (~0.92 correlation)

        Args:
            img_before: First image (H, W, 3) in [0, 1]
            img_after: Second image (H, W, 3) in [0, 1]

        Returns:
            Dictionary with LPIPS distance and similarity
        """
        # Lazy load LPIPS model on first use
        self._ensure_lpips_loaded()

        # Lazy load torch
        import torch

        # Prepare tensors
        tensor_before = self.prepare_tensor(img_before)
        tensor_after = self.prepare_tensor(img_after)

        # Compute LPIPS distance (lower = more similar)
        with torch.no_grad():
            distance = self.lpips_model(tensor_before, tensor_after).item()

        # Convert to similarity score (0-1, higher = more similar)
        # Using sigmoid-like transformation
        similarity = 1.0 / (1.0 + distance)

        return {
            "lpips_distance": float(distance),
            "lpips_similarity": float(similarity),
            "human_aligned": True,  # LPIPS correlates ~0.92 with human judgments
            "interpretation": "Pre-trained on ImageNet, understands human perception"
        }

    def compute_difference_heatmap(
        self,
        img_before: np.ndarray,
        img_after: np.ndarray
    ) -> np.ndarray:
        """
        Generate visual difference heatmap

        Args:
            img_before: First image (H, W, 3) in [0, 1]
            img_after: Second image (H, W, 3) in [0, 1]

        Returns:
            Heatmap as RGB image (H, W, 3) in [0, 1]
        """
        # Compute absolute difference
        diff = np.abs(img_before - img_after)

        # Convert to grayscale difference magnitude
        diff_magnitude = np.mean(diff, axis=2)

        # Normalize to [0, 1]
        diff_magnitude = (diff_magnitude - diff_magnitude.min()) / (diff_magnitude.max() - diff_magnitude.min() + 1e-8)

        # Apply colormap (red = high difference, blue = low difference)
        # Convert to 8-bit for OpenCV colormap
        diff_8bit = (diff_magnitude * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(diff_8bit, cv2.COLORMAP_JET)

        # Convert BGR to RGB and normalize to [0, 1]
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        return heatmap

    def aggregate_similarity_score(self, results: Dict[str, Any]) -> float:
        """
        Aggregate multiple metrics into overall similarity score

        Weights:
        - LPIPS: 50% (most human-aligned)
        - SSIM: 30% (structural similarity)
        - Histogram: 20% (color distribution)

        Args:
            results: Dictionary with traditional and LPIPS metrics

        Returns:
            Overall similarity score in [0, 1]
        """
        # Extract metrics
        lpips_sim = results["lpips"]["lpips_similarity"]
        ssim_score = results["traditional"]["ssim"]
        hist_corr = results["traditional"]["histogram_correlation"]

        # Weighted average
        overall = (
            0.50 * lpips_sim +
            0.30 * ssim_score +
            0.20 * hist_corr
        )

        return float(overall)

    def interpret_results(
        self,
        results: Dict[str, Any],
        img_before: np.ndarray,
        img_after: np.ndarray
    ) -> str:
        """
        Generate human-readable interpretation of comparison results

        Args:
            results: Comparison results dictionary
            img_before: First image
            img_after: Second image

        Returns:
            Human-readable interpretation string
        """
        overall = results["overall_similarity"]
        lpips_dist = results["lpips"]["lpips_distance"]
        ssim_score = results["traditional"]["ssim"]
        mse_score = results["traditional"]["mse"]

        # Classify similarity level
        if overall >= 0.95:
            similarity_level = "nearly identical"
        elif overall >= 0.85:
            similarity_level = "very similar"
        elif overall >= 0.70:
            similarity_level = "moderately similar"
        elif overall >= 0.50:
            similarity_level = "somewhat different"
        else:
            similarity_level = "significantly different"

        # Build interpretation
        lines = []
        lines.append(f"Images are {similarity_level} (overall similarity: {overall:.2%})")
        lines.append("")
        lines.append(f"Perceptual similarity (LPIPS): {results['lpips']['lpips_similarity']:.2%}")
        lines.append(f"  - Distance: {lpips_dist:.4f} (lower = more similar)")
        lines.append(f"  - Human-aligned metric (~92% correlation with human judgment)")
        lines.append("")
        lines.append(f"Structural similarity (SSIM): {ssim_score:.2%}")
        lines.append(f"  - Measures structural changes (edges, textures)")
        lines.append("")
        lines.append(f"Color distribution (histogram): {results['traditional']['histogram_correlation']:.2%}")
        lines.append(f"  - Measures color distribution similarity")
        lines.append("")

        # Detect specific types of differences
        if lpips_dist > 0.20:
            lines.append("⚠️ Significant perceptual differences detected")
            lines.append("  - Humans would easily notice these changes")
        elif lpips_dist > 0.10:
            lines.append("⚠️ Moderate perceptual differences")
            lines.append("  - Noticeable differences in visual quality")

        if ssim_score < 0.90:
            lines.append("⚠️ Structural changes detected")
            lines.append("  - Layout, edges, or textures have changed")

        if mse_score > 0.01:
            lines.append("⚠️ Pixel-level differences")
            lines.append(f"  - MSE: {mse_score:.6f}, PSNR: {results['traditional']['psnr']:.2f} dB")

        # Compute difference statistics
        diff = np.abs(img_before - img_after)
        diff_mean = np.mean(diff)
        diff_max = np.max(diff)
        changed_pixels = np.sum(diff > 0.01) / diff.size

        lines.append("")
        lines.append("Pixel-level statistics:")
        lines.append(f"  - Mean absolute difference: {diff_mean:.4f}")
        lines.append(f"  - Max absolute difference: {diff_max:.4f}")
        lines.append(f"  - Changed pixels (>1% threshold): {changed_pixels:.2%}")

        return "\n".join(lines)


async def compare_screenshots_ml(
    before_path: str,
    after_path: str,
    save_heatmap: bool = True,
    project_root: Optional[str] = None
) -> Dict[str, Any]:
    """
    ML-powered screenshot comparison with perceptual similarity

    Args:
        before_path: Path to "before" screenshot
        after_path: Path to "after" screenshot
        save_heatmap: Save difference heatmap to file
        project_root: Project root directory (defaults to env var)

    Returns:
        {
            "overall_similarity": 0.92,
            "traditional": {...},
            "lpips": {...},
            "difference_stats": {...},
            "heatmap_path": "path/to/heatmap.png" (if saved),
            "interpretation": "Human-readable analysis..."
        }
    """
    # Get project root
    if not project_root:
        project_root = os.getenv("PROJECT_ROOT", "/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean")

    # Convert to Path objects
    before = Path(before_path)
    after = Path(after_path)

    # Validate paths
    if not before.exists():
        return {
            "error": f"Before image not found: {before_path}"
        }

    if not after.exists():
        return {
            "error": f"After image not found: {after_path}"
        }

    # Initialize comparison
    comparator = MLVisualComparison()

    # Load images
    img_before = comparator.load_image(before)
    img_after = comparator.load_image(after)

    # Check dimensions match - auto-resize if needed
    resize_warning = None
    if img_before.shape != img_after.shape:
        # Auto-resize to smallest common dimensions
        target_h = min(img_before.shape[0], img_after.shape[0])
        target_w = min(img_before.shape[1], img_after.shape[1])

        original_before = f"{img_before.shape[1]}x{img_before.shape[0]}"
        original_after = f"{img_after.shape[1]}x{img_after.shape[0]}"

        # Resize using high-quality Lanczos interpolation
        img_before = cv2.resize(img_before, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        img_after = cv2.resize(img_after, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

        resize_warning = {
            "resized": True,
            "original_before": original_before,
            "original_after": original_after,
            "common_size": f"{target_w}x{target_h}",
            "message": f"⚠️  Images resized from {original_before} and {original_after} to {target_w}x{target_h} for comparison"
        }

    # Compute all metrics
    results = {}

    # Add resize warning if applicable
    if resize_warning:
        results["resize_warning"] = resize_warning

    # Traditional metrics
    results["traditional"] = comparator.traditional_metrics(img_before, img_after)

    # LPIPS perceptual similarity
    results["lpips"] = comparator.lpips_similarity(img_before, img_after)

    # Compute difference statistics
    diff = np.abs(img_before - img_after)
    results["difference_stats"] = {
        "mean_absolute_difference": float(np.mean(diff)),
        "max_absolute_difference": float(np.max(diff)),
        "changed_pixels_percent": float(np.sum(diff > 0.01) / diff.size * 100)
    }

    # Aggregate overall similarity
    results["overall_similarity"] = comparator.aggregate_similarity_score(results)

    # Generate difference heatmap
    heatmap = comparator.compute_difference_heatmap(img_before, img_after)

    if save_heatmap:
        # Save heatmap
        heatmaps_dir = Path(project_root) / "PIX" / "heatmaps"
        heatmaps_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename based on input names
        before_name = before.stem
        after_name = after.stem
        heatmap_filename = f"diff_{before_name}_vs_{after_name}.png"
        heatmap_path = heatmaps_dir / heatmap_filename

        # Save heatmap
        heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8))
        heatmap_img.save(heatmap_path)

        results["heatmap_path"] = str(heatmap_path)

    # Generate interpretation
    results["interpretation"] = comparator.interpret_results(results, img_before, img_after)

    # Add metadata
    results["metadata"] = {
        "before_path": str(before),
        "after_path": str(after),
        "image_size": f"{img_before.shape[1]}x{img_before.shape[0]}",
        "model": "LPIPS VGG (pre-trained on ImageNet)"
    }

    return results


def format_comparison_report(results: Dict[str, Any]) -> str:
    """
    Format ML comparison results as readable report

    Args:
        results: Results from compare_screenshots_ml()

    Returns:
        Formatted text report
    """
    if "error" in results:
        return f"Error: {results['error']}"

    lines = []
    lines.append("=" * 80)
    lines.append("ML-POWERED VISUAL COMPARISON REPORT")
    lines.append("=" * 80)
    lines.append("")

    # Metadata
    lines.append(f"Before: {results['metadata']['before_path']}")
    lines.append(f"After:  {results['metadata']['after_path']}")
    lines.append(f"Size:   {results['metadata']['image_size']}")
    lines.append(f"Model:  {results['metadata']['model']}")
    lines.append("")

    # Overall similarity
    lines.append("=" * 80)
    lines.append(f"OVERALL SIMILARITY: {results['overall_similarity']:.2%}")
    lines.append("=" * 80)
    lines.append("")

    # LPIPS results
    lines.append("PERCEPTUAL SIMILARITY (LPIPS):")
    lines.append("-" * 80)
    lpips = results["lpips"]
    lines.append(f"  Similarity: {lpips['lpips_similarity']:.2%}")
    lines.append(f"  Distance:   {lpips['lpips_distance']:.4f}")
    lines.append(f"  Human-aligned: {lpips['human_aligned']}")
    lines.append(f"  Note: {lpips['interpretation']}")
    lines.append("")

    # Traditional metrics
    lines.append("TRADITIONAL METRICS:")
    lines.append("-" * 80)
    trad = results["traditional"]
    lines.append(f"  SSIM (structural):     {trad['ssim']:.4f}")
    lines.append(f"  MSE (pixel error):     {trad['mse']:.6f}")
    lines.append(f"  PSNR (signal/noise):   {trad['psnr']:.2f} dB")
    lines.append(f"  Histogram correlation: {trad['histogram_correlation']:.4f}")
    lines.append("")

    # Difference statistics
    lines.append("DIFFERENCE STATISTICS:")
    lines.append("-" * 80)
    stats = results["difference_stats"]
    lines.append(f"  Mean absolute difference:  {stats['mean_absolute_difference']:.4f}")
    lines.append(f"  Max absolute difference:   {stats['max_absolute_difference']:.4f}")
    lines.append(f"  Changed pixels (>1%):      {stats['changed_pixels_percent']:.2f}%")
    lines.append("")

    # Heatmap
    if "heatmap_path" in results:
        lines.append("DIFFERENCE HEATMAP:")
        lines.append("-" * 80)
        lines.append(f"  Saved to: {results['heatmap_path']}")
        lines.append("")

    # Interpretation
    lines.append("=" * 80)
    lines.append("INTERPRETATION")
    lines.append("-" * 80)
    lines.append(results["interpretation"])
    lines.append("")

    lines.append("=" * 80)

    return "\n".join(lines)
