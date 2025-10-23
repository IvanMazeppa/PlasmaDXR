"""
Performance comparison tool for RTXDI analysis

Compares FPS, frame times, and GPU metrics between:
- Legacy renderer
- RTXDI M4 (weighted sampling)
- RTXDI M5 (temporal accumulation)
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import os

from ..utils.metrics_parser import (
    parse_performance_metrics,
    compare_metrics,
    PerformanceMetrics
)


async def compare_performance(
    legacy_log: Optional[str] = None,
    rtxdi_m4_log: Optional[str] = None,
    rtxdi_m5_log: Optional[str] = None,
    logs_dir: Optional[str] = None,
    project_root: Optional[str] = None
) -> Dict[str, Any]:
    """
    Compare performance metrics between renderer modes

    Args:
        legacy_log: Path to legacy renderer log file
        rtxdi_m4_log: Path to RTXDI M4 log file
        rtxdi_m5_log: Path to RTXDI M5 log file
        logs_dir: Directory containing log files (auto-detect latest)
        project_root: Project root directory (defaults to env var)

    Returns:
        Dictionary with comparison results, bottleneck analysis, and recommendations
    """
    # Get project root from environment if not provided
    if not project_root:
        project_root = os.getenv("PROJECT_ROOT", "/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean")

    # Convert logs_dir to Path if provided
    logs_path = Path(logs_dir) if logs_dir else Path(project_root) / "logs"

    # Convert log paths to Path objects
    legacy_path = Path(legacy_log) if legacy_log else None
    rtxdi_m4_path = Path(rtxdi_m4_log) if rtxdi_m4_log else None
    rtxdi_m5_path = Path(rtxdi_m5_log) if rtxdi_m5_log else None

    # Parse metrics
    metrics = parse_performance_metrics(
        legacy_log=legacy_path,
        rtxdi_m4_log=rtxdi_m4_path,
        rtxdi_m5_log=rtxdi_m5_path,
        logs_dir=logs_path
    )

    # Build comparison results
    results = {
        "metrics": {},
        "comparisons": [],
        "fastest": None,
        "recommendations": []
    }

    # Add metrics for each mode
    for mode, data in metrics.items():
        if data:
            results["metrics"][mode] = {
                "fps_avg": data.fps_avg,
                "fps_min": data.fps_min,
                "fps_max": data.fps_max,
                "frame_time_avg": data.frame_time_avg,
                "frame_time_p95": data.frame_time_p95,
                "frame_time_p99": data.frame_time_p99,
                "gpu_timings": data.gpu_timings,
                "particle_count": data.particle_count,
                "light_count": data.light_count,
                "resolution": data.resolution
            }

    # Compare RTXDI modes against legacy baseline
    if "legacy" in metrics and metrics["legacy"]:
        baseline = metrics["legacy"]

        if "rtxdi_m4" in metrics and metrics["rtxdi_m4"]:
            comparison = compare_metrics(baseline, metrics["rtxdi_m4"])
            results["comparisons"].append({
                "name": "Legacy vs RTXDI M4",
                **comparison
            })

        if "rtxdi_m5" in metrics and metrics["rtxdi_m5"]:
            comparison = compare_metrics(baseline, metrics["rtxdi_m5"])
            results["comparisons"].append({
                "name": "Legacy vs RTXDI M5",
                **comparison
            })

    # Compare M4 vs M5 directly
    if "rtxdi_m4" in metrics and "rtxdi_m5" in metrics:
        if metrics["rtxdi_m4"] and metrics["rtxdi_m5"]:
            comparison = compare_metrics(metrics["rtxdi_m4"], metrics["rtxdi_m5"])
            results["comparisons"].append({
                "name": "RTXDI M4 vs M5",
                **comparison
            })

    # Identify fastest renderer
    fastest_fps = 0.0
    fastest_mode = None
    for mode, data in metrics.items():
        if data and data.fps_avg > fastest_fps:
            fastest_fps = data.fps_avg
            fastest_mode = mode

    results["fastest"] = {
        "mode": fastest_mode,
        "fps": fastest_fps
    }

    # Generate recommendations
    results["recommendations"] = generate_recommendations(metrics)

    return results


def generate_recommendations(metrics: Dict[str, Optional[PerformanceMetrics]]) -> List[str]:
    """
    Generate optimization recommendations based on metrics comparison

    Args:
        metrics: Dictionary of performance metrics by mode

    Returns:
        List of recommendation strings
    """
    recommendations = []

    # Check if RTXDI is slower than legacy
    if "legacy" in metrics and "rtxdi_m4" in metrics:
        legacy = metrics["legacy"]
        rtxdi = metrics["rtxdi_m4"]

        if legacy and rtxdi and rtxdi.fps_avg < legacy.fps_avg:
            recommendations.append(
                "⚠️ RTXDI M4 is slower than legacy renderer. Primary causes likely include:"
            )

            # Check GPU timings
            if "blas_rebuild" in rtxdi.gpu_timings:
                blas_time = rtxdi.gpu_timings["blas_rebuild"]
                if blas_time > 2.0:
                    recommendations.append(
                        f"  - BLAS rebuild is expensive ({blas_time:.1f}ms). "
                        "Consider BLAS updates instead of full rebuild (src/lighting/RTLightingSystem_RayQuery.cpp:456)"
                    )

            if "rt_lighting" in rtxdi.gpu_timings:
                rt_time = rtxdi.gpu_timings["rt_lighting"]
                if rt_time > 3.0:
                    recommendations.append(
                        f"  - RT lighting is expensive ({rt_time:.1f}ms). "
                        "Consider adaptive ray sampling or spatial reuse (RTXDI M6)"
                    )

            recommendations.append(
                "  - Profile with PIX to identify exact bottleneck. Use: @pix-debugging-agent"
            )

    # Check M5 temporal accumulation overhead
    if "rtxdi_m4" in metrics and "rtxdi_m5" in metrics:
        m4 = metrics["rtxdi_m4"]
        m5 = metrics["rtxdi_m5"]

        if m4 and m5 and m5.fps_avg < m4.fps_avg:
            fps_loss = m4.fps_avg - m5.fps_avg
            if fps_loss > 5.0:
                recommendations.append(
                    f"⚠️ RTXDI M5 temporal accumulation adds {fps_loss:.1f} FPS overhead. "
                    "Consider reducing accumulation samples or optimizing ping-pong buffer transitions "
                    "(src/lighting/RTXDILightingSystem.cpp:789)"
                )

    # Target performance recommendations
    target_fps = 60.0
    target_particle_count = 25000

    for mode, data in metrics.items():
        if data and data.fps_avg < target_fps:
            recommendations.append(
                f"🎯 {mode} is below target {target_fps} FPS @ {target_particle_count}K particles. "
                f"Current: {data.fps_avg:.1f} FPS @ {data.particle_count} particles"
            )

    if not recommendations:
        recommendations.append("✅ All renderers performing within expected parameters")

    return recommendations


def format_comparison_report(results: Dict[str, Any]) -> str:
    """
    Format comparison results as a readable report

    Args:
        results: Results from compare_performance()

    Returns:
        Formatted text report
    """
    lines = []
    lines.append("=" * 80)
    lines.append("RTXDI PERFORMANCE COMPARISON REPORT")
    lines.append("=" * 80)
    lines.append("")

    # Metrics summary
    lines.append("METRICS SUMMARY")
    lines.append("-" * 80)

    if not results["metrics"]:
        lines.append("\nNo performance metrics found.")
        lines.append("\nTo use this tool, provide log file paths:")
        lines.append("  - legacy_log: Path to legacy renderer log")
        lines.append("  - rtxdi_m4_log: Path to RTXDI M4 log")
        lines.append("  - rtxdi_m5_log: Path to RTXDI M5 log")
        lines.append("\nExample: compare_performance(legacy_log='logs/legacy.log')")

    for mode, data in results["metrics"].items():
        lines.append(f"\n{mode.upper()}:")
        lines.append(f"  FPS: {data['fps_avg']:.1f} (min: {data['fps_min']:.1f}, max: {data['fps_max']:.1f})")
        lines.append(f"  Frame Time: {data['frame_time_avg']:.2f}ms (p95: {data['frame_time_p95']:.2f}ms, p99: {data['frame_time_p99']:.2f}ms)")
        lines.append(f"  Particles: {data['particle_count']:,} | Lights: {data['light_count']}")
        lines.append(f"  Resolution: {data['resolution']}")

        if data['gpu_timings']:
            lines.append("  GPU Timings:")
            for component, time_ms in data['gpu_timings'].items():
                lines.append(f"    - {component}: {time_ms:.2f}ms")

    # Comparisons
    if results["comparisons"]:
        lines.append("\n" + "=" * 80)
        lines.append("COMPARISONS")
        lines.append("-" * 80)
        for comp in results["comparisons"]:
            lines.append(f"\n{comp['name']}:")
            lines.append(f"  {comp['summary']}")
            lines.append(f"  FPS: {comp['fps']['baseline']:.1f} → {comp['fps']['comparison']:.1f} "
                        f"({comp['fps']['delta']:+.1f}, {comp['fps']['percent_change']:+.1f}%)")
            lines.append(f"  Frame Time: {comp['frame_time']['baseline']:.2f}ms → {comp['frame_time']['comparison']:.2f}ms "
                        f"({comp['frame_time']['delta']:+.2f}ms, {comp['frame_time']['percent_change']:+.1f}%)")

    # Fastest renderer
    if results["fastest"] and results["fastest"]["mode"]:
        lines.append("\n" + "=" * 80)
        lines.append(f"FASTEST: {results['fastest']['mode'].upper()} @ {results['fastest']['fps']:.1f} FPS")
        lines.append("=" * 80)

    # Recommendations
    if results["recommendations"]:
        lines.append("\n" + "=" * 80)
        lines.append("RECOMMENDATIONS")
        lines.append("-" * 80)
        for rec in results["recommendations"]:
            lines.append(f"\n{rec}")

    lines.append("\n" + "=" * 80)

    return "\n".join(lines)
