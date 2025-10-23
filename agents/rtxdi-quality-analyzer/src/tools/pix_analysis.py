"""
PIX capture analysis tool for RTXDI debugging

Analyzes PIX .wpix captures to identify RTXDI-specific bottlenecks:
- Light grid utilization and spatial coverage
- Temporal accumulation overhead
- Reservoir buffer usage
- Ray tracing dispatch costs
- BLAS/TLAS rebuild times
"""

from pathlib import Path
from typing import Dict, List, Optional, Any
import os

from ..utils.pix_parser import (
    parse_pix_capture,
    parse_buffer_dump,
    analyze_light_grid_coverage,
    identify_bottleneck,
    export_pix_events,
    RTXDIMetrics
)


async def analyze_pix_capture(
    capture_path: Optional[str] = None,
    captures_dir: Optional[str] = None,
    project_root: Optional[str] = None,
    analyze_buffers: bool = True
) -> Dict[str, Any]:
    """
    Analyze a PIX capture for RTXDI performance bottlenecks

    Args:
        capture_path: Path to specific .wpix capture file
        captures_dir: Directory containing PIX captures (auto-detect latest)
        project_root: Project root directory (defaults to env var)
        analyze_buffers: Also analyze buffer dumps if available

    Returns:
        Dictionary with RTXDI metrics, bottleneck analysis, and recommendations
    """
    # Get project root from environment if not provided
    if not project_root:
        project_root = os.getenv("PROJECT_ROOT", "/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean")

    # Determine capture path
    if capture_path:
        capture = Path(capture_path)
    elif captures_dir:
        capture = find_latest_capture(Path(captures_dir))
    else:
        # Default to project's PIX/Captures directory
        captures_path = Path(project_root) / "PIX" / "Captures"
        capture = find_latest_capture(captures_path)

    if not capture or not capture.exists():
        return {
            "error": "No PIX capture file found",
            "searched_paths": [
                capture_path,
                captures_dir,
                str(Path(project_root) / "PIX" / "Captures")
            ]
        }

    # Parse PIX capture
    metrics = parse_pix_capture(capture)

    if not metrics:
        return {
            "error": f"Failed to parse PIX capture: {capture}",
            "capture_path": str(capture)
        }

    # Build results
    results = {
        "capture_file": str(capture),
        "rtxdi_metrics": {
            "light_grid": {
                "size": metrics.light_grid_size,
                "coverage_percent": metrics.light_grid_coverage,
                "avg_lights_per_cell": metrics.lights_per_cell_avg,
                "max_lights_per_cell": metrics.lights_per_cell_max
            },
            "timings": {
                "temporal_accumulation": metrics.temporal_accumulation_time,
                "ray_dispatch": metrics.ray_dispatch_time,
                "blas_rebuild": metrics.blas_rebuild_time,
                "tlas_rebuild": metrics.tlas_rebuild_time,
                "total_rtxdi_overhead": metrics.total_rtxdi_overhead
            },
            "buffers": {
                "reservoir_size_mb": metrics.reservoir_buffer_size / (1024 * 1024)
            }
        },
        "bottleneck": None,
        "recommendations": []
    }

    # Identify bottleneck
    bottleneck_name, bottleneck_time, recommendation = identify_bottleneck(metrics)
    results["bottleneck"] = {
        "name": bottleneck_name,
        "time_ms": bottleneck_time,
        "percent_of_total": (bottleneck_time / metrics.total_rtxdi_overhead) * 100,
        "recommendation": recommendation
    }

    # Generate recommendations
    results["recommendations"] = generate_pix_recommendations(metrics, bottleneck_name)

    # Analyze buffer dumps if requested
    if analyze_buffers:
        buffer_dumps_dir = Path(project_root) / "PIX" / "buffer_dumps"
        if buffer_dumps_dir.exists():
            buffer_analysis = analyze_buffer_dumps(buffer_dumps_dir)
            results["buffer_analysis"] = buffer_analysis

    return results


def find_latest_capture(captures_dir: Path) -> Optional[Path]:
    """
    Find the most recent .wpix capture file in a directory

    Args:
        captures_dir: Directory to search

    Returns:
        Path to latest capture or None if not found
    """
    if not captures_dir.exists():
        return None

    wpix_files = list(captures_dir.glob("*.wpix"))
    if not wpix_files:
        return None

    # Sort by modification time, newest first
    wpix_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return wpix_files[0]


def analyze_buffer_dumps(buffer_dumps_dir: Path) -> Dict[str, Any]:
    """
    Analyze buffer dumps associated with PIX captures

    Args:
        buffer_dumps_dir: Directory containing buffer dumps

    Returns:
        Dictionary with buffer analysis results
    """
    results = {
        "buffers_found": [],
        "total_size_mb": 0.0,
        "issues": []
    }

    # Look for common buffer dumps
    buffer_names = [
        "g_particles.bin",
        "g_lights.bin",
        "g_currentReservoirs.bin",
        "g_prevReservoirs.bin",
        "g_rtLighting.bin"
    ]

    for buffer_name in buffer_names:
        buffer_path = buffer_dumps_dir / buffer_name
        if buffer_path.exists():
            info = parse_buffer_dump(buffer_path)
            if info:
                results["buffers_found"].append({
                    "name": info.buffer_name,
                    "size_mb": info.size_bytes / (1024 * 1024),
                    "element_count": info.element_count,
                    "element_size": info.element_size
                })
                results["total_size_mb"] += info.size_bytes / (1024 * 1024)

    # Check for issues
    for buffer in results["buffers_found"]:
        # Reservoir buffers are very large (126 MB each)
        if "Reservoirs" in buffer["name"] and buffer["size_mb"] > 100:
            results["issues"].append(
                f"⚠️ {buffer['name']} is {buffer['size_mb']:.1f} MB. "
                "Consider reducing resolution or using R16G16B16A16_FLOAT instead of R32G32B32A32_FLOAT"
            )

    return results


def generate_pix_recommendations(metrics: RTXDIMetrics, bottleneck: str) -> List[str]:
    """
    Generate optimization recommendations based on PIX analysis

    Args:
        metrics: RTXDIMetrics from PIX capture
        bottleneck: Primary bottleneck identified

    Returns:
        List of recommendation strings with file:line references
    """
    recommendations = []

    # BLAS rebuild bottleneck
    if bottleneck == "BLAS rebuild" and metrics.blas_rebuild_time > 2.0:
        recommendations.append(
            f"🔴 BLAS rebuild is primary bottleneck ({metrics.blas_rebuild_time:.1f}ms)"
        )
        recommendations.append(
            "   Fix: Implement BLAS updates instead of full rebuild"
        )
        recommendations.append(
            "   File: src/lighting/RTLightingSystem_RayQuery.cpp:456"
        )
        recommendations.append(
            "   Expected impact: +25% FPS (reduce from 2.1ms to ~0.5ms)"
        )
        recommendations.append(
            "   Estimated time: 4-6 hours"
        )

    # TLAS rebuild bottleneck
    elif bottleneck == "TLAS rebuild" and metrics.tlas_rebuild_time > 0.8:
        recommendations.append(
            f"🔴 TLAS rebuild is primary bottleneck ({metrics.tlas_rebuild_time:.1f}ms)"
        )
        recommendations.append(
            "   Fix: Implement instance culling for distant particles"
        )
        recommendations.append(
            "   File: src/lighting/RTLightingSystem_RayQuery.cpp:512"
        )
        recommendations.append(
            "   Expected impact: +15% FPS"
        )
        recommendations.append(
            "   Estimated time: 6-8 hours"
        )

    # Ray dispatch bottleneck
    elif bottleneck == "Ray dispatch" and metrics.ray_dispatch_time > 3.0:
        recommendations.append(
            f"🔴 Ray dispatch is primary bottleneck ({metrics.ray_dispatch_time:.1f}ms)"
        )
        recommendations.append(
            "   Fix: Implement adaptive ray sampling (reduce rays for low-detail regions)"
        )
        recommendations.append(
            "   File: shaders/rtxdi/rtxdi_raygen.hlsl:145"
        )
        recommendations.append(
            "   Expected impact: +20% FPS"
        )
        recommendations.append(
            "   Estimated time: 8-10 hours (requires spatial heuristics)"
        )

    # Temporal accumulation overhead
    elif bottleneck == "Temporal accumulation" and metrics.temporal_accumulation_time > 1.5:
        recommendations.append(
            f"🔴 Temporal accumulation is primary bottleneck ({metrics.temporal_accumulation_time:.1f}ms)"
        )
        recommendations.append(
            "   Fix: Optimize ping-pong buffer transitions or reduce sample count"
        )
        recommendations.append(
            "   File: src/lighting/RTXDILightingSystem.cpp:789"
        )
        recommendations.append(
            "   Expected impact: +10% FPS"
        )
        recommendations.append(
            "   Estimated time: 2-3 hours"
        )

    # Light grid coverage issues
    if metrics.light_grid_coverage < 50.0:
        recommendations.append(
            f"⚠️ Light grid coverage is low ({metrics.light_grid_coverage:.1f}%). "
            "Many cells are empty, wasting grid structure overhead"
        )
        recommendations.append(
            "   Consider: Adaptive grid sizing or hierarchical grid structure"
        )

    if metrics.lights_per_cell_max > 10:
        recommendations.append(
            f"⚠️ Some cells have {metrics.lights_per_cell_max} lights. "
            "This can cause sampling bias"
        )
        recommendations.append(
            "   Consider: Larger grid size or importance-based cell assignment"
        )

    # Overall RTXDI overhead
    if metrics.total_rtxdi_overhead > 8.0:
        recommendations.append(
            f"⚠️ Total RTXDI overhead is {metrics.total_rtxdi_overhead:.1f}ms. "
            "This is significant for real-time rendering"
        )
        recommendations.append(
            "   Consider: Spatial reuse (RTXDI M6) to amortize cost across pixels"
        )

    if not recommendations:
        recommendations.append("✅ RTXDI pipeline is well-optimized")

    return recommendations


def format_pix_report(results: Dict[str, Any]) -> str:
    """
    Format PIX analysis results as a readable report

    Args:
        results: Results from analyze_pix_capture()

    Returns:
        Formatted text report
    """
    lines = []
    lines.append("=" * 80)
    lines.append("PIX CAPTURE ANALYSIS - RTXDI BOTTLENECK REPORT")
    lines.append("=" * 80)
    lines.append(f"\nCapture File: {results['capture_file']}")
    lines.append("")

    # RTXDI Metrics
    lines.append("RTXDI METRICS")
    lines.append("-" * 80)

    metrics = results["rtxdi_metrics"]

    lines.append("\nLight Grid:")
    grid = metrics["light_grid"]
    lines.append(f"  Size: {grid['size'][0]}×{grid['size'][1]}×{grid['size'][2]} cells")
    lines.append(f"  Coverage: {grid['coverage_percent']:.1f}%")
    lines.append(f"  Avg lights/cell: {grid['avg_lights_per_cell']:.1f}")
    lines.append(f"  Max lights/cell: {grid['max_lights_per_cell']}")

    lines.append("\nGPU Timings:")
    timings = metrics["timings"]
    for component, time_ms in timings.items():
        if component != "total_rtxdi_overhead":
            lines.append(f"  {component}: {time_ms:.2f}ms")
    lines.append(f"  {'─' * 40}")
    lines.append(f"  TOTAL RTXDI OVERHEAD: {timings['total_rtxdi_overhead']:.2f}ms")

    lines.append("\nBuffer Memory:")
    buffers = metrics["buffers"]
    lines.append(f"  Reservoir buffers: {buffers['reservoir_size_mb']:.1f} MB")

    # Bottleneck
    if results["bottleneck"]:
        lines.append("\n" + "=" * 80)
        lines.append("PRIMARY BOTTLENECK")
        lines.append("-" * 80)
        bn = results["bottleneck"]
        lines.append(f"\n🔴 {bn['name']}: {bn['time_ms']:.2f}ms ({bn['percent_of_total']:.1f}% of total overhead)")
        lines.append(f"\n{bn['recommendation']}")

    # Recommendations
    if results["recommendations"]:
        lines.append("\n" + "=" * 80)
        lines.append("OPTIMIZATION RECOMMENDATIONS")
        lines.append("-" * 80)
        for rec in results["recommendations"]:
            lines.append(f"\n{rec}")

    # Buffer analysis
    if "buffer_analysis" in results:
        ba = results["buffer_analysis"]
        lines.append("\n" + "=" * 80)
        lines.append("BUFFER DUMP ANALYSIS")
        lines.append("-" * 80)
        lines.append(f"\nTotal buffer size: {ba['total_size_mb']:.1f} MB")
        lines.append(f"Buffers found: {len(ba['buffers_found'])}")

        for buffer in ba["buffers_found"]:
            lines.append(f"\n  {buffer['name']}:")
            lines.append(f"    Size: {buffer['size_mb']:.1f} MB")
            lines.append(f"    Elements: {buffer['element_count']:,} × {buffer['element_size']} bytes")

        if ba["issues"]:
            lines.append("\nIssues:")
            for issue in ba["issues"]:
                lines.append(f"  {issue}")

    lines.append("\n" + "=" * 80)

    return "\n".join(lines)
