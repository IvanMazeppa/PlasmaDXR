"""
Performance metrics parser for PlasmaDX-Clean

Parses log files, console output, and performance data to extract:
- FPS (frames per second)
- Frame times (milliseconds)
- GPU timings (BLAS/TLAS rebuild, shader dispatches)
- Memory usage
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime


@dataclass
class PerformanceMetrics:
    """Performance metrics for a single renderer configuration"""
    renderer_mode: str  # "legacy", "rtxdi_m4", "rtxdi_m5"
    fps_avg: float
    fps_min: float
    fps_max: float
    frame_time_avg: float  # milliseconds
    frame_time_p95: float  # 95th percentile
    frame_time_p99: float  # 99th percentile
    gpu_timings: Dict[str, float]  # Component -> time in ms
    particle_count: int
    light_count: int
    resolution: str
    timestamp: datetime


def parse_log_file(log_path: Path) -> Optional[PerformanceMetrics]:
    """
    Parse a PlasmaDX log file to extract performance metrics

    Args:
        log_path: Path to log file

    Returns:
        PerformanceMetrics object or None if parsing fails
    """
    if not log_path.exists():
        return None

    # TODO: Implement log file parsing
    # Expected log format from PlasmaDX-Clean:
    # [INFO] FPS: 120.5 (avg: 118.2, min: 95.3, max: 142.7)
    # [INFO] Frame time: 8.3ms (avg: 8.5ms, p95: 10.2ms, p99: 12.8ms)
    # [INFO] BLAS rebuild: 2.1ms
    # [INFO] TLAS rebuild: 0.8ms
    # [INFO] RT lighting: 3.2ms

    with open(log_path, 'r') as f:
        content = f.read()

    # Stub implementation - returns sample data
    return PerformanceMetrics(
        renderer_mode="unknown",
        fps_avg=120.0,
        fps_min=95.0,
        fps_max=142.0,
        frame_time_avg=8.3,
        frame_time_p95=10.2,
        frame_time_p99=12.8,
        gpu_timings={
            "blas_rebuild": 2.1,
            "tlas_rebuild": 0.8,
            "rt_lighting": 3.2
        },
        particle_count=10000,
        light_count=13,
        resolution="1920x1080",
        timestamp=datetime.now()
    )


def parse_performance_metrics(
    legacy_log: Optional[Path] = None,
    rtxdi_m4_log: Optional[Path] = None,
    rtxdi_m5_log: Optional[Path] = None,
    logs_dir: Optional[Path] = None
) -> Dict[str, Optional[PerformanceMetrics]]:
    """
    Parse performance metrics from multiple renderer modes

    Args:
        legacy_log: Path to legacy renderer log
        rtxdi_m4_log: Path to RTXDI M4 log
        rtxdi_m5_log: Path to RTXDI M5 log
        logs_dir: Directory to search for latest logs (if specific paths not provided)

    Returns:
        Dictionary mapping renderer mode to PerformanceMetrics
    """
    metrics = {}

    if legacy_log:
        result = parse_log_file(legacy_log)
        if result:
            result.renderer_mode = "legacy"
            metrics["legacy"] = result

    if rtxdi_m4_log:
        result = parse_log_file(rtxdi_m4_log)
        if result:
            result.renderer_mode = "rtxdi_m4"
            metrics["rtxdi_m4"] = result

    if rtxdi_m5_log:
        result = parse_log_file(rtxdi_m5_log)
        if result:
            result.renderer_mode = "rtxdi_m5"
            metrics["rtxdi_m5"] = result

    # If logs_dir provided, search for latest logs
    if logs_dir and logs_dir.exists():
        # TODO: Implement automatic log file detection
        # Look for patterns like:
        # - PlasmaDX-Clean_*_legacy.log
        # - PlasmaDX-Clean_*_rtxdi_m4.log
        # - PlasmaDX-Clean_*_rtxdi_m5.log
        pass

    return metrics


def compare_metrics(
    baseline: PerformanceMetrics,
    comparison: PerformanceMetrics
) -> Dict[str, Any]:
    """
    Compare two performance metric sets

    Args:
        baseline: Baseline metrics (e.g., legacy renderer)
        comparison: Comparison metrics (e.g., RTXDI M4)

    Returns:
        Dictionary with comparison results including deltas and percentages
    """
    fps_delta = comparison.fps_avg - baseline.fps_avg
    fps_percent = (fps_delta / baseline.fps_avg) * 100

    frame_time_delta = comparison.frame_time_avg - baseline.frame_time_avg
    frame_time_percent = (frame_time_delta / baseline.frame_time_avg) * 100

    return {
        "fps": {
            "baseline": baseline.fps_avg,
            "comparison": comparison.fps_avg,
            "delta": fps_delta,
            "percent_change": fps_percent,
            "faster": fps_delta > 0
        },
        "frame_time": {
            "baseline": baseline.frame_time_avg,
            "comparison": comparison.frame_time_avg,
            "delta": frame_time_delta,
            "percent_change": frame_time_percent,
            "faster": frame_time_delta < 0  # Lower is better
        },
        "summary": f"{comparison.renderer_mode} is {abs(fps_percent):.1f}% "
                   f"{'faster' if fps_delta > 0 else 'slower'} than {baseline.renderer_mode}"
    }
