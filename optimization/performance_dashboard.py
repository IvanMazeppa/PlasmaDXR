#!/usr/bin/env python3
"""
Real-Time Performance Monitoring Dashboard for PlasmaDX-Clean
Tracks FPS, GPU metrics, and optimization progress
"""

import json
import time
import re
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class FrameMetrics:
    """Per-frame performance metrics"""
    frame_number: int
    fps: float
    frame_time_ms: float
    gpu_time_ms: float
    blas_rebuild_ms: float
    froxel_pass_ms: float
    lighting_pass_ms: float
    particle_count: int
    light_count: int
    timestamp: float = field(default_factory=time.time)

@dataclass
class OptimizationMilestone:
    """Track optimization milestones"""
    name: str
    target_fps: float
    achieved_fps: Optional[float] = None
    completion_time: Optional[float] = None
    status: str = "pending"  # pending, in_progress, completed, failed

class PerformanceDashboard:
    """Real-time performance monitoring and visualization"""

    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.log_dir = self.project_root / "logs"
        self.metrics_history: List[FrameMetrics] = []
        self.baselines: Dict[str, float] = {}
        self.milestones: List[OptimizationMilestone] = []

        # Initialize baselines from CLAUDE.md targets
        self._initialize_baselines()

    def _initialize_baselines(self):
        """Load performance baselines from project documentation"""
        self.baselines = {
            "raster_only": 245.0,
            "rt_lighting": 165.0,
            "rt_shadows": 142.0,
            "dlss_performance": 190.0,
            "target_with_pinn": 280.0
        }

        # Define optimization milestones
        self.milestones = [
            OptimizationMilestone("BLAS Update Implementation", 178.0),  # +25% from 142
            OptimizationMilestone("Particle LOD Culling", 213.0),        # +50% from 142
            OptimizationMilestone("PINN ML Physics", 280.0),             # Target
            OptimizationMilestone("Froxel R16 Optimization", 150.0),     # +8 FPS
        ]

    def parse_log_file(self, log_path: Path) -> Optional[FrameMetrics]:
        """Extract performance metrics from application log"""
        if not log_path.exists():
            return None

        try:
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Example log parsing (adjust regex to actual log format)
            fps_match = re.search(r'FPS:\s*([\d.]+)', content)
            frame_time_match = re.search(r'Frame Time:\s*([\d.]+)\s*ms', content)
            blas_match = re.search(r'BLAS Rebuild:\s*([\d.]+)\s*ms', content)

            if fps_match and frame_time_match:
                return FrameMetrics(
                    frame_number=0,
                    fps=float(fps_match.group(1)),
                    frame_time_ms=float(frame_time_match.group(1)),
                    gpu_time_ms=float(frame_time_match.group(1)),
                    blas_rebuild_ms=float(blas_match.group(1)) if blas_match else 2.1,
                    froxel_pass_ms=1.75,  # Average from docs
                    lighting_pass_ms=0.0,
                    particle_count=100000,
                    light_count=13
                )
        except Exception as e:
            print(f"⚠️  Log parsing error: {e}")

        return None

    def analyze_optimization_report(self, report_path: Path) -> Dict:
        """Analyze multi-agent optimization results"""
        if not report_path.exists():
            return {}

        with open(report_path, 'r') as f:
            report = json.load(f)

        analysis = {
            "timestamp": datetime.fromtimestamp(report['timestamp']).isoformat(),
            "total_agents": report['total_agents'],
            "success_rate": report['successful_agents'] / report['total_agents'],
            "token_efficiency": report['token_efficiency'],
            "projected_improvements": {}
        }

        for opt in report['optimizations']:
            domain = opt['domain']
            for metric in opt['metrics']:
                if metric['target'] is not None and metric['baseline'] is not None:
                    improvement_pct = ((metric['target'] - metric['baseline']) / metric['baseline']) * 100
                    analysis['projected_improvements'][metric['name']] = {
                        "current": metric['value'],
                        "target": metric['target'],
                        "improvement_pct": improvement_pct,
                        "unit": metric['unit']
                    }

        return analysis

    def calculate_fps_delta(self, current_fps: float, baseline_key: str = "rt_shadows") -> Dict:
        """Calculate FPS improvement vs baseline"""
        baseline = self.baselines.get(baseline_key, 142.0)

        return {
            "current_fps": current_fps,
            "baseline_fps": baseline,
            "delta_fps": current_fps - baseline,
            "improvement_pct": ((current_fps - baseline) / baseline) * 100,
            "target_fps": self.baselines.get("target_with_pinn", 280.0),
            "progress_to_target": ((current_fps - baseline) / (self.baselines['target_with_pinn'] - baseline)) * 100
        }

    def generate_dashboard_text(self, latest_metrics: Optional[FrameMetrics] = None) -> str:
        """Generate ASCII dashboard for terminal output"""
        lines = []
        lines.append("=" * 100)
        lines.append("🎮 PlasmaDX-Clean Performance Dashboard".center(100))
        lines.append("=" * 100)

        # Current Performance
        if latest_metrics:
            lines.append("\n📊 Current Performance:")
            lines.append(f"   FPS: {latest_metrics.fps:.1f} | Frame Time: {latest_metrics.frame_time_ms:.2f}ms")
            lines.append(f"   Particles: {latest_metrics.particle_count:,} | Lights: {latest_metrics.light_count}")
            lines.append(f"   BLAS Rebuild: {latest_metrics.blas_rebuild_ms:.2f}ms | Froxel: {latest_metrics.froxel_pass_ms:.2f}ms")

            delta = self.calculate_fps_delta(latest_metrics.fps)
            symbol = "📈" if delta['delta_fps'] > 0 else "📉" if delta['delta_fps'] < 0 else "➖"
            lines.append(f"\n   {symbol} vs Baseline: {delta['delta_fps']:+.1f} FPS ({delta['improvement_pct']:+.1f}%)")
            lines.append(f"   🎯 Progress to Target (280 FPS): {delta['progress_to_target']:.1f}%")

        # Optimization Milestones
        lines.append("\n🎯 Optimization Milestones:")
        for milestone in self.milestones:
            status_icon = {
                "pending": "⏳",
                "in_progress": "🔄",
                "completed": "✅",
                "failed": "❌"
            }.get(milestone.status, "❓")

            achieved_text = f"{milestone.achieved_fps:.1f} FPS" if milestone.achieved_fps else "Not Started"
            lines.append(f"   {status_icon} {milestone.name}: Target {milestone.target_fps:.0f} FPS | {achieved_text}")

        # Baseline Comparisons
        lines.append("\n📈 Performance Baselines:")
        if latest_metrics:
            current = latest_metrics.fps
            for name, baseline_fps in self.baselines.items():
                delta_pct = ((current - baseline_fps) / baseline_fps) * 100 if baseline_fps > 0 else 0
                symbol = "✅" if current >= baseline_fps else "⚠️"
                lines.append(f"   {symbol} {name.replace('_', ' ').title()}: {baseline_fps:.0f} FPS | Current: {current:.1f} ({delta_pct:+.1f}%)")

        lines.append("\n" + "=" * 100)

        return "\n".join(lines)

    def monitor_realtime(self, duration_seconds: int = 60, interval_seconds: int = 5):
        """Monitor performance in real-time"""
        print(f"🔍 Starting real-time monitoring for {duration_seconds}s...")

        start_time = time.time()
        while time.time() - start_time < duration_seconds:
            # Find latest log file
            log_files = sorted(self.log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)

            if log_files:
                metrics = self.parse_log_file(log_files[0])
                if metrics:
                    self.metrics_history.append(metrics)
                    print("\033[2J\033[H")  # Clear screen
                    print(self.generate_dashboard_text(metrics))

            time.sleep(interval_seconds)

    def export_metrics(self, output_path: str):
        """Export metrics history to JSON"""
        data = {
            "baselines": self.baselines,
            "milestones": [
                {
                    "name": m.name,
                    "target_fps": m.target_fps,
                    "achieved_fps": m.achieved_fps,
                    "status": m.status
                }
                for m in self.milestones
            ],
            "metrics_history": [
                {
                    "frame": m.frame_number,
                    "fps": m.fps,
                    "frame_time_ms": m.frame_time_ms,
                    "timestamp": m.timestamp
                }
                for m in self.metrics_history
            ]
        }

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"📊 Metrics exported to {output_path}")

def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python performance_dashboard.py <project_root> [--monitor] [--analyze-report <path>]")
        sys.exit(1)

    project_root = sys.argv[1]
    dashboard = PerformanceDashboard(project_root)

    if "--monitor" in sys.argv:
        dashboard.monitor_realtime(duration_seconds=60, interval_seconds=5)
    elif "--analyze-report" in sys.argv:
        report_idx = sys.argv.index("--analyze-report") + 1
        if report_idx < len(sys.argv):
            report_path = Path(sys.argv[report_idx])
            analysis = dashboard.analyze_optimization_report(report_path)
            print(json.dumps(analysis, indent=2))
    else:
        # Static dashboard from latest log
        log_dir = Path(project_root) / "logs"
        if log_dir.exists():
            log_files = sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
            if log_files:
                metrics = dashboard.parse_log_file(log_files[0])
                print(dashboard.generate_dashboard_text(metrics))

if __name__ == "__main__":
    main()
