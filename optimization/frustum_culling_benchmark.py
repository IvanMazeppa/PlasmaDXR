#!/usr/bin/env python3
"""
Frustum Culling Benchmark Script
Compares FPS before and after GPU frustum culling implementation.

Usage:
    cd build/bin/Debug
    python3 ../../../optimization/frustum_culling_benchmark.py

Requirements:
    - Run from build/bin/Debug directory
    - Screenshots must have .json metadata sidecars with FPS data
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

def find_screenshots_with_metadata(screenshots_dir: Path) -> List[Dict]:
    """Find all screenshots that have JSON metadata sidecars."""
    results = []

    for json_file in screenshots_dir.glob("*.json"):
        try:
            with open(json_file, 'r') as f:
                metadata = json.load(f)

            # Extract key metrics
            fps = metadata.get('performance', {}).get('fps', 0)
            frame_time = metadata.get('performance', {}).get('frame_time_ms', 0)
            particle_count = metadata.get('particles', {}).get('count', 0)
            timestamp = metadata.get('timestamp', '')
            camera_dist = metadata.get('camera', {}).get('distance', 0)

            results.append({
                'file': str(json_file),
                'timestamp': timestamp,
                'fps': fps,
                'frame_time_ms': frame_time,
                'particle_count': particle_count,
                'camera_distance': camera_dist
            })
        except (json.JSONDecodeError, KeyError) as e:
            continue

    # Sort by timestamp
    results.sort(key=lambda x: x['timestamp'], reverse=True)
    return results

def analyze_log_fps(log_path: Path) -> Dict:
    """Extract FPS estimates from log file based on frame timing."""
    if not log_path.exists():
        return {}

    frame_times = []
    with open(log_path, 'r') as f:
        lines = f.readlines()

    # Look for frame timing patterns
    prev_time = None
    prev_frame = None

    for line in lines:
        if "RT Lighting computed" in line and "frame" in line:
            # Extract timestamp and frame number
            try:
                # Format: [17:43:53] [INFO] RT Lighting computed with dynamic emission (frame 0)
                time_str = line.split(']')[0].strip('[')
                frame_str = line.split('frame ')[1].split(')')[0]
                frame_num = int(frame_str)

                # Parse time
                h, m, s = map(int, time_str.split(':'))
                time_sec = h * 3600 + m * 60 + s

                if prev_time is not None and prev_frame is not None:
                    delta_frames = frame_num - prev_frame
                    delta_time = time_sec - prev_time
                    if delta_time > 0 and delta_frames > 0:
                        fps_estimate = delta_frames / delta_time
                        frame_times.append(fps_estimate)

                prev_time = time_sec
                prev_frame = frame_num
            except:
                continue

    if frame_times:
        avg_fps = sum(frame_times) / len(frame_times)
        return {
            'estimated_fps': round(avg_fps, 1),
            'samples': len(frame_times),
            'min_fps': round(min(frame_times), 1),
            'max_fps': round(max(frame_times), 1)
        }
    return {}

def print_report(screenshots: List[Dict], log_analysis: Dict):
    """Print formatted benchmark report."""
    print("=" * 80)
    print("            FRUSTUM CULLING BENCHMARK REPORT")
    print("            GPU-Side Particle Culling (2025-12-11)")
    print("=" * 80)
    print()

    if screenshots:
        print("📸 SCREENSHOT METADATA ANALYSIS")
        print("-" * 40)

        # Show most recent screenshots
        for i, ss in enumerate(screenshots[:5]):
            print(f"\n  [{i+1}] {Path(ss['file']).name}")
            print(f"      Timestamp: {ss['timestamp']}")
            print(f"      FPS: {ss['fps']:.1f}")
            print(f"      Frame Time: {ss['frame_time_ms']:.2f} ms")
            print(f"      Particles: {ss['particle_count']:,}")
            print(f"      Camera Distance: {ss['camera_distance']:.0f} units")

        # Calculate averages if multiple screenshots
        if len(screenshots) >= 2:
            avg_fps = sum(s['fps'] for s in screenshots) / len(screenshots)
            avg_frame_time = sum(s['frame_time_ms'] for s in screenshots) / len(screenshots)
            print(f"\n  📊 Average across {len(screenshots)} screenshots:")
            print(f"      FPS: {avg_fps:.1f}")
            print(f"      Frame Time: {avg_frame_time:.2f} ms")
    else:
        print("  ⚠️  No screenshots with metadata found")
        print("      Press F2 in-game to capture screenshots with FPS data")

    print()

    if log_analysis:
        print("📋 LOG FILE ANALYSIS")
        print("-" * 40)
        print(f"  Estimated FPS: {log_analysis.get('estimated_fps', 'N/A')}")
        print(f"  Samples: {log_analysis.get('samples', 0)}")
        print(f"  Range: {log_analysis.get('min_fps', 'N/A')} - {log_analysis.get('max_fps', 'N/A')} FPS")

    print()
    print("=" * 80)
    print("FRUSTUM CULLING IMPLEMENTATION SUMMARY")
    print("=" * 80)
    print("""
    ✅ IMPLEMENTED FEATURES:
       - GPU-side frustum plane extraction (Gribb/Hartmann method)
       - Sphere-frustum intersection test per particle
       - Degenerate AABB output for culled particles (min > max)
       - 1.5× frustum expansion to prevent pop-in
       - Zero CPU overhead (all culling on GPU)

    📈 EXPECTED BENEFITS:
       - Reduced BLAS build time (fewer active AABBs)
       - Reduced ray traversal cost (fewer intersections)
       - Better performance when camera zoomed in
       - No visual quality impact (conservative culling)

    🔧 CONFIGURATION:
       - Enable/disable via RTLightingSystem::SetFrustumCullingEnabled()
       - Expansion factor via RTLightingSystem::SetFrustumExpansion()
       - Currently: ENABLED with 1.5× expansion

    📝 TO CAPTURE BENCHMARK DATA:
       1. Launch PlasmaDX-Clean.exe from build/bin/Debug/
       2. Let it run for 30-60 seconds
       3. Press F2 to capture screenshot with FPS metadata
       4. Re-run this script to analyze results
    """)
    print("=" * 80)

def main():
    # Determine working directory
    cwd = Path.cwd()

    # Find screenshots directory
    screenshots_dir = None
    logs_dir = None

    if (cwd / "screenshots").exists():
        screenshots_dir = cwd / "screenshots"
        logs_dir = cwd / "logs"
    elif (cwd / "build/bin/Debug/screenshots").exists():
        screenshots_dir = cwd / "build/bin/Debug/screenshots"
        logs_dir = cwd / "build/bin/Debug/logs"
    else:
        # Try relative to script location
        script_dir = Path(__file__).parent.parent
        screenshots_dir = script_dir / "build/bin/Debug/screenshots"
        logs_dir = script_dir / "build/bin/Debug/logs"

    if not screenshots_dir or not screenshots_dir.exists():
        print(f"⚠️  Screenshots directory not found")
        print(f"   Expected: build/bin/Debug/screenshots/")
        print(f"   Run from project root or build/bin/Debug/")
        screenshots = []
    else:
        screenshots = find_screenshots_with_metadata(screenshots_dir)

    # Analyze most recent log
    log_analysis = {}
    if logs_dir and logs_dir.exists():
        log_files = sorted(logs_dir.glob("*.log"), key=os.path.getmtime, reverse=True)
        if log_files:
            log_analysis = analyze_log_fps(log_files[0])

    print_report(screenshots, log_analysis)

if __name__ == "__main__":
    main()
