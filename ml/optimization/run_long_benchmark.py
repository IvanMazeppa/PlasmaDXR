#!/usr/bin/env python3
"""
Long-Term Benchmark Runner for PlasmaDX-Clean

Runs a controlled benchmark with:
- Frame limit (e.g., 10-12K frames)
- Timeout limit (e.g., 25 minutes)
- Whichever comes first

Loads config from JSON file and converts to command-line args.
"""

import subprocess
import json
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load config from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)


def config_to_args(config: Dict[str, Any], frames: int, output_path: str) -> list:
    """Convert JSON config to command-line arguments"""
    args = []

    # Physics parameters
    physics = config.get('physics', {})
    if 'gm' in physics:
        args.extend(['--gm', str(physics['gm'])])
    if 'bh_mass' in physics:
        args.extend(['--bh-mass', str(physics['bh_mass'])])
    if 'alpha' in physics:
        args.extend(['--alpha', str(physics['alpha'])])
    if 'damping' in physics:
        args.extend(['--damping', str(physics['damping'])])
    if 'angular_boost' in physics:
        args.extend(['--angular-boost', str(physics['angular_boost'])])
    if 'disk_thickness' in physics:
        args.extend(['--disk-thickness', str(physics['disk_thickness'])])
    if 'inner_radius' in physics:
        args.extend(['--inner-radius', str(physics['inner_radius'])])
    if 'outer_radius' in physics:
        args.extend(['--outer-radius', str(physics['outer_radius'])])
    if 'density_scale' in physics:
        args.extend(['--density-scale', str(physics['density_scale'])])
    if 'force_clamp' in physics:
        args.extend(['--force-clamp', str(physics['force_clamp'])])
    if 'velocity_clamp' in physics:
        args.extend(['--velocity-clamp', str(physics['velocity_clamp'])])
    if 'boundary_mode' in physics:
        args.extend(['--boundary-mode', str(physics['boundary_mode'])])
    if 'time_multiplier' in physics:
        args.extend(['--physics-time-multiplier', str(physics['time_multiplier'])])

    # PINN configuration
    pinn = config.get('pinn', {})
    if pinn.get('enabled', False):
        model = pinn.get('model', 'pinn_v4_turbulence_robust')
        if 'v4' in model or 'turbulence' in model:
            args.extend(['--pinn', 'v4'])
        elif 'v3' in model:
            args.extend(['--pinn', 'v3'])
        elif 'v2' in model:
            args.extend(['--pinn', 'v2'])
        else:
            args.extend(['--pinn', 'v4'])

        if pinn.get('enforce_boundaries', False):
            args.append('--enforce-boundaries')

    # SIREN configuration
    siren = config.get('siren', {})
    if siren.get('enabled', False):
        args.append('--siren')
        if 'intensity' in siren:
            args.extend(['--siren-intensity', str(siren['intensity'])])
        if 'vortex_scale' in siren:
            args.extend(['--vortex-scale', str(siren['vortex_scale'])])
        if 'vortex_decay' in siren:
            args.extend(['--vortex-decay', str(siren['vortex_decay'])])

    # Particle count
    particles = config.get('particles', {})
    particle_count = particles.get('count', 5000)
    args.extend(['--particles', str(particle_count)])

    # Frames and output
    args.extend(['--frames', str(frames)])
    args.extend(['--output', output_path])

    return args


def wsl_to_windows_path(path: str) -> str:
    """Convert WSL path to Windows path for .exe"""
    if path.startswith('/mnt/'):
        drive = path[5]
        rest = path[6:].replace('/', '\\')
        return f"{drive.upper()}:{rest}"
    return path


def run_benchmark(
    executable_path: Path,
    config_path: Path,
    frames: int = 10000,
    timeout_minutes: float = 25.0,
    output_dir: Optional[Path] = None
) -> Dict[str, Any]:
    """
    Run benchmark with frame limit AND timeout protection.
    """
    config = load_config(config_path)
    config_name = config_path.stem

    if output_dir is None:
        output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"longterm_{config_name}_{frames}f_{timestamp}.json"
    output_path_win = wsl_to_windows_path(str(output_file.absolute()))

    args = config_to_args(config, frames, output_path_win)
    cmd = [str(executable_path), '--benchmark'] + args

    print(f"\n{'='*80}")
    print(f"LONG-TERM BENCHMARK")
    print(f"{'='*80}")
    print(f"Config: {config_path}")
    print(f"Frames: {frames}")
    print(f"Timeout: {timeout_minutes} minutes")
    print(f"Output: {output_file}")
    print(f"{'='*80}")
    print(f"\nCommand: {' '.join(cmd[:5])} ...")
    print(f"{'='*80}\n")

    timeout_seconds = timeout_minutes * 60
    start_time = time.time()

    result = {
        'status': 'unknown',
        'config_file': str(config_path),
        'frames_requested': frames,
        'timeout_minutes': timeout_minutes,
        'output_file': str(output_file),
        'start_time': timestamp,
        'duration_seconds': 0,
        'exit_code': None,
        'error': None
    }

    try:
        exe_dir = executable_path.parent
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=str(exe_dir)
        )

        result['exit_code'] = process.returncode
        result['duration_seconds'] = time.time() - start_time

        if output_file.exists():
            result['status'] = 'completed'
            print(f"\nBENCHMARK COMPLETED")
            print(f"   Duration: {result['duration_seconds']:.1f}s ({result['duration_seconds']/60:.1f} min)")
            print(f"   Output: {output_file}")

            with open(output_file, 'r') as f:
                bench_results = json.load(f)

            result['frames_simulated'] = bench_results.get('frames', frames)
            result['final_particle_count'] = bench_results.get('final_particle_count', 0)
            result['stability_score'] = bench_results.get('stability_score', 0)
            result['accuracy_score'] = bench_results.get('accuracy_score', 0)

            print(f"\nKey Results:")
            print(f"   Frames Simulated: {result['frames_simulated']}")
            print(f"   Stability Score: {result['stability_score']:.2f}")
            print(f"   Accuracy Score: {result['accuracy_score']:.2f}")
        else:
            result['status'] = 'failed'
            result['error'] = 'Output file not created'
            print(f"\nBENCHMARK FAILED")
            print(f"   Exit code: {process.returncode}")
            if process.stderr:
                print(f"   Stderr: {process.stderr[:500]}")
            if process.stdout:
                print(f"   Stdout: {process.stdout[:500]}")

    except subprocess.TimeoutExpired:
        result['status'] = 'timeout'
        result['duration_seconds'] = timeout_seconds
        result['error'] = f'Exceeded {timeout_minutes} minute timeout'
        print(f"\nBENCHMARK TIMEOUT")
        print(f"   Exceeded {timeout_minutes} minute limit")

    except Exception as e:
        result['status'] = 'error'
        result['duration_seconds'] = time.time() - start_time
        result['error'] = str(e)
        print(f"\nBENCHMARK ERROR: {e}")

    meta_file = output_dir / f"longterm_{config_name}_{frames}f_{timestamp}_meta.json"
    with open(meta_file, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nMetadata saved to: {meta_file}")

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Run long-term benchmark')
    parser.add_argument('--config', type=str, default='configs/ga_rank1.json',
                       help='Path to config JSON file')
    parser.add_argument('--frames', type=int, default=10000,
                       help='Number of frames (default: 10000)')
    parser.add_argument('--timeout', type=float, default=25.0,
                       help='Timeout in minutes (default: 25)')
    parser.add_argument('--exe', type=str, default=None,
                       help='Path to PlasmaDX-Clean.exe')

    args = parser.parse_args()

    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent.parent

    if args.exe:
        executable = Path(args.exe)
    else:
        executable = project_root / "build/bin/Debug/PlasmaDX-Clean.exe"

    if not executable.exists():
        print(f"ERROR: Executable not found: {executable}")
        sys.exit(1)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path

    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}")
        sys.exit(1)

    result = run_benchmark(
        executable_path=executable,
        config_path=config_path,
        frames=args.frames,
        timeout_minutes=args.timeout
    )

    print(f"\n{'='*80}")
    print(f"FINAL STATUS: {result['status'].upper()}")
    print(f"{'='*80}")

    if result['status'] == 'completed':
        sys.exit(0)
    elif result['status'] == 'timeout':
        sys.exit(2)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
