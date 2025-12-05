#!/usr/bin/env python3
"""Test different physics time multipliers to find maximum stable value."""

import subprocess
import json
import os
import tempfile
import sys

def test_multiplier(multiplier: float, frames: int = 1000) -> dict:
    """Run benchmark with given time multiplier and return results."""

    # Create temp config
    config = {
        "benchmark": {"enabled": True, "frames": frames},
        "physics": {"time_multiplier": multiplier}
    }

    config_path = tempfile.mktemp(suffix='.json')
    with open(config_path, 'w') as f:
        json.dump(config, f)

    exe_path = "build/bin/Debug/PlasmaDX-Clean.exe"
    result_path = "build/bin/Debug/benchmark_results.json"

    try:
        # Run benchmark
        subprocess.run(
            [exe_path, f"--config={config_path}", "--benchmark"],
            capture_output=True,
            timeout=120,
            cwd="/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean"
        )

        # Read results
        with open(f"/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/{result_path}") as f:
            results = json.load(f)

        return {
            "multiplier": multiplier,
            "stability": results.get("stability", {}).get("score", 0),
            "energy_drift": results.get("stability", {}).get("energy_drift_percent", 999),
            "escape_rate": results.get("stability", {}).get("particle_escape_rate", 999),
            "keplerian_error": results.get("accuracy", {}).get("keplerian_error_percent", 999),
            "overall": results.get("overall_score", 0)
        }
    except Exception as e:
        return {"multiplier": multiplier, "error": str(e)}
    finally:
        if os.path.exists(config_path):
            os.remove(config_path)

def main():
    print("=" * 60)
    print("Physics Time Multiplier Stability Test")
    print("=" * 60)

    multipliers = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0]
    results = []

    for mult in multipliers:
        print(f"\nTesting {mult}x physics time multiplier...")
        result = test_multiplier(mult, frames=1000)
        results.append(result)

        if "error" in result:
            print(f"  ERROR: {result['error']}")
        else:
            print(f"  Stability: {result['stability']:.1f}/100")
            print(f"  Energy drift: {result['energy_drift']:.2f}%")
            print(f"  Escape rate: {result['escape_rate']:.4f}%")
            print(f"  Keplerian error: {result['keplerian_error']:.2f}%")
            print(f"  Overall: {result['overall']:.1f}/100")

    print("\n" + "=" * 60)
    print("SUMMARY - Finding Maximum Stable Multiplier")
    print("=" * 60)
    print(f"{'Mult':>6} | {'Stability':>9} | {'Energy%':>8} | {'Escape%':>8} | {'Kepler%':>8} | {'Overall':>7}")
    print("-" * 60)

    for r in results:
        if "error" not in r:
            print(f"{r['multiplier']:>6.1f} | {r['stability']:>9.1f} | {r['energy_drift']:>8.2f} | {r['escape_rate']:>8.4f} | {r['keplerian_error']:>8.2f} | {r['overall']:>7.1f}")

    # Find maximum stable multiplier (stability > 90, energy < 10%)
    stable = [r for r in results if "error" not in r and r["stability"] > 90 and r["energy_drift"] < 10]
    if stable:
        best = max(stable, key=lambda x: x["multiplier"])
        print(f"\n✅ RECOMMENDED: {best['multiplier']}x (Stability {best['stability']:.1f}, Energy drift {best['energy_drift']:.2f}%)")
    else:
        print("\n⚠️ No stable multiplier found above 1.0x")

if __name__ == "__main__":
    main()
