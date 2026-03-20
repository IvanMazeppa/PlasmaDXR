"""Run a single aesthetic benchmark and report results.

Usage:
    python scripts/run_benchmark.py --prompt "Close-up of red wine pouring..." --effect water
    python scripts/run_benchmark.py --benchmark bench_tabletop_glass
"""

import argparse
import asyncio
import json
import sys
import os
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))


BENCHMARK_FILE = Path(__file__).parent.parent / "tests" / "fixtures" / "aesthetic_benchmarks.json"


def load_benchmark(benchmark_id: str) -> dict:
    """Load a benchmark by ID from the fixture file."""
    with open(BENCHMARK_FILE) as f:
        data = json.load(f)
    bench_list = data.get("benchmarks", data) if isinstance(data, dict) else data
    for b in bench_list:
        if b["id"] == benchmark_id:
            return b
    raise ValueError(f"Benchmark '{benchmark_id}' not found. Available: {[b['id'] for b in bench_list]}")


async def run_benchmark(
    prompt: str,
    effect_type: str,
    asset_name: str,
    max_iterations: int = 3,
    resolution: int = 96,
    quality_threshold: float = 50.0,
):
    """Run a single benchmark and return session state."""
    from orchestrator import create_vfx_asset

    print(f"\n{'='*60}", file=sys.stderr)
    print(f"BENCHMARK: {asset_name}", file=sys.stderr)
    print(f"PROMPT: {prompt[:80]}...", file=sys.stderr)
    print(f"EFFECT: {effect_type}", file=sys.stderr)
    print(f"MAX ITERS: {max_iterations}, RES: {resolution}", file=sys.stderr)
    print(f"{'='*60}\n", file=sys.stderr)

    # Verify enhancement pipeline fires
    from utils.prompt_enhancer import enhance_description, extract_style_spec
    enhanced = enhance_description(prompt, effect_type=effect_type)
    spec = extract_style_spec(prompt, effect_type=effect_type)

    print(f"[Benchmark] StyleSpec: mood={spec['mood']}, heroes={spec['hero_objects']}, "
          f"camera={spec['camera_distance_class']}, lighting={spec['world_lighting_mode']}", file=sys.stderr)
    print(f"[Benchmark] Enhancement: LOOK-DEV={'YES' if 'LOOK-DEV BRIEF' in enhanced else 'NO'}, "
          f"CAMERA={'YES' if 'CAMERA READABILITY' in enhanced else 'NO'}", file=sys.stderr)

    from tools.hero_object_refinement import get_refinement_instructions
    hero_inst = get_refinement_instructions(spec.get('hero_objects', []), spec.get('camera_distance_class', 'medium'))
    print(f"[Benchmark] Hero refinement: {'YES' if hero_inst else 'NO'} "
          f"({len(hero_inst)} chars)", file=sys.stderr)

    session = await create_vfx_asset(
        asset_name=asset_name,
        description=prompt,
        effect_type=effect_type,
        resolution=resolution,
        max_iterations=max_iterations,
        quality_threshold=quality_threshold,
        enable_diagnostics=True,
    )

    # Report results
    print(f"\n{'='*60}", file=sys.stderr)
    print(f"RESULTS: {asset_name}", file=sys.stderr)
    print(f"  Status: {session.status.value}", file=sys.stderr)
    print(f"  Iterations: {session.current_iteration}", file=sys.stderr)
    print(f"  Best score: {session.best_score:.1f} (iter {session.best_iteration})", file=sys.stderr)
    print(f"  Final render: {session.final_render_path}", file=sys.stderr)

    if session.iterations:
        print(f"\n  Iteration History:", file=sys.stderr)
        for it in session.iterations:
            print(f"    Iter {it.iteration}: score={it.score:.1f}, "
                  f"change={it.change_label or 'unknown'}, "
                  f"hash={it.script_hash or 'n/a'}, "
                  f"issues={it.issue_kinds or []}", file=sys.stderr)

    print(f"{'='*60}\n", file=sys.stderr)
    return session


def main():
    parser = argparse.ArgumentParser(description="Run aesthetic benchmark")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--benchmark", help="Benchmark ID from fixtures")
    group.add_argument("--prompt", help="Custom prompt text")
    parser.add_argument("--effect", default="water", help="Effect type")
    parser.add_argument("--name", default="benchmark_run", help="Asset name")
    parser.add_argument("--max-iters", type=int, default=3, help="Max iterations")
    parser.add_argument("--resolution", type=int, default=96, help="Sim resolution")
    parser.add_argument("--threshold", type=float, default=50.0, help="Quality threshold")
    args = parser.parse_args()

    if args.benchmark:
        bench = load_benchmark(args.benchmark)
        prompt = bench["prompt"]
        effect = bench["effect_type"]
        name = bench["id"]
    else:
        prompt = args.prompt
        effect = args.effect
        name = args.name

    session = asyncio.run(run_benchmark(
        prompt=prompt,
        effect_type=effect,
        asset_name=name,
        max_iterations=args.max_iters,
        resolution=args.resolution,
        quality_threshold=args.threshold,
    ))

    # Print JSON summary to stdout for programmatic consumption
    summary = {
        "asset_name": name,
        "status": session.status.value,
        "iterations": session.current_iteration,
        "best_score": session.best_score,
        "best_iteration": session.best_iteration,
        "final_render": session.final_render_path,
        "iteration_details": [
            {
                "iteration": it.iteration,
                "score": it.score,
                "change_label": it.change_label,
                "script_hash": it.script_hash,
                "issue_kinds": it.issue_kinds,
                "passed": it.passed,
            }
            for it in session.iterations
        ],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
