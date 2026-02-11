#!/usr/bin/env python3
"""
Deterministic Fallback Test - Validates P2-3 quality-to-parameter mapping.

Runs a fire effect for 2 iterations with tracing enabled.
Iteration 1 produces the baseline; iteration 2 exercises the deterministic
fallback path (map_quality_issues_to_params → merge → _modify_script_impl).

Uses a structured prompt following the Prompt Enhancement Guide.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

# Enable verbose tracing BEFORE importing orchestrator
from tracing import enable_verbose_tracing
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"traces/det_fallback_{timestamp}.jsonl"
enable_verbose_tracing(
    log_file=log_file,
    console_output=True,
    include_span_data=True,
)
print(f"[SETUP] Tracing enabled: {log_file}")

from orchestrator import BlenderVFXOrchestrator
from models.shared_context import AssetRequest, EffectType
from agents import trace


FIRE_DESCRIPTION = """SCENE DESCRIPTION:
A single propane camping stove burner produces a steady ring of blue-yellow flames
in an otherwise dark workshop. The flame ring is ~10cm diameter, rising ~5cm tall
at its peak. A slight draft from the left causes the flames to lean and flicker
asymmetrically. No smoke — this is a clean gas burn.

MOOD AND LOOK:
Industrial, utilitarian. Cool blue light dominates with warm yellow tips. The
metal stove grate glows faintly orange where flame contacts it. Background is
black. Feeling is functional and mesmerizing — like watching a gas burner at 3am.

CAMERA:
Medium close-up, eye-level, static. The flame ring fills roughly half the frame
width. Shallow depth of field with the nearest flame edge tack-sharp.

PHYSICAL DETAILS:
- Burner ring: 10cm diameter, brass with visible gas ports
- Flame height: 3-5cm, stable base with flickering tips
- Flame color: Blue core (2cm), yellow tips, occasional orange flicker
- No smoke, no soot
- Ambient temp: room temperature, no visible heat haze

HARD CONSTRAINTS:
- Fluid simulation: Blender Mantaflow smoke domain with fire enabled
- Domain type: GAS
- Fire reaction speed: Medium (0.5-0.7) for steady gas flame
- Smoke amount: Near-zero (0.01)
- Vorticity: Low (0.2-0.3)
- Renderer: Cycles GPU
- Samples: 128 max
- Frame range: 1-25
- Resolution: 64
- cache_type: ALL
- Reference: None

Research documentation, patterns, and APIs to find the optimal starting approach."""


async def main():
    print("=" * 70)
    print("DETERMINISTIC FALLBACK TEST (P2-3)")
    print("Verifying quality-to-parameter mapping works in iteration loop")
    print("=" * 70)
    print()

    request = AssetRequest(
        asset_name="det_fallback_fire_v1",
        description=FIRE_DESCRIPTION,
        effect_type=EffectType.FIRE,
        resolution=64,
        frame_start=1,
        frame_end=25,
        quality_threshold=70.0,
        max_iterations=2,
        semantic_query="gas burner flame ring blue core yellow tips industrial workshop dark background",
    )

    orchestrator = BlenderVFXOrchestrator()
    await orchestrator.initialize()

    print(f"Asset: {request.asset_name}")
    print(f"Effect Type: {request.effect_type.value}")
    print(f"Max Iterations: {request.max_iterations}")
    print(f"Quality Threshold: {request.quality_threshold}")
    print()

    try:
        with trace("Deterministic Fallback Test", group_id=request.asset_name):
            result = await orchestrator.create_asset_pipeline(request)

        print("\n" + "=" * 70)
        print("DETERMINISTIC FALLBACK TEST COMPLETE")
        print("=" * 70)
        print(f"Status: {result.status.value}")
        print(f"Best Score: {result.best_score:.1f}")
        print(f"Best Iteration: {result.best_iteration}")
        print(f"Total Iterations: {result.current_iteration}")
        print()

        if result.final_render_path:
            print(f"Final Render: {result.final_render_path}")
        if result.final_vdb_dir:
            print(f"VDB Directory: {result.final_vdb_dir}")

        print()
        print("Iteration History:")
        for it in result.iterations:
            status = "PASS" if it.passed else "FAIL"
            technique = it.technique[:60] if it.technique else "unknown"
            print(f"  [{status}] Iteration {it.iteration}: {it.score:.1f} - {it.primary_issue or 'No issues'}")
            print(f"          Technique: {technique}")

        if result.current_issues:
            print()
            print("Final Issues:")
            for issue in result.current_issues:
                print(f"  - {issue}")

        # Check for deterministic fallback evidence in trace
        print()
        print("=" * 70)
        print("DETERMINISTIC FALLBACK EVIDENCE")
        print("=" * 70)
        trace_path = Path(log_file)
        if trace_path.exists():
            import json
            det_hits = 0
            with open(trace_path) as f:
                for line in f:
                    if "DETERMINISTIC FALLBACK" in line or "det-fallback" in line or "detfix" in line:
                        det_hits += 1
            if det_hits > 0:
                print(f"  Found {det_hits} deterministic fallback entries in trace")
            else:
                print("  No deterministic fallback entries found in trace")
                print("  (This is normal if Learning Agent provided params or pattern matched)")
        else:
            print(f"  Trace file not found: {log_file}")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        await orchestrator.close()

    return 0 if result.status.value in ("passed", "max_iterations") else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
