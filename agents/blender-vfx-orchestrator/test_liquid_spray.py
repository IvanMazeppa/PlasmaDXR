#!/usr/bin/env python3
"""
Liquid Spray Test - Water spraying from a broken plumbing pipe.

Runs a LIQUID simulation for 3 iterations with tracing enabled.
Tests the Mantaflow LIQUID domain pipeline end-to-end.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

# Enable verbose tracing BEFORE importing orchestrator
from tracing import enable_verbose_tracing
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"traces/liquid_spray_{timestamp}.jsonl"
enable_verbose_tracing(
    log_file=log_file,
    console_output=True,
    include_span_data=True,
)
print(f"[SETUP] Tracing enabled: {log_file}")

from orchestrator import BlenderVFXOrchestrator
from models.shared_context import AssetRequest, EffectType
from agents import trace


WATER_DESCRIPTION = """SCENE DESCRIPTION:
A copper plumbing pipe running horizontally along a basement wall has ruptured at an elbow joint. Water sprays outward in a pressurized fan-shaped jet from the cracked fitting, splattering against the concrete floor below and pooling into a growing puddle. Droplets scatter outward from the impact zone. Over the 50-frame duration, the spray is continuous and the puddle expands outward across the floor.

MOOD AND LOOK:
Industrial urgency — the harsh overhead fluorescent light catches the water mid-spray, making the jet appear silvery-white against the dim basement environment. The wet concrete floor reflects the overhead light with a slick sheen. Color palette: cold steel grays, patina green-brown copper, blue-white water highlights, dark concrete. The mood is utilitarian and slightly ominous — a plumbing emergency in progress.

CAMERA:
Medium close-up, eye-level with the pipe breach. Camera positioned ~0.6m from the rupture point, angled slightly downward to capture both the spray jet and the splash impact on the floor. Moderate depth of field (f/5.6) — the pipe joint and spray are sharp, background wall softly out of focus. Static camera. 50mm lens equivalent for natural perspective.

PHYSICAL DETAILS:
- Pipe: Copper tube, 22mm outer diameter, ~0.4m visible horizontal run, patina green-brown surface with dull metallic sheen
- Elbow joint: 90-degree copper elbow fitting at the rupture point, ~3cm long, slightly separated from the pipe with a 3mm gap where the breach is
- Spray: Pressurized water fan ~15cm wide at 20cm from the breach, exits at ~45 degrees downward from the pipe axis
- Floor: Flat concrete slab, matte dark gray (0.3, 0.3, 0.32), positioned 0.4m below the pipe
- Puddle: Water pooling on concrete, ~25cm radius by frame 50, thin layer with reflective surface
- Wall: Flat concrete surface behind pipe, vertical, matte medium gray

MOTION/TIMING DETAILS:
The spray is continuous from frame 1 — no ramp-up needed, the pipe is already broken. Water exits the breach as a pressurized stream that fans out into droplets. On floor impact, water splashes radially outward with small secondary droplets. The puddle grows steadily from the impact zone.

HARD CONSTRAINTS:
- Fluid simulation: Blender Mantaflow liquid domain
- Domain type: LIQUID
- Liquid viscosity: Water (default, no viscosity override)
- use_plane_init: True (required for inflow emitters)
- Inflow velocity: Moderate pressure (~2-3 m/s initial velocity at breach)
- Resolution: 96 (balance detail vs bake time)
- Renderer: Cycles GPU
- Samples: 128 max
- Frame range: 1-50
- cache_type: ALL
- Render frame: 35 (puddle established, spray mid-action)
- Reference: None

Research documentation, patterns, and APIs to find the optimal starting approach."""


async def main():
    print("=" * 70)
    print("LIQUID SPRAY TEST - Broken Pipe Water Simulation")
    print("3 iterations, tracing enabled")
    print("=" * 70)
    print()

    request = AssetRequest(
        asset_name="liquid_spray_pipe_v1",
        description=WATER_DESCRIPTION,
        effect_type=EffectType.WATER,
        resolution=96,
        frame_start=1,
        frame_end=50,
        quality_threshold=65.0,
        max_iterations=3,
        semantic_query="water spray broken pipe plumbing liquid splash puddle mantaflow inflow",
    )

    orchestrator = BlenderVFXOrchestrator()
    await orchestrator.initialize()

    print(f"Asset: {request.asset_name}")
    print(f"Effect Type: {request.effect_type.value}")
    print(f"Max Iterations: {request.max_iterations}")
    print(f"Quality Threshold: {request.quality_threshold}")
    print()

    try:
        with trace("Liquid Spray Test", group_id=request.asset_name):
            result = await orchestrator.create_asset_pipeline(request)

        print("\n" + "=" * 70)
        print("LIQUID SPRAY TEST COMPLETE")
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
                print("  (Normal if Learning Agent provided params or pattern matched)")
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
