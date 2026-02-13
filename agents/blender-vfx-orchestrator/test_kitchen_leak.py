#!/usr/bin/env python3
"""
Kitchen Leak Test - Rich environment scene with water simulation.

Tests:
1. Script Writer produces rich scene (8+ objects, 5+ materials, 3+ lights)
2. Environment matches prompt description (not simplified to grey void)
3. Lighting appropriate for close-up/medium shot (no washout)
4. Water simulation bakes and renders correctly
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from tracing import enable_verbose_tracing
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"traces/kitchen_leak_{timestamp}.jsonl"
enable_verbose_tracing(
    log_file=log_file,
    console_output=True,
    include_span_data=True,
)
print(f"[SETUP] Tracing enabled: {log_file}")

from orchestrator import BlenderVFXOrchestrator
from models.shared_context import AssetRequest, EffectType
from agents import trace


KITCHEN_LEAK = """SCENE DESCRIPTION:
Under the kitchen sink, the flexible braided-steel supply hose for the hot water tap has burst at the crimp fitting where it meets the shutoff valve. A steady stream of water sprays sideways from the failed connection, hitting the interior wall of the sink cabinet and pooling across the cabinet floor. The cabinet doors are open (removed from the scene — we are looking directly into the under-sink cavity). Cleaning products, a bucket, and sponges sit in the pooling water. The spray has been running long enough that a shallow puddle covers the entire cabinet floor and drips over the front lip.

MOOD AND LOOK:
Domestic emergency — a harried homeowner has just discovered the leak. The overhead kitchen pendant light above the counter casts warm amber light downward. Inside the cabinet, the light is indirect — mostly bounced off the tiled backsplash and countertop above, creating soft warm shadows. A small LED strip under the upper cabinets (not visible, above frame) provides a cool blue-white accent along the countertop edge. Color palette: warm honey-oak wood cabinet interior, dark slate-green tiled backsplash visible above, chrome and steel fixtures, blue-tinted water catching the cool accent light. The mood is tense but well-lit — this is a modern kitchen, not a dark basement.

CAMERA:
Medium shot from floor level, looking slightly upward into the open cabinet cavity. Camera is ~1.2m from the burst fitting, positioned on the kitchen floor in front of the open cabinet. The camera sees: the burst hose connection (center-left of frame), the spray hitting the cabinet side wall (center), the pooling water on the cabinet floor (bottom third), and the underside of the countertop with the sink drain pipe (top of frame). Moderate depth of field (f/5.6) — hose fitting and spray in sharp focus, cleaning bottles in background slightly soft. 35mm lens equivalent for slight wide-angle that captures the enclosed space. Static camera.

PHYSICAL DETAILS:
- Cabinet interior: Honey-oak plywood panels, ~0.6m wide x 0.5m deep x 0.5m tall interior cavity. Floor, back wall, two side walls. No front doors (open/removed). Visible wood grain texture with satin finish.
- Countertop underside: Dark grey stone (granite), ~3cm thick slab overhanging the cabinet by 2cm. Visible along top edge of frame.
- Sink drain pipe: White PVC P-trap, ~4cm diameter, descending from countertop center into cabinet space. The P-trap makes a U-bend and exits through the back wall.
- Supply hoses: Two braided-steel flexible hoses (~1cm diameter, 30cm long) running from shutoff valves on the back wall up through holes in the countertop. The LEFT hose (hot water) has burst at its lower crimp fitting — the hose has separated 5mm from the brass compression fitting on the shutoff valve.
- Shutoff valves: Two chrome quarter-turn ball valves mounted on copper stub-outs emerging from the back wall, ~15cm apart, ~10cm above the cabinet floor.
- Spray: Water exits the gap as a pressurized fan ~8cm wide at 15cm distance, angled sideways toward the left cabinet wall. Flow rate is moderate — not a firehose, but a steady vigorous spray.
- Cabinet floor: Plywood with slight warp from water damage, ~2cm deep puddle covering the full floor area by frame 40.
- Props in cabinet: Yellow rubber cleaning gloves (draped over the P-trap), a blue plastic bucket (10cm tall, sitting in the puddle, partially filled), a green scouring sponge (floating in the puddle), a white spray bottle of cleaner (standing upright near the back wall, label facing camera).

MOTION/TIMING DETAILS:
The spray is continuous from frame 1 — the hose is already burst. Water hits the left cabinet side wall and runs down in rivulets. On the cabinet floor, the puddle is already established and slowly deepening. Small ripples propagate outward from where the spray runoff enters the puddle. The cleaning gloves sway very slightly from water contact. By frame 40, the puddle has reached the front lip of the cabinet and begins to drip over the edge (one or two drips visible).

HARD CONSTRAINTS:
- Fluid simulation: Blender Mantaflow liquid domain
- Domain type: LIQUID
- Liquid viscosity: Water (default, no viscosity override needed)
- use_plane_init: True (required for inflow emitters)
- Inflow velocity: Moderate pressure (~1.5-2.0 m/s at the burst fitting)
- Resolution: 96 (balance detail vs bake time)
- sampling_substeps: 6 (prevent discretized jet artifacts)
- Renderer: Cycles GPU
- Samples: 128 max
- Frame range: 1-50
- cache_type: ALL
- Render frame: 40 (puddle deep, spray mid-action, drip beginning)
- Reference: None

Research documentation, patterns, and APIs to find the optimal starting approach."""


async def main():
    print("=" * 70)
    print("KITCHEN LEAK TEST - Rich Environment Scene")
    print("1 iteration, tracing enabled")
    print("=" * 70)
    print()

    request = AssetRequest(
        asset_name="kitchen_leak",
        description=KITCHEN_LEAK,
        effect_type=EffectType.WATER,
        resolution=96,
        frame_start=1,
        frame_end=50,
        quality_threshold=65.0,
        max_iterations=1,
        semantic_query="water spray broken hose pipe kitchen liquid splash puddle mantaflow inflow cabinet sink",
    )

    orchestrator = BlenderVFXOrchestrator()
    await orchestrator.initialize()

    print(f"Asset: {request.asset_name}")
    print(f"Effect Type: {request.effect_type.value}")
    print(f"Max Iterations: {request.max_iterations}")
    print(f"Quality Threshold: {request.quality_threshold}")
    print()

    try:
        with trace("Kitchen Leak Test", group_id=request.asset_name):
            result = await orchestrator.create_asset_pipeline(request)

        print("\n" + "=" * 70)
        print("KITCHEN LEAK TEST COMPLETE")
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

        if result.current_issues:
            print()
            print("Issues:")
            for issue in result.current_issues:
                print(f"  - {issue}")

        # Trace analysis
        print()
        print("=" * 70)
        print("TRACE ANALYSIS")
        print("=" * 70)
        trace_path = Path(log_file)
        if trace_path.exists():
            with open(trace_path) as f:
                content = f.read()

            # Volume material check
            if "Injected volume material" in content:
                print("  [FAIL] Volume material injected on liquid domain")
            else:
                print("  [PASS] No volume material injected")

            # Error recovery check
            if "Error Recovery" in content:
                print("  [WARN] Error recovery was triggered")
            else:
                print("  [PASS] No error recovery triggered")

            # Render discovery
            if "OVERRIDE: Executor reported failure but render exists" in content:
                print("  [INFO] Render discovery override was used")
            if "Render discovered:" in content:
                print("  [INFO] Render discovery fallback was used")

            # Experiment tracking
            if "could not convert string to float" in content:
                print("  [FAIL] record_experiment_result still crashing")
            else:
                print("  [PASS] No record_experiment_result crashes")

            # Script complexity — count the generated script lines
            import re
            script_matches = re.findall(r'"script_path":\s*"([^"]*kitchen_leak[^"]*\.py)"', content)
            if script_matches:
                for sp in set(script_matches):
                    try:
                        lines = Path(sp).read_text().count('\n')
                        print(f"  [INFO] Script {Path(sp).name}: {lines} lines")
                        if lines < 500:
                            print(f"         [WARN] Script under 500 lines — may be too simple")
                        elif lines >= 600:
                            print(f"         [PASS] Script 600+ lines — likely rich scene")
                    except Exception:
                        pass

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
