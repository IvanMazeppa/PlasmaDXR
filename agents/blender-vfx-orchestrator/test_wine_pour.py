#!/usr/bin/env python3
"""
Wine pour 3-iteration test for Blender VFX Orchestrator.

Tests liquid simulation with enhanced prompt for wine pouring scene.
S1.5 benchmark: liquid scenario #2 (alongside kitchen_leak).
"""

import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

# Enable verbose tracing BEFORE importing orchestrator
from tracing import enable_verbose_tracing
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"traces/wine_pour_{timestamp}.jsonl"
enable_verbose_tracing(
    log_file=log_file,
    console_output=True,
    include_span_data=True,
)
print(f"[SETUP] Tracing enabled: {log_file}")

from orchestrator import BlenderVFXOrchestrator
from models.shared_context import AssetRequest, EffectType
from agents import trace


WINE_DESCRIPTION = """SCENE DESCRIPTION:
A stream of deep red wine pours from an unseen bottle (positioned just above frame) into an elegant stemmed wine glass resting on a dark wooden surface. The pour is steady but vigorous — not a delicate trickle, but a confident stream that creates dynamic splashing as it impacts the rising liquid surface. As the glass fills past the halfway point, wine droplets leap from the surface on impact, some catching the warm light as they arc through the air. A few droplets escape over the rim and trail down the outside of the glass.

MOOD AND LOOK:
Warm, intimate atmosphere evoking a cozy evening scene. Rich amber/golden key lighting from one side (as if from a nearby fireplace or candle cluster) creates dramatic highlights on the glass rim and wine surface, with the deep burgundy liquid glowing where light passes through it. The background falls into soft, warm darkness. The wine should exhibit convincing subsurface scattering — appearing darker at depth, more translucent and ruby-red where thin against the glass walls. Visible caustics pooling on the table surface beneath the glass.

CAMERA:
Close-up, slightly low angle looking up at the glass to emphasize elegance. Shallow depth of field optional. 35mm lens equivalent. Static camera.

PHYSICAL DETAILS:
- Wine glass: Standard Bordeaux style, clear glass with subtle reflections, ~22cm tall
- Wine color: Deep burgundy/cabernet red with ruby highlights in thin areas
- Pour stream: ~8mm diameter, continuous with slight surface tension wobble
- Table surface: Dark polished wood with subtle reflection of the glass base
- Background: Warm out-of-focus bokeh suggesting a candlelit room

MOTION/TIMING DETAILS:
The pour is continuous from frame 1. Wine enters from above frame and impacts the liquid surface already in the glass. By frame 30, the glass is about half full with active splashing. By frame 45, the glass is nearly 3/4 full and small waves propagate across the surface. Render frame 40 for peak action.

HARD CONSTRAINTS:
- Fluid simulation: Blender Mantaflow liquid domain (FLIP solver, not shader-only)
- Domain type: LIQUID with mesh generation enabled
- use_plane_init: True (required for inflow emitters)
- Inflow velocity: ~1.0-1.5 m/s downward pour
- Resolution: 96 (balance detail vs bake time)
- sampling_substeps: 6 (prevent discretized stream artifacts)
- Renderer: Cycles GPU
- Samples: 128 max
- Frame range: 1-60
- cache_type: ALL
- Render frame: 40 (mid-pour with active surface dynamics)
- Reference: None

Research documentation, patterns, and APIs to find the optimal starting approach."""


async def main():
    print("=" * 70)
    print("WINE POUR TEST - S1.5 Benchmark Liquid #2")
    print("3 iterations, tracing enabled")
    print(f"Trace log: {log_file}")
    print("=" * 70)
    print()

    request = AssetRequest(
        asset_name="wine_pour",
        description=WINE_DESCRIPTION,
        effect_type=EffectType.WATER,
        resolution=96,
        frame_start=1,
        frame_end=60,
        quality_threshold=60.0,
        max_iterations=3,
        semantic_query="wine pour liquid glass splash mantaflow inflow red wine stream impact surface",
    )

    orchestrator = BlenderVFXOrchestrator()
    await orchestrator.initialize()

    print(f"Asset: {request.asset_name}")
    print(f"Effect Type: {request.effect_type.value}")
    print(f"Max Iterations: {request.max_iterations}")
    print(f"Quality Threshold: {request.quality_threshold}")
    print()

    try:
        with trace("Wine Pour S1.5 Benchmark", group_id=request.asset_name):
            result = await orchestrator.create_asset_pipeline(request)

        print("\n" + "=" * 70)
        print("WINE POUR TEST COMPLETE")
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

            if "Injected volume material" in content:
                print("  [FAIL] Volume material injected on liquid domain")
            else:
                print("  [PASS] No volume material injected")

            if "Injected water material" in content:
                print("  [INFO] Liquid material injector triggered")
            else:
                print("  [INFO] No liquid material injection needed")

            if "LIGHT_ENERGY" in content:
                print("  [INFO] Light energy scaler triggered")

            if "Error Recovery" in content:
                print("  [WARN] Error recovery was triggered")
            else:
                print("  [PASS] No error recovery triggered")

            if "OVERRIDE: Executor reported failure but render exists" in content:
                print("  [INFO] Render discovery override was used")

            if "could not convert string to float" in content:
                print("  [FAIL] record_experiment_result still crashing")
            else:
                print("  [PASS] No record_experiment_result crashes")

        print("=" * 70)

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
