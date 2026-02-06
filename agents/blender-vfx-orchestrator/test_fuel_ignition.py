#!/usr/bin/env python3
"""
Fuel Ignition Test - Flammable Liquid Fire Spread

Tests Mantaflow GAS domain with fire for realistic flame front propagation.
Uses detailed prompt following the enhancement guide for best results.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Enable verbose tracing BEFORE importing orchestrator
from tracing import enable_verbose_tracing
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"traces/fuel_ignition_{timestamp}.jsonl"
enable_verbose_tracing(
    log_file=log_file,
    console_output=True,
    include_span_data=True,
)
print(f"[SETUP] Tracing enabled: {log_file}")

from orchestrator import BlenderVFXOrchestrator
from models.shared_context import AssetRequest, EffectType
from agents import trace


FUEL_IGNITION_DESCRIPTION = """SCENE DESCRIPTION:
A shallow pool of denatured alcohol (ethanol) sits in a weathered steel drip tray, approximately 30cm x 20cm, resting on a concrete surface in what appears to be an industrial workshop setting. The clear liquid reflects dim ambient light, its surface perfectly still. At frame 1, a spark or small flame source at one edge of the pool initiates combustion. The invisible alcohol vapors above the liquid surface ignite first, creating a characteristic blue-tinged flame that races across the pool in a sweeping wave. As the flame front propagates, it transitions from the initial blue flash to a more sustained pale yellow-orange fire that dances above the liquid surface. The fire consumes the alcohol over the duration of the shot, with flames gradually diminishing as fuel depletes.

MOOD AND LOOK:
Tense, industrial atmosphere with an undercurrent of danger. The scene evokes controlled laboratory accidents or industrial safety training footage. Low ambient lighting creates dramatic contrast between the bright flames and the shadowed surroundings. The concrete floor and steel tray ground the scene in utilitarian reality rather than fantasy. Color palette: cold blue-grays of the environment contrast sharply with the warm oranges and blues of the alcohol fire. The flame itself should appear authentic to ethanol combustion — predominantly blue at the base where combustion is hottest and most complete, transitioning to pale yellow-orange at the tips where unburned carbon particles glow.

CAMERA:
Medium-wide shot capturing the entire drip tray with comfortable margins to show the flame spread. Camera positioned at approximately 45-degree angle above the scene, looking down at the tray — this angle best reveals the flame front propagation across the liquid surface. Shallow depth of field with focus on the pool surface, allowing background workshop elements to fall into soft blur. Static camera on tripod for stability. Equivalent focal length of 50mm for naturalistic perspective without distortion.

PHYSICAL DETAILS:
- Fuel: Denatured ethanol, clear liquid with slight shimmer, density ~0.79 g/cm³
- Pool dimensions: 30cm x 20cm x 1cm deep (~600ml volume)
- Tray: Galvanized steel drip tray with rolled edges, showing light corrosion and oil stains
- Surface: Concrete floor with industrial gray finish, visible texture
- Flame height: 8-15cm above liquid surface, varying with turbulence
- Flame structure: Blue base (2-3cm), transitioning to pale yellow-orange body
- Flame spread rate: Approximately 50cm/second for ethanol vapour (fast flash)
- Burn duration: Initial flash 0.5s, sustained burn 3-4 seconds before diminishing

MOTION/TIMING DETAILS:
Frame 1-12: Ignition point appears at left edge of pool. Initial blue flash begins propagating.
Frame 12-24: Flame front races across the pool surface in a sweeping arc, characteristic of vapor-phase ignition. This should look like a wave of blue fire washing over the surface.
Frame 24-72: Full-surface fire established. Flames dance and flicker with turbulent motion. Gradual transition from intense to moderate as fuel begins depleting.
Frame 72-96: Fire diminishes. Flames shrink and become more sporadic as liquid level drops. Final wisps of blue flame chase the last fuel patches.

The flame front propagation is the KEY visual — it must show the fire spreading across the surface, not just appearing everywhere at once. This requires the inflow/source to animate or use expanding geometry.

HARD CONSTRAINTS:
- Fluid simulation: Blender Mantaflow smoke domain with FIRE ENABLED
- Domain type: GAS (not liquid)
- Fire settings:
  - flame_smoke: 0.3-0.5 (moderate smoke production from alcohol)
  - flame_vorticity: 0.7-1.0 (moderate turbulence)
  - flame_max_temp: 1500-2000K (alcohol burns cooler than hydrocarbons)
  - burning_rate: 0.5-0.8 (fast-burning fuel)
- Smoke settings:
  - smoke_adaptive_domain: True (efficiency)
  - use_noise: True for fire detail at 2-4x
- Emitter behavior: ANIMATED source that expands from ignition point to simulate flame front
  - Option A: Animated mesh emitter that scales/grows over time
  - Option B: Multiple point emitters activated in sequence
  - Option C: Use flow velocity to push ignition across surface
- Resolution: 128-192 (fire needs detail for realistic flickering)
- Renderer: Cycles GPU
- Samples: 256 max
- Frame range: 1-96 (4 seconds at 24fps)
- cache_type: ALL

MATERIAL REQUIREMENTS:
- Fire shader: Principled Volume with:
  - Blackbody emission temperature-mapped to flame_temp attribute
  - Blue tint at base (higher temp), yellow-orange at tips
  - Density attribute controls visibility
- Steel tray: Metallic with roughness 0.4-0.6, subtle rust coloring
- Concrete: Diffuse with procedural noise for texture, roughness 0.8+
- Ethanol liquid (if visible): Near-clear with IOR 1.36, slight reflection

BACKGROUND:
Dark industrial gray background. Not pure black — enough ambient to see steel tray edges and concrete texture. Suggests workshop environment without requiring detailed modeling.

Research documentation, patterns, and APIs to find the optimal approach for animated fire emitters that can simulate flame front propagation across a liquid surface."""


async def main():
    print("=" * 70)
    print("FUEL IGNITION TEST - FLAME FRONT PROPAGATION")
    print("Testing Mantaflow GAS domain with animated fire emitter")
    print("=" * 70)
    print()

    # Create request
    request = AssetRequest(
        asset_name=f"fuel_ignition_{timestamp}",
        description=FUEL_IGNITION_DESCRIPTION,
        effect_type=EffectType.FIRE,
        resolution=128,  # Good balance for fire
        frame_start=1,
        frame_end=96,  # 4 seconds
        quality_threshold=60.0,
        max_iterations=3,
        semantic_query="Mantaflow fire simulation, flame spread animation, animated emitter, burning_rate, flame_smoke, vorticity, blackbody emission",
    )

    # Initialize orchestrator
    orchestrator = BlenderVFXOrchestrator()
    await orchestrator.initialize()

    print(f"Asset: {request.asset_name}")
    print(f"Effect Type: {request.effect_type.value}")
    print(f"Max Iterations: {request.max_iterations}")
    print(f"Frame Range: {request.frame_start}-{request.frame_end}")
    print(f"Resolution: {request.resolution}")
    print(f"Quality Threshold: {request.quality_threshold}")
    print()

    # Run pipeline with tracing
    try:
        with trace("Fuel Ignition Test", group_id=request.asset_name):
            result = await orchestrator.create_asset_pipeline(request)

        print("\n" + "=" * 70)
        print("FUEL IGNITION TEST COMPLETE")
        print("=" * 70)
        print(f"Status: {result.status.value}")
        print(f"Best Score: {result.best_score:.1f}")
        print(f"Total Iterations: {result.current_iteration}")
        print()

        if result.final_render_path:
            print(f"Final Render: {result.final_render_path}")

        print()
        print("Iteration History:")
        for it in result.iterations:
            status = "✓" if it.passed else "✗"
            print(f"  [{status}] Iteration {it.iteration}: {it.score:.1f} - {it.primary_issue or 'No issues'}")

        if result.current_issues:
            print()
            print("Final Issues:")
            for issue in result.current_issues:
                print(f"  - {issue}")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        await orchestrator.close()

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
