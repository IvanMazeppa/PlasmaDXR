#!/usr/bin/env python3
"""
Oil Pour Test - 3 Iterations via create_asset_pipeline()

Tests the Mantaflow LIQUID domain with a viscous oil pour simulation.
Validates P0 hallucination prevention changes with a different effect type.
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
log_file = f"traces/oil_pour_3iter_{timestamp}.jsonl"
enable_verbose_tracing(
    log_file=log_file,
    console_output=True,
    include_span_data=True,
)
print(f"[SETUP] Tracing enabled: {log_file}")

from orchestrator import BlenderVFXOrchestrator
from models.shared_context import AssetRequest, EffectType
from agents import trace


OIL_POUR_DESCRIPTION = """SCENE DESCRIPTION:
A stream of golden extra-virgin olive oil pours from an unseen vessel above frame, descending in a smooth, continuous ribbon into a clear cylindrical glass container positioned center-frame. The oil enters the glass from approximately 15cm above the rim, creating an unbroken column that thickens slightly as it approaches the liquid surface below. As the stream meets the rising pool of oil in the glass, it creates subtle ripples that propagate outward in concentric circles. The oil level rises visibly over the duration of the shot, filling approximately the bottom third to two-thirds of the glass. Small air bubbles occasionally form where the stream penetrates the surface, rising slowly through the viscous liquid before reaching the top.

MOOD AND LOOK:
Warm, appetizing, culinary atmosphere evocative of high-end food photography or a cooking show intro. Strong backlighting from a soft golden source creates a luminous glow through the oil and glass, emphasizing the liquid's rich amber-gold color and transparency. The scene reads as premium, artisanal, Mediterranean — suggesting quality ingredients and slow-food philosophy. Soft fill light from front-left prevents the shadows from going completely black while maintaining drama. The overall palette is warm golds, soft creams, and clean whites with deep shadows that enhance the product-shot aesthetic.

CAMERA:
Medium close-up framing the entire glass with comfortable headroom for the incoming oil stream. Eye-level camera angle, perpendicular to the glass surface to minimize distortion and maximize clarity of the liquid inside. Shallow depth of field with focus locked on the glass — the stream above softly blurs as it enters from out of focus. Static camera on tripod for stability befitting a product shot. Focal length equivalent to 85mm for slight compression that flatters the cylindrical glass shape.

PHYSICAL DETAILS:
- Oil: Extra-virgin olive oil, deep golden-amber color with slight green undertones
- Oil viscosity: Medium-high (kinematic viscosity ~80 centistokes at room temperature)
- Oil stream: ~8mm diameter, continuous unbroken column, slight narrowing due to gravity acceleration
- Glass container: Clear borosilicate glass cylinder, 8cm diameter, 12cm tall, 3mm wall thickness
- Glass capacity: ~500ml, filling from ~150ml to ~350ml during shot
- Pour height: Stream enters frame 15cm above glass rim
- Surface behavior: Gentle ripples, occasional small bubble (2-3mm) where stream enters
- Light interaction: Caustics visible on the surface below glass, refraction through curved glass walls

MOTION/TIMING DETAILS:
The pour should feel graceful and deliberate — not rushed. The oil stream maintains consistent flow rate throughout. Surface ripples should propagate naturally, dampening as they reach the glass walls. The fill rate should be visible but not dramatic — approximately 200ml over 48 frames at 24fps (2 seconds real-time). Bubbles, when they form, should rise at approximately 1-2cm/second due to oil viscosity.

HARD CONSTRAINTS:
- Fluid simulation: Blender Mantaflow liquid domain
- Domain type: LIQUID
- Resolution: 96-128 for good surface detail
- Viscosity: Use viscosity_base = 5.0, viscosity_exponent = 3 for olive oil behavior
- use_plane_init: True (CRITICAL for inflow emitters)
- Inflow type: INFLOW with initial velocity downward (-Z direction)
- Surface tension: Enable mesh_smoothness for proper liquid surface
- Renderer: Cycles GPU
- Samples: 256 max
- Frame range: 1-48
- cache_type: ALL
- Adaptive domain: Enabled to optimize simulation bounds
- Reference: None

MATERIAL REQUIREMENTS:
- Oil material: Principled BSDF with transmission = 1.0, IOR = 1.47
- Oil color: RGB approximately (0.85, 0.65, 0.15) with subsurface scattering
- Glass material: Principled BSDF with transmission = 1.0, IOR = 1.52, roughness = 0.0
- Enable caustics for realistic light behavior through liquid and glass

Research documentation, patterns, and APIs to find the optimal starting approach for Mantaflow liquid simulation with viscous fluids and glass containers."""


async def main():
    print("=" * 70)
    print("OIL POUR TEST - 3 ITERATIONS")
    print("Testing Mantaflow LIQUID domain with viscous fluid")
    print("=" * 70)
    print()

    # Create request
    request = AssetRequest(
        asset_name=f"oil_pour_{timestamp}",
        description=OIL_POUR_DESCRIPTION,
        effect_type=EffectType.WATER,  # WATER covers all liquid simulations
        resolution=96,
        frame_start=1,
        frame_end=48,
        quality_threshold=60.0,
        max_iterations=3,
        semantic_query="Mantaflow liquid simulation, FLIP solver, viscosity, FluidDomainSettings, FluidFlowSettings, inflow",
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
        with trace("Oil Pour 3-Iteration Test", group_id=request.asset_name):
            result = await orchestrator.create_asset_pipeline(request)

        print("\n" + "=" * 70)
        print("OIL POUR TEST COMPLETE")
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
