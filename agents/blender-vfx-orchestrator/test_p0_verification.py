#!/usr/bin/env python3
"""
P0 Verification Test - Single Iteration with Doc Query Enforcement

Tests that the P0 changes (re-enabled doc query enforcement) are working:
1. Script Writer must call doc query before write_script
2. Negative examples in instructions should help prevent hallucination

Run with: python test_p0_verification.py
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
log_file = f"traces/p0_verification_{timestamp}.jsonl"
enable_verbose_tracing(
    log_file=log_file,
    console_output=True,
    include_span_data=True,
)
print(f"[SETUP] Tracing enabled: {log_file}")

from orchestrator import BlenderVFXOrchestrator
from models.shared_context import AssetRequest, EffectType
from agents import trace


# Same prompt as candle test
CANDLE_DESCRIPTION = """SCENE DESCRIPTION:
A single, well-worn beeswax taper candle stands in an ornate brass candlestick holder on the corner of a mahogany reading desk. The flame burns with hypnotic slowness — each flicker and dance stretched into a graceful, meditative movement. The flame's core glows intense blue-white near the wick, transitioning through brilliant yellow to soft orange at the dancing tips. Thin wisps of smoke curl upward from the flame's peak, twisting lazily before dissipating into the darkness above. The wax near the flame has formed a shallow molten pool that occasionally trembles with the flame's movement, its surface catching warm reflections.

MOOD AND LOOK:
Deep, contemplative atmosphere of a Victorian-era private library at night. The candle is the sole light source, creating an intimate sphere of warm illumination that fades into rich, velvety darkness. Leather-bound book spines are barely visible at the edges of frame, their gold-leaf lettering catching occasional glints. Dust motes drift lazily through the warm updraft. The overall feeling is one of quiet scholarship and timeless solitude — the kind of scene where hours pass unnoticed. Color palette dominated by warm ambers, deep browns, and shadows that aren't quite black but carry hints of deep burgundy and forest green.

CAMERA:
Tight close-up on the flame and upper portion of the candle. Macro-style framing that makes the flame feel monumental. Very shallow depth of field — the flame is tack-sharp while the candlestick base and background books blur into soft shapes. Camera positioned slightly below flame height, looking subtly upward to emphasize the flame's upward reach. Static camera to let the flame's movement be the sole focus.

PHYSICAL DETAILS:
- Candle: Beeswax taper, cream/honey colored, ~2cm diameter, showing some drip texture from previous burns
- Candlestick: Tarnished brass with patina, ornate Victorian design
- Flame height: ~3-4cm, teardrop shape with dancing tip
- Flame structure: Blue core (1cm), yellow body, orange-tipped wisps that occasionally detach
- Smoke: Thin, wispy trails — NOT heavy smoke, just delicate thermal disturbance
- Wax pool: ~1.5cm diameter molten area at top, slightly concave
- Wick: Black, charred, slightly curved, ~5mm visible above wax

SLOW MOTION BEHAVIOR:
At slow motion, the flame should reveal its internal structure — the stable blue cone, the turbulent yellow convection, the way orange wisps form, stretch, and break away. The smoke should show clear vortex structures as it rises. The flame should "breathe" — expanding and contracting subtly as air currents affect it. Occasional larger flickers where the flame leans to one side before recovering.

HARD CONSTRAINTS:
- Fluid simulation: Blender Mantaflow smoke domain with fire enabled
- Domain type: GAS with use_noise for detail
- Fire reaction speed: Low (0.3-0.5) for slow, graceful burn
- Smoke amount: Minimal (0.05-0.1) — this is a clean-burning candle
- Vorticity: Medium (0.5-0.7) for visible turbulence patterns
- Temperature difference: High for realistic buoyancy
- Renderer: Cycles GPU
- Samples: 256 max
- Frame range: 1-120 (slow motion needs more frames)
- Resolution: 64-96 for good smoke detail
- Adaptive domain: Enabled to optimize simulation
- cache_type: ALL
- Reference: None

Research documentation, patterns, and APIs to find the optimal starting approach."""


async def main():
    print("=" * 70)
    print("P0 VERIFICATION TEST - SINGLE ITERATION")
    print("Testing doc query enforcement after P0 changes")
    print("=" * 70)
    print()

    # Create request with MAX 1 ITERATION
    request = AssetRequest(
        asset_name=f"p0_test_{timestamp}",
        description=CANDLE_DESCRIPTION,
        effect_type=EffectType.FIRE,
        resolution=64,  # Lower for faster test
        frame_start=1,
        frame_end=30,  # Shorter for faster test
        quality_threshold=60.0,
        max_iterations=1,  # SINGLE ITERATION ONLY
        semantic_query="candle flame, fire simulation, Mantaflow, FluidDomainSettings",
    )

    # Initialize orchestrator
    orchestrator = BlenderVFXOrchestrator()
    await orchestrator.initialize()

    print(f"Asset: {request.asset_name}")
    print(f"Effect Type: {request.effect_type.value}")
    print(f"Max Iterations: {request.max_iterations}")
    print(f"Quality Threshold: {request.quality_threshold}")
    print()
    print("WATCHING FOR:")
    print("  1. Doc query calls BEFORE write_script")
    print("  2. No hallucinated attributes (resolution_divisions, use_dissolve, etc.)")
    print()

    # Run pipeline with tracing
    try:
        with trace("P0 Verification Test", group_id=request.asset_name):
            result = await orchestrator.create_asset_pipeline(request)

        print("\n" + "=" * 70)
        print("P0 VERIFICATION TEST COMPLETE")
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

        # P0 Verification Summary
        print()
        print("=" * 70)
        print("P0 VERIFICATION SUMMARY")
        print("=" * 70)
        print(f"Trace file: {log_file}")
        print()
        print("Check the trace file for:")
        print("  - 'semantic_search_blender_docs' or 'blender_doc_search_bundle' calls")
        print("  - These should appear BEFORE any 'write_script' calls")
        print("  - If DocQueryRequiredError was raised, enforcement is working")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()

        # Check if it's the enforcement error (which would be a GOOD sign)
        if "DocQueryRequiredError" in str(type(e).__name__) or "without first querying" in str(e):
            print()
            print("=" * 70)
            print("P0 ENFORCEMENT WORKING!")
            print("The DocQueryRequiredError was raised, meaning the enforcement is active.")
            print("This is expected if the Script Writer tried to skip doc queries.")
            print("=" * 70)

        return 1

    finally:
        await orchestrator.close()

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
