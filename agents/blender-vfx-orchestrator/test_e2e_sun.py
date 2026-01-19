"""
End-to-end test: Sun/Solar VFX with NASA SDO Reference Images.

Tests the full pipeline with high-quality NASA Solar Dynamics Observatory
reference imagery showing solar flares, prominences, and surface activity.

Reference dataset:
- 840 frames from NASA SDO at 2048p resolution
- Shows sun rotation and activity evolution over time
- Location: assets/reference_images/star/Eruptions_20241008_Activity_2048p30/
"""

import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Verify API key is set
if not os.getenv("OPENAI_API_KEY"):
    print("ERROR: OPENAI_API_KEY not found in environment or .env file")
    sys.exit(1)

# Use gpt-5.2 for full reasoning capabilities
os.environ.setdefault("ORCHESTRATOR_MODEL", "gpt-5.2")

from models.shared_context import AssetRequest, EffectType, SharedContext
from orchestrator import BlenderVFXOrchestrator, create_session_from_request, generate_session_id
from agents import Runner, trace, enable_verbose_stdout_logging

# Enable verbose logging to see agent behavior
enable_verbose_stdout_logging()

# Reference image paths
PROJECT_ROOT = Path("/home/maz3ppa/projects/PlasmaDXR")
NASA_REFERENCE_DIR = PROJECT_ROOT / "assets/reference_images/star/Eruptions_20241008_Activity_2048p30"


def get_nasa_reference_sample() -> str:
    """Get a sample NASA reference image with good solar activity."""
    # Pick a frame from the middle of the sequence for good activity
    candidates = [
        "frame_00200.jpg",  # Mid-sequence
        "frame_00300.jpg",
        "frame_00400.jpg",
        "frame_00100.jpg",
    ]

    for candidate in candidates:
        path = NASA_REFERENCE_DIR / candidate
        if path.exists():
            return str(path)

    # Fallback to first available
    frames = sorted(NASA_REFERENCE_DIR.glob("frame_*.jpg"))
    if frames:
        return str(frames[len(frames) // 2])  # Middle frame

    return None


async def test_sun_e2e():
    """Run end-to-end test of sun VFX generation with NASA reference."""
    print("=" * 70)
    print("BLENDER VFX ORCHESTRATOR - SUN/SOLAR E2E TEST")
    print("Using NASA Solar Dynamics Observatory Reference Imagery")
    print("=" * 70)
    print()

    # Check reference images exist
    if not NASA_REFERENCE_DIR.exists():
        print(f"ERROR: NASA reference directory not found:")
        print(f"  {NASA_REFERENCE_DIR}")
        return None

    frame_count = len(list(NASA_REFERENCE_DIR.glob("frame_*.jpg")))
    print(f"Reference Dataset:")
    print(f"  - Location: {NASA_REFERENCE_DIR}")
    print(f"  - Frames: {frame_count}")
    print()

    # Get a reference image
    reference_path = get_nasa_reference_sample()
    if reference_path:
        print(f"  - Selected Reference: {Path(reference_path).name}")
    else:
        print("  - WARNING: No reference image found, proceeding without")
    print()

    # Create sun asset request
    request = AssetRequest(
        asset_name="nasa_sun_test_v1",
        description="""Create a realistic sun with active surface features.

Key features to match from NASA SDO reference:
1. SOLAR CORONA: Bright outer glow with gradual falloff
2. SURFACE GRANULATION: Mottled texture from convection cells
3. PROMINENCES: Bright arcs extending from the limb (edge)
4. ACTIVE REGIONS: Brighter patches indicating magnetic activity
5. LIMB DARKENING: Edge appears slightly darker than center

The reference shows real solar activity captured by NASA's Solar Dynamics
Observatory. The volumetric simulation should capture the hot plasma emission
and the characteristic orange-red color temperature of the sun's surface.""",
        effect_type=EffectType.SUN,
        reference_path=reference_path,
        resolution=96,  # Good quality
        frame_start=1,
        frame_end=50,  # Animation to show activity
        quality_threshold=55.0,  # Realistic threshold for sun
        max_iterations=5,
    )

    print(f"Asset Request:")
    print(f"  - Asset: {request.asset_name}")
    print(f"  - Effect: {request.effect_type.value}")
    print(f"  - Resolution: {request.resolution}")
    print(f"  - Frames: {request.frame_start}-{request.frame_end}")
    print(f"  - Quality Threshold: {request.quality_threshold}")
    print(f"  - Max Iterations: {request.max_iterations}")
    print(f"  - Reference: {Path(reference_path).name if reference_path else 'None'}")
    print()

    # Initialize orchestrator
    print("[1/4] Initializing orchestrator...")
    orchestrator = BlenderVFXOrchestrator()
    await orchestrator.initialize()
    budget = orchestrator.get_budget_status()
    print(f"  ✓ Orchestrator initialized with 5 agents")
    print(f"  ✓ Budget: ${budget['total_remaining']:.2f} remaining")
    print()

    # Test context creation
    print("[2/4] Creating session context...")
    session_id = generate_session_id(request.asset_name)
    context = create_session_from_request(request, session_id)
    print(f"  ✓ Session ID: {session_id}")
    print(f"  ✓ Effect Type: {context.session.request.effect_type.value}")
    if reference_path:
        print(f"  ✓ Reference loaded: {Path(reference_path).name}")
    print()

    # Run the orchestrator with CODE-BASED PIPELINE
    print("[3/4] Running CODE-BASED PIPELINE orchestration...")
    print("  (Python controls: Research → Script → Execute → Evaluate → Learn → Loop)")
    print()
    print("-" * 70)

    try:
        # Use create_asset_pipeline() for deterministic code-based orchestration
        # This ensures the pipeline sequence is followed exactly (no LLM deviation)
        result = await orchestrator.create_asset_pipeline(request)

        print("-" * 70)
        print()
        print("[4/4] Results:")
        print(f"  - Status: {result.status.value}")
        print(f"  - Iterations completed: {result.current_iteration}")
        print(f"  - Best score: {result.best_score:.1f}")
        print(f"  - Best iteration: {result.best_iteration}")

        if result.final_render_path:
            print(f"  - Final render: {result.final_render_path}")
        if result.final_vdb_dir:
            print(f"  - VDB directory: {result.final_vdb_dir}")
        if result.current_issues:
            print(f"  - Issues: {result.current_issues[:200]}...")

        # Show iteration history
        if result.iterations:
            print()
            print("Iteration History:")
            for it in result.iterations:
                status = "✓" if it.passed else "✗"
                print(f"  [{it.iteration}] {status} Score: {it.score:.1f}")
                # Access primary_issue from nested quality model
                primary_issue = it.quality.primary_issue if it.quality else None
                if primary_issue:
                    issue_short = primary_issue[:60] + "..." if len(primary_issue) > 60 else primary_issue
                    print(f"      Issue: {issue_short}")
                # Access LPIPS from nested quality model if available
                if it.quality and it.quality.lpips_score is not None:
                    print(f"      LPIPS: {it.quality.lpips_score:.3f}")

        print()
        print("=" * 70)
        print("TEST COMPLETE")
        print("=" * 70)

        # Quality assessment
        if result.best_score >= 70:
            print("EXCELLENT: High quality sun render achieved!")
        elif result.best_score >= 55:
            print("GOOD: Quality threshold met, room for improvement")
        elif result.best_score >= 40:
            print("ACCEPTABLE: Basic sun render, needs iteration")
        else:
            print("NEEDS WORK: Significant quality issues remain")

        return result

    except Exception as e:
        print(f"\n  ✗ Error during orchestration: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        await orchestrator.close()


async def quick_quality_test():
    """Quick test of just the Quality Analyst against NASA reference."""
    print("=" * 70)
    print("QUICK TEST: Quality Analyst vs NASA Reference")
    print("=" * 70)
    print()

    from tools.asset_evaluator_tools import (
        _find_reference_images_impl,
        _analyze_with_vision_impl,
    )

    # Find NASA sun references
    print("[1] Finding NASA sun references...")
    refs_json = await _find_reference_images_impl("sun", limit=3)
    import json
    refs = json.loads(refs_json)
    print(f"  Found {refs['count']} references")

    if refs["results"]:
        ref_path = refs["results"][0]["path"]
        print(f"  Selected: {Path(ref_path).name}")
        print()

        # Analyze the reference image itself
        print("[2] Analyzing NASA reference quality...")
        analysis = await _analyze_with_vision_impl(
            render_path=ref_path,
            analysis_type="quality",
            effect_type="sun"
        )
        result = json.loads(analysis)
        print(f"  Score: {result.get('score', 'N/A')}")
        print(f"  Assessment: {result.get('overall_assessment', '')[:200]}...")
        print()

        return result

    return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sun VFX E2E Test")
    parser.add_argument("--quick", action="store_true", help="Run quick quality test only")
    args = parser.parse_args()

    if args.quick:
        result = asyncio.run(quick_quality_test())
    else:
        result = asyncio.run(test_sun_e2e())

    # Exit with appropriate code
    if result:
        if isinstance(result, dict):
            # Quick test result
            sys.exit(0 if result.get("score", 0) > 50 else 1)
        elif hasattr(result, "status"):
            # Full test result
            sys.exit(0 if result.status.value in ("passed", "max_iterations") else 1)

    sys.exit(1)
