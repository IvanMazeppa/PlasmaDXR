"""
Real-world reference image comparison test.

Compares actual renders from previous agent tests against web reference images.
"""

import asyncio
import json
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from agents import Agent, Runner, trace
from specialized_agents.quality_analyst import create_quality_analyst

# Paths
PROJECT_ROOT = Path("/home/maz3ppa/projects/PlasmaDXR")
REFERENCE_DIR = PROJECT_ROOT / "assets/reference_images"
RENDER_DIR = PROJECT_ROOT / "build/vdb_output"


async def test_explosion_comparison():
    """Compare explosion render against web reference images."""
    print("=" * 70)
    print("TEST: EXPLOSION RENDER vs REFERENCE IMAGES")
    print("=" * 70)
    print()

    # Explosion render from agent system
    render_path = RENDER_DIR / "dramatic_explosion_e2e_v4/render_0045.png"

    # Reference images
    references = list((REFERENCE_DIR / "explosions").glob("explosion_reference_web_*.jpg"))

    print(f"[1] Render: {render_path}")
    print(f"    Exists: {render_path.exists()}")
    print()
    print(f"[2] Available references ({len(references)}):")
    for ref in references[:3]:
        print(f"    - {ref.name}")
    print()

    if not render_path.exists():
        print("ERROR: Render not found!")
        return False

    # Create Quality Analyst and run comparison
    agent = create_quality_analyst()

    print("[3] Running Quality Analyst comparison...")
    print()

    with trace("Explosion Reference Comparison"):
        result = await Runner.run(
            agent,
            f"""Evaluate this explosion render and compare it to reference images.

Render path: {render_path}
Effect type: explosion

Please:
1. First analyze the render quality with analyze_with_vision()
2. Find explosion reference images with find_reference_images('explosion')
3. Compare to the best reference with compare_to_reference()
4. Return a comprehensive quality assessment including:
   - Overall visual quality score
   - How the render compares to the reference
   - Specific gaps between render and reference
   - What improvements would close the gap""",
            max_turns=25
        )

    print("[4] Quality Analyst Response:")
    print("-" * 60)
    print(result.final_output[:2000] if len(str(result.final_output)) > 2000 else result.final_output)
    print("-" * 60)
    print()

    return True


async def test_sun_comparison():
    """Compare sun render against NASA SDO reference images."""
    print()
    print("=" * 70)
    print("TEST: SUN RENDER vs NASA SDO REFERENCE")
    print("=" * 70)
    print()

    # Sun render from agent system
    render_path = RENDER_DIR / "stellar_sun_test_v1/render_0030.png"

    # NASA SDO reference
    reference_path = REFERENCE_DIR / "star/SDO_11-14-25_0830UT_131redS-171_4k.jpg"

    print(f"[1] Render: {render_path}")
    print(f"    Exists: {render_path.exists()}")
    print()
    print(f"[2] Reference: {reference_path.name}")
    print(f"    Exists: {reference_path.exists()}")
    print()

    if not render_path.exists():
        print("ERROR: Render not found!")
        return False

    # Create Quality Analyst and run comparison
    agent = create_quality_analyst()

    print("[3] Running Quality Analyst comparison...")
    print()

    with trace("Sun Reference Comparison"):
        result = await Runner.run(
            agent,
            f"""Evaluate this sun/star render and compare it to NASA SDO imagery.

Render path: {render_path}
Effect type: sun

The reference is real NASA Solar Dynamics Observatory footage showing solar activity.

Please:
1. Analyze the render quality with analyze_with_vision()
2. Find sun/star reference images with find_reference_images('star')
3. Compare to NASA SDO reference with compare_to_reference()
4. Assess how realistic the render looks compared to real solar footage
5. Identify what makes the NASA footage look real vs what the render is missing""",
            max_turns=25
        )

    print("[4] Quality Analyst Response:")
    print("-" * 60)
    print(result.final_output[:2000] if len(str(result.final_output)) > 2000 else result.final_output)
    print("-" * 60)
    print()

    return True


async def test_recent_orchestrator_output():
    """Test the most recent orchestrator output against appropriate references."""
    print()
    print("=" * 70)
    print("TEST: RECENT ORCHESTRATOR OUTPUT EVALUATION")
    print("=" * 70)
    print()

    # Most recent orchestrator test
    render_path = RENDER_DIR / "test_fixed_orchestrator_v1_iter1/render_0012.png"

    print(f"[1] Render: {render_path}")
    print(f"    Exists: {render_path.exists()}")
    print()

    if not render_path.exists():
        print("ERROR: Render not found!")
        return False

    # Create Quality Analyst
    agent = create_quality_analyst()

    print("[2] Running full Quality Analyst evaluation...")
    print()

    with trace("Orchestrator Output Evaluation"):
        result = await Runner.run(
            agent,
            f"""Evaluate this VFX render from a recent orchestrator test.

Render path: {render_path}

I don't know what effect type this is - please:
1. First analyze what type of effect this appears to be using analyze_with_vision()
2. Based on the effect type, find appropriate reference images
3. If references exist, compare the render to them
4. Provide a comprehensive quality assessment with:
   - What effect type you think this is
   - Overall quality score
   - Visual issues identified
   - How it compares to professional references (if found)
   - Specific suggestions for improvement""",
            max_turns=25
        )

    print("[3] Quality Analyst Response:")
    print("-" * 60)
    print(result.final_output[:2500] if len(str(result.final_output)) > 2500 else result.final_output)
    print("-" * 60)
    print()

    return True


if __name__ == "__main__":
    print()
    print("REAL REFERENCE IMAGE COMPARISON TEST")
    print("=" * 70)
    print()

    # Test 1: Explosion comparison
    test1 = asyncio.run(test_explosion_comparison())

    # Test 2: Sun comparison
    test2 = asyncio.run(test_sun_comparison())

    # Test 3: Recent orchestrator output
    test3 = asyncio.run(test_recent_orchestrator_output())

    print()
    print("=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    print(f"  Explosion comparison: {'Completed ✓' if test1 else 'Failed ✗'}")
    print(f"  Sun comparison:       {'Completed ✓' if test2 else 'Failed ✗'}")
    print(f"  Orchestrator output:  {'Completed ✓' if test3 else 'Failed ✗'}")
    print("=" * 70)
