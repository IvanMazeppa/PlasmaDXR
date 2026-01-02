#!/usr/bin/env python3
"""
Test script for Blender VFX Orchestrator - Explosion Scene

This tests the full pipeline:
1. Script generation via script-generator MCP
2. Blender execution via blender-executor MCP
3. Quality evaluation via asset-evaluator MCP
4. Iteration based on VFX quality scores
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from orchestrator import BlenderOrchestratorAgent
from state import AssetRequest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("test-explosion")


async def main():
    """Run explosion scene test."""

    print("=" * 80)
    print("BLENDER VFX ORCHESTRATOR - EXPLOSION SCENE TEST")
    print("=" * 80)
    print()

    # Reference images for ground truth comparison
    reference_images = [
        "assets/reference_images/explosions/explosion_reference_web_1.jpg",
        "assets/reference_images/explosions/explosion_reference_web_2.jpg",
        "assets/reference_images/explosions/explosion_reference_web_3.jpg",
        "assets/reference_images/explosions/explosion_reference_web_4.jpg",
    ]

    # Use first reference for LPIPS/evaluation
    reference_path = str(Path(__file__).parent.parent.parent / reference_images[0])

    print(f"Reference image: {reference_path}")
    print()

    # Create orchestrator
    agent = BlenderOrchestratorAgent()

    print(f"Orchestrator Status:")
    print(f"  Trust Score: {agent.autonomy.get_trust_score():.2f}")
    print(f"  Autonomy Level: {agent.autonomy.get_level().value}")
    print(f"  Max Budget: ${agent.guardrail_config.cost_limits.per_session:.2f}")
    print()

    try:
        # Start the agent
        print("Starting orchestrator agent...")
        await agent.start()
        print("✓ Agent started")
        print()

        # Create explosion asset
        print("=" * 80)
        print("CREATING EXPLOSION ASSET")
        print("=" * 80)

        result = await agent.create_asset(
            asset_name="test_explosion_v1",
            effect_type="pyro",
            description="""A dramatic fiery explosion with:
- Intense orange and red flames at the core
- Rising mushroom cloud shape
- Dark smoke billowing outward
- Bright hot center fading to cooler edges
- Debris and particles at the base
Similar to a gasoline or fuel explosion.""",
            reference_path=reference_path,
            semantic_query="dramatic fiery explosion with orange flames and dark smoke",
            resolution=96,  # Start with moderate resolution
            frame_end=50,   # 50 frames of animation
            technique_name=None  # Let it choose
        )

        print()
        print("=" * 80)
        print("RESULT")
        print("=" * 80)
        print(f"Status: {result.get('status', 'unknown')}")
        print(f"Best Score: {result.get('best_score', 0):.1f}/100")
        print(f"Iterations: {result.get('iterations', 0)}")
        print(f"Final Stage: {result.get('final_stage', 'unknown')}")

        if result.get('output_path'):
            print(f"Output: {result.get('output_path')}")

        if result.get('errors'):
            print(f"Errors: {result.get('errors')}")

    except KeyboardInterrupt:
        print("\n\n⚠ Test interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Stop agent
        print("\nStopping orchestrator...")
        await agent.stop()
        print("✓ Agent stopped")


if __name__ == "__main__":
    asyncio.run(main())
