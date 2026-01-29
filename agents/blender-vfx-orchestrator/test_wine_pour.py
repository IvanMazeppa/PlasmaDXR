"""
Wine pour 3-iteration test for Blender VFX Orchestrator.

Tests liquid simulation with enhanced prompt for wine pouring scene.
"""

import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set tracing and model config
os.environ["ORCHESTRATOR_MODEL"] = "gpt-5-mini"
os.environ["VERBOSE_TRACING"] = "1"

# Configure preset
from config.agent_config import AgentConfigManager, PresetConfig, reset_config

WINE_PRESET = PresetConfig(
    name="wine_pour_3iter",
    description="3-iteration wine pour test with enhanced prompt",
    default_model="gpt-5-mini",
    reasoning_effort="low",
    verbosity="low",
    max_output_tokens=2000,
    max_turns=6,
    max_iterations=3,
    verbose=True
)

# Enable verbose tracing
from tracing import enable_verbose_tracing
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"traces/wine_pour_3iter_{timestamp}.jsonl"
enable_verbose_tracing(
    log_file=log_file,
    console_output=True,
    include_span_data=True,
)

from orchestrator import BlenderVFXOrchestrator
from models.shared_context import AssetRequest, EffectType
from agents import trace


WINE_DESCRIPTION = """SCENE DESCRIPTION:
A stream of deep red wine pours from an unseen bottle (positioned just above frame) into an elegant stemmed wine glass resting on a dark wooden surface. The pour is steady but vigorous — not a delicate trickle, but a confident stream that creates dynamic splashing as it impacts the rising liquid surface. As the glass fills past the halfway point, wine droplets leap from the surface on impact, some catching the warm light as they arc through the air. A few droplets escape over the rim and trail down the outside of the glass.

MOOD AND LOOK:
Warm, intimate atmosphere evoking a cozy evening scene. Rich amber/golden key lighting from one side (as if from a nearby fireplace or candle cluster) creates dramatic highlights on the glass rim and wine surface, with the deep burgundy liquid glowing where light passes through it. The background falls into soft, warm darkness. The wine should exhibit convincing subsurface scattering — appearing darker at depth, more translucent and ruby-red where thin against the glass walls. Visible caustics pooling on the table surface beneath the glass.

CAMERA:
Close-up, slightly low angle looking up at the glass to emphasize elegance. Shallow depth of field optional.

PHYSICAL DETAILS:
- Wine glass: Standard Bordeaux style, clear glass with subtle reflections
- Wine color: Deep burgundy/cabernet red with ruby highlights in thin areas
- Pour stream: ~8mm diameter, continuous with slight surface tension wobble
- Table surface: Dark polished wood with subtle reflection of the glass base

HARD CONSTRAINTS:
- Fluid simulation: Blender Mantaflow liquid domain (FLIP solver, not shader-only)
- Domain type: LIQUID with mesh generation enabled
- Renderer: Cycles GPU
- Samples: 256 max
- Frame range: 1-60 (captures fill from ~1/4 to ~3/4 full)
- Resolution: 64-96 for good splash detail
- use_plane_init: True (for inflow emitter)
- cache_type: ALL"""


async def run_wine_pour_test():
    print("=" * 70)
    print("WINE POUR 3-ITERATION TEST")
    print(f"Model: {WINE_PRESET.default_model} | Iterations: {WINE_PRESET.max_iterations}")
    print(f"Trace log: {log_file}")
    print("=" * 70)
    print()

    # Initialize Orchestrator
    print("[1] Initializing orchestrator...")
    orchestrator = BlenderVFXOrchestrator()

    # Inject custom config
    from config import agent_config
    agent_config._global_config = AgentConfigManager(preset=WINE_PRESET)

    await orchestrator.initialize()
    print("    ✓ Orchestrator initialized")

    # Create test request
    request = AssetRequest(
        asset_name="wine_pour_3iter",
        description=WINE_DESCRIPTION,
        effect_type=EffectType.WATER,  # Using WATER for liquid simulations
        resolution=64,
        frame_start=1,
        frame_end=60,
        quality_threshold=45.0,
        max_iterations=3,
    )

    # Run the pipeline with outer trace
    print(f"\n[2] Running 3-iteration pipeline for '{request.asset_name}'...")
    print(f"    Effect type: {request.effect_type.value}")
    print(f"    Quality threshold: {request.quality_threshold}")
    print()

    try:
        # Single outer trace per SDK protocol
        with trace("Wine Pour 3-Iteration Test", group_id=request.asset_name):
            result = await orchestrator.create_asset_pipeline(request)

        print("\n" + "=" * 70)
        print("WINE POUR TEST COMPLETE")
        print("=" * 70)
        print(f"Status: {result.status.value}")
        print(f"Best Score: {result.best_score:.1f}")
        print(f"Iterations Run: {result.current_iteration}")
        print(f"Trace log: {log_file}")

        if result.best_render_path:
            print(f"Best Render: {result.best_render_path}")
        if result.output_dir:
            print(f"Output Dir: {result.output_dir}")

        # Check artifact gates results
        if hasattr(result, 'artifact_gate_results') and result.artifact_gate_results:
            print(f"\nArtifact Gates:")
            for gate in result.artifact_gate_results:
                status = "✓" if gate.passed else "✗"
                print(f"  {status} {gate.gate_name}: {gate.reason}")

        print("=" * 70)
        return result

    except Exception as e:
        print(f"\n[!] Error: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        await orchestrator.close()


if __name__ == "__main__":
    asyncio.run(run_wine_pour_test())
