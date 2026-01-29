"""
2-iteration shakedown test for Blender VFX Orchestrator.

Purpose: Verify Phase 4 gate items:
- Spec-first completion without fallback
- Modification contract in iteration 2
- Doc grounding with real DocPath
- Trace correlation (single outer trace)
- Artifact gates integration
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

SHAKEDOWN_PRESET = PresetConfig(
    name="2iter_shakedown",
    description="2-iteration shakedown test for Phase 4 gate verification",
    default_model="gpt-5-mini",
    reasoning_effort="low",
    verbosity="low",
    max_output_tokens=2000,
    max_turns=5,
    max_iterations=2,  # 2 iterations for modification contract test
    verbose=True
)

# Enable verbose tracing
from tracing import enable_verbose_tracing
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"traces/shakedown_2iter_{timestamp}.jsonl"
enable_verbose_tracing(
    log_file=log_file,
    console_output=True,
    include_span_data=True,
)

from orchestrator import BlenderVFXOrchestrator
from models.shared_context import AssetRequest, EffectType
from agents import trace


async def run_2iter_shakedown():
    print("=" * 70)
    print("2-ITERATION SHAKEDOWN TEST")
    print(f"Model: {SHAKEDOWN_PRESET.default_model} | Iterations: {SHAKEDOWN_PRESET.max_iterations}")
    print(f"Trace log: {log_file}")
    print("=" * 70)
    print()

    # Initialize Orchestrator
    print("[1] Initializing orchestrator...")
    orchestrator = BlenderVFXOrchestrator()

    # Inject custom config
    from config import agent_config
    agent_config._global_config = AgentConfigManager(preset=SHAKEDOWN_PRESET)

    await orchestrator.initialize()
    print("    ✓ Orchestrator initialized")

    # Create test request - simple fire for reliability
    request = AssetRequest(
        asset_name="shakedown_fire_2iter",
        description="Simple campfire effect. Use Quick Smoke with fire type. Resolution 32, 10 frames.",
        effect_type=EffectType.FIRE,
        resolution=32,
        frame_start=1,
        frame_end=10,
        quality_threshold=40.0,  # Lower threshold to allow iteration 2
        max_iterations=2,
    )

    # Run the pipeline with outer trace
    print(f"\n[2] Running 2-iteration pipeline for '{request.asset_name}'...")
    print(f"    Quality threshold: {request.quality_threshold}")
    print()

    iteration_results = []

    try:
        # Single outer trace per SDK protocol
        with trace("2-Iteration Shakedown", group_id=request.asset_name):
            result = await orchestrator.create_asset_pipeline(request)

        print("\n" + "=" * 70)
        print("SHAKEDOWN COMPLETE")
        print("=" * 70)
        print(f"Status: {result.status.value}")
        print(f"Best Score: {result.best_score:.1f}")
        print(f"Iterations Run: {result.current_iteration}")
        print(f"Trace log: {log_file}")

        # Check artifact gates results
        if hasattr(result, 'artifact_gate_results'):
            print(f"\nArtifact Gates:")
            for gate in result.artifact_gate_results:
                status = "✓" if gate.passed else "✗"
                print(f"  {status} {gate.gate_name}: {gate.message}")

        # Check modification contract (iteration 2)
        if result.current_iteration >= 2:
            print(f"\n✓ Iteration 2 reached - modification contract can be verified in trace")
        else:
            print(f"\n⚠ Only {result.current_iteration} iteration(s) - check if quality threshold was met early")

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
    asyncio.run(run_2iter_shakedown())
