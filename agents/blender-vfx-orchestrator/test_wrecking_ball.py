#!/usr/bin/env python3
"""
Wrecking ball E2E test — rigid body destruction physics.

Tests rigid body pipeline with section patching (Wave 2).
Uses codex_upgrade preset with gpt-5.4 for code-critical agents.
"""

import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

# Force codex_upgrade preset
os.environ["ORCHESTRATOR_PRESET"] = "codex_upgrade"

# Enable verbose tracing BEFORE importing orchestrator
from tracing import enable_verbose_tracing
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"traces/wrecking_ball_{timestamp}.jsonl"
enable_verbose_tracing(
    log_file=log_file,
    console_output=True,
    include_span_data=True,
)
print(f"[SETUP] Tracing enabled: {log_file}")
print(f"[SETUP] Preset: codex_upgrade")

from config import get_config
from config.agent_config import reset_config
from orchestrator import BlenderVFXOrchestrator
from models.shared_context import AssetRequest, EffectType
from agents import trace

# Override HITL to non-interactive/autonomous so test doesn't block
reset_config()  # Clear any cached config
config = get_config()
config.preset.hitl_interactive = False
config.preset.hitl_autonomy_level = 3  # Autonomous — only L4 escalation pauses

# Log model assignments
print()
print("=" * 70)
print("AGENT MODEL ASSIGNMENTS (codex_upgrade preset)")
print("=" * 70)
for agent_name in [
    "script_writer", "modification_coordinator", "research_agent",
    "quality_analyst", "technique_coordinator", "quality_gate_coordinator",
    "learning_agent", "docs_expert", "executor",
]:
    settings = config.get_agent_settings(agent_name)
    rollout_marker = " <<< gpt-5.4" if settings.model == "gpt-5.4" else ""
    print(f"  {agent_name:30s} model={settings.model:16s} reasoning={settings.reasoning_effort}{rollout_marker}")
print("=" * 70)
print()


WRECKING_BALL_DESCRIPTION = """SCENE DESCRIPTION:
A heavy steel wrecking ball on a chain swings into a brick wall from the left side of frame. The wall is part of an old building being demolished. On impact the bricks shatter and scatter — fragments fly toward the camera as the ball punches through the wall. Dust and small debris fill the air around the breach.

MOOD AND LOOK:
Harsh midday demolition site. Overcast sky gives flat, even lighting with no strong shadows. Gritty, industrial, documentary feel. Muted earth tones — red-brown brick, gray concrete, dark steel.

CAMERA:
Medium shot from inside the building, facing the wall. The camera is a few meters back at chest height. Static camera, deep focus. Standard lens. The wrecking ball enters from outside, punching through the wall toward the viewer.

PHYSICAL DETAILS:
- Wrecking ball: Heavy dark steel sphere, roughly 0.8m diameter
- Chain: Steel chain links connecting ball to an off-screen crane arm
- Brick wall: Standard red clay bricks with gray mortar joints, roughly 3m wide x 2.5m tall x 0.25m thick
- Ground: Cracked concrete floor with rubble and dust

MOTION/TIMING DETAILS:
Ball swings in from the left. Impact at frame 8. Main destruction frames 8-20. Debris scattering and settling frames 20-50. Render frame 15 for peak destruction.

HARD CONSTRAINTS:
- Physics simulation: Blender Rigid Body
- Bake frames: 1-60
- Renderer: Cycles GPU
- Samples: 256 max
- Render frame: 15
- Reference: None

Research documentation, patterns, and APIs to find the best approach for rigid body destruction with fracturing."""


async def main():
    print("=" * 70)
    print("WRECKING BALL TEST — Rigid Body Destruction")
    print("3 iterations, codex_upgrade preset, tracing enabled")
    print(f"Trace log: {log_file}")
    print("=" * 70)
    print()

    request = AssetRequest(
        asset_name="wrecking_ball",
        description=WRECKING_BALL_DESCRIPTION,
        effect_type=EffectType.DESTRUCTION,
        resolution=96,
        frame_start=1,
        frame_end=60,
        quality_threshold=60.0,
        max_iterations=3,
        semantic_query="wrecking ball demolition rigid body brick wall destruction fracture simulation",
    )

    orchestrator = BlenderVFXOrchestrator()
    await orchestrator.initialize()

    print(f"Asset: {request.asset_name}")
    print(f"Effect Type: {request.effect_type.value}")
    print(f"Max Iterations: {request.max_iterations}")
    print(f"Quality Threshold: {request.quality_threshold}")
    print()

    try:
        with trace("Wrecking Ball E2E Test", group_id=request.asset_name):
            result = await orchestrator.create_asset_pipeline(request)

        print("\n" + "=" * 70)
        print("WRECKING BALL TEST COMPLETE")
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
