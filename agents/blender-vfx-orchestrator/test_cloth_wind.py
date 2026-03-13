#!/usr/bin/env python3
"""
Cloth wind E2E test — cloth physics simulation.

Tests cloth simulation pipeline with truth pack type resolution fixes.
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
log_file = f"traces/cloth_wind_{timestamp}.jsonl"
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


CLOTH_WIND_DESCRIPTION = """SCENE DESCRIPTION:
An outdoor backyard scene in bright daylight. A simple rope clothes line is strung between two weathered wooden posts about 3 meters apart, standing about 1.8 meters tall. A large white cotton bed sheet is draped over the line and pinned at two points with wooden clothespins. Strong gusting wind catches the fabric, causing dramatic billowing, rippling waves, and occasional snapping motions. The sheet flaps violently in the wind.

MOOD AND LOOK:
Bright, sunny afternoon. Clear blue sky with a few wispy clouds. Warm natural sunlight from above-right creates soft shadows on the fabric folds. Clean, domestic, slightly nostalgic atmosphere.

CAMERA:
Medium shot from the side, about 4 meters away at eye level (1.6m). Static camera. 50mm lens equivalent. The full clothes line and sheet are visible in frame.

PHYSICAL DETAILS:
- Sheet: White cotton, ~2m x 2m, slightly translucent where light passes through thin folds
- Clothes line: Simple rope/cord, slight sag under the sheet weight
- Posts: Wooden, ~10cm diameter, slightly weathered gray
- Clothespins: Simple wooden spring-clip type
- Ground: Green grass lawn

MOTION/TIMING DETAILS:
Wind blows from screen-left at ~8-12 m/s with gusts. Sheet starts mostly hanging, wind picks up by frame 10. Peak billowing frames 20-40. Render frame 30 for peak drama.

HARD CONSTRAINTS:
- Physics simulation: Blender Cloth
- Bake frames: 1-60
- Renderer: Cycles GPU
- Samples: 256 max
- Render frame: 30
- Reference: None

Research documentation, patterns, and APIs to find the best approach for realistic cloth simulation with wind forces."""


async def main():
    print("=" * 70)
    print("CLOTH WIND TEST — Cloth Physics Simulation")
    print("2 iterations, codex_upgrade preset, tracing enabled")
    print(f"Trace log: {log_file}")
    print("=" * 70)
    print()

    request = AssetRequest(
        asset_name="cloth_wind_sheet",
        description=CLOTH_WIND_DESCRIPTION,
        effect_type=EffectType.CLOTH,
        resolution=96,
        frame_start=1,
        frame_end=60,
        quality_threshold=60.0,
        max_iterations=2,
        semantic_query="white sheet blowing in strong wind on a clothes line outdoors cloth simulation",
    )

    orchestrator = BlenderVFXOrchestrator()
    await orchestrator.initialize()

    print(f"Asset: {request.asset_name}")
    print(f"Effect Type: {request.effect_type.value}")
    print(f"Max Iterations: {request.max_iterations}")
    print(f"Quality Threshold: {request.quality_threshold}")
    print()

    try:
        with trace("Cloth Wind E2E Test", group_id=request.asset_name):
            result = await orchestrator.create_asset_pipeline(request)

        print("\n" + "=" * 70)
        print("CLOTH WIND TEST COMPLETE")
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
