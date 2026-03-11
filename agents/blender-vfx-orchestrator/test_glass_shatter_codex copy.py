#!/usr/bin/env python3
"""
Glass shatter E2E test — rigid body + cell fracture physics.

Tests a completely different physics pipeline from the wine pour (Mantaflow liquid).
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
log_file = f"traces/glass_shatter_codex_{timestamp}.jsonl"
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


GLASS_SHATTER_DESCRIPTION = """SCENE DESCRIPTION:
Inside a derelict industrial warehouse at night. A large single-pane glass window (~2m wide, ~1.5m tall) is mounted in a rusted steel frame set into an aged concrete wall. A heavy red clay brick flies in from the left side of frame at high speed and strikes the center of the window. The glass explodes outward — shards erupt from the impact point, catching light as they tumble and spin through the air. The brick continues through the window trailing glass fragments. Remaining glass attached to the frame cracks and collapses.

MOOD AND LOOK:
Gritty, cinematic industrial atmosphere. Cold blue-white moonlight streams through warehouse skylights above, creating hard directional shadows. A single sodium-vapor security light (warm orange, mounted high on the left wall) provides harsh rim lighting that makes the glass fragments glow as they scatter. The moment of impact should feel violent and percussive.

CAMERA:
Medium-wide shot from inside the warehouse, facing the window. Camera is ~4m back from the window, at chest height (1.3m), angled to catch rim lighting on the glass spray. Static camera. 50mm lens equivalent. Deep focus.

PHYSICAL DETAILS:
- Window glass: Float glass, 6mm thick, clear with subtle green tint at edges
- Steel frame: I-beam profile, ~8cm wide, rust-brown with peeling gray paint
- Concrete wall: Aged industrial, gray with moisture staining
- Brick: Red clay engineering brick, ~2.5kg
- Warehouse floor: Poured concrete with oil stains
- Glass shards should look like real broken glass — irregular, sharp, varied sizes

MOTION/TIMING DETAILS:
Brick enters frame at ~15 m/s from screen-left. Impact around frame 5. Frames 5-15: main shattering event. Frames 15-30: secondary collapse, fragments bouncing. Frame 40+: settling. Render frame 12 for peak action.

HARD CONSTRAINTS:
- Physics simulation: Blender Rigid Body
- Bake frames: 1-60
- Renderer: Cycles GPU
- Samples: 256 max
- Render frame: 12
- Reference: None

Research documentation, patterns, and APIs to find the best approach for realistic glass shattering simulation."""


async def main():
    print("=" * 70)
    print("GLASS SHATTER GPT-5.4 TEST — Rigid Body + Cell Fracture")
    print("2 iterations, codex_upgrade preset, tracing enabled")
    print(f"Trace log: {log_file}")
    print("=" * 70)
    print()

    request = AssetRequest(
        asset_name="glass_shatter_codex",
        description=GLASS_SHATTER_DESCRIPTION,
        effect_type=EffectType.SHATTER,  # Correct type for glass shattering/destruction
        resolution=96,
        frame_start=1,
        frame_end=60,
        quality_threshold=60.0,
        max_iterations=2,
        semantic_query="rigid body glass shatter break fragments realistic destruction simulation",
    )

    orchestrator = BlenderVFXOrchestrator()
    await orchestrator.initialize()

    print(f"Asset: {request.asset_name}")
    print(f"Effect Type: {request.effect_type.value}")
    print(f"Max Iterations: {request.max_iterations}")
    print(f"Quality Threshold: {request.quality_threshold}")
    print()

    try:
        with trace("Glass Shatter Codex E2E Test", group_id=request.asset_name):
            result = await orchestrator.create_asset_pipeline(request)

        print("\n" + "=" * 70)
        print("GLASS SHATTER GPT-5.4 TEST COMPLETE")
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

        # Decomposition verification
        print()
        print("=" * 70)
        print("DECOMPOSITION GATE VERIFICATION")
        print("-" * 40)
        from phases import research as _r, execution as _e
        import coordinator_agents as _c
        import models.pipeline_models as _m
        print("  [PASS] phases.research imported")
        print("  [PASS] phases.execution imported")
        print("  [PASS] coordinator_agents imported")
        print("  [PASS] models.pipeline_models imported")

        # Check artifacts
        artifacts_dir = Path("sessions/artifacts")
        if artifacts_dir.exists():
            sessions = list(artifacts_dir.iterdir())
            matching = [s for s in sessions if "glass_shatter" in s.name]
            if matching:
                session_dir = matching[-1]
                artifacts = list(session_dir.glob("*.json"))
                print(f"\n  Artifacts written ({len(artifacts)}):")
                for a in sorted(artifacts):
                    size = a.stat().st_size
                    print(f"    {a.name} ({size:,} bytes)")

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
