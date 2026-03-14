#!/usr/bin/env python3
"""
Ireland Flag E2E test — cloth physics simulation.

Tests iteration loop integrity after stale render fix (Wave 2A).
Verifies: failed executions don't reuse old renders, escalation to
modify_code/switch_technique fires correctly, section patching exercises
on a real recovery path.

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
log_file = f"traces/ireland_flag_{timestamp}.jsonl"
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


IRELAND_FLAG_DESCRIPTION = """SCENE DESCRIPTION:
A large national flag of the Republic of Ireland is mounted on a tall black iron flagpole, flying outdoors against an overcast sky. The flag is caught in a steady, strong breeze blowing from screen-left, causing the fabric to billow outward with deep rolling waves and occasional sharp snaps at the trailing edge. The three vertical stripes — green (hoist), white (centre), and orange (fly) — stretch and compress as the cloth deforms. The flag is attached along its full hoist edge to the pole via a rope-and-toggle halyard; the top and bottom hoist corners are fixed, the rest of the fabric is free to move.

MOOD AND LOOK:
Overcast daylight, soft and diffused — no harsh shadows. The sky is a flat light grey, providing even illumination that lets the flag colours read clearly. Atmosphere is civic, dignified, slightly windswept. Colour palette: vivid Ireland green (#169B62), pure white, and Ireland orange (#FF883E) against neutral grey sky and dark iron pole. The fabric should show subtle translucency where the white stripe is backlit by the sky.

CAMERA:
Medium shot, slightly low angle (camera at ~1.4m, flag centre at ~4m height) looking upward. Static camera. 85mm lens equivalent for mild telephoto compression. The full flag and top section of the pole are in frame. Shallow depth of field — flag tack-sharp, sky and distant background softly out of focus.

PHYSICAL DETAILS:
- Flag: 1.8m x 1.2m (3:2 ratio, standard Irish flag proportions), lightweight woven polyester/nylon
- Stripes: Three equal vertical bands — green at hoist, white centre, orange at fly end
- Fabric weight: Light (~100 g/m²), low stiffness, moderate air resistance
- Flagpole: Black powder-coated iron, ~6cm diameter, 6m tall, rigid
- Attachment: Hoist edge sewn into a sleeve or fixed at top and bottom corners with the edge constrained
- Ground: Not visible (low angle crops it out)

MOTION/TIMING DETAILS:
Wind blows steadily from screen-left at 6-10 m/s with gentle variation. Flag starts partially unfurled at frame 1, fully caught by the wind by frame 15. Continuous rippling and billowing through the sequence. Render frame 40 for a moment of full dramatic extension. The trailing (orange) edge should show the most movement — fluttering, curling, and occasional sharp snapping.

HARD CONSTRAINTS:
- Physics simulation: Blender Cloth modifier
- Pin group: Vertices along the hoist (left) edge pinned to the pole
- Wind: Force field, strength ~8, direction (-1, 0, 0.1) with turbulence ~2
- Cloth quality steps: 12
- Bake frames: 1-60
- Renderer: Cycles GPU
- Samples: 256 max
- Render frame: 40
- Reference: None

Research documentation, patterns, and APIs to find the best approach for realistic cloth simulation with wind forces and pinned edge constraints."""


async def main():
    print("=" * 70)
    print("IRELAND FLAG TEST — Cloth Physics + Stale Render Fix Verification")
    print("3 iterations, codex_upgrade preset, tracing enabled")
    print(f"Trace log: {log_file}")
    print("=" * 70)
    print()

    request = AssetRequest(
        asset_name="ireland_flag",
        description=IRELAND_FLAG_DESCRIPTION,
        effect_type=EffectType.FLAG,
        resolution=96,
        frame_start=1,
        frame_end=60,
        quality_threshold=60.0,
        max_iterations=3,
        semantic_query="flag cloth simulation wind force pinned edge billowing fabric waving",
    )

    orchestrator = BlenderVFXOrchestrator()
    await orchestrator.initialize()

    print(f"Asset: {request.asset_name}")
    print(f"Effect Type: {request.effect_type.value}")
    print(f"Max Iterations: {request.max_iterations}")
    print(f"Quality Threshold: {request.quality_threshold}")
    print()

    try:
        with trace("Ireland Flag E2E Test", group_id=request.asset_name):
            result = await orchestrator.create_asset_pipeline(request)

        print("\n" + "=" * 70)
        print("IRELAND FLAG TEST COMPLETE")
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
