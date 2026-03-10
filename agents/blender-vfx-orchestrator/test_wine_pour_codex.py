#!/usr/bin/env python3
"""
Wine pour E2E test — validates orchestrator decomposition + GPT-5.4 rollout.

Uses the codex_upgrade preset which upgrades script_writer, modification_coordinator,
and research_agent to gpt-5.4 while keeping other agents on gpt-5-mini.

max_iterations=1 for quick feedback. HITL overridden to non-interactive/autonomous.
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
log_file = f"traces/wine_pour_codex_{timestamp}.jsonl"
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

# Log model assignments for all agents to confirm codex rollout
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


WINE_DESCRIPTION = """SCENE DESCRIPTION:
A stream of deep red wine pours from an unseen bottle (positioned just above frame) into an elegant stemmed wine glass resting on a dark wooden surface. The pour is steady but vigorous — not a delicate trickle, but a confident stream that creates dynamic splashing as it impacts the rising liquid surface. As the glass fills past the halfway point, wine droplets leap from the surface on impact, some catching the warm light as they arc through the air. A few droplets escape over the rim and trail down the outside of the glass.

MOOD AND LOOK:
Warm, intimate atmosphere evoking a cozy evening scene. Rich amber/golden key lighting from one side (as if from a nearby fireplace or candle cluster) creates dramatic highlights on the glass rim and wine surface, with the deep burgundy liquid glowing where light passes through it. The background falls into soft, warm darkness. The wine should exhibit convincing subsurface scattering — appearing darker at depth, more translucent and ruby-red where thin against the glass walls. Visible caustics pooling on the table surface beneath the glass.

CAMERA:
Close-up, slightly low angle looking up at the glass to emphasize elegance. Shallow depth of field optional. 35mm lens equivalent. Static camera.

PHYSICAL DETAILS:
- Wine glass: Standard Bordeaux style, clear glass with subtle reflections, ~22cm tall
- Wine color: Deep burgundy/cabernet red with ruby highlights in thin areas
- Pour stream: ~8mm diameter, continuous with slight surface tension wobble
- Table surface: Dark polished wood with subtle reflection of the glass base
- Background: Warm out-of-focus bokeh suggesting a candlelit room

MOTION/TIMING DETAILS:
The pour is continuous from frame 1. Wine enters from above frame and impacts the liquid surface already in the glass. By frame 30, the glass is about half full with active splashing. By frame 45, the glass is nearly 3/4 full and small waves propagate across the surface. Render frame 40 for peak action.

HARD CONSTRAINTS:
- Fluid simulation: Blender Mantaflow liquid domain (FLIP solver, not shader-only)
- Domain type: LIQUID with mesh generation enabled
- use_plane_init: True (required for inflow emitters)
- Inflow velocity: ~1.0-1.5 m/s downward pour
- Resolution: 96 (balance detail vs bake time)
- sampling_substeps: 6 (prevent discretized stream artifacts)
- Renderer: Cycles GPU
- Samples: 128 max
- Frame range: 1-60
- cache_type: ALL
- Render frame: 40 (mid-pour with active surface dynamics)
- Reference: None

Research documentation, patterns, and APIs to find the optimal starting approach."""


async def main():
    print("=" * 70)
    print("WINE POUR GPT-5.4 TEST — Decomposition Gate + GPT-5.4 Rollout")
    print("1 iteration, codex_upgrade preset, tracing enabled")
    print(f"Trace log: {log_file}")
    print("=" * 70)
    print()

    request = AssetRequest(
        asset_name="wine_pour_codex",
        description=WINE_DESCRIPTION,
        effect_type=EffectType.WATER,
        resolution=96,
        frame_start=1,
        frame_end=60,
        quality_threshold=60.0,
        max_iterations=1,
        semantic_query="wine pour liquid glass splash mantaflow inflow red wine stream impact surface",
    )

    orchestrator = BlenderVFXOrchestrator()
    await orchestrator.initialize()

    print(f"Asset: {request.asset_name}")
    print(f"Effect Type: {request.effect_type.value}")
    print(f"Max Iterations: {request.max_iterations}")
    print(f"Quality Threshold: {request.quality_threshold}")
    print()

    try:
        with trace("Wine Pour Codex E2E Test", group_id=request.asset_name):
            result = await orchestrator.create_asset_pipeline(request)

        print("\n" + "=" * 70)
        print("WINE POUR CODEX TEST COMPLETE")
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

        # Verify rollout model usage in trace
        print()
        print("=" * 70)
        print("GPT-5.4 ROLLOUT ANALYSIS")
        print("=" * 70)
        trace_path = Path(log_file)
        if trace_path.exists():
            with open(trace_path) as f:
                content = f.read()

            if "gpt-5.4" in content:
                print("  [PASS] gpt-5.4 referenced in trace")
            else:
                print("  [WARN] gpt-5.4 NOT found in trace — check model assignment")

            if "read_artifact" in content:
                print("  [PASS] read_artifact tool invoked (artifact sharing)")
            else:
                print("  [INFO] read_artifact not invoked")

            if "list_session_artifacts" in content:
                print("  [PASS] list_session_artifacts tool invoked")
            else:
                print("  [INFO] list_session_artifacts not invoked")

        # Check decomposition: phases/ modules should be imported
        print()
        print("DECOMPOSITION GATE VERIFICATION")
        print("-" * 40)
        from phases import research as _research_mod
        from phases import execution as _execution_mod
        import coordinator_agents as _coord_mod
        import models.pipeline_models as _models_mod
        print("  [PASS] phases.research imported")
        print("  [PASS] phases.execution imported")
        print("  [PASS] coordinator_agents imported")
        print("  [PASS] models.pipeline_models imported")

        # Check artifacts directory
        artifacts_dir = Path("sessions/artifacts")
        if artifacts_dir.exists():
            sessions = list(artifacts_dir.iterdir())
            matching = [s for s in sessions if "wine_pour_codex" in s.name]
            if matching:
                session_dir = matching[0]
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
