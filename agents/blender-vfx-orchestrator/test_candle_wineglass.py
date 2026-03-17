#!/usr/bin/env python3
"""
Candle + Wine Glass E2E test — validate aesthetic realism changes.

Tests: StyleSpec extraction, prompt enhancement, AgX floor, look-dev hints,
hero object detection, structural pattern routing, code pattern tool exposure.

2 iterations. Prompt deliberately exercises multiple aesthetic dimensions:
- fire hints (candle)
- hero objects (candle, wine glass, wood table)
- palette extraction (amber)
- close-up DOF
- dark_world_practicals lighting mode
- intimate mood
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
log_file = f"traces/candle_wineglass_{timestamp}.jsonl"
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
reset_config()
config = get_config()
config.preset.hitl_interactive = False
config.preset.hitl_autonomy_level = 3

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


CANDLE_WINEGLASS_DESCRIPTION = (
    "A candle burns on a dark wooden table. "
    "Warm amber light reflects off a nearby wine glass. "
    "Close-up, intimate mood."
)


async def main():
    print("=" * 70)
    print("CANDLE + WINE GLASS E2E TEST — Aesthetic Realism Validation")
    print("2 iterations, codex_upgrade preset, tracing enabled")
    print(f"Trace log: {log_file}")
    print("=" * 70)
    print()

    # Pre-flight: verify prompt enhancement works on our description
    from utils.prompt_enhancer import enhance_description, extract_style_spec
    enhanced = enhance_description(CANDLE_WINEGLASS_DESCRIPTION, effect_type="fire")
    style = extract_style_spec(CANDLE_WINEGLASS_DESCRIPTION, effect_type="fire")

    print("PRE-FLIGHT: Prompt Enhancement Check")
    print("-" * 40)
    print(f"  Enhanced has LOOK-DEV BRIEF: {'LOOK-DEV BRIEF' in enhanced}")
    print(f"  Enhanced has AgX hint: {'AgX' in enhanced}")
    print(f"  Enhanced has fire hints: {'blue-white base' in enhanced}")
    print(f"  StyleSpec mood: {style['mood']}")
    print(f"  StyleSpec heroes: {style['hero_objects']}")
    print(f"  StyleSpec lighting: {style['world_lighting_mode']}")
    print(f"  StyleSpec camera: {style['camera_distance_class']}")
    print(f"  StyleSpec palette: {style['palette']}")
    print(f"  StyleSpec DOF: {style['dof_intent']}")
    print()

    request = AssetRequest(
        asset_name="candle_wineglass_aesthetic",
        description=CANDLE_WINEGLASS_DESCRIPTION,
        effect_type=EffectType.FIRE,
        resolution=64,
        frame_start=1,
        frame_end=60,
        quality_threshold=60.0,
        max_iterations=2,
        semantic_query="candle flame wine glass wooden table warm amber intimate close-up mantaflow fire",
    )

    orchestrator = BlenderVFXOrchestrator()
    await orchestrator.initialize()

    print(f"Asset: {request.asset_name}")
    print(f"Effect Type: {request.effect_type.value}")
    print(f"Max Iterations: {request.max_iterations}")
    print(f"Quality Threshold: {request.quality_threshold}")
    print()

    try:
        with trace("Candle WineGlass Aesthetic E2E", group_id=request.asset_name):
            result = await orchestrator.create_asset_pipeline(request)

        print("\n" + "=" * 70)
        print("CANDLE + WINE GLASS E2E TEST COMPLETE")
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

        # Aesthetic verification
        print()
        print("=" * 70)
        print("AESTHETIC REALISM VERIFICATION")
        print("=" * 70)

        # Check generated scripts for aesthetic markers
        artifacts_dir = Path("sessions/artifacts")
        if artifacts_dir.exists():
            sessions = list(artifacts_dir.iterdir())
            matching = [s for s in sessions if "candle_wineglass" in s.name]
            if matching:
                session_dir = matching[-1]
                scripts = list(session_dir.glob("*.py"))
                if not scripts:
                    # Check parent assets dir
                    scripts = list(Path("assets/blender_scripts/generated").glob("candle_wineglass*.py"))

        # Also check generated scripts directory
        gen_scripts = list(Path("assets/blender_scripts/generated").glob("candle_wineglass*.py"))
        all_scripts = gen_scripts

        if all_scripts:
            for script_path in all_scripts:
                print(f"\n  Script: {script_path.name}")
                script_text = script_path.read_text()
                script_len = len(script_text.splitlines())
                print(f"  Lines: {script_len}")

                # Aesthetic markers
                checks = {
                    "AgX color management": "AgX" in script_text,
                    "DOF enabled": "use_dof" in script_text or "aperture_fstop" in script_text,
                    "World shader": "Background" in script_text or "world" in script_text.lower(),
                    "Material creation": "materials.new(" in script_text,
                    "ShaderNode usage": "ShaderNode" in script_text,
                    "Glass BSDF": "Glass" in script_text or "glass" in script_text,
                    "Wine glass object": "wine" in script_text.lower() or "glass" in script_text.lower(),
                    "Candle object": "candle" in script_text.lower(),
                    "Wood material": "wood" in script_text.lower(),
                    "Principled Volume": "PrincipledVolume" in script_text or "Principled Volume" in script_text,
                }

                for check_name, passed in checks.items():
                    status = "PASS" if passed else "MISS"
                    print(f"    [{status}] {check_name}")

                # Count distinct materials
                mat_count = script_text.count("materials.new(")
                node_count = script_text.count("nodes.new(")
                print(f"    Materials created: {mat_count}")
                print(f"    Shader nodes created: {node_count}")
        else:
            print("  [WARN] No generated scripts found")

        # Trace analysis
        trace_path = Path(log_file)
        if trace_path.exists():
            import json as _json
            pipeline_events = []
            for line in trace_path.read_text().splitlines():
                try:
                    entry = _json.loads(line)
                    if entry.get("event_type") == "pipeline_event":
                        pipeline_events.append(entry)
                except _json.JSONDecodeError:
                    continue

            # Truth pack
            tp_events = [e for e in pipeline_events if e.get("event") == "truth_pack_fix"]
            if tp_events:
                total_fixes = sum(e.get("fixes_count", 0) for e in tp_events)
                print(f"\n  [PASS] Truth pack: {total_fixes} fixes across {len(tp_events)} runs")

            # Repair routing
            repair_events = [e for e in pipeline_events if e.get("event") == "repair_intent"]
            if repair_events:
                modes = [e.get("mode") for e in repair_events]
                print(f"  [PASS] RepairIntent: {', '.join(modes)}")

            # Iteration results
            iter_events = [e for e in pipeline_events if e.get("event") == "iteration_result"]
            if iter_events:
                print()
                print("  Iteration Summary:")
                for ie in iter_events:
                    print(f"    Iter {ie.get('iteration')}: score={ie.get('score', 0):.1f} "
                          f"passed={ie.get('passed')} "
                          f"issues={ie.get('structured_issues_count', 0)}")

        print()
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
