#!/usr/bin/env python3
"""
Campfire Mantaflow E2E test — re-baseline Mantaflow gas domain with full Wave 2 stack.

Validates: truth pack, repair routing, HITL (non-interactive), stale render fix,
section patching, GPT-5.4 rollout, technique contracts, artifact gates.

3 iterations to exercise the feedback loop. HITL overridden to autonomous.
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
log_file = f"traces/campfire_mantaflow_{timestamp}.jsonl"
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


CAMPFIRE_DESCRIPTION = """SCENE DESCRIPTION:
A small campfire burns steadily in a ring of rough river stones on bare earth. Three short logs (birch, ~30cm long, ~8cm diameter) are arranged in a loose teepee formation, their ends charred black where flames lick the wood. The fire is past its roaring phase — now a warm, settled burn with a bed of glowing embers beneath. Flames rise 30-40cm from the ember bed, occasionally licking higher when they catch a pocket of resin. Thin smoke drifts upward in lazy curls, dispersing above the flame tips. Small orange sparks occasionally lift from the embers and float upward before winking out.

MOOD AND LOOK:
Warm, peaceful night-time atmosphere — the feeling of sitting around a fire after a long day outdoors. The campfire is the sole light source, casting a warm orange-amber glow that illuminates the stone ring and fades rapidly into deep blue-black darkness beyond. Firelight dances across the stones, creating flickering shadows. Color palette: deep amber and orange flames, white-hot ember cores with red-orange halos, blue-tinged smoke against the dark sky, cool grey stones warmed by firelight.

CAMERA:
Medium close-up framing the full fire and stone ring. Eye-level perspective, as if sitting on the ground across from the fire. Shallow depth of field — flames and center logs tack-sharp, stones at the edges soften slightly. Static camera. 50mm lens equivalent.

PHYSICAL DETAILS:
- Stone ring: 6-8 rough river stones, grey granite, ~15-20cm each, arranged in a ~60cm diameter circle
- Logs: 3 birch logs, white bark with dark knots, ~30cm long, ~8cm diameter, charred ends
- Ember bed: ~40cm diameter, 5cm deep, glowing orange-white coals
- Flame height: 30-40cm from ember bed, tapering to points
- Flame structure: Blue-white base (2-3cm) near embers, bright yellow body, orange-tipped wisps that curl and detach
- Smoke: Thin, grey-white, rising in laminar wisps that become turbulent ~50cm above flames
- Ground plane: Dark packed earth/dirt, slightly reflective from moisture

HARD CONSTRAINTS:
- Fluid simulation: Blender Mantaflow gas domain with fire enabled
- Domain type: GAS with use_noise enabled for fine detail
- Fire reaction speed: Medium (0.4-0.6) for a settled, steady burn
- Smoke amount: Low (0.05-0.15) — this is a clean-burning fire, not a smoker
- Vorticity: Medium (0.4-0.6) for natural flame movement
- burning_rate: 0.6-0.8 (steady combustion)
- use_dissolve_smoke: True with dissolve_speed ~20 (smoke dissipates fairly quickly)
- Renderer: Cycles GPU
- Samples: 256 max
- Frame range: 1-90
- Resolution: 64 (balance bake time vs detail for E2E test)
- cache_type: ALL
- Reference: None

Research documentation, patterns, and APIs to find the optimal starting approach."""


async def main():
    print("=" * 70)
    print("CAMPFIRE MANTAFLOW E2E TEST — Wave 2 Re-Baseline")
    print("3 iterations, codex_upgrade preset, tracing enabled")
    print(f"Trace log: {log_file}")
    print("=" * 70)
    print()

    request = AssetRequest(
        asset_name="campfire_mantaflow_baseline",
        description=CAMPFIRE_DESCRIPTION,
        effect_type=EffectType.FIRE,
        resolution=64,
        frame_start=1,
        frame_end=90,
        quality_threshold=60.0,
        max_iterations=3,
        semantic_query="campfire fire flames embers smoke stones night mantaflow gas domain warm glow",
    )

    orchestrator = BlenderVFXOrchestrator()
    await orchestrator.initialize()

    print(f"Asset: {request.asset_name}")
    print(f"Effect Type: {request.effect_type.value}")
    print(f"Max Iterations: {request.max_iterations}")
    print(f"Quality Threshold: {request.quality_threshold}")
    print()

    try:
        with trace("Campfire Mantaflow Baseline E2E", group_id=request.asset_name):
            result = await orchestrator.create_asset_pipeline(request)

        print("\n" + "=" * 70)
        print("CAMPFIRE MANTAFLOW E2E TEST COMPLETE")
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

        # Wave 2 verification
        print()
        print("=" * 70)
        print("WAVE 2 STACK VERIFICATION")
        print("=" * 70)

        trace_path = Path(log_file)
        if trace_path.exists():
            # Parse pipeline events from JSONL trace
            import json as _json
            pipeline_events = []
            for line in trace_path.read_text().splitlines():
                try:
                    entry = _json.loads(line)
                    if entry.get("event_type") == "pipeline_event":
                        pipeline_events.append(entry)
                except _json.JSONDecodeError:
                    continue

            event_names = {e.get("event") for e in pipeline_events}

            # GPT-5.4 rollout — check orchestrator_init models
            init_events = [e for e in pipeline_events if e.get("event") == "orchestrator_init"]
            if init_events:
                models = init_events[-1].get("models", {})
                model_values = list(models.values())
                if any("gpt-5.4" in m for m in model_values):
                    print(f"  [PASS] gpt-5.4 in use: {', '.join(f'{k}={v}' for k, v in models.items() if 'gpt-5.4' in v)}")
                else:
                    print(f"  [WARN] gpt-5.4 NOT found in model assignments: {models}")
            else:
                print("  [WARN] No orchestrator_init event in trace")

            # Stale render fix: check execution_result events for false success
            exec_events = [e for e in pipeline_events if e.get("event") == "execution_result"]
            stale_renders = [e for e in exec_events if e.get("exit_code") != 0 and e.get("has_render")]
            if stale_renders:
                print("  [FAIL] Stale render promotion detected — fix not working!")
            else:
                print("  [PASS] No stale render promotion (fix working)")

            # Repair routing
            repair_events = [e for e in pipeline_events if e.get("event") == "repair_intent"]
            if repair_events:
                modes = [e.get("mode") for e in repair_events]
                print(f"  [PASS] RepairIntent routing active: {', '.join(modes)}")
            else:
                print("  [INFO] RepairIntent not triggered (may be normal for iter 1)")

            # HITL — still use content scan for now since HITL events are not yet instrumented
            content = trace_path.read_text()
            if "HITL" in content:
                print("  [PASS] HITL framework active")
            else:
                print("  [INFO] No HITL checkpoints triggered (may be normal)")

            # Truth pack
            tp_events = [e for e in pipeline_events if e.get("event") == "truth_pack_fix"]
            if tp_events:
                total_fixes = sum(e.get("fixes_count", 0) for e in tp_events)
                print(f"  [PASS] Truth pack validation active ({total_fixes} fixes across {len(tp_events)} runs)")
            else:
                print("  [WARN] Truth pack not detected in trace")

            # Iteration results summary
            iter_events = [e for e in pipeline_events if e.get("event") == "iteration_result"]
            if iter_events:
                print()
                print("  Pipeline Event Summary:")
                for ie in iter_events:
                    print(f"    Iter {ie.get('iteration')}: score={ie.get('score', 0):.1f} "
                          f"passed={ie.get('passed')} "
                          f"structured_issues={ie.get('structured_issues_count', 0)} "
                          f"plateau={ie.get('plateau_count', 0)}")

        # Score analysis
        print()
        print("MANTAFLOW BASELINE SCORES")
        print("-" * 40)
        for it in result.iterations:
            score = it.score if hasattr(it, 'score') else 0
            technique = it.script.technique if hasattr(it, 'script') and hasattr(it.script, 'technique') else "unknown"
            passed = "PASS" if score >= 60 else "FAIL"
            print(f"  Iteration {it.iteration}: score={score:.1f} [{passed}] technique={technique}")

        # Check artifacts
        artifacts_dir = Path("sessions/artifacts")
        if artifacts_dir.exists():
            sessions = list(artifacts_dir.iterdir())
            matching = [s for s in sessions if "campfire_mantaflow" in s.name]
            if matching:
                session_dir = matching[-1]
                artifacts = list(session_dir.glob("*.json"))
                print(f"\n  Artifacts written ({len(artifacts)}):")
                for a in sorted(artifacts)[:15]:
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
