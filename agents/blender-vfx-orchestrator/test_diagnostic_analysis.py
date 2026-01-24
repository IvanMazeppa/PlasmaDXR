#!/usr/bin/env python3
"""
Diagnostic Analysis Test

This script runs a VFX generation with diagnostic hooks enabled
to analyze what's causing repetitive patterns in agent outputs.

Captures:
- All prompts sent to agents
- All tool calls with fingerprinting
- Output patterns with similarity detection
- Potential training data reversion indicators

Usage:
    python test_diagnostic_analysis.py [--effect fire|explosion|smoke] [--iterations 2]
"""

import asyncio
import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from orchestrator import create_vfx_asset
from models.shared_context import AssetRequest, EffectType


async def run_diagnostic_analysis(
    effect_type: str = "fire",
    max_iterations: int = 2,
    output_log: str = None,
):
    """Run asset generation with diagnostic hooks."""

    print("=" * 60)
    print("DIAGNOSTIC ANALYSIS TEST")
    print("=" * 60)
    print(f"Effect Type: {effect_type}")
    print(f"Max Iterations: {max_iterations}")
    print()

    # Create diagnostic trace paths
    timestamp = datetime.now().strftime("%H%M%S")
    log_file = output_log or f"traces/diagnostic_trace_{timestamp}.jsonl"

    print(f"Diagnostic log: {log_file}")
    print(f"Flow tracker: traces/communication_flow.jsonl")
    print()

    # Asset name
    asset_name = f"diagnostic_{timestamp}_{effect_type}"

    print("Starting asset generation with LOCAL TRACING ENABLED...")
    print("-" * 60)

    try:
        # Use the high-level create_vfx_asset function with diagnostics enabled
        # This enables local AI-parseable trace logging
        session = await create_vfx_asset(
            asset_name=asset_name,
            description=f"A vibrant {effect_type} effect for diagnostic analysis - BE CREATIVE AND UNIQUE",
            effect_type=effect_type,
            quality_threshold=50.0,  # Lower threshold to allow iterations
            max_iterations=max_iterations,
            enable_diagnostics=True,  # Enable local trace logging
            diagnostic_log=log_file,
        )

        print("-" * 60)
        print("Generation complete!")
        print(f"Status: {session.status}")
        print(f"Best Score: {session.best_score}")
        print(f"Iterations: {len(session.iterations)}")
        print(f"Final Render: {session.final_render_path}")

        # Show iteration details
        print("\nIteration Details:")
        for i, iter_result in enumerate(session.iterations):
            print(f"  [{i+1}] Score: {iter_result.score:.1f} | "
                  f"Technique: {iter_result.script.technique_name if iter_result.script else 'N/A'}")

    except Exception as e:
        print(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Print diagnostic summary
        print()

        # Try to read flow tracker data
        flow_log = Path("traces/communication_flow.jsonl")
        if flow_log.exists():
            print("=" * 60)
            print("COMMUNICATION FLOW ANALYSIS (AI-PARSEABLE)")
            print("=" * 60)
            print(f"Flow log: {flow_log}")
            lines = flow_log.read_text().strip().split('\n')
            if lines and lines[0]:
                import json
                flows = [json.loads(line) for line in lines if line.strip()]
                print(f"Total flow events: {len(flows)}")

                # Analyze breakdowns
                breakdowns = [f for f in flows if f.get("event") == "flow_complete" and not f.get("success")]
                successes = [f for f in flows if f.get("event") == "flow_complete" and f.get("success")]

                print(f"Successful flows: {len(successes)}")
                print(f"Breakdown flows: {len(breakdowns)}")

                if breakdowns:
                    print("\n⚠️ COMMUNICATION BREAKDOWNS DETECTED:")
                    for bd in breakdowns[:5]:
                        print(f"  Coordinator: {bd.get('coordinator', '?')}")
                        print(f"  Decision: {bd.get('decision', '?')}")
                        print(f"  Parameters: {bd.get('parameters', {})}")
                        print(f"  Modify Result: {bd.get('modify_result', {})}")
                        print()
            print()

        # Diagnostic hooks summary if available
        diag_log = Path(log_file)

        print("\nDETAILED PATTERN REPORT:")
        print("-" * 40)

        if diag_log.exists():
            import json
            diag_lines = diag_log.read_text().strip().split('\n')
            diag_events = [json.loads(line) for line in diag_lines if line.strip()]

            # Count pattern types
            pattern_events = [e for e in diag_events if e.get("type") == "pattern_detected"]
            repeated_prompts = [e for e in pattern_events if e.get("pattern_type") == "repeated_prompt"]
            repeated_tools = [e for e in pattern_events if e.get("pattern_type") == "repeated_tool_call"]
            repeated_outputs = [e for e in pattern_events if e.get("pattern_type") == "repeated_output"]

            print(f"\nDiagnostic trace: {log_file} ({len(diag_events)} events)")

            if repeated_prompts:
                print("\n🔴 REPEATED PROMPTS (potential loop/template issue):")
                for p in repeated_prompts[:5]:
                    print(f"  [{p.get('count', '?')}x] {p.get('preview', '?')[:80]}...")

            if repeated_tools:
                print("\n🔴 REPEATED TOOL CALLS (potential stuck agent):")
                for t in repeated_tools[:5]:
                    print(f"  [{t.get('count', '?')}x] {t.get('tool_name', '?')}")

            if repeated_outputs:
                print("\n🔴 REPEATED OUTPUTS (potential training data reversion):")
                for o in repeated_outputs[:5]:
                    print(f"  [{o.get('count', '?')}x] {o.get('output_type', '?')}: {o.get('preview', '?')[:60]}...")

            if not pattern_events:
                print("\n✓ No repeated patterns detected")
        else:
            print(f"\nNote: Diagnostic log not found at {log_file}")

        # Analysis suggestions
        print("\n" + "=" * 60)
        print("ANALYSIS SUGGESTIONS")
        print("=" * 60)

        if len(report.get("repeated_prompts", [])) > 0:
            print("⚠️  Repeated prompts detected - check if the same prompt")
            print("   structure is being sent to agents across iterations.")

        if len(report.get("repeated_tool_calls", [])) > 0:
            print("⚠️  Repeated tool calls detected - agents may be stuck")
            print("   in a pattern. Check enforcement hooks configuration.")

        if len(report.get("repeated_outputs", [])) > 0:
            print("⚠️  Repeated outputs detected - this could indicate:")
            print("   1. Training data patterns overriding instructions")
            print("   2. Insufficient variety in prompts/research")
            print("   3. Legacy code paths being triggered")

        # Check generated scripts for similarity
        print("\n" + "=" * 60)
        print("SCRIPT ANALYSIS")
        print("=" * 60)

        scripts_dir = Path("assets/blender_scripts/generated")
        if scripts_dir.exists():
            recent_scripts = sorted(
                scripts_dir.glob(f"*diagnostic_{timestamp}*.py"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )[:5]

            if recent_scripts:
                print(f"Generated scripts ({len(recent_scripts)}):")
                for script in recent_scripts:
                    print(f"  - {script.name}")

                # Compare first two if multiple exist
                if len(recent_scripts) >= 2:
                    s1 = recent_scripts[0].read_text()
                    s2 = recent_scripts[1].read_text()

                    # Simple similarity check
                    lines1 = set(s1.split('\n'))
                    lines2 = set(s2.split('\n'))
                    common = lines1 & lines2
                    total = lines1 | lines2

                    if total:
                        similarity = len(common) / len(total) * 100
                        print(f"\nScript similarity: {similarity:.1f}%")
                        if similarity > 70:
                            print("⚠️  HIGH SIMILARITY - scripts may be following same pattern")
            else:
                print("No diagnostic scripts found")
        else:
            print(f"Scripts directory not found: {scripts_dir}")


def main():
    parser = argparse.ArgumentParser(description="Diagnostic Analysis Test")
    parser.add_argument("--effect", default="fire", choices=["fire", "explosion", "smoke", "sun"])
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--output", help="Output log file path")
    args = parser.parse_args()

    asyncio.run(run_diagnostic_analysis(
        effect_type=args.effect,
        max_iterations=args.iterations,
        output_log=args.output,
    ))


if __name__ == "__main__":
    main()
