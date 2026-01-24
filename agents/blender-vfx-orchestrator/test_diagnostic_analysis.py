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
from hooks import DiagnosticHooks


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

    # Create diagnostic hooks
    timestamp = datetime.now().strftime("%H%M%S")
    log_file = output_log or f"diagnostic_trace_{timestamp}.jsonl"
    diag_hooks = DiagnosticHooks(
        log_file=log_file,
        verbose=True,
        track_patterns=True,
    )

    print(f"Diagnostic log: {log_file}")
    print()

    # Asset name
    asset_name = f"diagnostic_{timestamp}_{effect_type}"

    print("Starting asset generation...")
    print("-" * 60)

    try:
        # Use the high-level create_vfx_asset function
        # This handles all the setup internally
        session = await create_vfx_asset(
            asset_name=asset_name,
            description=f"A vibrant {effect_type} effect for diagnostic analysis - BE CREATIVE AND UNIQUE",
            effect_type=effect_type,
            quality_threshold=50.0,  # Lower threshold to allow iterations
            max_iterations=max_iterations,
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
        diag_hooks.print_summary()

        # Get pattern report
        report = diag_hooks.get_pattern_report()

        print("\nDETAILED PATTERN REPORT:")
        print("-" * 40)

        if report["repeated_prompts"]:
            print("\n🔴 REPEATED PROMPTS (potential loop/template issue):")
            for p in report["repeated_prompts"]:
                print(f"  [{p['count']}x] {p['preview'][:80]}...")

        if report["repeated_tool_calls"]:
            print("\n🔴 REPEATED TOOL CALLS (potential stuck agent):")
            for t in report["repeated_tool_calls"]:
                print(f"  [{t['count']}x] {t['tool']}")

        if report["repeated_outputs"]:
            print("\n🔴 REPEATED OUTPUTS (potential training data reversion):")
            for o in report["repeated_outputs"]:
                print(f"  [{o['count']}x] {o['type']}: {o['preview'][:60]}...")

        # Check if log file has events
        log_path = Path(log_file)
        if log_path.exists():
            line_count = sum(1 for _ in open(log_path))
            print(f"\nDiagnostic trace: {log_file} ({line_count} events)")
        else:
            print(f"\nNote: Diagnostic hooks weren't integrated into pipeline.")
            print("To fully trace, hooks need to be passed to Runner.run() calls.")

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
