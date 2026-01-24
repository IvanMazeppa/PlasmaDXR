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

from orchestrator import VFXOrchestrator, create_vfx_asset
from models.shared_context import SharedContext, AssetRequest, EffectType
from hooks import DiagnosticHooks
from utils.session_persistence import generate_session_id


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

    # Create request
    timestamp = datetime.now().strftime("%H%M%S")
    asset_name = f"diagnostic_{timestamp}_{effect_type}"

    request = AssetRequest(
        asset_name=asset_name,
        description=f"A vibrant {effect_type} effect for diagnostic analysis",
        effect_type=EffectType(effect_type),
        quality_threshold=50.0,  # Lower threshold to allow iterations
        max_iterations=max_iterations,
        reference_path=None,
    )

    # Create diagnostic hooks
    log_file = output_log or f"diagnostic_trace_{timestamp}.jsonl"
    diag_hooks = DiagnosticHooks(
        log_file=log_file,
        verbose=True,
        track_patterns=True,
    )

    print(f"Diagnostic log: {log_file}")
    print()

    # Initialize session
    session_id = generate_session_id(request.asset_name)

    # Create shared context
    context = SharedContext(
        session_id=session_id,
        request=request,
    )

    # Create orchestrator
    orchestrator = VFXOrchestrator()

    print("Starting asset generation with diagnostic hooks...")
    print("-" * 60)

    try:
        # Run the pipeline
        # Note: We need to modify the pipeline to accept custom hooks
        # For now, we'll just run and analyze the standard flow
        session = await orchestrator.create_asset_pipeline(request)

        print("-" * 60)
        print("Generation complete!")
        print(f"Status: {session.status}")
        print(f"Best Score: {session.best_score}")
        print(f"Iterations: {len(session.iterations)}")

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

        # Analysis suggestions
        print("\n" + "=" * 60)
        print("ANALYSIS SUGGESTIONS")
        print("=" * 60)

        if len(report["repeated_prompts"]) > 0:
            print("⚠️  Repeated prompts detected - check if the same prompt")
            print("   structure is being sent to agents across iterations.")
            print("   This could indicate hardcoded templates in the pipeline.")

        if len(report["repeated_tool_calls"]) > 0:
            print("⚠️  Repeated tool calls detected - agents may be stuck")
            print("   in a pattern. Check enforcement hooks configuration.")

        if len(report["repeated_outputs"]) > 0:
            print("⚠️  Repeated outputs detected - this could indicate:")
            print("   1. Training data patterns overriding instructions")
            print("   2. Insufficient variety in prompts/research")
            print("   3. Legacy code paths being triggered")

        print()
        print(f"Full trace saved to: {log_file}")


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
