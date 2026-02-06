#!/usr/bin/env python3
"""
Oil Pour Resume Test - Continue with FIXED domain positioning

Resumes the oil_pour_20260202_021508 session with the manually corrected
domain geometry (was: z=0.25, scale=0.25 → now: z=0.15, scale=0.34).

This tests the resume functionality and validates that the domain fix resolves
the "no liquid visible" issue that Quality Analyst misdiagnosed as overexposure.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Enable verbose tracing BEFORE importing orchestrator
from tracing import enable_verbose_tracing
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"traces/oil_pour_resume_{timestamp}.jsonl"
enable_verbose_tracing(
    log_file=log_file,
    console_output=True,
    include_span_data=True,
)
print(f"[SETUP] Tracing enabled: {log_file}")

from orchestrator import BlenderVFXOrchestrator
from agents import trace


SESSION_ID = "oil_pour_20260202_021508_20260202_021509"
FIXED_SCRIPT = "/home/maz3ppa/projects/PlasmaDXR/assets/blender_scripts/generated/oil_pour_20260202_021508_iter3_paramfix.py"


async def main():
    print("=" * 70)
    print("OIL POUR RESUME TEST - FIXED DOMAIN POSITIONING")
    print("=" * 70)
    print()
    print("FIX APPLIED:")
    print("  OLD: domain at z=0.25, scale=(0.15,0.15,0.25) → z range 0.125-0.375")
    print("  NEW: domain at z=0.15, scale=(0.12,0.12,0.34) → z range -0.02 to 0.32")
    print("  Glass is at z=0-0.12, inflow at z=0.27 — now all encompassed!")
    print()

    # Initialize orchestrator
    orchestrator = BlenderVFXOrchestrator()
    await orchestrator.initialize()

    print(f"Session ID: {SESSION_ID}")
    print(f"Fixed Script: {FIXED_SCRIPT}")
    print(f"Additional Iterations: 2")
    print()

    # Resume session with 2 more iterations (original had 3, so total becomes 5)
    # The fixed script is already saved at iter3_paramfix.py path
    try:
        with trace("Oil Pour Resume Test", group_id=SESSION_ID):
            result = await orchestrator.resume_session(
                session_id=SESSION_ID,
                max_iterations_override=5,  # Was 3, adding 2 more
            )

        print("\n" + "=" * 70)
        print("OIL POUR RESUME COMPLETE")
        print("=" * 70)
        print(f"Status: {result.status.value}")
        print(f"Best Score: {result.best_score:.1f}")
        print(f"Total Iterations: {result.current_iteration}")
        print()

        if result.final_render_path:
            print(f"Final Render: {result.final_render_path}")

        if hasattr(result, 'iterations') and result.iterations:
            print()
            print("Iteration History:")
            for it in result.iterations:
                status = "✓" if it.passed else "✗"
                issue = getattr(it, 'primary_issue', None) or getattr(it, 'issues', ['No issues'])[0] if hasattr(it, 'issues') else 'No issues'
                print(f"  [{status}] Iteration {it.iteration}: {it.score:.1f} - {issue}")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        await orchestrator.close()

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
