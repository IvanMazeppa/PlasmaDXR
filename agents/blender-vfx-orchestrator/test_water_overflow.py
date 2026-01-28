#!/usr/bin/env python3
"""
Test: Water overflow from scratch with enriched prompt.
Uses development preset with tuned agent overrides (2026-01-28).
"""

import asyncio
import os
import sys

# Use development preset (gpt-5-mini base + strategic gpt-5.2 overrides)
os.environ["ORCHESTRATOR_PRESET"] = "development"

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from orchestrator import create_vfx_asset

ENRICHED_PROMPT = """\
SCENE DESCRIPTION:
Water spraying from a wall-mounted metal pipe into a drinking glass sitting on a wooden table.
The stream is pressurized — not a gentle pour, but a forceful spray that causes splashing
on impact with the glass and the water surface inside it. As the glass overfills, water
spills over the rim and pools on the table surface.

MOOD AND LOOK:
Moody, cinematic lighting. The scene should feel like a close-up shot from a short film —
dramatic highlights on the water and metal, deep shadows elsewhere. The water should look
physically convincing with visible refraction and caustics.

HARD CONSTRAINTS:
- Fluid simulation: Blender Mantaflow liquid domain (not shader-only, not static mesh)
- Renderer: Cycles GPU
- Samples: 256 max
- Frame range: 1-48
"""

async def main():
    print("=" * 70)
    print("WATER OVERFLOW TEST — Fresh Start")
    print("Preset: development (tuned agent overrides)")
    print("Session compaction: ENABLED (OpenAIResponsesCompactionSession)")
    print("=" * 70)
    print(f"\nPrompt:\n{ENRICHED_PROMPT[:200]}...\n")

    result = await create_vfx_asset(
        asset_name="water_overflow_v1",
        description=ENRICHED_PROMPT,
        effect_type="water",
        frame_end=48,
        quality_threshold=60.0,
        max_iterations=5,
        enable_diagnostics=True,
    )

    print(f"\n{'=' * 70}")
    print(f"RESULT: {result.status.value}")
    print(f"Best Score: {result.best_score:.1f} (iteration {result.best_iteration})")
    print(f"Total Iterations: {len(result.iterations)}")
    print(f"{'=' * 70}")

    for i, iter_result in enumerate(result.iterations):
        print(f"  Iter {iter_result.iteration}: score={iter_result.score:.1f}, "
              f"technique={iter_result.script.technique_name or 'unknown'}, "
              f"passed={iter_result.passed}")

    return result

if __name__ == "__main__":
    result = asyncio.run(main())
