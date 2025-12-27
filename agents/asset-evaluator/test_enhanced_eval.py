#!/usr/bin/env python3
"""
Test script for Enhanced VFX Evaluation System

Run from the asset-evaluator directory with venv activated:
    cd agents/asset-evaluator
    source venv/bin/activate
    python test_enhanced_eval.py
"""

import asyncio
import json
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Test images - update these paths as needed
PROJECT_ROOT = Path(__file__).parent.parent.parent

TEST_RENDER = PROJECT_ROOT / "build/vdb_output/fireball_test_v3/render_0035.png"
TEST_REFERENCE = PROJECT_ROOT / "assets/reference_images/explosions/explosion_reference_web_1.jpg"
TEST_FRAMES_DIR = PROJECT_ROOT / "build/vdb_output/fireball_test_v2"


def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


async def test_multi_prompt_clip():
    """Test the multi-prompt CLIP analysis."""
    print_section("TEST 1: Multi-Prompt CLIP Analysis")

    from enhanced_evaluation import evaluate_multi_prompt_clip

    if not TEST_RENDER.exists():
        print(f"SKIP: Test render not found: {TEST_RENDER}")
        return

    print(f"Analyzing: {TEST_RENDER.name}")
    print("This provides fine-grained quality gradient instead of simple pass/fail...\n")

    result = evaluate_multi_prompt_clip(
        str(TEST_RENDER),
        "a bright fiery explosion with flames and smoke",
        effect_type="explosion"
    )

    print(f"Overall CLIP match: {result.overall_match:.4f}")
    print(f"Gradient signal:    {result.gradient_signal:.4f}")
    print(f"\nQuality dimension scores:")
    for dim, score in result.quality_dimension_scores.items():
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  {dim:12s}: {score:.3f} [{bar}]")

    print(f"\nBest match:  '{result.best_matching_description}'")
    print(f"Worst match: '{result.worst_matching_description}'")


async def test_aesthetic_scoring():
    """Test reference-free aesthetic scoring."""
    print_section("TEST 2: LAION Aesthetic Scoring (Reference-Free)")

    from enhanced_evaluation import evaluate_aesthetic_quality

    if not TEST_RENDER.exists():
        print(f"SKIP: Test render not found: {TEST_RENDER}")
        return

    print(f"Analyzing: {TEST_RENDER.name}")
    print("This works WITHOUT a reference image...\n")

    score = evaluate_aesthetic_quality(str(TEST_RENDER))

    bar = "█" * int(score) + "░" * (10 - int(score))
    print(f"Aesthetic score: {score:.2f}/10 [{bar}]")

    if score >= 7:
        print("Interpretation: EXCELLENT - High aesthetic quality")
    elif score >= 5.5:
        print("Interpretation: GOOD - Acceptable quality")
    elif score >= 4:
        print("Interpretation: FAIR - Needs improvement")
    else:
        print("Interpretation: POOR - Significant issues")


async def test_vlm_diagnostics():
    """Test VLM-based diagnostic analysis."""
    print_section("TEST 3: Moondream VLM Diagnostics")

    from enhanced_evaluation import analyze_vfx_diagnostics

    if not TEST_RENDER.exists():
        print(f"SKIP: Test render not found: {TEST_RENDER}")
        return

    print(f"Analyzing: {TEST_RENDER.name}")
    print("Getting actionable feedback with Blender parameter suggestions...")
    print("(This may take a moment on first run - downloading model...)\n")

    try:
        diagnostics, suggestions = analyze_vfx_diagnostics(str(TEST_RENDER), "explosion")

        print("DIAGNOSTICS:")
        print(f"  Color temperature: {diagnostics.color_temperature}")
        print(f"  Density:          {diagnostics.density}")
        print(f"  Shape:            {diagnostics.shape}")
        print(f"  Brightness:       {diagnostics.brightness}")
        print(f"  Smoke ratio:      {diagnostics.smoke_ratio}")
        print(f"  Contrast:         {diagnostics.contrast}")
        print(f"  Detail level:     {diagnostics.detail_level}")

        print("\nSUGGESTED BLENDER PARAMETERS:")
        if suggestions.flame_max_temp:
            print(f"  flame_max_temp: {suggestions.flame_max_temp}")
        if suggestions.flame_smoke:
            print(f"  flame_smoke:    {suggestions.flame_smoke}")
        if suggestions.vorticity:
            print(f"  vorticity:      {suggestions.vorticity}")
        if suggestions.heat:
            print(f"  heat:           {suggestions.heat}")
        if suggestions.density:
            print(f"  density:        {suggestions.density}")
        if suggestions.resolution:
            print(f"  resolution:     {suggestions.resolution}")
        if suggestions.turbulence:
            print(f"  turbulence:     {suggestions.turbulence}")

        if not any([suggestions.flame_max_temp, suggestions.flame_smoke,
                    suggestions.vorticity, suggestions.heat]):
            print("  (No changes suggested - looking good!)")

    except Exception as e:
        print(f"VLM analysis error (this is OK, fallback will be used): {e}")


async def test_temporal_analysis():
    """Test temporal consistency analysis."""
    print_section("TEST 4: Temporal Animation Analysis")

    from enhanced_evaluation import analyze_temporal_quality

    if not TEST_FRAMES_DIR.exists():
        print(f"SKIP: Frames directory not found: {TEST_FRAMES_DIR}")
        return

    frames = sorted(list(TEST_FRAMES_DIR.glob("render_*.png")))
    if len(frames) < 2:
        print(f"SKIP: Need at least 2 frames, found {len(frames)}")
        return

    print(f"Analyzing {len(frames)} frames from: {TEST_FRAMES_DIR.name}")
    print("Checking for flickering and motion consistency...\n")

    result = analyze_temporal_quality([str(f) for f in frames], sample_rate=3)

    print(f"Frames analyzed:       {result['sampled_count']} of {result['frame_count']}")
    print(f"Temporal consistency:  {result['temporal_consistency']:.3f}")
    print(f"Flicker risk:          {result['flicker_risk'].upper()}")
    print(f"Average motion:        {result['average_motion']:.2f}")
    print(f"Static frames:         {result['static_frame_count']}")

    print("\nRecommendations:")
    for rec in result['recommendations']:
        print(f"  • {rec}")


async def test_full_enhanced_evaluation():
    """Test the complete enhanced evaluation pipeline."""
    print_section("TEST 5: Full Enhanced Evaluation")

    from enhanced_evaluation import enhanced_evaluate_render

    if not TEST_RENDER.exists():
        print(f"SKIP: Test render not found: {TEST_RENDER}")
        return

    print(f"Render:    {TEST_RENDER.name}")
    print(f"Reference: {TEST_REFERENCE.name if TEST_REFERENCE.exists() else 'None'}")
    print("\nRunning comprehensive multi-modal evaluation...\n")

    result_json = await enhanced_evaluate_render(
        render_path=str(TEST_RENDER),
        semantic_query="a bright fiery explosion with orange flames and dark smoke",
        effect_type="explosion",
        reference_path=str(TEST_REFERENCE) if TEST_REFERENCE.exists() else None,
        lpips_threshold=0.35,
        clip_threshold=0.60,
        aesthetic_threshold=5.5
    )

    result = json.loads(result_json)

    # Overall result
    status = "✅ PASSED" if result['passed'] else "❌ FAILED"
    print(f"Result: {status}")
    print(f"Overall Score: {result['overall_score']:.1f}/100\n")

    # Individual scores
    print("SCORES:")
    scores = result.get('scores', {})
    for metric, score in scores.items():
        if score is not None:
            bar = "█" * int(score * 10 / 100) + "░" * (10 - int(score * 10 / 100))
            print(f"  {metric:15s}: {score:.3f} [{bar}]")
        else:
            print(f"  {metric:15s}: N/A")

    # Metrics that passed/failed
    print(f"\nPassed metrics: {result.get('passed_metrics', [])}")
    print(f"Failed metrics: {result.get('failed_metrics', [])}")

    # Diagnostics
    if result.get('diagnostics'):
        print("\nDIAGNOSTICS:")
        for key, value in result['diagnostics'].items():
            if key != 'raw_analysis' and value != 'good':
                print(f"  ⚠ {key}: {value}")

    # Parameter suggestions
    if result.get('suggested_parameters'):
        suggestions = result['suggested_parameters']
        has_suggestions = any(v for v in suggestions.values())
        if has_suggestions:
            print("\nSUGGESTED CHANGES:")
            for param, change in suggestions.items():
                if change:
                    print(f"  → {param}: {change}")

    # Recommendations
    if result.get('recommendations'):
        print("\nRECOMMENDATIONS:")
        for rec in result['recommendations'][:5]:
            print(f"  • {rec}")


async def main():
    print("\n" + "="*60)
    print("  ENHANCED VFX EVALUATION SYSTEM - TEST SUITE")
    print("="*60)

    print(f"\nTest render:  {TEST_RENDER}")
    print(f"Test frames:  {TEST_FRAMES_DIR}")
    print(f"Reference:    {TEST_REFERENCE}")

    # Run tests
    await test_multi_prompt_clip()
    await test_aesthetic_scoring()
    await test_vlm_diagnostics()
    await test_temporal_analysis()
    await test_full_enhanced_evaluation()

    print_section("TESTS COMPLETE")
    print("The enhanced evaluation system is ready to use!")
    print("\nTo use via MCP, call these tools:")
    print("  • enhanced_evaluate")
    print("  • multi_prompt_clip_analysis")
    print("  • analyze_temporal_quality")
    print("  • get_alternative_approaches")


if __name__ == "__main__":
    asyncio.run(main())
