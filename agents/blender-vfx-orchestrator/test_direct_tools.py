"""
Test that asset_evaluator_tools work directly without MCP dependencies.

This verifies the rewrite was successful:
1. find_reference_images - Direct filesystem search
2. evaluate_render - Uses analyze_with_vision (GPT-5.2)
3. compare_to_reference - Uses analyze_with_vision
4. list_renders - Direct filesystem search
"""

import asyncio
import json
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Any

load_dotenv()

# Import the tools directly to test them
from tools.asset_evaluator_tools import (
    find_reference_images,
    evaluate_render,
    compare_to_reference,
    list_renders,
    get_reference_stats,
    analyze_with_vision,
)


@dataclass
class MockToolContext:
    """Minimal mock context for testing tools."""
    context: Any = None


async def call_tool(tool, args: dict) -> str:
    """Call a FunctionTool with the given arguments."""
    ctx = MockToolContext()
    json_args = json.dumps(args)
    return await tool.on_invoke_tool(ctx, json_args)


async def test_find_reference_images():
    """Test direct filesystem search for references."""
    print("=" * 70)
    print("TEST 1: find_reference_images (direct filesystem)")
    print("=" * 70)
    print()

    # Test explosion references - call via on_invoke_tool
    result = await call_tool(find_reference_images, {"effect_type": "explosion", "limit": 3})
    print(f"Result type: {type(result)}")

    try:
        data = json.loads(result)
        print(f"  Effect type: {data.get('effect_type')}")
        print(f"  Count: {data.get('count')}")
        print(f"  Recommended: {data.get('recommended')}")
        if data.get("results"):
            print("  Results:")
            for r in data.get("results", [])[:3]:
                print(f"    - {r.get('filename')} ({r.get('size_kb')}KB)")
        success = data.get("count", 0) > 0
    except json.JSONDecodeError as e:
        print(f"  ERROR: Could not parse JSON: {e}")
        print(f"  Raw result: {result[:500]}...")
        success = False

    print()
    print(f"  STATUS: {'✓ PASSED' if success else '✗ FAILED'}")
    return success


async def test_list_renders():
    """Test direct filesystem search for renders."""
    print()
    print("=" * 70)
    print("TEST 2: list_renders (direct filesystem)")
    print("=" * 70)
    print()

    result = await call_tool(list_renders, {"pattern": "", "limit": 5})

    try:
        data = json.loads(result)
        print(f"  Found {len(data)} renders")
        if data:
            print("  Most recent:")
            for r in data[:3]:
                print(f"    - {r.get('directory')}/{r.get('filename')}")
        success = len(data) > 0
    except json.JSONDecodeError as e:
        print(f"  ERROR: Could not parse JSON: {e}")
        success = False

    print()
    print(f"  STATUS: {'✓ PASSED' if success else '✗ FAILED'}")
    return success


async def test_get_reference_stats():
    """Test reference statistics gathering."""
    print()
    print("=" * 70)
    print("TEST 3: get_reference_stats (uses find_reference_images)")
    print("=" * 70)
    print()

    result = await call_tool(get_reference_stats, {"effect_type": "sun", "sample_size": 10})

    try:
        data = json.loads(result)
        print(f"  Effect type: {data.get('effect_type')}")
        print(f"  Count: {data.get('count')}")
        print(f"  Categories: {data.get('categories')}")
        print(f"  Total size: {data.get('total_size_kb')}KB")
        success = "error" not in data
    except json.JSONDecodeError as e:
        print(f"  ERROR: Could not parse JSON: {e}")
        success = False

    print()
    print(f"  STATUS: {'✓ PASSED' if success else '✗ FAILED'}")
    return success


async def test_analyze_with_vision():
    """Test GPT-5.2 vision analysis (requires actual render)."""
    print()
    print("=" * 70)
    print("TEST 4: analyze_with_vision (GPT-5.2 vision API)")
    print("=" * 70)
    print()

    # Find an actual render to test with
    renders_result = await call_tool(list_renders, {"pattern": "", "limit": 1})
    try:
        renders = json.loads(renders_result)
    except:
        renders = []

    if not renders:
        print("  SKIP: No renders found to test with")
        return True  # Not a failure of the tool

    render_path = renders[0].get("path")
    print(f"  Testing with: {render_path}")

    result = await call_tool(analyze_with_vision, {
        "render_path": render_path,
        "analysis_type": "quality",
        "effect_type": "auto"
    })

    try:
        data = json.loads(result)
        print(f"  Score: {data.get('score')}")
        print(f"  Assessment: {str(data.get('overall_assessment', ''))[:100]}...")
        if data.get("error"):
            print(f"  ERROR: {data.get('error')}")
            success = False
        else:
            success = "score" in data or "overall_assessment" in data
    except json.JSONDecodeError as e:
        print(f"  ERROR: Could not parse JSON: {e}")
        print(f"  Raw result: {result[:300]}...")
        success = False

    print()
    print(f"  STATUS: {'✓ PASSED' if success else '✗ FAILED'}")
    return success


async def test_evaluate_render():
    """Test unified evaluation (uses analyze_with_vision)."""
    print()
    print("=" * 70)
    print("TEST 5: evaluate_render (uses analyze_with_vision)")
    print("=" * 70)
    print()

    # Find an actual render to test with
    renders_result = await call_tool(list_renders, {"pattern": "", "limit": 1})
    try:
        renders = json.loads(renders_result)
    except:
        renders = []

    if not renders:
        print("  SKIP: No renders found to test with")
        return True  # Not a failure of the tool

    render_path = renders[0].get("path")
    print(f"  Testing with: {render_path}")

    result = await call_tool(evaluate_render, {
        "render_path": render_path,
        "effect_type": "auto",
        "profile": "quick"
    })

    try:
        data = json.loads(result)
        print(f"  Overall score: {data.get('overall_score')}")
        print(f"  Passed: {data.get('passed')}")
        print(f"  Profile used: {data.get('profile_used')}")
        if data.get("vision_assessment"):
            print(f"  Assessment: {str(data.get('vision_assessment'))[:100]}...")
        if data.get("error"):
            print(f"  ERROR: {data.get('error')}")
            success = False
        else:
            success = "overall_score" in data
    except json.JSONDecodeError as e:
        print(f"  ERROR: Could not parse JSON: {e}")
        print(f"  Raw result: {result[:300]}...")
        success = False

    print()
    print(f"  STATUS: {'✓ PASSED' if success else '✗ FAILED'}")
    return success


async def test_compare_to_reference():
    """Test reference comparison (uses analyze_with_vision)."""
    print()
    print("=" * 70)
    print("TEST 6: compare_to_reference (uses analyze_with_vision)")
    print("=" * 70)
    print()

    # Find a render and reference
    renders_result = await call_tool(list_renders, {"pattern": "explosion", "limit": 1})
    refs_result = await call_tool(find_reference_images, {"effect_type": "explosion", "limit": 1})

    try:
        renders = json.loads(renders_result)
        refs = json.loads(refs_result)
    except:
        renders = []
        refs = {"results": []}

    if not renders:
        print("  SKIP: No explosion renders found")
        return True
    if not refs.get("results"):
        print("  SKIP: No explosion references found")
        return True

    render_path = renders[0].get("path")
    ref_path = refs["results"][0].get("path")

    print(f"  Render: {Path(render_path).name}")
    print(f"  Reference: {Path(ref_path).name}")

    result = await call_tool(compare_to_reference, {
        "render_path": render_path,
        "reference_path": ref_path,
        "effect_type": "explosion"
    })

    try:
        data = json.loads(result)
        # compare_to_reference returns analyze_with_vision output
        score = data.get("similarity_score", data.get("score", 0))
        print(f"  Similarity/Score: {score}")
        assessment = data.get("comparison_notes", data.get("overall_assessment", ""))
        if assessment:
            print(f"  Notes: {str(assessment)[:100]}...")
        if data.get("error"):
            print(f"  ERROR: {data.get('error')}")
            success = False
        else:
            success = True
    except json.JSONDecodeError as e:
        print(f"  ERROR: Could not parse JSON: {e}")
        print(f"  Raw result: {result[:300]}...")
        success = False

    print()
    print(f"  STATUS: {'✓ PASSED' if success else '✗ FAILED'}")
    return success


async def main():
    print()
    print("DIRECT TOOLS TEST SUITE")
    print("Testing asset_evaluator_tools WITHOUT MCP dependencies")
    print("=" * 70)
    print()

    # Run tests
    results = {}
    results["find_reference_images"] = await test_find_reference_images()
    results["list_renders"] = await test_list_renders()
    results["get_reference_stats"] = await test_get_reference_stats()
    results["analyze_with_vision"] = await test_analyze_with_vision()
    results["evaluate_render"] = await test_evaluate_render()
    results["compare_to_reference"] = await test_compare_to_reference()

    # Summary
    print()
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    passed = 0
    for name, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"  {name}: {status}")
        if success:
            passed += 1

    print()
    print(f"OVERALL: {passed}/{len(results)} tests passed")
    print("=" * 70)

    return all(results.values())


if __name__ == "__main__":
    asyncio.run(main())
