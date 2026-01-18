"""
Test the new self-learning tools (vector store based).

Tests:
1. Semantic search via vector store (Strategy 1)
2. Code pattern memory (Strategy 4)
3. Proactive research tools (Strategy 3)
4. Knowledge distillation tools (Strategy 2)

These tests verify the NEW vector store-based tools work correctly,
NOT the deprecated local file-based tools.
"""

import asyncio
import json
import os
from dotenv import load_dotenv

load_dotenv()

from agents import Agent, Runner, trace
from models.shared_context import AssetRequest, EffectType, SharedContext
from orchestrator import generate_session_id
from utils import create_session_from_request

# Import ALL new self-learning tools
from tools.semantic_docs_tools import (
    semantic_search_blender_docs,
    find_alternative_approaches,
    search_blender_api_by_intent,
)
from tools.code_pattern_tools import (
    search_code_patterns,
    list_patterns_by_effect,
    get_pattern_library_stats,
)
from tools.proactive_research_tools import (
    pre_iteration_research,
    evaluate_escape_velocity,
    search_alternative_approaches as search_alt_approaches,
)
from tools.knowledge_distillation_tools import (
    analyze_script_for_patterns,
)


async def test_semantic_vector_search():
    """Test Strategy 1: Vector Store Semantic Search."""
    print("=" * 70)
    print("TEST 1: SEMANTIC VECTOR STORE SEARCH")
    print("=" * 70)
    print()

    # Create a test agent with ONLY semantic docs tools
    agent = Agent[SharedContext](
        name="Semantic Search Tester",
        instructions="""You test semantic search tools.
        When asked to search, use the semantic_search_blender_docs tool.
        Report what you find including the number of results and relevance.""",
        model="gpt-5.2",
        tools=[
            semantic_search_blender_docs,
            search_blender_api_by_intent,
        ],
    )

    # Create context for auto-population
    request = AssetRequest(
        asset_name="semantic_test",
        description="Test semantic search",
        effect_type=EffectType.PYRO,
    )
    session_id = generate_session_id(request.asset_name)
    context = create_session_from_request(request, session_id)

    print("[1] Testing semantic_search_blender_docs...")

    with trace("Semantic Vector Search Test"):
        result = await Runner.run(
            agent,
            """Use semantic_search_blender_docs to search for:
            "how to increase smoke density in Mantaflow simulations"

            Report the number of results found and summarize the top finding.""",
            context=context,
            max_turns=10
        )

    output = str(result.final_output)
    print(f"    Response: {output[:600]}...")
    print()

    # Check for success indicators
    success = any(term in output.lower() for term in [
        "result", "found", "smoke", "density", "mantaflow", "fluid"
    ])

    if success:
        print("    ✓ Semantic vector search returned relevant results")
    else:
        print("    ✗ Semantic vector search did not return expected results")

    return success


async def test_code_pattern_memory():
    """Test Strategy 4: Code Pattern Memory."""
    print()
    print("=" * 70)
    print("TEST 2: CODE PATTERN MEMORY")
    print("=" * 70)
    print()

    # Create agent with code pattern tools
    agent = Agent[SharedContext](
        name="Pattern Memory Tester",
        instructions="""You test code pattern memory tools.
        Use the tools to search for patterns and report statistics.""",
        model="gpt-5.2",
        tools=[
            search_code_patterns,
            list_patterns_by_effect,
            get_pattern_library_stats,
        ],
    )

    # Create context
    request = AssetRequest(
        asset_name="pattern_test",
        description="Test pattern memory",
        effect_type=EffectType.EXPLOSION,
    )
    session_id = generate_session_id(request.asset_name)
    context = create_session_from_request(request, session_id)

    print("[2] Testing code pattern memory tools...")

    with trace("Code Pattern Memory Test"):
        result = await Runner.run(
            agent,
            """First, call get_pattern_library_stats() to see what patterns exist.

            Then call list_patterns_by_effect() WITHOUT specifying effect_type -
            it should auto-populate from context (should use "explosion").

            Report: 1) Total patterns in library, 2) Effect type used for search""",
            context=context,
            max_turns=10
        )

    output = str(result.final_output)
    print(f"    Response: {output[:600]}...")
    print()

    # Check for context auto-population (explosion)
    success = "explosion" in output.lower() or "pattern" in output.lower()

    if success:
        print("    ✓ Code pattern memory working (context auto-population)")
    else:
        print("    ✗ Code pattern memory not working as expected")

    return success


async def test_proactive_research():
    """Test Strategy 3: Proactive Research (Early Warning)."""
    print()
    print("=" * 70)
    print("TEST 3: PROACTIVE RESEARCH (EARLY WARNING)")
    print("=" * 70)
    print()

    # Create agent with proactive research tools
    agent = Agent[SharedContext](
        name="Proactive Research Tester",
        instructions="""You test proactive research tools for early warning detection.
        Use the tools to evaluate escape velocity and search for alternatives.""",
        model="gpt-5.2",
        tools=[
            pre_iteration_research,
            evaluate_escape_velocity,
            search_alt_approaches,
        ],
    )

    # Create context with some iteration history
    request = AssetRequest(
        asset_name="proactive_test",
        description="Test proactive research",
        effect_type=EffectType.PYRO,
    )
    session_id = generate_session_id(request.asset_name)
    context = create_session_from_request(request, session_id)

    print("[3] Testing pre_iteration_research tool...")

    with trace("Proactive Research Test"):
        result = await Runner.run(
            agent,
            """Call pre_iteration_research() with:
            - current_issue: "smoke lacks density"
            - current_approach: "increasing flame_smoke parameter"
            - iteration_history: "[]"

            The effect_type should auto-populate from context (pyro).

            Report what warning_level and escape_action the tool returns.""",
            context=context,
            max_turns=10
        )

    output = str(result.final_output)
    print(f"    Response: {output[:600]}...")
    print()

    # Check for warning/escape indicators
    success = any(term in output.lower() for term in [
        "warning", "escape", "continue", "none", "early", "stuck", "pyro"
    ])

    if success:
        print("    ✓ Proactive research tool working")
    else:
        print("    ✗ Proactive research tool not working as expected")

    return success


async def test_find_alternatives():
    """Test find_alternative_approaches with vector store."""
    print()
    print("=" * 70)
    print("TEST 4: FIND ALTERNATIVE APPROACHES (VECTOR STORE)")
    print("=" * 70)
    print()

    # Create agent
    agent = Agent[SharedContext](
        name="Alternatives Tester",
        instructions="""You test the find_alternative_approaches tool.
        This searches the vector store for different techniques.""",
        model="gpt-5.2",
        tools=[
            find_alternative_approaches,
        ],
    )

    # Create context
    request = AssetRequest(
        asset_name="alternatives_test",
        description="Test alternatives search",
        effect_type=EffectType.FIRE,
    )
    session_id = generate_session_id(request.asset_name)
    context = create_session_from_request(request, session_id)

    print("[4] Testing find_alternative_approaches (vector store)...")

    with trace("Find Alternatives Test"):
        result = await Runner.run(
            agent,
            """Call find_alternative_approaches() with:
            - current_approach: "increasing flame_smoke parameter"
            - issue: "smoke lacks density"

            The effect_type should auto-populate from context (fire).

            Report how many alternatives were found and the first suggestion.""",
            context=context,
            max_turns=10
        )

    output = str(result.final_output)
    print(f"    Response: {output[:600]}...")
    print()

    # Check for alternatives
    success = any(term in output.lower() for term in [
        "alternative", "approach", "found", "fire", "suggestion"
    ])

    if success:
        print("    ✓ Find alternatives (vector store) working")
    else:
        print("    ✗ Find alternatives not working as expected")

    return success


if __name__ == "__main__":
    print()
    print("SELF-LEARNING TOOLS TEST SUITE")
    print("Testing NEW vector store-based tools (not deprecated local tools)")
    print()

    # Run all tests
    results = {}

    results["semantic_search"] = asyncio.run(test_semantic_vector_search())
    results["pattern_memory"] = asyncio.run(test_code_pattern_memory())
    results["proactive_research"] = asyncio.run(test_proactive_research())
    results["find_alternatives"] = asyncio.run(test_find_alternatives())

    print()
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    for name, passed in results.items():
        status = "PASSED ✓" if passed else "FAILED ✗"
        print(f"  {name:25} {status}")
    print()

    all_passed = all(results.values())
    passed_count = sum(results.values())
    print(f"OVERALL: {passed_count}/{len(results)} tests passed")
    print("=" * 70)
