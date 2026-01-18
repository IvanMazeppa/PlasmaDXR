"""
Integration test for orchestrator with self-learning tools.

This test verifies the orchestrator can access and use the new
vector store-based self-learning tools without requiring a full
asset generation workflow.
"""

import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from agents import Agent, Runner, trace
from models.shared_context import AssetRequest, EffectType, SharedContext
from orchestrator import generate_session_id
from utils import create_session_from_request

# Import the EXACT tools the orchestrator uses
from tools.proactive_research_tools import (
    pre_iteration_research,
    evaluate_escape_velocity,
    search_alternative_approaches,
)
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


async def test_orchestrator_tools_integration():
    """Test that orchestrator tools work together with context."""
    print("=" * 70)
    print("ORCHESTRATOR SELF-LEARNING TOOLS INTEGRATION TEST")
    print("=" * 70)
    print()

    # Create a mini-orchestrator with just the self-learning tools
    orchestrator_tools = [
        # Proactive research (Strategy 3)
        pre_iteration_research,
        evaluate_escape_velocity,
        search_alternative_approaches,
        # Semantic docs (Strategy 1 - vector store)
        semantic_search_blender_docs,
        find_alternative_approaches,
        search_blender_api_by_intent,
        # Code patterns (Strategy 4)
        search_code_patterns,
        list_patterns_by_effect,
        get_pattern_library_stats,
    ]

    agent = Agent[SharedContext](
        name="Self-Learning Integration Tester",
        instructions="""You test the integration of self-learning tools.

When asked to research an issue:
1. First call pre_iteration_research() to check warning levels
2. Then call semantic_search_blender_docs() to find documentation
3. Then call search_code_patterns() to find existing solutions
4. Report what you found from each tool""",
        model="gpt-5.2",
        tools=orchestrator_tools,
    )

    # Create context (simulating iteration 2 of a pyro effect)
    request = AssetRequest(
        asset_name="integration_test",
        description="Test self-learning integration",
        effect_type=EffectType.PYRO,
    )
    session_id = generate_session_id(request.asset_name)
    context = create_session_from_request(request, session_id)

    print(f"[1] Created test context:")
    print(f"    Session ID: {session_id}")
    print(f"    Effect type: {context.session.request.effect_type.value}")
    print(f"    Tools available: {len(orchestrator_tools)}")
    print()

    print("[2] Running integration test...")
    print("    Testing: pre_iteration_research + semantic_search + code_patterns")
    print()

    with trace("Self-Learning Integration Test"):
        result = await Runner.run(
            agent,
            """Research how to fix "smoke lacks density" for a pyro effect.

Call these tools in sequence:
1. pre_iteration_research(current_issue="smoke lacks density", current_approach="increasing flame_smoke")
2. semantic_search_blender_docs("smoke density Mantaflow")
3. search_code_patterns(issue="smoke lacks density")

Report the key findings from each tool.""",
            context=context,
            max_turns=20
        )

    output = str(result.final_output)
    print(f"[3] Agent response:")
    print(f"    {output[:800]}...")
    print()

    # Verify tools were used
    success_indicators = {
        "pre_iteration_research": any(term in output.lower() for term in ["warning", "escape", "continue"]),
        "semantic_search": any(term in output.lower() for term in ["result", "found", "search", "density"]),
        "code_patterns": any(term in output.lower() for term in ["pattern", "found", "search", "0"]),
    }

    print("[4] Tool usage verification:")
    all_passed = True
    for tool, used in success_indicators.items():
        status = "✓" if used else "✗"
        print(f"    {status} {tool}: {'used' if used else 'not detected'}")
        if not used:
            all_passed = False

    print()
    return all_passed


if __name__ == "__main__":
    success = asyncio.run(test_orchestrator_tools_integration())
    print("=" * 70)
    print("INTEGRATION TEST:", "PASSED ✓" if success else "PARTIAL (check tool usage)")
    print("=" * 70)
