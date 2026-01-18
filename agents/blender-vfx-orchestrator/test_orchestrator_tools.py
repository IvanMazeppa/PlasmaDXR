"""
Test that orchestrator tools can access SharedContext via RunContextWrapper.
"""

import asyncio
import json
from dotenv import load_dotenv

load_dotenv()

from models.shared_context import AssetRequest, EffectType, SharedContext
from orchestrator import generate_session_id
from utils import create_session_from_request
from agents import Agent, Runner, trace

# Import the actual orchestrator tools
from tools.proactive_research_tools import pre_iteration_research
from tools.code_pattern_tools import search_code_patterns, list_patterns_by_effect


async def test_orchestrator_tools_with_context():
    """Test that orchestrator tools can access context."""
    print("=" * 70)
    print("ORCHESTRATOR TOOLS CONTEXT ACCESS TEST")
    print("=" * 70)
    print()
    
    # Create test context
    request = AssetRequest(
        asset_name="tool_context_test",
        description="Test for tool context access",
        effect_type=EffectType.EXPLOSION,  # Using explosion to verify auto-population
    )
    
    session_id = generate_session_id(request.asset_name)
    context = create_session_from_request(request, session_id)
    
    print(f"[1] Created SharedContext:")
    print(f"    Session ID: {session_id}")
    print(f"    Effect Type: {context.session.request.effect_type.value}")
    print()
    
    # Create agent with orchestrator tools
    test_agent = Agent[SharedContext](
        name="Tool Test Agent",
        instructions="""You are testing that tools can access context.

Call list_patterns_by_effect() WITHOUT providing the effect_type parameter.
The tool should auto-populate effect_type from the session context (should be "explosion").

Report the tool's output including what effect_type it used.""",
        model="gpt-5.2",
        tools=[
            search_code_patterns,
            list_patterns_by_effect,
        ],
    )
    
    print(f"[2] Created Agent with orchestrator tools")
    print()
    
    # Run the test
    print(f"[3] Running list_patterns_by_effect (should auto-populate effect_type=explosion)...")
    
    with trace("Orchestrator Tools Context Test"):
        result = await Runner.run(
            test_agent,
            """Call list_patterns_by_effect() with NO parameters.
The tool should auto-populate effect_type from context (it should use "explosion").
Report exactly what the tool returned, especially the effect_type field.""",
            context=context,
            max_turns=3
        )
    
    output = str(result.final_output)
    print(f"    Output: {output}")
    print()
    
    # Check for explosion effect type (from context)
    success = "explosion" in output.lower()
    
    print(f"[4] Context Auto-Population Test:")
    if success:
        print(f"    ✓ effect_type=explosion was auto-populated from context!")
    else:
        print(f"    ✗ effect_type was not auto-populated (expected 'explosion' in output)")
    
    return success


if __name__ == "__main__":
    success = asyncio.run(test_orchestrator_tools_with_context())
    print()
    print("=" * 70)
    print("TEST RESULT:", "PASSED ✓" if success else "FAILED ✗")
    print("=" * 70)
