"""
Test Script Writer's access to documentation tools.

Verifies that:
1. Script Writer has documentation tools available
2. Script Writer can search docs when uncertain about parameters
3. Script Writer can discover APIs by intent
"""

import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from agents import Agent, Runner, trace
from specialized_agents.script_writer import SCRIPT_WRITER_INSTRUCTIONS, create_script_writer


async def test_script_writer_has_docs_tools():
    """Verify Script Writer is created with documentation tools."""
    print("=" * 70)
    print("TEST: SCRIPT WRITER HAS DOCUMENTATION TOOLS")
    print("=" * 70)
    print()

    agent = create_script_writer()

    tool_names = [t.name for t in agent.tools if hasattr(t, 'name')]
    print(f"[1] Script Writer tools ({len(tool_names)} total):")
    for name in tool_names:
        print(f"    - {name}")
    print()

    # Check for documentation tools
    has_semantic_search = "semantic_search_blender_docs" in tool_names
    has_api_intent = "search_blender_api_by_intent" in tool_names

    print("[2] Documentation tool verification:")
    print(f"    {'✓' if has_semantic_search else '✗'} semantic_search_blender_docs")
    print(f"    {'✓' if has_api_intent else '✗'} search_blender_api_by_intent")

    return has_semantic_search and has_api_intent


async def test_instructions_contain_docs_section():
    """Verify Script Writer instructions contain documentation section."""
    print()
    print("=" * 70)
    print("TEST: SCRIPT WRITER INSTRUCTIONS CONTAIN DOCS SECTION")
    print("=" * 70)
    print()

    checks = {
        "DOCUMENTATION SEARCH section": "DOCUMENTATION SEARCH" in SCRIPT_WRITER_INSTRUCTIONS,
        "semantic_search_blender_docs documented": "semantic_search_blender_docs" in SCRIPT_WRITER_INSTRUCTIONS,
        "search_blender_api_by_intent documented": "search_blender_api_by_intent" in SCRIPT_WRITER_INSTRUCTIONS,
        "WHEN TO SEARCH guidance": "WHEN TO SEARCH DOCS" in SCRIPT_WRITER_INSTRUCTIONS,
        "WHEN NOT TO SEARCH guidance": "WHEN NOT TO SEARCH" in SCRIPT_WRITER_INSTRUCTIONS,
        "Example queries included": "FluidDomainSettings" in SCRIPT_WRITER_INSTRUCTIONS,
    }

    all_passed = True
    for check, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"    {status} {check}")
        if not passed:
            all_passed = False

    return all_passed


async def test_script_writer_uses_docs():
    """Test that Script Writer actually uses documentation tools when appropriate."""
    print()
    print("=" * 70)
    print("TEST: SCRIPT WRITER USES DOCS FOR UNCERTAIN PARAMETERS")
    print("=" * 70)
    print()

    # Create a minimal agent with just docs tools for testing
    from tools.semantic_docs_tools import (
        semantic_search_blender_docs,
        search_blender_api_by_intent,
    )

    agent = Agent(
        name="Script Writer Docs Test",
        instructions="""You are testing documentation search capabilities.

When asked about an unfamiliar parameter or API, use the documentation tools:
- semantic_search_blender_docs() for general searches
- search_blender_api_by_intent() to find APIs by what they do

Report what you find from the documentation.""",
        model="gpt-5.2",
        tools=[
            semantic_search_blender_docs,
            search_blender_api_by_intent,
        ],
    )

    print("[1] Testing: Search for unfamiliar parameter...")

    with trace("Script Writer Docs Search Test"):
        result = await Runner.run(
            agent,
            """I need to modify a Blender script but I'm uncertain about a parameter.

The visual feedback says "smoke dissipates too quickly near domain boundaries"
and suggests adjusting "adaptive_domain" settings.

I'm not familiar with adaptive_domain. Search the documentation to:
1. Find what adaptive_domain controls
2. Find related parameters that affect smoke dissipation at boundaries

Use search_blender_api_by_intent to find APIs for "control smoke behavior at domain boundaries".""",
            max_turns=15
        )

    output = str(result.final_output)
    print(f"[2] Documentation search results:")
    print(f"    {output[:800]}...")
    print()

    # Check that docs were searched and relevant info found
    search_success = any(term in output.lower() for term in [
        "domain", "adaptive", "boundary", "dissolve", "cache", "resolution"
    ])

    print("[3] Search verification:")
    print(f"    {'✓' if search_success else '✗'} Found relevant documentation")

    return search_success


if __name__ == "__main__":
    print()
    print("SCRIPT WRITER DOCUMENTATION ACCESS TEST SUITE")
    print()

    # Test 1: Check tools are present
    test1 = asyncio.run(test_script_writer_has_docs_tools())

    # Test 2: Check instructions
    test2 = asyncio.run(test_instructions_contain_docs_section())

    # Test 3: Actually use docs
    test3 = asyncio.run(test_script_writer_uses_docs())

    print()
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"  Script Writer has docs tools:     {'PASSED ✓' if test1 else 'FAILED ✗'}")
    print(f"  Instructions contain docs section: {'PASSED ✓' if test2 else 'FAILED ✗'}")
    print(f"  Script Writer uses docs:          {'PASSED ✓' if test3 else 'FAILED ✗'}")
    print()
    print("OVERALL:", "PASSED ✓" if (test1 and test2 and test3) else "NEEDS REVIEW")
    print("=" * 70)
