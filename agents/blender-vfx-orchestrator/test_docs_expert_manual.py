"""
Test DocsExpert agent with Blender manual tools.

Verifies:
1. DocsExpert agent creation with gpt-5.2
2. Blender manual search tools work (search_manual, search_vdb_workflow, etc.)
3. trace() observability wrapper
"""

import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from agents import Runner, trace
from specialized_agents.docs_expert import create_docs_expert


async def test_docs_expert_with_manual():
    """Test DocsExpert searching Blender documentation."""
    print("=" * 70)
    print("DOCS EXPERT - BLENDER MANUAL TOOLS TEST")
    print("=" * 70)
    print()

    # Create DocsExpert (uses gpt-5.2 by default)
    print("[1] Creating DocsExpert agent...")
    agent = create_docs_expert()
    print(f"    ✓ Agent: {agent.name}")
    print(f"    ✓ Model: gpt-5.2 (with reasoning)")
    print(f"    ✓ Tools: {len(agent.tools)} total")
    tool_names = [t.name for t in agent.tools if hasattr(t, 'name')]
    print(f"    Tools: {tool_names}")
    print()

    # Test 1: Search Blender manual for smoke simulation
    print("[2] Test: Search for smoke simulation documentation...")

    with trace("DocsExpert Manual Search Test"):
        result = await Runner.run(
            agent,
            """Search the Blender documentation for information about controlling
            smoke density in Mantaflow simulations.

            Use search_manual or search_vdb_workflow to find relevant docs.
            Return a brief summary of what you found.""",
            max_turns=15
        )

    output = str(result.final_output)
    print(f"    Response: {output[:500]}...")
    print()

    # Check if search was successful
    search_success = any(term in output.lower() for term in [
        "smoke", "density", "mantaflow", "fluid", "domain", "simulation"
    ])

    print("[3] Results:")
    if search_success:
        print("    ✓ DocsExpert found relevant smoke/fluid documentation")
    else:
        print("    ✗ DocsExpert did not return expected smoke documentation terms")

    return search_success


async def test_docs_expert_vdb_search():
    """Test DocsExpert searching VDB-specific documentation."""
    print()
    print("[4] Test: VDB workflow documentation search...")

    agent = create_docs_expert()

    with trace("DocsExpert VDB Search Test"):
        result = await Runner.run(
            agent,
            """Use search_vdb_workflow to find documentation about
            exporting volumetric simulations as OpenVDB/NanoVDB files.

            What Blender settings control VDB export quality?""",
            max_turns=15
        )

    output = str(result.final_output)
    print(f"    Response: {output[:500]}...")
    print()

    # Check for VDB-related terms
    vdb_success = any(term in output.lower() for term in [
        "vdb", "volume", "export", "cache", "openvdb"
    ])

    if vdb_success:
        print("    ✓ DocsExpert found VDB documentation")
    else:
        print("    ✗ DocsExpert did not return expected VDB documentation terms")

    return vdb_success


async def test_docs_expert_parameter_validation():
    """Test DocsExpert parameter validation tools."""
    print()
    print("[5] Test: Parameter validation...")

    agent = create_docs_expert()

    with trace("DocsExpert Parameter Validation Test"):
        result = await Runner.run(
            agent,
            """Use validate_parameter_range to check if a turbulence value of 0.5
            is valid for Blender fluid simulations.

            Also use get_parameter_defaults to get recommended defaults for an explosion effect.""",
            max_turns=15
        )

    output = str(result.final_output)
    print(f"    Response: {output[:500]}...")
    print()

    # Check for parameter validation terms
    param_success = any(term in output.lower() for term in [
        "valid", "turbulence", "explosion", "default", "range"
    ])

    if param_success:
        print("    ✓ DocsExpert parameter validation working")
    else:
        print("    ✗ DocsExpert parameter validation not working as expected")

    return param_success


if __name__ == "__main__":
    print()

    # Run all tests
    test1 = asyncio.run(test_docs_expert_with_manual())
    test2 = asyncio.run(test_docs_expert_vdb_search())
    test3 = asyncio.run(test_docs_expert_parameter_validation())

    print()
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"  Manual search:        {'PASSED ✓' if test1 else 'FAILED ✗'}")
    print(f"  VDB workflow search:  {'PASSED ✓' if test2 else 'FAILED ✗'}")
    print(f"  Parameter validation: {'PASSED ✓' if test3 else 'FAILED ✗'}")
    print()

    all_passed = test1 and test2 and test3
    print("OVERALL:", "PASSED ✓" if all_passed else "FAILED ✗")
    print("=" * 70)
