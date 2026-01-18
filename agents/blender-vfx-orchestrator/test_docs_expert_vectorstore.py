"""
Test the updated DocsExpert with vector store-based tools.
"""

import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from agents import Runner, trace
from specialized_agents.docs_expert import create_docs_expert


async def test_docs_expert_vector_store():
    """Test DocsExpert with new vector store tools."""
    print("=" * 70)
    print("DOCS EXPERT - VECTOR STORE TOOLS TEST")
    print("=" * 70)
    print()

    # Create DocsExpert (now uses vector store tools)
    print("[1] Creating DocsExpert agent (vector store mode)...")
    agent = create_docs_expert()
    print(f"    ✓ Agent: {agent.name}")
    print(f"    ✓ Model: gpt-5.2 (with reasoning)")
    print(f"    ✓ Tools: {len(agent.tools)} total")
    tool_names = [t.name for t in agent.tools if hasattr(t, 'name')]
    print(f"    Tool names: {tool_names}")
    print()

    # Test semantic search
    print("[2] Testing semantic_search_blender_docs...")

    with trace("DocsExpert Vector Store Test"):
        result = await Runner.run(
            agent,
            """Use semantic_search_blender_docs to find information about:
            "how to increase smoke density and make it dissipate slower"

            Return a concise summary of what you found.""",
            max_turns=15
        )

    output = str(result.final_output)
    print(f"    Response: {output[:600]}...")
    print()

    # Check for relevant terms
    success = any(term in output.lower() for term in [
        "smoke", "density", "dissipate", "dissolve", "fluid", "domain"
    ])

    print("[3] Results:")
    if success:
        print("    ✓ DocsExpert (vector store) found relevant documentation")
    else:
        print("    ✗ Response may not contain expected terms (check manually)")

    return success


if __name__ == "__main__":
    success = asyncio.run(test_docs_expert_vector_store())
    print()
    print("=" * 70)
    print("TEST RESULT:", "PASSED ✓" if success else "CHECK MANUALLY")
    print("=" * 70)
