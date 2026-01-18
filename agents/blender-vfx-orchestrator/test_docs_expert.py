"""
Test DocsExpert with Blender manual tools.
"""

import asyncio
from dotenv import load_dotenv

load_dotenv()

from specialized_agents.docs_expert import create_docs_expert
from agents import Runner, trace


async def test_docs_expert():
    """Test DocsExpert with blender-manual tools."""
    print("=" * 70)
    print("DOCS EXPERT + BLENDER MANUAL TOOLS TEST")
    print("=" * 70)
    print()
    
    # Create DocsExpert
    print("[1] Creating DocsExpert agent...")
    docs_expert = create_docs_expert()
    print(f"    ✓ Agent created with {len(docs_expert.tools)} tools")
    print(f"    Tools: {[t.name for t in docs_expert.tools]}")
    print()
    
    # Run a documentation search
    print("[2] Searching Blender docs for 'smoke density simulation'...")
    
    with trace("DocsExpert Test"):
        result = await Runner.run(
            docs_expert,
            """Search the Blender documentation for information about smoke density in fluid simulations.
Use search_manual or search_python_api to find relevant documentation.
Summarize what you find about controlling smoke density.""",
            max_turns=5
        )
    
    output = str(result.final_output)
    print(f"    Output preview: {output[:500]}...")
    print()
    
    # Check for success indicators
    success_indicators = []
    output_lower = output.lower()
    
    if "density" in output_lower:
        success_indicators.append("density mentioned in results")
    if "smoke" in output_lower or "fluid" in output_lower:
        success_indicators.append("smoke/fluid documentation found")
    if "flame_smoke" in output_lower or "dissolve" in output_lower:
        success_indicators.append("specific parameters found")
    if len(output) > 200:
        success_indicators.append("substantial response received")
    
    print("[3] Results:")
    for indicator in success_indicators:
        print(f"    ✓ {indicator}")
    
    return len(success_indicators) >= 2


if __name__ == "__main__":
    success = asyncio.run(test_docs_expert())
    print()
    print("=" * 70)
    print("TEST RESULT:", "PASSED ✓" if success else "FAILED ✗")
    print("=" * 70)
