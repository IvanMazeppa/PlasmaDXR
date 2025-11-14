#!/usr/bin/env python3
"""
Quick test to verify MCP server can list tools and execute them
"""
import asyncio
import sys
from pathlib import Path

# Add src directory to path
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR / "src"))

from pyro_server import server, list_tools, call_tool

async def test_server():
    """Test MCP server functionality"""

    print("=" * 60)
    print("Testing DXR Volumetric Pyro Specialist MCP Server")
    print("=" * 60)

    # Test 1: List tools
    print("\n[TEST 1] Listing available tools...")
    tools = await list_tools()
    print(f"✅ Found {len(tools)} tools:")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description[:60]}...")

    # Test 2: Execute a simple tool
    print("\n[TEST 2] Testing design_explosion_effect tool...")
    result = await call_tool(
        name="design_explosion_effect",
        arguments={
            "effect_type": "supernova",
            "duration_seconds": 5.0,
            "max_radius_meters": 500.0,
            "peak_temperature_kelvin": 100000.0,
            "particle_budget": 10000
        }
    )

    if result and len(result) > 0:
        output = result[0].text
        print("✅ Tool executed successfully!")
        print(f"   Output length: {len(output)} characters")
        print(f"   First 200 chars: {output[:200]}...")
    else:
        print("❌ Tool execution failed")
        return False

    # Test 3: Execute performance estimator
    print("\n[TEST 3] Testing estimate_pyro_performance tool...")
    result = await call_tool(
        name="estimate_pyro_performance",
        arguments={
            "effect_complexity": "moderate",
            "particle_count": 10000,
            "noise_octaves": 3
        }
    )

    if result and len(result) > 0:
        print("✅ Performance estimator working!")
    else:
        print("❌ Performance estimator failed")
        return False

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED - MCP Server is fully functional!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = asyncio.run(test_server())
    sys.exit(0 if success else 1)
