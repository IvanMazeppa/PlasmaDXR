#!/usr/bin/env python3
"""
Quick test script for blender-vfx-orchestrator MCP tools.

Run this to verify the server works after the get_status fix.
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from server import (
    get_status,
    list_sessions,
    get_effect_types,
)


async def main():
    print("=" * 60)
    print("Testing blender-vfx-orchestrator MCP Tools")
    print("=" * 60)

    # Test 1: get_status (this was the one that hung)
    print("\n1. Testing get_status()...")
    try:
        result = await get_status()
        data = json.loads(result)
        print(f"   ✓ Success!")
        print(f"   - Orchestrator version: {data.get('orchestrator_version')}")
        print(f"   - Initialized: {data.get('initialized')}")
        print(f"   - Budget: ${data.get('budget', {}).get('total_spent_usd', 0):.2f} / ${data.get('budget', {}).get('monthly_limit_usd', 0):.2f}")
        print(f"   - DocsExpert connected: {data.get('docs_expert_connected')}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return 1

    # Test 2: list_sessions
    print("\n2. Testing list_sessions()...")
    try:
        result = await list_sessions()
        data = json.loads(result)
        print(f"   ✓ Success!")
        print(f"   - Total sessions: {data.get('total', 0)}")
        print(f"   - Sessions: {len(data.get('sessions', []))}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return 1

    # Test 3: get_effect_types
    print("\n3. Testing get_effect_types()...")
    try:
        result = await get_effect_types()
        data = json.loads(result)
        print(f"   ✓ Success!")
        print(f"   - Effect types: {data.get('effect_types', [])}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return 1

    print("\n" + "=" * 60)
    print("All MCP tool tests PASSED!")
    print("=" * 60)
    print("\nThe server is ready for use. Add to your MCP config and test:")
    print('  - get_status: Returns immediately (was hanging before)')
    print('  - list_sessions: Lists saved sessions')
    print('  - get_effect_types: Lists available effect types')
    print('  - create_asset: Starts asset generation (triggers initialization)')

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
