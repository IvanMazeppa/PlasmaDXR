#!/usr/bin/env python3
"""Display sample tool output to show what the MCP server generates"""
import asyncio
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR / "src"))

from pyro_server import call_tool

async def show_sample_output():
    """Show sample output from explosion designer"""

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

    print(result[0].text)

if __name__ == "__main__":
    asyncio.run(show_sample_output())
