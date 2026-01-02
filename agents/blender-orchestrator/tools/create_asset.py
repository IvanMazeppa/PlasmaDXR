"""
Create Asset Tool

Starts a new VFX asset generation session with the Blender orchestrator.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from claude_agent_sdk import tool

# Add parent to path for orchestrator import
sys.path.insert(0, str(Path(__file__).parent.parent))


@tool(
    name="create_asset",
    description="""Start a new VFX asset generation session with the Blender orchestrator.

    The orchestrator will:
    1. Generate a Blender Python script using script-generator
    2. Execute the simulation using blender-executor
    3. Evaluate quality using asset-evaluator (VFX metrics)
    4. Iterate until quality threshold met or max iterations reached

    Returns session info with progress updates.""",
    input_schema={
        "asset_name": str,
        "effect_type": str,
        "description": str,
        "reference_path": str,
        "resolution": int,
        "frame_end": int,
    },
)
async def create_asset(args: dict[str, Any]) -> dict[str, Any]:
    """
    Create a new VFX asset through the Blender pipeline.

    Args:
        args: Dictionary containing:
            - asset_name (str): Name for the asset
            - effect_type (str): Type of effect (pyro, explosion, fire, etc.)
            - description (str): Description of what to create
            - reference_path (str): Optional path to reference image
            - resolution (int): Blender simulation resolution (default 96)
            - frame_end (int): Animation end frame (default 50)

    Returns:
        Dictionary with session status and results
    """
    from orchestrator import BlenderOrchestratorAgent

    asset_name = args.get("asset_name", "unnamed_asset")
    effect_type = args.get("effect_type", "pyro")
    description = args.get("description", "A VFX asset")
    reference_path = args.get("reference_path", "")
    resolution = args.get("resolution", 96)
    frame_end = args.get("frame_end", 50)

    try:
        # Create and start orchestrator
        agent = BlenderOrchestratorAgent()
        await agent.start()

        # Run asset creation
        result = await agent.create_asset(
            asset_name=asset_name,
            effect_type=effect_type,
            description=description,
            reference_path=reference_path if reference_path else None,
            resolution=resolution,
            frame_end=frame_end,
        )

        await agent.stop()

        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps({
                        "status": "completed",
                        "session_id": result.get("session_id", "unknown"),
                        "best_score": result.get("best_score", 0),
                        "iterations": result.get("iterations", 0),
                        "final_stage": result.get("final_stage", "unknown"),
                        "output_path": result.get("output_path", ""),
                        "errors": result.get("errors", []),
                    }, indent=2)
                }
            ]
        }

    except Exception as e:
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps({
                        "status": "error",
                        "error": str(e),
                        "asset_name": asset_name,
                    }, indent=2)
                }
            ]
        }
