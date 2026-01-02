"""
List Sessions Tool

Returns all VFX asset generation sessions tracked by the orchestrator.
"""

import json
import sys
from pathlib import Path
from typing import Any

from claude_agent_sdk import tool

sys.path.insert(0, str(Path(__file__).parent.parent))


@tool(
    name="list_sessions",
    description="""List all VFX asset generation sessions.

    Returns sessions with their:
    - Session ID
    - Asset name
    - Status (in_progress, completed, failed, paused)
    - Best score achieved
    - Iteration count
    - Creation timestamp""",
    input_schema={
        "status_filter": str,
        "limit": int,
    },
)
async def list_sessions(args: dict[str, Any]) -> dict[str, Any]:
    """
    List all orchestrator sessions.

    Args:
        args: Dictionary containing:
            - status_filter (str): Optional filter by status
            - limit (int): Maximum number of sessions to return (default 20)

    Returns:
        Dictionary with list of session summaries
    """
    from orchestrator import BlenderOrchestratorAgent

    status_filter = args.get("status_filter", None)
    limit = args.get("limit", 20)

    try:
        agent = BlenderOrchestratorAgent()

        # list_sessions already returns List[Dict[str, Any]]
        sessions = agent.list_sessions(
            status_filter=status_filter,
            limit=limit
        )

        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps({
                        "count": len(sessions),
                        "sessions": sessions,
                    }, indent=2)
                }
            ]
        }

    except Exception as e:
        return {
            "content": [
                {"type": "text", "text": json.dumps({"error": str(e)}, indent=2)}
            ]
        }
