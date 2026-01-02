"""
Resume Session Tool

Resumes a paused or incomplete VFX asset generation session.
"""

import json
import sys
from pathlib import Path
from typing import Any

from claude_agent_sdk import tool

sys.path.insert(0, str(Path(__file__).parent.parent))


@tool(
    name="resume_session",
    description="""Resume a paused or incomplete VFX asset generation session.

    Continues from the last checkpoint:
    - Loads session state (parameters, scores, iteration count)
    - Continues iteration loop from where it stopped
    - Uses same quality thresholds and configuration

    Returns updated session status upon completion or pause.""",
    input_schema={
        "session_id": str,
    },
)
async def resume_session(args: dict[str, Any]) -> dict[str, Any]:
    """
    Resume an existing session.

    Args:
        args: Dictionary containing:
            - session_id (str): ID of the session to resume

    Returns:
        Dictionary with resumed session results
    """
    from orchestrator import BlenderOrchestratorAgent

    session_id = args.get("session_id")

    if not session_id:
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps({
                        "status": "error",
                        "error": "session_id is required",
                    }, indent=2)
                }
            ]
        }

    try:
        agent = BlenderOrchestratorAgent()
        await agent.start()

        result = await agent.resume_session(session_id)

        await agent.stop()

        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps({
                        "status": "resumed",
                        "session_id": session_id,
                        "best_score": result.get("best_score", 0),
                        "total_iterations": result.get("total_iterations", 0),
                        "quality_passed": result.get("quality_passed", False),
                        "trust_score": result.get("trust_score", 0),
                        "best_render_path": result.get("best_render_path", ""),
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
                        "session_id": session_id,
                    }, indent=2)
                }
            ]
        }
