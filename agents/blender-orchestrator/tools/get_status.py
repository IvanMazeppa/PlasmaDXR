"""
Get Status Tool

Returns the current status of the Blender orchestrator including
trust score, autonomy level, and active sessions.
"""

import json
import sys
from pathlib import Path
from typing import Any

from claude_agent_sdk import tool

sys.path.insert(0, str(Path(__file__).parent.parent))


@tool(
    name="get_status",
    description="""Get the current status of the Blender VFX orchestrator.

    Returns:
    - Trust score (0.0 - 1.0)
    - Current autonomy level (supervised, guided, autonomous, trusted)
    - Token usage limits and current usage
    - Active session info if any""",
    input_schema={},
)
async def get_status(args: dict[str, Any]) -> dict[str, Any]:
    """
    Get orchestrator status.

    Returns:
        Dictionary with orchestrator status information
    """
    from orchestrator import BlenderOrchestratorAgent

    try:
        agent = BlenderOrchestratorAgent()

        trust_score = agent.autonomy.get_trust_score()
        autonomy_level = agent.autonomy.get_level()

        status = {
            "trust_score": trust_score,
            "autonomy_level": autonomy_level.value,
            "token_limits": {
                "per_session": agent.guardrail_config.token_limits.per_session,
                "per_day": agent.guardrail_config.token_limits.per_day,
            },
            "cost_limits": {
                "per_session": agent.guardrail_config.cost_limits.per_session,
                "per_day": agent.guardrail_config.cost_limits.per_day,
            },
            "workflow_config": {
                "max_iterations": agent.workflow_config.max_iterations,
                "max_retries": agent.workflow_config.max_retries,
            },
            "quality_thresholds": {
                "vfx_pass": agent.quality_config.vfx_pass_threshold,
                "temporal": agent.quality_config.temporal_threshold,
            },
        }

        # Check for active sessions (list_sessions returns List[Dict])
        sessions = agent.list_sessions(status_filter="in_progress")
        if sessions:
            status["active_sessions"] = [
                {"session_id": s["session_id"], "asset_name": s.get("asset_name", "unknown")}
                for s in sessions[:5]
            ]

        return {
            "content": [
                {"type": "text", "text": json.dumps(status, indent=2)}
            ]
        }

    except Exception as e:
        return {
            "content": [
                {"type": "text", "text": json.dumps({"error": str(e)}, indent=2)}
            ]
        }
