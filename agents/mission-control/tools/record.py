"""
Record Decision Tool

Logs strategic decisions with rationale and artifact links to session files.
"""

from datetime import datetime
from pathlib import Path
from typing import Any

from claude_agent_sdk import tool


@tool(
    name="record_decision",
    description="""Record a strategic decision with rationale and supporting artifacts.

    Decisions are logged to docs/sessions/SESSION_<date>.md with:
    - Decision description
    - Rationale explaining why this choice was made
    - Links to supporting artifacts (PIX captures, screenshots, buffer dumps)
    - Timestamp and agent context

    Returns confirmation of successful recording.""",
    input_schema={
        "decision": str,
        "rationale": str,
        "artifacts": list,
    },
)
async def record_decision(args: dict[str, Any]) -> dict[str, Any]:
    """
    Record a decision to the session log file.

    Args:
        args: Dictionary containing:
            - decision (str): What was decided
            - rationale (str): Why it was decided
            - artifacts (list[str]): Paths to supporting files

    Returns:
        Dictionary with:
            - status (str): "recorded" or "error"
            - file_path (str): Path to session log
            - timestamp (str): When recorded

    TODO: Add decision categorization (performance, quality, architecture)
    TODO: Cross-reference with Git commits
    """
    decision: str = args["decision"]
    rationale: str = args["rationale"]
    artifacts: list[str] = args.get("artifacts", [])

    # Generate session file path (relative to project root)
    project_root = Path(__file__).parent.parent.parent.parent
    today = datetime.now().strftime("%Y-%m-%d")
    sessions_dir = project_root / "docs" / "sessions"
    session_file = sessions_dir / f"SESSION_{today}.md"

    # Create sessions directory if it doesn't exist
    sessions_dir.mkdir(parents=True, exist_ok=True)

    # Format timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Format artifact links
    artifact_links = "\n".join(f"  - `{art}`" for art in artifacts) if artifacts else "  - None"

    # Create decision entry
    decision_entry = f"""
## Decision: {decision}

**Timestamp**: {timestamp}
**Agent**: mission-control

**Rationale**:
{rationale}

**Supporting Artifacts**:
{artifact_links}

---
"""

    try:
        # Check if this is a new session file
        is_new_file = not session_file.exists()

        # If new file, add header
        if is_new_file:
            header = f"""# Session Summary - {today}

**Project**: PlasmaDX-Clean Multi-Agent RAG System
**Date**: {today}
**Orchestrator**: mission-control

---

"""
            with open(session_file, "w", encoding="utf-8") as f:
                f.write(header)

        # Append decision to session file
        with open(session_file, "a", encoding="utf-8") as f:
            f.write(decision_entry)

        return {
            "content": [
                {
                    "type": "text",
                    "text": f"""✅ Decision recorded successfully

**File**: `{session_file.relative_to(project_root)}`
**Timestamp**: {timestamp}
**Status**: {"New session file created" if is_new_file else "Appended to existing session"}

**Decision**: {decision}

**Rationale**: {rationale[:200]}{'...' if len(rationale) > 200 else ''}

**Artifacts**: {len(artifacts)} file(s) linked

The decision has been written to the session log and is now part of the persistent record.
""",
                }
            ]
        }

    except Exception as e:
        # Error handling - return error status
        return {
            "content": [
                {
                    "type": "text",
                    "text": f"""❌ Failed to record decision

**Error**: {str(e)}
**File**: {session_file}
**Timestamp**: {timestamp}

Please check file permissions and directory structure.
""",
                }
            ]
        }
