"""
Artifact read tools for agents.

Phase 2B-8: Agents receive compact artifact REFERENCES in prompts (path + 1-line
summary). When they need full details, they call read_artifact() to fetch the
JSON content. This replaces inline data dumps and cuts handoff context by ~75%.

Usage in agent:
    result = await read_artifact(path="/path/to/quality_iter1.json")
    # Returns JSON string of artifact contents (capped at 4K chars)

Two-layer pattern:
    _read_artifact_impl()  — plain function for internal use
    read_artifact          — @function_tool wrapper for agents
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

from agents import function_tool

# Artifact directory root
ARTIFACTS_DIR = Path(__file__).parent.parent / "sessions" / "artifacts"

# Maximum characters returned to agents (prevents context bloat)
MAX_ARTIFACT_CHARS = 4000


def _read_artifact_impl(path: str) -> str:
    """
    Read an artifact JSON file and return its contents.

    Args:
        path: Absolute or relative path to the artifact JSON file.
              Relative paths are resolved from the artifacts directory.

    Returns:
        JSON string with artifact contents, capped at MAX_ARTIFACT_CHARS.
    """
    artifact_path = Path(path)

    # Resolve relative paths from artifacts dir
    if not artifact_path.is_absolute():
        artifact_path = ARTIFACTS_DIR / path

    if not artifact_path.exists():
        return json.dumps({
            "error": f"Artifact not found: {path}",
            "available_hint": "Use list_session_artifacts to see available files.",
        })

    if not artifact_path.suffix == ".json":
        return json.dumps({
            "error": f"Not a JSON artifact: {path}",
        })

    try:
        content = artifact_path.read_text()
        if len(content) > MAX_ARTIFACT_CHARS:
            # Truncate and note
            truncated = content[:MAX_ARTIFACT_CHARS]
            return json.dumps({
                "truncated": True,
                "total_chars": len(content),
                "returned_chars": MAX_ARTIFACT_CHARS,
                "content": json.loads(truncated + "..."),
            }, indent=2, default=str)
        return content
    except json.JSONDecodeError:
        # Return raw text if not valid JSON
        content = artifact_path.read_text()[:MAX_ARTIFACT_CHARS]
        return json.dumps({
            "raw_content": content,
            "note": "File is not valid JSON",
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


def _list_session_artifacts_impl(session_id: str) -> str:
    """
    List available artifact files for a session.

    Args:
        session_id: The session ID to list artifacts for.

    Returns:
        JSON with list of artifact filenames and their sizes.
    """
    session_dir = ARTIFACTS_DIR / session_id
    if not session_dir.exists():
        return json.dumps({
            "session_id": session_id,
            "error": f"No artifacts directory for session: {session_id}",
            "artifact_count": 0,
            "artifacts": [],
        })

    artifacts: List[dict] = []
    for f in sorted(session_dir.glob("*.json")):
        artifacts.append({
            "filename": f.name,
            "path": str(f),
            "size_bytes": f.stat().st_size,
        })

    return json.dumps({
        "session_id": session_id,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }, indent=2)


# =========================================================================
# FUNCTION TOOL WRAPPERS (exposed to agents)
# =========================================================================

@function_tool
async def read_artifact(path: str) -> str:
    """
    Read a pipeline artifact file (JSON).

    Call this when you need full details from an artifact referenced in your
    prompt. The prompt gives you a 1-line summary; this tool gives you the
    full content (capped at 4K chars).

    Args:
        path: Path to the artifact file (from your prompt context).

    Returns:
        JSON string with the full artifact contents.
    """
    return _read_artifact_impl(path)


@function_tool
async def list_session_artifacts(session_id: str) -> str:
    """
    List all available artifact files for a session.

    Use this to discover what artifacts are available before reading them.

    Args:
        session_id: The session ID.

    Returns:
        JSON with artifact filenames, paths, and sizes.
    """
    return _list_session_artifacts_impl(session_id)
