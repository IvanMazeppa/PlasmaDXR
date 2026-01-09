"""
Session persistence for the Blender VFX Orchestrator.

Provides save/load functionality for orchestration state, enabling:
- Resumption after context limits
- Recovery from crashes
- Multi-session continuity
- Progress tracking across runs

State is persisted as JSON files in a configurable directory.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from models.shared_context import (
    AssetRequest,
    SessionState,
    SessionStatus,
    SharedContext,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

DEFAULT_STATE_DIR = Path("build/orchestrator_state")
STATE_FILE_PREFIX = "session_"
STATE_FILE_SUFFIX = ".json"


# =============================================================================
# SESSION PERSISTENCE
# =============================================================================

class SessionPersistence:
    """
    Manages persistence of orchestration sessions.

    Sessions are saved as JSON files with automatic timestamping.
    Supports listing, loading, and cleanup of old sessions.
    """

    def __init__(self, state_dir: Optional[Path] = None):
        """
        Initialize session persistence.

        Args:
            state_dir: Directory for state files (default: build/orchestrator_state)
        """
        self.state_dir = state_dir or DEFAULT_STATE_DIR
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def _get_session_path(self, session_id: str) -> Path:
        """Get the file path for a session."""
        safe_id = session_id.replace("/", "_").replace("\\", "_")
        return self.state_dir / f"{STATE_FILE_PREFIX}{safe_id}{STATE_FILE_SUFFIX}"

    def save_session(self, session: SessionState) -> str:
        """
        Save a session state to disk.

        Args:
            session: SessionState to persist

        Returns:
            Path to saved file

        Example:
            persistence = SessionPersistence()
            path = persistence.save_session(context.session)
            # Returns: "build/orchestrator_state/session_explosion_v1_20250107.json"
        """
        session.update_timestamp()

        path = self._get_session_path(session.session_id)

        with open(path, "w") as f:
            json.dump(session.model_dump(), f, indent=2, default=str)

        return str(path)

    def save_context(self, context: SharedContext) -> str:
        """
        Save a full shared context to disk.

        Extracts and saves the session state from the context.

        Args:
            context: SharedContext to persist

        Returns:
            Path to saved file
        """
        return self.save_session(context.session)

    def load_session(self, session_id: str) -> Optional[SessionState]:
        """
        Load a session state from disk.

        Args:
            session_id: Session ID to load

        Returns:
            SessionState if found, None otherwise

        Example:
            session = persistence.load_session("explosion_v1_20250107")
            if session:
                context = SharedContext(session=session)
        """
        path = self._get_session_path(session_id)

        if not path.exists():
            return None

        try:
            with open(path, "r") as f:
                data = json.load(f)
            return SessionState.model_validate(data)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error loading session {session_id}: {e}")
            return None

    def load_context(self, session_id: str) -> Optional[SharedContext]:
        """
        Load a session and wrap it in a SharedContext.

        Args:
            session_id: Session ID to load

        Returns:
            SharedContext if session found, None otherwise
        """
        session = self.load_session(session_id)
        if session is None:
            return None
        return SharedContext(session=session)

    def list_sessions(
        self,
        status_filter: Optional[SessionStatus] = None,
        limit: int = 50
    ) -> List[dict]:
        """
        List available sessions with metadata.

        Args:
            status_filter: Optional filter by status
            limit: Maximum number of sessions to return

        Returns:
            List of session summaries with metadata

        Example:
            sessions = persistence.list_sessions(status_filter=SessionStatus.IN_PROGRESS)
            # Returns: [{"session_id": "...", "status": "in_progress", ...}, ...]
        """
        sessions = []

        for path in sorted(self.state_dir.glob(f"{STATE_FILE_PREFIX}*{STATE_FILE_SUFFIX}"),
                          key=lambda p: p.stat().st_mtime,
                          reverse=True):
            try:
                with open(path, "r") as f:
                    data = json.load(f)

                # Apply status filter
                if status_filter and data.get("status") != status_filter.value:
                    continue

                # Extract summary
                sessions.append({
                    "session_id": data.get("session_id", "unknown"),
                    "status": data.get("status", "unknown"),
                    "asset_name": data.get("request", {}).get("asset_name", "unknown"),
                    "effect_type": data.get("request", {}).get("effect_type", "unknown"),
                    "current_iteration": data.get("current_iteration", 0),
                    "best_score": data.get("best_score", 0.0),
                    "created_at": data.get("created_at", ""),
                    "updated_at": data.get("updated_at", ""),
                    "file_path": str(path),
                })

                if len(sessions) >= limit:
                    break

            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error reading session file {path}: {e}")
                continue

        return sessions

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session file.

        Args:
            session_id: Session ID to delete

        Returns:
            True if deleted, False if not found
        """
        path = self._get_session_path(session_id)

        if path.exists():
            path.unlink()
            return True
        return False

    def cleanup_old_sessions(
        self,
        max_age_days: int = 30,
        keep_completed: bool = True
    ) -> int:
        """
        Remove old session files.

        Args:
            max_age_days: Delete sessions older than this
            keep_completed: If True, don't delete completed sessions

        Returns:
            Number of sessions deleted
        """
        deleted = 0
        cutoff = datetime.now().timestamp() - (max_age_days * 24 * 60 * 60)

        for path in self.state_dir.glob(f"{STATE_FILE_PREFIX}*{STATE_FILE_SUFFIX}"):
            try:
                # Check age
                if path.stat().st_mtime > cutoff:
                    continue

                # Check if completed (and should be kept)
                if keep_completed:
                    with open(path, "r") as f:
                        data = json.load(f)
                    if data.get("status") == SessionStatus.PASSED.value:
                        continue

                path.unlink()
                deleted += 1

            except (json.JSONDecodeError, OSError) as e:
                print(f"Error processing {path}: {e}")
                continue

        return deleted

    def session_exists(self, session_id: str) -> bool:
        """Check if a session file exists."""
        return self._get_session_path(session_id).exists()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def generate_session_id(asset_name: str) -> str:
    """
    Generate a unique session ID from asset name and timestamp.

    Args:
        asset_name: Base name for the session

    Returns:
        Unique session ID (e.g., "explosion_v1_20250107_143022")
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = asset_name.replace(" ", "_").replace("-", "_").lower()
    return f"{safe_name}_{timestamp}"


def create_session_from_request(
    request: AssetRequest,
    session_id: Optional[str] = None
) -> SharedContext:
    """
    Create a new SharedContext from an asset request.

    Args:
        request: Asset generation request
        session_id: Optional custom session ID (auto-generated if not provided)

    Returns:
        Initialized SharedContext ready for iteration

    Example:
        request = AssetRequest(
            asset_name="supernova_burst",
            description="expanding stellar explosion",
            effect_type=EffectType.SUPERNOVA
        )
        context = create_session_from_request(request)
    """
    if session_id is None:
        session_id = generate_session_id(request.asset_name)

    return SharedContext.from_request(request, session_id)


def resume_or_create_session(
    request: AssetRequest,
    session_id: Optional[str] = None,
    persistence: Optional[SessionPersistence] = None
) -> tuple[SharedContext, bool]:
    """
    Resume an existing session or create a new one.

    Checks if a session exists and is resumable (in_progress or paused).
    If so, loads it. Otherwise creates a new session.

    Args:
        request: Asset generation request
        session_id: Optional session ID to try to resume
        persistence: Optional persistence instance (created if not provided)

    Returns:
        Tuple of (SharedContext, is_resumed)

    Example:
        context, resumed = resume_or_create_session(
            request,
            session_id="explosion_v1_20250107_143022"
        )
        if resumed:
            print(f"Resumed at iteration {context.session.current_iteration}")
    """
    if persistence is None:
        persistence = SessionPersistence()

    # Try to resume if session_id provided
    if session_id and persistence.session_exists(session_id):
        existing = persistence.load_session(session_id)
        if existing and existing.status in [
            SessionStatus.IN_PROGRESS,
            SessionStatus.PAUSED
        ]:
            # Update status to in_progress if paused
            existing.status = SessionStatus.IN_PROGRESS
            existing.update_timestamp()
            return SharedContext(session=existing), True

    # Create new session
    return create_session_from_request(request, session_id), False


# =============================================================================
# SINGLETON INSTANCE
# =============================================================================

_default_persistence: Optional[SessionPersistence] = None


def get_persistence() -> SessionPersistence:
    """Get the default persistence instance."""
    global _default_persistence
    if _default_persistence is None:
        _default_persistence = SessionPersistence()
    return _default_persistence
