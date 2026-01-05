"""
Session Manager for Cross-Session Learning.

Uses persistent storage to accumulate knowledge across agent runs.
Supports both JSON file and SQLite backends for flexibility.

Key capabilities:
1. Persist conversation history across agent runs
2. Accumulate learned parameter patterns
3. Remember successful fixes for similar issues
4. Provide context from previous sessions
5. Avoid repeating failed experiments
"""

from __future__ import annotations

import json
import os
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class SessionContext:
    """Persistent context that survives across sessions."""
    session_id: str
    created_at: str
    last_active: str
    effect_type: str = ""
    successful_fixes: List[Dict[str, Any]] = field(default_factory=list)
    failed_experiments: List[Dict[str, Any]] = field(default_factory=list)
    parameter_history: Dict[str, List[float]] = field(default_factory=dict)
    quality_trajectory: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionContext":
        """Create from dictionary."""
        return cls(**data)


class SessionBackend(ABC):
    """Abstract base class for session storage backends."""

    @abstractmethod
    def save_session(self, session: SessionContext) -> None:
        """Save a session to storage."""
        pass

    @abstractmethod
    def load_session(self, session_id: str) -> Optional[SessionContext]:
        """Load a session from storage."""
        pass

    @abstractmethod
    def list_sessions(self) -> List[str]:
        """List all available session IDs."""
        pass

    @abstractmethod
    def delete_session(self, session_id: str) -> bool:
        """Delete a session from storage."""
        pass


class JSONBackend(SessionBackend):
    """JSON file-based session storage backend."""

    def __init__(self, sessions_dir: Path):
        """Initialize with sessions directory."""
        self.sessions_dir = sessions_dir
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def save_session(self, session: SessionContext) -> None:
        """Save session to JSON file."""
        session_path = self.sessions_dir / f"{session.session_id}.json"
        with open(session_path, 'w') as f:
            json.dump(session.to_dict(), f, indent=2)

    def load_session(self, session_id: str) -> Optional[SessionContext]:
        """Load session from JSON file."""
        session_path = self.sessions_dir / f"{session_id}.json"
        if not session_path.exists():
            return None
        try:
            with open(session_path, 'r') as f:
                data = json.load(f)
            return SessionContext.from_dict(data)
        except (json.JSONDecodeError, KeyError):
            return None

    def list_sessions(self) -> List[str]:
        """List all session IDs."""
        return [
            p.stem for p in self.sessions_dir.glob("*.json")
        ]

    def delete_session(self, session_id: str) -> bool:
        """Delete a session file."""
        session_path = self.sessions_dir / f"{session_id}.json"
        if session_path.exists():
            session_path.unlink()
            return True
        return False


class SQLiteBackend(SessionBackend):
    """SQLite-based session storage backend."""

    def __init__(self, db_path: Path):
        """Initialize SQLite backend."""
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    last_active TEXT NOT NULL,
                    effect_type TEXT DEFAULT '',
                    data TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS successful_fixes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    issue TEXT NOT NULL,
                    modifications TEXT NOT NULL,
                    score_improvement REAL,
                    doc_sources TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS failed_experiments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    issue TEXT NOT NULL,
                    modifications TEXT NOT NULL,
                    reason TEXT,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_fixes_session
                ON successful_fixes(session_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_failures_session
                ON failed_experiments(session_id)
            """)
            conn.commit()

    def save_session(self, session: SessionContext) -> None:
        """Save session to SQLite."""
        with sqlite3.connect(self.db_path) as conn:
            # Save main session data
            data = json.dumps({
                "parameter_history": session.parameter_history,
                "quality_trajectory": session.quality_trajectory
            })
            conn.execute("""
                INSERT OR REPLACE INTO sessions
                (session_id, created_at, last_active, effect_type, data)
                VALUES (?, ?, ?, ?, ?)
            """, (
                session.session_id,
                session.created_at,
                session.last_active,
                session.effect_type,
                data
            ))

            # Save successful fixes
            conn.execute(
                "DELETE FROM successful_fixes WHERE session_id = ?",
                (session.session_id,)
            )
            for fix in session.successful_fixes:
                conn.execute("""
                    INSERT INTO successful_fixes
                    (session_id, timestamp, issue, modifications, score_improvement, doc_sources)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    session.session_id,
                    fix.get("timestamp", ""),
                    fix.get("issue", ""),
                    json.dumps(fix.get("modifications", {})),
                    fix.get("score_improvement", 0),
                    json.dumps(fix.get("doc_sources", []))
                ))

            # Save failed experiments
            conn.execute(
                "DELETE FROM failed_experiments WHERE session_id = ?",
                (session.session_id,)
            )
            for fail in session.failed_experiments:
                conn.execute("""
                    INSERT INTO failed_experiments
                    (session_id, timestamp, issue, modifications, reason)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    session.session_id,
                    fail.get("timestamp", ""),
                    fail.get("issue", ""),
                    json.dumps(fail.get("modifications", {})),
                    fail.get("reason", "")
                ))

            conn.commit()

    def load_session(self, session_id: str) -> Optional[SessionContext]:
        """Load session from SQLite."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Load main session
            cursor = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,)
            )
            row = cursor.fetchone()
            if not row:
                return None

            data = json.loads(row["data"])

            # Load successful fixes
            cursor = conn.execute(
                "SELECT * FROM successful_fixes WHERE session_id = ? ORDER BY timestamp",
                (session_id,)
            )
            fixes = []
            for fix_row in cursor:
                fixes.append({
                    "timestamp": fix_row["timestamp"],
                    "issue": fix_row["issue"],
                    "modifications": json.loads(fix_row["modifications"]),
                    "score_improvement": fix_row["score_improvement"],
                    "doc_sources": json.loads(fix_row["doc_sources"])
                })

            # Load failed experiments
            cursor = conn.execute(
                "SELECT * FROM failed_experiments WHERE session_id = ? ORDER BY timestamp",
                (session_id,)
            )
            failures = []
            for fail_row in cursor:
                failures.append({
                    "timestamp": fail_row["timestamp"],
                    "issue": fail_row["issue"],
                    "modifications": json.loads(fail_row["modifications"]),
                    "reason": fail_row["reason"]
                })

            return SessionContext(
                session_id=row["session_id"],
                created_at=row["created_at"],
                last_active=row["last_active"],
                effect_type=row["effect_type"],
                successful_fixes=fixes,
                failed_experiments=failures,
                parameter_history=data.get("parameter_history", {}),
                quality_trajectory=data.get("quality_trajectory", [])
            )

    def list_sessions(self) -> List[str]:
        """List all session IDs."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT session_id FROM sessions ORDER BY last_active DESC"
            )
            return [row[0] for row in cursor]

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and its related data."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "DELETE FROM successful_fixes WHERE session_id = ?",
                (session_id,)
            )
            conn.execute(
                "DELETE FROM failed_experiments WHERE session_id = ?",
                (session_id,)
            )
            cursor = conn.execute(
                "DELETE FROM sessions WHERE session_id = ?",
                (session_id,)
            )
            conn.commit()
            return cursor.rowcount > 0

    def query_similar_issues(
        self,
        issue_keywords: List[str],
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Query for similar issues across all sessions.

        This is an SQLite-specific feature for cross-session learning.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            # Build LIKE conditions for keywords
            conditions = " OR ".join(
                ["issue LIKE ?"] * len(issue_keywords)
            )
            params = [f"%{kw}%" for kw in issue_keywords]

            # Query successful fixes
            cursor = conn.execute(f"""
                SELECT session_id, timestamp, issue, modifications,
                       score_improvement, doc_sources
                FROM successful_fixes
                WHERE {conditions}
                ORDER BY score_improvement DESC
                LIMIT ?
            """, params + [limit])

            results = []
            for row in cursor:
                results.append({
                    "session_id": row["session_id"],
                    "timestamp": row["timestamp"],
                    "issue": row["issue"],
                    "modifications": json.loads(row["modifications"]),
                    "score_improvement": row["score_improvement"],
                    "doc_sources": json.loads(row["doc_sources"]),
                    "type": "successful_fix"
                })

            return results


class SessionManager:
    """
    Manages persistent sessions for cross-session learning.

    Key capabilities:
    - Automatic session restoration on agent start
    - Persistent storage of successful parameter patterns
    - Cross-session knowledge accumulation
    - Session context injection into agent prompts
    """

    DEFAULT_DIR = Path("agents/blender-librarian/sessions")

    def __init__(
        self,
        effect_type: str = "",
        backend: str = "sqlite",
        storage_path: Optional[Path] = None
    ):
        """
        Initialize session manager.

        Args:
            effect_type: Type of effect for this session
            backend: Storage backend ("json" or "sqlite")
            storage_path: Custom path for storage (default: DEFAULT_DIR)
        """
        self.effect_type = effect_type
        self.storage_path = storage_path or self.DEFAULT_DIR

        # Initialize appropriate backend
        if backend == "sqlite":
            db_path = self.storage_path / "sessions.db"
            self._backend = SQLiteBackend(db_path)
        else:
            self._backend = JSONBackend(self.storage_path)

        self.current_session: Optional[SessionContext] = None

    def get_or_create_session(
        self,
        session_id: Optional[str] = None
    ) -> SessionContext:
        """
        Get existing session or create new one.

        Args:
            session_id: Optional session ID to restore. If None, creates new session.

        Returns:
            SessionContext with accumulated learning
        """
        if session_id:
            session = self._backend.load_session(session_id)
            if session:
                session.last_active = datetime.now().isoformat()
                self.current_session = session
                self._save_session()
                return session

        # Create new session
        new_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_session = SessionContext(
            session_id=new_id,
            created_at=datetime.now().isoformat(),
            last_active=datetime.now().isoformat(),
            effect_type=self.effect_type
        )
        self._save_session()
        return self.current_session

    def _save_session(self) -> None:
        """Save current session to storage."""
        if self.current_session:
            self._backend.save_session(self.current_session)

    def record_successful_fix(
        self,
        issue: str,
        modifications: Dict[str, Any],
        score_improvement: float,
        doc_sources: List[str]
    ) -> None:
        """
        Record a successful fix for cross-session learning.

        Args:
            issue: The issue that was fixed
            modifications: Parameter changes that worked
            score_improvement: How much the quality score improved
            doc_sources: Documentation sources used
        """
        if not self.current_session:
            return

        self.current_session.successful_fixes.append({
            "timestamp": datetime.now().isoformat(),
            "issue": issue,
            "modifications": modifications,
            "score_improvement": score_improvement,
            "doc_sources": doc_sources
        })
        self._save_session()

    def record_failed_experiment(
        self,
        issue: str,
        modifications: Dict[str, Any],
        reason: str
    ) -> None:
        """
        Record a failed experiment to avoid repeating.

        Args:
            issue: The issue we tried to fix
            modifications: Parameter changes that didn't work
            reason: Why it failed
        """
        if not self.current_session:
            return

        self.current_session.failed_experiments.append({
            "timestamp": datetime.now().isoformat(),
            "issue": issue,
            "modifications": modifications,
            "reason": reason
        })
        self._save_session()

    def record_parameter_change(
        self,
        parameter: str,
        value: float
    ) -> None:
        """
        Record a parameter change for history tracking.

        Args:
            parameter: Parameter name
            value: New value
        """
        if not self.current_session:
            return

        if parameter not in self.current_session.parameter_history:
            self.current_session.parameter_history[parameter] = []
        self.current_session.parameter_history[parameter].append(value)
        self._save_session()

    def record_quality_score(self, score: float) -> None:
        """
        Record a quality score for trajectory tracking.

        Args:
            score: Quality score (0-100)
        """
        if not self.current_session:
            return

        self.current_session.quality_trajectory.append(score)
        self._save_session()

    def get_relevant_history(
        self,
        current_issue: str,
        max_fixes: int = 5,
        max_failures: int = 3
    ) -> Dict[str, Any]:
        """
        Get relevant history for the current issue.

        Args:
            current_issue: Description of current issue
            max_fixes: Maximum successful fixes to return
            max_failures: Maximum failed experiments to return

        Returns:
            Dict with relevant past fixes and failures
        """
        if not self.current_session:
            return {"fixes": [], "failures": []}

        # Simple keyword matching
        keywords = current_issue.lower().split()

        relevant_fixes = []
        for fix in self.current_session.successful_fixes:
            if any(kw in fix["issue"].lower() for kw in keywords):
                relevant_fixes.append(fix)

        relevant_failures = []
        for fail in self.current_session.failed_experiments:
            if any(kw in fail["issue"].lower() for kw in keywords):
                relevant_failures.append(fail)

        # For SQLite backend, also search across sessions
        if isinstance(self._backend, SQLiteBackend):
            cross_session = self._backend.query_similar_issues(keywords[:5])
            for item in cross_session:
                if item not in relevant_fixes:
                    relevant_fixes.append(item)

        return {
            "fixes": relevant_fixes[-max_fixes:],
            "failures": relevant_failures[-max_failures:]
        }

    def inject_session_context(self, base_instructions: str) -> str:
        """
        Inject session context into agent instructions.

        Args:
            base_instructions: Base agent instructions

        Returns:
            Enhanced instructions with session context
        """
        if not self.current_session:
            return base_instructions

        context_parts = [base_instructions]

        # Add session info
        context_parts.append(f"\n\nSESSION CONTEXT:")
        context_parts.append(f"- Session ID: {self.current_session.session_id}")
        context_parts.append(f"- Effect Type: {self.current_session.effect_type or 'general'}")

        # Add successful fixes summary
        if self.current_session.successful_fixes:
            context_parts.append("\nPREVIOUS SUCCESSFUL FIXES:")
            for fix in self.current_session.successful_fixes[-3:]:
                context_parts.append(
                    f"- {fix['issue']}: {json.dumps(fix['modifications'])} "
                    f"(+{fix['score_improvement']:.1f} improvement)"
                )

        # Add failed experiments warning
        if self.current_session.failed_experiments:
            context_parts.append("\nAVOID (previously failed):")
            for fail in self.current_session.failed_experiments[-3:]:
                context_parts.append(
                    f"- {fail['issue']}: {json.dumps(fail['modifications'])} "
                    f"(Reason: {fail['reason']})"
                )

        # Add quality trajectory
        if self.current_session.quality_trajectory:
            recent = self.current_session.quality_trajectory[-5:]
            context_parts.append(f"\nQUALITY TRAJECTORY (recent): {recent}")

        return "\n".join(context_parts)

    def list_sessions(self) -> List[str]:
        """List all available session IDs."""
        return self._backend.list_sessions()

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session."""
        if not self.current_session:
            return {"error": "No active session"}

        return {
            "session_id": self.current_session.session_id,
            "effect_type": self.current_session.effect_type,
            "created_at": self.current_session.created_at,
            "last_active": self.current_session.last_active,
            "successful_fixes_count": len(self.current_session.successful_fixes),
            "failed_experiments_count": len(self.current_session.failed_experiments),
            "parameters_tracked": list(self.current_session.parameter_history.keys()),
            "quality_scores_recorded": len(self.current_session.quality_trajectory)
        }


# For testing
if __name__ == "__main__":
    # Test JSON backend
    print("Testing JSON backend...")
    json_manager = SessionManager(effect_type="sun", backend="json")
    session = json_manager.get_or_create_session()
    print(f"Created session: {session.session_id}")

    json_manager.record_successful_fix(
        issue="color too cool",
        modifications={"flame_max_temp": 5778},
        score_improvement=15.0,
        doc_sources=["physics/fluid/settings.html"]
    )
    print(f"Summary: {json_manager.get_session_summary()}")

    # Test SQLite backend
    print("\nTesting SQLite backend...")
    sqlite_manager = SessionManager(effect_type="explosion", backend="sqlite")
    session = sqlite_manager.get_or_create_session()
    print(f"Created session: {session.session_id}")

    sqlite_manager.record_successful_fix(
        issue="brightness too low",
        modifications={"emission_intensity": 5.0},
        score_improvement=20.0,
        doc_sources=["render/materials/emission.html"]
    )
    print(f"Summary: {sqlite_manager.get_session_summary()}")

    # Test cross-session query
    history = sqlite_manager.get_relevant_history("brightness issue")
    print(f"Relevant history: {history}")
