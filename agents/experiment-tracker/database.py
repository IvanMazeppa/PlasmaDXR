#!/usr/bin/env python3
"""
Experiment Tracking Database

SQLite database for recording experiments, parameter changes, observed effects,
and accumulated knowledge about what works and what doesn't.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict
from contextlib import contextmanager

# Database location
DB_PATH = Path(__file__).parent / "experiments.db"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ParameterChange:
    """Records a single parameter change in an experiment."""
    parameter: str          # e.g., "domain_scale"
    old_value: Any          # e.g., 6.0
    new_value: Any          # e.g., 10.0
    change_type: str        # "increase", "decrease", "modify"
    magnitude: float        # e.g., 1.67 (ratio) or absolute change
    rationale: str = ""     # Why this change was made


@dataclass
class ObservedEffect:
    """Records an observed effect from an experiment."""
    metric: str             # e.g., "framing_position", "clip_score"
    direction: str          # e.g., "shifted_up", "increased", "decreased"
    magnitude: float        # How much change
    expected: bool          # Was this the intended effect?
    side_effect: bool       # Unintended consequence?
    description: str = ""   # Human-readable description


@dataclass
class Experiment:
    """Complete record of a single experiment."""
    id: str                                     # Unique experiment ID
    session_id: str                             # Parent asset session
    timestamp: str                              # ISO format datetime

    # Hypothesis
    hypothesis: str                             # What we're testing
    issue_addressed: str                        # The problem we're trying to fix

    # Parameters
    baseline_params: Dict[str, Any]             # Before parameters
    result_params: Dict[str, Any]               # After parameters
    parameter_changes: List[ParameterChange]    # What changed

    # Scores
    baseline_scores: Dict[str, float]           # Before evaluation scores
    result_scores: Dict[str, float]             # After evaluation scores

    # Artifacts
    baseline_render: str                        # Path to before image
    result_render: str                          # Path to after image
    baseline_script: str                        # Path to before script
    result_script: str                          # Path to after script

    # Analysis
    observed_effects: List[ObservedEffect]      # What actually changed
    success: bool                               # Did it achieve the goal?
    partial_success: bool                       # Partially worked?

    # Learnings
    learnings: List[str]                        # What we learned
    warnings: List[str]                         # Gotchas discovered

    # Human feedback
    human_rating: Optional[int] = None          # 1-5 rating from user
    human_notes: str = ""                       # User observations


@dataclass
class ParameterKnowledge:
    """Accumulated knowledge about a parameter."""
    parameter: str                              # e.g., "domain_scale"

    # Statistics
    experiments_count: int                      # How many experiments involved this
    success_rate: float                         # % of successful experiments

    # Learned relationships
    effects: List[Dict[str, Any]]               # Cause-effect relationships
    rules: List[str]                            # Learned rules
    warnings: List[str]                         # Known gotchas

    # Recommended values
    safe_range: tuple                           # (min, max) safe values
    optimal_value: Optional[float]              # Best known value
    default_value: Any                          # Starting point

    # Confidence
    confidence: float                           # 0-1, how sure are we
    last_updated: str                           # ISO datetime


@dataclass
class CausalRelationship:
    """A learned cause-effect relationship."""
    cause: str                                  # e.g., "increase domain_scale"
    effect: str                                 # e.g., "content shifts upward"
    parameter: str                              # Which parameter
    confidence: float                           # 0-1
    evidence_count: int                         # Supporting experiments
    counterexamples: int                        # Contradicting experiments
    context: str = ""                           # When this applies


# =============================================================================
# Database Manager
# =============================================================================

class ExperimentDatabase:
    """SQLite database manager for experiment tracking."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DB_PATH
        self._ensure_tables()

    @contextmanager
    def _connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _ensure_tables(self):
        """Create tables if they don't exist."""
        with self._connection() as conn:
            conn.executescript("""
                -- Experiments table
                CREATE TABLE IF NOT EXISTS experiments (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    hypothesis TEXT,
                    issue_addressed TEXT,
                    baseline_params TEXT,  -- JSON
                    result_params TEXT,    -- JSON
                    baseline_scores TEXT,  -- JSON
                    result_scores TEXT,    -- JSON
                    baseline_render TEXT,
                    result_render TEXT,
                    baseline_script TEXT,
                    result_script TEXT,
                    success INTEGER,
                    partial_success INTEGER,
                    learnings TEXT,        -- JSON array
                    warnings TEXT,         -- JSON array
                    human_rating INTEGER,
                    human_notes TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                -- Parameter changes within experiments
                CREATE TABLE IF NOT EXISTS parameter_changes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    parameter TEXT NOT NULL,
                    old_value TEXT,
                    new_value TEXT,
                    change_type TEXT,
                    magnitude REAL,
                    rationale TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
                );

                -- Observed effects from experiments
                CREATE TABLE IF NOT EXISTS observed_effects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    metric TEXT NOT NULL,
                    direction TEXT,
                    magnitude REAL,
                    expected INTEGER,
                    side_effect INTEGER,
                    description TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
                );

                -- Accumulated parameter knowledge
                CREATE TABLE IF NOT EXISTS parameter_knowledge (
                    parameter TEXT PRIMARY KEY,
                    experiments_count INTEGER DEFAULT 0,
                    success_rate REAL DEFAULT 0.0,
                    effects TEXT,          -- JSON array
                    rules TEXT,            -- JSON array
                    warnings TEXT,         -- JSON array
                    safe_range TEXT,       -- JSON [min, max]
                    optimal_value REAL,
                    default_value TEXT,
                    confidence REAL DEFAULT 0.0,
                    last_updated TEXT
                );

                -- Causal relationships
                CREATE TABLE IF NOT EXISTS causal_relationships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cause TEXT NOT NULL,
                    effect TEXT NOT NULL,
                    parameter TEXT NOT NULL,
                    confidence REAL DEFAULT 0.0,
                    evidence_count INTEGER DEFAULT 0,
                    counterexamples INTEGER DEFAULT 0,
                    context TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(cause, effect, parameter)
                );

                -- Session summaries
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    asset_name TEXT NOT NULL,
                    effect_type TEXT,
                    description TEXT,
                    started_at TEXT,
                    completed_at TEXT,
                    final_status TEXT,
                    best_score REAL,
                    experiments_count INTEGER DEFAULT 0,
                    reference_path TEXT,
                    semantic_query TEXT
                );

                -- Indexes for common queries
                CREATE INDEX IF NOT EXISTS idx_experiments_session
                    ON experiments(session_id);
                CREATE INDEX IF NOT EXISTS idx_experiments_timestamp
                    ON experiments(timestamp);
                CREATE INDEX IF NOT EXISTS idx_param_changes_experiment
                    ON parameter_changes(experiment_id);
                CREATE INDEX IF NOT EXISTS idx_param_changes_parameter
                    ON parameter_changes(parameter);
                CREATE INDEX IF NOT EXISTS idx_effects_experiment
                    ON observed_effects(experiment_id);
                CREATE INDEX IF NOT EXISTS idx_causal_parameter
                    ON causal_relationships(parameter);
            """)

            # Phase 2B-6: Migration — add Ebbinghaus decay columns if missing
            cols = [
                r[1] for r in conn.execute(
                    "PRAGMA table_info(parameter_knowledge)"
                ).fetchall()
            ]
            if "last_reinforced" not in cols:
                conn.execute(
                    "ALTER TABLE parameter_knowledge "
                    "ADD COLUMN last_reinforced TEXT DEFAULT ''"
                )
            if "reinforcement_count" not in cols:
                conn.execute(
                    "ALTER TABLE parameter_knowledge "
                    "ADD COLUMN reinforcement_count INTEGER DEFAULT 0"
                )

            # Phase 2B-7: Migration — add effect_type column if missing
            if "effect_type" not in cols:
                conn.execute(
                    "ALTER TABLE parameter_knowledge "
                    "ADD COLUMN effect_type TEXT DEFAULT ''"
                )

    # =========================================================================
    # Experiment CRUD
    # =========================================================================

    def save_experiment(self, experiment: Experiment) -> str:
        """Save a complete experiment record."""
        with self._connection() as conn:
            # Save main experiment
            conn.execute("""
                INSERT OR REPLACE INTO experiments (
                    id, session_id, timestamp, hypothesis, issue_addressed,
                    baseline_params, result_params, baseline_scores, result_scores,
                    baseline_render, result_render, baseline_script, result_script,
                    success, partial_success, learnings, warnings,
                    human_rating, human_notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                experiment.id,
                experiment.session_id,
                experiment.timestamp,
                experiment.hypothesis,
                experiment.issue_addressed,
                json.dumps(experiment.baseline_params),
                json.dumps(experiment.result_params),
                json.dumps(experiment.baseline_scores),
                json.dumps(experiment.result_scores),
                experiment.baseline_render,
                experiment.result_render,
                experiment.baseline_script,
                experiment.result_script,
                int(experiment.success),
                int(experiment.partial_success),
                json.dumps(experiment.learnings),
                json.dumps(experiment.warnings),
                experiment.human_rating,
                experiment.human_notes
            ))

            # Save parameter changes
            for change in experiment.parameter_changes:
                conn.execute("""
                    INSERT INTO parameter_changes (
                        experiment_id, parameter, old_value, new_value,
                        change_type, magnitude, rationale
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    experiment.id,
                    change.parameter,
                    json.dumps(change.old_value),
                    json.dumps(change.new_value),
                    change.change_type,
                    change.magnitude,
                    change.rationale
                ))

            # Save observed effects
            for effect in experiment.observed_effects:
                conn.execute("""
                    INSERT INTO observed_effects (
                        experiment_id, metric, direction, magnitude,
                        expected, side_effect, description
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    experiment.id,
                    effect.metric,
                    effect.direction,
                    effect.magnitude,
                    int(effect.expected),
                    int(effect.side_effect),
                    effect.description
                ))

        return experiment.id

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Retrieve a single experiment by ID."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM experiments WHERE id = ?",
                (experiment_id,)
            ).fetchone()

            if not row:
                return None

            # Get parameter changes
            changes = conn.execute(
                "SELECT * FROM parameter_changes WHERE experiment_id = ?",
                (experiment_id,)
            ).fetchall()

            # Get observed effects
            effects = conn.execute(
                "SELECT * FROM observed_effects WHERE experiment_id = ?",
                (experiment_id,)
            ).fetchall()

            return Experiment(
                id=row['id'],
                session_id=row['session_id'],
                timestamp=row['timestamp'],
                hypothesis=row['hypothesis'],
                issue_addressed=row['issue_addressed'],
                baseline_params=json.loads(row['baseline_params'] or '{}'),
                result_params=json.loads(row['result_params'] or '{}'),
                baseline_scores=json.loads(row['baseline_scores'] or '{}'),
                result_scores=json.loads(row['result_scores'] or '{}'),
                baseline_render=row['baseline_render'],
                result_render=row['result_render'],
                baseline_script=row['baseline_script'],
                result_script=row['result_script'],
                success=bool(row['success']),
                partial_success=bool(row['partial_success']),
                learnings=json.loads(row['learnings'] or '[]'),
                warnings=json.loads(row['warnings'] or '[]'),
                human_rating=row['human_rating'],
                human_notes=row['human_notes'] or '',
                parameter_changes=[
                    ParameterChange(
                        parameter=c['parameter'],
                        old_value=json.loads(c['old_value']),
                        new_value=json.loads(c['new_value']),
                        change_type=c['change_type'],
                        magnitude=c['magnitude'],
                        rationale=c['rationale'] or ''
                    ) for c in changes
                ],
                observed_effects=[
                    ObservedEffect(
                        metric=e['metric'],
                        direction=e['direction'],
                        magnitude=e['magnitude'],
                        expected=bool(e['expected']),
                        side_effect=bool(e['side_effect']),
                        description=e['description'] or ''
                    ) for e in effects
                ]
            )

    def get_session_experiments(self, session_id: str) -> List[Experiment]:
        """Get all experiments for a session."""
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT id FROM experiments WHERE session_id = ? ORDER BY timestamp",
                (session_id,)
            ).fetchall()

            return [self.get_experiment(row['id']) for row in rows]

    def get_recent_experiments(self, limit: int = 20) -> List[Experiment]:
        """Get most recent experiments."""
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT id FROM experiments ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            ).fetchall()

            return [self.get_experiment(row['id']) for row in rows]

    # =========================================================================
    # Knowledge Base
    # =========================================================================

    def update_parameter_knowledge(
        self,
        parameter: str,
        experiment: Experiment,
        effects: List[CausalRelationship]
    ):
        """Update knowledge about a parameter based on an experiment."""
        with self._connection() as conn:
            # Get existing knowledge
            row = conn.execute(
                "SELECT * FROM parameter_knowledge WHERE parameter = ?",
                (parameter,)
            ).fetchone()

            if row:
                existing_rules = json.loads(row['rules'] or '[]')
                existing_warnings = json.loads(row['warnings'] or '[]')
                existing_effects = json.loads(row['effects'] or '[]')
                experiments_count = row['experiments_count'] + 1

                # Calculate new success rate
                successes = row['success_rate'] * row['experiments_count']
                if experiment.success:
                    successes += 1
                success_rate = successes / experiments_count
            else:
                existing_rules = []
                existing_warnings = []
                existing_effects = []
                experiments_count = 1
                success_rate = 1.0 if experiment.success else 0.0

            # Add new learnings
            new_rules = existing_rules + experiment.learnings
            new_warnings = existing_warnings + experiment.warnings
            new_effects = existing_effects + [asdict(e) for e in effects]

            # Remove duplicates while preserving order
            new_rules = list(dict.fromkeys(new_rules))
            new_warnings = list(dict.fromkeys(new_warnings))

            # Calculate confidence based on evidence
            confidence = min(1.0, experiments_count / 10)  # Max confidence at 10 experiments

            conn.execute("""
                INSERT OR REPLACE INTO parameter_knowledge (
                    parameter, experiments_count, success_rate,
                    effects, rules, warnings, confidence, last_updated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                parameter,
                experiments_count,
                success_rate,
                json.dumps(new_effects),
                json.dumps(new_rules),
                json.dumps(new_warnings),
                confidence,
                datetime.now().isoformat()
            ))

    def get_parameter_knowledge(self, parameter: str) -> Optional[ParameterKnowledge]:
        """Get accumulated knowledge about a parameter."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM parameter_knowledge WHERE parameter = ?",
                (parameter,)
            ).fetchone()

            if not row:
                return None

            return ParameterKnowledge(
                parameter=row['parameter'],
                experiments_count=row['experiments_count'],
                success_rate=row['success_rate'],
                effects=json.loads(row['effects'] or '[]'),
                rules=json.loads(row['rules'] or '[]'),
                warnings=json.loads(row['warnings'] or '[]'),
                safe_range=tuple(json.loads(row['safe_range'] or '[null, null]')),
                optimal_value=row['optimal_value'],
                default_value=json.loads(row['default_value'] or 'null'),
                confidence=row['confidence'],
                last_updated=row['last_updated']
            )

    def query_knowledge(
        self, query: str, effect_type: str = ""
    ) -> List[Dict[str, Any]]:
        """Search knowledge base for relevant information.

        Args:
            query: Search string (matched against parameter, rules, warnings)
            effect_type: If non-empty, only return entries whose effect_type
                         matches OR is empty (universal entries).
        """
        with self._connection() as conn:
            # Search parameter knowledge
            if effect_type:
                param_results = conn.execute("""
                    SELECT * FROM parameter_knowledge
                    WHERE (parameter LIKE ?
                           OR rules LIKE ?
                           OR warnings LIKE ?)
                      AND (effect_type = '' OR effect_type = ?)
                """, (f'%{query}%', f'%{query}%', f'%{query}%',
                      effect_type)).fetchall()
            else:
                param_results = conn.execute("""
                    SELECT * FROM parameter_knowledge
                    WHERE parameter LIKE ?
                       OR rules LIKE ?
                       OR warnings LIKE ?
                """, (f'%{query}%', f'%{query}%', f'%{query}%')).fetchall()

            # Search causal relationships
            causal_results = conn.execute("""
                SELECT * FROM causal_relationships
                WHERE cause LIKE ?
                   OR effect LIKE ?
                   OR context LIKE ?
            """, (f'%{query}%', f'%{query}%', f'%{query}%')).fetchall()

            results = []

            for row in param_results:
                results.append({
                    'type': 'parameter_knowledge',
                    'parameter': row['parameter'],
                    'rules': json.loads(row['rules'] or '[]'),
                    'warnings': json.loads(row['warnings'] or '[]'),
                    'confidence': row['confidence'],
                    # Phase 2B-6: Ebbinghaus decay fields
                    'last_reinforced': row['last_reinforced'] if 'last_reinforced' in row.keys() else '',
                    'reinforcement_count': row['reinforcement_count'] if 'reinforcement_count' in row.keys() else 0,
                    'last_updated': row['last_updated'] if 'last_updated' in row.keys() else '',
                    # Phase 2B-7: Effect type scope
                    'effect_type': row['effect_type'] if 'effect_type' in row.keys() else '',
                })

            for row in causal_results:
                results.append({
                    'type': 'causal_relationship',
                    'cause': row['cause'],
                    'effect': row['effect'],
                    'parameter': row['parameter'],
                    'confidence': row['confidence'],
                    'evidence_count': row['evidence_count']
                })

            return results

    def add_causal_relationship(self, relationship: CausalRelationship):
        """Add or update a causal relationship."""
        with self._connection() as conn:
            # Check if exists
            existing = conn.execute("""
                SELECT * FROM causal_relationships
                WHERE cause = ? AND effect = ? AND parameter = ?
            """, (relationship.cause, relationship.effect, relationship.parameter)).fetchone()

            if existing:
                # Update existing
                conn.execute("""
                    UPDATE causal_relationships
                    SET confidence = ?, evidence_count = ?, counterexamples = ?, context = ?
                    WHERE cause = ? AND effect = ? AND parameter = ?
                """, (
                    relationship.confidence,
                    relationship.evidence_count,
                    relationship.counterexamples,
                    relationship.context,
                    relationship.cause,
                    relationship.effect,
                    relationship.parameter
                ))
            else:
                # Insert new
                conn.execute("""
                    INSERT INTO causal_relationships (
                        cause, effect, parameter, confidence,
                        evidence_count, counterexamples, context
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    relationship.cause,
                    relationship.effect,
                    relationship.parameter,
                    relationship.confidence,
                    relationship.evidence_count,
                    relationship.counterexamples,
                    relationship.context
                ))

    def get_warnings_for_change(self, parameter: str, change_type: str) -> List[str]:
        """Get warnings relevant to a proposed parameter change."""
        warnings = []

        # Get parameter knowledge warnings
        knowledge = self.get_parameter_knowledge(parameter)
        if knowledge:
            warnings.extend(knowledge.warnings)

        # Get relevant causal relationships
        with self._connection() as conn:
            relationships = conn.execute("""
                SELECT * FROM causal_relationships
                WHERE parameter = ? AND confidence > 0.5
            """, (parameter,)).fetchall()

            for rel in relationships:
                if change_type in rel['cause'].lower():
                    warnings.append(f"Warning: {rel['cause']} tends to {rel['effect']} (confidence: {rel['confidence']:.0%})")

        return warnings

    # =========================================================================
    # Statistics
    # =========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get overall database statistics."""
        with self._connection() as conn:
            stats = {}

            stats['total_experiments'] = conn.execute(
                "SELECT COUNT(*) FROM experiments"
            ).fetchone()[0]

            stats['successful_experiments'] = conn.execute(
                "SELECT COUNT(*) FROM experiments WHERE success = 1"
            ).fetchone()[0]

            stats['total_sessions'] = conn.execute(
                "SELECT COUNT(DISTINCT session_id) FROM experiments"
            ).fetchone()[0]

            stats['parameters_tracked'] = conn.execute(
                "SELECT COUNT(*) FROM parameter_knowledge"
            ).fetchone()[0]

            stats['causal_relationships'] = conn.execute(
                "SELECT COUNT(*) FROM causal_relationships"
            ).fetchone()[0]

            if stats['total_experiments'] > 0:
                stats['overall_success_rate'] = (
                    stats['successful_experiments'] / stats['total_experiments']
                )
            else:
                stats['overall_success_rate'] = 0.0

            return stats


# =============================================================================
# Convenience Functions
# =============================================================================

def get_db() -> ExperimentDatabase:
    """Get default database instance."""
    return ExperimentDatabase()


def record_experiment(
    session_id: str,
    hypothesis: str,
    issue: str,
    baseline_params: Dict,
    result_params: Dict,
    baseline_scores: Dict,
    result_scores: Dict,
    parameter_changes: List[Dict],
    observed_effects: List[Dict],
    success: bool,
    learnings: List[str],
    warnings: List[str],
    baseline_render: str = "",
    result_render: str = "",
    baseline_script: str = "",
    result_script: str = ""
) -> str:
    """
    Convenience function to record an experiment.

    Returns experiment ID.
    """
    import uuid

    db = get_db()

    experiment = Experiment(
        id=str(uuid.uuid4())[:8],
        session_id=session_id,
        timestamp=datetime.now().isoformat(),
        hypothesis=hypothesis,
        issue_addressed=issue,
        baseline_params=baseline_params,
        result_params=result_params,
        baseline_scores=baseline_scores,
        result_scores=result_scores,
        baseline_render=baseline_render,
        result_render=result_render,
        baseline_script=baseline_script,
        result_script=result_script,
        success=success,
        partial_success=not success and any(
            result_scores.get(k, 0) > baseline_scores.get(k, 0)
            for k in result_scores
        ),
        learnings=learnings,
        warnings=warnings,
        parameter_changes=[
            ParameterChange(**c) for c in parameter_changes
        ],
        observed_effects=[
            ObservedEffect(**e) for e in observed_effects
        ]
    )

    return db.save_experiment(experiment)


if __name__ == "__main__":
    # Test database creation
    db = ExperimentDatabase()
    print(f"Database created at: {db.db_path}")
    print(f"Statistics: {db.get_statistics()}")
