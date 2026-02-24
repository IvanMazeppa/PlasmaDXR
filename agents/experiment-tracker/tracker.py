#!/usr/bin/env python3
"""
Experiment Tracker

High-level interface for tracking experiments, learning from results,
and providing recommendations based on accumulated knowledge.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, asdict

from database import (
    ExperimentDatabase, Experiment, ParameterChange, ObservedEffect,
    CausalRelationship, ParameterKnowledge, get_db
)


def _normalize_observed_effect(effect_dict: Dict[str, Any]) -> ObservedEffect:
    """
    Normalize an observed effect dict into an ObservedEffect dataclass.
    Handles cases where agents pass side_effect as a string description.
    """
    # Extract fields with defaults
    metric = effect_dict.get('metric', 'unknown')
    direction = effect_dict.get('direction', 'unknown')

    # LLM often passes qualitative strings like "high", "medium", "low"
    _mag_raw = effect_dict.get('magnitude', 0.0)
    _QUAL_TO_FLOAT = {"high": 0.9, "medium": 0.5, "low": 0.2, "none": 0.0}
    if isinstance(_mag_raw, str):
        magnitude = _QUAL_TO_FLOAT.get(_mag_raw.lower().strip(), 0.5)
    else:
        try:
            magnitude = float(_mag_raw)
        except (TypeError, ValueError):
            magnitude = 0.0

    expected = bool(effect_dict.get('expected', False))

    # Handle side_effect - can be bool, string, or missing
    side_effect_raw = effect_dict.get('side_effect', False)
    description = effect_dict.get('description', '')

    if isinstance(side_effect_raw, str):
        # Agent passed a description as side_effect - this is common
        # Treat non-empty string as True (is a side effect), use string as description
        side_effect = bool(side_effect_raw.strip())
        if not description:
            description = side_effect_raw
    else:
        side_effect = bool(side_effect_raw)

    return ObservedEffect(
        metric=metric,
        direction=direction,
        magnitude=magnitude,
        expected=expected,
        side_effect=side_effect,
        description=description
    )


@dataclass
class ExperimentPlan:
    """A planned experiment to test a hypothesis."""
    hypothesis: str
    issue_addressed: str
    parameter_changes: List[Dict[str, Any]]
    expected_effects: List[str]
    risks: List[str]
    confidence: float


@dataclass
class ExperimentResult:
    """Result of a completed experiment."""
    experiment_id: str
    success: bool
    partial_success: bool
    score_changes: Dict[str, float]  # metric -> change
    observed_effects: List[Dict[str, Any]]
    learnings: List[str]
    warnings: List[str]
    recommendations: List[str]


class ExperimentTracker:
    """
    Main interface for experiment tracking and learning.

    Responsibilities:
    - Record experiments with full context
    - Learn from results and build knowledge base
    - Provide warnings before making changes
    - Suggest experiments based on issues
    - Query knowledge base for guidance
    """

    def __init__(self, db: Optional[ExperimentDatabase] = None, auto_load_knowledge: bool = True):
        self.db = db or get_db()
        self._current_session: Optional[str] = None
        self._baseline_state: Optional[Dict] = None

        # Load knowledge from JSON files on initialization
        if auto_load_knowledge:
            self._load_knowledge_files()

    def _load_knowledge_files(self):
        """Load knowledge from JSON files in knowledge_base directory."""
        knowledge_dir = Path(__file__).parent / "knowledge_base"
        if not knowledge_dir.exists():
            return

        for json_file in knowledge_dir.glob("*.json"):
            try:
                with open(json_file, 'r') as f:
                    knowledge = json.load(f)

                # Extract effect type or use filename
                effect_type = knowledge.get("metadata", {}).get("effect_type", json_file.stem)

                # Load rules as parameter knowledge
                for rule in knowledge.get("rules", []):
                    self.add_manual_learning(
                        parameter=f"{effect_type}_general",
                        rule=rule
                    )

                # Load warnings
                for warning in knowledge.get("warnings", []):
                    self.add_manual_learning(
                        parameter=f"{effect_type}_general",
                        rule="",
                        warning=warning
                    )

                # Load recommended settings as parameter knowledge
                for param_name, param_data in knowledge.get("recommended_settings", {}).items():
                    if isinstance(param_data, dict):
                        rule = f"Recommended value: {param_data.get('value')} (default: {param_data.get('default')})"
                        if param_data.get("reason"):
                            rule += f". Reason: {param_data['reason']}"
                        self.add_manual_learning(
                            parameter=f"{effect_type}_{param_name}",
                            rule=rule,
                            warning=param_data.get("warning")
                        )

                # Load common issues as warnings
                for issue in knowledge.get("common_issues", []):
                    if isinstance(issue, dict):
                        warning = f"Issue: {issue.get('symptom')} | Cause: {issue.get('cause')} | Fix: {issue.get('fix')}"
                        self.add_manual_learning(
                            parameter=f"{effect_type}_issues",
                            rule="",
                            warning=warning
                        )

            except Exception as e:
                # Don't fail initialization if a knowledge file is malformed
                print(f"[tracker] Warning: Could not load {json_file}: {e}")

    # =========================================================================
    # Session Management
    # =========================================================================

    def start_session(
        self,
        asset_name: str,
        effect_type: str,
        description: str,
        reference_path: Optional[str] = None,
        semantic_query: Optional[str] = None
    ) -> str:
        """Start a new experiment session for an asset."""
        session_id = f"{asset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._current_session = session_id

        # Record session start
        with self.db._connection() as conn:
            conn.execute("""
                INSERT INTO sessions (
                    id, asset_name, effect_type, description,
                    started_at, reference_path, semantic_query
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                session_id, asset_name, effect_type, description,
                datetime.now().isoformat(), reference_path, semantic_query
            ))

        return session_id

    def end_session(self, final_status: str, best_score: float):
        """End the current session."""
        if not self._current_session:
            return

        with self.db._connection() as conn:
            conn.execute("""
                UPDATE sessions
                SET completed_at = ?, final_status = ?, best_score = ?,
                    experiments_count = (
                        SELECT COUNT(*) FROM experiments
                        WHERE session_id = ?
                    )
                WHERE id = ?
            """, (
                datetime.now().isoformat(),
                final_status,
                best_score,
                self._current_session,
                self._current_session
            ))

        self._current_session = None
        self._baseline_state = None

    # =========================================================================
    # Baseline Recording
    # =========================================================================

    def record_baseline(
        self,
        params: Dict[str, Any],
        scores: Dict[str, float],
        render_path: str,
        script_path: str
    ):
        """Record baseline state before an experiment."""
        self._baseline_state = {
            'params': params,
            'scores': scores,
            'render_path': render_path,
            'script_path': script_path,
            'timestamp': datetime.now().isoformat()
        }

    # =========================================================================
    # Experiment Recording
    # =========================================================================

    def record_experiment(
        self,
        hypothesis: str,
        issue_addressed: str,
        result_params: Dict[str, Any],
        result_scores: Dict[str, float],
        result_render: str,
        result_script: str,
        success: bool,
        observed_effects: List[Dict[str, Any]],
        learnings: List[str],
        warnings: List[str],
        human_notes: str = ""
    ) -> ExperimentResult:
        """
        Record a completed experiment and update knowledge base.

        Args:
            hypothesis: What we were testing
            issue_addressed: The problem we tried to fix
            result_params: Parameters after the change
            result_scores: Evaluation scores after the change
            result_render: Path to result render
            result_script: Path to result script
            success: Did the experiment achieve its goal?
            observed_effects: List of effects observed
            learnings: What we learned
            warnings: Gotchas discovered
            human_notes: Optional human observations

        Returns:
            ExperimentResult with analysis
        """
        if not self._baseline_state:
            raise ValueError("No baseline recorded. Call record_baseline() first.")

        if not self._current_session:
            self._current_session = f"adhoc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Calculate parameter changes
        parameter_changes = self._calculate_param_changes(
            self._baseline_state['params'],
            result_params
        )

        # Calculate score changes (coerce to float — LLM may pass string values)
        def _to_float(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return 0.0

        score_changes = {
            metric: _to_float(result_scores.get(metric, 0)) - _to_float(self._baseline_state['scores'].get(metric, 0))
            for metric in set(result_scores.keys()) | set(self._baseline_state['scores'].keys())
        }

        # Create experiment record
        experiment_id = str(uuid.uuid4())[:8]
        experiment = Experiment(
            id=experiment_id,
            session_id=self._current_session,
            timestamp=datetime.now().isoformat(),
            hypothesis=hypothesis,
            issue_addressed=issue_addressed,
            baseline_params=self._baseline_state['params'],
            result_params=result_params,
            baseline_scores=self._baseline_state['scores'],
            result_scores=result_scores,
            baseline_render=self._baseline_state['render_path'],
            result_render=result_render,
            baseline_script=self._baseline_state['script_path'],
            result_script=result_script,
            success=success,
            partial_success=not success and any(v > 0 for v in score_changes.values()),
            learnings=learnings,
            warnings=warnings,
            human_notes=human_notes,
            parameter_changes=[ParameterChange(**c) for c in parameter_changes],
            observed_effects=[_normalize_observed_effect(e) for e in observed_effects]
        )

        # Save to database
        self.db.save_experiment(experiment)

        # Update knowledge base
        self._update_knowledge_from_experiment(experiment)

        # Generate recommendations
        recommendations = self._generate_recommendations(experiment, score_changes)

        # Update baseline for next experiment
        self._baseline_state = {
            'params': result_params,
            'scores': result_scores,
            'render_path': result_render,
            'script_path': result_script,
            'timestamp': datetime.now().isoformat()
        }

        return ExperimentResult(
            experiment_id=experiment_id,
            success=success,
            partial_success=experiment.partial_success,
            score_changes=score_changes,
            observed_effects=observed_effects,
            learnings=learnings,
            warnings=warnings,
            recommendations=recommendations
        )

    def _calculate_param_changes(
        self,
        baseline: Dict[str, Any],
        result: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Calculate what parameters changed between baseline and result."""
        changes = []

        all_params = set(baseline.keys()) | set(result.keys())

        for param in all_params:
            old_val = baseline.get(param)
            new_val = result.get(param)

            if old_val != new_val:
                # Determine change type and magnitude
                if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
                    if old_val == 0:
                        magnitude = float('inf') if new_val != 0 else 0
                    else:
                        magnitude = new_val / old_val

                    change_type = "increase" if new_val > old_val else "decrease"
                else:
                    magnitude = 1.0
                    change_type = "modify"

                changes.append({
                    'parameter': param,
                    'old_value': old_val,
                    'new_value': new_val,
                    'change_type': change_type,
                    'magnitude': magnitude,
                    'rationale': ''
                })

        return changes

    def _update_knowledge_from_experiment(self, experiment: Experiment):
        """Update knowledge base based on experiment results."""
        for change in experiment.parameter_changes:
            # Create causal relationships from observed effects
            causal_effects = []

            for effect in experiment.observed_effects:
                relationship = CausalRelationship(
                    cause=f"{change.change_type} {change.parameter}",
                    effect=f"{effect.metric} {effect.direction}",
                    parameter=change.parameter,
                    confidence=0.5 if experiment.success else 0.3,
                    evidence_count=1,
                    counterexamples=0,
                    context=experiment.issue_addressed
                )
                causal_effects.append(relationship)

                # Add to database
                self.db.add_causal_relationship(relationship)

            # Update parameter knowledge
            self.db.update_parameter_knowledge(
                change.parameter,
                experiment,
                causal_effects
            )

    def _generate_recommendations(
        self,
        experiment: Experiment,
        score_changes: Dict[str, float]
    ) -> List[str]:
        """Generate recommendations based on experiment results."""
        recommendations = []

        if experiment.success:
            recommendations.append(
                f"SUCCESS: {experiment.hypothesis} - approach validated"
            )
        elif experiment.partial_success:
            recommendations.append(
                f"PARTIAL: Some improvement seen, but goal not fully achieved"
            )

            # Suggest refinements
            for effect in experiment.observed_effects:
                if effect.side_effect:
                    recommendations.append(
                        f"Side effect detected: {effect.metric} {effect.direction}. "
                        f"Consider compensating adjustment."
                    )
        else:
            recommendations.append(
                f"FAILED: {experiment.hypothesis} - try different approach"
            )

            # Check knowledge base for alternatives
            for change in experiment.parameter_changes:
                knowledge = self.db.get_parameter_knowledge(change.parameter)
                if knowledge and knowledge.rules:
                    recommendations.append(
                        f"Known rule for {change.parameter}: {knowledge.rules[0]}"
                    )

        return recommendations

    # =========================================================================
    # Pre-experiment Guidance
    # =========================================================================

    def get_warnings_for_change(
        self,
        parameter: str,
        change_type: str,
        new_value: Any = None
    ) -> List[str]:
        """Get warnings before making a parameter change."""
        return self.db.get_warnings_for_change(parameter, change_type)

    def suggest_experiments(
        self,
        issue: str,
        current_params: Dict[str, Any],
        current_scores: Dict[str, float]
    ) -> List[ExperimentPlan]:
        """
        Suggest experiments to address an issue.

        Uses knowledge base to propose informed experiments.
        """
        suggestions = []

        # Query knowledge base for relevant information
        knowledge_results = self.db.query_knowledge(issue)

        # Common issue patterns and their typical solutions
        issue_patterns = {
            'clipping': [
                {
                    'param': 'domain_scale',
                    'change': 'increase',
                    'companion': {'domain_location_z': 'decrease proportionally'},
                    'confidence': 0.8
                },
                {
                    'param': 'use_adaptive_domain',
                    'change': 'enable',
                    'confidence': 0.7
                }
            ],
            'framing': [
                {
                    'param': 'camera_location',
                    'change': 'pull back',
                    'confidence': 0.7
                },
                {
                    'param': 'domain_location',
                    'change': 'adjust',
                    'confidence': 0.6
                }
            ],
            'smoke': [
                {
                    'param': 'flame_smoke',
                    'change': 'increase',
                    'confidence': 0.8
                },
                {
                    'param': 'smoke_density',
                    'change': 'increase',
                    'confidence': 0.7
                }
            ],
            'intensity': [
                {
                    'param': 'flame_max_temp',
                    'change': 'increase',
                    'confidence': 0.7
                },
                {
                    'param': 'blackbody_intensity',
                    'change': 'increase',
                    'confidence': 0.8
                }
            ]
        }

        # Match issue to patterns
        issue_lower = issue.lower()
        for pattern, solutions in issue_patterns.items():
            if pattern in issue_lower:
                for sol in solutions:
                    # Check knowledge base for warnings
                    warnings = self.get_warnings_for_change(sol['param'], sol['change'])

                    # Adjust confidence based on knowledge
                    knowledge = self.db.get_parameter_knowledge(sol['param'])
                    if knowledge:
                        confidence = (sol['confidence'] + knowledge.confidence) / 2
                    else:
                        confidence = sol['confidence']

                    plan = ExperimentPlan(
                        hypothesis=f"{sol['change'].title()} {sol['param']} to fix {pattern}",
                        issue_addressed=issue,
                        parameter_changes=[{
                            'parameter': sol['param'],
                            'change_type': sol['change'],
                            'rationale': f"Common fix for {pattern} issues"
                        }],
                        expected_effects=[f"Reduce {pattern} problem"],
                        risks=warnings,
                        confidence=confidence
                    )
                    suggestions.append(plan)

        # Sort by confidence
        suggestions.sort(key=lambda x: x.confidence, reverse=True)

        return suggestions[:5]  # Top 5 suggestions

    # =========================================================================
    # Knowledge Queries
    # =========================================================================

    def query_knowledge(self, query: str) -> Dict[str, Any]:
        """Query the knowledge base for information."""
        results = self.db.query_knowledge(query)

        return {
            'query': query,
            'results_count': len(results),
            'results': results,
            'summary': self._summarize_knowledge_results(results)
        }

    def _summarize_knowledge_results(self, results: List[Dict]) -> str:
        """Summarize knowledge query results."""
        if not results:
            return "No relevant knowledge found."

        summary_parts = []

        for r in results:
            if r['type'] == 'parameter_knowledge':
                if r['rules']:
                    summary_parts.append(f"Rules for {r['parameter']}: {r['rules'][0]}")
                if r['warnings']:
                    summary_parts.append(f"Warning: {r['warnings'][0]}")

            elif r['type'] == 'causal_relationship':
                summary_parts.append(
                    f"{r['cause']} -> {r['effect']} "
                    f"(confidence: {r['confidence']:.0%})"
                )

        return " | ".join(summary_parts[:3])  # Top 3 points

    def get_parameter_info(self, parameter: str) -> Dict[str, Any]:
        """Get all known information about a parameter."""
        knowledge = self.db.get_parameter_knowledge(parameter)

        if not knowledge:
            return {
                'parameter': parameter,
                'known': False,
                'message': f"No experiments recorded for {parameter}"
            }

        return {
            'parameter': parameter,
            'known': True,
            'experiments_count': knowledge.experiments_count,
            'success_rate': f"{knowledge.success_rate:.0%}",
            'rules': knowledge.rules,
            'warnings': knowledge.warnings,
            'confidence': f"{knowledge.confidence:.0%}",
            'safe_range': knowledge.safe_range,
            'optimal_value': knowledge.optimal_value
        }

    # =========================================================================
    # Human Feedback
    # =========================================================================

    def add_human_feedback(
        self,
        experiment_id: str,
        rating: int,
        notes: str
    ):
        """Add human feedback to an experiment."""
        with self.db._connection() as conn:
            conn.execute("""
                UPDATE experiments
                SET human_rating = ?, human_notes = ?
                WHERE id = ?
            """, (rating, notes, experiment_id))

    def add_manual_learning(
        self,
        parameter: str,
        rule: str,
        warning: Optional[str] = None,
        context: str = ""
    ):
        """Manually add a learning to the knowledge base."""
        # Create a synthetic experiment to record this
        with self.db._connection() as conn:
            # Get or create parameter knowledge
            existing = conn.execute(
                "SELECT * FROM parameter_knowledge WHERE parameter = ?",
                (parameter,)
            ).fetchone()

            if existing:
                rules = json.loads(existing['rules'] or '[]')
                warnings_list = json.loads(existing['warnings'] or '[]')

                if rule and rule not in rules:
                    rules.append(rule)
                if warning and warning not in warnings_list:
                    warnings_list.append(warning)

                conn.execute("""
                    UPDATE parameter_knowledge
                    SET rules = ?, warnings = ?, last_updated = ?
                    WHERE parameter = ?
                """, (
                    json.dumps(rules),
                    json.dumps(warnings_list),
                    datetime.now().isoformat(),
                    parameter
                ))
            else:
                now_iso = datetime.now().isoformat()
                conn.execute("""
                    INSERT INTO parameter_knowledge (
                        parameter, rules, warnings, last_updated, confidence,
                        last_reinforced, reinforcement_count
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    parameter,
                    json.dumps([rule] if rule else []),
                    json.dumps([warning] if warning else []),
                    now_iso,
                    0.5,  # Manual entries start with moderate confidence
                    now_iso,  # Phase 2B-6: Initialize reinforcement so seeds don't decay
                    1,        # Phase 2B-6: Count=1 prevents immediate decay
                ))

    # =========================================================================
    # Statistics and Reports
    # =========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get overall statistics."""
        return self.db.get_statistics()

    def get_session_report(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate a report for a session."""
        session_id = session_id or self._current_session

        if not session_id:
            return {'error': 'No session specified'}

        experiments = self.db.get_session_experiments(session_id)

        if not experiments:
            return {
                'session_id': session_id,
                'experiments_count': 0,
                'message': 'No experiments recorded'
            }

        successful = sum(1 for e in experiments if e.success)
        partial = sum(1 for e in experiments if e.partial_success)

        all_learnings = []
        all_warnings = []
        for exp in experiments:
            all_learnings.extend(exp.learnings)
            all_warnings.extend(exp.warnings)

        return {
            'session_id': session_id,
            'experiments_count': len(experiments),
            'successful': successful,
            'partial_success': partial,
            'failed': len(experiments) - successful - partial,
            'success_rate': f"{successful / len(experiments):.0%}" if experiments else "N/A",
            'learnings': list(set(all_learnings)),
            'warnings': list(set(all_warnings)),
            'experiments': [
                {
                    'id': e.id,
                    'hypothesis': e.hypothesis,
                    'success': e.success,
                    'timestamp': e.timestamp
                }
                for e in experiments
            ]
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def get_tracker() -> ExperimentTracker:
    """Get default tracker instance."""
    return ExperimentTracker()


if __name__ == "__main__":
    # Test tracker
    tracker = ExperimentTracker()
    print(f"Statistics: {tracker.get_statistics()}")

    # Test suggest experiments
    suggestions = tracker.suggest_experiments(
        issue="clipping at top edge",
        current_params={'domain_scale': 6.0},
        current_scores={'clip': 0.64}
    )
    print(f"\nSuggestions for 'clipping at top edge':")
    for s in suggestions:
        print(f"  - {s.hypothesis} (confidence: {s.confidence:.0%})")
        if s.risks:
            print(f"    Risks: {s.risks}")
