"""
Knowledge Base Client for Script Generator

Phase 4.1: Provides access to experiment-tracker knowledge base for
mandatory warning checks before parameter modifications.

This module enables the script-generator to consult accumulated knowledge
before making parameter changes, helping avoid known pitfalls.

Usage:
    from knowledge_client import KnowledgeClient

    client = KnowledgeClient()
    result = client.check_before_modify("domain_scale", "increase", 10.0)

    if result.has_critical_warnings:
        # Apply mitigation or skip modification
        pass
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add experiment-tracker to path for importing its modules
EXPERIMENT_TRACKER_PATH = Path(__file__).parent.parent / "experiment-tracker"
if str(EXPERIMENT_TRACKER_PATH) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_TRACKER_PATH))

# Import experiment-tracker modules
try:
    from database import ExperimentDatabase, get_db
    from tracker import ExperimentTracker
    TRACKER_AVAILABLE = True
except ImportError as e:
    print(f"[knowledge_client] Warning: Could not import experiment-tracker: {e}")
    TRACKER_AVAILABLE = False


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class WarningCheck:
    """Result of checking warnings before a parameter change."""
    parameter: str
    change_type: str
    new_value: Optional[Any]

    # Warnings from knowledge base
    warnings: List[str] = field(default_factory=list)
    rules: List[str] = field(default_factory=list)

    # Severity assessment
    has_critical_warnings: bool = False
    severity: str = "none"  # none, low, medium, high, critical

    # Recommendations
    mitigation: Optional[str] = None
    should_proceed: bool = True
    alternative_suggestions: List[str] = field(default_factory=list)

    # Knowledge base info
    experiments_count: int = 0
    success_rate: float = 0.0
    confidence: float = 0.0
    safe_range: Optional[Tuple[float, float]] = None
    optimal_value: Optional[float] = None


@dataclass
class ConflictResolution:
    """Result of resolving conflicting knowledge."""
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    resolution: str = ""
    confidence: float = 0.0
    requires_human_review: bool = False
    winning_rule: Optional[str] = None


# =============================================================================
# CONFLICT RESOLUTION WEIGHTS (Phase 4.2)
# =============================================================================

CONFLICT_WEIGHTS = {
    "RECENCY": 2.0,       # Recent experiments are 2× more valuable
    "SUCCESS_RATE": 1.0,   # Higher success rate wins
    "HUMAN_FEEDBACK": 3.0, # Human feedback is 3× weighted
    "CONTEXT_MATCH": 5.0,  # Exact context match is 5× weighted
}

# Threshold for requesting human review
CONFIDENCE_DIFF_THRESHOLD = 0.20  # If difference < 20%, request human review


# =============================================================================
# KNOWLEDGE CLIENT
# =============================================================================

class KnowledgeClient:
    """
    Client for accessing experiment-tracker knowledge base.

    Provides mandatory warning checks before parameter modifications
    and conflict resolution for contradictory knowledge.
    """

    def __init__(self, db: Optional[Any] = None):
        """
        Initialize knowledge client.

        Args:
            db: Optional ExperimentDatabase instance (creates default if None)
        """
        self._db = None
        self._tracker = None

        if TRACKER_AVAILABLE:
            try:
                self._db = db or get_db()
                self._tracker = ExperimentTracker(db=self._db, auto_load_knowledge=True)
            except Exception as e:
                print(f"[knowledge_client] Warning: Could not initialize tracker: {e}")

    @property
    def available(self) -> bool:
        """Check if knowledge base is available."""
        return self._tracker is not None

    def check_before_modify(
        self,
        parameter: str,
        change_type: str,
        new_value: Optional[Any] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> WarningCheck:
        """
        Check knowledge base before making a parameter modification.

        This is the PRIMARY function for Phase 4.1 mandatory warning checks.

        Args:
            parameter: Parameter being modified (e.g., "domain_scale")
            change_type: Type of change ("increase", "decrease", "modify", "set")
            new_value: The new value being set (optional)
            context: Optional context dict (effect_type, current_params, etc.)

        Returns:
            WarningCheck with warnings, severity, and recommendations
        """
        result = WarningCheck(
            parameter=parameter,
            change_type=change_type,
            new_value=new_value
        )

        if not self.available:
            # No knowledge base - allow proceeding but note it
            result.warnings.append("Knowledge base not available - no historical data")
            return result

        # Get warnings from tracker
        try:
            warnings = self._tracker.get_warnings_for_change(
                parameter=parameter,
                change_type=change_type,
                new_value=new_value
            )
            result.warnings = warnings
        except Exception as e:
            result.warnings.append(f"Error querying warnings: {e}")

        # Get parameter info
        try:
            param_info = self._tracker.get_parameter_info(parameter)
            if param_info.get('known', False):
                result.experiments_count = param_info.get('experiments_count', 0)
                result.success_rate = float(param_info.get('success_rate', '0%').rstrip('%')) / 100
                result.rules = param_info.get('rules', [])
                result.confidence = float(param_info.get('confidence', '0%').rstrip('%')) / 100
                result.safe_range = param_info.get('safe_range')
                result.optimal_value = param_info.get('optimal_value')

                # Add rules to warnings if relevant
                for rule in result.rules:
                    if change_type.lower() in rule.lower():
                        result.warnings.append(f"Rule: {rule}")
        except Exception as e:
            pass  # Silently continue if param info fails

        # Assess severity based on warnings
        result = self._assess_severity(result, context)

        return result

    def _assess_severity(
        self,
        result: WarningCheck,
        context: Optional[Dict[str, Any]] = None
    ) -> WarningCheck:
        """
        Assess the severity of warnings and provide recommendations.

        Severity levels:
        - none: No warnings
        - low: Informational warnings only
        - medium: Caution advised
        - high: Significant risk
        - critical: Should not proceed without mitigation
        """
        if not result.warnings:
            result.severity = "none"
            result.should_proceed = True
            return result

        # Check for critical patterns in warnings
        critical_patterns = [
            "MUST", "NEVER", "ALWAYS", "explosion", "crash", "fail",
            "INSUFFICIENT", "unstable", "unreliable"
        ]

        high_patterns = [
            "should", "risk", "may cause", "can cause", "warning"
        ]

        medium_patterns = [
            "recommend", "suggest", "consider", "might", "could"
        ]

        warning_text = " ".join(result.warnings).lower()

        critical_count = sum(1 for p in critical_patterns if p.lower() in warning_text)
        high_count = sum(1 for p in high_patterns if p.lower() in warning_text)
        medium_count = sum(1 for p in medium_patterns if p.lower() in warning_text)

        # Determine severity
        if critical_count >= 2 or "must" in warning_text:
            result.severity = "critical"
            result.has_critical_warnings = True
            result.should_proceed = False
            result.mitigation = self._generate_mitigation(result)
        elif critical_count >= 1 or high_count >= 2:
            result.severity = "high"
            result.should_proceed = True  # Proceed with caution
            result.mitigation = self._generate_mitigation(result)
        elif high_count >= 1 or medium_count >= 2:
            result.severity = "medium"
            result.should_proceed = True
        elif medium_count >= 1 or len(result.warnings) > 0:
            result.severity = "low"
            result.should_proceed = True
        else:
            result.severity = "none"
            result.should_proceed = True

        # Check if new_value is outside safe range
        if result.safe_range and result.new_value is not None:
            try:
                val = float(result.new_value)
                min_val, max_val = result.safe_range
                if val < min_val or val > max_val:
                    result.warnings.append(
                        f"Value {val} is outside safe range [{min_val}, {max_val}]"
                    )
                    if result.severity in ("none", "low"):
                        result.severity = "medium"
            except (ValueError, TypeError):
                pass  # Can't compare non-numeric values

        return result

    def _generate_mitigation(self, result: WarningCheck) -> str:
        """Generate mitigation recommendation based on warnings."""
        mitigations = []

        # Parameter-specific mitigations
        param = result.parameter.lower()
        change = result.change_type.lower()

        if "domain_scale" in param and "increase" in change:
            mitigations.append(
                "When increasing domain_scale, also adjust domain_location_z "
                "proportionally to keep the effect centered"
            )

        if "step_min" in param and "decrease" in change:
            mitigations.append(
                "Decreasing step_min can cause simulation instability. "
                "Consider using step_min >= 20 for soft body physics"
            )

        if "damping" in param and "decrease" in change:
            mitigations.append(
                "Low damping can cause energy buildup and mesh explosion. "
                "Use damping >= 2.0 for stable soft body simulation"
            )

        if result.optimal_value is not None:
            mitigations.append(
                f"Consider using optimal value: {result.optimal_value}"
            )

        if result.safe_range is not None:
            mitigations.append(
                f"Safe range: {result.safe_range[0]} to {result.safe_range[1]}"
            )

        return " | ".join(mitigations) if mitigations else "Proceed with caution"

    def resolve_conflicts(
        self,
        parameter: str,
        conflicting_rules: List[Dict[str, Any]]
    ) -> ConflictResolution:
        """
        Resolve conflicting knowledge about a parameter (Phase 4.2).

        Uses weighted scoring:
        - RECENCY: Recent experiments 2× weight
        - SUCCESS_RATE: Higher rate wins
        - HUMAN_FEEDBACK: 3× weight
        - CONTEXT_MATCH: 5× weight if exact match

        Args:
            parameter: Parameter with conflicting knowledge
            conflicting_rules: List of conflicting rules with metadata

        Returns:
            ConflictResolution with winning rule and confidence
        """
        if not conflicting_rules:
            return ConflictResolution()

        if len(conflicting_rules) == 1:
            return ConflictResolution(
                conflicts=conflicting_rules,
                resolution="Single rule - no conflict",
                confidence=1.0,
                winning_rule=conflicting_rules[0].get("rule", "")
            )

        # Score each rule
        scored_rules = []
        for rule in conflicting_rules:
            score = 0.0

            # Recency (based on timestamp if available)
            recency_score = rule.get("recency_score", 0.5)
            score += recency_score * CONFLICT_WEIGHTS["RECENCY"]

            # Success rate
            success_rate = rule.get("success_rate", 0.5)
            score += success_rate * CONFLICT_WEIGHTS["SUCCESS_RATE"]

            # Human feedback
            if rule.get("has_human_feedback", False):
                human_rating = rule.get("human_rating", 3) / 5.0  # Normalize to 0-1
                score += human_rating * CONFLICT_WEIGHTS["HUMAN_FEEDBACK"]

            # Context match
            if rule.get("context_match", False):
                score += 1.0 * CONFLICT_WEIGHTS["CONTEXT_MATCH"]

            scored_rules.append((score, rule))

        # Sort by score descending
        scored_rules.sort(reverse=True, key=lambda x: x[0])

        winner_score, winner = scored_rules[0]
        second_score = scored_rules[1][0] if len(scored_rules) > 1 else 0

        # Calculate confidence
        total_score = sum(s for s, _ in scored_rules)
        confidence = winner_score / total_score if total_score > 0 else 0.5

        # Check if human review needed
        score_diff = (winner_score - second_score) / winner_score if winner_score > 0 else 0
        requires_review = score_diff < CONFIDENCE_DIFF_THRESHOLD

        return ConflictResolution(
            conflicts=conflicting_rules,
            resolution=f"Winner: {winner.get('rule', 'Unknown')} (score: {winner_score:.2f})",
            confidence=confidence,
            requires_human_review=requires_review,
            winning_rule=winner.get("rule", "")
        )

    def preload_knowledge_for_effect(self, effect_type: str) -> Dict[str, Any]:
        """
        Preload relevant knowledge for an effect type (Phase 4.3).

        Called at session start to cache relevant knowledge.

        Args:
            effect_type: Type of effect being created

        Returns:
            Dict with preloaded knowledge
        """
        if not self.available:
            return {"available": False}

        preloaded = {
            "available": True,
            "effect_type": effect_type,
            "parameters": {},
            "rules": [],
            "warnings": [],
            "suggestions": []
        }

        # Common parameters for each effect type
        effect_params = {
            "pyro": ["flame_smoke", "vorticity", "burning_rate", "temperature", "domain_scale"],
            "explosion": ["flame_smoke", "vorticity", "burning_rate", "domain_scale", "frame_end"],
            "fire": ["flame_max_temp", "flame_smoke", "burning_rate", "vorticity"],
            "smoke": ["vorticity", "dissolve_speed", "domain_scale"],
            "soft_body": ["step_min", "step_max", "damping", "goal_spring", "friction"],
            "cloth": ["quality", "mass", "air_damping", "collision_quality"],
            "rigid_body": ["friction", "bounciness", "collision_shape"],
            "nebula": ["noise_scale", "noise_strength", "vorticity"],
            "sun": ["temperature", "emission_strength", "noise_scale"],
        }

        params_to_load = effect_params.get(effect_type.lower(), [])

        for param in params_to_load:
            try:
                info = self._tracker.get_parameter_info(param)
                if info.get('known', False):
                    preloaded["parameters"][param] = info
                    preloaded["rules"].extend(info.get("rules", []))
                    preloaded["warnings"].extend(info.get("warnings", []))
            except Exception:
                pass

        # Deduplicate
        preloaded["rules"] = list(set(preloaded["rules"]))
        preloaded["warnings"] = list(set(preloaded["warnings"]))

        return preloaded


# =============================================================================
# MODULE-LEVEL CONVENIENCE FUNCTIONS
# =============================================================================

_client: Optional[KnowledgeClient] = None


def get_knowledge_client() -> KnowledgeClient:
    """Get or create global knowledge client instance."""
    global _client
    if _client is None:
        _client = KnowledgeClient()
    return _client


def check_before_modify(
    parameter: str,
    change_type: str,
    new_value: Optional[Any] = None,
    context: Optional[Dict[str, Any]] = None
) -> WarningCheck:
    """
    Convenience function for mandatory warning check.

    Call this before every modify_script() call.
    """
    return get_knowledge_client().check_before_modify(
        parameter=parameter,
        change_type=change_type,
        new_value=new_value,
        context=context
    )


def preload_knowledge(effect_type: str) -> Dict[str, Any]:
    """Convenience function for knowledge preloading."""
    return get_knowledge_client().preload_knowledge_for_effect(effect_type)


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("=== Knowledge Client Test ===\n")

    client = KnowledgeClient()
    print(f"Knowledge base available: {client.available}\n")

    # Test warning checks
    test_cases = [
        ("domain_scale", "increase", 15.0),
        ("step_min", "decrease", 3),
        ("damping", "decrease", 0.1),
        ("resolution", "increase", 256),
    ]

    print("=== Warning Check Tests ===\n")
    for param, change, value in test_cases:
        result = client.check_before_modify(param, change, value)
        print(f"{param} ({change} to {value}):")
        print(f"  Severity: {result.severity}")
        print(f"  Should proceed: {result.should_proceed}")
        print(f"  Warnings: {len(result.warnings)}")
        if result.mitigation:
            print(f"  Mitigation: {result.mitigation}")
        print()

    # Test knowledge preloading
    print("=== Knowledge Preloading ===\n")
    for effect in ["pyro", "soft_body", "unknown"]:
        preloaded = client.preload_knowledge_for_effect(effect)
        print(f"{effect}:")
        print(f"  Parameters loaded: {len(preloaded.get('parameters', {}))}")
        print(f"  Rules: {len(preloaded.get('rules', []))}")
        print(f"  Warnings: {len(preloaded.get('warnings', []))}")
        print()
