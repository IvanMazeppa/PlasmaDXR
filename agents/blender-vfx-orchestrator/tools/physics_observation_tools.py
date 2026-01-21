"""
Physics Observation Tools for Self-Learning VFX Orchestrator.

These tools enable the Quality Analyst to record physics anomalies that
feed into the learning system. Instead of hardcoding physics rules, we
observe what happens and let the system learn from experimentation.

KEY PRINCIPLE: Observe, don't assume. Record anomalies with full context
so the Learning Agent can correlate parameters with outcomes.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents import function_tool

# Path setup
SCRIPT_DIR = Path(__file__).parent
ORCHESTRATOR_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = ORCHESTRATOR_ROOT.parent.parent

# Physics observations storage
OBSERVATIONS_DIR = ORCHESTRATOR_ROOT / "data" / "physics_observations"
OBSERVATIONS_DIR.mkdir(parents=True, exist_ok=True)

# In-memory cache of recent observations (for correlation)
_recent_observations: List[Dict[str, Any]] = []


def _save_observation(observation: Dict[str, Any]) -> str:
    """Save observation to disk and return observation ID."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    obs_id = f"obs_{timestamp}_{observation.get('effect_type', 'unknown')}"

    filepath = OBSERVATIONS_DIR / f"{obs_id}.json"
    with open(filepath, "w") as f:
        json.dump(observation, f, indent=2, default=str)

    # Also keep in memory for quick access
    _recent_observations.append(observation)
    if len(_recent_observations) > 100:
        _recent_observations.pop(0)

    return obs_id


def _load_observations(
    effect_type: Optional[str] = None,
    limit: int = 20
) -> List[Dict[str, Any]]:
    """Load observations from disk."""
    observations = []

    if OBSERVATIONS_DIR.exists():
        for filepath in sorted(OBSERVATIONS_DIR.glob("*.json"), reverse=True):
            if len(observations) >= limit:
                break
            try:
                with open(filepath) as f:
                    obs = json.load(f)
                    if effect_type is None or obs.get("effect_type") == effect_type:
                        observations.append(obs)
            except Exception:
                continue

    return observations


# =============================================================================
# IMPLEMENTATION FUNCTIONS
# =============================================================================

def _observe_physics_anomaly_impl(
    effect_type: str,
    observation: str,
    expected_behavior: str,
    actual_behavior: str,
    suspected_parameters: Dict[str, Any],
    severity: str = "medium",
    render_path: Optional[str] = None,
    script_path: Optional[str] = None,
    iteration: int = 0
) -> str:
    """
    Record a physics anomaly observation.

    This is the core function that captures what the Quality Analyst sees
    when something doesn't behave as expected. The Learning Agent uses these
    observations to build causal models.

    Args:
        effect_type: Type of effect (sun, explosion, fire, etc.)
        observation: Description of what was observed
        expected_behavior: What should have happened
        actual_behavior: What actually happened
        suspected_parameters: Parameters that might be causing this
        severity: How bad is this? (low, medium, high, critical)
        render_path: Path to the render showing this behavior
        script_path: Path to the script that produced this
        iteration: Current iteration number

    Returns:
        JSON with observation_id and confirmation
    """
    observation_data = {
        "timestamp": datetime.now().isoformat(),
        "effect_type": effect_type,
        "observation": observation,
        "expected_behavior": expected_behavior,
        "actual_behavior": actual_behavior,
        "suspected_parameters": suspected_parameters,
        "severity": severity,
        "render_path": render_path,
        "script_path": script_path,
        "iteration": iteration,
        "status": "pending_analysis",  # Learning Agent will update
        "correlation_found": False,
        "causal_params": [],  # Filled in by Learning Agent
    }

    obs_id = _save_observation(observation_data)

    return json.dumps({
        "success": True,
        "observation_id": obs_id,
        "effect_type": effect_type,
        "severity": severity,
        "message": f"Physics anomaly recorded: {observation[:50]}...",
        "next_step": "Learning Agent should analyze this observation"
    }, indent=2)


def _get_pending_observations_impl(effect_type: Optional[str] = None) -> str:
    """
    Get observations pending analysis by the Learning Agent.

    Args:
        effect_type: Filter by effect type (optional)

    Returns:
        JSON with pending observations
    """
    observations = _load_observations(effect_type=effect_type, limit=50)

    pending = [
        obs for obs in observations
        if obs.get("status") == "pending_analysis"
    ]

    return json.dumps({
        "total_pending": len(pending),
        "observations": pending[:10],  # Return max 10
        "effect_type_filter": effect_type
    }, indent=2, default=str)


def _correlate_observation_impl(
    observation_id: str,
    causal_parameters: List[str],
    correlation_confidence: float,
    recommended_fix: str,
    notes: str = ""
) -> str:
    """
    Mark an observation as analyzed with correlation findings.

    The Learning Agent calls this after analyzing an observation to record
    which parameters actually caused the behavior.

    Args:
        observation_id: ID of the observation to update
        causal_parameters: Parameters that caused the behavior
        correlation_confidence: How confident are we? (0.0-1.0)
        recommended_fix: What parameter changes would fix this?
        notes: Additional analysis notes

    Returns:
        JSON confirmation
    """
    # Find and update the observation
    filepath = OBSERVATIONS_DIR / f"{observation_id}.json"

    if not filepath.exists():
        return json.dumps({
            "success": False,
            "error": f"Observation not found: {observation_id}"
        })

    try:
        with open(filepath) as f:
            obs = json.load(f)

        obs["status"] = "analyzed"
        obs["correlation_found"] = True
        obs["causal_params"] = causal_parameters
        obs["correlation_confidence"] = correlation_confidence
        obs["recommended_fix"] = recommended_fix
        obs["analysis_notes"] = notes
        obs["analyzed_at"] = datetime.now().isoformat()

        with open(filepath, "w") as f:
            json.dump(obs, f, indent=2, default=str)

        return json.dumps({
            "success": True,
            "observation_id": observation_id,
            "causal_params": causal_parameters,
            "confidence": correlation_confidence,
            "message": "Observation analyzed and correlation recorded"
        }, indent=2)

    except Exception as e:
        return json.dumps({
            "success": False,
            "error": str(e)
        })


def _get_physics_patterns_impl(effect_type: str) -> str:
    """
    Get established physics patterns from analyzed observations.

    This aggregates correlations from multiple observations to find
    patterns that appear consistently.

    Args:
        effect_type: Type of effect to get patterns for

    Returns:
        JSON with established patterns and their confidence
    """
    observations = _load_observations(effect_type=effect_type, limit=100)

    # Only use analyzed observations
    analyzed = [
        obs for obs in observations
        if obs.get("status") == "analyzed" and obs.get("correlation_found")
    ]

    # Aggregate patterns by causal parameters
    patterns: Dict[str, Dict[str, Any]] = {}

    for obs in analyzed:
        causal_params = tuple(sorted(obs.get("causal_params", [])))
        if not causal_params:
            continue

        key = str(causal_params)
        if key not in patterns:
            patterns[key] = {
                "parameters": list(causal_params),
                "observations": [],
                "avg_confidence": 0,
                "recommended_fixes": [],
                "behaviors": [],
            }

        patterns[key]["observations"].append(obs.get("observation_id"))
        patterns[key]["avg_confidence"] += obs.get("correlation_confidence", 0)
        patterns[key]["behaviors"].append(obs.get("actual_behavior", ""))

        fix = obs.get("recommended_fix")
        if fix and fix not in patterns[key]["recommended_fixes"]:
            patterns[key]["recommended_fixes"].append(fix)

    # Calculate averages
    for key in patterns:
        n = len(patterns[key]["observations"])
        if n > 0:
            patterns[key]["avg_confidence"] /= n
            patterns[key]["observation_count"] = n

    # Sort by observation count (more observations = more reliable)
    sorted_patterns = sorted(
        patterns.values(),
        key=lambda p: (p["observation_count"], p["avg_confidence"]),
        reverse=True
    )

    return json.dumps({
        "effect_type": effect_type,
        "pattern_count": len(sorted_patterns),
        "patterns": sorted_patterns[:10],  # Top 10
        "note": "Patterns with higher observation_count are more reliable"
    }, indent=2, default=str)


# =============================================================================
# FUNCTION TOOL WRAPPERS
# =============================================================================

@function_tool
async def observe_physics_anomaly(
    effect_type: str,
    observation: str,
    expected_behavior: str,
    actual_behavior: str,
    suspected_parameters: str,
    severity: str = "medium",
    render_path: str = "",
    script_path: str = "",
    iteration: int = 0
) -> str:
    """
    Record a physics anomaly for the Learning Agent to analyze.

    Use this when you observe unexpected physical behavior in a render.
    The Learning Agent will correlate these observations with parameters
    to build a causal model.

    Examples:
    - Sun effect drifts upward when it should be stationary
    - Explosion dissipates too quickly
    - Fire doesn't rise (no buoyancy effect)
    - Fluid clips through domain boundaries

    Args:
        effect_type: Type of effect (sun, explosion, fire, smoke, nebula)
        observation: Brief description of what you observed
        expected_behavior: What should have happened
        actual_behavior: What actually happened
        suspected_parameters: JSON of parameters that might cause this
            Example: '{"beta": 1.0, "alpha": 0.5}'
        severity: How bad is this? (low, medium, high, critical)
        render_path: Path to the render showing this behavior
        script_path: Path to the script that produced this
        iteration: Current iteration number

    Returns:
        JSON with observation_id and confirmation
    """
    try:
        params = json.loads(suspected_parameters) if suspected_parameters else {}
    except json.JSONDecodeError:
        params = {"raw": suspected_parameters}

    return _observe_physics_anomaly_impl(
        effect_type=effect_type,
        observation=observation,
        expected_behavior=expected_behavior,
        actual_behavior=actual_behavior,
        suspected_parameters=params,
        severity=severity,
        render_path=render_path or None,
        script_path=script_path or None,
        iteration=iteration
    )


@function_tool
async def get_pending_observations(effect_type: str = "") -> str:
    """
    Get physics observations pending analysis.

    The Learning Agent uses this to find observations that need
    correlation analysis.

    Args:
        effect_type: Filter by effect type (optional, empty = all)

    Returns:
        JSON with pending observations
    """
    return _get_pending_observations_impl(
        effect_type=effect_type if effect_type else None
    )


@function_tool
async def correlate_observation(
    observation_id: str,
    causal_parameters: str,
    correlation_confidence: float,
    recommended_fix: str,
    notes: str = ""
) -> str:
    """
    Record the analysis results for a physics observation.

    After analyzing an observation, use this to record which parameters
    actually caused the behavior and how to fix it.

    Args:
        observation_id: ID of the observation to update
        causal_parameters: JSON array of parameter names that caused this
            Example: '["beta", "alpha", "scene.gravity"]'
        correlation_confidence: How confident are we? (0.0-1.0)
        recommended_fix: What parameter changes would fix this?
        notes: Additional analysis notes

    Returns:
        JSON confirmation
    """
    try:
        params = json.loads(causal_parameters) if causal_parameters else []
    except json.JSONDecodeError:
        params = [causal_parameters]

    return _correlate_observation_impl(
        observation_id=observation_id,
        causal_parameters=params,
        correlation_confidence=correlation_confidence,
        recommended_fix=recommended_fix,
        notes=notes
    )


@function_tool
async def get_physics_patterns(effect_type: str) -> str:
    """
    Get established physics patterns from past observations.

    Returns patterns that have been observed multiple times and
    have high confidence correlations. Use these to inform script
    modifications.

    Args:
        effect_type: Type of effect to get patterns for

    Returns:
        JSON with established patterns and their confidence
    """
    return _get_physics_patterns_impl(effect_type)
