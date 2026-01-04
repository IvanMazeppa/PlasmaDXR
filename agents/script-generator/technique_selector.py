"""
Intelligent Technique Selection for Blender VFX Pipeline

Phase 3: Uses UCB1 (Upper Confidence Bound) algorithm for exploration/exploitation
balance when selecting techniques for VFX generation.

Key Features:
1. Tracks technique performance (success rate, average score, iterations to pass)
2. UCB1 balances trying proven techniques vs exploring new ones
3. Untried techniques get priority (exploration bonus)
4. Keyword relevance filtering before UCB1 selection

References:
- MULTI_AGENT_IMPROVEMENT_PLAN_V2.md: Phase 3 specification
- Upper Confidence Bound algorithm for multi-armed bandit problem
"""

import json
import math
import random
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Import technique catalog
from technique_catalog import PYRO_TECHNIQUES, list_all_techniques


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TechniquePerformance:
    """Performance record for a technique."""
    technique_name: str
    effect_type: str
    success_count: int = 0
    failure_count: int = 0
    total_score: float = 0.0  # Sum of all final scores
    total_iterations: int = 0  # Sum of iterations to pass
    exploration_bonus: float = 1.0
    last_used: Optional[str] = None

    @property
    def trials(self) -> int:
        """Total number of times technique was used."""
        return self.success_count + self.failure_count

    @property
    def avg_score(self) -> float:
        """Average final score when using this technique."""
        if self.trials == 0:
            return 0.0
        return self.total_score / self.trials

    @property
    def avg_iterations(self) -> float:
        """Average iterations to pass when successful."""
        if self.success_count == 0:
            return float('inf')
        return self.total_iterations / self.success_count

    @property
    def success_rate(self) -> float:
        """Success rate (0-1)."""
        if self.trials == 0:
            return 0.0
        return self.success_count / self.trials


@dataclass
class TechniqueRecommendation:
    """Result of technique selection."""
    technique_name: str
    confidence: float  # 0-1, how confident we are in this choice
    selection_reason: str  # Why this technique was selected
    ucb_score: float
    alternatives: List[Dict[str, Any]] = field(default_factory=list)
    keyword_matches: int = 0
    exploration_mode: bool = False  # True if this is an untried technique


# =============================================================================
# PERSISTENCE
# =============================================================================

class TechniquePerformanceStore:
    """
    Persistent storage for technique performance data.

    Stores performance in JSON format for cross-session learning.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        if storage_path is None:
            storage_path = Path(__file__).parent.parent / "experiment-tracker" / "data" / "technique_performance.json"
        self.storage_path = storage_path
        self._cache: Dict[str, TechniquePerformance] = {}
        self._load()

    def _key(self, technique_name: str, effect_type: str) -> str:
        """Generate storage key."""
        return f"{effect_type}:{technique_name}"

    def _load(self):
        """Load performance data from disk."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                for key, record in data.get("techniques", {}).items():
                    self._cache[key] = TechniquePerformance(**record)
            except Exception as e:
                print(f"[technique_selector] Warning: Could not load performance data: {e}")

    def _save(self):
        """Save performance data to disk."""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "updated": datetime.now().isoformat(),
            "techniques": {
                key: asdict(perf) for key, perf in self._cache.items()
            }
        }
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)

    def get(self, technique_name: str, effect_type: str) -> TechniquePerformance:
        """Get performance record, creating if not exists."""
        key = self._key(technique_name, effect_type)
        if key not in self._cache:
            self._cache[key] = TechniquePerformance(
                technique_name=technique_name,
                effect_type=effect_type
            )
        return self._cache[key]

    def record_success(
        self,
        technique_name: str,
        effect_type: str,
        final_score: float,
        iterations: int
    ):
        """Record a successful technique use."""
        perf = self.get(technique_name, effect_type)
        perf.success_count += 1
        perf.total_score += final_score
        perf.total_iterations += iterations
        perf.last_used = datetime.now().isoformat()
        self._save()

    def record_failure(
        self,
        technique_name: str,
        effect_type: str,
        final_score: float
    ):
        """Record a failed technique use (didn't pass threshold)."""
        perf = self.get(technique_name, effect_type)
        perf.failure_count += 1
        perf.total_score += final_score
        perf.last_used = datetime.now().isoformat()
        self._save()

    def get_all_for_effect(self, effect_type: str) -> List[TechniquePerformance]:
        """Get all performance records for an effect type."""
        return [
            perf for key, perf in self._cache.items()
            if key.startswith(f"{effect_type}:")
        ]


# =============================================================================
# UCB1 ALGORITHM
# =============================================================================

def calculate_ucb1_score(
    avg_reward: float,
    trials: int,
    total_trials: int,
    exploration_constant: float = 2.0
) -> float:
    """
    Calculate UCB1 score for a technique.

    UCB1 = avg_reward + sqrt(2 * ln(total_trials) / trials)

    Higher score = should be selected.

    Args:
        avg_reward: Average reward (normalized score 0-1)
        trials: Number of times this technique was tried
        total_trials: Total trials across all techniques
        exploration_constant: Controls exploration vs exploitation (default sqrt(2))

    Returns:
        UCB1 score
    """
    if trials == 0:
        return float('inf')  # Untried techniques get infinite score

    if total_trials <= 0:
        return avg_reward

    exploration_term = math.sqrt(exploration_constant * math.log(total_trials) / trials)
    return avg_reward + exploration_term


def select_technique_ucb1(
    effect_type: str,
    techniques: List[str],
    store: TechniquePerformanceStore,
    exploration_constant: float = 2.0
) -> Tuple[str, float, bool]:
    """
    Select a technique using UCB1 algorithm.

    Args:
        effect_type: Type of effect (pyro, liquid, etc.)
        techniques: List of candidate technique names
        store: Performance data store
        exploration_constant: UCB1 exploration parameter

    Returns:
        (technique_name, ucb_score, is_exploration)
    """
    if not techniques:
        return ("", 0.0, False)

    # Get total trials across all techniques
    total_trials = sum(
        store.get(t, effect_type).trials for t in techniques
    )

    # Calculate UCB1 scores
    scores = []
    for technique in techniques:
        perf = store.get(technique, effect_type)

        # Normalize average score to 0-1 (assuming 100-point scale)
        avg_reward = perf.avg_score / 100.0 if perf.trials > 0 else 0.0

        ucb_score = calculate_ucb1_score(
            avg_reward=avg_reward,
            trials=perf.trials,
            total_trials=total_trials,
            exploration_constant=exploration_constant
        )

        is_exploration = perf.trials == 0
        scores.append((ucb_score, technique, is_exploration))

    # Sort by UCB score descending
    scores.sort(reverse=True, key=lambda x: x[0])

    best_score, best_technique, is_exploration = scores[0]
    return (best_technique, best_score, is_exploration)


# =============================================================================
# KEYWORD FILTERING
# =============================================================================

def filter_by_keywords(
    description: str,
    effect_type: str,
    min_matches: int = 0
) -> List[Tuple[str, int]]:
    """
    Filter techniques by keyword relevance to description.

    Args:
        description: User's description of desired effect
        effect_type: Type of effect (pyro, etc.)
        min_matches: Minimum keyword matches required (0 = return all)

    Returns:
        List of (technique_name, match_count) sorted by match count descending
    """
    if effect_type.lower() != "pyro":
        return [(t, 0) for t in list_all_techniques(effect_type)]

    description_lower = description.lower()
    results = []

    for name, technique in PYRO_TECHNIQUES.items():
        match_count = 0
        for keyword in technique.get("keywords", []):
            if keyword.lower() in description_lower:
                match_count += 1
        results.append((name, match_count))

    # Sort by match count descending
    results.sort(key=lambda x: x[1], reverse=True)

    # Filter by minimum matches if specified
    if min_matches > 0:
        results = [(name, count) for name, count in results if count >= min_matches]

    return results


# =============================================================================
# MAIN SELECTION FUNCTION
# =============================================================================

def recommend_technique(
    effect_type: str,
    description: str,
    store: Optional[TechniquePerformanceStore] = None,
    keyword_weight: float = 0.3,
    prefer_untried: bool = True
) -> TechniqueRecommendation:
    """
    Recommend a technique using keyword filtering + UCB1.

    Strategy:
    1. Filter by keyword relevance (if strong matches exist)
    2. Apply UCB1 to filtered candidates
    3. Combine keyword score with UCB1 score

    Args:
        effect_type: Type of effect (pyro, liquid, etc.)
        description: User's description of desired effect
        store: Performance data store (creates default if None)
        keyword_weight: How much to weight keyword matches (0-1)
        prefer_untried: If True, untried techniques get boosted

    Returns:
        TechniqueRecommendation with selected technique and confidence
    """
    if store is None:
        store = TechniquePerformanceStore()

    # Get all techniques for this effect type
    all_techniques = list_all_techniques(effect_type)
    if not all_techniques:
        return TechniqueRecommendation(
            technique_name="",
            confidence=0.0,
            selection_reason="No techniques available for this effect type",
            ucb_score=0.0
        )

    # Filter by keywords
    keyword_results = filter_by_keywords(description, effect_type)

    # Check if we have strong keyword matches (2+ matches)
    strong_matches = [(name, count) for name, count in keyword_results if count >= 2]

    if strong_matches:
        # Use only strong keyword matches as candidates
        candidates = [name for name, _ in strong_matches]
        keyword_matched = True
    else:
        # No strong matches - use all techniques (random exploration)
        candidates = all_techniques
        keyword_matched = False

    # Apply UCB1 to candidates
    best_technique, ucb_score, is_exploration = select_technique_ucb1(
        effect_type=effect_type,
        techniques=candidates,
        store=store
    )

    # Calculate confidence
    # Higher if: strong keyword match + good UCB score + not exploration
    confidence = 0.5  # Base confidence

    if keyword_matched:
        # Boost for keyword matches
        best_keywords = next(
            (count for name, count in keyword_results if name == best_technique),
            0
        )
        confidence += min(best_keywords * 0.1, 0.3)

    if not is_exploration:
        # Boost for tried techniques with good performance
        perf = store.get(best_technique, effect_type)
        if perf.success_rate > 0.5:
            confidence += 0.15
        if perf.avg_score > 70:
            confidence += 0.1

    confidence = min(confidence, 0.95)  # Cap at 95%

    # Build reason string
    if is_exploration:
        reason = f"Exploration: '{best_technique}' has never been tried"
    elif keyword_matched:
        reason = f"Keyword match + good performance: '{best_technique}'"
    else:
        reason = f"UCB1 selected: '{best_technique}' (no keyword matches)"

    # Get alternatives
    alternatives = []
    for name, count in keyword_results[:5]:
        if name != best_technique:
            perf = store.get(name, effect_type)
            alternatives.append({
                "technique": name,
                "keyword_matches": count,
                "trials": perf.trials,
                "success_rate": round(perf.success_rate, 2),
                "avg_score": round(perf.avg_score, 1),
            })

    return TechniqueRecommendation(
        technique_name=best_technique,
        confidence=round(confidence, 2),
        selection_reason=reason,
        ucb_score=round(ucb_score, 3),
        alternatives=alternatives[:3],  # Top 3 alternatives
        keyword_matches=next(
            (count for name, count in keyword_results if name == best_technique),
            0
        ),
        exploration_mode=is_exploration
    )


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("=== Technique Selector Test ===\n")

    # Create a test store
    store = TechniquePerformanceStore()

    # Simulate some historical data
    print("Simulating performance data...")
    store.record_success("rising_mushroom", "pyro", 85.0, 3)
    store.record_success("rising_mushroom", "pyro", 78.0, 4)
    store.record_failure("ground_hugger", "pyro", 45.0)
    store.record_success("aerial_burst", "pyro", 92.0, 2)

    # Test recommendations
    test_descriptions = [
        "A rising mushroom cloud explosion with bright orange flames",
        "Low spreading fire that hugs the ground like napalm",
        "A bright spherical fireball in mid-air",
        "Some kind of explosion",  # Generic - should explore
        "Volcanic eruption with ash clouds",
    ]

    print("\n=== Recommendation Tests ===\n")
    for desc in test_descriptions:
        result = recommend_technique("pyro", desc, store)
        print(f"Description: {desc[:50]}...")
        print(f"  Selected: {result.technique_name}")
        print(f"  Confidence: {result.confidence}")
        print(f"  Reason: {result.selection_reason}")
        print(f"  UCB Score: {result.ucb_score}")
        print(f"  Exploration: {result.exploration_mode}")
        print(f"  Alternatives: {[a['technique'] for a in result.alternatives]}")
        print()

    # Show stored performance data
    print("\n=== Stored Performance Data ===\n")
    for perf in store.get_all_for_effect("pyro"):
        print(f"{perf.technique_name}:")
        print(f"  Trials: {perf.trials}")
        print(f"  Success rate: {perf.success_rate:.1%}")
        print(f"  Avg score: {perf.avg_score:.1f}")
        print(f"  Avg iterations: {perf.avg_iterations:.1f}")
