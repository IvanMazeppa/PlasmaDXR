"""
Budget Tracker - API cost tracking with monthly limits.

Tracks spending across categories (vision, docs, reasoning) to stay
within $20/month budget for GPT-5.2 API calls.

Adapted from blender-librarian/server.py
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

_logger = logging.getLogger(__name__)


# =============================================================================
# GPT-5.2 PRICING (as of Jan 2026)
# =============================================================================

# Token costs
COST_INPUT_1K = 0.00175    # $1.75 per 1M input tokens
COST_OUTPUT_1K = 0.014     # $14 per 1M output tokens

# Image costs (vision)
COST_IMAGE_LOW = 0.00015   # 85 tokens at low detail
COST_IMAGE_HIGH = 0.00045  # ~255 tokens at high detail

# Reasoning costs (extended thinking)
COST_REASONING_1K = 0.028  # $28 per 1M reasoning tokens (2x output)

# Typical operation costs (estimates)
VISION_COST_ESTIMATE = 0.05     # Single image analysis
DOC_COST_ESTIMATE = 0.02        # Doc synthesis query
REASONING_COST_ESTIMATE = 0.10  # Extended thinking operation


# =============================================================================
# BUDGET LIMITS
# =============================================================================

MONTHLY_BUDGET = 20.0  # Total monthly budget

# Category allocations
BUDGET_VISION = 10.0       # Vision analysis (render diagnosis)
BUDGET_DOCS = 6.0          # Doc synthesis (blender-manual queries)
BUDGET_REASONING = 2.0     # Extended thinking operations
BUFFER = 2.0               # Emergency buffer (remaining)


# =============================================================================
# BUDGET STATE
# =============================================================================

@dataclass
class BudgetState:
    """
    Persistent budget tracking state.

    Automatically resets at the start of each month.
    """
    month: str = ""
    vision_spent: float = 0.0
    doc_spent: float = 0.0
    reasoning_spent: float = 0.0
    total_calls: int = 0
    last_updated: str = ""

    # Per-category call counts for analytics
    vision_calls: int = 0
    doc_calls: int = 0
    reasoning_calls: int = 0

    def total_spent(self) -> float:
        """Total spending across all categories."""
        return self.vision_spent + self.doc_spent + self.reasoning_spent

    def remaining(self) -> float:
        """Remaining budget for the month."""
        return max(0.0, MONTHLY_BUDGET - self.total_spent())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BudgetState":
        """Create from dictionary, handling missing fields gracefully."""
        # Handle older state files missing new fields
        defaults = cls()
        for key in asdict(defaults).keys():
            if key not in data:
                data[key] = getattr(defaults, key)
        return cls(**{k: v for k, v in data.items() if k in asdict(defaults)})


# =============================================================================
# BUDGET TRACKER CLASS
# =============================================================================

class BudgetTracker:
    """
    Singleton budget tracker with persistent state.

    Tracks API spending across categories:
    - vision: Image analysis (render diagnosis)
    - docs: Documentation synthesis
    - reasoning: Extended thinking operations

    Usage:
        tracker = get_budget_tracker()
        if tracker.can_afford("vision", 0.05):
            # Make API call
            tracker.record_spend("vision", actual_cost)
    """

    _instance: Optional["BudgetTracker"] = None

    def __new__(cls, budget_file: Optional[Path] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, budget_file: Optional[Path] = None):
        if self._initialized:
            return

        if budget_file is None:
            # Default to orchestrator directory
            budget_file = Path(__file__).parent.parent / "budget_tracker.json"

        self._budget_file = budget_file
        self._state: Optional[BudgetState] = None
        self._initialized = True

    @property
    def state(self) -> BudgetState:
        """Get current budget state, loading/resetting as needed."""
        if self._state is None:
            self._state = self._load_state()
        return self._state

    def _load_state(self) -> BudgetState:
        """Load budget state from file, reset if new month."""
        current_month = datetime.now().strftime("%Y-%m")

        if self._budget_file.exists():
            try:
                data = json.loads(self._budget_file.read_text())
                state = BudgetState.from_dict(data)

                # Reset if new month
                if state.month != current_month:
                    _logger.info(f"New month ({current_month}), resetting budget")
                    state = BudgetState(month=current_month)
            except Exception as e:
                _logger.warning(f"Failed to load budget state: {e}")
                state = BudgetState(month=current_month)
        else:
            state = BudgetState(month=current_month)

        return state

    def _save_state(self) -> None:
        """Persist budget state to file."""
        if self._state is None:
            return

        self._state.last_updated = datetime.now().isoformat()

        try:
            self._budget_file.parent.mkdir(parents=True, exist_ok=True)
            self._budget_file.write_text(
                json.dumps(self._state.to_dict(), indent=2)
            )
        except Exception as e:
            _logger.error(f"Failed to save budget state: {e}")

    def get_category_limit(self, category: str) -> float:
        """Get budget limit for a category."""
        limits = {
            "vision": BUDGET_VISION,
            "docs": BUDGET_DOCS,
            "reasoning": BUDGET_REASONING,
        }
        return limits.get(category, 0.0)

    def get_category_spent(self, category: str) -> float:
        """Get amount spent in a category."""
        state = self.state
        spent = {
            "vision": state.vision_spent,
            "docs": state.doc_spent,
            "reasoning": state.reasoning_spent,
        }
        return spent.get(category, 0.0)

    def can_afford(self, category: str, estimated_cost: float) -> bool:
        """
        Check if we can afford an operation in a category.

        Args:
            category: "vision", "docs", or "reasoning"
            estimated_cost: Estimated cost of the operation

        Returns:
            True if within budget
        """
        limit = self.get_category_limit(category)
        spent = self.get_category_spent(category)
        return spent + estimated_cost <= limit

    def can_afford_vision(self, estimated_cost: float = VISION_COST_ESTIMATE) -> bool:
        """Check if we have vision budget remaining."""
        return self.can_afford("vision", estimated_cost)

    def can_afford_docs(self, estimated_cost: float = DOC_COST_ESTIMATE) -> bool:
        """Check if we have doc synthesis budget remaining."""
        return self.can_afford("docs", estimated_cost)

    def can_afford_reasoning(self, estimated_cost: float = REASONING_COST_ESTIMATE) -> bool:
        """Check if we have reasoning budget remaining."""
        return self.can_afford("reasoning", estimated_cost)

    def can_afford_evaluation(self) -> bool:
        """Check if we can afford a typical evaluation operation (vision-based)."""
        return self.can_afford_vision()

    def record_spend(self, category: str, cost: float) -> None:
        """
        Record API spending.

        Args:
            category: "vision", "docs", or "reasoning"
            cost: Actual cost incurred
        """
        state = self.state

        if category == "vision":
            state.vision_spent += cost
            state.vision_calls += 1
        elif category == "docs":
            state.doc_spent += cost
            state.doc_calls += 1
        elif category == "reasoning":
            state.reasoning_spent += cost
            state.reasoning_calls += 1
        else:
            _logger.warning(f"Unknown budget category: {category}")
            return

        state.total_calls += 1
        self._save_state()

        _logger.info(
            f"Budget: {category} +${cost:.4f} "
            f"(${self.get_category_spent(category):.2f}/{self.get_category_limit(category):.2f})"
        )

    @property
    def monthly_limit(self) -> float:
        """Get monthly budget limit."""
        return MONTHLY_BUDGET

    def get_spent(self) -> float:
        """Get total amount spent this month."""
        return self.state.total_spent()

    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive budget status.

        Returns dict with:
        - month: Current tracking month
        - categories: Per-category spent/limit/remaining
        - total_spent: Total across all categories
        - total_remaining: Budget remaining
        - total_calls: Number of API calls
        - can_afford: Dict of category -> bool
        """
        state = self.state

        categories = {}
        for cat in ["vision", "docs", "reasoning"]:
            limit = self.get_category_limit(cat)
            spent = self.get_category_spent(cat)
            categories[cat] = {
                "spent": round(spent, 4),
                "limit": limit,
                "remaining": round(max(0, limit - spent), 4),
                "calls": getattr(state, f"{cat}_calls", 0),
            }

        return {
            "month": state.month,
            "categories": categories,
            "total_spent": round(state.total_spent(), 4),
            "total_remaining": round(state.remaining(), 4),
            "total_calls": state.total_calls,
            "can_afford": {
                "vision": self.can_afford_vision(),
                "docs": self.can_afford_docs(),
                "reasoning": self.can_afford_reasoning(),
            },
            "last_updated": state.last_updated,
        }

    def reset(self) -> None:
        """Force reset budget state (for testing)."""
        self._state = BudgetState(month=datetime.now().strftime("%Y-%m"))
        self._save_state()


# =============================================================================
# MODULE-LEVEL SINGLETON ACCESSOR
# =============================================================================

_budget_tracker: Optional[BudgetTracker] = None


def get_budget_tracker(budget_file: Optional[Path] = None) -> BudgetTracker:
    """Get the global budget tracker singleton."""
    global _budget_tracker
    if _budget_tracker is None:
        _budget_tracker = BudgetTracker(budget_file)
    return _budget_tracker


# =============================================================================
# COST ESTIMATION HELPERS
# =============================================================================

def estimate_token_cost(input_tokens: int, output_tokens: int) -> float:
    """Estimate cost for a text-only API call."""
    return (input_tokens / 1000 * COST_INPUT_1K) + (output_tokens / 1000 * COST_OUTPUT_1K)


def estimate_vision_cost(
    input_tokens: int = 500,
    output_tokens: int = 500,
    image_count: int = 1,
    high_detail: bool = False
) -> float:
    """Estimate cost for a vision API call."""
    text_cost = estimate_token_cost(input_tokens, output_tokens)
    image_cost = image_count * (COST_IMAGE_HIGH if high_detail else COST_IMAGE_LOW)
    return text_cost + image_cost


def estimate_reasoning_cost(
    input_tokens: int = 1000,
    output_tokens: int = 500,
    reasoning_tokens: int = 2000
) -> float:
    """Estimate cost for an extended thinking API call."""
    text_cost = estimate_token_cost(input_tokens, output_tokens)
    reasoning_cost = reasoning_tokens / 1000 * COST_REASONING_1K
    return text_cost + reasoning_cost
