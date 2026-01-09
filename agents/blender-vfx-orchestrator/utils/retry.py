"""
Retry utilities with exponential backoff and reasoning settings.

Provides:
- retry_with_backoff: Async retry with exponential backoff
- estimate_query_complexity: Analyze query to determine reasoning effort
- get_reasoning_settings: Generate ModelSettings for reasoning level
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Dict, Optional, TypeVar

from agents import ModelSettings
from openai.types.shared import Reasoning


# =============================================================================
# TYPE VARIABLES AND CONFIGURATION
# =============================================================================

T = TypeVar('T')

# Retry configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_BASE = 2.0  # seconds
DEFAULT_BACKOFF_MAX = 30.0  # max wait between retries

# Complexity indicators for query analysis
COMPLEXITY_INDICATORS = {
    # High complexity - need deep reasoning
    "high": [
        "why does", "why is", "debug", "diagnose", "analyze", "compare",
        "investigate", "root cause", "broken", "not working", "failing",
        "optimize", "improve", "refactor", "architecture", "design",
        "multiple", "several", "combination", "together with",
        "tradeoff", "trade-off", "best approach", "recommend",
    ],
    # Medium complexity - moderate reasoning
    "medium": [
        "how do i", "how to", "what is the", "explain", "difference between",
        "parameter for", "configure", "setup", "integrate", "implement",
        "example", "tutorial", "guide", "workflow",
    ],
    # Low complexity - quick lookup
    "low": [
        "what is", "where is", "list", "show", "get", "find",
        "default", "range", "value", "path", "location",
    ],
}


# =============================================================================
# RETRY WITH BACKOFF
# =============================================================================

async def retry_with_backoff(
    func: Callable[[], T],
    max_retries: int = DEFAULT_MAX_RETRIES,
    backoff_base: float = DEFAULT_BACKOFF_BASE,
    backoff_max: float = DEFAULT_BACKOFF_MAX,
    retryable_exceptions: tuple = (Exception,),
    operation_name: str = "operation"
) -> T:
    """
    Retry an async operation with exponential backoff.

    Args:
        func: Async callable to retry
        max_retries: Maximum number of retry attempts
        backoff_base: Base for exponential backoff (seconds)
        backoff_max: Maximum backoff wait time (seconds)
        retryable_exceptions: Tuple of exception types to retry on
        operation_name: Name for logging purposes

    Returns:
        Result of successful func() call

    Raises:
        Last exception if all retries exhausted

    Example:
        result = await retry_with_backoff(
            lambda: api_call(),
            max_retries=3,
            operation_name="API call"
        )
    """
    logger = logging.getLogger("vfx_orchestrator")
    last_exception = None

    for attempt in range(max_retries + 1):
        try:
            return await func()
        except retryable_exceptions as e:
            last_exception = e

            if attempt == max_retries:
                logger.error(f"{operation_name} failed after {max_retries + 1} attempts: {e}")
                raise

            # Calculate backoff with exponential increase
            backoff = min(backoff_base * (2 ** attempt), backoff_max)
            logger.warning(
                f"{operation_name} failed (attempt {attempt + 1}/{max_retries + 1}): {e}. "
                f"Retrying in {backoff:.1f}s..."
            )
            await asyncio.sleep(backoff)

    # Should never reach here, but just in case
    raise last_exception if last_exception else RuntimeError("Unexpected retry state")


# =============================================================================
# QUERY COMPLEXITY ESTIMATION
# =============================================================================

def estimate_query_complexity(query: str, context: Optional[Dict[str, Any]] = None) -> str:
    """
    Estimate the complexity of a query to determine reasoning effort.

    Uses heuristics based on:
    - Keywords indicating high/medium/low complexity tasks
    - Query length
    - Context complexity (number of issues, parameters, etc.)

    Args:
        query: The user's query string
        context: Optional context dict (presence of certain keys increases complexity)

    Returns:
        Reasoning effort level: "low", "medium", or "high"

    Example:
        effort = estimate_query_complexity(
            "Why is my explosion not showing smoke?",
            {"known_issues": ["color_too_cool", "density_low"]}
        )
        # Returns: "high"
    """
    query_lower = query.lower()
    query_length = len(query.split())

    # Start with base score
    score = 0

    # Check complexity indicators
    for indicator in COMPLEXITY_INDICATORS["high"]:
        if indicator in query_lower:
            score += 3

    for indicator in COMPLEXITY_INDICATORS["medium"]:
        if indicator in query_lower:
            score += 2

    for indicator in COMPLEXITY_INDICATORS["low"]:
        if indicator in query_lower:
            score += 1

    # Query length factor
    if query_length > 50:
        score += 3
    elif query_length > 25:
        score += 2
    elif query_length > 10:
        score += 1

    # Context complexity factors
    if context:
        # Multiple issues = more complex
        issues = context.get("known_issues", context.get("issues", []))
        if isinstance(issues, list) and len(issues) > 2:
            score += 3
        elif isinstance(issues, list) and len(issues) > 0:
            score += 1

        # Vision analysis needed = more complex
        if context.get("render_path") or context.get("reference_path"):
            score += 2

        # Multiple parameters = more complex
        params = context.get("current_params", {})
        if isinstance(params, dict) and len(params) > 5:
            score += 2

    # Map score to effort level
    if score >= 8:
        return "high"
    elif score >= 4:
        return "medium"
    else:
        return "low"


# =============================================================================
# REASONING SETTINGS
# =============================================================================

def get_reasoning_settings(effort: str) -> ModelSettings:
    """
    Get ModelSettings with appropriate reasoning effort.

    Used with OpenAI Agents SDK to control model reasoning depth.
    Higher effort = more thorough but slower/costlier responses.

    Args:
        effort: "low", "medium", or "high"

    Returns:
        ModelSettings configured for the effort level

    Example:
        settings = get_reasoning_settings("high")
        # Use with Agent: Agent(..., model_settings=settings)
    """
    return ModelSettings(
        reasoning=Reasoning(effort=effort),
        verbosity="low"
    )
