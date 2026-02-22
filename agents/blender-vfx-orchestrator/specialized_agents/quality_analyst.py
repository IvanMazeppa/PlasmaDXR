"""
Quality Analyst Agent using OpenAI Agents SDK.

Specialized agent for evaluating VFX render quality using ML metrics.
Wraps the asset-evaluator MCP server's consolidated v2 API.

Key capabilities:
- Multi-metric quality evaluation (LPIPS, SigLIP, TOPIQ, DINOv2)
- VLM-powered issue diagnosis
- Iteration comparison for improvement tracking
- Temporal quality analysis for animations
- Physics anomaly observation for self-learning system

SELF-LEARNING: This agent can observe physics anomalies and feed them into
the learning system. Instead of hardcoding physics expectations, we observe
what happens and let the system learn from experimentation.
"""

from __future__ import annotations

import os
from typing import Optional, TYPE_CHECKING

from agents import Agent, ModelSettings

if TYPE_CHECKING:
    from tools.asset_evaluator_tools import (
        analyze_with_vision,
        evaluate_render,
        compare_renders,
        diagnose_issues,
        get_reference_stats,
        list_renders,
        analyze_temporal_quality,
        find_reference_images,
        compare_to_reference,
    )

# Import tools at runtime
from tools.asset_evaluator_tools import (
    analyze_with_vision,
    evaluate_render,
    compare_renders,
    diagnose_issues,
    get_reference_stats,
    list_renders,
    analyze_temporal_quality,
    # Reference image comparison tools
    find_reference_images,
    compare_to_reference,
)

# Import physics observation tools for self-learning
from tools.physics_observation_tools import (
    observe_physics_anomaly,
    get_physics_patterns,
)

# Import dynamic instructions for self-learning
from tools.dynamic_instructions import (
    dynamic_quality_analyst_instructions,
    get_quality_analyst_instructions_static,
    QUALITY_ANALYST_BASE_INSTRUCTIONS,
)

# Phase 2A-1: Conditional tool visibility
from utils.tool_visibility import budget_allows_vision


# DEPRECATED: Hardcoded instructions replaced by dynamic_instructions.py
# Keeping for reference only - the actual instructions come from QUALITY_ANALYST_BASE_INSTRUCTIONS
QUALITY_ANALYST_INSTRUCTIONS_DEPRECATED = """
[DEPRECATED - See tools/dynamic_instructions.py for current instructions]
"""


class QualityAnalystAgent:
    """
    Quality Analyst Agent for VFX render evaluation.

    Uses gpt-5.2 with high reasoning for intelligent quality analysis and issue diagnosis.

    SELF-LEARNING: This agent can observe physics anomalies and feed them into
    the learning system. Uses dynamic instructions that include known physics
    behaviors from past experiments.
    """

    def __init__(self, model: str = "gpt-5.2", use_dynamic_instructions: bool = True):
        """
        Initialize the quality analyst agent.

        Args:
            model: OpenAI model to use (default: gpt-5.2 for deep quality analysis)
            use_dynamic_instructions: If True, use dynamic instructions that include
                                      known physics behaviors from past experiments.
        """
        self.model = os.getenv("QUALITY_ANALYST_MODEL", model)
        self.use_dynamic_instructions = use_dynamic_instructions
        self._agent: Optional[Agent] = None

    def initialize(self, custom_instructions: str = "") -> None:
        """
        Initialize the agent with tools.

        Args:
            custom_instructions: Additional context (e.g., quality thresholds)
        """
        # Choose instruction source
        if self.use_dynamic_instructions:
            # Dynamic instructions: function that generates instructions at runtime
            instructions = dynamic_quality_analyst_instructions
        else:
            # Static fallback
            instructions = get_quality_analyst_instructions_static()
            if custom_instructions:
                instructions = instructions + "\n\n" + custom_instructions

        # Phase 2A-1: Apply is_enabled callbacks to expensive vision tools.
        # When budget is exhausted, these tools are hidden from the LLM entirely.
        # Cheap tools (find_reference_images, observe_physics_anomaly, get_physics_patterns,
        # get_reference_stats, list_renders) remain always visible.
        analyze_with_vision.is_enabled = budget_allows_vision
        evaluate_render.is_enabled = budget_allows_vision
        compare_renders.is_enabled = budget_allows_vision
        compare_to_reference.is_enabled = budget_allows_vision
        diagnose_issues.is_enabled = budget_allows_vision
        analyze_temporal_quality.is_enabled = budget_allows_vision

        self._agent = Agent(
            name="Quality Analyst",
            instructions=instructions,  # Can be function OR string
            model=self.model,
            model_settings=ModelSettings(
                reasoning={
                    "effort": "high"  # High reasoning for quality judgment
                },
            ),
            tools=[
                # PRIMARY: Vision-based analysis (use first) -- budget-gated
                analyze_with_vision,
                # REFERENCE: Compare against real footage/examples
                find_reference_images,            # Cheap: filesystem only
                compare_to_reference,             # Budget-gated: uses vision API
                # PHYSICS OBSERVATION: Feed anomalies into learning system
                observe_physics_anomaly,          # Cheap: in-memory
                get_physics_patterns,             # Cheap: in-memory
                # BACKUP: ML-based metrics (optional) -- budget-gated where applicable
                evaluate_render,
                compare_renders,
                diagnose_issues,
                get_reference_stats,              # Cheap: filesystem only
                list_renders,                     # Cheap: filesystem only
                analyze_temporal_quality,
            ],
        )

    @property
    def agent(self) -> Agent:
        """Get the underlying Agent instance for handoffs."""
        if not self._agent:
            raise RuntimeError("QualityAnalystAgent not initialized. Call initialize() first.")
        return self._agent


def create_quality_analyst(
    custom_instructions: str = "",
    use_dynamic_instructions: bool = True
) -> Agent:
    """
    Factory function to create and initialize a quality analyst agent.

    Args:
        custom_instructions: Additional context to append
        use_dynamic_instructions: If True (default), use dynamic instructions that
                                  include known physics behaviors from past experiments.

    Returns:
        Initialized Agent instance ready for use
    """
    analyst = QualityAnalystAgent(use_dynamic_instructions=use_dynamic_instructions)
    analyst.initialize(custom_instructions=custom_instructions)
    return analyst.agent


# Convenience alias
QualityAnalystAgent = QualityAnalystAgent


# For testing
if __name__ == "__main__":
    import asyncio
    from agents import Runner

    async def test():
        print("Testing QualityAnalystAgent...")
        print("-" * 60)

        agent = create_quality_analyst()

        # Test evaluation query
        result = await Runner.run(
            agent,
            "Evaluate the render at build/vdb_output/explosion_v1/render_0030.png with standard profile"
        )

        print(f"Response: {result.final_output}")

    asyncio.run(test())
