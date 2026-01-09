"""
Quality Analyst Agent using OpenAI Agents SDK.

Specialized agent for evaluating VFX render quality using ML metrics.
Wraps the asset-evaluator MCP server's consolidated v2 API.

Key capabilities:
- Multi-metric quality evaluation (LPIPS, SigLIP, TOPIQ, DINOv2)
- VLM-powered issue diagnosis
- Iteration comparison for improvement tracking
- Temporal quality analysis for animations
"""

from __future__ import annotations

import os
from typing import Optional, TYPE_CHECKING

from agents import Agent, ModelSettings

if TYPE_CHECKING:
    from tools.asset_evaluator_tools import (
        evaluate_render,
        compare_renders,
        diagnose_issues,
        get_reference_stats,
        list_renders,
        analyze_temporal_quality,
    )

# Import tools at runtime
from tools.asset_evaluator_tools import (
    evaluate_render,
    compare_renders,
    diagnose_issues,
    get_reference_stats,
    list_renders,
    analyze_temporal_quality,
)


# Agent instructions for quality analysis
QUALITY_ANALYST_INSTRUCTIONS = """You evaluate VFX render quality using ML metrics.

Your role:
1. Evaluate renders using the consolidated v2 API (not legacy tools)
2. Diagnose specific issues using VLM analysis
3. Compare iterations to track improvement
4. Determine pass/fail based on quality gates

TOOLS (v2 Consolidated API - USE THESE):
- evaluate_render() - Primary evaluation (LPIPS, SigLIP, TOPIQ, etc.)
- diagnose_issues() - VLM-powered issue detection
- compare_renders() - A/B comparison between iterations
- list_renders() - Find recent renders
- get_reference_stats() - Get reference dataset statistics
- analyze_temporal_quality() - Animation frame consistency

EVALUATION PROFILES:
- "quick": LPIPS + SigLIP only (~2 seconds) - Use for fast iteration
- "standard": + TOPIQ, feature_cv (~10 seconds) - Normal evaluation
- "comprehensive": + DINOv2, VLM diagnosis (~30 seconds) - When stuck

QUALITY GATES:
Pass when ALL conditions met:
1. overall_score >= 60
2. No critical issues (ZERO lights, BLACK screen, etc.)
3. If reference provided: LPIPS similarity acceptable

METRICS INTERPRETATION:
- LPIPS: Lower = more similar to reference (< 0.35 is good)
- SigLIP: Higher = better semantic match (> 0.5 is good)
- TOPIQ: Higher = better image quality (> 50 is good)
- Feature CV: Higher = more texture variety (> 10 is good)
- DINOv2: Higher = better structural similarity

ISSUE SEVERITY:
- critical: Must fix immediately (e.g., "ZERO LIGHTS ACTIVE")
- high: Significantly impacts quality
- medium: Noticeable but not blocking
- low: Minor improvement opportunity

WORKFLOW:
1. For NEW evaluation:
   a. Call evaluate_render() with appropriate profile
   b. If score < 60 or critical issues, call diagnose_issues()
   c. Return structured result with pass/fail and issues

2. For COMPARISON (iterations):
   a. Call compare_renders() with both render paths
   b. Report which is better and why
   c. Note improvements and regressions

OUTPUT FORMAT:
Return JSON with:
- passed: bool (met quality gates)
- overall_score: 0-100
- metric_scores: {lpips, siglip, topiq, feature_cv, structural_dino}
- issues: List of {category, severity, description}
- primary_issue: Most important issue to fix
- recommendations: Specific parameter changes
- evaluation_profile: Which profile was used
"""


class QualityAnalystAgent:
    """
    Quality Analyst Agent for VFX render evaluation.

    Uses gpt-4.1-mini for fast metric interpretation and issue categorization.
    """

    def __init__(self, model: str = "gpt-4.1-mini"):
        """
        Initialize the quality analyst agent.

        Args:
            model: OpenAI model to use (default: gpt-4.1-mini)
        """
        self.model = os.getenv("QUALITY_ANALYST_MODEL", model)
        self._agent: Optional[Agent] = None

    def initialize(self, custom_instructions: str = "") -> None:
        """
        Initialize the agent with tools.

        Args:
            custom_instructions: Additional context (e.g., quality thresholds)
        """
        instructions = QUALITY_ANALYST_INSTRUCTIONS
        if custom_instructions:
            instructions = instructions + "\n\n" + custom_instructions

        self._agent = Agent(
            name="Quality Analyst",
            instructions=instructions,
            model=self.model,
            model_settings=ModelSettings(
                # No reasoning needed for evaluation (gpt-4.1-mini)
            ),
            tools=[
                evaluate_render,
                compare_renders,
                diagnose_issues,
                get_reference_stats,
                list_renders,
                analyze_temporal_quality,
            ],
        )

    @property
    def agent(self) -> Agent:
        """Get the underlying Agent instance for handoffs."""
        if not self._agent:
            raise RuntimeError("QualityAnalystAgent not initialized. Call initialize() first.")
        return self._agent


def create_quality_analyst(custom_instructions: str = "") -> Agent:
    """
    Factory function to create and initialize a quality analyst agent.

    Args:
        custom_instructions: Additional context to append

    Returns:
        Initialized Agent instance ready for use
    """
    analyst = QualityAnalystAgent()
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
