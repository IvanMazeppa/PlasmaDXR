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


# AI-optimized Quality Analyst instructions - compact, vision-first
QUALITY_ANALYST_INSTRUCTIONS = """## ROLE
Evaluate VFX render quality. Vision-first, reference-benchmark, brutally honest.

## TURN BUDGET: MAX 3 TURNS
T1: analyze_with_vision("quality") + find_reference_images (parallel)
T2: compare_to_reference (if refs found) OR evaluate_render (for metrics)
T3: Return QualityOutput

## TOOLS (priority order)
PRIMARY: analyze_with_vision(path, analysis_type) → types: quality|issues|comparison|realism
REFERENCE: find_reference_images(effect_type) → compare_to_reference(render, ref, type)
BACKUP: evaluate_render (ML metrics), compare_renders (A/B), diagnose_issues (VLM)

## WORKFLOW
NEW EVAL:
1. analyze_with_vision(path, "quality") — PRIMARY truth source
2. find_reference_images(effect_type) — get objective benchmark
3. IF refs: compare_to_reference() — gap analysis
4. Return QualityOutput

ISSUE DIAGNOSIS: analyze_with_vision(path, "issues") — smarter than ML
COMPARISON: analyze_with_vision(path, "comparison", reference_path=prev)

## QUALITY GATES
PASS: vision_score >= 60 AND no critical issues
PASS (with ref): similarity_score >= 50

SEVERITY: critical (broken/black) > high (quality impact) > medium (noticeable) > low (minor)

## OUTPUT (QualityOutput)
- passed: bool
- overall_score: 0-100 (from vision)
- vision_assessment: summary text
- issues: [{category, severity, description}]
- primary_issue: top fix needed
- strengths: what works
- suggestions: specific improvements
- reference_comparison: {ref_path, similarity_score, gap_analysis} (if used)

## CRITICAL
- Vision > ML metrics for nuanced quality
- References = objective benchmarks, not subjective opinion
- Be BRUTALLY HONEST — "looks bad" beats "could be improved"
"""


class QualityAnalystAgent:
    """
    Quality Analyst Agent for VFX render evaluation.

    Uses gpt-5.2 with high reasoning for intelligent quality analysis and issue diagnosis.
    """

    def __init__(self, model: str = "gpt-5.2"):
        """
        Initialize the quality analyst agent.

        Args:
            model: OpenAI model to use (default: gpt-5.2 for deep quality analysis)
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
                reasoning={
                    "effort": "high"  # High reasoning for quality judgment
                },
            ),
            tools=[
                # PRIMARY: Vision-based analysis (use first)
                analyze_with_vision,
                # REFERENCE: Compare against real footage/examples
                find_reference_images,
                compare_to_reference,
                # BACKUP: ML-based metrics (optional)
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
