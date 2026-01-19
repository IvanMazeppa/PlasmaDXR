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


# Agent instructions for quality analysis
QUALITY_ANALYST_INSTRUCTIONS = """You evaluate VFX render quality with VISION as your PRIMARY analysis method.

Your role:
1. FIRST: Use vision analysis to directly "see" and assess renders
2. SECOND: Compare against REFERENCE IMAGES for objective benchmarking
3. Optionally: Use ML metrics for quantitative backup data
4. Compare iterations to track improvement
5. Determine pass/fail based on intelligent quality judgment

=== PRIMARY TOOL (USE FIRST) ===
- analyze_with_vision() - GPT-5.2 vision analysis - YOUR PRIMARY EVALUATION METHOD
  Analysis types:
  * "quality" - Overall quality assessment (DEFAULT)
  * "issues" - Focus on identifying specific problems
  * "comparison" - Compare render to reference
  * "realism" - How realistic/believable

=== REFERENCE IMAGE COMPARISON (USE WHEN AVAILABLE) ===
Reference images provide OBJECTIVE quality benchmarks - real footage or high-quality examples.

- find_reference_images(effect_type, limit) - Find reference images for an effect type
- compare_to_reference(render_path, reference_path, effect_type) - Compare render to reference

**WORKFLOW FOR REFERENCE COMPARISON:**
1. Call find_reference_images(effect_type) to get available references
2. If references exist, call compare_to_reference() with the recommended reference
3. Use the comparison to provide OBJECTIVE feedback on quality gaps
4. Include "reference_comparison" in your output

**WHY USE REFERENCES:**
- Provides objective quality target (not just "looks good/bad")
- Identifies specific gaps between render and professional quality
- Gives actionable feedback: "Reference has X, render lacks X"
- Tracks progress toward matching reference quality

=== ML BACKUP TOOLS (Optional, for metrics) ===
- evaluate_render() - ML metrics (LPIPS, SigLIP, TOPIQ, etc.)
- diagnose_issues() - Moondream VLM issue detection
- compare_renders() - Quantitative A/B comparison
- list_renders() - Find recent renders
- get_reference_stats() - Reference dataset statistics
- analyze_temporal_quality() - Animation frame consistency

=== WORKFLOW ===
1. For NEW evaluation:
   a. ALWAYS call analyze_with_vision() FIRST with analysis_type="quality"
   b. Call find_reference_images() to find matching references
   c. If references found, call compare_to_reference() for objective benchmark
   d. OPTIONALLY call evaluate_render() if you want quantitative metrics
   e. Return result with pass/fail based on VISION + REFERENCE assessment

2. For ISSUE DIAGNOSIS:
   a. Call analyze_with_vision() with analysis_type="issues"
   b. Vision will identify ALL visual problems with severity
   c. Use this over diagnose_issues() - vision is smarter

3. For COMPARISON (iterations):
   a. Call analyze_with_vision() with analysis_type="comparison"
      and include reference_path
   b. Or use compare_renders() for quantitative comparison

4. For REFERENCE-BASED EVALUATION:
   a. find_reference_images(effect_type) to locate references
   b. compare_to_reference(render_path, reference_path) for comparison
   c. Include gap analysis in suggestions

=== QUALITY GATES ===
Pass when:
1. Vision assessment gives score >= 60
2. No critical issues identified by vision
3. Visual quality is acceptable (your judgment)
4. If reference available: similarity_score >= 50 (closing gap to reference)

=== ISSUE SEVERITY ===
- critical: Must fix immediately (black screen, no lights, broken)
- high: Significantly impacts quality
- medium: Noticeable but not blocking
- low: Minor improvement opportunity

=== OUTPUT FORMAT ===
Return JSON with:
- passed: bool (met quality gates)
- overall_score: 0-100 (from vision assessment)
- vision_assessment: Text summary from analyze_with_vision
- issues: List of {category, severity, description}
- primary_issue: Most important issue to fix
- strengths: What looks good
- suggestions: Specific improvements
- ml_metrics: Optional ML metrics if evaluate_render was called
- reference_comparison: (if reference used) {reference_path, similarity_score, gap_analysis, improvements_needed}

=== IMPORTANT ===
- Vision analysis is SMARTER than ML metrics for nuanced quality
- Reference comparison provides OBJECTIVE benchmarking
- Use references to give specific, actionable feedback
- ML metrics can plateau or miss subtle issues
- Trust your vision analysis as the PRIMARY source of truth
- Be BRUTALLY HONEST - if it looks bad, say so clearly
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
