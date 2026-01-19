"""
Learning Agent using OpenAI Agents SDK.

Specialized agent for experiment tracking and accumulated learning.
Maintains a knowledge base of what works and what doesn't.

Key capabilities:
- Record experiment outcomes (success and failure)
- Query knowledge base for relevant learnings
- Suggest experiments based on past success
- Warn about known gotchas before changes
"""

from __future__ import annotations

import os
from typing import Optional, TYPE_CHECKING

from agents import Agent, ModelSettings

if TYPE_CHECKING:
    from tools.experiment_tracker_tools import (
        start_experiment_session,
        end_experiment_session,
        get_session_report,
        record_baseline,
        record_experiment_result,
        get_warnings_before_change,
        suggest_experiments,
        query_knowledge_base,
        get_parameter_knowledge,
        add_manual_learning,
        get_experiment_statistics,
    )

# Import tools at runtime
from tools.experiment_tracker_tools import (
    start_experiment_session,
    end_experiment_session,
    get_session_report,
    record_baseline,
    record_experiment_result,
    get_warnings_before_change,
    suggest_experiments,
    query_knowledge_base,
    get_parameter_knowledge,
    add_manual_learning,
    get_experiment_statistics,
)

# Import knowledge distillation tools for automatic pattern extraction
from tools.knowledge_distillation_tools import (
    extract_successful_pattern,
    analyze_script_for_patterns,
)

# Import code pattern tools for storing reusable patterns
from tools.code_pattern_tools import (
    record_code_pattern,
    search_code_patterns,
    report_pattern_outcome,
)


# Agent instructions for learning and experiment tracking
LEARNING_AGENT_INSTRUCTIONS = """You maintain the learning system for VFX generation.

Your role:
1. Record ALL experiment outcomes (success AND failure)
2. Check knowledge base before suggesting changes
3. Warn about known gotchas
4. Suggest experiments based on accumulated learning
5. AUTOMATICALLY extract and store successful patterns for reuse

BEFORE MAKING CHANGES:
1. ALWAYS call get_warnings_before_change(parameter, change_type)
2. Check query_knowledge_base(issue) for similar past issues
3. Get get_parameter_knowledge(parameter) for accumulated wisdom
4. Call search_code_patterns(issue) to find reusable fixes

AFTER EXPERIMENTS:
1. Call record_experiment_result() with full details
2. Extract learnings for future sessions
3. Add manual learnings for discovered gotchas via add_manual_learning()
4. **IF SUCCESSFUL (score_delta >= 5): AUTO-EXTRACT PATTERN** (see below)

## AUTOMATIC PATTERN EXTRACTION (CRITICAL FOR SELF-IMPROVEMENT)

When an experiment succeeds (score improves by >= 5 points), you MUST:

### Step 1: Extract the Pattern
Call extract_successful_pattern() with:
- script_path: Path to the successful script
- issue_type: What problem was fixed (e.g., "too_dark", "thin_smoke")
- parameters_changed: Dict of param changes {param: {from: x, to: y}}
- score_improvement: How much the score improved

### Step 2: Store in Code Pattern Library
Call record_code_pattern() with:
- pattern_name: Descriptive name (e.g., "fix_dim_flames_increase_emission")
- issue_type: Problem category
- code_snippet: The key code changes that fixed it
- parameters: Dict of effective parameters
- effect_types: Which effects this applies to (e.g., ["explosion", "fire"])
- success_rate: Initial 1.0 (first success)
- context: When to apply this pattern

### Step 3: Update Statistics
The pattern library tracks:
- How often this pattern works
- Which effect types it works for
- Average improvement it provides

## PATTERN REUSE WORKFLOW

When facing a known issue:
1. search_code_patterns(issue_type) - Find existing fixes
2. If pattern found with high success_rate:
   - Recommend using that pattern first
   - Skip experimentation for known solutions
3. After applying a pattern, call report_pattern_outcome():
   - success: Did it work this time?
   - score_delta: How much did it improve?
   - This updates the pattern's success statistics

## KNOWLEDGE DISTILLATION

Use analyze_script_for_patterns() to:
- Analyze successful scripts for reusable techniques
- Identify parameter combinations that work well together
- Extract generalizable rules from specific fixes

RECORDING EXPERIMENTS:
Every change should be recorded with:
- hypothesis: What you were testing
- issue_addressed: The problem being fixed
- result_params: Parameters after change
- result_scores: Evaluation scores after change
- success: Did it improve the target metric?
- observed_effects: What changed (good and bad)
- learnings: What we learned
- warnings: Gotchas discovered

SUGGESTING EXPERIMENTS:
When asked for suggestions:
1. FIRST: search_code_patterns(issue) - Check for known fixes
2. If no pattern found: suggest_experiments(issue, current_params)
3. Check for warnings on suggested changes
4. Prioritize experiments with higher confidence
5. Avoid experiments that have failed before

SESSION MANAGEMENT:
- start_experiment_session() at the beginning of asset creation
- record_baseline() to capture starting state
- end_experiment_session() when complete with final status

KNOWLEDGE BASE STRUCTURE:
- Rules: "Always adjust X when changing Y"
- Warnings: "Changing X without Y causes Z"
- Statistics: Success rate, average improvement for each parameter
- Context: When rules apply (effect_type, issue_type)
- Code Patterns: Reusable fixes with tracked success rates

OUTPUT FORMAT:
When suggesting experiments, return:
- pattern_matches: Any known patterns that might help
- experiments: List of suggested experiments with confidence
- warnings: Known gotchas for each suggestion
- rationale: Why these experiments might help
- avoid: Experiments known to fail for this issue

When recording results, return:
- recorded: bool
- learnings_extracted: List of new learnings
- knowledge_updated: What was added to knowledge base
- pattern_extracted: bool (if success, was pattern stored?)
- pattern_name: Name of stored pattern (if applicable)
"""


class LearningAgent:
    """
    Learning Agent for experiment tracking and knowledge accumulation.

    Uses gpt-5.2 with high reasoning for intelligent learning and pattern recognition.
    """

    def __init__(self, model: str = "gpt-5.2"):
        """
        Initialize the learning agent.

        Args:
            model: OpenAI model to use (default: gpt-5.2 for deep learning insights)
        """
        self.model = os.getenv("LEARNING_AGENT_MODEL", model)
        self._agent: Optional[Agent] = None

    def initialize(self, custom_instructions: str = "") -> None:
        """
        Initialize the agent with tools.

        Args:
            custom_instructions: Additional context (e.g., session ID)
        """
        instructions = LEARNING_AGENT_INSTRUCTIONS
        if custom_instructions:
            instructions = instructions + "\n\n" + custom_instructions

        self._agent = Agent(
            name="Learning Agent",
            instructions=instructions,
            model=self.model,
            model_settings=ModelSettings(
                reasoning={
                    "effort": "high"  # High reasoning for pattern recognition and learning
                },
            ),
            tools=[
                # Experiment tracking tools
                start_experiment_session,
                end_experiment_session,
                get_session_report,
                record_baseline,
                record_experiment_result,
                get_warnings_before_change,
                suggest_experiments,
                query_knowledge_base,
                get_parameter_knowledge,
                add_manual_learning,
                get_experiment_statistics,
                # Knowledge distillation tools (auto-extract patterns)
                extract_successful_pattern,
                analyze_script_for_patterns,
                # Code pattern tools (store/retrieve reusable patterns)
                record_code_pattern,
                search_code_patterns,
                report_pattern_outcome,
            ],
        )

    @property
    def agent(self) -> Agent:
        """Get the underlying Agent instance for handoffs."""
        if not self._agent:
            raise RuntimeError("LearningAgent not initialized. Call initialize() first.")
        return self._agent


def create_learning_agent(custom_instructions: str = "") -> Agent:
    """
    Factory function to create and initialize a learning agent.

    Args:
        custom_instructions: Additional context to append

    Returns:
        Initialized Agent instance ready for use
    """
    agent = LearningAgent()
    agent.initialize(custom_instructions=custom_instructions)
    return agent.agent


# Convenience alias
LearningAgent = LearningAgent


# For testing
if __name__ == "__main__":
    import asyncio
    from agents import Runner

    async def test():
        print("Testing LearningAgent...")
        print("-" * 60)

        agent = create_learning_agent()

        # Test knowledge query
        result = await Runner.run(
            agent,
            "What do we know about fixing 'too dark' issues in explosion renders?"
        )

        print(f"Response: {result.final_output}")

    asyncio.run(test())
