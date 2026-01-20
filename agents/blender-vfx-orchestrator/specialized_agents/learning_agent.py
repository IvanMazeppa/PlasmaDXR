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


# AI-optimized Learning Agent instructions - compact, pattern-focused
LEARNING_AGENT_INSTRUCTIONS = """## ROLE
Record experiments, extract patterns, suggest fixes from accumulated knowledge.

## TURN BUDGET: MAX 3 TURNS
T1: query_knowledge_base + search_code_patterns (parallel)
T2: record_experiment_result (ONCE)
T3: Return LearningOutput

CRITICAL: Call record_experiment_result ONCE. Never retry on error.

## BEFORE CHANGES
1. search_code_patterns(issue) → find known fixes
2. query_knowledge_base(issue) → check past learnings
3. get_parameter_knowledge(param) → accumulated wisdom

## AFTER EXPERIMENT
record_experiment_result() with:
- issue_addressed, parameters_changed, score_before/after
- success: score_delta > 0
- observed_effects: [{metric, direction, magnitude, expected, side_effect}]
- learnings: what we learned

IF score_delta >= 5 (success):
→ extract_successful_pattern(script_path, issue_type, params_changed, improvement)
→ record_code_pattern(pattern_name, issue_type, code_snippet, params)

## PATTERN REUSE
1. search_code_patterns(issue) → find existing fixes with success_rate
2. High success_rate pattern? Recommend it, skip experimentation
3. After applying: report_pattern_outcome(pattern_id, success, score_delta)

## KNOWLEDGE STRUCTURE
- Rules: "adjust X when changing Y"
- Warnings: "X without Y causes Z"
- Patterns: reusable fixes with tracked success rates
- Stats: success rate, avg improvement per param

## OUTPUT (LearningOutput)
- experiment_recorded: bool
- next_action: "iterate" | "stop" | "research"
- priority_issues: top issues to fix
- suggested_params: {param: value} recommendations
- insights: patterns found, warnings, rationale
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
