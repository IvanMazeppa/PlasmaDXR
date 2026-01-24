"""
Learning Agent using OpenAI Agents SDK.

Specialized agent for experiment tracking and accumulated learning.
Maintains a knowledge base of what works and what doesn't.

Key capabilities:
- Record experiment outcomes (success and failure)
- Query knowledge base for relevant learnings
- Suggest experiments based on past success
- Warn about known gotchas before changes
- Process physics observations from Quality Analyst

SELF-LEARNING: This agent is the core of the self-learning system. It:
1. Processes physics observations from the Quality Analyst
2. Correlates parameters with outcomes
3. Extracts patterns with success rates
4. Builds a knowledge base that dynamic instructions query
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

# Import physics observation tools for processing Quality Analyst observations
from tools.physics_observation_tools import (
    get_pending_observations,
    correlate_observation,
    get_physics_patterns,
)

# Doc mining tools (optional fallback when KB is empty)
from tools.semantic_docs_tools import (
    blender_doc_search_bundle,
)
# Import dynamic instructions for self-learning
from tools.dynamic_instructions import (
    dynamic_learning_agent_instructions,
    get_learning_agent_instructions_static,
    LEARNING_AGENT_BASE_INSTRUCTIONS,
)


# DEPRECATED: Hardcoded instructions replaced by dynamic_instructions.py
# Keeping for reference only
LEARNING_AGENT_INSTRUCTIONS_DEPRECATED = """
[DEPRECATED - See tools/dynamic_instructions.py for current instructions]
"""


class LearningAgent:
    """
    Learning Agent for experiment tracking and knowledge accumulation.

    Uses gpt-5.2 with high reasoning for intelligent learning and pattern recognition.

    SELF-LEARNING: This is the core of the self-learning system. It:
    1. Processes physics observations from the Quality Analyst
    2. Correlates parameters with outcomes
    3. Extracts patterns with success rates
    4. Builds knowledge base that dynamic instructions query
    """

    def __init__(self, model: str = "gpt-5.2", use_dynamic_instructions: bool = True):
        """
        Initialize the learning agent.

        Args:
            model: OpenAI model to use (default: gpt-5.2 for deep learning insights)
            use_dynamic_instructions: If True, use dynamic instructions that include
                                      current knowledge state and areas needing data.
        """
        self.model = os.getenv("LEARNING_AGENT_MODEL", model)
        self.use_dynamic_instructions = use_dynamic_instructions
        self._agent: Optional[Agent] = None

    def initialize(self, custom_instructions: str = "") -> None:
        """
        Initialize the agent with tools.

        Args:
            custom_instructions: Additional context (e.g., session ID)
        """
        # Choose instruction source
        if self.use_dynamic_instructions:
            # Dynamic instructions: function that generates instructions at runtime
            instructions = dynamic_learning_agent_instructions
        else:
            # Static fallback
            instructions = get_learning_agent_instructions_static()
            if custom_instructions:
                instructions = instructions + "\n\n" + custom_instructions

        self._agent = Agent(
            name="Learning Agent",
            instructions=instructions,  # Can be function OR string
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
                # Physics observation tools (process Quality Analyst observations)
                get_pending_observations,
                correlate_observation,
                get_physics_patterns,
                # Doc mining fallback (used only when KB is empty)
                blender_doc_search_bundle,
            ],
        )

    @property
    def agent(self) -> Agent:
        """Get the underlying Agent instance for handoffs."""
        if not self._agent:
            raise RuntimeError("LearningAgent not initialized. Call initialize() first.")
        return self._agent


def create_learning_agent(
    custom_instructions: str = "",
    use_dynamic_instructions: bool = True
) -> Agent:
    """
    Factory function to create and initialize a learning agent.

    Args:
        custom_instructions: Additional context to append
        use_dynamic_instructions: If True (default), use dynamic instructions that
                                  include current knowledge state and areas needing data.

    Returns:
        Initialized Agent instance ready for use
    """
    agent = LearningAgent(use_dynamic_instructions=use_dynamic_instructions)
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
