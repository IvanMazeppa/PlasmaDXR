"""
Specialized agents for the Blender VFX orchestration pipeline.

Each agent is an OpenAI Agent with specific tools and capabilities:
- ScriptWriterAgent: Script generation and modification
- QualityAnalystAgent: ML-powered quality evaluation
- LearningAgent: Experiment tracking and learning
- DocsExpert: Documentation search (uses function_tools, not MCP)
- APIValidator: Blender 5.0 API validation (validates code before execution)
- APISpecAgent: DEPRECATED (2A-6) — replaced by truth pack + tool guardrails
- CodeWriterAgent: DEPRECATED (2A-6) — replaced by truth pack + tool guardrails
- ExecutorAgent: DEPRECATED (2A-7) — execution is deterministic via _execute_blender_script_impl()

NOTE: This package is named 'specialized_agents' to avoid conflict with
the 'agents' package from openai-agents-python SDK.
"""

from .script_writer import ScriptWriterAgent, create_script_writer
from .quality_analyst import QualityAnalystAgent, create_quality_analyst
from .learning_agent import LearningAgent, create_learning_agent
from .docs_expert import create_docs_expert  # No class, just factory function
from .api_validator import (
    create_api_validator,
    get_api_validator_as_tool,
    validate_code_api,
    CodeValidationResult,
    APICallValidation,
)
# Deprecated agents — kept for historical reference, not imported by orchestrator
# from .executor import ExecutorAgent, create_executor  # DEPRECATED 2A-7
# from .api_spec_agent import APISpecAgent, create_api_spec_agent  # DEPRECATED 2A-6
# from .code_writer_agent import CodeWriterAgent, create_code_writer_agent  # DEPRECATED 2A-6

__all__ = [
    "ScriptWriterAgent",
    "QualityAnalystAgent",
    "LearningAgent",
    "create_script_writer",
    "create_quality_analyst",
    "create_learning_agent",
    "create_docs_expert",
    # API Validator exports
    "create_api_validator",
    "get_api_validator_as_tool",
    "validate_code_api",
    "CodeValidationResult",
    "APICallValidation",
]
