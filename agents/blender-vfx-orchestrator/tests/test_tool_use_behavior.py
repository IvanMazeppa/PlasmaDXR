"""
Tests for Phase 2A-2: Deterministic Agent Control (tool_use_behavior).

Verifies that agents with stop_on_first_tool are configured correctly,
and that agents requiring multi-tool sequences do NOT have it set.
"""

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Path setup (same as conftest.py)
_orchestrator_root = str(Path(__file__).parent.parent)
if _orchestrator_root not in sys.path:
    sys.path.insert(0, _orchestrator_root)


class TestExecutorStopOnFirstTool(unittest.TestCase):
    """Executor should use stop_on_first_tool to skip LLM post-processing."""

    def test_executor_has_stop_on_first_tool(self):
        from specialized_agents.executor import create_executor
        agent = create_executor()
        self.assertEqual(agent.tool_use_behavior, "stop_on_first_tool")

    def test_executor_still_has_all_tools(self):
        from specialized_agents.executor import create_executor
        agent = create_executor()
        tool_names = [t.name for t in agent.tools if hasattr(t, "name")]
        self.assertIn("execute_blender_script", tool_names)
        self.assertIn("parse_blender_errors", tool_names)
        self.assertIn("list_run_outputs", tool_names)
        self.assertIn("get_latest_run", tool_names)


class TestApiValidatorNoStopOnFirstTool(unittest.TestCase):
    """API Validator needs extract -> check -> search -> format (multi-tool)."""

    def test_api_validator_uses_default_behavior(self):
        from specialized_agents.api_validator import create_api_validator
        agent = create_api_validator()
        self.assertEqual(agent.tool_use_behavior, "run_llm_again")


class TestDocsExpertNoStopOnFirstTool(unittest.TestCase):
    """Docs Expert needs search -> validate -> return (multi-tool)."""

    def test_docs_expert_uses_default_behavior(self):
        from specialized_agents.docs_expert import create_docs_expert
        agent = create_docs_expert()
        self.assertEqual(agent.tool_use_behavior, "run_llm_again")


class TestQualityAnalystNoStopOnFirstTool(unittest.TestCase):
    """Quality Analyst needs vision analysis -> metrics -> diagnosis (multi-tool)."""

    def test_quality_analyst_uses_default_behavior(self):
        from specialized_agents.quality_analyst import create_quality_analyst
        agent = create_quality_analyst()
        self.assertEqual(agent.tool_use_behavior, "run_llm_again")


class TestLearningAgentNoStopOnFirstTool(unittest.TestCase):
    """Learning Agent needs record -> query -> extract -> correlate (multi-tool)."""

    def test_learning_agent_uses_default_behavior(self):
        from specialized_agents.learning_agent import create_learning_agent
        agent = create_learning_agent()
        self.assertEqual(agent.tool_use_behavior, "run_llm_again")


class TestQualityGateNoStopAtTools(unittest.TestCase):
    """Quality Gate needs to process tool results into QualityDecision structured output."""

    def test_quality_gate_uses_default_behavior(self):
        # Import create_quality_gate_coordinator directly from the module file
        # to avoid __init__.py's relative import chain
        spec = importlib.util.spec_from_file_location(
            "orchestrator_module",
            os.path.join(_orchestrator_root, "orchestrator.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        agent = mod.create_quality_gate_coordinator()
        self.assertEqual(agent.tool_use_behavior, "run_llm_again")


if __name__ == "__main__":
    unittest.main()
