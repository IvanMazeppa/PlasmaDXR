"""
Tests for Phase 2A-2: Deterministic Agent Control (tool_use_behavior).

Verifies that agents with stop_on_first_tool are configured correctly,
and that agents requiring multi-tool sequences do NOT have it set.

Phase 2A-7: Executor agent removed — execution is deterministic via
_execute_blender_script_impl(). Tests for executor stop_on_first_tool removed.
"""

import ast
import importlib.util
import os
import sys
import unittest
from pathlib import Path

# Path setup (same as conftest.py)
_orchestrator_root = str(Path(__file__).parent.parent)
if _orchestrator_root not in sys.path:
    sys.path.insert(0, _orchestrator_root)


class TestExecutorDeprecated(unittest.TestCase):
    """Phase 2A-7: Executor agent is deprecated — orchestrator must NOT import it."""

    def test_orchestrator_does_not_import_executor_agent(self):
        """Verify orchestrator.py has no import of the executor AGENT module.

        Note: importing from tools.blender_executor_tools is CORRECT — that's
        the direct execution path. We only forbid specialized_agents.executor.
        """
        orchestrator_path = os.path.join(_orchestrator_root, "orchestrator.py")
        with open(orchestrator_path, "r") as f:
            source = f.read()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                # Only flag the agent module, not executor tools
                if node.module and node.module.endswith(".executor"):
                    self.fail(
                        f"orchestrator.py still imports executor agent: "
                        f"'from {node.module} import ...' at line {node.lineno}"
                    )
                if node.names:
                    for alias in node.names:
                        if alias.name == "create_executor":
                            self.fail(
                                f"orchestrator.py still imports create_executor "
                                f"at line {node.lineno}"
                            )

    def test_executor_file_has_deprecation_notice(self):
        """Verify executor.py has DEPRECATED notice."""
        executor_path = os.path.join(
            _orchestrator_root, "specialized_agents", "executor.py"
        )
        with open(executor_path, "r") as f:
            first_lines = f.read(500)
        self.assertIn("DEPRECATED", first_lines)


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
