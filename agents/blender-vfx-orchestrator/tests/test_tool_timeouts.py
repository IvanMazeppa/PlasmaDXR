"""
Tests for Phase 2A-8: Function Tool Timeouts and Error Handlers.

Verifies that:
1. execute_blender_script has SDK-level timeout configured
2. failure_error_function produces actionable error messages
3. Internal timeout parameter is preserved
"""

import asyncio
import json
import os
import sys
import unittest
from pathlib import Path

# Path setup
_orchestrator_root = str(Path(__file__).parent.parent)
if _orchestrator_root not in sys.path:
    sys.path.insert(0, _orchestrator_root)


class TestExecutorToolTimeout(unittest.TestCase):
    """execute_blender_script should have SDK-level timeout configured."""

    def test_execute_blender_script_has_timeout(self):
        from tools.blender_executor_tools import execute_blender_script

        self.assertIsNotNone(execute_blender_script.timeout_seconds)
        self.assertGreater(execute_blender_script.timeout_seconds, 0)

    def test_timeout_is_above_internal_default(self):
        """SDK timeout should be >= internal 600s to avoid double-timeout race."""
        from tools.blender_executor_tools import execute_blender_script

        self.assertGreaterEqual(execute_blender_script.timeout_seconds, 600.0)

    def test_other_tools_have_no_timeout(self):
        """Non-execution tools don't need SDK timeout (they're fast)."""
        from tools.blender_executor_tools import (
            parse_blender_errors,
            list_run_outputs,
            get_latest_run,
        )

        # These tools are fast — they shouldn't have a timeout
        self.assertIsNone(parse_blender_errors.timeout_seconds)
        self.assertIsNone(list_run_outputs.timeout_seconds)
        self.assertIsNone(get_latest_run.timeout_seconds)


class TestExecutionErrorHandler(unittest.TestCase):
    """blender_execution_error_handler should produce actionable messages."""

    def _call_handler(self, exc):
        from tools.blender_executor_tools import blender_execution_error_handler

        return blender_execution_error_handler(None, exc)

    def test_file_not_found(self):
        result = self._call_handler(FileNotFoundError("/tmp/missing.py"))
        self.assertIn("FILE_NOT_FOUND", result)
        self.assertIn("script_path", result)

    def test_permission_error(self):
        result = self._call_handler(PermissionError("Access denied"))
        self.assertIn("PERMISSION_DENIED", result)

    def test_json_decode_error(self):
        result = self._call_handler(
            json.JSONDecodeError("Expecting value", "{bad", 0)
        )
        self.assertIn("INVALID_JSON", result)
        self.assertIn("script_args_json", result)

    def test_memory_error(self):
        result = self._call_handler(MemoryError("Cannot allocate memory"))
        self.assertIn("OUT_OF_MEMORY", result)
        self.assertIn("resolution_max", result)

    def test_timeout_error(self):
        result = self._call_handler(asyncio.TimeoutError())
        self.assertIn("TIMEOUT", result)
        self.assertIn("resolution_max", result)

    def test_generic_error_includes_type(self):
        result = self._call_handler(RuntimeError("Something unexpected"))
        self.assertIn("EXECUTION_ERROR", result)
        self.assertIn("RuntimeError", result)
        self.assertIn("Something unexpected", result)

    def test_os_error_with_memory_keyword(self):
        """OSError mentioning memory should trigger OUT_OF_MEMORY advice."""
        result = self._call_handler(OSError("Not enough memory"))
        self.assertIn("OUT_OF_MEMORY", result)


class TestGenericToolErrorHandler(unittest.TestCase):
    """blender_tool_error_handler should produce concise messages."""

    def test_returns_tool_error_prefix(self):
        from tools.blender_executor_tools import blender_tool_error_handler

        result = blender_tool_error_handler(None, ValueError("bad value"))
        self.assertIn("TOOL_ERROR", result)
        self.assertIn("ValueError", result)

    def test_truncates_long_messages(self):
        from tools.blender_executor_tools import blender_tool_error_handler

        long_msg = "x" * 1000
        result = blender_tool_error_handler(None, ValueError(long_msg))
        # Should be truncated to ~500 chars + prefix
        self.assertLessEqual(len(result), 600)


class TestInternalTimeoutStillWorks(unittest.TestCase):
    """The internal asyncio.wait_for timeout should still function."""

    def test_impl_has_timeout_parameter(self):
        """_execute_blender_script_impl should accept timeout_seconds."""
        import inspect
        from tools.blender_executor_tools import _execute_blender_script_impl

        sig = inspect.signature(_execute_blender_script_impl)
        self.assertIn("timeout_seconds", sig.parameters)
        self.assertEqual(sig.parameters["timeout_seconds"].default, 600)

    def test_tool_wrapper_passes_timeout_through(self):
        """execute_blender_script function should accept timeout_seconds."""
        import inspect
        from tools.blender_executor_tools import execute_blender_script

        # The tool's JSON schema should include timeout_seconds
        schema = execute_blender_script.params_json_schema
        self.assertIn("timeout_seconds", schema.get("properties", {}))


if __name__ == "__main__":
    unittest.main()
