"""
Tests for Phase 2A-3: Tool-level guardrails for truth pack enforcement.

Tests the three guardrails:
1. truth_pack_input_guardrail — validates scripts against truth pack BEFORE execution
2. critical_failure_output_guardrail — catches BLACK_SCREEN/ZERO_LIGHTS in evaluation output
3. script_length_output_guardrail — warns when generated scripts are too basic (<500 lines)
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

# Path setup (same as conftest.py)
_orchestrator_root = str(Path(__file__).parent.parent)
if _orchestrator_root not in sys.path:
    sys.path.insert(0, _orchestrator_root)

from guardrails.tool_guardrails import (
    truth_pack_input_guardrail,
    critical_failure_output_guardrail,
    script_length_output_guardrail,
    set_truth_pack,
    CRITICAL_FAILURE_KEYWORDS,
    MIN_SCRIPT_LINES_WARNING,
    MIN_SCRIPT_LINES_BASIC,
)


def _make_input_data(tool_arguments: dict):
    """Create a mock ToolInputGuardrailData-like object."""
    ctx = SimpleNamespace(tool_arguments=json.dumps(tool_arguments))
    return SimpleNamespace(context=ctx, agent=None)


def _make_output_data(output):
    """Create a mock ToolOutputGuardrailData-like object."""
    return SimpleNamespace(output=output)


def _is_allowed(result) -> bool:
    """Check if a ToolGuardrailFunctionOutput allows execution."""
    return result.behavior["type"] == "allow"


def _is_rejected(result) -> bool:
    """Check if a ToolGuardrailFunctionOutput rejects content."""
    return result.behavior["type"] == "reject_content"


# Minimal truth pack with FluidDomainSettings that has resolution_max but NOT resolution_divisions
_MINIMAL_TRUTH_PACK = {
    "FluidDomainSettings": {
        "properties": {
            "resolution_max": {"type": "int", "default": 64},
            "use_adaptive_timesteps": {"type": "bool", "default": False},
            "timesteps_max": {"type": "int", "default": 4},
            "use_dissolve_smoke": {"type": "bool", "default": False},
            "cache_type": {"type": "enum", "default": "REPLAY"},
        }
    }
}


class TestTruthPackInputGuardrail(unittest.TestCase):
    """Tests for truth_pack_input_guardrail on execute_blender_script."""

    def setUp(self):
        set_truth_pack(_MINIMAL_TRUTH_PACK)

    def tearDown(self):
        set_truth_pack(None)

    def test_hallucinated_attribute_caught_and_auto_fixed(self):
        """Script with resolution_divisions → guardrail catches and auto-fixes to resolution_max."""
        script = (
            "import bpy\n"
            "domain = bpy.context.object\n"
            "domain_settings = domain.modifiers['Fluid'].domain_settings\n"
            "domain_settings.resolution_divisions = 128\n"
            "domain_settings.use_adaptive_timesteps = True\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(script)
            f.flush()
            script_path = f.name

        try:
            data = _make_input_data({"script_path": script_path})
            result = truth_pack_input_guardrail.guardrail_function(data)

            # resolution_divisions → resolution_max is a KNOWN_HALLUCINATION with auto-fix
            # So the guardrail should auto-fix and allow
            fixed_script = Path(script_path).read_text()
            self.assertIn("resolution_max", fixed_script)
            self.assertNotIn("resolution_divisions", fixed_script)
            self.assertTrue(_is_allowed(result))
        finally:
            os.unlink(script_path)

    def test_clean_script_allowed(self):
        """Script with all valid attributes → guardrail allows execution."""
        script = (
            "import bpy\n"
            "domain = bpy.context.object\n"
            "domain_settings = domain.modifiers['Fluid'].domain_settings\n"
            "domain_settings.resolution_max = 128\n"
            "domain_settings.use_adaptive_timesteps = True\n"
            "domain_settings.timesteps_max = 4\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(script)
            f.flush()
            script_path = f.name

        try:
            data = _make_input_data({"script_path": script_path})
            result = truth_pack_input_guardrail.guardrail_function(data)
            self.assertTrue(_is_allowed(result))
        finally:
            os.unlink(script_path)

    def test_no_truth_pack_allows_all(self):
        """When truth pack is not set, guardrail allows all scripts."""
        set_truth_pack(None)
        data = _make_input_data({"script_path": "/nonexistent/script.py"})
        result = truth_pack_input_guardrail.guardrail_function(data)
        self.assertTrue(_is_allowed(result))

    def test_missing_script_path_allows(self):
        """When script_path is missing from tool arguments, guardrail allows."""
        data = _make_input_data({})
        result = truth_pack_input_guardrail.guardrail_function(data)
        self.assertTrue(_is_allowed(result))

    def test_nonexistent_file_allows(self):
        """When script file doesn't exist, guardrail lets the tool handle the error."""
        data = _make_input_data({"script_path": "/tmp/nonexistent_script_12345.py"})
        result = truth_pack_input_guardrail.guardrail_function(data)
        self.assertTrue(_is_allowed(result))

    def test_kill_switch_disables_guardrail(self):
        """ENABLE_TOOL_GUARDRAILS=false disables the guardrail."""
        with patch("guardrails.tool_guardrails.ENABLE_TOOL_GUARDRAILS", False):
            data = _make_input_data({"script_path": "/some/script.py"})
            result = truth_pack_input_guardrail.guardrail_function(data)
            self.assertTrue(_is_allowed(result))


class TestCriticalFailureOutputGuardrail(unittest.TestCase):
    """Tests for critical_failure_output_guardrail on evaluate_render/analyze_with_vision."""

    def test_black_screen_rejected(self):
        """BLACK_SCREEN in output → guardrail rejects with targeted message."""
        data = _make_output_data("Quality score: 0.0. Issues: BLACK_SCREEN detected, no visible content.")
        result = critical_failure_output_guardrail.guardrail_function(data)
        self.assertTrue(_is_rejected(result))
        self.assertIn("fundamental fix", result.behavior["message"])

    def test_zero_lights_rejected(self):
        """ZERO_LIGHTS in output → guardrail rejects."""
        data = _make_output_data("Evaluation: ZERO_LIGHTS active in scene")
        result = critical_failure_output_guardrail.guardrail_function(data)
        self.assertTrue(_is_rejected(result))

    def test_white_screen_rejected(self):
        """WHITE_SCREEN in output → guardrail rejects."""
        data = _make_output_data("Render shows WHITE_SCREEN overexposure")
        result = critical_failure_output_guardrail.guardrail_function(data)
        self.assertTrue(_is_rejected(result))

    def test_clipping_artifacts_rejected(self):
        """CLIPPING_ARTIFACTS in output → guardrail rejects."""
        data = _make_output_data("Visual issues: CLIPPING_ARTIFACTS at domain boundary")
        result = critical_failure_output_guardrail.guardrail_function(data)
        self.assertTrue(_is_rejected(result))

    def test_normal_output_allowed(self):
        """Normal evaluation output → guardrail allows."""
        data = _make_output_data("Quality score: 72.5. Good volumetric density, minor noise.")
        result = critical_failure_output_guardrail.guardrail_function(data)
        self.assertTrue(_is_allowed(result))

    def test_none_output_allowed(self):
        """None output → guardrail allows."""
        data = _make_output_data(None)
        result = critical_failure_output_guardrail.guardrail_function(data)
        self.assertTrue(_is_allowed(result))

    def test_all_critical_keywords_covered(self):
        """All defined critical failure keywords are caught."""
        for kw in CRITICAL_FAILURE_KEYWORDS:
            data = _make_output_data(f"Some context {kw} some more context")
            result = critical_failure_output_guardrail.guardrail_function(data)
            self.assertTrue(_is_rejected(result), f"Failed to catch: {kw}")

    def test_kill_switch_disables(self):
        """ENABLE_TOOL_GUARDRAILS=false disables the guardrail."""
        with patch("guardrails.tool_guardrails.ENABLE_TOOL_GUARDRAILS", False):
            data = _make_output_data("BLACK_SCREEN everywhere")
            result = critical_failure_output_guardrail.guardrail_function(data)
            self.assertTrue(_is_allowed(result))


class TestScriptLengthOutputGuardrail(unittest.TestCase):
    """Tests for script_length_output_guardrail on generate_script."""

    def _write_script(self, line_count: int) -> str:
        """Write a temp script with the given number of lines."""
        lines = [f"# Line {i}" for i in range(line_count)]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("\n".join(lines))
            return f.name

    def test_short_script_rejected(self):
        """Script under 300 lines → guardrail rejects."""
        script_path = self._write_script(200)
        try:
            data = _make_output_data(json.dumps({"script_path": script_path}))
            result = script_length_output_guardrail.guardrail_function(data)
            self.assertTrue(_is_rejected(result))
            self.assertIn("200 lines", result.behavior["message"])
        finally:
            os.unlink(script_path)

    def test_medium_script_warns(self):
        """Script between 300-500 lines → guardrail warns but allows."""
        script_path = self._write_script(400)
        try:
            data = _make_output_data(json.dumps({"script_path": script_path}))
            result = script_length_output_guardrail.guardrail_function(data)
            self.assertTrue(_is_allowed(result))
            self.assertIsNotNone(result.output_info)
            self.assertIn("warning", result.output_info)
        finally:
            os.unlink(script_path)

    def test_long_script_allowed_no_warning(self):
        """Script over 500 lines → guardrail allows with no warning."""
        script_path = self._write_script(600)
        try:
            data = _make_output_data(json.dumps({"script_path": script_path}))
            result = script_length_output_guardrail.guardrail_function(data)
            self.assertTrue(_is_allowed(result))
            self.assertIsNone(result.output_info)
        finally:
            os.unlink(script_path)

    def test_dict_output_format(self):
        """Output as dict (not JSON string) is handled correctly."""
        script_path = self._write_script(200)
        try:
            data = _make_output_data({"script_path": script_path})
            result = script_length_output_guardrail.guardrail_function(data)
            self.assertTrue(_is_rejected(result))
        finally:
            os.unlink(script_path)

    def test_missing_script_path_allows(self):
        """No script_path in output → guardrail allows."""
        data = _make_output_data(json.dumps({"status": "ok"}))
        result = script_length_output_guardrail.guardrail_function(data)
        self.assertTrue(_is_allowed(result))

    def test_kill_switch_disables(self):
        """ENABLE_TOOL_GUARDRAILS=false disables the guardrail."""
        with patch("guardrails.tool_guardrails.ENABLE_TOOL_GUARDRAILS", False):
            data = _make_output_data(json.dumps({"script_path": "/some/short.py"}))
            result = script_length_output_guardrail.guardrail_function(data)
            self.assertTrue(_is_allowed(result))


class TestAttachToolGuardrails(unittest.TestCase):
    """Tests for the attach_tool_guardrails() wiring function."""

    def test_attach_returns_dict(self):
        """attach_tool_guardrails() returns a dict of attached guardrails."""
        from guardrails.tool_guardrails import attach_tool_guardrails

        result = attach_tool_guardrails()
        self.assertIsInstance(result, dict)
        self.assertIn("execute_blender_script", result)
        self.assertIn("generate_script", result)

    def test_execute_blender_script_has_input_guardrail(self):
        """execute_blender_script should have truth_pack_input_guardrail attached."""
        from guardrails.tool_guardrails import attach_tool_guardrails
        from tools.blender_executor_tools import execute_blender_script

        # Clear any existing guardrails first
        execute_blender_script.tool_input_guardrails = None
        attach_tool_guardrails()

        self.assertIsNotNone(execute_blender_script.tool_input_guardrails)
        self.assertTrue(len(execute_blender_script.tool_input_guardrails) > 0)


class TestSectionNamingGuardrail(unittest.TestCase):
    """Test the section naming check on write_script output."""

    def _make_output(self, sections, success=True):
        return json.dumps({
            "success": success,
            "script_path": "/tmp/test.py",
            "sections_found": len(sections),
            "sections": sections,
        })

    def test_fully_sectioned_passes(self):
        from guardrails.script_guardrails import _check_section_naming
        output = self._make_output([
            "setup_scene", "create_geometry", "setup_materials",
            "setup_physics", "setup_lighting", "setup_camera",
            "bake_and_render",
        ])
        triggered, info = _check_section_naming(output)
        self.assertFalse(triggered)

    def test_unsectioned_warns(self):
        from guardrails.script_guardrails import _check_section_naming
        output = self._make_output([])
        triggered, info = _check_section_naming(output)
        # Warning only, not tripwire
        self.assertFalse(triggered)
        self.assertIn("warning", info)

    def test_partial_warns_with_missing(self):
        from guardrails.script_guardrails import _check_section_naming
        output = self._make_output(["setup_scene", "setup_physics", "bake_and_render"])
        triggered, info = _check_section_naming(output)
        self.assertFalse(triggered)
        self.assertIn("missing_sections", info)

    def test_failed_write_skips_check(self):
        from guardrails.script_guardrails import _check_section_naming
        output = self._make_output([], success=False)
        triggered, info = _check_section_naming(output)
        self.assertFalse(triggered)
        self.assertEqual(info["status"], "script_write_failed")


if __name__ == "__main__":
    unittest.main()
