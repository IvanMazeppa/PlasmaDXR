"""
Unit tests for Phase 2B-3: Deterministic quality checks.

Tests Tier 1 render validation — the $0 checks that run before LLM vision eval.
"""

import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from tools.deterministic_quality_checks import (
    DeterministicCheckResult,
    run_deterministic_checks,
    _check_valid_image,
    _check_minimum_size,
    _check_not_blank,
    _check_reasonable_file_size,
    _check_lights_in_script,
    _check_camera_in_script,
    PNG_MAGIC,
    MIN_RENDER_FILE_SIZE,
)

from config.agent_config import AgentConfigManager, PresetConfig


def _make_png(width: int = 256, height: int = 256, body_size: int = 50000) -> bytes:
    """
    Create a minimal valid PNG file with given dimensions.

    Structure: 8-byte signature + IHDR chunk + dummy IDAT chunk + IEND chunk.
    """
    sig = PNG_MAGIC  # 8 bytes

    # IHDR chunk: width (4B) + height (4B) + bit_depth (1B) + color_type (1B)
    # + compression (1B) + filter (1B) + interlace (1B) = 13 bytes
    ihdr_data = struct.pack(">II", width, height) + b"\x08\x02\x00\x00\x00"
    ihdr_crc = b"\x00\x00\x00\x00"  # Dummy CRC (not validated by our code)
    ihdr = struct.pack(">I", 13) + b"IHDR" + ihdr_data + ihdr_crc

    # IDAT chunk: dummy compressed data to reach target file size
    idat_data = b"\x78\x01" + b"\x00" * max(0, body_size - 30) + b"\x00\x00\x00\x01"
    idat = struct.pack(">I", len(idat_data)) + b"IDAT" + idat_data + b"\x00\x00\x00\x00"

    # IEND chunk
    iend = struct.pack(">I", 0) + b"IEND" + b"\xae\x42\x60\x82"

    return sig + ihdr + idat + iend


def _make_tiny_png(width: int = 256, height: int = 256) -> bytes:
    """Create a PNG that's valid but extremely small (blank heuristic trigger)."""
    return _make_png(width, height, body_size=50)


class TestCheckResult(unittest.TestCase):
    """Test DeterministicCheckResult dataclass."""

    def test_default_fields(self):
        r = DeterministicCheckResult(passed=True)
        self.assertTrue(r.passed)
        self.assertEqual(r.critical_issues, [])
        self.assertEqual(r.warnings, [])
        self.assertEqual(r.checks_run, 0)
        self.assertEqual(r.checks_passed, 0)
        self.assertEqual(r.evaluation_tier, "deterministic")

    def test_str_pass(self):
        r = DeterministicCheckResult(passed=True, checks_run=6, checks_passed=6)
        self.assertIn("PASS", str(r))
        self.assertIn("6/6", str(r))

    def test_str_fail(self):
        r = DeterministicCheckResult(
            passed=False,
            checks_run=1,
            checks_passed=0,
            critical_issues=["[render_valid_image] bad"],
        )
        self.assertIn("FAIL", str(r))
        self.assertIn("critical=", str(r))


class TestImageValidation(unittest.TestCase):
    """Test render_valid_image check."""

    def test_valid_png(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(_make_png())
            f.flush()
            passed, issue = _check_valid_image(Path(f.name))
            self.assertTrue(passed)
            self.assertIsNone(issue)
            Path(f.name).unlink()

    def test_valid_jpeg(self):
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
            # JPEG magic + enough data
            f.write(b"\xff\xd8\xff\xe0" + b"\x00" * 100)
            f.flush()
            passed, issue = _check_valid_image(Path(f.name))
            self.assertTrue(passed)
            self.assertIsNone(issue)
            Path(f.name).unlink()

    def test_invalid_file(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"this is not an image file")
            f.flush()
            passed, issue = _check_valid_image(Path(f.name))
            self.assertFalse(passed)
            self.assertIn("Unrecognized", issue)
            Path(f.name).unlink()

    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.flush()
            passed, issue = _check_valid_image(Path(f.name))
            self.assertFalse(passed)
            self.assertIn("empty", issue)
            Path(f.name).unlink()

    def test_nonexistent_file(self):
        passed, issue = _check_valid_image(Path("/nonexistent/render.png"))
        self.assertFalse(passed)
        self.assertIn("does not exist", issue)


class TestMinimumSize(unittest.TestCase):
    """Test render_minimum_size check."""

    def test_normal_size(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(_make_png(1920, 1080))
            f.flush()
            passed, issue = _check_minimum_size(Path(f.name))
            self.assertTrue(passed)
            Path(f.name).unlink()

    def test_too_small(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(_make_png(32, 32))
            f.flush()
            passed, issue = _check_minimum_size(Path(f.name))
            self.assertFalse(passed)
            self.assertIn("too small", issue)
            Path(f.name).unlink()

    def test_edge_case_minimum(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(_make_png(64, 64))
            f.flush()
            passed, issue = _check_minimum_size(Path(f.name))
            self.assertTrue(passed)
            Path(f.name).unlink()


class TestBlankDetection(unittest.TestCase):
    """Test render_not_blank check."""

    def test_normal_render_passes(self):
        """Large enough file passes the bytes-per-pixel heuristic."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            # 256x256 with 50KB body = ~0.76 bytes/pixel — well above threshold
            f.write(_make_png(256, 256, body_size=50000))
            f.flush()
            passed, issue = _check_not_blank(Path(f.name))
            self.assertTrue(passed)
            Path(f.name).unlink()

    def test_tiny_file_detected_as_blank(self):
        """Very small file relative to dimensions triggers blank detection."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            # 1920x1080 with tiny body = very low bytes/pixel
            f.write(_make_tiny_png(1920, 1080))
            f.flush()
            passed, issue = _check_not_blank(Path(f.name))
            self.assertFalse(passed)
            self.assertIn("blank", issue.lower())
            Path(f.name).unlink()


class TestFileSize(unittest.TestCase):
    """Test render_reasonable_file_size check."""

    def test_reasonable_size(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(_make_png(1920, 1080, body_size=100000))
            f.flush()
            passed, issue = _check_reasonable_file_size(Path(f.name))
            self.assertTrue(passed)
            Path(f.name).unlink()

    def test_too_small(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(_make_png(256, 256, body_size=100))
            f.flush()
            passed, issue = _check_reasonable_file_size(Path(f.name))
            # File < 15KB → warning
            if Path(f.name).stat().st_size < MIN_RENDER_FILE_SIZE:
                self.assertFalse(passed)
                self.assertIn("small", issue)
            Path(f.name).unlink()


class TestScriptChecks(unittest.TestCase):
    """Test script-level checks for lights and camera."""

    def test_lights_present(self):
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("""
import bpy
bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))
light = bpy.context.active_object
light.data.energy = 5.0
""")
            f.flush()
            passed, issue = _check_lights_in_script(Path(f.name))
            self.assertTrue(passed)
            Path(f.name).unlink()

    def test_lights_via_emission(self):
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("""
import bpy
mat = bpy.data.materials.new("glow")
mat.use_nodes = True
emission = mat.node_tree.nodes.new("ShaderNodeEmission")
emission.inputs['Strength'].default_value = 10.0
# emission_strength reference
""")
            f.flush()
            passed, issue = _check_lights_in_script(Path(f.name))
            self.assertTrue(passed)
            Path(f.name).unlink()

    def test_lights_missing(self):
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("""
import bpy
bpy.ops.mesh.primitive_cube_add()
bpy.ops.object.modifier_add(type='FLUID')
""")
            f.flush()
            passed, issue = _check_lights_in_script(Path(f.name))
            self.assertFalse(passed)
            self.assertIn("No light", issue)
            Path(f.name).unlink()

    def test_camera_present(self):
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("""
import bpy
cam = bpy.data.cameras.new("Camera")
cam_obj = bpy.data.objects.new("Camera", cam)
bpy.context.scene.camera = cam_obj
""")
            f.flush()
            passed, issue = _check_camera_in_script(Path(f.name))
            self.assertTrue(passed)
            Path(f.name).unlink()

    def test_camera_missing(self):
        with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
            f.write("""
import bpy
bpy.ops.mesh.primitive_cube_add()
""")
            f.flush()
            passed, issue = _check_camera_in_script(Path(f.name))
            self.assertFalse(passed)
            self.assertIn("No camera", issue)
            Path(f.name).unlink()


class TestShortCircuit(unittest.TestCase):
    """Test that critical Tier 1 failures short-circuit correctly."""

    def test_critical_fail_returns_early(self):
        """Critical failure should stop after first critical check."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(b"not an image")
            f.flush()
            result = run_deterministic_checks(render_path=f.name)
            self.assertFalse(result.passed)
            self.assertEqual(len(result.critical_issues), 1)
            self.assertIn("render_valid_image", result.critical_issues[0])
            # Fast-fail: only 1 check ran
            self.assertEqual(result.checks_run, 1)
            self.assertEqual(result.checks_passed, 0)
            Path(f.name).unlink()

    def test_passing_render_runs_all_checks(self):
        """A valid render should run all checks including warnings."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as render_f:
            # 1920x1080 = 2,073,600 pixels. Need bytes/pixel > 0.05 → > 103,680 bytes
            render_f.write(_make_png(1920, 1080, body_size=200000))
            render_f.flush()

            with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as script_f:
                script_f.write("""
import bpy
bpy.ops.object.light_add(type='SUN')
scene.camera = cam_obj
""")
                script_f.flush()

                result = run_deterministic_checks(
                    render_path=render_f.name,
                    script_path=script_f.name,
                )
                self.assertTrue(result.passed)
                # 3 critical + 3 warning checks = 6
                self.assertEqual(result.checks_run, 6)
                self.assertEqual(result.checks_passed, 6)
                self.assertEqual(result.critical_issues, [])

                Path(script_f.name).unlink()
            Path(render_f.name).unlink()

    def test_warnings_preserved_on_pass(self):
        """Warnings should be recorded but not block passing."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as render_f:
            # Valid PNG but small file size → warning (but NOT blank)
            # 256x256 = 65536 pixels. Need bytes/pixel > 0.05 → > 3277 bytes
            # Use 5000 body to pass blank check but still < 15KB for file size warning
            render_f.write(_make_png(256, 256, body_size=5000))
            render_f.flush()

            with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as script_f:
                # No lights, no camera → 2 warnings
                script_f.write("import bpy\nbpy.ops.mesh.primitive_cube_add()\n")
                script_f.flush()

                result = run_deterministic_checks(
                    render_path=render_f.name,
                    script_path=script_f.name,
                )
                self.assertTrue(result.passed)
                # Should have warnings for small file, no lights, no camera
                self.assertGreaterEqual(len(result.warnings), 2)
                self.assertEqual(result.critical_issues, [])

                Path(script_f.name).unlink()
            Path(render_f.name).unlink()

    def test_no_script_skips_script_checks(self):
        """Without script_path, only image checks run."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            f.write(_make_png(1920, 1080, body_size=200000))
            f.flush()
            result = run_deterministic_checks(render_path=f.name)
            self.assertTrue(result.passed)
            # 3 critical + 1 warning (file size) = 4 checks, no script checks
            self.assertEqual(result.checks_run, 4)
            Path(f.name).unlink()


class TestDeterministicConfig(unittest.TestCase):
    """Test feature flag in config system."""

    def test_preset_default_enabled(self):
        """Default PresetConfig has multi_grader_eval=True."""
        preset = PresetConfig(name="test")
        self.assertTrue(preset.multi_grader_eval)

    def test_from_dict_enabled(self):
        preset = PresetConfig.from_dict("test", {"multi_grader_eval": True})
        self.assertTrue(preset.multi_grader_eval)

    def test_from_dict_disabled(self):
        preset = PresetConfig.from_dict("test", {"multi_grader_eval": False})
        self.assertFalse(preset.multi_grader_eval)

    def test_config_manager_accessor(self):
        preset = PresetConfig(name="test", multi_grader_eval=True)
        config = AgentConfigManager(preset=preset)
        self.assertTrue(config.use_multi_grader_eval())

        preset_off = PresetConfig(name="test", multi_grader_eval=False)
        config_off = AgentConfigManager(preset=preset_off)
        self.assertFalse(config_off.use_multi_grader_eval())


if __name__ == "__main__":
    unittest.main()
