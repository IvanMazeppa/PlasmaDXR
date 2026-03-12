"""Tests for the patch_script_section tool."""

import sys
import os
import json
import tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from pathlib import Path
from utils.script_sections import CANONICAL_SECTIONS


# Test the implementation function directly (not the @function_tool wrapper)
from tools.script_generator_tools import _patch_script_section_impl


SECTIONED_SCRIPT = '''import bpy

ASSET_NAME = "test_patch"
OUTPUT_DIR = "/tmp/test_patch"

def setup_scene():
    """Scene setup."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

def create_geometry():
    """Geometry."""
    bpy.ops.mesh.primitive_cube_add(size=1)

def setup_materials():
    """Materials."""
    mat = bpy.data.materials.new("Mat")

def setup_physics():
    """Physics."""
    pass

def setup_lighting():
    """Lighting."""
    bpy.ops.object.light_add(type='SUN')

def setup_camera():
    """Camera."""
    bpy.ops.object.camera_add()

def bake_and_render():
    """Render."""
    bpy.context.scene.render.filepath = OUTPUT_DIR + "/test.png"
    bpy.ops.render.render(write_still=True)

setup_scene()
create_geometry()
setup_materials()
setup_physics()
setup_lighting()
setup_camera()
bake_and_render()
'''


class TestPatchScriptSection:
    @pytest.fixture
    def script_file(self, tmp_path):
        p = tmp_path / "test_script.py"
        p.write_text(SECTIONED_SCRIPT)
        return str(p)

    def test_patch_replaces_section(self, script_file):
        new_lighting = """def setup_lighting():
    bpy.ops.object.light_add(type='AREA', location=(0, 0, 5))
    light = bpy.context.active_object
    light.data.energy = 500"""
        result_json = _patch_script_section_impl(
            script_path=script_file,
            section_name="setup_lighting",
            new_code=new_lighting,
        )
        result = json.loads(result_json)
        assert result["success"] is True
        assert result["section_patched"] == "setup_lighting"
        # Read the output file
        patched = Path(result["script_path"]).read_text()
        assert "type='AREA'" in patched
        assert "energy = 500" in patched
        # Other sections preserved
        assert "def setup_scene():" in patched
        assert "def create_geometry():" in patched

    def test_patch_invalid_section_fails(self, script_file):
        result_json = _patch_script_section_impl(
            script_path=script_file,
            section_name="nonexistent_section",
            new_code="def nonexistent_section(): pass",
        )
        result = json.loads(result_json)
        assert result["success"] is False
        assert "not a canonical section" in result["error"]

    def test_patch_syntax_error_fails(self, script_file):
        bad_code = "def setup_lighting(\n    this is not valid python"
        result_json = _patch_script_section_impl(
            script_path=script_file,
            section_name="setup_lighting",
            new_code=bad_code,
        )
        result = json.loads(result_json)
        assert result["success"] is False
        assert "syntax" in result["error"].lower() or "compile" in result["error"].lower()

    def test_patch_creates_named_output(self, script_file):
        new_code = "def setup_lighting():\n    pass\n"
        result_json = _patch_script_section_impl(
            script_path=script_file,
            section_name="setup_lighting",
            new_code=new_code,
            output_name="test_patched_v2",
        )
        result = json.loads(result_json)
        assert result["success"] is True
        assert "test_patched_v2" in result["script_path"]

    def test_patch_preserves_preamble_and_epilogue(self, script_file):
        new_code = "def setup_physics():\n    bpy.ops.object.modifier_add(type='CLOTH')\n"
        result_json = _patch_script_section_impl(
            script_path=script_file,
            section_name="setup_physics",
            new_code=new_code,
        )
        result = json.loads(result_json)
        patched = Path(result["script_path"]).read_text()
        # Preamble
        assert "import bpy" in patched
        assert 'ASSET_NAME = "test_patch"' in patched
        # Epilogue
        assert "setup_scene()" in patched
        assert "bake_and_render()" in patched
