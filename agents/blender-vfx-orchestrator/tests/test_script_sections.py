"""Tests for script section parsing, extraction, and patching."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from utils.script_sections import (
    CANONICAL_SECTIONS,
    parse_sections,
    extract_section,
    replace_section,
    validate_section_structure,
    list_sections,
)


MINIMAL_SECTIONED_SCRIPT = '''
import bpy

ASSET_NAME = "test_asset"

def setup_scene():
    """Scene cleanup and render settings."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    bpy.context.scene.render.engine = 'CYCLES'

def create_geometry():
    """Create meshes."""
    bpy.ops.mesh.primitive_cube_add(size=1)

def setup_materials():
    """Create materials."""
    mat = bpy.data.materials.new("Base")
    mat.use_nodes = True

def setup_physics():
    """Physics setup."""
    pass

def setup_lighting():
    """Lighting."""
    bpy.ops.object.light_add(type='SUN')

def setup_camera():
    """Camera."""
    bpy.ops.object.camera_add()

def bake_and_render():
    """Bake and render."""
    bpy.context.scene.render.filepath = "/tmp/test.png"
    bpy.ops.render.render(write_still=True)

# Main execution
setup_scene()
create_geometry()
setup_materials()
setup_physics()
setup_lighting()
setup_camera()
bake_and_render()
'''

UNSECTIONED_SCRIPT = '''
import bpy

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
bpy.ops.mesh.primitive_cube_add(size=1)
bpy.ops.render.render(write_still=True)
'''

PARTIAL_SECTIONED_SCRIPT = '''
import bpy

def setup_scene():
    bpy.ops.object.select_all(action='SELECT')

def setup_lighting():
    bpy.ops.object.light_add(type='SUN')

# Inline code — no section
bpy.ops.mesh.primitive_cube_add(size=1)
bpy.ops.render.render(write_still=True)
'''


class TestParseSections:
    def test_finds_all_canonical_sections(self):
        sections = parse_sections(MINIMAL_SECTIONED_SCRIPT)
        assert len(sections) == 7
        for name in CANONICAL_SECTIONS:
            assert name in sections

    def test_unsectioned_returns_empty(self):
        sections = parse_sections(UNSECTIONED_SCRIPT)
        assert len(sections) == 0

    def test_partial_finds_existing_only(self):
        sections = parse_sections(PARTIAL_SECTIONED_SCRIPT)
        assert "setup_scene" in sections
        assert "setup_lighting" in sections
        assert "create_geometry" not in sections

    def test_section_has_line_range(self):
        sections = parse_sections(MINIMAL_SECTIONED_SCRIPT)
        scene = sections["setup_scene"]
        assert "start_line" in scene
        assert "end_line" in scene
        assert scene["start_line"] < scene["end_line"]

    def test_section_has_source(self):
        sections = parse_sections(MINIMAL_SECTIONED_SCRIPT)
        scene = sections["setup_scene"]
        assert "source" in scene
        assert "bpy.ops.object.select_all" in scene["source"]


class TestExtractSection:
    def test_extract_existing(self):
        source = extract_section(MINIMAL_SECTIONED_SCRIPT, "setup_lighting")
        assert "light_add" in source

    def test_extract_missing_returns_none(self):
        source = extract_section(MINIMAL_SECTIONED_SCRIPT, "nonexistent_section")
        assert source is None

    def test_extract_from_unsectioned_returns_none(self):
        source = extract_section(UNSECTIONED_SCRIPT, "setup_scene")
        assert source is None


class TestReplaceSection:
    def test_replace_preserves_other_sections(self):
        new_lighting = """def setup_lighting():
    \"\"\"Better lighting.\"\"\"
    bpy.ops.object.light_add(type='AREA', location=(0, 0, 5))
    light = bpy.context.active_object
    light.data.energy = 500
"""
        result = replace_section(MINIMAL_SECTIONED_SCRIPT, "setup_lighting", new_lighting)
        # New section present
        assert "type='AREA'" in result
        assert "energy = 500" in result
        # Old sections preserved
        assert "def setup_scene():" in result
        assert "def create_geometry():" in result
        assert "def setup_physics():" in result
        assert "def setup_camera():" in result
        assert "def bake_and_render():" in result
        # Old lighting gone
        assert "type='SUN'" not in result

    def test_replace_missing_section_raises(self):
        with pytest.raises(KeyError):
            replace_section(MINIMAL_SECTIONED_SCRIPT, "nonexistent", "def nonexistent(): pass")

    def test_replace_keeps_preamble(self):
        new_scene = """def setup_scene():
    pass
"""
        result = replace_section(MINIMAL_SECTIONED_SCRIPT, "setup_scene", new_scene)
        assert "import bpy" in result
        assert 'ASSET_NAME = "test_asset"' in result

    def test_replace_is_syntactically_valid(self):
        new_lighting = """def setup_lighting():
    bpy.ops.object.light_add(type='POINT')
"""
        result = replace_section(MINIMAL_SECTIONED_SCRIPT, "setup_lighting", new_lighting)
        compile(result, "<test>", "exec")  # Should not raise


class TestValidateSectionStructure:
    def test_fully_sectioned_passes(self):
        missing, found = validate_section_structure(MINIMAL_SECTIONED_SCRIPT)
        assert len(missing) == 0
        assert len(found) == 7

    def test_unsectioned_reports_all_missing(self):
        missing, found = validate_section_structure(UNSECTIONED_SCRIPT)
        assert len(missing) == 7
        assert len(found) == 0

    def test_partial_reports_missing(self):
        missing, found = validate_section_structure(PARTIAL_SECTIONED_SCRIPT)
        assert "create_geometry" in missing
        assert "setup_scene" in found


class TestListSections:
    def test_list_returns_names_and_lines(self):
        result = list_sections(MINIMAL_SECTIONED_SCRIPT)
        assert len(result) == 7
        assert all("name" in s and "start_line" in s for s in result)
