"""Tests for deterministic look-dev coverage metrics."""

import tempfile
from pathlib import Path

import pytest

from tools.script_analysis_tools import compute_lookdev_coverage, LookDevCoverage


def _write_script(content: str) -> str:
    """Write content to a temp .py file and return path."""
    f = tempfile.NamedTemporaryFile(suffix=".py", delete=False, mode="w")
    f.write(content)
    f.close()
    return f.name


# -- Color management --

def test_agx_detected():
    path = _write_script("""
scene.view_settings.view_transform = 'AgX'
""")
    cov = compute_lookdev_coverage(path)
    assert cov.has_color_management is True
    assert cov.color_management_type == "agx"


def test_filmic_detected():
    path = _write_script("""
scene.view_settings.view_transform = 'Filmic'
""")
    cov = compute_lookdev_coverage(path)
    assert cov.has_color_management is True
    assert cov.color_management_type == "filmic"


def test_no_color_management():
    path = _write_script("import bpy\nbpy.ops.render.render()")
    cov = compute_lookdev_coverage(path)
    assert cov.has_color_management is False


# -- DOF --

def test_dof_detected():
    path = _write_script("cam.data.dof.use_dof = True\n")
    cov = compute_lookdev_coverage(path)
    assert cov.has_dof is True


def test_no_dof():
    path = _write_script("cam.data.lens = 50\n")
    cov = compute_lookdev_coverage(path)
    assert cov.has_dof is False


# -- World setup --

def test_dark_world():
    path = _write_script("""
world = bpy.data.worlds.new('World')
world.color = (0.0, 0.0, 0.0)
""")
    cov = compute_lookdev_coverage(path)
    assert cov.has_world_setup is True
    assert cov.world_strategy == "dark_world"


def test_procedural_sky():
    path = _write_script("""
world = bpy.data.worlds['World']
world.use_nodes = True
sky = nodes.new('ShaderNodeTexSky')
""")
    cov = compute_lookdev_coverage(path)
    assert cov.has_world_setup is True
    assert cov.world_strategy == "procedural_sky"


# -- Materials --

def test_material_count():
    path = _write_script("""
mat1 = bpy.data.materials.new('Glass')
mat2 = bpy.data.materials.new('Metal')
mat3 = bpy.data.materials.new('Wood')
""")
    cov = compute_lookdev_coverage(path)
    assert cov.material_count == 3


def test_transmission_detected():
    path = _write_script("""
mat = bpy.data.materials.new('Glass')
principled.inputs['Transmission Weight'].default_value = 1.0
principled.inputs['IOR'].default_value = 1.45
""")
    cov = compute_lookdev_coverage(path)
    assert cov.has_transmission is True


def test_roughness_variation():
    path = _write_script("""
principled1.inputs['Roughness'].default_value = 0.1
principled2.inputs['Roughness'].default_value = 0.8
""")
    cov = compute_lookdev_coverage(path)
    assert cov.has_roughness_variation is True


# -- Modifiers (hero refinement) --

def test_subdivision_detected():
    path = _write_script("""
mod = obj.modifiers.new('Subsurf', type='SUBSURF')
mod.levels = 2
""")
    cov = compute_lookdev_coverage(path)
    assert cov.has_subdivision is True
    assert cov.modifier_count >= 1


def test_bevel_detected():
    path = _write_script("""
mod = obj.modifiers.new('Bevel', type='BEVEL')
mod.width = 0.02
""")
    cov = compute_lookdev_coverage(path)
    assert cov.has_bevel is True


def test_multiple_modifiers():
    path = _write_script("""
obj.modifiers.new('Subsurf', type='SUBSURF')
obj.modifiers.new('Bevel', type='BEVEL')
obj.modifiers.new('Solidify', type='SOLIDIFY')
""")
    cov = compute_lookdev_coverage(path)
    assert cov.modifier_count >= 3
    assert cov.has_subdivision
    assert cov.has_bevel
    assert cov.has_solidify


# -- Atmosphere --

def test_volume_scatter():
    path = _write_script("""
vol = nodes.new('ShaderNodeVolumeScatter')
""")
    cov = compute_lookdev_coverage(path)
    assert cov.has_atmosphere is True


# -- Lighting --

def test_light_count():
    path = _write_script("""
key = bpy.data.lights.new('Key', type='AREA')
fill = bpy.data.lights.new('Fill', type='POINT')
rim = bpy.data.lights.new('Rim', type='SPOT')
""")
    cov = compute_lookdev_coverage(path)
    assert cov.light_count == 3
    assert "AREA" in cov.light_types
    assert "POINT" in cov.light_types
    assert "SPOT" in cov.light_types


# -- Coverage score --

def test_minimal_script_low_score():
    path = _write_script("import bpy\nbpy.ops.render.render()\n")
    cov = compute_lookdev_coverage(path)
    assert cov.coverage_score < 20


def test_rich_script_high_score():
    path = _write_script("""
import bpy

scene = bpy.context.scene
scene.view_settings.view_transform = 'AgX'

cam = bpy.data.cameras.new('Camera')
cam.dof.use_dof = True

world = bpy.data.worlds.new('World')
world.use_nodes = True
bg = nodes.new('ShaderNodeBackground')

mat1 = bpy.data.materials.new('Glass')
mat2 = bpy.data.materials.new('Metal')
principled = nodes.new('ShaderNodeBsdfPrincipled')
principled.inputs['Transmission Weight'].default_value = 1.0
principled.inputs['Roughness'].default_value = 0.05
principled2.inputs['Roughness'].default_value = 0.7

obj.modifiers.new('Subsurf', type='SUBSURF')
obj.modifiers.new('Bevel', type='BEVEL')

vol = nodes.new('ShaderNodeVolumeScatter')

key = bpy.data.lights.new('Key', type='AREA')
fill = bpy.data.lights.new('Fill', type='POINT')
rim = bpy.data.lights.new('Rim', type='SPOT')
""")
    cov = compute_lookdev_coverage(path)
    assert cov.coverage_score >= 70


def test_nonexistent_file():
    cov = compute_lookdev_coverage("/nonexistent/file.py")
    assert cov.coverage_score == 0.0
    assert cov.material_count == 0
