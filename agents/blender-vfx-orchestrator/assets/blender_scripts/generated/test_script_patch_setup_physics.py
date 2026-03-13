import bpy

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
    bpy.ops.object.modifier_add(type='CLOTH')

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
