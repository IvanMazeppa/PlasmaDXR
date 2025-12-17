"""
Hydrogen Cloud (Mantaflow → OpenVDB) — Blender 5.x

This file is a *canonical copy* of the original recipe automation script that lives under
`docs/blender_recipes/scripts/` in older layouts.

For new work, prefer keeping Blender scripts under:
- `assets/blender_scripts/`
"""

import bpy
import math


def clear_scene():
    """Remove default objects."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()


def create_hydrogen_cloud(
    name: str = "HydrogenCloud",
    domain_size: float = 4.0,
    resolution: int = 64,
    frame_end: int = 100,
    cache_dir: str = "//vdb_cache/"
):
    """
    Create a complete hydrogen cloud nebula simulation.

    Args:
        name: Base name for objects
        domain_size: Size of domain cube in meters
        resolution: Voxel resolution (higher = more detail)
        frame_end: Last frame to simulate
        cache_dir: Directory for VDB cache files
    """
    print(f"[Nebula] Creating hydrogen cloud '{name}'...")

    # Create domain
    bpy.ops.mesh.primitive_cube_add(
        size=domain_size,
        location=(0, 0, domain_size / 2)
    )
    domain = bpy.context.active_object
    domain.name = f"{name}_Domain"

    # Add fluid modifier
    bpy.ops.object.modifier_add(type='FLUID')
    domain.modifiers['Fluid'].fluid_type = 'DOMAIN'
    settings = domain.modifiers['Fluid'].domain_settings

    # Domain settings
    settings.domain_type = 'GAS'
    settings.resolution_max = resolution
    settings.use_adaptive_domain = True

    # Gas settings
    settings.alpha = 0.5  # Buoyancy
    settings.beta = 1.0   # Heat buoyancy
    settings.vorticity = 0.3

    # Dissolve
    settings.use_dissolve_smoke = True
    settings.dissolve_speed = 80

    # VDB Export settings (Blender 5.x)
    #
    # NOTE:
    # - Some Blender 5 builds do NOT expose `cache_precision` on FluidDomainSettings.
    # - Keep this script cross-build by guarding optional properties.
    settings.cache_data_format = 'OPENVDB'
    if hasattr(settings, "openvdb_cache_compress_type"):
        settings.openvdb_cache_compress_type = 'ZIP'  # Options typically: ZIP, NONE
    if hasattr(settings, "cache_precision"):
        settings.cache_precision = 'HALF'  # Options may include: FULL, HALF, MINI (when present)
    settings.cache_directory = cache_dir
    settings.cache_frame_start = 1
    settings.cache_frame_end = frame_end

    print(f"  Domain created: {resolution}^3 resolution")

    # Create emitter
    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=domain_size * 0.1,
        location=(0, 0, domain_size * 0.4)
    )
    emitter = bpy.context.active_object
    emitter.name = f"{name}_Emitter"

    # Add flow modifier
    bpy.ops.object.modifier_add(type='FLUID')
    emitter.modifiers['Fluid'].fluid_type = 'FLOW'
    flow = emitter.modifiers['Fluid'].flow_settings

    flow.flow_type = 'SMOKE'
    flow.flow_behavior = 'INFLOW'
    flow.smoke_color = (0.8, 0.3, 0.4)  # Pink
    flow.temperature = 2.0
    flow.density = 1.0

    print(f"  Emitter created")

    # Create material
    mat = bpy.data.materials.new(name=f"{name}_Material")
    mat.use_nodes = True
    domain.data.materials.append(mat)

    # Set up volume shader
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear default nodes
    nodes.clear()

    # Add Principled Volume
    volume = nodes.new(type='ShaderNodeVolumePrincipled')
    volume.location = (0, 0)
    volume.inputs['Color'].default_value = (0.8, 0.4, 0.5, 1.0)
    volume.inputs['Density'].default_value = 0.8
    volume.inputs['Anisotropy'].default_value = -0.3
    volume.inputs['Absorption Color'].default_value = (0.1, 0.1, 0.2, 1.0)
    volume.inputs['Emission Strength'].default_value = 2.0
    volume.inputs['Emission Color'].default_value = (0.9, 0.4, 0.5, 1.0)

    # Add output
    output = nodes.new(type='ShaderNodeOutputMaterial')
    output.location = (300, 0)

    # Connect
    links.new(volume.outputs['Volume'], output.inputs['Volume'])

    print(f"  Material created")

    print(f"\n[Nebula] Setup complete!")
    print(f"  Domain: {domain.name}")
    print(f"  Emitter: {emitter.name}")
    print(f"  Cache: {cache_dir}")
    print(f"  Frames: 1-{frame_end}")
    print(f"\nNext steps:")
    print(f"  1. Save your .blend file (Ctrl+S)")
    print(f"  2. Select '{domain.name}'")
    print(f"  3. Go to Physics > Fluid > Cache")
    print(f"  4. Click 'Bake All'")
    print(f"  5. Find VDB files in {cache_dir}")

    return domain, emitter


# Run if executed directly
if __name__ == "__main__":
    clear_scene()
    domain, emitter = create_hydrogen_cloud(
        name="HydrogenCloud",
        resolution=64,
        frame_end=100
    )


