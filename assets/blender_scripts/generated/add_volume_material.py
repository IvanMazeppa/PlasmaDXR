#!/usr/bin/env python3
"""Add Principled Volume material to fluid domain for fire/smoke visualization."""

import bpy

# Find domain
domain = None
for obj in bpy.data.objects:
    if obj.type == 'MESH':
        for mod in obj.modifiers:
            if mod.type == 'FLUID' and mod.fluid_type == 'DOMAIN':
                domain = obj
                break

if not domain:
    print("[material] ERROR: No fluid domain found!")
else:
    print(f"[material] Found domain: {domain.name}")

    # Create volumetric material
    mat = bpy.data.materials.new(name="FireSmokeMaterial")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    # Clear default nodes
    nodes.clear()

    # Add Principled Volume shader
    volume = nodes.new('ShaderNodeVolumePrincipled')
    volume.location = (0, 0)
    volume.inputs['Color'].default_value = (1.0, 0.3, 0.05, 1.0)  # Orange-red
    volume.inputs['Density'].default_value = 2.0
    volume.inputs['Blackbody Intensity'].default_value = 2.0
    volume.inputs['Temperature'].default_value = 1800.0

    # Add output node
    output = nodes.new('ShaderNodeOutputMaterial')
    output.location = (300, 0)

    # Connect
    links.new(volume.outputs['Volume'], output.inputs['Volume'])

    # Assign to domain
    if domain.data.materials:
        domain.data.materials[0] = mat
    else:
        domain.data.materials.append(mat)

    print("[material] Volumetric material applied!")

# Hide emitter
for obj in bpy.data.objects:
    if 'Emitter' in obj.name or 'Flow' in obj.name:
        obj.hide_render = True
        print(f"[material] Hidden emitter: {obj.name}")

# Save
bpy.ops.wm.save_mainfile()
print("[material] Saved!")
