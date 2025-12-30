#!/usr/bin/env python3
"""
Export Blender mesh as raw binary format for PlasmaDX-Clean water rendering.

Binary format:
- Header: uint32 vertex_count, uint32 index_count
- Vertices: interleaved float3 position + float3 normal (6 floats per vertex)
- Indices: uint32 triangle indices

Usage in Blender:
    import sys
    sys.path.append('/path/to/PlasmaDXR/scripts')
    from export_water_mesh import export_water_frame, export_water_sequence

    # Export single frame
    export_water_frame("water_mesh.bin", bpy.context.active_object)

    # Export animation sequence
    export_water_sequence("water_cache/", bpy.context.active_object, 1, 100)
"""

import struct
import os

def export_water_frame(filepath, obj, apply_modifiers=True):
    """
    Export a mesh object as raw binary format.

    Args:
        filepath: Output path for .bin file
        obj: Blender mesh object
        apply_modifiers: Apply modifiers before export (default True)

    Returns:
        Tuple of (vertex_count, index_count) on success, None on failure
    """
    try:
        import bpy
        import bmesh
    except ImportError:
        print("Error: Must run from within Blender")
        return None

    if obj is None:
        print("Error: No object provided")
        return None

    if obj.type != 'MESH':
        print(f"Error: Object '{obj.name}' is not a mesh (type: {obj.type})")
        return None

    # Get evaluated mesh (with modifiers applied)
    if apply_modifiers:
        depsgraph = bpy.context.evaluated_depsgraph_get()
        obj_eval = obj.evaluated_get(depsgraph)
        mesh = obj_eval.to_mesh()
    else:
        mesh = obj.data.copy()

    if mesh is None or len(mesh.polygons) == 0:
        print(f"  No mesh geometry at this frame")
        if apply_modifiers:
            obj_eval.to_mesh_clear()
        return None

    # Triangulate using bmesh
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])

    # Collect data directly from bmesh (more reliable)
    vertices = []
    for v in bm.verts:
        # Position
        vertices.extend([v.co.x, v.co.y, v.co.z])
        # Normal
        vertices.extend([v.normal.x, v.normal.y, v.normal.z])

    # Collect triangle indices
    indices = []
    for face in bm.faces:
        if len(face.verts) != 3:
            print(f"Warning: Non-triangle face found (vertices: {len(face.verts)})")
            continue
        indices.extend([v.index for v in face.verts])

    vertex_count = len(bm.verts)
    index_count = len(indices)

    bm.free()

    # Clean up evaluated mesh if we created one
    if apply_modifiers:
        obj_eval.to_mesh_clear()

    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)

    with open(filepath, 'wb') as f:
        # Header
        f.write(struct.pack('II', vertex_count, index_count))
        # Vertices (6 floats per vertex)
        f.write(struct.pack(f'{len(vertices)}f', *vertices))
        # Indices
        f.write(struct.pack(f'{len(indices)}I', *indices))

    print(f"Exported: {filepath}")
    print(f"  Vertices: {vertex_count}")
    print(f"  Triangles: {index_count // 3}")
    print(f"  File size: {os.path.getsize(filepath)} bytes")

    return (vertex_count, index_count)


def export_water_sequence(output_dir, obj, frame_start, frame_end, name_pattern="water_frame_{:04d}.bin"):
    """
    Export an animated mesh sequence.

    Args:
        output_dir: Directory to write frame files
        obj: Blender mesh object (should have animation/simulation)
        frame_start: First frame to export
        frame_end: Last frame to export (inclusive)
        name_pattern: Filename pattern with {} for frame number

    Returns:
        Number of frames exported
    """
    try:
        import bpy
    except ImportError:
        print("Error: Must run from within Blender")
        return 0

    os.makedirs(output_dir, exist_ok=True)

    original_frame = bpy.context.scene.frame_current
    exported = 0

    for frame in range(frame_start, frame_end + 1):
        bpy.context.scene.frame_set(frame)
        filepath = os.path.join(output_dir, name_pattern.format(frame))

        result = export_water_frame(filepath, obj)
        if result:
            exported += 1
            print(f"Frame {frame}/{frame_end} exported")

    # Restore original frame
    bpy.context.scene.frame_set(original_frame)

    print(f"\nSequence export complete: {exported} frames")
    return exported


# Command-line interface for blender --background mode
def find_liquid_domain():
    """Find the liquid domain object in the scene."""
    import bpy
    for obj in bpy.context.scene.objects:
        if obj.type == 'MESH':
            for mod in obj.modifiers:
                if mod.type == 'FLUID':
                    if hasattr(mod, 'fluid_type') and mod.fluid_type == 'DOMAIN':
                        return obj
                    if hasattr(mod, 'domain_settings'):
                        if hasattr(mod.domain_settings, 'domain_type'):
                            if mod.domain_settings.domain_type == 'LIQUID':
                                return obj
    return None


def main():
    """Command-line entry point for blender --background mode."""
    import sys

    # Parse arguments after '--'
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    output_dir = None
    frame = None
    frame_start = None
    frame_end = None
    step = 1
    object_name = None

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--output_dir" and i + 1 < len(argv):
            output_dir = argv[i + 1]
            i += 2
        elif arg == "--frame" and i + 1 < len(argv):
            frame = int(argv[i + 1])
            i += 2
        elif arg == "--frame_start" and i + 1 < len(argv):
            frame_start = int(argv[i + 1])
            i += 2
        elif arg == "--frame_end" and i + 1 < len(argv):
            frame_end = int(argv[i + 1])
            i += 2
        elif arg == "--step" and i + 1 < len(argv):
            step = int(argv[i + 1])
            i += 2
        elif arg == "--object" and i + 1 < len(argv):
            object_name = argv[i + 1]
            i += 2
        else:
            i += 1

    # If no CLI args, print usage
    if not output_dir and not argv:
        print("Water Mesh Exporter for PlasmaDX-Clean")
        print("")
        print("CLI Usage:")
        print("  blender --background sim.blend --python export_water_mesh.py -- --output_dir ./meshes --frame 50")
        print("  blender --background sim.blend --python export_water_mesh.py -- --output_dir ./meshes --frame_start 20 --frame_end 95 --step 5")
        print("")
        print("Python Usage (in Blender console):")
        print("  from export_water_mesh import export_water_frame")
        print("  export_water_frame('output.bin', bpy.context.active_object)")
        return

    import bpy

    print("=" * 60)
    print("Water Mesh Exporter")
    print("=" * 60)

    # Find mesh object
    if object_name:
        obj = bpy.data.objects.get(object_name)
        if not obj:
            print(f"ERROR: Object '{object_name}' not found")
            return
    else:
        obj = find_liquid_domain()
        if not obj:
            # Try active object as fallback
            obj = bpy.context.active_object
            if not obj or obj.type != 'MESH':
                print("ERROR: No liquid domain found. Use --object to specify mesh name.")
                return

    print(f"Mesh object: {obj.name}")
    print(f"Output: {output_dir}")

    # Determine frames to export
    if frame is not None:
        # Single frame mode
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"water_mesh_{frame:04d}.bin")
        bpy.context.scene.frame_set(frame)
        export_water_frame(filepath, obj)
    elif frame_start is not None and frame_end is not None:
        # Sequence mode
        print(f"Frames: {frame_start} - {frame_end} (step {step})")
        os.makedirs(output_dir, exist_ok=True)
        exported = 0
        for f in range(frame_start, frame_end + 1, step):
            bpy.context.scene.frame_set(f)
            filepath = os.path.join(output_dir, f"water_mesh_{f:04d}.bin")
            result = export_water_frame(filepath, obj)
            if result:
                exported += 1
        print(f"\nExported {exported} frames")
    else:
        # Current frame
        os.makedirs(output_dir, exist_ok=True)
        frame = bpy.context.scene.frame_current
        filepath = os.path.join(output_dir, f"water_mesh_{frame:04d}.bin")
        export_water_frame(filepath, obj)


if __name__ == "__main__":
    main()
