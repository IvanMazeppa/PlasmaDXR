#!/usr/bin/env python3
"""Quick render of a single frame from a .blend file."""

import bpy
import sys
from pathlib import Path

def parse_args():
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    args = {"frame": 20, "output": None}
    i = 0
    while i < len(argv):
        if argv[i] == "--frame" and i + 1 < len(argv):
            args["frame"] = int(argv[i + 1])
            i += 2
        elif argv[i] == "--output" and i + 1 < len(argv):
            args["output"] = argv[i + 1]
            i += 2
        else:
            i += 1
    return args

args = parse_args()

# Set frame
bpy.context.scene.frame_set(args["frame"])

# Configure render
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.device = 'GPU'
scene.cycles.samples = 64  # Quick preview
scene.render.resolution_x = 512
scene.render.resolution_y = 512
scene.render.image_settings.file_format = 'PNG'

# Output path
if args["output"]:
    output_path = args["output"]
else:
    output_path = f"/home/maz3ppa/projects/PlasmaDXR/build/renders/frame_{args['frame']:04d}.png"

Path(output_path).parent.mkdir(parents=True, exist_ok=True)
scene.render.filepath = output_path

# Add camera if missing
if not scene.camera:
    bpy.ops.object.camera_add(location=(8, -8, 5))
    cam = bpy.context.active_object
    cam.rotation_euler = (1.1, 0, 0.8)
    scene.camera = cam

# Add light if missing
lights = [obj for obj in bpy.data.objects if obj.type == 'LIGHT']
if not lights:
    bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))

print(f"[render] Rendering frame {args['frame']} to {output_path}")
bpy.ops.render.render(write_still=True)
print(f"[render] Done: {output_path}")
