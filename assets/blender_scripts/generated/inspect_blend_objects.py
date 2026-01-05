#!/usr/bin/env python3
"""
Inspect a loaded .blend and print object/material/render visibility info.

Usage:
  blender --background some.blend --python inspect_blend_objects.py
"""

import bpy


def _fluid_modifier_summary(obj):
    out = []
    for m in obj.modifiers:
        if m.type != "FLUID":
            continue
        try:
            ft = m.fluid_type
        except Exception:
            ft = "<?>"
        out.append(ft)
    return out


print("=" * 80)
print("BLEND OBJECT INSPECTION")
print("=" * 80)

scene = bpy.context.scene
print(f"[scene] frame range: {scene.frame_start}-{scene.frame_end}")
print(f"[scene] render engine: {scene.render.engine}")
print("")

objs = list(bpy.data.objects)
print(f"[objects] count: {len(objs)}")

for obj in sorted(objs, key=lambda o: o.name.lower()):
    mats = []
    if getattr(obj, "data", None) and hasattr(obj.data, "materials"):
        mats = [m.name for m in obj.data.materials if m is not None]

    print(f"- {obj.name:32s} type={obj.type:6s} hide_render={obj.hide_render} hide_viewport={obj.hide_viewport}")
    if mats:
        print(f"    materials: {mats}")
    fm = _fluid_modifier_summary(obj)
    if fm:
        print(f"    fluid_mods: {fm}")

print("=" * 80)


