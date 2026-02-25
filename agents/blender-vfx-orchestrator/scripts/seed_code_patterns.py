"""
Seed Code Pattern Memory with proven techniques from successful wine_pour_v1.

These patterns are extracted from the wine_pour_v1_script_iter2_specfix.py script
which produced a near-photorealistic wine pour render. The key insight is that
the agent used Blender modifiers (Solidify, Subdivision Surface) and proper smooth
shading to achieve smooth, realistic glass surfaces — techniques the agent has
NOT been using in more recent runs.

Run from the orchestrator root:
    python scripts/seed_code_patterns.py
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.code_pattern_memory import get_pattern_memory


def seed_patterns():
    memory = get_pattern_memory()
    now_iso = __import__("datetime").datetime.now().isoformat()
    seeded = []

    # =========================================================================
    # Pattern 1: Glass mesh with Solidify + Subdivision Surface
    # =========================================================================
    pid = memory.record_successful_pattern(
        issue="glass mesh looks faceted, low-poly, unrealistic; wine glass appears as crude chalice",
        code_snippet="""# Build glass via bmesh profile spin, then add modifiers for quality
import bmesh, math
from mathutils import Vector

mesh = bpy.data.meshes.new("WineGlassMesh")
obj = bpy.data.objects.new("WineGlass", mesh)
bpy.context.collection.objects.link(obj)

bm = bmesh.new()
# Profile points (radius, height) — Bordeaux glass shape
profile = [
    (0.002, 0.000),  # center base
    (0.028, 0.004),  # base flare
    (0.030, 0.006),  # base edge
    (0.024, 0.008),  # base-to-stem taper
    (0.010, 0.010),  # stem start
    (0.007, 0.030),  # stem
    (0.007, 0.110),  # stem top
    (0.014, 0.125),  # bowl taper start
    (0.022, 0.135),  # bowl expanding
    (0.036, 0.150),  # bowl widening
    (0.048, 0.170),  # bowl near max
    (0.054, 0.190),  # max radius
    (0.052, 0.210),  # slight taper in
    (0.048, 0.220),  # rim
]
verts = [bm.verts.new((r, 0.0, z)) for r, z in profile]
edges = [bm.edges.new((verts[i], verts[i+1])) for i in range(len(verts)-1)]
bmesh.ops.spin(bm, geom=list(edges)+list(verts),
               cent=Vector((0,0,0)), axis=Vector((0,0,1)),
               angle=math.tau, steps=96)
bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.0001)
bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
bm.to_mesh(mesh)
bm.free()

# KEY: Solidify gives wall thickness, Subdivision smooths geometry
solid = obj.modifiers.new(name="Solidify", type='SOLIDIFY')
solid.thickness = 0.002   # 2mm glass wall
solid.offset = 1.0        # grow inward

subs = obj.modifiers.new(name="Subsurf", type='SUBSURF')
subs.levels = 2           # viewport
subs.render_levels = 2    # render

# Smooth shading eliminates faceting
for poly in obj.data.polygons:
    poly.use_smooth = True""",
        effect_type="water",
        improvement=30.0,
        experiment_id="seed_wine_pour_v1",
        name="glass_mesh_solidify_subdivision",
        context_before="Raw bmesh spin profile produces a faceted surface that looks like a low-poly chalice",
        context_after="Solidify + SubdivisionSurface + smooth shading produces photo-realistic glass with proper curvature and wall thickness",
    )
    seeded.append(("glass_mesh_solidify_subdivision", pid))

    # =========================================================================
    # Pattern 2: Glass material — Principled BSDF transmission
    # =========================================================================
    pid = memory.record_successful_pattern(
        issue="glass material looks opaque or plastic instead of transparent glass",
        code_snippet="""# Realistic glass material using Principled BSDF transmission
mat = bpy.data.materials.new("M_Glass")
mat.use_nodes = True
nt = mat.node_tree
# Clear defaults
for n in list(nt.nodes):
    nt.nodes.remove(n)
out = nt.nodes.new("ShaderNodeOutputMaterial")
bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
bsdf.inputs["Base Color"].default_value = (1, 1, 1, 1)       # White for clear glass
bsdf.inputs["Roughness"].default_value = 0.01                 # Near-perfect smooth
bsdf.inputs["Transmission Weight"].default_value = 1.0        # Fully transparent
bsdf.inputs["IOR"].default_value = 1.47                       # Glass IOR
nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
obj.data.materials.append(mat)""",
        effect_type="water",
        improvement=20.0,
        experiment_id="seed_wine_pour_v1",
        name="glass_material_principled_transmission",
        context_before="Flat color or missing transmission makes glass look like plastic or metal",
        context_after="Principled BSDF with Transmission Weight=1.0 and IOR=1.47 gives convincing clear glass with Fresnel reflections",
    )
    seeded.append(("glass_material_principled_transmission", pid))

    # =========================================================================
    # Pattern 3: Glass bottom collision disc (prevents liquid through stem)
    # =========================================================================
    pid = memory.record_successful_pattern(
        issue="liquid flows through glass stem; wine leaks out bottom of glass bowl",
        code_snippet="""# Invisible collision disc at bowl-stem junction prevents liquid escape
bpy.ops.mesh.primitive_cylinder_add(
    vertices=32,
    radius=0.022,   # Match inner bowl radius at junction
    depth=0.008,    # Thin disc
    location=(0.0, 0.0, 0.125),  # At bowl-stem junction height
)
disc = bpy.context.active_object
disc.name = "GlassBottom"

# Make invisible but still collide
disc.hide_render = True
disc.display_type = 'WIRE'

# Add as fluid effector for collision
mod = disc.modifiers.new(name='Fluid', type='FLUID')
mod.fluid_type = 'EFFECTOR'
mod.effector_settings.effector_type = 'COLLISION'""",
        effect_type="water",
        improvement=25.0,
        experiment_id="seed_wine_pour_v1",
        name="glass_bottom_collision_disc",
        context_before="Hollow glass mesh allows liquid to flow through stem, producing unrealistic leak",
        context_after="Hidden collision disc at bowl-stem junction contains liquid correctly; invisible to render",
    )
    seeded.append(("glass_bottom_collision_disc", pid))

    # =========================================================================
    # Pattern 4: Liquid mesh quality settings (smoothen, concave)
    # =========================================================================
    pid = memory.record_successful_pattern(
        issue="liquid surface looks blobby, chunky, low-resolution; no fine detail in splashes",
        code_snippet="""# Liquid domain mesh quality settings for smooth, detailed liquid surface
dset = mod.domain_settings
dset.domain_type = 'LIQUID'
dset.use_mesh = True           # Enable mesh generation (required for render)
dset.mesh_concave_upper = 3.0  # Concavity threshold — higher = more detail in concave areas
dset.mesh_smoothen_pos = 4     # Positive smoothing passes — smooths outward bumps
dset.mesh_smoothen_neg = 4     # Negative smoothing passes — smooths inward dents
dset.use_flip_particles = True # FLIP solver for better splashes
dset.flip_ratio = 0.95         # High ratio = more particles retained
dset.particle_radius = 1.0     # Default particle size""",
        effect_type="water",
        improvement=15.0,
        experiment_id="seed_wine_pour_v1",
        name="liquid_mesh_quality_smoothing",
        context_before="Default mesh generation produces chunky, blobby liquid surfaces with visible facets",
        context_after="mesh_smoothen_pos/neg=4 with mesh_concave_upper=3.0 produces smooth, detailed liquid surface",
    )
    seeded.append(("liquid_mesh_quality_smoothing", pid))

    # =========================================================================
    # Pattern 5: Smooth shading on all visible meshes
    # =========================================================================
    pid = memory.record_successful_pattern(
        issue="surfaces look faceted, angular; flat shading makes primitives look artificial",
        code_snippet="""# Apply smooth shading to mesh for better specular highlights and realism
# Method 1: Per-polygon (works on any mesh)
for poly in obj.data.polygons:
    poly.use_smooth = True

# Method 2: Bulk set via foreach_set (faster for large meshes)
mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))""",
        effect_type="water",
        improvement=10.0,
        experiment_id="seed_wine_pour_v1",
        name="smooth_shading_all_meshes",
        context_before="Default flat shading makes cubes/cylinders/spun profiles look faceted and artificial",
        context_after="Smooth shading on all visible geometry gives proper specular highlights and surface continuity",
    )
    seeded.append(("smooth_shading_all_meshes", pid))

    # =========================================================================
    # Pattern 6: Filmic color management for photorealism
    # =========================================================================
    pid = memory.record_successful_pattern(
        issue="render looks washed out, unrealistic colors, poor dynamic range",
        code_snippet="""# Filmic color management for photorealistic tone mapping
try:
    scene.view_settings.view_transform = 'Filmic'
    scene.view_settings.look = 'Medium High Contrast'
    scene.view_settings.exposure = 0.25
except Exception:
    pass  # Fall back to defaults if Filmic not available""",
        effect_type="water",
        improvement=10.0,
        experiment_id="seed_wine_pour_v1",
        name="filmic_color_management",
        context_before="Default color management produces washed-out, unrealistic renders with poor dynamic range",
        context_after="Filmic with Medium High Contrast gives cinematic look with better highlight rolloff and richer colors",
    )
    seeded.append(("filmic_color_management", pid))

    # =========================================================================
    # Pattern 7: Subdivision Surface for any smooth object
    # =========================================================================
    pid = memory.record_successful_pattern(
        issue="object surface looks low-poly, faceted, not smooth enough",
        code_snippet="""# Subdivision Surface modifier smooths any mesh geometry
# Use on: glass, bottles, curved containers, organic shapes, any smooth object
subs = obj.modifiers.new(name="Subsurf", type='SUBSURF')
subs.levels = 2           # Viewport subdivisions
subs.render_levels = 2    # Render subdivisions (can go to 3 for hero objects)

# ALWAYS combine with smooth shading
for poly in obj.data.polygons:
    poly.use_smooth = True

# Optional: add Solidify first if object needs wall thickness (glasses, bowls, shells)
# solid = obj.modifiers.new(name="Solidify", type='SOLIDIFY')
# solid.thickness = 0.002""",
        effect_type="water",
        improvement=20.0,
        experiment_id="seed_wine_pour_v1",
        name="subdivision_surface_smooth_objects",
        context_before="Primitive-based meshes (cubes, cylinders, spun profiles) show visible edges/facets",
        context_after="SubdivisionSurface level 2 + smooth shading produces curved, professional-looking surfaces",
    )
    # Mark this as applicable to ALL effect types
    pattern = memory.patterns[pid]
    pattern.effect_types = ["water", "fire", "smoke", "explosion", "nebula", "solar"]
    memory._save_pattern(pattern)
    seeded.append(("subdivision_surface_smooth_objects", pid))

    # =========================================================================
    # Summary
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"SEEDED {len(seeded)} CODE PATTERNS")
    print(f"{'='*60}")
    for name, pid in seeded:
        print(f"  {pid}: {name}")
    print(f"\nStorage: {memory.storage_dir}")
    print(f"Total patterns in memory: {len(memory.patterns)}")


if __name__ == "__main__":
    seed_patterns()
