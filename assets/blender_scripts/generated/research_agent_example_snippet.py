import bpy
import math

# ---------- Clean scene ----------
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.cycles.feature_set = 'SUPPORTED'
scene.cycles.samples = 128
scene.cycles.use_adaptive_sampling = True

# Color management: let HDR bloom/glare pop
scene.view_settings.view_transform = 'Filmic'
scene.view_settings.look = 'High Contrast'
scene.view_settings.exposure = 2.0

# ---------- Create Sun sphere ----------
bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, location=(0,0,0))
sun = bpy.context.active_object
sun.name = "Sun_Photosphere"

# ---------- Material: Photosphere (emission + noise) ----------
mat_sun = bpy.data.materials.new("MAT_SunPhotosphere")
mat_sun.use_nodes = True
nt = mat_sun.node_tree
nodes = nt.nodes
links = nt.links
nodes.clear()

out = nodes.new("ShaderNodeOutputMaterial")
out.location = (700, 0)

emis = nodes.new("ShaderNodeEmission")
emis.location = (450, 0)
links.new(emis.outputs["Emission"], out.inputs["Surface"])

# Coordinate + mapping (to animate noise drift)
texcoord = nodes.new("ShaderNodeTexCoord")
texcoord.location = (-800, 0)

mapping = nodes.new("ShaderNodeMapping")
mapping.location = (-600, 0)
mapping.vector_type = 'POINT'
links.new(texcoord.outputs["Object"], mapping.inputs["Vector"])

# Noise for granulation
noise = nodes.new("ShaderNodeTexNoise")
noise.location = (-350, 0)
noise.inputs["Scale"].default_value = 18.0
noise.inputs["Detail"].default_value = 8.0
noise.inputs["Roughness"].default_value = 0.5
links.new(mapping.outputs["Vector"], noise.inputs["Vector"])

# Color ramp: map noise -> hot spots
ramp = nodes.new("ShaderNodeValToRGB")
ramp.location = (-100, 0)
ramp.color_ramp.elements[0].position = 0.35
ramp.color_ramp.elements[1].position = 0.75
ramp.color_ramp.elements[0].color = (1.0, 0.25, 0.05, 1.0)  # darker orange
ramp.color_ramp.elements[1].color = (1.0, 1.0, 0.85, 1.0)   # near-white hot
links.new(noise.outputs["Fac"], ramp.inputs["Fac"])

# Use ramp color as emission color
links.new(ramp.outputs["Color"], emis.inputs["Color"])

# HDR-ish strength (Filmic will compress; glare uses bright values)
emis.inputs["Strength"].default_value = 80.0

sun.data.materials.append(mat_sun)

# Animate the mapping location (simple drift)
# (Optional, but makes it feel alive)
mapping.inputs["Location"].default_value = (0.0, 0.0, 0.0)
mapping.inputs["Location"].keyframe_insert(data_path="default_value", frame=1)
mapping.inputs["Location"].default_value = (0.6, 0.2, 0.1)
mapping.inputs["Location"].keyframe_insert(data_path="default_value", frame=120)

# ---------- Corona volume shell ----------
bpy.ops.mesh.primitive_uv_sphere_add(radius=1.15, location=(0,0,0))
corona = bpy.context.active_object
corona.name = "Sun_CoronaShell"

mat_corona = bpy.data.materials.new("MAT_SunCoronaVolume")
mat_corona.use_nodes = True
ntc = mat_corona.node_tree
nodes = ntc.nodes
links = ntc.links
nodes.clear()

outc = nodes.new("ShaderNodeOutputMaterial")
outc.location = (750, 0)

# Volume scatter for soft halo
vscatter = nodes.new("ShaderNodeVolumeScatter")
vscatter.location = (400, 70)
vscatter.inputs["Density"].default_value = 0.08
vscatter.inputs["Anisotropy"].default_value = 0.65  # forward scattering "glow"

# Add a bit of volume emission to keep corona bright
vemis = nodes.new("ShaderNodeVolumeEmission")
vemis.location = (400, -120)
vemis.inputs["Strength"].default_value = 2.0
vemis.inputs["Color"].default_value = (1.0, 0.75, 0.35, 1.0)

addv = nodes.new("ShaderNodeAddShader")
addv.location = (600, 0)

links.new(vscatter.outputs["Volume"], addv.inputs[0])
links.new(vemis.outputs["Volume"], addv.inputs[1])
links.new(addv.outputs["Shader"], outc.inputs["Volume"])

# Modulate density with a noise so corona isn’t uniform
texcoord2 = nodes.new("ShaderNodeTexCoord")
texcoord2.location = (-800, 0)
mapping2 = nodes.new("ShaderNodeMapping")
mapping2.location = (-600, 0)
links.new(texcoord2.outputs["Object"], mapping2.inputs["Vector"])

noise2 = nodes.new("ShaderNodeTexNoise")
noise2.location = (-350, 0)
noise2.inputs["Scale"].default_value = 6.0
noise2.inputs["Detail"].default_value = 10.0
noise2.inputs["Roughness"].default_value = 0.7
links.new(mapping2.outputs["Vector"], noise2.inputs["Vector"])

# Use noise to drive density (multiply)
mul = nodes.new("ShaderNodeMath")
mul.location = (120, 60)
mul.operation = 'MULTIPLY'
mul.inputs[1].default_value = 0.12
links.new(noise2.outputs["Fac"], mul.inputs[0])
links.new(mul.outputs["Value"], vscatter.inputs["Density"])

corona.data.materials.append(mat_corona)

# Make corona shell render as volume only (no surface)
corona.display_type = 'WIRE'

# ---------- World (dark space) ----------
world = bpy.data.worlds.new("World")
scene.world = world
world.use_nodes = True
wn = world.node_tree.nodes
wl = world.node_tree.links
for n in wn: wn.remove(n)
wout = wn.new("ShaderNodeOutputWorld")
wbg  = wn.new("ShaderNodeBackground")
wbg.inputs["Color"].default_value = (0,0,0,1)
wbg.inputs["Strength"].default_value = 1.0
wl.new(wbg.outputs["Background"], wout.inputs["Surface"])

# ---------- Camera ----------
bpy.ops.object.camera_add(location=(0, -4.0, 0.2), rotation=(math.radians(90), 0, 0))
cam = bpy.context.active_object
scene.camera = cam
cam.data.lens = 70

# ---------- Compositor: HDR Glare/Bloom ----------
scene.use_nodes = True
tree = scene.node_tree
nodes = tree.nodes
links = tree.links
nodes.clear()

rl = nodes.new("CompositorNodeRLayers")
rl.location = (-400, 0)

glare = nodes.new("CompositorNodeGlare")
glare.location = (-100, 0)
glare.glare_type = 'FOG_GLOW'     # good for bloom-like halos
glare.quality = 'HIGH'
glare.threshold = 1.0             # lower => more bloom
glare.size = 9                    # larger => softer/bigger glow
glare.mix = 0.0                   # 0 mixes original + glare (good default)

comp = nodes.new("CompositorNodeComposite")
comp.location = (250, 0)

viewer = nodes.new("CompositorNodeViewer")
viewer.location = (250, -150)

links.new(rl.outputs["Image"], glare.inputs["Image"])
links.new(glare.outputs["Image"], comp.inputs["Image"])
links.new(glare.outputs["Image"], viewer.inputs["Image"])

print("Sun photosphere + volumetric corona created.")