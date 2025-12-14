import bpy
import math
import os

def create_bipolar_nebula():
    # 1. Setup Scene
    bpy.ops.wm.read_factory_settings(use_empty=True)
    
    # Save the file to enable relative paths for baking
    # Using absolute path to ensure we know where it is
    project_root = os.path.abspath(os.path.dirname(__file__))
    output_blend = os.path.join(project_root, "bipolar_nebula.blend")
    bpy.ops.wm.save_as_mainfile(filepath=output_blend)
    
    scene = bpy.context.scene
    scene.frame_start = 1
    scene.frame_end = 120
    
    # 2. Create Domain
    bpy.ops.mesh.primitive_cube_add(size=12, enter_editmode=False, align='WORLD', location=(0, 0, 0))
    domain_obj = bpy.context.active_object
    domain_obj.name = "Nebula_Domain"
    
    # Add Fluid Modifier - DOMAIN
    fluid_mod = domain_obj.modifiers.new(name="Fluid", type='FLUID')
    fluid_mod.fluid_type = 'DOMAIN'
    
    # Configure Domain Settings
    domain_settings = fluid_mod.domain_settings
    domain_settings.domain_type = 'GAS'
    domain_settings.resolution_max = 128  # decent quality
    domain_settings.use_adaptive_domain = True
    domain_settings.use_noise = True
    domain_settings.noise_scale = 2.0
    
    # Cache Settings
    domain_settings.cache_type = 'ALL'
    domain_settings.cache_data_format = 'OPENVDB'
    # Use relative path for portability
    domain_settings.cache_directory = "//VDBs/NanoVDB/BipolarNebula/"
    
    # Material for Volume (Basic Setup)
    mat = bpy.data.materials.new(name="Nebula_Volume")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    
    # Add Principled Volume
    node_vol = nodes.new(type='ShaderNodeVolumePrincipled')
    node_vol.inputs['Density'].default_value = 0.5
    node_out = nodes.new(type='ShaderNodeOutputMaterial')
    
    links = mat.node_tree.links
    links.new(node_vol.outputs['Volume'], node_out.inputs['Volume'])
    
    domain_obj.data.materials.append(mat)
    
    # 3. Create Emitters (Two Jets)
    # Emitter 1 (Top Jet)
    bpy.ops.mesh.primitive_cone_add(radius1=1, radius2=0, depth=2, location=(0, 0, 1))
    emitter1 = bpy.context.active_object
    emitter1.name = "Emitter_Top"
    
    # Add Fluid Modifier - FLOW
    fmod1 = emitter1.modifiers.new(name="Fluid", type='FLUID')
    fmod1.fluid_type = 'FLOW'
    flow1 = fmod1.flow_settings
    flow1.flow_type = 'SMOKE' # or BOTH for fire+smoke
    flow1.flow_behavior = 'INFLOW'
    flow1.density = 2.0
    flow1.velocity_coord = (0, 0, 10) # Shoot up
    flow1.use_initial_velocity = True
    
    # Animate Emitter 1 Rotation (Swirl)
    emitter1.rotation_mode = 'XYZ'
    emitter1.keyframe_insert(data_path="rotation_euler", frame=1)
    emitter1.rotation_euler = (0.2, 0, 0) # Slight tilt
    emitter1.keyframe_insert(data_path="rotation_euler", frame=120)
    
    # Emitter 2 (Bottom Jet)
    bpy.ops.mesh.primitive_cone_add(radius1=1, radius2=0, depth=2, location=(0, 0, -1))
    emitter2 = bpy.context.active_object
    emitter2.name = "Emitter_Bottom"
    emitter2.rotation_euler = (math.pi, 0, 0) # Point down
    
    # Add Fluid Modifier - FLOW
    fmod2 = emitter2.modifiers.new(name="Fluid", type='FLUID')
    fmod2.fluid_type = 'FLOW'
    flow2 = fmod2.flow_settings
    flow2.flow_type = 'SMOKE'
    flow2.flow_behavior = 'INFLOW'
    flow2.density = 2.0
    flow2.velocity_coord = (0, 0, 10) # Local Z is down now? No, velocity_coord is global or local? 
    # Usually velocity_coord is added to object velocity. 
    # Let's use 'use_initial_velocity' with the object's normal/velocity, 
    # but 'velocity_coord' is explicit. 
    # Actually, let's just use Normal velocity for simplicity + consistency
    flow2.velocity_normal = 5.0 
    # And fix Flow 1 too
    flow1.velocity_coord = (0,0,0)
    flow1.velocity_normal = 5.0
    
    # Animate Emitter 2 Rotation
    emitter2.rotation_mode = 'XYZ'
    emitter2.keyframe_insert(data_path="rotation_euler", frame=1)
    emitter2.rotation_euler = (math.pi - 0.2, 0, 0)
    emitter2.keyframe_insert(data_path="rotation_euler", frame=120)

    # 4. Add Turbulence Force Field
    bpy.ops.object.effector_add(type='TURBULENCE', location=(0, 0, 0))
    turb = bpy.context.active_object
    turb.field.strength = 1.5
    turb.field.noise = 1.0
    
    # 5. Bake (Optional - Uncomment to bake immediately)
    print("Scene setup complete. Ready to bake.")
    print(f"Resolution: {domain_settings.resolution_max}")
    print(f"Cache Path: {os.path.abspath(domain_settings.cache_directory)}")
    
    # To bake automatically:
    # bpy.ops.fluid.bake_all()

if __name__ == "__main__":
    create_bipolar_nebula()

