"""
Extract all relevant settings from a .blend file for comparison/reproduction.
Outputs material nodes, render settings, color management, and fluid settings.
"""
import bpy
import json

def extract_settings():
    """Extract all settings from current .blend file."""
    settings = {
        "render": {},
        "color_management": {},
        "materials": {},
        "fluid_domain": {},
        "camera": {},
        "world": {}
    }

    # === Render Settings ===
    scene = bpy.context.scene
    render = scene.render
    settings["render"] = {
        "engine": render.engine,
        "resolution_x": render.resolution_x,
        "resolution_y": render.resolution_y,
        "resolution_percentage": render.resolution_percentage,
        "film_transparent": render.film_transparent,
    }

    # Cycles-specific
    if hasattr(scene, 'cycles'):
        cycles = scene.cycles
        settings["render"]["cycles"] = {
            "samples": cycles.samples,
            "preview_samples": cycles.preview_samples,
            "use_denoising": cycles.use_denoising,
            "volume_step_rate": cycles.volume_step_rate,
            "volume_preview_step_rate": cycles.volume_preview_step_rate,
            "volume_max_steps": cycles.volume_max_steps,
        }

    # === Color Management ===
    view = scene.view_settings
    settings["color_management"] = {
        "view_transform": view.view_transform,
        "look": view.look,
        "exposure": view.exposure,
        "gamma": view.gamma,
        "use_curve_mapping": view.use_curve_mapping,
    }

    # === Materials ===
    for mat in bpy.data.materials:
        if mat.use_nodes and mat.node_tree:
            mat_data = {
                "name": mat.name,
                "nodes": []
            }
            for node in mat.node_tree.nodes:
                node_data = {
                    "type": node.type,
                    "name": node.name,
                    "bl_idname": node.bl_idname,
                    "location": list(node.location),
                    "inputs": {},
                    "outputs": {}
                }

                # Extract input values
                for inp in node.inputs:
                    if hasattr(inp, 'default_value'):
                        val = inp.default_value
                        if hasattr(val, '__iter__') and not isinstance(val, str):
                            node_data["inputs"][inp.name] = list(val)
                        else:
                            node_data["inputs"][inp.name] = val

                # Special handling for Color Ramp
                if node.type == 'VALTORGB' and hasattr(node, 'color_ramp'):
                    ramp = node.color_ramp
                    node_data["color_ramp"] = {
                        "interpolation": ramp.interpolation,
                        "elements": []
                    }
                    for elem in ramp.elements:
                        node_data["color_ramp"]["elements"].append({
                            "position": elem.position,
                            "color": list(elem.color)
                        })

                mat_data["nodes"].append(node_data)

            # Extract links
            mat_data["links"] = []
            for link in mat.node_tree.links:
                mat_data["links"].append({
                    "from_node": link.from_node.name,
                    "from_socket": link.from_socket.name,
                    "to_node": link.to_node.name,
                    "to_socket": link.to_socket.name
                })

            settings["materials"][mat.name] = mat_data

    # === Fluid Domain Settings ===
    for obj in bpy.data.objects:
        if obj.type == 'MESH':
            for mod in obj.modifiers:
                if mod.type == 'FLUID':
                    if mod.fluid_type == 'DOMAIN':
                        domain = mod.domain_settings
                        settings["fluid_domain"] = {
                            "object_name": obj.name,
                            "domain_type": domain.domain_type,
                            "resolution_max": domain.resolution_max,
                            "time_scale": domain.time_scale,
                            "cfl_condition": domain.cfl_condition,
                            "use_adaptive_domain": domain.use_adaptive_domain,
                            # Fire settings
                            "burning_rate": domain.burning_rate,
                            "flame_smoke": domain.flame_smoke,
                            "flame_vorticity": domain.flame_vorticity,
                            "flame_max_temp": domain.flame_max_temp,
                            "flame_smoke_color": list(domain.flame_smoke_color),
                            # Noise
                            "use_noise": domain.use_noise,
                            "noise_scale": domain.noise_scale if domain.use_noise else None,
                            "noise_strength": domain.noise_strength if domain.use_noise else None,
                        }
                    elif mod.fluid_type == 'FLOW':
                        flow = mod.flow_settings
                        if "fluid_flows" not in settings:
                            settings["fluid_flows"] = []
                        settings["fluid_flows"].append({
                            "object_name": obj.name,
                            "flow_type": flow.flow_type,
                            "flow_behavior": flow.flow_behavior,
                            "temperature": flow.temperature,
                            "fuel_amount": flow.fuel_amount,
                        })

    # === World Settings ===
    world = scene.world
    if world and world.use_nodes:
        settings["world"]["use_nodes"] = True
        for node in world.node_tree.nodes:
            if node.type == 'BACKGROUND':
                settings["world"]["background_color"] = list(node.inputs['Color'].default_value)
                settings["world"]["background_strength"] = node.inputs['Strength'].default_value

    # === Camera ===
    cam = scene.camera
    if cam and cam.data:
        cam_data = cam.data
        settings["camera"] = {
            "name": cam.name,
            "type": cam_data.type,
            "location": list(cam.location),
            "rotation": list(cam.rotation_euler),
        }
        if cam_data.type == 'ORTHO':
            settings["camera"]["ortho_scale"] = cam_data.ortho_scale
        else:
            settings["camera"]["lens"] = cam_data.lens

    return settings


if __name__ == "__main__":
    settings = extract_settings()

    # Pretty print
    print("=" * 80)
    print("EXTRACTED BLEND FILE SETTINGS")
    print("=" * 80)
    print(json.dumps(settings, indent=2, default=str))
    print("=" * 80)

    # Also save to file
    output_path = bpy.path.abspath("//extracted_settings.json")
    with open(output_path, 'w') as f:
        json.dump(settings, f, indent=2, default=str)
    print(f"Settings saved to: {output_path}")
