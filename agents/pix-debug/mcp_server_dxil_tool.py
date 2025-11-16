async def analyze_dxil_root_signature(args: dict) -> list[TextContent]:
    """
    Disassemble DXIL shader and extract root signature information.
    Compares expected root parameters against C++ definitions for Volumetric ReSTIR shaders.
    """
    dxil_path = args.get("dxil_path")
    shader_name = args.get("shader_name", "Unknown")

    if not dxil_path:
        return [TextContent(type="text", text=json.dumps({
            "error": "dxil_path parameter required",
            "usage": "analyze_dxil_root_signature on <path-to-dxil-file>"
        }, indent=2))]

    # Convert to Windows path if needed
    if dxil_path.startswith('/mnt/'):
        # WSL path to Windows path
        dxil_path_win = dxil_path.replace('/mnt/d/', 'D:\\').replace('/mnt/c/', 'C:\\').replace('/', '\\')
    else:
        dxil_path_win = dxil_path

    dxc_path = "/mnt/c/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64/dxc.exe"

    try:
        # Run dxc -dumpbin
        result = subprocess.run(
            [dxc_path, "-dumpbin", dxil_path_win],
            capture_output=True,
            text=True,
            timeout=10,
            cwd="/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean"
        )

        if result.returncode != 0:
            return [TextContent(type="text", text=json.dumps({
                "error": f"dxc.exe failed with return code {result.returncode}",
                "stderr": result.stderr,
                "dxil_path": dxil_path
            }, indent=2))]

        disassembly = result.stdout

        # Parse the disassembly
        analysis = {
            "dxil_path": dxil_path,
            "shader_name": shader_name,
            "shader_model": None,
            "thread_group_size": None,
            "resource_bindings": {
                "cbuffers": [],
                "srvs": [],
                "uavs": []
            },
            "issues": [],
            "recommendations": []
        }

        # Extract shader model and thread group size
        for line in disassembly.split('\n'):
            if 'NumThreads=' in line:
                import re
                match = re.search(r'NumThreads=\((\d+),(\d+),(\d+)\)', line)
                if match:
                    analysis["thread_group_size"] = f"({match.group(1)},{match.group(2)},{match.group(3)})"
            if 'target triple' in line:
                analysis["shader_model"] = line.strip()

        # Parse Resource Bindings section
        in_bindings = False
        for line in disassembly.split('\n'):
            if 'Resource Bindings:' in line:
                in_bindings = True
                continue
            elif in_bindings and line.strip().startswith(';'):
                # Comment line, skip header
                continue
            elif in_bindings and line.strip() == '':
                # Empty line marks end of bindings
                in_bindings = False
                continue
            elif in_bindings and not line.strip().startswith(';'):
                # Parse binding line
                parts = line.split()
                if len(parts) >= 6:
                    name = parts[0]
                    res_type = parts[1]
                    hlsl_bind = parts[-2] if len(parts) >= 7 else "unknown"
                    
                    binding_info = {
                        "name": name,
                        "type": res_type,
                        "register": hlsl_bind
                    }
                    
                    if res_type == "cbuffer":
                        analysis["resource_bindings"]["cbuffers"].append(binding_info)
                    elif res_type == "texture":
                        analysis["resource_bindings"]["srvs"].append(binding_info)
                    elif res_type == "UAV":
                        analysis["resource_bindings"]["uavs"].append(binding_info)

        # Expected bindings for Volumetric ReSTIR shaders
        expected_bindings = {
            "PopulateVolumeMip2": {
                "cbuffers": [{"name": "PopulationConstants", "register": "cb0"}],
                "srvs": [{"name": "g_particles", "register": "t0"}],
                "uavs": [
                    {"name": "g_volumeTexture", "register": "u0"},
                    {"name": "g_diagnosticCounters", "register": "u1"}
                ]
            },
            "GenerateCandidates": {
                "cbuffers": [{"name": "PathGenConstants", "register": "cb0"}],
                "srvs": [
                    {"name": "g_particles", "register": "t0"},
                    {"name": "g_volumeTexture", "register": "t1"}
                ],
                "uavs": [{"name": "g_reservoirs", "register": "u0"}]
            },
            "ShadeSelectedPaths": {
                "cbuffers": [{"name": "ShadingConstants", "register": "cb0"}],
                "srvs": [
                    {"name": "g_particles", "register": "t0"},
                    {"name": "g_volumeTexture", "register": "t1"},
                    {"name": "g_reservoirs", "register": "t2"}
                ],
                "uavs": [{"name": "g_outputTexture", "register": "u0"}]
            }
        }

        # Compare to expected bindings if we know the shader
        if shader_name in expected_bindings:
            expected = expected_bindings[shader_name]
            actual = analysis["resource_bindings"]

            # Check cbuffers
            for exp_cb in expected["cbuffers"]:
                found = any(cb["register"] == exp_cb["register"] for cb in actual["cbuffers"])
                if not found:
                    analysis["issues"].append({
                        "severity": "error",
                        "resource": exp_cb["name"],
                        "expected": f"CBV at {exp_cb['register']}",
                        "actual": "NOT FOUND",
                        "impact": "Shader will not execute - root signature mismatch"
                    })

            # Check SRVs
            for exp_srv in expected["srvs"]:
                found = any(srv["register"] == exp_srv["register"] for srv in actual["srvs"])
                if not found:
                    analysis["issues"].append({
                        "severity": "error",
                        "resource": exp_srv["name"],
                        "expected": f"SRV at {exp_srv['register']}",
                        "actual": "NOT FOUND",
                        "impact": "Shader will not execute - root signature mismatch"
                    })

            # Check UAVs
            for exp_uav in expected["uavs"]:
                found = any(uav["register"] == exp_uav["register"] for uav in actual["uavs"])
                if not found:
                    analysis["issues"].append({
                        "severity": "error",
                        "resource": exp_uav["name"],
                        "expected": f"UAV at {exp_uav['register']}",
                        "actual": "NOT FOUND",
                        "impact": "Shader will not execute - root signature mismatch"
                    })

        # Generate recommendations
        if analysis["issues"]:
            analysis["recommendations"].append("❌ CRITICAL: Resource binding mismatches detected")
            analysis["recommendations"].append("")
            analysis["recommendations"].append("Root Signature Mismatch Analysis:")
            
            for issue in analysis["issues"]:
                analysis["recommendations"].append(
                    f"  - {issue['resource']}: Expected {issue['expected']}, got {issue['actual']}"
                )
            
            analysis["recommendations"].append("")
            analysis["recommendations"].append("Next Steps:")
            analysis["recommendations"].append("1. Check C++ root signature creation in VolumetricReSTIRSystem.cpp")
            analysis["recommendations"].append("2. Verify root parameter order matches shader expectations")
            analysis["recommendations"].append("3. Ensure descriptor types match (root descriptor vs descriptor table)")
            analysis["recommendations"].append("4. Check SetComputeRoot*() calls bind to correct parameter indices")
            
        else:
            analysis["recommendations"].append("✅ All expected resources found in shader")
            analysis["recommendations"].append("")
            analysis["recommendations"].append("If shader still not executing:")
            analysis["recommendations"].append("1. Verify C++ root signature matches this layout")
            analysis["recommendations"].append("2. Check parameter ORDER (not just presence)")
            analysis["recommendations"].append("3. Verify descriptor TYPES (root descriptor vs descriptor table)")
            analysis["recommendations"].append("4. Use PIX to inspect actual bound resources")

        # Add resource summary
        analysis["summary"] = {
            "total_cbuffers": len(analysis["resource_bindings"]["cbuffers"]),
            "total_srvs": len(analysis["resource_bindings"]["srvs"]),
            "total_uavs": len(analysis["resource_bindings"]["uavs"]),
            "issues_found": len(analysis["issues"])
        }

        return [TextContent(type="text", text=json.dumps(analysis, indent=2))]

    except subprocess.TimeoutExpired:
        return [TextContent(type="text", text=json.dumps({
            "error": "dxc.exe timed out after 10 seconds",
            "dxil_path": dxil_path
        }, indent=2))]
    except Exception as e:
        return [TextContent(type="text", text=json.dumps({
            "error": f"Error analyzing DXIL: {str(e)}",
            "dxil_path": dxil_path
        }, indent=2))]
