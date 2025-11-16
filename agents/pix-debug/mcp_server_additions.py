# Addition to list_tools() - insert before the closing ]
        Tool(
            name="validate_shader_execution",
            description="Validate that compute shaders are actually executing by analyzing diagnostic counters, dispatch logs, and buffer states. Critical for detecting silent shader execution failures (shaders dispatch but don't run). Parses PopulateVolumeMip2 diagnostic counters to confirm GPU execution.",
            inputSchema={
                "type": "object",
                "properties": {
                    "log_path": {"type": "string", "description": "Path to log file (optional, uses latest if not provided)"},
                    "buffer_dir": {"type": "string", "description": "Path to buffer dump directory (optional)"}
                },
                "required": []
            }
        ),

# Addition to call_tool() - insert before the else clause
    elif name == "validate_shader_execution":
        return await validate_shader_execution(arguments)

# New function implementation - add at end of file before main()
async def validate_shader_execution(args: dict) -> list[TextContent]:
    """
    Validate that compute shaders are actually executing by checking:
    1. Diagnostic counter values (should be non-zero if shader runs)
    2. Dispatch logs (verify dispatches occurred)
    3. Common shader execution failure patterns
    """
    log_path = args.get("log_path")
    buffer_dir = args.get("buffer_dir", BUFFER_DUMP_DIR)

    if not log_path:
        # Find latest log
        log_dir = os.path.join(PLASMA_DX_PATH, "build/bin/Debug/logs")
        if os.path.exists(log_dir):
            log_files = sorted(
                [os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.startswith("PlasmaDX")],
                key=lambda x: os.path.getmtime(x),
                reverse=True
            )
            if log_files:
                log_path = log_files[0]

    if not log_path or not os.path.exists(log_path):
        return [TextContent(type="text", text=json.dumps({
            "error": "No log file found",
            "searched_directory": os.path.join(PLASMA_DX_PATH, "build/bin/Debug/logs") if not log_path else None
        }, indent=2))]

    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            log_lines = f.readlines()

        analysis = {
            "log_path": log_path,
            "log_timestamp": datetime.fromtimestamp(os.path.getmtime(log_path)).isoformat(),
            "shaders_analyzed": [],
            "execution_failures": [],
            "dispatches_found": [],
            "recommendations": []
        }

        # Look for PopulateVolumeMip2 diagnostic counters
        for i, line in enumerate(log_lines):
            if "PopulateVolumeMip2 Diagnostic Counters" in line:
                # Extract next 4-5 lines
                counter_lines = log_lines[i:i+6]

                # Parse counter values
                import re
                total_threads = None
                early_returns = None
                voxel_writes = None
                max_voxels = None

                for cl in counter_lines:
                    if "[0] Total threads executed:" in cl:
                        match = re.search(r': (\d+)', cl)
                        if match:
                            total_threads = int(match.group(1))
                    elif "[1] Early returns:" in cl:
                        match = re.search(r': (\d+)', cl)
                        if match:
                            early_returns = int(match.group(1))
                    elif "[2] Total voxel writes:" in cl:
                        match = re.search(r': (\d+)', cl)
                        if match:
                            voxel_writes = int(match.group(1))
                    elif "[3] Max voxels per particle:" in cl:
                        match = re.search(r': (\d+)', cl)
                        if match:
                            max_voxels = int(match.group(1))

                shader_result = {
                    "shader": "PopulateVolumeMip2",
                    "counters": {
                        "total_threads": total_threads,
                        "early_returns": early_returns,
                        "voxel_writes": voxel_writes,
                        "max_voxels_per_particle": max_voxels
                    },
                    "status": "unknown",
                    "severity": "info"
                }

                # Analyze results
                if total_threads == 0 and early_returns == 0 and voxel_writes == 0:
                    shader_result["status"] = "NOT_EXECUTING"
                    shader_result["severity"] = "critical"
                    analysis["execution_failures"].append({
                        "shader": "PopulateVolumeMip2",
                        "issue": "Shader dispatched but never executed (all counters zero)",
                        "likely_causes": [
                            "Root signature mismatch between C++ and compiled DXIL",
                            "PSO binding failure (silent failure in D3D12)",
                            "Resource binding slot collision"
                        ],
                        "severity": "critical"
                    })
                elif total_threads > 0 and voxel_writes == 0:
                    shader_result["status"] = "EXECUTING_BUT_NO_OUTPUT"
                    shader_result["severity"] = "warning"
                    analysis["execution_failures"].append({
                        "shader": "PopulateVolumeMip2",
                        "issue": "Shader executing but producing no output",
                        "likely_causes": [
                            "Early return condition triggered for all threads",
                            "UAV binding failure (can't write to volume texture)",
                            "Invalid voxel bounds (all particles outside volume)"
                        ],
                        "severity": "warning"
                    })
                else:
                    shader_result["status"] = "EXECUTING_NORMALLY"
                    shader_result["severity"] = "info"

                analysis["shaders_analyzed"].append(shader_result)

        # Look for dispatch calls
        for line in log_lines:
            if "Dispatching" in line and "thread groups" in line:
                analysis["dispatches_found"].append(line.strip())

        # Look for --restir flag confirmation
        restir_enabled = False
        for line in log_lines:
            if "Lighting system: Volumetric ReSTIR" in line:
                restir_enabled = True
                break

        analysis["volumetric_restir_enabled"] = restir_enabled

        # Check for resource state errors
        state_errors = []
        for line in log_lines:
            if "Resource state" in line and ("ERROR" in line or "WARNING" in line):
                state_errors.append(line.strip())

        if state_errors:
            analysis["resource_state_errors"] = state_errors

        # Generate recommendations
        if analysis["execution_failures"]:
            critical_failures = [f for f in analysis["execution_failures"] if f["severity"] == "critical"]

            if critical_failures:
                analysis["recommendations"].extend([
                    "❌ CRITICAL: Shader execution failure detected",
                    "",
                    "Root Cause Analysis Steps:",
                    "1. Use 'analyze_dxil_root_signature' tool on PopulateVolumeMip2 shader",
                    "2. Compare DXIL root signature to C++ CreateRootSignature() code",
                    "3. Check for parameter order, type, or binding mismatches",
                    "",
                    "Quick Fix Attempts:",
                    "1. Force shader recompilation: Delete .dxil files and rebuild",
                    "2. Embed root signature in HLSL using [RootSignature(...)] attribute",
                    "3. Try minimal test shader (write-only UAV) to isolate issue",
                    "",
                    "Expected Values (at 2045 particles with 32 thread groups):",
                    "  [0] Total threads executed: 2048 (64 × 32)",
                    "  [1] Early returns: 3 (2048 - 2045)",
                    "  [2] Total voxel writes: 500,000+ (depends on particle distribution)",
                    "  [3] Max voxels per particle: ~512 (8×8×8 limit)"
                ])
            else:
                # Warning-level failures
                analysis["recommendations"].extend([
                    "⚠️  WARNING: Shader executing but behavior abnormal",
                    "",
                    "Check UAV bindings in C++ code:",
                    "1. Verify volume texture is bound to correct UAV slot (u0)",
                    "2. Check resource state transitions (SRV↔UAV)",
                    "3. Verify voxel bounds calculation (WorldToVoxel function)",
                    "",
                    "Check shader logic:",
                    "1. Review early return conditions",
                    "2. Verify AABB clamping to [0,63] range",
                    "3. Check density calculation (should be > 0.0001 for some particles)"
                ])

        elif not analysis["shaders_analyzed"]:
            analysis["recommendations"].append(
                "ℹ️  No diagnostic counters found in log. Either:"
            )
            analysis["recommendations"].append(
                "  1. PopulateVolumeMip2 was not dispatched (check if ReSTIR is enabled)"
            )
            analysis["recommendations"].append(
                "  2. Diagnostic instrumentation not added to shader yet"
            )
            analysis["recommendations"].append(
                "  3. Log file is from run without diagnostic counters"
            )

            if not restir_enabled:
                analysis["recommendations"].append("")
                analysis["recommendations"].append(
                    "⚠️  Volumetric ReSTIR not enabled in this run. Use '--restir' flag."
                )
        else:
            # No failures
            analysis["recommendations"].extend([
                "✅ All shaders executing normally",
                "",
                "If you're experiencing other issues:",
                "1. Check reservoir buffer contents with 'analyze_volumetric_restir_reservoirs'",
                "2. Create PIX capture to analyze GPU timeline",
                "3. Review visual output for artifacts"
            ])

        # Add dispatch summary
        if analysis["dispatches_found"]:
            analysis["dispatch_summary"] = {
                "total_dispatches": len(analysis["dispatches_found"]),
                "sample_dispatches": analysis["dispatches_found"][:5]  # First 5 examples
            }
            # Don't include full list in main output to avoid clutter
            del analysis["dispatches_found"]

        return [TextContent(type="text", text=json.dumps(analysis, indent=2))]

    except Exception as e:
        return [TextContent(type="text", text=json.dumps({
            "error": f"Error validating shader execution: {str(e)}",
            "log_path": log_path
        }, indent=2))]

