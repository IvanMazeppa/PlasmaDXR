#!/usr/bin/env python3
"""
PIX Debugging MCP Server for Claude Code
Run this to expose DXR debugging tools directly in your Claude Code session
"""

import asyncio
import json
import os
import subprocess
import struct
import time
from datetime import datetime
from typing import Any, Dict

import numpy as np
import pyautogui
from dotenv import load_dotenv
from mcp.server import Server
from mcp.types import TextContent, Tool

# Load environment variables
load_dotenv()

# Configuration
PIXTOOL_PATH = os.getenv("PIXTOOL_PATH", "/mnt/c/Program Files/Microsoft PIX/2509.25/pixtool.exe")
PLASMA_DX_PATH = os.getenv("PLASMA_DX_PATH", "/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean")
BUFFER_DUMP_DIR = os.getenv("BUFFER_DUMP_DIR", os.path.join(PLASMA_DX_PATH, "PIX/buffer_dumps"))
PIX_CAPTURES_DIR = os.getenv("PIX_CAPTURES_DIR", os.path.join(PLASMA_DX_PATH, "PIX/Captures"))

# Create MCP server
server = Server("pix-debug-tools")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available DXR debugging tools"""
    return [
        Tool(
            name="capture_buffers",
            description="Trigger in-app buffer dump from PlasmaDX-Clean at specific frame",
            inputSchema={
                "type": "object",
                "properties": {
                    "frame": {"type": "integer", "description": "Frame number to capture (optional for manual Ctrl+D)"},
                    "mode": {"type": "string", "description": "Rendering mode (gaussian/traditional/billboard)", "default": "gaussian"},
                    "output_dir": {"type": "string", "description": "Custom output directory (optional)"}
                },
                "required": []
            }
        ),
        Tool(
            name="analyze_restir_reservoirs",
            description="Parse and analyze ReSTIR reservoir buffers for lighting statistics",
            inputSchema={
                "type": "object",
                "properties": {
                    "current_path": {"type": "string", "description": "Path to g_currentReservoirs.bin"},
                    "sample_size": {"type": "integer", "description": "Number of reservoirs to sample", "default": 1000}
                },
                "required": ["current_path"]
            }
        ),
        Tool(
            name="analyze_particle_buffers",
            description="Validate particle buffer data (position, velocity, lifetime)",
            inputSchema={
                "type": "object",
                "properties": {
                    "particles_path": {"type": "string", "description": "Path to g_particles.bin"},
                    "expected_count": {"type": "integer", "description": "Expected particle count (optional)"}
                },
                "required": ["particles_path"]
            }
        ),
        Tool(
            name="pix_capture",
            description="Create PIX .wpix capture using pixtool for deep GPU analysis",
            inputSchema={
                "type": "object",
                "properties": {
                    "frames": {"type": "integer", "description": "Number of frames to capture", "default": 1},
                    "output_name": {"type": "string", "description": "Output filename (auto-generated if not provided)"},
                    "auto_open": {"type": "boolean", "description": "Open capture in PIX GUI", "default": False}
                },
                "required": []
            }
        ),
        Tool(
            name="pix_list_captures",
            description="List available PIX .wpix captures with metadata",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": []
            }
        ),
        Tool(
            name="diagnose_visual_artifact",
            description="Automated diagnosis of rendering issues from symptom description",
            inputSchema={
                "type": "object",
                "properties": {
                    "symptom": {"type": "string", "description": "Description of visual artifact (e.g., 'black dots at far distances')"},
                    "buffer_dump_dir": {"type": "string", "description": "Path to buffer dump directory (optional)"}
                },
                "required": ["symptom"]
            }
        ),
        Tool(
            name="analyze_dxil_root_signature",
            description="Disassemble DXIL shader and extract root signature for debugging resource binding mismatches. Critical for diagnosing shader execution failures. Analyzes cbuffers, SRVs, UAVs and compares to expected Volumetric ReSTIR bindings.",
            inputSchema={
                "type": "object",
                "properties": {
                    "dxil_path": {"type": "string", "description": "Path to compiled DXIL shader file (e.g., build/bin/Debug/shaders/volumetric_restir/populate_volume_mip2.dxil)"},
                    "shader_name": {"type": "string", "description": "Shader name for context (PopulateVolumeMip2, GenerateCandidates, or ShadeSelectedPaths)", "default": "Unknown"}
                },
                "required": ["dxil_path"]
            }
        ),
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
        Tool(
            name="diagnose_gpu_hang",
            description="Autonomous GPU hang/TDR crash diagnosis - launches PlasmaDX with specified settings, monitors for crashes, captures logs, and analyzes failure patterns. Ideal for debugging compute shader hangs, resource state issues, and particle count thresholds.",
            inputSchema={
                "type": "object",
                "properties": {
                    "particle_count": {"type": "integer", "description": "Number of particles to test (e.g., 2045)", "default": 2045},
                    "render_mode": {"type": "string", "description": "Rendering mode: gaussian, volumetric_restir, multi_light", "default": "volumetric_restir"},
                    "timeout_seconds": {"type": "integer", "description": "Timeout in seconds before considering app hung (default: 10)", "default": 10},
                    "test_threshold": {"type": "boolean", "description": "Test multiple particle counts around threshold (e.g., 2040, 2044, 2045, 2050)", "default": False},
                    "capture_logs": {"type": "boolean", "description": "Capture application logs for analysis", "default": True}
                },
                "required": []
            }
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Execute a debugging tool"""

    if name == "capture_buffers":
        return await capture_buffers(arguments)
    elif name == "analyze_restir_reservoirs":
        return await analyze_restir_reservoirs(arguments)
    elif name == "analyze_particle_buffers":
        return await analyze_particle_buffers(arguments)
    elif name == "pix_capture":
        return await pix_capture(arguments)
    elif name == "pix_list_captures":
        return await pix_list_captures(arguments)
    elif name == "diagnose_visual_artifact":
        return await diagnose_visual_artifact(arguments)
    elif name == "diagnose_gpu_hang":
        return await diagnose_gpu_hang(arguments)
    elif name == "validate_shader_execution":
        return await validate_shader_execution(arguments)
    elif name == "analyze_dxil_root_signature":
        return await analyze_dxil_root_signature(arguments)
    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def capture_buffers(args: dict) -> list[TextContent]:
    """Capture GPU buffers using in-app buffer dumping"""
    frame = args.get("frame")
    mode = args.get("mode", "gaussian")
    output_dir = args.get("output_dir")

    exe_path = os.path.join(PLASMA_DX_PATH, "build/bin/Debug/PlasmaDX-Clean.exe")
    dump_dir = output_dir or BUFFER_DUMP_DIR

    cmd = [exe_path, "--dump-buffers"]
    if frame is not None:
        cmd.append(str(frame))
    if mode:
        cmd.append(f"--{mode}")
    if output_dir:
        cmd.extend(["--dump-dir", output_dir])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=PLASMA_DX_PATH)

        metadata_path = os.path.join(dump_dir, "metadata.json")
        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

        captured_files = []
        if os.path.exists(dump_dir):
            captured_files = [f for f in os.listdir(dump_dir) if f.endswith('.bin') or f == 'metadata.json']

        result_data = {
            "status": "success" if result.returncode == 0 else "failed",
            "frame": frame,
            "dump_dir": dump_dir,
            "captured_files": captured_files,
            "file_count": len(captured_files),
            "metadata": metadata
        }

        return [TextContent(type="text", text=json.dumps(result_data, indent=2))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error capturing buffers: {str(e)}")]


async def analyze_restir_reservoirs(args: dict) -> list[TextContent]:
    """Analyze ReSTIR lighting reservoirs"""
    current_path = args.get("current_path")
    sample_size = args.get("sample_size", 1000)

    if not current_path:
        return [TextContent(type="text", text="Error: current_path is required")]

    try:
        with open(current_path, 'rb') as f:
            data = f.read()

        num_reservoirs = len(data) // 32
        reservoirs = []

        for i in range(0, min(num_reservoirs, sample_size)):
            offset = i * 32
            lightPos = struct.unpack('fff', data[offset:offset+12])
            weightSum = struct.unpack('f', data[offset+12:offset+16])[0]
            M = struct.unpack('I', data[offset+16:offset+20])[0]
            W = struct.unpack('f', data[offset+20:offset+24])[0]

            reservoirs.append({'lightPos': lightPos, 'weightSum': weightSum, 'M': M, 'W': W})

        W_values = [r['W'] for r in reservoirs]
        M_values = [r['M'] for r in reservoirs]
        weightSum_values = [r['weightSum'] for r in reservoirs]

        stats = {
            'total_reservoirs': num_reservoirs,
            'sampled': len(reservoirs),
            'W': {
                'mean': float(np.mean(W_values)),
                'std': float(np.std(W_values)),
                'min': float(np.min(W_values)),
                'max': float(np.max(W_values)),
                'median': float(np.median(W_values))
            },
            'M': {
                'mean': float(np.mean(M_values)),
                'std': float(np.std(M_values)),
                'min': int(np.min(M_values)),
                'max': int(np.max(M_values)),
                'median': float(np.median(M_values))
            }
        }

        issues = []
        recommendations = []

        if stats['M']['mean'] < 5.0:
            issues.append(f"Low average M = {stats['M']['mean']:.1f} (expect >8 for good convergence)")
            recommendations.append("Check spatial reuse: ensure neighbor search radius is adequate")

        if stats['W']['max'] > 0.01:
            issues.append(f"Very large W values detected (max={stats['W']['max']:.6f})")
            recommendations.append("Possible overflow - consider clamping W")

        result = {
            "summary": f"Analyzed {num_reservoirs} ReSTIR reservoirs. Avg W={stats['W']['mean']:.6f}, Avg M={stats['M']['mean']:.1f}",
            "statistics": stats,
            "issues_found": issues,
            "recommendations": recommendations
        }

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        return [TextContent(type="text", text=f"Error analyzing reservoirs: {str(e)}")]


async def analyze_particle_buffers(args: dict) -> list[TextContent]:
    """Analyze particle buffers"""
    particles_path = args.get("particles_path")

    if not particles_path:
        return [TextContent(type="text", text="Error: particles_path is required")]

    try:
        with open(particles_path, 'rb') as f:
            data = f.read()

        particle_size = 32
        num_particles = len(data) // particle_size

        positions = []
        for i in range(min(num_particles, 1000)):
            offset = i * particle_size
            pos = struct.unpack('fff', data[offset:offset+12])
            positions.append(pos)

        positions_arr = np.array(positions)

        result = {
            "total_particles": num_particles,
            "sampled": len(positions),
            "position_mean": positions_arr.mean(axis=0).tolist(),
            "position_std": positions_arr.std(axis=0).tolist()
        }

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def pix_capture(args: dict) -> list[TextContent]:
    """Create PIX capture"""
    frames = args.get("frames", 1)
    output_name = args.get("output_name")

    if output_name is None:
        timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
        output_name = f"mcp_capture_{timestamp}.wpix"

    output_path = os.path.join(PIX_CAPTURES_DIR, output_name)
    exe_path = os.path.join(PLASMA_DX_PATH, "build/bin/Debug/PlasmaDX-Clean.exe")

    cmd = [
        PIXTOOL_PATH, "launch", exe_path,
        f"--working-directory={PLASMA_DX_PATH}",
        "take-capture", f"--frames={frames}",
        "save-capture", output_path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        result_data = {
            "status": "success" if result.returncode == 0 else "failed",
            "capture_path": output_path,
            "exists": os.path.exists(output_path)
        }

        return [TextContent(type="text", text=json.dumps(result_data, indent=2))]
    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def pix_list_captures(args: dict) -> list[TextContent]:
    """List PIX captures"""
    captures = []
    if os.path.exists(PIX_CAPTURES_DIR):
        for filename in os.listdir(PIX_CAPTURES_DIR):
            if filename.endswith('.wpix'):
                filepath = os.path.join(PIX_CAPTURES_DIR, filename)
                stat = os.stat(filepath)
                captures.append({
                    'name': filename,
                    'size_mb': round(stat.st_size / (1024*1024), 2),
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                })

    captures = sorted(captures, key=lambda x: x['modified'], reverse=True)

    return [TextContent(type="text", text=json.dumps({
        "total_captures": len(captures),
        "captures": captures[:20]
    }, indent=2))]


async def diagnose_visual_artifact(args: dict) -> list[TextContent]:
    """Diagnose visual artifacts"""
    symptom = args.get("symptom", "").lower()

    if "black dot" in symptom or "dark particle" in symptom:
        result = {
            "summary": "Black dots typically caused by ReSTIR under-sampling (low M values)",
            "recommendations": [
                "Scale contribution by sqrt(M):",
                "float visibility_scale = sqrt(reservoir.M / 16.0);",
                "float3 contribution = reservoir.lightPos * reservoir.W * visibility_scale;"
            ]
        }
    elif "overexposure" in symptom:
        result = {
            "summary": "Overexposure from weight overflow",
            "recommendations": ["Clamp W values: W = min(W, 0.01)"]
        }
    else:
        result = {"summary": f"Unknown symptom: {symptom}"}

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


async def main():
    """Run MCP server"""
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())


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


async def diagnose_gpu_hang(args: dict) -> list[TextContent]:
    """
    Autonomous GPU hang diagnosis - launches PlasmaDX with specified settings,
    monitors for crashes/hangs, and provides diagnostic recommendations.
    
    Specifically designed for debugging:
    - Compute shader hangs (like PopulateVolumeMip2)
    - Resource state transition issues
    - Particle count thresholds
    - TDR crashes
    """
    particle_count = args.get("particle_count", 2045)
    render_mode = args.get("render_mode", "volumetric_restir")
    timeout_seconds = args.get("timeout_seconds", 10)
    test_threshold = args.get("test_threshold", False)
    capture_logs = args.get("capture_logs", True)
    
    exe_path = os.path.join(PLASMA_DX_PATH, "build/bin/Debug/PlasmaDX-Clean.exe")
    exe_dir = os.path.dirname(exe_path)
    log_dir = os.path.join(exe_dir, "logs")  # Logs written relative to exe location
    
    results = {
        "test_config": {
            "particle_count": particle_count,
            "render_mode": render_mode,
            "timeout_seconds": timeout_seconds,
            "test_threshold": test_threshold
        },
        "tests": [],
        "analysis": {},
        "recommendations": []
    }
    
    # Determine particle counts to test
    if test_threshold:
        # Test around threshold (e.g., 2040, 2044, 2045, 2048, 2050)
        test_counts = [
            particle_count - 5,
            particle_count - 1,
            particle_count,
            particle_count + 3,
            particle_count + 5
        ]
    else:
        test_counts = [particle_count]
    
    for count in test_counts:
        test_result = {
            "particle_count": count,
            "status": "unknown",
            "runtime_seconds": 0,
            "crash_detected": False,
            "hang_detected": False,
            "logs_captured": []
        }
        
        # Build command (launches in Gaussian mode by default)
        cmd = [exe_path, "--particles", str(count), "--restir"]
        
        try:
            # Launch process
            start_time = datetime.now()
            
            # Get exe directory for correct working directory (shader paths)
            exe_dir = os.path.dirname(exe_path)
            
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=exe_dir  # Run from debug folder, not project root
            )
            
            # Wait for window initialization
            time.sleep(3.0)  # Increased for window focus
            
            # Press F7 using PowerShell SendKeys (WSL2 → Windows compatible)
            try:
                # PowerShell script to activate window and send F7
                ps_script = '''
$wshell = New-Object -ComObject wscript.shell
Start-Sleep -Milliseconds 500
if ($wshell.AppActivate("PlasmaDX-Clean")) {
    Start-Sleep -Milliseconds 300
    $wshell.SendKeys("{F7}")
} else {
    Write-Host "Failed to activate PlasmaDX window"
}
'''
                result = subprocess.run(
                    ["powershell.exe", "-Command", ps_script],
                    capture_output=True,
                    text=True,
                    timeout=3
                )
                time.sleep(1.0)  # Wait for mode switch
                if result.stderr:
                    test_result["keyboard_automation_warning"] = result.stderr
            except Exception as e:
                test_result["keyboard_automation_error"] = str(e)
            
            try:
                # Wait for process with timeout
                stdout, stderr = proc.communicate(timeout=timeout_seconds)
                end_time = datetime.now()
                
                test_result["runtime_seconds"] = (end_time - start_time).total_seconds()
                test_result["status"] = "completed"
                
                # Check return code
                if proc.returncode != 0:
                    test_result["crash_detected"] = True
                    test_result["status"] = "crashed"
                    test_result["exit_code"] = proc.returncode
                
            except subprocess.TimeoutExpired:
                # Process exceeded timeout - kill it forcefully
                # Windows: Kill by image name to catch any child processes
                try:
                    kill_result = subprocess.run(
                        ["taskkill.exe", "/F", "/IM", "PlasmaDX-Clean.exe"], 
                        capture_output=True, 
                        text=True,
                        timeout=5
                    )
                    # Also kill the subprocess handle
                    proc.kill()
                    proc.wait(timeout=2)
                except:
                    # Best effort - process might already be dead
                    pass
                
                end_time = datetime.now()
                test_result["runtime_seconds"] = (end_time - start_time).total_seconds()
                test_result["hang_detected"] = False  # Not a hang, just timeout
                test_result["status"] = "timeout"
                
        except Exception as e:
            test_result["status"] = "error"
            test_result["error"] = str(e)
        
        # Capture latest log file if requested
        if capture_logs and os.path.exists(log_dir):
            log_files = sorted(
                [f for f in os.listdir(log_dir) if f.startswith("PlasmaDX")],
                key=lambda x: os.path.getmtime(os.path.join(log_dir, x)),
                reverse=True
            )
            
            if log_files:
                latest_log = os.path.join(log_dir, log_files[0])
                try:
                    with open(latest_log, 'r', encoding='utf-8', errors='ignore') as f:
                        # Read first 50 lines (initialization) + last 100 lines (hang analysis)
                        lines = f.readlines()
                        first_lines = lines[:50] if len(lines) > 50 else lines
                        last_lines = lines[-100:] if len(lines) > 100 else lines
                        
                        # Combine first and last, avoiding duplicates if log is short
                        if len(lines) <= 150:
                            captured = lines
                        else:
                            captured = first_lines + ["... (middle section omitted) ..."] + last_lines
                        
                        test_result["logs_captured"] = [line.strip() for line in captured if line.strip()]
                except:
                    pass
        
        results["tests"].append(test_result)
    
    # Analyze results
    hung_counts = [t["particle_count"] for t in results["tests"] if t["hang_detected"]]
    crashed_counts = [t["particle_count"] for t in results["tests"] if t["crash_detected"]]
    success_counts = [t["particle_count"] for t in results["tests"] if t["status"] == "completed" and not t["crash_detected"]]
    
    results["analysis"] = {
        "total_tests": len(results["tests"]),
        "hung_tests": len(hung_counts),
        "crashed_tests": len(crashed_counts),
        "successful_tests": len(success_counts),
        "hung_particle_counts": hung_counts,
        "crashed_particle_counts": crashed_counts,
        "success_particle_counts": success_counts
    }
    
    # Generate recommendations
    if test_threshold and hung_counts:
        # Find exact threshold
        min_hung = min(hung_counts)
        max_success = max(success_counts) if success_counts else 0
        
        if max_success > 0 and min_hung > max_success:
            results["analysis"]["crash_threshold"] = f"{max_success} → {min_hung}"
            results["recommendations"].append(
                f"CRITICAL: GPU hang threshold identified between {max_success} and {min_hung} particles"
            )
            results["recommendations"].append(
                f"This is exactly {min_hung - max_success} particles difference - suggests power-of-2 boundary issue"
            )
            
            # Check if near power of 2
            import math
            nearest_power = 2 ** math.ceil(math.log2(min_hung))
            if abs(nearest_power - min_hung) <= 5:
                results["recommendations"].append(
                    f"Threshold {min_hung} is near power-of-2 boundary {nearest_power} - likely GPU scheduling issue"
                )
    
    elif hung_counts:
        results["recommendations"].append(
            f"GPU hang detected at {hung_counts[0]} particles - likely compute shader infinite loop or resource deadlock"
        )
    
    # Analyze log patterns
    for test in results["tests"]:
        if test["hang_detected"] and test["logs_captured"]:
            logs = test["logs_captured"]
            
            # Look for common patterns
            if any("PopulateVolumeMip2" in log for log in logs):
                results["recommendations"].append(
                    "PopulateVolumeMip2 shader detected in logs - check for nested loop hangs or UAV write deadlocks"
                )
            
            if any("Resource state transition" in log for log in logs):
                results["recommendations"].append(
                    "Resource state transitions detected - verify UAV barriers and SRV↔UAV transitions"
                )
            
            if any("m_volumeFirstFrame" in log for log in logs):
                results["recommendations"].append(
                    "Volume first frame flag detected - check frame 1 vs frame 2 state transition logic"
                )
    
    # Add general recommendations
    if hung_counts:
        results["recommendations"].extend([
            "",
            "Recommended debugging steps:",
            "1. Use PIX GPU capture at exact threshold particle count",
            "2. Inspect resource state transitions frame-by-frame",
            "3. Check for UAV write conflicts or missing barriers",
            "4. Verify thread bounds checking in compute shaders",
            "5. Look for nested loops with variable iteration counts"
        ])
    
    return [TextContent(type="text", text=json.dumps(results, indent=2))]
