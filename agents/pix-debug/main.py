#!/usr/bin/env python3
"""
PIX Debugging Agent v4 - Hybrid DXR Debugging
Combines in-app buffer dumping with PIX capture analysis for comprehensive DirectX Raytracing debugging
"""

import json
import os
import subprocess
import struct
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import anyio
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from claude_agent_sdk import (
    tool,
    create_sdk_mcp_server,
    query,
    ClaudeAgentOptions,
)

# Load environment variables
load_dotenv()

# API key is optional - will use Claude Code CLI if not provided
# This allows running directly in your Claude Code session without a separate API key

# Configuration
PIXTOOL_PATH = os.getenv("PIXTOOL_PATH", "/mnt/c/Program Files/Microsoft PIX/2509.25/pixtool.exe")
PLASMA_DX_PATH = os.getenv("PLASMA_DX_PATH", "/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean")
BUFFER_DUMP_DIR = os.getenv("BUFFER_DUMP_DIR", os.path.join(PLASMA_DX_PATH, "PIX/buffer_dumps"))
PIX_CAPTURES_DIR = os.getenv("PIX_CAPTURES_DIR", os.path.join(PLASMA_DX_PATH, "PIX/Captures"))
ANALYSIS_SCRIPTS_DIR = os.getenv("ANALYSIS_SCRIPTS_DIR", os.path.join(PLASMA_DX_PATH, "PIX/scripts/analysis"))

# Agent System Prompt
SYSTEM_PROMPT = """You are a DirectX Raytracing (DXR) debugging specialist with expertise in:

1. **Acceleration Structure Analysis**
   - Bottom-Level Acceleration Structures (BLAS) for particle geometry
   - Top-Level Acceleration Structures (TLAS) for scene hierarchy
   - AABB (Axis-Aligned Bounding Box) generation and validation
   - Build flags, update modes, and performance optimization

2. **Ray Tracing Pipeline Debugging**
   - State object configuration and validation
   - Shader Binding Table (SBT) layout and indexing
   - Ray generation, closest hit, any hit, miss, and intersection shaders
   - Shader record alignment and resource binding

3. **Buffer Analysis**
   - GPU buffer readback and parsing (binary struct analysis)
   - ReSTIR reservoir analysis (lightPos, weightSum, M, W)
   - Particle data validation (position, velocity, color, lifetime)
   - Lighting and emission grid buffers

4. **PIX Integration (Hybrid Approach)**
   - **In-App Buffer Dumping**: Fast 2-second captures using --dump-buffers flag
   - **PIX Capture Analysis**: Deep GPU state inspection using pixtool for complex issues
   - Programmatic capture triggers
   - Event timeline analysis and shader debugging

5. **Problem Solving**
   - Root cause identification from buffer statistics
   - Visual artifact diagnosis (black dots, overexposure, banding)
   - Performance bottleneck detection (stalls, memory bandwidth)
   - Shader debugging (invalid intersections, NaN propagation)

**Tools Available:**
- `capture_buffers`: Trigger in-app buffer dump at specific frame
- `analyze_restir_reservoirs`: Parse and analyze ReSTIR lighting reservoirs
- `analyze_particle_buffers`: Validate particle data (position, velocity, lifetime)
- `pix_capture`: Create PIX .wpix capture using pixtool
- `pix_list_captures`: List available PIX captures
- `compare_captures`: Compare multiple buffer dumps to identify regressions
- `diagnose_visual_artifact`: Automated diagnosis of rendering issues

**Workflow:**
1. Start with fast buffer dumps for quick analysis
2. Escalate to PIX captures for deep GPU state inspection when needed
3. Cross-reference buffer statistics with visual artifacts
4. Provide actionable recommendations with HLSL code snippets

Always provide quantitative analysis (avg W, M distribution, spatial patterns) and cite specific buffer locations when identifying bugs.
"""

# MCP Tools Implementation

@tool(
    "capture_buffers",
    "Trigger in-app buffer dump from PlasmaDX-Clean at specific frame",
    {
        "frame": int,
        "mode": str,
        "output_dir": str
    }
)
async def capture_buffers_tool(args):
    """Capture GPU buffers using in-app buffer dumping"""
    frame = args.get("frame")
    mode = args.get("mode", "gaussian")
    output_dir = args.get("output_dir")

    exe_path = os.path.join(PLASMA_DX_PATH, "build/bin/Debug/PlasmaDX-Clean.exe")
    dump_dir = output_dir or BUFFER_DUMP_DIR

    # Build command
    cmd = [exe_path, "--dump-buffers"]
    if frame is not None:
        cmd.append(str(frame))
    if mode:
        cmd.append(f"--{mode}")
    if output_dir:
        cmd.extend(["--dump-dir", output_dir])

    try:
        # Run capture
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=PLASMA_DX_PATH
        )

        # Check for metadata file
        metadata_path = os.path.join(dump_dir, "metadata.json")
        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

        # List captured files
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

        if result.stderr:
            result_data["stderr"] = result.stderr

        return {
            "content": [{
                "type": "text",
                "text": json.dumps(result_data, indent=2)
            }]
        }
    except Exception as e:
        return {
            "content": [{
                "type": "text",
                "text": f"Error capturing buffers: {str(e)}"
            }],
            "is_error": True
        }


@tool(
    "analyze_restir_reservoirs",
    "Parse and analyze ReSTIR reservoir buffers for lighting statistics",
    {
        "current_path": str,
        "prev_path": str,
        "sample_size": int
    }
)
async def analyze_restir_reservoirs_tool(args):
    """Analyze ReSTIR lighting reservoirs"""
    current_path = args.get("current_path")
    prev_path = args.get("prev_path")
    sample_size = args.get("sample_size", 1000)

    if not current_path:
        return {
            "content": [{
                "type": "text",
                "text": "Error: current_path is required"
            }],
            "is_error": True
        }

    try:
        # Parse current reservoirs
        with open(current_path, 'rb') as f:
            data = f.read()

        # Struct: float3 lightPos (12), float weightSum (4), uint M (4), float W (4), padding (8) = 32 bytes
        num_reservoirs = len(data) // 32
        reservoirs = []

        for i in range(0, min(num_reservoirs, sample_size)):
            offset = i * 32
            lightPos = struct.unpack('fff', data[offset:offset+12])
            weightSum = struct.unpack('f', data[offset+12:offset+16])[0]
            M = struct.unpack('I', data[offset+16:offset+20])[0]
            W = struct.unpack('f', data[offset+20:offset+24])[0]

            reservoirs.append({
                'lightPos': lightPos,
                'weightSum': weightSum,
                'M': M,
                'W': W
            })

        # Calculate statistics
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
            },
            'weightSum': {
                'mean': float(np.mean(weightSum_values)),
                'std': float(np.std(weightSum_values)),
                'min': float(np.min(weightSum_values)),
                'max': float(np.max(weightSum_values))
            }
        }

        # Analyze for issues
        issues = []
        recommendations = []

        # Check for low M values (ReSTIR bug indicator)
        if stats['M']['mean'] < 5.0:
            issues.append(f"Low average M = {stats['M']['mean']:.1f} (expect >8 for good convergence)")
            recommendations.append("Check spatial reuse: ensure neighbor search radius is adequate")
            recommendations.append("Verify temporal reuse: check if previous frame reservoirs are valid")

        # Check for extreme W values
        if stats['W']['max'] > 0.01:
            issues.append(f"Very large W values detected (max={stats['W']['max']:.6f})")
            recommendations.append("Possible unbiased contribution overflow - consider clamping W")

        # Check for zero weight reservoirs
        zero_W_count = sum(1 for w in W_values if w == 0.0)
        if zero_W_count > len(W_values) * 0.1:
            issues.append(f"{zero_W_count}/{len(W_values)} reservoirs have W=0 ({zero_W_count/len(W_values)*100:.1f}%)")
            recommendations.append("High zero-weight ratio indicates poor light sampling or invalid reservoirs")

        result = {
            "summary": f"Analyzed {num_reservoirs} ReSTIR reservoirs. Avg W={stats['W']['mean']:.6f}, Avg M={stats['M']['mean']:.1f}",
            "statistics": stats,
            "issues_found": issues,
            "recommendations": recommendations
        }

        return {
            "content": [{
                "type": "text",
                "text": json.dumps(result, indent=2)
            }]
        }

    except Exception as e:
        return {
            "content": [{
                "type": "text",
                "text": f"Error analyzing reservoirs: {str(e)}"
            }],
            "is_error": True
        }


@tool(
    "analyze_particle_buffers",
    "Validate particle buffer data (position, velocity, lifetime)",
    {
        "particles_path": str,
        "expected_count": int
    }
)
async def analyze_particle_buffers_tool(args):
    """Analyze particle buffer for validation"""
    particles_path = args.get("particles_path")
    expected_count = args.get("expected_count")

    if not particles_path:
        return {
            "content": [{
                "type": "text",
                "text": "Error: particles_path is required"
            }],
            "is_error": True
        }

    try:
        with open(particles_path, 'rb') as f:
            data = f.read()

        # Particle struct: float3 pos, float3 vel, float lifetime, float age (32 bytes)
        particle_size = 32
        num_particles = len(data) // particle_size

        particles = []
        for i in range(min(num_particles, 1000)):  # Sample first 1000
            offset = i * particle_size
            pos = struct.unpack('fff', data[offset:offset+12])
            vel = struct.unpack('fff', data[offset+12:offset+24])
            lifetime = struct.unpack('f', data[offset+24:offset+28])[0]
            age = struct.unpack('f', data[offset+28:offset+32])[0]

            particles.append({
                'pos': pos,
                'vel': vel,
                'lifetime': lifetime,
                'age': age
            })

        # Statistics
        positions = np.array([p['pos'] for p in particles])
        velocities = np.array([p['vel'] for p in particles])
        lifetimes = [p['lifetime'] for p in particles]
        ages = [p['age'] for p in particles]

        stats = {
            'total_particles': num_particles,
            'sampled': len(particles),
            'position': {
                'mean': positions.mean(axis=0).tolist(),
                'std': positions.std(axis=0).tolist(),
                'min': positions.min(axis=0).tolist(),
                'max': positions.max(axis=0).tolist()
            },
            'velocity': {
                'mean': velocities.mean(axis=0).tolist(),
                'std': velocities.std(axis=0).tolist()
            },
            'lifetime': {
                'mean': float(np.mean(lifetimes)),
                'min': float(np.min(lifetimes)),
                'max': float(np.max(lifetimes))
            },
            'age': {
                'mean': float(np.mean(ages)),
                'min': float(np.min(ages)),
                'max': float(np.max(ages))
            }
        }

        issues = []
        recommendations = []

        # Check for NaN positions
        if np.isnan(positions).any():
            nan_count = int(np.isnan(positions).sum())
            issues.append(f"Found {nan_count} NaN values in particle positions")
            recommendations.append("Check particle spawn and physics update shaders for NaN propagation")

        # Check for dead particles
        dead_count = sum(1 for a, l in zip(ages, lifetimes) if a >= l)
        if dead_count > num_particles * 0.1:
            issues.append(f"{dead_count}/{num_particles} particles are dead (age >= lifetime)")
            recommendations.append("Particle recycling may not be working correctly")

        # Check expected count
        if expected_count and num_particles != expected_count:
            issues.append(f"Particle count mismatch: got {num_particles}, expected {expected_count}")

        result = {
            "summary": f"Analyzed {num_particles} particles",
            "statistics": stats,
            "issues_found": issues,
            "recommendations": recommendations
        }

        return {
            "content": [{
                "type": "text",
                "text": json.dumps(result, indent=2)
            }]
        }

    except Exception as e:
        return {
            "content": [{
                "type": "text",
                "text": f"Error analyzing particles: {str(e)}"
            }],
            "is_error": True
        }


@tool(
    "pix_capture",
    "Create PIX .wpix capture using pixtool for deep GPU analysis",
    {
        "frames": int,
        "output_name": str,
        "auto_open": bool
    }
)
async def pix_capture_tool(args):
    """Create PIX capture"""
    frames = args.get("frames", 1)
    output_name = args.get("output_name")
    auto_open = args.get("auto_open", False)

    if output_name is None:
        timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
        output_name = f"agent_capture_{timestamp}.wpix"

    output_path = os.path.join(PIX_CAPTURES_DIR, output_name)
    exe_path = os.path.join(PLASMA_DX_PATH, "build/bin/Debug/PlasmaDX-Clean.exe")

    # Build pixtool command
    cmd = [
        PIXTOOL_PATH,
        "launch",
        exe_path,
        f"--working-directory={PLASMA_DX_PATH}",
        "take-capture",
        f"--frames={frames}",
        "save-capture",
        output_path
    ]

    if auto_open:
        cmd.insert(cmd.index("take-capture"), "--open")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120
        )

        result_data = {
            "status": "success" if result.returncode == 0 else "failed",
            "capture_path": output_path,
            "exists": os.path.exists(output_path),
            "size_mb": round(os.path.getsize(output_path) / (1024*1024), 2) if os.path.exists(output_path) else 0
        }

        if result.stderr:
            result_data["stderr"] = result.stderr

        return {
            "content": [{
                "type": "text",
                "text": json.dumps(result_data, indent=2)
            }]
        }
    except Exception as e:
        return {
            "content": [{
                "type": "text",
                "text": f"Error creating PIX capture: {str(e)}"
            }],
            "is_error": True
        }


@tool(
    "pix_list_captures",
    "List available PIX .wpix captures with metadata",
    {}
)
async def pix_list_captures_tool(args):
    """List PIX captures"""
    captures = []
    if os.path.exists(PIX_CAPTURES_DIR):
        for filename in os.listdir(PIX_CAPTURES_DIR):
            if filename.endswith('.wpix'):
                filepath = os.path.join(PIX_CAPTURES_DIR, filename)
                stat = os.stat(filepath)
                captures.append({
                    'name': filename,
                    'path': filepath,
                    'size_mb': round(stat.st_size / (1024*1024), 2),
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                })

    captures = sorted(captures, key=lambda x: x['modified'], reverse=True)

    return {
        "content": [{
            "type": "text",
            "text": json.dumps({
                "total_captures": len(captures),
                "captures": captures[:20]  # Return most recent 20
            }, indent=2)
        }]
    }


@tool(
    "diagnose_visual_artifact",
    "Automated diagnosis of rendering issues from symptom description",
    {
        "symptom": str,
        "buffer_dump_dir": str
    }
)
async def diagnose_visual_artifact_tool(args):
    """Diagnose visual artifacts"""
    symptom = args.get("symptom", "")
    buffer_dump_dir = args.get("buffer_dump_dir")

    symptom_lower = symptom.lower()
    result = {}

    # Symptom database
    if "black dot" in symptom_lower or "dark particle" in symptom_lower:
        result = {
            "summary": "Black dots typically caused by ReSTIR under-sampling (low M values)",
            "issues_found": [
                "Possible ReSTIR convergence issue",
                "Insufficient spatial/temporal reuse",
                "Particles too far from light sources"
            ],
            "recommendations": [
                "Scale contribution by sqrt(M) to maintain visibility:",
                "float visibility_scale = sqrt(reservoir.M / 16.0);",
                "float3 contribution = reservoir.lightPos * reservoir.W * visibility_scale;",
                "",
                "Increase spatial reuse radius in shader",
                "Verify light placement for far camera distances"
            ]
        }

        # Analyze buffer if provided
        if buffer_dump_dir:
            reservoir_path = os.path.join(buffer_dump_dir, "g_currentReservoirs.bin")
            if os.path.exists(reservoir_path):
                # Call analyze tool
                analysis_result = await analyze_restir_reservoirs_tool({
                    "current_path": reservoir_path,
                    "sample_size": 1000
                })
                if analysis_result.get("content"):
                    result["buffer_analysis"] = json.loads(analysis_result["content"][0]["text"])

    elif "overexposure" in symptom_lower or "too bright" in symptom_lower:
        result = {
            "summary": "Overexposure typically from accumulation overflow",
            "issues_found": ["Possible weight accumulation overflow", "Missing tone mapping"],
            "recommendations": [
                "Clamp W values: W = min(W, 0.01)",
                "Add exposure control in final pass",
                "Check for NaN propagation in weightSum calculation"
            ]
        }

    elif "banding" in symptom_lower or "color band" in symptom_lower:
        result = {
            "summary": "Color banding from insufficient precision",
            "issues_found": ["Low bit depth intermediate buffers", "Premature quantization"],
            "recommendations": [
                "Use DXGI_FORMAT_R16G16B16A16_FLOAT for HDR buffers",
                "Apply dithering before final output",
                "Check for 8-bit intermediate storage"
            ]
        }

    else:
        result = {
            "summary": f"Unknown symptom: {symptom}",
            "recommendations": [
                "Provide more specific symptom description",
                "Capture buffer dump for automated analysis",
                "Check for shader compilation errors or warnings"
            ]
        }

    return {
        "content": [{
            "type": "text",
            "text": json.dumps(result, indent=2)
        }]
    }


# Create MCP server with all tools
pix_server = create_sdk_mcp_server(
    name="pix-debug-tools",
    version="4.0.0",
    tools=[
        capture_buffers_tool,
        analyze_restir_reservoirs_tool,
        analyze_particle_buffers_tool,
        pix_capture_tool,
        pix_list_captures_tool,
        diagnose_visual_artifact_tool,
    ]
)


async def main():
    """Main entry point for PIX Debugging Agent"""
    print("🔧 PIX Debugging Agent v4 - DirectX Raytracing Specialist")
    print("=" * 70)
    print("\nInitializing Claude Agent SDK...")

    # Configure options
    options = ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        mcp_servers={"pix_tools": pix_server},
    )

    print("✅ Agent ready! Starting interactive session...\n")
    print("Example queries:")
    print("  - 'Capture buffers at frame 120 in gaussian mode'")
    print("  - 'I'm seeing black dots at far distances'")
    print("  - 'Analyze the latest buffer dump'")
    print("  - 'Create a PIX capture for detailed analysis'")
    print("\nType your question or request:\n")

    # Use query function for simple interaction
    user_prompt = input("You: ")

    async for message in query(prompt=user_prompt, options=options):
        # Print assistant messages
        if hasattr(message, 'content'):
            for block in message.content:
                if hasattr(block, 'text'):
                    print(f"\n{block.text}\n")


if __name__ == "__main__":
    anyio.run(main)
