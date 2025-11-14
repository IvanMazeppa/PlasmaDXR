#!/usr/bin/env python3
"""
DXR Volumetric Pyro Specialist - Specialized MCP Server
Design consultant for pyrotechnic and explosive volumetric effects in DirectX 12 DXR 1.1

Translates high-level effect requests (e.g., "add supernova explosions") into
implementation-ready material specifications for volumetric 3D Gaussian particle systems.

Position in Multi-Agent Pipeline:
- Receives from: gaussian-analyzer (baseline volumetric analysis)
- Provides to: material-system-engineer (detailed pyro specs for code generation)
- Validated by: dxr-image-quality-analyst (visual quality assessment)
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src directory to Python path for direct imports
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR / "src"))

from dotenv import load_dotenv
from mcp.server import Server
from mcp.types import TextContent, Tool

# Import pyro engineering tools
from tools.pyro_research import research_pyro_techniques, format_research_report
from tools.explosion_designer import design_explosion_effect, format_explosion_design
from tools.fire_designer import design_fire_effect, format_fire_design
from tools.performance_estimator import estimate_pyro_performance, format_performance_estimate
from tools.technique_comparator import compare_pyro_techniques, format_comparison_report

# Load environment
load_dotenv()

PROJECT_ROOT = os.getenv("PROJECT_ROOT", "/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean")

# Create MCP server
server = Server("dxr-volumetric-pyro-specialist")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available volumetric pyro design tools"""
    return [
        Tool(
            name="research_pyro_techniques",
            description="Research cutting-edge volumetric pyro techniques using web search and academic sources. Provides recommendations for NVIDIA RTX Volumetrics, neural rendering, GPU pyro solvers, and real-time explosive simulations.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (e.g., 'GPU volumetric explosions real-time', 'stellar flare rendering')"
                    },
                    "focus": {
                        "type": "string",
                        "description": "Specific focus area",
                        "enum": ["explosions", "fire_smoke", "stellar_phenomena", "procedural_noise", "performance"],
                        "default": "explosions"
                    },
                    "include_papers": {
                        "type": "boolean",
                        "description": "Include academic papers and technical articles (default: true)",
                        "default": True
                    },
                    "include_implementations": {
                        "type": "boolean",
                        "description": "Include code examples and shader implementations (default: true)",
                        "default": True
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="design_explosion_effect",
            description="Design complete explosion effect specification (supernova, stellar flare, accretion disk burst). Provides temporal dynamics (expansion curves, temperature decay), material properties, procedural noise parameters, color profiles, and performance estimates.",
            inputSchema={
                "type": "object",
                "properties": {
                    "effect_type": {
                        "type": "string",
                        "description": "Type of explosion effect",
                        "enum": ["supernova", "stellar_flare", "accretion_burst", "shockwave", "custom"],
                        "default": "supernova"
                    },
                    "duration_seconds": {
                        "type": "number",
                        "description": "Effect duration in seconds (default: 5.0)",
                        "default": 5.0,
                        "minimum": 0.5,
                        "maximum": 30.0
                    },
                    "max_radius_meters": {
                        "type": "number",
                        "description": "Maximum explosion radius in meters (default: 500.0)",
                        "default": 500.0,
                        "minimum": 50.0,
                        "maximum": 5000.0
                    },
                    "peak_temperature_kelvin": {
                        "type": "number",
                        "description": "Peak temperature in Kelvin (default: 100000K for supernova)",
                        "default": 100000.0,
                        "minimum": 5000.0,
                        "maximum": 500000.0
                    },
                    "particle_budget": {
                        "type": "integer",
                        "description": "Number of particles for effect (default: 10000)",
                        "default": 10000,
                        "minimum": 1000,
                        "maximum": 100000
                    }
                },
                "required": ["effect_type"]
            }
        ),
        Tool(
            name="design_fire_effect",
            description="Design fire/smoke effect specification (stellar fire, nebula wisps, combustion). Provides turbulence parameters, scattering profiles, opacity curves, procedural noise (Simplex/Perlin/Worley), and color temperature gradients.",
            inputSchema={
                "type": "object",
                "properties": {
                    "fire_type": {
                        "type": "string",
                        "description": "Type of fire/smoke effect",
                        "enum": ["stellar_fire", "nebula_wisp", "combustion_cloud", "plasma_jet", "custom"],
                        "default": "stellar_fire"
                    },
                    "turbulence_intensity": {
                        "type": "number",
                        "description": "Turbulence intensity (0.0 = calm, 1.0 = violent)",
                        "default": 0.5,
                        "minimum": 0.0,
                        "maximum": 1.0
                    },
                    "temperature_range_kelvin": {
                        "type": "array",
                        "description": "Temperature range [min, max] in Kelvin",
                        "items": {"type": "number"},
                        "minItems": 2,
                        "maxItems": 2,
                        "default": [3000.0, 15000.0]
                    },
                    "opacity_profile": {
                        "type": "string",
                        "description": "Opacity distribution profile",
                        "enum": ["dense_core", "wispy", "uniform", "layered"],
                        "default": "dense_core"
                    },
                    "particle_budget": {
                        "type": "integer",
                        "description": "Number of particles for effect (default: 10000)",
                        "default": 10000,
                        "minimum": 1000,
                        "maximum": 100000
                    }
                },
                "required": ["fire_type"]
            }
        ),
        Tool(
            name="estimate_pyro_performance",
            description="Estimate FPS impact of pyro effects. Provides ALU operation counts per particle, memory bandwidth requirements, temporal buffer overhead, and FPS estimates on RTX 4060 Ti @ 1080p.",
            inputSchema={
                "type": "object",
                "properties": {
                    "effect_complexity": {
                        "type": "string",
                        "description": "Effect complexity level",
                        "enum": ["minimal", "moderate", "complex", "extreme"],
                        "default": "moderate"
                    },
                    "particle_count": {
                        "type": "integer",
                        "description": "Number of particles (default: 10000)",
                        "default": 10000,
                        "minimum": 1000,
                        "maximum": 100000
                    },
                    "noise_octaves": {
                        "type": "integer",
                        "description": "Procedural noise octaves (default: 3)",
                        "default": 3,
                        "minimum": 1,
                        "maximum": 8
                    },
                    "temporal_animation": {
                        "type": "boolean",
                        "description": "Include temporal animation overhead (default: true)",
                        "default": True
                    },
                    "base_fps": {
                        "type": "number",
                        "description": "Baseline FPS before pyro effects (default: 120)",
                        "default": 120.0,
                        "minimum": 30.0,
                        "maximum": 240.0
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="compare_pyro_techniques",
            description="Compare pyro implementation approaches (particle-based vs OpenVDB vs hybrid, temporal effects vs static, GPU pyro solvers). Evaluates quality, performance, implementation complexity, and suitability for real-time rendering.",
            inputSchema={
                "type": "object",
                "properties": {
                    "techniques": {
                        "type": "array",
                        "description": "Techniques to compare",
                        "items": {
                            "type": "string",
                            "enum": ["particle_gaussian", "openvdb", "hybrid", "gpu_pyro_solver", "baked_simulation"]
                        },
                        "default": ["particle_gaussian", "hybrid"]
                    },
                    "criteria": {
                        "type": "array",
                        "description": "Comparison criteria",
                        "items": {
                            "type": "string",
                            "enum": ["visual_quality", "performance", "implementation_complexity", "memory_usage", "temporal_coherence"]
                        },
                        "default": ["visual_quality", "performance", "implementation_complexity"]
                    },
                    "target_fps": {
                        "type": "number",
                        "description": "Target FPS for performance comparison (default: 90)",
                        "default": 90.0,
                        "minimum": 30.0,
                        "maximum": 240.0
                    }
                },
                "required": []
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Execute a pyro design tool"""

    if name == "research_pyro_techniques":
        results = await research_pyro_techniques(
            query=arguments.get("query"),
            focus=arguments.get("focus", "explosions"),
            include_papers=arguments.get("include_papers", True),
            include_implementations=arguments.get("include_implementations", True)
        )
        report = await format_research_report(results)
        return [TextContent(type="text", text=report)]

    elif name == "design_explosion_effect":
        results = await design_explosion_effect(
            effect_type=arguments.get("effect_type", "supernova"),
            duration_seconds=arguments.get("duration_seconds", 5.0),
            max_radius_meters=arguments.get("max_radius_meters", 500.0),
            peak_temperature_kelvin=arguments.get("peak_temperature_kelvin", 100000.0),
            particle_budget=arguments.get("particle_budget", 10000)
        )
        report = await format_explosion_design(results)
        return [TextContent(type="text", text=report)]

    elif name == "design_fire_effect":
        results = await design_fire_effect(
            fire_type=arguments.get("fire_type", "stellar_fire"),
            turbulence_intensity=arguments.get("turbulence_intensity", 0.5),
            temperature_range_kelvin=arguments.get("temperature_range_kelvin", [3000.0, 15000.0]),
            opacity_profile=arguments.get("opacity_profile", "dense_core"),
            particle_budget=arguments.get("particle_budget", 10000)
        )
        report = await format_fire_design(results)
        return [TextContent(type="text", text=report)]

    elif name == "estimate_pyro_performance":
        results = await estimate_pyro_performance(
            effect_complexity=arguments.get("effect_complexity", "moderate"),
            particle_count=arguments.get("particle_count", 10000),
            noise_octaves=arguments.get("noise_octaves", 3),
            temporal_animation=arguments.get("temporal_animation", True),
            base_fps=arguments.get("base_fps", 120.0)
        )
        report = await format_performance_estimate(results)
        return [TextContent(type="text", text=report)]

    elif name == "compare_pyro_techniques":
        results = await compare_pyro_techniques(
            techniques=arguments.get("techniques", ["particle_gaussian", "hybrid"]),
            criteria=arguments.get("criteria", ["visual_quality", "performance", "implementation_complexity"]),
            target_fps=arguments.get("target_fps", 90.0)
        )
        report = await format_comparison_report(results)
        return [TextContent(type="text", text=report)]

    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def main():
    """Run MCP server using stdio transport"""
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
