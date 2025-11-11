#!/usr/bin/env python3
"""
Gaussian Analyzer - Flat MCP Server
Specialized tools for 3D Gaussian volumetric particle analysis and optimization
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Optional

# Add src directory to Python path
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR / "src"))

from dotenv import load_dotenv
from mcp.server import Server
from mcp.types import TextContent, Tool

# Import tools directly (flat structure)
from tools.parameter_analyzer import analyze_gaussian_parameters
from tools.material_simulator import simulate_material_properties
from tools.performance_estimator import estimate_performance_impact
from tools.technique_comparator import compare_rendering_techniques
from tools.struct_validator import validate_particle_struct

# Load environment
load_dotenv()

PROJECT_ROOT = os.getenv("PROJECT_ROOT", "/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean")

# Create MCP server
server = Server("gaussian-analyzer")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List all available Gaussian analysis tools"""
    return [
        Tool(
            name="analyze_gaussian_parameters",
            description="Analyze current 3D Gaussian particle structure, identify gaps, and propose extensions for diverse celestial body rendering",
            inputSchema={
                "type": "object",
                "properties": {
                    "analysis_depth": {
                        "type": "string",
                        "enum": ["quick", "detailed", "comprehensive"],
                        "description": "Analysis depth: quick (structure only), detailed (+ shader analysis), comprehensive (+ performance impact)",
                        "default": "detailed"
                    },
                    "focus_area": {
                        "type": "string",
                        "enum": ["structure", "shaders", "materials", "performance", "all"],
                        "description": "Specific area to focus on",
                        "default": "all"
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="simulate_material_properties",
            description="Simulate how proposed material property changes would affect rendering (opacity, scattering, emission, albedo)",
            inputSchema={
                "type": "object",
                "properties": {
                    "material_type": {
                        "type": "string",
                        "enum": ["PLASMA", "STAR_MAIN_SEQUENCE", "STAR_GIANT", "GAS_CLOUD", "DUST", "ROCKY", "ICY", "NEUTRON_STAR", "CUSTOM"],
                        "description": "Material type to simulate"
                    },
                    "properties": {
                        "type": "object",
                        "description": "Material properties to test (opacity, scattering_coefficient, emission_multiplier, albedo_rgb, phase_function_g)",
                        "properties": {
                            "opacity": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            "scattering_coefficient": {"type": "number", "minimum": 0.0},
                            "emission_multiplier": {"type": "number", "minimum": 0.0},
                            "albedo_rgb": {
                                "type": "array",
                                "items": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                                "minItems": 3,
                                "maxItems": 3
                            },
                            "phase_function_g": {"type": "number", "minimum": -1.0, "maximum": 1.0}
                        }
                    },
                    "render_mode": {
                        "type": "string",
                        "enum": ["volumetric_only", "hybrid_surface_volume", "comparison"],
                        "description": "Rendering mode to simulate",
                        "default": "volumetric_only"
                    }
                },
                "required": ["material_type"]
            }
        ),
        Tool(
            name="estimate_performance_impact",
            description="Calculate estimated FPS impact of proposed particle structure or shader modifications",
            inputSchema={
                "type": "object",
                "properties": {
                    "particle_struct_bytes": {
                        "type": "integer",
                        "description": "Proposed particle structure size in bytes (current: 32 bytes)",
                        "minimum": 32,
                        "maximum": 128
                    },
                    "material_types_count": {
                        "type": "integer",
                        "description": "Number of material types to support",
                        "minimum": 1,
                        "maximum": 16,
                        "default": 5
                    },
                    "shader_complexity": {
                        "type": "string",
                        "enum": ["minimal", "moderate", "complex"],
                        "description": "Shader modification complexity (minimal=lookup, moderate=branches, complex=per-pixel raymarching)",
                        "default": "moderate"
                    },
                    "particle_counts": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Particle counts to test (default: [10000, 50000, 100000])",
                        "default": [10000, 50000, 100000]
                    }
                },
                "required": ["particle_struct_bytes"]
            }
        ),
        Tool(
            name="compare_rendering_techniques",
            description="Compare different volumetric rendering approaches (pure volumetric vs hybrid vs billboard) for performance and quality trade-offs",
            inputSchema={
                "type": "object",
                "properties": {
                    "techniques": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["pure_volumetric_gaussian", "hybrid_surface_volume", "billboard_impostors", "adaptive_lod", "current_implementation"]
                        },
                        "description": "Techniques to compare",
                        "default": ["pure_volumetric_gaussian", "hybrid_surface_volume"]
                    },
                    "criteria": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["performance", "visual_quality", "memory_usage", "implementation_complexity", "material_flexibility"]
                        },
                        "description": "Comparison criteria",
                        "default": ["performance", "visual_quality", "material_flexibility"]
                    }
                },
                "required": []
            }
        ),
        Tool(
            name="validate_particle_struct",
            description="Validate proposed particle structure for alignment, size constraints, and backward compatibility",
            inputSchema={
                "type": "object",
                "properties": {
                    "struct_definition": {
                        "type": "string",
                        "description": "C++ struct definition to validate (paste complete struct code)"
                    },
                    "check_backward_compatibility": {
                        "type": "boolean",
                        "description": "Check if structure is backward compatible with 32-byte legacy format",
                        "default": True
                    },
                    "check_gpu_alignment": {
                        "type": "boolean",
                        "description": "Validate 16-byte GPU alignment requirements",
                        "default": True
                    }
                },
                "required": ["struct_definition"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls"""
    try:
        if name == "analyze_gaussian_parameters":
            result = await analyze_gaussian_parameters(
                PROJECT_ROOT,
                analysis_depth=arguments.get("analysis_depth", "detailed"),
                focus_area=arguments.get("focus_area", "all")
            )
            return [TextContent(type="text", text=result)]

        elif name == "simulate_material_properties":
            result = await simulate_material_properties(
                material_type=arguments["material_type"],
                properties=arguments.get("properties", {}),
                render_mode=arguments.get("render_mode", "volumetric_only")
            )
            return [TextContent(type="text", text=result)]

        elif name == "estimate_performance_impact":
            result = await estimate_performance_impact(
                particle_struct_bytes=arguments["particle_struct_bytes"],
                material_types_count=arguments.get("material_types_count", 5),
                shader_complexity=arguments.get("shader_complexity", "moderate"),
                particle_counts=arguments.get("particle_counts", [10000, 50000, 100000])
            )
            return [TextContent(type="text", text=result)]

        elif name == "compare_rendering_techniques":
            result = await compare_rendering_techniques(
                techniques=arguments.get("techniques", ["pure_volumetric_gaussian", "hybrid_surface_volume"]),
                criteria=arguments.get("criteria", ["performance", "visual_quality", "material_flexibility"])
            )
            return [TextContent(type="text", text=result)]

        elif name == "validate_particle_struct":
            result = await validate_particle_struct(
                struct_definition=arguments["struct_definition"],
                check_backward_compatibility=arguments.get("check_backward_compatibility", True),
                check_gpu_alignment=arguments.get("check_gpu_alignment", True)
            )
            return [TextContent(type="text", text=result)]

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

    except Exception as e:
        return [TextContent(type="text", text=f"Error executing {name}: {str(e)}")]


async def main():
    """Run the MCP server"""
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
