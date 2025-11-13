#!/usr/bin/env python3
"""
Material System Engineer - MCP Server
Autonomous implementation agent for material system design and build orchestration
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

# Import all tools
from tools.file_operations import (
    read_codebase_file,
    write_codebase_file,
    search_codebase
)
from tools.code_generator import (
    generate_material_shader,
    generate_particle_struct,
    generate_material_config
)
from tools.integration_tools import (
    create_test_scenario,
    generate_imgui_controls,
    validate_file_syntax
)

# Load environment
load_dotenv()

PROJECT_ROOT = os.getenv("PROJECT_ROOT", "/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean")

# Create MCP server
server = Server("material-system-engineer")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List all available material engineering tools"""
    return [
        # === File Operations (Tools 6-8) ===
        Tool(
            name="read_codebase_file",
            description="Read any project file (shader/C++/header/JSON/config)",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Relative path from project root (e.g., 'src/particles/ParticleSystem.h')"
                    }
                },
                "required": ["file_path"]
            }
        ),
        Tool(
            name="write_codebase_file",
            description="Write file with automatic backup to .backups/ directory",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Relative path from project root"
                    },
                    "content": {
                        "type": "string",
                        "description": "File content to write"
                    },
                    "create_backup": {
                        "type": "boolean",
                        "description": "Create timestamped backup before writing (default: true)",
                        "default": True
                    }
                },
                "required": ["file_path", "content"]
            }
        ),
        Tool(
            name="search_codebase",
            description="Search for pattern in codebase files (grep-like functionality)",
            inputSchema={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Text pattern to search for (supports regex)"
                    },
                    "file_glob": {
                        "type": "string",
                        "description": "File pattern to search (e.g., '**/*.hlsl', '**/*.cpp', 'src/**/*.h')",
                        "default": "**/*"
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of matches to return",
                        "default": 100
                    }
                },
                "required": ["pattern"]
            }
        ),

        # === Code Generation (Tools 9-11) ===
        Tool(
            name="generate_material_shader",
            description="Generate complete HLSL shader code for material type",
            inputSchema={
                "type": "object",
                "properties": {
                    "material_type": {
                        "type": "string",
                        "description": "Material type (e.g., GAS_CLOUD, STAR_MAIN_SEQUENCE)"
                    },
                    "properties": {
                        "type": "object",
                        "description": "Material properties (opacity, scattering_coefficient, emission_multiplier, albedo_rgb, phase_function_g)",
                        "properties": {
                            "opacity": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            "scattering_coefficient": {"type": "number", "minimum": 0.0},
                            "emission_multiplier": {"type": "number", "minimum": 0.0},
                            "albedo_rgb": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 3,
                                "maxItems": 3
                            },
                            "phase_function_g": {"type": "number", "minimum": -1.0, "maximum": 1.0}
                        }
                    },
                    "base_shader_template": {
                        "type": "string",
                        "description": "Shader template to use",
                        "default": "volumetric_raytracing"
                    }
                },
                "required": ["material_type", "properties"]
            }
        ),
        Tool(
            name="generate_particle_struct",
            description="Generate C++ particle struct with proper GPU alignment",
            inputSchema={
                "type": "object",
                "properties": {
                    "base_struct": {
                        "type": "string",
                        "description": "Existing struct definition or 'minimal' for new struct"
                    },
                    "new_fields": {
                        "type": "array",
                        "description": "List of field objects with type, name, size_bytes, comment",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string"},
                                "name": {"type": "string"},
                                "size_bytes": {"type": "integer"},
                                "comment": {"type": "string"}
                            },
                            "required": ["type", "name", "size_bytes"]
                        }
                    },
                    "target_alignment": {
                        "type": "integer",
                        "description": "GPU alignment requirement (default: 16 bytes)",
                        "default": 16
                    }
                },
                "required": ["base_struct", "new_fields"]
            }
        ),
        Tool(
            name="generate_material_config",
            description="Generate material property configuration file (JSON/C++/HLSL)",
            inputSchema={
                "type": "object",
                "properties": {
                    "material_definitions": {
                        "type": "array",
                        "description": "List of material definition objects",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string"},
                                "opacity": {"type": "number"},
                                "scattering_coefficient": {"type": "number"},
                                "emission_multiplier": {"type": "number"},
                                "albedo_rgb": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "minItems": 3,
                                    "maxItems": 3
                                },
                                "phase_function_g": {"type": "number"},
                                "description": {"type": "string"}
                            },
                            "required": ["type"]
                        }
                    },
                    "output_format": {
                        "type": "string",
                        "enum": ["json", "cpp_array", "hlsl_constants"],
                        "description": "Output format",
                        "default": "json"
                    }
                },
                "required": ["material_definitions"]
            }
        ),

        # === Integration Tools (Tools 12-14) ===
        Tool(
            name="create_test_scenario",
            description="Generate test scenario configuration for material system validation",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Scenario name (e.g., 'gas_cloud_test')"
                    },
                    "particle_count": {
                        "type": "integer",
                        "description": "Total number of particles"
                    },
                    "material_distribution": {
                        "type": "object",
                        "description": "Dict of material types to percentage (must sum to 1.0)",
                        "additionalProperties": {"type": "number"}
                    },
                    "lighting_preset": {
                        "type": "string",
                        "description": "Light configuration preset",
                        "default": "stellar_ring"
                    },
                    "camera_distance": {
                        "type": "number",
                        "description": "Camera distance from origin",
                        "default": 800.0
                    },
                    "output_format": {
                        "type": "string",
                        "enum": ["json", "markdown"],
                        "description": "Output format",
                        "default": "json"
                    }
                },
                "required": ["name", "particle_count", "material_distribution"]
            }
        ),
        Tool(
            name="generate_imgui_controls",
            description="Generate ImGui control code for material property editing",
            inputSchema={
                "type": "object",
                "properties": {
                    "material_types": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of material type names"
                    },
                    "output_format": {
                        "type": "string",
                        "enum": ["cpp", "markdown"],
                        "description": "Output format",
                        "default": "cpp"
                    }
                },
                "required": ["material_types"]
            }
        ),
        Tool(
            name="validate_file_syntax",
            description="Basic syntax validation for generated code (C++/HLSL/JSON)",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "File path (for error messages)"
                    },
                    "file_content": {
                        "type": "string",
                        "description": "File content to validate"
                    },
                    "file_type": {
                        "type": "string",
                        "enum": ["auto", "cpp", "hlsl", "json"],
                        "description": "File type (auto detects from extension)",
                        "default": "auto"
                    }
                },
                "required": ["file_path", "file_content"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls"""
    try:
        # File Operations
        if name == "read_codebase_file":
            result = await read_codebase_file(
                PROJECT_ROOT,
                arguments["file_path"]
            )
            return [TextContent(type="text", text=result)]

        elif name == "write_codebase_file":
            result = await write_codebase_file(
                PROJECT_ROOT,
                arguments["file_path"],
                arguments["content"],
                arguments.get("create_backup", True)
            )
            return [TextContent(type="text", text=result)]

        elif name == "search_codebase":
            result = await search_codebase(
                PROJECT_ROOT,
                arguments["pattern"],
                arguments.get("file_glob", "**/*"),
                arguments.get("max_results", 100)
            )
            return [TextContent(type="text", text=result)]

        # Code Generation
        elif name == "generate_material_shader":
            result = await generate_material_shader(
                arguments["material_type"],
                arguments["properties"],
                arguments.get("base_shader_template", "volumetric_raytracing")
            )
            return [TextContent(type="text", text=result)]

        elif name == "generate_particle_struct":
            result = await generate_particle_struct(
                arguments["base_struct"],
                arguments["new_fields"],
                arguments.get("target_alignment", 16)
            )
            return [TextContent(type="text", text=result)]

        elif name == "generate_material_config":
            result = await generate_material_config(
                arguments["material_definitions"],
                arguments.get("output_format", "json")
            )
            return [TextContent(type="text", text=result)]

        # Integration Tools
        elif name == "create_test_scenario":
            result = await create_test_scenario(
                arguments["name"],
                arguments["particle_count"],
                arguments["material_distribution"],
                arguments.get("lighting_preset", "stellar_ring"),
                arguments.get("camera_distance", 800.0),
                arguments.get("output_format", "json")
            )
            return [TextContent(type="text", text=result)]

        elif name == "generate_imgui_controls":
            result = await generate_imgui_controls(
                arguments["material_types"],
                arguments.get("output_format", "cpp")
            )
            return [TextContent(type="text", text=result)]

        elif name == "validate_file_syntax":
            result = await validate_file_syntax(
                arguments["file_path"],
                arguments["file_content"],
                arguments.get("file_type", "auto")
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
