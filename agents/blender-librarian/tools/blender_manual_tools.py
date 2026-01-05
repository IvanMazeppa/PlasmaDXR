"""
OpenAI Tool Definitions for blender-manual MCP Server

These tool definitions allow GPT-5.2 to call the blender-manual MCP server
through the OpenAI Responses API with function calling.

The 12 tools mirror the MCP server's capabilities:
- search_manual, search_tutorials, browse_hierarchy (general navigation)
- search_vdb_workflow, search_python_api (specialized searches)
- search_nodes, search_modifiers (shader/modifier docs)
- read_page (content retrieval)
- list_api_modules, search_bpy_operators, search_bpy_types (API reference)
- search_semantic (AI-powered semantic search)

NOTE: OpenAI strict mode requires ALL properties to be in 'required' array.
Default values are handled by the MCP bridge when arguments are omitted.
"""

from typing import List, Dict, Any

# OpenAI function tool definitions for blender-manual MCP server
# Using strict mode - all properties must be in 'required' array
BLENDER_MANUAL_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "name": "search_manual",
        "description": "Search the Blender 5.0 Manual for documentation. Best for general queries about features, settings, or workflows. Returns paths, snippets, and relevance scores.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (keywords, phrases). Examples: 'volume rendering', 'export openvdb', 'python api fluid simulation'"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results (default: 5)"
                },
                "offset": {
                    "type": "integer",
                    "description": "Skip first N results for pagination (default: 0)"
                },
                "compact": {
                    "type": "boolean",
                    "description": "If true, returns minimal output (titles/paths only)"
                }
            },
            "required": ["query", "limit", "offset", "compact"],
            "additionalProperties": False
        },
        "strict": True
    },
    {
        "type": "function",
        "name": "search_tutorials",
        "description": "Search for tutorials and getting started guides. Optimized for finding learning resources, especially for volumetrics and VDB workflows.",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Main topic (e.g., 'volumetrics', 'fluid simulation', 'rendering')"
                },
                "technique": {
                    "type": ["string", "null"],
                    "description": "Optional specific technique (e.g., 'mantaflow', 'pyro', 'smoke'). Pass null if not needed."
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results (default: 5)"
                },
                "compact": {
                    "type": "boolean",
                    "description": "Minimal output mode"
                }
            },
            "required": ["topic", "technique", "limit", "compact"],
            "additionalProperties": False
        },
        "strict": True
    },
    {
        "type": "function",
        "name": "browse_hierarchy",
        "description": "Browse the Blender Manual's hierarchical structure like a file tree. Navigate through categories and sections to discover content.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": ["string", "null"],
                    "description": "Optional path prefix to browse (e.g., 'render', 'physics/fluid'). Pass null to show top-level categories."
                }
            },
            "required": ["path"],
            "additionalProperties": False
        },
        "strict": True
    },
    {
        "type": "function",
        "name": "search_vdb_workflow",
        "description": "Specialized search for VDB/OpenVDB/NanoVDB workflows. Optimized for finding export settings, baking simulations, and volume handling.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "VDB-related query (e.g., 'export vdb', 'bake mantaflow', 'cache smoke')"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results (default: 5)"
                },
                "offset": {
                    "type": "integer",
                    "description": "Pagination offset (default: 0)"
                },
                "compact": {
                    "type": "boolean",
                    "description": "Minimal output mode"
                }
            },
            "required": ["query", "limit", "offset", "compact"],
            "additionalProperties": False
        },
        "strict": True
    },
    {
        "type": "function",
        "name": "search_python_api",
        "description": "Search for Python API documentation (bpy.ops, bpy.types, bpy.data). Searches BOTH the official Python API reference AND manual scripting pages.",
        "parameters": {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "description": "API operation or concept (e.g., 'bpy.ops.fluid', 'export script', 'volume')"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results (default: 5)"
                },
                "compact": {
                    "type": "boolean",
                    "description": "Minimal output mode"
                },
                "api_only": {
                    "type": "boolean",
                    "description": "If true, only search official API docs (not manual)"
                }
            },
            "required": ["operation", "limit", "compact", "api_only"],
            "additionalProperties": False
        },
        "strict": True
    },
    {
        "type": "function",
        "name": "search_nodes",
        "description": "Search for shader nodes, compositor nodes, and geometry nodes. Optimized for finding node documentation, especially volume-related nodes.",
        "parameters": {
            "type": "object",
            "properties": {
                "node_type": {
                    "type": "string",
                    "description": "Node name or type (e.g., 'Principled Volume', 'Volume Scatter', 'Math')"
                },
                "category": {
                    "type": ["string", "null"],
                    "description": "Optional node category ('shader', 'compositor', 'geometry'). Pass null if not needed."
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results (default: 5)"
                },
                "compact": {
                    "type": "boolean",
                    "description": "Minimal output mode"
                }
            },
            "required": ["node_type", "category", "limit", "compact"],
            "additionalProperties": False
        },
        "strict": True
    },
    {
        "type": "function",
        "name": "search_modifiers",
        "description": "Search for modifier documentation. Find information about mesh, volume, and simulation modifiers.",
        "parameters": {
            "type": "object",
            "properties": {
                "modifier_name": {
                    "type": ["string", "null"],
                    "description": "Optional specific modifier (e.g., 'Volume Displace', 'Fluid'). Pass null to get overview of all modifiers."
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results (default: 5)"
                },
                "compact": {
                    "type": "boolean",
                    "description": "Minimal output mode"
                }
            },
            "required": ["modifier_name", "limit", "compact"],
            "additionalProperties": False
        },
        "strict": True
    },
    {
        "type": "function",
        "name": "read_page",
        "description": "Read the full content of a manual or API page with enhanced formatting. Use paths returned from search results.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to HTML file (e.g., 'render/cycles/world_settings.html', 'bpy.ops.mesh.html')"
                },
                "max_length": {
                    "type": "integer",
                    "description": "Maximum characters to return (default: 4000, use 0 for unlimited)"
                },
                "source": {
                    "type": "string",
                    "description": "'manual', 'python_api', or 'auto' (tries both)"
                }
            },
            "required": ["path", "max_length", "source"],
            "additionalProperties": False
        },
        "strict": True
    },
    {
        "type": "function",
        "name": "list_api_modules",
        "description": "List available Python API modules from the official Blender Python API reference.",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": ["string", "null"],
                    "description": "Optional filter by module category (e.g., 'bpy', 'bmesh', 'aud'). Pass null for all."
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum modules to list (default: 30)"
                }
            },
            "required": ["category", "limit"],
            "additionalProperties": False
        },
        "strict": True
    },
    {
        "type": "function",
        "name": "search_bpy_operators",
        "description": "Search for Blender Python operators (bpy.ops.*) by category.",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "description": "Operator category (e.g., 'mesh', 'object', 'fluid', 'curve', 'anim')"
                },
                "operation": {
                    "type": ["string", "null"],
                    "description": "Optional specific operation name to search for. Pass null for all in category."
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results (default: 5)"
                }
            },
            "required": ["category", "operation", "limit"],
            "additionalProperties": False
        },
        "strict": True
    },
    {
        "type": "function",
        "name": "search_bpy_types",
        "description": "Search for Blender Python types (bpy.types.*) by name.",
        "parameters": {
            "type": "object",
            "properties": {
                "typename": {
                    "type": "string",
                    "description": "Type name to search for (e.g., 'Object', 'Mesh', 'FluidModifier', 'Volume')"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results (default: 5)"
                }
            },
            "required": ["typename", "limit"],
            "additionalProperties": False
        },
        "strict": True
    },
    {
        "type": "function",
        "name": "search_semantic",
        "description": "Semantic similarity search using AI embeddings. Finds conceptually related content even without exact keyword matches. Best for natural language questions.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language query (e.g., 'how to make realistic smoke effects', 'rendering transparent volumetric clouds')"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum results (default: 5)"
                },
                "compact": {
                    "type": "boolean",
                    "description": "Minimal output mode"
                }
            },
            "required": ["query", "limit", "compact"],
            "additionalProperties": False
        },
        "strict": True
    }
]

# Tool name to function mapping (for MCP bridge)
TOOL_NAME_MAP = {
    "search_manual": "search_manual",
    "search_tutorials": "search_tutorials",
    "browse_hierarchy": "browse_hierarchy",
    "search_vdb_workflow": "search_vdb_workflow",
    "search_python_api": "search_python_api",
    "search_nodes": "search_nodes",
    "search_modifiers": "search_modifiers",
    "read_page": "read_page",
    "list_api_modules": "list_api_modules",
    "search_bpy_operators": "search_bpy_operators",
    "search_bpy_types": "search_bpy_types",
    "search_semantic": "search_semantic"
}


def get_tool_names() -> List[str]:
    """Get list of all available tool names."""
    return list(TOOL_NAME_MAP.keys())


def get_tool_definition(name: str) -> Dict[str, Any] | None:
    """Get tool definition by name."""
    for tool in BLENDER_MANUAL_TOOLS:
        if tool["name"] == name:
            return tool
    return None
