"""
Shared module for blender-vfx-orchestrator.

This module contains extracted functionality from blender-manual MCP server
as standalone functions wrapped with OpenAI Agents SDK function_tool decorator.

This solves the MCP nesting architectural issue where spawning child MCP clients
from within MCP tool handlers causes anyio TaskGroup conflicts.

Key modules:
- blender_docs_tools: 12 Blender documentation search tools as function_tools
"""

from .blender_docs_tools import (
    # Index management
    ensure_index_loaded,
    get_index_stats,

    # Search tools (function_tools for agents)
    search_manual,
    search_tutorials,
    browse_hierarchy,
    search_vdb_workflow,
    search_python_api,
    search_nodes,
    search_modifiers,
    read_page,
    list_api_modules,
    search_bpy_operators,
    search_bpy_types,
    search_semantic,

    # Direct callable implementations (for non-agent code)
    search_semantic_impl,
)

__all__ = [
    # Index management
    "ensure_index_loaded",
    "get_index_stats",

    # Search tools (function_tools for agents)
    "search_manual",
    "search_tutorials",
    "browse_hierarchy",
    "search_vdb_workflow",
    "search_python_api",
    "search_nodes",
    "search_modifiers",
    "read_page",
    "list_api_modules",
    "search_bpy_operators",
    "search_bpy_types",
    "search_semantic",

    # Direct callable implementations (for non-agent code)
    "search_semantic_impl",
]
