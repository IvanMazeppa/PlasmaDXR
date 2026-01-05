"""
MCP Bridge Client for blender-manual Server

This client provides async access to the blender-manual MCP server's tools,
allowing the GPT-5.2 tool calling agent to execute documentation searches.

Two modes of operation:
1. Direct import: Import and call blender_server functions directly (faster)
2. HTTP bridge: Connect via MCP HTTP transport (for remote/isolated operation)

Default mode is direct import for performance and simplicity.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass


@dataclass
class MCPToolResult:
    """Result from calling an MCP tool."""
    success: bool
    data: Any
    error: Optional[str] = None
    tool_name: str = ""


class MCPBridgeClient:
    """
    Client for calling blender-manual MCP server tools.

    Uses direct import mode by default for best performance.
    """

    def __init__(self, mode: str = "direct"):
        """
        Initialize the MCP bridge client.

        Args:
            mode: "direct" for local import, "http" for HTTP transport (future)
        """
        self.mode = mode
        self._blender_server = None
        self._initialized = False

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def initialize(self):
        """Initialize the client and load blender-manual module."""
        if self._initialized:
            return

        if self.mode == "direct":
            await self._init_direct()
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        self._initialized = True

    async def _init_direct(self):
        """Initialize direct import mode."""
        # Add blender-manual to path if needed
        project_root = Path(__file__).resolve().parents[3]
        manual_dir = project_root / "agents" / "blender-manual"

        if str(manual_dir) not in sys.path:
            sys.path.insert(0, str(manual_dir))

        # Import blender_server (this triggers index loading)
        import blender_server as blender_manual
        self._blender_server = blender_manual

    async def close(self):
        """Close the client."""
        self._initialized = False

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> MCPToolResult:
        """
        Call an MCP tool by name with given arguments.

        Args:
            tool_name: Name of the tool to call
            arguments: Dictionary of arguments for the tool

        Returns:
            MCPToolResult with success/failure and data
        """
        if not self._initialized:
            await self.initialize()

        if self.mode == "direct":
            return await self._call_direct(tool_name, arguments)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    async def _call_direct(self, tool_name: str, arguments: Dict[str, Any]) -> MCPToolResult:
        """Call tool using direct import."""
        try:
            # Get the function from blender_server
            if not hasattr(self._blender_server, tool_name):
                return MCPToolResult(
                    success=False,
                    data=None,
                    error=f"Unknown tool: {tool_name}",
                    tool_name=tool_name
                )

            func = getattr(self._blender_server, tool_name)

            # Call the function (it returns a string)
            # Run in executor since some functions may do I/O
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: func(**arguments))

            # Try to parse as JSON
            try:
                parsed = json.loads(result)
            except (json.JSONDecodeError, TypeError):
                parsed = result

            return MCPToolResult(
                success=True,
                data=parsed,
                tool_name=tool_name
            )

        except Exception as e:
            return MCPToolResult(
                success=False,
                data=None,
                error=str(e),
                tool_name=tool_name
            )

    # Convenience methods for each tool

    async def search_manual(self, query: str, limit: int = 5, offset: int = 0, compact: bool = False) -> MCPToolResult:
        """Search the Blender manual."""
        return await self.call_tool("search_manual", {
            "query": query,
            "limit": limit,
            "offset": offset,
            "compact": compact
        })

    async def search_tutorials(self, topic: str, technique: Optional[str] = None, limit: int = 5, compact: bool = False) -> MCPToolResult:
        """Search for tutorials."""
        args = {"topic": topic, "limit": limit, "compact": compact}
        if technique:
            args["technique"] = technique
        return await self.call_tool("search_tutorials", args)

    async def browse_hierarchy(self, path: Optional[str] = None) -> MCPToolResult:
        """Browse the manual hierarchy."""
        args = {}
        if path:
            args["path"] = path
        return await self.call_tool("browse_hierarchy", args)

    async def search_vdb_workflow(self, query: str, limit: int = 5, offset: int = 0, compact: bool = False) -> MCPToolResult:
        """Search for VDB workflows."""
        return await self.call_tool("search_vdb_workflow", {
            "query": query,
            "limit": limit,
            "offset": offset,
            "compact": compact
        })

    async def search_python_api(self, operation: str, limit: int = 5, compact: bool = False, api_only: bool = False) -> MCPToolResult:
        """Search Python API documentation."""
        return await self.call_tool("search_python_api", {
            "operation": operation,
            "limit": limit,
            "compact": compact,
            "api_only": api_only
        })

    async def search_nodes(self, node_type: str, category: Optional[str] = None, limit: int = 5, compact: bool = False) -> MCPToolResult:
        """Search for shader/compositor/geometry nodes."""
        args = {"node_type": node_type, "limit": limit, "compact": compact}
        if category:
            args["category"] = category
        return await self.call_tool("search_nodes", args)

    async def search_modifiers(self, modifier_name: Optional[str] = None, limit: int = 5, compact: bool = False) -> MCPToolResult:
        """Search for modifiers."""
        args = {"limit": limit, "compact": compact}
        if modifier_name:
            args["modifier_name"] = modifier_name
        return await self.call_tool("search_modifiers", args)

    async def read_page(self, path: str, max_length: int = 4000, source: str = "auto") -> MCPToolResult:
        """Read a manual page."""
        return await self.call_tool("read_page", {
            "path": path,
            "max_length": max_length,
            "source": source
        })

    async def list_api_modules(self, category: Optional[str] = None, limit: int = 30) -> MCPToolResult:
        """List API modules."""
        args = {"limit": limit}
        if category:
            args["category"] = category
        return await self.call_tool("list_api_modules", args)

    async def search_bpy_operators(self, category: str, operation: Optional[str] = None, limit: int = 5) -> MCPToolResult:
        """Search bpy.ops operators."""
        args = {"category": category, "limit": limit}
        if operation:
            args["operation"] = operation
        return await self.call_tool("search_bpy_operators", args)

    async def search_bpy_types(self, typename: str, limit: int = 5) -> MCPToolResult:
        """Search bpy.types types."""
        return await self.call_tool("search_bpy_types", {
            "typename": typename,
            "limit": limit
        })

    async def search_semantic(self, query: str, limit: int = 5, compact: bool = False) -> MCPToolResult:
        """Semantic similarity search."""
        return await self.call_tool("search_semantic", {
            "query": query,
            "limit": limit,
            "compact": compact
        })


# Synchronous wrapper for non-async contexts
class MCPBridgeClientSync:
    """
    Synchronous wrapper for MCPBridgeClient.

    Use this when you need to call MCP tools from synchronous code.
    """

    def __init__(self, mode: str = "direct"):
        self._async_client = MCPBridgeClient(mode=mode)
        self._loop = None

    def __enter__(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._async_client.initialize())
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._loop.run_until_complete(self._async_client.close())
        self._loop.close()

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> MCPToolResult:
        """Call an MCP tool synchronously."""
        return self._loop.run_until_complete(
            self._async_client.call_tool(tool_name, arguments)
        )


# Test the client
if __name__ == "__main__":
    async def test():
        async with MCPBridgeClient() as client:
            # Test search_manual
            result = await client.search_manual("volume rendering", limit=3)
            print(f"search_manual success: {result.success}")
            if result.success:
                print(f"  Found results about volume rendering")
            else:
                print(f"  Error: {result.error}")

            # Test search_vdb_workflow
            result = await client.search_vdb_workflow("export openvdb", limit=2)
            print(f"search_vdb_workflow success: {result.success}")

            # Test search_python_api
            result = await client.search_python_api("bpy.ops.fluid", limit=2)
            print(f"search_python_api success: {result.success}")

    asyncio.run(test())
