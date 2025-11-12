#!/usr/bin/env python3
"""
Flat MCP server test - exact pix-debug structure
Single file, module-level server, SDK 0.1.1
"""

import asyncio
import os
from dotenv import load_dotenv
from mcp.server import Server
from mcp.types import TextContent, Tool

# Load environment
load_dotenv()

PROJECT_ROOT = os.getenv("PROJECT_ROOT", "/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean")

# Create MCP server at module level (like pix-debug)
server = Server("rtxdi-flat-test")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools"""
    return [
        Tool(
            name="test_echo",
            description="Simple echo test to verify MCP connection",
            inputSchema={
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Message to echo"}
                },
                "required": ["message"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Execute a tool"""
    if name == "test_echo":
        msg = arguments.get("message", "No message provided")
        return [TextContent(type="text", text=f"✓ Echo successful: {msg}")]
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
