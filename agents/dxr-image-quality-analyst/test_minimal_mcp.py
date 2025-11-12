#!/usr/bin/env python3
"""
Minimal MCP server test - exactly matching pix-debug structure
"""
import asyncio
from mcp.server import Server
from mcp.types import TextContent, Tool

# Create server at module level
server = Server("test-minimal")

@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="test_echo",
            description="Simple echo test",
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
    if name == "test_echo":
        msg = arguments.get("message", "No message")
        return [TextContent(type="text", text=f"Echo: {msg}")]
    return [TextContent(type="text", text=f"Unknown tool: {name}")]

async def main():
    from mcp.server.stdio import stdio_server
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    asyncio.run(main())
