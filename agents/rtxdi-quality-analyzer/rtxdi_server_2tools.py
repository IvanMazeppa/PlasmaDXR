#!/usr/bin/env python3
"""
RTXDI Quality Analyzer - 2 Tools Only (No ML)
Testing if ML tool causes timeout
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

# Import only 2 tools (skip ML)
from tools.performance_comparison import compare_performance, format_comparison_report
from tools.pix_analysis import analyze_pix_capture, format_pix_report

# Load environment
load_dotenv()

PROJECT_ROOT = os.getenv("PROJECT_ROOT", "/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean")

# Create MCP server at module level
server = Server("rtxdi-2tools-test")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools"""
    return [
        Tool(
            name="compare_performance",
            description="Compare RTXDI performance metrics between legacy renderer, RTXDI M4, and RTXDI M5",
            inputSchema={
                "type": "object",
                "properties": {
                    "legacy_log": {
                        "type": "string",
                        "description": "Path to legacy renderer log file (optional)"
                    },
                    "rtxdi_m4_log": {
                        "type": "string",
                        "description": "Path to RTXDI M4 log file (optional)"
                    },
                    "rtxdi_m5_log": {
                        "type": "string",
                        "description": "Path to RTXDI M5 log file (optional)"
                    }
                }
            }
        ),
        Tool(
            name="analyze_pix_capture",
            description="Analyze PIX GPU capture for RTXDI bottlenecks and performance issues",
            inputSchema={
                "type": "object",
                "properties": {
                    "capture_path": {
                        "type": "string",
                        "description": "Path to .wpix capture file (optional, will auto-detect latest if not provided)"
                    },
                    "analyze_buffers": {
                        "type": "boolean",
                        "description": "Also analyze buffer dumps if available (default: true)"
                    }
                }
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Execute a tool"""

    if name == "compare_performance":
        results = await compare_performance(
            legacy_log=arguments.get("legacy_log"),
            rtxdi_m4_log=arguments.get("rtxdi_m4_log"),
            rtxdi_m5_log=arguments.get("rtxdi_m5_log"),
            project_root=PROJECT_ROOT
        )
        report = format_comparison_report(results)
        return [TextContent(type="text", text=report)]

    elif name == "analyze_pix_capture":
        results = await analyze_pix_capture(
            capture_path=arguments.get("capture_path"),
            project_root=PROJECT_ROOT,
            analyze_buffers=arguments.get("analyze_buffers", True)
        )
        report = format_pix_report(results)
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
