"""
RTXDI Quality Analyzer Agent - MCP Server Mode (FIXED)

Provides diagnostic tools for RTXDI rendering performance and quality analysis
via the Claude Agent SDK MCP protocol.
"""

import asyncio
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Claude Agent SDK imports
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

# Import custom tools
from .tools.performance_comparison import compare_performance, format_comparison_report
from .tools.pix_analysis import analyze_pix_capture, format_pix_report
from .tools.ml_visual_comparison import compare_screenshots_ml, format_comparison_report as format_ml_report

# Load environment variables
load_dotenv()

# Initialize analyzer (module-level)
PROJECT_ROOT = os.getenv("PROJECT_ROOT", "/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean")
PIX_PATH = os.getenv("PIX_PATH", "/mnt/c/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64")

# Create MCP server (module-level, NOT inside main())
server = Server("rtxdi-quality-analyzer")


# Register tools using decorators at module level
@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available RTXDI diagnostic tools"""
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
        ),
        Tool(
            name="compare_screenshots_ml",
            description="ML-powered before/after screenshot comparison using LPIPS perceptual similarity (pre-trained, no training required). Provides human-aligned visual analysis with ~92% correlation to human judgment. Generates difference heatmaps and detailed similarity metrics.",
            inputSchema={
                "type": "object",
                "properties": {
                    "before_path": {
                        "type": "string",
                        "description": "Path to 'before' screenshot (e.g., /mnt/c/Users/dilli/Pictures/Screenshots/screenshot1.png)"
                    },
                    "after_path": {
                        "type": "string",
                        "description": "Path to 'after' screenshot (e.g., /mnt/c/Users/dilli/Pictures/Screenshots/screenshot2.png)"
                    },
                    "save_heatmap": {
                        "type": "boolean",
                        "description": "Save difference heatmap to PIX/heatmaps/ directory (default: true)",
                        "default": True
                    }
                },
                "required": ["before_path", "after_path"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool invocations"""

    if name == "compare_performance":
        # Run performance comparison
        results = await compare_performance(
            legacy_log=arguments.get("legacy_log"),
            rtxdi_m4_log=arguments.get("rtxdi_m4_log"),
            rtxdi_m5_log=arguments.get("rtxdi_m5_log"),
            project_root=PROJECT_ROOT
        )
        report = format_comparison_report(results)
        return [TextContent(type="text", text=report)]

    elif name == "analyze_pix_capture":
        # Run PIX analysis
        results = await analyze_pix_capture(
            capture_path=arguments.get("capture_path"),
            project_root=PROJECT_ROOT,
            analyze_buffers=arguments.get("analyze_buffers", True)
        )
        report = format_pix_report(results)
        return [TextContent(type="text", text=report)]

    elif name == "compare_screenshots_ml":
        # Run ML-powered visual comparison
        results = await compare_screenshots_ml(
            before_path=arguments.get("before_path"),
            after_path=arguments.get("after_path"),
            save_heatmap=arguments.get("save_heatmap", True),
            project_root=PROJECT_ROOT
        )
        report = format_ml_report(results)
        return [TextContent(type="text", text=report)]

    else:
        raise ValueError(f"Unknown tool: {name}")


async def main():
    """Run MCP server using stdio transport"""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
