#!/usr/bin/env python3
"""
RTXDI Quality Analyzer - Flat MCP Server
Single-file server with direct imports (no package structure)
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

# Import tools directly (not as packages)
from tools.performance_comparison import compare_performance, format_comparison_report
from tools.pix_analysis import analyze_pix_capture, format_pix_report
from tools.ml_visual_comparison import compare_screenshots_ml, format_comparison_report as format_ml_report

# Load environment
load_dotenv()

PROJECT_ROOT = os.getenv("PROJECT_ROOT", "/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean")
PIX_PATH = os.getenv("PIX_PATH", "/mnt/c/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64")

# Create MCP server at module level (like pix-debug)
server = Server("rtxdi-quality-analyzer")


async def list_recent_screenshots(limit: int = 10) -> str:
    """List recent screenshots from project directory"""
    screenshots_dir = Path(PROJECT_ROOT) / "screenshots"

    if not screenshots_dir.exists():
        return "No screenshots directory found. Use Ctrl+Shift+S to capture screenshots."

    # Get all PNG files sorted by modification time
    screenshots = sorted(
        screenshots_dir.glob("*.png"),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )[:limit]

    if not screenshots:
        return "No screenshots found. Use Ctrl+Shift+S to capture screenshots."

    result = f"Recent screenshots (showing {len(screenshots)} most recent):\n\n"
    for i, screenshot in enumerate(screenshots, 1):
        mtime = screenshot.stat().st_mtime
        from datetime import datetime
        timestamp = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        size_mb = screenshot.stat().st_size / (1024 * 1024)
        result += f"{i}. {screenshot.name}\n"
        result += f"   Path: {screenshot}\n"
        result += f"   Time: {timestamp}\n"
        result += f"   Size: {size_mb:.2f} MB\n\n"

    return result


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools"""
    return [
        Tool(
            name="list_recent_screenshots",
            description="List recent screenshots from project directory (sorted by time, newest first). Useful for finding screenshots to compare.",
            inputSchema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of screenshots to list (default: 10)",
                        "default": 10
                    }
                }
            }
        ),
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
    """Execute a tool"""

    if name == "list_recent_screenshots":
        result = await list_recent_screenshots(
            limit=arguments.get("limit", 10)
        )
        return [TextContent(type="text", text=result)]

    elif name == "compare_performance":
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

    elif name == "compare_screenshots_ml":
        results = await compare_screenshots_ml(
            before_path=arguments.get("before_path"),
            after_path=arguments.get("after_path"),
            save_heatmap=arguments.get("save_heatmap", True),
            project_root=PROJECT_ROOT
        )
        report = format_ml_report(results)
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
