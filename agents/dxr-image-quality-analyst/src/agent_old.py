"""
RTXDI Quality Analyzer Agent - MCP Server Mode

Provides diagnostic tools for RTXDI rendering performance and quality analysis
via the Claude Agent SDK MCP protocol.

Usage:
    # When using Claude Code with Claude Max subscription (recommended):
    # 1. Register this agent in Claude Code's MCP config
    # 2. Claude Code will automatically start and connect to this agent
    # 3. No API key needed - authentication handled by Claude Code

    # Or run standalone for testing:
    python -m src.agent

Note: This agent is designed to run within Claude Code's context.
Authentication is handled automatically when running via Claude Code.
"""

import asyncio
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Claude Agent SDK imports
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import Tool, TextContent
    import mcp.server.stdio
except ImportError as e:
    print(f"Error importing MCP SDK: {e}")
    print("Make sure claude-agent-sdk is installed: pip install claude-agent-sdk")
    raise

# Import custom tools
from .tools.performance_comparison import compare_performance, format_comparison_report
from .tools.pix_analysis import analyze_pix_capture, format_pix_report
from .tools.ml_visual_comparison import compare_screenshots_ml, format_comparison_report as format_ml_report


# Load environment variables
load_dotenv()


class RTXDIAnalyzer:
    """
    RTXDI Quality Analyzer Agent

    Provides two core diagnostic capabilities:
    1. Performance comparison between legacy, RTXDI M4, and RTXDI M5
    2. PIX capture analysis for bottleneck identification
    """

    def __init__(self, project_root: Optional[str] = None):
        """
        Initialize the RTXDI analyzer

        Args:
            project_root: Path to PlasmaDX-Clean project root
        """
        self.project_root = project_root or os.getenv(
            "PROJECT_ROOT",
            "/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean"
        )
        self.pix_path = os.getenv(
            "PIX_PATH",
            "/mnt/c/Program Files (x86)/Windows Kits/10/bin/10.0.26100.0/x64"
        )

    async def run_performance_comparison(
        self,
        legacy_log: Optional[str] = None,
        rtxdi_m4_log: Optional[str] = None,
        rtxdi_m5_log: Optional[str] = None
    ) -> str:
        """
        Compare performance between renderer modes

        Args:
            legacy_log: Path to legacy renderer log
            rtxdi_m4_log: Path to RTXDI M4 log
            rtxdi_m5_log: Path to RTXDI M5 log

        Returns:
            Formatted comparison report
        """
        results = await compare_performance(
            legacy_log=legacy_log,
            rtxdi_m4_log=rtxdi_m4_log,
            rtxdi_m5_log=rtxdi_m5_log,
            project_root=self.project_root
        )

        return format_comparison_report(results)

    async def run_pix_analysis(
        self,
        capture_path: Optional[str] = None,
        analyze_buffers: bool = True
    ) -> str:
        """
        Analyze PIX capture for RTXDI bottlenecks

        Args:
            capture_path: Path to .wpix capture file (auto-detect if not provided)
            analyze_buffers: Also analyze buffer dumps

        Returns:
            Formatted PIX analysis report
        """
        results = await analyze_pix_capture(
            capture_path=capture_path,
            project_root=self.project_root,
            analyze_buffers=analyze_buffers
        )

        return format_pix_report(results)

    async def run_ml_comparison(
        self,
        before_path: str,
        after_path: str,
        save_heatmap: bool = True
    ) -> str:
        """
        ML-powered screenshot comparison using LPIPS perceptual similarity

        Args:
            before_path: Path to "before" screenshot
            after_path: Path to "after" screenshot
            save_heatmap: Save difference heatmap to file

        Returns:
            Formatted ML comparison report
        """
        results = await compare_screenshots_ml(
            before_path=before_path,
            after_path=after_path,
            save_heatmap=save_heatmap,
            project_root=self.project_root
        )

        return format_ml_report(results)

    async def diagnose(self, query: str) -> str:
        """
        Main diagnostic entry point - interprets user query and runs appropriate analysis

        Args:
            query: User's diagnostic question

        Returns:
            Diagnostic response
        """
        query_lower = query.lower()

        # Performance comparison queries
        if any(keyword in query_lower for keyword in ["performance", "fps", "faster", "slower", "compare"]):
            return await self.run_performance_comparison()

        # PIX analysis queries
        elif any(keyword in query_lower for keyword in ["pix", "bottleneck", "capture", "timing"]):
            return await self.run_pix_analysis()

        # General diagnostic
        else:
            return (
                "RTXDI Quality Analyzer - Available Diagnostics:\n\n"
                "1. Performance Comparison - Compare FPS/frame times between legacy, RTXDI M4, M5\n"
                "   Keywords: 'performance', 'fps', 'faster', 'slower', 'compare'\n\n"
                "2. PIX Capture Analysis - Identify bottlenecks in RTXDI pipeline\n"
                "   Keywords: 'pix', 'bottleneck', 'capture', 'timing'\n\n"
                "Example queries:\n"
                "  - 'Why isn't RTXDI faster than the legacy renderer?'\n"
                "  - 'Analyze the latest PIX capture for bottlenecks'\n"
                "  - 'Compare performance between all three renderers'\n"
            )


async def main():
    """
    Main entry point for MCP server mode

    Creates and runs an MCP server with RTXDI diagnostic tools.
    """
    # Initialize analyzer
    analyzer = RTXDIAnalyzer()

    # Create MCP server
    server = Server("rtxdi-quality-analyzer")

    # Register tool: compare_performance
    @server.list_tools()
    async def list_tools() -> list[Tool]:
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
                            "description": "Path to 'before' screenshot (e.g., /mnt/c/Users/dilli/Pictures/PlasmaDX-Screenshots/rtxdi_m4_before.png)"
                        },
                        "after_path": {
                            "type": "string",
                            "description": "Path to 'after' screenshot (e.g., /mnt/c/Users/dilli/Pictures/PlasmaDX-Screenshots/rtxdi_m5_after.png)"
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

    # Register tool handler: compare_performance
    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        if name == "compare_performance":
            # Run performance comparison
            results = await analyzer.run_performance_comparison(
                legacy_log=arguments.get("legacy_log"),
                rtxdi_m4_log=arguments.get("rtxdi_m4_log"),
                rtxdi_m5_log=arguments.get("rtxdi_m5_log")
            )

            return [TextContent(
                type="text",
                text=results
            )]

        elif name == "analyze_pix_capture":
            # Run PIX analysis
            results = await analyzer.run_pix_analysis(
                capture_path=arguments.get("capture_path"),
                analyze_buffers=arguments.get("analyze_buffers", True)
            )

            return [TextContent(
                type="text",
                text=results
            )]

        elif name == "compare_screenshots_ml":
            # Run ML-powered visual comparison
            results = await analyzer.run_ml_comparison(
                before_path=arguments.get("before_path"),
                after_path=arguments.get("after_path"),
                save_heatmap=arguments.get("save_heatmap", True)
            )

            return [TextContent(
                type="text",
                text=results
            )]

        else:
            raise ValueError(f"Unknown tool: {name}")

    # Run the server using stdio transport
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
