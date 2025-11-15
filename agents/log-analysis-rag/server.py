#!/usr/bin/env python3
"""
PlasmaDX Log Analysis RAG - Standard MCP Server
Multi-agent RAG system for DirectX 12 rendering diagnostics
"""

import asyncio
import sys
from pathlib import Path

# Add directories to Python path
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from dotenv import load_dotenv
from mcp.server import Server
from mcp.types import TextContent, Tool

# Import MCP tools
from mcp_server.tools import (
    ingest_logs_tool,
    diagnose_issue_tool,
    query_logs_tool,
    analyze_pix_capture_tool,
    read_buffer_dump_tool,
    route_to_specialist_tool
)

# Load environment
load_dotenv()

# Create MCP server
server = Server("log-analysis-rag")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available diagnostic tools"""
    return [
        Tool(
            name="ingest_logs",
            description="Index logs/PIX/buffers into RAG database",
            inputSchema={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to log directory"
                    },
                    "include_pix": {
                        "type": "boolean",
                        "description": "Also parse PIX captures (default: true)"
                    },
                    "max_files": {
                        "type": "number",
                        "description": "Maximum number of log files (default: 10)"
                    }
                }
            }
        ),
        Tool(
            name="diagnose_issue",
            description="Run full LangGraph self-correcting diagnostic workflow",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Diagnostic question (required)"
                    },
                    "confidence_threshold": {
                        "type": "number",
                        "description": "Minimum confidence (default: 0.7)"
                    },
                    "context": {
                        "type": "object",
                        "description": "Optional context dict"
                    }
                },
                "required": ["question"]
            }
        ),
        Tool(
            name="query_logs",
            description="Direct hybrid retrieval (BM25 + FAISS) bypassing full workflow",
            inputSchema={
                "type": "object",
                "properties": {
                    "semantic_query": {
                        "type": "string",
                        "description": "Natural language query (required)"
                    },
                    "top_k": {
                        "type": "number",
                        "description": "Number of results (default: 10)"
                    },
                    "filters": {
                        "type": "object",
                        "description": "Optional metadata filters"
                    }
                },
                "required": ["semantic_query"]
            }
        ),
        Tool(
            name="analyze_pix_capture",
            description="Extract PIX GPU capture metadata and timeline events",
            inputSchema={
                "type": "object",
                "properties": {
                    "capture_path": {
                        "type": "string",
                        "description": "Path to .wpix file (optional, auto-detects latest)"
                    },
                    "extract_events": {
                        "type": "boolean",
                        "description": "Export event timeline (default: true)"
                    }
                }
            }
        ),
        Tool(
            name="read_buffer_dump",
            description="Parse binary GPU buffer dumps",
            inputSchema={
                "type": "object",
                "properties": {
                    "buffer_path": {
                        "type": "string",
                        "description": "Path to .bin file (required)"
                    },
                    "buffer_type": {
                        "type": "string",
                        "description": "Buffer type (particles, reservoirs, rtLighting)"
                    },
                    "max_entries": {
                        "type": "number",
                        "description": "Max entries to show (default: 10)"
                    }
                },
                "required": ["buffer_path"]
            }
        ),
        Tool(
            name="route_to_specialist",
            description="Recommend which specialist agent should handle the issue",
            inputSchema={
                "type": "object",
                "properties": {
                    "issue_description": {
                        "type": "string",
                        "description": "Description of rendering issue (required)"
                    },
                    "symptoms": {
                        "type": "array",
                        "description": "List of observed symptoms"
                    },
                    "context": {
                        "type": "object",
                        "description": "Optional context"
                    }
                },
                "required": ["issue_description"]
            }
        )
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls"""
    try:
        # Route to appropriate tool function
        if name == "ingest_logs":
            result = await ingest_logs_tool(arguments)
        elif name == "diagnose_issue":
            result = await diagnose_issue_tool(arguments)
        elif name == "query_logs":
            result = await query_logs_tool(arguments)
        elif name == "analyze_pix_capture":
            result = await analyze_pix_capture_tool(arguments)
        elif name == "read_buffer_dump":
            result = await read_buffer_dump_tool(arguments)
        elif name == "route_to_specialist":
            result = await route_to_specialist_tool(arguments)
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

        # Extract text from result content
        if isinstance(result, dict) and "content" in result:
            return result["content"]
        else:
            return [TextContent(type="text", text=str(result))]

    except Exception as e:
        return [TextContent(type="text", text=f"Error executing {name}: {str(e)}")]


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
