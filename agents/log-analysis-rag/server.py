#!/usr/bin/env python3
"""
PlasmaDX Log Analysis RAG - FastMCP Server
Multi-agent RAG system for DirectX 12 rendering diagnostics
"""

import sys
from pathlib import Path
from typing import Optional

# Add directories to Python path
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

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

# Create FastMCP server
mcp = FastMCP("log-analysis-rag")


@mcp.tool()
async def ingest_logs(
    path: Optional[str] = None,
    include_pix: bool = True,
    max_files: int = 10
) -> str:
    """Index logs/PIX/buffers into RAG database"""
    result = await ingest_logs_tool({
        "path": path,
        "include_pix": include_pix,
        "max_files": max_files
    })
    if isinstance(result, dict) and "content" in result:
        return str(result["content"])
    return str(result)


@mcp.tool()
async def diagnose_issue(
    question: str,
    confidence_threshold: float = 0.7,
    context: Optional[dict] = None
) -> str:
    """Run full LangGraph self-correcting diagnostic workflow"""
    result = await diagnose_issue_tool({
        "question": question,
        "confidence_threshold": confidence_threshold,
        "context": context
    })
    if isinstance(result, dict) and "content" in result:
        return str(result["content"])
    return str(result)


@mcp.tool()
async def query_logs(
    semantic_query: str,
    top_k: int = 10,
    filters: Optional[dict] = None
) -> str:
    """Direct hybrid retrieval (BM25 + FAISS) bypassing full workflow"""
    result = await query_logs_tool({
        "semantic_query": semantic_query,
        "top_k": top_k,
        "filters": filters
    })
    if isinstance(result, dict) and "content" in result:
        return str(result["content"])
    return str(result)


@mcp.tool()
async def analyze_pix_capture(
    capture_path: Optional[str] = None,
    extract_events: bool = True
) -> str:
    """Extract PIX GPU capture metadata and timeline events"""
    result = await analyze_pix_capture_tool({
        "capture_path": capture_path,
        "extract_events": extract_events
    })
    if isinstance(result, dict) and "content" in result:
        return str(result["content"])
    return str(result)


@mcp.tool()
async def read_buffer_dump(
    buffer_path: str,
    buffer_type: Optional[str] = None,
    max_entries: int = 10
) -> str:
    """Parse binary GPU buffer dumps"""
    result = await read_buffer_dump_tool({
        "buffer_path": buffer_path,
        "buffer_type": buffer_type,
        "max_entries": max_entries
    })
    if isinstance(result, dict) and "content" in result:
        return str(result["content"])
    return str(result)


@mcp.tool()
async def route_to_specialist(
    issue_description: str,
    symptoms: Optional[list] = None,
    context: Optional[dict] = None
) -> str:
    """Recommend which specialist agent should handle the issue"""
    result = await route_to_specialist_tool({
        "issue_description": issue_description,
        "symptoms": symptoms,
        "context": context
    })
    if isinstance(result, dict) and "content" in result:
        return str(result["content"])
    return str(result)


if __name__ == "__main__":
    mcp.run()
