#!/usr/bin/env python3
"""
PlasmaDX Log Analysis RAG Agent
Multi-agent RAG system for DirectX 12 rendering diagnostics

This agent analyzes application logs, PIX GPU captures, and buffer dumps
to diagnose RT lighting, RTXDI, shadow, and performance issues using
self-correcting workflow with hybrid retrieval and hallucination detection.

Usage:
    python main.py "Why does GPU hang at 2045 particles?"
    python main.py --ingest /path/to/logs
    python main.py --query "Find Map() failures"
"""

import asyncio
import sys
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, AssistantMessage, TextBlock
from claude_agent_sdk import create_sdk_mcp_server, tool

# Load environment variables
load_dotenv()

# Import MCP tools
from mcp_server.tools import (
    ingest_logs_tool,
    diagnose_issue_tool,
    query_logs_tool,
    analyze_pix_capture_tool,
    read_buffer_dump_tool,
    route_to_specialist_tool
)

# ============================================================================
# MCP Tool Decorators (wrap async functions)
# ============================================================================

@tool(
    "ingest_logs",
    "Index logs/PIX/buffers into RAG database",
    {
        "path": {"type": "string", "description": "Path to log directory"},
        "include_pix": {"type": "boolean", "description": "Also parse PIX captures (default: true)"},
        "max_files": {"type": "number", "description": "Maximum number of log files (default: 10)"}
    }
)
async def ingest_logs(args):
    return await ingest_logs_tool(args)


@tool(
    "diagnose_issue",
    "Run full LangGraph self-correcting diagnostic workflow",
    {
        "question": {"type": "string", "description": "Diagnostic question (required)"},
        "confidence_threshold": {"type": "number", "description": "Minimum confidence (default: 0.7)"},
        "context": {"type": "object", "description": "Optional context dict"}
    }
)
async def diagnose_issue(args):
    return await diagnose_issue_tool(args)


@tool(
    "query_logs",
    "Direct hybrid retrieval (BM25 + FAISS) bypassing full workflow",
    {
        "semantic_query": {"type": "string", "description": "Natural language query (required)"},
        "top_k": {"type": "number", "description": "Number of results (default: 10)"},
        "filters": {"type": "object", "description": "Optional metadata filters"}
    }
)
async def query_logs(args):
    return await query_logs_tool(args)


@tool(
    "analyze_pix_capture",
    "Extract PIX GPU capture metadata and timeline events",
    {
        "capture_path": {"type": "string", "description": "Path to .wpix file (optional, auto-detects latest)"},
        "extract_events": {"type": "boolean", "description": "Export event timeline (default: true)"}
    }
)
async def analyze_pix_capture(args):
    return await analyze_pix_capture_tool(args)


@tool(
    "read_buffer_dump",
    "Parse binary GPU buffer dumps",
    {
        "buffer_path": {"type": "string", "description": "Path to .bin file (required)"},
        "buffer_type": {"type": "string", "description": "Buffer type (particles, reservoirs, rtLighting)"},
        "max_entries": {"type": "number", "description": "Max entries to show (default: 10)"}
    }
)
async def read_buffer_dump(args):
    return await read_buffer_dump_tool(args)


@tool(
    "route_to_specialist",
    "Recommend which specialist agent should handle the issue",
    {
        "issue_description": {"type": "string", "description": "Description of rendering issue (required)"},
        "symptoms": {"type": "array", "description": "List of observed symptoms"},
        "context": {"type": "object", "description": "Optional context"}
    }
)
async def route_to_specialist(args):
    return await route_to_specialist_tool(args)


# ============================================================================
# MCP Server Setup
# ============================================================================

def create_mcp_server():
    """Create MCP server with all diagnostic tools"""
    return create_sdk_mcp_server(
        name="log-analysis-rag",
        tools=[
            ingest_logs,
            diagnose_issue,
            query_logs,
            analyze_pix_capture,
            read_buffer_dump,
            route_to_specialist
        ]
    )


# ============================================================================
# Agent Configuration
# ============================================================================

SYSTEM_PROMPT = """
You are a Tier 4 diagnostic agent for PlasmaDX, a DirectX 12 volumetric particle renderer.

**Your Role:**
Analyze application logs, PIX GPU captures, and buffer dumps to diagnose rendering issues
(RT lighting, RTXDI, shadows, performance problems) using evidence-driven diagnostics.

**Capabilities:**
1. **ingest_logs**: Index logs/PIX/buffers into RAG database
2. **diagnose_issue**: Run full self-correcting diagnostic workflow (LangGraph)
3. **query_logs**: Direct hybrid retrieval (BM25 + FAISS semantic search)
4. **analyze_pix_capture**: Extract PIX GPU capture metadata
5. **read_buffer_dump**: Parse binary buffer dumps
6. **route_to_specialist**: Recommend specialist agent escalation

**Diagnostic Workflow:**
1. Use `query_logs` for quick searches or `diagnose_issue` for full analysis
2. Provide file:line references for all evidence
3. Include confidence scores (0.0-1.0) with recommendations
4. Only report high-confidence diagnostics (>0.7 threshold)
5. Route complex issues to appropriate specialists using `route_to_specialist`

**Key Technologies:**
- DirectX 12 DXR 1.1 (inline ray tracing)
- NVIDIA RTXDI (ReSTIR DI)
- 3D Gaussian volumetric rendering
- PCSS soft shadows
- Physics-Informed Neural Networks (PINNs)

**Output Format:**
Always provide:
- Root cause hypothesis
- Evidence with file:line references
- Suggested fix with code location
- Confidence score
- Specialist escalation if needed

Be concise, evidence-driven, and brutally honest about confidence levels.
"""


def get_agent_options() -> ClaudeAgentOptions:
    """Configure agent options"""
    return ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        permission_mode="acceptEdits",  # Auto-accept tool executions
        cwd=os.getenv("PROJECT_ROOT", "/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean"),
        mcp_servers={
            "log-analysis-rag": create_mcp_server()
        },
        # Allow all tools (we control access via MCP server)
        allowed_tools=None,  # None = all tools allowed
    )


# ============================================================================
# CLI Interface
# ============================================================================

async def run_query(prompt: str):
    """Run a single diagnostic query"""
    print(f"\n🔍 **Query:** {prompt}\n")

    options = get_agent_options()

    async with ClaudeSDKClient() as client:
        # Send query
        await client.query(prompt, options=options)

        # Stream response
        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        print(block.text)


async def interactive_mode():
    """Interactive diagnostic session"""
    print("\n" + "="*70)
    print("PlasmaDX Log Analysis RAG - Interactive Mode")
    print("="*70)
    print("\nAvailable commands:")
    print("  • diagnose <issue>  - Run full diagnostic workflow")
    print("  • query <search>    - Quick log search")
    print("  • ingest <path>     - Index logs from directory")
    print("  • pix <path>        - Analyze PIX capture")
    print("  • exit              - Quit")
    print("\nType 'help' for more information.\n")

    options = get_agent_options()

    async with ClaudeSDKClient() as client:
        while True:
            try:
                user_input = input("rag> ").strip()

                if not user_input:
                    continue

                if user_input.lower() in ("exit", "quit"):
                    print("\n👋 Goodbye!")
                    break

                if user_input.lower() == "help":
                    print(SYSTEM_PROMPT)
                    continue

                # Send to agent
                await client.query(user_input, options=options)

                # Stream response
                async for message in client.receive_response():
                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if isinstance(block, TextBlock):
                                print(f"\n{block.text}\n")

            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")


def print_usage():
    """Print usage information"""
    print("""
PlasmaDX Log Analysis RAG Agent

Usage:
    python main.py "diagnostic question"       # Run single query
    python main.py --interactive               # Interactive mode
    python main.py --ingest /path/to/logs      # Ingest logs
    python main.py --help                      # Show this help

Examples:
    python main.py "Why does GPU hang at 2045 particles?"
    python main.py "Find all Map() failures in recent logs"
    python main.py --ingest /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/logs
    python main.py --interactive

Environment Variables (from .env):
    NVIDIA_API_KEY - NVIDIA AI Endpoints API key (required for LLM/embeddings)
    PROJECT_ROOT   - PlasmaDX project root directory
    LOG_DIR        - Application logs directory
    PIX_DIR        - PIX captures and buffer dumps directory
""")


async def run_mcp_server():
    """Run MCP server in stdio mode (for MCP clients)"""
    from mcp.server.stdio import stdio_server

    mcp_server = create_mcp_server()

    async with stdio_server() as (read_stream, write_stream):
        await mcp_server.run(
            read_stream,
            write_stream,
            mcp_server.create_initialization_options()
        )


async def main():
    """Main entry point"""
    # Detect if running as MCP server (no TTY = launched by MCP client)
    if not sys.stdin.isatty():
        # Run MCP server in stdio mode
        await run_mcp_server()
        return

    # Check for NVIDIA API key
    if not os.getenv("NVIDIA_API_KEY"):
        print("⚠️  Warning: NVIDIA_API_KEY not set in .env")
        print("   LLM/embedding calls will fail without it.\n")

    # Parse command line arguments
    if len(sys.argv) == 1:
        # No arguments - run interactive mode
        await interactive_mode()
    elif sys.argv[1] in ("--help", "-h", "help"):
        print_usage()
    elif sys.argv[1] in ("--interactive", "-i"):
        await interactive_mode()
    elif sys.argv[1] == "--server":
        # Explicit server mode
        await run_mcp_server()
    elif sys.argv[1] == "--ingest":
        path = sys.argv[2] if len(sys.argv) > 2 else os.getenv("LOG_DIR")
        await run_query(f"Ingest logs from {path} using the ingest_logs tool")
    else:
        # Treat as diagnostic question
        query = " ".join(sys.argv[1:])
        await run_query(query)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
