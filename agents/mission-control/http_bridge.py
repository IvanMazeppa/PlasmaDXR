#!/usr/bin/env python3
"""
HTTP Bridge for Mission-Control Autonomous Agent

Allows Claude Code to interact with the autonomous agent via HTTP.

Architecture:
    Claude Code → MCP Bridge → HTTP → Autonomous Agent → Specialist MCP Tools

Usage:
    uvicorn http_bridge:app --host 0.0.0.0 --port 8001

    Then from Claude Code, you can call:
    curl -X POST http://localhost:8001/query -d '{"prompt": "analyze probe grid"}'
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Dict

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from autonomous_agent import MissionControlAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("http-bridge")

# Global agent instance
agent: MissionControlAgent = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage agent lifecycle (startup/shutdown).
    """
    global agent

    # Startup
    logger.info("Starting Mission-Control autonomous agent...")
    agent = MissionControlAgent()
    await agent.start()
    logger.info("✅ HTTP bridge ready on port 8001")

    yield

    # Shutdown
    logger.info("Shutting down Mission-Control autonomous agent...")
    await agent.stop()


app = FastAPI(
    title="Mission-Control Autonomous Agent",
    description="HTTP bridge for autonomous strategic orchestrator",
    version="1.0.0",
    lifespan=lifespan
)


class QueryRequest(BaseModel):
    """Query request model"""
    prompt: str
    stream: bool = False


class QueryResponse(BaseModel):
    """Query response model"""
    response: str
    status: str = "success"


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "agent": "mission-control",
        "mode": "autonomous"
    }


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Send query to autonomous agent.

    The agent will autonomously:
    1. Analyze the query
    2. Call appropriate specialist MCP tools
    3. Synthesize findings
    4. Make recommendations

    Args:
        request: Query request with prompt

    Returns:
        Autonomous agent response
    """
    try:
        logger.info(f"Received query: {request.prompt[:100]}...")

        # Non-streaming response
        response = await agent.query(request.prompt)

        return QueryResponse(
            response=response,
            status="success"
        )

    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query/stream")
async def query_stream(request: QueryRequest):
    """
    Send query to autonomous agent with streaming response.

    Streams agent reasoning and tool calls in real-time.

    Args:
        request: Query request with prompt

    Returns:
        Server-Sent Events stream
    """
    async def event_generator():
        try:
            logger.info(f"Streaming query: {request.prompt[:100]}...")

            await agent.client.query(request.prompt)

            # Stream autonomous response
            async for message in agent.client.receive_response():
                yield f"data: {message}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Streaming failed: {e}", exc_info=True)
            yield f"data: ERROR: {str(e)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )


@app.get("/status")
async def get_status():
    """
    Get agent status and available capabilities.

    Returns:
        Agent status, MCP servers, tool counts
    """
    if not agent or not agent.options:
        return {
            "status": "not_ready",
            "error": "Agent not initialized"
        }

    return {
        "status": "ready",
        "agent": "mission-control",
        "mode": "autonomous",
        "mcp_servers": list(agent.options.mcp_servers.keys()),
        "tools_available": len(agent.options.allowed_tools),
        "project_root": str(agent.options.cwd)
    }


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Mission-Control HTTP bridge...")
    logger.info("Access at: http://localhost:8001")
    logger.info("Docs at: http://localhost:8001/docs")

    uvicorn.run(
        "http_bridge:app",
        host="0.0.0.0",
        port=8001,
        log_level="info",
        reload=False  # Don't reload in production
    )
