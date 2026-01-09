"""
MCP Connection Pool - Persistent connections to MCP servers.

Provides singleton connection pools for each MCP server type,
eliminating the 15-30s startup time that would occur if we
spawned a new subprocess for each query.

Adapted from blender-librarian/librarian_agents/doc_expert.py
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Dict, Optional

from agents.mcp import MCPServerStdio


class MCPConnectionPool:
    """
    Singleton connection pool for an MCP server.

    Instead of spawning a new MCP server subprocess for each query,
    this pool maintains a persistent connection that can be reused.

    Benefits:
    - Eliminates 15-30s startup time for each query
    - Keeps any warm state (embeddings, caches) in the server process
    - Reduces resource usage from repeated subprocess spawning

    Usage:
        pool = MCPConnectionPool(
            name="script-generator",
            server_dir=Path("/path/to/agents/script-generator"),
            server_script="server.py"
        )
        server = await pool.get_server()
        # Use server in agents...
        await pool.close()
    """

    def __init__(
        self,
        name: str,
        server_dir: Path,
        server_script: str = "server.py",
        venv_name: str = "venv",
        timeout_seconds: float = 120.0,
        env_vars: Optional[Dict[str, str]] = None
    ):
        """
        Initialize a connection pool for an MCP server.

        Args:
            name: Unique name for this pool (e.g., "script-generator")
            server_dir: Path to the MCP server directory
            server_script: Name of the server script (default: "server.py")
            venv_name: Name of the virtual environment directory
            timeout_seconds: Client session timeout (default: 120s)
            env_vars: Additional environment variables to pass
        """
        self.name = name
        self.server_dir = server_dir
        self.server_script = server_script
        self.venv_name = venv_name
        self.timeout_seconds = timeout_seconds
        self.env_vars = env_vars or {}

        self._lock: Optional[asyncio.Lock] = None
        self._mcp_server: Optional[MCPServerStdio] = None
        self._connected = False

    async def get_server(self) -> MCPServerStdio:
        """
        Get the pooled MCP server connection, creating it if necessary.

        Returns:
            Connected MCPServerStdio instance

        Thread-safe via asyncio.Lock.
        """
        if self._lock is None:
            self._lock = asyncio.Lock()

        async with self._lock:
            if self._mcp_server is not None and self._connected:
                return self._mcp_server

            # Build paths
            venv_python = self.server_dir / self.venv_name / "bin" / "python"
            server_script_path = self.server_dir / self.server_script

            # Verify paths exist
            if not venv_python.exists():
                raise FileNotFoundError(
                    f"Virtual environment not found: {venv_python}\n"
                    f"Run: cd {self.server_dir} && python3 -m venv venv && ./venv/bin/pip install -r requirements.txt"
                )
            if not server_script_path.exists():
                raise FileNotFoundError(f"Server script not found: {server_script_path}")

            # Prepare environment
            env = os.environ.copy()
            env.update(self.env_vars)

            # Create MCP server connection
            self._mcp_server = MCPServerStdio(
                name=f"{self.name}-pooled",
                params={
                    "command": str(venv_python),
                    "args": [str(server_script_path)],
                    "cwd": str(self.server_dir),
                    "env": env
                },
                cache_tools_list=True,
                client_session_timeout_seconds=self.timeout_seconds
            )

            await self._mcp_server.connect()
            self._connected = True
            return self._mcp_server

    async def close(self) -> None:
        """Close the pooled connection."""
        if self._lock is None:
            self._lock = asyncio.Lock()

        async with self._lock:
            if self._mcp_server:
                try:
                    await self._mcp_server.cleanup()
                except Exception:
                    pass  # Ignore cleanup errors
                self._mcp_server = None
                self._connected = False

    @property
    def is_connected(self) -> bool:
        """Check if the pool has an active connection."""
        return self._connected and self._mcp_server is not None


# =============================================================================
# GLOBAL POOL REGISTRY
# =============================================================================

class PoolRegistry:
    """
    Global registry of MCP connection pools.

    Manages pools for all MCP servers used by the orchestrator:
    - script-generator
    - blender-executor
    - asset-evaluator
    - experiment-tracker
    - blender-manual (for DocsExpert)
    """

    _instance: Optional["PoolRegistry"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._pools: Dict[str, MCPConnectionPool] = {}
        self._project_root = Path(__file__).resolve().parents[3]
        self._agents_dir = self._project_root / "agents"
        self._initialized = True

    def get_pool(self, server_name: str) -> MCPConnectionPool:
        """
        Get or create a connection pool for an MCP server.

        Args:
            server_name: Name of the MCP server (e.g., "script-generator")

        Returns:
            MCPConnectionPool for the specified server
        """
        if server_name not in self._pools:
            server_dir = self._agents_dir / server_name

            # Server-specific configuration
            env_vars = {}
            if server_name == "blender-manual":
                # Pre-warm embeddings for semantic search
                env_vars["BLENDER_MCP_PREWARM_EMBEDDINGS"] = "1"

            self._pools[server_name] = MCPConnectionPool(
                name=server_name,
                server_dir=server_dir,
                env_vars=env_vars
            )

        return self._pools[server_name]

    async def get_server(self, server_name: str) -> MCPServerStdio:
        """
        Get a connected MCP server instance.

        Convenience method that gets the pool and returns the connected server.

        Args:
            server_name: Name of the MCP server

        Returns:
            Connected MCPServerStdio instance
        """
        pool = self.get_pool(server_name)
        return await pool.get_server()

    async def close_all(self) -> None:
        """Close all pooled connections."""
        for pool in self._pools.values():
            await pool.close()

    @property
    def connected_servers(self) -> list[str]:
        """List of currently connected server names."""
        return [name for name, pool in self._pools.items() if pool.is_connected]


# Module-level singleton accessor
_pool_registry: Optional[PoolRegistry] = None


def get_connection_pool() -> PoolRegistry:
    """Get the global MCP connection pool registry singleton."""
    global _pool_registry
    if _pool_registry is None:
        _pool_registry = PoolRegistry()
    return _pool_registry


# Convenience function for direct server access
async def get_mcp_server(server_name: str) -> MCPServerStdio:
    """
    Get a connected MCP server by name.

    Args:
        server_name: Name of the MCP server (e.g., "script-generator")

    Returns:
        Connected MCPServerStdio instance

    Example:
        server = await get_mcp_server("script-generator")
    """
    return await get_connection_pool().get_server(server_name)
