#!/usr/bin/env python3
"""
MCP Server Health Checks for Blender VFX Orchestrator

Provides health monitoring for the 6 MCP servers used by the orchestrator:
1. script-generator
2. blender-executor
3. asset-evaluator
4. experiment-tracker
5. iteration-controller
6. blender-manual

Health checks can be run:
- At session start (verify all servers available)
- Before critical operations (verify specific server)
- Periodically during long sessions (detect degradation)
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("blender-orchestrator.health_check")


class HealthState(Enum):
    """Health state of an MCP server."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"  # Slow responses but functional
    UNHEALTHY = "unhealthy"  # Not responding
    UNKNOWN = "unknown"  # Not yet checked


@dataclass
class HealthStatus:
    """Health status for a single MCP server."""
    server_name: str
    state: HealthState = HealthState.UNKNOWN
    last_check: Optional[str] = None
    latency_ms: Optional[float] = None
    error: Optional[str] = None
    consecutive_failures: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "server": self.server_name,
            "state": self.state.value,
            "last_check": self.last_check,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "consecutive_failures": self.consecutive_failures,
        }


@dataclass
class ServerConfig:
    """Configuration for an MCP server health check."""
    name: str
    ping_tool: str  # Tool to call for health check
    ping_params: Dict[str, Any] = field(default_factory=dict)
    timeout_seconds: float = 10.0
    degraded_threshold_ms: float = 5000.0  # Above this = degraded
    critical: bool = True  # If True, unhealthy blocks workflow


# Server configurations
MCP_SERVERS: Dict[str, ServerConfig] = {
    "script-generator": ServerConfig(
        name="script-generator",
        ping_tool="list_techniques",  # Lightweight tool
        ping_params={"effect_type": "pyro"},
        timeout_seconds=10.0,
        critical=True,
    ),
    "blender-executor": ServerConfig(
        name="blender-executor",
        ping_tool="list_available_scripts",  # Just lists files
        ping_params={},
        timeout_seconds=10.0,
        critical=True,
    ),
    "asset-evaluator": ServerConfig(
        name="asset-evaluator",
        ping_tool="list_recent_renders",  # Lists files
        ping_params={"limit": 1},
        timeout_seconds=10.0,
        critical=True,
    ),
    "experiment-tracker": ServerConfig(
        name="experiment-tracker",
        ping_tool="get_experiment_statistics",  # Reads local state
        ping_params={},
        timeout_seconds=5.0,
        critical=False,  # Workflow can proceed without experiment tracking
    ),
    "iteration-controller": ServerConfig(
        name="iteration-controller",
        ping_tool="list_sessions",  # Reads local state
        ping_params={},
        timeout_seconds=5.0,
        critical=True,
    ),
    "blender-manual": ServerConfig(
        name="blender-manual",
        ping_tool="browse_hierarchy",  # Local documentation
        ping_params={},
        timeout_seconds=5.0,
        critical=False,  # Research optional
    ),
}


class HealthChecker:
    """
    Performs health checks on MCP servers.

    Usage:
        checker = HealthChecker(tool_executor)

        # Check all servers at session start
        all_healthy = await checker.check_all_servers()

        # Check specific server before critical operation
        status = await checker.check_server("blender-executor")

        # Get overall health
        if checker.all_critical_healthy():
            proceed_with_workflow()
    """

    def __init__(
        self,
        tool_executor: Optional[Callable] = None,
        max_consecutive_failures: int = 3,
    ):
        """
        Initialize health checker.

        Args:
            tool_executor: Async function to execute MCP tool calls
            max_consecutive_failures: Failures before marking unhealthy
        """
        self._tool_executor = tool_executor
        self._max_failures = max_consecutive_failures

        # Server health states
        self._server_status: Dict[str, HealthStatus] = {
            name: HealthStatus(server_name=name)
            for name in MCP_SERVERS
        }

        # Health history for trend analysis
        self._health_history: List[Dict[str, Any]] = []
        self._max_history = 100

    def set_tool_executor(self, executor: Callable) -> None:
        """Set the tool executor function."""
        self._tool_executor = executor

    async def check_server(self, server_name: str) -> HealthStatus:
        """
        Check health of a specific MCP server.

        Args:
            server_name: Name of server to check

        Returns:
            HealthStatus for the server
        """
        if server_name not in MCP_SERVERS:
            logger.warning(f"Unknown server: {server_name}")
            return HealthStatus(
                server_name=server_name,
                state=HealthState.UNKNOWN,
                error=f"Unknown server: {server_name}"
            )

        config = MCP_SERVERS[server_name]
        status = self._server_status[server_name]
        timestamp = datetime.now().isoformat()

        try:
            start_time = time.time()

            # Execute health check tool
            if self._tool_executor:
                result = await asyncio.wait_for(
                    self._tool_executor(config.ping_tool, config.ping_params),
                    timeout=config.timeout_seconds
                )
            else:
                # Mock mode for testing
                await asyncio.sleep(0.1)
                result = {"mock": True}

            latency_ms = (time.time() - start_time) * 1000

            # Check for degraded performance
            if latency_ms > config.degraded_threshold_ms:
                status.state = HealthState.DEGRADED
                logger.warning(
                    f"Server {server_name} degraded: {latency_ms:.0f}ms latency"
                )
            else:
                status.state = HealthState.HEALTHY

            status.last_check = timestamp
            status.latency_ms = latency_ms
            status.error = None
            status.consecutive_failures = 0

            logger.debug(f"Health check passed: {server_name} ({latency_ms:.0f}ms)")

        except asyncio.TimeoutError:
            status.state = HealthState.UNHEALTHY
            status.last_check = timestamp
            status.error = f"Timeout after {config.timeout_seconds}s"
            status.consecutive_failures += 1
            logger.error(f"Health check timeout: {server_name}")

        except Exception as e:
            status.state = HealthState.UNHEALTHY
            status.last_check = timestamp
            status.error = str(e)
            status.consecutive_failures += 1
            logger.error(f"Health check failed: {server_name} - {e}")

        # Record history
        self._record_health(server_name, status)

        return status

    async def check_all_servers(self) -> Dict[str, HealthStatus]:
        """
        Check health of all MCP servers.

        Returns:
            Dictionary of server name -> HealthStatus
        """
        logger.info("Running health checks on all MCP servers...")

        # Run checks in parallel
        tasks = [
            self.check_server(name)
            for name in MCP_SERVERS
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

        # Log summary
        healthy = sum(
            1 for s in self._server_status.values()
            if s.state == HealthState.HEALTHY
        )
        degraded = sum(
            1 for s in self._server_status.values()
            if s.state == HealthState.DEGRADED
        )
        unhealthy = sum(
            1 for s in self._server_status.values()
            if s.state == HealthState.UNHEALTHY
        )

        logger.info(
            f"Health check complete: {healthy} healthy, "
            f"{degraded} degraded, {unhealthy} unhealthy"
        )

        return dict(self._server_status)

    async def check_critical_servers(self) -> Tuple[bool, List[str]]:
        """
        Check only critical servers (those that block workflow).

        Returns:
            Tuple of (all_healthy: bool, unhealthy_servers: List[str])
        """
        critical_servers = [
            name for name, config in MCP_SERVERS.items()
            if config.critical
        ]

        unhealthy = []
        for name in critical_servers:
            status = await self.check_server(name)
            if status.state == HealthState.UNHEALTHY:
                unhealthy.append(name)

        return len(unhealthy) == 0, unhealthy

    def all_critical_healthy(self) -> bool:
        """Check if all critical servers are healthy (without new check)."""
        for name, config in MCP_SERVERS.items():
            if config.critical:
                status = self._server_status.get(name)
                if status and status.state == HealthState.UNHEALTHY:
                    return False
        return True

    def get_unhealthy_servers(self) -> List[str]:
        """Get list of currently unhealthy servers."""
        return [
            name for name, status in self._server_status.items()
            if status.state == HealthState.UNHEALTHY
        ]

    def get_degraded_servers(self) -> List[str]:
        """Get list of currently degraded servers."""
        return [
            name for name, status in self._server_status.items()
            if status.state == HealthState.DEGRADED
        ]

    def get_status_summary(self) -> Dict[str, Any]:
        """Get summary of all server health statuses."""
        return {
            "checked_at": datetime.now().isoformat(),
            "servers": {
                name: status.to_dict()
                for name, status in self._server_status.items()
            },
            "summary": {
                "total": len(self._server_status),
                "healthy": sum(
                    1 for s in self._server_status.values()
                    if s.state == HealthState.HEALTHY
                ),
                "degraded": sum(
                    1 for s in self._server_status.values()
                    if s.state == HealthState.DEGRADED
                ),
                "unhealthy": sum(
                    1 for s in self._server_status.values()
                    if s.state == HealthState.UNHEALTHY
                ),
                "unknown": sum(
                    1 for s in self._server_status.values()
                    if s.state == HealthState.UNKNOWN
                ),
                "all_critical_healthy": self.all_critical_healthy(),
            }
        }

    def _record_health(self, server_name: str, status: HealthStatus) -> None:
        """Record health check result to history."""
        record = {
            "timestamp": status.last_check,
            "server": server_name,
            "state": status.state.value,
            "latency_ms": status.latency_ms,
        }
        self._health_history.append(record)

        # Trim history
        if len(self._health_history) > self._max_history:
            self._health_history = self._health_history[-self._max_history:]

    def get_health_trends(self, server_name: str, limit: int = 10) -> List[Dict]:
        """Get recent health history for a server."""
        return [
            h for h in self._health_history[-limit:]
            if h["server"] == server_name
        ]

    def print_status(self) -> None:
        """Print human-readable status to console."""
        summary = self.get_status_summary()

        print("\n" + "=" * 50)
        print("MCP SERVER HEALTH STATUS")
        print("=" * 50)

        for name, status_dict in summary["servers"].items():
            config = MCP_SERVERS.get(name)
            critical_mark = " [CRITICAL]" if config and config.critical else ""
            state = status_dict["state"].upper()

            # Color indicators
            if state == "HEALTHY":
                indicator = "✅"
            elif state == "DEGRADED":
                indicator = "⚠️"
            elif state == "UNHEALTHY":
                indicator = "❌"
            else:
                indicator = "❓"

            latency = status_dict.get("latency_ms")
            latency_str = f" ({latency:.0f}ms)" if latency else ""

            print(f"  {indicator} {name}{critical_mark}: {state}{latency_str}")

            if status_dict.get("error"):
                print(f"      Error: {status_dict['error']}")

        print("-" * 50)
        s = summary["summary"]
        print(f"Total: {s['healthy']}/{s['total']} healthy")
        if not s["all_critical_healthy"]:
            print("⚠️  CRITICAL SERVERS UNHEALTHY - Workflow blocked")
        print("=" * 50 + "\n")


# Singleton instance
_checker_instance: Optional[HealthChecker] = None


def get_health_checker() -> HealthChecker:
    """Get or create the global health checker instance."""
    global _checker_instance
    if _checker_instance is None:
        _checker_instance = HealthChecker()
    return _checker_instance


if __name__ == "__main__":
    # Quick test in mock mode
    async def test_health_checker():
        checker = HealthChecker()

        # Run mock health checks
        await checker.check_all_servers()

        # Print status
        checker.print_status()

        # Check critical servers
        all_healthy, unhealthy = await checker.check_critical_servers()
        print(f"All critical healthy: {all_healthy}")
        if unhealthy:
            print(f"Unhealthy critical servers: {unhealthy}")

    asyncio.run(test_health_checker())
