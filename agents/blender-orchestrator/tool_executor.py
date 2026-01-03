#!/usr/bin/env python3
"""
Tool Execution Verifier for Blender VFX Orchestrator

Provides verification layer for MCP tool calls:
- Pre/post execution hooks
- Result schema validation
- Execution logging with latency
- Mismatch detection between claimed and executed tools

This addresses Blocker 2: Orchestrator Execution Detection
"""

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger("blender-orchestrator.tool_executor")


class ToolStatus(Enum):
    """Status of a tool execution."""
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    VALIDATION_ERROR = "validation_error"


@dataclass
class ToolExecutionRecord:
    """Record of a single tool execution."""
    timestamp: str
    tool: str
    params: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    status: ToolStatus = ToolStatus.SUCCESS
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "tool": self.tool,
            "params": self.params,
            "result": self.result,
            "error": self.error,
            "status": self.status.value,
            "latency_ms": self.latency_ms,
        }


@dataclass
class ToolSchema:
    """Schema definition for validating tool results."""
    required_fields: List[str] = field(default_factory=list)
    optional_fields: List[str] = field(default_factory=list)
    result_type: str = "dict"  # dict, list, str, bool


# Known tool result schemas for validation
TOOL_SCHEMAS: Dict[str, ToolSchema] = {
    # script-generator tools
    "generate_script": ToolSchema(
        required_fields=["script_path", "content"],
        optional_fields=["technique", "parameters"]
    ),
    "modify_script": ToolSchema(
        required_fields=["script_path", "modifications_applied"],
        optional_fields=["warnings"]
    ),
    "validate_parameters": ToolSchema(
        required_fields=["valid", "validated_params"],
        optional_fields=["warnings", "clamped"]
    ),
    "list_techniques": ToolSchema(
        required_fields=["techniques"],
        result_type="dict"
    ),

    # blender-executor tools
    "execute_blender_script": ToolSchema(
        required_fields=["success", "run_dir"],
        optional_fields=["stdout", "stderr", "vdb_files", "render_files"]
    ),
    "parse_blender_errors": ToolSchema(
        required_fields=["errors"],
        optional_fields=["suggested_fixes"]
    ),

    # asset-evaluator tools
    "evaluate_vfx_quality": ToolSchema(
        required_fields=["composite_score", "passed"],
        optional_fields=["dimension_scores", "issues", "critical_issues"]
    ),
    "evaluate_ground_truth": ToolSchema(
        required_fields=["overall_score", "passed"],
        optional_fields=["distribution_comparison", "recommendations"]
    ),
    "analyze_temporal_quality": ToolSchema(
        required_fields=["temporal_consistency"],
        optional_fields=["flicker_risk", "recommendations"]
    ),

    # experiment-tracker tools
    "get_warnings_before_change": ToolSchema(
        required_fields=["warnings"],
        optional_fields=["relevant_knowledge"]
    ),
    "suggest_experiments": ToolSchema(
        required_fields=["suggestions"],
        optional_fields=["rationale"]
    ),
    "record_experiment_result": ToolSchema(
        required_fields=["recorded"],
        optional_fields=["experiment_id", "knowledge_updated"]
    ),

    # iteration-controller tools
    "save_iteration_state": ToolSchema(
        required_fields=["saved"],
        optional_fields=["state_file"]
    ),
    "load_iteration_state": ToolSchema(
        required_fields=["loaded"],
        optional_fields=["state"]
    ),
    "diagnose_vfx_issues": ToolSchema(
        required_fields=["diagnosed_issues", "suggested_fixes"],
        optional_fields=["priority_order"]
    ),
}


class ToolExecutionVerifier:
    """
    Verifies MCP tool execution with pre/post hooks and result validation.

    Usage:
        verifier = ToolExecutionVerifier()

        # Wrap tool calls
        result = await verifier.wrap_tool_call(
            "generate_script",
            {"effect_type": "pyro", "description": "..."},
            executor_func
        )

        # Verify all claimed tools were executed
        if not verifier.verify_execution(["generate_script", "execute_blender_script"]):
            logger.error("Tool execution mismatch detected")
    """

    def __init__(self, log_dir: Optional[Path] = None):
        """
        Initialize the verifier.

        Args:
            log_dir: Directory for persisting execution logs (optional)
        """
        self.call_log: List[ToolExecutionRecord] = []
        self.log_dir = log_dir
        self._session_id: Optional[str] = None

        # Pre/post execution hooks
        self._pre_hooks: List[Callable] = []
        self._post_hooks: List[Callable] = []

    def set_session_id(self, session_id: str) -> None:
        """Set current session ID for log organization."""
        self._session_id = session_id

    def add_pre_hook(self, hook: Callable[[str, Dict], None]) -> None:
        """Add a pre-execution hook (called before tool execution)."""
        self._pre_hooks.append(hook)

    def add_post_hook(self, hook: Callable[[str, Dict, Any], None]) -> None:
        """Add a post-execution hook (called after tool execution)."""
        self._post_hooks.append(hook)

    async def wrap_tool_call(
        self,
        tool_name: str,
        params: Dict[str, Any],
        executor: Callable,
        timeout_seconds: float = 300.0,
    ) -> Dict[str, Any]:
        """
        Wrap a tool call with verification, logging, and hooks.

        Args:
            tool_name: Name of the MCP tool (without server prefix)
            params: Parameters to pass to the tool
            executor: Async function that executes the actual tool call
            timeout_seconds: Maximum execution time

        Returns:
            Tool execution result

        Raises:
            Exception: Re-raises any execution errors after logging
        """
        start_time = time.time()
        timestamp = datetime.now().isoformat()

        # Run pre-execution hooks
        for hook in self._pre_hooks:
            try:
                hook(tool_name, params)
            except Exception as e:
                logger.warning(f"Pre-hook failed for {tool_name}: {e}")

        try:
            # Execute the tool
            result = await executor()
            latency_ms = (time.time() - start_time) * 1000

            # Parse result if it's a JSON string
            parsed_result = result
            if isinstance(result, str):
                try:
                    parsed_result = json.loads(result)
                except json.JSONDecodeError:
                    parsed_result = {"raw": result}

            # Validate result schema
            validation_status = self._validate_result(tool_name, parsed_result)

            # Create execution record
            record = ToolExecutionRecord(
                timestamp=timestamp,
                tool=tool_name,
                params=params,
                result=parsed_result,
                status=validation_status,
                latency_ms=latency_ms,
            )
            self.call_log.append(record)

            # Log execution
            logger.info(
                f"Tool executed: {tool_name} "
                f"(latency: {latency_ms:.1f}ms, status: {validation_status.value})"
            )

            # Run post-execution hooks
            for hook in self._post_hooks:
                try:
                    hook(tool_name, params, parsed_result)
                except Exception as e:
                    logger.warning(f"Post-hook failed for {tool_name}: {e}")

            return parsed_result

        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000

            # Create failure record
            record = ToolExecutionRecord(
                timestamp=timestamp,
                tool=tool_name,
                params=params,
                error=str(e),
                status=ToolStatus.FAILED,
                latency_ms=latency_ms,
            )
            self.call_log.append(record)

            logger.error(f"Tool failed: {tool_name} - {e}")
            raise

    def _validate_result(
        self,
        tool_name: str,
        result: Any
    ) -> ToolStatus:
        """
        Validate tool result against expected schema.

        Args:
            tool_name: Name of the tool
            result: Result to validate

        Returns:
            Validation status
        """
        # Check for error in result
        if isinstance(result, dict) and result.get("error"):
            return ToolStatus.FAILED

        # Get schema for this tool
        schema = TOOL_SCHEMAS.get(tool_name)
        if not schema:
            # No schema defined, accept any non-error result
            return ToolStatus.SUCCESS

        # Validate required fields
        if isinstance(result, dict):
            for field in schema.required_fields:
                if field not in result:
                    logger.warning(
                        f"Validation warning: {tool_name} missing required field '{field}'"
                    )
                    return ToolStatus.VALIDATION_ERROR

        return ToolStatus.SUCCESS

    def verify_execution(self, claimed_tools: List[str]) -> bool:
        """
        Verify that all claimed tools were actually executed.

        Args:
            claimed_tools: List of tool names that should have been executed

        Returns:
            True if all claimed tools were executed successfully
        """
        executed = {
            log.tool for log in self.call_log
            if log.status == ToolStatus.SUCCESS
        }
        missing = set(claimed_tools) - executed

        if missing:
            logger.error(f"Tools claimed but not executed: {missing}")
            return False

        return True

    def get_execution_summary(self) -> Dict[str, Any]:
        """
        Get summary of all tool executions.

        Returns:
            Summary including counts, latencies, and status breakdown
        """
        if not self.call_log:
            return {
                "total_calls": 0,
                "successful": 0,
                "failed": 0,
                "avg_latency_ms": 0,
            }

        successful = sum(1 for log in self.call_log if log.status == ToolStatus.SUCCESS)
        failed = sum(1 for log in self.call_log if log.status == ToolStatus.FAILED)
        total_latency = sum(log.latency_ms for log in self.call_log)

        return {
            "total_calls": len(self.call_log),
            "successful": successful,
            "failed": failed,
            "validation_errors": sum(
                1 for log in self.call_log
                if log.status == ToolStatus.VALIDATION_ERROR
            ),
            "avg_latency_ms": total_latency / len(self.call_log),
            "max_latency_ms": max(log.latency_ms for log in self.call_log),
            "tools_called": list({log.tool for log in self.call_log}),
        }

    def get_recent_calls(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the most recent tool calls."""
        return [log.to_dict() for log in self.call_log[-limit:]]

    def clear_log(self) -> None:
        """Clear the execution log (e.g., between sessions)."""
        self.call_log = []

    def save_log(self, filepath: Optional[Path] = None) -> Path:
        """
        Save execution log to file.

        Args:
            filepath: Optional specific path (uses default if not provided)

        Returns:
            Path where log was saved
        """
        if filepath is None:
            if self.log_dir is None:
                self.log_dir = Path("build/orchestrator_state/tool_logs")
            self.log_dir.mkdir(parents=True, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_part = f"_{self._session_id}" if self._session_id else ""
            filepath = self.log_dir / f"tool_execution{session_part}_{timestamp}.json"

        log_data = {
            "session_id": self._session_id,
            "generated_at": datetime.now().isoformat(),
            "summary": self.get_execution_summary(),
            "calls": [log.to_dict() for log in self.call_log],
        }

        with open(filepath, "w") as f:
            json.dump(log_data, f, indent=2)

        logger.info(f"Tool execution log saved to {filepath}")
        return filepath


# Singleton instance for global access
_verifier_instance: Optional[ToolExecutionVerifier] = None


def get_verifier(log_dir: Optional[Path] = None) -> ToolExecutionVerifier:
    """Get or create the global verifier instance."""
    global _verifier_instance
    if _verifier_instance is None:
        _verifier_instance = ToolExecutionVerifier(log_dir=log_dir)
    return _verifier_instance


if __name__ == "__main__":
    # Quick test
    import asyncio

    async def test_verifier():
        verifier = ToolExecutionVerifier()

        # Mock executor
        async def mock_executor():
            return {"script_path": "/test/path.py", "content": "print('hello')"}

        # Test wrap_tool_call
        result = await verifier.wrap_tool_call(
            "generate_script",
            {"effect_type": "pyro"},
            mock_executor
        )

        print(f"Result: {result}")
        print(f"Summary: {verifier.get_execution_summary()}")
        print(f"Verified: {verifier.verify_execution(['generate_script'])}")

    asyncio.run(test_verifier())
