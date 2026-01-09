"""
Function tool wrappers for blender-executor MCP server.

Exposes blender-executor capabilities to OpenAI Agents SDK agents:
- execute_blender_script: Run Blender scripts with CLI runner
- parse_blender_errors: Structure error output for diagnosis
- list_run_outputs: Enumerate files from a run
- get_latest_run: Get most recent execution info

These wrappers translate MCP tool calls into @function_tool decorated
functions that agents can call directly.
"""

from __future__ import annotations

import json
from typing import Optional

from agents import function_tool

from utils.mcp_connection_pool import get_mcp_server


# =============================================================================
# SCRIPT EXECUTION
# =============================================================================

@function_tool
async def execute_blender_script(
    script_path: str,
    script_args_json: str = "{}",
    output_dir: Optional[str] = None,
    blend_file: Optional[str] = None,
    timeout_seconds: int = 600,
    use_ui: bool = False
) -> str:
    """
    Execute a Blender Python script via CLI with full output capture.

    Runs scripts through the CLI runner which handles:
    - Background mode execution (no UI by default)
    - Argument passing to scripts
    - Output directory management
    - Automatic log capture

    Args:
        script_path: Path to .py script (absolute or relative to project root)
        script_args_json: JSON string of arguments to pass
            Example: '{"bake": "1", "resolution": "96", "frame_end": "50"}'
        output_dir: Override output directory for VDB/renders
        blend_file: Optional .blend file to open before running script
        timeout_seconds: Max execution time (default 600 = 10 minutes)
        use_ui: Run with Blender UI visible (default False = background)

    Returns:
        JSON with ExecutionResult containing:
        - success: True if exit code was 0
        - exit_code: Process exit code
        - stdout: Standard output
        - stderr: Standard error
        - duration_seconds: Execution time
        - run_dir: Directory containing outputs
        - vdb_files: List of generated VDB files
        - render_files: List of rendered images
        - blend_files: List of .blend files
        - errors: Parsed error objects

    Example:
        execute_blender_script(
            script_path="assets/blender_scripts/generated/explosion_v1.py",
            script_args_json='{"bake": "1", "resolution": "96", "frame_end": "50"}',
            output_dir="build/vdb_output/explosion_v1"
        )
    """
    server = await get_mcp_server("blender-executor")

    # Parse JSON string to dict for MCP call
    script_args = json.loads(script_args_json) if script_args_json else None

    result = await server.call_tool(
        "execute_blender_script",
        {
            "script_path": script_path,
            "script_args": script_args,
            "output_dir": output_dir,
            "blend_file": blend_file,
            "timeout_seconds": timeout_seconds,
            "use_ui": use_ui,
        }
    )

    if hasattr(result, 'content') and result.content:
        return result.content[0].text if result.content else "{}"
    return str(result)


# =============================================================================
# ERROR PARSING
# =============================================================================

@function_tool
async def parse_blender_errors(
    stderr: str,
    stdout: str = ""
) -> str:
    """
    Parse Blender/Python errors into structured format with suggested fixes.

    Analyzes error output to identify:
    - Python exceptions (AttributeError, RuntimeError, etc.)
    - Blender-specific errors (context, API changes)
    - Module import errors
    - Key/attribute access errors

    Each error includes suggested fixes for common problems like
    Blender API changes between versions.

    Args:
        stderr: Standard error output from Blender execution
        stdout: Standard output (optional, sometimes errors appear here)

    Returns:
        JSON array of BlenderError objects with:
        - error_type: PYTHON, BLENDER, CONTEXT, API, KEY, MODULE
        - message: Error message
        - file: Source file (if available)
        - line: Line number (if available)
        - traceback: Full traceback (if available)
        - suggested_fix: Recommended fix for common errors

    Example errors detected:
        - AttributeError for removed Blender 5.0 API properties
        - RuntimeError for incorrect context (poll failed)
        - ModuleNotFoundError for missing dependencies
    """
    server = await get_mcp_server("blender-executor")

    result = await server.call_tool(
        "parse_blender_errors",
        {
            "stderr": stderr,
            "stdout": stdout,
        }
    )

    if hasattr(result, 'content') and result.content:
        return result.content[0].text if result.content else "[]"
    return str(result)


# =============================================================================
# OUTPUT MANAGEMENT
# =============================================================================

@function_tool
async def list_run_outputs(
    run_dir: Optional[str] = None
) -> str:
    """
    List all output files from a Blender CLI run.

    Scans a run directory for all generated artifacts:
    - VDB volumetric files
    - Rendered images (PNG, JPG, EXR)
    - Blend files (scene state)
    - Log files (stdout, Blender internal)

    Args:
        run_dir: Path to run directory. If not specified, uses latest run.

    Returns:
        JSON with RunOutputs containing:
        - run_dir: Full path to run directory
        - timestamp: Extracted from directory name (YYYYMMDD_HHMMSS)
        - script_name: Script that was executed
        - stdout_log: Path to stdout/stderr log
        - blender_log: Path to Blender's internal log
        - vdb_files: List of .vdb files found
        - render_files: List of image files
        - blend_files: List of .blend files
        - other_files: Other output files

    Example:
        list_run_outputs("build/blender_logs/20250107_143022_explosion_v1")
    """
    server = await get_mcp_server("blender-executor")

    result = await server.call_tool(
        "list_run_outputs",
        {
            "run_dir": run_dir,
        }
    )

    if hasattr(result, 'content') and result.content:
        return result.content[0].text if result.content else "{}"
    return str(result)


@function_tool
async def get_latest_run() -> str:
    """
    Get information about the most recent Blender execution.

    Convenience function that finds the latest run directory and
    returns its full output listing. Useful for checking results
    immediately after an execution.

    Returns:
        JSON with latest run details including all output files,
        or error if no runs found.

    Example:
        # After executing a script
        latest = await get_latest_run()
        # Returns same format as list_run_outputs
    """
    server = await get_mcp_server("blender-executor")

    result = await server.call_tool(
        "get_latest_run",
        {}
    )

    if hasattr(result, 'content') and result.content:
        return result.content[0].text if result.content else "{}"
    return str(result)
