#!/usr/bin/env python3
"""
Blender Executor MCP Server

Executes Blender Python scripts via CLI with full output capture,
structured error parsing, and output file discovery.

Part of the Self-Improving NanoVDB Asset Generation Pipeline.

Tools:
    - execute_blender_script: Run script with args, capture all output
    - parse_blender_errors: Parse stderr into structured errors
    - list_run_outputs: List VDB, renders, logs from a run
    - get_latest_run: Get most recent execution directory

Usage:
    python server.py  # Run as MCP server (stdio transport)
"""

import asyncio
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# Load environment
load_dotenv()

# Project paths
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = Path(os.getenv(
    "PROJECT_ROOT",
    SCRIPT_DIR.parent.parent  # agents/blender-executor -> PlasmaDXR
))

# Blender executable (Linux installation - more stable with Mantaflow)
DEFAULT_BLENDER_EXE = "/home/maz3ppa/apps/blender-5.0.1-linux-x64/blender"
BLENDER_EXE = os.getenv("BLENDER_EXE", DEFAULT_BLENDER_EXE)

# CLI runner script
CLI_RUNNER = PROJECT_ROOT / "assets/blender_scripts/GPT-5.2/run_blender_cli.sh"

# Log directory
LOG_DIR = PROJECT_ROOT / "build/blender_cli_logs"

# Create FastMCP server
mcp = FastMCP("blender-executor")


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class BlenderError:
    """Structured Blender/Python error."""
    error_type: str  # "PYTHON", "BLENDER", "CONTEXT", "API", "FILE"
    message: str
    file: Optional[str] = None
    line: Optional[int] = None
    traceback: Optional[str] = None
    suggested_fix: Optional[str] = None


@dataclass
class ExecutionResult:
    """Result of executing a Blender script."""
    success: bool
    exit_code: int
    stdout: str
    stderr: str
    duration_seconds: float
    run_dir: str
    vdb_files: list[str]
    render_files: list[str]
    blend_files: list[str]
    log_files: list[str]
    errors: list[dict]  # List of BlenderError as dicts


@dataclass
class RunOutputs:
    """Outputs from a Blender CLI run."""
    run_dir: str
    timestamp: str
    script_name: str
    stdout_log: Optional[str]
    blender_log: Optional[str]
    vdb_files: list[str]
    render_files: list[str]
    blend_files: list[str]
    other_files: list[str]


# =============================================================================
# Error Parsing
# =============================================================================

# Common Blender error patterns
ERROR_PATTERNS = [
    # Python exceptions
    (r"(?P<type>\w+Error): (?P<message>.+)", "PYTHON"),
    (r"(?P<type>\w+Exception): (?P<message>.+)", "PYTHON"),

    # File/line info
    (r'File "(?P<file>[^"]+)", line (?P<line>\d+)', "LOCATION"),

    # Blender-specific errors
    (r"Error: (?P<message>.+)", "BLENDER"),
    (r"Warning: (?P<message>.+)", "WARNING"),

    # Context errors
    (r"RuntimeError: Operator bpy\.ops\.(?P<op>\w+\.\w+) poll failed, context is incorrect", "CONTEXT"),

    # Attribute/Key errors (common API mistakes)
    (r"AttributeError: '(?P<type>\w+)' object has no attribute '(?P<attr>\w+)'", "API"),
    (r"KeyError: '(?P<key>\w+)'", "KEY"),

    # Module not found
    (r"ModuleNotFoundError: No module named '(?P<module>\w+)'", "MODULE"),
]

# Suggested fixes for common errors
ERROR_FIXES = {
    "CONTEXT": "Ensure correct mode (OBJECT/EDIT) and active object is set before operator call.",
    "API": "Property may have been renamed or removed in Blender 5.0. Use blender-manual MCP to verify.",
    "KEY": "Object/modifier name doesn't exist. Check spelling or iterate with 'for obj in bpy.data.objects'.",
    "MODULE": "Module not available. Check if it's a Blender addon that needs enabling.",
    "openvdb_cache_compress_type": "BLOSC compression removed in Blender 5.0. Use 'ZIP' or 'NONE' instead.",
}


def parse_blender_errors_impl(stderr: str, stdout: str = "") -> list[BlenderError]:
    """
    Parse Blender/Python errors from stderr into structured format.

    Args:
        stderr: Standard error output from Blender
        stdout: Standard output (sometimes errors go here too)

    Returns:
        List of structured BlenderError objects
    """
    errors = []
    combined = stderr + "\n" + stdout
    lines = combined.split("\n")

    current_error = None
    current_traceback = []

    for i, line in enumerate(lines):
        # Check for Python exception patterns
        for pattern, error_type in ERROR_PATTERNS:
            match = re.search(pattern, line)
            if match:
                # Save previous error if exists
                if current_error:
                    current_error.traceback = "\n".join(current_traceback) if current_traceback else None
                    errors.append(current_error)
                    current_traceback = []

                groups = match.groupdict()

                if error_type == "LOCATION":
                    # This is file/line info, attach to previous or next error
                    if errors:
                        errors[-1].file = groups.get("file")
                        errors[-1].line = int(groups.get("line", 0))
                    continue

                # Create new error
                message = groups.get("message", line)

                # Determine suggested fix
                suggested_fix = ERROR_FIXES.get(error_type)

                # Check for specific known issues in message
                if "openvdb_cache_compress_type" in message:
                    suggested_fix = ERROR_FIXES["openvdb_cache_compress_type"]
                elif "has no attribute" in message:
                    suggested_fix = ERROR_FIXES.get("API")

                current_error = BlenderError(
                    error_type=error_type,
                    message=message,
                    suggested_fix=suggested_fix
                )
                break

        # Collect traceback lines
        if current_error and (line.startswith("  ") or line.startswith("\t")):
            current_traceback.append(line)

    # Don't forget the last error
    if current_error:
        current_error.traceback = "\n".join(current_traceback) if current_traceback else None
        errors.append(current_error)

    return errors


# =============================================================================
# File Discovery
# =============================================================================

def find_output_files(directory: Path, output_dir: Optional[str] = None) -> dict:
    """
    Find all output files from a Blender run.

    Args:
        directory: Run directory containing logs
        output_dir: Optional explicit output directory for VDB/renders

    Returns:
        Dict with categorized file lists
    """
    vdb_files = []
    render_files = []
    blend_files = []
    log_files = []
    other_files = []

    # Search in run directory
    if directory.exists():
        for f in directory.rglob("*"):
            if f.is_file():
                ext = f.suffix.lower()
                if ext == ".vdb":
                    vdb_files.append(str(f))
                elif ext in [".png", ".jpg", ".jpeg", ".exr", ".tiff"]:
                    render_files.append(str(f))
                elif ext == ".blend":
                    blend_files.append(str(f))
                elif ext in [".log", ".txt"]:
                    log_files.append(str(f))
                else:
                    other_files.append(str(f))

    # Also search in output_dir if specified
    if output_dir:
        output_path = Path(output_dir)
        if output_path.exists():
            for f in output_path.rglob("*"):
                if f.is_file():
                    ext = f.suffix.lower()
                    fstr = str(f)
                    if ext == ".vdb" and fstr not in vdb_files:
                        vdb_files.append(fstr)
                    elif ext in [".png", ".jpg", ".jpeg", ".exr", ".tiff"] and fstr not in render_files:
                        render_files.append(fstr)
                    elif ext == ".blend" and fstr not in blend_files:
                        blend_files.append(fstr)

    return {
        "vdb_files": sorted(vdb_files),
        "render_files": sorted(render_files),
        "blend_files": sorted(blend_files),
        "log_files": sorted(log_files),
        "other_files": sorted(other_files),
    }


def get_latest_run_dir() -> Optional[Path]:
    """Get the most recent run directory."""
    if not LOG_DIR.exists():
        return None

    runs = sorted(LOG_DIR.iterdir(), reverse=True)
    for run in runs:
        if run.is_dir() and run.name[0].isdigit():
            return run
    return None


# =============================================================================
# MCP Tools
# =============================================================================

@mcp.tool()
async def execute_blender_script(
    script_path: str,
    script_args: Optional[dict] = None,
    output_dir: Optional[str] = None,
    blend_file: Optional[str] = None,
    timeout_seconds: int = 600,
    use_ui: bool = False
) -> str:
    """
    Execute a Blender Python script via CLI with full output capture.

    Args:
        script_path: Path to .py script (absolute or relative to project root)
        script_args: Dict of arguments to pass (e.g., {"--bake": "1", "--resolution": "128"})
        output_dir: Override output directory for VDB/renders
        blend_file: Optional .blend file to open before running script
        timeout_seconds: Max execution time (default 600 = 10 minutes)
        use_ui: Run with Blender UI visible (default False = background)

    Returns:
        JSON string with ExecutionResult

    Example:
        execute_blender_script(
            script_path="assets/blender_scripts/GPT-5.2/blender_hydrogen_cloud.py",
            script_args={"--bake": "1", "--resolution": "96", "--frame_end": "24"},
            output_dir="build/vdb_output/hydrogen_test"
        )
    """
    start_time = datetime.now()

    # Resolve script path
    script = Path(script_path)
    if not script.is_absolute():
        script = PROJECT_ROOT / script_path

    if not script.exists():
        return json.dumps({
            "success": False,
            "exit_code": -1,
            "error": f"Script not found: {script}",
            "stdout": "",
            "stderr": "",
            "duration_seconds": 0,
            "run_dir": "",
            "vdb_files": [],
            "render_files": [],
            "blend_files": [],
            "log_files": [],
            "errors": []
        })

    # Check Blender executable
    if not Path(BLENDER_EXE).exists():
        return json.dumps({
            "success": False,
            "exit_code": -1,
            "error": f"Blender not found at: {BLENDER_EXE}. Set BLENDER_EXE env var.",
            "stdout": "",
            "stderr": "",
            "duration_seconds": 0,
            "run_dir": "",
            "vdb_files": [],
            "render_files": [],
            "blend_files": [],
            "log_files": [],
            "errors": []
        })

    # Build command
    cmd = [str(CLI_RUNNER)]

    if use_ui:
        cmd.append("--ui")

    if blend_file:
        cmd.extend(["--blend", blend_file])

    cmd.append(str(script))

    # Add script arguments
    if script_args:
        cmd.append("--")
        for key, value in script_args.items():
            # Handle keys that may or may not have --
            if not key.startswith("--"):
                key = f"--{key}"
            cmd.append(key)
            cmd.append(str(value))

        # Add output_dir to script args if specified
        if output_dir:
            if "--output_dir" not in script_args:
                cmd.extend(["--output_dir", output_dir])
    elif output_dir:
        cmd.extend(["--", "--output_dir", output_dir])

    # Execute
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(PROJECT_ROOT)
        )

        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout_seconds
        )

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        exit_code = proc.returncode

    except asyncio.TimeoutError:
        return json.dumps({
            "success": False,
            "exit_code": -1,
            "error": f"Execution timed out after {timeout_seconds} seconds",
            "stdout": "",
            "stderr": "",
            "duration_seconds": timeout_seconds,
            "run_dir": "",
            "vdb_files": [],
            "render_files": [],
            "blend_files": [],
            "log_files": [],
            "errors": []
        })
    except Exception as e:
        return json.dumps({
            "success": False,
            "exit_code": -1,
            "error": str(e),
            "stdout": "",
            "stderr": "",
            "duration_seconds": 0,
            "run_dir": "",
            "vdb_files": [],
            "render_files": [],
            "blend_files": [],
            "log_files": [],
            "errors": []
        })

    duration = (datetime.now() - start_time).total_seconds()

    # Find the run directory from stdout
    run_dir = ""
    run_dir_match = re.search(r"\[run_blender_cli\] Run dir:\s+(.+)", stdout)
    if run_dir_match:
        run_dir = run_dir_match.group(1).strip()
    else:
        # Try to find latest run
        latest = get_latest_run_dir()
        if latest:
            run_dir = str(latest)

    # Parse errors
    errors = parse_blender_errors_impl(stderr, stdout)

    # Find output files
    outputs = find_output_files(Path(run_dir) if run_dir else Path("."), output_dir)

    result = ExecutionResult(
        success=exit_code == 0,
        exit_code=exit_code,
        stdout=stdout,
        stderr=stderr,
        duration_seconds=duration,
        run_dir=run_dir,
        vdb_files=outputs["vdb_files"],
        render_files=outputs["render_files"],
        blend_files=outputs["blend_files"],
        log_files=outputs["log_files"],
        errors=[asdict(e) for e in errors]
    )

    return json.dumps(asdict(result), indent=2)


@mcp.tool()
async def parse_blender_errors(
    stderr: str,
    stdout: str = ""
) -> str:
    """
    Parse Blender/Python errors into structured format with suggested fixes.

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
        - "AttributeError: 'FluidDomainSettings' has no attribute 'openvdb_cache_compress_type'"
          -> Suggests checking Blender 5.0 API changes

        - "RuntimeError: Operator bpy.ops.fluid.bake_all poll failed, context is incorrect"
          -> Suggests setting active object and mode
    """
    errors = parse_blender_errors_impl(stderr, stdout)
    return json.dumps([asdict(e) for e in errors], indent=2)


@mcp.tool()
async def list_run_outputs(
    run_dir: Optional[str] = None
) -> str:
    """
    List all output files from a Blender CLI run.

    Args:
        run_dir: Path to run directory. If not specified, uses latest run.

    Returns:
        JSON with RunOutputs containing:
        - run_dir: Full path to run directory
        - timestamp: Extracted from directory name
        - script_name: Script that was executed
        - stdout_log: Path to stdout/stderr log
        - blender_log: Path to Blender's internal log
        - vdb_files: List of .vdb files found
        - render_files: List of image files (png, jpg, exr)
        - blend_files: List of .blend files
        - other_files: Other output files
    """
    if run_dir:
        run_path = Path(run_dir)
    else:
        run_path = get_latest_run_dir()
        if not run_path:
            return json.dumps({
                "error": f"No run directories found in {LOG_DIR}"
            })

    if not run_path.exists():
        return json.dumps({
            "error": f"Run directory not found: {run_path}"
        })

    # Parse timestamp and script name from directory name
    # Format: YYYYMMDD_HHMMSS_scriptname
    dir_name = run_path.name
    parts = dir_name.split("_", 2)
    timestamp = f"{parts[0]}_{parts[1]}" if len(parts) >= 2 else dir_name
    script_name = parts[2] if len(parts) >= 3 else "unknown"

    # Find log files
    stdout_log = run_path / "stdout_stderr.txt"
    blender_log = run_path / "blender.log"

    # Find all outputs
    outputs = find_output_files(run_path)

    result = RunOutputs(
        run_dir=str(run_path),
        timestamp=timestamp,
        script_name=script_name,
        stdout_log=str(stdout_log) if stdout_log.exists() else None,
        blender_log=str(blender_log) if blender_log.exists() else None,
        vdb_files=outputs["vdb_files"],
        render_files=outputs["render_files"],
        blend_files=outputs["blend_files"],
        other_files=outputs["other_files"]
    )

    return json.dumps(asdict(result), indent=2)


@mcp.tool()
async def get_latest_run() -> str:
    """
    Get information about the most recent Blender execution.

    Returns:
        JSON with latest run details, or error if no runs found.
    """
    latest = get_latest_run_dir()
    if not latest:
        return json.dumps({
            "error": f"No run directories found in {LOG_DIR}",
            "log_dir": str(LOG_DIR)
        })

    # Use list_run_outputs to get full details
    return await list_run_outputs(str(latest))


@mcp.tool()
async def list_available_scripts(
    directory: Optional[str] = None
) -> str:
    """
    List available Blender scripts in the project.

    Args:
        directory: Directory to search. Default: assets/blender_scripts/GPT-5.2/

    Returns:
        JSON array of script paths with basic info.
    """
    if directory:
        search_dir = Path(directory)
    else:
        search_dir = PROJECT_ROOT / "assets/blender_scripts/GPT-5.2"

    if not search_dir.exists():
        return json.dumps({
            "error": f"Directory not found: {search_dir}"
        })

    scripts = []
    for f in search_dir.rglob("*.py"):
        # Skip __pycache__ and test files
        if "__pycache__" in str(f) or f.name.startswith("test_"):
            continue

        # Read first docstring if available
        description = ""
        try:
            content = f.read_text()
            # Look for module docstring
            if content.startswith('"""'):
                end = content.find('"""', 3)
                if end > 0:
                    description = content[3:end].strip().split("\n")[0]
            elif content.startswith("'''"):
                end = content.find("'''", 3)
                if end > 0:
                    description = content[3:end].strip().split("\n")[0]
        except:
            pass

        scripts.append({
            "path": str(f.relative_to(PROJECT_ROOT)),
            "name": f.stem,
            "description": description[:100] if description else ""
        })

    return json.dumps(scripts, indent=2)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    mcp.run()
