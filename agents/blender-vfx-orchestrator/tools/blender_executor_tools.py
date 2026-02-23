"""
Function tool wrappers for Blender script execution.

IN-PROCESS IMPLEMENTATION - Does NOT spawn MCP subprocesses.

Exposes Blender execution capabilities to OpenAI Agents SDK agents:
- execute_blender_script: Run Blender scripts with CLI runner
- parse_blender_errors: Structure error output for diagnosis
- list_run_outputs: Enumerate files from a run
- get_latest_run: Get most recent execution info

This uses the two-layer pattern:
- _impl functions: Plain async functions with actual logic (for internal use)
- @function_tool wrappers: Exposed to agents, call the _impl functions

IMPORTANT: This replaces the previous MCP-based implementation which caused
anyio TaskGroup conflicts when called from within OpenAI Agents SDK context.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

from agents import function_tool


# =============================================================================
# CONFIGURATION (from blender-executor MCP server)
# =============================================================================

# Project paths
SCRIPT_DIR = Path(__file__).parent
ORCHESTRATOR_ROOT = SCRIPT_DIR.parent  # tools -> blender-vfx-orchestrator
PROJECT_ROOT = Path(os.getenv(
    "PROJECT_ROOT",
    ORCHESTRATOR_ROOT.parent.parent  # blender-vfx-orchestrator -> agents -> PlasmaDXR
))

# Blender executable
DEFAULT_BLENDER_EXE = "/home/maz3ppa/apps/blender-5.0.1-linux-x64/blender"
BLENDER_EXE = os.getenv("BLENDER_EXE", DEFAULT_BLENDER_EXE)

# CLI runner script
CLI_RUNNER = PROJECT_ROOT / "assets/blender_scripts/GPT-5.2/run_blender_cli.sh"

# Log directory
LOG_DIR = PROJECT_ROOT / "build/blender_cli_logs"


# =============================================================================
# DATA CLASSES (from blender-executor MCP server)
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
    vdb_files: List[str]
    render_files: List[str]
    blend_files: List[str]
    log_files: List[str]
    errors: List[Dict]  # List of BlenderError as dicts


@dataclass
class RunOutputs:
    """Outputs from a Blender CLI run."""
    run_dir: str
    timestamp: str
    script_name: str
    stdout_log: Optional[str]
    blender_log: Optional[str]
    vdb_files: List[str]
    render_files: List[str]
    blend_files: List[str]
    other_files: List[str]


# =============================================================================
# ERROR PATTERNS (from blender-executor MCP server)
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

# Suggested fixes for common errors (general category fixes)
ERROR_FIXES = {
    "CONTEXT": "Ensure correct mode (OBJECT/EDIT) and active object is set before operator call.",
    "API": "Property may have been renamed or removed in Blender 5.0. Use blender-manual MCP to verify.",
    "KEY": "Object/modifier name doesn't exist. Check spelling or iterate with 'for obj in bpy.data.objects'.",
    "MODULE": "Module not available. Check if it's a Blender addon that needs enabling.",
    "openvdb_cache_compress_type": "BLOSC compression removed in Blender 5.0. Use 'ZIP' or 'NONE' instead.",
}

# =============================================================================
# BLENDER 5.0 SPECIFIC FIXES (pattern-matched for auto-repair)
# =============================================================================
# Each entry: (pattern_in_message, fix_description, code_replacement)
# code_replacement is a tuple: (old_code_pattern, new_code_pattern) or None if manual
BLENDER_5_FIXES = {
    # Shader node socket renames
    "'Specular'": (
        "Specular socket renamed in Blender 5.0. Use 'Specular IOR Level' instead, or check if socket exists with hasattr check.",
        ("inputs['Specular']", "inputs.get('Specular IOR Level', inputs.get('Specular', None))")
    ),
    "'Specular Tint'": (
        "Specular Tint socket renamed in Blender 5.0. Use 'Specular Tint' is now controlled differently.",
        None
    ),
    "'Subsurface'": (
        "Subsurface input renamed in Blender 5.0 to 'Subsurface Weight'. Use inputs.get() with fallback.",
        ("inputs['Subsurface']", "inputs.get('Subsurface Weight', inputs.get('Subsurface', None))")
    ),
    "'Transmission'": (
        "Transmission input renamed in Blender 5.0 to 'Transmission Weight'. Use inputs.get() with fallback.",
        ("inputs['Transmission']", "inputs.get('Transmission Weight', inputs.get('Transmission', None))")
    ),
    "'Sheen'": (
        "Sheen input renamed in Blender 5.0 to 'Sheen Weight'. Use inputs.get() with fallback.",
        ("inputs['Sheen']", "inputs.get('Sheen Weight', inputs.get('Sheen', None))")
    ),
    "'Clearcoat'": (
        "Clearcoat removed in Blender 5.0. Use 'Coat Weight' and 'Coat Roughness' instead.",
        ("inputs['Clearcoat']", "inputs.get('Coat Weight', None)")
    ),

    # Principled BSDF Emission sockets renamed in Blender 4.0+
    "'Emission'": (
        "Principled BSDF 'Emission' socket renamed in Blender 4.0+. Use 'Emission Color' for color and 'Emission Strength' for intensity. Safe pattern: sock = bsdf.inputs.get('Emission') or bsdf.inputs.get('Emission Color'); if sock: sock.default_value = color",
        ("inputs['Emission']", "inputs.get('Emission Color', inputs.get('Emission'))")
    ),
    'key "Emission" not found': (
        "Principled BSDF 'Emission' renamed to 'Emission Color' in Blender 4.0+. Use inputs.get('Emission Color') or inputs.get('Emission') with fallback.",
        None
    ),

    # Emission node has no Normal input (multiple patterns)
    "Emission': 'Normal'": (
        "Emission shader has no 'Normal' input in Blender 5.0. Remove the normal link or use Mix Shader with Diffuse for bump effect.",
        None  # Requires manual restructuring
    ),
    "'Normal' not found": (
        "The 'Normal' input doesn't exist on this node type. Check node type - Emission nodes don't have Normal input.",
        None
    ),
    # More generic Normal patterns for KeyError cases (handle both quote styles)
    "KeyError: 'Normal'": (
        "The 'Normal' input doesn't exist on this node type. Emission and Volume shaders don't have Normal inputs - only surface shaders like Principled BSDF do. Remove the normal connection for emission/volume nodes.",
        None
    ),
    'key "normal" not found': (
        "The 'Normal' input doesn't exist on this shader node. Emission and Volume Principled shaders don't accept Normal input. Only use Normal on surface shaders like Principled BSDF. Remove: links.new(bump.outputs['Normal'], emission.inputs['Normal'])",
        None
    ),
    "'Normal'": (
        "The 'Normal' input likely doesn't exist on this shader node. Emission and Volume Principled shaders don't accept Normal input. Only use Normal input on surface shaders like Principled BSDF or Diffuse BSDF.",
        None
    ),

    # Cycles volume stepping API changes
    "volume_step_size": (
        "volume_step_size deprecated in Blender 5.0. Use volume_step_rate with hasattr guard: if hasattr(scene.cycles, 'volume_step_rate'): scene.cycles.volume_step_rate = 0.5",
        ("volume_step_size", "volume_step_rate")
    ),
    "volume_max_steps": (
        "volume_max_steps may not exist in all Blender versions. Use hasattr guard: if hasattr(scene.cycles, 'volume_max_steps'): scene.cycles.volume_max_steps = 256",
        None
    ),

    # OpenVDB compression
    "openvdb_cache_compress_type": (
        "BLOSC compression removed in Blender 5.0. Use 'ZIP' or 'NONE' for cache_file_format.",
        ("'BLOSC'", "'ZIP'")
    ),
    "cache_compress_type": (
        "cache_compress_type property renamed/removed. Use cache_data_format and cache_type instead.",
        None
    ),

    # Fluid domain changes
    "use_adaptive_domain": (
        "use_adaptive_domain behavior changed. Ensure domain is properly configured with bpy.ops.fluid.bake_all().",
        None
    ),

    # Material output changes
    "'Displacement'": (
        "Displacement output may need different setup. Ensure Material Output has 'Displacement' socket and material uses displacement in settings.",
        None
    ),

    # Color management
    "'AgX'": (
        "AgX color transform available in Blender 4.0+. Use hasattr check: if 'AgX' in scene.view_settings.bl_rna.properties['view_transform'].enum_items.keys()",
        None
    ),
    "'Filmic'": (
        "Filmic may not be available. Use try/except or check available transforms.",
        None
    ),

    # Context/operator issues
    "context is incorrect": (
        "Operator poll failed. Use context override: with bpy.context.temp_override(area=area, region=region): bpy.ops.xxx()",
        None
    ),
    "override context": (
        "Context override syntax changed in Blender 3.2+. Use temp_override instead of passing dict.",
        ("bpy.ops.xxx(override,", "with bpy.context.temp_override(**override): bpy.ops.xxx(")
    ),

    # Object visibility
    "visible_shadow": (
        "visible_shadow is a valid property but may need object to be renderable. Check hide_render = False.",
        None
    ),
    "cycles_visibility": (
        "cycles_visibility removed in Blender 5.0. Use direct object properties: obj.visible_shadow, obj.visible_camera, obj.visible_diffuse. Guard with: if hasattr(obj, 'cycles_visibility'): obj.cycles_visibility.shadow = False elif hasattr(obj, 'visible_shadow'): obj.visible_shadow = False",
        None
    ),
    "shadow_method": (
        "mat.shadow_method removed in Blender 5.0. Use hasattr guard: if hasattr(mat, 'shadow_method'): mat.shadow_method = 'NONE'. For shadows, disable at object level with obj.visible_shadow = False instead.",
        None
    ),

    # Denoising
    "use_denoising": (
        "use_denoising location changed. In Blender 4.0+ use scene.cycles.use_denoising, in earlier versions it was on render layer.",
        None
    ),

    # Deprecation warnings for Blender 6.0
    "use_nodes' is expected to be removed": (
        "use_nodes will be removed in Blender 6.0. Nodes are always enabled now - simply remove the use_nodes = True line.",
        ("use_nodes = True", "# use_nodes removed in Blender 6.0 - nodes always enabled")
    ),
    "'use_nodes'": (
        "use_nodes property is deprecated. In Blender 5.0+ materials and worlds always use nodes. Remove use_nodes assignments.",
        None
    ),

    # Scene compositor access
    "Scene' object has no attribute 'node_tree'": (
        "Compositor node_tree access requires scene.use_nodes = True first. In Blender 5.0+, add hasattr check: if hasattr(scene, 'node_tree') and scene.node_tree: ...",
        None
    ),
    "'node_tree'": (
        "node_tree may not be available. For compositor: ensure scene.use_nodes = True. For materials/worlds: nodes are always enabled in Blender 5.0+.",
        None
    ),
}


def _get_specific_fix(message: str, error_type: str) -> Optional[str]:
    """
    Get a specific fix suggestion based on the error message content.

    This function matches against known Blender 5.0 API changes and returns
    detailed fix suggestions with code examples where available.

    Args:
        message: The error message text
        error_type: The general error category

    Returns:
        Specific fix suggestion or None if no specific fix found
    """
    # Check for specific Blender 5.0 patterns
    for pattern, fix_info in BLENDER_5_FIXES.items():
        if pattern.lower() in message.lower():
            description = fix_info[0] if isinstance(fix_info, tuple) else fix_info
            code_fix = fix_info[1] if isinstance(fix_info, tuple) and len(fix_info) > 1 else None

            if code_fix:
                old_code, new_code = code_fix
                return f"{description}\n\nCode fix: Replace '{old_code}' with '{new_code}'"
            return description

    # Fall back to category fix
    return ERROR_FIXES.get(error_type)


# =============================================================================
# INTERNAL IMPLEMENTATION FUNCTIONS (directly callable)
# =============================================================================

def _parse_blender_errors_impl(stderr: str, stdout: str = "") -> List[BlenderError]:
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

                # Determine suggested fix using enhanced pattern matching
                # This checks against BLENDER_5_FIXES for specific API changes
                suggested_fix = _get_specific_fix(message, error_type)

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


def _find_output_files(directory: Path, output_dir: Optional[str] = None) -> Dict[str, List[str]]:
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


def _get_latest_run_dir() -> Optional[Path]:
    """Get the most recent run directory."""
    if not LOG_DIR.exists():
        return None

    runs = sorted(LOG_DIR.iterdir(), reverse=True)
    for run in runs:
        if run.is_dir() and run.name[0].isdigit():
            return run
    return None


async def _execute_blender_script_impl(
    script_path: str,
    script_args: Optional[Dict[str, Any]] = None,
    output_dir: Optional[str] = None,
    blend_file: Optional[str] = None,
    timeout_seconds: int = 600,
    use_ui: bool = False
) -> str:
    """
    Execute a Blender Python script via CLI with full output capture.

    This is the internal implementation - directly callable.

    Args:
        script_path: Path to .py script (absolute or relative to project root)
        script_args: Dict of arguments to pass (e.g., {"bake": "1", "resolution": "128"})
        output_dir: Override output directory for VDB/renders
        blend_file: Optional .blend file to open before running script
        timeout_seconds: Max execution time (default 600 = 10 minutes)
        use_ui: Run with Blender UI visible (default False = background)

    Returns:
        JSON string with ExecutionResult
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

    # PRE-EXECUTION FIX: Apply known Blender 5.0 API fixes
    # This breaks the infinite loop when Coordinator diagnoses API issues
    # that _modify_script_impl cannot fix (it only handles Config class params)
    try:
        from tools.blender_api_fixer import validate_and_fix_script
        fix_result = validate_and_fix_script(str(script))
        if fix_result.get("fixed"):
            print(f"[Executor] Applied {len(fix_result['fixes_applied'])} API fixes before execution", file=sys.stderr)
            for fix in fix_result.get("fixes_applied", []):
                print(f"  - {fix}", file=sys.stderr)
    except ImportError:
        pass  # Fixer not available, continue without
    except Exception as e:
        print(f"[Executor] WARN: API fixer failed: {e}", file=sys.stderr)

    # Generate unique output directory for this execution
    script_basename = Path(script_path).stem
    unique_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_output_dir = str(PROJECT_ROOT / "build" / "vfx_output" / f"{unique_ts}_{script_basename}")
    os.makedirs(unique_output_dir, exist_ok=True)

    # Use unique dir if no explicit output_dir was provided
    if not output_dir:
        output_dir = unique_output_dir
        print(f"[Executor] Unique output dir: {output_dir}", file=sys.stderr)

    cache_dir = None
    if output_dir:
        cache_dir = os.path.join(output_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)

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

    # Check CLI runner exists
    if not CLI_RUNNER.exists():
        return json.dumps({
            "success": False,
            "exit_code": -1,
            "error": f"CLI runner not found: {CLI_RUNNER}",
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
    cmd = [str(CLI_RUNNER), "--save-blend"]

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
        # Pass output dir via env so scripts/wrappers can use it
        env = os.environ.copy()
        env['BLENDER_OUTPUT_DIR'] = output_dir or ""
        env['BLENDER_CACHE_DIR'] = cache_dir or ""

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(PROJECT_ROOT),
            env=env,
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
        latest = _get_latest_run_dir()
        if latest:
            run_dir = str(latest)

    # Parse errors
    errors = _parse_blender_errors_impl(stderr, stdout)

    # Find output files
    outputs = _find_output_files(Path(run_dir) if run_dir else Path("."), output_dir)

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


async def _list_run_outputs_impl(run_dir: Optional[str] = None) -> str:
    """
    List all output files from a Blender CLI run.

    Internal implementation - directly callable.

    Args:
        run_dir: Path to run directory. If not specified, uses latest run.

    Returns:
        JSON with RunOutputs
    """
    if run_dir:
        run_path = Path(run_dir)
    else:
        run_path = _get_latest_run_dir()
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
    outputs = _find_output_files(run_path)

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


async def _get_latest_run_impl() -> str:
    """
    Get information about the most recent Blender execution.

    Internal implementation - directly callable.

    Returns:
        JSON with latest run details, or error if no runs found.
    """
    latest = _get_latest_run_dir()
    if not latest:
        return json.dumps({
            "error": f"No run directories found in {LOG_DIR}",
            "log_dir": str(LOG_DIR)
        })

    # Use list_run_outputs to get full details
    return await _list_run_outputs_impl(str(latest))


# =============================================================================
# TOOL ERROR HANDLERS (Phase 2A-8)
# =============================================================================

def blender_execution_error_handler(ctx: Any, exc: Exception) -> str:
    """Convert execution exceptions into actionable error messages for the agent.

    Instead of a generic traceback, the agent sees a structured diagnosis with
    suggested next steps. This prevents the agent from hallucinating fixes.
    """
    exc_type = type(exc).__name__
    msg = str(exc)

    # Categorize by exception type with actionable guidance
    if isinstance(exc, FileNotFoundError):
        return (
            f"FILE_NOT_FOUND: {msg}. "
            "Check script_path exists. Use absolute paths or paths relative to project root."
        )
    if isinstance(exc, PermissionError):
        return (
            f"PERMISSION_DENIED: {msg}. "
            "Check file permissions on script and Blender executable."
        )
    if isinstance(exc, json.JSONDecodeError):
        return (
            f"INVALID_JSON: script_args_json is malformed: {msg}. "
            "Ensure script_args_json is valid JSON, e.g. '{\"bake\": \"1\"}'."
        )
    if isinstance(exc, (MemoryError, OSError)) and "memory" in msg.lower():
        return (
            f"OUT_OF_MEMORY: {msg}. "
            "Reduce resolution_max (try 64-96), reduce frame_end, "
            "or lower timesteps_max to decrease memory usage."
        )
    if isinstance(exc, asyncio.TimeoutError):
        return (
            f"TIMEOUT: Blender execution exceeded time limit. "
            "Reduce resolution_max, frame_end, or timesteps_max. "
            "For liquid sims, reduce flip_ratio or disable secondary particles."
        )
    # Generic fallback — still more useful than a raw traceback
    return f"EXECUTION_ERROR ({exc_type}): {msg}"


def blender_tool_error_handler(ctx: Any, exc: Exception) -> str:
    """Generic error handler for non-execution Blender tools."""
    return f"TOOL_ERROR ({type(exc).__name__}): {str(exc)[:500]}"


# =============================================================================
# FUNCTION TOOL WRAPPERS (exposed to agents)
# =============================================================================

@function_tool(
    failure_error_function=blender_execution_error_handler,
    timeout=660.0,  # SDK safety net — slightly above internal 600s default
)
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
    # Parse JSON string to dict for internal call
    script_args = json.loads(script_args_json) if script_args_json else None

    return await _execute_blender_script_impl(
        script_path=script_path,
        script_args=script_args,
        output_dir=output_dir,
        blend_file=blend_file,
        timeout_seconds=timeout_seconds,
        use_ui=use_ui
    )


@function_tool(failure_error_function=blender_tool_error_handler)
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
    errors = _parse_blender_errors_impl(stderr, stdout)
    return json.dumps([asdict(e) for e in errors], indent=2)


@function_tool(failure_error_function=blender_tool_error_handler)
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
    return await _list_run_outputs_impl(run_dir)


@function_tool(failure_error_function=blender_tool_error_handler)
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
    return await _get_latest_run_impl()
