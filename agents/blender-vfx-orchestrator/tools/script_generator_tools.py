"""
Function tool wrappers for script-generator MCP server.

Exposes script-generator capabilities to OpenAI Agents SDK agents:
- generate_script: Create new Blender script
- modify_script: Improve script based on feedback
- validate_script: Pre-execution validation
- list_techniques: Show available techniques
- recommend_technique: UCB1-based selection
- record_technique_outcome: Learning feedback

These wrappers translate MCP tool calls into @function_tool decorated
functions that agents can call directly.
"""

from __future__ import annotations

import json
from typing import Optional

from agents import function_tool

from utils.mcp_connection_pool import get_mcp_server


# =============================================================================
# SCRIPT GENERATION
# =============================================================================

@function_tool
async def generate_script(
    effect_type: str,
    description: str,
    output_name: str,
    resolution: int = 96,
    frame_start: int = 1,
    frame_end: int = 50,
    template_name: Optional[str] = None,
    technique_name: Optional[str] = None,
    force_random_technique: bool = False
) -> str:
    """
    Generate a new Blender Python script for VFX asset creation.

    Uses the Technique Catalog to select categorically different approaches,
    ensuring variety across generated scripts.

    Args:
        effect_type: Type of effect (pyro, explosion, fire, smoke, nebula, sun)
        description: Natural language description of what to create
        output_name: Name for the output script (without .py)
        resolution: Blender simulation resolution (default 96)
        frame_start: Animation start frame (default 1)
        frame_end: Animation end frame (default 50)
        template_name: Optional specific template to base on
        technique_name: Optional specific technique from catalog
        force_random_technique: If True, ignore keywords for maximum variety

    Returns:
        JSON with generated script path, content preview, and technique info
    """
    server = await get_mcp_server("script-generator")

    result = await server.call_tool(
        "generate_script",
        {
            "effect_type": effect_type,
            "description": description,
            "output_name": output_name,
            "resolution": resolution,
            "frame_start": frame_start,
            "frame_end": frame_end,
            "template_name": template_name,
            "technique_name": technique_name,
            "force_random_technique": force_random_technique,
        }
    )

    # Extract text content from MCP response
    if hasattr(result, 'content') and result.content:
        return result.content[0].text if result.content else "{}"
    return str(result)


@function_tool
async def modify_script(
    script_path: str,
    modifications_json: str,
    output_name: Optional[str] = None
) -> str:
    """
    Modify an existing Blender script based on evaluation feedback.

    Applies parameter changes and includes knowledge base warnings
    for potentially problematic modifications.

    Args:
        script_path: Path to the script to modify
        modifications_json: JSON string of changes to make:
            - resolution: New simulation resolution
            - frame_end: New end frame
            - turbulence: Turbulence/vorticity value (0-1)
            - density: Density multiplier
            - temperature: Temperature value
            - flame_smoke: Flame smoke ratio
            - domain_scale: Domain size multiplier
            - custom_code: Dict of {search_pattern: replacement}
            Example: '{"resolution": 128, "turbulence": 0.8}'
        output_name: Optional new filename (default: adds "_modified" suffix)

    Returns:
        JSON with modified script path, changes made, and warnings
    """
    server = await get_mcp_server("script-generator")

    # Parse JSON string to dict for MCP call
    modifications = json.loads(modifications_json)

    result = await server.call_tool(
        "modify_script",
        {
            "script_path": script_path,
            "modifications": modifications,
            "output_name": output_name,
        }
    )

    if hasattr(result, 'content') and result.content:
        return result.content[0].text if result.content else "{}"
    return str(result)


# =============================================================================
# VALIDATION
# =============================================================================

@function_tool
async def validate_script(
    script_path: str,
    strict: bool = False
) -> str:
    """
    Validate a Blender script before execution.

    Performs comprehensive pre-execution validation:
    - Python syntax validation using AST
    - Parameter extraction and range checking
    - Required pattern detection (domain, flow, physics)
    - Security/safety checks
    - Output path validation

    Args:
        script_path: Path to the Blender Python script to validate
        strict: If True, treat warnings as errors (default False)

    Returns:
        JSON with:
        - valid: True if script passes validation
        - issues: List of validation issues found
        - extracted_params: Parameters detected in script
        - detected_effect_type: volumetric, mesh, or unknown
        - error_count: Number of errors
        - warning_count: Number of warnings
    """
    server = await get_mcp_server("script-generator")

    result = await server.call_tool(
        "validate_script",
        {
            "script_path": script_path,
            "strict": strict,
        }
    )

    if hasattr(result, 'content') and result.content:
        return result.content[0].text if result.content else "{}"
    return str(result)


# =============================================================================
# TECHNIQUE SELECTION
# =============================================================================

@function_tool
async def list_techniques(
    effect_type: str = "pyro"
) -> str:
    """
    List available techniques from the catalog for variety in generation.

    Each technique produces categorically different visual results.

    Args:
        effect_type: Type of effect (pyro, explosion, etc.)

    Returns:
        JSON with available techniques and their descriptions
    """
    server = await get_mcp_server("script-generator")

    result = await server.call_tool(
        "list_techniques",
        {
            "effect_type": effect_type,
        }
    )

    if hasattr(result, 'content') and result.content:
        return result.content[0].text if result.content else "{}"
    return str(result)


@function_tool
async def recommend_technique(
    effect_type: str,
    description: str,
    keyword_weight: float = 0.3,
    prefer_untried: bool = True
) -> str:
    """
    Recommend a technique using UCB1 algorithm for exploration/exploitation balance.

    Uses keyword filtering combined with Upper Confidence Bound (UCB1)
    to select techniques. Untried techniques get exploration bonus.
    Performance data persists across sessions for learning.

    Args:
        effect_type: Type of effect (pyro, explosion, fire, etc.)
        description: Description of desired effect (used for keyword matching)
        keyword_weight: How much to weight keyword matches (0-1, default 0.3)
        prefer_untried: If True, untried techniques get exploration bonus

    Returns:
        JSON with:
        - technique_name: Recommended technique
        - confidence: 0-1 confidence score
        - selection_reason: Why this technique was selected
        - ucb_score: Raw UCB1 score
        - alternatives: Top 3 alternative techniques
        - exploration_mode: True if this is an untried technique
    """
    server = await get_mcp_server("script-generator")

    result = await server.call_tool(
        "recommend_technique",
        {
            "effect_type": effect_type,
            "description": description,
            "keyword_weight": keyword_weight,
            "prefer_untried": prefer_untried,
        }
    )

    if hasattr(result, 'content') and result.content:
        return result.content[0].text if result.content else "{}"
    return str(result)


@function_tool
async def record_technique_outcome(
    technique_name: str,
    effect_type: str,
    success: bool,
    final_score: float,
    iterations: int = 1
) -> str:
    """
    Record the outcome of using a technique for learning.

    Call this after an asset generation session completes to update
    technique performance statistics. Enables UCB1 to learn which
    techniques work best for different effect types.

    Args:
        technique_name: Name of the technique that was used
        effect_type: Type of effect (pyro, explosion, etc.)
        success: True if the technique passed quality thresholds
        final_score: Final quality score achieved (0-100)
        iterations: Number of iterations needed to pass

    Returns:
        JSON with updated technique statistics
    """
    server = await get_mcp_server("script-generator")

    result = await server.call_tool(
        "record_technique_outcome",
        {
            "technique_name": technique_name,
            "effect_type": effect_type,
            "success": success,
            "final_score": final_score,
            "iterations": iterations,
        }
    )

    if hasattr(result, 'content') and result.content:
        return result.content[0].text if result.content else "{}"
    return str(result)
