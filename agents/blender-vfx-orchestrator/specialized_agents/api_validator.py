"""
API Validator Agent using OpenAI Agents SDK.

Validates Blender 5.0 API calls in Python scripts against documentation.
This agent is called BEFORE code is written/executed to catch API errors
at the source rather than discovering them through runtime failures.

Key capabilities:
- Extract bpy.types.*, bpy.ops.*, bpy.context.* calls from code
- Validate each call against Blender 5.0 documentation via vector store
- Return structured validation results with corrections
- Callable via as_tool() for integration with Script Writer

SDK Pattern: This agent uses structured output (Pydantic) and is designed
to be called as a tool from the orchestrator via agent.as_tool().

PROBLEM ADDRESSED: Problem 1 from Architecture Optimization Plan
- LLM generates code from training data (Blender 2.8x-4.x) instead of 5.0
- This agent REQUIRES documentation lookup before approving any API usage
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from pydantic import BaseModel, Field
from agents import Agent, ModelSettings, function_tool

if TYPE_CHECKING:
    from agents import RunContextWrapper

# Add parent directory to path for imports
_parent_dir = str(Path(__file__).parent.parent)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

# Import semantic docs tools for API validation
from tools.semantic_docs_tools import (
    semantic_search_blender_docs,
    search_blender_api_by_intent,
)


# =============================================================================
# STRUCTURED OUTPUT MODELS (SDK Pattern: Pydantic for type safety)
# =============================================================================

class APICallValidation(BaseModel):
    """Validation result for a single API call."""
    api_call: str = Field(description="The original API call (e.g., 'bpy.types.FluidDomainSettings.flame_smoke')")
    is_valid: bool = Field(description="Whether the API call exists in Blender 5.0")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in validation (0.0-1.0)")
    correction: Optional[str] = Field(default=None, description="Corrected API call if invalid")
    documentation_source: Optional[str] = Field(default=None, description="Source of validation info")
    notes: Optional[str] = Field(default=None, description="Additional validation notes")


class CodeValidationResult(BaseModel):
    """Complete validation result for a code snippet."""
    is_valid: bool = Field(description="True if ALL API calls are valid")
    total_calls_checked: int = Field(description="Number of API calls validated")
    valid_calls: int = Field(description="Number of valid API calls")
    invalid_calls: int = Field(description="Number of invalid API calls")
    validations: List[APICallValidation] = Field(description="Individual validation results")
    corrections_needed: List[str] = Field(description="List of corrections to apply")
    summary: str = Field(description="Human-readable summary of validation")


# =============================================================================
# KNOWN BLENDER 5.0 API CHANGES (High-confidence corrections)
# =============================================================================

# These are KNOWN breaking changes from Blender 4.x to 5.0
# Used as high-confidence corrections when documentation search is ambiguous
KNOWN_API_CHANGES: Dict[str, Dict[str, str]] = {
    # Principled Volume shader socket renames
    "inputs[\"Smoke\"]": {
        "correction": "inputs[\"Grid\"]",
        "reason": "Renamed in Blender 5.0 Principled Volume shader"
    },
    "inputs[\"Smoke Color\"]": {
        "correction": "inputs[\"Grid Color\"]",
        "reason": "Renamed in Blender 5.0 Principled Volume shader"
    },
    # FluidDomainSettings changes
    "modifier.effector_weights": {
        "correction": "effector_weights",
        "reason": "effector_weights moved from modifier to domain settings in 5.0"
    },
    # FluidDomainSettings.use_caching removed (Blender 5.0)
    ".use_caching": {
        "correction": "# use_caching removed in Blender 5.0 - delete this line",
        "reason": "FluidDomainSettings.use_caching was removed in Blender 5.0"
    },
    # FluidDomainSettings.adaptive_domain renamed (Blender 5.0)
    ".adaptive_domain": {
        "correction": ".use_adaptive_domain",
        "reason": "FluidDomainSettings.adaptive_domain renamed to use_adaptive_domain in Blender 5.0"
    },
    # FluidDomainSettings.noise_res_factor removed (Blender 5.0)
    ".noise_res_factor": {
        "correction": "# noise_res_factor removed in Blender 5.0 - delete this line",
        "reason": "FluidDomainSettings.noise_res_factor removed in Blender 5.0"
    },
    # FluidDomainSettings.resolution_divisions does NOT exist - use resolution_max
    ".resolution_divisions": {
        "correction": ".resolution_max",
        "reason": "FluidDomainSettings.resolution_divisions does not exist. Use resolution_max instead."
    },
    "resolution_divisions": {
        "correction": "resolution_max",
        "reason": "FluidDomainSettings.resolution_divisions does not exist. Use resolution_max instead."
    },
    # LLM TYPO: use_adaptive_time_steps → use_adaptive_timesteps (no underscore)
    ".use_adaptive_time_steps": {
        "correction": ".use_adaptive_timesteps",
        "reason": "TYPO: use_adaptive_time_steps does not exist. Correct spelling is use_adaptive_timesteps (no underscore between time and steps)."
    },
    "use_adaptive_time_steps": {
        "correction": "use_adaptive_timesteps",
        "reason": "TYPO: use_adaptive_time_steps does not exist. Correct spelling is use_adaptive_timesteps (no underscore between time and steps)."
    },
    # FluidDomainSettings.use_dissolve does NOT exist - correct is use_dissolve_smoke
    ".use_dissolve": {
        "correction": ".use_dissolve_smoke",
        "reason": "use_dissolve does not exist. Use use_dissolve_smoke for gas domain dissolve. Related: use_dissolve_smoke_log for logarithmic dissolve."
    },
    "use_dissolve": {
        "correction": "use_dissolve_smoke",
        "reason": "use_dissolve does not exist. Use use_dissolve_smoke for gas domain dissolve. Related: use_dissolve_smoke_log for logarithmic dissolve."
    },
    # FluidDomainSettings.cache_format does NOT exist - correct is cache_data_format
    ".cache_format": {
        "correction": ".cache_data_format",
        "reason": "cache_format does not exist. Correct attribute is cache_data_format (with 'data' in the name)."
    },
    "cache_format": {
        "correction": "cache_data_format",
        "reason": "cache_format does not exist. Correct attribute is cache_data_format (with 'data' in the name)."
    },
    # FluidFlowSettings.velocity does NOT exist - use velocity_factor, velocity_normal, etc.
    # NOTE: Use " =" suffix to avoid matching valid attrs like velocity_normal, velocity_factor
    ".velocity =": {
        "correction": ".velocity_factor =",
        "reason": "FluidFlowSettings.velocity does not exist. Use velocity_factor, velocity_normal, velocity_random, or use_initial_velocity."
    },
    ".velocity=": {
        "correction": ".velocity_factor =",
        "reason": "FluidFlowSettings.velocity does not exist (no space variant)."
    },
    # =============================================================================
    # LLM HALLUCINATED ATTRIBUTES (Phase 3: 2026-01-25)
    # These are attributes that LLMs commonly hallucinate but DO NOT EXIST
    # =============================================================================
    # FluidDomainSettings.timesteps_per_frame does NOT exist
    ".timesteps_per_frame": {
        "correction": ".timesteps_max",
        "reason": "HALLUCINATED: timesteps_per_frame does not exist. Use timesteps_max (int 1-45) or cfl_condition (float 0-10) for simulation stability."
    },
    "timesteps_per_frame": {
        "correction": "timesteps_max",
        "reason": "HALLUCINATED: timesteps_per_frame does not exist. Use timesteps_max (int 1-45) or cfl_condition (float 0-10) for simulation stability."
    },
    # FluidDomainSettings.timesteps_maximum does NOT exist (also hallucinated)
    ".timesteps_maximum": {
        "correction": ".timesteps_max",
        "reason": "HALLUCINATED: timesteps_maximum does not exist. The correct attribute is timesteps_max (int 1-45)."
    },
    "timesteps_maximum": {
        "correction": "timesteps_max",
        "reason": "HALLUCINATED: timesteps_maximum does not exist. The correct attribute is timesteps_max (int 1-45)."
    },
    # =============================================================================
    # HALLUCINATED ATTRIBUTES - FluidFlowSettings (Phase 4: 2026-01-26)
    # =============================================================================
    # FluidFlowSettings.velocity_multi does NOT exist - use velocity_factor
    ".velocity_multi": {
        "correction": ".velocity_factor",
        "reason": "HALLUCINATED: velocity_multi does not exist. Use velocity_factor (float 0-1000) for initial velocity."
    },
    "velocity_multi": {
        "correction": "velocity_factor",
        "reason": "HALLUCINATED: velocity_multi does not exist. Use velocity_factor (float 0-1000) for initial velocity."
    },
    # =============================================================================
    # TYPE WARNINGS (Phase 3: 2026-01-25, Updated Phase 4: 2026-01-26)
    # These attributes EXIST but have specific type requirements that LLMs miss
    # =============================================================================
    # FluidDomainSettings.noise_scale expects INT, not float
    "noise_scale = 1.0": {
        "correction": "noise_scale = 1  # Must be int, not float",
        "reason": "TYPE ERROR: noise_scale expects int, not float. Use int(value) or integer literals."
    },
    "noise_scale = 2.0": {
        "correction": "noise_scale = 2  # Must be int, not float",
        "reason": "TYPE ERROR: noise_scale expects int, not float. Use int(value) or integer literals."
    },
    "noise_scale = 0.5": {
        "correction": "noise_scale = 1  # Must be int >= 1, not float",
        "reason": "TYPE ERROR: noise_scale expects int >= 1, not float. Minimum value is 1."
    },
    "noise_scale = 3.0": {
        "correction": "noise_scale = 3  # Must be int, not float",
        "reason": "TYPE ERROR: noise_scale expects int, not float. Use int(value) or integer literals."
    },
    "noise_scale = 4.0": {
        "correction": "noise_scale = 4  # Must be int, not float",
        "reason": "TYPE ERROR: noise_scale expects int, not float. Use int(value) or integer literals."
    },
    # bpy.app.build_options changes (Blender 5.0)
    "bpy.app.build_options.engines": {
        "correction": "getattr(bpy.app.build_options, 'cycles', False)",
        "reason": "bpy.app.build_options.engines removed in Blender 5.0. Check render engines via boolean attributes like 'cycles'"
    },
    "build_options.engines": {
        "correction": "getattr(bpy.app.build_options, 'cycles', False)",
        "reason": "bpy.app.build_options.engines removed in Blender 5.0"
    },
    # Common render engine check patterns
    "'CYCLES' in bpy.app.build_options.engines": {
        "correction": "getattr(bpy.app.build_options, 'cycles', False)",
        "reason": "Use boolean attribute check instead of 'in engines' in Blender 5.0"
    },
    "\"CYCLES\" in bpy.app.build_options.engines": {
        "correction": "getattr(bpy.app.build_options, 'cycles', False)",
        "reason": "Use boolean attribute check instead of 'in engines' in Blender 5.0"
    },
    # CyclesRenderSettings removed attributes (Blender 5.0)
    "volume_samples": {
        "correction": "# volume_samples removed - Cycles uses automatic volume sampling in 5.0",
        "reason": "CyclesRenderSettings.volume_samples removed in Blender 5.0. Volume sampling is now automatic."
    },
    "CyclesRenderSettings.volume_samples": {
        "correction": "# volume_samples removed - Cycles uses automatic volume sampling in 5.0",
        "reason": "CyclesRenderSettings.volume_samples removed in Blender 5.0. Volume sampling is now automatic."
    },
    ".volume_samples": {
        "correction": "# volume_samples removed - Cycles uses automatic volume sampling in 5.0",
        "reason": "scene.cycles.volume_samples removed in Blender 5.0. Volume sampling is now automatic."
    },
    # Compositor node_tree access (Blender 5.0 - CRITICAL)
    "scene.node_tree": {
        "correction": "bpy.context.scene.node_tree",
        "reason": "In Blender 5.0, scene.node_tree is not a direct attribute. Use bpy.context.scene.node_tree after scene.use_nodes=True"
    },
    "nt = scene.node_tree": {
        "correction": "nt = bpy.context.scene.node_tree",
        "reason": "In Blender 5.0, access compositor node_tree via bpy.context.scene.node_tree"
    },
    # Mantaflow baking issues (Blender 5.0)
    "bpy.ops.fluid.free_all()": {
        "correction": "# Skip free_all() on fresh scenes - call bpy.context.view_layer.update() instead",
        "reason": "free_all() fails with 'grids still in use' on fresh scenes. Update depsgraph first, then bake directly."
    },
    # ShaderNodeSeparateRGB/CombineRGB REMOVED (Blender 5.0 - CRITICAL)
    "ShaderNodeSeparateRGB": {
        "correction": "ShaderNodeSeparateColor",
        "reason": "ShaderNodeSeparateRGB removed in Blender 5.0. Use ShaderNodeSeparateColor with mode='RGB'. Inputs: 'Color' (not 'Image'). Outputs: 'Red', 'Green', 'Blue', 'Alpha' (not 'R', 'G', 'B')."
    },
    "'ShaderNodeSeparateRGB'": {
        "correction": "'ShaderNodeSeparateColor'",
        "reason": "ShaderNodeSeparateRGB removed in Blender 5.0. Use ShaderNodeSeparateColor."
    },
    "ShaderNodeCombineRGB": {
        "correction": "ShaderNodeCombineColor",
        "reason": "ShaderNodeCombineRGB removed in Blender 5.0. Use ShaderNodeCombineColor with mode='RGB'. Inputs: 'Red', 'Green', 'Blue', 'Alpha' (not 'R', 'G', 'B')."
    },
    "'ShaderNodeCombineRGB'": {
        "correction": "'ShaderNodeCombineColor'",
        "reason": "ShaderNodeCombineRGB removed in Blender 5.0. Use ShaderNodeCombineColor."
    },
    # Node socket name changes for SeparateColor/CombineColor
    ".outputs['R']": {
        "correction": ".outputs['Red']",
        "reason": "ShaderNodeSeparateColor uses 'Red' not 'R' in Blender 5.0"
    },
    ".outputs['G']": {
        "correction": ".outputs['Green']",
        "reason": "ShaderNodeSeparateColor uses 'Green' not 'G' in Blender 5.0"
    },
    ".outputs['B']": {
        "correction": ".outputs['Blue']",
        "reason": "ShaderNodeSeparateColor uses 'Blue' not 'B' in Blender 5.0"
    },
    ".inputs['Image']": {
        "correction": ".inputs['Color']",
        "reason": "ShaderNodeSeparateColor uses 'Color' input not 'Image' in Blender 5.0"
    },
    # CyclesRenderSettings.feature_set REMOVED (Blender 5.0)
    "scene.cycles.feature_set": {
        "correction": "# scene.cycles.feature_set removed in Blender 5.0 - experimental features are always available",
        "reason": "CyclesRenderSettings.feature_set removed in Blender 5.0. Experimental features like adaptive subdivision are enabled differently."
    },
    ".feature_set": {
        "correction": "# .feature_set removed in Blender 5.0",
        "reason": "CyclesRenderSettings.feature_set removed in Blender 5.0."
    },
    # bpy.ops.object.forcefield_add REMOVED (Blender 5.0) — use effector_add
    "forcefield_add": {
        "correction": "effector_add",
        "reason": "bpy.ops.object.forcefield_add does not exist in Blender 5.0. Use bpy.ops.object.effector_add() instead."
    },
    "bpy.ops.object.forcefield_add": {
        "correction": "bpy.ops.object.effector_add",
        "reason": "forcefield_add removed in Blender 5.0. Use effector_add."
    },
    # CRITICAL: Fluid modifier setup sequence (Blender 5.0)
    # domain_settings is None until fluid_type='DOMAIN' is set
    "mod.domain_settings.domain_type": {
        "correction": "mod.fluid_type = 'DOMAIN'; mod.domain_settings.domain_type",
        "reason": "CRITICAL: In Blender 5.0, you MUST set mod.fluid_type='DOMAIN' BEFORE accessing domain_settings. The domain_settings attribute is None until fluid_type is set."
    },
    "modifier.domain_settings": {
        "correction": "modifier.fluid_type = 'DOMAIN' # Set first, then access modifier.domain_settings",
        "reason": "CRITICAL: In Blender 5.0, domain_settings is None until fluid_type='DOMAIN' is set on the FLUID modifier."
    },
}

# =============================================================================
# BACKWARDS-COMPATIBILITY ANTI-PATTERNS (Flag these as warnings)
# =============================================================================

# Patterns that indicate version-compatibility code (we only support Blender 5.0)
VERSION_COMPAT_ANTIPATTERNS = {
    r"if\s+hasattr\s*\([^,]+,\s*['\"](\w+)['\"]\s*\)": {
        "severity": "warning",
        "reason": "hasattr() for version detection - use exact Blender 5.0 API instead"
    },
    r"set_attr_if_exists": {
        "severity": "warning",
        "reason": "Version compatibility helper - use exact Blender 5.0 API instead"
    },
    r"#.*[Bb]lender\s+[34]\.[x\d]": {
        "severity": "warning",
        "reason": "Reference to old Blender version in comments - we only support 5.0"
    },
    r"if\s+not\s+set_\w+\s*\([^)]+\)\s*:\s*set_": {
        "severity": "warning",
        "reason": "Fallback chain pattern - use exact Blender 5.0 API instead"
    },
    r"try:\s*\n\s*\w+\.\w+\s*=.*\nexcept\s+AttributeError": {
        "severity": "warning",
        "reason": "Try/except for API detection - use exact Blender 5.0 API instead"
    },
}

# Known valid Blender 5.0 API patterns (don't warn about these)
KNOWN_VALID_5_0_PATTERNS = [
    r"bpy\.types\.FluidDomainSettings\.\w+",
    r"bpy\.types\.FluidFlowSettings\.\w+",
    r"bpy\.types\.FluidEffectorSettings\.\w+",
    r"bpy\.ops\.fluid\.\w+",
    r"bpy\.ops\.object\.\w+",
    r"bpy\.context\.\w+",
    r"bpy\.data\.\w+",
]


# =============================================================================
# API EXTRACTION TOOLS
# =============================================================================

def extract_api_calls_from_code(code: str) -> List[str]:
    """
    Extract Blender API calls from Python code.

    Looks for patterns:
    - bpy.types.* (type references)
    - bpy.ops.* (operators)
    - bpy.context.* (context access)
    - bpy.data.* (data access)
    - .inputs["*"] (node socket access)
    - domain.* settings (FluidDomainSettings)

    Args:
        code: Python source code

    Returns:
        List of unique API calls found
    """
    api_calls = set()

    # Pattern 1: bpy.types.ClassName
    for match in re.finditer(r'bpy\.types\.(\w+)', code):
        api_calls.add(f"bpy.types.{match.group(1)}")

    # Pattern 2: bpy.types.ClassName.property
    for match in re.finditer(r'bpy\.types\.(\w+)\.(\w+)', code):
        api_calls.add(f"bpy.types.{match.group(1)}.{match.group(2)}")

    # Pattern 3: bpy.ops.category.operator
    for match in re.finditer(r'bpy\.ops\.(\w+)\.(\w+)', code):
        api_calls.add(f"bpy.ops.{match.group(1)}.{match.group(2)}")

    # Pattern 4: Node socket access .inputs["Name"]
    for match in re.finditer(r'\.inputs\["([^"]+)"\]', code):
        api_calls.add(f'inputs["{match.group(1)}"]')

    # Pattern 5: domain.property (FluidDomainSettings)
    for match in re.finditer(r'domain\.(\w+)', code):
        prop = match.group(1)
        # Skip common non-API attributes
        if prop not in ['name', 'data', 'location', 'scale', 'rotation']:
            api_calls.add(f"domain.{prop}")

    # Pattern 6: flow_settings.property (FluidFlowSettings)
    for match in re.finditer(r'flow_settings\.(\w+)', code):
        api_calls.add(f"flow_settings.{match.group(1)}")

    # Pattern 7: modifier.effector_weights (deprecated pattern)
    if 'modifier.effector_weights' in code:
        api_calls.add('modifier.effector_weights')

    # Pattern 8: bpy.app.* (application info/build options)
    for match in re.finditer(r'bpy\.app\.(\w+)\.(\w+)', code):
        api_calls.add(f"bpy.app.{match.group(1)}.{match.group(2)}")

    # Pattern 9: bpy.app.build_options.engines (specific deprecated pattern)
    if 'bpy.app.build_options.engines' in code or 'build_options.engines' in code:
        api_calls.add('bpy.app.build_options.engines')

    # Pattern 10: Check for "in engines" pattern (common mistake)
    if re.search(r'in\s+bpy\.app\.build_options\.engines', code):
        api_calls.add('"CYCLES" in bpy.app.build_options.engines')

    # Pattern 11: CyclesRenderSettings.volume_samples (removed in 5.0)
    if '.volume_samples' in code or 'volume_samples' in code:
        api_calls.add('.volume_samples')

    # Pattern 12: scene.cycles.* properties
    for match in re.finditer(r'scene\.cycles\.(\w+)', code):
        prop = match.group(1)
        if prop == 'volume_samples':
            api_calls.add('CyclesRenderSettings.volume_samples')

    # Pattern 13: ShaderNodeSeparateRGB/CombineRGB (REMOVED in Blender 5.0)
    if 'ShaderNodeSeparateRGB' in code:
        api_calls.add('ShaderNodeSeparateRGB')
    if 'ShaderNodeCombineRGB' in code:
        api_calls.add('ShaderNodeCombineRGB')

    # Pattern 14: Old socket names for Separate/Combine nodes
    for match in re.finditer(r"\.outputs\['([RGB])'\]", code):
        letter = match.group(1)
        api_calls.add(f".outputs['{letter}']")
    if ".inputs['Image']" in code:
        api_calls.add(".inputs['Image']")

    # Pattern 15: CRITICAL - Detect domain_settings access without fluid_type='DOMAIN'
    # This is a sequence issue: domain_settings is None until fluid_type is set
    if '.domain_settings' in code:
        # Check if fluid_type = 'DOMAIN' is set BEFORE accessing domain_settings
        fluid_type_set = re.search(r"\.fluid_type\s*=\s*['\"]DOMAIN['\"]", code)

        # Match patterns like:
        # - mod.domain_settings.xxx
        # - xxx = mod.domain_settings
        # - modifiers["Fluid"].domain_settings
        domain_settings_access = re.search(
            r"(\w+\.domain_settings[.\[\]]|=\s*\w+\.domain_settings\s*$|modifiers\[.+\]\.domain_settings)",
            code,
            re.MULTILINE
        )

        if domain_settings_access and not fluid_type_set:
            api_calls.add("mod.domain_settings.domain_type")

    # Pattern 16: TYPE ERROR - noise_scale assigned with float (must be int)
    # Match: noise_scale = X.Y or noise_scale = float(X)
    for match in re.finditer(r'noise_scale\s*=\s*(\d+\.\d+|\d+\.)', code):
        float_val = match.group(1)
        api_calls.add(f"noise_scale = {float_val}")

    # Pattern 17: HALLUCINATED - velocity_multi does not exist
    if 'velocity_multi' in code:
        api_calls.add("velocity_multi")

    return sorted(api_calls)


@function_tool
def extract_blender_api_calls(code: str) -> str:
    """
    Extract all Blender API calls from Python code for validation.

    Args:
        code: Python source code to analyze

    Returns:
        JSON with extracted API calls and their locations
    """
    api_calls = extract_api_calls_from_code(code)

    # Check for known problematic patterns
    known_issues = []
    for call in api_calls:
        for pattern, fix in KNOWN_API_CHANGES.items():
            if pattern in call:
                known_issues.append({
                    "api_call": call,
                    "known_issue": pattern,
                    "correction": fix["correction"],
                    "reason": fix["reason"]
                })

    return json.dumps({
        "total_api_calls": len(api_calls),
        "api_calls": api_calls,
        "known_issues_found": len(known_issues),
        "known_issues": known_issues
    }, indent=2)


@function_tool
def check_known_api_changes(api_call: str) -> str:
    """
    Check if an API call matches known Blender 5.0 breaking changes.

    Args:
        api_call: The API call to check (e.g., 'inputs["Smoke"]')

    Returns:
        JSON with known correction if exists, otherwise "no_known_issue"
    """
    for pattern, fix in KNOWN_API_CHANGES.items():
        if pattern in api_call:
            return json.dumps({
                "has_known_issue": True,
                "original": api_call,
                "pattern_matched": pattern,
                "correction": fix["correction"],
                "reason": fix["reason"],
                "confidence": 1.0  # Known issues are high confidence
            })

    return json.dumps({
        "has_known_issue": False,
        "api_call": api_call,
        "note": "No known breaking changes for this API call"
    })


@function_tool
def validate_api_call_against_docs(api_call: str) -> str:
    """
    Validate a single API call against Blender 5.0 documentation.

    This tool searches the vector store for documentation about the API call
    and determines if it exists in Blender 5.0.

    Args:
        api_call: The API call to validate (e.g., 'bpy.types.FluidDomainSettings.flame_smoke')

    Returns:
        JSON with validation result, confidence, and any corrections
    """
    # First check known issues (high confidence)
    for pattern, fix in KNOWN_API_CHANGES.items():
        if pattern in api_call:
            return json.dumps({
                "api_call": api_call,
                "is_valid": False,
                "confidence": 1.0,
                "correction": api_call.replace(pattern, fix["correction"]),
                "source": "known_breaking_changes",
                "reason": fix["reason"]
            })

    # For other calls, we need to rely on the agent to use semantic search
    # This tool just marks the call as "needs_verification"
    return json.dumps({
        "api_call": api_call,
        "status": "needs_verification",
        "instruction": "Use semantic_search_blender_docs to verify this API exists in Blender 5.0",
        "search_query": f"Blender 5.0 API {api_call}"
    })


@function_tool
def format_validation_report(
    code: str,
    validations: str  # JSON string of validation results
) -> str:
    """
    Format a complete validation report for code review.

    Args:
        code: The original code being validated
        validations: JSON string containing validation results

    Returns:
        Formatted report with summary and corrections
    """
    try:
        results = json.loads(validations)
    except json.JSONDecodeError:
        return json.dumps({"error": "Invalid validations JSON"})

    # Count results
    total = len(results) if isinstance(results, list) else 0
    valid = sum(1 for r in results if r.get("is_valid", False)) if isinstance(results, list) else 0
    invalid = total - valid

    corrections = []
    if isinstance(results, list):
        for r in results:
            if not r.get("is_valid", True) and r.get("correction"):
                corrections.append({
                    "original": r.get("api_call"),
                    "corrected": r.get("correction"),
                    "reason": r.get("reason", "See Blender 5.0 docs")
                })

    report = {
        "summary": {
            "total_api_calls": total,
            "valid": valid,
            "invalid": invalid,
            "pass": invalid == 0
        },
        "corrections_needed": corrections,
        "recommendation": (
            "Code is valid for Blender 5.0" if invalid == 0
            else f"Apply {len(corrections)} corrections before execution"
        )
    }

    return json.dumps(report, indent=2)


# =============================================================================
# AGENT INSTRUCTIONS
# =============================================================================

API_VALIDATOR_INSTRUCTIONS = """## ROLE
You are the Blender 5.0 API Validator. Your job is to validate Python code
against Blender 5.0 documentation BEFORE it is executed.

## CRITICAL: Blender 5.0 API Changes
Blender 5.0 has BREAKING CHANGES from 4.x. Common issues:
- inputs["Smoke"] → inputs["Grid"] (Principled Volume shader)
- modifier.effector_weights → effector_weights (direct on domain)
- flow_type → flow_behavior (FluidFlowSettings)

## TURN BUDGET: MAX 4 TURNS
T1: extract_blender_api_calls(code) - get all API calls
T2: check_known_api_changes for each suspicious call
T3: semantic_search_blender_docs for unknown APIs (ONE search max)
T4: Return CodeValidationResult

## WORKFLOW
1. Extract API calls from code
2. Check each against known breaking changes (FAST, no LLM)
3. For unknown APIs, search docs ONCE (avoid research loops)
4. Return structured validation result

## OUTPUT: CodeValidationResult
- is_valid: bool (True only if ALL calls valid)
- total_calls_checked: int
- valid_calls: int
- invalid_calls: int
- validations: List[APICallValidation]
- corrections_needed: List[str] (specific corrections to apply)
- summary: str

## RULES
- NEVER approve code without checking known_api_changes
- ONE semantic search max (avoid loops)
- If uncertain about an API, mark it as needs_manual_review
- Confidence > 0.8 for known issues, < 0.5 for uncertain
- Always provide specific corrections, not vague suggestions
"""


# =============================================================================
# AGENT FACTORY
# =============================================================================

def create_api_validator(custom_instructions: str = "") -> Agent:
    """
    Create an API Validator agent for Blender 5.0 code validation.

    This agent is designed to be called via as_tool() from the orchestrator
    or Script Writer to validate code BEFORE execution.

    Args:
        custom_instructions: Additional context (e.g., current effect type)

    Returns:
        Agent instance ready for use
    """
    instructions = API_VALIDATOR_INSTRUCTIONS
    if custom_instructions:
        instructions = instructions + "\n\n" + custom_instructions

    return Agent(
        name="API Validator",
        instructions=instructions,
        model=os.getenv("API_VALIDATOR_MODEL", "gpt-4.1"),  # Fast model for validation
        model_settings=ModelSettings(
            temperature=0.0,  # Deterministic for validation
        ),
        tools=[
            # API extraction and validation
            extract_blender_api_calls,
            check_known_api_changes,
            validate_api_call_against_docs,
            format_validation_report,
            # Documentation search (use sparingly - one search max)
            semantic_search_blender_docs,
            search_blender_api_by_intent,
        ],
    )


# =============================================================================
# AS_TOOL WRAPPER FOR ORCHESTRATOR INTEGRATION
# =============================================================================

def get_api_validator_as_tool():
    """
    Get the API Validator agent wrapped as a tool for orchestrator use.

    SDK Pattern: Uses function_tool wrapper with Runner.run() and explicit
    max_turns instead of agent.as_tool() which cannot enforce turn limits.

    Turn limit: 3 (validate code, report results)

    Returns:
        function_tool instance that can be added to another agent's tools list

    Usage:
        orchestrator = Agent(
            tools=[
                get_api_validator_as_tool(),
                # ... other tools
            ]
        )
    """
    from agents import Runner

    validator = create_api_validator()

    @function_tool
    async def validate_blender_api(code: str) -> str:
        """Validate Blender Python code against Blender 5.0 API documentation.
        Call BEFORE writing or executing scripts to catch API errors.
        Returns validation result with specific corrections for any invalid APIs."""
        result = await Runner.run(
            validator,
            f"Validate this Blender Python code:\n\n```python\n{code}\n```",
            max_turns=3,
        )
        return str(result.final_output)

    return validate_blender_api


# =============================================================================
# STANDALONE VALIDATION FUNCTION (for direct Python calls)
# =============================================================================

async def validate_code_api(code: str) -> CodeValidationResult:
    """
    Validate code API calls without running the full agent.

    This is a lightweight validation that only checks known issues.
    For full validation with documentation search, use the agent.

    Args:
        code: Python source code to validate

    Returns:
        CodeValidationResult with validation details
    """
    api_calls = extract_api_calls_from_code(code)
    validations = []
    corrections_needed = []

    for call in api_calls:
        validation = APICallValidation(
            api_call=call,
            is_valid=True,
            confidence=0.5,  # Default confidence for unknown
        )

        # Check known issues
        for pattern, fix in KNOWN_API_CHANGES.items():
            if pattern in call:
                corrected = call.replace(pattern, fix["correction"])
                validation = APICallValidation(
                    api_call=call,
                    is_valid=False,
                    confidence=1.0,
                    correction=corrected,
                    documentation_source="known_breaking_changes",
                    notes=fix["reason"]
                )
                corrections_needed.append(f"{call} → {corrected}")
                break

        validations.append(validation)

    # Check for version compatibility anti-patterns
    version_compat_warnings = []
    for pattern, info in VERSION_COMPAT_ANTIPATTERNS.items():
        if re.search(pattern, code, re.MULTILINE):
            version_compat_warnings.append(f"[{info['severity'].upper()}] {info['reason']}")

    valid_count = sum(1 for v in validations if v.is_valid)
    invalid_count = len(validations) - valid_count

    # Add version compat warnings to summary
    summary_parts = []
    if validations:
        summary_parts.append(f"Validated {len(validations)} API calls: {valid_count} valid, {invalid_count} invalid")
    else:
        summary_parts.append("No API calls found to validate")

    if version_compat_warnings:
        summary_parts.append(f"VERSION COMPAT WARNINGS: {'; '.join(version_compat_warnings[:3])}")

    return CodeValidationResult(
        is_valid=(invalid_count == 0 and len(version_compat_warnings) == 0),
        total_calls_checked=len(validations),
        valid_calls=valid_count,
        invalid_calls=invalid_count,
        validations=validations,
        corrections_needed=corrections_needed,
        summary=" | ".join(summary_parts)
    )


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import asyncio
    from agents import Runner

    async def test():
        print("Testing API Validator Agent...")
        print("-" * 60)

        # Test code with known issues
        test_code = '''
import bpy

# Create domain
bpy.ops.mesh.primitive_cube_add()
domain = bpy.context.active_object
bpy.ops.object.modifier_add(type='FLUID')
domain.modifiers["Fluid"].fluid_type = 'DOMAIN'

# BAD: Old API pattern (should be caught)
domain.modifier.effector_weights.wind = 1.0

# Set up shader
mat = bpy.data.materials.new("VolumeShader")
mat.use_nodes = True
principled_volume = mat.node_tree.nodes.new('ShaderNodeVolumePrincipled')

# BAD: Old socket name (should be caught)
principled_volume.inputs["Smoke"].default_value = 0.5
principled_volume.inputs["Smoke Color"].default_value = (1, 0.5, 0.2, 1)

# GOOD: New socket name
principled_volume.inputs["Grid"].default_value = 0.8
'''

        print("Test code:")
        print(test_code[:200] + "...")
        print()

        # Test 1: Lightweight validation (no agent)
        print("Test 1: Lightweight validation (known issues only)")
        result = await validate_code_api(test_code)
        print(f"  Valid: {result.is_valid}")
        print(f"  Checked: {result.total_calls_checked} calls")
        print(f"  Invalid: {result.invalid_calls}")
        print(f"  Corrections: {result.corrections_needed}")
        print()

        # Test 2: Extract API calls
        print("Test 2: API call extraction")
        api_calls = extract_api_calls_from_code(test_code)
        print(f"  Found {len(api_calls)} API calls:")
        for call in api_calls[:10]:
            print(f"    - {call}")
        print()

        # Test 3: Full agent test (requires API key)
        if os.getenv("OPENAI_API_KEY"):
            print("Test 3: Full agent validation")
            agent = create_api_validator()
            result = await Runner.run(
                agent,
                f"Validate this Blender Python code:\n\n```python\n{test_code}\n```",
                max_turns=5
            )
            print(f"  Response: {str(result.final_output)[:500]}...")
        else:
            print("Test 3: Skipped (OPENAI_API_KEY not set)")

        print("-" * 60)
        print("Tests complete!")

    asyncio.run(test())
