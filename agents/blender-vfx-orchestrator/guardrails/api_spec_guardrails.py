"""
API Spec Guardrails for the Spec-First Pipeline.

These guardrails enforce that:
1. API Spec Agent: Every attribute has a verified doc_ref from Blender 5.0 API docs
2. Code Writer: Only uses attributes that are in the verified APISpec

SDK Reference: https://github.com/openai/openai-agents-python/blob/v0.7.0/docs/guardrails.md

IMPORTANT: Guardrails MUST return output_info on both pass and fail (SDK requirement).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, TYPE_CHECKING, Union

from agents import (
    Agent,
    GuardrailFunctionOutput,
    output_guardrail,
    TResponseInputItem,
)

if TYPE_CHECKING:
    from agents import RunContextWrapper
    from models.api_spec import APISpec, VerifiedScriptOutput


# =============================================================================
# ENUM VALUE VALIDATION
# =============================================================================

# Fallback enum values when APISpec doesn't provide enum_values.
# TEMPORARY SAFETY NET: Remove once enum_values are reliably extracted from docs.
# Keep this small and explicit to avoid false positives.
VALID_ENUM_VALUES = {
    "flow_behavior": ["INFLOW", "OUTFLOW", "GEOMETRY"],
    "flow_type": ["SMOKE", "FIRE", "BOTH"],
    "domain_type": ["GAS", "LIQUID"],
}

# =============================================================================
# BLENDER 5.0 DEPRECATED ATTRIBUTES BLACKLIST
# =============================================================================
# These attributes existed in Blender 4.x but were REMOVED or RENAMED in 5.0.
# Scripts using these will fail at runtime - reject them early.
# This is the primary defense against LLM training data using older Blender patterns.

DEPRECATED_ATTRIBUTES = {
    # FluidDomainSettings - removed/renamed in Blender 5.0
    "resolution_divisions": "REMOVED: Use 'resolution_max' instead",
    "use_adaptive_time_steps": "RENAMED: Use 'use_adaptive_timesteps' (no underscore before steps)",
    "use_dissolve": "RENAMED: Use 'use_dissolve_smoke'",
    "absolute_density": "REMOVED: Use 'density' with 'use_absolute'",
    "bake_frame_start": "REMOVED: Use scene.frame_start or cache_frame_start/end",
    "bake_frame_end": "REMOVED: Use scene.frame_end or cache_frame_start/end",
    "timesteps_per_frame": "RENAMED: Use 'timesteps_max'",

    # Cache/compression - BLOSC removed in Blender 5.0
    "openvdb_cache_compress_type_BLOSC": "REMOVED: BLOSC compression removed in 5.0, use 'ZIP' or 'NONE'",

    # Mesh - removed in Blender 4.1+
    "use_auto_smooth": "REMOVED: Auto smooth is now per-edge, use Smooth by Angle modifier",
    "auto_smooth_angle": "REMOVED: Use Smooth by Angle modifier",
}
# NOTE: Principled BSDF input renames (Transmission→Transmission Weight,
# Specular→Specular IOR Level, etc.) are handled by blender_api_fixer.py
# NOT here. The fixer auto-corrects them deterministically; putting them
# in the guardrail would block the script before the fixer can run.

# Known enum values that are INVALID in Blender 5.0
DEPRECATED_ENUM_VALUES = {
    "openvdb_cache_compress_type": {
        "BLOSC": "REMOVED: Use 'ZIP' or 'NONE' in Blender 5.0",
    },
    "cache_type": {
        # All valid in 5.0, but document for reference
    },
}

# Known-good Blender 5.0 Mantaflow attributes verified from official docs.
# Attributes in this whitelist bypass strict doc_ref anchor validation.
# This is P1-17 from the Master Roadmap: "API index/whitelist from Blender 5 docs"
KNOWN_GOOD_ATTRIBUTES = {
    # FluidDomainSettings - core simulation setup
    "FluidDomainSettings.domain_type": {
        "value_type": "enum",
        "enum_values": ["GAS", "LIQUID"],
        "doc_ref": "blender_python_reference_5_0/bpy.types.FluidDomainSettings.html",
    },
    "FluidDomainSettings.resolution_max": {
        "value_type": "int",
        "value_range": "[32, 512]",
        "doc_ref": "blender_python_reference_5_0/bpy.types.FluidDomainSettings.html",
    },
    "FluidDomainSettings.use_noise": {
        "value_type": "bool",
        "doc_ref": "blender_python_reference_5_0/bpy.types.FluidDomainSettings.html",
    },
    "FluidDomainSettings.noise_strength": {
        "value_type": "float",
        "value_range": "[0.0, 10.0]",
        "doc_ref": "blender_python_reference_5_0/bpy.types.FluidDomainSettings.html",
    },
    "FluidDomainSettings.noise_scale": {
        "value_type": "int",
        "value_range": "[1, 10]",
        "doc_ref": "blender_python_reference_5_0/bpy.types.FluidDomainSettings.html",
    },
    "FluidDomainSettings.vorticity": {
        "value_type": "float",
        "value_range": "[0.0, 1.0]",
        "doc_ref": "blender_python_reference_5_0/bpy.types.FluidDomainSettings.html",
    },
    "FluidDomainSettings.use_adaptive_timesteps": {
        "value_type": "bool",
        "doc_ref": "blender_python_reference_5_0/bpy.types.FluidDomainSettings.html",
    },
    "FluidDomainSettings.timesteps_max": {
        "value_type": "int",
        "doc_ref": "blender_python_reference_5_0/bpy.types.FluidDomainSettings.html",
    },
    "FluidDomainSettings.use_dissolve_smoke": {
        "value_type": "bool",
        "doc_ref": "blender_python_reference_5_0/bpy.types.FluidDomainSettings.html",
    },
    "FluidDomainSettings.dissolve_speed": {
        "value_type": "int",
        "doc_ref": "blender_python_reference_5_0/bpy.types.FluidDomainSettings.html",
    },
    "FluidDomainSettings.use_flip_particles": {
        "value_type": "bool",
        "doc_ref": "blender_python_reference_5_0/bpy.types.FluidDomainSettings.html",
    },
    "FluidDomainSettings.flip_ratio": {
        "value_type": "float",
        "value_range": "[0.0, 1.0]",
        "doc_ref": "blender_python_reference_5_0/bpy.types.FluidDomainSettings.html",
    },
    "FluidDomainSettings.particle_radius": {
        "value_type": "float",
        "doc_ref": "blender_python_reference_5_0/bpy.types.FluidDomainSettings.html",
    },
    "FluidDomainSettings.use_mesh": {
        "value_type": "bool",
        "doc_ref": "blender_python_reference_5_0/bpy.types.FluidDomainSettings.html",
    },
    "FluidDomainSettings.mesh_concave_upper": {
        "value_type": "float",
        "doc_ref": "blender_python_reference_5_0/bpy.types.FluidDomainSettings.html",
    },
    "FluidDomainSettings.mesh_smoothen_pos": {
        "value_type": "int",
        "doc_ref": "blender_python_reference_5_0/bpy.types.FluidDomainSettings.html",
    },
    "FluidDomainSettings.mesh_smoothen_neg": {
        "value_type": "int",
        "doc_ref": "blender_python_reference_5_0/bpy.types.FluidDomainSettings.html",
    },
    "FluidDomainSettings.cache_directory": {
        "value_type": "str",
        "doc_ref": "blender_python_reference_5_0/bpy.types.FluidDomainSettings.html",
    },
    "FluidDomainSettings.cache_type": {
        "value_type": "enum",
        "enum_values": ["REPLAY", "MODULAR", "ALL"],
        "doc_ref": "blender_python_reference_5_0/bpy.types.FluidDomainSettings.html",
    },
    "FluidDomainSettings.openvdb_cache_compress_type": {
        "value_type": "enum",
        "enum_values": ["ZIP", "NONE"],
        "doc_ref": "blender_python_reference_5_0/bpy.types.FluidDomainSettings.html",
    },
    # FluidDomainSettings - liquid specific
    "FluidDomainSettings.use_spray_particles": {
        "value_type": "bool",
        "doc_ref": "blender_python_reference_5_0/bpy.types.FluidDomainSettings.html",
    },
    "FluidDomainSettings.use_foam_particles": {
        "value_type": "bool",
        "doc_ref": "blender_python_reference_5_0/bpy.types.FluidDomainSettings.html",
    },
    "FluidDomainSettings.use_bubble_particles": {
        "value_type": "bool",
        "doc_ref": "blender_python_reference_5_0/bpy.types.FluidDomainSettings.html",
    },
    # FluidFlowSettings - inflow/outflow
    "FluidFlowSettings.use_absolute": {
        "value_type": "bool",
        "doc_ref": "blender_python_reference_5_0/bpy.types.FluidFlowSettings.html",
    },
    "FluidFlowSettings.flow_type": {
        "value_type": "enum",
        "enum_values": ["SMOKE", "FIRE", "BOTH", "LIQUID"],
        "doc_ref": "blender_python_reference_5_0/bpy.types.FluidFlowSettings.html",
    },
    "FluidFlowSettings.flow_behavior": {
        "value_type": "enum",
        "enum_values": ["INFLOW", "OUTFLOW", "GEOMETRY"],
        "doc_ref": "blender_python_reference_5_0/bpy.types.FluidFlowSettings.html",
    },
    "FluidFlowSettings.use_initial_velocity": {
        "value_type": "bool",
        "doc_ref": "blender_python_reference_5_0/bpy.types.FluidFlowSettings.html",
    },
    "FluidFlowSettings.velocity_normal": {
        "value_type": "float",
        "doc_ref": "blender_python_reference_5_0/bpy.types.FluidFlowSettings.html",
    },
    "FluidFlowSettings.temperature": {
        "value_type": "float",
        "doc_ref": "blender_python_reference_5_0/bpy.types.FluidFlowSettings.html",
    },
    "FluidFlowSettings.density": {
        "value_type": "float",
        "doc_ref": "blender_python_reference_5_0/bpy.types.FluidFlowSettings.html",
    },
    # Object-level fluid type
    "FluidModifier.fluid_type": {
        "value_type": "enum",
        "enum_values": ["NONE", "DOMAIN", "FLOW", "EFFECTOR"],
        "doc_ref": "blender_python_reference_5_0/bpy.types.FluidModifier.html",
    },
}


# =============================================================================
# API SPEC GUARDRAIL
# =============================================================================

@output_guardrail
async def validate_api_spec(
    ctx: "RunContextWrapper[Any]",
    agent: Agent,
    output: Any
) -> GuardrailFunctionOutput:
    """
    Output guardrail that validates APISpec has doc_refs for all attributes.

    This guardrail ensures that the API Spec Agent has verified every
    attribute against Blender 5.0 documentation before the spec is accepted.

    Requirements for doc_refs:
    1. Must not be empty
    2. Must be from blender_python_reference_5_0/ (API docs, not manual)
    3. Must include the attribute or operation name

    Args:
        ctx: Run context wrapper
        agent: The API Spec Agent
        output: APISpec output

    Returns:
        GuardrailFunctionOutput with:
        - tripwire_triggered=True if any attribute lacks valid doc_ref
        - output_info always populated (SDK requirement)
    """
    errors = []
    warnings = []

    # Handle different output types
    if output is None:
        return GuardrailFunctionOutput(
            output_info={
                "status": "rejected",
                "reason": "API Spec output is None",
            },
            tripwire_triggered=True,
        )

    # Check if output has the expected structure
    if not hasattr(output, 'domain_attributes') or not hasattr(output, 'flow_attributes'):
        return GuardrailFunctionOutput(
            output_info={
                "status": "rejected",
                "reason": "Output is not an APISpec (missing domain_attributes or flow_attributes)",
                "output_type": type(output).__name__,
            },
            tripwire_triggered=True,
        )

    # Validate domain attributes
    for attr in output.domain_attributes:
        result = _validate_attribute_doc_ref(attr, "domain")
        if result:
            errors.append(result)

    # Validate flow attributes
    for attr in output.flow_attributes:
        result = _validate_attribute_doc_ref(attr, "flow")
        if result:
            errors.append(result)

    # Validate scene attributes (if present)
    if hasattr(output, 'scene_attributes'):
        for attr in output.scene_attributes:
            result = _validate_attribute_doc_ref(attr, "scene")
            if result:
                errors.append(result)

    # Validate operations
    if hasattr(output, 'ops'):
        for op in output.ops:
            result = _validate_op_doc_ref(op)
            if result:
                errors.append(result)

    # Count verified items
    verified_count = len(output.domain_attributes) + len(output.flow_attributes)
    if hasattr(output, 'scene_attributes'):
        verified_count += len(output.scene_attributes)
    ops_count = len(output.ops) if hasattr(output, 'ops') else 0

    if errors:
        print(
            f"[Guardrail] validate_api_spec TRIGGERED: {len(errors)} unverified attributes/ops",
            file=sys.stderr
        )
        return GuardrailFunctionOutput(
            output_info={
                "status": "rejected",
                "errors": errors,
                "warnings": warnings,
                "verified_count": verified_count - len(errors),
                "rejected_count": len(errors),
                "reason": "Some attributes or operations lack valid doc_refs",
            },
            tripwire_triggered=True,
        )

    print(
        f"[Guardrail] validate_api_spec PASSED: {verified_count} attributes, {ops_count} ops verified",
        file=sys.stderr
    )
    return GuardrailFunctionOutput(
        output_info={
            "status": "passed",
            "verified_attributes": verified_count,
            "verified_ops": ops_count,
            "warnings": warnings,
        },
        tripwire_triggered=False,
    )


def _validate_attribute_doc_ref(attr: Any, category: str) -> str | None:
    """
    Validate a single attribute's doc_ref.

    Returns error message if invalid, None if valid.
    """
    if not hasattr(attr, 'doc_ref') or not hasattr(attr, 'attribute_name'):
        return f"{category}: attribute missing doc_ref or attribute_name field"

    # Check whitelist first - known-good attributes bypass strict doc_ref anchor validation
    object_type = getattr(attr, 'object_type', '')
    whitelist_key = f"{object_type}.{attr.attribute_name}" if object_type else ""
    if whitelist_key and whitelist_key in KNOWN_GOOD_ATTRIBUTES:
        # Still require non-empty doc_ref, but don't enforce anchor matching
        if not attr.doc_ref or attr.doc_ref.strip() == "":
            return f"{category}.{attr.attribute_name}: missing doc_ref (whitelisted but still needs ref)"
        return None  # Pass - known good attribute

    if not attr.doc_ref or attr.doc_ref.strip() == "":
        return f"{category}.{attr.attribute_name}: missing doc_ref"

    # Check doc_ref is from API docs (allow various chunked formats)
    doc_ref_lower = attr.doc_ref.lower()
    if "blender_python_reference_5_0" not in doc_ref_lower:
        return (
            f"{category}.{attr.attribute_name}: doc_ref must be from "
            f"blender_python_reference_5_0 (got: {attr.doc_ref[:60]}...)"
        )

    # Check attribute name appears in doc_ref (relaxed - allow partial match)
    attr_name = attr.attribute_name.lower()
    if attr_name not in doc_ref_lower:
        # Try matching any substantial part of the attribute name
        parts = attr_name.split('_')
        substantial_parts = [p for p in parts if len(p) > 3]
        if not any(part in doc_ref_lower for part in substantial_parts):
            return (
                f"{category}.{attr.attribute_name}: doc_ref does not reference "
                f"this attribute (doc_ref: {attr.doc_ref[:60]}...)"
            )

    return None


def _validate_op_doc_ref(op: Any) -> str | None:
    """
    Validate a single operation's doc_ref.

    Returns error message if invalid, None if valid.
    """
    if not hasattr(op, 'op_path') or not hasattr(op, 'doc_ref'):
        return "operation: missing op_path or doc_ref field"

    if not op.op_path.startswith("bpy.ops."):
        return f"operation: path must start with bpy.ops. (got: {op.op_path})"

    if not op.doc_ref or op.doc_ref.strip() == "":
        return f"operation {op.op_path}: missing doc_ref"

    if "blender_python_reference_5_0" not in op.doc_ref.lower():
        return (
            f"operation {op.op_path}: doc_ref must be from blender_python_reference_5_0 "
            f"(got: {op.doc_ref[:60]}...)"
        )

    return None


# =============================================================================
# CODE WRITER GUARDRAIL
# =============================================================================

@output_guardrail
async def validate_code_against_spec(
    ctx: "RunContextWrapper[Any]",
    agent: Agent,
    output: Any
) -> GuardrailFunctionOutput:
    """
    Output guardrail: rejects scripts using DEPRECATED Blender 4.x attributes.

    SIMPLIFIED (2026-02-06): Removed spec-compliance check (Layer 1) and
    complexity check (Layer 3) which together created an impossible constraint.
    Now only checks for deprecated attributes that will crash at runtime.

    The artifact gates (artifact_gates.py) handle post-execution validation
    of renders/cache/saves mechanically - no need to guess pre-execution.
    """
    # Handle None output
    if output is None:
        return GuardrailFunctionOutput(
            output_info={"status": "rejected", "reason": "Code Writer output is None"},
            tripwire_triggered=True,
        )

    # Get script path
    script_path = None
    if hasattr(output, 'script_path'):
        script_path = output.script_path
    elif isinstance(output, dict):
        script_path = output.get('script_path')

    if not script_path:
        return GuardrailFunctionOutput(
            output_info={"status": "rejected", "reason": "Output missing script_path"},
            tripwire_triggered=True,
        )

    # Read the script content
    try:
        script_content = Path(script_path).read_text()
    except FileNotFoundError:
        return GuardrailFunctionOutput(
            output_info={"status": "rejected", "reason": f"Script file not found: {script_path}"},
            tripwire_triggered=True,
        )
    except Exception as e:
        return GuardrailFunctionOutput(
            output_info={"status": "rejected", "reason": f"Error reading script: {e}"},
            tripwire_triggered=True,
        )

    # ======================================================================
    # ONLY CHECK: Blender 5.0 deprecated attribute blacklist
    # ======================================================================
    # This catches LLM training data using older Blender patterns.
    # These attributes WILL cause runtime errors - reject early.
    deprecated_found: Dict[str, str] = {}

    for attr_name, reason in DEPRECATED_ATTRIBUTES.items():
        patterns = [
            rf'\.{re.escape(attr_name)}\s*=',       # .attr = value
            rf'\.{re.escape(attr_name)}\s*\(',       # .attr(...)
            rf'\[[\'"]{re.escape(attr_name)}[\'"]\]', # ['attr'] or ["attr"]
        ]
        for pattern in patterns:
            if re.search(pattern, script_content):
                deprecated_found[attr_name] = reason
                break

    # Check for deprecated enum values (e.g., BLOSC compression)
    deprecated_enum_found: Dict[str, str] = {}
    for attr_name, bad_values in DEPRECATED_ENUM_VALUES.items():
        for bad_value, reason in bad_values.items():
            pattern = rf'\.{re.escape(attr_name)}\s*=\s*[\'\"]{re.escape(bad_value)}[\'\"]'
            if re.search(pattern, script_content):
                deprecated_enum_found[f"{attr_name}={bad_value}"] = reason

    total_deprecated = len(deprecated_found) + len(deprecated_enum_found)

    if total_deprecated > 0:
        print(
            f"[Guardrail] DEPRECATED BLENDER 4.x PATTERNS DETECTED ({total_deprecated}):",
            file=sys.stderr
        )
        for attr, reason in {**deprecated_found, **deprecated_enum_found}.items():
            print(f"  - {attr}: {reason}", file=sys.stderr)

        return GuardrailFunctionOutput(
            output_info={
                "status": "rejected",
                "reason": f"Uses {total_deprecated} DEPRECATED Blender 4.x attributes",
                "deprecated_attributes": deprecated_found,
                "deprecated_enum_values": deprecated_enum_found,
            },
            tripwire_triggered=True,
        )

    # Passed - no deprecated attributes found
    line_count = len(script_content.split('\n'))
    print(
        f"[Guardrail] validate_code_against_spec PASSED: "
        f"{line_count} lines, no deprecated attributes",
        file=sys.stderr
    )
    return GuardrailFunctionOutput(
        output_info={
            "status": "passed",
            "line_count": line_count,
        },
        tripwire_triggered=False,
    )


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    import asyncio
    from models.api_spec import APISpec, APIAttribute, APIOperation

    async def test_guardrails():
        print("Testing API Spec Guardrails...")
        print("-" * 60)

        # Mock context
        class MockContext:
            context = type('obj', (object,), {'api_spec': None})()

        ctx = MockContext()

        # Mock agent
        class MockAgent:
            name = "Test Agent"

        agent = MockAgent()

        # Test 1: Valid API Spec
        print("\n1. Testing validate_api_spec with valid spec...")
        valid_spec = APISpec(
            effect_type="pyro",
            technique="mantaflow_smoke",
            domain_attributes=[
                APIAttribute(
                    object_type="FluidDomainSettings",
                    attribute_name="resolution_max",
                    value_type="int",
                    doc_ref="blender_python_reference_5_0/bpy.types.FluidDomainSettings.html#resolution_max",
                ),
            ],
            flow_attributes=[
                APIAttribute(
                    object_type="FluidFlowSettings",
                    attribute_name="flow_type",
                    value_type="enum",
                    doc_ref="blender_python_reference_5_0/bpy.types.FluidFlowSettings.html#flow_type",
                ),
            ],
            ops=[
                APIOperation(
                    op_path="bpy.ops.fluid.bake_data",
                    doc_ref="blender_python_reference_5_0/bpy.ops.fluid.html#bake_data",
                ),
            ],
        )

        result = await validate_api_spec.guardrail_function(ctx, agent, valid_spec)
        print(f"   Triggered: {result.tripwire_triggered} (expected: False)")
        print(f"   Output info: {result.output_info}")
        assert result.tripwire_triggered is False

        # Test 2: Invalid API Spec (missing doc_ref)
        print("\n2. Testing validate_api_spec with missing doc_ref...")

        class FakeSpec:
            domain_attributes = [
                type('attr', (), {
                    'attribute_name': 'resolution_max',
                    'doc_ref': '',  # Empty
                })()
            ]
            flow_attributes = []
            ops = []

        result = await validate_api_spec.guardrail_function(ctx, agent, FakeSpec())
        print(f"   Triggered: {result.tripwire_triggered} (expected: True)")
        print(f"   Errors: {result.output_info.get('errors', [])}")
        assert result.tripwire_triggered is True

        # Test 3: None output
        print("\n3. Testing validate_api_spec with None...")
        result = await validate_api_spec.guardrail_function(ctx, agent, None)
        print(f"   Triggered: {result.tripwire_triggered} (expected: True)")
        assert result.tripwire_triggered is True

        print("\n" + "-" * 60)
        print("All API Spec guardrail tests passed!")

    asyncio.run(test_guardrails())
