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
    "FluidDomainSettings.timesteps_maximum": {
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
        "enum_values": ["ZIP", "BLOSC", "NONE"],
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
    Output guardrail that validates code only uses APIs from the verified spec.

    This guardrail parses the generated script and checks that:
    1. All dset.* assignments use attributes from APISpec.domain_attributes
    2. All fset.* assignments use attributes from APISpec.flow_attributes
    3. All bpy.ops.* calls use operations from APISpec.ops

    The Code Writer MUST use these variable naming conventions:
    - dset = mod.domain_settings (FluidDomainSettings)
    - fset = mod.flow_settings (FluidFlowSettings)

    Args:
        ctx: Run context wrapper (must have ctx.context.api_spec)
        agent: The Code Writer Agent
        output: VerifiedScriptOutput

    Returns:
        GuardrailFunctionOutput with:
        - tripwire_triggered=True if code uses unverified APIs
        - output_info always populated (SDK requirement)
    """
    # Handle None output
    if output is None:
        return GuardrailFunctionOutput(
            output_info={
                "status": "rejected",
                "reason": "Code Writer output is None",
            },
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
            output_info={
                "status": "rejected",
                "reason": "Output missing script_path",
            },
            tripwire_triggered=True,
        )

    # Read the script content
    try:
        script_content = Path(script_path).read_text()
    except FileNotFoundError:
        return GuardrailFunctionOutput(
            output_info={
                "status": "rejected",
                "reason": f"Script file not found: {script_path}",
            },
            tripwire_triggered=True,
        )
    except Exception as e:
        return GuardrailFunctionOutput(
            output_info={
                "status": "rejected",
                "reason": f"Error reading script: {e}",
            },
            tripwire_triggered=True,
        )

    # Get API spec from context
    api_spec = None
    if hasattr(ctx, 'context') and hasattr(ctx.context, 'api_spec'):
        api_spec = ctx.context.api_spec

    if api_spec is None:
        # If no API spec in context, we can't validate - pass with warning
        print(
            "[Guardrail] validate_code_against_spec: No API spec in context, skipping validation",
            file=sys.stderr
        )
        return GuardrailFunctionOutput(
            output_info={
                "status": "passed_no_spec",
                "warning": "No API spec in context - code not validated against spec",
            },
            tripwire_triggered=False,
        )

    # Extract attribute assignments from code
    # Patterns for dset.* and fset.* assignments (the enforced variable names)
    domain_pattern = r'\bdset\.(\w+)\s*='
    flow_pattern = r'\bfset\.(\w+)\s*='

    # Also check for common variations (domain_settings, flow_settings)
    alt_domain_pattern = r'\bdomain_settings\.(\w+)\s*='
    alt_flow_pattern = r'\bflow_settings\.(\w+)\s*='

    used_domain_attrs = set(re.findall(domain_pattern, script_content))
    used_domain_attrs.update(re.findall(alt_domain_pattern, script_content))

    used_flow_attrs = set(re.findall(flow_pattern, script_content))
    used_flow_attrs.update(re.findall(alt_flow_pattern, script_content))

    # Extract bpy.ops calls
    ops_pattern = r'\b(bpy\.ops\.\w+\.\w+)'
    used_ops = set(re.findall(ops_pattern, script_content))

    # Build allowed sets from API spec
    allowed_domain = set()
    allowed_flow = set()
    allowed_ops = set()

    if hasattr(api_spec, 'domain_attributes'):
        for attr in api_spec.domain_attributes:
            allowed_domain.add(attr.attribute_name)

    if hasattr(api_spec, 'flow_attributes'):
        for attr in api_spec.flow_attributes:
            allowed_flow.add(attr.attribute_name)

    if hasattr(api_spec, 'ops'):
        for op in api_spec.ops:
            allowed_ops.add(op.op_path)

    # Build enum validation map (attribute_name -> allowed values)
    enum_allowed: Dict[str, Set[str]] = {}
    spec_attrs = []
    if hasattr(api_spec, 'domain_attributes'):
        spec_attrs.extend(api_spec.domain_attributes)
    if hasattr(api_spec, 'flow_attributes'):
        spec_attrs.extend(api_spec.flow_attributes)
    if hasattr(api_spec, 'scene_attributes'):
        spec_attrs.extend(api_spec.scene_attributes)
    if hasattr(api_spec, 'object_attributes'):
        spec_attrs.extend(api_spec.object_attributes)

    for attr in spec_attrs:
        value_type = str(getattr(attr, "value_type", "")).lower()
        if value_type == "enum":
            values = []
            enum_values = getattr(attr, "enum_values", None)
            if enum_values:
                values = list(enum_values)
            elif attr.attribute_name in VALID_ENUM_VALUES:
                values = list(VALID_ENUM_VALUES[attr.attribute_name])
            if values:
                enum_allowed[attr.attribute_name] = set(values)

    # Extract enum assignments (string literals only)
    enum_assign_pattern = r'\b(?:dset|fset|domain_settings|flow_settings|scene)\.(\w+)\s*=\s*(["\'])([^"\']+)\2'
    enum_assignments: Dict[str, str] = {}
    for attr_name, _quote, value in re.findall(enum_assign_pattern, script_content):
        enum_assignments[attr_name] = value

    # Find violations
    domain_violations = used_domain_attrs - allowed_domain
    flow_violations = used_flow_attrs - allowed_flow
    op_violations = used_ops - allowed_ops

    # Enum value validation (reject unknown or missing enum values)
    enum_value_violations: Set[str] = set()
    enum_value_missing: Set[str] = set()
    for attr_name, allowed_values in enum_allowed.items():
        if attr_name in used_domain_attrs or attr_name in used_flow_attrs:
            if attr_name not in enum_assignments:
                enum_value_missing.add(attr_name)
                continue
            value = enum_assignments[attr_name]
            if value not in allowed_values:
                enum_value_violations.add(f"{attr_name}={value}")

    # Also allow common safe ops that don't need to be in spec
    safe_ops = {
        # Object operations
        "bpy.ops.object.select_all",
        "bpy.ops.object.delete",
        "bpy.ops.object.modifier_add",
        "bpy.ops.object.mode_set",
        "bpy.ops.object.origin_set",
        "bpy.ops.object.shade_smooth",
        "bpy.ops.object.shade_flat",
        "bpy.ops.object.transform_apply",
        # Mesh primitives
        "bpy.ops.mesh.primitive_cube_add",
        "bpy.ops.mesh.primitive_cylinder_add",
        "bpy.ops.mesh.primitive_sphere_add",
        "bpy.ops.mesh.primitive_ico_sphere_add",
        "bpy.ops.mesh.primitive_uv_sphere_add",
        "bpy.ops.mesh.primitive_plane_add",
        # File operations
        "bpy.ops.render.render",
        "bpy.ops.wm.save_as_mainfile",
        "bpy.ops.wm.open_mainfile",
        # Fluid/physics (common setup ops)
        "bpy.ops.ptcache.bake_all",
        "bpy.ops.ptcache.free_bake_all",
    }
    op_violations = op_violations - safe_ops

    # Verbose logging for debugging
    if op_violations:
        print(f"[Guardrail DEBUG] Rejected ops: {op_violations}", file=sys.stderr)
        print(f"[Guardrail DEBUG] Allowed spec ops: {allowed_ops}", file=sys.stderr)

    total_violations = (
        len(domain_violations)
        + len(flow_violations)
        + len(op_violations)
        + len(enum_value_violations)
        + len(enum_value_missing)
    )

    if total_violations > 0:
        print(
            f"[Guardrail] validate_code_against_spec TRIGGERED: "
            f"{len(domain_violations)} domain, {len(flow_violations)} flow, "
            f"{len(op_violations)} ops, {len(enum_value_violations)} enum value, "
            f"{len(enum_value_missing)} enum missing violations",
            file=sys.stderr
        )
        return GuardrailFunctionOutput(
            output_info={
                "status": "rejected",
                "reason": "Code uses APIs not in verified spec",
                "domain_violations": list(domain_violations),
                "flow_violations": list(flow_violations),
                "op_violations": list(op_violations),
                "enum_value_violations": list(enum_value_violations),
                "enum_value_missing": list(enum_value_missing),
                "allowed_enum_values": {k: sorted(list(v)) for k, v in enum_allowed.items()},
                "allowed_domain": list(allowed_domain),
                "allowed_flow": list(allowed_flow),
                "allowed_ops": list(allowed_ops),
            },
            tripwire_triggered=True,
        )

    # All good
    verified_count = len(used_domain_attrs) + len(used_flow_attrs)
    print(
        f"[Guardrail] validate_code_against_spec PASSED: "
        f"{verified_count} attributes, {len(used_ops)} ops verified",
        file=sys.stderr
    )
    return GuardrailFunctionOutput(
        output_info={
            "status": "passed",
            "verified_domain_attrs": list(used_domain_attrs),
            "verified_flow_attrs": list(used_flow_attrs),
            "verified_ops": list(used_ops),
            "verified_enum_values": {
                k: enum_assignments.get(k) for k in enum_allowed.keys()
                if k in enum_assignments
            },
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
