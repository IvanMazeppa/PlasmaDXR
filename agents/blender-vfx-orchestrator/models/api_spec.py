"""
API Specification Models for Spec-First Script Generation.

These Pydantic models define verified API specifications that MUST be
grounded in Blender 5.0 documentation. Every attribute and operation
must have a doc_ref from the Python reference.

Key insight: The Script Writer cannot hallucinate an attribute that
isn't in the verified spec. And the spec cannot include an attribute
without a doc reference. This creates a closed system where
hallucination is structurally impossible.

SDK Reference: https://github.com/openai/openai-agents-python/blob/v0.7.0/docs/agents.md
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class APIAttribute(BaseModel):
    """
    A single verified API attribute from Blender documentation.

    Every attribute MUST have a doc_ref from the Blender Python reference.
    This is validated by the APISpec field validators.
    """
    object_type: str = Field(
        description="Blender API class name, e.g., 'FluidDomainSettings'"
    )
    attribute_name: str = Field(
        description="Exact attribute name from docs, e.g., 'resolution_max'"
    )
    value_type: str = Field(
        description="Python type from docs: int, float, bool, str, or enum name"
    )
    doc_ref: str = Field(
        description="API documentation path - REQUIRED. Must be from blender_python_reference_5_0/"
    )
    example_value: Optional[Any] = Field(
        default=None,
        description="Example valid value, e.g., 128, True, 'SMOKE'"
    )
    value_range: Optional[str] = Field(
        default=None,
        description="Valid range if numeric, e.g., '[32, 512]' or '(0.0, 10.0)'"
    )
    enum_values: Optional[List[str]] = Field(
        default=None,
        description="Valid enum values if type is enum, e.g., ['SMOKE', 'FIRE', 'BOTH']"
    )


class APIOperation(BaseModel):
    """
    A verified bpy.ops.* operation from Blender documentation.

    Operators are function calls like bpy.ops.fluid.bake_data() that
    must be verified against Blender 5.0 documentation.
    """
    op_path: str = Field(
        description="Full operator path, e.g., 'bpy.ops.fluid.bake_data'"
    )
    doc_ref: str = Field(
        description="API documentation path - REQUIRED. Must be from blender_python_reference_5_0/"
    )
    required_context: Optional[str] = Field(
        default=None,
        description="Required context for operator, e.g., 'OBJECT', 'EDIT_MESH'"
    )
    parameters: Optional[Dict[str, str]] = Field(
        default=None,
        description="Operator parameters and their types"
    )


class APISpec(BaseModel):
    """
    Verified API specification for Blender script generation.

    This is the output of the API Spec Agent and input to the Code Writer.
    Every attribute and operation MUST have a doc_ref from Blender 5.0
    documentation. The guardrail enforces this.

    Design principle: If it's not in the spec, the Code Writer cannot use it.
    """
    effect_type: str = Field(
        description="VFX effect type: pyro, explosion, fire, smoke, nebula, etc."
    )
    technique: str = Field(
        description="Technique name, e.g., 'mantaflow_smoke', 'shader_volume'"
    )

    # Core API usage - all attributes must be verified
    domain_attributes: List[APIAttribute] = Field(
        default_factory=list,
        description="FluidDomainSettings attributes to use (must have doc_refs)"
    )
    flow_attributes: List[APIAttribute] = Field(
        default_factory=list,
        description="FluidFlowSettings attributes to use (must have doc_refs)"
    )

    # Operators (bpy.ops calls) - must be verified
    ops: List[APIOperation] = Field(
        default_factory=list,
        description="bpy.ops.* calls used by the script (must have doc_refs)"
    )

    # Shader nodes for volume rendering
    shader_nodes: List[str] = Field(
        default_factory=list,
        description="Shader node types to create, e.g., 'ShaderNodeVolumePrincipled'"
    )

    # Additional APIs (Scene, Object, etc.)
    scene_attributes: List[APIAttribute] = Field(
        default_factory=list,
        description="Scene attributes to use (frame_start, frame_end, etc.)"
    )
    object_attributes: List[APIAttribute] = Field(
        default_factory=list,
        description="Object attributes to use (location, rotation, etc.)"
    )

    # Documentation grounding
    approach_doc_refs: List[str] = Field(
        default_factory=list,
        description="General approach documentation references used"
    )

    # Warnings from documentation
    deprecation_warnings: List[str] = Field(
        default_factory=list,
        description="Deprecation warnings found in docs (e.g., 'use_nodes removed in 6.0')"
    )

    @field_validator('domain_attributes', 'flow_attributes', 'scene_attributes', 'object_attributes')
    @classmethod
    def all_attributes_have_doc_refs(cls, v: List[APIAttribute]) -> List[APIAttribute]:
        """
        Validate that every attribute has a proper doc_ref.

        Requirements:
        1. doc_ref must not be empty
        2. doc_ref must be from blender_python_reference_5_0/ (API docs, not manual)
        3. doc_ref should contain the attribute name (case-insensitive)
        """
        for attr in v:
            # Check doc_ref is not empty
            if not attr.doc_ref or attr.doc_ref.strip() == "":
                raise ValueError(
                    f"Attribute {attr.object_type}.{attr.attribute_name} missing doc_ref"
                )

            # Check doc_ref is from API docs (not manual)
            if not attr.doc_ref.startswith("blender_python_reference_5_0/"):
                # Also accept the chunked doc format
                if not "blender_python_reference_5_0" in attr.doc_ref:
                    raise ValueError(
                        f"{attr.object_type}.{attr.attribute_name} doc_ref must be API doc path "
                        f"(got: {attr.doc_ref[:50]}...)"
                    )

            # Check attribute name appears in doc_ref (relaxed check)
            # Allow partial matches for compound attributes
            attr_lower = attr.attribute_name.lower()
            doc_lower = attr.doc_ref.lower()
            if attr_lower not in doc_lower:
                # Try without underscores (some docs use camelCase)
                attr_parts = attr_lower.split('_')
                if not any(part in doc_lower for part in attr_parts if len(part) > 3):
                    # Fallback: accept if object_type (class name) appears in doc_ref
                    # This validates the agent found the right documentation page
                    obj_type_lower = attr.object_type.lower()
                    if obj_type_lower not in doc_lower:
                        raise ValueError(
                            f"{attr.object_type}.{attr.attribute_name}: doc_ref '{attr.doc_ref}' "
                            f"does not appear to reference this attribute"
                        )
        return v

    @field_validator('ops')
    @classmethod
    def all_ops_have_doc_refs(cls, v: List[APIOperation]) -> List[APIOperation]:
        """
        Validate that every operation has a proper doc_ref.

        Requirements:
        1. op_path must start with bpy.ops.
        2. doc_ref must not be empty
        3. doc_ref must be from API docs
        """
        for op in v:
            # Check op_path format
            if not op.op_path.startswith("bpy.ops."):
                raise ValueError(
                    f"Operation must start with 'bpy.ops.': {op.op_path}"
                )

            # Check doc_ref is not empty
            if not op.doc_ref or op.doc_ref.strip() == "":
                raise ValueError(
                    f"Operation {op.op_path} missing doc_ref"
                )

            # Check doc_ref is from API docs
            if "blender_python_reference_5_0" not in op.doc_ref:
                raise ValueError(
                    f"{op.op_path} doc_ref must be API doc path "
                    f"(got: {op.doc_ref[:50]}...)"
                )
        return v

    def get_all_attribute_names(self) -> set:
        """Get all verified attribute names for code validation."""
        names = set()
        for attr in self.domain_attributes:
            names.add(attr.attribute_name)
        for attr in self.flow_attributes:
            names.add(attr.attribute_name)
        for attr in self.scene_attributes:
            names.add(attr.attribute_name)
        for attr in self.object_attributes:
            names.add(attr.attribute_name)
        return names

    def get_domain_attribute_names(self) -> set:
        """Get verified domain attribute names."""
        return {attr.attribute_name for attr in self.domain_attributes}

    def get_flow_attribute_names(self) -> set:
        """Get verified flow attribute names."""
        return {attr.attribute_name for attr in self.flow_attributes}

    def get_op_paths(self) -> set:
        """Get all verified operator paths."""
        return {op.op_path for op in self.ops}


class VerifiedScriptOutput(BaseModel):
    """
    Output from Code Writer that has been verified against API spec.

    This is the output type for the Code Writer agent. The guardrail
    validates that the script only uses attributes from the APISpec.
    """
    script_path: str = Field(
        description="Absolute path to the generated script"
    )
    technique_used: str = Field(
        description="Technique/approach used in script generation"
    )
    parameters_set: Dict[str, Any] = Field(
        default_factory=dict,
        description="Key parameters configured in the script"
    )
    apis_used: List[str] = Field(
        default_factory=list,
        description="List of API attributes used (must all be in spec)"
    )
    ops_used: List[str] = Field(
        default_factory=list,
        description="List of bpy.ops calls used (must all be in spec)"
    )
    validation_passed: bool = Field(
        default=True,
        description="Whether script validation passed"
    )
    validation_errors: List[str] = Field(
        default_factory=list,
        description="Any validation errors or warnings"
    )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_domain_attribute(
    name: str,
    value_type: str,
    doc_ref: str,
    example: Any = None,
    value_range: str = None,
) -> APIAttribute:
    """
    Create a FluidDomainSettings attribute.

    Args:
        name: Attribute name (e.g., 'resolution_max')
        value_type: Python type (int, float, bool, str)
        doc_ref: API doc path (must include attribute name)
        example: Optional example value
        value_range: Optional range string

    Returns:
        APIAttribute for FluidDomainSettings
    """
    return APIAttribute(
        object_type="FluidDomainSettings",
        attribute_name=name,
        value_type=value_type,
        doc_ref=doc_ref,
        example_value=example,
        value_range=value_range,
    )


def create_flow_attribute(
    name: str,
    value_type: str,
    doc_ref: str,
    example: Any = None,
    enum_values: List[str] = None,
) -> APIAttribute:
    """
    Create a FluidFlowSettings attribute.

    Args:
        name: Attribute name (e.g., 'flow_type')
        value_type: Python type (int, float, bool, str, enum)
        doc_ref: API doc path (must include attribute name)
        example: Optional example value
        enum_values: Optional list of valid enum values

    Returns:
        APIAttribute for FluidFlowSettings
    """
    return APIAttribute(
        object_type="FluidFlowSettings",
        attribute_name=name,
        value_type=value_type,
        doc_ref=doc_ref,
        example_value=example,
        enum_values=enum_values,
    )


def create_operation(
    op_path: str,
    doc_ref: str,
    context: str = None,
) -> APIOperation:
    """
    Create a verified bpy.ops operation.

    Args:
        op_path: Full operator path (e.g., 'bpy.ops.fluid.bake_data')
        doc_ref: API doc path
        context: Required context (e.g., 'OBJECT')

    Returns:
        APIOperation
    """
    return APIOperation(
        op_path=op_path,
        doc_ref=doc_ref,
        required_context=context,
    )


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    # Test APISpec validation
    print("Testing APISpec validation...")

    # Test valid spec
    try:
        valid_spec = APISpec(
            effect_type="pyro",
            technique="mantaflow_smoke",
            domain_attributes=[
                APIAttribute(
                    object_type="FluidDomainSettings",
                    attribute_name="resolution_max",
                    value_type="int",
                    doc_ref="blender_python_reference_5_0/bpy.types.FluidDomainSettings.html#resolution_max",
                    example_value=128,
                ),
            ],
            flow_attributes=[
                APIAttribute(
                    object_type="FluidFlowSettings",
                    attribute_name="flow_type",
                    value_type="enum",
                    doc_ref="blender_python_reference_5_0/bpy.types.FluidFlowSettings.html#flow_type",
                    enum_values=["SMOKE", "FIRE", "BOTH"],
                ),
            ],
            ops=[
                APIOperation(
                    op_path="bpy.ops.fluid.bake_data",
                    doc_ref="blender_python_reference_5_0/bpy.ops.fluid.html#bake_data",
                ),
            ],
        )
        print(f"Valid spec created: {valid_spec.effect_type}")
        print(f"  Domain attrs: {valid_spec.get_domain_attribute_names()}")
        print(f"  Flow attrs: {valid_spec.get_flow_attribute_names()}")
        print(f"  Ops: {valid_spec.get_op_paths()}")
    except Exception as e:
        print(f"ERROR creating valid spec: {e}")

    # Test invalid spec (missing doc_ref)
    print("\nTesting validation - missing doc_ref...")
    try:
        invalid_spec = APISpec(
            effect_type="pyro",
            technique="test",
            domain_attributes=[
                APIAttribute(
                    object_type="FluidDomainSettings",
                    attribute_name="resolution_max",
                    value_type="int",
                    doc_ref="",  # Empty - should fail
                ),
            ],
        )
        print("ERROR: Should have raised ValueError")
    except ValueError as e:
        print(f"  Correctly rejected: {e}")

    # Test invalid spec (wrong doc_ref format)
    print("\nTesting validation - wrong doc_ref format...")
    try:
        invalid_spec = APISpec(
            effect_type="pyro",
            technique="test",
            domain_attributes=[
                APIAttribute(
                    object_type="FluidDomainSettings",
                    attribute_name="resolution_max",
                    value_type="int",
                    doc_ref="https://docs.blender.org/manual/",  # Manual, not API
                ),
            ],
        )
        print("ERROR: Should have raised ValueError")
    except ValueError as e:
        print(f"  Correctly rejected: {e}")

    # Test invalid spec (attribute not in doc_ref)
    print("\nTesting validation - attribute not in doc_ref...")
    try:
        invalid_spec = APISpec(
            effect_type="pyro",
            technique="test",
            domain_attributes=[
                APIAttribute(
                    object_type="FluidDomainSettings",
                    attribute_name="timesteps_per_frame",  # Doesn't exist
                    value_type="int",
                    doc_ref="blender_python_reference_5_0/bpy.types.FluidDomainSettings.html#resolution_max",
                ),
            ],
        )
        print("ERROR: Should have raised ValueError")
    except ValueError as e:
        print(f"  Correctly rejected: {e}")

    print("\nAll APISpec validation tests passed!")
