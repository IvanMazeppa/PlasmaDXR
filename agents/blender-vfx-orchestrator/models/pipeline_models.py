"""
Pipeline output models for agent-to-agent data passing.

These Pydantic models define structured outputs for each pipeline phase.
Extracted from orchestrator.py to break circular imports between
orchestrator and phase modules.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

from utils.script_sections import CANONICAL_SECTIONS  # noqa: F401 — re-export


# =============================================================================
# AGENT OUTPUT MODELS
# =============================================================================


class ResearchOutput(BaseModel):
    """Output from Research Agent - provides starting parameters for script generation.

    Phase 3: Structured schema for deterministic research outputs.
    All fields must be populated from actual tool results, not invented.
    """

    @model_validator(mode="before")
    @classmethod
    def normalize_field_names(cls, data: Any) -> Any:
        """Normalize title-case keys to snake_case.

        LLMs sometimes return keys like "Recommended Approach" instead of
        "recommended_approach" despite the JSON schema specifying snake_case.
        """
        if isinstance(data, dict):
            return {
                key.lower().replace(" ", "_").replace("-", "_"): value
                for key, value in data.items()
            }
        return data

    recommended_approach: str = Field(description="Best approach for the effect type (from documentation)")
    key_parameters: Dict[str, Any] = Field(default_factory=dict, description="Recommended parameter values (from patterns or docs)")
    api_modules: List[str] = Field(default_factory=list, description="Blender API modules to use (e.g., bpy.types.FluidDomainSettings)")
    code_patterns: List[Dict[str, Any]] = Field(default_factory=list, description="Proven patterns from library: [{pattern_id, issue, code_snippet}]")
    warnings: List[str] = Field(default_factory=list, description="Potential pitfalls to avoid (from knowledge base)")
    alternative_approaches: List[str] = Field(default_factory=list, description="Backup approaches if primary fails")
    doc_refs: List[str] = Field(default_factory=list, description="Blender 5.0 documentation references used (URLs or section names)")


class ScriptOutput(BaseModel):
    """Output from Script Writer - path to generated/modified script."""
    script_path: str = Field(description="Absolute path to the generated script")
    technique_used: str = Field(description="Technique/approach used in script generation")
    parameters_set: Dict[str, Any] = Field(default_factory=dict, description="Key parameters configured")
    validation_passed: bool = Field(default=True, description="Whether script validation passed")
    validation_errors: List[Any] = Field(default_factory=list, description="Any validation errors (strings or structured dicts)")
    sections_found: List[str] = Field(default_factory=list, description="Canonical section names found in the script")


class ExecutionOutput(BaseModel):
    """Output from Executor - render path and execution status."""
    success: bool = Field(description="Whether Blender execution succeeded")
    render_path: Optional[str] = Field(default=None, description="Path to rendered output (current iteration only)")
    partial_render_path: Optional[str] = Field(default=None, description="Render from failed execution (diagnostic only, not for scoring)")
    vdb_path: Optional[str] = Field(default=None, description="Path to VDB volume data")
    run_dir: Optional[str] = Field(default=None, description="Executor output directory containing cache and logs")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    execution_time_seconds: float = Field(default=0.0, description="Total execution time")


class QualityOutput(BaseModel):
    """Output from Quality Analyst - evaluation results with feedback."""
    overall_score: float = Field(description="Quality score 0-100")
    passed: bool = Field(description="Whether quality threshold was met")
    primary_issue: Optional[str] = Field(default=None, description="Most critical issue to fix")
    issues: List[str] = Field(default_factory=list, description="All identified issues")
    suggestions: List[str] = Field(default_factory=list, description="Specific improvement suggestions")
    vision_assessment: str = Field(default="", description="Detailed visual quality description")
    reference_similarity: Optional[float] = Field(default=None, description="Similarity to reference (if available)")


class LearningOutput(BaseModel):
    """Output from Learning Agent - experiment recorded, next action suggested."""
    experiment_recorded: bool = Field(default=True, description="Whether experiment was logged")
    pattern_extracted: bool = Field(default=False, description="Whether a new pattern was extracted")
    pattern_id: Optional[str] = Field(default=None, description="ID of extracted pattern")
    next_action: str = Field(description="Recommended next action: 'iterate', 'switch_technique', 'complete'")
    suggested_modifications: List[str] = Field(default_factory=list, description="Text descriptions of changes for context")
    parameter_modifications: Dict[str, Any] = Field(
        default_factory=dict,
        description="Concrete parameter changes as {param_name: new_value}, e.g., {'temperature': 3.0, 'density': 5.0}"
    )


# =============================================================================
# COORDINATOR DECISION MODELS (Phase 2: Agents-as-Tools Pattern)
# =============================================================================


class TechniqueDecision(BaseModel):
    """Output from Coordinator for initial technique selection."""
    selected_technique: str = Field(description="The technique to use (e.g., 'mantaflow_fire', 'shader_based_volume')")
    reasoning: str = Field(description="Why this technique was selected")
    key_parameters: Dict[str, Any] = Field(default_factory=dict, description="Starting parameters for the technique")
    alternative_techniques: List[str] = Field(default_factory=list, description="Backup techniques if primary fails")
    research_summary: str = Field(default="", description="Summary of research findings")


class ModificationDecision(BaseModel):
    """Output from Coordinator for parameter modification strategy."""
    action: str = Field(description="Action to take: 'modify_params', 'modify_code', 'switch_technique', 'continue'")
    parameter_changes: Dict[str, Any] = Field(default_factory=dict, description="Concrete parameter changes (for modify_params)")
    code_change_description: Optional[str] = Field(default=None, description="Description of structural code changes needed (for modify_code)")
    new_technique: Optional[str] = Field(default=None, description="New technique if switching")
    reasoning: str = Field(default="See code_change_description or parameter_changes", description="Why this modification strategy was chosen")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence in this decision (0.0-1.0)")


class QualityDecision(BaseModel):
    """Output from Coordinator for quality gate interpretation."""
    passed: bool = Field(description="Whether quality gate is passed")
    should_continue: bool = Field(description="Whether to continue iterating")
    next_action: str = Field(description="Next action: 'complete', 'iterate', 'switch_technique', 'request_guidance'")
    escape_level: int = Field(ge=0, le=4, description="Current escape velocity level (0-4)")
    reasoning: str = Field(description="Explanation of the quality decision")


# =============================================================================
# TECHNIQUE CONTRACT (Binding generation constraint)
# =============================================================================


class PhysicsSystem(str, Enum):
    """Blender physics systems available for VFX generation."""
    MANTAFLOW_GAS = "mantaflow_gas"
    MANTAFLOW_LIQUID = "mantaflow_liquid"
    RIGID_BODY = "rigid_body"
    PARTICLE_SYSTEM = "particle_system"
    CLOTH = "cloth"
    GEOMETRY_NODES = "geometry_nodes"
    SHADER_ONLY = "shader_only"


class RequiredOperator(BaseModel):
    """A Blender operator that MUST be called in the generated script."""
    operator: str = Field(description="Full operator path, e.g. 'bpy.ops.object.add_fracture_cell_objects'")
    purpose: str = Field(description="What this operator does in the workflow")
    context_requirements: Optional[str] = Field(
        default=None,
        description="Context setup needed before calling (e.g. 'object must be selected and active')"
    )


class RequiredAddon(BaseModel):
    """A Blender addon/extension that must be enabled before script execution."""
    module_name: str = Field(description="Addon module name for addon_utils.enable(), e.g. 'bl_ext.blender_org.cell_fracture'")
    enable_code: str = Field(description="Exact Python code to enable the addon")


class HeadlessConstraint(BaseModel):
    """A constraint for headless Blender execution."""
    description: str = Field(description="What must be avoided or handled differently")
    workaround: str = Field(description="How to handle this in headless mode")


class TechniqueContract(BaseModel):
    """Binding contract for script generation.

    This replaces advisory prose with structured constraints that the Script
    Writer MUST follow. The contract is produced by Phase 0.5 (technique
    selection) and consumed by Phase 1 (script generation).

    The Script Writer has creative freedom WITHIN the contract (scene
    composition, materials, lighting, camera) but CANNOT substitute a
    different technique, skip required operators, or ignore headless
    constraints.
    """
    technique_name: str = Field(
        description="Canonical technique identifier (e.g. 'cell_fracture_rigid_body', 'mantaflow_fire')"
    )
    physics_systems: List[PhysicsSystem] = Field(
        description="Which Blender physics systems this technique uses"
    )
    required_operators: List[RequiredOperator] = Field(
        default_factory=list,
        description="Operators the script MUST call (e.g. Cell Fracture, bake_all)"
    )
    required_addons: List[RequiredAddon] = Field(
        default_factory=list,
        description="Addons that must be enabled before execution"
    )
    headless_constraints: List[HeadlessConstraint] = Field(
        default_factory=list,
        description="Things that don't work in headless mode and their workarounds"
    )
    key_parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Starting parameter values for the technique"
    )
    code_scaffolding: Optional[str] = Field(
        default=None,
        description="Required code snippet that MUST appear in the script (e.g. addon enable boilerplate)"
    )
    forbidden_patterns: List[str] = Field(
        default_factory=list,
        description="Patterns the script must NOT use (e.g. 'manual mesh cutting' when Cell Fracture is required)"
    )
    reasoning: str = Field(
        default="",
        description="Why this technique was selected (for logging/debugging)"
    )
    alternative_techniques: List[str] = Field(
        default_factory=list,
        description="Fallback techniques if this one fails"
    )

    def to_script_constraints(self) -> str:
        """Render the contract as structured constraints for the Script Writer prompt.

        This produces a dense, LLM-optimized constraint block — NOT advisory prose.
        The format is designed so the Script Writer cannot misinterpret or ignore it.
        """
        lines = [
            "## TECHNIQUE CONTRACT (BINDING — DO NOT DEVIATE)",
            f"Technique: {self.technique_name}",
            f"Physics: {', '.join(p.value for p in self.physics_systems)}",
        ]

        if self.required_addons:
            lines.append("\n### REQUIRED ADDONS (must appear at top of script)")
            for addon in self.required_addons:
                lines.append(f"- {addon.module_name}: {addon.enable_code}")

        if self.required_operators:
            lines.append("\n### REQUIRED OPERATORS (script MUST call these)")
            for op in self.required_operators:
                lines.append(f"- {op.operator} — {op.purpose}")
                if op.context_requirements:
                    lines.append(f"  Context: {op.context_requirements}")

        if self.headless_constraints:
            lines.append("\n### HEADLESS CONSTRAINTS (will crash if violated)")
            for hc in self.headless_constraints:
                lines.append(f"- AVOID: {hc.description}")
                lines.append(f"  USE: {hc.workaround}")

        if self.forbidden_patterns:
            lines.append("\n### FORBIDDEN (do NOT use these)")
            for fp in self.forbidden_patterns:
                lines.append(f"- {fp}")

        if self.code_scaffolding:
            lines.append(f"\n### REQUIRED CODE (include verbatim)")
            lines.append(f"```python\n{self.code_scaffolding}\n```")

        if self.key_parameters:
            lines.append(f"\n### STARTING PARAMETERS")
            for k, v in self.key_parameters.items():
                lines.append(f"- {k}: {v}")

        return "\n".join(lines)

    def check_adherence(self, script_content: str) -> tuple[bool, List[str]]:
        """Check if a generated script adheres to this contract.

        Returns (passed, violations) where violations is a list of
        specific contract violations found in the script.
        """
        violations = []

        # Check required addons
        for addon in self.required_addons:
            if addon.module_name not in script_content:
                violations.append(
                    f"MISSING ADDON: '{addon.module_name}' not found in script. "
                    f"Required code: {addon.enable_code}"
                )

        # Check required operators
        for op in self.required_operators:
            # Extract the operator call (e.g. 'add_fracture_cell_objects' from
            # 'bpy.ops.object.add_fracture_cell_objects')
            op_parts = op.operator.split(".")
            op_call = op_parts[-1] if op_parts else op.operator
            if op_call not in script_content:
                violations.append(
                    f"MISSING OPERATOR: '{op.operator}' not called. "
                    f"Purpose: {op.purpose}"
                )

        # Check forbidden patterns
        for pattern in self.forbidden_patterns:
            pattern_lower = pattern.lower()
            # Check for common implementations of forbidden patterns
            if pattern_lower in script_content.lower():
                violations.append(
                    f"FORBIDDEN PATTERN: '{pattern}' found in script"
                )

        passed = len(violations) == 0
        return passed, violations


# =============================================================================
# CAPABILITY PACK REGISTRY (known technique adapters)
# =============================================================================


# Registry of known technique contracts that can be selected by the pipeline.
# This replaces the "teaching via prompt injection" approach with verified
# structured adapters. New techniques are added here, not in dynamic_instructions.
CAPABILITY_PACKS: Dict[str, "TechniqueContract"] = {}


def register_capability_pack(name: str, contract: "TechniqueContract") -> None:
    """Register a technique contract as a known capability pack."""
    CAPABILITY_PACKS[name] = contract


def get_capability_pack(name: str) -> Optional["TechniqueContract"]:
    """Look up a registered capability pack by name."""
    return CAPABILITY_PACKS.get(name)


def list_capability_packs() -> List[str]:
    """List all registered capability pack names."""
    return list(CAPABILITY_PACKS.keys())


def _register_builtin_packs() -> None:
    """Register built-in capability packs for known techniques."""

    # --- Cell Fracture + Rigid Body (destruction/shatter) ---
    register_capability_pack("cell_fracture_rigid_body", TechniqueContract(
        technique_name="cell_fracture_rigid_body",
        physics_systems=[PhysicsSystem.RIGID_BODY],
        required_addons=[
            RequiredAddon(
                module_name="bl_ext.blender_org.cell_fracture",
                enable_code="import addon_utils; addon_utils.enable('bl_ext.blender_org.cell_fracture', default_set=True, persistent=True)",
            ),
        ],
        required_operators=[
            RequiredOperator(
                operator="bpy.ops.object.add_fracture_cell_objects",
                purpose="Fracture the target mesh into cell-shaped pieces",
                context_requirements="Target object must be selected and active in OBJECT mode",
            ),
        ],
        headless_constraints=[
            HeadlessConstraint(
                description="bpy.ops.anim.keyframe_insert_by_name requires animation context",
                workaround="Use obj.keyframe_insert(data_path='rigid_body.enabled', frame=N) instead",
            ),
            HeadlessConstraint(
                description="bpy.ops.rigidbody.bake_to_keyframes requires specific context override",
                workaround="Use bpy.ops.ptcache.bake_all(bake=True) for rigid body simulation, or iterate frames manually",
            ),
        ],
        key_parameters={
            "cell_fracture_source": "{'PARTICLE_OWN'}",
            "cell_fracture_source_limit": 100,
            "cell_fracture_noise": 0.05,
            "rigid_body_steps_per_second": 120,
            "rigid_body_solver_iterations": 20,
        },
        code_scaffolding="""import addon_utils
addon_utils.enable('bl_ext.blender_org.cell_fracture', default_set=True, persistent=True)""",
        forbidden_patterns=[
            "manual mesh cutting with bmesh",
            "manual Voronoi tessellation",
            "custom fracture_mesh function",
            "fracture_glass() helper function",
        ],
        reasoning="Cell Fracture addon produces realistic fracture patterns; manual mesh cutting produces inferior results",
        alternative_techniques=["voronoi_fracture_geometry_nodes", "simple_rigid_body"],
    ))

    # --- Mantaflow Gas (fire/smoke/explosion) ---
    register_capability_pack("mantaflow_fire", TechniqueContract(
        technique_name="mantaflow_fire",
        physics_systems=[PhysicsSystem.MANTAFLOW_GAS],
        required_operators=[
            RequiredOperator(
                operator="bpy.ops.object.quick_smoke",
                purpose="Set up Mantaflow domain and flow objects quickly",
                context_requirements="Emitter object must be selected",
            ),
        ],
        headless_constraints=[
            HeadlessConstraint(
                description="Bake requires active domain object",
                workaround="bpy.context.view_layer.objects.active = domain_obj; bpy.ops.fluid.bake_all()",
            ),
        ],
        key_parameters={
            "domain_type": "GAS",
            "resolution_max": 128,
            "use_noise": True,
            "noise_strength": 1.0,
            "cache_type": "ALL",
        },
        reasoning="Mantaflow gas simulation is the standard approach for fire/smoke effects",
        alternative_techniques=["shader_based_fire", "mantaflow_high_res_fire"],
    ))

    # --- Mantaflow Liquid (water/pour/splash) ---
    register_capability_pack("mantaflow_liquid", TechniqueContract(
        technique_name="mantaflow_liquid",
        physics_systems=[PhysicsSystem.MANTAFLOW_LIQUID],
        required_operators=[
            RequiredOperator(
                operator="bpy.ops.object.quick_liquid",
                purpose="Set up Mantaflow liquid domain and flow",
                context_requirements="Emitter object must be selected",
            ),
        ],
        headless_constraints=[
            HeadlessConstraint(
                description="Liquid needs use_plane_init=True for non-empty bakes",
                workaround="Set flow.use_plane_init = True on all liquid flow objects",
            ),
            HeadlessConstraint(
                description="cache_type REPLAY doesn't produce full bake",
                workaround="Use cache_type = 'ALL' and bpy.ops.fluid.bake_all()",
            ),
        ],
        key_parameters={
            "domain_type": "LIQUID",
            "resolution_max": 128,
            "use_mesh": True,
            "use_plane_init": True,
            "cache_type": "ALL",
        },
        reasoning="Mantaflow liquid simulation for water/liquid effects",
        alternative_techniques=["mantaflow_high_res_liquid", "shader_based_water"],
    ))

    # --- Simple Rigid Body (no Cell Fracture — falling/stacking objects) ---
    register_capability_pack("simple_rigid_body", TechniqueContract(
        technique_name="simple_rigid_body",
        physics_systems=[PhysicsSystem.RIGID_BODY],
        required_operators=[
            RequiredOperator(
                operator="bpy.ops.rigidbody.object_add",
                purpose="Add rigid body physics to objects",
                context_requirements="Object must be selected and active",
            ),
        ],
        headless_constraints=[
            HeadlessConstraint(
                description="Rigid body world may not exist by default",
                workaround="Check bpy.context.scene.rigidbody_world; if None, add via bpy.ops.rigidbody.world_add()",
            ),
        ],
        key_parameters={
            "substeps_per_frame": 10,
            "solver_iterations": 10,
        },
        reasoning="Simple rigid body for scenes without fracturing (falling objects, dominos, etc.)",
        alternative_techniques=["cell_fracture_rigid_body"],
    ))

    # --- Particles Core (rain, snow, sparks, dust, embers) ---
    register_capability_pack("particles_core", TechniqueContract(
        technique_name="particles_core",
        physics_systems=[PhysicsSystem.PARTICLE_SYSTEM],
        required_operators=[
            RequiredOperator(
                operator="bpy.ops.object.particle_system_add",
                purpose="Add a particle system to the emitter object",
                context_requirements="Emitter object must be selected and active",
            ),
        ],
        headless_constraints=[
            HeadlessConstraint(
                description="Particle cache must be baked before rendering",
                workaround="Use bpy.ops.ptcache.bake_all(bake=True) after configuring particle settings",
            ),
            HeadlessConstraint(
                description="Particle instance objects need to exist before bake",
                workaround="Create instance geometry first, assign to particle settings.instance_object before baking",
            ),
        ],
        key_parameters={
            "particle_count": 1000,
            "frame_start": 1,
            "frame_end": 120,
            "lifetime": 50,
            "physics_type": "NEWTON",
            "emit_from": "FACE",
        },
        reasoning="Blender particle system for effects like rain, snow, sparks, dust, embers, confetti",
        alternative_techniques=["geometry_nodes_particles", "mantaflow_fire"],
    ))

    # --- Cloth / Soft Body (fabric, curtains, flags, soft deformation) ---
    register_capability_pack("cloth_softbody", TechniqueContract(
        technique_name="cloth_softbody",
        physics_systems=[PhysicsSystem.CLOTH],
        required_operators=[
            RequiredOperator(
                operator="bpy.ops.object.modifier_add",
                purpose="Add CLOTH modifier to the target mesh",
                context_requirements="Target mesh must be selected and active; call with type='CLOTH'",
            ),
        ],
        headless_constraints=[
            HeadlessConstraint(
                description="Cloth cache must be baked before rendering",
                workaround="Use bpy.ops.ptcache.bake_all(bake=True) after configuring cloth settings",
            ),
            HeadlessConstraint(
                description="Collision objects need COLLISION modifier for cloth interaction",
                workaround="Add bpy.ops.object.modifier_add(type='COLLISION') to all collision objects before baking",
            ),
        ],
        key_parameters={
            "quality": 5,
            "mass": 0.3,
            "air_damping": 1.0,
            "tension_stiffness": 15.0,
            "compression_stiffness": 15.0,
            "bending_stiffness": 0.5,
        },
        reasoning="Cloth simulation for fabric, curtains, flags, banners, tablecloths, soft body deformation",
        alternative_techniques=["softbody_modifier", "geometry_nodes_deformation"],
    ))

    # --- Geometry Nodes Environment (procedural landscapes, scatter, foliage) ---
    register_capability_pack("geometry_nodes_environment", TechniqueContract(
        technique_name="geometry_nodes_environment",
        physics_systems=[PhysicsSystem.GEOMETRY_NODES],
        required_operators=[
            RequiredOperator(
                operator="bpy.ops.object.modifier_add",
                purpose="Add NODES modifier (Geometry Nodes) to the target mesh",
                context_requirements="Target mesh must be selected; call with type='NODES'",
            ),
        ],
        headless_constraints=[
            HeadlessConstraint(
                description="Geometry Nodes modifiers must be applied or evaluated before render in some cases",
                workaround="Ensure the node tree is assigned: modifier.node_group = bpy.data.node_groups['GroupName']",
            ),
        ],
        key_parameters={
            "scatter_density": 10.0,
            "seed": 42,
            "scale_min": 0.8,
            "scale_max": 1.2,
        },
        code_scaffolding="""# Create geometry nodes modifier
mod = obj.modifiers.new(name="GeometryNodes", type='NODES')
# Create or assign a node group
node_group = bpy.data.node_groups.new(name="EnvironmentSetup", type='GeometryNodeTree')
mod.node_group = node_group""",
        reasoning="Geometry Nodes for procedural environments: terrain, foliage scatter, rocks, procedural cities, abstract shapes",
        alternative_techniques=["manual_mesh_placement", "particle_system_scatter"],
    ))


# Auto-register on import
_register_builtin_packs()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def compute_sica_utility(score: float, cost_usd: float, time_seconds: float) -> float:
    """
    SICA utility function for selecting best iteration.

    U = 0.5 * score_norm + 0.25 * (1 - cost/10) + 0.25 * (1 - time/300)
    score_norm is score / 100.
    """
    score_norm = max(0.0, min(1.0, score / 100.0))
    cost_component = 1.0 - min(1.0, max(0.0, cost_usd) / 10.0)
    time_component = 1.0 - min(1.0, max(0.0, time_seconds) / 300.0)
    utility = 0.5 * score_norm + 0.25 * cost_component + 0.25 * time_component
    return max(0.0, min(1.0, utility))
