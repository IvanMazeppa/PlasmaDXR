"""Tests for TechniqueContract — binding generation constraints.

Verifies:
- Contract creation and serialization
- Capability pack registry
- Script constraint rendering
- Adherence checking (pass/fail)
- Fuzzy matching from TechniqueDecision to capability pack
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from models.pipeline_models import (
    TechniqueContract,
    TechniqueDecision,
    PhysicsSystem,
    RequiredOperator,
    RequiredAddon,
    HeadlessConstraint,
    CAPABILITY_PACKS,
    get_capability_pack,
    list_capability_packs,
    register_capability_pack,
)


class TestCapabilityPackRegistry:
    """Test the built-in capability pack registry."""

    def test_builtin_packs_registered(self):
        packs = list_capability_packs()
        assert "cell_fracture_rigid_body" in packs
        assert "mantaflow_fire" in packs
        assert "mantaflow_liquid" in packs
        assert "simple_rigid_body" in packs
        assert "particles_core" in packs
        assert "cloth_softbody" in packs
        assert "geometry_nodes_environment" in packs

    def test_mvp_pack_count(self):
        """The MVP target is 7 packs (Wave 1 exit criteria)."""
        packs = list_capability_packs()
        assert len(packs) >= 7, f"MVP requires 7+ packs, got {len(packs)}: {packs}"

    def test_get_cell_fracture_pack(self):
        pack = get_capability_pack("cell_fracture_rigid_body")
        assert pack is not None
        assert pack.technique_name == "cell_fracture_rigid_body"
        assert PhysicsSystem.RIGID_BODY in pack.physics_systems
        assert len(pack.required_addons) == 1
        assert pack.required_addons[0].module_name == "bl_ext.blender_org.cell_fracture"
        assert len(pack.required_operators) >= 1
        assert any("add_fracture_cell_objects" in op.operator for op in pack.required_operators)
        assert len(pack.headless_constraints) >= 1
        assert len(pack.forbidden_patterns) >= 1

    def test_get_mantaflow_fire_pack(self):
        pack = get_capability_pack("mantaflow_fire")
        assert pack is not None
        assert PhysicsSystem.MANTAFLOW_GAS in pack.physics_systems
        assert pack.key_parameters.get("domain_type") == "GAS"

    def test_get_particles_core_pack(self):
        pack = get_capability_pack("particles_core")
        assert pack is not None
        assert PhysicsSystem.PARTICLE_SYSTEM in pack.physics_systems
        assert any("particle_system_add" in op.operator for op in pack.required_operators)
        assert len(pack.headless_constraints) >= 1
        assert pack.key_parameters.get("physics_type") == "NEWTON"

    def test_get_cloth_softbody_pack(self):
        pack = get_capability_pack("cloth_softbody")
        assert pack is not None
        assert PhysicsSystem.CLOTH in pack.physics_systems
        assert any("modifier_add" in op.operator for op in pack.required_operators)
        assert len(pack.headless_constraints) >= 2
        assert "quality" in pack.key_parameters

    def test_get_geometry_nodes_environment_pack(self):
        pack = get_capability_pack("geometry_nodes_environment")
        assert pack is not None
        assert PhysicsSystem.GEOMETRY_NODES in pack.physics_systems
        assert any("modifier_add" in op.operator for op in pack.required_operators)
        assert pack.code_scaffolding is not None
        assert "GeometryNodeTree" in pack.code_scaffolding

    def test_get_nonexistent_pack(self):
        pack = get_capability_pack("nonexistent_technique")
        assert pack is None

    def test_register_custom_pack(self):
        custom = TechniqueContract(
            technique_name="test_custom",
            physics_systems=[PhysicsSystem.SHADER_ONLY],
            reasoning="test",
        )
        register_capability_pack("test_custom", custom)
        assert get_capability_pack("test_custom") is not None
        # Cleanup
        del CAPABILITY_PACKS["test_custom"]


class TestTechniqueContractModel:
    """Test the TechniqueContract Pydantic model."""

    def test_minimal_contract(self):
        contract = TechniqueContract(
            technique_name="test",
            physics_systems=[PhysicsSystem.MANTAFLOW_GAS],
        )
        assert contract.technique_name == "test"
        assert len(contract.required_operators) == 0
        assert len(contract.required_addons) == 0

    def test_full_contract_serialization(self):
        pack = get_capability_pack("cell_fracture_rigid_body")
        data = pack.model_dump()
        restored = TechniqueContract(**data)
        assert restored.technique_name == pack.technique_name
        assert len(restored.required_operators) == len(pack.required_operators)
        assert len(restored.required_addons) == len(pack.required_addons)
        assert len(restored.headless_constraints) == len(pack.headless_constraints)

    def test_physics_system_enum(self):
        for ps in PhysicsSystem:
            assert isinstance(ps.value, str)


class TestScriptConstraintRendering:
    """Test to_script_constraints() output."""

    def test_cell_fracture_constraints(self):
        pack = get_capability_pack("cell_fracture_rigid_body")
        text = pack.to_script_constraints()
        assert "TECHNIQUE CONTRACT (BINDING" in text
        assert "cell_fracture_rigid_body" in text
        assert "REQUIRED ADDONS" in text
        assert "bl_ext.blender_org.cell_fracture" in text
        assert "REQUIRED OPERATORS" in text
        assert "add_fracture_cell_objects" in text
        assert "HEADLESS CONSTRAINTS" in text
        assert "FORBIDDEN" in text
        assert "manual mesh cutting" in text.lower() or "manual Voronoi" in text

    def test_mantaflow_constraints(self):
        pack = get_capability_pack("mantaflow_fire")
        text = pack.to_script_constraints()
        assert "TECHNIQUE CONTRACT (BINDING" in text
        assert "mantaflow_gas" in text
        assert "STARTING PARAMETERS" in text

    def test_constraints_are_dense(self):
        """Contract output should be structured, not verbose prose."""
        pack = get_capability_pack("cell_fracture_rigid_body")
        text = pack.to_script_constraints()
        lines = text.strip().split("\n")
        # Should be reasonably compact (under 40 lines for the full cell fracture pack)
        assert len(lines) < 40, f"Contract too verbose: {len(lines)} lines"


class TestAdherenceChecking:
    """Test check_adherence() against real-ish scripts."""

    GOOD_CELL_FRACTURE_SCRIPT = """
import bpy
import addon_utils
addon_utils.enable('bl_ext.blender_org.cell_fracture', default_set=True, persistent=True)

# Create glass object
bpy.ops.mesh.primitive_cube_add(size=1)
glass = bpy.context.active_object
glass.name = "Glass"

# Fracture it
bpy.ops.object.select_all(action='DESELECT')
glass.select_set(True)
bpy.context.view_layer.objects.active = glass
bpy.ops.object.add_fracture_cell_objects(
    source={'PARTICLE_OWN'},
    source_limit=100,
    noise=0.05,
)

# Setup rigid body
for obj in bpy.data.objects:
    if obj.name.startswith("Glass_cell"):
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.rigidbody.object_add(type='ACTIVE')
"""

    BAD_MANUAL_FRACTURE_SCRIPT = """
import bpy
import bmesh

def fracture_glass(obj, num_pieces=50):
    # Manual Voronoi tessellation to split mesh
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    # ... manual mesh cutting with bmesh ...
    for i in range(num_pieces):
        # Create fragment manually
        bpy.ops.mesh.primitive_cube_add(size=0.1)
        fragment = bpy.context.active_object
        fragment.name = f"Fragment_{i}"

# Create glass and fracture manually
bpy.ops.mesh.primitive_cube_add(size=1)
glass = bpy.context.active_object
fracture_glass(glass)
"""

    MISSING_ADDON_SCRIPT = """
import bpy

# Uses Cell Fracture operator but forgot to enable addon
bpy.ops.object.select_all(action='DESELECT')
bpy.ops.object.add_fracture_cell_objects(
    source={'PARTICLE_OWN'},
    source_limit=50,
)
"""

    def test_good_script_passes(self):
        pack = get_capability_pack("cell_fracture_rigid_body")
        passed, violations = pack.check_adherence(self.GOOD_CELL_FRACTURE_SCRIPT)
        assert passed, f"Good script should pass, got violations: {violations}"
        assert len(violations) == 0

    def test_manual_fracture_fails(self):
        pack = get_capability_pack("cell_fracture_rigid_body")
        passed, violations = pack.check_adherence(self.BAD_MANUAL_FRACTURE_SCRIPT)
        assert not passed
        # Should detect missing addon
        addon_violations = [v for v in violations if "ADDON" in v]
        assert len(addon_violations) >= 1, f"Should detect missing addon: {violations}"
        # Should detect missing operator
        op_violations = [v for v in violations if "OPERATOR" in v]
        assert len(op_violations) >= 1, f"Should detect missing operator: {violations}"

    def test_missing_addon_fails(self):
        pack = get_capability_pack("cell_fracture_rigid_body")
        passed, violations = pack.check_adherence(self.MISSING_ADDON_SCRIPT)
        assert not passed
        addon_violations = [v for v in violations if "ADDON" in v]
        assert len(addon_violations) >= 1

    def test_mantaflow_adherence(self):
        pack = get_capability_pack("mantaflow_fire")
        good_script = """
import bpy
bpy.ops.mesh.primitive_cube_add(size=2)
emitter = bpy.context.active_object
emitter.select_set(True)
bpy.ops.object.quick_smoke(style='FIRE')
domain = bpy.data.objects['Smoke Domain']
"""
        passed, violations = pack.check_adherence(good_script)
        assert passed, f"Mantaflow script should pass: {violations}"


class TestBuildTechniqueContract:
    """Test _build_technique_contract from phases/research.py."""

    def test_exact_match(self):
        from phases.research import _build_technique_contract
        decision = TechniqueDecision(
            selected_technique="cell_fracture_rigid_body",
            reasoning="Best for glass shatter",
            key_parameters={"custom_param": 42},
        )
        contract = _build_technique_contract(decision, None)
        assert contract is not None
        assert contract.technique_name == "cell_fracture_rigid_body"
        # Decision params should be merged
        assert contract.key_parameters.get("custom_param") == 42
        # Pack defaults should be preserved
        assert "cell_fracture_source" in contract.key_parameters

    def test_fuzzy_match(self):
        from phases.research import _build_technique_contract
        decision = TechniqueDecision(
            selected_technique="cell_fracture_constraint_breaking",
            reasoning="Glass shatter with constraints",
        )
        contract = _build_technique_contract(decision, None)
        assert contract is not None
        assert contract.technique_name == "cell_fracture_rigid_body"

    def test_keyword_match_shatter(self):
        from phases.research import _build_technique_contract
        decision = TechniqueDecision(
            selected_technique="constraint_breaking_glass_shatter",
            reasoning="Shatter glass",
        )
        contract = _build_technique_contract(decision, None)
        assert contract is not None
        assert contract.technique_name == "cell_fracture_rigid_body"

    def test_keyword_match_fire(self):
        from phases.research import _build_technique_contract
        decision = TechniqueDecision(
            selected_technique="pyro_fire_with_smoke",
            reasoning="Fire and smoke",
        )
        contract = _build_technique_contract(decision, None)
        assert contract is not None
        assert contract.technique_name == "mantaflow_fire"

    def test_keyword_match_liquid(self):
        from phases.research import _build_technique_contract
        decision = TechniqueDecision(
            selected_technique="wine_pour_liquid",
            reasoning="Pour wine",
        )
        contract = _build_technique_contract(decision, None)
        assert contract is not None
        assert contract.technique_name == "mantaflow_liquid"

    def test_keyword_match_particles_rain(self):
        from phases.research import _build_technique_contract
        decision = TechniqueDecision(
            selected_technique="rain_particle_system",
            reasoning="Rain effect",
        )
        contract = _build_technique_contract(decision, None)
        assert contract is not None
        assert contract.technique_name == "particles_core"

    def test_keyword_match_particles_sparks(self):
        from phases.research import _build_technique_contract
        decision = TechniqueDecision(
            selected_technique="sparks_from_welding",
            reasoning="Welding sparks",
        )
        contract = _build_technique_contract(decision, None)
        assert contract is not None
        assert contract.technique_name == "particles_core"

    def test_keyword_match_cloth(self):
        from phases.research import _build_technique_contract
        decision = TechniqueDecision(
            selected_technique="cloth_simulation_curtain",
            reasoning="Curtain blowing in wind",
        )
        contract = _build_technique_contract(decision, None)
        assert contract is not None
        assert contract.technique_name == "cloth_softbody"

    def test_keyword_match_flag(self):
        from phases.research import _build_technique_contract
        decision = TechniqueDecision(
            selected_technique="waving_flag_in_wind",
            reasoning="Flag simulation",
        )
        contract = _build_technique_contract(decision, None)
        assert contract is not None
        assert contract.technique_name == "cloth_softbody"

    def test_keyword_match_geometry_nodes(self):
        from phases.research import _build_technique_contract
        decision = TechniqueDecision(
            selected_technique="geometry_nodes_scatter_rocks",
            reasoning="Procedural environment",
        )
        contract = _build_technique_contract(decision, None)
        assert contract is not None
        assert contract.technique_name == "geometry_nodes_environment"

    def test_keyword_match_terrain(self):
        from phases.research import _build_technique_contract
        decision = TechniqueDecision(
            selected_technique="procedural_terrain_generation",
            reasoning="Landscape",
        )
        contract = _build_technique_contract(decision, None)
        assert contract is not None
        assert contract.technique_name == "geometry_nodes_environment"

    def test_no_match_returns_none(self):
        from phases.research import _build_technique_contract
        decision = TechniqueDecision(
            selected_technique="completely_novel_technique_xyz",
            reasoning="Unknown",
        )
        contract = _build_technique_contract(decision, None)
        assert contract is None

    def test_none_decision(self):
        from phases.research import _build_technique_contract
        contract = _build_technique_contract(None, None)
        assert contract is None

    def test_alternatives_merged(self):
        from phases.research import _build_technique_contract
        decision = TechniqueDecision(
            selected_technique="mantaflow_fire",
            reasoning="Fire",
            alternative_techniques=["custom_alt_1", "custom_alt_2"],
        )
        contract = _build_technique_contract(decision, None)
        assert contract is not None
        # Pack alternatives + decision alternatives
        assert "custom_alt_1" in contract.alternative_techniques
        assert "custom_alt_2" in contract.alternative_techniques


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
