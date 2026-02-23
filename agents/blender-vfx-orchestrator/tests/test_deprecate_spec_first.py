"""Tests for Phase 2A-6: Deprecate Spec-First Pipeline.

Verifies:
1. Script Writer still receives truth pack data (via context)
2. truth_pack_to_api_spec() produces valid APISpec from truth pack
3. Technique selection still runs standalone (no parallel API spec)
4. Spec-first attributes removed from orchestrator
"""

import os
import sys
import pytest

# Ensure the orchestrator package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from tools.truth_pack import truth_pack_to_api_spec
from models.api_spec import APISpec, APIAttribute, APIOperation


# ===== truth_pack_to_api_spec =====


class TestTruthPackToApiSpec:
    """Verify truth_pack_to_api_spec() produces valid APISpec."""

    @pytest.fixture
    def sample_truth_pack(self):
        """Minimal truth pack mimicking real build_truth_pack() output."""
        return {
            "FluidDomainSettings": {
                "properties": {
                    "domain_type": {
                        "type": "ENUM",
                        "items": ["GAS", "LIQUID"],
                        "default": "GAS",
                        "is_readonly": False,
                    },
                    "resolution_max": {
                        "type": "INT",
                        "range": [6, 10000],
                        "default": 32,
                        "is_readonly": False,
                    },
                    "cache_type": {
                        "type": "ENUM",
                        "items": ["REPLAY", "MODULAR", "ALL"],
                        "default": "REPLAY",
                        "is_readonly": False,
                    },
                    # Read-only props should be excluded
                    "is_baking": {
                        "type": "BOOLEAN",
                        "default": False,
                        "is_readonly": True,
                    },
                },
            },
            "FluidFlowSettings": {
                "properties": {
                    "flow_type": {
                        "type": "ENUM",
                        "items": ["SMOKE", "FIRE", "BOTH", "LIQUID"],
                        "default": "SMOKE",
                        "is_readonly": False,
                    },
                },
            },
            "Scene": {
                "properties": {
                    "frame_start": {
                        "type": "INT",
                        "range": [-500000, 500000],
                        "default": 1,
                        "is_readonly": False,
                    },
                },
            },
        }

    def test_returns_api_spec(self, sample_truth_pack):
        """truth_pack_to_api_spec returns an APISpec instance."""
        spec = truth_pack_to_api_spec(sample_truth_pack, "fire", "mantaflow_gas")
        assert isinstance(spec, APISpec)

    def test_effect_type_set(self, sample_truth_pack):
        """Effect type is correctly set on the spec."""
        spec = truth_pack_to_api_spec(sample_truth_pack, "fire", "mantaflow_gas")
        assert spec.effect_type == "fire"

    def test_technique_set(self, sample_truth_pack):
        """Technique is correctly set on the spec."""
        spec = truth_pack_to_api_spec(sample_truth_pack, "fire", "mantaflow_gas")
        assert spec.technique == "mantaflow_gas"

    def test_domain_attributes_populated(self, sample_truth_pack):
        """FluidDomainSettings props become domain_attributes."""
        spec = truth_pack_to_api_spec(sample_truth_pack, "fire", "mantaflow_gas")
        domain_names = [a.attribute_name for a in spec.domain_attributes]
        assert "domain_type" in domain_names
        assert "resolution_max" in domain_names
        assert "cache_type" in domain_names

    def test_readonly_excluded(self, sample_truth_pack):
        """Read-only properties are excluded from the spec."""
        spec = truth_pack_to_api_spec(sample_truth_pack, "fire", "mantaflow_gas")
        all_attr_names = [
            a.attribute_name
            for a in spec.domain_attributes + spec.flow_attributes + spec.scene_attributes + spec.object_attributes
        ]
        assert "is_baking" not in all_attr_names

    def test_flow_attributes_populated(self, sample_truth_pack):
        """FluidFlowSettings props become flow_attributes."""
        spec = truth_pack_to_api_spec(sample_truth_pack, "fire", "mantaflow_gas")
        flow_names = [a.attribute_name for a in spec.flow_attributes]
        assert "flow_type" in flow_names

    def test_scene_attributes_populated(self, sample_truth_pack):
        """Scene props become scene_attributes."""
        spec = truth_pack_to_api_spec(sample_truth_pack, "fire", "mantaflow_gas")
        scene_names = [a.attribute_name for a in spec.scene_attributes]
        assert "frame_start" in scene_names

    def test_ops_populated(self, sample_truth_pack):
        """Default ops list is always present."""
        spec = truth_pack_to_api_spec(sample_truth_pack, "fire", "mantaflow_gas")
        assert len(spec.ops) >= 1
        assert any("bake_all" in op.op_path for op in spec.ops)

    def test_doc_refs_populated(self, sample_truth_pack):
        """Each attribute has a doc_ref."""
        spec = truth_pack_to_api_spec(sample_truth_pack, "fire", "mantaflow_gas")
        for attr in spec.domain_attributes:
            assert attr.doc_ref, f"Missing doc_ref for {attr.attribute_name}"

    def test_empty_truth_pack(self):
        """Empty truth pack produces empty (but valid) APISpec."""
        spec = truth_pack_to_api_spec({}, "fire", "mantaflow_gas")
        assert isinstance(spec, APISpec)
        assert spec.effect_type == "fire"
        assert len(spec.domain_attributes) == 0

    def test_enum_values_preserved(self, sample_truth_pack):
        """Enum values from truth pack flow through to APIAttribute."""
        spec = truth_pack_to_api_spec(sample_truth_pack, "fire", "mantaflow_gas")
        domain_type_attr = next(
            (a for a in spec.domain_attributes if a.attribute_name == "domain_type"),
            None,
        )
        assert domain_type_attr is not None
        assert domain_type_attr.enum_values == ["GAS", "LIQUID"]


# ===== Orchestrator spec-first removal =====


class TestSpecFirstRemoval:
    """Verify spec-first attributes and methods are removed from orchestrator."""

    @pytest.fixture
    def orchestrator_source(self):
        """Read orchestrator.py source for static analysis."""
        orch_path = os.path.join(os.path.dirname(__file__), "..", "orchestrator.py")
        with open(orch_path) as f:
            return f.read()

    def test_no_spec_first_pipeline_attr(self, orchestrator_source):
        """Orchestrator should not set self._use_spec_first_pipeline."""
        assert "self._use_spec_first_pipeline" not in orchestrator_source

    def test_no_api_spec_agent_attr(self, orchestrator_source):
        """Orchestrator should not set self._api_spec_agent."""
        assert "self._api_spec_agent" not in orchestrator_source

    def test_no_code_writer_agent_attr(self, orchestrator_source):
        """Orchestrator should not set self._code_writer_agent."""
        assert "self._code_writer_agent" not in orchestrator_source

    def test_no_run_spec_first_methods(self, orchestrator_source):
        """Orchestrator should not define _run_spec_first_* methods."""
        assert "def _run_spec_first_pipeline" not in orchestrator_source
        assert "def _run_spec_first_modification" not in orchestrator_source
        assert "def _run_parallel_technique_and_spec" not in orchestrator_source
        assert "def _run_api_spec_only" not in orchestrator_source

    def test_no_spec_first_pipeline_calls(self, orchestrator_source):
        """No runtime calls to removed methods remain."""
        # Exclude comment lines (which mention them in deprecation note)
        live_lines = [
            line for line in orchestrator_source.split("\n")
            if not line.strip().startswith("#")
        ]
        live_code = "\n".join(live_lines)
        assert "self._use_spec_first_pipeline" not in live_code
        assert "self._code_writer_agent" not in live_code
        assert "self._api_spec_agent" not in live_code
        assert "await self._run_spec_first_pipeline(" not in live_code
        assert "await self._run_spec_first_modification(" not in live_code

    def test_truth_pack_to_api_spec_still_importable(self):
        """truth_pack_to_api_spec is still importable (backward compat)."""
        from tools.truth_pack import truth_pack_to_api_spec as fn
        assert callable(fn)

    def test_api_spec_models_still_importable(self):
        """APISpec, APIAttribute, APIOperation models still exist."""
        from models.api_spec import APISpec, APIAttribute, APIOperation
        assert APISpec is not None
        assert APIAttribute is not None
        assert APIOperation is not None


# ===== Script Writer context =====


class TestScriptWriterContext:
    """Verify Script Writer still receives truth pack data via context."""

    def test_context_has_truth_pack_field(self):
        """SharedContext defines truth_pack field."""
        ctx_path = os.path.join(
            os.path.dirname(__file__), "..", "models", "shared_context.py"
        )
        with open(ctx_path) as f:
            source = f.read()
        assert "truth_pack" in source

    def test_context_has_api_spec_field(self):
        """SharedContext defines api_spec field (backward compat)."""
        ctx_path = os.path.join(
            os.path.dirname(__file__), "..", "models", "shared_context.py"
        )
        with open(ctx_path) as f:
            source = f.read()
        assert "api_spec" in source

    def test_orchestrator_sets_truth_pack_on_context(self):
        """Orchestrator pipeline sets context.truth_pack from build_truth_pack()."""
        orch_path = os.path.join(os.path.dirname(__file__), "..", "orchestrator.py")
        with open(orch_path) as f:
            source = f.read()
        assert "context.truth_pack = truth_pack" in source

    def test_orchestrator_sets_api_spec_from_truth_pack(self):
        """Orchestrator pipeline sets context.api_spec via truth_pack_to_api_spec()."""
        orch_path = os.path.join(os.path.dirname(__file__), "..", "orchestrator.py")
        with open(orch_path) as f:
            source = f.read()
        assert "context.api_spec = truth_pack_to_api_spec(" in source


# ===== Deprecation notices =====


class TestDeprecationNotices:
    """Verify deprecation notices are present on old modules."""

    def test_api_spec_agent_deprecated(self):
        """api_spec_agent.py has deprecation notice."""
        agent_path = os.path.join(
            os.path.dirname(__file__), "..", "specialized_agents", "api_spec_agent.py"
        )
        with open(agent_path) as f:
            content = f.read(500)
        assert "DEPRECATED" in content
        assert "Phase 2A-6" in content

    def test_code_writer_agent_deprecated(self):
        """code_writer_agent.py has deprecation notice."""
        agent_path = os.path.join(
            os.path.dirname(__file__), "..", "specialized_agents", "code_writer_agent.py"
        )
        with open(agent_path) as f:
            content = f.read(500)
        assert "DEPRECATED" in content
        assert "Phase 2A-6" in content
