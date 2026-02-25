"""
Tests for Phase 2B-8: Artifact-Based Sharing.

Tests that:
- read_artifact returns correct JSON content
- read_artifact caps output at 4K chars
- Missing artifact returns graceful error
- list_session_artifacts discovers files
- Config flag exists and presets parse it
- Artifact tools are wired into agents
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

# Add parent dir for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# =============================================================================
# TestReadArtifact — core read functionality
# =============================================================================

class TestReadArtifact(unittest.TestCase):
    """Test _read_artifact_impl reads and caps artifact content."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.artifact_path = os.path.join(self.tmpdir, "quality_iter1.json")

        self.sample_data = {
            "overall_score": 67.0,
            "passed": True,
            "primary_issue": "flat liquid surface",
            "issues": ["weak refraction", "slightly faceted glass"],
            "suggestions": ["increase Transmission Weight", "add SubdivisionSurface"],
            "vision_assessment": "The render shows a wine glass with liquid.",
            "iteration": 1,
        }
        with open(self.artifact_path, "w") as f:
            json.dump(self.sample_data, f, indent=2)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_read_existing_artifact(self):
        """Reading an existing artifact returns its JSON content."""
        from tools.artifact_tools import _read_artifact_impl
        result = _read_artifact_impl(self.artifact_path)
        data = json.loads(result)
        self.assertEqual(data["overall_score"], 67.0)
        self.assertEqual(data["primary_issue"], "flat liquid surface")

    def test_read_missing_artifact(self):
        """Reading a missing artifact returns error JSON."""
        from tools.artifact_tools import _read_artifact_impl
        result = _read_artifact_impl("/nonexistent/path/artifact.json")
        data = json.loads(result)
        self.assertIn("error", data)
        self.assertIn("not found", data["error"].lower())

    def test_read_non_json_file(self):
        """Reading a non-JSON file returns error."""
        from tools.artifact_tools import _read_artifact_impl
        txt_path = os.path.join(self.tmpdir, "notes.txt")
        with open(txt_path, "w") as f:
            f.write("hello")
        result = _read_artifact_impl(txt_path)
        data = json.loads(result)
        self.assertIn("error", data)

    def test_write_then_read_roundtrip(self):
        """Write artifact via ArtifactManager, read via read_artifact."""
        from utils.artifact_manager import ArtifactManager
        from tools.artifact_tools import _read_artifact_impl

        mgr = ArtifactManager("test_roundtrip")
        path = mgr.write_quality_from_output(
            iteration=1,
            overall_score=72.5,
            passed=True,
            primary_issue=None,
            issues=[],
            suggestions=[],
            vision_assessment="Looks great",
        )

        result = _read_artifact_impl(path)
        data = json.loads(result)
        self.assertEqual(data["overall_score"], 72.5)
        self.assertTrue(data["passed"])

        # Cleanup
        import shutil
        shutil.rmtree(mgr.artifact_dir, ignore_errors=True)


# =============================================================================
# TestReadArtifactCap — truncation at 4K chars
# =============================================================================

class TestReadArtifactCap(unittest.TestCase):
    """Test that large artifacts are capped at MAX_ARTIFACT_CHARS."""

    def test_large_artifact_truncated(self):
        """Artifacts larger than 4K chars should be truncated."""
        from tools.artifact_tools import _read_artifact_impl, MAX_ARTIFACT_CHARS

        tmpdir = tempfile.mkdtemp()
        large_path = os.path.join(tmpdir, "large.json")

        # Create a large JSON file
        large_data = {"data": "x" * (MAX_ARTIFACT_CHARS + 1000)}
        with open(large_path, "w") as f:
            json.dump(large_data, f)

        result = _read_artifact_impl(large_path)
        # Result should be truncated — either a truncation wrapper or raw content <= 4K
        self.assertLessEqual(len(result), MAX_ARTIFACT_CHARS + 500)  # Allow for wrapper overhead

        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


# =============================================================================
# TestListSessionArtifacts — discovery
# =============================================================================

class TestListSessionArtifacts(unittest.TestCase):
    """Test _list_session_artifacts_impl discovers files."""

    def test_list_existing_session(self):
        """Listing artifacts for a session with files returns them."""
        from utils.artifact_manager import ArtifactManager
        from tools.artifact_tools import _list_session_artifacts_impl

        mgr = ArtifactManager("test_list_session")
        mgr.write_research_from_output(
            recommended_approach="test approach",
            key_parameters={"param1": 1.0},
            api_modules=["bpy.ops.fluid"],
            warnings=[],
            alternative_approaches=[],
            doc_refs=[],
        )
        mgr.write_quality_from_output(
            iteration=1, overall_score=50.0, passed=False,
            primary_issue="test", issues=[], suggestions=[],
        )

        result = _list_session_artifacts_impl("test_list_session")
        data = json.loads(result)
        self.assertEqual(data["artifact_count"], 2)
        filenames = [a["filename"] for a in data["artifacts"]]
        self.assertIn("research.json", filenames)
        self.assertIn("quality_iter1.json", filenames)

        # Cleanup
        import shutil
        shutil.rmtree(mgr.artifact_dir, ignore_errors=True)

    def test_list_missing_session(self):
        """Listing artifacts for nonexistent session returns error."""
        from tools.artifact_tools import _list_session_artifacts_impl
        result = _list_session_artifacts_impl("nonexistent_session_xyz")
        data = json.loads(result)
        self.assertIn("error", data)
        self.assertEqual(data["artifact_count"], 0)


# =============================================================================
# TestConfigFlag — artifact_sharing flag
# =============================================================================

class TestConfigFlag(unittest.TestCase):
    """Test that the artifact_sharing config flag works."""

    def test_preset_has_flag(self):
        """PresetConfig should have artifact_sharing field."""
        from config.agent_config import PresetConfig
        p = PresetConfig(name="test")
        self.assertTrue(p.artifact_sharing)

    def test_from_dict_reads_flag(self):
        """from_dict should parse artifact_sharing."""
        from config.agent_config import PresetConfig
        p = PresetConfig.from_dict("test", {"artifact_sharing": False})
        self.assertFalse(p.artifact_sharing)

    def test_config_manager_accessor(self):
        """AgentConfigManager should expose use_artifact_sharing()."""
        from config.agent_config import AgentConfigManager, PresetConfig
        config = AgentConfigManager(
            preset=PresetConfig(name="test", artifact_sharing=True)
        )
        self.assertTrue(config.use_artifact_sharing())

    def test_all_presets_have_flag(self):
        """All YAML presets should include artifact_sharing."""
        from config.agent_config import AgentConfigManager
        config = AgentConfigManager.load_preset("development")
        for name in config.list_presets():
            preset_config = AgentConfigManager.load_preset(name)
            self.assertIsInstance(
                preset_config.preset.artifact_sharing,
                bool,
                f"Preset '{name}' missing artifact_sharing",
            )


# =============================================================================
# TestAgentToolWiring — artifact tools are in agent tool lists
# =============================================================================

class TestAgentToolWiring(unittest.TestCase):
    """Test that read_artifact is wired into the right agents."""

    def test_quality_gate_has_read_artifact(self):
        """Quality gate coordinator should have read_artifact tool."""
        from orchestrator import create_quality_gate_coordinator
        agent = create_quality_gate_coordinator()
        tool_names = [t.name for t in agent.tools]
        self.assertIn("read_artifact", tool_names)

    def test_quality_gate_instructions_mention_artifact(self):
        """Quality gate instructions should mention read_artifact."""
        from orchestrator import create_quality_gate_coordinator
        agent = create_quality_gate_coordinator()
        instructions = agent.instructions
        if callable(instructions):
            pass  # Dynamic instructions — skip text check
        else:
            self.assertIn("read_artifact", instructions)


if __name__ == "__main__":
    unittest.main()
