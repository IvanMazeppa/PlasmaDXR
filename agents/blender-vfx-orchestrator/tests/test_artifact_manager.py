"""
Unit tests for artifact manager.

These tests verify artifact-first handoffs work correctly.
"""

import tempfile
import json
import os
from pathlib import Path
import unittest

# Import directly from module file (avoid SDK dependencies in utils/__init__.py)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

import artifact_manager
ArtifactManager = artifact_manager.ArtifactManager
ResearchArtifact = artifact_manager.ResearchArtifact
QualityArtifact = artifact_manager.QualityArtifact
IterationArtifact = artifact_manager.IterationArtifact
ScorecardArtifact = artifact_manager.ScorecardArtifact
ARTIFACTS_DIR = artifact_manager.ARTIFACTS_DIR


class TestArtifactManager(unittest.TestCase):
    """Test artifact manager functionality."""

    def setUp(self):
        """Create a temporary session for each test."""
        self.test_session_id = f"test_session_{os.getpid()}"
        self.manager = ArtifactManager(self.test_session_id)

    def tearDown(self):
        """Clean up test artifacts."""
        import shutil
        artifact_dir = ARTIFACTS_DIR / self.test_session_id
        if artifact_dir.exists():
            shutil.rmtree(artifact_dir)

    def test_artifact_dir_created(self):
        """Artifact directory should be created on init."""
        self.assertTrue(self.manager.artifact_dir.exists())
        self.assertTrue(self.manager.artifact_dir.is_dir())

    def test_write_research_artifact(self):
        """Writing research artifact should create JSON file."""
        path = self.manager.write_research_from_output(
            recommended_approach="mantaflow_fire",
            key_parameters={"temperature": 3.0, "density": 5.0},
            api_modules=["bpy.types.FluidDomainSettings"],
            warnings=["Watch for overflow"],
            alternative_approaches=["shader_based"],
            doc_refs=["https://docs.blender.org/manual/en/5.0/physics/fluid/"],
        )

        self.assertTrue(Path(path).exists())

        # Verify content
        with open(path) as f:
            data = json.load(f)

        self.assertEqual(data["recommended_approach"], "mantaflow_fire")
        self.assertEqual(data["key_parameters"]["temperature"], 3.0)
        self.assertIn("timestamp", data)

    def test_write_quality_artifact(self):
        """Writing quality artifact should create JSON file."""
        path = self.manager.write_quality_from_output(
            iteration=1,
            overall_score=75.5,
            passed=True,
            primary_issue=None,
            issues=[],
            suggestions=["Increase brightness"],
            vision_assessment="Good volumetric density",
            render_path="/path/to/render.png",
        )

        self.assertTrue(Path(path).exists())
        self.assertIn("quality_iter1.json", path)

        with open(path) as f:
            data = json.load(f)

        self.assertEqual(data["overall_score"], 75.5)
        self.assertTrue(data["passed"])

    def test_write_iteration_artifact(self):
        """Writing iteration artifact should create JSON file."""
        path = self.manager.write_iteration_from_values(
            iteration=2,
            script_path="/path/to/script.py",
            technique_used="mantaflow_fire",
            score=65.0,
            passed=True,
            primary_issue="Brightness low",
            render_path="/path/to/render.png",
            parameter_changes={"emission": 2.0},
            escape_level=1,
        )

        self.assertTrue(Path(path).exists())
        self.assertIn("iteration_2.json", path)

    def test_write_scorecard_artifact(self):
        """Writing scorecard artifact should create JSON file."""
        path = self.manager.write_scorecard_from_values(
            iteration=1,
            overall_score=70.0,
            passed=True,
            critical_issues=[],
            warnings=["Brightness could be higher"],
            cache_size_mb=15.5,
            render_count=3,
            iteration_cost_usd=0.25,
            iteration_time_seconds=45.0,
            utility_score=0.75,
        )

        self.assertTrue(Path(path).exists())

        with open(path) as f:
            data = json.load(f)

        self.assertEqual(data["cache_size_mb"], 15.5)
        self.assertEqual(data["render_count"], 3)
        self.assertEqual(data["utility_score"], 0.75)

    def test_get_artifact_paths(self):
        """Getting artifact paths should return correct paths."""
        # Write artifacts
        self.manager.write_research_from_output(
            recommended_approach="test",
            key_parameters={},
            api_modules=[],
            warnings=[],
            alternative_approaches=[],
            doc_refs=[],
        )
        self.manager.write_quality_from_output(
            iteration=1,
            overall_score=50.0,
            passed=False,
            primary_issue="Test",
            issues=[],
            suggestions=[],
        )

        # Check paths
        research_path = self.manager.get_research_path()
        self.assertIsNotNone(research_path)
        self.assertIn("research.json", research_path)

        quality_path = self.manager.get_quality_path(1)
        self.assertIsNotNone(quality_path)
        self.assertIn("quality_iter1.json", quality_path)

        # Non-existent should return None
        self.assertIsNone(self.manager.get_quality_path(99))

    def test_get_iteration_summary(self):
        """Iteration summary should show recent iterations."""
        # Write some iterations
        for i in range(1, 4):
            self.manager.write_iteration_from_values(
                iteration=i,
                script_path=f"/path/script_{i}.py",
                technique_used="mantaflow",
                score=50.0 + i * 5,
                passed=i == 3,
                primary_issue=f"Issue {i}" if i < 3 else None,
            )

        summary = self.manager.get_iteration_summary()

        self.assertIn("Recent Iterations", summary)
        self.assertIn("Iter 1", summary)
        self.assertIn("Iter 3", summary)
        self.assertIn("score=65.0", summary)

    def test_get_artifact_paths_summary(self):
        """Artifact paths summary should list available artifacts."""
        # Write artifacts
        self.manager.write_research_from_output(
            recommended_approach="test",
            key_parameters={},
            api_modules=[],
            warnings=[],
            alternative_approaches=[],
            doc_refs=[],
        )
        self.manager.write_quality_from_output(
            iteration=1,
            overall_score=50.0,
            passed=False,
            primary_issue="Test",
            issues=[],
            suggestions=[],
        )

        summary = self.manager.get_artifact_paths_summary()

        self.assertIn("Available Artifacts", summary)
        self.assertIn("Research:", summary)
        self.assertIn("Quality:", summary)


class TestArtifactDataclasses(unittest.TestCase):
    """Test artifact dataclasses."""

    def test_research_artifact_timestamp(self):
        """Research artifact should auto-generate timestamp."""
        artifact = ResearchArtifact(
            recommended_approach="test",
            key_parameters={},
            api_modules=[],
            warnings=[],
            alternative_approaches=[],
            doc_refs=[],
        )

        self.assertIsNotNone(artifact.timestamp)
        self.assertIn("T", artifact.timestamp)  # ISO format

    def test_quality_artifact_fields(self):
        """Quality artifact should have all required fields."""
        artifact = QualityArtifact(
            overall_score=75.0,
            passed=True,
            primary_issue=None,
            issues=[],
            suggestions=[],
            vision_assessment="Good",
            reference_similarity=0.85,
            render_path="/path/render.png",
            iteration=1,
        )

        self.assertEqual(artifact.overall_score, 75.0)
        self.assertEqual(artifact.reference_similarity, 0.85)


if __name__ == "__main__":
    unittest.main()
