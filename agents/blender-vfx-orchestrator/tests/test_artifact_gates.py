"""
Unit tests for artifact gates.

These tests verify the deterministic validation of execution outputs
without requiring Blender or the full orchestrator.
"""

import tempfile
import os
from pathlib import Path
import unittest

# Make guardrails package importable without going through __init__.py
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "guardrails"))

# Now we can import artifact_gates as a standalone module
import artifact_gates

validate_execution_artifacts = artifact_gates.validate_execution_artifacts
discover_execution_artifacts = artifact_gates.discover_execution_artifacts
gate_cache_size = artifact_gates.gate_cache_size
gate_render_count = artifact_gates.gate_render_count
format_gate_failure_for_diagnosis = artifact_gates.format_gate_failure_for_diagnosis
MIN_CACHE_SIZE_BYTES = artifact_gates.MIN_CACHE_SIZE_BYTES


class TestArtifactDiscovery(unittest.TestCase):
    """Test artifact discovery from directories."""

    def test_empty_directory(self):
        """Empty directory should return empty summary."""
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = discover_execution_artifacts(tmpdir)
            self.assertFalse(summary.cache_exists)
            self.assertEqual(summary.cache_size_bytes, 0)
            self.assertEqual(summary.render_count, 0)

    def test_cache_discovery(self):
        """Cache directory with files should be discovered."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create cache directory with some files
            cache_dir = Path(tmpdir) / "cache"
            cache_dir.mkdir()

            # Create data files
            (cache_dir / "data_0001.vdb").write_bytes(b"x" * 50000)  # 50KB
            (cache_dir / "data_0002.vdb").write_bytes(b"x" * 50000)  # 50KB

            summary = discover_execution_artifacts(tmpdir)

            self.assertTrue(summary.cache_exists)
            self.assertEqual(summary.cache_file_count, 2)
            self.assertEqual(summary.cache_size_bytes, 100000)

    def test_render_discovery(self):
        """Render files should be discovered."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create render files
            (Path(tmpdir) / "render_0001.png").write_bytes(b"PNG")
            (Path(tmpdir) / "render_0002.png").write_bytes(b"PNG")
            (Path(tmpdir) / "render_0003.exr").write_bytes(b"EXR")

            summary = discover_execution_artifacts(tmpdir)

            self.assertEqual(summary.render_count, 3)
            self.assertEqual(len(summary.render_paths), 3)

    def test_vdb_discovery(self):
        """VDB files in output dir should be discovered."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create VDB file outside cache
            (Path(tmpdir) / "volume.vdb").write_bytes(b"x" * 20000)

            summary = discover_execution_artifacts(tmpdir)

            self.assertEqual(summary.vdb_count, 1)


class TestCacheGate(unittest.TestCase):
    """Test cache size gate."""

    def test_missing_cache_fails_when_required(self):
        """Missing cache should fail when required."""
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = discover_execution_artifacts(tmpdir)
            result = gate_cache_size(summary, require_cache=True)

            self.assertFalse(result.passed)
            self.assertEqual(result.gate_name, "CACHE_SIZE")
            self.assertIn("does not exist", result.reason)

    def test_missing_cache_passes_when_not_required(self):
        """Missing cache should pass when not required (shader effects)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = discover_execution_artifacts(tmpdir)
            result = gate_cache_size(summary, require_cache=False)

            self.assertTrue(result.passed)

    def test_small_cache_fails(self):
        """Cache below threshold should fail."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            cache_dir.mkdir()
            # Create small cache (100KB, below 1MB threshold)
            (cache_dir / "data.vdb").write_bytes(b"x" * 100000)

            summary = discover_execution_artifacts(tmpdir)
            result = gate_cache_size(summary, min_size_bytes=1_000_000)

            self.assertFalse(result.passed)
            self.assertIn("too small", result.reason)

    def test_sufficient_cache_passes(self):
        """Cache above threshold should pass."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            cache_dir.mkdir()
            # Create large cache (2MB)
            (cache_dir / "data.vdb").write_bytes(b"x" * 2_000_000)

            summary = discover_execution_artifacts(tmpdir)
            result = gate_cache_size(summary, min_size_bytes=1_000_000)

            self.assertTrue(result.passed)
            self.assertIn("OK", result.reason)


class TestRenderGate(unittest.TestCase):
    """Test render count gate."""

    def test_no_renders_fails(self):
        """No renders should fail."""
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = discover_execution_artifacts(tmpdir)
            result = gate_render_count(summary, min_count=1)

            self.assertFalse(result.passed)
            self.assertIn("Not enough renders", result.reason)

    def test_sufficient_renders_passes(self):
        """Meeting render count should pass."""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "render.png").write_bytes(b"PNG")

            summary = discover_execution_artifacts(tmpdir)
            result = gate_render_count(summary, min_count=1)

            self.assertTrue(result.passed)


class TestCombinedGates(unittest.TestCase):
    """Test combined gate validation."""

    def test_all_gates_pass(self):
        """All gates should pass with valid artifacts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create valid cache
            cache_dir = Path(tmpdir) / "cache"
            cache_dir.mkdir()
            (cache_dir / "data.vdb").write_bytes(b"x" * 2_000_000)  # 2MB

            # Create render
            (Path(tmpdir) / "render.png").write_bytes(b"PNG")

            passed, results, summary = validate_execution_artifacts(
                output_dir=tmpdir,
                effect_type="explosion",
                verbose=False,
            )

            self.assertTrue(passed)
            self.assertTrue(all(r.passed for r in results))

    def test_shader_effect_no_cache_required(self):
        """Shader effects should not require cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Only render, no cache
            (Path(tmpdir) / "render.png").write_bytes(b"PNG")

            passed, results, summary = validate_execution_artifacts(
                output_dir=tmpdir,
                effect_type="shader_based_volume",  # Shader effect
                verbose=False,
            )

            self.assertTrue(passed)

    def test_simulation_effect_requires_cache(self):
        """Simulation effects should require cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Only render, no cache
            (Path(tmpdir) / "render.png").write_bytes(b"PNG")

            passed, results, summary = validate_execution_artifacts(
                output_dir=tmpdir,
                effect_type="explosion",  # Simulation effect
                verbose=False,
            )

            self.assertFalse(passed)

    def test_failure_diagnosis_formatting(self):
        """Gate failure should produce useful diagnosis text."""
        with tempfile.TemporaryDirectory() as tmpdir:
            summary = discover_execution_artifacts(tmpdir)

            # Run with required cache to get failure
            passed, results, _ = validate_execution_artifacts(
                output_dir=tmpdir,
                effect_type="explosion",
                verbose=False,
            )

            diagnosis = format_gate_failure_for_diagnosis(results, summary)

            self.assertIn("ARTIFACT GATE FAILURE", diagnosis)
            self.assertIn("CACHE_SIZE", diagnosis)
            self.assertIn("Likely Causes", diagnosis)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def test_none_output_dir(self):
        """None output_dir should not crash."""
        summary = discover_execution_artifacts(None)
        self.assertFalse(summary.cache_exists)
        self.assertEqual(summary.render_count, 0)

    def test_nonexistent_directory(self):
        """Non-existent directory should not crash."""
        summary = discover_execution_artifacts("/nonexistent/path/that/does/not/exist")
        self.assertFalse(summary.cache_exists)
        self.assertEqual(summary.render_count, 0)


if __name__ == "__main__":
    unittest.main()
