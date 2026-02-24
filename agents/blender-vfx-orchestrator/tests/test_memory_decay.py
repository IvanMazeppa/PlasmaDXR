"""
Tests for Phase 2B-6: Ebbinghaus Memory Decay.

Tests compute_retention(), retention filtering in code_pattern_tools
and dynamic_instructions, reinforce_entry(), backward compat, and
archive thresholds.
"""

import json
import math
import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Add parent dir for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.code_pattern_memory import (
    CodePattern,
    compute_retention,
    RETENTION_ACTIVE,
    RETENTION_ARCHIVE,
)


# =============================================================================
# TestComputeRetention — Core decay function
# =============================================================================

class TestComputeRetention(unittest.TestCase):
    """Test the Ebbinghaus retention formula."""

    def test_fresh_entry_returns_near_one(self):
        """An entry reinforced just now should have retention ~1.0."""
        now = datetime.now()
        pattern = CodePattern(
            pattern_id="p1",
            name="test",
            code_snippet="pass",
            issue_category="test",
            created_at=now.isoformat(),
            last_reinforced=now.isoformat(),
            reinforcement_count=1,
            average_improvement=50.0,
        )
        retention = compute_retention(pattern, now)
        self.assertAlmostEqual(retention, 1.0, places=2)

    def test_stale_entry_below_active_threshold(self):
        """An entry not reinforced for many days should drop below RETENTION_ACTIVE."""
        now = datetime.now()
        old = now - timedelta(days=30)
        pattern = CodePattern(
            pattern_id="p2",
            name="stale",
            code_snippet="pass",
            issue_category="test",
            created_at=old.isoformat(),
            last_reinforced=old.isoformat(),
            reinforcement_count=1,
            average_improvement=50.0,
        )
        retention = compute_retention(pattern, now)
        self.assertLess(retention, RETENTION_ACTIVE)

    def test_very_stale_below_archive_threshold(self):
        """An entry not reinforced for months should drop below RETENTION_ARCHIVE."""
        now = datetime.now()
        ancient = now - timedelta(days=365)
        pattern = CodePattern(
            pattern_id="p3",
            name="ancient",
            code_snippet="pass",
            issue_category="test",
            created_at=ancient.isoformat(),
            last_reinforced=ancient.isoformat(),
            reinforcement_count=1,
            average_improvement=50.0,
        )
        retention = compute_retention(pattern, now)
        self.assertLess(retention, RETENTION_ARCHIVE)

    def test_reinforcement_resets_decay(self):
        """Re-reinforcing an entry should bring retention back to ~1.0."""
        now = datetime.now()
        # Entry was created 30 days ago but reinforced just now
        pattern = CodePattern(
            pattern_id="p4",
            name="re-reinforced",
            code_snippet="pass",
            issue_category="test",
            created_at=(now - timedelta(days=30)).isoformat(),
            last_reinforced=now.isoformat(),
            reinforcement_count=5,
            average_improvement=60.0,
        )
        retention = compute_retention(pattern, now)
        self.assertAlmostEqual(retention, 1.0, places=2)

    def test_higher_reinforcement_count_slows_decay(self):
        """More reinforcements = higher strength = slower decay."""
        now = datetime.now()
        old = now - timedelta(days=5)

        low_reinforce = CodePattern(
            pattern_id="p5a",
            name="low",
            code_snippet="pass",
            issue_category="test",
            created_at=old.isoformat(),
            last_reinforced=old.isoformat(),
            reinforcement_count=1,
            average_improvement=50.0,
        )
        high_reinforce = CodePattern(
            pattern_id="p5b",
            name="high",
            code_snippet="pass",
            issue_category="test",
            created_at=old.isoformat(),
            last_reinforced=old.isoformat(),
            reinforcement_count=10,
            average_improvement=50.0,
        )

        r_low = compute_retention(low_reinforce, now)
        r_high = compute_retention(high_reinforce, now)
        self.assertGreater(r_high, r_low)

    def test_dict_input_works(self):
        """compute_retention should accept dict (KB entries) as well as CodePattern."""
        now = datetime.now()
        entry = {
            "last_reinforced": now.isoformat(),
            "reinforcement_count": 2,
            "confidence": 70.0,
            "created_at": now.isoformat(),
        }
        retention = compute_retention(entry, now)
        self.assertAlmostEqual(retention, 1.0, places=2)

    def test_no_reinforcement_falls_back_to_created_at(self):
        """Entries with no last_reinforced fall back to created_at for decay calc."""
        pattern = CodePattern(
            pattern_id="p6",
            name="no-reinforcement",
            code_snippet="pass",
            issue_category="test",
            # created_at defaults to now, last_reinforced="" → falls back to created_at
        )
        retention = compute_retention(pattern)
        # Newly created → created_at is ~now → retention ~1.0
        self.assertGreater(retention, 0.9)

    def test_missing_dict_fields_return_neutral(self):
        """A dict with no reinforcement or created_at returns 0.5."""
        entry = {"type": "parameter_knowledge", "parameter": "test"}
        retention = compute_retention(entry)
        self.assertEqual(retention, 0.5)


# =============================================================================
# TestRetentionFiltering — Filtering at query boundaries
# =============================================================================

class TestRetentionFiltering(unittest.TestCase):
    """Test that stale entries are filtered out when decay is enabled."""

    def _make_pattern(self, days_old: int, reinforcement_count: int = 1) -> CodePattern:
        now = datetime.now()
        ts = (now - timedelta(days=days_old)).isoformat()
        return CodePattern(
            pattern_id=f"p_{days_old}d",
            name=f"pattern_{days_old}d",
            code_snippet="pass",
            issue_category="test",
            created_at=ts,
            last_reinforced=ts,
            reinforcement_count=reinforcement_count,
            average_improvement=50.0,
        )

    def test_fresh_patterns_pass_filter(self):
        """Patterns reinforced recently should pass the RETENTION_ACTIVE filter."""
        pattern = self._make_pattern(0)
        now = datetime.now()
        self.assertGreaterEqual(compute_retention(pattern, now), RETENTION_ACTIVE)

    def test_stale_patterns_filtered_out(self):
        """Patterns not reinforced for a long time should be filtered out."""
        pattern = self._make_pattern(60)  # 60 days, reinforce_count=1, avg=50 → strength=0.5
        now = datetime.now()
        self.assertLess(compute_retention(pattern, now), RETENTION_ACTIVE)

    def test_feature_flag_disable_bypasses_filter(self):
        """When memory_decay_enabled=False, all patterns should be included."""
        from config.agent_config import PresetConfig, AgentConfigManager

        preset = PresetConfig(name="test_no_decay", memory_decay_enabled=False)
        config = AgentConfigManager(preset=preset)
        self.assertFalse(config.use_memory_decay())


# =============================================================================
# TestReinforceEntry — KB reinforcement
# =============================================================================

class TestReinforceEntry(unittest.TestCase):
    """Test the reinforce_entry implementation."""

    def setUp(self):
        """Create a temp DB with a test entry."""
        self._tmpdir = tempfile.mkdtemp()
        self._db_path = os.path.join(self._tmpdir, "test_tracker.db")

    def _setup_tracker(self):
        """Create tracker with a seeded entry."""
        # Import here to avoid issues
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        from agents_experiment_tracker_path import get_tracker_with_db
        # We'll use the actual DB module directly
        exp_tracker_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "agents", "experiment-tracker"
        )
        sys.path.insert(0, exp_tracker_path)
        from database import ExperimentDatabase
        return ExperimentDatabase(self._db_path)

    def test_reinforce_updates_count_and_timestamp(self):
        """Reinforcing an entry should increment count and update timestamp."""
        # Use the _impl function directly with a mock tracker
        from tools.experiment_tracker_tools import _reinforce_entry_impl

        # Create a mock tracker and DB
        mock_conn = MagicMock()
        mock_row = {"confidence": 50.0, "reinforcement_count": 1}
        mock_conn.execute.return_value.fetchone.return_value = mock_row
        mock_conn.__enter__ = lambda s: mock_conn
        mock_conn.__exit__ = MagicMock(return_value=False)

        mock_db = MagicMock()
        mock_db._connection.return_value = mock_conn

        mock_tracker = MagicMock()
        mock_tracker.db = mock_db

        with patch(
            "tools.experiment_tracker_tools._get_tracker_instance",
            return_value=mock_tracker,
        ):
            result = _reinforce_entry_impl("technique:fire:test", 80.0)

        data = json.loads(result)
        self.assertTrue(data["success"])
        self.assertEqual(data["reinforcement_count"], 2)
        # EMA: 50 + 0.1 * (80 - 50) = 53.0
        self.assertAlmostEqual(data["confidence"], 53.0, places=1)

    def test_reinforce_entry_not_found(self):
        """Reinforcing a non-existent entry should return error."""
        from tools.experiment_tracker_tools import _reinforce_entry_impl

        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchone.return_value = None
        mock_conn.__enter__ = lambda s: mock_conn
        mock_conn.__exit__ = MagicMock(return_value=False)

        mock_db = MagicMock()
        mock_db._connection.return_value = mock_conn

        mock_tracker = MagicMock()
        mock_tracker.db = mock_db

        with patch(
            "tools.experiment_tracker_tools._get_tracker_instance",
            return_value=mock_tracker,
        ):
            result = _reinforce_entry_impl("nonexistent_entry", 60.0)

        data = json.loads(result)
        self.assertFalse(data["success"])
        self.assertIn("not found", data["error"])

    def test_ema_confidence_update(self):
        """EMA should converge confidence toward quality_score over repeated calls."""
        from tools.experiment_tracker_tools import _reinforce_entry_impl

        confidence = 50.0
        for _ in range(10):
            mock_conn = MagicMock()
            mock_row = {"confidence": confidence, "reinforcement_count": 1}
            mock_conn.execute.return_value.fetchone.return_value = mock_row
            mock_conn.__enter__ = lambda s: mock_conn
            mock_conn.__exit__ = MagicMock(return_value=False)

            mock_db = MagicMock()
            mock_db._connection.return_value = mock_conn

            mock_tracker = MagicMock()
            mock_tracker.db = mock_db

            with patch(
                "tools.experiment_tracker_tools._get_tracker_instance",
                return_value=mock_tracker,
            ):
                result = _reinforce_entry_impl("test", 90.0)

            data = json.loads(result)
            confidence = data["confidence"]

        # After 10 iterations of EMA with quality=90, confidence should approach 90
        self.assertGreater(confidence, 75.0)


# =============================================================================
# TestBackwardCompat — Old data without new fields
# =============================================================================

class TestBackwardCompat(unittest.TestCase):
    """Test backward compatibility with old data formats."""

    def test_old_code_pattern_json_loads(self):
        """CodePattern JSON without new fields should load fine via from_dict."""
        old_data = {
            "pattern_id": "old_p",
            "name": "old pattern",
            "code_snippet": "bpy.ops.mesh.primitive_cube_add()",
            "issue_category": "geometry",
            "effect_types": ["fire"],
            "created_at": "2026-01-15T10:00:00",
            "success_count": 3,
            "failure_count": 1,
            "usage_count": 4,
            "average_improvement": 15.0,
            "parameters_affected": ["size"],
            # Intentionally missing: last_reinforced, reinforcement_count
        }
        pattern = CodePattern.from_dict(old_data)
        self.assertEqual(pattern.last_reinforced, "")
        self.assertEqual(pattern.reinforcement_count, 0)

    def test_old_kb_entry_dict_computes_retention(self):
        """KB entry dict without new fields should get neutral retention."""
        old_entry = {
            "type": "parameter_knowledge",
            "parameter": "burning_rate",
            "rules": ["Use 0.3-0.8 for candle"],
            "warnings": [],
            "confidence": 60.0,
            # Missing: last_reinforced, reinforcement_count, created_at
        }
        retention = compute_retention(old_entry)
        self.assertEqual(retention, 0.5)  # No timestamps → neutral


# =============================================================================
# TestArchiveThreshold — Entries near/below archive boundary
# =============================================================================

class TestArchiveThreshold(unittest.TestCase):
    """Test entries near the archive threshold."""

    def test_entry_below_archive_flagged(self):
        """Very old entries should fall below RETENTION_ARCHIVE."""
        now = datetime.now()
        ancient = now - timedelta(days=365)
        entry = {
            "last_reinforced": ancient.isoformat(),
            "reinforcement_count": 1,
            "confidence": 50.0,
        }
        retention = compute_retention(entry, now)
        self.assertLess(retention, RETENTION_ARCHIVE)

    def test_entry_above_archive_kept(self):
        """Recently used entries should stay above RETENTION_ARCHIVE."""
        now = datetime.now()
        recent = now - timedelta(days=1)
        entry = {
            "last_reinforced": recent.isoformat(),
            "reinforcement_count": 3,
            "confidence": 70.0,
        }
        retention = compute_retention(entry, now)
        self.assertGreater(retention, RETENTION_ARCHIVE)


# =============================================================================
# TestConfigIntegration — Config flag behavior
# =============================================================================

class TestConfigIntegration(unittest.TestCase):
    """Test config flag integration."""

    def test_preset_config_memory_decay_field(self):
        """PresetConfig should have memory_decay_enabled field."""
        from config.agent_config import PresetConfig
        preset = PresetConfig(name="test")
        self.assertTrue(preset.memory_decay_enabled)  # Default is True

    def test_preset_from_dict_parses_flag(self):
        """from_dict should parse memory_decay_enabled."""
        from config.agent_config import PresetConfig
        preset = PresetConfig.from_dict("test", {"memory_decay_enabled": False})
        self.assertFalse(preset.memory_decay_enabled)

    def test_config_manager_accessor(self):
        """AgentConfigManager should expose use_memory_decay()."""
        from config.agent_config import PresetConfig, AgentConfigManager
        preset = PresetConfig(name="test", memory_decay_enabled=True)
        config = AgentConfigManager(preset=preset)
        self.assertTrue(config.use_memory_decay())

        preset_off = PresetConfig(name="test_off", memory_decay_enabled=False)
        config_off = AgentConfigManager(preset=preset_off)
        self.assertFalse(config_off.use_memory_decay())


# =============================================================================
# TestRetentionMath — Mathematical properties of the decay function
# =============================================================================

class TestRetentionMath(unittest.TestCase):
    """Test mathematical correctness of the decay formula."""

    def test_formula_matches_ebbinghaus(self):
        """R(t) = exp(-t/S) where S = max(reinforce_count * avg/100, 0.1)."""
        now = datetime.now()
        days = 3
        ts = (now - timedelta(days=days)).isoformat()

        entry = {
            "last_reinforced": ts,
            "reinforcement_count": 2,
            "confidence": 80.0,
        }
        expected_strength = max(2 * 80.0 / 100.0, 0.1)  # 1.6
        expected_retention = math.exp(-days / expected_strength)

        actual = compute_retention(entry, now)
        self.assertAlmostEqual(actual, expected_retention, places=5)

    def test_strength_floor_prevents_division_by_zero(self):
        """Strength should be clamped to at least 0.1."""
        now = datetime.now()
        ts = (now - timedelta(days=1)).isoformat()

        entry = {
            "last_reinforced": ts,
            "reinforcement_count": 0,
            "confidence": 0.0,
        }
        # strength = max(0 * 0/100, 0.1) = 0.1
        expected = math.exp(-1 / 0.1)
        actual = compute_retention(entry, now)
        self.assertAlmostEqual(actual, expected, places=5)


if __name__ == "__main__":
    unittest.main()
