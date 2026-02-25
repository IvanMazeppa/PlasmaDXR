"""
Tests for Phase 2B-7: Effect-Type-Scoped Evidence Gating.

Tests that KB entries are scoped by effect_type so that fire patterns
don't leak into liquid scripts (and vice versa).  Universal entries
(effect_type='') are always returned regardless of scope.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent dir for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Add experiment-tracker to path
EXPERIMENT_TRACKER_DIR = (
    Path(__file__).parent.parent.parent.parent / "agents" / "experiment-tracker"
)
sys.path.insert(0, str(EXPERIMENT_TRACKER_DIR))


# =============================================================================
# TestDatabaseMigration — effect_type column exists after init
# =============================================================================

class TestDatabaseMigration(unittest.TestCase):
    """Verify that the effect_type column is created during init."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        from database import ExperimentDatabase
        self.db = ExperimentDatabase(db_path=Path(self.tmp.name))

    def tearDown(self):
        os.unlink(self.tmp.name)

    def test_effect_type_column_exists(self):
        """parameter_knowledge should have an effect_type column after init."""
        with self.db._connection() as conn:
            cols = [
                r[1] for r in conn.execute(
                    "PRAGMA table_info(parameter_knowledge)"
                ).fetchall()
            ]
        self.assertIn("effect_type", cols)

    def test_effect_type_defaults_to_empty(self):
        """New rows should default to empty string effect_type."""
        with self.db._connection() as conn:
            conn.execute("""
                INSERT INTO parameter_knowledge (parameter, rules, warnings, last_updated, confidence)
                VALUES ('test_param', '["rule1"]', '[]', '2026-01-01', 0.5)
            """)
            row = conn.execute(
                "SELECT effect_type FROM parameter_knowledge WHERE parameter = 'test_param'"
            ).fetchone()
        self.assertEqual(row['effect_type'], '')


# =============================================================================
# TestQueryKnowledgeScoping — query_knowledge respects effect_type filter
# =============================================================================

class TestQueryKnowledgeScoping(unittest.TestCase):
    """Test that query_knowledge filters by effect_type correctly."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        from database import ExperimentDatabase
        self.db = ExperimentDatabase(db_path=Path(self.tmp.name))

        # Seed: one fire entry, one liquid entry, one universal entry
        with self.db._connection() as conn:
            for param, et in [
                ("technique:fire:mantaflow_gas", "fire"),
                ("technique:liquid:mantaflow_flip", "liquid"),
                ("technique:general:subdivision_surface", ""),
            ]:
                conn.execute("""
                    INSERT INTO parameter_knowledge
                        (parameter, rules, warnings, last_updated, confidence, effect_type)
                    VALUES (?, '["use this technique"]', '[]', '2026-01-01', 0.8, ?)
                """, (param, et))

    def tearDown(self):
        os.unlink(self.tmp.name)

    def test_no_filter_returns_all(self):
        """Without effect_type filter, all entries matching query are returned."""
        results = self.db.query_knowledge("technique")
        params = [r['parameter'] for r in results]
        self.assertEqual(len(results), 3)

    def test_fire_filter_excludes_liquid(self):
        """With effect_type='fire', liquid entry should be excluded."""
        results = self.db.query_knowledge("technique", effect_type="fire")
        params = [r['parameter'] for r in results]
        self.assertIn("technique:fire:mantaflow_gas", params)
        self.assertIn("technique:general:subdivision_surface", params)  # universal
        self.assertNotIn("technique:liquid:mantaflow_flip", params)

    def test_liquid_filter_excludes_fire(self):
        """With effect_type='liquid', fire entry should be excluded."""
        results = self.db.query_knowledge("technique", effect_type="liquid")
        params = [r['parameter'] for r in results]
        self.assertIn("technique:liquid:mantaflow_flip", params)
        self.assertIn("technique:general:subdivision_surface", params)  # universal
        self.assertNotIn("technique:fire:mantaflow_gas", params)

    def test_universal_always_included(self):
        """Universal entries (effect_type='') are included regardless of filter."""
        for et in ["fire", "liquid", "smoke", "explosion"]:
            results = self.db.query_knowledge("technique", effect_type=et)
            params = [r['parameter'] for r in results]
            self.assertIn("technique:general:subdivision_surface", params)

    def test_result_includes_effect_type_field(self):
        """Each result dict should include effect_type field."""
        results = self.db.query_knowledge("technique")
        for r in results:
            self.assertIn('effect_type', r)


# =============================================================================
# TestAddManualLearningWithEffectType — effect_type stored on insert
# =============================================================================

class TestAddManualLearningWithEffectType(unittest.TestCase):
    """Test that add_manual_learning stores effect_type."""

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        from database import ExperimentDatabase
        self.db = ExperimentDatabase(db_path=Path(self.tmp.name))

        # Create tracker with test DB
        from tracker import ExperimentTracker
        self.tracker = ExperimentTracker()
        self.tracker.db = self.db

    def tearDown(self):
        os.unlink(self.tmp.name)

    def test_new_entry_stores_effect_type(self):
        """A new manual learning should store effect_type."""
        self.tracker.add_manual_learning(
            parameter="test:fire:param",
            rule="use high resolution for fire",
            effect_type="fire",
        )
        with self.db._connection() as conn:
            row = conn.execute(
                "SELECT effect_type FROM parameter_knowledge WHERE parameter = 'test:fire:param'"
            ).fetchone()
        self.assertEqual(row['effect_type'], 'fire')

    def test_new_entry_defaults_to_empty(self):
        """Without explicit effect_type, should default to empty."""
        self.tracker.add_manual_learning(
            parameter="test:universal",
            rule="always use smooth shading",
        )
        with self.db._connection() as conn:
            row = conn.execute(
                "SELECT effect_type FROM parameter_knowledge WHERE parameter = 'test:universal'"
            ).fetchone()
        self.assertEqual(row['effect_type'], '')


# =============================================================================
# TestConfigFlag — effect_type_scoping flag in presets
# =============================================================================

class TestConfigFlag(unittest.TestCase):
    """Test that the effect_type_scoping config flag works."""

    def test_preset_has_flag(self):
        """PresetConfig should have effect_type_scoping field."""
        from config.agent_config import PresetConfig
        p = PresetConfig(name="test")
        self.assertTrue(p.effect_type_scoping)

    def test_from_dict_reads_flag(self):
        """from_dict should parse effect_type_scoping."""
        from config.agent_config import PresetConfig
        p = PresetConfig.from_dict("test", {"effect_type_scoping": False})
        self.assertFalse(p.effect_type_scoping)

    def test_config_manager_accessor(self):
        """AgentConfigManager should expose use_effect_type_scoping()."""
        from config.agent_config import AgentConfigManager, PresetConfig
        config = AgentConfigManager(
            preset=PresetConfig(name="test", effect_type_scoping=True)
        )
        self.assertTrue(config.use_effect_type_scoping())

    def test_debug_preset_disables_scoping(self):
        """Debug preset should have effect_type_scoping=false for visibility."""
        from config.agent_config import AgentConfigManager
        config = AgentConfigManager.load_preset("debug")
        self.assertFalse(config.use_effect_type_scoping())


# =============================================================================
# TestDynamicInstructionsScoping — query_validated_learnings passes effect_type
# =============================================================================

class TestDynamicInstructionsScoping(unittest.TestCase):
    """Test that dynamic instructions pass effect_type to KB query."""

    @patch("tools.dynamic_instructions._get_knowledge_base")
    @patch("config.agent_config.get_config")
    def test_scoping_enabled_passes_effect_type(self, mock_config, mock_kb_fn):
        """When scoping enabled, effect_type is passed to KB query."""
        mock_config.return_value.use_effect_type_scoping.return_value = True

        # Track calls to the KB query function
        calls = []
        def fake_kb_query(query, effect_type=""):
            calls.append({"query": query, "effect_type": effect_type})
            return json.dumps({"results": []})

        mock_kb_fn.return_value = fake_kb_query

        from tools.dynamic_instructions import query_validated_learnings
        query_validated_learnings("fire", category="physics")

        self.assertTrue(len(calls) > 0)
        self.assertEqual(calls[0]["effect_type"], "fire")

    @patch("tools.dynamic_instructions._get_knowledge_base")
    @patch("config.agent_config.get_config")
    def test_scoping_disabled_no_effect_type(self, mock_config, mock_kb_fn):
        """When scoping disabled, effect_type is empty."""
        mock_config.return_value.use_effect_type_scoping.return_value = False

        calls = []
        def fake_kb_query(query, effect_type=""):
            calls.append({"query": query, "effect_type": effect_type})
            return json.dumps({"results": []})

        mock_kb_fn.return_value = fake_kb_query

        from tools.dynamic_instructions import query_validated_learnings
        query_validated_learnings("liquid", category="physics")

        self.assertTrue(len(calls) > 0)
        self.assertEqual(calls[0]["effect_type"], "")


if __name__ == "__main__":
    unittest.main()
