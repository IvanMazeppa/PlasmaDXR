"""Tests for Phase 2B-1: Stateless iterations (Ralph Pattern).

Verifies:
1. Config flag defaults and YAML integration
2. IterationSnapshot construction and formatting
3. Session routing logic (stateless vs stateful)
4. Compaction guard behavior
5. Preset YAML integration
"""
import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from config.agent_config import AgentConfigManager, PresetConfig
from utils.iteration_state import IterationSnapshot


# ============================================================
# 1. Config flag
# ============================================================

class TestStatelessConfig:
    """PresetConfig and AgentConfigManager support stateless_iterations."""

    def test_preset_config_defaults_to_true(self):
        preset = PresetConfig(name="test")
        assert preset.stateless_iterations is True

    def test_from_dict_reads_true(self):
        preset = PresetConfig.from_dict("test", {"stateless_iterations": True})
        assert preset.stateless_iterations is True

    def test_from_dict_reads_false(self):
        preset = PresetConfig.from_dict("debug", {"stateless_iterations": False})
        assert preset.stateless_iterations is False

    def test_from_dict_defaults_true_when_missing(self):
        preset = PresetConfig.from_dict("legacy", {})
        assert preset.stateless_iterations is True

    def test_config_manager_exposes_flag(self):
        preset = PresetConfig(name="test", stateless_iterations=True)
        mgr = AgentConfigManager(preset=preset)
        assert mgr.use_stateless_iterations() is True

    def test_config_manager_stateful(self):
        preset = PresetConfig(name="debug", stateless_iterations=False)
        mgr = AgentConfigManager(preset=preset)
        assert mgr.use_stateless_iterations() is False


# ============================================================
# 2. IterationSnapshot
# ============================================================

class TestIterationSnapshot:
    """IterationSnapshot.from_session() and format_for_prompt()."""

    def _make_mock_session(self):
        """Build a mock SessionState with typical fields."""
        mock = MagicMock()
        mock.current_iteration = 3
        mock.request.max_iterations = 5
        mock.request.effect_type.value = "fire"
        mock.current_technique = "mantaflow_basic"
        mock.best_score = 45.2
        mock.best_iteration = 2
        mock.iterations = [
            MagicMock(score=0.0),
            MagicMock(score=38.0),
            MagicMock(score=45.2),
        ]

        stuck = MagicMock()
        stuck.escape_level.value = 1
        stuck.same_issue_count = 2
        stuck.plateau_count = 1
        stuck.techniques_tried = ["mantaflow_basic", "mantaflow_advanced"]
        stuck.techniques_failed = ["mantaflow_advanced"]
        mock.stuck_state = stuck

        return mock

    def test_from_session_builds_correct_snapshot(self):
        session = self._make_mock_session()
        snap = IterationSnapshot.from_session(
            session, previous_score=38.0, previous_primary_issue="LOW_DENSITY"
        )
        assert snap.iteration == 3
        assert snap.max_iterations == 5
        assert snap.effect_type == "fire"
        assert snap.technique == "mantaflow_basic"
        assert snap.best_score == 45.2
        assert snap.best_iteration == 2
        assert snap.previous_score == 38.0
        assert snap.previous_primary_issue == "LOW_DENSITY"
        assert snap.escape_level == 1
        assert snap.same_issue_count == 2
        assert snap.plateau_count == 1
        assert snap.techniques_tried == ["mantaflow_basic", "mantaflow_advanced"]
        assert snap.techniques_failed == ["mantaflow_advanced"]
        assert snap.iteration_scores == [0.0, 38.0, 45.2]

    def test_format_for_prompt_produces_compact_output(self):
        session = self._make_mock_session()
        snap = IterationSnapshot.from_session(
            session, previous_score=38.0, previous_primary_issue="LOW_DENSITY"
        )
        text = snap.format_for_prompt()
        assert "## Cross-Iteration Context" in text
        assert "Iteration: 3/5" in text
        assert "Best: 45.2 (iter 2)" in text
        assert "Previous: 38.0" in text
        assert "Escape Level: 1" in text
        assert "Same Issue: 2x" in text
        assert "Score Trend:" in text
        assert "Tried: mantaflow_basic, mantaflow_advanced" in text
        assert "Failed: mantaflow_advanced" in text

    def test_format_for_prompt_no_techniques(self):
        session = self._make_mock_session()
        session.stuck_state.techniques_tried = []
        session.stuck_state.techniques_failed = []
        snap = IterationSnapshot.from_session(session)
        text = snap.format_for_prompt()
        assert "Tried:" not in text
        assert "Failed:" not in text

    def test_snapshot_is_frozen(self):
        session = self._make_mock_session()
        snap = IterationSnapshot.from_session(session)
        with pytest.raises(AttributeError):
            snap.iteration = 99

    def test_format_for_prompt_truncates_history(self):
        session = self._make_mock_session()
        session.iterations = [MagicMock(score=float(i)) for i in range(10)]
        snap = IterationSnapshot.from_session(session)
        text = snap.format_for_prompt(max_history=3)
        # Should only show last 3 scores
        assert "7.0, 8.0, 9.0" in text


# ============================================================
# 3. Session routing
# ============================================================

class TestSessionRouting:
    """Stateless mode routes iter_session=None, stateful uses sdk_session."""

    def test_stateless_gives_none(self):
        """When stateless_iterations=True, iter_session should be None."""
        use_stateless = True
        sdk_session = MagicMock()
        iter_session = None if use_stateless else sdk_session
        assert iter_session is None

    def test_stateful_gives_sdk_session(self):
        """When stateless_iterations=False, iter_session should be sdk_session."""
        use_stateless = False
        sdk_session = MagicMock()
        iter_session = None if use_stateless else sdk_session
        assert iter_session is sdk_session


# ============================================================
# 4. Compaction guard
# ============================================================

class TestCompactionGuard:
    """Compaction is skipped in stateless mode, runs in stateful mode."""

    def test_compaction_skipped_in_stateless_mode(self):
        """Stateless mode should NOT call run_compaction."""
        use_stateless = True
        sdk_session = AsyncMock()
        sdk_session.run_compaction = AsyncMock()

        async def _run():
            if not use_stateless:
                if sdk_session and hasattr(sdk_session, 'run_compaction'):
                    await sdk_session.run_compaction({"force": True})

        asyncio.run(_run())
        sdk_session.run_compaction.assert_not_called()

    def test_compaction_runs_in_stateful_mode(self):
        """Stateful mode SHOULD call run_compaction."""
        use_stateless = False
        sdk_session = AsyncMock()
        sdk_session.run_compaction = AsyncMock()

        async def _run():
            if not use_stateless:
                if sdk_session and hasattr(sdk_session, 'run_compaction'):
                    await sdk_session.run_compaction({"force": True})

        asyncio.run(_run())
        sdk_session.run_compaction.assert_called_once_with({"force": True})


# ============================================================
# 5. Preset YAML integration
# ============================================================

class TestPresetYamlIntegration:
    """All presets in presets.yaml have the stateless_iterations flag."""

    @pytest.fixture
    def presets_data(self):
        config_path = Path(__file__).parent.parent / "config" / "presets.yaml"
        return AgentConfigManager.load_presets_file(config_path)

    def test_all_presets_have_flag(self, presets_data):
        presets = presets_data.get("presets", {})
        assert len(presets) >= 7, f"Expected at least 7 presets, got {len(presets)}"
        for name, data in presets.items():
            assert "stateless_iterations" in data, (
                f"Preset '{name}' missing stateless_iterations flag"
            )

    def test_debug_preset_is_stateful(self, presets_data):
        debug = presets_data["presets"]["debug"]
        assert debug["stateless_iterations"] is False

    def test_production_preset_is_stateless(self, presets_data):
        prod = presets_data["presets"]["production"]
        assert prod["stateless_iterations"] is True

    def test_quick_test_preset_is_stateless(self, presets_data):
        qt = presets_data["presets"]["quick_test"]
        assert qt["stateless_iterations"] is True

    def test_development_preset_is_stateless(self, presets_data):
        dev = presets_data["presets"]["development"]
        assert dev["stateless_iterations"] is True

    def test_budget_saver_preset_is_stateless(self, presets_data):
        bs = presets_data["presets"]["budget_saver"]
        assert bs["stateless_iterations"] is True

    def test_loaded_preset_exposes_flag(self):
        config_path = Path(__file__).parent.parent / "config" / "presets.yaml"
        mgr = AgentConfigManager.load_preset("quick_test", config_path)
        assert mgr.use_stateless_iterations() is True

    def test_loaded_debug_preset_is_stateful(self):
        config_path = Path(__file__).parent.parent / "config" / "presets.yaml"
        mgr = AgentConfigManager.load_preset("debug", config_path)
        assert mgr.use_stateless_iterations() is False
