"""
Tests for codex rollout preset wiring.

Verifies that:
- codex_upgrade preset exists in YAML
- preset-specific overrides can supersede global agent_overrides
- non-rollout presets keep existing model assignments
- GPT-5.4 rollout settings apply the expected temperature gating
"""

import os
import sys
import unittest
from pathlib import Path

# Add parent dir for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.agent_config import AgentConfigManager, AgentSettings


class TestCodexRolloutPreset(unittest.TestCase):
    """Validate safe rollout behavior for the historical codex_upgrade preset."""

    @classmethod
    def setUpClass(cls):
        cls.config_path = Path(__file__).parent.parent / "config" / "presets.yaml"

    def test_codex_upgrade_preset_exists(self):
        """presets.yaml should define codex_upgrade preset."""
        data = AgentConfigManager.load_presets_file(self.config_path)
        self.assertIn("codex_upgrade", data.get("presets", {}))

    def test_codex_upgrade_overrides_script_writer_model(self):
        """Preset-specific overrides should move script_writer to gpt-5.4."""
        mgr = AgentConfigManager.load_preset("codex_upgrade", self.config_path)
        settings = mgr.get_agent_settings("script_writer")
        self.assertEqual(settings.model, "gpt-5.4")
        self.assertEqual(settings.reasoning_effort, "high")

    def test_codex_upgrade_keeps_non_target_agent_on_global_default(self):
        """Agents not in codex_upgrade override block keep global override model."""
        mgr = AgentConfigManager.load_preset("codex_upgrade", self.config_path)
        # quality_analyst is now upgraded to gpt-5.4 in codex_upgrade preset
        settings = mgr.get_agent_settings("quality_analyst")
        self.assertEqual(settings.model, "gpt-5.4")
        # technique_coordinator stays on default (not in override block)
        settings = mgr.get_agent_settings("technique_coordinator")
        self.assertEqual(settings.model, "gpt-5-mini")

    def test_non_rollout_preset_keeps_script_writer_on_gpt_5_2(self):
        """Existing presets should remain unchanged."""
        mgr = AgentConfigManager.load_preset("production", self.config_path)
        settings = mgr.get_agent_settings("script_writer")
        self.assertEqual(settings.model, "gpt-5.2")

    def test_rollout_model_settings_exclude_temperature_when_reasoning_enabled(self):
        """High-reasoning GPT-5.4 settings should ignore temperature."""
        settings = AgentSettings(
            model="gpt-5.4",
            reasoning_effort="high",
            temperature=0.7,  # Should be ignored when reasoning is enabled
            verbosity="medium",
        )
        model_settings = settings.to_model_settings()
        self.assertNotIn("temperature", model_settings)
        self.assertEqual(model_settings.get("reasoning", {}).get("effort"), "high")
        self.assertEqual(model_settings.get("text", {}).get("verbosity"), "medium")

    def test_gpt_5_4_supports_temperature_when_reasoning_disabled(self):
        """GPT-5.4 should allow temperature when reasoning_effort is none."""
        settings = AgentSettings(
            model="gpt-5.4",
            reasoning_effort="none",
            temperature=0.2,
        )
        model_settings = settings.to_model_settings()
        self.assertEqual(model_settings.get("temperature"), 0.2)


if __name__ == "__main__":
    unittest.main()
