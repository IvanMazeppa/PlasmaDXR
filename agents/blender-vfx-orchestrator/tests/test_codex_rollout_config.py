"""
Tests for codex rollout preset wiring.

Verifies that:
- codex_upgrade preset exists in YAML
- preset-specific overrides can supersede global agent_overrides
- non-rollout presets keep existing model assignments
- codex models use text verbosity path (no temperature)
"""

import os
import sys
import unittest
from pathlib import Path

# Add parent dir for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.agent_config import AgentConfigManager, AgentSettings


class TestCodexRolloutPreset(unittest.TestCase):
    """Validate safe codex rollout behavior in config system."""

    @classmethod
    def setUpClass(cls):
        cls.config_path = Path(__file__).parent.parent / "config" / "presets.yaml"

    def test_codex_upgrade_preset_exists(self):
        """presets.yaml should define codex_upgrade preset."""
        data = AgentConfigManager.load_presets_file(self.config_path)
        self.assertIn("codex_upgrade", data.get("presets", {}))

    def test_codex_upgrade_overrides_script_writer_model(self):
        """Preset-specific overrides should move script_writer to gpt-5.3-codex."""
        mgr = AgentConfigManager.load_preset("codex_upgrade", self.config_path)
        settings = mgr.get_agent_settings("script_writer")
        self.assertEqual(settings.model, "gpt-5.3-codex")
        self.assertEqual(settings.reasoning_effort, "high")

    def test_codex_upgrade_keeps_non_target_agent_on_global_default(self):
        """Agents not in codex_upgrade override block keep global override model."""
        mgr = AgentConfigManager.load_preset("codex_upgrade", self.config_path)
        settings = mgr.get_agent_settings("quality_analyst")
        self.assertEqual(settings.model, "gpt-5.2")

    def test_non_rollout_preset_keeps_script_writer_on_gpt_5_2(self):
        """Existing presets should remain unchanged."""
        mgr = AgentConfigManager.load_preset("production", self.config_path)
        settings = mgr.get_agent_settings("script_writer")
        self.assertEqual(settings.model, "gpt-5.2")

    def test_codex_model_settings_exclude_temperature(self):
        """Codex models should map to reasoning + text verbosity without temperature."""
        settings = AgentSettings(
            model="gpt-5.3-codex",
            reasoning_effort="high",
            temperature=0.7,  # Should be ignored for codex models
            verbosity="medium",
        )
        model_settings = settings.to_model_settings()
        self.assertNotIn("temperature", model_settings)
        self.assertEqual(model_settings.get("reasoning", {}).get("effort"), "high")
        self.assertEqual(model_settings.get("text", {}).get("verbosity"), "medium")


if __name__ == "__main__":
    unittest.main()
