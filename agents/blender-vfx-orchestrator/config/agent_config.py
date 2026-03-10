"""
Agent Configuration Manager.

Loads presets from YAML and provides agent-specific settings with overrides.

GPT-5 Parameter Compatibility (from OpenAI Platform docs):
- gpt-5.4/gpt-5.2/gpt-5.1: temperature/top_p/logprobs ONLY supported with reasoning_effort=none
- gpt-5-codex/gpt-5.3-codex/gpt-5.2-codex/gpt-5/gpt-5-mini/gpt-5-nano:
  NO temperature support, use text.verbosity and max_output_tokens
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)

# Default config file path
CONFIG_DIR = Path(__file__).parent
PRESETS_FILE = CONFIG_DIR / "presets.yaml"

# Models that support temperature (only with reasoning_effort=none)
TEMPERATURE_SUPPORTED_MODELS = {"gpt-5.4", "gpt-5.2", "gpt-5.1"}

# Models that DON'T support temperature at all
NO_TEMPERATURE_MODELS = {
    "codex",
    "gpt-5-codex",
    "gpt-5.3-codex",
    "gpt-5.2-codex",
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
}


@dataclass
class AgentSettings:
    """Settings for a single agent."""
    model: str = "gpt-5-nano"
    reasoning_effort: str = "none"  # none, low, medium, high, xhigh
    temperature: Optional[float] = None  # Only for gpt-5.4/5.2/5.1 with reasoning=none
    verbosity: str = "medium"  # low, medium, high - for text output control
    max_output_tokens: Optional[int] = None
    max_turns: int = 10
    verbose: bool = False

    def to_model_settings(self) -> Dict[str, Any]:
        """
        Convert to SDK ModelSettings kwargs.

        Handles parameter compatibility based on model type:
        - gpt-5.4/gpt-5.2/gpt-5.1: temperature only with reasoning_effort=none
        - codex/gpt-5-codex/gpt-5.3-codex/gpt-5.2-codex/gpt-5/gpt-5-mini/gpt-5-nano:
          use text.verbosity, max_output_tokens
        """
        settings = {}

        # Determine model category
        model_supports_temperature = any(
            self.model.startswith(m) for m in TEMPERATURE_SUPPORTED_MODELS
        )
        model_no_temperature = any(
            self.model == m or self.model.startswith(m + "-")
            for m in NO_TEMPERATURE_MODELS
        )

        # Handle reasoning effort
        if self.reasoning_effort != "none":
            settings["reasoning"] = {"effort": self.reasoning_effort}

        # Handle temperature (only for gpt-5.4/5.2/5.1 with reasoning=none)
        if model_supports_temperature and self.reasoning_effort == "none":
            if self.temperature is not None:
                settings["temperature"] = self.temperature

        # Handle text verbosity (alternative to temperature for codex/gpt-5/mini/nano)
        if model_no_temperature or self.reasoning_effort != "none":
            if self.verbosity:
                settings["text"] = {"verbosity": self.verbosity}

        # Handle max_output_tokens
        if self.max_output_tokens:
            settings["max_output_tokens"] = self.max_output_tokens

        return settings


@dataclass
class PresetConfig:
    """Configuration from a preset."""
    name: str
    description: str = ""
    default_model: str = "gpt-5-nano"
    reasoning_effort: str = "none"
    temperature: Optional[float] = None
    verbosity: str = "medium"
    max_output_tokens: Optional[int] = None
    max_turns: int = 10
    max_iterations: int = 3
    verbose: bool = False
    stateless_iterations: bool = True  # Phase 2B-1: Don't share SDK session across iterations
    multi_grader_eval: bool = True  # Phase 2B-3: Deterministic checks before LLM vision
    hitl_enabled: bool = True       # Phase 2B-5: Enable HITL checkpoints
    hitl_autonomy_level: int = 1    # Phase 2B-5: 0=guided, 1=supervised, 2=semi-auto, 3=autonomous, 4=full
    hitl_interactive: bool = True   # Phase 2B-5: CLI prompts vs file-based decisions
    memory_decay_enabled: bool = True  # Phase 2B-6: Ebbinghaus decay on KB entries
    effect_type_scoping: bool = True   # Phase 2B-7: KB entries scoped by effect type
    artifact_sharing: bool = True      # Phase 2B-8: Agents read artifacts via tool

    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> "PresetConfig":
        """Create PresetConfig from dictionary."""
        return cls(
            name=name,
            description=data.get("description", ""),
            default_model=data.get("default_model", "gpt-5.2"),
            reasoning_effort=data.get("reasoning_effort", "none"),
            temperature=data.get("temperature"),  # None if not specified
            verbosity=data.get("verbosity", "medium"),
            max_output_tokens=data.get("max_output_tokens"),
            max_turns=data.get("max_turns", 10),
            max_iterations=data.get("max_iterations", 3),
            verbose=data.get("verbose", False),
            stateless_iterations=data.get("stateless_iterations", True),
            multi_grader_eval=data.get("multi_grader_eval", True),
            hitl_enabled=data.get("hitl_enabled", True),
            hitl_autonomy_level=data.get("hitl_autonomy_level", 1),
            hitl_interactive=data.get("hitl_interactive", True),
            memory_decay_enabled=data.get("memory_decay_enabled", True),
            effect_type_scoping=data.get("effect_type_scoping", True),
            artifact_sharing=data.get("artifact_sharing", True),
        )


@dataclass
class AgentConfigManager:
    """
    Manages agent configurations with preset selection and per-agent overrides.

    Usage:
        config = AgentConfigManager.from_env()
        settings = config.get_agent_settings("script_writer")
    """
    preset: PresetConfig
    agent_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    # Optional per-preset overrides (e.g., codex rollout) layered on top
    # of global agent_overrides.
    preset_agent_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    _raw_presets: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load_presets_file(cls, path: Path = PRESETS_FILE) -> Dict[str, Any]:
        """Load the presets YAML file."""
        if not path.exists():
            logger.warning(f"Presets file not found: {path}, using defaults")
            return {"presets": {}, "agent_overrides": {}, "preset_agent_overrides": {}}

        with open(path, "r") as f:
            return yaml.safe_load(f)

    @classmethod
    def load_preset(
        cls,
        preset_name: str,
        presets_file: Path = PRESETS_FILE
    ) -> "AgentConfigManager":
        """
        Load a specific preset by name.

        Args:
            preset_name: Name of preset (e.g., "quick_test", "production")
            presets_file: Path to presets YAML file

        Returns:
            AgentConfigManager configured with the preset
        """
        data = cls.load_presets_file(presets_file)
        presets = data.get("presets", {})
        overrides = data.get("agent_overrides", {})
        preset_agent_overrides: Dict[str, Dict[str, Any]] = {}

        if preset_name not in presets:
            available = list(presets.keys())
            logger.warning(
                f"Preset '{preset_name}' not found. Available: {available}. "
                f"Using 'development' defaults."
            )
            preset_config = PresetConfig(name=preset_name)
        else:
            preset_data = presets[preset_name]
            preset_config = PresetConfig.from_dict(preset_name, preset_data)
            preset_agent_overrides = preset_data.get("agent_overrides", {})

        logger.info(f"Loaded preset: {preset_name} ({preset_config.description})")

        return cls(
            preset=preset_config,
            agent_overrides=overrides,
            preset_agent_overrides=preset_agent_overrides,
            _raw_presets=presets,
        )

    @classmethod
    def from_env(cls, presets_file: Path = PRESETS_FILE) -> "AgentConfigManager":
        """
        Load preset from ORCHESTRATOR_PRESET environment variable.

        Defaults to "development" if not set.
        """
        preset_name = os.getenv("ORCHESTRATOR_PRESET", "development")
        return cls.load_preset(preset_name, presets_file)

    def get_agent_settings(self, agent_name: str) -> AgentSettings:
        """
        Get settings for a specific agent.

        Applies preset defaults, then agent-specific overrides.

        Args:
            agent_name: Name of agent (e.g., "script_writer", "research_agent")

        Returns:
            AgentSettings with all overrides applied
        """
        # Start with preset defaults
        settings = AgentSettings(
            model=self.preset.default_model,
            reasoning_effort=self.preset.reasoning_effort,
            temperature=self.preset.temperature,
            verbosity=self.preset.verbosity,
            max_output_tokens=self.preset.max_output_tokens,
            max_turns=self.preset.max_turns,
            verbose=self.preset.verbose,
        )

        def _apply_overrides(override_block: Dict[str, Any]) -> None:
            if not override_block:
                return
            if "model" in override_block:
                settings.model = override_block["model"]
            if "reasoning_effort" in override_block:
                settings.reasoning_effort = override_block["reasoning_effort"]
            if "temperature" in override_block:
                settings.temperature = override_block["temperature"]
            if "verbosity" in override_block:
                settings.verbosity = override_block["verbosity"]
            if "max_output_tokens" in override_block:
                settings.max_output_tokens = override_block["max_output_tokens"]
            if "max_turns" in override_block:
                settings.max_turns = override_block["max_turns"]
            if "verbose" in override_block:
                settings.verbose = override_block["verbose"]

        # Apply global overrides first, then preset-specific rollout overrides.
        _apply_overrides(self.agent_overrides.get(agent_name, {}))
        _apply_overrides(self.preset_agent_overrides.get(agent_name, {}))

        return settings

    def get_max_iterations(self) -> int:
        """Get max iterations from preset."""
        return self.preset.max_iterations

    def is_verbose(self) -> bool:
        """Check if verbose mode is enabled."""
        return self.preset.verbose

    def use_stateless_iterations(self) -> bool:
        """Phase 2B-1: Check if stateless iterations are enabled."""
        return self.preset.stateless_iterations

    def use_multi_grader_eval(self) -> bool:
        """Phase 2B-3: Check if deterministic quality checks are enabled."""
        return self.preset.multi_grader_eval

    def use_hitl(self) -> bool:
        """Phase 2B-5: Check if HITL checkpoints are enabled."""
        return self.preset.hitl_enabled

    def get_hitl_autonomy_level(self) -> int:
        """Phase 2B-5: Get HITL autonomy level (0-4)."""
        return self.preset.hitl_autonomy_level

    def is_hitl_interactive(self) -> bool:
        """Phase 2B-5: Check if HITL uses interactive CLI prompts."""
        return self.preset.hitl_interactive

    def use_memory_decay(self) -> bool:
        """Phase 2B-6: Check if Ebbinghaus memory decay is enabled."""
        return self.preset.memory_decay_enabled

    def use_effect_type_scoping(self) -> bool:
        """Phase 2B-7: Check if KB entries are scoped by effect type."""
        return self.preset.effect_type_scoping

    def use_artifact_sharing(self) -> bool:
        """Phase 2B-8: Check if agents use artifact tools instead of inline data."""
        return self.preset.artifact_sharing

    def list_presets(self) -> Dict[str, str]:
        """List available presets with descriptions."""
        return {
            name: data.get("description", "")
            for name, data in self._raw_presets.items()
        }

    def __repr__(self) -> str:
        return (
            f"AgentConfigManager(preset='{self.preset.name}', "
            f"model={self.preset.default_model}, "
            f"reasoning={self.preset.reasoning_effort})"
        )


# Global config instance (lazy loaded)
_global_config: Optional[AgentConfigManager] = None


def get_config() -> AgentConfigManager:
    """
    Get the global config instance.

    Loads from ORCHESTRATOR_PRESET env var on first call.
    """
    global _global_config
    if _global_config is None:
        _global_config = AgentConfigManager.from_env()
    return _global_config


def reset_config() -> None:
    """Reset global config (useful for testing)."""
    global _global_config
    _global_config = None


# CLI for testing
if __name__ == "__main__":
    import sys

    preset = sys.argv[1] if len(sys.argv) > 1 else "development"
    config = AgentConfigManager.load_preset(preset)

    print(f"\n{'='*60}")
    print(f"Preset: {config.preset.name}")
    print(f"Description: {config.preset.description}")
    print(f"{'='*60}")
    print(f"\nDefault Settings:")
    print(f"  Model: {config.preset.default_model}")
    print(f"  Reasoning: {config.preset.reasoning_effort}")
    print(f"  Temperature: {config.preset.temperature}")
    print(f"  Verbosity: {config.preset.verbosity}")
    print(f"  Max Output Tokens: {config.preset.max_output_tokens}")
    print(f"  Max Turns: {config.preset.max_turns}")
    print(f"  Max Iterations: {config.preset.max_iterations}")

    print(f"\nAgent-Specific Settings:")
    for agent in ["research_agent", "script_writer", "executor",
                  "quality_analyst", "learning_agent",
                  "technique_coordinator", "modification_coordinator"]:
        settings = config.get_agent_settings(agent)
        model_settings = settings.to_model_settings()
        print(f"  {agent}:")
        print(f"    model={settings.model}, reasoning={settings.reasoning_effort}")
        print(f"    SDK settings: {model_settings}")

    print(f"\nAvailable Presets:")
    for name, desc in config.list_presets().items():
        marker = " <-- active" if name == preset else ""
        print(f"  {name}: {desc}{marker}")
    print()
