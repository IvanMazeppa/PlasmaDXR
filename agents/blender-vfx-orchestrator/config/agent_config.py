"""
Agent Configuration Manager.

Loads presets from YAML and provides agent-specific settings with overrides.

GPT-5 Parameter Compatibility (from OpenAI Platform docs):
- gpt-5.2/gpt-5.1: temperature/top_p/logprobs ONLY supported with reasoning_effort=none
- gpt-5/gpt-5-mini/gpt-5-nano: NO temperature support, use text.verbosity and max_output_tokens
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
TEMPERATURE_SUPPORTED_MODELS = {"gpt-5.2", "gpt-5.1"}

# Models that DON'T support temperature at all
NO_TEMPERATURE_MODELS = {"gpt-5", "gpt-5-mini", "gpt-5-nano"}


@dataclass
class AgentSettings:
    """Settings for a single agent."""
    model: str = "gpt-5-nano"
    reasoning_effort: str = "none"  # none, low, medium, high, xhigh
    temperature: Optional[float] = None  # Only for gpt-5.2/5.1 with reasoning=none
    verbosity: str = "medium"  # low, medium, high - for text output control
    max_output_tokens: Optional[int] = None
    max_turns: int = 10
    verbose: bool = False

    def to_model_settings(self) -> Dict[str, Any]:
        """
        Convert to SDK ModelSettings kwargs.

        Handles parameter compatibility based on model type:
        - gpt-5.2/gpt-5.1: temperature only with reasoning_effort=none
        - gpt-5/gpt-5-mini/gpt-5-nano: use text.verbosity, max_output_tokens
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

        # Handle temperature (only for gpt-5.2/5.1 with reasoning=none)
        if model_supports_temperature and self.reasoning_effort == "none":
            if self.temperature is not None:
                settings["temperature"] = self.temperature

        # Handle text verbosity (alternative to temperature for gpt-5-mini/nano)
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
    _raw_presets: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def load_presets_file(cls, path: Path = PRESETS_FILE) -> Dict[str, Any]:
        """Load the presets YAML file."""
        if not path.exists():
            logger.warning(f"Presets file not found: {path}, using defaults")
            return {"presets": {}, "agent_overrides": {}}

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

        if preset_name not in presets:
            available = list(presets.keys())
            logger.warning(
                f"Preset '{preset_name}' not found. Available: {available}. "
                f"Using 'development' defaults."
            )
            preset_config = PresetConfig(name=preset_name)
        else:
            preset_config = PresetConfig.from_dict(preset_name, presets[preset_name])

        logger.info(f"Loaded preset: {preset_name} ({preset_config.description})")

        return cls(
            preset=preset_config,
            agent_overrides=overrides,
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

        # Apply agent-specific overrides
        overrides = self.agent_overrides.get(agent_name, {})
        if overrides:
            if "model" in overrides:
                settings.model = overrides["model"]
            if "reasoning_effort" in overrides:
                settings.reasoning_effort = overrides["reasoning_effort"]
            if "temperature" in overrides:
                settings.temperature = overrides["temperature"]
            if "verbosity" in overrides:
                settings.verbosity = overrides["verbosity"]
            if "max_output_tokens" in overrides:
                settings.max_output_tokens = overrides["max_output_tokens"]
            if "max_turns" in overrides:
                settings.max_turns = overrides["max_turns"]
            if "verbose" in overrides:
                settings.verbose = overrides["verbose"]

        return settings

    def get_max_iterations(self) -> int:
        """Get max iterations from preset."""
        return self.preset.max_iterations

    def is_verbose(self) -> bool:
        """Check if verbose mode is enabled."""
        return self.preset.verbose

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
