"""
Agent Configuration System for VFX Orchestrator.

Provides preset-based configuration for all agents with per-agent overrides.

Usage:
    # Load preset from environment variable
    config = AgentConfigManager.from_env()  # Reads ORCHESTRATOR_PRESET

    # Or load specific preset
    config = AgentConfigManager.load_preset("quick_test")

    # Get settings for an agent
    settings = config.get_agent_settings("script_writer")

    # Use in agent creation
    agent = Agent(
        name="Script Writer",
        model=settings.model,
        model_settings=ModelSettings(
            temperature=settings.temperature,
            reasoning={"effort": settings.reasoning_effort} if settings.reasoning_effort != "none" else None,
        ),
        ...
    )
"""

from .agent_config import (
    AgentSettings,
    PresetConfig,
    AgentConfigManager,
    get_config,
)

__all__ = [
    "AgentSettings",
    "PresetConfig",
    "AgentConfigManager",
    "get_config",
]
