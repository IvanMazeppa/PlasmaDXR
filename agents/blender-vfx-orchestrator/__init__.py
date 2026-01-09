"""
Blender VFX Orchestrator - Unified Autonomous Agent System

Consolidates 11 MCP servers into a single GPT-5.2 powered orchestrator
using the OpenAI Agents SDK for multi-agent coordination.

Key Agents:
- ScriptWriter: Generates/modifies Blender Python scripts
- Executor: Runs Blender simulations and captures outputs
- QualityAnalyst: Evaluates renders using ML metrics
- LearningAgent: Tracks experiments and learns from outcomes
- DocsExpert: Searches Blender documentation via MCP

Usage:
    from blender_vfx_orchestrator import BlenderVFXOrchestrator

    orchestrator = BlenderVFXOrchestrator()
    await orchestrator.initialize()
    result = await orchestrator.create_asset(
        name="supernova_v1",
        description="A bright stellar explosion with expanding shockwave",
        effect_type="explosion",
        max_iterations=5
    )
"""

__version__ = "0.1.0"
__author__ = "PlasmaDX Team"

from .orchestrator import BlenderVFXOrchestrator
from .models.shared_context import SharedContext, IterationResult, AssetRequest, SessionState

__all__ = [
    "BlenderVFXOrchestrator",
    "SharedContext",
    "IterationResult",
    "AssetRequest",
    "SessionState",
]
