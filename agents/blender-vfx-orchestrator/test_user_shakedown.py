"""
User-requested shakedown test for Blender VFX Orchestrator.

Parameters:
- Model: gpt-5-mini
- Reasoning: low
- Verbosity: low
- Tracing: verbose-tracing
- Iterations: 1
- Knowledge Sources: LLM_PRIMER, SDK_ENFORCEMENT_PROTOCOL, VERSION_TRUTH
"""

import asyncio
import os
import sys
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set requested environment variables
os.environ["ORCHESTRATOR_MODEL"] = "gpt-5-mini"
os.environ["VERBOSE_TRACING"] = "1"

# Force configuration to match user request
from config.agent_config import AgentConfigManager, PresetConfig, reset_config

# Create a custom preset for this test
USER_PRESET = PresetConfig(
    name="user_shakedown",
    description="User-requested shakedown test",
    default_model="gpt-5-mini",
    reasoning_effort="low",
    verbosity="low",
    max_output_tokens=2000,
    max_turns=5,
    max_iterations=1,
    verbose=True
)

# Enable verbose tracing
from tracing import enable_verbose_tracing
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"traces/user_shakedown_{timestamp}.jsonl"
enable_verbose_tracing(
    log_file=log_file,
    console_output=True,
    include_span_data=True,
)

from orchestrator import BlenderVFXOrchestrator, create_session_from_request, generate_session_id
from models.shared_context import AssetRequest, EffectType
from agents import Runner, trace


async def run_user_shakedown():
    print("=" * 70)
    print("USER SHAKEDOWN TEST")
    print(f"Model: {USER_PRESET.default_model} | Reasoning: {USER_PRESET.reasoning_effort}")
    print(f"Verbosity: {USER_PRESET.verbosity} | Iterations: {USER_PRESET.max_iterations}")
    print("=" * 70)
    print()

    # Step 1: Read the requested documentation files
    docs_paths = [
        "agents/blender-vfx-orchestrator/docs/LLM_PRIMER_2026-01-26.md",
        "agents/blender-vfx-orchestrator/docs/SDK_ENFORCEMENT_PROTOCOL.md",
        "agents/blender-vfx-orchestrator/docs/VERSION_TRUTH.md"
    ]
    
    docs_content = ""
    for path in docs_paths:
        p = Path(path)
        if p.exists():
            print(f"[1] Loading context from {p.name}...")
            docs_content += f"\n\n--- {p.name} ---\n"
            docs_content += p.read_text()
        else:
            print(f"[!] Warning: {path} not found")

    # Step 2: Initialize Orchestrator
    print("\n[2] Initializing orchestrator...")
    orchestrator = BlenderVFXOrchestrator()
    
    # Manually inject our custom config to bypass the preset loader
    from config import agent_config
    agent_config._global_config = AgentConfigManager(preset=USER_PRESET)
    
    await orchestrator.initialize()
    print("    âœ” Orchestrator initialized")

    # Step 3: Create a test request
    # Use a simple effect to ensure success in 1 iteration
    request = AssetRequest(
        asset_name="user_shakedown_fire",
        description="Simple fire effect for shakedown test. Follow the loaded documentation strictly.",
        effect_type=EffectType.FIRE,
        resolution=32,
        frame_start=1,
        frame_end=10,
        quality_threshold=30.0,
        max_iterations=1,
    )

    # Step 4: Run the pipeline
    print(f"\n[3] Running asset pipeline for '{request.asset_name}'...")
    print(f"    Knowledge payload size: {len(docs_content)} characters")
    
    # We'll prepend the docs to the request description to ensure the agent sees them
    # as the user's "onboarding" context.
    request.description = f"CONTEXT FROM DOCUMENTATION:\n{docs_content}\n\nUSER REQUEST: {request.description}"

    try:
        with trace("User Shakedown Run"):
            result = await orchestrator.create_asset_pipeline(request)
        
        print("\n" + "=" * 70)
        print("SHAKEDOWN COMPLETE")
        print(f"Status: {result.status.value}")
        print(f"Score: {result.best_score:.1f}")
        print(f"Trace saved to: {log_file}")
        print("=" * 70)
        
        return result
    except Exception as e:
        print(f"[!] Error: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        await orchestrator.close()


if __name__ == "__main__":
    asyncio.run(run_user_shakedown())
