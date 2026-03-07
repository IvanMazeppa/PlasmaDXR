"""Quick E2E test to verify orchestrator pipeline works."""

import argparse
import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from models.shared_context import AssetRequest, EffectType, SharedContext
from orchestrator import BlenderVFXOrchestrator, create_session_from_request, generate_session_id
from config import AgentConfigManager
from agents import enable_verbose_stdout_logging


# Effect type configurations
EFFECT_CONFIGS = {
    'sun': {
        'effect_type': EffectType.SUN,
        'description': 'NASA SDO-style sun with surface granulation and corona',
    },
    'fire': {
        'effect_type': EffectType.FIRE,
        'description': 'Small orange fire plume using Mantaflow with blackbody shading',
    },
    'explosion': {
        'effect_type': EffectType.EXPLOSION,
        'description': 'Fiery explosion with expanding fireball and rising smoke',
    },
    'smoke': {
        'effect_type': EffectType.SMOKE,
        'description': 'Billowing smoke with turbulent motion',
    },
    'nebula': {
        'effect_type': EffectType.NEBULA,
        'description': 'Colorful space nebula with swirling gases and stars',
    },
    'water': {
        'effect_type': EffectType.WATER,
        'description': """SCENE DESCRIPTION:
A stream of deep red wine pours from an unseen bottle (positioned just above frame) into an elegant stemmed wine glass resting on a dark wooden surface. The pour is steady but vigorous — not a delicate trickle, but a confident stream that creates dynamic splashing as it impacts the rising liquid surface. As the glass fills past the halfway point, wine droplets leap from the surface on impact, some catching the warm light as they arc through the air. A few droplets escape over the rim and trail down the outside of the glass.

MOOD AND LOOK:
Warm, intimate atmosphere evoking a cozy evening scene. Rich amber/golden key lighting from one side (as if from a nearby fireplace or candle cluster) creates dramatic highlights on the glass rim and wine surface, with the deep burgundy liquid glowing where light passes through it. The background falls into soft, warm darkness. The wine should exhibit convincing subsurface scattering — appearing darker at depth, more translucent and ruby-red where thin areas against the glass walls. Visible caustics pooling on the table surface beneath the glass.

CAMERA:
Close-up, slightly low angle looking up at the glass to emphasize elegance. Shallow depth of field optional.

PHYSICAL DETAILS:
Wine glass: Standard Bordeaux style, clear glass with subtle reflections
Wine color: Deep burgundy/cabernet red with ruby highlights in thin areas
Pour stream: ~8mm diameter, continuous with slight surface tension wobble
Table surface: Dark polished wood with subtle reflection of the glass base

HARD CONSTRAINTS:
Fluid simulation: Blender Mantaflow liquid domain (FLIP solver, not shader-only)
Domain type: LIQUID with mesh generation enabled
Renderer: Cycles GPU
Samples: 256 max
Frame range: 1-60 (captures fill from ~1/4 to ~3/4 full)
Resolution: 64-96 for good splash detail
use_plane_init: True (for inflow emitter)
cache_type: ALL""",
    },
}


def list_presets():
    """List available presets."""
    config = AgentConfigManager.load_preset("development")  # Load any to get list
    print("\nAvailable presets:")
    print("-" * 50)
    for name, desc in config.list_presets().items():
        print(f"  {name:15} - {desc}")
    print()


async def run_quick_test(
    name: str,
    effect: str,
    preset: str,
    max_iterations: int = 3,
    verbose: bool = False
):
    # Set preset via environment BEFORE importing orchestrator config
    os.environ['ORCHESTRATOR_PRESET'] = preset

    # Reset config to pick up new preset
    from config.agent_config import reset_config
    reset_config()

    # Enable verbose logging if requested
    if verbose:
        enable_verbose_stdout_logging()

    print('=' * 70)
    print(f'VFX ORCHESTRATOR E2E TEST')
    print(f'Preset: {preset} | Effect: {effect} | Iterations: {max_iterations}')
    print('=' * 70)

    config = EFFECT_CONFIGS.get(effect.lower(), EFFECT_CONFIGS['sun'])

    # Get max_iterations from preset if not overridden
    preset_config = AgentConfigManager.load_preset(preset)
    if max_iterations == 3:  # Default value, use preset
        max_iterations = preset_config.get_max_iterations()

    # Effect-specific frame ranges
    frame_end = 60 if effect.lower() == 'water' else 25
    resolution = 96 if effect.lower() == 'water' else 64

    request = AssetRequest(
        asset_name=name,
        description=config['description'],
        effect_type=config['effect_type'],
        resolution=resolution,
        frame_start=1,
        frame_end=frame_end,
        quality_threshold=70.0,
        max_iterations=max_iterations,
    )

    print(f'\nEffect: {request.effect_type.value}')
    print(f'Max iterations: {request.max_iterations}')
    print(f'Model: {preset_config.preset.default_model}')
    print(f'Reasoning: {preset_config.preset.reasoning_effort}')
    print()

    orchestrator = BlenderVFXOrchestrator()
    await orchestrator.initialize()

    session_id = generate_session_id(request.asset_name)
    print(f'Session: {session_id}')

    print('\nStarting create_asset_pipeline...\n')
    result = await orchestrator.create_asset_pipeline(request)

    print()
    print('=' * 70)
    print('TEST COMPLETE')
    print('=' * 70)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Quick E2E test for VFX orchestrator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with budget_saver preset
  python test_quick_e2e.py --preset budget_saver --effect fire

  # Full quality test
  python test_quick_e2e.py --preset production --effect sun --name sun_hq_v1

  # List available presets
  python test_quick_e2e.py --list-presets
"""
    )
    parser.add_argument('--name', type=str, default='quick_test_v1', help='Asset name')
    parser.add_argument('--effect', type=str, default='sun',
                        choices=['sun', 'fire', 'explosion', 'smoke', 'nebula', 'water'],
                        help='Effect type to test')
    parser.add_argument('--preset', type=str, default='quick_test',
                        help='Config preset (quick_test, development, codex_upgrade, production, budget_saver, debug)')
    parser.add_argument('--iterations', type=int, default=3, help='Max iterations (overrides preset)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose SDK logging')
    parser.add_argument('--list-presets', action='store_true', help='List available presets and exit')

    args = parser.parse_args()

    if args.list_presets:
        list_presets()
        sys.exit(0)

    try:
        result = asyncio.run(run_quick_test(
            name=args.name,
            effect=args.effect,
            preset=args.preset,
            max_iterations=args.iterations,
            verbose=args.verbose
        ))
        print(f'\nFinal result: {result}')
    except KeyboardInterrupt:
        print('\nTest interrupted by user')
    except Exception as e:
        print(f'\nERROR: {e}')
        import traceback
        traceback.print_exc()
