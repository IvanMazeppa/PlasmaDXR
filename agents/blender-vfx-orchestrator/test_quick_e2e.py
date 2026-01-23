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

    request = AssetRequest(
        asset_name=name,
        description=config['description'],
        effect_type=config['effect_type'],
        resolution=64,
        frame_start=1,
        frame_end=25,
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
                        choices=['sun', 'fire', 'explosion', 'smoke', 'nebula'],
                        help='Effect type to test')
    parser.add_argument('--preset', type=str, default='quick_test',
                        help='Config preset (quick_test, development, production, budget_saver, debug)')
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
