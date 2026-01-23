"""Quick 3-iteration E2E test to verify script modifications work."""

import argparse
import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ.setdefault('ORCHESTRATOR_MODEL', 'gpt-5.2')

from models.shared_context import AssetRequest, EffectType, SharedContext
from orchestrator import BlenderVFXOrchestrator, create_session_from_request, generate_session_id
from agents import enable_verbose_stdout_logging

enable_verbose_stdout_logging()


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


async def run_quick_test(name: str, effect: str, max_iterations: int = 3, timeout: int = 300):
    print('=' * 70)
    print('QUICK E2E TEST - 3 ITERATIONS - VERIFY SCRIPT MODIFICATIONS')
    print('=' * 70)

    config = EFFECT_CONFIGS.get(effect.lower(), EFFECT_CONFIGS['sun'])

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

    print(f'Running with max_iterations={request.max_iterations}')
    print(f'Effect type: {request.effect_type.value}')

    orchestrator = BlenderVFXOrchestrator()
    await orchestrator.initialize()
    print('Orchestrator initialized')

    session_id = generate_session_id(request.asset_name)
    context = create_session_from_request(request, session_id)
    print(f'Session: {session_id}')

    print('Starting create_asset_pipeline...')
    result = await orchestrator.create_asset_pipeline(request)

    print()
    print('=' * 70)
    print('TEST COMPLETE')
    print('=' * 70)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Quick E2E test for VFX orchestrator')
    parser.add_argument('--name', type=str, default='quick_test_v1', help='Asset name')
    parser.add_argument('--effect', type=str, default='sun',
                        choices=['sun', 'fire', 'explosion', 'smoke', 'nebula'],
                        help='Effect type to test')
    parser.add_argument('--iterations', type=int, default=3, help='Max iterations')
    parser.add_argument('--timeout', type=int, default=300, help='Timeout in seconds')

    args = parser.parse_args()

    try:
        result = asyncio.run(run_quick_test(
            name=args.name,
            effect=args.effect,
            max_iterations=args.iterations,
            timeout=args.timeout
        ))
        print(f'Final result: {result}')
    except Exception as e:
        print(f'ERROR: {e}')
        import traceback
        traceback.print_exc()
