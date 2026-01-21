"""Quick 3-iteration E2E test to verify script modifications work."""

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


async def run_quick_test():
    print('=' * 70)
    print('QUICK E2E TEST - 3 ITERATIONS - VERIFY SCRIPT MODIFICATIONS')
    print('=' * 70)

    request = AssetRequest(
        asset_name='sun_fix_test_v1',
        description='NASA SDO-style sun with surface granulation and corona',
        effect_type=EffectType.SUN,
        resolution=64,
        frame_start=1,
        frame_end=25,
        quality_threshold=70.0,
        max_iterations=3,
    )

    print(f'Running with max_iterations={request.max_iterations}')

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
    try:
        result = asyncio.run(run_quick_test())
        print(f'Final result: {result}')
    except Exception as e:
        print(f'ERROR: {e}')
        import traceback
        traceback.print_exc()
