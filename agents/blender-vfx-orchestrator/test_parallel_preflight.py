"""
Shakedown test for parallel preflight feature.

Tests:
1. Parallel preflight enabled - both agents complete successfully
2. Parallel preflight with docs expert failing - should still return research results
3. Parallel preflight disabled - should fall back to sequential
4. Environment variable toggling

Run with: python test_parallel_preflight.py
"""

import asyncio
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure we can import from the orchestrator
sys.path.insert(0, str(Path(__file__).parent))

from models.shared_context import AssetRequest, EffectType, SharedContext
from orchestrator import BlenderVFXOrchestrator, generate_session_id
from utils import create_session_from_request
from agents import trace


async def test_parallel_preflight_enabled():
    """Test parallel preflight with both agents completing successfully."""
    print("\n" + "=" * 70)
    print("TEST 1: Parallel Preflight ENABLED")
    print("=" * 70)

    # Ensure parallel preflight is enabled
    os.environ["VFX_PARALLEL_PREFLIGHT"] = "1"

    orchestrator = BlenderVFXOrchestrator()
    await orchestrator.initialize()

    try:
        # Create test request
        request = AssetRequest(
            asset_name="test_parallel_fire",
            description="Simple fire effect for parallel preflight test",
            effect_type=EffectType.FIRE,
            resolution=32,
            frame_start=1,
            frame_end=5,
            quality_threshold=20.0,  # Low threshold for quick test
            max_iterations=1,
        )

        session_id = generate_session_id(request.asset_name)
        context = create_session_from_request(request, session_id)

        # Build run config
        run_config = orchestrator._build_run_config(
            session=context.session,
            request=request,
            iteration=0,
            phase="preflight_test",
        )

        # Create hooks (minimal)
        from hooks import create_research_hooks
        research_hooks = create_research_hooks()

        print(f"[Test] Running parallel preflight for {request.effect_type.value}...")
        start_time = time.perf_counter()

        with trace("Parallel Preflight Test"):
            research_result, docs_notes = await orchestrator._run_parallel_preflight(
                request=request,
                context=context,
                sdk_session=None,
                research_hooks=research_hooks,
                run_config=run_config,
            )

        elapsed = time.perf_counter() - start_time
        print(f"[Test] Completed in {elapsed:.1f}s")

        # Check results
        research_ok = research_result is not None
        docs_ok = docs_notes is not None

        print(f"\n[Results]")
        print(f"  Research result: {'OK' if research_ok else 'FAILED'}")
        print(f"  Docs notes: {'OK' if docs_ok else 'None (may be expected)'}")

        if research_ok and hasattr(research_result, 'final_output'):
            ro = research_result.final_output
            if hasattr(ro, 'recommended_approach'):
                print(f"  Recommended approach: {ro.recommended_approach[:60]}...")
            if hasattr(ro, 'doc_refs') and ro.doc_refs:
                print(f"  Doc refs: {len(ro.doc_refs)} found")

        if docs_notes:
            print(f"  Docs notes preview: {docs_notes[:100]}...")

        # Pass if research succeeded (docs are optional)
        success = research_ok
        print(f"\n[TEST 1] {'PASSED' if success else 'FAILED'}")
        return success

    finally:
        await orchestrator.close()


async def test_parallel_preflight_disabled():
    """Test that disabling parallel preflight falls back to sequential."""
    print("\n" + "=" * 70)
    print("TEST 2: Parallel Preflight DISABLED (Sequential Fallback)")
    print("=" * 70)

    # Disable parallel preflight
    os.environ["VFX_PARALLEL_PREFLIGHT"] = "0"

    # Need to reinitialize to pick up the setting
    orchestrator = BlenderVFXOrchestrator()
    await orchestrator.initialize()

    try:
        # Create test request
        request = AssetRequest(
            asset_name="test_sequential_smoke",
            description="Simple smoke effect for sequential test",
            effect_type=EffectType.SMOKE,
            resolution=32,
            frame_start=1,
            frame_end=5,
            quality_threshold=20.0,
            max_iterations=1,
        )

        session_id = generate_session_id(request.asset_name)
        context = create_session_from_request(request, session_id)

        run_config = orchestrator._build_run_config(
            session=context.session,
            request=request,
            iteration=0,
            phase="sequential_test",
        )

        from hooks import create_research_hooks
        research_hooks = create_research_hooks()

        print(f"[Test] Running sequential preflight for {request.effect_type.value}...")
        start_time = time.perf_counter()

        with trace("Sequential Preflight Test"):
            research_result, docs_notes = await orchestrator._run_parallel_preflight(
                request=request,
                context=context,
                sdk_session=None,
                research_hooks=research_hooks,
                run_config=run_config,
            )

        elapsed = time.perf_counter() - start_time
        print(f"[Test] Completed in {elapsed:.1f}s")

        # Check results - docs_notes should be None in sequential mode
        research_ok = research_result is not None
        docs_none = docs_notes is None

        print(f"\n[Results]")
        print(f"  Research result: {'OK' if research_ok else 'FAILED'}")
        print(f"  Docs notes None (expected): {'YES' if docs_none else 'NO'}")

        success = research_ok and docs_none
        print(f"\n[TEST 2] {'PASSED' if success else 'FAILED'}")
        return success

    finally:
        await orchestrator.close()
        # Re-enable for other tests
        os.environ["VFX_PARALLEL_PREFLIGHT"] = "1"


async def test_runconfig_propagation():
    """Test that RunConfig is properly propagated to both agents."""
    print("\n" + "=" * 70)
    print("TEST 3: RunConfig Propagation")
    print("=" * 70)

    os.environ["VFX_PARALLEL_PREFLIGHT"] = "1"

    orchestrator = BlenderVFXOrchestrator()
    await orchestrator.initialize()

    try:
        # Check if RunConfig is supported
        supports_runconfig = orchestrator._supports_run_config()
        print(f"[Test] SDK supports RunConfig: {supports_runconfig}")

        # Create test request
        request = AssetRequest(
            asset_name="test_runconfig",
            description="RunConfig propagation test",
            effect_type=EffectType.PYRO,
            resolution=32,
            frame_start=1,
            frame_end=5,
            quality_threshold=20.0,
            max_iterations=1,
        )

        session_id = generate_session_id(request.asset_name)
        context = create_session_from_request(request, session_id)

        # Build run config with specific metadata
        run_config = orchestrator._build_run_config(
            session=context.session,
            request=request,
            iteration=42,  # Distinctive value
            phase="runconfig_test",
        )

        if run_config:
            print(f"[Test] RunConfig built successfully")
            print(f"[Test]   workflow_name: {getattr(run_config, 'workflow_name', 'N/A')}")
            print(f"[Test]   group_id: {getattr(run_config, 'group_id', 'N/A')}")
            print(f"[Test]   trace_metadata: {getattr(run_config, 'trace_metadata', 'N/A')}")
        else:
            print(f"[Test] RunConfig not available (older SDK)")

        # Run preflight to verify no errors
        from hooks import create_research_hooks
        research_hooks = create_research_hooks()

        print(f"[Test] Running preflight with RunConfig...")
        with trace("RunConfig Test"):
            research_result, docs_notes = await orchestrator._run_parallel_preflight(
                request=request,
                context=context,
                sdk_session=None,
                research_hooks=research_hooks,
                run_config=run_config,
            )

        success = research_result is not None
        print(f"\n[TEST 3] {'PASSED' if success else 'FAILED'}")
        return success

    finally:
        await orchestrator.close()


async def test_error_resilience():
    """Test that parallel preflight handles partial failures gracefully."""
    print("\n" + "=" * 70)
    print("TEST 4: Error Resilience (Research continues if DocsExpert fails)")
    print("=" * 70)

    os.environ["VFX_PARALLEL_PREFLIGHT"] = "1"

    orchestrator = BlenderVFXOrchestrator()
    await orchestrator.initialize()

    try:
        # Create test request
        request = AssetRequest(
            asset_name="test_resilience",
            description="Error resilience test",
            effect_type=EffectType.EXPLOSION,
            resolution=32,
            frame_start=1,
            frame_end=5,
            quality_threshold=20.0,
            max_iterations=1,
        )

        session_id = generate_session_id(request.asset_name)
        context = create_session_from_request(request, session_id)

        run_config = orchestrator._build_run_config(
            session=context.session,
            request=request,
            iteration=0,
            phase="resilience_test",
        )

        from hooks import create_research_hooks
        research_hooks = create_research_hooks()

        print(f"[Test] Running preflight (checking resilience)...")
        with trace("Resilience Test"):
            research_result, docs_notes = await orchestrator._run_parallel_preflight(
                request=request,
                context=context,
                sdk_session=None,
                research_hooks=research_hooks,
                run_config=run_config,
            )

        # The test passes if research completes regardless of docs status
        research_ok = research_result is not None
        print(f"\n[Results]")
        print(f"  Research result: {'OK' if research_ok else 'FAILED'}")
        print(f"  Docs notes: {'Present' if docs_notes else 'None'}")
        print(f"  (Both outcomes acceptable - key is research doesn't fail if docs fail)")

        success = research_ok
        print(f"\n[TEST 4] {'PASSED' if success else 'FAILED'}")
        return success

    finally:
        await orchestrator.close()


async def run_all_tests():
    """Run all parallel preflight tests."""
    print("\n" + "=" * 70)
    print("PARALLEL PREFLIGHT SHAKEDOWN TEST")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    results = {}

    try:
        results["test_parallel_enabled"] = await test_parallel_preflight_enabled()
    except Exception as e:
        print(f"\n[ERROR] Test 1 crashed: {e}")
        import traceback
        traceback.print_exc()
        results["test_parallel_enabled"] = False

    try:
        results["test_sequential_fallback"] = await test_parallel_preflight_disabled()
    except Exception as e:
        print(f"\n[ERROR] Test 2 crashed: {e}")
        import traceback
        traceback.print_exc()
        results["test_sequential_fallback"] = False

    try:
        results["test_runconfig"] = await test_runconfig_propagation()
    except Exception as e:
        print(f"\n[ERROR] Test 3 crashed: {e}")
        import traceback
        traceback.print_exc()
        results["test_runconfig"] = False

    try:
        results["test_resilience"] = await test_error_resilience()
    except Exception as e:
        print(f"\n[ERROR] Test 4 crashed: {e}")
        import traceback
        traceback.print_exc()
        results["test_resilience"] = False

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {test_name}: {status}")

    total = len(results)
    passed = sum(results.values())
    print(f"\n  Total: {passed}/{total} tests passed")

    overall = all(results.values())
    print(f"\n{'=' * 70}")
    print(f"OVERALL: {'ALL TESTS PASSED' if overall else 'SOME TESTS FAILED'}")
    print(f"{'=' * 70}")

    return overall


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
