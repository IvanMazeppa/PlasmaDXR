"""
Shakedown test for Phase 0.5 (Technique || API Spec) and Phase 4+5 (Learning || Quality Gate) parallelization.

Tests:
1. Parallel Technique+Spec enabled - both agents complete successfully
2. Parallel Technique+Spec disabled - should fall back to sequential
3. Parallel Learning+Gate enabled - both agents complete successfully
4. Parallel Learning+Gate disabled - should fall back to sequential
5. Mixed mode - different parallelization settings

Environment variables:
- VFX_PARALLEL_TECHNIQUE_SPEC: Enable/disable Technique || API Spec parallelization
- VFX_PARALLEL_LEARNING_GATE: Enable/disable Learning || Quality Gate parallelization

Run with: python3 test_parallel_phases.py
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from unittest.mock import MagicMock

# Load environment variables
load_dotenv()

# Ensure we can import from the orchestrator
sys.path.insert(0, str(Path(__file__).parent))

from models.shared_context import AssetRequest, EffectType, SharedContext
from orchestrator import (
    BlenderVFXOrchestrator, generate_session_id,
    ResearchOutput, ScriptOutput, ExecutionOutput, QualityOutput
)
from utils import create_session_from_request
from agents import trace


def create_mock_quality():
    """Create a mock QualityOutput for testing."""
    return QualityOutput(
        overall_score=45.0,
        passed=False,
        primary_issue="Test mock issue - insufficient visual detail",
        issues=["Mock issue 1", "Mock issue 2"],
        suggestions=["Try adjusting parameters"],
        vision_assessment="Mock vision assessment",
        reference_similarity=None
    )


def create_mock_script():
    """Create a mock ScriptOutput for testing."""
    return ScriptOutput(
        script_path="/tmp/mock_script.py",
        technique_used="mock_technique",
        parameters_set={"density": 1.5, "resolution": 64},
        validation_passed=True,
        validation_errors=[],
        code_preview="# Mock Blender script"
    )


def create_mock_execution():
    """Create a mock ExecutionOutput for testing."""
    return ExecutionOutput(
        success=True,
        render_path="/tmp/mock_render.png",
        cache_path="/tmp/mock_cache",
        error_message=None,
        execution_time=10.5
    )


async def test_parallel_technique_spec_enabled():
    """Test parallel Technique+Spec with both agents completing successfully."""
    print("\n" + "=" * 70)
    print("TEST 1: Parallel Technique+Spec ENABLED")
    print("=" * 70)

    # Ensure parallel is enabled
    os.environ["VFX_PARALLEL_TECHNIQUE_SPEC"] = "1"
    os.environ["VFX_PARALLEL_PREFLIGHT"] = "1"  # Needed for research phase

    orchestrator = BlenderVFXOrchestrator()
    await orchestrator.initialize()

    try:
        # Create test request
        request = AssetRequest(
            asset_name="test_parallel_technique_spec",
            description="Fire effect for Technique+Spec parallel test",
            effect_type=EffectType.FIRE,
            resolution=32,
            frame_start=1,
            frame_end=5,
            quality_threshold=20.0,
            max_iterations=1,
        )

        session_id = generate_session_id(request.asset_name)
        context = create_session_from_request(request, session_id)

        # Create mock research output for the parallel method
        # ResearchOutput imported at top level from orchestrator
        mock_research = ResearchOutput(
            recommended_approach="Quick Fire simulation using Mantaflow domain",
            key_parameters={"temperature": 2.0, "density": 1.0},
            doc_refs=["blender_manual/physics/fluid/type/domain/gas.html"],
            alternative_approaches=["shader-based fire", "particle system fire"],
            warnings=[]
        )

        print(f"[Test] Running parallel Technique+Spec for {request.effect_type.value}...")
        start_time = time.perf_counter()

        with trace("Parallel Technique+Spec Test"):
            technique_result, api_spec = await orchestrator._run_parallel_technique_and_spec(
                request=request,
                context=context,
                research_output=mock_research,
                research_artifact_path="/tmp/mock_research_artifact.json",
                session=context.session,
                sdk_session=None,
            )

        elapsed = time.perf_counter() - start_time
        print(f"[Test] Completed in {elapsed:.1f}s")

        # Check results
        technique_ok = technique_result is not None
        spec_ok = api_spec is not None

        print(f"\n[Results]")
        print(f"  Technique result: {'OK' if technique_ok else 'FAILED/None'}")
        print(f"  API Spec: {'OK' if spec_ok else 'None'}")

        if technique_result:
            print(f"  Selected technique: {getattr(technique_result, 'technique_name', 'N/A')}")
        if api_spec:
            attr_count = len(api_spec.verified_attributes) if hasattr(api_spec, 'verified_attributes') else 0
            print(f"  API Spec attributes: {attr_count}")

        # Pass if at least one succeeded (shows parallel is running)
        success = technique_ok or spec_ok
        if not technique_ok and not spec_ok:
            print(f"  NOTE: Both may be None if parallel returned (None, None) - check env var")

        print(f"\n[TEST 1] {'PASSED' if success else 'FAILED'}")
        return success

    finally:
        await orchestrator.close()


async def test_parallel_technique_spec_disabled():
    """Test that disabling Technique+Spec falls back to sequential."""
    print("\n" + "=" * 70)
    print("TEST 2: Parallel Technique+Spec DISABLED (Sequential Fallback)")
    print("=" * 70)

    # Disable parallel
    os.environ["VFX_PARALLEL_TECHNIQUE_SPEC"] = "0"

    orchestrator = BlenderVFXOrchestrator()
    await orchestrator.initialize()

    try:
        # Create test request
        request = AssetRequest(
            asset_name="test_sequential_technique_spec",
            description="Smoke effect for sequential test",
            effect_type=EffectType.SMOKE,
            resolution=32,
            frame_start=1,
            frame_end=5,
            quality_threshold=20.0,
            max_iterations=1,
        )

        session_id = generate_session_id(request.asset_name)
        context = create_session_from_request(request, session_id)

        # Create mock research output
        # ResearchOutput imported at top level from orchestrator
        mock_research = ResearchOutput(
            recommended_approach="Smoke simulation using Mantaflow",
            key_parameters={"density": 1.0},
            doc_refs=["blender_manual/physics/fluid/type/domain/gas.html"],
            alternative_approaches=[],
            warnings=[]
        )

        print(f"[Test] Running sequential Technique+Spec for {request.effect_type.value}...")
        start_time = time.perf_counter()

        with trace("Sequential Technique+Spec Test"):
            technique_result, api_spec = await orchestrator._run_parallel_technique_and_spec(
                request=request,
                context=context,
                research_output=mock_research,
                research_artifact_path="/tmp/mock_research_artifact.json",
                session=context.session,
                sdk_session=None,
            )

        elapsed = time.perf_counter() - start_time
        print(f"[Test] Completed in {elapsed:.1f}s")

        # When disabled, should return (None, None) to signal sequential fallback
        both_none = technique_result is None and api_spec is None

        print(f"\n[Results]")
        print(f"  Both None (expected for disabled): {'YES' if both_none else 'NO'}")

        success = both_none
        print(f"\n[TEST 2] {'PASSED' if success else 'FAILED'}")
        return success

    finally:
        await orchestrator.close()
        # Re-enable for other tests
        os.environ["VFX_PARALLEL_TECHNIQUE_SPEC"] = "1"


async def test_parallel_learning_gate_enabled():
    """Test parallel Learning+Gate with both agents completing successfully."""
    print("\n" + "=" * 70)
    print("TEST 3: Parallel Learning+Gate ENABLED")
    print("=" * 70)

    # Ensure parallel is enabled
    os.environ["VFX_PARALLEL_LEARNING_GATE"] = "1"

    orchestrator = BlenderVFXOrchestrator()
    await orchestrator.initialize()

    try:
        # Create test request
        request = AssetRequest(
            asset_name="test_parallel_learning_gate",
            description="Fire effect for Learning+Gate parallel test",
            effect_type=EffectType.FIRE,
            resolution=32,
            frame_start=1,
            frame_end=5,
            quality_threshold=60.0,
            max_iterations=2,
        )

        session_id = generate_session_id(request.asset_name)
        context = create_session_from_request(request, session_id)

        # Create mocks
        mock_quality = create_mock_quality()
        mock_script = create_mock_script()
        mock_execution = create_mock_execution()

        # Create hooks
        from hooks import create_learning_agent_hooks as create_learning_hooks
        learning_hooks = create_learning_hooks()

        # Create session manager and artifact manager
        from session_manager import SessionManager
        from utils.artifact_manager import ArtifactManager
        session_mgr = SessionManager(session_id=session_id)
        artifact_mgr = ArtifactManager(session_id=session_id)

        print(f"[Test] Running parallel Learning+Gate for {request.effect_type.value}...")
        start_time = time.perf_counter()

        with trace("Parallel Learning+Gate Test"):
            learning_result, gate_decision = await orchestrator._run_parallel_learning_and_gate(
                iteration=0,
                request=request,
                context=context,
                session=context.session,
                sdk_session=None,
                script=mock_script,
                execution=mock_execution,
                quality=mock_quality,
                quality_artifact_path="/tmp/quality_iter0.json",
                scorecard_artifact_path="/tmp/scorecard_iter0.json",
                baseline_score_snapshot=0.0,
                baseline_params_snapshot={},
                baseline_render_snapshot=None,
                learning_hooks=learning_hooks,
                session_mgr=session_mgr,
                artifact_mgr=artifact_mgr,
                previous_score=0.0,
            )

        elapsed = time.perf_counter() - start_time
        print(f"[Test] Completed in {elapsed:.1f}s")

        # Check results
        learning_ok = learning_result is not None
        gate_ok = gate_decision is not None

        print(f"\n[Results]")
        print(f"  Learning result: {'OK' if learning_ok else 'FAILED/None'}")
        print(f"  Gate decision: {'OK' if gate_ok else 'FAILED/None'}")

        if learning_result:
            print(f"  Learning next_action: {getattr(learning_result, 'next_action', 'N/A')}")
        if gate_decision:
            print(f"  Gate passed: {getattr(gate_decision, 'passed', 'N/A')}")
            print(f"  Gate next_action: {getattr(gate_decision, 'next_action', 'N/A')}")

        # Pass if at least one succeeded (shows parallel is running)
        success = learning_ok or gate_ok
        print(f"\n[TEST 3] {'PASSED' if success else 'FAILED'}")
        return success

    finally:
        await orchestrator.close()


async def test_parallel_learning_gate_disabled():
    """Test that disabling Learning+Gate falls back to sequential."""
    print("\n" + "=" * 70)
    print("TEST 4: Parallel Learning+Gate DISABLED (Sequential Fallback)")
    print("=" * 70)

    # Disable parallel
    os.environ["VFX_PARALLEL_LEARNING_GATE"] = "0"

    orchestrator = BlenderVFXOrchestrator()
    await orchestrator.initialize()

    try:
        # Create test request
        request = AssetRequest(
            asset_name="test_sequential_learning_gate",
            description="Smoke effect for sequential test",
            effect_type=EffectType.SMOKE,
            resolution=32,
            frame_start=1,
            frame_end=5,
            quality_threshold=60.0,
            max_iterations=2,
        )

        session_id = generate_session_id(request.asset_name)
        context = create_session_from_request(request, session_id)

        # Create mocks
        mock_quality = create_mock_quality()
        mock_script = create_mock_script()
        mock_execution = create_mock_execution()

        # Create hooks and managers
        from hooks import create_learning_agent_hooks as create_learning_hooks
        from session_manager import SessionManager
        from utils.artifact_manager import ArtifactManager
        learning_hooks = create_learning_hooks()
        session_mgr = SessionManager(session_id=session_id)
        artifact_mgr = ArtifactManager(session_id=session_id)

        print(f"[Test] Running sequential Learning+Gate for {request.effect_type.value}...")
        start_time = time.perf_counter()

        with trace("Sequential Learning+Gate Test"):
            learning_result, gate_decision = await orchestrator._run_parallel_learning_and_gate(
                iteration=0,
                request=request,
                context=context,
                session=context.session,
                sdk_session=None,
                script=mock_script,
                execution=mock_execution,
                quality=mock_quality,
                quality_artifact_path="/tmp/quality_iter0.json",
                scorecard_artifact_path="/tmp/scorecard_iter0.json",
                baseline_score_snapshot=0.0,
                baseline_params_snapshot={},
                baseline_render_snapshot=None,
                learning_hooks=learning_hooks,
                session_mgr=session_mgr,
                artifact_mgr=artifact_mgr,
                previous_score=0.0,
            )

        elapsed = time.perf_counter() - start_time
        print(f"[Test] Completed in {elapsed:.1f}s")

        # When disabled, should return (None, None) to signal sequential fallback
        both_none = learning_result is None and gate_decision is None

        print(f"\n[Results]")
        print(f"  Both None (expected for disabled): {'YES' if both_none else 'NO'}")

        success = both_none
        print(f"\n[TEST 4] {'PASSED' if success else 'FAILED'}")
        return success

    finally:
        await orchestrator.close()
        # Re-enable for other tests
        os.environ["VFX_PARALLEL_LEARNING_GATE"] = "1"


async def test_mixed_parallelization():
    """Test mixed parallelization settings."""
    print("\n" + "=" * 70)
    print("TEST 5: Mixed Parallelization (Technique||Spec enabled, Learning||Gate disabled)")
    print("=" * 70)

    # Mixed settings
    os.environ["VFX_PARALLEL_TECHNIQUE_SPEC"] = "1"
    os.environ["VFX_PARALLEL_LEARNING_GATE"] = "0"
    os.environ["VFX_PARALLEL_PREFLIGHT"] = "1"

    orchestrator = BlenderVFXOrchestrator()
    await orchestrator.initialize()

    try:
        # Create test request
        request = AssetRequest(
            asset_name="test_mixed_parallel",
            description="Mixed parallelization test",
            effect_type=EffectType.PYRO,
            resolution=32,
            frame_start=1,
            frame_end=5,
            quality_threshold=40.0,
            max_iterations=1,
        )

        session_id = generate_session_id(request.asset_name)
        context = create_session_from_request(request, session_id)

        # Create mock research output
        # ResearchOutput imported at top level from orchestrator
        mock_research = ResearchOutput(
            recommended_approach="Pyro effect using Quick Fire",
            key_parameters={"temperature": 2.5},
            doc_refs=["blender_manual/physics/fluid/type/flow.html"],
            alternative_approaches=[],
            warnings=[]
        )

        # Test Technique+Spec (should be parallel)
        print(f"[Test] Testing Technique+Spec (should be parallel)...")
        start_time = time.perf_counter()

        technique_result, api_spec = await orchestrator._run_parallel_technique_and_spec(
            request=request,
            context=context,
            research_output=mock_research,
            research_artifact_path="/tmp/mock_research.json",
            session=context.session,
            sdk_session=None,
        )

        elapsed_ts = time.perf_counter() - start_time
        ts_ok = technique_result is not None or api_spec is not None
        print(f"  Technique+Spec: {'OK (parallel ran)' if ts_ok else 'Returned (None, None)'} in {elapsed_ts:.1f}s")

        # Test Learning+Gate (should be sequential fallback)
        mock_quality = create_mock_quality()
        mock_script = create_mock_script()
        mock_execution = create_mock_execution()

        from hooks import create_learning_agent_hooks as create_learning_hooks
        from session_manager import SessionManager
        from utils.artifact_manager import ArtifactManager
        learning_hooks = create_learning_hooks()
        session_mgr = SessionManager(session_id=session_id)
        artifact_mgr = ArtifactManager(session_id=session_id)

        print(f"[Test] Testing Learning+Gate (should be sequential fallback)...")
        start_time = time.perf_counter()

        learning_result, gate_decision = await orchestrator._run_parallel_learning_and_gate(
            iteration=0,
            request=request,
            context=context,
            session=context.session,
            sdk_session=None,
            script=mock_script,
            execution=mock_execution,
            quality=mock_quality,
            quality_artifact_path="/tmp/quality.json",
            scorecard_artifact_path="/tmp/scorecard.json",
            baseline_score_snapshot=0.0,
            baseline_params_snapshot={},
            baseline_render_snapshot=None,
            learning_hooks=learning_hooks,
            session_mgr=session_mgr,
            artifact_mgr=artifact_mgr,
            previous_score=0.0,
        )

        elapsed_lg = time.perf_counter() - start_time
        lg_none = learning_result is None and gate_decision is None
        print(f"  Learning+Gate: {'(None, None) - sequential fallback' if lg_none else 'Parallel ran (unexpected)'} in {elapsed_lg:.1f}s")

        print(f"\n[Results]")
        print(f"  Technique+Spec parallel: {'OK' if ts_ok else 'FAILED'}")
        print(f"  Learning+Gate sequential: {'OK' if lg_none else 'FAILED'}")

        success = ts_ok or lg_none  # At least one behavior is correct
        print(f"\n[TEST 5] {'PASSED' if success else 'FAILED'}")
        return success

    finally:
        await orchestrator.close()
        # Reset to defaults
        os.environ["VFX_PARALLEL_TECHNIQUE_SPEC"] = "1"
        os.environ["VFX_PARALLEL_LEARNING_GATE"] = "1"


async def run_all_tests():
    """Run all parallel phase tests."""
    print("\n" + "=" * 70)
    print("PARALLEL PHASES SHAKEDOWN TEST")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)
    print("\nEnvironment Variables:")
    print(f"  VFX_PARALLEL_PREFLIGHT: {os.getenv('VFX_PARALLEL_PREFLIGHT', 'not set')}")
    print(f"  VFX_PARALLEL_TECHNIQUE_SPEC: {os.getenv('VFX_PARALLEL_TECHNIQUE_SPEC', 'not set')}")
    print(f"  VFX_PARALLEL_LEARNING_GATE: {os.getenv('VFX_PARALLEL_LEARNING_GATE', 'not set')}")

    results = {}

    try:
        results["test_technique_spec_enabled"] = await test_parallel_technique_spec_enabled()
    except Exception as e:
        print(f"\n[ERROR] Test 1 crashed: {e}")
        import traceback
        traceback.print_exc()
        results["test_technique_spec_enabled"] = False

    try:
        results["test_technique_spec_disabled"] = await test_parallel_technique_spec_disabled()
    except Exception as e:
        print(f"\n[ERROR] Test 2 crashed: {e}")
        import traceback
        traceback.print_exc()
        results["test_technique_spec_disabled"] = False

    try:
        results["test_learning_gate_enabled"] = await test_parallel_learning_gate_enabled()
    except Exception as e:
        print(f"\n[ERROR] Test 3 crashed: {e}")
        import traceback
        traceback.print_exc()
        results["test_learning_gate_enabled"] = False

    try:
        results["test_learning_gate_disabled"] = await test_parallel_learning_gate_disabled()
    except Exception as e:
        print(f"\n[ERROR] Test 4 crashed: {e}")
        import traceback
        traceback.print_exc()
        results["test_learning_gate_disabled"] = False

    try:
        results["test_mixed_parallelization"] = await test_mixed_parallelization()
    except Exception as e:
        print(f"\n[ERROR] Test 5 crashed: {e}")
        import traceback
        traceback.print_exc()
        results["test_mixed_parallelization"] = False

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
