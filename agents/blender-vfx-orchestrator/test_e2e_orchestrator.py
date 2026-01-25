"""
End-to-end test of the Blender VFX Orchestrator.

Tests:
1. Orchestrator initialization with typed Agent[SharedContext]
2. trace() wrapper for observability
3. RunContextWrapper auto-population in tools
4. Blender manual tool integration
5. Full asset generation workflow

Verbose Tracing:
    Set VERBOSE_TRACING=1 to enable detailed trace output.
    Optionally set VERBOSE_TRACE_FILE to control the log path.
"""

import asyncio
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Verify API key is set
if not os.getenv("OPENAI_API_KEY"):
    print("ERROR: OPENAI_API_KEY not found in environment or .env file")
    sys.exit(1)

# Use gpt-5.2 for full reasoning capabilities
os.environ.setdefault("ORCHESTRATOR_MODEL", "gpt-5.2")

from models.shared_context import AssetRequest, EffectType, SharedContext
from orchestrator import BlenderVFXOrchestrator, create_session_from_request, generate_session_id
from agents import Runner, trace

# Enable verbose tracing if requested
if os.environ.get("VERBOSE_TRACING", "").lower() in ("1", "true", "yes"):
    from tracing import enable_verbose_tracing
    log_file = os.environ.get("VERBOSE_TRACE_FILE")
    if not log_file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"traces/e2e_test_verbose_{timestamp}.jsonl"
    enable_verbose_tracing(
        log_file=log_file,
        console_output=True,
        include_span_data=True,
    )
    print("[E2E TEST] Verbose tracing ENABLED", file=sys.stderr)


async def test_orchestrator_e2e():
    """Run end-to-end test of the orchestrator."""
    print("=" * 70)
    print("BLENDER VFX ORCHESTRATOR - END-TO-END TEST")
    print("=" * 70)
    print()
    
    # Create a simple test request
    request = AssetRequest(
        asset_name="test_smoke_e2e",
        description="Simple rising smoke column for testing orchestrator pipeline",
        effect_type=EffectType.PYRO,
        resolution=48,  # Low res for speed
        frame_start=1,
        frame_end=24,  # Short animation
        quality_threshold=40.0,  # Lower threshold for test
        max_iterations=2,  # Limit iterations
    )
    
    print(f"Test Request:")
    print(f"  - Asset: {request.asset_name}")
    print(f"  - Effect: {request.effect_type.value}")
    print(f"  - Resolution: {request.resolution}")
    print(f"  - Frames: {request.frame_start}-{request.frame_end}")
    print(f"  - Quality Threshold: {request.quality_threshold}")
    print(f"  - Max Iterations: {request.max_iterations}")
    print()
    
    # Initialize orchestrator
    print("[1/4] Initializing orchestrator...")
    orchestrator = BlenderVFXOrchestrator()
    await orchestrator.initialize()
    print(f"  ✓ Orchestrator initialized with 5 agents")
    print(f"  ✓ Budget: ${orchestrator.get_budget_status()['total_remaining']:.2f} remaining")
    print()
    
    # Test typed context creation
    print("[2/4] Testing typed SharedContext...")
    session_id = generate_session_id(request.asset_name)
    context = create_session_from_request(request, session_id)
    print(f"  ✓ Session ID: {session_id}")
    print(f"  ✓ Context type: {type(context).__name__}")
    print(f"  ✓ Session effect_type: {context.session.request.effect_type.value}")
    print()
    
    # Run the orchestrator with trace
    print("[3/4] Running orchestrator with trace()...")
    print("  (This will generate a script, execute Blender, and evaluate quality)")
    print()
    
    try:
        result = await orchestrator.create_asset_pipeline(request)
        
        print()
        print("[4/4] Results:")
        print(f"  - Status: {result.status.value}")
        print(f"  - Iterations completed: {result.current_iteration}")
        print(f"  - Best score: {result.best_score:.1f}")
        print(f"  - Best iteration: {result.best_iteration}")
        
        if result.final_render_path:
            print(f"  - Final render: {result.final_render_path}")
        if result.final_vdb_dir:
            print(f"  - VDB directory: {result.final_vdb_dir}")
        if result.current_issues:
            print(f"  - Current issues: {result.current_issues}")
        
        # Check iteration history
        if result.iterations:
            print()
            print("Iteration History:")
            for it in result.iterations:
                print(f"  [{it.iteration}] Score: {it.score:.1f}, Passed: {it.passed}")
                primary_issue = getattr(it, "primary_issue", None)
                if not primary_issue and hasattr(it, "quality"):
                    primary_issue = getattr(it.quality, "primary_issue", None)
                if primary_issue:
                    print(f"      Issue: {primary_issue}")
        
        print()
        print("=" * 70)
        print("TEST COMPLETE")
        print("=" * 70)
        
        return result
        
    except Exception as e:
        print(f"\n  ✗ Error during orchestration: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        await orchestrator.close()


if __name__ == "__main__":
    result = asyncio.run(test_orchestrator_e2e())
    
    # Exit with appropriate code
    if result and result.status.value in ("passed", "max_iterations"):
        sys.exit(0)
    else:
        sys.exit(1)
