"""
Test SDK features: trace(), RunContextWrapper, typed Agent[SharedContext]
"""

import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from models.shared_context import AssetRequest, EffectType, SharedContext
from orchestrator import generate_session_id
from utils import create_session_from_request
from agents import Agent, Runner, trace, function_tool, RunContextWrapper


# Test tool that uses RunContextWrapper
@function_tool  
def test_context_access(
    wrapper: RunContextWrapper[SharedContext],
    message: str
) -> str:
    """Test tool that accesses the SharedContext and returns session info."""
    context = wrapper.context
    if context and hasattr(context, 'session'):
        session = context.session
        return f"CONTEXT_ACCESS_SUCCESS: session_id={session.session_id}, effect_type={session.request.effect_type.value}, message={message}"
    return f"CONTEXT_ACCESS_FAILED: no context available, message={message}"


async def test_trace_and_context():
    """Test trace() wrapper and RunContextWrapper."""
    print("=" * 70)
    print("SDK FEATURES TEST")
    print("=" * 70)
    print()
    
    # Create test request
    request = AssetRequest(
        asset_name="sdk_test",
        description="Test for SDK features",
        effect_type=EffectType.PYRO,
    )
    
    session_id = generate_session_id(request.asset_name)
    context = create_session_from_request(request, session_id)
    
    print(f"[1] Created SharedContext:")
    print(f"    Session ID: {session_id}")
    print(f"    Effect Type: {context.session.request.effect_type.value}")
    print()
    
    # Create a simple agent with our test tool
    test_agent = Agent[SharedContext](
        name="SDK Test Agent",
        instructions="""You are a test agent. When asked to test context access:
1. Call the test_context_access tool with the provided message
2. Report the EXACT tool output back to the user""",
        model="gpt-5.2",
        tools=[test_context_access],
    )
    
    print(f"[2] Created typed Agent[SharedContext]")
    print()
    
    # Run with trace
    print(f"[3] Running with trace() wrapper...")
    
    with trace("SDK Features Test"):
        result = await Runner.run(
            test_agent,
            "Test context access with message 'hello_test_123'. Report the exact tool output.",
            context=context,
            max_turns=3
        )
    
    output = str(result.final_output)
    print(f"    Agent output: {output[:200]}...")
    print()
    
    # Check various indicators of success
    success_indicators = [
        "CONTEXT_ACCESS_SUCCESS" in output,
        "session_id=" in output,
        "effect_type=" in output,
        "pyro" in output.lower(),
        "sdk_test" in output.lower(),
        "hello_test_123" in output,
        "successful" in output.lower(),
    ]
    
    print(f"[4] Success Indicators:")
    indicator_names = [
        "CONTEXT_ACCESS_SUCCESS marker",
        "session_id present",
        "effect_type present", 
        "pyro effect detected",
        "sdk_test session detected",
        "test message present",
        "success mentioned"
    ]
    
    for name, passed in zip(indicator_names, success_indicators):
        status = "✓" if passed else "✗"
        print(f"    {status} {name}")
    
    # Consider success if at least 3 indicators pass
    passed_count = sum(success_indicators)
    overall_success = passed_count >= 3
    
    print()
    print(f"[5] Result: {passed_count}/7 indicators passed")
    
    return overall_success


if __name__ == "__main__":
    success = asyncio.run(test_trace_and_context())
    print()
    print("=" * 70)
    print("TEST RESULT:", "PASSED ✓" if success else "FAILED ✗")
    print("=" * 70)
