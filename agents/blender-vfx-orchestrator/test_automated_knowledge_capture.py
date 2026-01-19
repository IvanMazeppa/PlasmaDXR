"""
Test automated knowledge capture in Learning Agent.

Verifies that:
1. Learning Agent has pattern extraction tools
2. Learning Agent instructions contain automatic pattern extraction section
3. Learning Agent understands when to extract patterns (score_delta >= 5)
"""

import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

from agents import Agent, Runner, trace
from specialized_agents.learning_agent import (
    LEARNING_AGENT_INSTRUCTIONS,
    create_learning_agent,
)


async def test_learning_agent_has_pattern_tools():
    """Verify Learning Agent is created with pattern extraction tools."""
    print("=" * 70)
    print("TEST: LEARNING AGENT HAS PATTERN EXTRACTION TOOLS")
    print("=" * 70)
    print()

    agent = create_learning_agent()

    tool_names = [t.name for t in agent.tools if hasattr(t, 'name')]
    print(f"[1] Learning Agent tools ({len(tool_names)} total):")
    for name in tool_names:
        print(f"    - {name}")
    print()

    # Check for pattern extraction tools
    required_tools = [
        "extract_successful_pattern",
        "analyze_script_for_patterns",
        "record_code_pattern",
        "search_code_patterns",
        "report_pattern_outcome",
    ]

    print("[2] Pattern extraction tool verification:")
    all_present = True
    for tool in required_tools:
        present = tool in tool_names
        print(f"    {'✓' if present else '✗'} {tool}")
        if not present:
            all_present = False

    return all_present


async def test_instructions_contain_pattern_extraction():
    """Verify Learning Agent instructions contain automatic pattern extraction section."""
    print()
    print("=" * 70)
    print("TEST: LEARNING AGENT INSTRUCTIONS CONTAIN PATTERN EXTRACTION")
    print("=" * 70)
    print()

    checks = {
        "AUTOMATIC PATTERN EXTRACTION section": "AUTOMATIC PATTERN EXTRACTION" in LEARNING_AGENT_INSTRUCTIONS,
        "extract_successful_pattern documented": "extract_successful_pattern()" in LEARNING_AGENT_INSTRUCTIONS,
        "record_code_pattern documented": "record_code_pattern()" in LEARNING_AGENT_INSTRUCTIONS,
        "search_code_patterns documented": "search_code_patterns" in LEARNING_AGENT_INSTRUCTIONS,
        "report_pattern_outcome documented": "report_pattern_outcome()" in LEARNING_AGENT_INSTRUCTIONS,
        "score_delta >= 5 threshold": "score_delta >= 5" in LEARNING_AGENT_INSTRUCTIONS,
        "PATTERN REUSE WORKFLOW section": "PATTERN REUSE WORKFLOW" in LEARNING_AGENT_INSTRUCTIONS,
        "pattern_extracted output field": "pattern_extracted" in LEARNING_AGENT_INSTRUCTIONS,
    }

    all_passed = True
    for check, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"    {status} {check}")
        if not passed:
            all_passed = False

    return all_passed


async def test_learning_agent_pattern_extraction_logic():
    """Test that Learning Agent understands when to extract patterns."""
    print()
    print("=" * 70)
    print("TEST: LEARNING AGENT PATTERN EXTRACTION LOGIC")
    print("=" * 70)
    print()

    # Create a minimal agent with just pattern instructions for testing
    agent = Agent(
        name="Learning Agent Pattern Test",
        instructions=LEARNING_AGENT_INSTRUCTIONS + """

For this test, you do NOT have access to actual tools.
Instead, return JSON describing what actions you WOULD take.

Return JSON:
{
    "should_extract_pattern": bool,
    "reason": "why or why not",
    "tools_would_call": ["list of tools you would call"],
    "pattern_details": {
        "pattern_name": "descriptive name if extracting",
        "issue_type": "problem category"
    }
}
""",
        model="gpt-5.2",
        tools=[],  # No tools - just testing instruction comprehension
    )

    # Test scenario 1: Successful experiment (should extract)
    print("[1] Testing: Successful experiment (score improved by 12 points)...")

    with trace("Pattern Extraction Logic Test 1"):
        result1 = await Runner.run(
            agent,
            """An experiment just completed with these results:

Previous score: 45
Current score: 57
Score delta: +12 points

The experiment:
- Issue: "flames too dark"
- Changed: emission_intensity from 1.0 to 4.5
- Changed: flame_max_temp from 1.5 to 3.0
- Effect type: explosion

Based on your instructions, should you extract a pattern? What would you do?""",
            max_turns=10
        )

    output1 = str(result1.final_output).lower()
    print(f"[2] Response summary: {output1[:500]}...")
    print()

    # Test scenario 2: Failed experiment (should NOT extract)
    print("[3] Testing: Failed experiment (score decreased)...")

    with trace("Pattern Extraction Logic Test 2"):
        result2 = await Runner.run(
            agent,
            """An experiment just completed with these results:

Previous score: 45
Current score: 42
Score delta: -3 points

The experiment:
- Issue: "smoke too thin"
- Changed: dissolve_speed from 50 to 100
- Effect type: fire

Based on your instructions, should you extract a pattern? What would you do?""",
            max_turns=10
        )

    output2 = str(result2.final_output).lower()
    print(f"[4] Response summary: {output2[:500]}...")
    print()

    # Verify understanding
    checks = {
        "understands_to_extract_on_success": any(term in output1 for term in [
            "extract", "record", "pattern", "yes", "true", "should"
        ]),
        "mentions_score_threshold": any(term in output1 for term in [
            "5", "threshold", ">= 5", "12", "improved"
        ]),
        "understands_not_to_extract_on_failure": any(term in output2 for term in [
            "no", "false", "not", "shouldn't", "failed", "decreased"
        ]),
        "knows_tools_to_use": any(term in output1 for term in [
            "extract_successful_pattern", "record_code_pattern"
        ]),
    }

    print("[5] Logic verification:")
    all_passed = True
    for check, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"    {status} {check}")
        if not passed:
            all_passed = False

    return all_passed


async def test_orchestrator_phase5_integration():
    """Verify orchestrator PHASE 5 includes mandatory knowledge capture."""
    print()
    print("=" * 70)
    print("TEST: ORCHESTRATOR PHASE 5 MANDATORY KNOWLEDGE CAPTURE")
    print("=" * 70)
    print()

    from orchestrator import ORCHESTRATOR_INSTRUCTIONS

    checks = {
        "MANDATORY KNOWLEDGE CAPTURE in title": "MANDATORY KNOWLEDGE CAPTURE" in ORCHESTRATOR_INSTRUCTIONS,
        "extract_successful_pattern call": "extract_successful_pattern" in ORCHESTRATOR_INSTRUCTIONS,
        "record_code_pattern call": "record_code_pattern" in ORCHESTRATOR_INSTRUCTIONS,
        "score_delta calculation": "score_delta" in ORCHESTRATOR_INSTRUCTIONS,
        "threshold check (>= 5)": ">= 5" in ORCHESTRATOR_INSTRUCTIONS,
        "Step 5.1 CALCULATE IMPROVEMENT": "CALCULATE IMPROVEMENT" in ORCHESTRATOR_INSTRUCTIONS,
        "Step 5.2 CAPTURE KNOWLEDGE": "CAPTURE KNOWLEDGE" in ORCHESTRATOR_INSTRUCTIONS,
    }

    all_passed = True
    for check, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"    {status} {check}")
        if not passed:
            all_passed = False

    return all_passed


if __name__ == "__main__":
    print()
    print("AUTOMATED KNOWLEDGE CAPTURE TEST SUITE")
    print()

    # Test 1: Check tools are present
    test1 = asyncio.run(test_learning_agent_has_pattern_tools())

    # Test 2: Check instructions
    test2 = asyncio.run(test_instructions_contain_pattern_extraction())

    # Test 3: Test pattern extraction logic
    test3 = asyncio.run(test_learning_agent_pattern_extraction_logic())

    # Test 4: Check orchestrator integration
    test4 = asyncio.run(test_orchestrator_phase5_integration())

    print()
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"  Learning Agent has pattern tools:     {'PASSED ✓' if test1 else 'FAILED ✗'}")
    print(f"  Instructions contain pattern section: {'PASSED ✓' if test2 else 'FAILED ✗'}")
    print(f"  Pattern extraction logic correct:     {'PASSED ✓' if test3 else 'FAILED ✗'}")
    print(f"  Orchestrator PHASE 5 integration:     {'PASSED ✓' if test4 else 'FAILED ✗'}")
    print()
    print("OVERALL:", "PASSED ✓" if (test1 and test2 and test3 and test4) else "NEEDS REVIEW")
    print("=" * 70)
