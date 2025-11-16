#!/usr/bin/env python3
"""
Quick test for record_decision file I/O functionality.

This directly tests the tool function without requiring the full SDK client.
"""

import asyncio
from pathlib import Path
from datetime import datetime

# Import the tool function
from tools.record import record_decision


async def test_record_decision():
    """Test that record_decision creates and writes to session files."""

    print("Testing record_decision file I/O...")
    print("-" * 60)

    # Test data
    test_args = {
        "decision": "Mission-Control Agent Phase 1.1 Complete",
        "rationale": """Successfully implemented the record_decision tool with:
- Proper markdown formatting
- Error handling for file operations
- Session file management (create/append)
- Project-relative path resolution
- mypy strict mode type checking (0 errors)""",
        "artifacts": [
            "agents/mission-control/tools/record.py",
            "agents/mission-control/venv/lib/python3.12/site-packages/claude_agent_sdk/"
        ]
    }

    # Call the tool
    result = await record_decision(test_args)

    # Display result
    print("\n✅ Tool execution completed!")
    print("\nResult:")
    print(result["content"][0]["text"])

    # Verify file was created
    project_root = Path(__file__).parent.parent.parent
    today = datetime.now().strftime("%Y-%m-%d")
    session_file = project_root / "docs" / "sessions" / f"SESSION_{today}.md"

    if session_file.exists():
        print(f"\n✅ Session file created: {session_file.relative_to(project_root)}")
        print(f"   File size: {session_file.stat().st_size} bytes")

        # Show first few lines
        with open(session_file, "r") as f:
            lines = f.readlines()[:15]
        print("\n📄 First 15 lines of session file:")
        print("".join(lines))
    else:
        print(f"\n❌ Session file not created at expected location: {session_file}")

    print("\n" + "=" * 60)
    print("Test complete! Check the session file for full content.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_record_decision())
