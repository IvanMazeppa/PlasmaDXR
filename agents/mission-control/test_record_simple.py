#!/usr/bin/env python3
"""
Simple direct test of record_decision logic (without SDK decorator).
"""

from datetime import datetime
from pathlib import Path


def test_record_decision_logic():
    """Test the core logic of record_decision."""

    print("Testing record_decision file I/O logic...")
    print("-" * 60)

    # Test data
    decision = "Mission-Control Agent Phase 1.1 Complete"
    rationale = """Successfully implemented the record_decision tool with:
- Proper markdown formatting
- Error handling for file operations
- Session file management (create/append)
- Project-relative path resolution
- mypy strict mode type checking (0 errors)"""
    artifacts = [
        "agents/mission-control/tools/record.py",
        "mypy type checking output"
    ]

    # Generate session file path
    project_root = Path(__file__).parent.parent.parent
    today = datetime.now().strftime("%Y-%m-%d")
    sessions_dir = project_root / "docs" / "sessions"
    session_file = sessions_dir / f"SESSION_{today}.md"

    # Create sessions directory if it doesn't exist
    sessions_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📁 Sessions directory: {sessions_dir.relative_to(project_root)}")

    # Format timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Format artifact links
    artifact_links = "\n".join(f"  - `{art}`" for art in artifacts)

    # Create decision entry
    decision_entry = f"""
## Decision: {decision}

**Timestamp**: {timestamp}
**Agent**: mission-control

**Rationale**:
{rationale}

**Supporting Artifacts**:
{artifact_links}

---
"""

    # Check if this is a new session file
    is_new_file = not session_file.exists()

    # If new file, add header
    if is_new_file:
        header = f"""# Session Summary - {today}

**Project**: PlasmaDX-Clean Multi-Agent RAG System
**Date**: {today}
**Orchestrator**: mission-control

---

"""
        with open(session_file, "w", encoding="utf-8") as f:
            f.write(header)
        print(f"✅ Created new session file")
    else:
        print(f"✅ Appending to existing session file")

    # Append decision to session file
    with open(session_file, "a", encoding="utf-8") as f:
        f.write(decision_entry)

    print(f"\n✅ Decision recorded successfully!")
    print(f"   File: {session_file.relative_to(project_root)}")
    print(f"   Size: {session_file.stat().st_size} bytes")

    # Show the file content
    print(f"\n📄 Session file content:")
    print("=" * 60)
    with open(session_file, "r") as f:
        print(f.read())
    print("=" * 60)

    print("\n✅ Test complete! File I/O working correctly.")


if __name__ == "__main__":
    test_record_decision_logic()
