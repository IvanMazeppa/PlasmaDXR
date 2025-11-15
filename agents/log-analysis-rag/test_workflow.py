#!/usr/bin/env python3
"""
Test Complete LangGraph RAG Workflow
Validates end-to-end diagnostic pipeline
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from src.graph import get_workflow, GraphState


def test_diagnostic_workflow(question: str):
    """
    Test complete RAG workflow with a diagnostic question

    Args:
        question: Diagnostic question to test
    """
    print("=" * 80)
    print("TESTING LANGRAPH RAG WORKFLOW")
    print("=" * 80)
    print(f"\nQuestion: {question}\n")

    # Get compiled workflow
    workflow = get_workflow()

    # Create initial state
    initial_state: GraphState = {
        "path": "/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean/build/bin/Debug/logs",
        "question": question,
        "documents": [],
        "generation": "",
        "confidence": 0.0,
        "recommended_specialist": None,
        "file_line_refs": [],
        "artifact_paths": []
    }

    # Run workflow
    try:
        print("🚀 Starting workflow...\n")

        final_state = workflow.invoke(initial_state)

        print("\n" + "=" * 80)
        print("WORKFLOW COMPLETE")
        print("=" * 80)

        # Display results
        print(f"\n📊 DIAGNOSIS:")
        print(f"{final_state.get('generation', 'No generation produced')}\n")

        print(f"🎯 Confidence: {final_state.get('confidence', 0.0):.2f}")

        specialist = final_state.get('recommended_specialist')
        if specialist:
            print(f"👤 Recommended Specialist: {specialist}")
        else:
            print("👤 Recommended Specialist: None (no escalation)")

        file_refs = final_state.get('file_line_refs', [])
        if file_refs:
            print(f"\n📂 File References ({len(file_refs)}):")
            for ref in file_refs[:5]:  # Show first 5
                print(f"  - {ref}")

        artifacts = final_state.get('artifact_paths', [])
        if artifacts:
            print(f"\n📎 Artifacts ({len(artifacts)}):")
            for artifact in artifacts[:5]:  # Show first 5
                print(f"  - {artifact}")

        # Check hallucination
        hallucination_check = final_state.get('hallucination_check', 'unknown')
        if hallucination_check == 'yes':
            print("\n✅ Hallucination Check: PASSED (diagnosis is grounded)")
        elif hallucination_check == 'no':
            print("\n⚠️ Hallucination Check: FAILED (diagnosis not grounded)")
        else:
            print(f"\n❓ Hallucination Check: {hallucination_check}")

        return final_state

    except Exception as e:
        print(f"\n❌ Workflow failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Test with common PlasmaDX diagnostic questions
    test_questions = [
        "Why is DLSS Super Resolution not working?",
        "What caused the particle flashing issue?",
        "RTXDI M5 temporal accumulation shows patchwork pattern",
    ]

    # Run first test question
    test_diagnostic_workflow(test_questions[0])
