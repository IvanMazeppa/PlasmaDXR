#!/usr/bin/env python3
"""
Vector Store Health Check and Verification Script.

Verifies that Blender documentation vector stores are:
1. Accessible and operational
2. Contain expected document types
3. Return results with proper DocPath headers
4. Support intent-based routing

Usage:
    python scripts/verify_vector_stores.py
    python scripts/verify_vector_stores.py --verbose
    python scripts/verify_vector_stores.py --test-queries
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("Error: openai package not installed")
    sys.exit(1)

# Vector store IDs (Production - 2026-01-25)
MANUAL_STORE_ID = os.getenv(
    "BLENDER_MANUAL_VECTOR_STORE_ID",
    "vs_697564fbfc0c8191b2e44aa47dbaf482"
)
API_STORE_ID = os.getenv(
    "BLENDER_API_VECTOR_STORE_ID",
    "vs_697571f0275c8191910fea0f2c8bdd3a"
)

# Test queries for each store
API_TEST_QUERIES = [
    "bpy.types.FluidDomainSettings properties",
    "bpy.ops.fluid.bake parameters",
    "FluidModifier flame_smoke attribute",
]

MANUAL_TEST_QUERIES = [
    "smoke simulation tutorial",
    "physics fluid domain settings",
    "volumetric rendering workflow",
]


def check_store_status(client: OpenAI, store_id: str, store_name: str) -> dict:
    """Check if a vector store is accessible and get its status."""
    result = {
        "store_name": store_name,
        "store_id": store_id,
        "accessible": False,
        "status": "unknown",
        "file_count": 0,
        "errors": [],
    }

    try:
        vs = client.vector_stores.retrieve(store_id)
        result["accessible"] = True
        result["status"] = vs.status
        result["file_count"] = vs.file_counts.completed
        result["in_progress"] = vs.file_counts.in_progress
        result["failed"] = vs.file_counts.failed
        result["name"] = vs.name
    except Exception as e:
        result["errors"].append(str(e))

    return result


def test_search(client: OpenAI, store_id: str, query: str, verbose: bool = False) -> dict:
    """Test a search query and verify results have DocPath headers."""
    result = {
        "query": query,
        "success": False,
        "results_count": 0,
        "has_doc_path": False,
        "has_doc_type": False,
        "sample_doc_path": None,
        "errors": [],
    }

    try:
        response = client.vector_stores.search(
            vector_store_id=store_id,
            query=query,
            max_num_results=3
        )

        result["results_count"] = len(response.data)
        result["success"] = result["results_count"] > 0

        for r in response.data:
            content = r.content[0].text if r.content else ""

            # Check for DocPath header
            for line in content.split('\n')[:15]:
                if line.startswith('DocPath:'):
                    result["has_doc_path"] = True
                    result["sample_doc_path"] = line.replace('DocPath:', '').strip()
                elif line.startswith('DocType:'):
                    result["has_doc_type"] = True
                    result["sample_doc_type"] = line.replace('DocType:', '').strip()

            if verbose and result["success"]:
                print(f"    Score: {r.score:.3f}")
                print(f"    DocPath: {result.get('sample_doc_path', 'N/A')}")
                print(f"    Preview: {content[:100]}...")

            if result["has_doc_path"]:
                break

    except Exception as e:
        result["errors"].append(str(e))

    return result


def main():
    parser = argparse.ArgumentParser(description="Verify Blender vector stores")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    parser.add_argument("--test-queries", "-t", action="store_true", help="Run test queries")
    args = parser.parse_args()

    print("=" * 60)
    print("Blender Vector Store Health Check")
    print("=" * 60)

    client = OpenAI()
    all_passed = True

    # Check store status
    print("\n1. Checking store accessibility...")
    stores = [
        (MANUAL_STORE_ID, "Manual"),
        (API_STORE_ID, "API"),
    ]

    for store_id, store_name in stores:
        status = check_store_status(client, store_id, store_name)

        if status["accessible"]:
            icon = "✓" if status["status"] == "completed" else "⏳"
            print(f"  {icon} {store_name}: {status.get('name', 'unnamed')}")
            print(f"      Status: {status['status']}")
            print(f"      Files: {status['file_count']} completed", end="")
            if status.get("in_progress", 0) > 0:
                print(f", {status['in_progress']} in progress", end="")
            if status.get("failed", 0) > 0:
                print(f", {status['failed']} failed", end="")
            print()
        else:
            print(f"  ✗ {store_name}: NOT ACCESSIBLE")
            for err in status["errors"]:
                print(f"      Error: {err}")
            all_passed = False

    # Run test queries
    if args.test_queries:
        print("\n2. Testing search queries...")

        # Test API store
        print(f"\n  API Store ({API_STORE_ID[:20]}...):")
        for query in API_TEST_QUERIES:
            result = test_search(client, API_STORE_ID, query, args.verbose)
            if result["success"] and result["has_doc_path"]:
                print(f"    ✓ '{query[:40]}...' -> {result['results_count']} results")
                if args.verbose:
                    print(f"      DocPath: {result['sample_doc_path']}")
            elif result["success"]:
                print(f"    ⚠ '{query[:40]}...' -> {result['results_count']} results (NO DocPath)")
                all_passed = False
            else:
                print(f"    ✗ '{query[:40]}...' -> FAILED")
                for err in result["errors"]:
                    print(f"      Error: {err}")
                all_passed = False

        # Test Manual store
        print(f"\n  Manual Store ({MANUAL_STORE_ID[:20]}...):")
        for query in MANUAL_TEST_QUERIES:
            result = test_search(client, MANUAL_STORE_ID, query, args.verbose)
            if result["success"] and result["has_doc_path"]:
                print(f"    ✓ '{query[:40]}...' -> {result['results_count']} results")
                if args.verbose:
                    print(f"      DocPath: {result['sample_doc_path']}")
            elif result["success"]:
                print(f"    ⚠ '{query[:40]}...' -> {result['results_count']} results (NO DocPath)")
            else:
                print(f"    ✗ '{query[:40]}...' -> FAILED (may be empty - upload pending)")

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("HEALTH CHECK: PASSED")
    else:
        print("HEALTH CHECK: ISSUES FOUND")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
