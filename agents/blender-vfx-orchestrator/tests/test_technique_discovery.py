"""Integration tests for technique discovery via semantic doc search.

These tests require a populated vector store. They are skipped when the
BLENDER_MANUAL_VECTOR_STORE_ID env var is not set (CI-safe).
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from tools.semantic_docs_tools import _search_vector_store

_HAS_STORE = bool(os.getenv("BLENDER_MANUAL_VECTOR_STORE_ID"))
skip_no_store = pytest.mark.skipif(not _HAS_STORE, reason="Vector store env var not set")


@skip_no_store
class TestTechniqueDiscovery:
    """Validate that semantic search returns diverse technique results."""

    def test_alternative_fire_techniques(self):
        results = _search_vector_store("alternative fire techniques", max_results=5, intent="manual")
        assert len(results) >= 2, f"Expected >=2 results, got {len(results)}"
        for r in results:
            assert r["score"] > 0, "Each result must have a positive score"

    def test_rigid_body_destruction_methods(self):
        results = _search_vector_store("rigid body destruction methods", max_results=5, intent="manual")
        assert len(results) >= 1, f"Expected >=1 result, got {len(results)}"
        for r in results:
            assert r["score"] > 0

    def test_api_query_no_regression(self):
        results = _search_vector_store("bpy.types.FluidDomainSettings", max_results=3, intent="api")
        assert len(results) >= 1, "API reference query must return results"
        for r in results:
            assert r["score"] > 0
