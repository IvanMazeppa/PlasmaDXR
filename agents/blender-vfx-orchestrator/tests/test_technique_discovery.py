"""Integration tests for technique discovery via semantic doc search.

These tests require a populated vector store and an OpenAI API key.
They are skipped when OPENAI_API_KEY is not available (CI-safe).
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import pytest

from tools.semantic_docs_tools import _search_vector_store

_HAS_API_KEY = bool(os.getenv("OPENAI_API_KEY"))
skip_no_api = pytest.mark.skipif(not _HAS_API_KEY, reason="OPENAI_API_KEY not set")


@skip_no_api
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
