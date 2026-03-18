"""Tests for the aesthetic benchmark fixture and baseline capture.

These tests validate the benchmark fixture structure and provide utilities
for running baseline comparisons. The actual E2E runs are expensive and
should be triggered manually, not in CI.
"""

import json
from pathlib import Path

import pytest


BENCHMARK_PATH = Path(__file__).parent / "fixtures" / "aesthetic_benchmarks.json"

REQUIRED_FIELDS = {
    "id", "prompt", "effect_type", "category", "camera_distance", "difficulty",
}

VALID_EFFECT_TYPES = {
    "none", "fire", "smoke", "explosion", "water", "destruction",
    "cloth", "rigid_body", "particles",
}

VALID_CAMERA_DISTANCES = {"macro", "close", "medium", "wide", "extreme_wide"}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}


@pytest.fixture
def benchmarks():
    data = json.loads(BENCHMARK_PATH.read_text())
    return data["benchmarks"]


def test_fixture_exists():
    assert BENCHMARK_PATH.exists(), f"Benchmark fixture not found at {BENCHMARK_PATH}"


def test_fixture_loads_valid_json():
    data = json.loads(BENCHMARK_PATH.read_text())
    assert "benchmarks" in data
    assert "version" in data
    assert len(data["benchmarks"]) >= 8, "Need at least 8 benchmark prompts"


def test_all_benchmarks_have_required_fields(benchmarks):
    for bench in benchmarks:
        missing = REQUIRED_FIELDS - set(bench.keys())
        assert not missing, f"Benchmark {bench.get('id', '?')} missing fields: {missing}"


def test_ids_are_unique(benchmarks):
    ids = [b["id"] for b in benchmarks]
    assert len(ids) == len(set(ids)), f"Duplicate benchmark IDs: {[x for x in ids if ids.count(x) > 1]}"


def test_effect_types_valid(benchmarks):
    for bench in benchmarks:
        assert bench["effect_type"] in VALID_EFFECT_TYPES, (
            f"Benchmark {bench['id']}: invalid effect_type '{bench['effect_type']}'"
        )


def test_camera_distances_valid(benchmarks):
    for bench in benchmarks:
        assert bench["camera_distance"] in VALID_CAMERA_DISTANCES, (
            f"Benchmark {bench['id']}: invalid camera_distance '{bench['camera_distance']}'"
        )


def test_difficulties_valid(benchmarks):
    for bench in benchmarks:
        assert bench["difficulty"] in VALID_DIFFICULTIES, (
            f"Benchmark {bench['id']}: invalid difficulty '{bench['difficulty']}'"
        )


def test_prompts_are_nonempty(benchmarks):
    for bench in benchmarks:
        assert len(bench["prompt"]) > 20, (
            f"Benchmark {bench['id']}: prompt too short"
        )


def test_category_coverage(benchmarks):
    """Ensure benchmark set covers required categories."""
    categories = {b["category"] for b in benchmarks}
    required = {"tabletop_glass", "candlelit_macro", "cloth", "product_closeup", "explosion"}
    missing = required - categories
    assert not missing, f"Missing required benchmark categories: {missing}"


def test_at_least_one_no_physics_benchmark(benchmarks):
    """At least one benchmark should test pure geometry+material (no physics)."""
    no_physics = [b for b in benchmarks if b["effect_type"] == "none"]
    assert len(no_physics) >= 1
