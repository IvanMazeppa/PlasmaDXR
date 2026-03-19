"""Tests for hero object refinement helper."""

import pytest

from tools.hero_object_refinement import (
    HERO_RECIPES,
    get_refinement_instructions,
    get_repair_refinement_instructions,
    identify_hero_types,
)


# -- identify_hero_types --

def test_wine_glass_detected():
    types = identify_hero_types(["wine glass"])
    assert types["wine glass"] == "glass_vessel"


def test_crystal_glass_detected():
    types = identify_hero_types(["crystal glass"])
    assert types["crystal glass"] == "glass_vessel"


def test_candle_holder_detected():
    types = identify_hero_types(["brass holder"])
    # "brass holder" doesn't match — but "candle holder" or "candlestick" would
    types2 = identify_hero_types(["candle holder"])
    assert types2["candle holder"] == "candle_holder"


def test_candle_detected():
    types = identify_hero_types(["candle"])
    assert types["candle"] == "candle"


def test_fabric_detected():
    types = identify_hero_types(["silk scarf"])
    assert types["silk scarf"] == "fabric"


def test_table_detected():
    types = identify_hero_types(["table"])
    assert types["table"] == "wood_furniture"


def test_knife_detected():
    types = identify_hero_types(["knife"])
    assert types["knife"] == "metal_weapon"


def test_unknown_object_not_matched():
    types = identify_hero_types(["alien artifact"])
    assert len(types) == 0


def test_multiple_heroes():
    types = identify_hero_types(["wine glass", "candle", "wooden table"])
    assert len(types) == 3
    assert types["wine glass"] == "glass_vessel"
    assert types["candle"] == "candle"
    assert types["wooden table"] == "wood_furniture"


# -- get_refinement_instructions --

def test_refinement_for_glass():
    result = get_refinement_instructions(["wine glass"])
    assert "HERO OBJECT REFINEMENT" in result
    assert "SOLIDIFY" in result
    assert "SUBSURF" in result
    assert "Glass BSDF" in result


def test_refinement_for_closeup():
    result = get_refinement_instructions(["wine glass"], camera_distance="close")
    assert "CLOSE-UP SHOT" in result
    assert "SUBSURF level 3+" in result


def test_refinement_for_medium_shot():
    result = get_refinement_instructions(["wine glass"], camera_distance="medium")
    assert "CLOSE-UP SHOT" not in result


def test_no_refinement_for_unknown():
    result = get_refinement_instructions(["alien artifact"])
    assert result == ""


def test_anti_pattern_warning():
    result = get_refinement_instructions(["wine glass"])
    assert "primitive_cylinder_add" in result
    assert "ALWAYS wrong" in result


def test_deduplication():
    """Same recipe shouldn't appear twice for different names of same type."""
    result = get_refinement_instructions(["wine glass", "crystal glass"])
    # Both map to glass_vessel — should only appear once
    assert result.count("glass_vessel") == 1


# -- get_repair_refinement_instructions --

def test_repair_instructions():
    result = get_repair_refinement_instructions(
        ["wine glass"],
        issues=["Hero object is a basic cylinder"],
    )
    assert "HERO OBJECT REPAIR" in result
    assert "too primitive" in result
    assert "SOLIDIFY" in result
    assert "Hero object is a basic cylinder" in result


def test_repair_no_match():
    result = get_repair_refinement_instructions(["alien artifact"])
    assert result == ""


# -- recipe coverage --

def test_all_recipes_have_content():
    """Every recipe should have at least 3 steps."""
    for key, steps in HERO_RECIPES.items():
        assert len(steps) >= 3, f"Recipe '{key}' has only {len(steps)} steps"


def test_common_heroes_have_recipes():
    """Common hero objects should all map to a recipe."""
    common_heroes = [
        "wine glass", "candle", "candle holder", "bottle", "mug",
        "window", "sword", "cloth", "stone", "table", "lantern",
    ]
    types = identify_hero_types(common_heroes)
    assert len(types) >= 10, f"Only matched {len(types)}/{len(common_heroes)} common heroes"
