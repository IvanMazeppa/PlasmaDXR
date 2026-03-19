"""Tests for camera readability hints in prompt_enhancer."""

import pytest

from utils.prompt_enhancer import enhance_description, extract_style_spec


# -- Camera hints in enhance_description --

def test_closeup_gets_closeup_camera_hints():
    enhanced = enhance_description("Close-up of a wine glass catching firelight", "fire")
    assert "CAMERA READABILITY" in enhanced
    assert "CLOSE-UP CAMERA" in enhanced
    assert "50-85mm" in enhanced
    assert "f/2.8" in enhanced


def test_wide_gets_wide_camera_hints():
    enhanced = enhance_description("Wide establishing shot of a burning village", "fire")
    assert "WIDE SHOT CAMERA" in enhanced
    assert "24-35mm" in enhanced
    assert "f/8.0" in enhanced


def test_medium_default_camera_hints():
    enhanced = enhance_description("A candle burns on a table", "fire")
    assert "MEDIUM SHOT CAMERA" in enhanced
    assert "35-50mm" in enhanced


def test_macro_triggers_closeup_hints():
    enhanced = enhance_description("Macro detail of wax dripping from a candle", "fire")
    assert "CLOSE-UP CAMERA" in enhanced


def test_camera_hints_after_effect_hints():
    """Camera hints should appear after effect-specific hints."""
    enhanced = enhance_description("Close-up of a campfire", "fire")
    look_dev_pos = enhanced.index("LOOK-DEV BRIEF")
    camera_pos = enhanced.index("CAMERA READABILITY")
    assert camera_pos > look_dev_pos


def test_no_double_camera_hints():
    """Double-enhancement should not duplicate camera hints."""
    enhanced = enhance_description("Close-up of a wine glass", "water")
    enhanced2 = enhance_description(enhanced, "water")
    assert enhanced2 == enhanced  # blocked by LOOK-DEV BRIEF check


# -- Camera fields in extract_style_spec --

def test_style_spec_close_distance():
    spec = extract_style_spec("Extreme close-up of a candle flame", "fire")
    assert spec["camera_distance_class"] == "close"
    assert spec["dof_intent"] == "shallow on hero"
    assert spec["camera_framing"] == "close-up"


def test_style_spec_wide_distance():
    spec = extract_style_spec("Wide panoramic shot of a forest fire", "fire")
    assert spec["camera_distance_class"] == "wide"
    assert spec["camera_framing"] == "wide shot"


def test_style_spec_medium_default():
    spec = extract_style_spec("A candle on a table", "fire")
    assert spec["camera_distance_class"] == "medium"
    assert spec["dof_intent"] == "moderate"


def test_style_spec_macro_framing():
    spec = extract_style_spec("Macro shot of water droplets", "water")
    assert spec["camera_distance_class"] == "close"
    assert spec["camera_framing"] == "macro"
