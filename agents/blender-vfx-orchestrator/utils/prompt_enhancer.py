"""
Prompt Enhancer: injects look-dev craft hints into scene descriptions
and extracts a structured StyleSpec for the pipeline.

Zero LLM cost — deterministic text injection based on keyword detection.
Ensures the Script Writer spends code budget on visual craft, not just
technique plumbing.

The enhancement is additive: it appends a LOOK-DEV BRIEF section to the
description if one is not already present. It never modifies the user's
original creative intent.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Keyword-to-craft-hint mappings
# ---------------------------------------------------------------------------

_FIRE_HINTS = [
    "Flame color gradient: blue-white base (hottest), bright yellow body, orange-tipped wisps that curl and detach",
    "Ember glow: point lights or emission meshes at the base, warm orange with slight flicker via keyframed intensity",
    "Smoke interaction: thin wisps rise from flame tips, use Volume Scatter for light-scattering in smoke column",
    "Ground bounce: warm orange area light below fire source for reflected firelight on surrounding surfaces",
]

_WATER_HINTS = [
    "Liquid material: Glass BSDF + Volume Absorption for depth-dependent color (clear at surface, tinted at depth)",
    "Caustics: enable light paths caustics or fake with projected texture; visible on receiving surfaces",
    "Meniscus: liquid surface tension visible at container contact edges",
    "Refraction: IOR 1.333 for water, 1.345 for wine; subsurface color shift at depth",
]

_SMOKE_HINTS = [
    "Volume scattering: Principled Volume with anisotropy 0.3-0.5 for directional light scatter",
    "Smoke color variation: use Color Ramp on density attribute for depth-based color shift",
    "Light shafts: area light or sun lamp through smoke creates visible god rays with volume scatter",
    "Dissipation: smoke thins and becomes more transparent as it rises — use dissolve_speed",
]

_EXPLOSION_HINTS = [
    "Fireball core: extremely bright emission (blackbody 2000-4000K), surrounded by orange convection",
    "Shockwave ring: brief expanding ring of compressed air (subtle distortion or dust ring)",
    "Debris: small rigid body fragments launched outward with drag, trailing thin smoke",
    "Temporal arc: bright flash → expanding fireball → rising mushroom → dissipating smoke column",
]

_GLASS_HINTS = [
    "Glass material: Glass BSDF with IOR 1.5, slight green tint in transmission for thick glass",
    "Fracture edges: emission or bright specular on shard edges to catch light",
    "Shard size distribution: mix of large structural pieces and small splinters",
    "Motion blur: enable for fast-moving shards to convey velocity",
]

_UNIVERSAL_HINTS = [
    "Color management: AgX view transform (preserves HDR detail in emissives)",
    "Depth of field: shallow DOF on hero subject, background elements soften naturally",
    "World lighting: gradient or environment texture — never flat single-color world shader",
    "Hero object finish: primary subject gets full PBR material (roughness, normal, clearcoat where appropriate)",
]

# Map effect keywords to their craft hint sets
_EFFECT_HINTS = {
    "fire": _FIRE_HINTS,
    "flame": _FIRE_HINTS,
    "candle": _FIRE_HINTS,
    "torch": _FIRE_HINTS,
    "campfire": _FIRE_HINTS,
    "bonfire": _FIRE_HINTS,
    "water": _WATER_HINTS,
    "liquid": _WATER_HINTS,
    "pour": _WATER_HINTS,
    "splash": _WATER_HINTS,
    "rain": _WATER_HINTS,
    "wine": _WATER_HINTS,
    "smoke": _SMOKE_HINTS,
    "fog": _SMOKE_HINTS,
    "mist": _SMOKE_HINTS,
    "steam": _SMOKE_HINTS,
    "explosion": _EXPLOSION_HINTS,
    "blast": _EXPLOSION_HINTS,
    "detonate": _EXPLOSION_HINTS,
    "glass": _GLASS_HINTS,
    "shatter": _GLASS_HINTS,
    "fracture": _GLASS_HINTS,
    "destruction": _GLASS_HINTS,
}

# ---------------------------------------------------------------------------
# Camera distance detection
# ---------------------------------------------------------------------------

_CLOSE_KEYWORDS = [
    "close-up", "closeup", "close up", "macro", "tight", "detail",
    "tabletop", "intimate", "hero shot",
]
_WIDE_KEYWORDS = [
    "wide", "establishing", "landscape", "aerial", "overhead",
    "panoramic", "full scene",
]

# ---------------------------------------------------------------------------
# Mood detection
# ---------------------------------------------------------------------------

_MOOD_KEYWORDS: Dict[str, List[str]] = {
    "intimate": ["candle", "intimate", "warm", "cozy", "close", "personal"],
    "dramatic": ["dramatic", "explosive", "intense", "violent", "powerful", "fierce"],
    "ethereal": ["ethereal", "dreamy", "soft", "gentle", "mystical", "cosmic"],
    "dark": ["dark", "shadow", "night", "noir", "gloomy", "sinister"],
    "bright": ["bright", "daylight", "sunny", "vivid", "cheerful"],
}

# ---------------------------------------------------------------------------
# World lighting mode detection
# ---------------------------------------------------------------------------

_LIGHTING_MODE_KEYWORDS: Dict[str, List[str]] = {
    "dark_world_practicals": ["candle", "campfire", "torch", "lantern", "firelight", "night", "dark room"],
    "procedural_sky": ["outdoor", "sky", "daylight", "sunset", "sunrise", "dawn", "dusk"],
    "env_texture": ["studio", "product", "hdri", "environment"],
}

# ---------------------------------------------------------------------------
# Hero object extraction
# ---------------------------------------------------------------------------

_HERO_OBJECT_PATTERNS = [
    # Containers/vessels
    (r"\b(wine\s*glass|crystal\s*glass|tumbler|goblet|chalice|bottle|vase|jar|mug|cup)\b", "glass_vessel"),
    (r"\b(candle|taper|pillar\s*candle|tea\s*light)\b", "candle"),
    # Architectural
    (r"\b(window|window\s*pane|glass\s*pane|mirror)\b", "glass_pane"),
    (r"\b(brick|stone|masonry|wall)\b", "masonry"),
    # Props
    (r"\b(sword|blade|knife|dagger)\b", "metal_weapon"),
    (r"\b(cloth|fabric|curtain|flag|banner|tablecloth)\b", "fabric"),
    (r"\b(wood|wooden|log|plank|table)\b", "wood"),
]

_HERO_MATERIAL_GOALS: Dict[str, List[str]] = {
    "glass_vessel": ["Glass BSDF", "IOR 1.5", "transmission", "SOLIDIFY for thickness", "SUBSURF for smoothness"],
    "candle": ["SSS wax material", "slight translucency", "warm color"],
    "glass_pane": ["Glass BSDF", "IOR 1.5", "thin panel thickness", "edge specular"],
    "masonry": ["roughness variation", "color noise", "BEVEL edge breakup"],
    "metal_weapon": ["metallic 1.0", "low roughness", "anisotropic highlights"],
    "fabric": ["velvet BSDF or diffuse+sheen", "subsurface for thin fabric", "CLOTH modifier wrinkles"],
    "wood": ["wood texture", "roughness variation", "grain direction"],
}


def _detect_effects(description: str) -> List[str]:
    """Find which effect categories are present in the description."""
    desc_lower = description.lower()
    seen = set()
    matched = []
    for keyword, hints in _EFFECT_HINTS.items():
        hint_id = id(hints)  # dedupe by hint set identity
        if keyword in desc_lower and hint_id not in seen:
            seen.add(hint_id)
            matched.append(keyword)
    return matched


def _detect_camera_distance(description: str) -> str:
    """Detect camera distance class from description keywords."""
    desc_lower = description.lower()
    for kw in _CLOSE_KEYWORDS:
        if kw in desc_lower:
            return "close"
    for kw in _WIDE_KEYWORDS:
        if kw in desc_lower:
            return "wide"
    return "medium"


def _detect_mood(description: str) -> str:
    """Detect dominant mood from description keywords."""
    desc_lower = description.lower()
    best_mood = ""
    best_count = 0
    for mood, keywords in _MOOD_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in desc_lower)
        if count > best_count:
            best_count = count
            best_mood = mood
    return best_mood or "cinematic"


def _detect_lighting_mode(description: str) -> str:
    """Detect world lighting mode from description keywords."""
    desc_lower = description.lower()
    for mode, keywords in _LIGHTING_MODE_KEYWORDS.items():
        if any(kw in desc_lower for kw in keywords):
            return mode
    return "dark_world_practicals"  # safe default for VFX


def _extract_hero_objects(description: str) -> tuple[List[str], List[str]]:
    """Extract hero objects and their material goals from description."""
    heroes = []
    material_goals = []
    seen_types = set()
    for pattern, obj_type in _HERO_OBJECT_PATTERNS:
        match = re.search(pattern, description, re.IGNORECASE)
        if match and obj_type not in seen_types:
            seen_types.add(obj_type)
            heroes.append(match.group(0).strip())
            material_goals.extend(_HERO_MATERIAL_GOALS.get(obj_type, []))
    return heroes, material_goals


def _extract_palette(description: str) -> List[str]:
    """Extract color keywords from description."""
    color_patterns = [
        r"\b(amber|gold|golden|warm\s+orange|deep\s+brown|burgundy|crimson|ruby)\b",
        r"\b(teal|turquoise|cobalt|navy|cerulean|azure)\b",
        r"\b(emerald|forest\s+green|olive|sage)\b",
        r"\b(ivory|cream|pearl|bone|chalk)\b",
        r"\b(charcoal|slate|ash|obsidian|jet\s+black)\b",
        r"\b(rose|blush|coral|salmon|magenta)\b",
    ]
    colors = []
    for pat in color_patterns:
        for match in re.finditer(pat, description, re.IGNORECASE):
            color = match.group(0).strip().lower()
            if color not in colors:
                colors.append(color)
    return colors


def extract_style_spec(
    description: str,
    effect_type: Optional[str] = None,
) -> Dict[str, object]:
    """Extract a StyleSpec dict from a scene description.

    Returns a dict suitable for StyleSpec(**result) construction.
    Zero LLM cost — pure keyword extraction.
    """
    heroes, material_goals = _extract_hero_objects(description)
    palette = _extract_palette(description)
    mood = _detect_mood(description)
    camera_distance = _detect_camera_distance(description)
    lighting_mode = _detect_lighting_mode(description)

    # Infer setting from description (first sentence as approximation)
    sentences = description.split(".")
    setting = sentences[0].strip() if sentences else ""

    # Infer DOF from camera distance
    dof_intent = "shallow on hero" if camera_distance in ("close", "macro") else "moderate"

    # Infer realism target
    realism = "cinematic"

    # Infer camera framing from description
    camera_framing = ""
    desc_lower = description.lower()
    if "close-up" in desc_lower or "closeup" in desc_lower:
        camera_framing = "close-up"
    elif "macro" in desc_lower:
        camera_framing = "macro"
    elif "wide" in desc_lower:
        camera_framing = "wide shot"
    elif "medium" in desc_lower:
        camera_framing = "medium shot"

    return {
        "setting": setting[:200],  # cap length
        "mood": mood,
        "palette": palette,
        "camera_framing": camera_framing,
        "camera_distance_class": camera_distance,
        "dof_intent": dof_intent,
        "world_lighting_mode": lighting_mode,
        "hero_objects": heroes,
        "hero_material_goals": material_goals,
        "support_prop_budget": "moderate" if len(heroes) > 1 else "minimal",
        "realism_target": realism,
    }


def enhance_description(
    description: str,
    effect_type: Optional[str] = None,
) -> str:
    """Enrich a scene description with look-dev craft hints.

    Returns the original description with a LOOK-DEV BRIEF appended.
    If the description already contains a LOOK-DEV BRIEF, returns it unchanged.

    Args:
        description: The user's scene description.
        effect_type: The EffectType value (e.g., "fire", "water"). Used as
            fallback if no effect keywords are detected in the description.
    """
    # Don't double-enhance
    if "LOOK-DEV BRIEF" in description:
        return description

    # Collect relevant hints
    hints: List[str] = list(_UNIVERSAL_HINTS)  # always include universal

    matched_effects = _detect_effects(description)

    # If no effect detected from description text, use the effect_type param
    if not matched_effects and effect_type:
        et = effect_type.lower()
        if et in _EFFECT_HINTS:
            matched_effects = [et]

    for effect_kw in matched_effects:
        effect_hints = _EFFECT_HINTS.get(effect_kw, [])
        for h in effect_hints:
            if h not in hints:
                hints.append(h)

    # Build the brief
    brief_lines = [
        "",
        "LOOK-DEV BRIEF (auto-generated — apply these visual craft defaults):",
    ]
    for hint in hints:
        brief_lines.append(f"- {hint}")

    return description + "\n".join(brief_lines)


# ---------------------------------------------------------------------------
# Tests (inline, run with: python -m utils.prompt_enhancer)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Test 1: Fire description gets fire hints
    desc = "A campfire burns in a stone ring. Flames rise 30cm."
    enhanced = enhance_description(desc, effect_type="fire")
    assert "LOOK-DEV BRIEF" in enhanced
    assert "AgX" in enhanced
    assert "blue-white base" in enhanced
    print("[PASS] Fire enhancement")

    # Test 2: Water description gets water hints
    desc = "Wine pours into a crystal glass"
    enhanced = enhance_description(desc, effect_type="water")
    assert "Glass BSDF" in enhanced
    assert "IOR 1.333" in enhanced
    print("[PASS] Water enhancement")

    # Test 3: No double-enhancement
    enhanced2 = enhance_description(enhanced, effect_type="water")
    assert enhanced2 == enhanced
    print("[PASS] No double-enhance")

    # Test 4: Universal hints always present
    desc = "A nebula in deep space"
    enhanced = enhance_description(desc, effect_type="nebula")
    assert "AgX" in enhanced
    assert "Depth of field" in enhanced
    print("[PASS] Universal hints")

    # Test 5: Effect type fallback when no keywords match
    desc = "A mysterious effect"
    enhanced = enhance_description(desc, effect_type="fire")
    assert "blue-white base" in enhanced
    print("[PASS] Effect type fallback")

    # Test 6: Multiple effects detected
    desc = "An explosion sends shards of glass flying through smoke"
    enhanced = enhance_description(desc, effect_type="explosion")
    assert "Fireball core" in enhanced
    assert "Glass BSDF" in enhanced or "Fracture edges" in enhanced
    assert "Volume scattering" in enhanced
    print("[PASS] Multiple effects")

    # Test 7: StyleSpec extraction
    desc = "Wine pours into a crystal glass on a dark wooden table. Candlelight illuminates the scene with warm amber tones."
    spec = extract_style_spec(desc, effect_type="water")
    assert spec["mood"] == "intimate" or spec["mood"] == "dark"
    assert "wine glass" in [h.lower() for h in spec["hero_objects"]] or "crystal glass" in [h.lower() for h in spec["hero_objects"]]
    assert spec["world_lighting_mode"] == "dark_world_practicals"
    assert len(spec["hero_material_goals"]) > 0
    print(f"[PASS] StyleSpec extraction: {len(spec['hero_objects'])} heroes, mood={spec['mood']}")

    # Test 8: StyleSpec with close-up
    desc = "Extreme close-up of a candle flame. Macro detail."
    spec = extract_style_spec(desc, effect_type="fire")
    assert spec["camera_distance_class"] == "close"
    assert spec["dof_intent"] == "shallow on hero"
    print(f"[PASS] StyleSpec close-up: distance={spec['camera_distance_class']}")

    # Test 9: Palette extraction
    desc = "Deep burgundy wine, warm amber candlelight, golden highlights on cream tablecloth"
    spec = extract_style_spec(desc, effect_type="water")
    assert len(spec["palette"]) >= 2
    print(f"[PASS] Palette extraction: {spec['palette']}")

    print("\nAll tests passed!")
