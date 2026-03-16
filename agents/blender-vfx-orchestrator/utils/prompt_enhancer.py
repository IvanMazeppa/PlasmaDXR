"""
Prompt Enhancer: injects look-dev craft hints into scene descriptions.

Zero LLM cost — deterministic text injection based on keyword detection.
Ensures the Script Writer spends code budget on visual craft, not just
technique plumbing.

The enhancement is additive: it appends a LOOK-DEV BRIEF section to the
description if one is not already present. It never modifies the user's
original creative intent.
"""

from __future__ import annotations

from typing import List, Optional


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

    print("\nAll tests passed!")
