"""
Hero Object Refinement Helper.

Provides deterministic refinement instructions for hero objects based on
their type. These instructions are injected into the Script Writer prompt
when StyleSpec identifies hero objects.

Zero LLM cost — keyword-based lookup of refinement recipes.

The goal is to make hero geometry stop defaulting to low-detail primitives.
Each recipe describes the modifiers and geometry operations that make
a specific object type look convincing at close range.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Refinement recipes by object category
# ---------------------------------------------------------------------------

# Each recipe is a list of concrete Blender instructions the Script Writer
# should follow when building that hero object type.

HERO_RECIPES: Dict[str, List[str]] = {
    "glass_vessel": [
        "Start from a UV sphere or lathe profile — NOT a cylinder",
        "Use SUBSURF (levels 2-3) for smooth curvature",
        "Use SOLIDIFY (thickness 0.002-0.004) for glass wall thickness — critical for refraction",
        "BEVEL on rim edge (width 0.001, segments 2) for realistic lip",
        "Material: Glass BSDF, IOR 1.5, Transmission Weight 1.0",
        "Slight green tint in transmission for thick glass: Volume Absorption (color (0.8, 0.95, 0.8), density 50)",
        "Caustic-ready: ensure light paths max bounces >= 8",
    ],
    "candle": [
        "Cylinder body with slight taper (scale top ring 0.95x)",
        "SUBSURF (level 2) for soft wax shape",
        "Drip detail: small displaced spheres or sculpted bumps on sides",
        "Wick: thin cylinder with dark material, slight bend",
        "Material: SSS shader (subsurface weight 0.3, subsurface radius (1.0, 0.5, 0.2))",
        "Translucency near flame: warm orange emission near top edge",
    ],
    "candle_holder": [
        "Lathe profile or cylinder with BEVEL and SUBSURF",
        "SOLIDIFY for hollow interior if open-top design",
        "Material: metallic brass (base color (0.7, 0.55, 0.2), metallic 1.0, roughness 0.3-0.5)",
        "Roughness variation via noise texture for aged look",
        "BEVEL on all hard edges (width 0.001-0.002, segments 2)",
    ],
    "bottle": [
        "Lathe profile from reference silhouette — NOT a cylinder with scale",
        "SUBSURF (levels 2-3) for smooth curves",
        "SOLIDIFY (thickness 0.003) for glass wall",
        "Separate cap/label geometry if visible",
        "Material: Glass BSDF with slight color tint in Volume Absorption",
        "BEVEL on lip and base edges",
    ],
    "mug_cup": [
        "Cylinder body with SUBSURF (level 2)",
        "Handle: torus or curve-to-mesh with SUBSURF",
        "SOLIDIFY for wall thickness (0.003-0.005)",
        "BEVEL on rim and base (width 0.001, segments 2)",
        "Material: ceramic glaze — roughness 0.15-0.25, slight SSS",
        "Interior may need separate darker material",
    ],
    "metal_weapon": [
        "Edge loops for blade profile, NOT boolean cuts",
        "BEVEL on cutting edges (narrow, segments 1) for specular catch",
        "SUBSURF (level 1-2) for handle/guard smoothness",
        "Material: metallic 1.0, roughness 0.05-0.15 for polished steel",
        "Anisotropic tangent direction along blade length",
        "Guard/pommel: separate objects with different roughness values",
    ],
    "fabric": [
        "CLOTH modifier for natural draping — NOT manually posed",
        "SUBSURF (level 1) BEFORE cloth for mesh density",
        "Pin group for attachment points (vertex group 'pin')",
        "Material: diffuse + sheen (Sheen Weight 0.5-0.8 for velvet/silk)",
        "Normal map or displacement for weave texture",
        "SOLIDIFY (tiny thickness 0.0005) to prevent single-face transparency",
    ],
    "stone_masonry": [
        "Cube base with BEVEL on all edges (width 0.002-0.005, segments 2)",
        "DISPLACE modifier with noise texture for surface irregularity",
        "Material: noise texture mixing 2-3 stone colors for variation",
        "Roughness variation: voronoi texture mixed into roughness channel",
        "Mortar lines: separate geometry or edge-wear effect",
    ],
    "wood_furniture": [
        "Boolean or inset for panel detail — NOT flat faces",
        "BEVEL on all visible edges (width 0.001-0.003, segments 2)",
        "Material: procedural wood texture (wave + noise, stretched along Y)",
        "Roughness variation: 0.3-0.6 range for natural wood",
        "Separate materials for different wood tones (top vs legs)",
        "Slight SUBSURF for rounded edges on turned legs/posts",
    ],
    "window_pane": [
        "Thin plane with SOLIDIFY for glass thickness (0.003-0.006)",
        "Frame: separate geometry with wood or metal material",
        "Material: Glass BSDF, IOR 1.5, Transmission Weight 1.0",
        "Frost effect: mix Glass BSDF with Diffuse via noise texture factor",
        "Mullions (dividers) as separate edge-beveled geometry",
    ],
    "lamp_lantern": [
        "Body: lathe profile or box with SUBSURF",
        "SOLIDIFY for hollow shell",
        "Glass panel inserts: separate objects with Glass BSDF",
        "Point light INSIDE the geometry for practical illumination",
        "Material: aged metal (roughness 0.5-0.8, slight color noise)",
        "BEVEL on all structural edges",
    ],
}

# Map detection patterns to recipe keys
_HERO_DETECTION: List[Tuple[str, str]] = [
    (r"\b(wine\s*glass|crystal\s*glass|tumbler|goblet|chalice)\b", "glass_vessel"),
    (r"\b(bottle|flask|decanter|carafe)\b", "bottle"),
    (r"\b(mug|cup|teacup|coffee\s*cup)\b", "mug_cup"),
    (r"\b(candle\s*holder|candlestick|candelabra|brass\s*holder)\b", "candle_holder"),
    (r"\b(candle|taper|pillar\s*candle|tea\s*light|votive)\b", "candle"),
    (r"\b(window|window\s*pane|glass\s*pane)\b", "window_pane"),
    (r"\b(sword|blade|knife|dagger|axe)\b", "metal_weapon"),
    (r"\b(cloth|fabric|curtain|flag|banner|tablecloth|scarf|silk|drape)\b", "fabric"),
    (r"\b(brick|stone|masonry|cobble|flagstone)\b", "stone_masonry"),
    (r"\b(table|chair|shelf|cabinet|desk|bench|stool)\b", "wood_furniture"),
    (r"\b(lamp|lantern|light\s*fixture|sconce)\b", "lamp_lantern"),
    (r"\b(vase|jar|pot|urn|bowl)\b", "glass_vessel"),  # similar refinement
]


def identify_hero_types(hero_objects: List[str]) -> Dict[str, str]:
    """Map hero object names to their refinement recipe category.

    Returns {object_name: recipe_key} for each hero that matches a known type.
    """
    result = {}
    for hero in hero_objects:
        for pattern, recipe_key in _HERO_DETECTION:
            if re.search(pattern, hero, re.IGNORECASE):
                result[hero] = recipe_key
                break
    return result


def get_refinement_instructions(
    hero_objects: List[str],
    camera_distance: str = "medium",
) -> str:
    """Generate hero-object refinement instructions for the Script Writer.

    Returns a formatted instruction block to inject into the prompt.
    Empty string if no heroes match known types.
    """
    hero_types = identify_hero_types(hero_objects)
    if not hero_types:
        return ""

    lines = [
        "",
        "## HERO OBJECT REFINEMENT (MANDATORY for create_geometry)",
        "These objects are the visual focal point. They MUST NOT be basic primitives.",
        "",
    ]

    # Increase detail requirements for close-up shots
    if camera_distance in ("close", "macro"):
        lines.append(
            "**CLOSE-UP SHOT** — hero objects will fill the frame. "
            "Every edge, bevel, and surface detail is visible. "
            "Use SUBSURF level 3+, add micro-bevels on all edges, "
            "and include surface imperfections in materials.\n"
        )

    seen_recipes = set()
    for hero_name, recipe_key in hero_types.items():
        if recipe_key in seen_recipes:
            continue
        seen_recipes.add(recipe_key)

        recipe = HERO_RECIPES.get(recipe_key, [])
        if not recipe:
            continue

        lines.append(f"### {hero_name} ({recipe_key})")
        for step in recipe:
            lines.append(f"- {step}")
        lines.append("")

    lines.append(
        "**ANTI-PATTERN:** A wine glass as `primitive_cylinder_add` with scale "
        "is ALWAYS wrong. A candle as a plain cylinder with no wax detail is ALWAYS wrong. "
        "If the hero object looks like a placeholder, the script has failed."
    )

    return "\n".join(lines)


def get_repair_refinement_instructions(
    hero_objects: List[str],
    issues: Optional[List[str]] = None,
) -> str:
    """Generate refinement instructions for repair passes targeting create_geometry.

    Focused version for when QA has flagged hero_object issues.
    """
    hero_types = identify_hero_types(hero_objects)
    if not hero_types:
        return ""

    lines = [
        "",
        "## HERO OBJECT REPAIR — QA flagged hero geometry as too primitive",
        "The current create_geometry() section needs MORE refinement, not parameter tweaks.",
        "Apply these specific improvements:",
        "",
    ]

    seen_recipes = set()
    for hero_name, recipe_key in hero_types.items():
        if recipe_key in seen_recipes:
            continue
        seen_recipes.add(recipe_key)

        recipe = HERO_RECIPES.get(recipe_key, [])
        if not recipe:
            continue

        lines.append(f"### Fix: {hero_name}")
        for step in recipe:
            lines.append(f"- {step}")
        lines.append("")

    if issues:
        lines.append("### QA Issues to Address")
        for issue in issues[:5]:
            lines.append(f"- {issue}")

    return "\n".join(lines)
