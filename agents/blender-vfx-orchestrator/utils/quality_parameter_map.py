"""
Deterministic quality-issue-to-parameter mapping.

Maps quality evaluation keywords to concrete parameter modifications
compatible with _modify_script_impl. Zero LLM cost, zero network.

Used as a fallback when:
- Learning Agent provides no parameter_modifications
- No code patterns matched from memory
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class ParameterSuggestion:
    """A single deterministic parameter suggestion from keyword matching."""
    issue_matched: str  # The keyword group that matched
    category: str  # e.g. "brightness", "resolution", "noise"
    parameter_modifications: Dict[str, Any]  # Compatible with _modify_script_impl
    confidence: float  # 0.0-1.0, boosted if matched on primary_issue
    explanation: str  # Human-readable reason


# Each entry: (keywords, category, params, explanation)
# keywords: list of substrings to match against issues (case-insensitive)
# params: dict compatible with _modify_script_impl
QUALITY_ISSUE_MAP: List[dict] = [
    {
        "keywords": ["overexposed", "too bright", "blown out", "white_screen", "white screen"],
        "category": "brightness",
        "params": {"emission_strength": 0.3, "density": 2.0},
        "explanation": "Reduce emission strength and increase density to darken overexposed volumes",
    },
    {
        "keywords": ["too dark", "underexposed", "black_screen", "black screen", "barely visible"],
        "category": "brightness",
        "params": {"emission_strength": 5.0, "density": 0.5, "light_energy": 500.0},
        "explanation": "Increase emission and light energy, reduce density for brighter volumes",
    },
    {
        "keywords": ["zero_lights", "no lights", "no lighting", "zero lights active"],
        "category": "lighting",
        "params": {"light_energy": 300.0},
        "explanation": "Add/increase light energy - scene has no active lighting",
    },
    {
        "keywords": ["domain boundary", "clipping", "cut off", "truncated", "volume clipping"],
        "category": "resolution",
        "params": {"resolution_max": 128, "domain_size": 4.0},
        "explanation": "Increase domain resolution and size to prevent boundary clipping",
    },
    {
        "keywords": ["low resolution", "blocky", "voxel", "pixelated", "chunky"],
        "category": "resolution",
        "params": {"resolution_max": 128},
        "explanation": "Increase simulation resolution for finer detail",
    },
    {
        "keywords": ["speckling", "noise", "speckle", "grainy", "grain", "fireflies"],
        "category": "noise",
        "params": {"noise_scale": 1.5, "render_samples": 256},
        "explanation": "Increase noise scale and render samples to reduce speckling artifacts",
    },
    {
        "keywords": ["weak temperature", "no temperature", "flat temperature", "temperature gradient"],
        "category": "temperature",
        "params": {"temperature_diff": 5.0, "heat": 2.0},
        "explanation": "Increase temperature differential and heat for visible thermal gradients",
    },
    {
        "keywords": ["no motion", "static", "frozen", "not moving", "no velocity"],
        "category": "dynamics",
        "params": {"initial_velocity": 3.0, "buoyancy_density": -0.5},
        "explanation": "Add initial velocity and buoyancy to create motion in the simulation",
    },
    {
        "keywords": ["dissipates too fast", "disappears", "fades too quick", "short lived"],
        "category": "lifetime",
        "params": {"use_dissolve_smoke": True, "dissolve_speed": 80},
        "explanation": "Slow dissolve rate to keep volumes visible longer",
    },
    {
        "keywords": ["too dense", "opaque", "solid", "can't see through", "no transparency"],
        "category": "density",
        "params": {"density": 0.3, "emission_strength": 2.0},
        "explanation": "Reduce density for more transparency while boosting emission for visibility",
    },
    {
        "keywords": ["too thin", "wispy", "barely there", "transparent", "faint"],
        "category": "density",
        "params": {"density": 5.0, "vorticity": 0.5},
        "explanation": "Increase density and add vorticity for more substantial volumes",
    },
    {
        "keywords": ["no color", "grey", "gray", "monochrome", "colorless"],
        "category": "color",
        "params": {"emission_color_r": 1.0, "emission_color_g": 0.4, "emission_color_b": 0.1},
        "explanation": "Add warm emission color to create visible color in the volume",
    },
    {
        "keywords": ["empty bake", "no simulation", "bake failed", "cache empty", "no fluid"],
        "category": "simulation",
        "params": {"resolution_max": 64, "timesteps_max": 4},
        "explanation": "Ensure basic simulation settings are present for bake to produce data",
    },
    {
        "keywords": ["slow", "sluggish", "heavy", "performance"],
        "category": "performance",
        "params": {"resolution_max": 64, "render_samples": 128},
        "explanation": "Reduce resolution and samples for faster iteration",
    },
    {
        "keywords": ["turbulence", "smooth", "uniform", "too regular", "no detail"],
        "category": "turbulence",
        "params": {"noise_scale": 2.5, "vorticity": 1.0},
        "explanation": "Increase noise scale and vorticity for more turbulent, detailed volumes",
    },
    # --- Liquid-specific issues ---
    {
        "keywords": ["coarse", "faceted", "bead-like", "beaded", "chunky stream", "low-res fluid",
                      "discretized", "stacked blobs", "chunky", "blobby", "stepped",
                      "instanced blobs", "lobe", "quantized"],
        "category": "liquid_resolution",
        "params": {"resolution_max": 200, "sampling_substeps": 8, "particle_radius": 1.2, "use_mesh": True},
        "explanation": "Increase liquid resolution and sampling substeps to reduce discretized/blobby jet appearance",
    },
    {
        "keywords": ["no water", "no fluid", "fluid not visible", "no liquid", "empty fluid", "no splash"],
        "category": "liquid_visibility",
        "params": {"resolution_max": 128, "use_plane_init": True},
        "explanation": "Ensure fluid source is active with sufficient resolution for visible output",
    },
    {
        "keywords": ["no spray", "no foam", "missing particles", "no mist", "no droplets",
                      "missing micro-droplets", "no micro", "no secondary"],
        "category": "liquid_particles",
        "params": {"use_spray_particles": True, "use_foam_particles": True, "use_bubble_particles": True},
        "explanation": "Enable secondary particle types (spray, foam, bubbles) for realistic liquid",
    },
    {
        "keywords": ["sharp edges", "jagged", "staircase", "aliased mesh", "mesh artifacts"],
        "category": "liquid_mesh",
        "params": {"mesh_concavity_upper": 3.5, "mesh_concavity_lower": 1.0, "mesh_particle_radius": 1.5},
        "explanation": "Smooth liquid mesh by adjusting concavity and particle radius settings",
    },
    {
        "keywords": ["weak splash", "no puddle", "puddle sheen", "flat puddle", "no reflection",
                      "no refraction", "dull water"],
        "category": "liquid_material",
        "params": {"ior": 1.33, "roughness": 0.02, "light_energy": 200.0},
        "explanation": "Improve liquid material (water IOR 1.33, low roughness for reflections, adequate lighting)",
    },
    # S1.5-C: Liquid-specific exposure and readability
    {
        "keywords": ["white water", "water overexposed", "liquid overexposed", "water blown out",
                      "indistinguishable", "white on white", "water not readable",
                      "water not visible", "can't see water"],
        "category": "liquid_exposure",
        "params": {"light_energy": 50.0, "world_strength": 0.1, "render_samples": 256},
        "explanation": "Reduce scene lighting for liquid close-ups — liquid scenes need much less light than volumes",
    },
    {
        "keywords": ["fluid exits domain", "all fluid gone", "fluid escaped", "empty domain",
                      "domain too small", "fluid out of bounds"],
        "category": "liquid_velocity",
        "params": {"velocity_normal": 1.5, "velocity_random": 0.2, "domain_size": 3.0},
        "explanation": "Reduce velocity and increase domain size — fluid is leaving the simulation domain",
    },
    {
        "keywords": ["no flow", "inflow not working", "emitter inactive", "source not emitting",
                      "fluid source empty"],
        "category": "liquid_inflow",
        "params": {"use_plane_init": True, "velocity_normal": 1.5, "use_initial_velocity": True},
        "explanation": "Enable initial velocity and plane init for liquid inflow to produce visible output",
    },
]


def map_quality_issues_to_params(
    issues: List[str],
    primary_issue: Optional[str] = None,
    effect_type: Optional[str] = None,  # noqa: ARG001 - reserved for per-effect tuning
) -> List[ParameterSuggestion]:
    """
    Map quality issue strings to concrete parameter suggestions.

    Uses substring matching against QUALITY_ISSUE_MAP keywords.
    Primary issue gets a confidence boost.

    Args:
        issues: List of issue strings from QualityOutput.issues
        primary_issue: The primary issue (gets confidence boost)
        effect_type: Optional effect type for future per-effect tuning

    Returns:
        List of ParameterSuggestion, sorted by confidence descending
    """
    suggestions: List[ParameterSuggestion] = []
    matched_categories: set = set()

    # Combine all issue text for matching
    all_issue_text = " ".join(issues).lower()
    primary_text = (primary_issue or "").lower()

    for entry in QUALITY_ISSUE_MAP:
        keywords = entry["keywords"]
        matched_keyword = None

        # Check if any keyword matches in combined issue text or primary issue
        for kw in keywords:
            kw_lower = kw.lower()
            if kw_lower in all_issue_text or kw_lower in primary_text:
                matched_keyword = kw
                break

        if matched_keyword is None:
            continue

        category = entry["category"]

        # Skip duplicate categories (first match wins per category)
        if category in matched_categories:
            continue
        matched_categories.add(category)

        # Confidence: base 0.6, boosted to 0.85 if matched on primary_issue
        is_primary_match = any(kw.lower() in primary_text for kw in keywords)
        confidence = 0.85 if is_primary_match else 0.6

        suggestions.append(ParameterSuggestion(
            issue_matched=matched_keyword,
            category=category,
            parameter_modifications=dict(entry["params"]),
            confidence=confidence,
            explanation=entry["explanation"],
        ))

    # Sort by confidence descending
    suggestions.sort(key=lambda s: s.confidence, reverse=True)
    return suggestions


def merge_suggestions_to_modifications(
    suggestions: List[ParameterSuggestion],
    max_params: int = 6,
) -> Dict[str, Any]:
    """
    Merge multiple suggestions into a single modifications dict.

    Deduplicates by keeping the highest-confidence value per parameter.

    Args:
        suggestions: Output from map_quality_issues_to_params
        max_params: Maximum parameters to include (prevents shotgun changes)

    Returns:
        Dict compatible with _modify_script_impl modifications parameter
    """
    if not suggestions:
        return {}

    # Track best confidence per parameter
    param_confidence: Dict[str, float] = {}
    param_values: Dict[str, Any] = {}

    for suggestion in suggestions:
        for param, value in suggestion.parameter_modifications.items():
            existing_conf = param_confidence.get(param, -1.0)
            if suggestion.confidence > existing_conf:
                param_confidence[param] = suggestion.confidence
                param_values[param] = value

    # Sort by confidence and take top max_params
    sorted_params = sorted(
        param_values.keys(),
        key=lambda p: param_confidence[p],
        reverse=True,
    )

    return {p: param_values[p] for p in sorted_params[:max_params]}


def format_suggestions_for_prompt(suggestions: List[ParameterSuggestion]) -> str:
    """
    Format suggestions as a human-readable string for injection into LLM prompts.

    Used to enrich Modification Coordinator and Learning Agent prompts.
    """
    if not suggestions:
        return ""

    lines = ["## Deterministic Parameter Suggestions (from quality analysis)"]
    for s in suggestions:
        params_str = ", ".join(f"{k}={v}" for k, v in s.parameter_modifications.items())
        lines.append(
            f"- [{s.category}] {s.explanation} → {params_str} "
            f"(confidence: {s.confidence:.0%}, matched: '{s.issue_matched}')"
        )
    lines.append("Use these as starting points. Override with script-specific values if you find better targets.")
    return "\n".join(lines)
