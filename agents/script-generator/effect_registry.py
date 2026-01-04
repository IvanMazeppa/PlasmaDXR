"""
Effect Type Registry for Blender VFX Multi-Agent Pipeline

Phase 2.5.1: Extends pipeline beyond volumetric-only simulations to support:
- Volumetric effects (pyro, nebula, sun) -> VDB output
- Mesh-based physics (soft body, cloth, rigid body) -> Blend/PNG output
- Custom/LLM-driven effects -> Auto-detected output

This registry determines:
1. Simulation category (volumetric, mesh, custom)
2. Output format (vdb, blend, auto)
3. Execution pattern (bake_export, live_render, auto)
4. Available templates for each effect type

References:
- MULTI_AGENT_IMPROVEMENT_PLAN_V2.md: Phase 2.5.1 specification
- GEMINI_FEEDBACK_ANALYSIS_AND_PLAN_AMENDMENTS.md: Mesh physics gap analysis
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum


class SimulationCategory(Enum):
    """Category of simulation determining processing pipeline."""
    VOLUMETRIC = "volumetric"  # Mantaflow fluid -> OpenVDB
    MESH = "mesh"              # Soft body, cloth, rigid body -> Blend/PNG
    CUSTOM = "custom"          # LLM-driven, auto-detect output


class OutputFormat(Enum):
    """Output file format for simulation results."""
    VDB = "vdb"      # OpenVDB volumetric data
    BLEND = "blend"  # Blender scene file with baked animation
    PNG = "png"      # Rendered image sequence
    AUTO = "auto"    # Auto-detect based on simulation type


class ExecutionPattern(Enum):
    """Execution pattern for running the simulation."""
    BAKE_EXPORT = "bake_export"  # bpy.ops.fluid.bake_all() then export VDB
    LIVE_RENDER = "live_render"  # Frame-by-frame render with physics update
    AUTO = "auto"                 # Auto-detect based on effect type


@dataclass
class EffectTypeDefinition:
    """Complete definition of an effect type."""
    category: SimulationCategory
    output: OutputFormat
    simulation_pattern: ExecutionPattern
    templates: List[str]
    description: str
    blender_physics_type: Optional[str] = None  # e.g., "FLUID", "SOFT_BODY", "CLOTH"
    requires_bake: bool = True
    evaluation_metrics: Optional[List[str]] = None  # Metrics for quality evaluation


# =============================================================================
# EFFECT TYPE REGISTRY
# =============================================================================

EFFECT_TYPES: Dict[str, Dict[str, Any]] = {
    # =========================================================================
    # VOLUMETRIC EFFECTS (VDB output, bake_export pattern)
    # =========================================================================

    "pyro": {
        "category": "volumetric",
        "output": "vdb",
        "simulation_pattern": "bake_export",
        "templates": ["explosion", "fire", "smoke", "fireball"],
        "description": "Fire and smoke simulations using Mantaflow gas solver",
        "blender_physics_type": "FLUID",
        "fluid_domain_type": "GAS",
        "requires_bake": True,
        "evaluation_metrics": ["edge_density", "noise_frequency", "dynamic_range", "warm_ratio"],
    },

    "explosion": {
        "category": "volumetric",
        "output": "vdb",
        "simulation_pattern": "bake_export",
        "templates": ["aerial_burst", "ground_explosion", "fireball", "shockwave"],
        "description": "Explosive detonation effects with fast expanding fireballs",
        "blender_physics_type": "FLUID",
        "fluid_domain_type": "GAS",
        "requires_bake": True,
        "evaluation_metrics": ["edge_density", "noise_frequency", "dynamic_range", "warm_ratio"],
    },

    "fire": {
        "category": "volumetric",
        "output": "vdb",
        "simulation_pattern": "bake_export",
        "templates": ["campfire", "torch", "bonfire", "wildfire"],
        "description": "Sustained fire effects with flames and heat distortion",
        "blender_physics_type": "FLUID",
        "fluid_domain_type": "GAS",
        "requires_bake": True,
        "evaluation_metrics": ["edge_density", "warm_ratio", "brightness"],
    },

    "smoke": {
        "category": "volumetric",
        "output": "vdb",
        "simulation_pattern": "bake_export",
        "templates": ["plume", "fog", "mist", "exhaust"],
        "description": "Smoke and fog volumetric effects",
        "blender_physics_type": "FLUID",
        "fluid_domain_type": "GAS",
        "requires_bake": True,
        "evaluation_metrics": ["coverage", "density_variance", "turbulence"],
    },

    "nebula": {
        "category": "volumetric",
        "output": "vdb",
        "simulation_pattern": "bake_export",
        "templates": ["emission_nebula", "reflection_nebula", "dark_nebula", "planetary_nebula"],
        "description": "Cosmic gas clouds and interstellar medium",
        "blender_physics_type": "FLUID",
        "fluid_domain_type": "GAS",
        "requires_bake": True,
        "evaluation_metrics": ["color_distribution", "structure_complexity", "coverage"],
    },

    "sun": {
        "category": "volumetric",
        "output": "vdb",
        "simulation_pattern": "bake_export",
        "templates": ["main_sequence_star", "solar_prominence", "stellar_flare", "corona"],
        "description": "Stellar surface and solar phenomena",
        "blender_physics_type": "FLUID",
        "fluid_domain_type": "GAS",
        "requires_bake": True,
        "evaluation_metrics": ["granulation", "limb_darkening", "prominence_coverage", "color_temperature"],
    },

    "liquid": {
        "category": "volumetric",
        "output": "vdb",
        "simulation_pattern": "bake_export",
        "templates": ["water", "splash", "pour", "wave"],
        "description": "Liquid fluid simulations (water, oil, etc.)",
        "blender_physics_type": "FLUID",
        "fluid_domain_type": "LIQUID",
        "requires_bake": True,
        "evaluation_metrics": ["surface_smoothness", "splash_detail", "transparency"],
    },

    # =========================================================================
    # MESH-BASED PHYSICS (Blend/PNG output, live_render pattern)
    # =========================================================================

    "soft_body": {
        "category": "mesh",
        "output": "blend",
        "simulation_pattern": "live_render",
        "templates": ["jelly", "bounce", "squish", "rubber", "gelatin"],
        "description": "Deformable mesh physics (jelly, rubber, organic)",
        "blender_physics_type": "SOFT_BODY",
        "requires_bake": False,  # Live render calculates per-frame
        "evaluation_metrics": ["surface_smoothness", "silhouette_clarity", "deformation_quality"],
        # Soft body stability settings (from Gemini 3 Pro discoveries)
        "recommended_settings": {
            "step_min": 20,      # Default 5 - prevents jitter explosions
            "step_max": 100,     # Default 10 - adaptive stepping
            "damping": 2.0,      # Default 0.5 - prevents energy buildup
            "friction": 0.5,
            "goal_spring": 0.5,
            "goal_friction": 0.5,
        },
        "preprocessing": ["voxel_remesh"],  # Required for joined primitives
    },

    "cloth": {
        "category": "mesh",
        "output": "blend",
        "simulation_pattern": "live_render",
        "templates": ["fabric", "flag", "curtain", "drape", "banner", "cape"],
        "description": "Fabric and cloth physics simulation",
        "blender_physics_type": "CLOTH",
        "requires_bake": False,
        "evaluation_metrics": ["fold_quality", "drape_realism", "motion_smoothness"],
        "recommended_settings": {
            "quality": 10,
            "collision_quality": 5,
            "self_collision": True,
        },
    },

    "rigid_body": {
        "category": "mesh",
        "output": "blend",
        "simulation_pattern": "live_render",
        "templates": ["destruction", "dominos", "pile", "shatter", "collapse"],
        "description": "Rigid body physics (destruction, collisions)",
        "blender_physics_type": "RIGID_BODY",
        "requires_bake": False,
        "evaluation_metrics": ["collision_accuracy", "fragment_distribution", "motion_realism"],
        "recommended_settings": {
            "collision_shape": "CONVEX_HULL",  # or MESH for precise
            "friction": 0.5,
            "bounciness": 0.5,
        },
        "addon_required": "cell_fracture",  # For shattering effects
    },

    # =========================================================================
    # CUSTOM/GENERIC (Auto-detect output, auto pattern)
    # =========================================================================

    "custom": {
        "category": "custom",
        "output": "auto",
        "simulation_pattern": "auto",
        "templates": [],
        "description": "LLM-driven custom effects - auto-detect simulation type",
        "blender_physics_type": None,  # Determined at runtime
        "requires_bake": None,  # Determined at runtime
        "evaluation_metrics": ["clip_score"],  # CLIP only for custom
    },

    "generic": {
        "category": "custom",
        "output": "auto",
        "simulation_pattern": "auto",
        "templates": [],
        "description": "Generic effects without specific physics type",
        "blender_physics_type": None,
        "requires_bake": None,
        "evaluation_metrics": ["clip_score"],
    },
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_effect_type(name: str) -> Optional[Dict[str, Any]]:
    """
    Get effect type definition by name.

    Args:
        name: Effect type name (e.g., "pyro", "soft_body")

    Returns:
        Effect type definition dict or None if not found
    """
    return EFFECT_TYPES.get(name.lower())


def get_category(effect_type: str) -> SimulationCategory:
    """
    Get simulation category for an effect type.

    Args:
        effect_type: Effect type name

    Returns:
        SimulationCategory enum value
    """
    effect = get_effect_type(effect_type)
    if effect:
        return SimulationCategory(effect["category"])
    return SimulationCategory.CUSTOM


def get_output_format(effect_type: str) -> OutputFormat:
    """
    Get output format for an effect type.

    Args:
        effect_type: Effect type name

    Returns:
        OutputFormat enum value
    """
    effect = get_effect_type(effect_type)
    if effect:
        return OutputFormat(effect["output"])
    return OutputFormat.AUTO


def get_execution_pattern(effect_type: str) -> ExecutionPattern:
    """
    Get execution pattern for an effect type.

    Args:
        effect_type: Effect type name

    Returns:
        ExecutionPattern enum value
    """
    effect = get_effect_type(effect_type)
    if effect:
        return ExecutionPattern(effect["simulation_pattern"])
    return ExecutionPattern.AUTO


def is_volumetric(effect_type: str) -> bool:
    """Check if effect type is volumetric (VDB output)."""
    return get_category(effect_type) == SimulationCategory.VOLUMETRIC


def is_mesh_physics(effect_type: str) -> bool:
    """Check if effect type is mesh-based physics."""
    return get_category(effect_type) == SimulationCategory.MESH


def requires_live_render(effect_type: str) -> bool:
    """Check if effect type requires live render pattern."""
    return get_execution_pattern(effect_type) == ExecutionPattern.LIVE_RENDER


def get_evaluation_metrics(effect_type: str) -> List[str]:
    """
    Get appropriate evaluation metrics for an effect type.

    Volumetric effects use VFX diagnostics (edge_density, warm_ratio, etc.)
    Mesh effects use surface quality metrics (smoothness, silhouette, etc.)
    Custom effects use CLIP only

    Args:
        effect_type: Effect type name

    Returns:
        List of metric names to use for evaluation
    """
    effect = get_effect_type(effect_type)
    if effect and effect.get("evaluation_metrics"):
        return effect["evaluation_metrics"]

    # Default fallback by category
    category = get_category(effect_type)
    if category == SimulationCategory.VOLUMETRIC:
        return ["edge_density", "noise_frequency", "dynamic_range"]
    elif category == SimulationCategory.MESH:
        return ["surface_smoothness", "silhouette_clarity", "shadow_definition"]
    else:
        return ["clip_score"]


def get_templates(effect_type: str) -> List[str]:
    """Get available templates for an effect type."""
    effect = get_effect_type(effect_type)
    if effect:
        return effect.get("templates", [])
    return []


def get_recommended_settings(effect_type: str) -> Dict[str, Any]:
    """
    Get recommended physics settings for an effect type.

    Particularly important for mesh physics where wrong settings
    cause simulation instability (jitter, explosions).
    """
    effect = get_effect_type(effect_type)
    if effect:
        return effect.get("recommended_settings", {})
    return {}


def list_volumetric_types() -> List[str]:
    """List all volumetric effect types."""
    return [name for name, effect in EFFECT_TYPES.items()
            if effect["category"] == "volumetric"]


def list_mesh_types() -> List[str]:
    """List all mesh-based physics effect types."""
    return [name for name, effect in EFFECT_TYPES.items()
            if effect["category"] == "mesh"]


def list_all_types() -> List[str]:
    """List all registered effect types."""
    return list(EFFECT_TYPES.keys())


def get_blender_physics_type(effect_type: str) -> Optional[str]:
    """
    Get the Blender physics modifier type for an effect.

    Returns:
        "FLUID", "SOFT_BODY", "CLOTH", "RIGID_BODY", or None
    """
    effect = get_effect_type(effect_type)
    if effect:
        return effect.get("blender_physics_type")
    return None


# =============================================================================
# EFFECT TYPE VALIDATION
# =============================================================================

def validate_effect_type(effect_type: str) -> Dict[str, Any]:
    """
    Validate and return normalized effect type information.

    Args:
        effect_type: User-provided effect type name

    Returns:
        Dict with validation result and normalized info
    """
    normalized = effect_type.lower().strip()

    # Direct match
    if normalized in EFFECT_TYPES:
        return {
            "valid": True,
            "normalized_name": normalized,
            "effect_info": EFFECT_TYPES[normalized],
            "suggestion": None,
        }

    # Check for partial matches (e.g., "explosion" might match "pyro")
    for name, effect in EFFECT_TYPES.items():
        if normalized in effect.get("templates", []):
            return {
                "valid": True,
                "normalized_name": name,
                "effect_info": effect,
                "suggestion": f"'{normalized}' is a template of '{name}'",
            }

    # No match - suggest similar or default to custom
    suggestions = []
    for name in EFFECT_TYPES.keys():
        if normalized[:3] in name or name[:3] in normalized:
            suggestions.append(name)

    return {
        "valid": False,
        "normalized_name": "custom",
        "effect_info": EFFECT_TYPES["custom"],
        "suggestion": f"Unknown effect type '{effect_type}'. Similar: {suggestions or ['custom']}",
    }


# =============================================================================
# MAIN - Testing
# =============================================================================

if __name__ == "__main__":
    print("=== Effect Type Registry ===\n")

    print("VOLUMETRIC EFFECTS (VDB output):")
    for name in list_volumetric_types():
        effect = EFFECT_TYPES[name]
        print(f"  - {name}: {effect['description']}")
        print(f"    Templates: {effect['templates']}")

    print("\nMESH PHYSICS (Blend output):")
    for name in list_mesh_types():
        effect = EFFECT_TYPES[name]
        print(f"  - {name}: {effect['description']}")
        print(f"    Templates: {effect['templates']}")
        if effect.get("recommended_settings"):
            print(f"    Recommended: {effect['recommended_settings']}")

    print("\n=== Category Tests ===")
    test_types = ["pyro", "soft_body", "cloth", "custom", "explosion"]
    for t in test_types:
        cat = get_category(t)
        out = get_output_format(t)
        pat = get_execution_pattern(t)
        print(f"{t}: category={cat.value}, output={out.value}, pattern={pat.value}")
