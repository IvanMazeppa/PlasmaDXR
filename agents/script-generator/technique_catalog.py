"""
Blender VFX Technique Catalog - Extracted from Blender 5.0 Manual

This catalog defines categorically different pyro/gas simulation techniques.
Each technique uses distinct parameter ranges to produce visually unique results.

Sources:
- bpy.types.FluidDomainSettings (Python API)
- bpy.types.FluidFlowSettings (Python API)
- physics/fluid/type/domain/gas/noise.html (Blender Manual)
- physics/fluid/type/domain/gas/adaptive_domain.html (Blender Manual)
- physics/fluid/type/effector.html (Blender Manual)

Parameter Ranges (from Blender 5.0 Python API):
- burning_rate: [0.01, 4.0], default 0.75
- flame_smoke: [0.0, 8.0], default 1.0
- flame_vorticity: [0.0, 2.0], default 0.5
- flame_max_temp: [1.0, 10.0], default 3.0
- flame_ignition: [0.5, 5.0], default 1.5
- alpha (density buoyancy): [-5.0, 5.0], default 1.0
- beta (heat buoyancy): [-5.0, 5.0], default 1.0
- dissolve_speed: [1, 10000], default 5
- fuel_amount: [0.0, 10.0], default 1.0
- temperature: [-10.0, 10.0], default 1.0
- velocity_normal: [-100.0, 100.0], default 0.0
- velocity_random: [0.0, 10.0], default 0.0

Noise Upres Parameters (from physics/fluid/type/domain/gas/noise.html):
- noise_scale: [1, 10], default 2 (resolution multiplier)
- noise_strength: [0.0, 10.0], default 1.0 (turbulence intensity)
- noise_pos_scale: [0.0001, 10.0], default 2.0 (vortex size)

Emission Dynamics Profiles:
- burst: Explosive initial emission that decays rapidly (explosions)
- sustained: Continuous emission at steady rate (fires, jets)
- pulsing: Rhythmic emission waves (volcanic, engine exhaust)
- decay_only: No new emission, existing fuel burns out (aftermath)
"""

import random
from typing import Dict, Any, Tuple, List, Optional

# =============================================================================
# PYRO TECHNIQUES - Categorically different fire/smoke effects
# =============================================================================

PYRO_TECHNIQUES: Dict[str, Dict[str, Any]] = {

    "rising_mushroom": {
        "description": "Classic mushroom cloud - fast upward expansion with curling cap",
        "visual_signature": "Rising column with distinctive mushroom cap formation",
        "keywords": ["mushroom", "nuclear", "rising", "upward", "classic"],

        # Domain settings - HIGH turbulence, FAST burn, MODERATE smoke
        "domain_params": {
            "burning_rate": (1.2, 2.0),      # Fast combustion
            "flame_smoke": (1.5, 2.5),        # Moderate-heavy smoke
            "flame_vorticity": (0.8, 1.2),    # High turbulence for cap curl
            "flame_max_temp": (4.0, 6.0),     # Hot flames
            "flame_ignition": (1.0, 2.0),     # Quick ignition
            "alpha": (0.8, 1.2),              # Standard density buoyancy
            "beta": (1.5, 2.5),               # High heat buoyancy (rises fast)
            "dissolve_speed": (80, 150),      # Moderate persistence
        },

        # Noise upres for fine detail (2-4x visual improvement)
        "noise_params": {
            "use_noise": True,
            "noise_scale": (2, 3),            # 2-3x resolution boost
            "noise_strength": (1.0, 1.5),     # Moderate turbulence detail
            "noise_pos_scale": (2.0, 3.0),    # Medium vortex size for cap curls
        },

        # Emission dynamics - burst profile (explosive start, rapid decay)
        "emission_dynamics": {
            "profile": "burst",
            "peak_frame": (1, 5),             # Explosion peaks early
            "decay_start": (8, 12),           # Start decay around frame 10
            "decay_end": (25, 35),            # Fully decayed by frame 30
            "peak_fuel_multiplier": (2.0, 3.0),  # 2-3x fuel at peak
            "decay_fuel_multiplier": (0.1, 0.3), # 10-30% fuel at decay
            "peak_velocity_multiplier": (2.0, 2.5),  # Strong initial burst
        },

        # Flow settings
        "flow_params": {
            "flow_type": "BOTH",              # Fire + Smoke
            "fuel_amount": (2.5, 4.0),        # High fuel for sustained flames
            "temperature": (2.0, 4.0),        # Hot emission
            "velocity_normal": (3.0, 6.0),    # Strong upward initial velocity
        },

        # Emitter configuration
        "emitter": {
            "shape": "sphere",
            "position_z": "ground",           # Emit from ground level
            "radius": (0.8, 1.2),
        },

        # Camera suggestion
        "camera": {
            "angle": "low_angle_upward",      # Look up at rising cloud
            "distance": "medium",
        },
    },

    "ground_hugger": {
        "description": "Low, spreading explosion that hugs terrain like napalm",
        "visual_signature": "Spreading ground fog with embedded orange fire tendrils",
        "keywords": ["ground", "spreading", "napalm", "low", "crawling", "terrain"],

        # Domain settings - LOW turbulence, SLOW burn, VERY HEAVY smoke
        "domain_params": {
            "burning_rate": (0.2, 0.5),       # Slow smoldering burn
            "flame_smoke": (4.0, 6.0),        # Very heavy smoke production
            "flame_vorticity": (0.1, 0.3),    # Minimal turbulence
            "flame_max_temp": (2.0, 3.0),     # Cooler flames
            "flame_ignition": (0.5, 1.0),     # Low ignition threshold
            "alpha": (1.5, 2.5),              # High density = sinks
            "beta": (-0.5, 0.3),              # NEGATIVE heat buoyancy = stays low
            "dissolve_speed": (200, 500),     # Long persistence
        },

        # Noise upres - subtle detail for creeping smoke
        "noise_params": {
            "use_noise": True,
            "noise_scale": (2, 2),            # 2x resolution boost
            "noise_strength": (0.5, 0.8),     # Low turbulence (ground hugging)
            "noise_pos_scale": (3.0, 5.0),    # Larger, slower vortices
        },

        # Emission dynamics - sustained profile (continuous burn)
        "emission_dynamics": {
            "profile": "sustained",
            "peak_frame": (5, 15),            # Gradual ramp up
            "decay_start": (40, 50),          # Late decay (long burn)
            "decay_end": (60, 80),            # Very gradual fade
            "peak_fuel_multiplier": (1.0, 1.2),  # Steady fuel
            "decay_fuel_multiplier": (0.3, 0.5), # Slow decay
            "peak_velocity_multiplier": (1.0, 1.2),  # Low velocity spread
        },

        "flow_params": {
            "flow_type": "BOTH",
            "fuel_amount": (1.0, 2.0),        # Moderate fuel
            "temperature": (0.5, 1.5),        # Cool emission
            "velocity_normal": (-1.0, 1.0),   # Minimal directional velocity
            "velocity_random": (2.0, 4.0),    # Random spreading motion
        },

        "emitter": {
            "shape": "plane",                 # Flat emission surface
            "position_z": "ground",
            "scale_xy": (2.0, 3.0),           # Wide emission area
        },

        "camera": {
            "angle": "ground_level",
            "distance": "close",
        },
    },

    "aerial_burst": {
        "description": "Mid-air spherical detonation with minimal smoke",
        "visual_signature": "Bright spherical fireball expanding outward",
        "keywords": ["aerial", "burst", "spherical", "fireball", "airburst", "midair"],

        # Domain settings - MODERATE turbulence, VERY FAST burn, MINIMAL smoke
        "domain_params": {
            "burning_rate": (2.5, 4.0),       # Very fast combustion
            "flame_smoke": (0.3, 0.8),        # Minimal smoke
            "flame_vorticity": (0.4, 0.7),    # Moderate turbulence
            "flame_max_temp": (7.0, 10.0),    # Extremely hot
            "flame_ignition": (2.0, 3.5),     # High ignition temp
            "alpha": (0.3, 0.6),              # Low density buoyancy
            "beta": (0.5, 1.0),               # Low heat buoyancy (expands, doesn't rise)
            "dissolve_speed": (20, 50),       # Fast dissipation
        },

        # Noise upres - high detail for fireball surface
        "noise_params": {
            "use_noise": True,
            "noise_scale": (3, 4),            # High resolution boost
            "noise_strength": (1.2, 1.8),     # Intense surface detail
            "noise_pos_scale": (1.0, 2.0),    # Small, fast vortices
        },

        # Emission dynamics - burst profile (instant flash, fast decay)
        "emission_dynamics": {
            "profile": "burst",
            "peak_frame": (1, 2),             # Instant explosion
            "decay_start": (3, 5),            # Immediate decay
            "decay_end": (15, 25),            # Fast burnout
            "peak_fuel_multiplier": (3.0, 4.0),  # Massive initial fuel
            "decay_fuel_multiplier": (0.0, 0.1), # Complete burnout
            "peak_velocity_multiplier": (3.0, 4.0),  # Explosive expansion
        },

        "flow_params": {
            "flow_type": "FIRE",              # Fire only, minimal smoke
            "flow_behavior": "GEOMETRY",      # Shape-based emission
            "fuel_amount": (5.0, 8.0),        # Very high fuel for intense flash
            "temperature": (5.0, 8.0),        # Very hot
            "velocity_normal": (8.0, 15.0),   # Strong outward expansion
        },

        "emitter": {
            "shape": "sphere",
            "position_z": "elevated",         # Mid-air
            "radius": (0.5, 0.8),             # Compact initial size
        },

        "camera": {
            "angle": "side_view",
            "distance": "far",
        },
    },

    "slow_smolder": {
        "description": "Slow-burning embers with heavy trailing smoke",
        "visual_signature": "Dense smoke columns with occasional flame wisps",
        "keywords": ["smolder", "smoke", "embers", "slow", "trailing", "aftermath"],

        # Domain settings - MINIMAL turbulence, VERY SLOW burn, MAXIMUM smoke
        "domain_params": {
            "burning_rate": (0.05, 0.15),     # Very slow burn
            "flame_smoke": (6.0, 8.0),        # Maximum smoke production
            "flame_vorticity": (0.05, 0.15),  # Almost no turbulence
            "flame_max_temp": (1.5, 2.5),     # Cool, smoldering flames
            "flame_ignition": (0.5, 0.8),     # Low ignition
            "alpha": (1.0, 1.5),              # Moderate density
            "beta": (0.3, 0.8),               # Gentle rise
            "dissolve_speed": (500, 2000),    # Very long persistence
        },

        # Noise upres - very subtle for lazy smoke wisps
        "noise_params": {
            "use_noise": True,
            "noise_scale": (2, 2),            # Minimal resolution boost
            "noise_strength": (0.3, 0.5),     # Very low turbulence
            "noise_pos_scale": (4.0, 6.0),    # Large, lazy vortices
        },

        # Emission dynamics - decay_only profile (aftermath, no new ignition)
        "emission_dynamics": {
            "profile": "decay_only",
            "peak_frame": (1, 1),             # Start with existing embers
            "decay_start": (1, 5),            # Already decaying
            "decay_end": (80, 120),           # Very long burnout
            "peak_fuel_multiplier": (0.5, 0.8),  # Low initial fuel
            "decay_fuel_multiplier": (0.05, 0.15), # Barely smoldering at end
            "peak_velocity_multiplier": (0.3, 0.5),  # Minimal velocity
        },

        "flow_params": {
            "flow_type": "SMOKE",             # Smoke dominant
            "fuel_amount": (0.3, 0.8),        # Low fuel
            "temperature": (0.2, 0.8),        # Cool
        },

        "emitter": {
            "shape": "scattered_points",      # Multiple small emission points
            "position_z": "ground",
        },

        "camera": {
            "angle": "atmospheric",
            "distance": "medium",
        },
    },

    "plasma_jet": {
        "description": "Directed high-temperature plasma stream",
        "visual_signature": "Intense focused flame beam with no smoke",
        "keywords": ["plasma", "jet", "beam", "directed", "intense", "torch"],

        # Domain settings - MAXIMUM turbulence, MAXIMUM burn, NO smoke
        "domain_params": {
            "burning_rate": (3.5, 4.0),       # Maximum combustion
            "flame_smoke": (0.0, 0.1),        # No smoke
            "flame_vorticity": (1.5, 2.0),    # Maximum turbulence
            "flame_max_temp": (9.0, 10.0),    # Maximum temperature
            "flame_ignition": (3.0, 5.0),     # High ignition threshold
            "alpha": (0.1, 0.3),              # Minimal density effect
            "beta": (0.2, 0.5),               # Low buoyancy (directional flow)
            "dissolve_speed": (5, 15),        # Very fast dissipation
        },

        # Noise upres - high frequency for plasma instabilities
        "noise_params": {
            "use_noise": True,
            "noise_scale": (3, 4),            # High resolution boost
            "noise_strength": (1.5, 2.0),     # Maximum turbulence detail
            "noise_pos_scale": (0.5, 1.0),    # Tiny, fast vortices
        },

        # Emission dynamics - sustained profile (continuous stream)
        "emission_dynamics": {
            "profile": "sustained",
            "peak_frame": (1, 3),             # Fast ramp up
            "decay_start": (45, 55),          # Late decay
            "decay_end": (50, 60),            # Sharp cutoff
            "peak_fuel_multiplier": (1.0, 1.0),  # Constant fuel
            "decay_fuel_multiplier": (0.0, 0.1), # Sharp cutoff
            "peak_velocity_multiplier": (1.0, 1.2),  # Steady velocity
        },

        "flow_params": {
            "flow_type": "FIRE",
            "fuel_amount": (8.0, 10.0),       # Maximum fuel
            "temperature": (8.0, 10.0),       # Maximum temperature
            "velocity_normal": (20.0, 40.0),  # Very strong directional velocity
        },

        "emitter": {
            "shape": "cone",                  # Directional emission
            "position_z": "mid",
            "direction": "horizontal",
        },

        "camera": {
            "angle": "side_profile",
            "distance": "medium",
        },
    },

    "volcanic_plume": {
        "description": "Sustained vertical eruption with ash and pyroclastic flow",
        "visual_signature": "Tall column with spreading umbrella top and falling debris",
        "keywords": ["volcanic", "eruption", "plume", "ash", "pyroclastic"],

        "domain_params": {
            "burning_rate": (0.8, 1.2),
            "flame_smoke": (3.0, 5.0),        # Heavy ash/smoke
            "flame_vorticity": (0.5, 0.8),
            "flame_max_temp": (5.0, 7.0),
            "alpha": (1.8, 2.5),              # Heavy particles
            "beta": (2.0, 3.0),               # Strong upward force
            "dissolve_speed": (300, 600),
        },

        # Noise upres - high detail for pyroclastic turbulence
        "noise_params": {
            "use_noise": True,
            "noise_scale": (3, 4),            # High resolution for ash detail
            "noise_strength": (1.5, 2.5),     # Strong turbulence (pyroclastic chaos)
            "noise_pos_scale": (1.5, 2.5),    # Medium vortices for rolling ash clouds
        },

        # Emission dynamics - pulsing profile (rhythmic eruption bursts)
        "emission_dynamics": {
            "profile": "pulsing",
            "peak_frame": (5, 10),            # Initial buildup
            "decay_start": (60, 80),          # Long sustained eruption
            "decay_end": (100, 150),          # Very gradual wind-down
            "peak_fuel_multiplier": (1.5, 2.0),  # Moderate peaks
            "decay_fuel_multiplier": (0.4, 0.6), # Still active during decay
            "peak_velocity_multiplier": (1.8, 2.2),  # Pulsing ejection strength
            "pulse_period": (8, 15),          # Frames between pulses
            "pulse_intensity": (0.3, 0.5),    # 30-50% variation
        },

        "flow_params": {
            "flow_type": "BOTH",
            "fuel_amount": (3.0, 5.0),
            "temperature": (3.0, 5.0),
            "velocity_normal": (10.0, 20.0),  # Strong upward jet
        },

        "emitter": {
            "shape": "cone",
            "position_z": "ground",
            "direction": "vertical",
        },

        "camera": {
            "angle": "distant_low",
            "distance": "very_far",
        },
    },

    "flashover": {
        "description": "Rapid room-scale fire spread (backdraft effect)",
        "visual_signature": "Sudden explosive expansion of flames across entire space",
        "keywords": ["flashover", "backdraft", "room", "rapid", "spread"],

        "domain_params": {
            "burning_rate": (2.0, 3.0),       # Fast spread
            "flame_smoke": (2.0, 3.0),
            "flame_vorticity": (1.0, 1.5),    # Chaotic
            "flame_max_temp": (6.0, 8.0),
            "alpha": (0.5, 1.0),
            "beta": (1.0, 1.5),
            "dissolve_speed": (40, 80),
        },

        # Noise upres - chaotic detail for backdraft turbulence
        "noise_params": {
            "use_noise": True,
            "noise_scale": (2, 3),            # Moderate resolution boost
            "noise_strength": (1.2, 1.8),     # High turbulence (chaotic spread)
            "noise_pos_scale": (1.0, 2.0),    # Fast, medium-sized vortices
        },

        # Emission dynamics - burst profile (sudden ignition, rapid spread)
        "emission_dynamics": {
            "profile": "burst",
            "peak_frame": (3, 8),             # Rapid buildup to flashover
            "decay_start": (20, 30),          # Starts consuming fuel
            "decay_end": (40, 60),            # Burns out
            "peak_fuel_multiplier": (2.5, 3.5),  # Massive fuel release at flashover
            "decay_fuel_multiplier": (0.2, 0.4), # Rapid consumption
            "peak_velocity_multiplier": (2.5, 3.0),  # Explosive spread
        },

        "flow_params": {
            "flow_type": "BOTH",
            "fuel_amount": (4.0, 6.0),
            "temperature": (4.0, 6.0),
            "velocity_random": (5.0, 10.0),   # Chaotic spread
        },

        "emitter": {
            "shape": "volume",                # Fill volume
            "position_z": "ceiling",          # Start high, spread down
        },

        "camera": {
            "angle": "interior",
            "distance": "close",
        },
    },

    "stellar_flare": {
        "description": "Cosmic-scale plasma eruption (for space scenes)",
        "visual_signature": "Arcing plasma loops with extreme temperature gradient",
        "keywords": ["stellar", "flare", "solar", "cosmic", "space", "plasma", "arc"],

        "domain_params": {
            "burning_rate": (1.5, 2.5),
            "flame_smoke": (0.0, 0.2),        # No smoke in space
            "flame_vorticity": (0.3, 0.6),    # Smooth arcs
            "flame_max_temp": (10.0, 10.0),   # Maximum temp
            "alpha": (0.0, 0.1),              # No gravity effects
            "beta": (0.0, 0.1),               # No buoyancy
            "dissolve_speed": (100, 200),
        },

        # Noise upres - smooth plasma instabilities
        "noise_params": {
            "use_noise": True,
            "noise_scale": (2, 3),            # Moderate resolution boost
            "noise_strength": (0.6, 1.0),     # Moderate - plasma is relatively smooth
            "noise_pos_scale": (2.0, 4.0),    # Large, smooth vortices (magnetic field scale)
        },

        # Emission dynamics - burst profile (flare eruption and decay)
        "emission_dynamics": {
            "profile": "burst",
            "peak_frame": (5, 15),            # Flare builds up
            "decay_start": (30, 50),          # Peak emission period
            "decay_end": (80, 120),           # Long cooling tail
            "peak_fuel_multiplier": (2.0, 3.0),  # Intense peak emission
            "decay_fuel_multiplier": (0.15, 0.3), # Long cooling plasma
            "peak_velocity_multiplier": (2.0, 2.5),  # Magnetic reconnection burst
        },

        "flow_params": {
            "flow_type": "FIRE",
            "fuel_amount": (6.0, 10.0),
            "temperature": (10.0, 10.0),      # Maximum
            "velocity_normal": (5.0, 15.0),
        },

        "emitter": {
            "shape": "arc",
            "position_z": "surface",
        },

        "material": {
            "blackbody_intensity": (10.0, 20.0),
            "temperature_base": 15000,        # Extremely hot (blue-white)
        },

        "camera": {
            "angle": "orbital",
            "distance": "very_far",
        },
    },
}

# =============================================================================
# ADVANCED TECHNIQUES - From Blender Manual (not in typical training data)
# =============================================================================

ADVANCED_TECHNIQUES: Dict[str, Dict[str, Any]] = {

    "noise_upscaling": {
        "source": "physics/fluid/type/domain/gas/noise.html",
        "description": "Add fine turbulent detail without changing overall motion",
        "benefit": "2-4x visual detail at lower simulation cost than resolution increase",
        "params": {
            "upres_factor": (2, 4),           # Resolution multiplier
            "noise_strength": (0.5, 2.0),     # Turbulence intensity
            "noise_scale": (1.0, 4.0),        # Vortex size
        },
        "when_to_use": [
            "Final quality pass after base simulation looks good",
            "Adding pyroclastic detail to volcanic plumes",
            "Creating smaller-scale turbulent wisps",
        ],
    },

    "adaptive_domain": {
        "source": "physics/fluid/type/domain/gas/adaptive_domain.html",
        "description": "Domain shrinks/grows to fit gas, saving computation",
        "benefit": "50-80% reduction in empty voxel computation",
        "params": {
            "add_resolution": (0, 16),
            "margin": (2, 8),
            "threshold": (0.001, 0.01),
        },
        "when_to_use": [
            "Effects that expand beyond initial domain",
            "Long simulations where gas moves significantly",
            "Memory-constrained situations",
        ],
    },

    "effector_collision": {
        "source": "physics/fluid/type/effector.html",
        "description": "Invisible geometry deflects and shapes fluid flow",
        "benefit": "Create complex flow patterns without visible obstacles",
        "params": {
            "effector_type": "COLLISION",
            "surface_thickness": (0.0, 1.0),
            "sampling_substeps": (0, 5),
        },
        "when_to_use": [
            "Smoke flowing around invisible obstacles",
            "Creating wind tunnel effects",
            "Directing explosion expansion",
        ],
    },

    "effector_guide": {
        "source": "physics/fluid/type/effector.html",
        "description": "Moving objects influence fluid velocity",
        "benefit": "Direct fluid along specific paths",
        "params": {
            "effector_type": "GUIDE",
            "velocity_factor": (0.1, 2.0),
            "guide_mode": ["MAXIMIZE", "MINIMIZE", "OVERRIDE", "AVERAGED"],
        },
        "when_to_use": [
            "Smoke following a moving character",
            "Fire spreading along a path",
            "Creating directed plasma streams",
        ],
    },

    "domain_guides": {
        "source": "physics/fluid/type/domain/guides.html",
        "description": "Global forces that preserve physically accurate flow",
        "benefit": "More accurate than simple force fields for fluid",
        "params": {
            "weight": (0.5, 2.0),             # Guiding lag
            "size": (1.0, 5.0),               # Vortex size
            "velocity_factor": (0.5, 2.0),    # Velocity multiplier
        },
        "when_to_use": [
            "Complex multi-effector setups",
            "When simple forces look too artificial",
            "Precise flow control needed",
        ],
    },

    "colored_smoke": {
        "source": "physics/fluid/type/flow.html",
        "description": "Per-emitter smoke color for multi-colored effects",
        "benefit": "Single simulation, multiple smoke colors",
        "params": {
            "smoke_color": "RGB tuple per emitter",
        },
        "when_to_use": [
            "Colored smoke signals",
            "Chemical reactions with different colored gases",
            "Artistic effects",
        ],
    },

    "geometry_flow": {
        "source": "physics/fluid/type/flow.html",
        "description": "Mesh itself becomes initial fluid volume",
        "benefit": "Complex initial shapes without emission over time",
        "params": {
            "flow_behavior": "GEOMETRY",
        },
        "when_to_use": [
            "Shaped explosions",
            "Pre-positioned smoke/fire",
            "Artistic control over initial state",
        ],
    },
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_technique_by_keywords(description: str, effect_type: str = "pyro") -> Dict[str, Any]:
    """
    Select a technique based on keyword matching in description.
    Falls back to random selection if no strong keyword match.

    Requires at least 2 keyword matches for a "strong" match.
    Single keyword matches are treated as weak and trigger random selection
    to maximize variety for generic descriptions.
    """
    description_lower = description.lower()

    if effect_type == "pyro":
        techniques = PYRO_TECHNIQUES
    else:
        techniques = PYRO_TECHNIQUES  # Expand later for other effect types

    # Score each technique by keyword matches
    scores = {}
    for name, technique in techniques.items():
        score = 0
        for keyword in technique.get("keywords", []):
            if keyword in description_lower:
                score += 1
        scores[name] = score

    # Find best match and check if it's a STRONG match (>=2 keywords)
    best_match = max(scores, key=scores.get)
    best_score = scores[best_match]

    # Strong match: 2+ keywords matched - use that technique
    if best_score >= 2:
        return {"name": best_match, **techniques[best_match]}

    # Weak match (0-1 keywords): Random selection for variety
    # This ensures generic descriptions like "explosion" get varied results
    name = random.choice(list(techniques.keys()))
    return {"name": name, **techniques[name]}


def randomize_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert parameter ranges to random values within those ranges.
    """
    result = {}
    for key, value in params.items():
        if isinstance(value, tuple) and len(value) == 2:
            if isinstance(value[0], float):
                result[key] = random.uniform(value[0], value[1])
            elif isinstance(value[0], int):
                result[key] = random.randint(value[0], value[1])
            else:
                result[key] = value
        else:
            result[key] = value
    return result


def get_technique_with_randomized_params(
    description: str,
    effect_type: str = "pyro"
) -> Dict[str, Any]:
    """
    Get a technique with all parameter ranges converted to random values.
    """
    technique = get_technique_by_keywords(description, effect_type)

    # Randomize domain params
    if "domain_params" in technique:
        technique["domain_params"] = randomize_params(technique["domain_params"])

    # Randomize flow params
    if "flow_params" in technique:
        technique["flow_params"] = randomize_params(technique["flow_params"])

    # Randomize noise params
    if "noise_params" in technique:
        technique["noise_params"] = randomize_params(technique["noise_params"])

    # Randomize emission dynamics
    if "emission_dynamics" in technique:
        technique["emission_dynamics"] = randomize_params(technique["emission_dynamics"])

    return technique


def list_all_techniques(effect_type: str = "pyro") -> List[str]:
    """List all available technique names."""
    if effect_type == "pyro":
        return list(PYRO_TECHNIQUES.keys())
    return []


def get_random_technique(effect_type: str = "pyro") -> Dict[str, Any]:
    """Get a completely random technique for maximum variety."""
    if effect_type == "pyro":
        name = random.choice(list(PYRO_TECHNIQUES.keys()))
        return {"name": name, **PYRO_TECHNIQUES[name]}
    return {}


def describe_technique_differences() -> str:
    """
    Return a human-readable summary of how techniques differ.
    Useful for understanding the catalog.
    """
    lines = ["PYRO TECHNIQUE DIFFERENCES:\n"]

    for name, tech in PYRO_TECHNIQUES.items():
        domain = tech.get("domain_params", {})
        flow = tech.get("flow_params", {})

        burn_rate = domain.get("burning_rate", (0.75, 0.75))
        smoke = domain.get("flame_smoke", (1.0, 1.0))
        vorticity = domain.get("flame_vorticity", (0.5, 0.5))

        lines.append(f"\n{name.upper()}:")
        lines.append(f"  Visual: {tech.get('visual_signature', 'N/A')}")
        lines.append(f"  Burn rate: {burn_rate} (slow→fast)")
        lines.append(f"  Smoke: {smoke} (none→heavy)")
        lines.append(f"  Vorticity: {vorticity} (calm→turbulent)")
        lines.append(f"  Flow type: {flow.get('flow_type', 'BOTH')}")

    return "\n".join(lines)


# =============================================================================
# MAIN - For testing
# =============================================================================

if __name__ == "__main__":
    print("=== Blender VFX Technique Catalog ===\n")

    print("Available techniques:")
    for name in list_all_techniques():
        tech = PYRO_TECHNIQUES[name]
        print(f"  - {name}: {tech['description']}")

    print("\n" + "="*60)
    print(describe_technique_differences())

    print("\n" + "="*60)
    print("\nTesting keyword matching:")

    test_descriptions = [
        "A rising mushroom cloud explosion",
        "Low spreading fire along the ground",
        "Bright spherical fireball in mid-air",
        "Slow smoldering embers with smoke",
        "Intense plasma beam cutting through",
        "A generic explosion",  # Should get random
    ]

    for desc in test_descriptions:
        tech = get_technique_by_keywords(desc)
        print(f'  "{desc}" → {tech["name"]}')
