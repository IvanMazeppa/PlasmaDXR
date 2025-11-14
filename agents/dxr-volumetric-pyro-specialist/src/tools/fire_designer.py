"""
Fire/Smoke Effect Designer
Designs fire and smoke effects with turbulence and procedural noise
"""

async def design_fire_effect(fire_type: str = "stellar_fire",
                             turbulence_intensity: float = 0.5,
                             temperature_range_kelvin: list = None,
                             opacity_profile: str = "dense_core",
                             particle_budget: int = 10000) -> dict:
    """Design fire/smoke effect with turbulence and scattering"""

    if temperature_range_kelvin is None:
        temperature_range_kelvin = [3000.0, 15000.0]

    # Noise parameters based on turbulence
    noise_frequency = 1.0 + (turbulence_intensity * 3.0)
    noise_amplitude = 0.2 + (turbulence_intensity * 0.6)

    # Opacity based on profile
    opacity_configs = {
        "dense_core": {"core": 0.9, "edge": 0.1},
        "wispy": {"core": 0.3, "edge": 0.05},
        "uniform": {"core": 0.5, "edge": 0.5},
        "layered": {"core": 0.7, "edge": 0.3}
    }

    opacity = opacity_configs.get(opacity_profile, opacity_configs["dense_core"])

    return {
        "fire_type": fire_type,
        "turbulence": turbulence_intensity,
        "temp_range": temperature_range_kelvin,
        "opacity_profile": opacity_profile,
        "particle_budget": particle_budget,
        "noise": {
            "frequency": noise_frequency,
            "amplitude": noise_amplitude,
            "octaves": max(2, int(turbulence_intensity * 4))
        },
        "opacity": opacity
    }


async def format_fire_design(results: dict) -> str:
    """Format fire design as specification"""

    report = f"""# {results['fire_type'].upper()} DESIGN SPECIFICATION

## Effect Overview
- **Type**: {results['fire_type'].replace('_', ' ').title()}
- **Turbulence Intensity**: {results['turbulence']:.2f} (0=calm, 1=violent)
- **Temperature Range**: {results['temp_range'][0]:.0f}K - {results['temp_range'][1]:.0f}K
- **Opacity Profile**: {results['opacity_profile'].replace('_', ' ').title()}
- **Particle Budget**: {results['particle_budget']:,}

## Procedural Noise (Turbulence)

### Perlin Noise Configuration
```hlsl
PerlinNoise3D turbulence;
turbulence.frequency = {results['noise']['frequency']:.2f};
turbulence.amplitude = {results['noise']['amplitude']:.2f};
turbulence.octaves = {results['noise']['octaves']};
turbulence.temporal_rate = 0.5; // Animation speed

// Apply to particle positions
particle.position += turbulence.sample(particle.position, time) * dt;
```

## Material Properties

### Scattering Profile
```hlsl
// Core: dense scattering
float scattering_core = 0.9;
// Edge: thin wispy
float scattering_edge = 0.3;

// Radial gradient
float distance_factor = length(particle.position - fire_center) / fire_radius;
float scattering = lerp(scattering_core, scattering_edge, distance_factor);
```

### Opacity Distribution
- **Core Opacity**: {results['opacity']['core']:.2f}
- **Edge Opacity**: {results['opacity']['edge']:.2f}

## Color Temperature Gradient

| Temperature | RGB Color | Region |
|-------------|-----------|--------|
| {results['temp_range'][1]:.0f}K | (1.0, 0.9, 0.7) | Hot core |
| {sum(results['temp_range'])/2:.0f}K | (1.0, 0.6, 0.3) | Mid region |
| {results['temp_range'][0]:.0f}K | (0.8, 0.3, 0.1) | Cool edge |

---
*Fire design by DXR Volumetric Pyro Specialist*
"""
    return report
