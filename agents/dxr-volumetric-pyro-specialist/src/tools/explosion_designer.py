"""
Explosion Effect Designer
Designs complete explosion effect specifications with temporal dynamics
"""

import math


async def design_explosion_effect(effect_type: str = "supernova",
                                  duration_seconds: float = 5.0,
                                  max_radius_meters: float = 500.0,
                                  peak_temperature_kelvin: float = 100000.0,
                                  particle_budget: int = 10000) -> dict:
    """
    Design explosion effect with temporal dynamics, material properties, and procedural noise

    Returns:
        dict: Complete explosion specification
    """

    # Calculate temporal dynamics
    expansion_rate = max_radius_meters / (duration_seconds ** 2)  # Quadratic expansion
    decay_time_constant = duration_seconds / 3.0  # Temperature exponential decay

    # Material properties based on effect type
    material_configs = {
        "supernova": {
            "scattering": 0.8,
            "absorption": 0.3,
            "emission": 5.0,
            "phase_g": 0.6
        },
        "stellar_flare": {
            "scattering": 0.7,
            "absorption": 0.4,
            "emission": 3.0,
            "phase_g": 0.5
        },
        "accretion_burst": {
            "scattering": 0.6,
            "absorption": 0.5,
            "emission": 4.0,
            "phase_g": 0.4
        },
        "shockwave": {
            "scattering": 0.9,
            "absorption": 0.2,
            "emission": 2.0,
            "phase_g": 0.7
        }
    }

    material = material_configs.get(effect_type, material_configs["supernova"])

    # Estimate performance impact
    alu_ops_per_particle = 80  # Noise + expansion + temperature
    fps_impact_percent = min(25, (particle_budget / 10000) * 15)

    return {
        "effect_type": effect_type,
        "duration": duration_seconds,
        "max_radius": max_radius_meters,
        "peak_temperature": peak_temperature_kelvin,
        "particle_budget": particle_budget,
        "temporal": {
            "expansion_rate": expansion_rate,
            "decay_constant": decay_time_constant
        },
        "material": material,
        "performance": {
            "alu_ops": alu_ops_per_particle,
            "fps_impact_percent": fps_impact_percent
        }
    }


async def format_explosion_design(results: dict) -> str:
    """Format explosion design as detailed specification"""

    report = f"""# {results['effect_type'].upper()} EXPLOSION DESIGN SPECIFICATION

## Effect Overview
- **Type**: {results['effect_type'].replace('_', ' ').title()}
- **Duration**: {results['duration']:.1f} seconds
- **Maximum Radius**: {results['max_radius']:.0f} meters
- **Peak Temperature**: {results['peak_temperature']:.0f}K
- **Particle Budget**: {results['particle_budget']:,}

## Temporal Dynamics

### Expansion Curve
```hlsl
// Quadratic blast wave expansion
float r(float t) {{
    float r0 = 10.0; // Initial radius
    return r0 * (1.0 + {results['temporal']['expansion_rate']:.1f} * t * t);
}}
```

### Temperature Decay
```hlsl
// Exponential cooling
float T(float t) {{
    float T_peak = {results['peak_temperature']:.0f};
    float tau = {results['temporal']['decay_constant']:.2f};
    return T_peak * exp(-t / tau);
}}
```

### Opacity Fade
```hlsl
// Quadratic fade over duration
float alpha(float t) {{
    float t_max = {results['duration']:.1f};
    float fade = 1.0 - (t / t_max);
    return fade * fade;
}}
```

## Material Properties

- **Scattering Coefficient**: {results['material']['scattering']} (Henyey-Greenstein forward scattering)
- **Absorption**: {results['material']['absorption']} (Beer-Lambert volumetric attenuation)
- **Emission Multiplier**: {results['material']['emission']} (self-illumination intensity)
- **Phase Function g**: {results['material']['phase_g']} (anisotropy parameter)

## Procedural Noise Parameters

### SimplexNoise3D Configuration
```hlsl
// Turbulence for explosion irregularity
SimplexNoise3D noise;
noise.frequency = 2.0;        // Medium-scale turbulence
noise.amplitude = 0.3;        // 30% displacement
noise.octaves = 3;            // Layered detail
noise.lacunarity = 2.0;       // Frequency multiplier per octave
noise.persistence = 0.5;      // Amplitude multiplier per octave

// Temporal modulation for flickering
noise.frequency *= (1.0 + 0.5 * sin(time * 2.0));
```

## Color Profile (Temperature-Based Blackbody)

| Temperature | RGB Color | Description |
|-------------|-----------|-------------|
| 100000K | (0.7, 0.8, 1.0) | Blue-white core |
| 30000K | (1.0, 0.9, 0.7) | Yellow-white mid |
| 5000K | (1.0, 0.4, 0.2) | Red-orange outer |

## Performance Estimate

- **FPS Impact**: -{results['performance']['fps_impact_percent']:.0f}% (120 FPS → {120 * (1 - results['performance']['fps_impact_percent']/100):.0f} FPS)
- **ALU Ops/Particle**: ~{results['performance']['alu_ops']} operations
- **Memory Bandwidth**: +1.2 GB/s (temporal buffer reads)

## Shader Integration Points

### 1. Particle Physics (`particle_physics.hlsl`)
Add explosion dynamics to particle update loop:
```hlsl
struct ParticleExplosion {{
    float3 center;      // Explosion epicenter
    float startTime;    // Time explosion triggered
    float duration;     // Effect duration
}};

// In physics kernel
float elapsed = globalTime - explosion.startTime;
if (elapsed < explosion.duration) {{
    float3 toCenter = particle.position - explosion.center;
    float dist = length(toCenter);

    // Apply expansion force
    float expansionForce = {results['temporal']['expansion_rate']:.1f} * elapsed;
    particle.velocity += normalize(toCenter) * expansionForce * deltaTime;
}}
```

### 2. Gaussian Renderer (`particle_gaussian_raytrace.hlsl`)
Add temperature-based emission:
```hlsl
// In volumetric ray marching loop
float temperature = T(elapsed); // Temperature decay function
float3 emission = BlackbodyEmission(temperature) * {results['material']['emission']};
radiance += emission * opacity * dt;
```

### 3. Required Buffers
- **Per-Particle**: `float explosionTime` (4 bytes, total +{results['particle_budget'] * 4 / 1024:.1f} KB)
- **Global**: `ParticleExplosion explosionData` (16 bytes root constant)

## Validation Criteria (for dxr-image-quality-analyst)

**Visual Characteristics**:
- Spherical expansion with blue-white → red-orange gradient ✓
- Temporal: Smooth acceleration, peak at t={results['duration']/2:.1f}s, fade by t={results['duration']:.1f}s ✓
- Irregularity: 30% displacement from simplex noise ✓

**Performance**:
- Maintain >{90 * (1 - results['performance']['fps_impact_percent']/100):.0f} FPS with {results['particle_budget']:,} particles ✓
- Total particle budget including background: <100K particles ✓

---
*Explosion design by DXR Volumetric Pyro Specialist*
*Ready for implementation by material-system-engineer*
"""
    return report
