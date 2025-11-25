# Learned Vortex Field System

Neural network-based turbulence that produces realistic vortex structures.

## Architecture

**SIREN (Sinusoidal Representation Network)**
- Input: position (x,y,z), time (t), seed (s) → 5 floats
- Hidden: 2 × 48 neurons with sin() activation
- Output: vorticity vector (ω_x, ω_y, ω_z)
- Velocity recovered via: v = curl(ψ) where ∇²ψ = -ω

**Parameters:** ~4,000 (16KB model file)

## Training Pipeline

1. `generate_vortex_training_data.py` - Create analytical/simulated vortex fields
2. `train_vortex_siren.py` - Train SIREN network
3. `export_to_hlsl.py` - Convert to HLSL compute shader with embedded weights

## Integration Options

### Option 1: HLSL Compute Shader (Recommended)
Embed trained weights directly in shader. Evaluate per-particle or on grid.
- Overhead: <0.01ms @ 10K particles
- No CPU/GPU sync needed

### Option 2: ONNX Runtime (Simpler)
Run alongside PINN on CPU, add velocity perturbations.
- Overhead: ~0.1ms
- Easier to iterate/retrain

### Option 3: Baked 4D Texture
Precompute vortex field into RGBA volume texture, sample at runtime.
- Overhead: 0ms (just texture fetch)
- Memory: 64-256MB depending on resolution
- Less flexible (can't change parameters at runtime)

## Vortex Types Supported

1. **Coherent vortex tubes** - Large-scale swirling structures
2. **Vortex sheets** - Shearing layers that roll up
3. **Hairpin vortices** - 3D structures common in turbulent boundary layers
4. **Kolmogorov cascade** - Multi-scale energy transfer

## Quick Start

```bash
# Generate training data (analytical vortices)
python generate_vortex_training_data.py --n_samples 100000 --output vortex_data.npz

# Train SIREN network
python train_vortex_siren.py --data vortex_data.npz --epochs 500

# Export to HLSL
python export_to_hlsl.py --model models/vortex_siren.onnx --output ../shaders/noise/learned_vortex.hlsl
```

## HLSL Integration Example

```hlsl
// In particle_physics.hlsl or particle_gaussian_raytrace.hlsl

#include "noise/learned_vortex.hlsl"

float3 GetTurbulentVelocity(float3 position, float time, float intensity)
{
    // Evaluate learned vortex field
    float3 vorticity = EvaluateVortexSIREN(position * 0.01, time * 0.1);

    // Scale by intensity parameter
    return vorticity * intensity * 10.0;
}
```

## Performance Notes

- SIREN with 2×48 neurons = 48*5 + 48*48 + 48*3 = 2,736 MAD ops per evaluation
- At 10K particles: 27.36M MAD ops = ~0.003ms on RTX 4060 Ti
- Effectively free compared to ray tracing costs

## Future Enhancements

1. **Conditional SIREN** - Add disk radius as input for position-dependent turbulence
2. **Multi-scale** - Stack multiple SIRENs at different frequencies
3. **Temporal coherence** - Add recurrent connection for smooth evolution
