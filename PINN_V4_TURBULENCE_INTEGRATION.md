# PINN v4 + SIREN Turbulence Integration Guide

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     Two-Model Turbulence System                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│   CPU Side (ONNX Runtime)              GPU Side (Compute Shader)         │
│   ~~~~~~~~~~~~~~~~~~~~~~~~              ~~~~~~~~~~~~~~~~~~~~~~~~         │
│                                                                           │
│   ┌──────────────────────┐             ┌──────────────────────┐         │
│   │   PINN v4 Model      │             │  SIREN Vortex Field  │         │
│   │   (67K params)       │             │  (4K params)         │         │
│   │                      │             │                      │         │
│   │   Input: pos, vel,   │             │  Input: pos, time,   │         │
│   │          time, params│             │          seed        │         │
│   │                      │             │                      │         │
│   │   Output: F_orbital  │             │  Output: ω (vorticity)│        │
│   │   (gravity+viscosity)│             │                      │         │
│   └──────────┬───────────┘             └──────────┬───────────┘         │
│              │                                    │                      │
│              │ ~8ms @ 10K particles               │ ~0.01ms @ 10K       │
│              │                                    │                      │
│              └────────────────┬───────────────────┘                      │
│                               │                                          │
│                               ▼                                          │
│                    ┌──────────────────────┐                              │
│                    │  Force Combination   │                              │
│                    │                      │                              │
│                    │  F_total = F_orbital │                              │
│                    │          + turbulence│                              │
│                    │            × intensity│                             │
│                    └──────────┬───────────┘                              │
│                               │                                          │
│                               ▼                                          │
│                    ┌──────────────────────┐                              │
│                    │  Velocity Verlet     │                              │
│                    │  Integration         │                              │
│                    └──────────────────────┘                              │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────┘
```

## Why Two Models?

### PINN v4 (Orbital Physics)
- **Trained on diverse velocities** (50%-150% of Keplerian)
- Handles particles that have been kicked by turbulence
- Outputs physically correct gravitational + viscous forces
- Runs on CPU via ONNX Runtime (~8ms for 10K particles)

### SIREN Vortex (Turbulence)
- **Specialized for oscillatory patterns** (sin activations)
- Outputs smooth, coherent vortex structures
- Can be baked into HLSL shader for zero GPU cost
- Independent of orbital mechanics

## Training Steps

### Step 1: Train PINN v4 (Turbulence-Robust)

```bash
cd /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean

# Generate training data with diverse velocities
ml/venv/bin/python3 ml/pinn_v4_with_turbulence.py --generate-data

# Train model (300 epochs, ~15 min on GPU)
ml/venv/bin/python3 ml/pinn_v4_with_turbulence.py --epochs 300

# Copy to runtime location
cp ml/models/pinn_v4_turbulence_robust.onnx build/bin/Debug/ml/models/
```

### Step 2: Train SIREN Vortex Field

```bash
cd ml/vortex_field

# Generate vortex training data
python generate_vortex_training_data.py --n_samples 100000 --output vortex_data.npz

# Train SIREN and export to HLSL
python train_vortex_siren.py --data vortex_data.npz --epochs 500 --export_hlsl

# Result: ml/models/vortex_siren.hlsl (embedded weights)
```

## C++ Integration

### Option A: SIREN on CPU (Simpler)

Add SIREN as second ONNX model, combine forces in `IntegrateForces`:

```cpp
// In ParticleSystem.cpp

void ParticleSystem::IntegrateForces(const std::vector<XMFLOAT3>& forces, float deltaTime)
{
    // 1. Get orbital forces from PINN v4 (already in 'forces' parameter)
    
    // 2. Get turbulence from SIREN (if enabled)
    std::vector<XMFLOAT3> turbForces;
    if (m_turbulenceIntensity > 0 && m_sirenVortex) {
        m_sirenVortex->PredictVorticity(
            m_cpuPositions.data(),
            m_activeParticleCount,
            m_totalTime,
            turbForces
        );
    }
    
    // 3. Combine and integrate
    for (uint32_t i = 0; i < m_activeParticleCount; i++) {
        float fx = forces[i].x;
        float fy = forces[i].y;
        float fz = forces[i].z;
        
        // Add turbulence (as force, not velocity!)
        if (m_turbulenceIntensity > 0) {
            fx += turbForces[i].x * m_turbulenceIntensity;
            fy += turbForces[i].y * m_turbulenceIntensity;
            fz += turbForces[i].z * m_turbulenceIntensity;
        }
        
        // Velocity Verlet integration
        m_cpuVelocities[i].x += fx * deltaTime;
        m_cpuVelocities[i].y += fy * deltaTime;
        m_cpuVelocities[i].z += fz * deltaTime;
        
        m_cpuPositions[i].x += m_cpuVelocities[i].x * deltaTime;
        m_cpuPositions[i].y += m_cpuVelocities[i].y * deltaTime;
        m_cpuPositions[i].z += m_cpuVelocities[i].z * deltaTime;
    }
}
```

### Option B: SIREN on GPU (Zero CPU Cost)

Embed SIREN weights in HLSL shader, add to particle_physics.hlsl:

```hlsl
// In shaders/particles/particle_physics.hlsl

#include "noise/learned_vortex.hlsl"  // Auto-generated from train_vortex_siren.py

float3 GetTurbulentForce(float3 position, float time, float intensity)
{
    // Evaluate learned vortex field (tiny ~4K param network)
    float3 vorticity = EvaluateVortexSIREN(
        position * 0.01,     // Scale position to [-1,1] range
        frac(time * 0.1),    // Cyclic time
        0.0                  // Seed (can vary per frame for chaos)
    );
    
    // Convert vorticity to force-like perturbation
    // curl(ω) gives velocity-like quantity
    return vorticity * intensity * 10.0;
}
```

## Boundary Volume Fix

For large particles (radius 16-30), expand the simulation space:

```cpp
// In ParticleSystem.h

// Increase outer radius for larger particles
static constexpr float OUTER_DISK_RADIUS = 500.0f;  // Was 300

// Adjust initialization to use more of the space
void ParticleSystem::InitializeAccretionDisk_CPU() {
    // Use 30-400 radius range instead of 10-200
    std::uniform_real_distribution<float> radiusDist(30.0f, OUTER_DISK_RADIUS * 0.8f);
    // ...
}
```

Also update the boundary enforcement in `IntegrateForces`:

```cpp
const float innerRadius = 20.0f;    // Larger inner boundary for big particles
const float outerRadius = 450.0f;   // Larger outer boundary
```

## Visual Speed Improvement

Particles far from center orbit slowly (physically correct). Options:

### Option 1: Spawn particles closer to center
```cpp
// More particles near inner disk where orbits are fast
std::uniform_real_distribution<float> radiusDist(15.0f, 100.0f);  // Not 300
```

### Option 2: Non-uniform radius distribution (dense inner disk)
```cpp
// Exponential distribution favoring inner radii
float u = std::uniform_real_distribution<float>(0.0f, 1.0f)(rng);
float radius = R_INNER + (R_OUTER - R_INNER) * (1.0f - std::pow(1.0f - u, 2.0f));
```

### Option 3: Higher GM (requires retraining)
```python
# In training script, use GM = 500 instead of 100
# Orbits 5× faster, but forces also 5× stronger
GM = 500.0
```

## Performance Summary

| Component | Time @ 10K particles | Notes |
|-----------|---------------------|-------|
| PINN v4 (CPU) | ~8ms | Same as v3 |
| SIREN (GPU) | ~0.01ms | Embedded in shader |
| Integration | ~0.5ms | Parallel CPU |
| **Total** | **~8.5ms** | 117 FPS physics |

## Quick Start

```bash
# 1. Train PINN v4
ml/venv/bin/python3 ml/pinn_v4_with_turbulence.py --generate-data
ml/venv/bin/python3 ml/pinn_v4_with_turbulence.py --epochs 300

# 2. Train SIREN vortex
cd ml/vortex_field
python generate_vortex_training_data.py
python train_vortex_siren.py --export_hlsl

# 3. Deploy
cp ml/models/pinn_v4_turbulence_robust.onnx* build/bin/Debug/ml/models/

# 4. Rebuild C++
# Update ParticleSystem.cpp to load pinn_v4_turbulence_robust.onnx

# 5. Run
./build/bin/Debug/PlasmaDX-Clean.exe
```

## Summary

- **Keep PINN for orbital physics** - It works correctly for gravity/viscosity
- **Add SIREN for turbulence** - Separate specialized network, runs on GPU
- **Train v4 to handle turbulent velocities** - Won't explode when particles get kicked
- **Increase boundary volume** - For larger particle radii
- **Spawn particles closer to center** - For visible orbital motion

