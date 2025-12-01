# Particle Size Awareness - PINN Training Enhancement

**Date:** 2025-12-01
**Priority:** HIGH (Phase 5 dependency)
**Estimated Time:** 1-2 hours (training + integration)

---

## Problem Statement

The current PINN models (v1-v4) treat particles as **point masses**, but PlasmaDX uses **volumetric 3D Gaussian particles** with:
- **Radius:** 18-24 units (typically 20)
- **Effective diameter:** 36-48 units (typically 40)
- **Volumetric interactions:** Particles overlap, scatter light, create pressure

This mismatch causes:
1. **Incorrect pressure/density forces** - PINN doesn't account for particle overlap
2. **Missing collision dynamics** - No repulsion when particles get too close
3. **Unrealistic clustering** - Particles can occupy same space without interaction
4. **GA optimization issues** - Optimizes for point particles, not volumetric Gaussians

---

## Current PINN Architecture

### v3/v4 Model Input (10D Cartesian)
```python
input = [
    x, y, z,              # Position (3D)
    vx, vy, vz,           # Velocity (3D)
    GM,                   # Gravitational parameter
    alpha,                # Viscosity
    disk_thickness,       # H/R ratio
    time                  # Simulation time
]
# Total: 10 dimensions
```

### What's Missing
```python
particle_radius  # NOT included!
```

The PINN has **no way to know**:
- How big particles are
- When they're overlapping
- What pressure forces to apply
- How density affects local dynamics

---

## Solution: Extend to 11D Input

### New Input Schema
```python
input = [
    x, y, z,              # Position (3D)
    vx, vy, vz,           # Velocity (3D)
    GM,                   # Gravitational parameter
    alpha,                # Viscosity
    disk_thickness,       # H/R ratio
    time,                 # Simulation time
    particle_radius       # NEW! (typically 20.0)
]
# Total: 11 dimensions
```

### Benefits
1. **Pressure forces** - PINN learns to repel overlapping particles
2. **Density-aware dynamics** - Forces scale with local particle density
3. **Collision avoidance** - Natural repulsion at close range
4. **Better GA optimization** - Optimizes for actual volumetric particles

---

## Implementation Plan

### Step 1: Update Training Data Generation (30 min)

**File:** `ml/pinn_accretion_disk.py` or new `ml/pinn_v5_volumetric.py`

```python
# === CHANGES TO TRAINING SCRIPT ===

# 1. Extend input dimension
INPUT_DIM = 11  # Was 10

# 2. Add particle radius to training data
def generate_training_data(num_samples=10000):
    # ... existing position/velocity generation ...
    
    # NEW: Add particle radius column
    particle_radii = np.random.uniform(15.0, 25.0, size=(num_samples, 1))  # Vary 15-25
    
    # Compute forces with volumetric effects
    forces = compute_forces_with_volume(
        positions, velocities, 
        GM, alpha, disk_thickness, 
        particle_radii  # NEW!
    )
    
    # Stack inputs
    inputs = np.hstack([
        positions,      # (N, 3)
        velocities,     # (N, 3)
        GM_col,         # (N, 1)
        alpha_col,      # (N, 1)
        thickness_col,  # (N, 1)
        time_col,       # (N, 1)
        particle_radii  # (N, 1) NEW!
    ])  # Total: (N, 11)
    
    return inputs, forces

# 3. Add volumetric force computation
def compute_forces_with_volume(pos, vel, GM, alpha, H_R, radii):
    """
    Compute forces accounting for particle size
    
    New physics:
    - Pressure force when particles overlap
    - Density-dependent viscosity
    - Collision avoidance
    """
    forces = np.zeros_like(pos)
    
    # Gravity (unchanged)
    r = np.linalg.norm(pos, axis=1, keepdims=True)
    F_gravity = -GM * pos / (r**3 + 1e-6)
    
    # Viscosity (density-weighted)
    local_density = estimate_local_density(pos, radii)  # NEW!
    F_viscosity = -alpha * vel * local_density  # Scaled by density
    
    # Pressure (NEW!)
    F_pressure = compute_pressure_forces(pos, radii)
    
    forces = F_gravity + F_viscosity + F_pressure
    return forces

def estimate_local_density(positions, radii):
    """
    Estimate particle density at each position
    
    Uses KD-tree for efficiency
    """
    from scipy.spatial import cKDTree
    
    tree = cKDTree(positions)
    densities = np.zeros(len(positions))
    
    for i, (pos, radius) in enumerate(zip(positions, radii)):
        # Count particles within 3× radius
        neighbors = tree.query_ball_point(pos, r=3.0 * radius)
        densities[i] = len(neighbors) / (4/3 * np.pi * (3*radius)**3)
    
    return densities.reshape(-1, 1)

def compute_pressure_forces(positions, radii):
    """
    Compute repulsive forces from overlapping particles
    
    F_pressure = k * overlap * direction
    where overlap = (r1 + r2) - distance
    """
    from scipy.spatial import cKDTree
    
    forces = np.zeros_like(positions)
    tree = cKDTree(positions)
    
    PRESSURE_STRENGTH = 5.0  # Tunable
    
    for i, (pos, radius) in enumerate(zip(positions, radii)):
        # Find nearby particles (within 2× radius)
        neighbors = tree.query_ball_point(pos, r=2.5 * radius)
        
        for j in neighbors:
            if i == j:
                continue
            
            delta = positions[i] - positions[j]
            dist = np.linalg.norm(delta) + 1e-6
            overlap = (radii[i] + radii[j]) - dist
            
            if overlap > 0:
                # Repulsive force proportional to overlap
                direction = delta / dist
                force_mag = PRESSURE_STRENGTH * overlap
                forces[i] += direction * force_mag
    
    return forces

# 4. Update network architecture
class VolumetricPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(11, 256),  # Was 10, now 11
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 3)  # Output: [Fx, Fy, Fz]
        )
    
    def forward(self, x):
        return self.network(x)

# 5. Training loop (unchanged except input dim)
def train_model():
    model = VolumetricPINN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Generate training data with particle radii
    X_train, y_train = generate_training_data(num_samples=50000)
    
    # ... standard training loop ...
    
    # Export to ONNX
    dummy_input = torch.randn(1, 11)  # Was 10, now 11
    torch.onnx.export(
        model, dummy_input,
        "ml/models/pinn_v5_volumetric.onnx",
        input_names=["particle_state"],
        output_names=["forces"],
        dynamic_axes={"particle_state": {0: "batch_size"}}
    )
```

---

### Step 2: Update C++ Inference (15 min)

**File:** `src/ml/PINNPhysicsSystem.cpp`

```cpp
// In PredictForcesBatch() method

// Prepare input tensor (11D instead of 10D)
std::vector<float> inputData(particleCount * 11);  // Was 10

for (uint32_t i = 0; i < particleCount; i++) {
    size_t offset = i * 11;  // Was 10
    
    // Position (Cartesian)
    inputData[offset + 0] = positions[i].x;
    inputData[offset + 1] = positions[i].y;
    inputData[offset + 2] = positions[i].z;
    
    // Velocity (Cartesian)
    inputData[offset + 3] = velocities[i].x;
    inputData[offset + 4] = velocities[i].y;
    inputData[offset + 5] = velocities[i].z;
    
    // Physics parameters
    inputData[offset + 6] = m_physicsParams.gm;
    inputData[offset + 7] = m_physicsParams.alphaViscosity;
    inputData[offset + 8] = m_physicsParams.diskThickness;
    inputData[offset + 9] = currentTime;
    
    // NEW: Particle radius
    inputData[offset + 10] = PARTICLE_RADIUS;  // e.g., 20.0f
}

// Update input shape
std::vector<int64_t> inputShape = {static_cast<int64_t>(particleCount), 11};  // Was 10
```

**File:** `src/ml/PINNPhysicsSystem.h`

```cpp
// Add particle radius parameter
struct PhysicsParams {
    float blackHoleMassNormalized = 1.0f;
    float alphaViscosity = 0.1f;
    float diskThickness = 0.1f;
    float particleRadius = 20.0f;  // NEW!
};

// Add accessor methods
void SetParticleRadius(float radius);
float GetParticleRadius() const { return m_physicsParams.particleRadius; }
```

---

### Step 3: Model Detection Logic (10 min)

**File:** `src/ml/PINNPhysicsSystem.cpp`

```cpp
bool PINNPhysicsSystem::Initialize(const std::string& modelPath) {
    // ... existing initialization ...
    
    // Detect model version by input shape
    if (m_inputShape.size() >= 2) {
        int64_t inputDim = m_inputShape[1];
        
        if (inputDim == 7) {
            m_isV1Model = true;  // Spherical coordinates
            LOG_INFO("[PINN] Detected v1 model (7D spherical)");
        }
        else if (inputDim == 10) {
            m_isV3Model = true;  // Cartesian, no particle size
            LOG_INFO("[PINN] Detected v3/v4 model (10D Cartesian)");
        }
        else if (inputDim == 11) {
            m_isV5Model = true;  // NEW! Cartesian with particle size
            LOG_INFO("[PINN] Detected v5 model (11D Cartesian + particle size)");
        }
    }
    
    return true;
}
```

---

### Step 4: Training Execution (20-30 min)

```bash
cd /mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean

# Create new training script (or modify existing)
python ml/pinn_v5_volumetric.py

# Expected output:
# - Training data: 50,000 samples with particle radii
# - Model: 11D input → 3D output
# - Training time: ~20 minutes (GPU) or ~60 minutes (CPU)
# - Output: ml/models/pinn_v5_volumetric.onnx
```

---

### Step 5: Testing & Validation (15 min)

```bash
# Rebuild C++ project
MSBuild.exe build/PlasmaDX-Clean.sln /p:Configuration=Debug /p:Platform=x64

# Test v5 model
./build/bin/Debug/PlasmaDX-Clean.exe --pinn v5 --particles 1000

# Run benchmark
./build/bin/Debug/PlasmaDX-Clean.exe \
    --benchmark \
    --pinn ml/models/pinn_v5_volumetric.onnx \
    --particles 5000 \
    --frames 500

# Compare to v4
./build/bin/Debug/PlasmaDX-Clean.exe \
    --benchmark \
    --pinn v4 \
    --particles 5000 \
    --frames 500
```

**Expected Improvements:**
- Less particle clustering (pressure forces prevent overlap)
- More realistic density distribution
- Better visual coherence (particles maintain spacing)
- Improved GA optimization (optimizes for actual particle size)

---

## Alternative: Post-PINN Collision Layer (Quick Fix)

If retraining is deferred, add collision forces in C++:

**File:** `src/particles/ParticleSystem.cpp`

```cpp
void ParticleSystem::UpdatePhysics_PINN(float deltaTime, float totalTime) {
    // Step 1: PINN inference (existing)
    m_pinnPhysics->PredictForcesBatch(
        m_cpuPositions.data(), m_cpuVelocities.data(),
        m_cpuForces.data(), m_activeParticleCount, totalTime
    );
    
    // Step 2: NEW - Add collision/pressure forces
    if (m_enableVolumetricCollisions) {
        AddVolumetricPressureForces(m_cpuPositions, m_cpuForces, PARTICLE_RADIUS);
    }
    
    // Step 3: SIREN turbulence (existing)
    if (m_sirenVortexField && m_sirenVortexField->IsEnabled()) {
        // ... existing SIREN code ...
    }
    
    // Step 4: Integration (existing)
    IntegrateForces(m_cpuForces, deltaTime);
}

void ParticleSystem::AddVolumetricPressureForces(
    const std::vector<XMFLOAT3>& positions,
    std::vector<XMFLOAT3>& forces,
    float particleRadius)
{
    const float PRESSURE_STRENGTH = 5.0f;
    const float INTERACTION_RADIUS = 2.5f * particleRadius;
    
    // Spatial hashing for O(n) instead of O(n²)
    SpatialHash hash(INTERACTION_RADIUS);
    hash.Build(positions);
    
    for (uint32_t i = 0; i < positions.size(); i++) {
        auto neighbors = hash.QueryNeighbors(positions[i]);
        
        for (uint32_t j : neighbors) {
            if (i >= j) continue;  // Avoid double-counting
            
            XMVECTOR delta = XMLoadFloat3(&positions[i]) - XMLoadFloat3(&positions[j]);
            float dist = XMVectorGetX(XMVector3Length(delta));
            float overlap = 2.0f * particleRadius - dist;
            
            if (overlap > 0) {
                XMVECTOR direction = XMVector3Normalize(delta);
                XMVECTOR repulsion = direction * (PRESSURE_STRENGTH * overlap);
                
                XMFLOAT3 repulsionF;
                XMStoreFloat3(&repulsionF, repulsion);
                
                forces[i].x += repulsionF.x;
                forces[i].y += repulsionF.y;
                forces[i].z += repulsionF.z;
                
                forces[j].x -= repulsionF.x;
                forces[j].y -= repulsionF.y;
                forces[j].z -= repulsionF.z;
            }
        }
    }
}
```

**Pros:** No retraining, works immediately
**Cons:** O(n²) without spatial hashing, not learned by PINN

---

## Recommendation

### For Phase 5 (Immediate)
1. **Start with world scale fix** (already done ✅)
2. **Test current setup** with corrected boundaries
3. **If particles cluster badly**, implement post-PINN collision layer

### For Phase 6 (Future)
1. **Train v5 model** with particle size awareness
2. **Compare v4 vs v5** in benchmark
3. **Re-run GA** with v5 model for optimal parameters

---

## Expected Performance Impact

| Metric | v4 (Point Mass) | v5 (Volumetric) | Change |
|--------|-----------------|-----------------|--------|
| Training time | 20 min | 25 min | +25% (density calc) |
| Inference time | 2.5 ms | 2.8 ms | +12% (11D vs 10D) |
| Visual quality | Good | **Excellent** | Better spacing |
| Physical accuracy | 85% | **95%** | Pressure forces |
| GA optimization | Suboptimal | **Optimal** | True particle size |

---

## Files to Create/Modify

### New Files
- `ml/pinn_v5_volumetric.py` - Training script with 11D input
- `ml/models/pinn_v5_volumetric.onnx` - Trained model

### Modified Files
- `src/ml/PINNPhysicsSystem.h` - Add v5 detection, particle radius param
- `src/ml/PINNPhysicsSystem.cpp` - 11D input tensor, v5 inference
- `src/particles/ParticleSystem.cpp` - Optional collision layer

---

## Success Criteria

✅ **Model trains successfully** (loss < 0.01)
✅ **Inference works** (no crashes, reasonable forces)
✅ **Particles maintain spacing** (no extreme clustering)
✅ **Visual quality improves** (coherent disk structure)
✅ **GA optimization improves** (fitness matches validation)

---

## Next Steps

1. **User decides:** Train v5 now or defer to Phase 6?
2. **If now:** Create `ml/pinn_v5_volumetric.py` based on template above
3. **If defer:** Implement post-PINN collision layer as stopgap
4. **Either way:** Continue Phase 5 with corrected world scale

---

**Status:** Awaiting user decision on v5 training timeline
**Estimated Total Time:** 1-2 hours (if training now)
**Complexity:** Medium (well-defined problem, clear solution)
