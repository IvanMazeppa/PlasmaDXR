# ML-Driven Physics Optimization - Implementation Roadmap

## Complete Step-by-Step Guide

**Project:** PlasmaDX-Clean - Ultimate Galaxy Physics Model Creator
**Date:** 2025-11-28 (Updated: 2025-11-30)
**Estimated Total Time:** 40-60 hours
**Current Progress:** 45% Complete (3 of 6 phases)

---

## 🎯 STATUS UPDATE (2025-11-30)

**✅ Phase 1: Runtime Controls** - COMPLETE (November 2025)
**⏭️ Phase 2: Enhanced Metrics** - SKIPPED (Metrics already sufficient)
**✅ Phase 3: Genetic Algorithm** - COMPLETE (2025-11-30)
**⏳ Phase 4: Active Learning** - NOT STARTED
**⏳ Phase 5: Constrained Turbulence** - NOT STARTED (RECOMMENDED NEXT)
**⏳ Phase 6: Vision Assessment** - NOT STARTED

**Latest Achievement:**
- Genetic algorithm successfully optimized 12 physics parameters
- Best fitness: 48.04 (validated: 46.24)
- Performance: 252.8 FPS @ 5000 particles (100/100 score)
- 270% improvement over baseline (12.5 → 46.24)

**See:** `ml/optimization/COMPREHENSIVE_PROGRESS_SUMMARY.md` for full details

---

## Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         IMPLEMENTATION PHASES                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ✅ PHASE 1: Runtime Controls (8-12 hours) - COMPLETE                       │
│  ├── ✅ 1.1 Add all parameter CLI flags                                     │
│  ├── ✅ 1.2 Extend BenchmarkConfig structure                                │
│  ├── ✅ 1.3 Add ParticleSystem setters/getters                              │
│  ├── ✅ 1.4 Add ImGui runtime controls                                      │
│  └── ✅ 1.5 Test all parameters                                             │
│                                                                             │
│  ⏭️ PHASE 2: Enhanced Metrics (6-8 hours) - SKIPPED                         │
│  ├── ⏭️ 2.1 Implement vortex detection (deferred to Phase 5)                │
│  ├── ⏭️ 2.2 Add turbulence quality metrics (deferred to Phase 5)            │
│  ├── ✅ 2.3 Extend benchmark JSON output (already done)                     │
│  └── ⏭️ 2.4 Add trajectory export (not needed yet)                          │
│                                                                             │
│  ✅ PHASE 3: Genetic Algorithm Optimizer (8-10 hours) - COMPLETE            │
│  ├── ✅ 3.1 Create Python optimization framework (DEAP, parallel)           │
│  ├── ✅ 3.2 Implement fitness function (multi-objective, fixed bugs)        │
│  ├── ✅ 3.3 Add parameter sweeping (genetic evolution)                      │
│  └── ✅ 3.4 Create convergence visualization (matplotlib)                   │
│      Achievement: 270% fitness improvement (12.5 → 46.24)                   │
│      Best FPS: 252.8 @ 5K particles (100/100 performance score)             │
│                                                                             │
│  ⏳ PHASE 4: Active Learning (10-12 hours) - NOT STARTED                    │
│  ├── ⏳ 4.1 Implement failure region detection                              │
│  ├── ⏳ 4.2 Create training data augmentation                               │
│  ├── ⏳ 4.3 Modify PINN training for weighted samples                       │
│  └── ⏳ 4.4 Implement iterative improvement loop                            │
│                                                                             │
│  ⏳ PHASE 5: Constrained Turbulence (8-10 hours) - RECOMMENDED NEXT         │
│  ├── ⏳ 5.1 Create physics-constrained SIREN                                │
│  ├── ⏳ 5.2 Implement conservation loss functions                           │
│  ├── ⏳ 5.3 Generate orbital-frame training data                            │
│  └── ⏳ 5.4 Train and validate new model                                    │
│                                                                             │
│  ⏳ PHASE 6: Vision Assessment (Optional, 6-8 hours) - NOT STARTED          │
│  ├── ⏳ 6.1 Add headless frame rendering                                    │
│  ├── ⏳ 6.2 Integrate vision API                                            │
│  └── ⏳ 6.3 Combine with numerical metrics                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Runtime Parameter Controls (8-12 hours)

### 1.1 Add CLI Flags to BenchmarkRunner (2-3 hours)

**File:** `src/benchmark/BenchmarkRunner.cpp`

**Task:** Extend `ParseCommandLine()` with all physics parameters.

```cpp
// Add these cases to ParseCommandLine()

// === PHYSICS PARAMETERS ===
else if (arg == "--gm" && i + 1 < argc) {
    outConfig.physics.gm = std::stof(argv[++i]);
}
else if (arg == "--alpha" && i + 1 < argc) {
    outConfig.physics.alphaViscosity = std::stof(argv[++i]);
}
else if (arg == "--angular-boost" && i + 1 < argc) {
    outConfig.physics.angularMomentumBoost = std::stof(argv[++i]);
}
else if (arg == "--damping" && i + 1 < argc) {
    outConfig.physics.damping = std::stof(argv[++i]);
}
else if (arg == "--bh-mass" && i + 1 < argc) {
    outConfig.physics.blackHoleMass = std::stof(argv[++i]);
}
else if (arg == "--disk-thickness" && i + 1 < argc) {
    outConfig.physics.diskThickness = std::stof(argv[++i]);
}
else if (arg == "--density-scale" && i + 1 < argc) {
    outConfig.physics.densityScale = std::stof(argv[++i]);
}
else if (arg == "--inner-radius" && i + 1 < argc) {
    outConfig.physics.innerRadius = std::stof(argv[++i]);
}
else if (arg == "--outer-radius" && i + 1 < argc) {
    outConfig.physics.outerRadius = std::stof(argv[++i]);
}
else if (arg == "--force-clamp" && i + 1 < argc) {
    outConfig.simulation.forceClamp = std::stof(argv[++i]);
}
else if (arg == "--velocity-clamp" && i + 1 < argc) {
    outConfig.simulation.velocityClamp = std::stof(argv[++i]);
}
else if (arg == "--boundary-mode" && i + 1 < argc) {
    outConfig.simulation.boundaryMode = std::stoi(argv[++i]);
}
```

**Checklist:**
- [ ] Add all 15+ parameter flags
- [ ] Add validation (min/max bounds)
- [ ] Update help text with all flags
- [ ] Test each flag individually

---

### 1.2 Extend BenchmarkConfig Structure (1 hour)

**File:** `src/benchmark/BenchmarkConfig.h`

**Task:** Add `PhysicsConfig`, `SimulationConfig`, `TurbulenceConfig` structs.

```cpp
// Add these structures

struct PhysicsConfig {
    float gm = 100.0f;
    float alphaViscosity = 0.1f;
    float angularMomentumBoost = 1.0f;
    float damping = 1.0f;
    float blackHoleMass = 1.0f;
    float diskThickness = 0.1f;
    float densityScale = 1.0f;
    float innerRadius = 6.0f;
    float outerRadius = 300.0f;
};

struct SimulationConfig {
    float forceClamp = 10.0f;
    float velocityClamp = 20.0f;
    int boundaryMode = 1;
};

struct TurbulenceConfig {
    bool sirenEnabled = false;
    float sirenIntensity = 0.5f;
    float sirenSeed = 0.0f;
    bool conserveAngularMomentum = true;
};

// Add to BenchmarkConfig:
PhysicsConfig physics;
SimulationConfig simulation;
TurbulenceConfig turbulence;
```

**Checklist:**
- [ ] Create `PhysicsConfig` struct
- [ ] Create `SimulationConfig` struct
- [ ] Create `TurbulenceConfig` struct
- [ ] Add to `BenchmarkConfig`

---

### 1.3 Add ParticleSystem Setters/Getters (2-3 hours)

**File:** `src/particles/ParticleSystem.h`

**Task:** Add setter/getter for each runtime parameter.

```cpp
// Add declarations
void SetGM(float gm);
float GetGM() const;

void SetAlphaViscosity(float alpha);
float GetAlphaViscosity() const;

void SetAngularMomentumBoost(float boost);
float GetAngularMomentumBoost() const;

// ... etc for all parameters
```

**File:** `src/particles/ParticleSystem.cpp`

**Task:** Implement setters with clamping.

```cpp
void ParticleSystem::SetGM(float gm) {
    m_pinnGM = std::clamp(gm, 10.0f, 500.0f);
    LOG_INFO("[Physics] GM set to: {:.1f}", m_pinnGM);
}

float ParticleSystem::GetGM() const {
    return m_pinnGM;
}

// ... etc
```

**Checklist:**
- [ ] Add member variables for new parameters
- [ ] Implement setters with bounds checking
- [ ] Implement getters
- [ ] Add logging for parameter changes
- [ ] Update `IntegrateForces()` to use new parameters

---

### 1.4 Add ImGui Runtime Controls (2-3 hours)

**File:** `src/core/Application.cpp`

**Task:** Add ImGui sliders/controls in `RenderImGui()`.

```cpp
if (ImGui::CollapsingHeader("Physics Parameters", ImGuiTreeNodeFlags_DefaultOpen)) {
    
    // Gravity
    float gm = m_particleSystem->GetGM();
    if (ImGui::SliderFloat("GM (Gravity)", &gm, 10.0f, 500.0f)) {
        m_particleSystem->SetGM(gm);
    }
    
    // Viscosity
    float alpha = m_particleSystem->GetAlphaViscosity();
    if (ImGui::SliderFloat("Alpha Viscosity", &alpha, 0.001f, 1.0f, "%.3f", ImGuiSliderFlags_Logarithmic)) {
        m_particleSystem->SetAlphaViscosity(alpha);
    }
    
    // ... add all other parameters
}
```

**Checklist:**
- [ ] Add "Physics Parameters" collapsing header
- [ ] Add sliders for all parameters
- [ ] Add tooltips/help markers
- [ ] Group related parameters (Gravity, Viscosity, Geometry, etc.)
- [ ] Add "Reset to Defaults" button

---

### 1.5 Test All Parameters (1-2 hours)

**Task:** Verify each parameter affects simulation correctly.

**Test Script:**
```bash
# Test each parameter
./PlasmaDX-Clean.exe --benchmark --pinn v4 --gm 50 --frames 200
./PlasmaDX-Clean.exe --benchmark --pinn v4 --gm 200 --frames 200
./PlasmaDX-Clean.exe --benchmark --pinn v4 --alpha 0.01 --frames 200
./PlasmaDX-Clean.exe --benchmark --pinn v4 --alpha 0.5 --frames 200
# ... etc
```

**Checklist:**
- [ ] Each parameter changes simulation behavior
- [ ] No crashes with extreme values
- [ ] JSON output includes all parameters
- [ ] ImGui controls update simulation in real-time

---

## Phase 2: Enhanced Metrics (6-8 hours)

### 2.1 Implement Vortex Detection (3-4 hours)

**File:** `src/particles/ParticleSystem.h`

```cpp
struct VortexMetrics {
    uint32_t vortexCount;
    float avgVortexStrength;
    float vortexCoherence;
    float angularVelocityVariance;
};

VortexMetrics DetectVortices() const;
```

**File:** `src/particles/ParticleSystem.cpp`

**Task:** Implement grid-based vorticity estimation.

**Algorithm:**
1. Create 20x20 grid over disk
2. Compute average velocity per cell
3. Estimate curl using finite differences
4. Identify high-vorticity regions
5. Cluster into vortex centers

**Checklist:**
- [ ] Implement `DetectVortices()`
- [ ] Test with SIREN turbulence enabled
- [ ] Verify vortex count is reasonable (0-20)
- [ ] Log vortex metrics

---

### 2.2 Add Turbulence Quality Metrics (1-2 hours)

**File:** `src/benchmark/BenchmarkMetrics.h`

```cpp
struct TurbulenceMetrics {
    MetricStats vortexCount;
    MetricStats vortexStrength;
    MetricStats vortexCoherence;
    MetricStats angularVelocityVariance;
    
    // Derived
    float turbulenceQualityScore;  // 0-100
};
```

**Checklist:**
- [ ] Add `TurbulenceMetrics` struct
- [ ] Add to `BenchmarkResults`
- [ ] Compute `turbulenceQualityScore`
- [ ] Include in overall score calculation

---

### 2.3 Extend Benchmark JSON Output (1 hour)

**File:** `src/benchmark/BenchmarkRunner.cpp`

**Task:** Add turbulence section to JSON.

```json
"turbulence": {
    "vortex_count": { "mean": 3.5, "max": 8 },
    "vortex_strength": { "mean": 0.023, "max": 0.045 },
    "vortex_coherence": 0.65,
    "turbulence_quality_score": 72.5
}
```

**Checklist:**
- [ ] Add turbulence section to JSON
- [ ] Add to CSV output
- [ ] Include all config parameters in output

---

### 2.4 Add Trajectory Export (1-2 hours)

**File:** `src/benchmark/BenchmarkRunner.cpp`

**Task:** Export particle positions/velocities at intervals.

```cpp
bool BenchmarkRunner::ExportTrajectory(const std::string& path) {
    std::ofstream file(path);
    file << "frame,particle_id,x,y,z,vx,vy,vz,r,v_mag\n";
    
    // Export every 10th frame, every 100th particle
    for (auto& snapshot : m_trajectorySnapshots) {
        for (int i = 0; i < particleCount; i += 100) {
            file << snapshot.frame << "," << i << ","
                 << snapshot.positions[i].x << "," 
                 << snapshot.positions[i].y << ","
                 << snapshot.positions[i].z << ","
                 << snapshot.velocities[i].x << ","
                 << snapshot.velocities[i].y << ","
                 << snapshot.velocities[i].z << ","
                 << snapshot.radii[i] << ","
                 << snapshot.speeds[i] << "\n";
        }
    }
    return true;
}
```

**Checklist:**
- [ ] Add `--export-trajectory <path>` flag
- [ ] Store trajectory snapshots during benchmark
- [ ] Export to CSV
- [ ] Test with Python analysis scripts

---

## Phase 3: Genetic Algorithm Optimizer (8-10 hours)

### 3.1 Create Python Optimization Framework (2-3 hours)

**File:** `ml/optimization/genetic_optimizer.py`

**Task:** Set up DEAP-based genetic algorithm.

```bash
# Install dependencies
pip install deap numpy scipy matplotlib
```

**Structure:**
```python
# Define parameter space
PARAM_BOUNDS = {
    'gm': (50.0, 200.0),
    'alpha': (0.01, 0.5),
    # ... all parameters
}

# Create individual (parameter set)
# Evaluate fitness (run benchmark)
# Evolve population
```

**Checklist:**
- [ ] Create `ml/optimization/` directory
- [ ] Install DEAP: `pip install deap`
- [ ] Define parameter bounds
- [ ] Create `Individual` class
- [ ] Test basic evolution loop

---

### 3.2 Implement Fitness Function (2-3 hours)

**File:** `ml/optimization/genetic_optimizer.py`

**Task:** Call benchmark and compute fitness.

```python
def evaluate(individual):
    params = decode_individual(individual)
    
    # Run benchmark
    result = run_benchmark(params)
    
    # Multi-objective fitness
    fitness = (
        0.35 * result['stability_score'] +
        0.30 * result['accuracy_score'] +
        0.20 * result['performance_score'] +
        0.15 * result['visual_score']
    )
    
    # Bonus for vortices
    if result.get('vortex_count', 0) > 0:
        fitness += 5.0
    
    return (fitness,)
```

**Checklist:**
- [ ] Implement `run_benchmark()` subprocess call
- [ ] Parse JSON output
- [ ] Compute weighted fitness
- [ ] Handle benchmark failures gracefully
- [ ] Add vortex bonus

---

### 3.3 Add Parameter Sweeping (2 hours)

**File:** `ml/optimization/param_sweep.py`

**Task:** Grid/random search for baseline comparison.

```python
def grid_sweep(param_name, values, base_params):
    results = []
    for value in values:
        params = base_params.copy()
        params[param_name] = value
        result = run_benchmark(params)
        results.append({
            'value': value,
            'score': result['overall_score']
        })
    return results
```

**Checklist:**
- [ ] Implement `grid_sweep()`
- [ ] Implement `random_sweep()`
- [ ] Save sweep results to CSV
- [ ] Create sweep visualization

---

### 3.4 Create Convergence Visualization (2 hours)

**File:** `ml/optimization/visualization.py`

**Task:** Plot fitness over generations.

```python
import matplotlib.pyplot as plt

def plot_convergence(logbook, output_path):
    gen = logbook.select("gen")
    avg = logbook.select("avg")
    max_ = logbook.select("max")
    
    plt.figure(figsize=(10, 6))
    plt.plot(gen, avg, label="Average")
    plt.plot(gen, max_, label="Best")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.savefig(output_path)
```

**Checklist:**
- [ ] Plot fitness vs generation
- [ ] Plot parameter evolution
- [ ] Save best individual per generation
- [ ] Create summary report

---

## Phase 4: Active Learning (10-12 hours)

### 4.1 Implement Failure Region Detection (3-4 hours)

**File:** `ml/active_learning/failure_detector.py`

**Task:** Analyze trajectories to find high-error regions.

**Algorithm:**
1. Load trajectory CSV
2. Compute Keplerian velocity error per particle
3. Cluster high-error regions with DBSCAN
4. Return region centers and sizes

```python
def detect_failures(trajectory_csv, error_threshold=0.3):
    df = pd.read_csv(trajectory_csv)
    
    failures = []
    for _, row in df.iterrows():
        v_kepler = sqrt(GM / row['r'])
        v_actual = row['v_mag']
        error = abs(v_actual - v_kepler) / v_kepler
        
        if error > error_threshold:
            failures.append({
                'position': (row['x'], row['y'], row['z']),
                'error': error
            })
    
    return cluster_failures(failures)
```

**Checklist:**
- [ ] Parse trajectory CSV
- [ ] Compute velocity errors
- [ ] Implement DBSCAN clustering
- [ ] Return failure regions

---

### 4.2 Create Training Data Augmentation (3-4 hours)

**File:** `ml/active_learning/data_augmentation.py`

**Task:** Generate additional training samples in failure regions.

```python
def augment_training_data(base_data_path, failure_regions, samples_per_region=2000):
    # Load base data
    data = np.load(base_data_path)
    states = data['states']
    forces = data['forces']
    
    # Generate new samples in failure regions
    new_states, new_forces = [], []
    for region in failure_regions:
        samples = generate_samples_in_region(region, samples_per_region)
        new_states.extend(samples['states'])
        new_forces.extend(samples['forces'])
    
    # Combine with original
    augmented_states = np.vstack([states, new_states])
    augmented_forces = np.vstack([forces, new_forces])
    
    return augmented_states, augmented_forces
```

**Checklist:**
- [ ] Load existing training data
- [ ] Generate samples in failure regions
- [ ] Compute ground-truth forces analytically
- [ ] Save augmented dataset

---

### 4.3 Modify PINN Training for Weighted Samples (2 hours)

**File:** `ml/pinn_v4_with_turbulence.py`

**Task:** Support sample weights in loss function.

```python
def weighted_mse_loss(pred, target, weights):
    return (weights * (pred - target) ** 2).mean()

# In training loop:
loss = weighted_mse_loss(F_pred, F_true, sample_weights)
```

**Checklist:**
- [ ] Add `--weighted-loss` flag
- [ ] Load sample weights from NPZ
- [ ] Modify loss computation
- [ ] Test weighted training

---

### 4.4 Implement Iterative Improvement Loop (3-4 hours)

**File:** `ml/active_learning/active_learning_loop.py`

**Task:** Orchestrate full active learning pipeline.

```python
def run_active_learning(max_iterations=10, target_score=85.0):
    current_model = initial_model
    
    for iteration in range(max_iterations):
        # 1. Benchmark
        result = run_benchmark(current_model)
        
        # 2. Detect failures
        failures = detect_failures(result['trajectory'])
        
        # 3. Augment data
        augmented_data = augment_training_data(failures)
        
        # 4. Retrain
        new_model = retrain_pinn(augmented_data)
        
        # 5. Evaluate
        new_score = run_benchmark(new_model)['overall_score']
        
        if new_score >= target_score:
            break
        
        current_model = new_model
    
    return current_model
```

**Checklist:**
- [ ] Implement main loop
- [ ] Add convergence detection
- [ ] Save intermediate models
- [ ] Create iteration report

---

## Phase 5: Constrained Turbulence (8-10 hours)

### 5.1 Create Physics-Constrained SIREN (3-4 hours)

**File:** `ml/vortex_field/constrained_siren.py`

**Task:** SIREN that conserves angular momentum.

**Key Constraint:**
```python
# Turbulent force: F = v × ω
# Angular momentum contribution: L = r × F
# Constraint: Σ L = 0 (no net torque)

F_turb = cross(velocity, omega_pred)
L_total = sum(cross(position, F_turb))

# Project out component causing net torque
omega_constrained = omega_pred - correction_factor * L_total
```

**Checklist:**
- [ ] Create `ConstrainedTurbulenceSIREN` class
- [ ] Implement angular momentum projection
- [ ] Test constraint effectiveness
- [ ] Export to ONNX

---

### 5.2 Implement Conservation Loss Functions (2 hours)

**File:** `ml/vortex_field/constrained_siren.py`

**Task:** Multi-objective physics-informed loss.

```python
class PhysicsInformedTurbulenceLoss:
    def forward(self, omega, positions, velocities):
        # Loss 1: Angular momentum conservation
        L_conservation = norm(sum(cross(positions, cross(velocities, omega))))
        
        # Loss 2: No radial drift
        drift_penalty = mean(dot(F_turb, r_hat)) ** 2
        
        # Loss 3: Vortex structure (reward)
        vortex_reward = -variance(omega)
        
        return L_conservation + drift_penalty + vortex_reward
```

**Checklist:**
- [ ] Implement `L_conservation` loss
- [ ] Implement `drift_penalty` loss
- [ ] Implement `vortex_reward` loss
- [ ] Tune loss weights

---

### 5.3 Generate Orbital-Frame Training Data (2-3 hours)

**File:** `ml/vortex_field/generate_orbital_vortex_data.py`

**Task:** Create training data with vortices that respect orbital dynamics.

```python
def generate_orbital_vortex_sample():
    # Particle on Keplerian orbit
    r = random.uniform(20, 250)
    v_kepler = sqrt(GM / r)
    
    # Position and velocity
    theta = random.uniform(0, 2*pi)
    x, z = r * cos(theta), r * sin(theta)
    vx, vz = -v_kepler * sin(theta), v_kepler * cos(theta)
    
    # Target vorticity (respects angular momentum)
    omega = generate_constrained_vorticity(x, z, r, theta)
    
    return {'position': [x, 0, z], 'velocity': [vx, 0, vz], 'omega': omega}
```

**Checklist:**
- [ ] Generate Keplerian orbit samples
- [ ] Create vorticity that respects constraints
- [ ] Validate angular momentum conservation
- [ ] Save training data

---

### 5.4 Train and Validate New Model (2 hours)

**Task:** Train constrained SIREN and verify it doesn't break orbits.

```bash
# Train
python ml/vortex_field/train_constrained_siren.py --epochs 200 --output ml/models/constrained_siren.onnx

# Validate
./PlasmaDX-Clean.exe --benchmark --pinn v4 --siren ml/models/constrained_siren.onnx --frames 1000
```

**Success Criteria:**
- Angular momentum drift < 5%
- Vortex count > 0
- Orbits remain Keplerian (error < 10%)

**Checklist:**
- [ ] Train model
- [ ] Export to ONNX
- [ ] Benchmark with new model
- [ ] Verify constraints are satisfied

---

## Phase 6: Vision Assessment (Optional, 6-8 hours)

### 6.1 Add Headless Frame Rendering (3-4 hours)

**File:** `src/particles/ParticleRenderer_Gaussian.cpp`

**Task:** Render to texture without window.

```cpp
bool RenderToTexture(ID3D12Resource* target, uint32_t width, uint32_t height) {
    // Create render target view
    // Set viewport
    // Render particles
    // Copy to output texture
}
```

**Checklist:**
- [ ] Create offscreen render target
- [ ] Render without swap chain
- [ ] Save to PNG file
- [ ] Add `--capture-frame <path>` flag

---

### 6.2 Integrate Vision API (2 hours)

**File:** `ml/quality_assessment/vision_assessor.py`

**Task:** Call AI vision API to assess frames.

```python
def assess_frame(frame_path):
    with open(frame_path, 'rb') as f:
        image_data = base64.b64encode(f.read())
    
    response = call_vision_api(image_data, ASSESSMENT_PROMPT)
    return parse_assessment(response)
```

**Checklist:**
- [ ] Create API wrapper
- [ ] Define assessment prompt
- [ ] Parse response to scores
- [ ] Handle API errors

---

### 6.3 Combine with Numerical Metrics (2 hours)

**File:** `ml/quality_assessment/combined_scorer.py`

**Task:** Weighted combination of numerical and vision scores.

```python
def combined_score(benchmark_results, vision_results):
    numerical = benchmark_results['overall_score']
    vision = vision_results['overall_score'] * 10  # Scale to 0-100
    
    return 0.7 * numerical + 0.3 * vision
```

**Checklist:**
- [ ] Load benchmark JSON
- [ ] Load vision assessment
- [ ] Compute combined score
- [ ] Generate combined report

---

## Milestone Checkpoints

### Milestone 1: Runtime Controls Complete
- [ ] All 15+ parameters controllable via CLI
- [ ] All parameters in ImGui
- [ ] Benchmark exports all parameters
- **Expected:** End of Week 1

### Milestone 2: Enhanced Metrics Complete
- [ ] Vortex detection working
- [ ] Turbulence quality score in output
- [ ] Trajectory export functional
- **Expected:** End of Week 2

### Milestone 3: Genetic Optimizer Working
- [ ] GA finds better parameters than random
- [ ] Convergence plots generated
- [ ] Can run overnight sweeps
- **Expected:** End of Week 3

### Milestone 4: Active Learning Loop Complete
- [ ] Failure regions identified automatically
- [ ] Model improves over iterations
- [ ] Score increases measurably
- **Expected:** End of Week 4

### Milestone 5: Constrained Turbulence Ready
- [ ] SIREN preserves angular momentum
- [ ] Visible vortex structures
- [ ] Orbits remain stable with turbulence
- **Expected:** End of Week 5

### Milestone 6: Full Pipeline Operational
- [ ] End-to-end optimization works
- [ ] Presets generated automatically
- [ ] Target score achievable
- **Expected:** End of Week 6

---

## Quick Reference: Key Files

| File | Purpose |
|------|---------|
| `src/benchmark/BenchmarkConfig.h` | All config structures |
| `src/benchmark/BenchmarkRunner.cpp` | Main benchmark logic |
| `src/particles/ParticleSystem.h` | Physics parameters |
| `src/core/Application.cpp` | ImGui controls |
| `ml/optimization/genetic_optimizer.py` | GA implementation |
| `ml/active_learning/failure_detector.py` | Failure region detection |
| `ml/vortex_field/constrained_siren.py` | Physics-constrained turbulence |
| `ml/quality_assessment/combined_scorer.py` | Combined scoring |

---

## Testing Commands

```bash
# Test runtime parameters
./PlasmaDX-Clean.exe --benchmark --pinn v4 --gm 150 --alpha 0.05 --frames 500

# Run genetic optimization
python ml/optimization/genetic_optimizer.py --generations 50 --population 30

# Run active learning
python ml/active_learning/active_learning_loop.py --iterations 5

# Train constrained SIREN
python ml/vortex_field/train_constrained_siren.py --epochs 200

# Full pipeline
python ml/optimization/master_pipeline.py --target-score 90
```

---

## Success Criteria

| Metric | Current | Target | Notes |
|--------|---------|--------|-------|
| Overall Score | 32.5 | 85.0+ | Primary goal |
| Energy Drift | -36% | < 5% | Conservation |
| Angular Momentum Drift | -54% | < 5% | Conservation |
| Keplerian Error | 24% | < 10% | Physical accuracy |
| Vortex Count | 0 | 3-10 | Turbulence quality |
| FPS | 94 | > 60 | Performance |

---

**End of Roadmap**

**Total Estimated Time:** 40-60 hours across 6 phases
**Recommended Order:** Phase 1 → 2 → 3 → 5 → 4 → 6

Phase 1 and 2 are prerequisites for all others. Phase 5 (constrained turbulence) can be done in parallel with Phase 3 and 4.

