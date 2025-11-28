# ML-Driven Physics Optimization System

## Complete Technical Specification v1.0

**Project:** PlasmaDX-Clean - Ultimate Galaxy Physics Model Creator
**Date:** 2025-11-28
**Status:** Specification Complete

---

## 1. Executive Summary

This specification defines a comprehensive ML-driven system for:
1. **Automated parameter optimization** using genetic algorithms and Bayesian optimization
2. **Active learning** to identify and fill gaps in PINN training data
3. **Physics-constrained turbulence models** that preserve orbital mechanics
4. **Multi-modal quality assessment** combining numerical metrics and AI vision
5. **Self-improving feedback loops** for continuous model refinement

The system builds on the existing PINN Benchmark infrastructure to create an autonomous physics simulation optimizer.

---

## 2. System Architecture

### 2.1 High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    GALAXY PHYSICS MODEL CREATOR                                  │
│                         Complete System Architecture                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                        LAYER 1: RUNTIME CONTROLS                         │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐        │   │
│  │  │ Gravity     │ │ Viscosity   │ │ Angular     │ │ Turbulence  │        │   │
│  │  │ (GM)        │ │ (α)         │ │ Momentum    │ │ (SIREN)     │        │   │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘        │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐        │   │
│  │  │ Damping     │ │ Density     │ │ Black Hole  │ │ Boundaries  │        │   │
│  │  │             │ │ Scale       │ │ Mass        │ │             │        │   │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘        │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                        │                                        │
│                                        ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                        LAYER 2: BENCHMARK ENGINE                         │   │
│  │                                                                           │   │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐       │   │
│  │  │ Headless         │  │ Metric           │  │ Quality          │       │   │
│  │  │ Simulation       │──│ Collection       │──│ Scoring          │       │   │
│  │  │ (100-2000 frames)│  │ (snapshots)      │  │ (0-100)          │       │   │
│  │  └──────────────────┘  └──────────────────┘  └──────────────────┘       │   │
│  │                                                                           │   │
│  │  Outputs: JSON results, CSV time-series, preset files                    │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                        │                                        │
│                                        ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                        LAYER 3: OPTIMIZATION ENGINE                      │   │
│  │                                                                           │   │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐       │   │
│  │  │ Genetic          │  │ Bayesian         │  │ Grid/Random      │       │   │
│  │  │ Algorithm        │  │ Optimization     │  │ Search           │       │   │
│  │  │ (exploration)    │  │ (exploitation)   │  │ (baseline)       │       │   │
│  │  └──────────────────┘  └──────────────────┘  └──────────────────┘       │   │
│  │                                                                           │   │
│  │  Fitness = f(stability, accuracy, performance, visual_quality)           │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                        │                                        │
│                                        ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                        LAYER 4: ACTIVE LEARNING                          │   │
│  │                                                                           │   │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐       │   │
│  │  │ Failure Region   │  │ Training Data    │  │ Model            │       │   │
│  │  │ Identification   │──│ Augmentation     │──│ Retraining       │       │   │
│  │  │                  │  │                  │  │                  │       │   │
│  │  └──────────────────┘  └──────────────────┘  └──────────────────┘       │   │
│  │                                                                           │   │
│  │  Identifies: weak force regions, high error zones, instabilities        │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                        │                                        │
│                                        ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │                        LAYER 5: QUALITY ASSESSMENT                       │   │
│  │                                                                           │   │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐       │   │
│  │  │ Numerical        │  │ Vortex           │  │ AI Vision        │       │   │
│  │  │ Metrics          │  │ Detection        │  │ Assessment       │       │   │
│  │  │ (physics)        │  │ (turbulence)     │  │ (aesthetics)     │       │   │
│  │  └──────────────────┘  └──────────────────┘  └──────────────────┘       │   │
│  │                                                                           │   │
│  │  Combined Score = 0.5*Physics + 0.3*Turbulence + 0.2*Visual              │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Runtime Parameter Controls

### 3.1 Complete Parameter List

All parameters controllable via command-line flags AND runtime ImGui controls:

| Parameter | Flag | Range | Default | Physics Effect |
|-----------|------|-------|---------|----------------|
| Gravitational Parameter | `--gm` | 10-500 | 100.0 | Orbital velocity, force magnitude |
| Alpha Viscosity | `--alpha` | 0.001-1.0 | 0.1 | Angular momentum transfer rate |
| Angular Momentum Boost | `--angular-boost` | 0.1-5.0 | 1.0 | Initial orbital velocity multiplier |
| Velocity Damping | `--damping` | 0.9-1.0 | 1.0 | Energy dissipation per frame |
| Black Hole Mass | `--bh-mass` | 0.1-10.0 | 1.0 | Gravitational well depth |
| Disk Thickness (H/R) | `--disk-thickness` | 0.01-0.5 | 0.1 | Vertical extent of disk |
| Density Scale | `--density-scale` | 0.1-10.0 | 1.0 | Global density multiplier |
| Inner Radius | `--inner-radius` | 1-50 | 6.0 | ISCO (innermost stable orbit) |
| Outer Radius | `--outer-radius` | 100-1000 | 300.0 | Disk edge |
| SIREN Intensity | `--siren-intensity` | 0-5.0 | 0.5 | Turbulence strength |
| SIREN Seed | `--siren-seed` | 0-1000 | 0.0 | Deterministic randomness |
| Timescale | `--timescale` | 0.1-100 | 1.0 | Simulation speed |
| Force Clamp | `--force-clamp` | 0.1-100 | 10.0 | Maximum force magnitude |
| Velocity Clamp | `--velocity-clamp` | 1-100 | 20.0 | Maximum velocity magnitude |
| Boundary Mode | `--boundary-mode` | 0-3 | 1 | 0=none, 1=reflect, 2=wrap, 3=respawn |

### 3.2 Command-Line Interface Extension

```cpp
// In BenchmarkRunner::ParseCommandLine()

// === Physics Parameters ===
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

// === Simulation Parameters ===
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

### 3.3 Extended Config Structure

```cpp
// In BenchmarkConfig.h

struct PhysicsConfig {
    // Gravitational
    float gm = 100.0f;                    // Gravitational parameter (G*M)
    float blackHoleMass = 1.0f;           // Mass multiplier (0.1-10.0)
    
    // Viscosity
    float alphaViscosity = 0.1f;          // Shakura-Sunyaev alpha (0.001-1.0)
    float damping = 1.0f;                 // Velocity damping (0.9-1.0)
    
    // Disk geometry
    float diskThickness = 0.1f;           // H/R ratio (0.01-0.5)
    float innerRadius = 6.0f;             // ISCO
    float outerRadius = 300.0f;           // Disk edge
    
    // Density
    float densityScale = 1.0f;            // Global density multiplier
    
    // Dynamics
    float angularMomentumBoost = 1.0f;    // Initial velocity boost
};

struct SimulationConfig {
    float forceClamp = 10.0f;             // Max force magnitude
    float velocityClamp = 20.0f;          // Max velocity magnitude
    int boundaryMode = 1;                 // Boundary handling
};

struct TurbulenceConfig {
    bool sirenEnabled = false;
    float sirenIntensity = 0.5f;
    float sirenSeed = 0.0f;
    
    // Physics constraints (NEW)
    bool conserveAngularMomentum = true;  // Project out net torque
    float vortexScale = 1.0f;             // Eddy size multiplier
    float vortexDecay = 0.1f;             // Temporal decay rate
};

struct BenchmarkConfig {
    // ... existing fields ...
    PhysicsConfig physics;
    SimulationConfig simulation;
    TurbulenceConfig turbulence;
};
```

### 3.4 ParticleSystem Integration

```cpp
// In ParticleSystem.h - Add setters for all runtime parameters

// === Physics Parameter Setters ===
void SetGM(float gm) { m_pinnGM = std::clamp(gm, 10.0f, 500.0f); }
float GetGM() const { return m_pinnGM; }

void SetAlphaViscosity(float alpha) { m_alphaViscosity = std::clamp(alpha, 0.001f, 1.0f); }
float GetAlphaViscosity() const { return m_alphaViscosity; }

void SetAngularMomentumBoost(float boost) { m_angularBoost = std::clamp(boost, 0.1f, 5.0f); }
float GetAngularMomentumBoost() const { return m_angularBoost; }

void SetDamping(float damping) { m_pinnDamping = std::clamp(damping, 0.9f, 1.0f); }
float GetDamping() const { return m_pinnDamping; }

void SetBlackHoleMassMultiplier(float mult) { m_bhMassMultiplier = std::clamp(mult, 0.1f, 10.0f); }
float GetBlackHoleMassMultiplier() const { return m_bhMassMultiplier; }

void SetDiskThickness(float hr) { m_diskThicknessHR = std::clamp(hr, 0.01f, 0.5f); }
float GetDiskThickness() const { return m_diskThicknessHR; }

void SetDensityScale(float scale) { m_densityScale = std::clamp(scale, 0.1f, 10.0f); }
float GetDensityScale() const { return m_densityScale; }

void SetInnerRadius(float r) { m_innerRadius = std::clamp(r, 1.0f, 50.0f); }
float GetInnerRadius() const { return m_innerRadius; }

void SetOuterRadius(float r) { m_outerRadius = std::clamp(r, 100.0f, 1000.0f); }
float GetOuterRadius() const { return m_outerRadius; }

void SetForceClamp(float clamp) { m_forceClamp = std::clamp(clamp, 0.1f, 100.0f); }
float GetForceClamp() const { return m_forceClamp; }

void SetVelocityClamp(float clamp) { m_velocityClamp = std::clamp(clamp, 1.0f, 100.0f); }
float GetVelocityClamp() const { return m_velocityClamp; }

void SetBoundaryMode(int mode) { m_boundaryMode = std::clamp(mode, 0, 3); }
int GetBoundaryMode() const { return m_boundaryMode; }
```

### 3.5 ImGui Runtime Controls

```cpp
// In Application::RenderImGui() - Physics Controls Section

if (ImGui::CollapsingHeader("Physics Parameters", ImGuiTreeNodeFlags_DefaultOpen)) {
    
    // === Gravity Section ===
    ImGui::Text("Gravity & Mass");
    ImGui::Separator();
    
    float gm = m_particleSystem->GetGM();
    if (ImGui::SliderFloat("GM (Gravity)", &gm, 10.0f, 500.0f, "%.1f")) {
        m_particleSystem->SetGM(gm);
    }
    ImGui::SameLine(); HelpMarker("Gravitational parameter. Higher = stronger gravity, faster orbits.");
    
    float bhMass = m_particleSystem->GetBlackHoleMassMultiplier();
    if (ImGui::SliderFloat("BH Mass Mult", &bhMass, 0.1f, 10.0f, "%.2f")) {
        m_particleSystem->SetBlackHoleMassMultiplier(bhMass);
    }
    
    // === Viscosity Section ===
    ImGui::Spacing();
    ImGui::Text("Viscosity & Damping");
    ImGui::Separator();
    
    float alpha = m_particleSystem->GetAlphaViscosity();
    if (ImGui::SliderFloat("Alpha Viscosity", &alpha, 0.001f, 1.0f, "%.3f", ImGuiSliderFlags_Logarithmic)) {
        m_particleSystem->SetAlphaViscosity(alpha);
    }
    ImGui::SameLine(); HelpMarker("Shakura-Sunyaev viscosity. Higher = faster angular momentum transfer.");
    
    float damping = m_particleSystem->GetDamping();
    if (ImGui::SliderFloat("Velocity Damping", &damping, 0.9f, 1.0f, "%.4f")) {
        m_particleSystem->SetDamping(damping);
    }
    
    // === Geometry Section ===
    ImGui::Spacing();
    ImGui::Text("Disk Geometry");
    ImGui::Separator();
    
    float diskH = m_particleSystem->GetDiskThickness();
    if (ImGui::SliderFloat("Disk H/R", &diskH, 0.01f, 0.5f, "%.3f")) {
        m_particleSystem->SetDiskThickness(diskH);
    }
    
    float innerR = m_particleSystem->GetInnerRadius();
    if (ImGui::SliderFloat("Inner Radius", &innerR, 1.0f, 50.0f, "%.1f")) {
        m_particleSystem->SetInnerRadius(innerR);
    }
    
    float outerR = m_particleSystem->GetOuterRadius();
    if (ImGui::SliderFloat("Outer Radius", &outerR, 100.0f, 1000.0f, "%.1f")) {
        m_particleSystem->SetOuterRadius(outerR);
    }
    
    // === Dynamics Section ===
    ImGui::Spacing();
    ImGui::Text("Dynamics");
    ImGui::Separator();
    
    float angBoost = m_particleSystem->GetAngularMomentumBoost();
    if (ImGui::SliderFloat("Angular Boost", &angBoost, 0.1f, 5.0f, "%.2f")) {
        m_particleSystem->SetAngularMomentumBoost(angBoost);
    }
    
    float densityScale = m_particleSystem->GetDensityScale();
    if (ImGui::SliderFloat("Density Scale", &densityScale, 0.1f, 10.0f, "%.2f")) {
        m_particleSystem->SetDensityScale(densityScale);
    }
    
    // === Safety Limits ===
    ImGui::Spacing();
    ImGui::Text("Safety Limits");
    ImGui::Separator();
    
    float forceClamp = m_particleSystem->GetForceClamp();
    if (ImGui::SliderFloat("Force Clamp", &forceClamp, 0.1f, 100.0f, "%.1f")) {
        m_particleSystem->SetForceClamp(forceClamp);
    }
    
    float velClamp = m_particleSystem->GetVelocityClamp();
    if (ImGui::SliderFloat("Velocity Clamp", &velClamp, 1.0f, 100.0f, "%.1f")) {
        m_particleSystem->SetVelocityClamp(velClamp);
    }
    
    const char* boundaryModes[] = { "None", "Reflect", "Wrap", "Respawn" };
    int boundaryMode = m_particleSystem->GetBoundaryMode();
    if (ImGui::Combo("Boundary Mode", &boundaryMode, boundaryModes, 4)) {
        m_particleSystem->SetBoundaryMode(boundaryMode);
    }
}
```

---

## 4. Genetic Algorithm Optimization

### 4.1 Parameter Genome

```python
# ml/optimization/genetic_optimizer.py

import numpy as np
from deap import base, creator, tools, algorithms
import subprocess
import json
import os

# Define the parameter space (genome)
PARAM_BOUNDS = {
    'gm': (50.0, 200.0),
    'alpha': (0.01, 0.5),
    'angular_boost': (0.5, 2.0),
    'damping': (0.95, 1.0),
    'bh_mass': (0.5, 2.0),
    'disk_thickness': (0.05, 0.2),
    'density_scale': (0.5, 2.0),
    'siren_intensity': (0.0, 1.0),
    'force_clamp': (5.0, 50.0),
    'velocity_clamp': (10.0, 50.0),
    'timescale': (1.0, 50.0),
}

# Create fitness and individual types
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

def create_individual():
    """Create random individual within bounds."""
    return creator.Individual([
        np.random.uniform(low, high)
        for low, high in PARAM_BOUNDS.values()
    ])

def decode_individual(individual):
    """Convert genome list to parameter dict."""
    keys = list(PARAM_BOUNDS.keys())
    return {keys[i]: individual[i] for i in range(len(keys))}

def run_benchmark(params, frames=500, exe_path="build/bin/Debug/PlasmaDX-Clean.exe"):
    """Run benchmark with given parameters and return results."""
    
    cmd = [
        exe_path,
        "--benchmark",
        "--pinn", "v4",
        "--frames", str(frames),
        "--output", "temp_benchmark.json",
        "--gm", str(params['gm']),
        "--alpha", str(params['alpha']),
        "--angular-boost", str(params['angular_boost']),
        "--damping", str(params['damping']),
        "--bh-mass", str(params['bh_mass']),
        "--disk-thickness", str(params['disk_thickness']),
        "--density-scale", str(params['density_scale']),
        "--force-clamp", str(params['force_clamp']),
        "--velocity-clamp", str(params['velocity_clamp']),
        "--timescale", str(params['timescale']),
    ]
    
    if params['siren_intensity'] > 0.01:
        cmd.extend(["--siren", "--siren-intensity", str(params['siren_intensity'])])
    
    try:
        subprocess.run(cmd, capture_output=True, timeout=120)
        with open("temp_benchmark.json", 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Benchmark failed: {e}")
        return None

def evaluate(individual):
    """Fitness function - runs benchmark and computes score."""
    params = decode_individual(individual)
    results = run_benchmark(params)
    
    if results is None:
        return (0.0,)  # Failed run
    
    # Extract metrics
    stability = results['summary']['stability_score']
    accuracy = results['summary']['accuracy_score']
    performance = results['summary']['performance_score']
    visual = results['summary']['visual_score']
    
    # Weighted fitness
    fitness = (
        0.35 * stability +
        0.30 * accuracy +
        0.20 * performance +
        0.15 * visual
    )
    
    # Bonus for vortex formation (if detected)
    if 'vortex_count' in results.get('turbulence', {}):
        vortex_bonus = min(10.0, results['turbulence']['vortex_count'] * 2.0)
        fitness += vortex_bonus
    
    # Penalty for extreme parameter values (prefer moderate settings)
    extremity_penalty = 0
    for key, (low, high) in PARAM_BOUNDS.items():
        val = params[key]
        mid = (low + high) / 2
        range_size = high - low
        deviation = abs(val - mid) / (range_size / 2)
        if deviation > 0.8:  # Very extreme
            extremity_penalty += 2.0
    
    fitness -= extremity_penalty
    
    return (max(0.0, fitness),)

def mutate(individual, indpb=0.2):
    """Mutate individual with Gaussian noise."""
    keys = list(PARAM_BOUNDS.keys())
    for i in range(len(individual)):
        if np.random.random() < indpb:
            low, high = PARAM_BOUNDS[keys[i]]
            sigma = (high - low) * 0.1  # 10% of range
            individual[i] += np.random.gauss(0, sigma)
            individual[i] = np.clip(individual[i], low, high)
    return (individual,)

def crossover(ind1, ind2):
    """Blend crossover."""
    alpha = 0.5
    for i in range(len(ind1)):
        if np.random.random() < 0.5:
            ind1[i], ind2[i] = (
                alpha * ind1[i] + (1 - alpha) * ind2[i],
                alpha * ind2[i] + (1 - alpha) * ind1[i]
            )
    return ind1, ind2

# Setup DEAP toolbox
toolbox = base.Toolbox()
toolbox.register("individual", create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", crossover)
toolbox.register("mutate", mutate, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

def run_evolution(population_size=30, generations=50, checkpoint_every=10):
    """Run genetic algorithm optimization."""
    
    print("="*60)
    print("GENETIC ALGORITHM PARAMETER OPTIMIZATION")
    print("="*60)
    print(f"Population: {population_size}")
    print(f"Generations: {generations}")
    print(f"Parameters: {len(PARAM_BOUNDS)}")
    print("="*60)
    
    # Create initial population
    pop = toolbox.population(n=population_size)
    
    # Statistics tracking
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    stats.register("min", np.min)
    
    # Hall of fame (best individuals)
    hof = tools.HallOfFame(10)
    
    # Run evolution
    pop, logbook = algorithms.eaSimple(
        pop, toolbox,
        cxpb=0.7,      # Crossover probability
        mutpb=0.2,     # Mutation probability
        ngen=generations,
        stats=stats,
        halloffame=hof,
        verbose=True
    )
    
    # Save results
    best = hof[0]
    best_params = decode_individual(best)
    
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)
    print(f"Best fitness: {best.fitness.values[0]:.2f}")
    print("Best parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value:.4f}")
    
    # Save to file
    with open("optimized_params.json", 'w') as f:
        json.dump({
            'best_params': best_params,
            'fitness': best.fitness.values[0],
            'generations': generations,
            'population_size': population_size
        }, f, indent=2)
    
    return best_params, logbook

if __name__ == "__main__":
    best_params, log = run_evolution(
        population_size=30,
        generations=50
    )
```

### 4.2 Bayesian Optimization Alternative

```python
# ml/optimization/bayesian_optimizer.py

from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import json

# Define search space
space = [
    Real(50.0, 200.0, name='gm'),
    Real(0.01, 0.5, name='alpha', prior='log-uniform'),
    Real(0.5, 2.0, name='angular_boost'),
    Real(0.95, 1.0, name='damping'),
    Real(0.5, 2.0, name='bh_mass'),
    Real(0.05, 0.2, name='disk_thickness'),
    Real(0.5, 2.0, name='density_scale'),
    Real(0.0, 1.0, name='siren_intensity'),
    Real(1.0, 50.0, name='timescale'),
]

@use_named_args(space)
def objective(**params):
    """Objective function for Bayesian optimization."""
    results = run_benchmark(params)
    
    if results is None:
        return 100.0  # High penalty for failed runs
    
    # Minimize negative score (= maximize score)
    return -results['summary']['overall_score']

def run_bayesian_optimization(n_calls=100):
    """Run Bayesian optimization."""
    
    print("="*60)
    print("BAYESIAN OPTIMIZATION")
    print("="*60)
    
    result = gp_minimize(
        objective,
        space,
        n_calls=n_calls,
        n_random_starts=20,
        random_state=42,
        verbose=True,
        n_jobs=1  # Sequential (benchmark runs are heavy)
    )
    
    # Extract best parameters
    best_params = {
        space[i].name: result.x[i]
        for i in range(len(space))
    }
    
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)
    print(f"Best score: {-result.fun:.2f}")
    print("Best parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value:.4f}")
    
    # Save convergence plot
    from skopt.plots import plot_convergence
    import matplotlib.pyplot as plt
    plot_convergence(result)
    plt.savefig("bayesian_convergence.png")
    
    return best_params, result

if __name__ == "__main__":
    best_params, result = run_bayesian_optimization(n_calls=100)
```

---

## 5. Active Learning System

### 5.1 Failure Region Detection

```python
# ml/active_learning/failure_detector.py

import numpy as np
import json

class FailureRegionDetector:
    """Identifies regions in parameter space where PINN predictions fail."""
    
    def __init__(self, benchmark_results_path):
        with open(benchmark_results_path, 'r') as f:
            self.results = json.load(f)
    
    def analyze_trajectory(self, trajectory_csv):
        """Analyze particle trajectories to find failure regions."""
        import pandas as pd
        
        df = pd.read_csv(trajectory_csv)
        failures = []
        
        for _, row in df.iterrows():
            r = np.sqrt(row['x']**2 + row['z']**2)
            v = np.sqrt(row['vx']**2 + row['vz']**2)
            
            # Expected Keplerian velocity
            GM = 100.0  # Default
            v_kepler = np.sqrt(GM / r)
            
            # Velocity error
            v_error = abs(v - v_kepler) / v_kepler
            
            if v_error > 0.3:  # >30% error = failure
                failures.append({
                    'position': (row['x'], row['y'], row['z']),
                    'velocity': (row['vx'], row['vy'], row['vz']),
                    'radius': r,
                    'error': v_error,
                    'error_type': 'keplerian_velocity'
                })
            
            # Check for escape/collapse
            if r < 6.0:  # Inside ISCO
                failures.append({
                    'position': (row['x'], row['y'], row['z']),
                    'radius': r,
                    'error_type': 'collapse'
                })
            elif r > 300.0:  # Outside disk
                failures.append({
                    'position': (row['x'], row['y'], row['z']),
                    'radius': r,
                    'error_type': 'escape'
                })
        
        return failures
    
    def cluster_failures(self, failures, eps=20.0):
        """Cluster failure points to find coherent regions."""
        from sklearn.cluster import DBSCAN
        
        if not failures:
            return []
        
        positions = np.array([f['position'] for f in failures])
        clustering = DBSCAN(eps=eps, min_samples=5).fit(positions)
        
        regions = []
        for label in set(clustering.labels_):
            if label == -1:  # Noise
                continue
            
            mask = clustering.labels_ == label
            cluster_positions = positions[mask]
            
            # Compute bounding region
            center = cluster_positions.mean(axis=0)
            radius = np.max(np.linalg.norm(cluster_positions - center, axis=1))
            
            error_types = [failures[i]['error_type'] for i in np.where(mask)[0]]
            dominant_error = max(set(error_types), key=error_types.count)
            
            regions.append({
                'center': center.tolist(),
                'radius': radius,
                'point_count': int(mask.sum()),
                'dominant_error': dominant_error
            })
        
        return regions
    
    def generate_augmentation_samples(self, failure_regions, samples_per_region=1000):
        """Generate training data samples focused on failure regions."""
        
        augmentation_data = []
        
        for region in failure_regions:
            center = np.array(region['center'])
            radius = region['radius']
            
            for _ in range(samples_per_region):
                # Sample position in region
                offset = np.random.randn(3) * radius * 0.5
                pos = center + offset
                
                # Compute correct Keplerian velocity
                r = np.linalg.norm(pos[[0, 2]])  # Cylindrical radius
                GM = 100.0
                v_kepler = np.sqrt(GM / r)
                
                # Velocity direction (tangent to orbit)
                theta = np.arctan2(pos[2], pos[0])
                vx = -v_kepler * np.sin(theta)
                vz = v_kepler * np.cos(theta)
                vy = 0.0  # Disk plane
                
                # Compute ground-truth force
                r_3d = np.linalg.norm(pos)
                F_grav = -GM * pos / (r_3d ** 3)
                
                augmentation_data.append({
                    'position': pos.tolist(),
                    'velocity': [vx, vy, vz],
                    'force': F_grav.tolist(),
                    'source_region': region['dominant_error']
                })
        
        return augmentation_data
```

### 5.2 Training Data Augmentation Pipeline

```python
# ml/active_learning/data_augmentation.py

import numpy as np
import json

class TrainingDataAugmenter:
    """Augments PINN training data based on identified failures."""
    
    def __init__(self, base_training_data_path):
        data = np.load(base_training_data_path)
        self.states = data['states']
        self.forces = data['forces']
    
    def add_failure_region_samples(self, augmentation_data, weight=2.0):
        """Add samples from failure regions with higher weight."""
        
        new_states = []
        new_forces = []
        weights = []
        
        for sample in augmentation_data:
            pos = sample['position']
            vel = sample['velocity']
            force = sample['force']
            
            # Create state vector [x, y, z, vx, vy, vz, t, M_bh, α, H/R]
            state = pos + vel + [0.0, 1.0, 0.1, 0.1]  # Default params
            
            new_states.append(state)
            new_forces.append(force)
            weights.append(weight)  # Higher weight for failure regions
        
        # Combine with original data
        augmented_states = np.vstack([
            self.states,
            np.array(new_states)
        ])
        augmented_forces = np.vstack([
            self.forces,
            np.array(new_forces)
        ])
        
        # Create sample weights (1.0 for original, 2.0 for augmented)
        sample_weights = np.concatenate([
            np.ones(len(self.states)),
            np.array(weights)
        ])
        
        return augmented_states, augmented_forces, sample_weights
    
    def save_augmented_data(self, output_path, augmentation_data, weight=2.0):
        """Save augmented training data."""
        
        states, forces, weights = self.add_failure_region_samples(
            augmentation_data, weight
        )
        
        np.savez(
            output_path,
            states=states.astype(np.float32),
            forces=forces.astype(np.float32),
            weights=weights.astype(np.float32)
        )
        
        print(f"Saved augmented data:")
        print(f"  Original samples: {len(self.states)}")
        print(f"  Augmented samples: {len(augmentation_data)}")
        print(f"  Total samples: {len(states)}")
        print(f"  Output: {output_path}")
```

### 5.3 Active Learning Loop

```python
# ml/active_learning/active_learning_loop.py

import os
import subprocess
from failure_detector import FailureRegionDetector
from data_augmentation import TrainingDataAugmenter

def run_active_learning_iteration(
    iteration,
    base_model_path,
    training_data_path,
    output_dir
):
    """Run one iteration of active learning."""
    
    print(f"\n{'='*60}")
    print(f"ACTIVE LEARNING - ITERATION {iteration}")
    print('='*60)
    
    # Step 1: Run benchmark with current model
    print("\n[Step 1] Running benchmark...")
    benchmark_output = os.path.join(output_dir, f"benchmark_iter{iteration}.json")
    trajectory_output = os.path.join(output_dir, f"trajectory_iter{iteration}.csv")
    
    subprocess.run([
        "build/bin/Debug/PlasmaDX-Clean.exe",
        "--benchmark",
        "--pinn", base_model_path,
        "--frames", "1000",
        "--output", benchmark_output,
        "--export-trajectory", trajectory_output,
        "--timescale", "10"
    ])
    
    # Step 2: Detect failure regions
    print("\n[Step 2] Detecting failure regions...")
    detector = FailureRegionDetector(benchmark_output)
    failures = detector.analyze_trajectory(trajectory_output)
    failure_regions = detector.cluster_failures(failures)
    
    print(f"  Found {len(failures)} failure points")
    print(f"  Clustered into {len(failure_regions)} regions")
    
    for i, region in enumerate(failure_regions):
        print(f"    Region {i}: {region['dominant_error']} at r={np.linalg.norm(region['center']):.1f}")
    
    # Step 3: Generate augmentation data
    print("\n[Step 3] Generating augmentation samples...")
    augmentation_data = detector.generate_augmentation_samples(
        failure_regions,
        samples_per_region=2000
    )
    print(f"  Generated {len(augmentation_data)} augmentation samples")
    
    # Step 4: Augment training data
    print("\n[Step 4] Augmenting training data...")
    augmenter = TrainingDataAugmenter(training_data_path)
    augmented_data_path = os.path.join(output_dir, f"training_data_iter{iteration}.npz")
    augmenter.save_augmented_data(augmented_data_path, augmentation_data)
    
    # Step 5: Retrain model
    print("\n[Step 5] Retraining PINN model...")
    new_model_path = os.path.join(output_dir, f"pinn_iter{iteration}.onnx")
    subprocess.run([
        "ml/venv/bin/python3",
        "ml/pinn_v4_with_turbulence.py",
        "--training-data", augmented_data_path,
        "--output", new_model_path,
        "--epochs", "100",
        "--weighted-loss"  # Use sample weights
    ])
    
    # Step 6: Evaluate improvement
    print("\n[Step 6] Evaluating improvement...")
    eval_output = os.path.join(output_dir, f"eval_iter{iteration}.json")
    subprocess.run([
        "build/bin/Debug/PlasmaDX-Clean.exe",
        "--benchmark",
        "--pinn", new_model_path,
        "--frames", "1000",
        "--output", eval_output,
        "--timescale", "10"
    ])
    
    with open(eval_output, 'r') as f:
        eval_results = json.load(f)
    
    new_score = eval_results['summary']['overall_score']
    print(f"\n  New model score: {new_score:.2f}")
    
    return new_model_path, new_score, failure_regions

def run_full_active_learning(
    initial_model_path="ml/models/pinn_v4_turbulence_robust.onnx",
    training_data_path="ml/training_data/pinn_v4_total_forces.npz",
    output_dir="ml/active_learning_output",
    max_iterations=10,
    score_threshold=85.0
):
    """Run full active learning pipeline until convergence."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    current_model = initial_model_path
    best_score = 0.0
    
    for iteration in range(max_iterations):
        new_model, score, regions = run_active_learning_iteration(
            iteration,
            current_model,
            training_data_path,
            output_dir
        )
        
        if score > best_score:
            best_score = score
            best_model = new_model
        
        if score >= score_threshold:
            print(f"\n{'='*60}")
            print(f"TARGET SCORE REACHED: {score:.2f} >= {score_threshold}")
            print(f"Best model: {best_model}")
            print('='*60)
            break
        
        if len(regions) == 0:
            print(f"\n{'='*60}")
            print("NO MORE FAILURE REGIONS - CONVERGENCE")
            print('='*60)
            break
        
        current_model = new_model
        training_data_path = os.path.join(output_dir, f"training_data_iter{iteration}.npz")
    
    return best_model, best_score

if __name__ == "__main__":
    run_full_active_learning()
```

---

## 6. Physics-Constrained Turbulence Model

### 6.1 Constrained SIREN Architecture

```python
# ml/vortex_field/constrained_siren.py

import torch
import torch.nn as nn
import numpy as np

class SineLayer(nn.Module):
    """SIREN sine activation layer."""
    
    def __init__(self, in_features, out_features, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.linear.in_features,
                                             1 / self.linear.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.linear.in_features) / self.omega_0,
                     np.sqrt(6 / self.linear.in_features) / self.omega_0
                )
    
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class ConstrainedTurbulenceSIREN(nn.Module):
    """
    SIREN model for turbulence that preserves angular momentum.
    
    Key constraint: The turbulent forces should not change the
    total angular momentum of the system (no net torque).
    """
    
    def __init__(self, hidden_dim=128, num_layers=4):
        super().__init__()
        
        # Input: [x, y, z, t, seed] = 5D
        layers = [SineLayer(5, hidden_dim, is_first=True)]
        
        for _ in range(num_layers - 1):
            layers.append(SineLayer(hidden_dim, hidden_dim))
        
        # Output: raw vorticity [ω_x, ω_y, ω_z]
        layers.append(nn.Linear(hidden_dim, 3))
        
        self.network = nn.Sequential(*layers)
        
        # Learnable constraint strength
        self.constraint_strength = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, positions, velocities, time, seed):
        """
        Forward pass with angular momentum conservation constraint.
        
        Args:
            positions: [N, 3] particle positions
            velocities: [N, 3] particle velocities
            time: scalar simulation time
            seed: scalar random seed
        
        Returns:
            omega_constrained: [N, 3] constrained vorticity field
        """
        N = positions.shape[0]
        
        # Prepare input [x, y, z, t, seed]
        t_tensor = torch.full((N, 1), time, device=positions.device)
        seed_tensor = torch.full((N, 1), seed, device=positions.device)
        x = torch.cat([positions, t_tensor, seed_tensor], dim=1)
        
        # Raw vorticity prediction
        omega_raw = self.network(x)
        
        # === ANGULAR MOMENTUM CONSERVATION CONSTRAINT ===
        # F_turb = v × ω (turbulent force from vorticity)
        # L = r × F (angular momentum contribution)
        # We want: Σ L = 0 (total angular momentum unchanged)
        
        # Compute turbulent forces
        F_turb = torch.cross(velocities, omega_raw, dim=1)
        
        # Compute angular momentum contribution per particle
        L_per_particle = torch.cross(positions, F_turb, dim=1)
        
        # Total angular momentum change (should be zero)
        L_total = L_per_particle.sum(dim=0)  # [3]
        
        # Project out the component that causes net torque
        # This is done by adjusting ω to cancel the net L
        
        # Compute how much each particle contributes to L_total
        # and reduce ω proportionally
        L_magnitude = torch.norm(L_total)
        
        if L_magnitude > 1e-6:
            # Correction factor
            correction = L_total / (N * L_magnitude + 1e-6)
            
            # Apply correction to omega
            # This is a simplified projection - more sophisticated
            # methods could use Lagrange multipliers
            omega_corrected = omega_raw - self.constraint_strength * correction.unsqueeze(0)
        else:
            omega_corrected = omega_raw
        
        return omega_corrected
    
    def compute_turbulent_forces(self, positions, velocities, time, seed):
        """Compute constrained turbulent forces."""
        omega = self.forward(positions, velocities, time, seed)
        F_turb = torch.cross(velocities, omega, dim=1)
        return F_turb


class PhysicsInformedTurbulenceLoss(nn.Module):
    """
    Loss function that enforces physics constraints on turbulence.
    """
    
    def __init__(
        self,
        lambda_conservation=10.0,    # Angular momentum conservation
        lambda_vortex=0.1,           # Vortex structure reward
        lambda_drift=5.0,            # Radial drift penalty
        lambda_smoothness=0.01       # Temporal smoothness
    ):
        super().__init__()
        self.lambda_conservation = lambda_conservation
        self.lambda_vortex = lambda_vortex
        self.lambda_drift = lambda_drift
        self.lambda_smoothness = lambda_smoothness
    
    def forward(self, omega, positions, velocities, omega_prev=None):
        """
        Compute physics-informed loss.
        
        Args:
            omega: [N, 3] predicted vorticity
            positions: [N, 3] particle positions
            velocities: [N, 3] particle velocities
            omega_prev: [N, 3] vorticity from previous timestep (optional)
        """
        N = omega.shape[0]
        
        # Turbulent forces
        F_turb = torch.cross(velocities, omega, dim=1)
        
        # === Loss 1: Angular Momentum Conservation ===
        L_per_particle = torch.cross(positions, F_turb, dim=1)
        L_total = L_per_particle.sum(dim=0)
        L_conservation = torch.norm(L_total) / N
        
        # === Loss 2: No Systematic Radial Drift ===
        # Radial component of force should average to zero
        r = torch.norm(positions[:, [0, 2]], dim=1, keepdim=True) + 1e-6
        r_hat = positions[:, [0, 2]] / r  # Radial direction in XZ plane
        F_radial = (F_turb[:, [0, 2]] * r_hat).sum(dim=1)
        drift_penalty = torch.mean(F_radial) ** 2
        
        # === Loss 3: Vortex Structure (Reward) ===
        # High variance in vorticity = interesting turbulence
        omega_var = torch.var(omega, dim=0).sum()
        vortex_reward = -omega_var  # Negative because we minimize loss
        
        # === Loss 4: Temporal Smoothness ===
        if omega_prev is not None:
            smoothness = torch.mean((omega - omega_prev) ** 2)
        else:
            smoothness = torch.tensor(0.0, device=omega.device)
        
        # Combined loss
        total_loss = (
            self.lambda_conservation * L_conservation +
            self.lambda_drift * drift_penalty +
            self.lambda_vortex * vortex_reward +
            self.lambda_smoothness * smoothness
        )
        
        return total_loss, {
            'conservation': L_conservation.item(),
            'drift': drift_penalty.item(),
            'vortex_var': omega_var.item(),
            'smoothness': smoothness.item()
        }
```

### 6.2 Constrained SIREN Training Script

```python
# ml/vortex_field/train_constrained_siren.py

import torch
import torch.optim as optim
import numpy as np
from constrained_siren import ConstrainedTurbulenceSIREN, PhysicsInformedTurbulenceLoss

def generate_orbital_training_data(num_samples=100000, GM=100.0):
    """Generate training data with particles on Keplerian orbits."""
    
    data = []
    
    for _ in range(num_samples):
        # Random orbital radius
        r = np.random.uniform(20.0, 250.0)
        
        # Random azimuthal angle
        theta = np.random.uniform(0, 2 * np.pi)
        
        # Position in disk plane
        x = r * np.cos(theta)
        z = r * np.sin(theta)
        y = np.random.normal(0, 0.1 * r)  # Disk thickness
        
        # Keplerian velocity (tangent to orbit)
        v_kepler = np.sqrt(GM / r)
        vx = -v_kepler * np.sin(theta)
        vz = v_kepler * np.cos(theta)
        vy = 0.0
        
        # Time and seed
        t = np.random.uniform(0, 100)
        seed = np.random.uniform(0, 1000)
        
        # Target vorticity: localized vortex structures
        # Create eddies at various scales
        omega_target = generate_vortex_field(x, y, z, t, seed)
        
        data.append({
            'position': [x, y, z],
            'velocity': [vx, vy, vz],
            'time': t,
            'seed': seed,
            'omega_target': omega_target
        })
    
    return data

def generate_vortex_field(x, y, z, t, seed):
    """Generate ground-truth vorticity with vortex structures."""
    
    # Multi-scale vortices using Perlin-like noise
    np.random.seed(int(seed) % 10000)
    
    # Large-scale vortex
    r = np.sqrt(x**2 + z**2)
    omega_large = np.array([
        0.1 * np.sin(r / 50 + t * 0.1) * z / (r + 1),
        0.05 * np.cos(r / 30 + t * 0.05),
        -0.1 * np.sin(r / 50 + t * 0.1) * x / (r + 1)
    ])
    
    # Small-scale turbulence
    omega_small = np.array([
        0.05 * np.sin(x / 10 + t) * np.cos(z / 10),
        0.02 * np.sin(y / 5 + t * 2),
        0.05 * np.cos(x / 10 + t) * np.sin(z / 10)
    ])
    
    # Combine
    omega = omega_large + omega_small
    
    # Ensure zero divergence (approximately)
    # div(ω) ≈ 0 for physical vorticity fields
    
    return omega

def train_constrained_siren(
    epochs=200,
    batch_size=1024,
    lr=0.0001,
    output_path="ml/models/constrained_siren.onnx"
):
    """Train the constrained SIREN model."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    
    # Create model and loss
    model = ConstrainedTurbulenceSIREN(hidden_dim=128, num_layers=4).to(device)
    loss_fn = PhysicsInformedTurbulenceLoss(
        lambda_conservation=10.0,
        lambda_vortex=0.1,
        lambda_drift=5.0,
        lambda_smoothness=0.01
    )
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    # Generate training data
    print("Generating training data...")
    training_data = generate_orbital_training_data(num_samples=100000)
    
    # Convert to tensors
    positions = torch.tensor(
        [d['position'] for d in training_data],
        dtype=torch.float32, device=device
    )
    velocities = torch.tensor(
        [d['velocity'] for d in training_data],
        dtype=torch.float32, device=device
    )
    times = torch.tensor(
        [d['time'] for d in training_data],
        dtype=torch.float32, device=device
    )
    seeds = torch.tensor(
        [d['seed'] for d in training_data],
        dtype=torch.float32, device=device
    )
    omega_targets = torch.tensor(
        [d['omega_target'] for d in training_data],
        dtype=torch.float32, device=device
    )
    
    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    
    for epoch in range(epochs):
        total_loss = 0
        total_conservation = 0
        total_vortex_var = 0
        num_batches = 0
        
        # Shuffle
        perm = torch.randperm(len(training_data))
        
        for i in range(0, len(training_data), batch_size):
            idx = perm[i:i+batch_size]
            
            pos_batch = positions[idx]
            vel_batch = velocities[idx]
            t_batch = times[idx].mean().item()
            seed_batch = seeds[idx].mean().item()
            omega_target_batch = omega_targets[idx]
            
            optimizer.zero_grad()
            
            # Forward pass
            omega_pred = model(pos_batch, vel_batch, t_batch, seed_batch)
            
            # Physics loss
            physics_loss, metrics = loss_fn(omega_pred, pos_batch, vel_batch)
            
            # Supervised loss (match target vorticity)
            supervised_loss = torch.mean((omega_pred - omega_target_batch) ** 2)
            
            # Combined loss
            loss = physics_loss + 0.5 * supervised_loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_conservation += metrics['conservation']
            total_vortex_var += metrics['vortex_var']
            num_batches += 1
        
        scheduler.step()
        
        avg_loss = total_loss / num_batches
        avg_conservation = total_conservation / num_batches
        avg_vortex_var = total_vortex_var / num_batches
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Loss: {avg_loss:.6f}")
            print(f"  Conservation: {avg_conservation:.6f}")
            print(f"  Vortex Var: {avg_vortex_var:.4f}")
            print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
    
    # Export to ONNX
    print(f"\nExporting to ONNX: {output_path}")
    model.eval()
    
    # Create dummy inputs for export
    dummy_pos = torch.randn(100, 3, device=device)
    dummy_vel = torch.randn(100, 3, device=device)
    dummy_time = torch.tensor(0.0, device=device)
    dummy_seed = torch.tensor(0.0, device=device)
    
    # Export with dynamic batch size
    torch.onnx.export(
        model,
        (dummy_pos, dummy_vel, dummy_time, dummy_seed),
        output_path,
        input_names=['positions', 'velocities', 'time', 'seed'],
        output_names=['vorticity'],
        dynamic_axes={
            'positions': {0: 'batch'},
            'velocities': {0: 'batch'},
            'vorticity': {0: 'batch'}
        },
        opset_version=17
    )
    
    print("Training complete!")
    return model

if __name__ == "__main__":
    train_constrained_siren()
```

---

## 7. Vortex Detection System

### 7.1 C++ Vortex Detection

```cpp
// In ParticleSystem.h - Add VortexMetrics struct

struct VortexMetrics {
    uint32_t vortexCount = 0;           // Number of detected eddies
    float avgVortexStrength = 0.0f;     // Mean |∇×v|
    float vortexCoherence = 0.0f;       // Spatial correlation
    float angularVelocityVariance = 0.0f;  // Differential rotation indicator
    std::vector<DirectX::XMFLOAT3> vortexCenters;  // Locations of detected vortices
};

VortexMetrics DetectVortices() const;
```

```cpp
// In ParticleSystem.cpp

ParticleSystem::VortexMetrics ParticleSystem::DetectVortices() const {
    VortexMetrics metrics;
    
    if (!m_particlesOnCPU || m_cpuPositions.empty() || m_cpuVelocities.empty()) {
        return metrics;
    }
    
    const uint32_t count = m_activeParticleCount;
    const float GM = m_pinnGM;
    
    // Step 1: Compute angular velocity for each particle
    std::vector<float> angularVelocities(count);
    float sumOmega = 0.0f;
    
    for (uint32_t i = 0; i < count; i++) {
        float r = sqrtf(m_cpuPositions[i].x * m_cpuPositions[i].x +
                       m_cpuPositions[i].z * m_cpuPositions[i].z);
        if (r > 1e-6f) {
            // Angular velocity: ω = v_tangential / r
            float v_tang = sqrtf(m_cpuVelocities[i].x * m_cpuVelocities[i].x +
                                m_cpuVelocities[i].z * m_cpuVelocities[i].z);
            angularVelocities[i] = v_tang / r;
            sumOmega += angularVelocities[i];
        }
    }
    
    float meanOmega = sumOmega / count;
    
    // Step 2: Compute variance of angular velocity (differential rotation)
    float varOmega = 0.0f;
    for (uint32_t i = 0; i < count; i++) {
        float diff = angularVelocities[i] - meanOmega;
        varOmega += diff * diff;
    }
    metrics.angularVelocityVariance = varOmega / count;
    
    // Step 3: Detect vortex regions using velocity curl estimation
    // Grid-based approach for efficiency
    const int GRID_SIZE = 20;
    const float CELL_SIZE = (OUTER_DISK_RADIUS * 2) / GRID_SIZE;
    
    struct GridCell {
        float vorticity = 0.0f;
        int particleCount = 0;
        float avgX = 0.0f, avgZ = 0.0f;
    };
    std::vector<std::vector<GridCell>> grid(GRID_SIZE, std::vector<GridCell>(GRID_SIZE));
    
    // Accumulate velocities in grid
    for (uint32_t i = 0; i < count; i++) {
        int gx = static_cast<int>((m_cpuPositions[i].x + OUTER_DISK_RADIUS) / CELL_SIZE);
        int gz = static_cast<int>((m_cpuPositions[i].z + OUTER_DISK_RADIUS) / CELL_SIZE);
        
        gx = std::clamp(gx, 0, GRID_SIZE - 1);
        gz = std::clamp(gz, 0, GRID_SIZE - 1);
        
        grid[gx][gz].particleCount++;
        grid[gx][gz].avgX += m_cpuVelocities[i].x;
        grid[gx][gz].avgZ += m_cpuVelocities[i].z;
    }
    
    // Compute average velocities and estimate curl
    float totalVorticity = 0.0f;
    int vortexCells = 0;
    
    for (int gx = 1; gx < GRID_SIZE - 1; gx++) {
        for (int gz = 1; gz < GRID_SIZE - 1; gz++) {
            if (grid[gx][gz].particleCount > 0) {
                grid[gx][gz].avgX /= grid[gx][gz].particleCount;
                grid[gx][gz].avgZ /= grid[gx][gz].particleCount;
            }
            
            // Estimate curl using finite differences
            // curl_y = ∂vz/∂x - ∂vx/∂z
            if (grid[gx+1][gz].particleCount > 0 && grid[gx-1][gz].particleCount > 0 &&
                grid[gx][gz+1].particleCount > 0 && grid[gx][gz-1].particleCount > 0) {
                
                float dvz_dx = (grid[gx+1][gz].avgZ - grid[gx-1][gz].avgZ) / (2 * CELL_SIZE);
                float dvx_dz = (grid[gx][gz+1].avgX - grid[gx][gz-1].avgX) / (2 * CELL_SIZE);
                
                float vorticity = dvz_dx - dvx_dz;
                grid[gx][gz].vorticity = vorticity;
                totalVorticity += std::abs(vorticity);
                
                // Detect vortex (high vorticity)
                if (std::abs(vorticity) > 0.01f) {
                    vortexCells++;
                    
                    // Record vortex center
                    float cx = (gx + 0.5f) * CELL_SIZE - OUTER_DISK_RADIUS;
                    float cz = (gz + 0.5f) * CELL_SIZE - OUTER_DISK_RADIUS;
                    metrics.vortexCenters.push_back({cx, 0.0f, cz});
                }
            }
        }
    }
    
    // Cluster nearby vortex centers
    // (Simplified - just count distinct regions)
    metrics.vortexCount = std::min(static_cast<uint32_t>(vortexCells / 4), 20u);  // Rough estimate
    metrics.avgVortexStrength = totalVorticity / ((GRID_SIZE - 2) * (GRID_SIZE - 2));
    
    // Compute vortex coherence (spatial correlation of vorticity)
    float coherenceSum = 0.0f;
    int coherencePairs = 0;
    for (int gx = 1; gx < GRID_SIZE - 2; gx++) {
        for (int gz = 1; gz < GRID_SIZE - 2; gz++) {
            coherenceSum += grid[gx][gz].vorticity * grid[gx+1][gz].vorticity;
            coherenceSum += grid[gx][gz].vorticity * grid[gx][gz+1].vorticity;
            coherencePairs += 2;
        }
    }
    metrics.vortexCoherence = coherenceSum / (coherencePairs * metrics.avgVortexStrength + 1e-6f);
    
    return metrics;
}
```

### 7.2 Benchmark Vortex Metrics Integration

```cpp
// In BenchmarkMetrics.h - Add to PhysicsSnapshot

struct PhysicsSnapshot {
    // ... existing fields ...
    
    // Vortex/Turbulence metrics
    uint32_t vortexCount = 0;
    float avgVortexStrength = 0.0f;
    float vortexCoherence = 0.0f;
    float angularVelocityVariance = 0.0f;
};

// In BenchmarkRunner.cpp - Capture vortex metrics

PhysicsSnapshot BenchmarkRunner::CaptureSnapshot() {
    auto psSnap = m_particleSystem->CapturePhysicsSnapshot();
    
    // ... existing conversion ...
    
    // Add vortex metrics
    auto vortexMetrics = m_particleSystem->DetectVortices();
    snap.vortexCount = vortexMetrics.vortexCount;
    snap.avgVortexStrength = vortexMetrics.avgVortexStrength;
    snap.vortexCoherence = vortexMetrics.vortexCoherence;
    snap.angularVelocityVariance = vortexMetrics.angularVelocityVariance;
    
    return snap;
}
```

---

## 8. AI Vision Quality Assessment

### 8.1 Headless Frame Rendering

```cpp
// In ParticleSystem.h

// Render a single frame to texture (no window required)
bool RenderFrameToTexture(
    ID3D12Resource* outputTexture,
    uint32_t width,
    uint32_t height,
    const DirectX::XMFLOAT4X4& viewMatrix,
    const DirectX::XMFLOAT4X4& projMatrix
);

// Save texture to PNG file
bool SaveTextureToPNG(
    ID3D12Resource* texture,
    uint32_t width,
    uint32_t height,
    const std::string& path
);
```

```cpp
// In BenchmarkRunner.cpp

bool BenchmarkRunner::CaptureFrameForVision(const std::string& outputPath) {
    // Create render target texture
    D3D12_RESOURCE_DESC texDesc = {};
    texDesc.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
    texDesc.Width = 1920;
    texDesc.Height = 1080;
    texDesc.DepthOrArraySize = 1;
    texDesc.MipLevels = 1;
    texDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    texDesc.SampleDesc.Count = 1;
    texDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET;
    
    // ... create texture, render, save ...
    
    return m_particleSystem->SaveTextureToPNG(renderTarget, 1920, 1080, outputPath);
}
```

### 8.2 Vision Assessment Integration

```python
# ml/quality_assessment/vision_assessor.py

import base64
import json
from pathlib import Path

class VisionQualityAssessor:
    """Uses AI vision to assess simulation quality."""
    
    def __init__(self, api_key=None):
        # Could use Claude, GPT-4V, or local model
        self.api_key = api_key
    
    def assess_frame(self, frame_path: str) -> dict:
        """Assess a single frame for visual quality."""
        
        # Read image
        with open(frame_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        prompt = """
        Analyze this accretion disk simulation frame for visual and physical quality.
        
        Evaluate on these criteria (score 1-10 each):
        
        1. ORBITAL STRUCTURE
           - Do particles form a clear disk shape?
           - Is there visible rotation/orbital motion?
           - Are there density gradients (denser toward center)?
        
        2. TURBULENCE QUALITY
           - Are there visible vortex/eddy structures?
           - Is motion chaotic but not random?
           - Do eddies have coherent swirling patterns?
        
        3. PHYSICAL REALISM
           - Does it look like a real accretion disk?
           - Is there visible shearing (differential rotation)?
           - Are particles maintaining orbital paths?
        
        4. COLOR/EMISSION
           - Is there a temperature gradient (hotter = blue/white near center)?
           - Does emission vary with density?
           - Is the color scheme astrophysically plausible?
        
        5. OVERALL AESTHETICS
           - Is it visually appealing?
           - Does it look "alive" with dynamics?
           - Would this work in a scientific visualization?
        
        Respond in JSON format:
        {
            "orbital_structure": {"score": X, "notes": "..."},
            "turbulence_quality": {"score": X, "notes": "..."},
            "physical_realism": {"score": X, "notes": "..."},
            "color_emission": {"score": X, "notes": "..."},
            "aesthetics": {"score": X, "notes": "..."},
            "overall_score": X,
            "key_issues": ["...", "..."],
            "recommendations": ["...", "..."]
        }
        """
        
        # Call vision API (placeholder - implement based on your API)
        response = self.call_vision_api(image_data, prompt)
        
        return json.loads(response)
    
    def assess_sequence(self, frame_paths: list, interval: int = 10) -> dict:
        """Assess a sequence of frames for temporal quality."""
        
        assessments = []
        for i, path in enumerate(frame_paths[::interval]):
            assessment = self.assess_frame(path)
            assessment['frame_index'] = i * interval
            assessments.append(assessment)
        
        # Compute temporal statistics
        avg_scores = {
            'orbital_structure': np.mean([a['orbital_structure']['score'] for a in assessments]),
            'turbulence_quality': np.mean([a['turbulence_quality']['score'] for a in assessments]),
            'physical_realism': np.mean([a['physical_realism']['score'] for a in assessments]),
            'color_emission': np.mean([a['color_emission']['score'] for a in assessments]),
            'aesthetics': np.mean([a['aesthetics']['score'] for a in assessments]),
            'overall': np.mean([a['overall_score'] for a in assessments])
        }
        
        # Temporal consistency (low variance = stable quality)
        temporal_consistency = 1.0 - np.std([a['overall_score'] for a in assessments]) / 10.0
        
        return {
            'frame_assessments': assessments,
            'average_scores': avg_scores,
            'temporal_consistency': temporal_consistency,
            'combined_score': avg_scores['overall'] * 0.8 + temporal_consistency * 20
        }
```

### 8.3 Combined Scoring (Numerical + Vision)

```python
# ml/quality_assessment/combined_scorer.py

class CombinedQualityScorer:
    """Combines numerical metrics with AI vision assessment."""
    
    def __init__(self, vision_weight=0.3, numerical_weight=0.7):
        self.vision_weight = vision_weight
        self.numerical_weight = numerical_weight
        self.vision_assessor = VisionQualityAssessor()
    
    def score(self, benchmark_results: dict, frame_path: str = None) -> dict:
        """Compute combined quality score."""
        
        # Numerical score (from benchmark)
        numerical_score = benchmark_results['summary']['overall_score']
        
        # Vision score (if frame available)
        if frame_path and Path(frame_path).exists():
            vision_result = self.vision_assessor.assess_frame(frame_path)
            vision_score = vision_result['overall_score'] * 10  # Scale to 0-100
        else:
            vision_score = numerical_score  # Fall back to numerical
            self.vision_weight = 0
            self.numerical_weight = 1.0
        
        # Combined score
        combined = (
            self.numerical_weight * numerical_score +
            self.vision_weight * vision_score
        )
        
        return {
            'numerical_score': numerical_score,
            'vision_score': vision_score,
            'combined_score': combined,
            'breakdown': {
                'stability': benchmark_results['summary']['stability_score'],
                'accuracy': benchmark_results['summary']['accuracy_score'],
                'performance': benchmark_results['summary']['performance_score'],
                'visual_numerical': benchmark_results['summary']['visual_score'],
                'visual_ai': vision_score if frame_path else None
            }
        }
```

---

## 9. Complete Optimization Pipeline

### 9.1 Master Orchestrator

```python
# ml/optimization/master_pipeline.py

import os
import json
import argparse
from datetime import datetime

from genetic_optimizer import run_evolution
from bayesian_optimizer import run_bayesian_optimization
from active_learning_loop import run_full_active_learning
from quality_assessment.combined_scorer import CombinedQualityScorer

class GalaxyPhysicsOptimizer:
    """
    Master orchestrator for the complete optimization pipeline.
    
    Combines:
    - Genetic algorithm for parameter exploration
    - Bayesian optimization for fine-tuning
    - Active learning for model improvement
    - Vision assessment for quality validation
    """
    
    def __init__(self, output_dir="optimization_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.scorer = CombinedQualityScorer()
        
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(output_dir, f"run_{self.run_id}")
        os.makedirs(self.run_dir, exist_ok=True)
    
    def run_full_pipeline(
        self,
        initial_model="ml/models/pinn_v4_turbulence_robust.onnx",
        target_score=90.0,
        max_iterations=5
    ):
        """Run the complete optimization pipeline."""
        
        print("="*70)
        print("GALAXY PHYSICS MODEL CREATOR - FULL OPTIMIZATION PIPELINE")
        print("="*70)
        print(f"Run ID: {self.run_id}")
        print(f"Target Score: {target_score}")
        print(f"Output: {self.run_dir}")
        print("="*70)
        
        current_model = initial_model
        current_params = None
        best_score = 0.0
        
        for iteration in range(max_iterations):
            print(f"\n{'='*70}")
            print(f"ITERATION {iteration + 1}/{max_iterations}")
            print("="*70)
            
            # Phase 1: Genetic Algorithm - Find good parameter region
            print("\n[Phase 1] Genetic Algorithm Exploration...")
            ga_params, ga_log = run_evolution(
                population_size=20,
                generations=30,
                model_path=current_model
            )
            
            # Phase 2: Bayesian Optimization - Fine-tune parameters
            print("\n[Phase 2] Bayesian Optimization Fine-Tuning...")
            bo_params, bo_result = run_bayesian_optimization(
                n_calls=50,
                initial_params=ga_params,
                model_path=current_model
            )
            
            # Phase 3: Benchmark with best parameters
            print("\n[Phase 3] Full Benchmark...")
            benchmark_result = self.run_full_benchmark(
                bo_params,
                model_path=current_model,
                output_prefix=f"iter{iteration}"
            )
            
            # Phase 4: Vision Assessment
            print("\n[Phase 4] Vision Quality Assessment...")
            frame_path = os.path.join(self.run_dir, f"frame_iter{iteration}.png")
            combined_score = self.scorer.score(benchmark_result, frame_path)
            
            print(f"\nIteration {iteration + 1} Results:")
            print(f"  Numerical Score: {combined_score['numerical_score']:.2f}")
            print(f"  Vision Score: {combined_score['vision_score']:.2f}")
            print(f"  Combined Score: {combined_score['combined_score']:.2f}")
            
            if combined_score['combined_score'] > best_score:
                best_score = combined_score['combined_score']
                best_params = bo_params
                best_model = current_model
            
            # Check if target reached
            if combined_score['combined_score'] >= target_score:
                print(f"\n{'='*70}")
                print(f"TARGET SCORE REACHED: {combined_score['combined_score']:.2f}")
                print("="*70)
                break
            
            # Phase 5: Active Learning - Improve model
            print("\n[Phase 5] Active Learning Model Improvement...")
            new_model, al_score = self.run_active_learning_step(
                current_model,
                benchmark_result,
                iteration
            )
            
            if al_score > combined_score['combined_score']:
                current_model = new_model
                print(f"  Model improved: {al_score:.2f} > {combined_score['combined_score']:.2f}")
            else:
                print(f"  No improvement from active learning")
        
        # Save final results
        self.save_final_results(best_model, best_params, best_score)
        
        return best_model, best_params, best_score
    
    def run_full_benchmark(self, params, model_path, output_prefix):
        """Run comprehensive benchmark with given parameters."""
        
        output_path = os.path.join(self.run_dir, f"{output_prefix}_benchmark.json")
        
        cmd = [
            "build/bin/Debug/PlasmaDX-Clean.exe",
            "--benchmark",
            "--pinn", model_path,
            "--frames", "1000",
            "--output", output_path,
            "--capture-frame", os.path.join(self.run_dir, f"{output_prefix}_frame.png"),
        ]
        
        # Add all parameters
        for key, value in params.items():
            cmd.extend([f"--{key.replace('_', '-')}", str(value)])
        
        subprocess.run(cmd)
        
        with open(output_path, 'r') as f:
            return json.load(f)
    
    def run_active_learning_step(self, model_path, benchmark_result, iteration):
        """Run one step of active learning."""
        
        al_output_dir = os.path.join(self.run_dir, f"active_learning_iter{iteration}")
        
        return run_full_active_learning(
            initial_model_path=model_path,
            training_data_path="ml/training_data/pinn_v4_total_forces.npz",
            output_dir=al_output_dir,
            max_iterations=3,
            score_threshold=benchmark_result['summary']['overall_score'] + 5
        )
    
    def save_final_results(self, model_path, params, score):
        """Save final optimization results."""
        
        results = {
            'run_id': self.run_id,
            'best_model': model_path,
            'best_params': params,
            'best_score': score,
            'timestamp': datetime.now().isoformat()
        }
        
        output_path = os.path.join(self.run_dir, "final_results.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Also save as preset
        preset_path = os.path.join(self.run_dir, "optimized_preset.json")
        preset = {
            'preset_name': f"optimized_{self.run_id}",
            'description': f"Auto-optimized preset (score: {score:.2f})",
            'physics': params,
            'model': model_path
        }
        with open(preset_path, 'w') as f:
            json.dump(preset, f, indent=2)
        
        print(f"\nFinal results saved to: {output_path}")
        print(f"Preset saved to: {preset_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-score", type=float, default=90.0)
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="optimization_results")
    args = parser.parse_args()
    
    optimizer = GalaxyPhysicsOptimizer(output_dir=args.output_dir)
    best_model, best_params, best_score = optimizer.run_full_pipeline(
        target_score=args.target_score,
        max_iterations=args.max_iterations
    )
```

---

## 10. Summary

This specification defines a complete system for:

1. **Runtime Parameter Control** - All physics parameters accessible via CLI and ImGui
2. **Genetic Algorithm Optimization** - Explore parameter space efficiently
3. **Bayesian Optimization** - Fine-tune promising regions
4. **Active Learning** - Automatically improve PINN training data
5. **Physics-Constrained Turbulence** - SIREN that preserves angular momentum
6. **Vortex Detection** - Quantify turbulence quality
7. **AI Vision Assessment** - Evaluate visual quality
8. **Master Pipeline** - Orchestrate all components

The result is an autonomous system capable of discovering optimal physics parameters and improving the underlying ML models - the **Ultimate Galaxy Physics Model Creator**.

