# ML-Driven Physics Optimization

Genetic algorithm optimizer for PlasmaDX accretion disk physics parameters.

## Quick Start

### 1. Install Dependencies

```bash
cd ml
source venv/bin/activate  # Or: ./venv/Scripts/activate on Windows
pip install -r requirements_optimization.txt
```

### 2. Run Optimization (Small Test)

```bash
cd ml/optimization
python genetic_optimizer.py
```

This runs a small test (10 individuals, 5 generations) to verify everything works.

### 3. View Results

```bash
python visualize_results.py
```

Generates:
- `results/convergence.png` - Fitness over generations
- `results/parameter_distribution.png` - Top parameters
- Console summary with top 3 individuals

## Configuration

Edit `genetic_optimizer.py` main() function to adjust:

```python
optimizer = GeneticOptimizer(
    executable_path="../build/bin/Debug/PlasmaDX-Clean.exe",
    pinn_model="v4",           # v1, v2, v3, or v4
    population_size=20,        # Individuals per generation (try 20-50)
    generations=50,            # Evolution iterations (try 30-100)
    mutation_prob=0.2,         # Mutation probability (0.1-0.3)
    crossover_prob=0.7         # Crossover probability (0.6-0.8)
)
```

## Parameter Space

Optimizes 12 physics parameters:

**Gravitational:**
- `gm` (50-200): Gravitational parameter
- `bh_mass` (0.1-10): Black hole mass multiplier

**Viscosity & Dynamics:**
- `alpha` (0.01-0.5): Shakura-Sunyaev viscosity
- `damping` (0.9-1.0): Velocity damping
- `angular_boost` (0.5-3.0): Angular momentum multiplier

**Disk Geometry:**
- `disk_thickness` (0.05-0.3): H/R ratio
- `inner_radius` (3-10): ISCO
- `outer_radius` (200-500): Disk edge

**Material:**
- `density_scale` (0.5-3.0): Density multiplier

**Safety:**
- `force_clamp` (5-50): Max force magnitude
- `velocity_clamp` (10-50): Max velocity magnitude

**Simulation:**
- `boundary_mode` (0-3): None/Reflect/Wrap/Respawn

## Fitness Function

Multi-objective weighted score:
- **35%** Stability (low variance, no NaN/Inf)
- **30%** Accuracy (realistic orbits, energy conservation)
- **20%** Performance (FPS)
- **15%** Visual quality

**Bonuses:**
- +10 for vortex formation (interesting dynamics)
- +5 for good particle retention

## Typical Runtime

- **Small test** (10 pop, 5 gen): ~15-30 minutes
- **Medium run** (20 pop, 30 gen): ~3-6 hours
- **Full run** (50 pop, 100 gen): ~24-48 hours

Each benchmark evaluation takes ~30-60 seconds (500 frames, 5000 particles).

## Output Files

`results/`
- `hall_of_fame.json` - Top 5 individuals with parameters and fitness
- `generation_stats.json` - Per-generation statistics
- `convergence.png` - Fitness convergence plot
- `parameter_distribution.png` - Parameter values for top individuals
- `tmp_eval_*.json` - Temporary benchmark results (auto-cleaned)

## Advanced Usage

### Run Longer Optimization

```python
optimizer = GeneticOptimizer(
    population_size=30,
    generations=100,
    # ... other params
)
```

### Use Different PINN Model

```python
optimizer = GeneticOptimizer(
    pinn_model="v3",  # Try v2 for parameter conditioning
    # ... other params
)
```

### Adjust Evolution Strategy

```python
optimizer = GeneticOptimizer(
    mutation_prob=0.3,    # Higher mutation = more exploration
    crossover_prob=0.5,   # Lower crossover = less recombination
    # ... other params
)
```

## Troubleshooting

**"Executable not found"**
- Build PlasmaDX-Clean first: `MSBuild.exe build/PlasmaDX-Clean.sln`
- Or update `executable_path` in `genetic_optimizer.py`

**"Benchmark timeout"**
- Some parameter combinations may cause hangs
- These individuals get fitness = 0 and are naturally selected out
- Monitor console output for warnings

**Low fitness scores**
- Increase `generations` for better convergence
- Adjust parameter bounds if needed
- Check benchmark is working: `./PlasmaDX-Clean.exe --benchmark --help`

## Next Steps

After finding good parameters:
1. Test them manually in PlasmaDX
2. Use them as starting point for fine-tuning
3. Consider Phase 4: Active Learning to improve PINN model
4. Consider Phase 5: Physics-Constrained SIREN turbulence

---

**Phase 3 Complete!** 🎉

Move on to Phase 4 (Active Learning) or Phase 5 (Constrained Turbulence) when ready.
