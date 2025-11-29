# Phase 3: Genetic Algorithm Optimizer - COMPLETE ✅

## Issues Fixed

### 1. Working Directory Problem ✅
**Issue**: Executable couldn't find shaders when run from outside its directory  
**Fix**: Added `cwd=str(executable_dir)` to subprocess.run() call

### 2. Output Path Duplication ✅
**Issue**: Output path was `/ml/optimization/ml/optimization/results/` (doubled)  
**Fix**: Made output_dir relative to script location using `Path(__file__).parent`

### 3. Windows Path Conversion ✅
**Issue**: Passing Linux path `/mnt/d/...` to Windows .exe  
**Fix**: Convert WSL paths to Windows format `D:\...` before passing to executable

### 4. Exit Code Handling ✅
**Issue**: Exit code 1 doesn't always mean failure (can mean "UNSUITABLE" score < 50)
**Fix**: Check for output file existence instead of relying solely on exit code

### 5. Executable Path Resolution ✅
**Issue**: Relative path `../build/bin/Debug/PlasmaDX-Clean.exe` failed when run from different directories
**Fix**: Resolve path relative to script location: `Path(__file__).parent.parent.parent / "build/bin/Debug/PlasmaDX-Clean.exe"`
**Benefit**: Optimizer now works when run from ANY directory (ml/optimization/, project root, ml/, etc.)

## Test Results

```
================================================================================
✅ ALL TESTS PASSED!

Ready to run optimization:
  python genetic_optimizer.py
================================================================================
```

## What Was Created

1. **genetic_optimizer.py** - Main GA framework with DEAP
   - 12 physics parameters optimized
   - Multi-objective fitness (stability, accuracy, performance, visual)
   - Population-based evolution with tournament selection
   - Automatic path handling for WSL/Windows compatibility

2. **visualize_results.py** - Convergence plots and analysis
   - Fitness evolution over generations
   - Parameter distribution heatmaps
   - Statistical summaries

3. **test_setup.py** - Verification script
   - Tests all dependencies
   - Verifies executable path
   - Runs single benchmark to validate pipeline

4. **requirements_optimization.txt** - Python dependencies
   - DEAP, NumPy, SciPy, Matplotlib, Pandas, tqdm

## Next Steps

### Quick Test (15-30 minutes)
```bash
cd ml/optimization
source ../venv/bin/activate
python genetic_optimizer.py  # Uses default small test (10 pop, 5 gen)
```

### Full Optimization (6-48 hours)
Edit `genetic_optimizer.py` main() function:
```python
optimizer = GeneticOptimizer(
    executable_path=executable,
    pinn_model="v4",
    population_size=30,    # Increase from 10
    generations=50,        # Increase from 5
    mutation_prob=0.2,
    crossover_prob=0.7
)
```

### Visualize Results
```bash
python visualize_results.py
```

Generates:
- `results/convergence_plot.png` - Fitness over generations
- `results/parameter_distribution.png` - Best individual parameters
- `results/hall_of_fame.json` - Top 5 individuals
- `results/generation_stats.json` - Full evolution history

## Parameter Space

| Parameter | Range | Description |
|-----------|-------|-------------|
| gm | 50.0 - 200.0 | Gravitational parameter |
| bh_mass | 0.1 - 10.0 | Black hole mass |
| alpha | 0.01 - 0.5 | Shakura-Sunyaev viscosity |
| damping | 0.9 - 1.0 | Velocity damping |
| angular_boost | 0.5 - 3.0 | Angular momentum boost |
| disk_thickness | 0.05 - 0.3 | H/R disk thickness |
| inner_radius | 3.0 - 10.0 | Inner edge (units) |
| outer_radius | 200.0 - 500.0 | Outer edge (units) |
| density_scale | 0.5 - 3.0 | Density multiplier |
| force_clamp | 5.0 - 50.0 | Max force magnitude |
| velocity_clamp | 10.0 - 50.0 | Max velocity |
| boundary_mode | 0 - 3 | 0=none, 1=reflect, 2=wrap, 3=respawn |

## Fitness Function

**Multi-objective weighted score:**
- 35% Stability (low variance, no NaN/Inf, energy conservation)
- 30% Accuracy (realistic orbits, Keplerian motion)
- 20% Performance (FPS, physics time)
- 15% Visual quality (coherent motion, disk structure)

**Bonuses:**
- +10 points for vortices (turbulent dynamics)
- +5 points for >90% particle retention

**Score range:** 0-100+ (bonuses can exceed 100)

