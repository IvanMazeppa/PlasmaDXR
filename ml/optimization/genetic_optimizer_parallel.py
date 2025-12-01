#!/usr/bin/env python3
"""
Parallel Genetic Algorithm Optimizer for PlasmaDX PINN Physics

Leverages multiprocessing to evaluate multiple individuals simultaneously.
Designed for multi-core CPUs (Ryzen 5950x: 16 cores, 32 threads).

Expected speedup: 10-16x on 16-core CPU

Usage:
    python genetic_optimizer_parallel.py --workers 16
"""

import json
import subprocess
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
from multiprocessing import Pool, cpu_count
import numpy as np

# DEAP imports
from deap import base, creator, tools, algorithms


@dataclass
class ParameterBounds:
    """Bounds for all physics parameters
    
    Phase 5 UPDATED: Realistic world scale for volumetric particles (radius ~20 units)
    - Inner radius must be >> particle diameter (40 units)
    - Outer radius must give particles room to orbit
    - Boundary should be off (mode 0) or very large
    """
    # Gravitational
    gm: Tuple[float, float] = (50.0, 200.0)
    bh_mass: Tuple[float, float] = (0.1, 10.0)

    # Viscosity & Dynamics
    alpha: Tuple[float, float] = (0.01, 0.5)
    damping: Tuple[float, float] = (0.95, 1.0)  # Tighter range - too much damping kills orbits
    angular_boost: Tuple[float, float] = (0.8, 2.0)  # Narrower range

    # Disk Geometry - PHASE 5: Scaled for volumetric particles
    disk_thickness: Tuple[float, float] = (0.05, 0.3)
    inner_radius: Tuple[float, float] = (30.0, 80.0)    # Was 3-10, now 1.5-4× particle diameter
    outer_radius: Tuple[float, float] = (500.0, 1500.0)  # Was 200-500, now 25-75× particle diameter

    # Material
    density_scale: Tuple[float, float] = (0.5, 3.0)

    # Safety Limits
    force_clamp: Tuple[float, float] = (5.0, 50.0)
    velocity_clamp: Tuple[float, float] = (10.0, 50.0)

    # Boundary - PHASE 5: Prefer OFF (mode 0) for realistic physics
    boundary_mode: Tuple[int, int] = (0, 1)  # Was 0-3, now only none(0) or reflect(1)


# Global variables for multiprocessing (set once, shared across workers)
_GLOBAL_OPTIMIZER = None


def _init_worker(optimizer):
    """Initialize worker process with optimizer instance"""
    global _GLOBAL_OPTIMIZER
    _GLOBAL_OPTIMIZER = optimizer


def _evaluate_individual_worker(individual):
    """Worker function for parallel evaluation"""
    global _GLOBAL_OPTIMIZER
    return _GLOBAL_OPTIMIZER.evaluate_individual(individual)


class ParallelGeneticOptimizer:
    """Genetic Algorithm optimizer with parallel evaluation"""

    def __init__(self,
                 executable_path: str,
                 pinn_model: str = "v4",
                 population_size: int = 20,
                 generations: int = 50,
                 mutation_prob: float = 0.2,
                 crossover_prob: float = 0.7,
                 output_dir: str = "results",
                 num_workers: int = None,
                 physics_time_multiplier: float = 1.0):
        """
        Args:
            executable_path: Path to PlasmaDX-Clean.exe
            pinn_model: PINN model to use (v1, v2, v3, v4)
            population_size: Number of individuals per generation
            generations: Number of generations to evolve
            mutation_prob: Probability of mutation per gene
            crossover_prob: Probability of crossover
            output_dir: Directory to save results (relative to script location)
            num_workers: Number of parallel workers (None = auto-detect CPU count)
        """
        # Path resolution
        self.executable_path = str(Path(executable_path).absolute())
        script_dir = Path(__file__).parent.absolute()
        project_root = script_dir.parent.parent

        # Make output_dir relative to script location
        self.output_dir = (script_dir / output_dir).absolute()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.pinn_model = pinn_model
        self.population_size = population_size
        self.generations = generations
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob
        self.physics_time_multiplier = physics_time_multiplier

        # Parallel configuration
        self.num_workers = num_workers or max(1, cpu_count() - 2)  # Leave 2 cores for OS
        print(f"[Parallel GA] Using {self.num_workers} worker processes")
        print(f"[Parallel GA] CPU count: {cpu_count()} (detected)")

        self.bounds = ParameterBounds()
        self.param_names = [
            'gm', 'bh_mass', 'alpha', 'damping', 'angular_boost',
            'disk_thickness', 'inner_radius', 'outer_radius',
            'density_scale', 'force_clamp', 'velocity_clamp', 'boundary_mode'
        ]

        self.toolbox = None
        self.evaluation_count = 0

        self._setup_deap()

    def _setup_deap(self):
        """Initialize DEAP framework"""
        # Create fitness and individual classes
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        self.toolbox = base.Toolbox()

        # Register genetic operators
        self.toolbox.register("individual", self._create_individual)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", self._mutate_individual)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

        # Evaluation will be registered differently for parallel execution
        # (done in run_evolution)

    def _create_individual(self):
        """Create random individual within parameter bounds"""
        individual = []
        for param_name in self.param_names:
            bounds = getattr(self.bounds, param_name)
            if param_name == 'boundary_mode':
                # Integer parameter
                value = np.random.randint(bounds[0], bounds[1] + 1)
            else:
                # Float parameter
                value = np.random.uniform(bounds[0], bounds[1])
            individual.append(value)
        return creator.Individual(individual)

    def _mutate_individual(self, individual):
        """Mutate individual genes"""
        for i, param_name in enumerate(self.param_names):
            if np.random.random() < self.mutation_prob:
                bounds = getattr(self.bounds, param_name)
                if param_name == 'boundary_mode':
                    # Random reset for integer
                    individual[i] = np.random.randint(bounds[0], bounds[1] + 1)
                else:
                    # Gaussian mutation for float
                    sigma = (bounds[1] - bounds[0]) * 0.1
                    individual[i] = np.clip(
                        individual[i] + np.random.normal(0, sigma),
                        bounds[0], bounds[1]
                    )
        return (individual,)

    def decode_individual(self, individual):
        """Convert individual genes to parameter dictionary"""
        params = {}
        for i, param_name in enumerate(self.param_names):
            value = individual[i]
            # Ensure boundary_mode is integer
            if param_name == 'boundary_mode':
                value = int(round(value))
            params[param_name] = value
        return params

    def run_benchmark(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run benchmark with given parameters

        Returns:
            Dictionary with benchmark results (or None if failed)
        """
        import os
        # Use process ID + timestamp for unique filenames in multiprocessing
        unique_id = f"{os.getpid()}_{int(time.time() * 1000000)}"
        output_file = self.output_dir / f"tmp_eval_{unique_id}.json"
        output_path_str = str(output_file)

        # Convert WSL path to Windows path if needed
        if output_path_str.startswith('/mnt/'):
            # /mnt/d/... -> D:/...
            drive = output_path_str[5]
            rest = output_path_str[6:].replace('/', '\\')
            output_path_str = f"{drive.upper()}:{rest}"

        # Build command line
        cmd = [
            self.executable_path,
            "--benchmark",
            "--pinn", self.pinn_model,
            "--frames", "500",  # Shorter for GA speed
            "--particles", "5000",  # Smaller for GA speed
            "--physics-time-multiplier", str(self.physics_time_multiplier),
            "--output", output_path_str,
            "--gm", str(params['gm']),
            "--bh-mass", str(params['bh_mass']),
            "--alpha", str(params['alpha']),
            "--damping", str(params['damping']),
            "--angular-boost", str(params['angular_boost']),
            "--disk-thickness", str(params['disk_thickness']),
            "--inner-radius", str(params['inner_radius']),
            "--outer-radius", str(params['outer_radius']),
            "--density-scale", str(params['density_scale']),
            "--force-clamp", str(params['force_clamp']),
            "--velocity-clamp", str(params['velocity_clamp']),
            "--boundary-mode", str(params['boundary_mode'])
        ]

        try:
            # Get executable directory (needed for shader/resource loading)
            executable_dir = Path(self.executable_path).parent.absolute()

            # Run benchmark from executable directory (timeout after 2 minutes)
            result = subprocess.run(cmd,
                                   capture_output=True,
                                   text=True,
                                   timeout=120,
                                   cwd=str(executable_dir))

            # Note: Exit code 1 can mean "UNSUITABLE" result (score < 50), not failure
            # Check if output file was created instead of relying on exit code
            # (output_file already defined above with unique_id)

            if not output_file.exists():
                print(f"[WARNING] Benchmark failed (exit code {result.returncode})")
                print(f"[WARNING] Output file not found: {output_file}")
                if result.stderr:
                    print(f"[STDERR] {result.stderr}")
                if result.stdout:
                    print(f"[STDOUT] {result.stdout[:1000]}")
                return None

            with open(output_file, 'r') as f:
                results = json.load(f)

            # Clean up temporary file
            output_file.unlink()

            return results

        except subprocess.TimeoutExpired:
            print(f"[WARNING] Benchmark timeout after 120s")
            return None
        except Exception as e:
            print(f"[ERROR] Benchmark failed: {e}")
            return None

    def compute_fitness(self, results: Dict[str, Any]) -> float:
        """
        Compute fitness from benchmark results

        Multi-objective weighted score (Phase 5 - VISUAL EMPHASIS):
        - 25% Stability (reduced from 35%)
        - 15% Accuracy (reduced from 30%)
        - 20% Performance (unchanged)
        - 40% Visual quality (increased from 15% - THIS IS WHAT MATTERS!)

        Bonuses:
        - +10 for vortices (FUTURE: turbulence not implemented yet)
        - +5 for high retention
        """
        if results is None:
            return 0.0

        fitness = 0.0

        # Get summary section (scores are nested here)
        summary = results.get('summary', {})

        # Stability score (0-100)
        stability = summary.get('stability_score', 0.0)
        fitness += 0.25 * stability  # Reduced from 0.35

        # Accuracy score (0-100)
        accuracy = summary.get('accuracy_score', 0.0)
        fitness += 0.15 * accuracy  # Reduced from 0.30

        # Performance score (0-100)
        performance = summary.get('performance_score', 0.0)
        fitness += 0.20 * performance

        # Visual score (0-100, if available) - KEY METRIC FOR TURBULENCE!
        visual = summary.get('visual_score', 50.0)  # Default to 50 if missing
        fitness += 0.40 * visual  # Increased from 0.15 - visual quality is PRIMARY goal

        # Bonus for vortices (turbulence) - NOT YET IMPLEMENTED
        # TODO: Phase 5 - SIREN turbulence integration
        # vortex_count = results.get('turbulence', {}).get('vortex_count', {}).get('mean', 0)
        # if vortex_count > 0:
        #     fitness += 10.0

        # Bonus for high retention
        escape_rate = results.get('stability', {}).get('escape_rate', {}).get('mean', 100.0)
        if escape_rate < 10.0:  # Less than 10% escaped
            fitness += 5.0

        return fitness

    def evaluate_individual(self, individual):
        """Evaluate fitness of individual (called by workers)"""
        params = self.decode_individual(individual)
        results = self.run_benchmark(params)
        fitness = self.compute_fitness(results)

        self.evaluation_count += 1

        return (fitness,)

    def run_evolution(self):
        """
        Run genetic algorithm evolution with parallel evaluation

        Returns:
            List of best individuals (Hall of Fame)
        """
        print(f"\n{'='*80}")
        print(f"PARALLEL GENETIC ALGORITHM OPTIMIZATION")
        print(f"{'='*80}")
        print(f"Population size: {self.population_size}")
        print(f"Generations: {self.generations}")
        print(f"Workers: {self.num_workers}")
        print(f"Expected evaluations: {self.population_size * self.generations}")
        print(f"{'='*80}\n")

        # Create initial population
        population = self.toolbox.population(n=self.population_size)

        # Hall of fame (best individuals)
        hof = tools.HallOfFame(5)

        # Statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # Logbook
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + stats.fields

        # Evaluate initial population IN PARALLEL
        print(f"[Gen 0] Evaluating initial population ({self.population_size} individuals)...")
        start_time = time.time()

        with Pool(processes=self.num_workers, initializer=_init_worker, initargs=(self,)) as pool:
            fitnesses = pool.map(_evaluate_individual_worker, population)

        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        elapsed = time.time() - start_time
        print(f"[Gen 0] Complete in {elapsed:.1f}s ({elapsed/self.population_size:.1f}s per individual)")

        hof.update(population)
        record = stats.compile(population)
        logbook.record(gen=0, nevals=len(population), **record)
        print(logbook.stream)

        # Evolution loop
        for gen in range(1, self.generations + 1):
            print(f"\n[Gen {gen}/{self.generations}] Starting evolution...")
            gen_start = time.time()

            # Select offspring
            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))

            # Crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if np.random.random() < self.crossover_prob:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # Mutation
            for mutant in offspring:
                if np.random.random() < self.mutation_prob:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate offspring IN PARALLEL
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            print(f"[Gen {gen}] Evaluating {len(invalid_ind)} new individuals...")
            eval_start = time.time()

            with Pool(processes=self.num_workers, initializer=_init_worker, initargs=(self,)) as pool:
                fitnesses = pool.map(_evaluate_individual_worker, invalid_ind)

            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            eval_elapsed = time.time() - eval_start
            print(f"[Gen {gen}] Evaluation complete in {eval_elapsed:.1f}s")

            # Update population
            population[:] = offspring
            hof.update(population)

            # Record statistics
            record = stats.compile(population)
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)

            gen_elapsed = time.time() - gen_start
            print(f"[Gen {gen}] {logbook.stream}")
            print(f"[Gen {gen}] Generation time: {gen_elapsed:.1f}s")
            print(f"[Gen {gen}] Best fitness so far: {hof[0].fitness.values[0]:.2f}")

        # Save results
        self.save_results(hof, logbook)

        return list(hof)

    def save_results(self, hof, logbook):
        """Save hall of fame and statistics"""
        # Save hall of fame
        hof_data = []
        for i, ind in enumerate(hof):
            params = self.decode_individual(ind)
            hof_data.append({
                'rank': i + 1,
                'fitness': ind.fitness.values[0],
                'parameters': params
            })

        hof_file = self.output_dir / "hall_of_fame.json"
        with open(hof_file, 'w') as f:
            json.dump(hof_data, f, indent=2)
        print(f"\n[Results] Hall of Fame saved to: {hof_file}")

        # Save generation statistics
        stats_data = []
        for record in logbook:
            stats_data.append(dict(record))

        stats_file = self.output_dir / "generation_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats_data, f, indent=2)
        print(f"[Results] Generation stats saved to: {stats_file}")

        # Print best individual
        print(f"\n{'='*80}")
        print(f"BEST INDIVIDUAL (Fitness: {hof[0].fitness.values[0]:.2f})")
        print(f"{'='*80}")
        best_params = self.decode_individual(hof[0])
        for param_name, value in best_params.items():
            print(f"  {param_name:20s}: {value:.3f}")
        print(f"{'='*80}\n")


def main():
    """Example usage with parallel evaluation"""
    import argparse

    parser = argparse.ArgumentParser(description="Parallel Genetic Algorithm Optimizer")
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of worker processes (default: CPU count - 2)')
    parser.add_argument('--population', type=int, default=10,
                       help='Population size (default: 10)')
    parser.add_argument('--generations', type=int, default=5,
                       help='Number of generations (default: 5)')
    parser.add_argument('--physics-time-multiplier', type=float, default=1.0,
                       help='Physics deltaTime multiplier for faster orbit settling (default: 1.0, range: 1-200)')
    args = parser.parse_args()

    # Path to PlasmaDX executable (resolve relative to this script's location)
    script_dir = Path(__file__).parent.absolute()
    project_root = script_dir.parent.parent  # ml/optimization -> ml -> project root
    executable = project_root / "build/bin/Debug/PlasmaDX-Clean.exe"

    if not executable.exists():
        print(f"ERROR: Executable not found: {executable}")
        print(f"Please build PlasmaDX-Clean first or update the path.")
        print(f"Expected location: {executable}")
        return

    # Create optimizer
    optimizer = ParallelGeneticOptimizer(
        executable_path=executable,
        pinn_model="v4",
        population_size=args.population,
        generations=args.generations,
        mutation_prob=0.2,
        crossover_prob=0.7,
        num_workers=args.workers,
        physics_time_multiplier=args.physics_time_multiplier
    )

    # Run optimization
    start_time = time.time()
    optimizer.run_evolution()
    elapsed = time.time() - start_time

    print(f"\n✅ Optimization complete!")
    print(f"   Total time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    print(f"   Best individuals saved to: {optimizer.output_dir}")
    print(f"\n   Speedup estimate: {optimizer.num_workers}x faster than serial")


if __name__ == "__main__":
    main()
