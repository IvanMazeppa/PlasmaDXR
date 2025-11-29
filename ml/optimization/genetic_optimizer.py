"""
Genetic Algorithm Optimizer for PlasmaDX Physics Parameters

Uses DEAP (Distributed Evolutionary Algorithms in Python) to optimize
physics parameters for accretion disk simulation quality.

Author: Claude Code
Date: 2025-11-29
"""

import subprocess
import json
import os
import sys
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
import time

# DEAP imports
from deap import base, creator, tools, algorithms

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))


@dataclass
class ParameterBounds:
    """Bounds for all physics parameters"""
    # Gravitational
    gm: Tuple[float, float] = (50.0, 200.0)
    bh_mass: Tuple[float, float] = (0.1, 10.0)

    # Viscosity & Dynamics
    alpha: Tuple[float, float] = (0.01, 0.5)
    damping: Tuple[float, float] = (0.9, 1.0)
    angular_boost: Tuple[float, float] = (0.5, 3.0)

    # Disk Geometry
    disk_thickness: Tuple[float, float] = (0.05, 0.3)
    inner_radius: Tuple[float, float] = (3.0, 10.0)
    outer_radius: Tuple[float, float] = (200.0, 500.0)

    # Material
    density_scale: Tuple[float, float] = (0.5, 3.0)

    # Safety Limits
    force_clamp: Tuple[float, float] = (5.0, 50.0)
    velocity_clamp: Tuple[float, float] = (10.0, 50.0)

    # Simulation
    boundary_mode: Tuple[int, int] = (0, 3)  # 0=none, 1=reflect, 2=wrap, 3=respawn


class GeneticOptimizer:
    """Genetic Algorithm optimizer for physics parameters"""

    def __init__(self,
                 executable_path: str,
                 pinn_model: str = "v4",
                 population_size: int = 20,
                 generations: int = 50,
                 mutation_prob: float = 0.2,
                 crossover_prob: float = 0.7,
                 output_dir: str = "ml/optimization/results"):
        """
        Args:
            executable_path: Path to PlasmaDX-Clean.exe
            pinn_model: PINN model to use (v1, v2, v3, v4)
            population_size: Number of individuals per generation
            generations: Number of generations to evolve
            mutation_prob: Probability of mutation per gene
            crossover_prob: Probability of crossover
            output_dir: Directory to save results
        """
        self.executable_path = executable_path
        self.pinn_model = pinn_model
        self.population_size = population_size
        self.generations = generations
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.bounds = ParameterBounds()
        self.param_names = [
            'gm', 'bh_mass', 'alpha', 'damping', 'angular_boost',
            'disk_thickness', 'inner_radius', 'outer_radius', 'density_scale',
            'force_clamp', 'velocity_clamp', 'boundary_mode'
        ]

        # Statistics tracking
        self.generation_stats = []
        self.best_individuals = []
        self.evaluation_count = 0

        # Setup DEAP
        self._setup_deap()

    def _setup_deap(self):
        """Initialize DEAP framework"""
        # Create fitness class (maximize single objective)
        if hasattr(creator, "FitnessMax"):
            del creator.FitnessMax
        if hasattr(creator, "Individual"):
            del creator.Individual

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        # Create toolbox
        self.toolbox = base.Toolbox()

        # Attribute generators for each parameter
        for param_name in self.param_names:
            bounds = getattr(self.bounds, param_name)
            if param_name == 'boundary_mode':
                # Integer parameter
                self.toolbox.register(f"attr_{param_name}",
                                     random.randint, bounds[0], bounds[1])
            else:
                # Float parameter
                self.toolbox.register(f"attr_{param_name}",
                                     random.uniform, bounds[0], bounds[1])

        # Individual generator
        self.toolbox.register("individual", self._create_individual)

        # Population generator
        self.toolbox.register("population", tools.initRepeat,
                            list, self.toolbox.individual)

        # Genetic operators
        self.toolbox.register("evaluate", self.evaluate_individual)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", self._mutate_individual)
        self.toolbox.register("select", tools.selTournament, tournsize=3)

    def _create_individual(self):
        """Create a random individual with all parameters"""
        genes = []
        for param_name in self.param_names:
            attr_func = getattr(self.toolbox, f"attr_{param_name}")
            genes.append(attr_func())
        return creator.Individual(genes)

    def _mutate_individual(self, individual) -> Tuple:
        """
        Mutate individual with Gaussian noise for floats,
        random reset for integers
        """
        for i, param_name in enumerate(self.param_names):
            if random.random() < self.mutation_prob:
                bounds = getattr(self.bounds, param_name)
                if param_name == 'boundary_mode':
                    # Integer: random reset
                    individual[i] = random.randint(bounds[0], bounds[1])
                else:
                    # Float: Gaussian mutation (10% of range)
                    sigma = (bounds[1] - bounds[0]) * 0.1
                    individual[i] += random.gauss(0, sigma)
                    # Clamp to bounds
                    individual[i] = max(bounds[0], min(bounds[1], individual[i]))
        return (individual,)

    def decode_individual(self, individual) -> Dict[str, Any]:
        """Convert individual genes to parameter dictionary"""
        params = {}
        for i, param_name in enumerate(self.param_names):
            value = individual[i]
            # Round boundary_mode to integer
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
        # Build command line
        cmd = [
            self.executable_path,
            "--benchmark",
            "--pinn", self.pinn_model,
            "--frames", "500",  # Shorter for GA speed
            "--particles", "5000",  # Smaller for GA speed
            "--output", str(self.output_dir / f"tmp_eval_{self.evaluation_count}.json"),
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
            # Run benchmark (timeout after 2 minutes)
            result = subprocess.run(cmd,
                                   capture_output=True,
                                   text=True,
                                   timeout=120)

            if result.returncode != 0:
                print(f"[WARNING] Benchmark failed: {result.stderr[:200]}")
                return None

            # Load results
            output_file = self.output_dir / f"tmp_eval_{self.evaluation_count}.json"
            if not output_file.exists():
                print(f"[WARNING] Output file not found: {output_file}")
                return None

            with open(output_file, 'r') as f:
                results = json.load(f)

            # Clean up temp file
            output_file.unlink()

            return results

        except subprocess.TimeoutExpired:
            print(f"[WARNING] Benchmark timeout (params: {params})")
            return None
        except Exception as e:
            print(f"[ERROR] Benchmark exception: {e}")
            return None

    def compute_fitness(self, results: Dict[str, Any]) -> float:
        """
        Compute fitness score from benchmark results

        Multi-objective weighted fitness:
        - 35% Stability (low variance, no NaN/Inf)
        - 30% Accuracy (realistic orbits, energy conservation)
        - 20% Performance (FPS)
        - 15% Visual quality (if available)

        Bonus:
        - +10 for having vortices (interesting dynamics)
        - +5 for good boundary behavior
        """
        if results is None:
            return 0.0

        fitness = 0.0

        # Stability score (0-100)
        stability = results.get('stability_score', 0.0)
        fitness += 0.35 * stability

        # Accuracy score (0-100)
        accuracy = results.get('accuracy_score', 0.0)
        fitness += 0.30 * accuracy

        # Performance score (0-100)
        performance = results.get('performance_score', 0.0)
        fitness += 0.20 * performance

        # Visual score (0-100, if available)
        visual = results.get('visual_score', 50.0)  # Default to neutral
        fitness += 0.15 * visual

        # Bonus for vortices (interesting turbulent dynamics)
        vortex_count = results.get('turbulence', {}).get('vortex_count', {}).get('mean', 0)
        if vortex_count > 0:
            fitness += 10.0

        # Bonus for particles staying in bounds
        final_particle_count = results.get('final_particle_count', 0)
        initial_particle_count = results.get('particle_count', 1)
        retention_rate = final_particle_count / max(initial_particle_count, 1)
        if retention_rate > 0.9:
            fitness += 5.0

        return fitness

    def evaluate_individual(self, individual) -> Tuple[float]:
        """Evaluate fitness of an individual"""
        self.evaluation_count += 1

        # Decode parameters
        params = self.decode_individual(individual)

        print(f"\n[Eval {self.evaluation_count}] Testing: GM={params['gm']:.1f}, "
              f"alpha={params['alpha']:.3f}, BH={params['bh_mass']:.1f}")

        # Run benchmark
        results = self.run_benchmark(params)

        # Compute fitness
        fitness = self.compute_fitness(results)

        print(f"  → Fitness: {fitness:.2f}")

        return (fitness,)

    def run_evolution(self):
        """Run genetic algorithm evolution"""
        print(f"\n{'='*80}")
        print(f"GENETIC ALGORITHM OPTIMIZATION")
        print(f"{'='*80}")
        print(f"Population size: {self.population_size}")
        print(f"Generations: {self.generations}")
        print(f"Mutation prob: {self.mutation_prob}")
        print(f"Crossover prob: {self.crossover_prob}")
        print(f"PINN model: {self.pinn_model}")
        print(f"{'='*80}\n")

        # Create initial population
        population = self.toolbox.population(n=self.population_size)

        # Statistics
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        # Hall of fame (best individuals)
        hof = tools.HallOfFame(5)

        # Run evolution
        population, logbook = algorithms.eaSimple(
            population,
            self.toolbox,
            cxpb=self.crossover_prob,
            mutpb=self.mutation_prob,
            ngen=self.generations,
            stats=stats,
            halloffame=hof,
            verbose=True
        )

        # Save statistics
        self.generation_stats = logbook
        self.best_individuals = hof

        # Save results
        self.save_results()

        return hof

    def save_results(self):
        """Save optimization results"""
        # Save hall of fame
        hof_file = self.output_dir / "hall_of_fame.json"
        hof_data = []
        for i, ind in enumerate(self.best_individuals):
            params = self.decode_individual(ind)
            hof_data.append({
                'rank': i + 1,
                'fitness': ind.fitness.values[0],
                'parameters': params
            })

        with open(hof_file, 'w') as f:
            json.dump(hof_data, f, indent=2)

        print(f"\n✅ Hall of Fame saved: {hof_file}")

        # Save generation statistics
        stats_file = self.output_dir / "generation_stats.json"
        stats_data = []
        for record in self.generation_stats:
            stats_data.append(dict(record))

        with open(stats_file, 'w') as f:
            json.dump(stats_data, f, indent=2)

        print(f"✅ Statistics saved: {stats_file}")

        # Print best individual
        if len(self.best_individuals) > 0:
            best = self.best_individuals[0]
            params = self.decode_individual(best)
            print(f"\n🏆 BEST INDIVIDUAL (Fitness: {best.fitness.values[0]:.2f})")
            print(f"{'='*80}")
            for param_name, value in params.items():
                print(f"  {param_name:20s}: {value:.3f}")
            print(f"{'='*80}\n")


def main():
    """Example usage"""
    # Path to PlasmaDX executable
    cd "../build/bin/Debug/"
    executable = ./PlasmaDX-Clean.exe"

    if not Path(executable).exists():
        print(f"ERROR: Executable not found: {executable}")
        print(f"Please build PlasmaDX-Clean first or update the path.")
        return

    # Create optimizer
    optimizer = GeneticOptimizer(
        executable_path=executable,
        pinn_model="v4",
        population_size=10,  # Small for testing
        generations=5,       # Small for testing
        mutation_prob=0.2,
        crossover_prob=0.7
    )

    # Run optimization
    best_individuals = optimizer.run_evolution()

    print(f"\n✅ Optimization complete!")
    print(f"   Best individuals saved to: {optimizer.output_dir}")


if __name__ == "__main__":
    main()
