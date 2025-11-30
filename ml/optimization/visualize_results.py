"""
Visualization tools for genetic algorithm optimization results

Plots convergence curves, parameter distributions, and fitness landscapes.

Author: Claude Code
Date: 2025-11-29
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List


def plot_convergence(stats_file: str, output_file: str = None):
    """
    Plot fitness convergence over generations

    Args:
        stats_file: Path to generation_stats.json
        output_file: Where to save plot (or None to display)
    """
    with open(stats_file, 'r') as f:
        stats = json.load(f)

    generations = [record['gen'] for record in stats]
    avg_fitness = [record['avg'] for record in stats]
    max_fitness = [record['max'] for record in stats]
    min_fitness = [record['min'] for record in stats]
    std_fitness = [record['std'] for record in stats]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Fitness over generations
    ax1.plot(generations, max_fitness, 'g-', linewidth=2, label='Best')
    ax1.plot(generations, avg_fitness, 'b-', linewidth=2, label='Average')
    ax1.fill_between(generations,
                     np.array(avg_fitness) - np.array(std_fitness),
                     np.array(avg_fitness) + np.array(std_fitness),
                     alpha=0.3, color='blue', label='±1 Std Dev')
    ax1.plot(generations, min_fitness, 'r--', linewidth=1, alpha=0.5, label='Worst')

    ax1.set_xlabel('Generation', fontsize=12)
    ax1.set_ylabel('Fitness Score', fontsize=12)
    ax1.set_title('Genetic Algorithm Convergence', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Improvement rate
    improvement = np.diff(max_fitness)
    ax2.bar(generations[1:], improvement, color='green', alpha=0.6)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Generation', fontsize=12)
    ax2.set_ylabel('Fitness Improvement', fontsize=12)
    ax2.set_title('Generation-to-Generation Improvement', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✅ Convergence plot saved: {output_file}")
    else:
        plt.show()


def plot_parameter_distribution(hof_file: str, output_file: str = None):
    """
    Plot parameter distributions for best individuals

    Args:
        hof_file: Path to hall_of_fame.json
        output_file: Where to save plot (or None to display)
    """
    with open(hof_file, 'r') as f:
        hof = json.load(f)

    if len(hof) == 0:
        print("No individuals in hall of fame")
        return

    # Extract parameters from top 5
    param_names = list(hof[0]['parameters'].keys())
    param_values = {name: [] for name in param_names}

    for individual in hof[:5]:  # Top 5
        for name, value in individual['parameters'].items():
            param_values[name].append(value)

    # Create subplots (4x3 grid for 12 parameters)
    fig, axes = plt.subplots(4, 3, figsize=(15, 12))
    axes = axes.flatten()

    for i, param_name in enumerate(param_names):
        ax = axes[i]
        values = param_values[param_name]

        # Bar chart for top 5
        x = np.arange(len(values))
        ax.bar(x, values, color='steelblue', alpha=0.7)
        ax.set_xlabel('Rank', fontsize=10)
        ax.set_ylabel(param_name, fontsize=10)
        ax.set_title(f'{param_name}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticks(x)
        ax.set_xticklabels([f"#{j+1}" for j in range(len(values))])

    # Hide unused subplots
    for i in range(len(param_names), len(axes)):
        axes[i].axis('off')

    plt.suptitle('Parameter Values for Top 5 Individuals', fontsize=16, fontweight='bold')
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✅ Parameter distribution plot saved: {output_file}")
    else:
        plt.show()


def print_summary(hof_file: str, stats_file: str):
    """Print text summary of optimization results"""
    with open(hof_file, 'r') as f:
        hof = json.load(f)

    with open(stats_file, 'r') as f:
        stats = json.load(f)

    print(f"\n{'='*80}")
    print(f"OPTIMIZATION SUMMARY")
    print(f"{'='*80}\n")

    # Overall stats
    final_gen = stats[-1]
    print(f"Total Generations: {final_gen['gen'] + 1}")
    print(f"Final Best Fitness: {final_gen['max']:.2f}")
    print(f"Final Avg Fitness: {final_gen['avg']:.2f}")
    print(f"Final Std Dev: {final_gen['std']:.2f}")

    # Improvement
    initial_best = stats[0]['max']
    final_best = final_gen['max']
    improvement = final_best - initial_best
    improvement_pct = (improvement / initial_best) * 100 if initial_best > 0 else 0

    print(f"\nImprovement: {improvement:.2f} ({improvement_pct:.1f}%)")
    print(f"  Initial Best: {initial_best:.2f}")
    print(f"  Final Best: {final_best:.2f}")

    # Top 3 individuals
    print(f"\n{'='*80}")
    print(f"TOP 3 INDIVIDUALS")
    print(f"{'='*80}\n")

    for i, individual in enumerate(hof[:3]):
        print(f"Rank #{i+1} (Fitness: {individual['fitness']:.2f})")
        print(f"{'-'*80}")
        for param, value in individual['parameters'].items():
            print(f"  {param:20s}: {value:.3f}")
        print()


def main():
    """Generate all visualizations"""
    # Resolve results directory relative to this script
    script_dir = Path(__file__).parent.absolute()
    results_dir = script_dir / "results"

    stats_file = results_dir / "generation_stats.json"
    hof_file = results_dir / "hall_of_fame.json"

    if not stats_file.exists():
        print(f"ERROR: Stats file not found: {stats_file}")
        print(f"Run genetic_optimizer.py first!")
        return

    if not hof_file.exists():
        print(f"ERROR: Hall of fame file not found: {hof_file}")
        print(f"Run genetic_optimizer.py first!")
        return

    # Generate plots
    print("\nGenerating visualizations...")

    plot_convergence(
        str(stats_file),
        str(results_dir / "convergence.png")
    )

    plot_parameter_distribution(
        str(hof_file),
        str(results_dir / "parameter_distribution.png")
    )

    print_summary(str(hof_file), str(stats_file))

    print(f"\n✅ All visualizations complete!")


if __name__ == "__main__":
    main()
