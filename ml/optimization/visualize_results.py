#!/usr/bin/env python3
"""
Visualization Tools for Genetic Algorithm Optimization Results

Modern, comprehensive visualization suite for PlasmaDX physics optimization.
Generates publication-quality plots with actionable insights.

Usage:
    python visualize_results.py                    # All visualizations
    python visualize_results.py --convergence      # Convergence plot only
    python visualize_results.py --params           # Parameter distribution only
    python visualize_results.py --radar            # Radar chart comparison
    python visualize_results.py --heatmap          # Parameter correlation heatmap
    python visualize_results.py --export-configs   # Export top N as JSON configs
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Use Agg backend for headless environments
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec


# Parameter metadata for better visualization
PARAM_METADATA = {
    'gm': {'label': 'GM', 'unit': '', 'category': 'gravity'},
    'bh_mass': {'label': 'BH Mass', 'unit': 'M☉', 'category': 'gravity'},
    'alpha': {'label': 'α Viscosity', 'unit': '', 'category': 'dynamics'},
    'damping': {'label': 'Damping', 'unit': '', 'category': 'dynamics'},
    'angular_boost': {'label': 'Angular Boost', 'unit': '', 'category': 'dynamics'},
    'disk_thickness': {'label': 'Disk H/R', 'unit': '', 'category': 'geometry'},
    'inner_radius': {'label': 'Inner R', 'unit': '', 'category': 'geometry'},
    'outer_radius': {'label': 'Outer R', 'unit': '', 'category': 'geometry'},
    'density_scale': {'label': 'Density', 'unit': '', 'category': 'material'},
    'force_clamp': {'label': 'Force Clamp', 'unit': '', 'category': 'limits'},
    'velocity_clamp': {'label': 'Vel Clamp', 'unit': '', 'category': 'limits'},
    'boundary_mode': {'label': 'Boundary', 'unit': '', 'category': 'limits'},
    'siren_intensity': {'label': 'SIREN Int', 'unit': '', 'category': 'turbulence'},
    'vortex_scale': {'label': 'Vortex Scale', 'unit': '', 'category': 'turbulence'},
    'vortex_decay': {'label': 'Vortex Decay', 'unit': '', 'category': 'turbulence'},
}

CATEGORY_COLORS = {
    'gravity': '#e41a1c',
    'dynamics': '#377eb8',
    'geometry': '#4daf4a',
    'material': '#984ea3',
    'limits': '#ff7f00',
    'turbulence': '#a65628',
}


def load_results(results_dir: Path) -> Tuple[Optional[List], Optional[List]]:
    """Load hall of fame and generation statistics"""
    hof_file = results_dir / "hall_of_fame.json"
    stats_file = results_dir / "generation_stats.json"

    hof = None
    stats = None

    if hof_file.exists():
        with open(hof_file, 'r') as f:
            hof = json.load(f)
    else:
        print(f"Warning: {hof_file} not found")

    if stats_file.exists():
        with open(stats_file, 'r') as f:
            stats = json.load(f)
    else:
        print(f"Warning: {stats_file} not found")

    return hof, stats


def plot_convergence(stats: List[Dict], output_file: Path, title_suffix: str = ""):
    """
    Plot comprehensive fitness convergence visualization

    Shows fitness trends, improvement rate, and population diversity.
    """
    if not stats:
        print("No statistics data available for convergence plot")
        return

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)

    # Extract data
    gens = [r['gen'] for r in stats]
    maxs = [r['max'] for r in stats]
    avgs = [r['avg'] for r in stats]
    mins = [r['min'] for r in stats]
    stds = [r['std'] for r in stats]
    nevals = [r.get('nevals', 0) for r in stats]

    # Plot 1: Main convergence curve (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(gens, maxs, 'g-', linewidth=2.5, label='Best', marker='o', markersize=3)
    ax1.plot(gens, avgs, 'b-', linewidth=1.5, label='Average', alpha=0.8)
    ax1.fill_between(gens, mins, maxs, alpha=0.15, color='green', label='Range')
    ax1.fill_between(gens,
                     np.array(avgs) - np.array(stds),
                     np.array(avgs) + np.array(stds),
                     alpha=0.2, color='blue', label='±1 Std')

    # Mark best generation
    best_gen_idx = np.argmax(maxs)
    ax1.axvline(x=gens[best_gen_idx], color='gold', linestyle='--', alpha=0.7, linewidth=1.5)
    ax1.scatter([gens[best_gen_idx]], [maxs[best_gen_idx]], s=150, color='gold',
                zorder=5, edgecolors='black', linewidths=1.5)
    ax1.annotate(f'Best: {maxs[best_gen_idx]:.2f}',
                xy=(gens[best_gen_idx], maxs[best_gen_idx]),
                xytext=(10, 10), textcoords='offset points',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

    ax1.set_xlabel('Generation', fontsize=11)
    ax1.set_ylabel('Fitness Score', fontsize=11)
    ax1.set_title('Fitness Convergence', fontsize=13, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(min(gens), max(gens))

    # Plot 2: Improvement rate (top-right)
    ax2 = fig.add_subplot(gs[0, 1])
    improvements = np.diff(maxs)
    colors = ['#2ecc71' if x > 0 else '#e74c3c' if x < 0 else '#95a5a6' for x in improvements]
    bars = ax2.bar(gens[1:], improvements, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)

    # Cumulative improvement line
    ax2_twin = ax2.twinx()
    cumulative = np.cumsum(np.maximum(improvements, 0))
    ax2_twin.plot(gens[1:], cumulative, 'purple', linewidth=2, linestyle='--', alpha=0.7)
    ax2_twin.set_ylabel('Cumulative Gain', fontsize=10, color='purple')
    ax2_twin.tick_params(axis='y', labelcolor='purple')

    ax2.set_xlabel('Generation', fontsize=11)
    ax2.set_ylabel('Fitness Change', fontsize=11)
    ax2.set_title('Generation Improvement', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Population diversity (bottom-left)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.fill_between(gens, 0, stds, alpha=0.6, color='#3498db')
    ax3.plot(gens, stds, 'b-', linewidth=2)

    # Add diversity trend line
    z = np.polyfit(gens, stds, 2)
    p = np.poly1d(z)
    ax3.plot(gens, p(gens), 'r--', linewidth=1.5, alpha=0.7, label='Trend')

    ax3.set_xlabel('Generation', fontsize=11)
    ax3.set_ylabel('Standard Deviation', fontsize=11)
    ax3.set_title('Population Diversity', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right')

    # Plot 4: Summary statistics (bottom-right)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    # Calculate summary stats
    total_improvement = maxs[-1] - maxs[0]
    improvement_pct = (total_improvement / maxs[0] * 100) if maxs[0] > 0 else 0
    avg_improvement = np.mean([x for x in improvements if x > 0]) if any(x > 0 for x in improvements) else 0
    stagnation_gens = sum(1 for x in improvements if x <= 0)
    total_evals = sum(nevals)

    summary_text = f"""
    OPTIMIZATION SUMMARY
    {'='*40}

    Initial Best:      {maxs[0]:.2f}
    Final Best:        {maxs[-1]:.2f}
    Total Improvement: {total_improvement:+.2f} ({improvement_pct:+.1f}%)

    Generations:       {len(gens)}
    Total Evaluations: {total_evals}
    Best at Gen:       {gens[best_gen_idx]}

    Avg Improvement:   {avg_improvement:.3f}/gen
    Stagnant Gens:     {stagnation_gens} ({stagnation_gens/max(1,len(gens)-1)*100:.1f}%)
    Final Diversity:   {stds[-1]:.2f}
    """

    ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#ecf0f1', alpha=0.8))

    fig.suptitle(f'GA Optimization Results{title_suffix}', fontsize=15, fontweight='bold', y=0.98)
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Convergence plot saved: {output_file}")


def plot_parameter_distribution(hof: List[Dict], output_file: Path, bounds: Dict = None):
    """
    Plot parameter value distributions for top individuals

    Color-coded by parameter category with normalized comparison.
    """
    if not hof:
        print("No hall of fame data available for parameter distribution")
        return

    param_names = list(hof[0]['parameters'].keys())
    n_params = len(param_names)
    n_individuals = min(5, len(hof))

    # Calculate grid dimensions
    n_cols = 4
    n_rows = (n_params + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3))
    axes = axes.flatten()

    for i, param_name in enumerate(param_names):
        ax = axes[i]
        values = [hof[j]['parameters'][param_name] for j in range(n_individuals)]

        # Get category color
        meta = PARAM_METADATA.get(param_name, {'label': param_name, 'category': 'other'})
        color = CATEGORY_COLORS.get(meta['category'], '#7f8c8d')

        # Create grouped bar chart
        x = np.arange(n_individuals)
        bars = ax.bar(x, values, color=color, alpha=0.8, edgecolor='black', linewidth=0.5)

        # Highlight best value
        best_idx = np.argmax(values) if param_name != 'boundary_mode' else 0
        bars[0].set_edgecolor('gold')
        bars[0].set_linewidth(2)

        ax.set_xlabel('Rank', fontsize=9)
        ax.set_ylabel(meta['label'], fontsize=9)
        ax.set_title(f"{meta['label']}", fontsize=10, fontweight='bold', color=color)
        ax.set_xticks(x)
        ax.set_xticklabels([f"#{j+1}" for j in range(n_individuals)], fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for j, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.annotate(f'{val:.2f}' if val >= 0.01 else f'{val:.4f}',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords='offset points',
                       ha='center', va='bottom', fontsize=7, rotation=45)

    # Hide unused subplots
    for i in range(n_params, len(axes)):
        axes[i].axis('off')

    # Add legend for categories
    legend_elements = [Patch(facecolor=color, edgecolor='black', label=cat.title())
                      for cat, color in CATEGORY_COLORS.items()]
    fig.legend(handles=legend_elements, loc='upper right', ncol=3, fontsize=9,
              bbox_to_anchor=(0.98, 0.02))

    fig.suptitle(f'Parameter Values for Top {n_individuals} Individuals',
                fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Parameter distribution saved: {output_file}")


def plot_radar_chart(hof: List[Dict], output_file: Path):
    """
    Radar chart comparing top 3 individuals across key parameters

    Normalized to 0-1 scale for visual comparison.
    """
    if not hof or len(hof) < 2:
        print("Need at least 2 individuals for radar chart")
        return

    # Key parameters for radar (subset for readability)
    key_params = ['gm', 'bh_mass', 'alpha', 'damping', 'angular_boost',
                  'siren_intensity', 'vortex_scale', 'inner_radius']
    key_params = [p for p in key_params if p in hof[0]['parameters']]

    n_params = len(key_params)
    n_individuals = min(3, len(hof))

    # Calculate angles
    angles = np.linspace(0, 2 * np.pi, n_params, endpoint=False).tolist()
    angles += angles[:1]  # Complete the loop

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

    colors = ['#e74c3c', '#3498db', '#2ecc71']

    for i in range(n_individuals):
        params = hof[i]['parameters']

        # Normalize values to 0-1 range
        values = []
        for p in key_params:
            val = params[p]
            # Use approximate bounds for normalization
            bounds = {
                'gm': (50, 200), 'bh_mass': (0.1, 10), 'alpha': (0.01, 0.5),
                'damping': (0.95, 1.0), 'angular_boost': (0.8, 2.0),
                'siren_intensity': (0, 1), 'vortex_scale': (0.5, 3),
                'inner_radius': (30, 80)
            }
            lo, hi = bounds.get(p, (0, val * 2 if val > 0 else 1))
            normalized = (val - lo) / (hi - lo) if hi > lo else 0.5
            values.append(np.clip(normalized, 0, 1))

        values += values[:1]  # Complete the loop

        ax.plot(angles, values, 'o-', linewidth=2, color=colors[i],
                label=f'Rank #{i+1} (Fit: {hof[i]["fitness"]:.1f})')
        ax.fill(angles, values, alpha=0.15, color=colors[i])

    # Set labels
    labels = [PARAM_METADATA.get(p, {'label': p})['label'] for p in key_params]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)

    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=8)
    ax.grid(True, alpha=0.3)

    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.set_title('Top Individuals Parameter Comparison\n(Normalized to parameter bounds)',
                fontsize=13, fontweight='bold', pad=20)

    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Radar chart saved: {output_file}")


def plot_parameter_correlation(hof: List[Dict], output_file: Path):
    """
    Heatmap showing correlation between parameters across top individuals
    """
    if not hof or len(hof) < 3:
        print("Need at least 3 individuals for correlation heatmap")
        return

    param_names = list(hof[0]['parameters'].keys())
    n_params = len(param_names)
    n_individuals = min(10, len(hof))

    # Build parameter matrix
    matrix = np.zeros((n_individuals, n_params))
    for i in range(n_individuals):
        for j, p in enumerate(param_names):
            matrix[i, j] = hof[i]['parameters'][p]

    # Calculate correlation matrix
    corr_matrix = np.corrcoef(matrix.T)

    fig, ax = plt.subplots(figsize=(12, 10))

    # Create heatmap
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Correlation Coefficient', fontsize=11)

    # Set labels
    labels = [PARAM_METADATA.get(p, {'label': p})['label'] for p in param_names]
    ax.set_xticks(np.arange(n_params))
    ax.set_yticks(np.arange(n_params))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    # Add correlation values as text
    for i in range(n_params):
        for j in range(n_params):
            val = corr_matrix[i, j]
            color = 'white' if abs(val) > 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                   color=color, fontsize=7)

    ax.set_title(f'Parameter Correlation Matrix (Top {n_individuals} Individuals)',
                fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Correlation heatmap saved: {output_file}")


def export_configs(hof: List[Dict], output_dir: Path, n_configs: int = 5,
                   physics_multiplier: float = 3.0):
    """
    Export top N individuals as JSON config files for testing
    """
    if not hof:
        print("No hall of fame data available for config export")
        return

    configs_dir = output_dir / "exported_configs"
    configs_dir.mkdir(exist_ok=True)

    for i, individual in enumerate(hof[:n_configs]):
        params = individual['parameters']

        config = {
            "description": f"GA Optimizer Rank {i+1} - Fitness {individual['fitness']:.2f}",
            "benchmark": {
                "enabled": True,
                "frames": 1000
            },
            "physics": {
                "gm": round(params.get('gm', 100), 2),
                "bh_mass": round(params.get('bh_mass', 5), 2),
                "alpha": round(params.get('alpha', 0.2), 3),
                "damping": round(params.get('damping', 0.97), 3),
                "angular_boost": round(params.get('angular_boost', 1.5), 2),
                "disk_thickness": round(params.get('disk_thickness', 0.1), 3),
                "inner_radius": round(params.get('inner_radius', 50), 2),
                "outer_radius": round(params.get('outer_radius', 1000), 2),
                "density_scale": round(params.get('density_scale', 2.5), 2),
                "force_clamp": round(params.get('force_clamp', 800), 2),
                "velocity_clamp": round(params.get('velocity_clamp', 200), 2),
                "boundary_mode": int(params.get('boundary_mode', 0)),
                "time_multiplier": physics_multiplier
            },
            "siren": {
                "enabled": params.get('siren_intensity', 0) > 0.01,
                "intensity": round(params.get('siren_intensity', 0), 3),
                "vortex_scale": round(params.get('vortex_scale', 1), 2),
                "vortex_decay": round(params.get('vortex_decay', 0.1), 3)
            },
            "particles": {
                "count": 10000
            }
        }

        config_file = configs_dir / f"ga_rank{i+1}.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Config exported: {config_file}")

    print(f"\nExported {min(n_configs, len(hof))} configs to {configs_dir}")


def print_summary(hof: List[Dict], stats: List[Dict]):
    """Print comprehensive text summary of optimization results"""
    print(f"\n{'='*80}")
    print(f"{'OPTIMIZATION RESULTS SUMMARY':^80}")
    print(f"{'='*80}\n")

    if stats:
        final_gen = stats[-1]
        print(f"CONVERGENCE METRICS")
        print(f"{'-'*40}")
        print(f"  Total Generations:    {final_gen['gen'] + 1}")
        print(f"  Final Best Fitness:   {final_gen['max']:.2f}")
        print(f"  Final Avg Fitness:    {final_gen['avg']:.2f}")
        print(f"  Final Population Div: {final_gen['std']:.2f}")

        if len(stats) > 1:
            improvement = final_gen['max'] - stats[0]['max']
            print(f"\n  Initial Best:         {stats[0]['max']:.2f}")
            print(f"  Improvement:          {improvement:+.2f} ({improvement/stats[0]['max']*100:+.1f}%)")

    if hof:
        print(f"\n\nTOP 5 INDIVIDUALS")
        print(f"{'='*80}")

        for i, ind in enumerate(hof[:5]):
            print(f"\nRank #{i+1} | Fitness: {ind['fitness']:.2f}")
            print(f"{'-'*40}")

            # Group parameters by category
            by_category = {}
            for param, value in ind['parameters'].items():
                meta = PARAM_METADATA.get(param, {'category': 'other', 'label': param})
                cat = meta['category']
                if cat not in by_category:
                    by_category[cat] = []
                by_category[cat].append((meta['label'], value))

            for cat, params in by_category.items():
                print(f"  [{cat.upper()}]")
                for label, val in params:
                    print(f"    {label:15s}: {val:.4f}")

        # Key insights
        print(f"\n\n{'KEY INSIGHTS':^80}")
        print(f"{'='*80}")

        # Convergence patterns
        best = hof[0]['parameters']
        print(f"\nBest Configuration Characteristics:")
        print(f"  - GM: {best.get('gm', 'N/A'):.1f} (gravitational strength)")
        print(f"  - Damping: {best.get('damping', 'N/A'):.3f} (energy retention)")
        print(f"  - SIREN Intensity: {best.get('siren_intensity', 'N/A'):.3f} (turbulence)")

        # Parameter consistency across top individuals
        if len(hof) >= 3:
            print(f"\nParameter Consistency (Top 3):")
            for param in ['gm', 'damping', 'siren_intensity']:
                if param in hof[0]['parameters']:
                    values = [hof[i]['parameters'][param] for i in range(3)]
                    std = np.std(values)
                    mean = np.mean(values)
                    cv = (std / mean * 100) if mean > 0 else 0
                    label = PARAM_METADATA.get(param, {'label': param})['label']
                    stability = "STABLE" if cv < 10 else "VARIABLE" if cv < 30 else "DIVERSE"
                    print(f"  - {label}: CV={cv:.1f}% ({stability})")


def main():
    parser = argparse.ArgumentParser(description="GA Optimization Visualization Suite")
    parser.add_argument('--results-dir', type=str, default=None,
                       help='Results directory (default: script_dir/results)')
    parser.add_argument('--convergence', action='store_true',
                       help='Generate convergence plot only')
    parser.add_argument('--params', action='store_true',
                       help='Generate parameter distribution only')
    parser.add_argument('--radar', action='store_true',
                       help='Generate radar chart only')
    parser.add_argument('--heatmap', action='store_true',
                       help='Generate correlation heatmap only')
    parser.add_argument('--export-configs', action='store_true',
                       help='Export top individuals as JSON configs')
    parser.add_argument('--n-configs', type=int, default=5,
                       help='Number of configs to export (default: 5)')
    parser.add_argument('--summary', action='store_true',
                       help='Print text summary only')
    args = parser.parse_args()

    # Resolve results directory
    script_dir = Path(__file__).parent.absolute()
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        results_dir = script_dir / "results"

    if not results_dir.exists():
        print(f"ERROR: Results directory not found: {results_dir}")
        print("Run genetic_optimizer_parallel.py first!")
        return

    # Load data
    hof, stats = load_results(results_dir)

    if not hof and not stats:
        print("ERROR: No data files found in results directory")
        return

    # Generate requested visualizations (or all if none specified)
    generate_all = not any([args.convergence, args.params, args.radar,
                           args.heatmap, args.export_configs, args.summary])

    print(f"\nGenerating visualizations from: {results_dir}")
    print(f"{'='*60}\n")

    if args.summary or generate_all:
        print_summary(hof, stats)

    if args.convergence or generate_all:
        if stats:
            plot_convergence(stats, results_dir / "convergence.png")

    if args.params or generate_all:
        if hof:
            plot_parameter_distribution(hof, results_dir / "parameter_distribution.png")

    if args.radar or generate_all:
        if hof:
            plot_radar_chart(hof, results_dir / "radar_comparison.png")

    if args.heatmap or generate_all:
        if hof:
            plot_parameter_correlation(hof, results_dir / "parameter_correlation.png")

    if args.export_configs:
        if hof:
            export_configs(hof, results_dir, args.n_configs)

    print(f"\nVisualization complete!")


if __name__ == "__main__":
    main()
