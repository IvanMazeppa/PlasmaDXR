#!/bin/bash
# Quick benchmark validation script
# Runs baseline vs optimized and compares results

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$PROJECT_ROOT/build/bin/Debug"

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║          GA Optimization Benchmark Validation                 ║"
echo "║          Baseline vs Optimized (Fitness 73.79)                ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Check if executable exists
if [ ! -f "$BUILD_DIR/PlasmaDX-Clean.exe" ]; then
    echo "❌ ERROR: Executable not found: $BUILD_DIR/PlasmaDX-Clean.exe"
    echo ""
    echo "Build first:"
    echo "  MSBuild.exe PlasmaDX-Clean.sln /p:Configuration=Debug /p:Platform=x64"
    exit 1
fi

cd "$BUILD_DIR"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Running Baseline Benchmark (default physics)..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
./PlasmaDX-Clean.exe --config=../../../configs/scenarios/benchmark_baseline.json

if [ $? -eq 0 ]; then
    echo "✅ Baseline benchmark complete"
else
    echo "❌ Baseline benchmark failed"
    exit 1
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Running Optimized Benchmark (GA parameters, fitness 73.79)..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
./PlasmaDX-Clean.exe --config=../../../configs/scenarios/benchmark_ga_validation.json

if [ $? -eq 0 ]; then
    echo "✅ Optimized benchmark complete"
else
    echo "❌ Optimized benchmark failed"
    exit 1
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Results Comparison"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Compare results using Python
python3 << 'PYTHON_EOF'
import json
import sys
from pathlib import Path

results_dir = Path(__file__).parent.parent.parent / "ml" / "optimization" / "results"
baseline_file = results_dir / "baseline_benchmark.json"
optimized_file = results_dir / "validation_benchmark.json"

def compute_fitness(results):
    """Compute fitness using GA formula"""
    summary = results.get('summary', {})
    stability = summary.get('stability_score', 0.0)
    accuracy = summary.get('accuracy_score', 0.0)
    performance = summary.get('performance_score', 0.0)
    visual = summary.get('visual_score', 50.0)

    fitness = 0.35 * stability + 0.30 * accuracy + 0.20 * performance + 0.15 * visual

    # Bonus for low escape rate
    escape_rate = results.get('stability', {}).get('escape_rate', {}).get('mean', 100.0)
    if escape_rate < 10.0:
        fitness += 5.0

    return fitness, stability, accuracy, performance, visual

try:
    with open(baseline_file) as f:
        baseline = json.load(f)
    with open(optimized_file) as f:
        optimized = json.load(f)

    base_fit, base_stab, base_acc, base_perf, base_vis = compute_fitness(baseline)
    opt_fit, opt_stab, opt_acc, opt_perf, opt_vis = compute_fitness(optimized)

    print("┌─────────────────┬──────────┬──────────┬──────────┐")
    print("│ Metric          │ Baseline │ Optimized│   Delta  │")
    print("├─────────────────┼──────────┼──────────┼──────────┤")
    print(f"│ Stability       │ {base_stab:8.2f} │ {opt_stab:8.2f} │ {opt_stab-base_stab:+8.2f} │")
    print(f"│ Accuracy        │ {base_acc:8.2f} │ {opt_acc:8.2f} │ {opt_acc-base_acc:+8.2f} │")
    print(f"│ Performance     │ {base_perf:8.2f} │ {opt_perf:8.2f} │ {opt_perf-base_perf:+8.2f} │")
    print(f"│ Visual          │ {base_vis:8.2f} │ {opt_vis:8.2f} │ {opt_vis-base_vis:+8.2f} │")
    print("├─────────────────┼──────────┼──────────┼──────────┤")
    print(f"│ FITNESS         │ {base_fit:8.2f} │ {opt_fit:8.2f} │ {opt_fit-base_fit:+8.2f} │")
    print("└─────────────────┴──────────┴──────────┴──────────┘")
    print("")
    print(f"Expected fitness (from GA): 73.79")
    print(f"Validated fitness:          {opt_fit:.2f}")
    print(f"Variance:                   {abs(opt_fit - 73.79):.2f} ({abs(opt_fit - 73.79) / 73.79 * 100:.1f}%)")
    print("")

    improvement = ((opt_fit - base_fit) / base_fit * 100)
    print(f"Overall improvement: {improvement:+.1f}%")
    print("")

    if opt_fit >= 70.0:
        print("✅ VALIDATION PASSED - Fitness meets expectations!")
    elif opt_fit >= 65.0:
        print("⚠️  PARTIAL SUCCESS - Fitness lower than expected but still good")
    else:
        print("❌ VALIDATION FAILED - Fitness significantly lower than expected")

except FileNotFoundError as e:
    print(f"❌ ERROR: Result file not found: {e.filename}")
    sys.exit(1)
except Exception as e:
    print(f"❌ ERROR: {e}")
    sys.exit(1)

PYTHON_EOF

echo ""
echo "Results saved to:"
echo "  - ml/optimization/results/baseline_benchmark.json"
echo "  - ml/optimization/results/validation_benchmark.json"
echo ""
