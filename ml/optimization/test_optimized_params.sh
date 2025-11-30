#!/bin/bash
# Visual test script for GA-optimized physics parameters
# Compares baseline vs optimized parameters side-by-side

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EXE="$PROJECT_ROOT/build/bin/Debug/PlasmaDX-Clean.exe"
PINN_MODEL="$PROJECT_ROOT/ml/models/pinn_accretion_disk.onnx"

echo "╔════════════════════════════════════════════════════════════╗"
echo "║   GA-Optimized Physics Visual Validation Test             ║"
echo "║   Fitness: 73.79 (Rank 1, Generation 13)                  ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check if executable exists
if [ ! -f "$EXE" ]; then
    echo "❌ ERROR: Executable not found: $EXE"
    echo "   Run: MSBuild.exe PlasmaDX-Clean.sln /p:Configuration=Debug /p:Platform=x64"
    exit 1
fi

# Check if PINN model exists
if [ ! -f "$PINN_MODEL" ]; then
    echo "⚠️  WARNING: PINN model not found: $PINN_MODEL"
    echo "   Continuing with GPU physics fallback..."
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 1: Baseline Physics (default parameters)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Run the renderer and take screenshots at:"
echo "  - Frame 120 (2 seconds)"
echo "  - Frame 600 (10 seconds)"
echo "  - Frame 1200 (20 seconds)"
echo ""
echo "Press F2 to capture screenshots"
echo ""
echo "Command:"
echo "  cd \"$PROJECT_ROOT/build/bin/Debug\""
echo "  ./PlasmaDX-Clean.exe --config=../../../configs/user/default.json"
echo ""
read -p "Press ENTER when baseline screenshots are captured..."

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 2: GA-Optimized Physics (fitness 73.79)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Key parameter changes:"
echo "  • gm: 100.0 → 165.52 (+65%)"
echo "  • bh_mass: 4.3 → 6.71 (+56%)"
echo "  • angular_boost: 1.0 → 2.58 (+158%)"
echo "  • velocity_clamp: 50.0 → 10.0 (-80%)"
echo "  • boundary_mode: 0 → 3 (reflective)"
echo ""
echo "Command:"
echo "  cd \"$PROJECT_ROOT/build/bin/Debug\""
echo "  ./PlasmaDX-Clean.exe --config=../../../configs/user/ga_optimized.json"
echo ""
read -p "Press ENTER when optimized screenshots are captured..."

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST 3: Benchmark Validation (5000 particles)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Running headless benchmark to validate fitness score..."
echo ""

BENCHMARK_OUTPUT="$SCRIPT_DIR/results/validation_benchmark.json"

# Convert WSL path to Windows path
BENCHMARK_WIN_PATH="${BENCHMARK_OUTPUT//\/mnt\//}"
BENCHMARK_WIN_PATH="${BENCHMARK_WIN_PATH/\/d\//D:/}"
BENCHMARK_WIN_PATH="${BENCHMARK_WIN_PATH//\//\\}"

# Run benchmark
cd "$(dirname "$EXE")"
./PlasmaDX-Clean.exe \
    --benchmark \
    --pinn "$PINN_MODEL" \
    --frames 500 \
    --particles 5000 \
    --output "$BENCHMARK_WIN_PATH" \
    --gm 165.52 \
    --bh-mass 6.71 \
    --alpha 0.276 \
    --damping 0.985 \
    --angular-boost 2.58 \
    --disk-thickness 0.098 \
    --inner-radius 3.41 \
    --outer-radius 463.65 \
    --density-scale 2.65 \
    --force-clamp 30.21 \
    --velocity-clamp 10.0 \
    --boundary-mode 3

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Validation Results:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -f "$BENCHMARK_OUTPUT" ]; then
    python3 << 'EOF'
import json
import sys

with open('$BENCHMARK_OUTPUT', 'r') as f:
    results = json.load(f)

summary = results.get('summary', {})
stability = summary.get('stability_score', 0.0)
performance = summary.get('performance_score', 0.0)
accuracy = summary.get('accuracy_score', 0.0)
visual = summary.get('visual_score', 50.0)
overall = summary.get('overall_score', 0.0)

# Compute fitness (same formula as GA)
fitness = 0.35 * stability + 0.30 * accuracy + 0.20 * performance + 0.15 * visual
escape_rate = results.get('stability', {}).get('escape_rate', {}).get('mean', 100.0)
if escape_rate < 10.0:
    fitness += 5.0

print(f"Stability:    {stability:6.2f}/100")
print(f"Accuracy:     {accuracy:6.2f}/100")
print(f"Performance:  {performance:6.2f}/100")
print(f"Visual:       {visual:6.2f}/100")
print(f"")
print(f"Overall Score: {overall:6.2f}/100")
print(f"Fitness:       {fitness:6.2f}/100")
print(f"")
print(f"Expected (GA): 73.79")
print(f"Variance:      {abs(fitness - 73.79):6.2f} ({abs(fitness - 73.79) / 73.79 * 100:.1f}%)")
EOF
else
    echo "❌ Benchmark output not found: $BENCHMARK_OUTPUT"
fi

echo ""
echo "✅ Visual validation test complete!"
echo ""
echo "Next steps:"
echo "  1. Compare screenshots in screenshots/ directory"
echo "  2. Use MCP tool: compare_screenshots_ml for perceptual analysis"
echo "  3. Document visual improvements in CLAUDE.md"
echo ""
