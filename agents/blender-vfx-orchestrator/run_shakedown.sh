#!/bin/bash
# Quick shakedown test - 2 iterations, quick_test preset
# Usage: ./run_shakedown.sh [effect]
# Effects: sun, fire, explosion, smoke, nebula

cd "$(dirname "$0")"
source venv/bin/activate

EFFECT="${1:-fire}"

echo "=============================================="
echo "SHAKEDOWN TEST - 2 iterations, quick_test mode"
echo "Effect: $EFFECT"
echo "=============================================="

python test_quick_e2e.py \
    --preset quick_test \
    --iterations 2 \
    --effect "$EFFECT" \
    --name "shakedown_$(date +%H%M%S)"
