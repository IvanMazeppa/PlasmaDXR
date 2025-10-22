#!/usr/bin/env python3
"""
Generate synthetic training data for initial model training.

This script creates a synthetic dataset based on performance models
to bootstrap the adaptive quality system before collecting real data.

Usage:
    python generate_training_data.py --output training_data/synthetic_performance_data.csv --samples 10000
"""

import numpy as np
import pandas as pd
import argparse
import os
from pathlib import Path


def generate_synthetic_data(num_samples=10000, random_seed=42):
    """
    Generate synthetic performance data based on cost models.

    Cost Model:
        frameTime = baseCost + particleCost + shadowCost + featureCost + distanceCost + noise
    """
    np.random.seed(random_seed)

    data = []

    for i in range(num_samples):
        # Sample scene parameters
        particleCount = np.random.choice([10000, 20000, 50000, 100000], p=[0.4, 0.3, 0.2, 0.1])
        lightCount = np.random.randint(1, 17)  # 1-16 lights
        cameraDistance = np.random.uniform(100, 2000)

        # Sample quality settings
        shadowRaysPerLight = np.random.choice([1, 2, 4, 8, 16], p=[0.3, 0.2, 0.25, 0.15, 0.1])
        useShadowRays = np.random.choice([0, 1], p=[0.2, 0.8])
        useInScattering = np.random.choice([0, 1], p=[0.9, 0.1])  # Rare (very expensive)
        usePhaseFunction = np.random.choice([0, 1], p=[0.3, 0.7])
        useAnisotropicGaussians = np.random.choice([0, 1], p=[0.4, 0.6])
        enableTemporalFiltering = np.random.choice([0, 1], p=[0.3, 0.7])
        useRTXDI = np.random.choice([0, 1], p=[0.7, 0.3])
        enableRTLighting = np.random.choice([0, 1], p=[0.2, 0.8])
        godRayDensity = np.random.uniform(0.0, 1.0)

        # --- Cost Model (empirical from PlasmaDX testing) ---

        # Base cost
        baseCost = 2.0

        # Particle cost (scales linearly)
        particleCost = (particleCount / 10000.0) * 0.8

        # Shadow ray cost (most expensive - scales with rays × lights)
        if useShadowRays:
            shadowCost = shadowRaysPerLight * lightCount * 0.35
        else:
            shadowCost = 0.0

        # RT lighting cost (particle-to-particle illumination)
        if enableRTLighting:
            rtCost = lightCount * 0.4
        else:
            rtCost = 0.0

        # Feature costs (additive)
        featureCost = 0.0
        if useInScattering:
            featureCost += 3.5  # Very expensive (volume scattering)
        if usePhaseFunction:
            featureCost += 0.6
        if useAnisotropicGaussians:
            featureCost += 0.4
        if enableTemporalFiltering:
            featureCost += 0.1  # Minimal cost
        if godRayDensity > 0:
            featureCost += godRayDensity * 2.5  # Proportional to density

        # Distance affects overdraw (close = more particles on screen)
        if cameraDistance < 400:
            distanceFactor = 1.8
        elif cameraDistance < 800:
            distanceFactor = 1.3
        else:
            distanceFactor = 1.0

        # RTXDI reduces cost (better light sampling)
        rtxdiMultiplier = 0.85 if useRTXDI else 1.0

        # Total frame time
        frameTime = (baseCost + particleCost + shadowCost + rtCost + featureCost) * distanceFactor * rtxdiMultiplier

        # Add realistic noise (±10%)
        noise = np.random.normal(0, frameTime * 0.1)
        frameTime += noise

        # Clamp to realistic range (1ms - 50ms)
        frameTime = np.clip(frameTime, 1.0, 50.0)

        # Store sample
        data.append({
            'particleCount': particleCount,
            'lightCount': lightCount,
            'cameraDistance': cameraDistance,
            'shadowRaysPerLight': shadowRaysPerLight,
            'useShadowRays': useShadowRays,
            'useInScattering': useInScattering,
            'usePhaseFunction': usePhaseFunction,
            'useAnisotropicGaussians': useAnisotropicGaussians,
            'enableTemporalFiltering': enableTemporalFiltering,
            'useRTXDI': useRTXDI,
            'enableRTLighting': enableRTLighting,
            'godRayDensity': godRayDensity,
            'frameTime': frameTime
        })

    return pd.DataFrame(data)


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic training data for adaptive quality ML')
    parser.add_argument('--output', type=str, default='training_data/synthetic_performance_data.csv',
                        help='Output CSV path')
    parser.add_argument('--samples', type=int, default=10000,
                        help='Number of samples to generate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    args = parser.parse_args()

    print(f"Generating {args.samples} synthetic performance samples...")

    # Generate data
    df = generate_synthetic_data(num_samples=args.samples, random_seed=args.seed)

    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Save to CSV
    df.to_csv(args.output, index=False)

    print(f"\nDataset saved to: {args.output}")
    print(f"Dataset shape: {df.shape}")
    print(f"\nSummary statistics:")
    print(df.describe())

    print(f"\nFrame time distribution:")
    print(f"  Min:    {df['frameTime'].min():.2f} ms")
    print(f"  25%:    {df['frameTime'].quantile(0.25):.2f} ms")
    print(f"  Median: {df['frameTime'].median():.2f} ms")
    print(f"  75%:    {df['frameTime'].quantile(0.75):.2f} ms")
    print(f"  Max:    {df['frameTime'].max():.2f} ms")

    print(f"\nTo train the model:")
    print(f"  python ml/train_adaptive_quality.py --data {args.output}")


if __name__ == '__main__':
    main()
