#!/usr/bin/env python3
"""
Test trained adaptive quality model with sample scenarios.

Usage:
    python test_model.py --model models/adaptive_quality.model
"""

import argparse
import struct
import numpy as np
from pathlib import Path


class ModelTester:
    def __init__(self, model_path):
        self.model_path = model_path
        self.decision_tree = []
        self.feature_scaling = []
        self.feature_names = [
            'particleCount', 'lightCount', 'cameraDistance', 'shadowRaysPerLight',
            'useShadowRays', 'useInScattering', 'usePhaseFunction', 'useAnisotropicGaussians',
            'enableTemporalFiltering', 'useRTXDI', 'enableRTLighting', 'godRayDensity'
        ]

    def load_model(self):
        """Load binary model file"""
        with open(self.model_path, 'rb') as f:
            # Read node count
            node_count = struct.unpack('I', f.read(4))[0]
            print(f"Loading model: {node_count} nodes")

            # Read decision tree nodes
            for i in range(node_count):
                feature_idx, threshold, left, right, prediction = struct.unpack('ifiif', f.read(20))
                self.decision_tree.append({
                    'feature_idx': feature_idx,
                    'threshold': threshold,
                    'left': left,
                    'right': right,
                    'prediction': prediction
                })

            # Read feature scaling
            for i in range(12):
                mean, std = struct.unpack('ff', f.read(8))
                self.feature_scaling.append({'mean': mean, 'std': std})

        print(f"Model loaded successfully")
        return True

    def normalize_features(self, features):
        """Normalize feature array"""
        normalized = []
        for i, value in enumerate(features):
            mean = self.feature_scaling[i]['mean']
            std = self.feature_scaling[i]['std']
            normalized.append((value - mean) / std)
        return normalized

    def predict(self, features):
        """Traverse decision tree to predict frame time"""
        normalized = self.normalize_features(features)

        # Traverse tree
        node_idx = 0
        while node_idx >= 0 and node_idx < len(self.decision_tree):
            node = self.decision_tree[node_idx]

            if node['feature_idx'] == -1:
                # Leaf node - return prediction
                return node['prediction']

            # Internal node - traverse
            if normalized[node['feature_idx']] <= node['threshold']:
                node_idx = node['left']
            else:
                node_idx = node['right']

        print("ERROR: Tree traversal failed")
        return 0.0

    def test_scenario(self, name, features):
        """Test a specific scenario"""
        prediction = self.predict(features)
        print(f"\n{name}")
        print(f"  Predicted Frame Time: {prediction:.2f} ms")
        print(f"  Predicted FPS: {1000.0/prediction:.1f}")
        return prediction


def main():
    parser = argparse.ArgumentParser(description='Test adaptive quality model')
    parser.add_argument('--model', type=str, default='models/adaptive_quality.model',
                        help='Path to trained model')
    args = parser.parse_args()

    # Check if model exists
    if not Path(args.model).exists():
        print(f"ERROR: Model not found: {args.model}")
        print("Run: python train_adaptive_quality.py")
        return

    # Create tester
    tester = ModelTester(args.model)
    tester.load_model()

    print("\n" + "="*60)
    print("Testing Adaptive Quality Model")
    print("="*60)

    # Test scenarios
    scenarios = [
        {
            'name': 'Scenario 1: Low Quality (10K particles, 1 shadow ray)',
            'features': [
                10000,  # particleCount
                13,     # lightCount
                800,    # cameraDistance
                1,      # shadowRaysPerLight
                1,      # useShadowRays
                0,      # useInScattering
                1,      # usePhaseFunction
                1,      # useAnisotropicGaussians
                1,      # enableTemporalFiltering
                0,      # useRTXDI
                1,      # enableRTLighting
                0.0     # godRayDensity
            ]
        },
        {
            'name': 'Scenario 2: Medium Quality (10K particles, 4 shadow rays)',
            'features': [
                10000,  # particleCount
                13,     # lightCount
                800,    # cameraDistance
                4,      # shadowRaysPerLight
                1,      # useShadowRays
                0,      # useInScattering
                1,      # usePhaseFunction
                1,      # useAnisotropicGaussians
                1,      # enableTemporalFiltering
                0,      # useRTXDI
                1,      # enableRTLighting
                0.2     # godRayDensity
            ]
        },
        {
            'name': 'Scenario 3: High Quality (10K particles, 8 shadow rays)',
            'features': [
                10000,  # particleCount
                13,     # lightCount
                800,    # cameraDistance
                8,      # shadowRaysPerLight
                1,      # useShadowRays
                0,      # useInScattering
                1,      # usePhaseFunction
                1,      # useAnisotropicGaussians
                1,      # enableTemporalFiltering
                0,      # useRTXDI
                1,      # enableRTLighting
                0.5     # godRayDensity
            ]
        },
        {
            'name': 'Scenario 4: Ultra Quality (10K particles, 16 shadow rays + all features)',
            'features': [
                10000,  # particleCount
                13,     # lightCount
                800,    # cameraDistance
                16,     # shadowRaysPerLight
                1,      # useShadowRays
                1,      # useInScattering (EXPENSIVE!)
                1,      # usePhaseFunction
                1,      # useAnisotropicGaussians
                1,      # enableTemporalFiltering
                0,      # useRTXDI
                1,      # enableRTLighting
                1.0     # godRayDensity (max)
            ]
        },
        {
            'name': 'Scenario 5: Stress Test (100K particles, 8 shadow rays)',
            'features': [
                100000, # particleCount
                13,     # lightCount
                800,    # cameraDistance
                8,      # shadowRaysPerLight
                1,      # useShadowRays
                0,      # useInScattering
                1,      # usePhaseFunction
                1,      # useAnisotropicGaussians
                1,      # enableTemporalFiltering
                0,      # useRTXDI
                1,      # enableRTLighting
                0.5     # godRayDensity
            ]
        },
        {
            'name': 'Scenario 6: RTXDI Mode (10K particles, 4 shadow rays)',
            'features': [
                10000,  # particleCount
                13,     # lightCount
                800,    # cameraDistance
                4,      # shadowRaysPerLight
                1,      # useShadowRays
                0,      # useInScattering
                1,      # usePhaseFunction
                1,      # useAnisotropicGaussians
                1,      # enableTemporalFiltering
                1,      # useRTXDI (enabled)
                1,      # enableRTLighting
                0.2     # godRayDensity
            ]
        }
    ]

    # Run tests
    for scenario in scenarios:
        tester.test_scenario(scenario['name'], scenario['features'])

    print("\n" + "="*60)
    print("Quality Level Recommendations (Target: 120 FPS = 8.33ms)")
    print("="*60)

    # Recommend quality levels
    target_fps = 120.0
    target_time = 1000.0 / target_fps

    quality_levels = [
        ('Ultra', scenarios[3]['features']),
        ('High', scenarios[2]['features']),
        ('Medium', scenarios[1]['features']),
        ('Low', scenarios[0]['features'])
    ]

    print(f"\nFor {scenarios[0]['features'][0]:.0f} particles at {scenarios[0]['features'][2]:.0f} units:")
    for name, features in quality_levels:
        pred_time = tester.predict(features)
        status = "✓ MEETS TARGET" if pred_time <= target_time * 1.1 else "✗ TOO SLOW"
        print(f"  {name:8s}: {pred_time:5.2f}ms ({1000.0/pred_time:5.1f} FPS) {status}")

    print(f"\nFor {scenarios[4]['features'][0]:.0f} particles at {scenarios[4]['features'][2]:.0f} units:")
    for name, features in quality_levels:
        # Scale for 100K particles
        scaled_features = features.copy()
        scaled_features[0] = 100000  # particleCount
        pred_time = tester.predict(scaled_features)
        status = "✓ MEETS TARGET" if pred_time <= target_time * 1.1 else "✗ TOO SLOW"
        print(f"  {name:8s}: {pred_time:5.2f}ms ({1000.0/pred_time:5.1f} FPS) {status}")

    print("\n" + "="*60)
    print("Model test complete!")
    print("="*60)


if __name__ == '__main__':
    main()
