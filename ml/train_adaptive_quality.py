#!/usr/bin/env python3
"""
Adaptive Quality ML Training Pipeline for PlasmaDX-Clean

Trains a model to predict frame time based on scene complexity and quality settings.
Exports a lightweight decision tree for C++ inference.

Usage:
    python train_adaptive_quality.py --data training_data/performance_data.csv --output models/adaptive_quality.model
"""

import numpy as np
import pandas as pd
import argparse
import struct
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class AdaptiveQualityTrainer:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'particleCount', 'lightCount', 'cameraDistance', 'shadowRaysPerLight',
            'useShadowRays', 'useInScattering', 'usePhaseFunction', 'useAnisotropicGaussians',
            'enableTemporalFiltering', 'useRTXDI', 'enableRTLighting', 'godRayDensity'
        ]
        self.target_name = 'frameTime'
        # Store training data for ensemble-to-tree conversion
        self.X_train_scaled = None
        self.y_train = None

    def load_data(self, csv_path):
        """Load performance data from CSV"""
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)

        print(f"Dataset shape: {df.shape}")
        print(f"Samples: {len(df)}")
        print(f"\nFeature ranges:")
        print(df[self.feature_names].describe())
        print(f"\nTarget range (frameTime):")
        print(df[self.target_name].describe())

        return df

    def preprocess_data(self, df):
        """Prepare features and target"""
        X = df[self.feature_names].values
        y = df[self.target_name].values

        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Normalize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test

    def train_model(self, X_train, y_train, model_type='gradient_boosting'):
        """Train regression model"""
        print(f"\nTraining {model_type} model...")

        # Store training data for later use (ensemble-to-tree conversion)
        self.X_train_scaled = X_train
        self.y_train = y_train

        if model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            )
        elif model_type == 'decision_tree':
            # Simple decision tree (easier to export to C++)
            self.model = DecisionTreeRegressor(
                max_depth=8,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        self.model.fit(X_train, y_train)
        print(f"Model trained successfully")

    def evaluate_model(self, X_test, y_test):
        """Evaluate model performance"""
        y_pred = self.model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print(f"\nModel Performance:")
        print(f"  MAE:  {mae:.3f} ms")
        print(f"  RMSE: {rmse:.3f} ms")
        print(f"  R²:   {r2:.3f}")

        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'y_pred': y_pred,
            'y_test': y_test
        }

    def analyze_feature_importance(self):
        """Analyze which features matter most"""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]

            print(f"\nFeature Importance:")
            for i in indices:
                print(f"  {self.feature_names[i]:30s}: {importances[i]:.4f}")

            return importances
        else:
            print("Model doesn't support feature importance")
            return None

    def export_model_cpp(self, output_path):
        """Export model to C++ compatible binary format"""
        print(f"\nExporting model to {output_path}...")

        # Get the tree to export
        if isinstance(self.model, DecisionTreeRegressor):
            # Direct export
            tree = self.model.tree_
            print("Exporting DecisionTreeRegressor directly...")
        else:
            # Need to convert ensemble to single tree
            print("WARNING: Only DecisionTreeRegressor can be exported to C++")
            print("Converting ensemble model to single decision tree...")
            print("(This approximates the ensemble with a simpler tree)")

            if self.X_train_scaled is None or self.y_train is None:
                raise ValueError("Training data not available for ensemble-to-tree conversion")

            # Train a decision tree to approximate the ensemble model
            # Use the ensemble's predictions as targets (model distillation)
            y_ensemble_predictions = self.model.predict(self.X_train_scaled)

            simple_tree = DecisionTreeRegressor(
                max_depth=10,  # Deeper tree to better approximate ensemble
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42
            )
            simple_tree.fit(self.X_train_scaled, y_ensemble_predictions)

            # Check approximation quality
            tree_predictions = simple_tree.predict(self.X_train_scaled)
            approximation_error = np.mean(np.abs(y_ensemble_predictions - tree_predictions))
            print(f"Approximation MAE: {approximation_error:.3f} ms")
            print(f"Tree depth: {simple_tree.get_depth()}, Nodes: {simple_tree.tree_.node_count}")

            tree = simple_tree.tree_

        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'wb') as f:
            # Write number of nodes
            node_count = tree.node_count
            f.write(struct.pack('I', node_count))

            # Write decision tree nodes
            # DecisionNode struct: int featureIndex, float threshold, int left, int right, float prediction
            for i in range(node_count):
                feature_idx = tree.feature[i]
                threshold = tree.threshold[i]
                left_child = tree.children_left[i]
                right_child = tree.children_right[i]

                # Leaf nodes have feature = -2
                if feature_idx == -2:
                    feature_idx = -1
                    prediction = tree.value[i][0]
                else:
                    prediction = 0.0

                # Pack struct (20 bytes): int, float, int, int, float
                f.write(struct.pack('ifiif', feature_idx, threshold, left_child, right_child, prediction))

            # Write feature scaling parameters (12 features × 8 bytes = 96 bytes)
            for i in range(12):
                mean = self.scaler.mean_[i]
                std = self.scaler.scale_[i]
                f.write(struct.pack('ff', mean, std))

        print(f"Model exported: {node_count} nodes, {os.path.getsize(output_path)} bytes")

    def plot_results(self, results, output_dir):
        """Generate analysis plots"""
        print(f"\nGenerating plots in {output_dir}...")
        os.makedirs(output_dir, exist_ok=True)

        # Prediction vs Actual
        plt.figure(figsize=(10, 6))
        plt.scatter(results['y_test'], results['y_pred'], alpha=0.5)
        plt.plot([results['y_test'].min(), results['y_test'].max()],
                 [results['y_test'].min(), results['y_test'].max()],
                 'r--', lw=2)
        plt.xlabel('Actual Frame Time (ms)')
        plt.ylabel('Predicted Frame Time (ms)')
        plt.title(f'Frame Time Prediction (R² = {results["r2"]:.3f})')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{output_dir}/prediction_vs_actual.png", dpi=150, bbox_inches='tight')
        plt.close()

        # Residuals
        residuals = results['y_test'] - results['y_pred']
        plt.figure(figsize=(10, 6))
        plt.scatter(results['y_pred'], residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--', lw=2)
        plt.xlabel('Predicted Frame Time (ms)')
        plt.ylabel('Residual (ms)')
        plt.title('Residual Plot')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{output_dir}/residuals.png", dpi=150, bbox_inches='tight')
        plt.close()

        # Feature importance (if available)
        importances = self.analyze_feature_importance()
        if importances is not None:
            plt.figure(figsize=(12, 6))
            indices = np.argsort(importances)[::-1]
            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(range(len(importances)),
                       [self.feature_names[i] for i in indices],
                       rotation=45, ha='right')
            plt.ylabel('Importance')
            plt.title('Feature Importance')
            plt.tight_layout()
            plt.savefig(f"{output_dir}/feature_importance.png", dpi=150, bbox_inches='tight')
            plt.close()

        print("Plots generated successfully")


def main():
    parser = argparse.ArgumentParser(description='Train adaptive quality ML model for PlasmaDX-Clean')
    parser.add_argument('--data', type=str, default='training_data/performance_data.csv',
                        help='Path to training data CSV')
    parser.add_argument('--output', type=str, default='models/adaptive_quality.model',
                        help='Output model path')
    parser.add_argument('--model', type=str, default='gradient_boosting',
                        choices=['random_forest', 'gradient_boosting', 'decision_tree'],
                        help='Model type')
    parser.add_argument('--plots', type=str, default='analysis',
                        help='Output directory for plots')

    args = parser.parse_args()

    # Create trainer
    trainer = AdaptiveQualityTrainer()

    # Load data
    df = trainer.load_data(args.data)

    # Preprocess
    X_train, X_test, y_train, y_test = trainer.preprocess_data(df)

    # Train
    trainer.train_model(X_train, y_train, model_type=args.model)

    # Evaluate
    results = trainer.evaluate_model(X_test, y_test)

    # Analyze
    trainer.analyze_feature_importance()

    # Export to C++
    trainer.export_model_cpp(args.output)

    # Generate plots
    trainer.plot_results(results, args.plots)

    print("\n=== Training Complete ===")
    print(f"Model saved to: {args.output}")
    print(f"Analysis plots in: {args.plots}")
    print(f"\nTo use in PlasmaDX-Clean:")
    print(f"  1. Copy {args.output} to ml/models/")
    print(f"  2. Enable adaptive quality in ImGui")
    print(f"  3. Set target FPS (60/120/144)")


if __name__ == '__main__':
    main()
