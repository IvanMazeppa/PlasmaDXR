#!/usr/bin/env python3
"""
Parameter-Conditioned PINN Training Script (Version 2)

Trains a Physics-Informed Neural Network that accepts physics parameters as inputs,
allowing runtime control without retraining.

Input: [r, θ, φ, v_r, v_θ, v_φ, t, M_bh_norm, α_visc, disk_thickness]
Output: [F_r, F_θ, F_φ]

The model learns to predict forces across a range of physics parameter values,
enabling real-time adjustment of black hole mass, viscosity, and disk thickness.

Usage:
    python train_pinn_v2.py --data training_data/ideal_accretion_disk.npz --epochs 1000
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import argparse
from pathlib import Path
import json
import time

# Physical constants
G = 6.674e-8
C = 2.998e10
M_SUN = 1.989e33
M_BH_DEFAULT = 10.0 * M_SUN
R_S = 2 * G * M_BH_DEFAULT / C**2
R_ISCO = 3 * R_S


class ParameterConditionedPINN(nn.Module):
    """
    Physics-Informed Neural Network with parameter conditioning.

    Takes particle state + physics parameters and predicts forces.
    The physics parameters allow runtime control without retraining.
    """

    def __init__(self, state_dim=7, param_dim=3, hidden_dim=128, num_layers=5):
        super().__init__()

        self.state_dim = state_dim
        self.param_dim = param_dim
        input_dim = state_dim + param_dim  # 7 state + 3 params = 10

        # Build network
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_dim, 3))  # Output: F_r, F_θ, F_φ

        self.network = nn.Sequential(*layers)

        # Initialize weights (Xavier for better gradient flow)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, state, params):
        """
        Forward pass.

        Args:
            state: (batch, 7) - [r, θ, φ, v_r, v_θ, v_φ, t]
            params: (batch, 3) - [M_bh_normalized, α_viscosity, disk_thickness]

        Returns:
            forces: (batch, 3) - [F_r, F_θ, F_φ]
        """
        x = torch.cat([state, params], dim=1)
        return self.network(x)


def physics_loss_keplerian(model, state, params, device):
    """
    Physics loss: Keplerian orbit stability.

    For circular orbits: F_r + v_φ²/r = 0 (gravity balanced by centrifugal)
    """
    state = state.requires_grad_(True)
    forces = model(state, params)

    r = state[:, 0:1]  # Keep dimension for broadcasting
    v_phi = state[:, 5:6]

    F_r = forces[:, 0:1]

    # Centrifugal acceleration
    a_centrifugal = v_phi**2 / (r + 1e-8)

    # For stable orbit: F_r + a_centrifugal ≈ 0
    # (gravity pulls in, centrifugal pushes out)
    residual = F_r + a_centrifugal

    return torch.mean(residual**2)


def physics_loss_angular_momentum(model, state, params, device):
    """
    Physics loss: Angular momentum conservation.

    For circular orbits without external torque: F_φ ≈ 0
    (small viscous torque allowed)
    """
    forces = model(state, params)
    F_phi = forces[:, 2]

    # Allow small torque (viscosity), penalize large torques
    return torch.mean(F_phi**2) * 0.1  # Weighted less than Keplerian


def physics_loss_vertical_confinement(model, state, params, device):
    """
    Physics loss: Vertical confinement (disk structure).

    Particles should experience restoring force toward disk midplane.
    F_θ should push particles toward θ = π/2 (equatorial plane).
    """
    forces = model(state, params)
    theta = state[:, 1]
    F_theta = forces[:, 1]

    # θ = π/2 is the disk midplane
    # F_θ should be negative when θ < π/2 (push up)
    # F_θ should be positive when θ > π/2 (push down)
    deviation = theta - np.pi / 2
    expected_sign = torch.sign(deviation)

    # F_theta should have same sign as deviation (restoring force)
    sign_violation = torch.relu(-F_theta * expected_sign)

    return torch.mean(sign_violation**2)


def generate_augmented_data(states, forces, num_param_variations=10):
    """
    Augment training data by generating variations with different physics parameters.

    This teaches the network how forces change with different parameter values.
    """
    print(f"Augmenting data with {num_param_variations} parameter variations...")

    augmented_states = []
    augmented_forces = []
    augmented_params = []

    for i in range(num_param_variations):
        # Random physics parameters
        M_bh_factor = np.random.uniform(0.5, 2.0)  # 0.5x to 2x default mass
        alpha_visc = np.random.uniform(0.01, 0.3)  # Viscosity range
        thickness = np.random.uniform(0.05, 0.2)   # H/R range

        # Scale forces based on mass (F ∝ M)
        scaled_forces = forces.copy()
        scaled_forces[:, 0] *= M_bh_factor  # F_r scales with mass

        # Create parameter array
        params = np.array([M_bh_factor, alpha_visc, thickness])
        params_batch = np.tile(params, (len(states), 1))

        augmented_states.append(states)
        augmented_forces.append(scaled_forces)
        augmented_params.append(params_batch)

    # Concatenate all variations
    all_states = np.vstack(augmented_states)
    all_forces = np.vstack(augmented_forces)
    all_params = np.vstack(augmented_params)

    print(f"  Total samples: {len(all_states)}")

    return all_states, all_forces, all_params


def train_model(
    model,
    train_loader,
    val_states,
    val_forces,
    val_params,
    device,
    epochs=1000,
    lr=0.001,
    physics_weight=0.1
):
    """
    Train the PINN model with combined data loss and physics loss.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.5)

    best_val_loss = float('inf')
    best_model_state = None

    print("\nTraining PINN v2 (parameter-conditioned)...")
    print(f"  Device: {device}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {lr}")
    print(f"  Physics loss weight: {physics_weight}")
    print()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_data_loss = 0.0
        epoch_physics_loss = 0.0

        for batch_states, batch_forces, batch_params in train_loader:
            batch_states = batch_states.to(device)
            batch_forces = batch_forces.to(device)
            batch_params = batch_params.to(device)

            optimizer.zero_grad()

            # Forward pass
            pred_forces = model(batch_states, batch_params)

            # Data loss (MSE)
            data_loss = nn.functional.mse_loss(pred_forces, batch_forces)

            # Physics losses
            phys_keplerian = physics_loss_keplerian(model, batch_states, batch_params, device)
            phys_angular = physics_loss_angular_momentum(model, batch_states, batch_params, device)
            phys_vertical = physics_loss_vertical_confinement(model, batch_states, batch_params, device)

            physics_loss = phys_keplerian + phys_angular + phys_vertical

            # Combined loss
            loss = data_loss + physics_weight * physics_loss

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_data_loss += data_loss.item()
            epoch_physics_loss += physics_loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(val_states, val_params)
            val_loss = nn.functional.mse_loss(val_pred, val_forces).item()

        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()

        # Log progress
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{epochs}: "
                  f"Loss={epoch_loss / len(train_loader):.6f}, "
                  f"Data={epoch_data_loss / len(train_loader):.6f}, "
                  f"Physics={epoch_physics_loss / len(train_loader):.6f}, "
                  f"Val={val_loss:.6f}")

    # Restore best model
    model.load_state_dict(best_model_state)
    print(f"\nTraining complete. Best validation loss: {best_val_loss:.6f}")

    return model


def export_to_onnx(model, output_path, device):
    """
    Export trained model to ONNX format for C++ inference.
    """
    model.eval()

    # Dummy inputs
    dummy_state = torch.randn(1, 7).to(device)
    dummy_params = torch.randn(1, 3).to(device)

    # Export
    torch.onnx.export(
        model,
        (dummy_state, dummy_params),
        output_path,
        input_names=['particle_state', 'physics_params'],
        output_names=['forces'],
        dynamic_axes={
            'particle_state': {0: 'batch'},
            'physics_params': {0: 'batch'},
            'forces': {0: 'batch'}
        },
        opset_version=11
    )

    print(f"Exported ONNX model to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Train parameter-conditioned PINN')
    parser.add_argument('--data', type=str, default='training_data/ideal_accretion_disk.npz',
                        help='Training data path')
    parser.add_argument('--output', type=str, default='models/pinn_v2_param_conditioned.onnx',
                        help='Output model path')
    parser.add_argument('--epochs', type=int, default=1000, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden layer dimension')
    parser.add_argument('--num_layers', type=int, default=5, help='Number of hidden layers')
    parser.add_argument('--physics_weight', type=float, default=0.1, help='Physics loss weight')
    parser.add_argument('--param_variations', type=int, default=10, help='Number of parameter variations')
    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Training data not found: {data_path}")
        print("Run: python generate_ideal_physics_data.py first")
        return

    print(f"Loading training data from: {data_path}")
    data = np.load(data_path)
    states = data['states_normalized']
    forces = data['forces_normalized']

    # Augment with parameter variations
    states, forces, params = generate_augmented_data(
        states, forces, num_param_variations=args.param_variations
    )

    # Split train/val
    n_samples = len(states)
    n_train = int(0.9 * n_samples)
    indices = np.random.permutation(n_samples)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    train_states = torch.FloatTensor(states[train_idx])
    train_forces = torch.FloatTensor(forces[train_idx])
    train_params = torch.FloatTensor(params[train_idx])

    val_states = torch.FloatTensor(states[val_idx]).to(device)
    val_forces = torch.FloatTensor(forces[val_idx]).to(device)
    val_params = torch.FloatTensor(params[val_idx]).to(device)

    # Create data loader
    train_dataset = TensorDataset(train_states, train_forces, train_params)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Create model
    model = ParameterConditionedPINN(
        state_dim=7,
        param_dim=3,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers
    ).to(device)

    print(f"\nModel architecture:")
    print(f"  Input: 7 (state) + 3 (params) = 10 dimensions")
    print(f"  Hidden: {args.num_layers} layers × {args.hidden_dim} neurons")
    print(f"  Output: 3 (F_r, F_θ, F_φ)")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Train
    start_time = time.time()
    model = train_model(
        model,
        train_loader,
        val_states,
        val_forces,
        val_params,
        device,
        epochs=args.epochs,
        lr=args.lr,
        physics_weight=args.physics_weight
    )
    train_time = time.time() - start_time
    print(f"Training time: {train_time / 60:.1f} minutes")

    # Export to ONNX
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_onnx(model, str(output_path), device)

    # Save config
    config = {
        'version': 2,
        'state_dim': 7,
        'param_dim': 3,
        'hidden_dim': args.hidden_dim,
        'num_layers': args.num_layers,
        'input_format': ['r', 'theta', 'phi', 'v_r', 'v_theta', 'v_phi', 't'],
        'param_format': ['M_bh_normalized', 'alpha_viscosity', 'disk_thickness'],
        'output_format': ['F_r', 'F_theta', 'F_phi'],
        'training_epochs': args.epochs,
        'physics_weight': args.physics_weight,
        'param_variations': args.param_variations
    }

    config_path = output_path.with_suffix('.config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Saved config to: {config_path}")
    print("\n✅ Training complete!")
    print(f"   Model: {output_path}")
    print(f"   Config: {config_path}")


if __name__ == '__main__':
    main()
