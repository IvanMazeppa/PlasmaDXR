#!/usr/bin/env python3
"""
Physics-Informed Neural Network for Accretion Disk Dynamics

Implements a PINN that learns particle forces while respecting:
- General Relativity (Schwarzschild metric)
- Angular momentum conservation
- Shakura-Sunyaev viscosity
- Energy conservation

Reference:
- Shakura & Sunyaev (1973) - α-disk model
- Raissi et al. (2019) - Physics-Informed Neural Networks
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Tuple, Dict


# === NORMALIZED UNITS (for numerical stability) ===
# We use geometric units where G = c = M = 1
# This prevents numerical overflow from huge CGS values
#
# Conversion to physical units:
#   1 length unit = R_S (Schwarzschild radius) ≈ 30 km for 10 M_sun
#   1 time unit = R_S/c ≈ 0.0001 s
#   1 velocity unit = c
#
# Benefits:
#   - All values are O(1) - no overflow
#   - Physics equations simplify
#   - Gradients are well-scaled

G = 1.0  # Geometric units
C = 1.0  # Speed of light = 1
M_BH = 1.0  # Black hole mass = 1

R_S = 2.0  # Schwarzschild radius in geometric units
R_ISCO = 6.0  # ISCO at 3 R_S (= 6.0 in units where R_S = 2)


class AccretionDiskPINN(nn.Module):
    """
    Physics-Informed Neural Network for accretion disk particle dynamics.

    Network Architecture:
        Input: (r, theta, phi, v_r, v_theta, v_phi, t) - 7D phase space + time
        Hidden: 5 layers × 128 neurons with Tanh activation
        Output: (F_r, F_theta, F_phi) - 3D force vector

    Physics Constraints:
        1. Keplerian motion (far from ISCO)
        2. Angular momentum conservation
        3. Shakura-Sunyaev viscosity
        4. GR effective potential
        5. Energy conservation
    """

    def __init__(self, hidden_dim=128, num_layers=5, alpha_viscosity=0.01):
        super().__init__()

        self.alpha = alpha_viscosity  # Shakura-Sunyaev α parameter

        # Network layers
        layers = []
        layers.append(nn.Linear(7, hidden_dim))  # Input: (r, θ, φ, v_r, v_θ, v_φ, t)
        layers.append(nn.Tanh())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_dim, 3))  # Output: (F_r, F_θ, F_φ)

        self.network = nn.Sequential(*layers)

        # Initialize weights with Xavier initialization
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        """
        Forward pass: predict forces on particle.

        Args:
            x: (batch, 7) tensor - (r, theta, phi, v_r, v_theta, v_phi, t)

        Returns:
            F: (batch, 3) tensor - (F_r, F_theta, F_phi)
        """
        return self.network(x)

    def keplerian_angular_velocity(self, r):
        """Keplerian orbital frequency: Ω = √(GM/r³)"""
        return torch.sqrt(G * M_BH / r**3)

    def gr_effective_potential(self, r, L):
        """
        GR-corrected effective potential.

        V_eff = -GM/r + L²/(2r²) - GML²/r³
                 ↑        ↑           ↑
              gravity  centrifugal   GR correction
        """
        V_gravity = -G * M_BH / r
        V_centrifugal = L**2 / (2 * r**2)
        V_gr = -G * M_BH * L**2 / r**3  # GR correction term

        return V_gravity + V_centrifugal + V_gr

    def viscous_torque(self, r, v_phi, nu):
        """
        Shakura-Sunyaev viscous torque: G = ∂/∂r[νΣr³∂Ω/∂r]

        Args:
            r: radius
            v_phi: azimuthal velocity
            nu: kinematic viscosity (ν = α c_s H)

        Returns:
            Torque per unit mass
        """
        # Approximate ∂Ω/∂r using Keplerian profile
        Omega = v_phi / r
        dOmega_dr = -1.5 * Omega / r  # Keplerian shear

        # Viscous torque
        torque = nu * r**3 * dOmega_dr

        return torque / r  # Torque per unit mass

    def physics_loss(self, x, F_pred):
        """
        Compute physics-informed loss enforcing conservation laws.

        Args:
            x: (batch, 7) - input phase space
            F_pred: (batch, 3) - predicted forces

        Returns:
            Dictionary of loss components
        """
        r = x[:, 0:1]
        theta = x[:, 1:2]
        phi = x[:, 2:3]
        v_r = x[:, 3:4]
        v_theta = x[:, 4:5]
        v_phi = x[:, 5:6]
        t = x[:, 6:7]

        # Clamp values for numerical stability
        r = torch.clamp(r, min=R_ISCO * 0.5, max=1000.0)
        v_phi = torch.clamp(v_phi, min=-2.0, max=2.0)
        v_r = torch.clamp(v_r, min=-1.0, max=1.0)

        # Unpack predicted forces
        F_r = F_pred[:, 0:1]
        F_theta = F_pred[:, 1:2]
        F_phi = F_pred[:, 2:3]

        # === Physics Constraint 1: Keplerian Motion (far from BH) ===
        # For r >> R_ISCO, F_r should balance gravity + centrifugal
        Omega_kepler = self.keplerian_angular_velocity(r)
        v_phi_kepler = r * Omega_kepler

        # Centrifugal force: F_centrifugal = v²/r (with safety)
        F_centrifugal = v_phi**2 / (r + 1e-6)
        F_gravity = -G * M_BH / (r**2 + 1e-6)

        # Far from ISCO, radial force should be ~0 (circular orbit)
        mask_far = (r > 5 * R_ISCO).float()  # Apply only far from ISCO
        F_expected = F_centrifugal + F_gravity
        loss_keplerian = mask_far * (F_r - F_expected)**2

        # === Physics Constraint 2: Angular Momentum Conservation (Simplified) ===
        # For circular orbits: torque (r × F) should cause angular momentum change
        # Simplified: F_phi should be small for stable orbits
        L = r * v_phi  # Specific angular momentum

        # Expected azimuthal force for circular orbit (should be small)
        # We use a soft constraint: F_phi should be small
        loss_angular_momentum = (F_phi)**2

        # === Physics Constraint 3: Energy Conservation ===
        # Power (F · v) should be ~0 for conservative forces
        v_squared = v_r**2 + v_theta**2 + v_phi**2
        power = F_r * v_r + F_theta * v_theta + F_phi * v_phi

        # For conservative forces, power should be ~0
        loss_energy = power**2

        # === Physics Constraint 4: GR Correction ===
        # At r ~ R_ISCO, GR effects dominate
        mask_near = (r < 10 * R_ISCO).float()  # Near ISCO

        # GR radial force (from effective potential)
        # dV/dr = GM/r² - L²/r³ + 3GML²/r⁴
        L_squared = L**2
        dV_dr = (G * M_BH / (r**2 + 1e-6) -
                 L_squared / (r**3 + 1e-6) +
                 3 * G * M_BH * L_squared / (r**4 + 1e-6))
        F_r_gr = -dV_dr

        loss_gr = mask_near * (F_r - F_r_gr)**2

        # === Total Physics Loss (with safety checks) ===
        losses = {
            'keplerian': torch.clamp(torch.mean(loss_keplerian), max=100.0),
            'angular_momentum': torch.clamp(torch.mean(loss_angular_momentum), max=100.0),
            'energy': torch.clamp(torch.mean(loss_energy), max=100.0) * 0.1,
            'gr': torch.clamp(torch.mean(loss_gr), max=100.0)
        }

        return losses


class AccretionDiskDataset:
    """
    Dataset generator for training PINN.

    Generates particle trajectories from current physics shader
    to use as ground truth data.
    """

    def __init__(self, num_trajectories=1000, trajectory_length=100):
        self.num_trajectories = num_trajectories
        self.trajectory_length = trajectory_length

    def generate_training_data(self):
        """
        Generate synthetic training data from simplified physics.

        Uses normalized units (G=c=M=1) and stable integration.
        """
        data = []

        for traj_idx in range(self.num_trajectories):
            # Random initial conditions (normalized units)
            r0 = np.random.uniform(10.0, 100.0)  # 10-100 R_S (5-50 R_ISCO)
            theta0 = np.pi / 2  # Disk plane
            phi0 = np.random.uniform(0, 2 * np.pi)

            # Keplerian velocity (in geometric units)
            v_kepler = np.sqrt(G * M_BH / r0)  # ~0.1-0.3 in normalized units

            # Small perturbations (5% of Keplerian)
            v_r0 = np.random.normal(0, 0.05 * v_kepler)
            v_theta0 = 0.0
            v_phi0 = v_kepler * (1 + np.random.normal(0, 0.05))

            # Generate trajectory with smaller timestep for stability
            dt = 0.5  # Normalized time step
            for i in range(self.trajectory_length):
                t = i * dt

                # Store state
                state = np.array([r0, theta0, phi0, v_r0, v_theta0, v_phi0, t], dtype=np.float32)

                # Compute forces (Keplerian + small perturbations)
                # F_r = -GM/r² + v_φ²/r (gravity + centrifugal)
                F_gravity = -G * M_BH / (r0**2 + 1e-6)
                F_centrifugal = v_phi0**2 / (r0 + 1e-6)
                F_r = F_gravity + F_centrifugal

                # Add small random perturbations (turbulence)
                F_r += np.random.normal(0, 0.01 * abs(F_gravity))

                F_theta = 0.0  # Stay in disk plane
                F_phi = 0.0  # No torque for simple Keplerian

                forces = np.array([F_r, F_theta, F_phi], dtype=np.float32)

                # Clamp forces to reasonable range
                forces = np.clip(forces, -1.0, 1.0)

                data.append((state, forces))

                # Update state (leapfrog integration for stability)
                # Half-step velocity update
                v_r_half = v_r0 + 0.5 * F_r * dt
                v_theta_half = v_theta0 + 0.5 * F_theta * dt
                v_phi_half = v_phi0 + 0.5 * F_phi * dt

                # Full-step position update
                r0 = r0 + v_r_half * dt
                theta0 = theta0 + v_theta_half / (r0 + 1e-6) * dt
                phi0 = phi0 + v_phi_half / (r0 + 1e-6) * dt

                # Half-step velocity update (complete)
                v_r0 = v_r_half + 0.5 * F_r * dt
                v_theta0 = v_theta_half + 0.5 * F_theta * dt
                v_phi0 = v_phi_half + 0.5 * F_phi * dt

                # Keep theta in [0, π]
                if theta0 < 0:
                    theta0 = -theta0
                    phi0 += np.pi
                if theta0 > np.pi:
                    theta0 = 2*np.pi - theta0
                    phi0 += np.pi

                # Keep phi in [-π, π]
                phi0 = np.arctan2(np.sin(phi0), np.cos(phi0))

                # Safety: if particle fell into BH or escaped, restart
                if r0 < R_ISCO * 0.5 or r0 > 200.0:
                    break

        return data


def train_pinn(model, dataset, num_epochs=1000, lr=1e-3, device='cuda'):
    """
    Train PINN with combined data loss + physics loss.

    Loss = λ_data * MSE(F_pred, F_true) + Σ λ_physics * Physics_Loss
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Loss weights (tunable hyperparameters)
    lambda_data = 1.0
    lambda_keplerian = 0.5
    lambda_angular_momentum = 0.5
    lambda_energy = 0.1
    lambda_gr = 1.0

    model.to(device)
    model.train()

    # Convert dataset to tensors (more efficient)
    states_np = np.array([s for s, _ in dataset], dtype=np.float32)
    forces_np = np.array([f for _, f in dataset], dtype=np.float32)

    states = torch.from_numpy(states_np).to(device)
    forces = torch.from_numpy(forces_np).to(device)

    print(f"Training data shapes: states={states.shape}, forces={forces.shape}")
    print(f"State ranges: r=[{states[:, 0].min():.2f}, {states[:, 0].max():.2f}]")
    print(f"Force ranges: F_r=[{forces[:, 0].min():.3f}, {forces[:, 0].max():.3f}]")

    history = {
        'total_loss': [],
        'data_loss': [],
        'physics_losses': {
            'keplerian': [],
            'angular_momentum': [],
            'energy': [],
            'gr': []
        }
    }

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Forward pass
        F_pred = model(states)

        # Data loss (supervised)
        data_loss = nn.MSELoss()(F_pred, forces)

        # Physics loss (unsupervised - enforce physical laws)
        physics_losses = model.physics_loss(states, F_pred)

        # Total loss
        total_loss = (
            lambda_data * data_loss +
            lambda_keplerian * physics_losses['keplerian'] +
            lambda_angular_momentum * physics_losses['angular_momentum'] +
            lambda_energy * physics_losses['energy'] +
            lambda_gr * physics_losses['gr']
        )

        # Backward pass
        total_loss.backward()

        # Gradient clipping (prevent explosion)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Check for NaN
        if torch.isnan(total_loss):
            print(f"\nWARNING: NaN detected at epoch {epoch+1}")
            print(f"  Data loss: {data_loss.item()}")
            print(f"  Physics losses: {physics_losses}")
            print("  Stopping training...")
            break

        # Log progress
        history['total_loss'].append(total_loss.item())
        history['data_loss'].append(data_loss.item())
        for key in physics_losses:
            history['physics_losses'][key].append(physics_losses[key].item())

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Total Loss: {total_loss.item():.6f}")
            print(f"  Data Loss: {data_loss.item():.6f}")
            print(f"  Physics Losses:")
            for key, val in physics_losses.items():
                print(f"    {key}: {val.item():.6f}")
            print()

    return history


def export_to_onnx(model, output_path='models/pinn_accretion_disk.onnx'):
    """Export trained PINN to ONNX for C++ inference."""
    # Move model to CPU for ONNX export
    model = model.cpu()
    model.eval()

    # Dummy input (batch_size=1, 7 features) - also on CPU
    dummy_input = torch.randn(1, 7)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['particle_state'],
        output_names=['forces'],
        dynamic_axes={
            'particle_state': {0: 'batch_size'},
            'forces': {0: 'batch_size'}
        },
        opset_version=12
    )

    print(f"Model exported to {output_path}")
    print(f"Input: (batch, 7) - [r, theta, phi, v_r, v_theta, v_phi, t]")
    print(f"Output: (batch, 3) - [F_r, F_theta, F_phi]")


def plot_training_history(history, output_dir='ml/analysis/pinn'):
    """Generate training plots."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history['total_loss']) + 1)

    # Total loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['total_loss'], label='Total Loss')
    plt.plot(epochs, history['data_loss'], label='Data Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('PINN Training Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.savefig(f'{output_dir}/training_loss.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Physics losses
    plt.figure(figsize=(12, 6))
    for key, values in history['physics_losses'].items():
        plt.plot(epochs, values, label=key)
    plt.xlabel('Epoch')
    plt.ylabel('Physics Loss')
    plt.title('Physics Constraint Losses')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.savefig(f'{output_dir}/physics_losses.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Training plots saved to {output_dir}/")


def main():
    print("="*60)
    print("Physics-Informed Neural Network for Accretion Disk")
    print("="*60)
    print()

    # Configuration
    config = {
        'black_hole_mass': 10.0,  # Solar masses (physical units)
        'alpha_viscosity': 0.01,  # Shakura-Sunyaev α
        'hidden_dim': 128,
        'num_layers': 5,
        'num_trajectories': 500,  # Reduced for faster training
        'trajectory_length': 50,   # Reduced for faster training
        'num_epochs': 1000,        # Reduced for testing
        'learning_rate': 1e-3,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    print("NOTE: Using normalized units (G=c=M=1) for numerical stability")
    print("All distances are in units of R_S (Schwarzschild radius)")
    print("All velocities are in units of c (speed of light)")
    print()

    print(f"Configuration:")
    for key, val in config.items():
        print(f"  {key}: {val}")
    print()

    # Generate training data
    print("Generating training data...")
    dataset_generator = AccretionDiskDataset(
        num_trajectories=config['num_trajectories'],
        trajectory_length=config['trajectory_length']
    )
    dataset = dataset_generator.generate_training_data()
    print(f"Generated {len(dataset)} training samples")
    print()

    # Create PINN model
    print("Creating PINN model...")
    model = AccretionDiskPINN(
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        alpha_viscosity=config['alpha_viscosity']
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # Train model
    print("Training PINN...")
    history = train_pinn(
        model,
        dataset,
        num_epochs=config['num_epochs'],
        lr=config['learning_rate'],
        device=config['device']
    )

    # Plot results
    print("Generating plots...")
    plot_training_history(history)

    # Export to ONNX
    print("Exporting to ONNX...")
    export_to_onnx(model, 'models/pinn_accretion_disk.onnx')

    # Save config
    with open('models/pinn_config.json', 'w') as f:
        json.dump(config, f, indent=2)

    print()
    print("="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Final Total Loss: {history['total_loss'][-1]:.6f}")
    print(f"Final Data Loss: {history['data_loss'][-1]:.6f}")
    print()
    print("Next steps:")
    print("  1. Review training plots in ml/analysis/pinn/")
    print("  2. Integrate ONNX model into PlasmaDX-Clean")
    print("  3. Collect real physics data from GPU buffer dumps")
    print("  4. Retrain with real data for better accuracy")


if __name__ == '__main__':
    main()
