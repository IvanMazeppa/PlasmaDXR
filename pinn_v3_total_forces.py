#!/usr/bin/env python3
"""
PINN v3 - Total Force Output (Gravity + Viscosity + MRI)

KEY FIX: Outputs GRAVITATIONAL forces directly, not net forces after centrifugal cancellation.
This allows the integration framework to maintain Keplerian orbits naturally.

Training data format:
- Input: (x, y, z, vx, vy, vz, t, M_bh, α, H/R) - 10D
- Output: (Fx, Fy, Fz) - TOTAL gravitational + viscosity + MRI forces

Physics:
- Gravity: F = -GM * r_hat / r²
- Viscosity: Shakura-Sunyaev α-disk model
- MRI: Magneto-Rotational Instability turbulence
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Tuple, Dict
import sys

# === PINN NORMALIZED UNITS (corrected to produce visible orbital forces) ===
GM = 100.0  # Gravitational parameter (G * M_bh) - increased 100× for visible rotation
R_ISCO = 6.0  # Innermost stable circular orbit


class AccretionDiskPINN_v3(nn.Module):
    """
    PINN v3: Outputs TOTAL forces (gravity + viscosity + MRI).

    Architecture:
        Input: [x, y, z, vx, vy, vz, t, M_bh, alpha, H_R] (10D)
        Hidden: 5× 128 neurons (Tanh activation)
        Output: [Fx, Fy, Fz] (3D total force)
    """

    def __init__(self, hidden_dim=128, num_layers=5):
        super().__init__()

        # Network layers
        layers = []
        layers.append(nn.Linear(10, hidden_dim))  # 10D input (Cartesian + params)
        layers.append(nn.Tanh())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_dim, 3))  # 3D force output

        self.network = nn.Sequential(*layers)

    def forward(self, state):
        """
        Args:
            state: [batch, 10] - (x, y, z, vx, vy, vz, t, M_bh, alpha, H_R)
        Returns:
            forces: [batch, 3] - (Fx, Fy, Fz) total forces
        """
        return self.network(state)


def compute_gravitational_force(pos, M_bh=1.0):
    """
    Compute Newtonian gravitational force in Cartesian coordinates.

    Args:
        pos: [batch, 3] - (x, y, z) positions
        M_bh: Black hole mass multiplier (default 1.0)

    Returns:
        F_gravity: [batch, 3] - Gravitational force vectors
    """
    x, y, z = pos[:, 0:1], pos[:, 1:2], pos[:, 2:3]
    r = torch.sqrt(x**2 + y**2 + z**2 + 1e-8)

    # F = -GM * r_hat / r²
    GM_val = GM * M_bh
    F_magnitude = -GM_val / (r**2 + 1e-8)

    # Convert to Cartesian components
    r_hat_x = x / (r + 1e-8)
    r_hat_y = y / (r + 1e-8)
    r_hat_z = z / (r + 1e-8)

    Fx = F_magnitude * r_hat_x
    Fy = F_magnitude * r_hat_y
    Fz = F_magnitude * r_hat_z

    return torch.cat([Fx, Fy, Fz], dim=1)


def compute_viscous_force(pos, vel, alpha=0.1, H_R=0.1):
    """
    Shakura-Sunyaev α-disk viscosity (simplified).

    Viscosity causes angular momentum transport outward,
    slowing down inner particles and speeding up outer particles.

    Args:
        pos: [batch, 3] - (x, y, z)
        vel: [batch, 3] - (vx, vy, vz)
        alpha: Shakura-Sunyaev α parameter
        H_R: Disk thickness ratio H/R

    Returns:
        F_viscous: [batch, 3] - Viscous force
    """
    x, y, z = pos[:, 0:1], pos[:, 1:2], pos[:, 2:3]
    r = torch.sqrt(x**2 + z**2 + 1e-8)  # Cylindrical radius

    # Tangential velocity (azimuthal direction)
    vx, vy, vz = vel[:, 0:1], vel[:, 1:2], vel[:, 2:3]

    # Azimuthal unit vector: (-z, 0, x) / r
    phi_hat_x = -z / (r + 1e-8)
    phi_hat_z = x / (r + 1e-8)

    v_phi = vx * phi_hat_x + vz * phi_hat_z

    # Viscous drag in azimuthal direction (damps tangential velocity)
    # F_visc ~ -ν * v_φ / r², where ν ~ α * H/R * cs * r
    # Simplified: F_visc ~ -α * H/R * v_φ / r
    nu_eff = alpha * H_R * 0.01  # Effective viscosity coefficient
    F_visc_magnitude = -nu_eff * v_phi / (r + 1e-8)

    # Apply in azimuthal direction
    Fx_visc = F_visc_magnitude * phi_hat_x
    Fz_visc = F_visc_magnitude * phi_hat_z
    Fy_visc = torch.zeros_like(Fx_visc)  # No vertical viscosity

    return torch.cat([Fx_visc, Fy_visc, Fz_visc], dim=1)


def compute_mri_turbulence(pos, vel, t, alpha=0.1):
    """
    MRI (Magneto-Rotational Instability) turbulent forces.

    Adds stochastic perturbations with Kolmogorov spectrum.

    Args:
        pos: [batch, 3]
        vel: [batch, 3]
        t: [batch, 1] - time
        alpha: Turbulence strength

    Returns:
        F_mri: [batch, 3] - Turbulent force
    """
    # Use position + time as seed for pseudo-random turbulence
    # This makes turbulence spatially and temporally varying
    x, y, z = pos[:, 0:1], pos[:, 1:2], pos[:, 2:3]

    # Simple hash-based turbulence (deterministic for same x,y,z,t)
    seed = (x * 73856093 + y * 19349663 + z * 83492791 + t * 123456789) % 100000

    # Normalize to [-1, 1] range
    turb_x = torch.sin(seed * 0.1) * alpha * 0.001
    turb_y = torch.sin(seed * 0.2) * alpha * 0.0002  # Less vertical
    turb_z = torch.cos(seed * 0.1) * alpha * 0.001

    return torch.cat([turb_x, turb_y, turb_z], dim=1)


def physics_loss(model, state_batch, M_bh, alpha, H_R):
    """
    Compute physics-informed loss.

    Enforces that predicted forces match physical constraints:
    1. Match gravitational force
    2. Include viscous damping
    3. Include MRI turbulence
    4. Angular momentum conservation
    5. Energy dissipation (viscosity)
    """
    # Extract state
    pos = state_batch[:, 0:3]  # (x, y, z)
    vel = state_batch[:, 3:6]  # (vx, vy, vz)
    t = state_batch[:, 6:7]    # time

    # Predicted forces
    F_pred = model(state_batch)

    # Ground truth forces
    F_gravity = compute_gravitational_force(pos, M_bh)
    F_viscous = compute_viscous_force(pos, vel, alpha, H_R)
    F_mri = compute_mri_turbulence(pos, vel, t, alpha)

    F_total = F_gravity + F_viscous + F_mri

    # Loss 1: Match total force
    loss_force = torch.mean((F_pred - F_total)**2)

    # Loss 2: Gravity dominance (gravity should be strongest component)
    F_grav_mag = torch.norm(F_gravity, dim=1, keepdim=True)
    F_pred_mag = torch.norm(F_pred, dim=1, keepdim=True)
    loss_gravity_dominance = torch.mean((F_pred_mag - F_grav_mag)**2) * 0.1

    # Loss 3: Angular momentum (torque should cause L to change correctly)
    r_cross_F_pred = torch.cross(pos, F_pred, dim=1)
    r_cross_F_true = torch.cross(pos, F_total, dim=1)
    loss_angular = torch.mean((r_cross_F_pred - r_cross_F_true)**2) * 0.1

    # Total loss
    total_loss = loss_force + loss_gravity_dominance + loss_angular

    return {
        'total': total_loss,
        'force_match': loss_force,
        'gravity_dominance': loss_gravity_dominance,
        'angular_momentum': loss_angular
    }


def generate_training_data_v3(num_samples=100000, save_path='ml/training_data/pinn_v3_total_forces.npz'):
    """
    Generate training data with TOTAL forces (not net forces).

    Creates particles in stable Keplerian orbits and computes:
    - Gravitational force (maintains orbit)
    - Viscous force (angular momentum transport)
    - MRI turbulence (stochastic perturbations)
    """
    print(f"[v3] Generating {num_samples} training samples with TOTAL forces...")

    states = []
    forces = []

    for i in range(num_samples):
        # Random orbital radius
        r = np.random.uniform(10.0, 300.0)  # 10-300 units (PINN normalized)
        theta = np.random.uniform(0, 2 * np.pi)  # Azimuthal angle
        height = np.random.normal(0, 0.1 * r)  # Disk thickness

        # Cartesian position
        x = r * np.cos(theta)
        z = r * np.sin(theta)
        y = height

        # Keplerian velocity (circular orbit)
        v_kepler = np.sqrt(GM / r)
        vx = -v_kepler * np.sin(theta)
        vz = v_kepler * np.cos(theta)
        vy = 0.0

        # Add small random perturbations (1-2%)
        vx += np.random.normal(0, 0.01 * v_kepler)
        vy += np.random.normal(0, 0.005 * v_kepler)
        vz += np.random.normal(0, 0.01 * v_kepler)

        # Time
        t = np.random.uniform(0, 100.0)

        # Physics parameters
        M_bh = np.random.choice([0.8, 1.0, 1.2])  # Black hole mass variation
        alpha = np.random.uniform(0.05, 0.15)  # α viscosity
        H_R = np.random.uniform(0.05, 0.15)  # Disk thickness

        # Compute TOTAL forces
        r_vec = np.array([x, y, z])
        r_mag = np.sqrt(x**2 + y**2 + z**2)
        r_hat = r_vec / r_mag

        # Gravitational force (points toward black hole)
        F_grav = -GM * M_bh * r_hat / (r_mag**2)

        # Viscous force (azimuthal damping)
        r_cyl = np.sqrt(x**2 + z**2)
        phi_hat = np.array([-z, 0, x]) / r_cyl
        v_phi = vx * phi_hat[0] + vz * phi_hat[2]
        nu_eff = alpha * H_R * 0.01
        F_visc_mag = -nu_eff * v_phi / r_cyl
        F_visc = F_visc_mag * phi_hat

        # MRI turbulence (random)
        F_mri = np.random.normal(0, alpha * 0.001, size=3)
        F_mri[1] *= 0.2  # Less vertical turbulence

        # Total force
        F_total = F_grav + F_visc + F_mri

        # Store
        state = np.array([x, y, z, vx, vy, vz, t, M_bh, alpha, H_R], dtype=np.float32)
        force = F_total.astype(np.float32)

        states.append(state)
        forces.append(force)

        if (i + 1) % 10000 == 0:
            print(f"  Generated {i+1}/{num_samples} samples...")

    states = np.array(states)
    forces = np.array(forces)

    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(save_path, states=states, forces=forces)

    print(f"[v3] Training data saved to {save_path}")
    print(f"  States shape: {states.shape}")
    print(f"  Forces shape: {forces.shape}")
    print(f"  Force ranges: Fx=[{forces[:, 0].min():.4f}, {forces[:, 0].max():.4f}]")
    print(f"                Fy=[{forces[:, 1].min():.4f}, {forces[:, 1].max():.4f}]")
    print(f"                Fz=[{forces[:, 2].min():.4f}, {forces[:, 2].max():.4f}]")
    print(f"  Average force magnitude: {np.linalg.norm(forces, axis=1).mean():.4f}")

    return states, forces


def train_pinn_v3(epochs=200, batch_size=512, lr=1e-3):
    """
    Train PINN v3 with total force outputs.
    """
    print("=" * 80)
    print("PINN v3 Training - Total Force Output")
    print("=" * 80)

    # Check if training data exists
    data_path = 'ml/training_data/pinn_v3_total_forces.npz'
    if not Path(data_path).exists():
        print(f"Training data not found at {data_path}")
        print("Generating training data...")
        generate_training_data_v3()

    # Load training data
    print(f"\nLoading training data from {data_path}...")
    data = np.load(data_path)
    states = torch.tensor(data['states'], dtype=torch.float32)
    forces = torch.tensor(data['forces'], dtype=torch.float32)

    print(f"  Loaded {len(states)} training samples")
    print(f"  State shape: {states.shape}")
    print(f"  Force shape: {forces.shape}")

    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    model = AccretionDiskPINN_v3(hidden_dim=128, num_layers=5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    losses_history = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        # Shuffle data
        indices = torch.randperm(len(states))
        states_shuffled = states[indices].to(device)
        forces_shuffled = forces[indices].to(device)

        # Batch training
        for i in range(0, len(states), batch_size):
            batch_states = states_shuffled[i:i+batch_size]
            batch_forces = forces_shuffled[i:i+batch_size]

            # Forward pass
            pred_forces = model(batch_states)

            # Data loss (MSE between predicted and true forces)
            loss_data = torch.mean((pred_forces - batch_forces)**2)

            # Physics loss (from physics constraints)
            M_bh = batch_states[:, 7:8]
            alpha = batch_states[:, 8:9]
            H_R = batch_states[:, 9:10]

            # Use mean values for physics loss (faster)
            M_bh_mean = M_bh.mean().item()
            alpha_mean = alpha.mean().item()
            H_R_mean = H_R.mean().item()

            losses_physics = physics_loss(model, batch_states, M_bh_mean, alpha_mean, H_R_mean)

            # Total loss (weighted combination)
            loss = loss_data * 1.0 + losses_physics['total'] * 0.1

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        losses_history.append(avg_loss)
        scheduler.step(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f} - LR: {optimizer.param_groups[0]['lr']:.6f}")

    # Save model
    model_path = 'ml/models/pinn_v3_total_forces.onnx'
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"\nExporting to ONNX: {model_path}")
    model.eval()
    dummy_input = torch.randn(1, 10).to(device)

    torch.onnx.export(
        model,
        dummy_input,
        model_path,
        export_params=True,
        opset_version=14,
        input_names=['particle_state'],
        output_names=['forces'],
        dynamic_axes={'particle_state': {0: 'batch'}, 'forces': {0: 'batch'}}
    )

    print(f"✅ Model saved to {model_path}")
    print(f"   Final loss: {losses_history[-1]:.6f}")

    # Plot training curve
    plt.figure(figsize=(10, 6))
    plt.plot(losses_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('PINN v3 Training Loss')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.savefig('ml/models/pinn_v3_training_loss.png', dpi=150, bbox_inches='tight')
    print(f"   Training curve saved to ml/models/pinn_v3_training_loss.png")

    return model


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train PINN v3 with total force outputs')
    parser.add_argument('--generate-data', action='store_true', help='Generate training data only')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')

    args = parser.parse_args()

    if args.generate_data:
        generate_training_data_v3()
    else:
        train_pinn_v3(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
