#!/usr/bin/env python3
"""
PINN v4 - Orbital Physics + Turbulence-Robust Training

KEY IMPROVEMENTS over v3:
1. Trained on WIDER velocity distribution (not just Keplerian)
2. Includes velocity perturbations up to ±30% from Keplerian
3. Forces remain physically correct even with non-circular orbits
4. Designed to work WITH separate SIREN turbulence model

The model learns: "Given ANY position and velocity, what is the physical force?"
Not just: "Given Keplerian orbit, what is the force?"

Training data covers:
- Circular orbits (Keplerian)
- Elliptical orbits (eccentric)
- Infalling particles
- Outward-moving particles
- Vertically oscillating particles
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict

# === PHYSICS CONSTANTS ===
GM = 100.0          # Gravitational parameter (must match C++ PINN_GM)
R_ISCO = 6.0        # Innermost stable circular orbit
R_INNER = 10.0      # Inner disk radius for training
R_OUTER = 300.0     # Outer disk radius for training

# Velocity perturbation range (fraction of Keplerian)
V_PERTURB_MIN = 0.5   # 50% of Keplerian (highly sub-Keplerian)
V_PERTURB_MAX = 1.5   # 150% of Keplerian (highly super-Keplerian)


class AccretionDiskPINN_v4(nn.Module):
    """
    PINN v4: Robust to non-Keplerian velocities.
    
    Same architecture as v3 but trained on diverse velocity states.
    """
    
    def __init__(self, hidden_dim=128, num_layers=5):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(10, hidden_dim))
        layers.append(nn.Tanh())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        
        layers.append(nn.Linear(hidden_dim, 3))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state):
        return self.network(state)


def generate_training_data_v4(num_samples=200000, save_path='ml/training_data/pinn_v4_turbulence_robust.npz'):
    """
    Generate training data with DIVERSE velocity states.
    
    Unlike v3 (only Keplerian velocities), v4 includes:
    - Circular orbits (Keplerian)
    - Elliptical orbits (eccentric)
    - Radially infalling/outflowing particles
    - Vertically oscillating particles
    - Random velocity perturbations
    """
    print(f"[v4] Generating {num_samples} TURBULENCE-ROBUST training samples...")
    print(f"     Velocity range: {V_PERTURB_MIN:.0%} to {V_PERTURB_MAX:.0%} of Keplerian")
    
    states = []
    forces = []
    
    np.random.seed(42)  # Reproducibility
    
    for i in range(num_samples):
        # Random orbital radius
        r = np.random.uniform(R_INNER, R_OUTER)
        theta = np.random.uniform(0, 2 * np.pi)
        
        # Disk thickness (Gaussian around midplane)
        H_R_sample = np.random.uniform(0.05, 0.15)
        height = np.random.normal(0, H_R_sample * r)
        
        # Cartesian position
        x = r * np.cos(theta)
        z = r * np.sin(theta)
        y = height
        
        # === DIVERSE VELOCITY SAMPLING ===
        v_kepler = np.sqrt(GM / r)
        
        # Choose velocity type
        velocity_type = np.random.choice([
            'keplerian',      # 30% - circular orbit
            'elliptical',     # 25% - eccentric orbit
            'perturbed',      # 25% - random perturbation
            'radial_motion',  # 10% - infalling/outflowing
            'vertical_osc',   # 10% - vertical oscillation
        ], p=[0.30, 0.25, 0.25, 0.10, 0.10])
        
        if velocity_type == 'keplerian':
            # Perfect circular orbit
            v_scale = 1.0
            vr = 0.0
            vy = 0.0
            
        elif velocity_type == 'elliptical':
            # Elliptical orbit: v varies from 0.7 to 1.3 Keplerian
            eccentricity = np.random.uniform(0.1, 0.4)
            true_anomaly = np.random.uniform(0, 2 * np.pi)
            v_scale = np.sqrt((1 + eccentricity * np.cos(true_anomaly)) / (1 - eccentricity**2))
            # Radial velocity component (non-zero for eccentric orbits)
            vr = v_kepler * eccentricity * np.sin(true_anomaly)
            vy = np.random.normal(0, 0.02 * v_kepler)
            
        elif velocity_type == 'perturbed':
            # Random perturbation (like after turbulence kick)
            v_scale = np.random.uniform(V_PERTURB_MIN, V_PERTURB_MAX)
            vr = np.random.normal(0, 0.2 * v_kepler)  # Random radial motion
            vy = np.random.normal(0, 0.1 * v_kepler)  # Random vertical motion
            
        elif velocity_type == 'radial_motion':
            # Infalling or outflowing
            v_scale = np.random.uniform(0.7, 1.1)
            vr = np.random.uniform(-0.5, 0.3) * v_kepler  # Mostly infall
            vy = np.random.normal(0, 0.05 * v_kepler)
            
        else:  # vertical_osc
            # Vertical oscillation (disk breathing mode)
            v_scale = np.random.uniform(0.95, 1.05)
            vr = np.random.normal(0, 0.05 * v_kepler)
            vy = np.random.uniform(-0.3, 0.3) * v_kepler
        
        # Tangential velocity (azimuthal)
        v_phi = v_kepler * v_scale
        
        # Convert to Cartesian velocities
        # Tangential direction: (-sin(θ), 0, cos(θ))
        vx = -v_phi * np.sin(theta) + vr * np.cos(theta)
        vz = v_phi * np.cos(theta) + vr * np.sin(theta)
        
        # Time (for temporal coherence in MRI)
        t = np.random.uniform(0, 100.0)
        
        # Physics parameters
        M_bh = np.random.choice([0.8, 1.0, 1.2])
        alpha = np.random.uniform(0.05, 0.15)
        H_R = H_R_sample
        
        # === COMPUTE PHYSICAL FORCES ===
        # These are the TRUE forces regardless of velocity state
        
        r_vec = np.array([x, y, z])
        r_mag = np.sqrt(x**2 + y**2 + z**2)
        r_hat = r_vec / (r_mag + 1e-8)
        
        # 1. Gravitational force (always toward center)
        F_grav = -GM * M_bh * r_hat / (r_mag**2)
        
        # 2. Viscous force (depends on actual velocity, not Keplerian)
        r_cyl = np.sqrt(x**2 + z**2)
        if r_cyl > 1e-6:
            phi_hat = np.array([-z, 0, x]) / r_cyl
            v_phi_actual = vx * phi_hat[0] + vz * phi_hat[2]
            nu_eff = alpha * H_R * 0.01
            F_visc_mag = -nu_eff * v_phi_actual / r_cyl
            F_visc = F_visc_mag * phi_hat
        else:
            F_visc = np.zeros(3)
        
        # 3. Vertical restoring force (disk gravity toward midplane)
        omega_kepler = np.sqrt(GM * M_bh / r_mag**3)
        F_vertical = -omega_kepler**2 * y * np.array([0, 1, 0]) * 0.5
        
        # 4. MRI turbulence (small random component)
        F_mri = np.random.normal(0, alpha * 0.001, size=3)
        F_mri[1] *= 0.2  # Less vertical
        
        # Total force
        F_total = F_grav + F_visc + F_vertical + F_mri
        
        # Store sample
        state = np.array([x, y, z, vx, vy, vz, t, M_bh, alpha, H_R], dtype=np.float32)
        force = F_total.astype(np.float32)
        
        states.append(state)
        forces.append(force)
        
        if (i + 1) % 20000 == 0:
            print(f"  Generated {i+1}/{num_samples} samples...")
    
    states = np.array(states)
    forces = np.array(forces)
    
    # Compute statistics
    v_mags = np.sqrt(states[:, 3]**2 + states[:, 4]**2 + states[:, 5]**2)
    r_mags = np.sqrt(states[:, 0]**2 + states[:, 1]**2 + states[:, 2]**2)
    v_kep_expected = np.sqrt(GM / r_mags)
    v_ratio = v_mags / v_kep_expected
    
    print(f"\n[v4] Training data generated!")
    print(f"  Samples: {len(states)}")
    print(f"  Position range: r = [{r_mags.min():.1f}, {r_mags.max():.1f}]")
    print(f"  Velocity ratio (v/v_kepler): [{v_ratio.min():.2f}, {v_ratio.max():.2f}]")
    print(f"  Force magnitude: [{np.linalg.norm(forces, axis=1).min():.4f}, {np.linalg.norm(forces, axis=1).max():.4f}]")
    
    # Save
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(save_path, states=states, forces=forces)
    print(f"\n  Saved to: {save_path}")
    
    return states, forces


def train_pinn_v4(epochs=300, batch_size=1024, lr=1e-3):
    """Train PINN v4 with turbulence-robust data."""
    
    print("=" * 80)
    print("PINN v4 Training - Turbulence-Robust Model")
    print("=" * 80)
    
    data_path = 'ml/training_data/pinn_v4_turbulence_robust.npz'
    if not Path(data_path).exists():
        print("Training data not found, generating...")
        generate_training_data_v4()
    
    # Load data
    data = np.load(data_path)
    states = torch.tensor(data['states'], dtype=torch.float32)
    forces = torch.tensor(data['forces'], dtype=torch.float32)
    
    print(f"\nLoaded {len(states)} samples")
    
    # Split train/val
    n_val = len(states) // 10
    indices = torch.randperm(len(states))
    train_states = states[indices[n_val:]]
    train_forces = forces[indices[n_val:]]
    val_states = states[indices[:n_val]]
    val_forces = forces[indices[:n_val]]
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model
    model = AccretionDiskPINN_v4(hidden_dim=128, num_layers=5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=15, factor=0.5)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training
    best_val_loss = float('inf')
    best_state = None
    history = {'train': [], 'val': []}
    
    for epoch in range(epochs):
        model.train()
        
        # Shuffle
        perm = torch.randperm(len(train_states))
        train_states_shuffled = train_states[perm].to(device)
        train_forces_shuffled = train_forces[perm].to(device)
        
        epoch_loss = 0
        n_batches = 0
        
        for i in range(0, len(train_states), batch_size):
            batch_s = train_states_shuffled[i:i+batch_size]
            batch_f = train_forces_shuffled[i:i+batch_size]
            
            pred = model(batch_s)
            loss = torch.mean((pred - batch_f)**2)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        train_loss = epoch_loss / n_batches
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(val_states.to(device))
            val_loss = torch.mean((val_pred - val_forces.to(device))**2).item()
        
        scheduler.step(val_loss)
        
        history['train'].append(train_loss)
        history['val'].append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1:3d}: Train={train_loss:.6f}, Val={val_loss:.6f}, LR={optimizer.param_groups[0]['lr']:.6f}")
    
    # Restore best
    if best_state:
        model.load_state_dict(best_state)
    
    # Export to ONNX
    model_path = 'ml/models/pinn_v4_turbulence_robust.onnx'
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    
    model.eval().cpu()
    dummy = torch.randn(1, 10)
    
    torch.onnx.export(
        model, dummy, model_path,
        opset_version=18,
        input_names=['particle_state'],
        output_names=['forces'],
        dynamic_axes={'particle_state': {0: 'batch'}, 'forces': {0: 'batch'}}
    )
    
    print(f"\n✅ Model saved to: {model_path}")
    print(f"   Best val loss: {best_val_loss:.6f}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(history['train'], label='Train')
    plt.plot(history['val'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.yscale('log')
    plt.legend()
    plt.title('PINN v4 Training (Turbulence-Robust)')
    plt.grid(True, alpha=0.3)
    plt.savefig('ml/models/pinn_v4_training_loss.png', dpi=150)
    print(f"   Training curve: ml/models/pinn_v4_training_loss.png")
    
    return model


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-data', action='store_true')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=1024)
    
    args = parser.parse_args()
    
    if args.generate_data:
        generate_training_data_v4()
    else:
        train_pinn_v4(epochs=args.epochs, batch_size=args.batch_size)

