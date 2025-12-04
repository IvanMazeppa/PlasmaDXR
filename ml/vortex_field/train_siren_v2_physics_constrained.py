#!/usr/bin/env python3
"""
Train Physics-Constrained SIREN v2 for Accretion Disk Turbulence.

PROBLEM WITH v1:
  - Vorticity → Force via F = cross(v, ω)
  - Arbitrary vorticity destroys angular momentum
  - High turbulence = particles spiral out/in = disk destruction

SOLUTION (v2):
  - Constrain vorticity to preserve angular momentum
  - Vorticity must be primarily vertical (z-axis) in disk plane
  - Radial vorticity component must be minimized
  - Energy budget constraint: turbulent forces do zero net work

PHYSICS CONSTRAINTS:
  1. Angular Momentum: L = r × p, dL/dt = r × F
     For L conservation: F_turb must be tangential (⊥ to r)
     This means ω must be ∥ to disk normal (z-axis)

  2. Energy Conservation: dE/dt = F · v
     For E conservation: F_turb · v ≈ 0

  3. Keplerian Coherence: Vortices should follow orbital motion
     Azimuthal structure should match orbital frequency

Reference: Sitzmann et al. "Implicit Neural Representations with Periodic Activation Functions" (2020)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import argparse
from pathlib import Path
from typing import Tuple, Dict
import json


class SirenLayer(nn.Module):
    """SIREN layer with sinusoidal activation."""

    def __init__(self, in_features: int, out_features: int,
                 omega: float = 30.0, is_first: bool = False):
        super().__init__()
        self.omega = omega
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.linear.in_features
            else:
                bound = np.sqrt(6.0 / self.linear.in_features) / self.omega
            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega * self.linear(x))


class PhysicsConstrainedSIREN(nn.Module):
    """
    Physics-Constrained SIREN v2 for accretion disk turbulence.

    Key difference from v1: Output is decomposed into:
      - ω_z (vertical vorticity) - PRIMARY, preserves angular momentum
      - ω_r (radial vorticity) - PENALIZED, causes L drift
      - ω_φ (azimuthal vorticity) - PENALIZED, causes vertical motion

    The network learns to suppress radial/azimuthal components while
    maintaining rich vertical vortex structure.
    """

    def __init__(self, hidden_dim: int = 64, n_layers: int = 3, omega: float = 30.0):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.omega = omega
        self.n_layers = n_layers

        # Input: [x, y, z, t, seed, r, phi] = 7 dimensions
        # Added cylindrical coords for physics-aware processing
        input_dim = 7

        # Build SIREN layers
        layers = []
        layers.append(SirenLayer(input_dim, hidden_dim, omega=omega, is_first=True))
        for _ in range(n_layers - 1):
            layers.append(SirenLayer(hidden_dim, hidden_dim, omega=omega))
        self.layers = nn.ModuleList(layers)

        # Separate output heads for each vorticity component
        # This allows independent control over each component's magnitude
        self.head_omega_z = nn.Linear(hidden_dim, 1)  # Vertical (good)
        self.head_omega_r = nn.Linear(hidden_dim, 1)  # Radial (bad)
        self.head_omega_phi = nn.Linear(hidden_dim, 1)  # Azimuthal (neutral)

        # Initialize output layers with small weights
        self._init_output_heads()

    def _init_output_heads(self):
        with torch.no_grad():
            bound = np.sqrt(6.0 / self.hidden_dim) / self.omega
            for head in [self.head_omega_z, self.head_omega_r, self.head_omega_phi]:
                head.weight.uniform_(-bound, bound)
                head.bias.zero_()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass returning cylindrical vorticity components.

        Args:
            x: [batch, 7] = [x, y, z, t, seed, r, phi]

        Returns:
            omega_z: [batch, 1] - Vertical vorticity (preserved)
            omega_r: [batch, 1] - Radial vorticity (penalized)
            omega_phi: [batch, 1] - Azimuthal vorticity (penalized)
        """
        h = x
        for layer in self.layers:
            h = layer(h)

        omega_z = self.head_omega_z(h)
        omega_r = self.head_omega_r(h)
        omega_phi = self.head_omega_phi(h)

        return omega_z, omega_r, omega_phi

    def get_cartesian_vorticity(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get vorticity in Cartesian coordinates for force computation.

        The cylindrical components are converted:
          ω_x = ω_r * cos(φ) - ω_φ * sin(φ)
          ω_y = ω_r * sin(φ) + ω_φ * cos(φ)
          ω_z = ω_z
        """
        omega_z, omega_r, omega_phi = self.forward(x)

        # Extract phi from input (index 6)
        phi = x[:, 6:7]
        cos_phi = torch.cos(phi)
        sin_phi = torch.sin(phi)

        omega_x = omega_r * cos_phi - omega_phi * sin_phi
        omega_y = omega_r * sin_phi + omega_phi * cos_phi

        return torch.cat([omega_x, omega_y, omega_z], dim=1)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


def generate_physics_aware_training_data(
    n_samples: int = 100000,
    r_min: float = 100.0,
    r_max: float = 500.0,
    z_scale: float = 30.0,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate training data with physics-aware vorticity targets.

    The target vorticity is designed to:
    1. Be primarily vertical (ω_z dominant)
    2. Follow Keplerian orbital structure
    3. Create visually interesting but stable turbulence
    """
    np.random.seed(seed)

    # Sample positions in cylindrical coordinates
    r = np.random.uniform(r_min, r_max, n_samples)
    phi = np.random.uniform(0, 2 * np.pi, n_samples)
    z = np.random.normal(0, z_scale, n_samples)

    # Convert to Cartesian
    x = r * np.cos(phi)
    y = r * np.sin(phi)

    # Time and seed
    t = np.random.uniform(0, 100, n_samples)
    vortex_seed = np.random.uniform(0, 1, n_samples)

    # === TARGET VORTICITY (Physics-Constrained) ===

    # 1. Vertical vorticity (ω_z) - PRIMARY COMPONENT
    # Creates horizontal swirling motion that preserves L
    # Pattern: Multiple vortex cells with Keplerian frequency modulation
    keplerian_freq = 1.0 / np.sqrt(r)  # ω ∝ r^(-3/2)

    # Spatial structure: Gaussian vortex cores
    n_vortices = 8
    vortex_strength = np.zeros(n_samples)
    for i in range(n_vortices):
        vortex_r = np.random.uniform(r_min + 50, r_max - 50)
        vortex_phi = 2 * np.pi * i / n_vortices
        vortex_x = vortex_r * np.cos(vortex_phi)
        vortex_y = vortex_r * np.sin(vortex_phi)

        # Distance to vortex center (rotating with Keplerian flow)
        phase = keplerian_freq * t
        rot_x = vortex_x * np.cos(phase) - vortex_y * np.sin(phase)
        rot_y = vortex_x * np.sin(phase) + vortex_y * np.cos(phase)

        dist_sq = (x - rot_x)**2 + (y - rot_y)**2
        core_radius = 30.0 + 20.0 * np.sin(vortex_seed * 2 * np.pi)

        # Gaussian vortex profile
        vortex_strength += np.exp(-dist_sq / (2 * core_radius**2))

    # Add turbulent noise
    noise = 0.3 * np.sin(x * 0.02 + t * 0.1) * np.cos(y * 0.02 - t * 0.05)
    omega_z_target = vortex_strength * (0.5 + 0.5 * noise)

    # Normalize to reasonable range
    omega_z_target = omega_z_target / (np.max(np.abs(omega_z_target)) + 1e-8)

    # 2. Radial vorticity (ω_r) - MINIMAL (causes angular momentum change)
    # Should be near-zero but with tiny variations for realism
    omega_r_target = 0.05 * np.sin(phi * 3 + t * 0.1) * np.exp(-np.abs(z) / z_scale)

    # 3. Azimuthal vorticity (ω_φ) - SMALL (causes vertical motion)
    # Allow small vertical mixing near disk midplane
    omega_phi_target = 0.1 * np.tanh(z / z_scale) * np.sin(r * 0.01 + vortex_seed * 10)

    # === ASSEMBLE DATA ===

    # Input features: [x, y, z, t, seed, r, phi]
    inputs = np.stack([x, y, z, t, vortex_seed, r, phi], axis=1).astype(np.float32)

    # Targets: [ω_z, ω_r, ω_φ]
    omega_z_target = omega_z_target.astype(np.float32).reshape(-1, 1)
    omega_r_target = omega_r_target.astype(np.float32).reshape(-1, 1)
    omega_phi_target = omega_phi_target.astype(np.float32).reshape(-1, 1)

    return inputs, omega_z_target, omega_r_target, omega_phi_target


class PhysicsConstrainedLoss(nn.Module):
    """
    Multi-objective loss function for physics-constrained SIREN.

    Components:
    1. Data fitting loss (MSE on target vorticity)
    2. Angular momentum conservation loss (penalize ω_r)
    3. Energy conservation loss (penalize F · v)
    4. Vertical coherence loss (encourage ω_z dominance)
    """

    def __init__(self,
                 lambda_data: float = 1.0,
                 lambda_angular: float = 10.0,  # Strong penalty on ω_r
                 lambda_energy: float = 5.0,
                 lambda_vertical: float = 2.0):
        super().__init__()
        self.lambda_data = lambda_data
        self.lambda_angular = lambda_angular
        self.lambda_energy = lambda_energy
        self.lambda_vertical = lambda_vertical

        self.mse = nn.MSELoss()

    def forward(self,
                omega_z_pred: torch.Tensor,
                omega_r_pred: torch.Tensor,
                omega_phi_pred: torch.Tensor,
                omega_z_target: torch.Tensor,
                omega_r_target: torch.Tensor,
                omega_phi_target: torch.Tensor,
                positions: torch.Tensor,
                velocities: torch.Tensor = None) -> Tuple[torch.Tensor, Dict]:
        """
        Compute total loss with breakdown.

        Args:
            omega_*_pred: Predicted vorticity components [batch, 1]
            omega_*_target: Target vorticity components [batch, 1]
            positions: [batch, 3] = [x, y, z]
            velocities: [batch, 3] = [vx, vy, vz] (optional, for energy loss)
        """
        losses = {}

        # 1. Data fitting loss
        loss_data_z = self.mse(omega_z_pred, omega_z_target)
        loss_data_r = self.mse(omega_r_pred, omega_r_target)
        loss_data_phi = self.mse(omega_phi_pred, omega_phi_target)
        losses['data_z'] = loss_data_z.item()
        losses['data_r'] = loss_data_r.item()
        losses['data_phi'] = loss_data_phi.item()

        loss_data = loss_data_z + loss_data_r + loss_data_phi

        # 2. Angular momentum conservation: penalize radial vorticity magnitude
        # ω_r causes dL/dt ≠ 0
        loss_angular = torch.mean(omega_r_pred ** 2)
        losses['angular'] = loss_angular.item()

        # 3. Vertical coherence: ω_z should dominate
        # Ratio of |ω_z| to total |ω|
        omega_total = torch.sqrt(omega_z_pred**2 + omega_r_pred**2 + omega_phi_pred**2 + 1e-8)
        vertical_ratio = torch.abs(omega_z_pred) / omega_total
        loss_vertical = torch.mean(1.0 - vertical_ratio)  # Penalize low ratio
        losses['vertical'] = loss_vertical.item()

        # 4. Energy conservation (if velocities provided)
        if velocities is not None:
            # F_turb = v × ω (cross product)
            # Energy change: dE/dt = F · v = (v × ω) · v = 0 (always!)
            # But we want to penalize large forces that could cause numerical issues
            # Proxy: penalize |ω| when velocity is high
            v_mag = torch.norm(velocities, dim=1, keepdim=True)
            loss_energy = torch.mean(omega_total * v_mag)
            losses['energy'] = loss_energy.item()
        else:
            loss_energy = torch.tensor(0.0, device=omega_z_pred.device)
            losses['energy'] = 0.0

        # Total weighted loss
        total = (self.lambda_data * loss_data +
                 self.lambda_angular * loss_angular +
                 self.lambda_vertical * loss_vertical +
                 self.lambda_energy * loss_energy)

        losses['total'] = total.item()

        return total, losses


def train_physics_constrained(
    model: PhysicsConstrainedSIREN,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    lambda_angular: float = 10.0,
    lambda_vertical: float = 2.0
) -> Dict:
    """Train with physics constraints."""

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=30, factor=0.5)

    criterion = PhysicsConstrainedLoss(
        lambda_angular=lambda_angular,
        lambda_vertical=lambda_vertical
    )

    history = {'train_loss': [], 'val_loss': [], 'angular_loss': [], 'vertical_ratio': []}
    best_val_loss = float('inf')
    best_state = None

    print(f"\n{'='*60}")
    print(f"Training Physics-Constrained SIREN v2")
    print(f"{'='*60}")
    print(f"Parameters: {model.count_parameters()}")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Learning rate: {lr}")
    print(f"Lambda angular: {lambda_angular}")
    print(f"Lambda vertical: {lambda_vertical}")
    print(f"{'='*60}\n")

    for epoch in range(epochs):
        # === Training ===
        model.train()
        train_loss = 0.0
        train_angular = 0.0
        train_vertical = 0.0

        for batch in train_loader:
            inputs, omega_z_t, omega_r_t, omega_phi_t = [b.to(device) for b in batch]
            positions = inputs[:, :3]  # x, y, z

            optimizer.zero_grad()

            omega_z_p, omega_r_p, omega_phi_p = model(inputs)

            loss, loss_dict = criterion(
                omega_z_p, omega_r_p, omega_phi_p,
                omega_z_t, omega_r_t, omega_phi_t,
                positions
            )

            loss.backward()
            optimizer.step()

            train_loss += loss_dict['total']
            train_angular += loss_dict['angular']
            train_vertical += loss_dict['vertical']

        train_loss /= len(train_loader)
        train_angular /= len(train_loader)
        train_vertical /= len(train_loader)

        # === Validation ===
        model.eval()
        val_loss = 0.0
        val_angular = 0.0

        with torch.no_grad():
            for batch in val_loader:
                inputs, omega_z_t, omega_r_t, omega_phi_t = [b.to(device) for b in batch]
                positions = inputs[:, :3]

                omega_z_p, omega_r_p, omega_phi_p = model(inputs)

                loss, loss_dict = criterion(
                    omega_z_p, omega_r_p, omega_phi_p,
                    omega_z_t, omega_r_t, omega_phi_t,
                    positions
                )

                val_loss += loss_dict['total']
                val_angular += loss_dict['angular']

        val_loss /= len(val_loader)
        val_angular /= len(val_loader)

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['angular_loss'].append(val_angular)
        history['vertical_ratio'].append(1.0 - train_vertical)

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Progress
        if (epoch + 1) % 25 == 0 or epoch == 0:
            lr_now = optimizer.param_groups[0]['lr']
            vert_ratio = 1.0 - train_vertical
            print(f"Epoch {epoch+1:4d}: Loss={train_loss:.5f}, Val={val_loss:.5f}, "
                  f"Angular={val_angular:.6f}, VertRatio={vert_ratio:.3f}, LR={lr_now:.1e}")

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nRestored best model (val_loss={best_val_loss:.5f})")

    return history


def export_to_onnx(model: PhysicsConstrainedSIREN, output_path: str):
    """Export to ONNX with Cartesian output for compatibility."""
    model.eval()
    model.cpu()

    # Create wrapper that outputs Cartesian vorticity
    class CartesianWrapper(nn.Module):
        def __init__(self, siren):
            super().__init__()
            self.siren = siren

        def forward(self, x):
            return self.siren.get_cartesian_vorticity(x)

    wrapper = CartesianWrapper(model)

    # Dummy input: [x, y, z, t, seed, r, phi]
    dummy = torch.randn(1, 7)

    torch.onnx.export(
        wrapper,
        dummy,
        output_path,
        input_names=['input'],
        output_names=['vorticity'],
        dynamic_axes={'input': {0: 'batch'}, 'vorticity': {0: 'batch'}},
        opset_version=17
    )

    print(f"Exported ONNX: {output_path}")


def export_to_hlsl(model: PhysicsConstrainedSIREN, output_path: str, output_scale: float = 1.0):
    """Export to HLSL shader for GPU evaluation."""
    model.eval()
    model.cpu()

    hlsl = []
    hlsl.append("// Physics-Constrained SIREN v2 - Accretion Disk Turbulence")
    hlsl.append(f"// Parameters: {model.count_parameters()}")
    hlsl.append("// Preserves angular momentum by constraining vorticity to z-axis")
    hlsl.append("")
    hlsl.append(f"static const float SIREN_V2_OMEGA = {model.omega}f;")
    hlsl.append(f"static const float SIREN_V2_OUTPUT_SCALE = {output_scale}f;")
    hlsl.append(f"static const int SIREN_V2_HIDDEN_DIM = {model.hidden_dim};")
    hlsl.append("")

    # Export weights
    for name, param in model.named_parameters():
        data = param.detach().numpy().flatten()
        array_name = f"SIREN_V2_{name.replace('.', '_').upper()}"

        hlsl.append(f"static const float {array_name}[{len(data)}] = {{")
        for i in range(0, len(data), 8):
            row = data[i:i+8]
            row_str = ", ".join(f"{v:.8f}f" for v in row)
            hlsl.append(f"    {row_str},")
        hlsl.append("};")
        hlsl.append("")

    # Evaluation function
    hd = model.hidden_dim
    nl = model.n_layers

    hlsl.append(f"""
// Evaluate physics-constrained vorticity
// Input: position (world space), time, seed
// Output: vorticity vector (primarily z-component)
float3 EvaluateSIRENv2(float3 position, float time, float seed)
{{
    // Convert to cylindrical coordinates
    float r = length(position.xy);
    float phi = atan2(position.y, position.x);

    // Input: [x, y, z, t, seed, r, phi]
    float input_vec[7] = {{
        position.x, position.y, position.z,
        time, seed, r, phi
    }};

    // Forward pass through SIREN layers
    float h[{hd}];
    float h_next[{hd}];

    // Layer 0 (first layer)
    for (int i = 0; i < {hd}; i++) {{
        float sum = SIREN_V2_LAYERS_0_LINEAR_BIAS[i];
        for (int j = 0; j < 7; j++) {{
            sum += SIREN_V2_LAYERS_0_LINEAR_WEIGHT[i * 7 + j] * input_vec[j];
        }}
        h[i] = sin(SIREN_V2_OMEGA * sum);
    }}
""")

    # Hidden layers
    for layer_idx in range(1, nl):
        hlsl.append(f"""
    // Layer {layer_idx}
    for (int i = 0; i < {hd}; i++) {{
        float sum = SIREN_V2_LAYERS_{layer_idx}_LINEAR_BIAS[i];
        for (int j = 0; j < {hd}; j++) {{
            sum += SIREN_V2_LAYERS_{layer_idx}_LINEAR_WEIGHT[i * {hd} + j] * h[j];
        }}
        h_next[i] = sin(SIREN_V2_OMEGA * sum);
    }}
    for (int i = 0; i < {hd}; i++) h[i] = h_next[i];
""")

    hlsl.append(f"""
    // Output heads (cylindrical vorticity)
    float omega_z = SIREN_V2_HEAD_OMEGA_Z_BIAS[0];
    float omega_r = SIREN_V2_HEAD_OMEGA_R_BIAS[0];
    float omega_phi = SIREN_V2_HEAD_OMEGA_PHI_BIAS[0];

    for (int j = 0; j < {hd}; j++) {{
        omega_z += SIREN_V2_HEAD_OMEGA_Z_WEIGHT[j] * h[j];
        omega_r += SIREN_V2_HEAD_OMEGA_R_WEIGHT[j] * h[j];
        omega_phi += SIREN_V2_HEAD_OMEGA_PHI_WEIGHT[j] * h[j];
    }}

    // Convert cylindrical to Cartesian vorticity
    float cos_phi = cos(phi);
    float sin_phi = sin(phi);

    float3 vorticity;
    vorticity.x = omega_r * cos_phi - omega_phi * sin_phi;
    vorticity.y = omega_r * sin_phi + omega_phi * cos_phi;
    vorticity.z = omega_z;  // Dominant component (preserves L)

    return vorticity * SIREN_V2_OUTPUT_SCALE;
}}

// Compute turbulent force from vorticity
// F_turb = velocity × vorticity (cross product)
// This is tangential when ω is vertical, preserving angular momentum
float3 ComputeTurbulentForce(float3 velocity, float3 vorticity, float intensity)
{{
    return cross(velocity, vorticity) * intensity;
}}
""")

    with open(output_path, 'w') as f:
        f.write('\n'.join(hlsl))

    print(f"Exported HLSL: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Train Physics-Constrained SIREN v2')
    parser.add_argument('--output', type=str, default='../models/siren_v2_physics.onnx')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--omega', type=float, default=30.0)
    parser.add_argument('--n_samples', type=int, default=200000)
    parser.add_argument('--lambda_angular', type=float, default=10.0,
                        help='Penalty weight for radial vorticity (angular momentum)')
    parser.add_argument('--lambda_vertical', type=float, default=2.0,
                        help='Penalty weight for non-vertical vorticity')
    parser.add_argument('--export_hlsl', action='store_true', help='Export HLSL shader')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Generate training data
    print(f"\nGenerating {args.n_samples} physics-aware training samples...")
    inputs, omega_z, omega_r, omega_phi = generate_physics_aware_training_data(
        n_samples=args.n_samples
    )

    print(f"  Input shape: {inputs.shape}")
    print(f"  Target shapes: ω_z={omega_z.shape}, ω_r={omega_r.shape}, ω_φ={omega_phi.shape}")
    print(f"  ω_z range: [{omega_z.min():.3f}, {omega_z.max():.3f}]")
    print(f"  ω_r range: [{omega_r.min():.3f}, {omega_r.max():.3f}]")
    print(f"  ω_φ range: [{omega_phi.min():.3f}, {omega_phi.max():.3f}]")

    # Convert to tensors
    inputs_t = torch.tensor(inputs, dtype=torch.float32)
    omega_z_t = torch.tensor(omega_z, dtype=torch.float32)
    omega_r_t = torch.tensor(omega_r, dtype=torch.float32)
    omega_phi_t = torch.tensor(omega_phi, dtype=torch.float32)

    # Split train/val
    n_val = len(inputs) // 10
    indices = torch.randperm(len(inputs))
    train_idx, val_idx = indices[n_val:], indices[:n_val]

    train_dataset = TensorDataset(
        inputs_t[train_idx], omega_z_t[train_idx],
        omega_r_t[train_idx], omega_phi_t[train_idx]
    )
    val_dataset = TensorDataset(
        inputs_t[val_idx], omega_z_t[val_idx],
        omega_r_t[val_idx], omega_phi_t[val_idx]
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Create model
    model = PhysicsConstrainedSIREN(
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        omega=args.omega
    )

    print(f"\nModel: PhysicsConstrainedSIREN v2")
    print(f"  Input: 7 (x, y, z, t, seed, r, φ)")
    print(f"  Hidden: {args.n_layers} layers × {args.hidden_dim} neurons")
    print(f"  Output: 3 heads (ω_z, ω_r, ω_φ)")
    print(f"  Parameters: {model.count_parameters()}")

    # Train
    history = train_physics_constrained(
        model, train_loader, val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=device,
        lambda_angular=args.lambda_angular,
        lambda_vertical=args.lambda_vertical
    )

    # Export
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    export_to_onnx(model, str(output_path))

    if args.export_hlsl:
        hlsl_path = output_path.with_suffix('.hlsl')
        export_to_hlsl(model, str(hlsl_path))

    # Save training info
    info = {
        'version': 2,
        'type': 'physics_constrained_siren',
        'parameters': model.count_parameters(),
        'hidden_dim': args.hidden_dim,
        'n_layers': args.n_layers,
        'omega': args.omega,
        'lambda_angular': args.lambda_angular,
        'lambda_vertical': args.lambda_vertical,
        'final_val_loss': history['val_loss'][-1],
        'final_angular_loss': history['angular_loss'][-1],
        'final_vertical_ratio': history['vertical_ratio'][-1]
    }

    info_path = output_path.with_suffix('.info.json')
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    print(f"Saved info: {info_path}")

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"  ONNX model: {output_path}")
    if args.export_hlsl:
        print(f"  HLSL shader: {hlsl_path}")
    print(f"  Final validation loss: {history['val_loss'][-1]:.5f}")
    print(f"  Final angular momentum loss: {history['angular_loss'][-1]:.6f}")
    print(f"  Final vertical ratio: {history['vertical_ratio'][-1]:.3f}")
    print("="*60)


if __name__ == '__main__':
    main()
