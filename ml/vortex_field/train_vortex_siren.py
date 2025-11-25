#!/usr/bin/env python3
"""
Train SIREN (Sinusoidal Representation Network) for vortex field.

SIREN uses sin() activations which naturally capture periodic/oscillatory
patterns like vortex structures. Much better than ReLU for this task.

Reference: Sitzmann et al. "Implicit Neural Representations with Periodic Activation Functions" (2020)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import argparse
from pathlib import Path


class SirenLayer(nn.Module):
    """
    SIREN layer with sinusoidal activation.

    Key insight: sin(ωx) can represent high-frequency patterns
    that ReLU networks struggle with.
    """

    def __init__(self, in_features: int, out_features: int,
                 omega: float = 30.0, is_first: bool = False):
        super().__init__()
        self.omega = omega
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)

        # Special initialization for SIREN
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            if self.is_first:
                # First layer: uniform in [-1/in, 1/in]
                bound = 1.0 / self.linear.in_features
            else:
                # Hidden layers: uniform in [-sqrt(6/in)/omega, sqrt(6/in)/omega]
                bound = np.sqrt(6.0 / self.linear.in_features) / self.omega

            self.linear.weight.uniform_(-bound, bound)
            self.linear.bias.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega * self.linear(x))


class VortexSIREN(nn.Module):
    """
    SIREN network for vortex field prediction.

    Input: [x, y, z, t, seed] → 5 dimensions
    Output: [ω_x, ω_y, ω_z] → 3 dimensions (vorticity)

    Very compact: ~4000 parameters
    """

    def __init__(self, hidden_dim: int = 48, n_layers: int = 2, omega: float = 30.0):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.omega = omega

        # Build network
        layers = []

        # First layer (special initialization)
        layers.append(SirenLayer(5, hidden_dim, omega=omega, is_first=True))

        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(SirenLayer(hidden_dim, hidden_dim, omega=omega))

        self.layers = nn.ModuleList(layers)

        # Output layer (linear, no activation)
        self.output = nn.Linear(hidden_dim, 3)

        # Initialize output layer
        with torch.no_grad():
            bound = np.sqrt(6.0 / hidden_dim) / omega
            self.output.weight.uniform_(-bound, bound)
            self.output.bias.zero_()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())


def train(model: VortexSIREN, train_loader: DataLoader, val_loader: DataLoader,
          epochs: int, lr: float, device: torch.device) -> dict:
    """
    Train the SIREN model.
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)

    criterion = nn.MSELoss()

    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    best_state = None

    print(f"\nTraining SIREN ({model.count_parameters()} parameters)")
    print(f"  Device: {device}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {lr}")
    print("-" * 50)

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()

        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()

        # Progress
        if (epoch + 1) % 50 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:4d}: Train={train_loss:.6f}, Val={val_loss:.6f}, LR={current_lr:.2e}")

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nRestored best model (val_loss={best_val_loss:.6f})")

    return history


def export_to_onnx(model: VortexSIREN, output_path: str, output_scale: float):
    """
    Export trained model to ONNX format.
    """
    model.eval()
    model.cpu()

    # Dummy input
    dummy_input = torch.randn(1, 5)

    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['vorticity'],
        dynamic_axes={'input': {0: 'batch'}, 'vorticity': {0: 'batch'}},
        opset_version=11
    )

    print(f"Exported ONNX model to: {output_path}")

    # Also save metadata
    meta_path = Path(output_path).with_suffix('.meta.npz')
    np.savez(meta_path, output_scale=output_scale, omega=model.omega,
             hidden_dim=model.hidden_dim)
    print(f"Saved metadata to: {meta_path}")


def export_to_hlsl(model: VortexSIREN, output_path: str, output_scale: float):
    """
    Export trained weights to HLSL header for GPU evaluation.

    This embeds the neural network directly in shader code - zero runtime cost!
    """
    model.eval()
    model.cpu()

    hlsl = []
    hlsl.append("// Auto-generated SIREN vortex field")
    hlsl.append(f"// Parameters: {model.count_parameters()}")
    hlsl.append(f"// Output scale: {output_scale}")
    hlsl.append("")
    hlsl.append(f"static const float VORTEX_OUTPUT_SCALE = {output_scale}f;")
    hlsl.append(f"static const float VORTEX_OMEGA = {model.omega}f;")
    hlsl.append("")

    # Extract weights
    layer_idx = 0
    for name, param in model.named_parameters():
        data = param.detach().numpy().flatten()
        array_name = name.replace('.', '_').upper()

        hlsl.append(f"static const float {array_name}[{len(data)}] = {{")

        # Format weights in rows of 8
        for i in range(0, len(data), 8):
            row = data[i:i+8]
            row_str = ", ".join(f"{v:.8f}f" for v in row)
            hlsl.append(f"    {row_str},")

        hlsl.append("};")
        hlsl.append("")

    # Add evaluation function
    hidden_dim = model.hidden_dim
    hlsl.append(f"""
float3 EvaluateVortexSIREN(float3 position, float time, float seed)
{{
    // Input: [x, y, z, t, seed]
    float input[5] = {{ position.x, position.y, position.z, time, seed }};

    // Layer 0 (first SIREN layer)
    float h0[{hidden_dim}];
    for (int i = 0; i < {hidden_dim}; i++) {{
        float sum = LAYERS_0_LINEAR_BIAS[i];
        for (int j = 0; j < 5; j++) {{
            sum += LAYERS_0_LINEAR_WEIGHT[i * 5 + j] * input[j];
        }}
        h0[i] = sin(VORTEX_OMEGA * sum);
    }}

    // Layer 1 (hidden SIREN layer)
    float h1[{hidden_dim}];
    for (int i = 0; i < {hidden_dim}; i++) {{
        float sum = LAYERS_1_LINEAR_BIAS[i];
        for (int j = 0; j < {hidden_dim}; j++) {{
            sum += LAYERS_1_LINEAR_WEIGHT[i * {hidden_dim} + j] * h0[j];
        }}
        h1[i] = sin(VORTEX_OMEGA * sum);
    }}

    // Output layer (linear)
    float3 vorticity;
    vorticity.x = OUTPUT_BIAS[0];
    vorticity.y = OUTPUT_BIAS[1];
    vorticity.z = OUTPUT_BIAS[2];

    for (int j = 0; j < {hidden_dim}; j++) {{
        vorticity.x += OUTPUT_WEIGHT[0 * {hidden_dim} + j] * h1[j];
        vorticity.y += OUTPUT_WEIGHT[1 * {hidden_dim} + j] * h1[j];
        vorticity.z += OUTPUT_WEIGHT[2 * {hidden_dim} + j] * h1[j];
    }}

    return vorticity * VORTEX_OUTPUT_SCALE;
}}
""")

    # Write file
    with open(output_path, 'w') as f:
        f.write('\n'.join(hlsl))

    print(f"Exported HLSL shader to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Train SIREN for vortex field')
    parser.add_argument('--data', type=str, default='vortex_data.npz', help='Training data')
    parser.add_argument('--output', type=str, default='../models/vortex_siren.onnx', help='Output model')
    parser.add_argument('--epochs', type=int, default=500, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=4096, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=48, help='Hidden layer dimension')
    parser.add_argument('--n_layers', type=int, default=2, help='Number of hidden layers')
    parser.add_argument('--omega', type=float, default=30.0, help='SIREN frequency')
    parser.add_argument('--export_hlsl', action='store_true', help='Also export HLSL shader')

    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load data
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Data not found: {data_path}")
        print("Run: python generate_vortex_training_data.py --output vortex_data.npz")
        return

    print(f"Loading data from: {data_path}")
    data = np.load(data_path)
    inputs = torch.tensor(data['inputs'], dtype=torch.float32)
    outputs = torch.tensor(data['outputs'], dtype=torch.float32)
    output_scale = float(data['output_scale'])

    print(f"  Samples: {len(inputs)}")
    print(f"  Output scale: {output_scale:.4f}")

    # Split train/val
    n_val = len(inputs) // 10
    indices = torch.randperm(len(inputs))
    train_idx, val_idx = indices[n_val:], indices[:n_val]

    train_dataset = TensorDataset(inputs[train_idx], outputs[train_idx])
    val_dataset = TensorDataset(inputs[val_idx], outputs[val_idx])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Create model
    model = VortexSIREN(
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        omega=args.omega
    )

    print(f"\nModel architecture:")
    print(f"  Input: 5 (x, y, z, t, seed)")
    print(f"  Hidden: {args.n_layers} layers × {args.hidden_dim} neurons")
    print(f"  Output: 3 (vorticity)")
    print(f"  Parameters: {model.count_parameters()}")
    print(f"  Omega: {args.omega}")

    # Train
    history = train(model, train_loader, val_loader, args.epochs, args.lr, device)

    # Save outputs
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Export ONNX
    export_to_onnx(model, str(output_path), output_scale)

    # Optionally export HLSL
    if args.export_hlsl:
        hlsl_path = output_path.with_suffix('.hlsl')
        export_to_hlsl(model, str(hlsl_path), output_scale)

    print("\nDone!")
    print(f"  ONNX model: {output_path}")
    if args.export_hlsl:
        print(f"  HLSL shader: {hlsl_path}")


if __name__ == '__main__':
    main()
