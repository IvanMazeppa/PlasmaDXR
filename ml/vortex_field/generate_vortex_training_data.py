#!/usr/bin/env python3
"""
Generate training data for learned vortex field (SIREN).

Creates velocity/vorticity fields from analytical vortex models:
1. Lamb-Oseen vortex (viscous decaying vortex)
2. Vortex rings
3. Vortex tubes with helical structure
4. Superposition of multiple vortices

Output: (position, time) → vorticity vector
"""

import numpy as np
from pathlib import Path
import argparse


def lamb_oseen_vortex(positions: np.ndarray, time: float,
                      center: np.ndarray, axis: np.ndarray,
                      circulation: float = 1.0, core_radius: float = 0.1,
                      viscosity: float = 0.01) -> np.ndarray:
    """
    Lamb-Oseen vortex - analytical solution for viscous decaying vortex.

    ω(r,t) = (Γ / (4πνt)) * exp(-r² / (4νt))

    Args:
        positions: [N, 3] sample positions
        time: Current time (affects decay)
        center: [3] vortex center position
        axis: [3] vortex axis direction (normalized)
        circulation: Vortex strength Γ
        core_radius: Initial core radius
        viscosity: Kinematic viscosity ν

    Returns:
        [N, 3] vorticity vectors
    """
    # Ensure axis is normalized
    axis = axis / np.linalg.norm(axis)

    # Vector from center to each position
    r_vec = positions - center

    # Distance from vortex axis
    # Project onto plane perpendicular to axis
    r_parallel = np.dot(r_vec, axis)[:, np.newaxis] * axis
    r_perp = r_vec - r_parallel
    r_dist = np.linalg.norm(r_perp, axis=1)

    # Effective radius including viscous spreading
    r_eff = np.sqrt(core_radius**2 + 4 * viscosity * (time + 0.1))

    # Vorticity magnitude (Gaussian profile)
    omega_mag = (circulation / (np.pi * r_eff**2)) * np.exp(-(r_dist / r_eff)**2)

    # Vorticity direction is along axis
    vorticity = omega_mag[:, np.newaxis] * axis

    return vorticity


def vortex_ring(positions: np.ndarray, time: float,
                center: np.ndarray, axis: np.ndarray,
                ring_radius: float = 0.5, core_radius: float = 0.1,
                circulation: float = 1.0) -> np.ndarray:
    """
    Vortex ring - toroidal vortex structure.

    The vorticity is concentrated in a torus around the ring.
    """
    axis = axis / np.linalg.norm(axis)

    # Vector from center to positions
    r_vec = positions - center

    # Decompose into axial and radial components
    z = np.dot(r_vec, axis)  # Axial distance
    r_vec_radial = r_vec - z[:, np.newaxis] * axis
    rho = np.linalg.norm(r_vec_radial, axis=1)  # Radial distance from axis

    # Distance from ring core
    # Ring is at (rho=ring_radius, z=0)
    d_ring = np.sqrt((rho - ring_radius)**2 + z**2)

    # Vorticity magnitude (Gaussian around ring)
    omega_mag = (circulation / (np.pi * core_radius**2)) * np.exp(-(d_ring / core_radius)**2)

    # Vorticity direction is tangent to ring (azimuthal)
    # Tangent = axis × radial_direction
    radial_dir = np.zeros_like(r_vec_radial)
    nonzero = rho > 1e-6
    radial_dir[nonzero] = r_vec_radial[nonzero] / rho[nonzero, np.newaxis]

    tangent = np.cross(axis, radial_dir)

    vorticity = omega_mag[:, np.newaxis] * tangent

    return vorticity


def helical_vortex_tube(positions: np.ndarray, time: float,
                        center: np.ndarray = np.array([0, 0, 0]),
                        helix_radius: float = 0.3,
                        helix_pitch: float = 0.5,
                        core_radius: float = 0.08,
                        circulation: float = 1.0,
                        rotation_speed: float = 1.0) -> np.ndarray:
    """
    Helical vortex tube - spiral vortex structure.

    Common in rotating flows (e.g., accretion disks).
    """
    # Helical centerline parametrized by arc length
    # x = R*cos(s), y = R*sin(s), z = pitch*s/(2π)

    n_points = len(positions)
    vorticity = np.zeros((n_points, 3))

    # Time evolution: helix rotates
    phase = time * rotation_speed

    # Sample points along helix to find nearest
    s_samples = np.linspace(0, 4*np.pi, 100)
    helix_points = np.column_stack([
        helix_radius * np.cos(s_samples + phase) + center[0],
        helix_radius * np.sin(s_samples + phase) + center[1],
        helix_pitch * s_samples / (2*np.pi) + center[2]
    ])

    # For each position, find distance to nearest helix point
    for i, pos in enumerate(positions):
        dists = np.linalg.norm(helix_points - pos, axis=1)
        min_idx = np.argmin(dists)
        min_dist = dists[min_idx]

        # Vorticity magnitude
        omega_mag = (circulation / (np.pi * core_radius**2)) * np.exp(-(min_dist / core_radius)**2)

        # Vorticity direction: tangent to helix at nearest point
        s = s_samples[min_idx]
        tangent = np.array([
            -helix_radius * np.sin(s + phase),
            helix_radius * np.cos(s + phase),
            helix_pitch / (2*np.pi)
        ])
        tangent = tangent / np.linalg.norm(tangent)

        vorticity[i] = omega_mag * tangent

    return vorticity


def generate_random_vortex_field(positions: np.ndarray, time: float,
                                  n_vortices: int = 5,
                                  bounds: float = 1.0,
                                  seed: int = None) -> np.ndarray:
    """
    Generate superposition of random vortices.

    Creates complex, realistic-looking turbulent field.
    """
    if seed is not None:
        np.random.seed(seed)

    vorticity = np.zeros((len(positions), 3))

    for _ in range(n_vortices):
        # Random vortex parameters
        vtype = np.random.choice(['tube', 'ring', 'helix'])
        center = np.random.uniform(-bounds, bounds, 3)
        axis = np.random.randn(3)
        axis = axis / np.linalg.norm(axis)
        circulation = np.random.uniform(0.5, 2.0) * np.random.choice([-1, 1])
        core_radius = np.random.uniform(0.05, 0.2)

        if vtype == 'tube':
            v = lamb_oseen_vortex(positions, time, center, axis,
                                  circulation, core_radius)
        elif vtype == 'ring':
            ring_radius = np.random.uniform(0.2, 0.6)
            v = vortex_ring(positions, time, center, axis,
                           ring_radius, core_radius, circulation)
        else:  # helix
            v = helical_vortex_tube(positions, time, center,
                                    helix_radius=np.random.uniform(0.2, 0.5),
                                    helix_pitch=np.random.uniform(0.3, 0.8),
                                    core_radius=core_radius,
                                    circulation=circulation)

        vorticity += v

    return vorticity


def generate_dataset(n_samples: int = 100000,
                     n_time_steps: int = 20,
                     bounds: float = 1.0,
                     n_vortices_range: tuple = (3, 8),
                     output_path: str = None) -> dict:
    """
    Generate complete training dataset for vortex SIREN.
    """
    print(f"Generating {n_samples} vortex field samples...")

    samples_per_timestep = n_samples // n_time_steps

    all_inputs = []  # [x, y, z, t, seed]
    all_outputs = []  # [ω_x, ω_y, ω_z]

    for t_idx in range(n_time_steps):
        time = t_idx * 0.1  # Time from 0 to 2.0

        # Random seed for this time step (for reproducibility)
        seed = t_idx * 1000

        # Generate random positions
        positions = np.random.uniform(-bounds, bounds, (samples_per_timestep, 3))

        # Number of vortices varies
        n_vortices = np.random.randint(n_vortices_range[0], n_vortices_range[1] + 1)

        # Generate vorticity field
        vorticity = generate_random_vortex_field(
            positions, time, n_vortices=n_vortices,
            bounds=bounds, seed=seed
        )

        # Create input features
        times = np.full(samples_per_timestep, time)
        seeds = np.full(samples_per_timestep, seed % 100)  # Normalized seed

        inputs = np.column_stack([positions, times, seeds / 100.0])

        all_inputs.append(inputs)
        all_outputs.append(vorticity)

        if (t_idx + 1) % 5 == 0:
            print(f"  Time step {t_idx + 1}/{n_time_steps}")

    # Combine
    inputs = np.vstack(all_inputs).astype(np.float32)
    outputs = np.vstack(all_outputs).astype(np.float32)

    # Normalize outputs for better training
    output_scale = np.std(outputs) + 1e-6
    outputs_normalized = outputs / output_scale

    print(f"\nDataset shape:")
    print(f"  Inputs: {inputs.shape} (x, y, z, t, seed)")
    print(f"  Outputs: {outputs.shape} (ω_x, ω_y, ω_z)")
    print(f"  Output scale: {output_scale:.4f}")

    # Save
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            output_path,
            inputs=inputs,
            outputs=outputs_normalized,
            output_scale=output_scale,
            bounds=bounds
        )
        print(f"\nSaved to: {output_path}")

    return {
        'inputs': inputs,
        'outputs': outputs_normalized,
        'output_scale': output_scale
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate vortex field training data')
    parser.add_argument('--n_samples', type=int, default=100000, help='Number of samples')
    parser.add_argument('--n_time_steps', type=int, default=20, help='Number of time steps')
    parser.add_argument('--bounds', type=float, default=1.0, help='Spatial bounds')
    parser.add_argument('--output', type=str, default='vortex_data.npz', help='Output path')

    args = parser.parse_args()

    generate_dataset(
        n_samples=args.n_samples,
        n_time_steps=args.n_time_steps,
        bounds=args.bounds,
        output_path=args.output
    )

    print("\nNext step: python train_vortex_siren.py --data vortex_data.npz")
