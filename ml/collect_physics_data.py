#!/usr/bin/env python3
"""
Collect physics training data from PlasmaDX GPU buffer dumps.

Reads particle buffer dumps and extracts (state, forces) pairs for PINN training.

Usage:
    1. Run PlasmaDX-Clean with --dump-buffers flag
    2. Run this script to process buffer dumps
    3. Generated dataset: ml/training_data/physics_trajectories.npz
"""

import numpy as np
import struct
from pathlib import Path
import json
from typing import List, Tuple
import argparse


# Particle structure (32 bytes, from ParticleSystem.h)
# struct Particle {
#     float3 position;     // 12 bytes
#     float mass;          // 4 bytes
#     float3 velocity;     // 12 bytes
#     float temperature;   // 4 bytes
# };

PARTICLE_SIZE = 32  # bytes


def read_particle_buffer(buffer_path: str) -> np.ndarray:
    """
    Read binary particle buffer from GPU dump.

    Args:
        buffer_path: Path to g_particles.bin

    Returns:
        Array of shape (N, 8) with columns:
        [pos_x, pos_y, pos_z, mass, vel_x, vel_y, vel_z, temperature]
    """
    with open(buffer_path, 'rb') as f:
        data = f.read()

    num_particles = len(data) // PARTICLE_SIZE

    if len(data) % PARTICLE_SIZE != 0:
        print(f"WARNING: Buffer size {len(data)} not multiple of {PARTICLE_SIZE}")
        num_particles = len(data) // PARTICLE_SIZE

    particles = []

    for i in range(num_particles):
        offset = i * PARTICLE_SIZE

        # Read particle data
        chunk = data[offset:offset + PARTICLE_SIZE]

        # Unpack: 3 floats (pos), 1 float (mass), 3 floats (vel), 1 float (temp)
        values = struct.unpack('ffffffff', chunk)

        particles.append(values)

    return np.array(particles, dtype=np.float32)


def cartesian_to_spherical(particles: np.ndarray) -> np.ndarray:
    """
    Convert Cartesian coordinates to spherical (r, θ, φ).

    Args:
        particles: (N, 8) array in Cartesian

    Returns:
        (N, 8) array with [r, theta, phi, mass, v_r, v_theta, v_phi, temp]
    """
    # Extract positions and velocities
    x, y, z = particles[:, 0], particles[:, 1], particles[:, 2]
    vx, vy, vz = particles[:, 4], particles[:, 5], particles[:, 6]

    # Spherical coordinates
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / (r + 1e-10))  # Polar angle (0 to π)
    phi = np.arctan2(y, x)  # Azimuthal angle (-π to π)

    # Convert velocities to spherical basis
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)

    # Transform velocity components
    v_r = (vx * sin_theta * cos_phi +
           vy * sin_theta * sin_phi +
           vz * cos_theta)

    v_theta = (vx * cos_theta * cos_phi +
               vy * cos_theta * sin_phi -
               vz * sin_theta)

    v_phi = (-vx * sin_phi + vy * cos_phi)

    # Reconstruct array
    spherical = np.column_stack([
        r, theta, phi,
        particles[:, 3],  # mass
        v_r, v_theta, v_phi,
        particles[:, 7]   # temperature
    ])

    return spherical


def compute_forces_from_trajectory(particles_t0: np.ndarray,
                                   particles_t1: np.ndarray,
                                   dt: float) -> np.ndarray:
    """
    Compute forces from consecutive frames using finite differences.

    F = m * (v_{t+1} - v_t) / dt

    Args:
        particles_t0: Particles at time t
        particles_t1: Particles at time t + dt
        dt: Time step (seconds)

    Returns:
        Forces (N, 3) - [F_r, F_theta, F_phi]
    """
    # Extract velocities (spherical)
    v_t0 = particles_t0[:, 4:7]  # (v_r, v_theta, v_phi) at t
    v_t1 = particles_t1[:, 4:7]  # (v_r, v_theta, v_phi) at t+dt

    # Masses
    mass = particles_t0[:, 3:4]

    # Compute acceleration
    dv_dt = (v_t1 - v_t0) / dt

    # F = ma
    forces = mass * dv_dt

    return forces


def collect_trajectories(buffer_dir: str, output_path: str):
    """
    Collect training data from buffer dumps.

    Args:
        buffer_dir: Directory containing buffer dumps (PIX/buffer_dumps/)
        output_path: Output .npz file for training data
    """
    buffer_dir = Path(buffer_dir)

    # Find all buffer dump sessions
    sessions = sorted(buffer_dir.glob('**/g_particles.bin'))

    if not sessions:
        print(f"No buffer dumps found in {buffer_dir}")
        print("Run PlasmaDX-Clean with --dump-buffers flag to generate data")
        return

    print(f"Found {len(sessions)} buffer dump sessions")

    all_states = []
    all_forces = []

    # Process each session (assumes sequential frames)
    for i, buffer_path in enumerate(sessions):
        print(f"Processing {buffer_path}...")

        # Read particles
        particles_cart = read_particle_buffer(str(buffer_path))

        # Convert to spherical
        particles_sph = cartesian_to_spherical(particles_cart)

        print(f"  Loaded {len(particles_sph)} particles")

        # If we have a previous frame, compute forces
        if i > 0:
            dt = 1.0 / 120.0  # Assume 120 FPS (8.33ms between frames)

            # Compute forces from velocity change
            forces = compute_forces_from_trajectory(prev_particles, particles_sph, dt)

            # Store state (previous frame) and forces
            # State: (r, theta, phi, v_r, v_theta, v_phi, t)
            # We'll add time as frame_index * dt
            time = (i - 1) * dt

            for j in range(len(prev_particles)):
                state = np.concatenate([
                    prev_particles[j, 0:3],  # r, theta, phi
                    prev_particles[j, 4:7],  # v_r, v_theta, v_phi
                    [time]
                ])

                force = forces[j]

                all_states.append(state)
                all_forces.append(force)

        prev_particles = particles_sph

    # Convert to numpy arrays
    states = np.array(all_states, dtype=np.float32)
    forces = np.array(all_forces, dtype=np.float32)

    print(f"\nCollected {len(states)} training samples")
    print(f"State shape: {states.shape}")
    print(f"Forces shape: {forces.shape}")

    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_path,
        states=states,
        forces=forces,
        metadata={
            'num_sessions': len(sessions),
            'num_samples': len(states),
            'dt': 1.0 / 120.0
        }
    )

    print(f"\nData saved to {output_path}")

    # Statistics
    print("\nDataset Statistics:")
    print(f"  Radial range: {states[:, 0].min():.2f} - {states[:, 0].max():.2f}")
    print(f"  Velocity range: {states[:, 3:6].min():.2f} - {states[:, 3:6].max():.2f}")
    print(f"  Force range: {forces.min():.2e} - {forces.max():.2e}")


def load_collected_data(data_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load collected physics data for PINN training.

    Args:
        data_path: Path to .npz file

    Returns:
        (states, forces) tuple
    """
    data = np.load(data_path)
    return data['states'], data['forces']


def main():
    parser = argparse.ArgumentParser(description='Collect physics data from buffer dumps')
    parser.add_argument('--input', type=str, default='PIX/buffer_dumps',
                        help='Directory containing buffer dumps')
    parser.add_argument('--output', type=str, default='ml/training_data/physics_trajectories.npz',
                        help='Output .npz file')

    args = parser.parse_args()

    print("="*60)
    print("Physics Data Collection for PINN Training")
    print("="*60)
    print()

    collect_trajectories(args.input, args.output)

    print()
    print("="*60)
    print("Next steps:")
    print("  1. Train PINN: python ml/pinn_accretion_disk.py --data", args.output)
    print("  2. Export to ONNX for C++ inference")
    print("  3. Integrate into PlasmaDX-Clean")
    print("="*60)


if __name__ == '__main__':
    main()
