#!/usr/bin/env python3
"""
Generate ideal accretion disk physics data for PINN training.

CRITICAL: Uses PlasmaDX NORMALIZED UNITS where:
- G * M_bh = 1 (gravitational parameter)
- R_ISCO = 6 (innermost stable circular orbit)
- Inner disk radius = 10
- Outer disk radius = 300-500
- Keplerian velocity at r: v_k = sqrt(1/r)
- Orbital period at r: T = 2π * r^(3/2)

Forces are computed analytically for stable Keplerian orbits with:
- Gravitational attraction: F_r = -1/r^2
- Centrifugal balance: a_cent = v_φ^2/r
- Viscous torque: F_φ = -α * v_φ / r^2
- Vertical confinement: F_θ restores to midplane

This produces training data that matches PlasmaDX's internal physics scale.
"""

import numpy as np
from pathlib import Path
import argparse

# PlasmaDX normalized units
GM = 1.0                # G * M_black_hole = 1
R_ISCO = 6.0            # Innermost Stable Circular Orbit
R_INNER = 10.0          # Inner disk edge
R_OUTER = 300.0         # Outer disk edge
DISK_THICKNESS = 0.1    # H/R scale height ratio


def keplerian_velocity(r: np.ndarray) -> np.ndarray:
    """Keplerian orbital velocity: v_k = sqrt(GM/r) = sqrt(1/r)"""
    return np.sqrt(GM / r)


def keplerian_angular_velocity(r: np.ndarray) -> np.ndarray:
    """Keplerian angular velocity: Ω = sqrt(GM/r^3) = sqrt(1/r^3)"""
    return np.sqrt(GM / r**3)


def gravitational_acceleration(r: np.ndarray) -> np.ndarray:
    """Gravitational acceleration: a_g = -GM/r^2 = -1/r^2"""
    return -GM / r**2


def generate_particle_trajectory(
    r0: float,
    duration: float = 10.0,
    dt: float = 0.05,
    alpha: float = 0.1,      # Shakura-Sunyaev viscosity
    H_R: float = 0.1,        # Disk thickness (H/R)
    M_bh_mult: float = 1.0,  # Black hole mass multiplier
    seed: int = None
) -> dict:
    """
    Generate a single particle trajectory in normalized units.

    Returns state (r, θ, φ, v_r, v_θ, v_φ, t) and forces (F_r, F_θ, F_φ).
    """
    if seed is not None:
        np.random.seed(seed)

    # Effective GM with mass multiplier
    GM_eff = GM * M_bh_mult

    n_steps = int(duration / dt)

    # State arrays
    r = np.zeros(n_steps)
    theta = np.zeros(n_steps)
    phi = np.zeros(n_steps)
    v_r = np.zeros(n_steps)
    v_theta = np.zeros(n_steps)
    v_phi = np.zeros(n_steps)
    time = np.zeros(n_steps)

    # Force arrays (what PINN learns to predict)
    F_r = np.zeros(n_steps)
    F_theta = np.zeros(n_steps)
    F_phi = np.zeros(n_steps)

    # Initial conditions
    r[0] = r0
    theta[0] = np.pi / 2 + np.random.uniform(-0.05, 0.05) * H_R  # Near midplane
    phi[0] = np.random.uniform(0, 2 * np.pi)

    # Start with Keplerian velocity + small perturbation
    v_k = np.sqrt(GM_eff / r0)
    v_r[0] = np.random.uniform(-0.02, 0.02) * v_k
    v_theta[0] = np.random.uniform(-0.01, 0.01) * v_k
    v_phi[0] = v_k * (1.0 + np.random.uniform(-0.01, 0.01))

    for i in range(n_steps - 1):
        time[i] = i * dt
        ri = r[i]
        thetai = theta[i]
        vri = v_r[i]
        vthetai = v_theta[i]
        vphii = v_phi[i]

        # === FORCES IN SPHERICAL COORDINATES ===

        # 1. Gravitational force (radial)
        F_grav = -GM_eff / ri**2

        # 2. Centrifugal "force" (appears in spherical coords)
        F_cent = vphii**2 / ri + vthetai**2 / ri

        # 3. Viscous torque (Shakura-Sunyaev α model)
        # Causes angular momentum transport outward, particle drifts inward
        Omega = np.sqrt(GM_eff / ri**3)
        nu = alpha * (H_R * ri)**2 * Omega  # kinematic viscosity

        # Viscous radial drift (inward spiral)
        F_visc_r = -3 * nu / ri * 0.01  # Small inward drift

        # Viscous azimuthal torque (angular momentum loss)
        F_visc_phi = -alpha * vphii / ri * 0.1

        # 4. Vertical confinement (restoring force to disk midplane)
        # z = r * cos(θ), midplane is θ = π/2
        z = ri * np.cos(thetai)
        H = H_R * ri  # scale height

        # Vertical gravity component restores to midplane
        # F_z = -Ω² * z approximately
        F_vert_z = -Omega**2 * z * 0.5

        # Convert to θ force: F_θ = F_z * sin(θ) / r (approximately)
        F_vert_theta = F_vert_z * np.sin(thetai) / ri if ri > 0.1 else 0

        # 5. Coriolis terms (appear in rotating frame / spherical coords)
        F_coriolis_theta = -2 * vri * vthetai / ri
        F_coriolis_phi = -2 * vri * vphii / ri

        # === TOTAL FORCES ===
        F_r[i] = F_grav + F_cent + F_visc_r
        F_theta[i] = F_coriolis_theta + F_vert_theta
        F_phi[i] = F_coriolis_phi + F_visc_phi

        # === INTEGRATION (semi-implicit Euler) ===
        # Update velocities
        v_r[i+1] = vri + F_r[i] * dt
        v_theta[i+1] = vthetai + F_theta[i] * dt
        v_phi[i+1] = vphii + F_phi[i] * dt

        # Update positions
        r[i+1] = ri + v_r[i+1] * dt
        theta[i+1] = thetai + v_theta[i+1] / ri * dt
        phi[i+1] = phi[i] + v_phi[i+1] / (ri * np.sin(thetai) + 1e-6) * dt

        # Boundary conditions
        r[i+1] = np.clip(r[i+1], R_ISCO * 1.01, R_OUTER * 0.99)
        theta[i+1] = np.clip(theta[i+1], 0.1, np.pi - 0.1)
        phi[i+1] = phi[i+1] % (2 * np.pi)

        # Damping for stability (small)
        v_r[i+1] *= 0.999
        v_theta[i+1] *= 0.999

    # Fill last step
    time[-1] = (n_steps - 1) * dt
    F_r[-1] = F_r[-2]
    F_theta[-1] = F_theta[-2]
    F_phi[-1] = F_phi[-2]

    return {
        'r': r, 'theta': theta, 'phi': phi,
        'v_r': v_r, 'v_theta': v_theta, 'v_phi': v_phi,
        'F_r': F_r, 'F_theta': F_theta, 'F_phi': F_phi,
        'time': time,
        'params': {'M_bh': M_bh_mult, 'alpha': alpha, 'H_R': H_R}
    }


def generate_dataset(
    n_particles: int = 200,
    duration: float = 10.0,
    dt: float = 0.05,
    output_path: str = None,
    include_param_variations: bool = True
) -> dict:
    """Generate complete training dataset."""

    print("Generating ideal accretion disk data (NORMALIZED UNITS)")
    print(f"  G*M = {GM}, R_ISCO = {R_ISCO}")
    print(f"  Inner radius = {R_INNER}, Outer radius = {R_OUTER}")

    all_states = []
    all_forces = []
    all_params = []

    # Parameter variations (smaller range, centered on defaults)
    if include_param_variations:
        M_bh_values = [0.7, 0.85, 1.0, 1.15, 1.3]  # Mass multipliers
        alpha_values = [0.05, 0.1, 0.15, 0.2]       # Viscosity
        H_R_values = [0.08, 0.1, 0.12]              # Thickness
    else:
        M_bh_values = [1.0]
        alpha_values = [0.1]
        H_R_values = [0.1]

    total_configs = len(M_bh_values) * len(alpha_values) * len(H_R_values)
    particles_per_config = max(1, n_particles // total_configs)

    print(f"  {total_configs} parameter configurations")
    print(f"  {particles_per_config} particles per config")
    print()

    config_idx = 0
    for M_bh in M_bh_values:
        for alpha in alpha_values:
            for H_R in H_R_values:
                config_idx += 1

                # Generate particles at different radii
                radii = np.linspace(R_INNER, R_OUTER * 0.7, particles_per_config)

                for j, r0 in enumerate(radii):
                    traj = generate_particle_trajectory(
                        r0=r0,
                        duration=duration,
                        dt=dt,
                        alpha=alpha,
                        H_R=H_R,
                        M_bh_mult=M_bh,
                        seed=config_idx * 10000 + j
                    )

                    n_steps = len(traj['r'])

                    # State: [r, θ, φ, v_r, v_θ, v_φ, t]
                    states = np.column_stack([
                        traj['r'], traj['theta'], traj['phi'],
                        traj['v_r'], traj['v_theta'], traj['v_phi'],
                        traj['time']
                    ])

                    # Forces: [F_r, F_θ, F_φ]
                    forces = np.column_stack([
                        traj['F_r'], traj['F_theta'], traj['F_phi']
                    ])

                    # Parameters: [M_bh_norm, α, H/R]
                    params = np.tile([M_bh, alpha, H_R], (n_steps, 1))

                    all_states.append(states)
                    all_forces.append(forces)
                    all_params.append(params)

                if config_idx % 10 == 0:
                    print(f"  Config {config_idx}/{total_configs}: M={M_bh:.2f}, α={alpha:.2f}, H/R={H_R:.2f}")

    # Combine
    states = np.vstack(all_states).astype(np.float32)
    forces = np.vstack(all_forces).astype(np.float32)
    params = np.vstack(all_params).astype(np.float32)

    print(f"\n=== Dataset Summary ===")
    print(f"Total samples: {len(states)}")
    print(f"State shape: {states.shape}")
    print(f"Forces shape: {forces.shape}")
    print(f"Params shape: {params.shape}")

    print(f"\nState ranges (NORMALIZED UNITS):")
    for i, name in enumerate(['r', 'θ', 'φ', 'v_r', 'v_θ', 'v_φ', 't']):
        print(f"  {name}: [{states[:,i].min():.3f}, {states[:,i].max():.3f}]")

    print(f"\nForce ranges (NORMALIZED UNITS):")
    for i, name in enumerate(['F_r', 'F_θ', 'F_φ']):
        print(f"  {name}: [{forces[:,i].min():.6f}, {forces[:,i].max():.6f}]")

    # Save
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            output_path,
            states=states,
            forces=forces,
            params=params
        )
        print(f"\nSaved to: {output_path}")

    return {'states': states, 'forces': forces, 'params': params}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate ideal physics training data (normalized units)')
    parser.add_argument('--n_particles', type=int, default=200, help='Particles per config')
    parser.add_argument('--duration', type=float, default=10.0, help='Simulation duration')
    parser.add_argument('--dt', type=float, default=0.05, help='Time step')
    parser.add_argument('--output', type=str, default='training_data/ideal_accretion_disk.npz',
                        help='Output path')
    parser.add_argument('--no_variations', action='store_true', help='Skip parameter variations')

    args = parser.parse_args()

    generate_dataset(
        n_particles=args.n_particles,
        duration=args.duration,
        dt=args.dt,
        output_path=args.output,
        include_param_variations=not args.no_variations
    )

    print("\n✅ Data generation complete!")
    print("\nNext: python train_pinn_v2.py --epochs 300 --batch_size 4096 --physics_weight 0.01")
