#!/usr/bin/env python3
"""
Generate ideal accretion disk physics data WITH turbulence for PINN v2 training.

Extends generate_ideal_physics_data.py with:
1. Kolmogorov-spectrum turbulent velocity perturbations
2. MRI-inspired (Magneto-Rotational Instability) turbulent forces
3. Spatiotemporally coherent noise (not random jitter)

The resulting PINN learns realistic chaotic motion while respecting physics.
"""

import numpy as np
from pathlib import Path
import argparse

# Physical constants (normalized units)
G = 1.0                 # Gravitational constant
M_BH = 10.0             # Black hole mass (10 solar masses normalized)
R_ISCO = 6.0            # Innermost Stable Circular Orbit
R_INNER = 8.0           # Inner disk radius (just outside ISCO)
R_OUTER = 500.0         # Outer disk radius
DISK_THICKNESS = 0.1    # H/R ratio (scale height)


def kolmogorov_spectrum_noise(n_samples: int, n_modes: int = 16, seed: int = None) -> np.ndarray:
    """
    Generate turbulent velocity field using Kolmogorov energy spectrum.

    E(k) ∝ k^(-5/3) for inertial range

    This produces realistic turbulence with energy cascade from large to small scales.
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate random phases and directions for each mode
    phases = np.random.uniform(0, 2*np.pi, (n_modes, 3))
    directions = np.random.randn(n_modes, 3)
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    # Wavenumbers with Kolmogorov scaling
    k = np.arange(1, n_modes + 1)
    amplitudes = k ** (-5.0/6.0)  # sqrt of E(k) ∝ k^(-5/3)
    amplitudes /= np.sum(amplitudes)  # Normalize

    # Sum modes to create turbulent field
    turbulence = np.zeros((n_samples, 3))
    positions = np.random.randn(n_samples, 3)  # Sample positions

    for i in range(n_modes):
        # Spatial variation
        spatial = np.sin(k[i] * np.sum(positions * directions[i], axis=1, keepdims=True) + phases[i])
        turbulence += amplitudes[i] * spatial * directions[i]

    return turbulence


def mri_turbulent_force(r: np.ndarray, v_phi: np.ndarray, alpha: float,
                        time: np.ndarray, seed: int = None) -> np.ndarray:
    """
    Generate MRI-inspired turbulent forces for accretion disk.

    The Magneto-Rotational Instability creates:
    1. Radial transport of angular momentum (outward)
    2. Turbulent stress proportional to pressure
    3. Time-varying chaotic structure

    Simplified model: F_turb ∝ α * v_phi^2 / r * noise
    """
    if seed is not None:
        np.random.seed(seed)

    n = len(r)

    # Base MRI strength scales with local shear
    Omega = np.sqrt(G * M_BH / r**3)  # Keplerian angular velocity
    shear_rate = -1.5 * Omega  # d(ln Ω)/d(ln r) = -3/2 for Keplerian

    # Turbulent amplitude proportional to alpha and local velocity
    turb_amplitude = alpha * np.abs(v_phi) * np.abs(shear_rate) * 0.1

    # Generate spatiotemporally coherent noise
    # Use time-dependent seed for temporal coherence
    time_phase = (time * 10.0) % (2 * np.pi)

    # Radial force (angular momentum transport)
    noise_r = np.sin(r * 0.1 + time_phase) * np.cos(time_phase * 2.3)
    noise_r += 0.5 * np.sin(r * 0.3 + time_phase * 1.7)
    F_r_turb = turb_amplitude * noise_r

    # Azimuthal force (energy dissipation)
    noise_phi = np.cos(r * 0.15 + time_phase * 0.8) * np.sin(time_phase * 1.9)
    noise_phi += 0.3 * np.cos(r * 0.25 + time_phase * 2.1)
    F_phi_turb = -turb_amplitude * noise_phi * 0.5  # Negative = energy loss

    # Vertical force (disk breathing mode)
    noise_theta = np.sin(r * 0.05 + time_phase * 0.5) * 0.3
    F_theta_turb = turb_amplitude * noise_theta * 0.2

    return np.column_stack([F_r_turb, F_theta_turb, F_phi_turb])


def generate_turbulent_trajectory(r0: float, duration: float, dt: float,
                                  M_bh: float = M_BH, alpha: float = 0.1,
                                  H_R: float = 0.1, turb_intensity: float = 1.0,
                                  seed: int = None) -> dict:
    """
    Generate a single particle trajectory with turbulence.

    Combines:
    1. Keplerian orbital mechanics
    2. Viscous angular momentum transport
    3. MRI-driven turbulent forces
    4. Vertical oscillations with turbulent excitation
    """
    if seed is not None:
        np.random.seed(seed)

    n_steps = int(duration / dt)

    # Initialize arrays
    r = np.zeros(n_steps)
    theta = np.zeros(n_steps)
    phi = np.zeros(n_steps)
    v_r = np.zeros(n_steps)
    v_theta = np.zeros(n_steps)
    v_phi = np.zeros(n_steps)
    time = np.zeros(n_steps)

    # Forces (what the PINN will learn to predict)
    F_r = np.zeros(n_steps)
    F_theta = np.zeros(n_steps)
    F_phi = np.zeros(n_steps)

    # Initial conditions
    r[0] = r0
    theta[0] = np.pi / 2 + np.random.uniform(-0.1, 0.1) * H_R  # Near midplane
    phi[0] = np.random.uniform(0, 2 * np.pi)

    # Keplerian velocity with small perturbation
    v_kep = np.sqrt(G * M_bh / r0)
    v_r[0] = np.random.uniform(-0.05, 0.05) * v_kep
    v_theta[0] = np.random.uniform(-0.02, 0.02) * v_kep
    v_phi[0] = v_kep * (1 + np.random.uniform(-0.02, 0.02))

    # Pre-generate turbulent noise for coherence
    turb_seeds = np.random.randint(0, 10000, n_steps)

    for i in range(n_steps - 1):
        time[i] = i * dt

        # Current state
        ri, thetai, phii = r[i], theta[i], phi[i]
        vri, vthetai, vphii = v_r[i], v_theta[i], v_phi[i]

        # === Gravitational force ===
        F_grav_r = -G * M_bh / ri**2

        # === Centrifugal force ===
        F_cent_r = vphii**2 / ri

        # === Coriolis forces ===
        F_coriolis_theta = -2 * vri * vthetai / ri
        F_coriolis_phi = -2 * vri * vphii / ri

        # === Viscous forces (Shakura-Sunyaev) ===
        Omega = np.sqrt(G * M_bh / ri**3)
        nu = alpha * (H_R * ri)**2 * Omega  # Kinematic viscosity

        # Viscous torque causes inward drift and angular momentum loss
        F_visc_r = -3 * nu * vri / ri**2 * 0.1
        F_visc_phi = -nu * vphii / ri**2

        # === Vertical confinement (disk gravity) ===
        z = ri * np.cos(thetai)
        H = H_R * ri
        F_vert = -G * M_bh * z / (ri**2 * H) * 0.5  # Restore to midplane
        F_vert_theta = F_vert * np.sin(thetai) / ri

        # === MRI Turbulent forces ===
        turb_forces = mri_turbulent_force(
            np.array([ri]), np.array([vphii]), alpha,
            np.array([time[i]]), seed=turb_seeds[i]
        )[0] * turb_intensity

        # Add Kolmogorov velocity perturbation (small-scale turbulence)
        if i % 10 == 0:  # Update every 10 steps for performance
            kolm_noise = kolmogorov_spectrum_noise(1, n_modes=8, seed=turb_seeds[i])[0]
            kolm_scale = 0.01 * v_kep * turb_intensity

        # === Total forces ===
        F_r[i] = F_grav_r + F_cent_r + F_visc_r + turb_forces[0]
        F_theta[i] = F_coriolis_theta + F_vert_theta + turb_forces[1]
        F_phi[i] = F_coriolis_phi + F_visc_phi + turb_forces[2]

        # === Integration (leapfrog) ===
        # Update velocities
        v_r[i+1] = vri + F_r[i] * dt + kolm_noise[0] * kolm_scale
        v_theta[i+1] = vthetai + F_theta[i] * dt + kolm_noise[1] * kolm_scale * 0.5
        v_phi[i+1] = vphii + F_phi[i] * dt + kolm_noise[2] * kolm_scale

        # Update positions
        r[i+1] = ri + 0.5 * (vri + v_r[i+1]) * dt
        theta[i+1] = thetai + 0.5 * (vthetai + v_theta[i+1]) * dt / ri
        phi[i+1] = phii + 0.5 * (vphii + v_phi[i+1]) * dt / (ri * np.sin(thetai))

        # Boundary conditions
        r[i+1] = np.clip(r[i+1], R_ISCO * 1.1, R_OUTER * 0.99)
        theta[i+1] = np.clip(theta[i+1], 0.1, np.pi - 0.1)
        phi[i+1] = phi[i+1] % (2 * np.pi)

    # Final step
    time[-1] = (n_steps - 1) * dt
    F_r[-1], F_theta[-1], F_phi[-1] = F_r[-2], F_theta[-2], F_phi[-2]

    return {
        'r': r, 'theta': theta, 'phi': phi,
        'v_r': v_r, 'v_theta': v_theta, 'v_phi': v_phi,
        'F_r': F_r, 'F_theta': F_theta, 'F_phi': F_phi,
        'time': time,
        'params': {'M_bh': M_bh, 'alpha': alpha, 'H_R': H_R, 'turb': turb_intensity}
    }


def generate_dataset(n_particles: int = 1000, duration: float = 10.0,
                     dt: float = 0.01, output_path: str = None,
                     include_param_variations: bool = True) -> dict:
    """
    Generate complete training dataset with turbulent physics.
    """
    print(f"Generating {n_particles} turbulent particle trajectories...")

    all_states = []
    all_forces = []
    all_params = []

    # Parameter variations for conditioning
    if include_param_variations:
        M_bh_values = [0.5, 0.75, 1.0, 1.25, 1.5]  # Mass multipliers
        alpha_values = [0.05, 0.1, 0.15, 0.2]       # Viscosity
        H_R_values = [0.08, 0.1, 0.12, 0.15]        # Disk thickness
        turb_values = [0.5, 1.0, 1.5, 2.0]          # Turbulence intensity
    else:
        M_bh_values = [1.0]
        alpha_values = [0.1]
        H_R_values = [0.1]
        turb_values = [1.0]

    particles_per_config = max(1, n_particles // (len(M_bh_values) * len(alpha_values)))

    total_configs = len(M_bh_values) * len(alpha_values) * len(H_R_values) * len(turb_values)
    config_idx = 0

    for M_bh_mult in M_bh_values:
        for alpha in alpha_values:
            for H_R in H_R_values:
                for turb in turb_values:
                    config_idx += 1
                    M_bh = M_BH * M_bh_mult

                    # Generate particles at different radii
                    radii = np.linspace(R_INNER, R_OUTER * 0.8, particles_per_config)

                    for j, r0 in enumerate(radii):
                        traj = generate_turbulent_trajectory(
                            r0=r0, duration=duration, dt=dt,
                            M_bh=M_bh, alpha=alpha, H_R=H_R,
                            turb_intensity=turb,
                            seed=config_idx * 1000 + j
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
                        params = np.tile([M_bh_mult, alpha, H_R], (n_steps, 1))

                        all_states.append(states)
                        all_forces.append(forces)
                        all_params.append(params)

                    if config_idx % 10 == 0:
                        print(f"  Config {config_idx}/{total_configs}: M={M_bh_mult:.2f}, α={alpha:.2f}, H/R={H_R:.2f}, turb={turb:.1f}")

    # Combine all data
    states = np.vstack(all_states).astype(np.float32)
    forces = np.vstack(all_forces).astype(np.float32)
    params = np.vstack(all_params).astype(np.float32)

    print(f"\nDataset shape: {states.shape[0]} samples")
    print(f"  States: {states.shape}")
    print(f"  Forces: {forces.shape}")
    print(f"  Params: {params.shape}")

    # Save
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            output_path,
            states=states,
            forces=forces,
            params=params,
            metadata={
                'n_particles': n_particles,
                'duration': duration,
                'dt': dt,
                'includes_turbulence': True,
                'turbulence_model': 'MRI + Kolmogorov'
            }
        )
        print(f"\nSaved to: {output_path}")

    return {'states': states, 'forces': forces, 'params': params}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate turbulent accretion disk training data')
    parser.add_argument('--n_particles', type=int, default=500, help='Particles per config')
    parser.add_argument('--duration', type=float, default=5.0, help='Trajectory duration')
    parser.add_argument('--dt', type=float, default=0.02, help='Time step')
    parser.add_argument('--output', type=str, default='training_data/turbulent_accretion_disk.npz',
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

    print("\nTo train with turbulent data:")
    print("  python train_pinn_v2.py --data training_data/turbulent_accretion_disk.npz --epochs 300 --batch_size 4096")
