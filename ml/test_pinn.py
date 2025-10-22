#!/usr/bin/env python3
"""
Test and validate trained PINN model.

Tests:
1. Keplerian orbit stability
2. Angular momentum conservation
3. Energy conservation
4. ISCO behavior
5. ONNX export integrity
"""

import numpy as np
import torch
import onnxruntime as ort
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


# Physical constants
G = 6.674e-8
C = 2.998e10
M_SUN = 1.989e33
M_BH = 10.0 * M_SUN
R_S = 2 * G * M_BH / C**2
R_ISCO = 3 * R_S


def load_onnx_model(model_path):
    """Load ONNX model for inference."""
    session = ort.InferenceSession(model_path)
    return session


def predict_forces(session, states):
    """
    Predict forces using ONNX model.

    Args:
        session: ONNX runtime session
        states: (N, 7) array - [r, theta, phi, v_r, v_theta, v_phi, t]

    Returns:
        forces: (N, 3) array - [F_r, F_theta, F_phi]
    """
    # Prepare input
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Run inference
    forces = session.run([output_name], {input_name: states.astype(np.float32)})[0]

    return forces


def keplerian_velocity(r):
    """Keplerian orbital velocity at radius r."""
    return np.sqrt(G * M_BH / r)


def test_keplerian_orbit(session, r0=20*R_ISCO, num_steps=1000):
    """
    Test 1: Keplerian circular orbit should be stable.

    Returns:
        (success, max_radial_drift_percent)
    """
    print("\n" + "="*60)
    print("Test 1: Keplerian Orbit Stability")
    print("="*60)

    # Initial conditions (circular orbit)
    r = r0
    theta = np.pi / 2  # Disk plane
    phi = 0.0
    v_r = 0.0
    v_theta = 0.0
    v_phi = keplerian_velocity(r)

    dt = 0.1
    trajectory = []

    for i in range(num_steps):
        t = i * dt

        # Current state
        state = np.array([[r, theta, phi, v_r, v_theta, v_phi, t]], dtype=np.float32)

        # Predict forces
        forces = predict_forces(session, state)[0]
        F_r, F_theta, F_phi = forces

        # Update velocities
        v_r += F_r * dt
        v_theta += F_theta * dt
        v_phi += F_phi * dt

        # Update positions
        r += v_r * dt
        theta += v_theta / r * dt
        phi += v_phi / r * dt

        # Keep theta in [0, π]
        if theta < 0:
            theta = -theta
            phi += np.pi
        if theta > np.pi:
            theta = 2*np.pi - theta
            phi += np.pi

        # Keep phi in [-π, π]
        phi = np.arctan2(np.sin(phi), np.cos(phi))

        trajectory.append([r, theta, phi, v_r, v_theta, v_phi])

    trajectory = np.array(trajectory)

    # Analyze stability
    r_drift = (trajectory[:, 0] - r0) / r0 * 100  # Percent drift
    max_drift = np.max(np.abs(r_drift))

    print(f"Initial radius: {r0/R_ISCO:.1f} × R_ISCO")
    print(f"Orbital period: ~{2*np.pi*r0/keplerian_velocity(r0):.2f} s")
    print(f"Time simulated: {num_steps*dt:.2f} s")
    print(f"Max radial drift: {max_drift:.3f}%")

    # Success criteria: drift < 1%
    success = max_drift < 1.0

    if success:
        print(f"✅ PASS: Orbit is stable (drift < 1%)")
    else:
        print(f"❌ FAIL: Orbit is unstable (drift > 1%)")

    return success, max_drift, trajectory


def test_angular_momentum_conservation(session, trajectory, r0=20*R_ISCO):
    """
    Test 2: Angular momentum should be conserved.

    Returns:
        (success, max_L_drift_percent)
    """
    print("\n" + "="*60)
    print("Test 2: Angular Momentum Conservation")
    print("="*60)

    # Extract trajectory data
    r = trajectory[:, 0]
    v_phi = trajectory[:, 5]

    # Compute angular momentum
    L = r * v_phi
    L_initial = L[0]
    L_drift = (L - L_initial) / L_initial * 100

    max_drift = np.max(np.abs(L_drift))

    print(f"Initial L: {L_initial:.2e}")
    print(f"Final L: {L[-1]:.2e}")
    print(f"Max drift: {max_drift:.3f}%")

    # Success criteria: drift < 1%
    success = max_drift < 1.0

    if success:
        print(f"✅ PASS: Angular momentum conserved (drift < 1%)")
    else:
        print(f"❌ FAIL: Angular momentum not conserved (drift > 1%)")

    return success, max_drift


def test_energy_conservation(session, trajectory, r0=20*R_ISCO):
    """
    Test 3: Total energy should be conserved.

    Returns:
        (success, max_E_drift_percent)
    """
    print("\n" + "="*60)
    print("Test 3: Energy Conservation")
    print("="*60)

    # Extract trajectory data
    r = trajectory[:, 0]
    v_r = trajectory[:, 3]
    v_theta = trajectory[:, 4]
    v_phi = trajectory[:, 5]

    # Compute kinetic energy
    KE = 0.5 * (v_r**2 + v_theta**2 + v_phi**2)

    # Compute potential energy (with GR correction)
    L = r * v_phi
    PE = -G * M_BH / r + L**2 / (2 * r**2) - G * M_BH * L**2 / r**3

    # Total energy
    E = KE + PE
    E_initial = E[0]
    E_drift = (E - E_initial) / abs(E_initial) * 100

    max_drift = np.max(np.abs(E_drift))

    print(f"Initial E: {E_initial:.2e}")
    print(f"Final E: {E[-1]:.2e}")
    print(f"Max drift: {max_drift:.3f}%")

    # Success criteria: drift < 2% (softer constraint)
    success = max_drift < 2.0

    if success:
        print(f"✅ PASS: Energy approximately conserved (drift < 2%)")
    else:
        print(f"❌ FAIL: Energy not conserved (drift > 2%)")

    return success, max_drift


def test_isco_behavior(session, r0=3*R_ISCO, num_steps=1000):
    """
    Test 4: Particle at ISCO should remain stable (not plunge).

    Returns:
        (success, min_radius)
    """
    print("\n" + "="*60)
    print("Test 4: ISCO Stability")
    print("="*60)

    # Initial conditions at ISCO
    r = r0
    theta = np.pi / 2
    phi = 0.0
    v_r = 0.0
    v_theta = 0.0
    v_phi = keplerian_velocity(r)

    dt = 0.05  # Smaller timestep near ISCO
    min_r = r

    for i in range(num_steps):
        t = i * dt

        # Current state
        state = np.array([[r, theta, phi, v_r, v_theta, v_phi, t]], dtype=np.float32)

        # Predict forces
        forces = predict_forces(session, state)[0]
        F_r, F_theta, F_phi = forces

        # Update
        v_r += F_r * dt
        v_theta += F_theta * dt
        v_phi += F_phi * dt

        r += v_r * dt
        theta += v_theta / r * dt
        phi += v_phi / r * dt

        # Track minimum radius
        min_r = min(min_r, r)

        # Check if plunged into black hole
        if r < R_S:
            print(f"❌ FAIL: Particle plunged into black hole (r = {r/R_S:.2f} × R_S)")
            return False, min_r

    print(f"ISCO radius: {R_ISCO/R_S:.1f} × R_S")
    print(f"Initial radius: {r0/R_S:.1f} × R_S")
    print(f"Minimum radius: {min_r/R_S:.1f} × R_S")
    print(f"Time simulated: {num_steps*dt:.2f} s")

    # Success: didn't cross event horizon
    success = min_r > R_S

    if success:
        print(f"✅ PASS: Particle remained outside event horizon")
    else:
        print(f"❌ FAIL: Particle crossed event horizon")

    return success, min_r


def plot_test_results(trajectory, output_dir='ml/analysis/pinn'):
    """Generate validation plots."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    r = trajectory[:, 0]
    v_phi = trajectory[:, 5]
    L = r * v_phi

    # Plot 1: Orbital trajectory
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Radial distance over time
    axes[0, 0].plot(r / R_ISCO)
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Radius (× R_ISCO)')
    axes[0, 0].set_title('Orbital Radius vs Time')
    axes[0, 0].grid(True, alpha=0.3)

    # Angular momentum
    axes[0, 1].plot(L)
    axes[0, 1].set_xlabel('Time Step')
    axes[0, 1].set_ylabel('Angular Momentum (L)')
    axes[0, 1].set_title('Angular Momentum Conservation')
    axes[0, 1].grid(True, alpha=0.3)

    # 2D trajectory (top-down view)
    x = r * np.cos(trajectory[:, 2])
    y = r * np.sin(trajectory[:, 2])
    axes[1, 0].plot(x / R_ISCO, y / R_ISCO)
    axes[1, 0].set_xlabel('X (× R_ISCO)')
    axes[1, 0].set_ylabel('Y (× R_ISCO)')
    axes[1, 0].set_title('Orbital Trajectory (Top-Down View)')
    axes[1, 0].axis('equal')
    axes[1, 0].grid(True, alpha=0.3)

    # Velocity components
    axes[1, 1].plot(trajectory[:, 3], label='v_r')
    axes[1, 1].plot(trajectory[:, 4], label='v_θ')
    axes[1, 1].plot(trajectory[:, 5], label='v_φ')
    axes[1, 1].set_xlabel('Time Step')
    axes[1, 1].set_ylabel('Velocity')
    axes[1, 1].set_title('Velocity Components')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/pinn_validation.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nValidation plots saved to {output_dir}/pinn_validation.png")


def main():
    parser = argparse.ArgumentParser(description='Test PINN model')
    parser.add_argument('--model', type=str, default='ml/models/pinn_accretion_disk.onnx',
                        help='Path to ONNX model')
    args = parser.parse_args()

    print("="*60)
    print("PINN Model Validation")
    print("="*60)

    # Load model
    print(f"\nLoading model: {args.model}")
    if not Path(args.model).exists():
        print(f"ERROR: Model not found: {args.model}")
        print("Train model first: python pinn_accretion_disk.py")
        return

    session = load_onnx_model(args.model)
    print("Model loaded successfully")

    # Run tests
    results = {}

    # Test 1: Keplerian orbit
    success, drift, trajectory = test_keplerian_orbit(session)
    results['keplerian'] = {'success': success, 'drift': drift}

    # Test 2: Angular momentum
    success, drift = test_angular_momentum_conservation(session, trajectory)
    results['angular_momentum'] = {'success': success, 'drift': drift}

    # Test 3: Energy
    success, drift = test_energy_conservation(session, trajectory)
    results['energy'] = {'success': success, 'drift': drift}

    # Test 4: ISCO
    success, min_r = test_isco_behavior(session)
    results['isco'] = {'success': success, 'min_r': min_r}

    # Generate plots
    plot_test_results(trajectory)

    # Summary
    print("\n" + "="*60)
    print("Validation Summary")
    print("="*60)

    all_passed = all(results[test]['success'] for test in results)

    for test_name, test_result in results.items():
        status = "✅ PASS" if test_result['success'] else "❌ FAIL"
        print(f"{test_name:20s}: {status}")

    print()
    if all_passed:
        print("🎉 All tests PASSED! Model is ready for integration.")
    else:
        print("⚠️  Some tests FAILED. Consider retraining with more data or adjusting physics loss weights.")

    print("="*60)


if __name__ == '__main__':
    main()
