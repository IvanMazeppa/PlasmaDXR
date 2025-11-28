#!/usr/bin/env python3
"""
ONNX Model Validator for PINN Accretion Disk

Verifies that the exported ONNX model is compatible with the C++ inference engine.
Checks input/output shapes, data types, and runs test inference.
"""

from pathlib import Path

import numpy as np

import onnx
import onnxruntime as ort


def validate_onnx_model(model_path: str = "models/pinn_accretion_disk.onnx"):
    """
    Validate ONNX model format and compatibility.

    Args:
        model_path: Path to .onnx model file

    Returns:
        bool: True if validation passes, False otherwise
    """
    print("=" * 60)
    print("ONNX Model Validation")
    print("=" * 60)

    model_path = Path(model_path)
    if not model_path.exists():
        print(f"❌ FAIL: Model file not found: {model_path}")
        print(f"   Run: python pinn_accretion_disk.py")
        return False

    # Load and check model
    try:
        model = onnx.load(str(model_path))
        onnx.checker.check_model(model)
        print(f"✅ ONNX model loaded successfully: {model_path}")
        print(f"   File size: {model_path.stat().st_size / 1024:.1f} KB")
    except Exception as e:
        print(f"❌ FAIL: Model check failed: {e}")
        return False

    # Check input/output metadata
    graph = model.graph

    print("\n" + "=" * 60)
    print("Model Metadata")
    print("=" * 60)

    # Inputs
    print("\nInputs:")
    for i, inp in enumerate(graph.input):
        print(f"  [{i}] {inp.name}")
        shape = [dim.dim_value if dim.dim_value > 0 else "dynamic" for dim in inp.type.tensor_type.shape.dim]
        print(f"      Shape: {shape}")
        print(f"      Type: {inp.type.tensor_type.elem_type}")

    # Outputs
    print("\nOutputs:")
    for i, out in enumerate(graph.output):
        print(f"  [{i}] {out.name}")
        shape = [dim.dim_value if dim.dim_value > 0 else "dynamic" for dim in out.type.tensor_type.shape.dim]
        print(f"      Shape: {shape}")
        print(f"      Type: {out.type.tensor_type.elem_type}")

    # Validate shape requirements for C++ integration
    print("\n" + "=" * 60)
    print("C++ Compatibility Check")
    print("=" * 60)

    # Expected: 1 input, 1 output
    if len(graph.input) != 1:
        print(f"❌ FAIL: Expected 1 input, found {len(graph.input)}")
        return False
    print(f"✅ Input count: 1")

    if len(graph.output) != 1:
        print(f"❌ FAIL: Expected 1 output, found {len(graph.output)}")
        return False
    print(f"✅ Output count: 1")

    # Check input shape: [batch_size, 7]
    input_shape = [dim.dim_value for dim in graph.input[0].type.tensor_type.shape.dim]
    if len(input_shape) != 2:
        print(f"❌ FAIL: Expected 2D input, found {len(input_shape)}D: {input_shape}")
        return False

    if input_shape[1] != 7:
        print(f"❌ FAIL: Expected 7 input features, found {input_shape[1]}")
        print(f"   Required features: (r, θ, φ, v_r, v_θ, v_φ, t)")
        return False
    print(f"✅ Input shape: [batch_size, 7]")

    # Check output shape: [batch_size, 3]
    output_shape = [dim.dim_value for dim in graph.output[0].type.tensor_type.shape.dim]
    if len(output_shape) != 2:
        print(f"❌ FAIL: Expected 2D output, found {len(output_shape)}D: {output_shape}")
        return False

    if output_shape[1] != 3:
        print(f"❌ FAIL: Expected 3 output forces, found {output_shape[1]}")
        print(f"   Required outputs: (F_r, F_θ, F_φ)")
        return False
    print(f"✅ Output shape: [batch_size, 3]")

    # Test inference with ONNX Runtime
    print("\n" + "=" * 60)
    print("Inference Test")
    print("=" * 60)

    try:
        # Create inference session
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = 4
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        session = ort.InferenceSession(str(model_path), sess_options)

        # Test data: single particle at r=100, circular orbit
        test_input = np.array([
            [100.0, np.pi/2, 0.0, 0.0, 0.0, 10.0, 0.0]  # (r, θ, φ, v_r, v_θ, v_φ, t)
        ], dtype=np.float32)

        print(f"Test input (1 particle):")
        print(f"  r={test_input[0, 0]:.1f}, θ={test_input[0, 1]:.3f}, φ={test_input[0, 2]:.3f}")
        print(f"  v_r={test_input[0, 3]:.1f}, v_θ={test_input[0, 4]:.1f}, v_φ={test_input[0, 5]:.1f}")
        print(f"  t={test_input[0, 6]:.1f}")

        # Run inference
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        result = session.run([output_name], {input_name: test_input})[0]

        print(f"\nPredicted forces:")
        print(f"  F_r={result[0, 0]:.6f}")
        print(f"  F_θ={result[0, 1]:.6f}")
        print(f"  F_φ={result[0, 2]:.6f}")

        # Sanity check: radial force should be negative (gravity pulls inward)
        if result[0, 0] >= 0:
            print(f"⚠️  WARNING: F_r is positive (expected negative for gravity)")
        else:
            print(f"✅ F_r is negative (gravity pulls inward)")

        print(f"\n✅ Inference test passed!")

    except Exception as e:
        print(f"❌ FAIL: Inference test failed: {e}")
        return False

    # Batch inference test
    print("\n" + "=" * 60)
    print("Batch Inference Test")
    print("=" * 60)

    try:
        batch_size = 1000

        # Create batch: particles in circular orbit at various radii
        radii = np.linspace(50, 200, batch_size)
        batch_input = np.zeros((batch_size, 7), dtype=np.float32)
        batch_input[:, 0] = radii  # r
        batch_input[:, 1] = np.pi / 2  # θ (equatorial plane)
        batch_input[:, 2] = np.linspace(0, 2*np.pi, batch_size)  # φ (distributed around disk)
        batch_input[:, 5] = 10.0  # v_φ (tangential velocity)

        print(f"Batch size: {batch_size}")
        print(f"Radius range: [{radii[0]:.1f}, {radii[-1]:.1f}]")

        # Run batch inference
        import time
        start_time = time.perf_counter()
        batch_result = session.run([output_name], {input_name: batch_input})[0]
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        print(f"\nInference time: {elapsed_ms:.2f} ms")
        print(f"Particles/sec: {batch_size / (elapsed_ms / 1000):.0f}")
        print(f"Time per particle: {elapsed_ms / batch_size:.4f} ms")

        # Validate output shape
        if batch_result.shape != (batch_size, 3):
            print(f"❌ FAIL: Expected output shape ({batch_size}, 3), got {batch_result.shape}")
            return False

        # Check force statistics
        F_r_mean = np.mean(batch_result[:, 0])
        F_theta_mean = np.mean(batch_result[:, 1])
        F_phi_mean = np.mean(batch_result[:, 2])

        print(f"\nForce statistics (mean):")
        print(f"  F_r: {F_r_mean:.6f}")
        print(f"  F_θ: {F_theta_mean:.6f}")
        print(f"  F_φ: {F_phi_mean:.6f}")

        print(f"\n✅ Batch inference test passed!")

    except Exception as e:
        print(f"❌ FAIL: Batch inference test failed: {e}")
        return False

    # Final summary
    print("\n" + "=" * 60)
    print("Validation Summary")
    print("=" * 60)
    print("✅ ONNX model is valid and compatible with C++ inference engine")
    print("✅ Input shape: [batch_size, 7]")
    print("✅ Output shape: [batch_size, 3]")
    print("✅ Single particle inference works")
    print("✅ Batch inference works")
    print("\nModel ready for C++ integration!")

    return True


if __name__ == "__main__":
    success = validate_onnx_model()
    exit(0 if success else 1)
