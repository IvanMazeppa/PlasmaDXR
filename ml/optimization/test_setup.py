"""
Quick test script to verify genetic optimizer setup

Tests:
1. DEAP installation
2. Executable path
3. Single benchmark run
4. Parameter encoding/decoding

Author: Claude Code
Date: 2025-11-29
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))


def test_deap_import():
    """Test DEAP import"""
    print("1. Testing DEAP import...")
    try:
        import deap
        from deap import base, creator, tools, algorithms
        print(f"   ✅ DEAP {deap.__version__} imported successfully")
        return True
    except ImportError as e:
        print(f"   ❌ DEAP import failed: {e}")
        print(f"   Install with: pip install deap")
        return False


def test_dependencies():
    """Test other dependencies"""
    print("\n2. Testing dependencies...")
    missing = []

    try:
        import numpy as np
        print(f"   ✅ NumPy {np.__version__}")
    except ImportError:
        print(f"   ❌ NumPy missing")
        missing.append("numpy")

    try:
        import matplotlib
        print(f"   ✅ Matplotlib {matplotlib.__version__}")
    except ImportError:
        print(f"   ❌ Matplotlib missing")
        missing.append("matplotlib")

    try:
        import scipy
        print(f"   ✅ SciPy {scipy.__version__}")
    except ImportError:
        print(f"   ❌ SciPy missing")
        missing.append("scipy")

    if missing:
        print(f"\n   Install missing: pip install {' '.join(missing)}")
        return False
    return True


def test_executable():
    """Test executable path"""
    print("\n3. Testing executable path...")

    # Try different possible paths
    possible_paths = [
        "../build/bin/Debug/PlasmaDX-Clean.exe",
        "../build/Debug/PlasmaDX-Clean.exe",
        "../../build/bin/Debug/PlasmaDX-Clean.exe",
        "../../build/Debug/PlasmaDX-Clean.exe"
    ]

    for path in possible_paths:
        full_path = Path(__file__).parent / path
        if full_path.exists():
            print(f"   ✅ Found executable: {full_path}")
            return str(full_path)

    print(f"   ❌ Executable not found in common locations")
    print(f"   Please build PlasmaDX-Clean first:")
    print(f"     MSBuild.exe build/PlasmaDX-Clean.sln /p:Configuration=Debug")
    return None


def test_optimizer_creation():
    """Test creating optimizer instance"""
    print("\n4. Testing optimizer creation...")

    try:
        from genetic_optimizer import GeneticOptimizer, ParameterBounds

        # Create optimizer
        optimizer = GeneticOptimizer(
            executable_path="dummy_path.exe",  # Won't run, just testing creation
            population_size=5,
            generations=2
        )

        print(f"   ✅ Optimizer created successfully")
        print(f"      Population size: {optimizer.population_size}")
        print(f"      Generations: {optimizer.generations}")
        print(f"      Parameters: {len(optimizer.param_names)}")

        # Test parameter bounds
        bounds = ParameterBounds()
        print(f"\n   Parameter bounds:")
        for param_name in optimizer.param_names:
            param_bounds = getattr(bounds, param_name)
            print(f"      {param_name:20s}: {param_bounds}")

        return True

    except Exception as e:
        print(f"   ❌ Failed to create optimizer: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_individual_creation():
    """Test creating and mutating individuals"""
    print("\n5. Testing individual creation...")

    try:
        from genetic_optimizer import GeneticOptimizer
        from deap import creator

        optimizer = GeneticOptimizer(
            executable_path="dummy_path.exe",
            population_size=5,
            generations=2
        )

        # Create individual
        individual = optimizer.toolbox.individual()
        print(f"   ✅ Individual created: {len(individual)} genes")

        # Decode individual
        params = optimizer.decode_individual(individual)
        print(f"   ✅ Decoded parameters:")
        for param, value in params.items():
            print(f"      {param:20s}: {value:.3f}")

        # Test mutation
        mutant, = optimizer.toolbox.mutate(individual)
        print(f"   ✅ Mutation successful")

        return True

    except Exception as e:
        print(f"   ❌ Individual creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_single_benchmark(executable_path):
    """Test running a single benchmark"""
    print(f"\n6. Testing single benchmark run...")
    print(f"   (This will take ~30-60 seconds)")

    try:
        from genetic_optimizer import GeneticOptimizer

        optimizer = GeneticOptimizer(
            executable_path=executable_path,
            pinn_model="v4",
            population_size=1,
            generations=1
        )

        # Create a test individual
        individual = optimizer.toolbox.individual()
        params = optimizer.decode_individual(individual)

        print(f"   Running benchmark with params:")
        print(f"      GM: {params['gm']:.1f}")
        print(f"      Alpha: {params['alpha']:.3f}")
        print(f"      BH Mass: {params['bh_mass']:.1f}")

        # Run benchmark
        results = optimizer.run_benchmark(params)

        if results is None:
            print(f"   ❌ Benchmark failed to run")
            return False

        # Check results
        fitness = optimizer.compute_fitness(results)
        print(f"   ✅ Benchmark completed successfully!")
        print(f"      Fitness: {fitness:.2f}")
        print(f"      Stability: {results.get('stability_score', 0):.2f}")
        print(f"      Accuracy: {results.get('accuracy_score', 0):.2f}")
        print(f"      Performance: {results.get('performance_score', 0):.2f}")

        return True

    except Exception as e:
        print(f"   ❌ Benchmark test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print(f"\n{'='*80}")
    print(f"GENETIC OPTIMIZER SETUP VERIFICATION")
    print(f"{'='*80}\n")

    all_passed = True

    # Test 1: DEAP
    if not test_deap_import():
        all_passed = False
        return

    # Test 2: Dependencies
    if not test_dependencies():
        all_passed = False
        return

    # Test 3: Executable
    executable_path = test_executable()
    if executable_path is None:
        print(f"\n⚠️  Skipping benchmark test (executable not found)")
        print(f"   Other tests can still run")
        executable_path = None

    # Test 4: Optimizer creation
    if not test_optimizer_creation():
        all_passed = False
        return

    # Test 5: Individual creation
    if not test_individual_creation():
        all_passed = False
        return

    # Test 6: Single benchmark (optional if executable found)
    if executable_path:
        benchmark_passed = test_single_benchmark(executable_path)
        if not benchmark_passed:
            all_passed = False
    else:
        print(f"\n6. Skipping benchmark test (no executable)")

    # Summary
    print(f"\n{'='*80}")
    if all_passed:
        print(f"✅ ALL TESTS PASSED!")
        print(f"\nReady to run optimization:")
        print(f"  python genetic_optimizer.py")
    else:
        print(f"❌ SOME TESTS FAILED")
        print(f"\nFix issues above before running optimization")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
