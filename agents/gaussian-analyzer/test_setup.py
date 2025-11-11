#!/usr/bin/env python3
"""
Test script to verify Gaussian Analyzer MCP server setup
Run this after installation to ensure everything works
"""

import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_all_tools():
    """Test each tool to verify setup"""

    print("🧪 Testing Gaussian Analyzer MCP Server Setup\n")
    print("=" * 60)

    PROJECT_ROOT = "/mnt/d/Users/dilli/AndroidStudioProjects/PlasmaDX-Clean"

    # Test 1: Parameter Analyzer
    print("\n1️⃣  Testing parameter_analyzer...")
    try:
        from tools.parameter_analyzer import analyze_gaussian_parameters
        result = await analyze_gaussian_parameters(PROJECT_ROOT, "quick", "structure")
        assert len(result) > 100, "Result too short"
        print("   ✅ parameter_analyzer works!")
    except Exception as e:
        print(f"   ❌ parameter_analyzer failed: {e}")
        return False

    # Test 2: Material Simulator
    print("\n2️⃣  Testing material_simulator...")
    try:
        from tools.material_simulator import simulate_material_properties
        result = await simulate_material_properties(
            "GAS_CLOUD",
            {"opacity": 0.3, "scattering_coefficient": 1.0},
            "volumetric_only"
        )
        assert "GAS_CLOUD" in result
        print("   ✅ material_simulator works!")
    except Exception as e:
        print(f"   ❌ material_simulator failed: {e}")
        return False

    # Test 3: Performance Estimator
    print("\n3️⃣  Testing performance_estimator...")
    try:
        from tools.performance_estimator import estimate_performance_impact
        result = await estimate_performance_impact(48, 5, "moderate", [10000])
        assert "FPS" in result
        print("   ✅ performance_estimator works!")
    except Exception as e:
        print(f"   ❌ performance_estimator failed: {e}")
        return False

    # Test 4: Technique Comparator
    print("\n4️⃣  Testing technique_comparator...")
    try:
        from tools.technique_comparator import compare_rendering_techniques
        result = await compare_rendering_techniques(
            ["pure_volumetric_gaussian"],
            ["performance"]
        )
        assert "Comparison" in result
        print("   ✅ technique_comparator works!")
    except Exception as e:
        print(f"   ❌ technique_comparator failed: {e}")
        return False

    # Test 5: Struct Validator
    print("\n5️⃣  Testing struct_validator...")
    try:
        from tools.struct_validator import validate_particle_struct
        test_struct = """
        struct ParticleData {
            XMFLOAT3 position;
            XMFLOAT3 velocity;
            float temperature;
            float radius;
        };
        """
        result = await validate_particle_struct(test_struct, True, True)
        assert "Validation" in result
        print("   ✅ struct_validator works!")
    except Exception as e:
        print(f"   ❌ struct_validator failed: {e}")
        return False

    print("\n" + "=" * 60)
    print("✅ All tests passed! Setup is complete.\n")
    print("Next steps:")
    print("1. Add gaussian-analyzer to Claude Code MCP settings")
    print("2. Launch: /agent 3d-gaussian-volumetric-engineer")
    print("3. See INTEGRATION_GUIDE.md for usage examples")

    return True

if __name__ == "__main__":
    success = asyncio.run(test_all_tools())
    sys.exit(0 if success else 1)
