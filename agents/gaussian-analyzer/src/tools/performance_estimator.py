"""
Performance Estimator Tool
Calculates FPS impact of particle structure and shader changes
"""


async def estimate_performance_impact(
    particle_struct_bytes: int,
    material_types_count: int = 5,
    shader_complexity: str = "moderate",
    particle_counts: list = None
) -> str:
    """
    Estimate FPS impact of proposed changes

    Args:
        particle_struct_bytes: Proposed particle size (32-128 bytes)
        material_types_count: Number of material types (1-16)
        shader_complexity: minimal/moderate/complex
        particle_counts: List of particle counts to test

    Returns:
        Performance impact report with FPS estimates
    """

    if particle_counts is None:
        particle_counts = [10000, 50000, 100000]

    report = "# Performance Impact Estimation\n\n"

    # Current baseline
    current_size = 32
    current_fps = {
        10000: 120,
        50000: 45,
        100000: 18
    }

    report += "## Baseline Performance (Current Implementation)\n\n"
    report += "| Particles | FPS | Frame Time |\n"
    report += "|-----------|-----|------------|\n"
    for count in particle_counts:
        fps = current_fps.get(count, current_fps[100000] * (100000 / count))
        frame_ms = 1000 / fps
        report += f"| {count:,} | {fps:.1f} | {frame_ms:.2f}ms |\n"

    report += "\n## Memory Impact Analysis\n\n"

    memory_overhead = ((particle_struct_bytes - current_size) / current_size) * 100

    report += f"**Particle Structure Growth:**\n"
    report += f"- Current: {current_size} bytes\n"
    report += f"- Proposed: {particle_struct_bytes} bytes\n"
    report += f"- Growth: +{particle_struct_bytes - current_size} bytes ({memory_overhead:+.1f}%)\n\n"

    report += "**Memory Usage per Particle Count:**\n\n"
    report += "| Particles | Current | Proposed | Delta |\n"
    report += "|-----------|---------|----------|-------|\n"
    for count in particle_counts:
        current_mb = (count * current_size) / (1024 * 1024)
        proposed_mb = (count * particle_struct_bytes) / (1024 * 1024)
        delta_mb = proposed_mb - current_mb
        report += f"| {count:,} | {current_mb:.2f} MB | {proposed_mb:.2f} MB | +{delta_mb:.2f} MB |\n"

    report += "\n## Shader Complexity Impact\n\n"

    # Shader overhead factors
    shader_overhead = {
        "minimal": 5,    # Just lookup, no branches
        "moderate": 12,  # Lookup + material switch + branches
        "complex": 25    # Per-pixel raymarching + complex material evaluation
    }

    overhead_ops = shader_overhead.get(shader_complexity, 12)

    report += f"**Shader Complexity: {shader_complexity}**\n\n"

    if shader_complexity == "minimal":
        report += "- Material property lookup from constant buffer\n"
        report += "- Minimal branching (compile-time optimized)\n"
        report += f"- Additional ALU ops: ~{overhead_ops}\n"
        report += "- Best case: Material properties baked into constant buffer indexed by type\n"

    elif shader_complexity == "moderate":
        report += "- Material property lookup + switch statement\n"
        report += "- Per-type emission/scattering/phase function\n"
        report += f"- Additional ALU ops: ~{overhead_ops}\n"
        report += "- Typical case: Per-material branches with early-out optimizations\n"

    else:  # complex
        report += "- Per-pixel ray marching with material evaluation\n"
        report += "- Complex BRDF calculations (hybrid surface/volume)\n"
        report += f"- Additional ALU ops: ~{overhead_ops}\n"
        report += "- Worst case: Full hybrid rendering with normal approximation\n"

    report += "\n## Material Type System Overhead\n\n"

    # Constant buffer size
    material_cb_size = material_types_count * 32  # 32 bytes per material
    report += f"**Material Properties Constant Buffer:**\n"
    report += f"- Material types: {material_types_count}\n"
    report += f"- Bytes per material: 32 bytes (8 floats)\n"
    report += f"- Total CB size: {material_cb_size} bytes ({material_cb_size / 1024:.2f} KB)\n"
    report += f"- GPU impact: Negligible (<1 KB constant buffer)\n\n"

    report += "## Estimated FPS Impact\n\n"

    # Calculate total overhead percentage
    memory_bandwidth_impact = memory_overhead * 0.3  # Memory overhead affects bandwidth (30% correlation)
    shader_cost_impact = (overhead_ops / 33) * 100   # Current shader ~33 ops baseline

    total_impact = memory_bandwidth_impact + shader_cost_impact

    report += "**Overhead Breakdown:**\n"
    report += f"- Memory bandwidth: {memory_bandwidth_impact:.1f}%\n"
    report += f"- Shader complexity: {shader_cost_impact:.1f}%\n"
    report += f"- **Total estimated overhead: {total_impact:.1f}%**\n\n"

    report += "**Projected FPS:**\n\n"
    report += "| Particles | Current FPS | Projected FPS | Change |\n"
    report += "|-----------|-------------|---------------|--------|\n"

    for count in particle_counts:
        current = current_fps.get(count, current_fps[100000] * (100000 / count))
        projected = current * (1 - total_impact / 100)
        change = projected - current

        status = "✅" if abs(change) < 10 else "⚠️" if abs(change) < 20 else "❌"

        report += f"| {count:,} | {current:.1f} | {projected:.1f} | {change:+.1f} {status} |\n"

    report += "\n## Performance Targets Validation\n\n"

    projected_10k = current_fps[10000] * (1 - total_impact / 100)
    projected_100k = current_fps[100000] * (1 - total_impact / 100)

    report += "**Performance Goals:**\n"
    report += f"- 10K particles: Target 90-120 FPS\n"
    report += f"  - Current: {current_fps[10000]:.1f} FPS ✅\n"
    report += f"  - Projected: {projected_10k:.1f} FPS {'✅' if projected_10k >= 90 else '❌'}\n\n"

    report += f"- 100K particles: Target 60+ FPS\n"
    report += f"  - Current: {current_fps[100000]:.1f} FPS ❌ (below target)\n"
    report += f"  - Projected: {projected_100k:.1f} FPS ❌ (below target)\n"
    report += f"  - Note: PINN ML physics expected to boost 100K to 110+ FPS\n\n"

    report += "## Bottleneck Analysis\n\n"

    report += "**Primary Bottlenecks:**\n"
    report += "1. **BLAS/TLAS Rebuild** (2.1ms @ 100K particles)\n"
    report += "   - Impact: 47.6 FPS ceiling (21 FPS per frame)\n"
    report += "   - Mitigation: BLAS update (not rebuild) → +25% FPS\n\n"

    report += "2. **Memory Bandwidth** (particle buffer reads)\n"
    report += f"   - Current: {current_size} bytes × {particle_counts[-1]:,} particles = {(current_size * particle_counts[-1]) / (1024*1024):.2f} MB\n"
    report += f"   - Proposed: {particle_struct_bytes} bytes × {particle_counts[-1]:,} particles = {(particle_struct_bytes * particle_counts[-1]) / (1024*1024):.2f} MB\n"
    report += f"   - Additional bandwidth: {((particle_struct_bytes - current_size) * particle_counts[-1]) / (1024*1024):.2f} MB per frame\n\n"

    report += "3. **Shader ALU Cost**\n"
    report += f"   - Baseline: ~33 ops per ray-particle intersection\n"
    report += f"   - With materials: ~{33 + overhead_ops} ops\n"
    report += f"   - Relative cost: {((33 + overhead_ops) / 33) * 100:.1f}%\n\n"

    report += "## Optimization Recommendations\n\n"

    if total_impact > 20:
        report += "⚠️  **High performance impact detected!** Consider these optimizations:\n\n"

    report += "**Immediate Optimizations:**\n"
    report += "- Use constant buffer for material properties (not per-particle)\n"
    report += "- Implement material type switch with branch prediction hints\n"
    report += "- Add compile-time specialization for common material types\n"
    report += "- Use 16-byte alignment to avoid cache line splits\n\n"

    report += "**Advanced Optimizations:**\n"
    report += "- Particle LOD: Distant particles use simplified materials\n"
    report += "- Frustum culling: Don't build BLAS for off-screen particles\n"
    report += "- Material batching: Group particles by type for coherent memory access\n"
    report += "- Async compute: Overlap BLAS build with previous frame's rendering\n\n"

    report += "**Long-term Optimizations:**\n"
    report += "- BLAS update API: Refit existing BLAS instead of rebuild (+25% FPS)\n"
    report += "- GPU-driven culling: Compute shader culls particles before BLAS build\n"
    report += "- Release build: Current measurements are Debug build (+30% FPS)\n"
    report += "- PINN ML physics: Replace GPU physics shader (5-10× speedup @ 100K)\n\n"

    report += "## Conclusion\n\n"

    if total_impact < 10:
        report += f"✅ **Low impact** ({total_impact:.1f}%) - Implementation is feasible with minimal performance cost\n"
    elif total_impact < 20:
        report += f"⚠️  **Moderate impact** ({total_impact:.1f}%) - Acceptable with optimizations, stays within performance targets\n"
    else:
        report += f"❌ **High impact** ({total_impact:.1f}%) - Requires significant optimizations or reduced scope\n"

    report += f"\n**Recommended Approach:**\n"

    if particle_struct_bytes <= 48 and shader_complexity in ["minimal", "moderate"]:
        report += "- Proceed with proposed changes\n"
        report += "- Implement constant buffer material lookup\n"
        report += "- Profile with PIX to identify actual bottlenecks\n"
        report += "- Validate FPS with real measurements\n"
    else:
        report += "- Start with minimal structure (48 bytes)\n"
        report += "- Use simplest shader complexity (constant buffer lookup only)\n"
        report += "- Measure real performance before expanding\n"
        report += "- Consider phased rollout (5 types → 8 types → 16 types)\n"

    report += "\n---\n"
    report += f"**Estimation based on: {particle_struct_bytes}B structure, {material_types_count} types, {shader_complexity} shader complexity**\n"

    return report
