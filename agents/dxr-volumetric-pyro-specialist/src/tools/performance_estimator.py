"""
Pyro Performance Estimator
Estimates FPS impact of volumetric pyro effects
"""

async def estimate_pyro_performance(effect_complexity: str = "moderate",
                                    particle_count: int = 10000,
                                    noise_octaves: int = 3,
                                    temporal_animation: bool = True,
                                    base_fps: float = 120.0) -> dict:
    """Estimate FPS impact of pyro effects"""

    # ALU operations per particle by complexity
    complexity_alu = {
        "minimal": 30,    # Simple expansion
        "moderate": 80,   # Noise + dynamics
        "complex": 150,   # Multi-octave noise + physics
        "extreme": 250    # Full pyro solver
    }

    alu_ops = complexity_alu.get(effect_complexity, 80)
    alu_ops += (noise_octaves - 1) * 20  # Each octave adds ALU cost

    # FPS impact estimation
    particle_factor = (particle_count / 10000) ** 0.8  # Sub-linear scaling
    complexity_factor = alu_ops / 80  # Relative to moderate
    temporal_factor = 1.1 if temporal_animation else 1.0

    total_factor = particle_factor * complexity_factor * temporal_factor
    fps_impact_percent = min(40, total_factor * 15)
    estimated_fps = base_fps * (1 - fps_impact_percent / 100)

    # Memory bandwidth
    bandwidth_gb_s = (particle_count * 32 * 60) / (1024 ** 3)  # 32 bytes/particle @ 60fps

    return {
        "complexity": effect_complexity,
        "particle_count": particle_count,
        "noise_octaves": noise_octaves,
        "alu_ops": alu_ops,
        "fps_impact_percent": fps_impact_percent,
        "base_fps": base_fps,
        "estimated_fps": estimated_fps,
        "bandwidth_gb_s": bandwidth_gb_s
    }


async def format_performance_estimate(results: dict) -> str:
    """Format performance estimate as report"""

    report = f"""# Pyro Performance Estimate

## Configuration
- **Effect Complexity**: {results['complexity'].title()}
- **Particle Count**: {results['particle_count']:,}
- **Noise Octaves**: {results['noise_octaves']}
- **Temporal Animation**: {'Yes' if results.get('temporal_animation', True) else 'No'}

## Performance Impact

### FPS Estimation (RTX 4060 Ti @ 1080p)
- **Baseline FPS**: {results['base_fps']:.0f}
- **FPS Impact**: -{results['fps_impact_percent']:.1f}%
- **Estimated FPS**: **{results['estimated_fps']:.0f} FPS**

### GPU Resource Usage
- **ALU Ops/Particle**: ~{results['alu_ops']} operations
- **Memory Bandwidth**: +{results['bandwidth_gb_s']:.2f} GB/s
- **Shader Complexity**: {results['complexity'].title()}

## Optimization Recommendations

{'✅ **WITHIN BUDGET**: Effect should run at target 90+ FPS' if results['estimated_fps'] >= 90 else '⚠️ **OPTIMIZATION NEEDED**: Consider reducing particle count or simplifying noise'}

### If Performance Issues:
1. Reduce noise octaves ({results['noise_octaves']} → {max(1, results['noise_octaves']-1)})
2. Use LOD: Full detail near camera, simplified far
3. Cache noise lookups for distant particles
4. Consider hybrid approach (pyro near, billboards far)

---
*Performance estimate by DXR Volumetric Pyro Specialist*
"""
    return report
