"""
Performance Analysis Tool
Analyze shadow performance characteristics and optimization opportunities
"""

from pathlib import Path
from typing import Dict, Optional, List
import re


async def analyze_shadow_performance(
    project_root: str,
    technique: str = "raytraced",
    particle_count: int = 10000,
    light_count: int = 13,
    include_bottleneck_analysis: bool = True,
    include_optimization_suggestions: bool = True
) -> Dict:
    """
    Analyze shadow performance characteristics

    Args:
        project_root: Path to PlasmaDX-Clean project root
        technique: Shadow technique to analyze (pcss, raytraced, hybrid)
        particle_count: Number of particles for estimation
        light_count: Number of lights
        include_bottleneck_analysis: Identify performance bottlenecks
        include_optimization_suggestions: Generate optimization recommendations

    Returns:
        Dict with performance analysis results
    """

    results = {
        "technique": technique,
        "configuration": {
            "particle_count": particle_count,
            "light_count": light_count,
            "target_fps": 115,
            "target_frame_time_ms": 8.7  # 115 FPS = 8.7ms per frame
        },
        "estimates": {},
        "bottlenecks": [],
        "optimizations": []
    }

    # Analyze based on technique
    if technique == "pcss":
        results["estimates"] = analyze_pcss_performance(particle_count, light_count)
    elif technique == "raytraced":
        results["estimates"] = analyze_raytraced_performance(particle_count, light_count)
    elif technique == "hybrid":
        results["estimates"] = analyze_hybrid_performance(particle_count, light_count)

    # Identify bottlenecks
    if include_bottleneck_analysis:
        results["bottlenecks"] = identify_bottlenecks(
            results["estimates"],
            results["configuration"]
        )

    # Generate optimization suggestions
    if include_optimization_suggestions:
        results["optimizations"] = generate_optimization_suggestions(
            technique,
            results["estimates"],
            results["bottlenecks"],
            particle_count,
            light_count
        )

    return results


def analyze_pcss_performance(particle_count: int, light_count: int) -> Dict:
    """Estimate PCSS performance"""

    # Known PCSS performance from CLAUDE.md
    estimates = {
        "technique": "PCSS with temporal filtering",
        "presets": {
            "Performance": {
                "rays_per_light": 1,
                "estimated_fps": 115,
                "frame_time_ms": 8.7,
                "shadow_pass_time_ms": 2.1,
                "breakdown": {
                    "ray_generation": 0.5,
                    "poisson_sampling": 0.3,
                    "temporal_blend": 0.2,
                    "blocker_search": 0.8,
                    "pcf_filtering": 0.3
                }
            },
            "Balanced": {
                "rays_per_light": 4,
                "estimated_fps": 95,
                "frame_time_ms": 10.5,
                "shadow_pass_time_ms": 4.3,
                "breakdown": {
                    "ray_generation": 2.0,
                    "poisson_sampling": 0.8,
                    "temporal_blend": 0.2,
                    "blocker_search": 1.0,
                    "pcf_filtering": 0.3
                }
            },
            "Quality": {
                "rays_per_light": 8,
                "estimated_fps": 68,
                "frame_time_ms": 14.7,
                "shadow_pass_time_ms": 8.5,
                "breakdown": {
                    "ray_generation": 4.0,
                    "poisson_sampling": 1.5,
                    "temporal_blend": 0.2,
                    "blocker_search": 2.0,
                    "pcf_filtering": 0.8
                }
            }
        },
        "scaling": {
            "per_particle": 0.00021,  # ms per particle (2.1ms / 10K particles)
            "per_light": 0.16,        # ms per light (2.1ms / 13 lights)
            "per_ray": 0.26           # ms per ray (2.1ms / 8 rays total across lights)
        }
    }

    # Scale to actual configuration
    scale_factor = (particle_count / 10000.0) * (light_count / 13.0)

    for preset in estimates["presets"].values():
        preset["frame_time_ms"] *= scale_factor
        preset["shadow_pass_time_ms"] *= scale_factor
        preset["estimated_fps"] = 1000.0 / preset["frame_time_ms"]

        for key in preset["breakdown"]:
            preset["breakdown"][key] *= scale_factor

    return estimates


def analyze_raytraced_performance(particle_count: int, light_count: int) -> Dict:
    """Estimate raytraced shadow performance"""

    estimates = {
        "technique": "DXR 1.1 inline RayQuery volumetric shadows",
        "presets": {
            "Performance": {
                "rays_per_light": 1,
                "estimated_fps": 110,  # Slightly slower than PCSS (more accurate)
                "frame_time_ms": 9.1,
                "shadow_pass_time_ms": 2.9,
                "breakdown": {
                    "ray_generation": 0.3,
                    "tlas_traversal": 1.5,     # Most expensive
                    "ray_intersection": 0.8,
                    "volumetric_attenuation": 0.2,
                    "temporal_blend": 0.1
                }
            },
            "Balanced": {
                "rays_per_light": 4,
                "estimated_fps": 92,
                "frame_time_ms": 10.9,
                "shadow_pass_time_ms": 5.7,
                "breakdown": {
                    "ray_generation": 1.2,
                    "tlas_traversal": 3.0,     # Scales with ray count
                    "ray_intersection": 1.2,
                    "volumetric_attenuation": 0.2,
                    "temporal_blend": 0.1
                }
            },
            "Quality": {
                "rays_per_light": 8,
                "estimated_fps": 65,
                "frame_time_ms": 15.4,
                "shadow_pass_time_ms": 10.2,
                "breakdown": {
                    "ray_generation": 2.4,
                    "tlas_traversal": 6.0,     # Dominates at high ray counts
                    "ray_intersection": 1.5,
                    "volumetric_attenuation": 0.2,
                    "temporal_blend": 0.1
                }
            }
        },
        "scaling": {
            "per_particle": 0.00029,  # ms per particle (slightly higher than PCSS)
            "per_light": 0.22,        # ms per light (ray tracing overhead)
            "per_ray": 0.36           # ms per ray (TLAS traversal cost)
        },
        "memory": {
            "tlas_size_mb": particle_count * 64 / (1024 * 1024),  # 64 bytes per particle
            "shadow_buffers_mb": 8,  # Reuses PCSS ping-pong buffers
            "total_mb": particle_count * 64 / (1024 * 1024) + 8
        }
    }

    # Scale to actual configuration
    scale_factor = (particle_count / 10000.0) * (light_count / 13.0)

    for preset in estimates["presets"].values():
        preset["frame_time_ms"] *= scale_factor
        preset["shadow_pass_time_ms"] *= scale_factor
        preset["estimated_fps"] = 1000.0 / preset["frame_time_ms"]

        for key in preset["breakdown"]:
            preset["breakdown"][key] *= scale_factor

    return estimates


def analyze_hybrid_performance(particle_count: int, light_count: int) -> Dict:
    """Estimate hybrid shadow performance"""

    estimates = {
        "technique": "Hybrid: Raytraced near + cached far",
        "presets": {
            "Performance": {
                "rays_per_light": 1,
                "near_particles_pct": 30,  # 30% within 500 units
                "estimated_fps": 128,       # Better than pure raytraced
                "frame_time_ms": 7.8,
                "shadow_pass_time_ms": 1.6,
                "breakdown": {
                    "near_raytraced": 0.9,
                    "far_cached": 0.3,
                    "lod_blending": 0.2,
                    "temporal_blend": 0.2
                }
            }
        },
        "scaling": {
            "near_cost_per_particle": 0.00029,  # Same as raytraced
            "far_cost_per_particle": 0.00003,   # 10× cheaper (cached)
            "lod_overhead": 0.0001              # Transition blending cost
        }
    }

    return estimates


def identify_bottlenecks(estimates: Dict, config: Dict) -> List[Dict]:
    """Identify performance bottlenecks"""

    bottlenecks = []

    # Analyze each preset
    for preset_name, preset_data in estimates.get("presets", {}).items():
        if preset_data["estimated_fps"] < config["target_fps"]:
            # Performance below target
            bottlenecks.append({
                "preset": preset_name,
                "severity": "high" if preset_data["estimated_fps"] < config["target_fps"] * 0.8 else "medium",
                "issue": f"FPS below target: {preset_data['estimated_fps']:.1f} < {config['target_fps']}",
                "frame_time_ms": preset_data["frame_time_ms"],
                "target_frame_time_ms": config["target_frame_time_ms"],
                "overhead_ms": preset_data["frame_time_ms"] - config["target_frame_time_ms"]
            })

        # Identify most expensive breakdown component
        breakdown = preset_data.get("breakdown", {})
        if breakdown:
            max_component = max(breakdown.items(), key=lambda x: x[1])
            if max_component[1] > 2.0:  # > 2ms is significant
                bottlenecks.append({
                    "preset": preset_name,
                    "severity": "medium",
                    "issue": f"Expensive operation: {max_component[0]}",
                    "time_ms": max_component[1],
                    "percentage": (max_component[1] / preset_data["shadow_pass_time_ms"]) * 100
                })

    return bottlenecks


def generate_optimization_suggestions(
    technique: str,
    estimates: Dict,
    bottlenecks: List[Dict],
    particle_count: int,
    light_count: int
) -> List[Dict]:
    """Generate optimization suggestions"""

    optimizations = []

    # General raytracing optimizations
    if technique == "raytraced":
        optimizations.append({
            "priority": "high",
            "optimization": "Temporal shadow caching",
            "description": "Reuse PCSS temporal accumulation (already implemented)",
            "expected_improvement": "8× quality at 1× cost (67ms convergence)",
            "implementation_cost": "Low (reuse existing buffers)",
            "tradeoffs": "67ms temporal lag, ghosting on fast motion"
        })

        optimizations.append({
            "priority": "high",
            "optimization": "Early ray termination",
            "description": "Abort shadow ray when accumulated opacity > 0.99",
            "expected_improvement": "15-25% faster (avoid unnecessary traversal)",
            "implementation_cost": "Low (already in generated code)",
            "tradeoffs": "None (slight approximation acceptable)"
        })

        optimizations.append({
            "priority": "medium",
            "optimization": "Distance-based ray budget",
            "description": "Cast fewer rays for distant particles (< 1000 units)",
            "expected_improvement": "30-40% faster at high particle counts",
            "implementation_cost": "Medium (add distance LOD logic)",
            "tradeoffs": "Lower quality on distant particles (often acceptable)"
        })

        optimizations.append({
            "priority": "medium",
            "optimization": "Checkerboard shadow rendering",
            "description": "Compute shadows for every other pixel, spatial interpolation",
            "expected_improvement": "2× faster shadow pass",
            "implementation_cost": "Medium (add checkerboard pattern + interpolation)",
            "tradeoffs": "Slight spatial blur, works well with temporal accumulation"
        })

        optimizations.append({
            "priority": "low",
            "optimization": "Light importance culling",
            "description": "Skip shadow rays for lights contributing < 5% to final color",
            "expected_improvement": "20-30% faster with many lights (16+)",
            "implementation_cost": "Medium (integrate with RTXDI importance sampling)",
            "tradeoffs": "Slight lighting inaccuracy (usually imperceptible)"
        })

        optimizations.append({
            "priority": "low",
            "optimization": "Hybrid near/far shadows",
            "description": "Raytrace near particles, cache or skip far particles",
            "expected_improvement": "50-100% faster at extreme particle counts (50K+)",
            "implementation_cost": "High (requires shadow map or caching infrastructure)",
            "tradeoffs": "Complex implementation, transition artifacts"
        })

    # TLAS traversal bottlenecks
    has_tlas_bottleneck = any(
        "tlas_traversal" in str(b) or "ray_intersection" in str(b)
        for b in bottlenecks
    )

    if has_tlas_bottleneck:
        optimizations.append({
            "priority": "high",
            "optimization": "TLAS optimization flags",
            "description": "Use FAST_TRACE build flags for acceleration structure",
            "expected_improvement": "10-15% faster TLAS traversal",
            "implementation_cost": "Very low (change build flags)",
            "tradeoffs": "Slightly lower quality BVH (usually imperceptible)"
        })

        optimizations.append({
            "priority": "medium",
            "optimization": "Instance culling",
            "description": "Don't include distant particles (>2000 units) in TLAS",
            "expected_improvement": "20-30% faster TLAS rebuild + traversal",
            "implementation_cost": "Medium (add culling logic to BLAS building)",
            "tradeoffs": "Distant particles can't cast/receive shadows (often acceptable)"
        })

    # Multi-light bottlenecks
    if light_count > 10:
        optimizations.append({
            "priority": "medium",
            "optimization": "RTXDI integration",
            "description": "Let RTXDI select most important light, shadow only that light",
            "expected_improvement": f"{light_count}× fewer shadow rays (importance sampling)",
            "implementation_cost": "Medium (modify RTXDI raygen shader)",
            "tradeoffs": "Only shadows for selected light (RTXDI already handles importance)"
        })

    # Memory bottlenecks
    memory_mb = estimates.get("memory", {}).get("total_mb", 0)
    if memory_mb > 100:
        optimizations.append({
            "priority": "low",
            "optimization": "Compressed shadow buffers",
            "description": "Use R8_UNORM instead of R16_FLOAT for shadow buffers",
            "expected_improvement": "50% less shadow buffer memory (8MB → 4MB)",
            "implementation_cost": "Low (change buffer format)",
            "tradeoffs": "8-bit precision (usually sufficient for shadows)"
        })

    return optimizations


async def format_performance_analysis_report(results: Dict) -> str:
    """Format performance analysis as markdown report"""

    report = f"""# Shadow Performance Analysis

## Configuration

- **Technique**: {results['technique']}
- **Particle Count**: {results['configuration']['particle_count']:,}
- **Light Count**: {results['configuration']['light_count']}
- **Target FPS**: {results['configuration']['target_fps']}
- **Target Frame Time**: {results['configuration']['target_frame_time_ms']:.1f} ms

---

## Performance Estimates

"""

    estimates = results.get("estimates", {})
    for preset_name, preset_data in estimates.get("presets", {}).items():
        report += f"""### {preset_name} Preset

- **Rays per light**: {preset_data['rays_per_light']}
- **Estimated FPS**: {preset_data['estimated_fps']:.1f}
- **Frame time**: {preset_data['frame_time_ms']:.2f} ms
- **Shadow pass time**: {preset_data['shadow_pass_time_ms']:.2f} ms

**Breakdown**:
"""
        for component, time_ms in preset_data.get("breakdown", {}).items():
            pct = (time_ms / preset_data["shadow_pass_time_ms"]) * 100
            report += f"- {component}: {time_ms:.2f} ms ({pct:.1f}%)\n"

        report += "\n"

    # Scaling factors
    scaling = estimates.get("scaling", {})
    if scaling:
        report += """### Scaling Factors

"""
        for factor, value in scaling.items():
            report += f"- **{factor}**: {value:.5f} ms\n"

        report += "\n"

    # Memory usage
    memory = estimates.get("memory", {})
    if memory:
        report += f"""### Memory Usage

- **TLAS size**: {memory.get('tlas_size_mb', 0):.2f} MB
- **Shadow buffers**: {memory.get('shadow_buffers_mb', 0):.2f} MB
- **Total**: {memory.get('total_mb', 0):.2f} MB

"""

    # Bottlenecks
    bottlenecks = results.get("bottlenecks", [])
    if bottlenecks:
        report += "---\n\n## Performance Bottlenecks\n\n"

        for bottleneck in sorted(bottlenecks, key=lambda b: {"high": 0, "medium": 1, "low": 2}[b["severity"]]):
            report += f"""### {bottleneck['issue']} ({bottleneck['severity'].upper()} severity)

**Preset**: {bottleneck.get('preset', 'N/A')}
"""

            if "time_ms" in bottleneck:
                report += f"**Time**: {bottleneck['time_ms']:.2f} ms ({bottleneck.get('percentage', 0):.1f}% of shadow pass)\n"

            if "overhead_ms" in bottleneck:
                report += f"**Overhead**: {bottleneck['overhead_ms']:.2f} ms above target\n"

            report += "\n"

    # Optimizations
    optimizations = results.get("optimizations", [])
    if optimizations:
        report += "---\n\n## Optimization Suggestions\n\n"

        for opt in sorted(optimizations, key=lambda o: {"high": 0, "medium": 1, "low": 2}[o["priority"]]):
            report += f"""### {opt['optimization']} (Priority: {opt['priority'].upper()})

**Description**: {opt['description']}

**Expected Improvement**: {opt['expected_improvement']}

**Implementation Cost**: {opt['implementation_cost']}

**Tradeoffs**: {opt['tradeoffs']}

---

"""

    return report
