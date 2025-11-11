"""
Technique Comparator Tool
Compares different volumetric rendering approaches
"""


async def compare_rendering_techniques(
    techniques: list,
    criteria: list
) -> str:
    """
    Compare different rendering techniques across multiple criteria

    Args:
        techniques: List of techniques to compare
        criteria: List of criteria (performance, visual_quality, memory, complexity, flexibility)

    Returns:
        Comparison table and analysis
    """

    report = "# Volumetric Rendering Technique Comparison\n\n"

    # Define technique properties
    technique_data = {
        "current_implementation": {
            "name": "Current Implementation (32-byte PLASMA)",
            "performance": {"score": 10, "fps_10k": 120, "fps_100k": 18},
            "visual_quality": {"score": 7, "note": "Excellent for plasma, limited to single material"},
            "memory_usage": {"score": 10, "mb_10k": 0.31, "mb_100k": 3.1},
            "implementation_complexity": {"score": 10, "note": "Baseline - already implemented"},
            "material_flexibility": {"score": 2, "note": "Homogeneous plasma only"}
        },
        "pure_volumetric_gaussian": {
            "name": "Pure Volumetric Gaussian (48-byte Multi-Material)",
            "performance": {"score": 8, "fps_10k": 108, "fps_100k": 16},
            "visual_quality": {"score": 9, "note": "Excellent for stars, gas, nebulae - all volumetric"},
            "memory_usage": {"score": 8, "mb_10k": 0.46, "mb_100k": 4.6},
            "implementation_complexity": {"score": 9, "note": "Moderate - extend structure + shader materials"},
            "material_flexibility": {"score": 8, "note": "8 material types via constant buffer"}
        },
        "hybrid_surface_volume": {
            "name": "Hybrid Surface/Volume (64-byte + Normal Approximation)",
            "performance": {"score": 6, "fps_10k": 90, "fps_100k": 13},
            "visual_quality": {"score": 10, "note": "Best quality - supports rocky/icy bodies with surfaces"},
            "memory_usage": {"score": 6, "mb_10k": 0.61, "mb_100k": 6.1},
            "implementation_complexity": {"score": 5, "note": "Complex - normal estimation + BRDF + hybrid shader"},
            "material_flexibility": {"score": 10, "note": "Full material range - volumetric + surface"}
        },
        "billboard_impostors": {
            "name": "Billboard Impostors (Sprite-based)",
            "performance": {"score": 10, "fps_10k": 240, "fps_100k": 60},
            "visual_quality": {"score": 4, "note": "Fast but loses volumetric depth and raytracing"},
            "memory_usage": {"score": 9, "mb_10k": 0.35, "mb_100k": 3.5},
            "implementation_complexity": {"score": 7, "note": "Requires new pipeline - rasterization instead of RT"},
            "material_flexibility": {"score": 5, "note": "Texture-based materials, no true volumetric"}
        },
        "adaptive_lod": {
            "name": "Adaptive LOD (Distance-Based Simplification)",
            "performance": {"score": 9, "fps_10k": 115, "fps_100k": 25},
            "visual_quality": {"score": 8, "note": "Smart quality/performance balance"},
            "memory_usage": {"score": 9, "mb_10k": 0.46, "mb_100k": 4.6},
            "implementation_complexity": {"score": 6, "note": "Moderate - requires LOD system + material switching"},
            "material_flexibility": {"score": 9, "note": "Full materials but simplified at distance"}
        }
    }

    # Filter to requested techniques
    selected_techniques = {k: v for k, v in technique_data.items() if k in techniques}

    if not selected_techniques:
        return "❌ No valid techniques selected. Available: " + ", ".join(technique_data.keys())

    report += "## Techniques Under Comparison\n\n"
    for i, (key, data) in enumerate(selected_techniques.items(), 1):
        report += f"{i}. **{data['name']}**\n"
    report += "\n"

    # Performance comparison
    if "performance" in criteria:
        report += "## Performance Comparison\n\n"
        report += "| Technique | Score | FPS @ 10K | FPS @ 100K | Notes |\n"
        report += "|-----------|-------|-----------|------------|-------|\n"

        for key, data in selected_techniques.items():
            perf = data["performance"]
            score_bar = "█" * perf["score"] + "░" * (10 - perf["score"])
            report += f"| {data['name'][:35]} | {score_bar} | {perf['fps_10k']} | {perf['fps_100k']} | "

            if perf["score"] >= 9:
                report += "Excellent\n"
            elif perf["score"] >= 7:
                report += "Good\n"
            else:
                report += "Acceptable\n"

        report += "\n"

    # Visual quality comparison
    if "visual_quality" in criteria:
        report += "## Visual Quality Comparison\n\n"
        report += "| Technique | Score | Assessment |\n"
        report += "|-----------|-------|------------|\n"

        for key, data in selected_techniques.items():
            quality = data["visual_quality"]
            score_bar = "★" * quality["score"] + "☆" * (10 - quality["score"])
            report += f"| {data['name'][:35]} | {score_bar} | {quality['note']} |\n"

        report += "\n"

    # Memory usage comparison
    if "memory_usage" in criteria:
        report += "## Memory Usage Comparison\n\n"
        report += "| Technique | Score | @ 10K Particles | @ 100K Particles | Efficiency |\n"
        report += "|-----------|-------|-----------------|------------------|------------|\n"

        for key, data in selected_techniques.items():
            mem = data["memory_usage"]
            score_bar = "█" * mem["score"] + "░" * (10 - mem["score"])
            efficiency = "Excellent" if mem["score"] >= 9 else "Good" if mem["score"] >= 7 else "Acceptable"
            report += f"| {data['name'][:35]} | {score_bar} | {mem['mb_10k']:.2f} MB | {mem['mb_100k']:.2f} MB | {efficiency} |\n"

        report += "\n"

    # Implementation complexity comparison
    if "implementation_complexity" in criteria:
        report += "## Implementation Complexity Comparison\n\n"
        report += "| Technique | Score | Effort | Details |\n"
        report += "|-----------|-------|--------|----------|\n"

        for key, data in selected_techniques.items():
            complexity = data["implementation_complexity"]
            score_bar = "●" * complexity["score"] + "○" * (10 - complexity["score"])

            if complexity["score"] >= 9:
                effort = "Minimal"
            elif complexity["score"] >= 7:
                effort = "Moderate"
            elif complexity["score"] >= 5:
                effort = "Significant"
            else:
                effort = "Major"

            report += f"| {data['name'][:35]} | {score_bar} | {effort} | {complexity['note']} |\n"

        report += "\n"

    # Material flexibility comparison
    if "material_flexibility" in criteria:
        report += "## Material Flexibility Comparison\n\n"
        report += "| Technique | Score | Capability | Supported Materials |\n"
        report += "|-----------|-------|------------|---------------------|\n"

        for key, data in selected_techniques.items():
            flexibility = data["material_flexibility"]
            score_bar = "▓" * flexibility["score"] + "░" * (10 - flexibility["score"])

            if flexibility["score"] >= 8:
                capability = "Excellent"
            elif flexibility["score"] >= 6:
                capability = "Good"
            elif flexibility["score"] >= 4:
                capability = "Limited"
            else:
                capability = "Minimal"

            report += f"| {data['name'][:35]} | {score_bar} | {capability} | {flexibility['note']} |\n"

        report += "\n"

    # Overall scores
    report += "## Overall Scores Summary\n\n"

    report += "| Technique | Performance | Quality | Memory | Complexity | Flexibility | **Total** |\n"
    report += "|-----------|-------------|---------|--------|------------|-------------|-------|\n"

    for key, data in selected_techniques.items():
        perf = data["performance"]["score"]
        quality = data["visual_quality"]["score"]
        memory = data["memory_usage"]["score"]
        complexity = data["implementation_complexity"]["score"]
        flexibility = data["material_flexibility"]["score"]
        total = perf + quality + memory + complexity + flexibility

        report += f"| {data['name'][:35]} | {perf}/10 | {quality}/10 | {memory}/10 | {complexity}/10 | {flexibility}/10 | **{total}/50** |\n"

    report += "\n"

    # Recommendations
    report += "## Recommendations\n\n"

    # Find best technique by criteria
    best_performance = max(selected_techniques.items(), key=lambda x: x[1]["performance"]["score"])
    best_quality = max(selected_techniques.items(), key=lambda x: x[1]["visual_quality"]["score"])
    best_flexibility = max(selected_techniques.items(), key=lambda x: x[1]["material_flexibility"]["score"])
    best_balance = max(selected_techniques.items(),
                       key=lambda x: sum([
                           x[1]["performance"]["score"],
                           x[1]["visual_quality"]["score"],
                           x[1]["material_flexibility"]["score"]
                       ]))

    report += f"**Best Performance:** {best_performance[1]['name']}\n"
    report += f"- {best_performance[1]['performance']['fps_10k']} FPS @ 10K particles\n"
    report += f"- {best_performance[1]['performance']['fps_100k']} FPS @ 100K particles\n\n"

    report += f"**Best Visual Quality:** {best_quality[1]['name']}\n"
    report += f"- {best_quality[1]['visual_quality']['note']}\n\n"

    report += f"**Best Material Flexibility:** {best_flexibility[1]['name']}\n"
    report += f"- {best_flexibility[1]['material_flexibility']['note']}\n\n"

    report += f"**Best Overall Balance:** {best_balance[1]['name']}\n"
    total_score = sum([
        best_balance[1]["performance"]["score"],
        best_balance[1]["visual_quality"]["score"],
        best_balance[1]["material_flexibility"]["score"]
    ])
    report += f"- Combined score: {total_score}/30 (performance + quality + flexibility)\n\n"

    # Strategic recommendations
    report += "## Strategic Recommendations\n\n"

    report += "**For Immediate Implementation:**\n"
    report += "- Start with **Pure Volumetric Gaussian (48-byte)** approach\n"
    report += "- Provides 8 material types with minimal performance impact (~10% FPS loss)\n"
    report += "- Excellent quality for stars, gas clouds, nebulae (80% of use cases)\n"
    report += "- Moderate complexity - achievable in 1-2 weeks\n\n"

    report += "**For Long-Term Quality:**\n"
    report += "- Upgrade to **Hybrid Surface/Volume (64-byte)** if rocky/icy bodies needed\n"
    report += "- Requires surface normal approximation and BRDF implementation\n"
    report += "- Best visual quality but ~25% FPS loss\n"
    report += "- Consider as Phase 2 after validating volumetric approach\n\n"

    report += "**For Maximum Performance:**\n"
    report += "- Implement **Adaptive LOD** system on top of chosen technique\n"
    report += "- Distant particles use simplified materials/shaders\n"
    report += "- Can recover 10-15% FPS with minimal quality loss\n"
    report += "- Compatible with both volumetric and hybrid approaches\n\n"

    report += "**Avoid:**\n"
    report += "- Billboard impostors - defeats purpose of volumetric Gaussian rendering\n"
    report += "- Loses ray tracing, volumetric depth, and physical accuracy\n"
    report += "- Only suitable if FPS requirements cannot be met with optimizations\n\n"

    report += "---\n"
    report += f"**Comparison completed across {len(criteria)} criteria for {len(selected_techniques)} techniques**\n"

    return report
