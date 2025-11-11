"""
Material Simulator Tool
Simulates how proposed material properties would affect rendering
"""


async def simulate_material_properties(
    material_type: str,
    properties: dict,
    render_mode: str = "volumetric_only"
) -> str:
    """
    Simulate material property changes and predict visual/performance impact

    Args:
        material_type: Material type to simulate (PLASMA, STAR, GAS, etc.)
        properties: Material properties dict (opacity, scattering, emission, albedo, phase_g)
        render_mode: Rendering mode (volumetric_only, hybrid_surface_volume, comparison)

    Returns:
        Simulation report with visual predictions and performance estimates
    """

    report = f"# Material Property Simulation: {material_type}\n\n"

    # Extract properties with defaults
    opacity = properties.get("opacity", 0.5)
    scattering = properties.get("scattering_coefficient", 1.0)
    emission = properties.get("emission_multiplier", 1.0)
    albedo = properties.get("albedo_rgb", [1.0, 1.0, 1.0])
    phase_g = properties.get("phase_function_g", 0.6)

    report += "## Input Parameters\n\n"
    report += f"**Material Type:** {material_type}\n"
    report += f"**Render Mode:** {render_mode}\n\n"

    report += "**Material Properties:**\n"
    report += f"- Opacity: {opacity:.2f}\n"
    report += f"- Scattering coefficient: {scattering:.2f}\n"
    report += f"- Emission multiplier: {emission:.2f}\n"
    report += f"- Albedo (RGB): ({albedo[0]:.2f}, {albedo[1]:.2f}, {albedo[2]:.2f})\n"
    report += f"- Phase function g: {phase_g:.2f} "

    if phase_g > 0.7:
        report += "(strongly forward scattering)\n"
    elif phase_g > 0.3:
        report += "(moderately forward scattering)\n"
    elif phase_g > -0.3:
        report += "(isotropic scattering)\n"
    else:
        report += "(backward scattering)\n"

    report += "\n## Visual Appearance Prediction\n\n"

    # Predict appearance based on properties
    if material_type in ["PLASMA_BLOB", "STAR_MAIN_SEQUENCE", "STAR_GIANT", "STAR_HYPERGIANT"]:
        report += "**Emission-Dominated Rendering:**\n"
        report += f"- Blackbody emission × {emission:.2f} multiplier\n"
        report += f"- Core brightness: {'Very high' if emission > 1.5 else 'High' if emission > 0.8 else 'Moderate'}\n"
        report += f"- Edge softness: {('Sharp' if opacity > 0.7 else 'Soft' if opacity > 0.3 else 'Very diffuse')}\n"

        if phase_g > 0.5:
            report += "- Halo effect: Strong (forward scattering creates bright rim)\n"
        else:
            report += "- Halo effect: Minimal\n"

    elif material_type in ["GAS_CLOUD", "DUST_PARTICLE"]:
        report += "**Scattering-Dominated Rendering:**\n"
        report += f"- Albedo color: RGB({albedo[0]:.2f}, {albedo[1]:.2f}, {albedo[2]:.2f})\n"
        report += f"- Scattering strength: {scattering:.2f}x baseline\n"

        if phase_g < 0:
            report += f"- Backward scattering (g={phase_g:.2f}): Creates wispy, diffuse appearance\n"
            report += "- Visual character: Nebula-like, soft edges, glow around backlighting\n"
        else:
            report += f"- Forward/isotropic scattering (g={phase_g:.2f}): More defined edges\n"

        report += f"- Opacity: {opacity:.2f} → {'Dense' if opacity > 0.7 else 'Medium' if opacity > 0.3 else 'Transparent'}\n"

    elif material_type in ["ROCKY_BODY", "ICY_BODY"]:
        report += "**Hybrid Surface/Volume Rendering:**\n"
        report += "⚠️  Note: Hybrid rendering requires additional shader modifications\n\n"
        report += f"- Albedo: RGB({albedo[0]:.2f}, {albedo[1]:.2f}, {albedo[2]:.2f})\n"
        report += f"- Surface reflection: {'High' if albedo[0] > 0.7 else 'Medium' if albedo[0] > 0.3 else 'Low'}\n"
        report += f"- Volume scattering: {scattering:.2f}x (for translucent regions)\n"
        report += "- Rendering approach: Surface normal approximation + volumetric core\n"

    else:  # CUSTOM
        report += "**Custom Material:**\n"
        report += "- Emission contribution: " + ("Primary" if emission > 0.5 else "Secondary" if emission > 0.1 else "Minimal") + "\n"
        report += "- Scattering contribution: " + ("Primary" if scattering > 0.5 else "Secondary" if scattering > 0.1 else "Minimal") + "\n"
        report += f"- Effective albedo influence: {'High' if emission < 0.3 and scattering > 0.5 else 'Medium' if emission < 0.6 else 'Low'}\n"

    report += "\n## Performance Impact Estimate\n\n"

    # Estimate shader cost
    shader_ops = 0

    # Base Gaussian intersection cost
    shader_ops += 20  # Ray-ellipsoid intersection

    # Emission cost
    if emission > 0.01:
        shader_ops += 5  # Temperature lookup + blackbody calculation

    # Scattering cost
    if scattering > 0.01:
        shader_ops += 8  # Albedo lookup + phase function + Beer-Lambert

    # Hybrid rendering cost
    if render_mode == "hybrid_surface_volume":
        shader_ops += 15  # Normal approximation + Fresnel + surface BRDF

    report += f"**Estimated Shader ALU Operations per Ray-Particle Intersection:**\n"
    report += f"- Base Gaussian intersection: 20 ops\n"
    report += f"- Emission calculation: {5 if emission > 0.01 else 0} ops\n"
    report += f"- Scattering calculation: {8 if scattering > 0.01 else 0} ops\n"
    report += f"- Hybrid surface cost: {15 if render_mode == 'hybrid_surface_volume' else 0} ops\n"
    report += f"- **Total: ~{shader_ops} ops**\n\n"

    report += f"**FPS Impact Estimate (@ 10K particles):**\n"
    baseline_ops = 33  # Current implementation
    overhead_percent = ((shader_ops - baseline_ops) / baseline_ops) * 100

    report += f"- Baseline: 120 FPS (current implementation, ~33 ops)\n"
    report += f"- Estimated: {120 * (baseline_ops / shader_ops):.1f} FPS ({overhead_percent:+.1f}% change)\n"
    report += f"- Status: {'✅ Acceptable' if overhead_percent < 15 else '⚠️ Moderate impact' if overhead_percent < 30 else '❌ Significant impact'}\n"

    report += "\n## Shader Modification Requirements\n\n"

    report += "**Required Changes to `particle_gaussian_raytrace.hlsl`:**\n\n"

    if material_type != "PLASMA_BLOB":
        report += "```hlsl\n"
        report += "// Material property lookup\n"
        report += f"MaterialProperties matProps = g_materialProperties[particle.materialType]; // {material_type}\n\n"

        report += "// Apply material-specific properties\n"
        if emission > 0.01:
            report += f"float3 emissionColor = CalculateBlackbody(particle.temperature) * {emission:.2f};\n"

        if scattering > 0.01:
            report += f"float3 albedo = float3({albedo[0]:.2f}, {albedo[1]:.2f}, {albedo[2]:.2f});\n"
            report += f"float scatter = {scattering:.2f};\n"
            report += f"float phaseFunctionG = {phase_g:.2f};\n"

        if opacity != 0.5:
            report += f"float effectiveOpacity = baseOpacity * {opacity:.2f};\n"

        report += "```\n\n"

    report += "## Recommendations\n\n"

    if material_type in ["PLASMA_BLOB", "STAR_MAIN_SEQUENCE"]:
        report += "✅ **This material is well-suited for pure volumetric Gaussian rendering**\n"
        report += "- No hybrid surface rendering needed\n"
        report += "- Standard Beer-Lambert absorption works well\n"
        report += "- Forward scattering creates natural star-like appearance\n"

    elif material_type in ["GAS_CLOUD", "DUST_PARTICLE"]:
        report += "✅ **Excellent candidate for albedo-based scattering**\n"
        report += "- Implement albedo lookup in shader\n"
        report += f"- Use phase_g={phase_g:.2f} for wispy appearance\n"
        report += "- Consider temporal variation (pulsing, turbulence)\n"

    elif material_type in ["ROCKY_BODY", "ICY_BODY"]:
        report += "⚠️  **Hybrid rendering required - significant shader modifications**\n"
        report += "- Approximate surface normal from Gaussian gradient\n"
        report += "- Implement simplified BRDF (Lambertian + Fresnel)\n"
        report += "- May need separate rendering pass for best quality\n"

    report += "\n---\n"
    report += f"**Simulation completed for {material_type} in {render_mode} mode**\n"

    return report
