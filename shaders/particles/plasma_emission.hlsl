// Physical Plasma Emission Model for Accretion Disk
// Based on blackbody radiation and Shakura-Sunyaev disk model

// Planck's blackbody radiation spectrum
// Returns emission intensity at given wavelength (nm) and temperature (K)
float PlanckEmission(float wavelength, float temperature) {
    const float h = 6.626e-34; // Planck constant
    const float c = 3e8;       // Speed of light
    const float k = 1.381e-23; // Boltzmann constant

    // Convert wavelength from nm to meters
    float lambda = wavelength * 1e-9;

    // Planck's law: B(λ,T) = (2hc²/λ⁵) / (exp(hc/λkT) - 1)
    // Simplified for GPU computation
    float factor = 14387.7 / (wavelength * temperature); // hc/λkT in practical units

    // Prevent overflow
    if (factor > 80.0) return 0.0;

    float intensity = 1.0 / (exp(factor) - 1.0);
    intensity *= pow(wavelength, -5.0);

    return intensity;
}

// Convert temperature to RGB using accurate blackbody radiation
float3 BlackbodyColor(float temperature) {
    // CIE 1931 color matching functions peaks (nm)
    const float redWavelength = 700.0;   // Red peak
    const float greenWavelength = 546.0; // Green peak
    const float blueWavelength = 435.0;  // Blue peak

    // Sample blackbody spectrum at RGB wavelengths
    float3 rgb;
    rgb.r = PlanckEmission(redWavelength, temperature);
    rgb.g = PlanckEmission(greenWavelength, temperature);
    rgb.b = PlanckEmission(blueWavelength, temperature);

    // Normalize to preserve color while allowing HDR values
    float maxComponent = max(rgb.r, max(rgb.g, rgb.b));
    if (maxComponent > 0.0) {
        rgb /= maxComponent;
    }

    // Apply Stefan-Boltzmann scaling for total intensity
    float intensity = pow(temperature / 5778.0, 4.0); // Normalized to Sun's temperature

    return rgb * intensity;
}

// Shakura-Sunyaev accretion disk temperature profile
float DiskTemperature(float radius, float innerRadius, float blackHoleMass, float accretionRate) {
    // Temperature ∝ (M_dot * M / r³)^(1/4)
    // Simplified for real-time computation

    // Avoid singularity at center
    radius = max(radius, innerRadius);

    // Disk temperature profile: T ∝ r^(-3/4)
    float tempFactor = pow(innerRadius / radius, 0.75);

    // Maximum temperature at inner edge (scaled for visuals)
    const float maxTemp = 30000.0; // Kelvin

    return maxTemp * tempFactor;
}

// Doppler shift for rotating disk with adjustable strength
float3 DopplerShift(float3 baseColor, float3 velocity, float3 viewDir, float strength) {
    const float c = 300000.0; // Speed of light (km/s) - scaled for game units

    // Radial velocity component
    float radialVelocity = dot(velocity, viewDir);

    // Relativistic Doppler factor (modulated by strength)
    float beta = (radialVelocity / c) * strength; // Apply strength multiplier
    float dopplerFactor = sqrt((1.0 - beta) / (1.0 + beta));

    // Shift spectrum (simplified - shifts RGB channels)
    float3 shiftedColor;

    if (beta > 0) {
        // Redshift (moving away)
        shiftedColor.r = baseColor.r * (1.0 + beta);
        shiftedColor.g = baseColor.g;
        shiftedColor.b = baseColor.b * (1.0 - beta);
    } else {
        // Blueshift (moving toward)
        shiftedColor.r = baseColor.r * (1.0 + beta);
        shiftedColor.g = baseColor.g;
        shiftedColor.b = baseColor.b * (1.0 - beta);
    }

    // Apply intensity change from Doppler effect
    shiftedColor *= dopplerFactor;

    // Lerp between original and shifted based on strength
    return lerp(baseColor, shiftedColor, saturate(strength));
}

// Complete plasma emission for particle
float3 ComputePlasmaEmission(
    float3 position,
    float3 velocity,
    float temperature,
    float density,
    float3 viewPos) {

    // Get blackbody emission color
    float3 emission = BlackbodyColor(temperature);

    // Apply density modulation (optical depth)
    float opacity = 1.0 - exp(-density * 0.1);
    emission *= opacity;

    // Apply Doppler shift based on velocity (using default strength 1.0)
    float3 viewDir = normalize(viewPos - position);
    emission = DopplerShift(emission, velocity, viewDir, 1.0);

    // Add emission line for hot plasma (simplified H-alpha at 656nm)
    if (temperature > 10000.0) {
        float lineStrength = (temperature - 10000.0) / 20000.0;
        emission.r += lineStrength * 2.0; // H-alpha is red
    }

    return emission;
}

// Gravitational redshift near black hole with adjustable strength
float3 GravitationalRedshift(float3 color, float radius, float schwarzschildRadius, float strength) {
    // Simplified gravitational redshift: z = 1/sqrt(1 - rs/r) - 1
    float ratio = (schwarzschildRadius / radius) * strength; // Apply strength
    ratio = clamp(ratio, 0.0, 0.9); // Prevent singularity

    float redshiftFactor = 1.0 / sqrt(1.0 - ratio);

    // Apply redshift (shifts spectrum toward red)
    float3 redshifted;
    redshifted.r = color.r * redshiftFactor;
    redshifted.g = color.g * sqrt(redshiftFactor);
    redshifted.b = color.b / redshiftFactor;

    // Lerp between original and redshifted based on strength
    return lerp(color, redshifted, saturate(strength));
}