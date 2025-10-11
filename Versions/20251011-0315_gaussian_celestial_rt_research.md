# 3D Gaussian Particles as Celestial Bodies — RT Interaction Research

## Goals

- Give each particle astrophysical or stylized (sci‑fi) attributes.
- Use DXR 1.1 RayQuery + volumetric marching to produce compelling light/shadow phenomena.
- Keep performance scalable to 100K+ particles via sampling, LOD, and reuse.

## Property model (per‑particle extensions)

- Physical core: temperatureK (1000–40000), mass, radius, density; metallicity (0–1), age, rotation axis, spin rate.
- Emissive/spectral: spectralType (O..M proxy), spectralTintRGB, emissionIntensityScale; variability (amplitude, period, phase, noise).
- Atmosphere/dust halo: albedo (σ_s/σ_t), absorption σ_a, scattering σ_s, g (HG anisotropy); dustColor, haloScale, cocoonDensity.
- Magnetic/jet features: jetAxis, jetSpread, jetIntensity, hotspotLongitude/Latitude, hotspotWidth.
- Relativistic/stylized: dopplerStrength, gravRedshiftStrength, lensingProxyStrength; sciFiGlowPower, neonEdgeFactor, diffractionRingsStrength.

## Rendering interactions (realistic track)

1. Blackbody + spectral tint

Base emission from temperature; modulate by spectralTintRGB/metallicity.

1. Single‑scatter volumetrics in the march

At each step s: T *= exp(-σ_t·Δs); add σ_a·L_e for emission, and σ_s·Σ L_i(s)·p(cosθ) for in‑scattering. L_i(s) via short RayQuery occlusion toward K reservoir emitters.

1. Directional effects

Phase via HG(g). Increase g for forward scattering in jets; reduce in dust shells. Limb darkening when radius is large on screen: I(μ) = I0(1 − a(1 − μ) − b(1 − μ)^2).

1. Dust cocoon halos

Elevate σ_a near star‑forming particles; push hue toward dustColor; attenuate transmittance to create glowing envelopes.

1. Relativistic cues

Doppler shift by view‑aligned velocity; gravitational redshift vs. local potential; simple lensing proxy by screen‑space deflection near compact mass.

1. Temporal accumulation + denoise

Accumulate single‑scatter color; NRD/SVGF denoise using history, approximate normals (Gaussian axes), and depth proxy.

## Rendering interactions (sci‑fi track)

1. Neon emission law

L_e ∝ temperature^p with sciFiGlowPower (p ∈ [2,6]); clamp with soft rolloff.

1. Diffractive/starburst effects

Post kernel guided by jetAxis/rotation to create star spikes; modulate by brightness.

1. Chromatic aberration halo

Expand halo radius by wavelength; split RGB contributions based on spectral type.

1. Volumetric glyphs and filaments

Procedurally attach field‑aligned filaments to rotation axis; sample density from glyph noise within Gaussian support.

1. Quantized phase scattering

Replace HG with stepped lobes for stylized forward/back scattering; animate via period/phase.

## Data and AS considerations

- AABB generation: reflect anisotropy when jets/hotspots enlarge support; use conservative bounds.
- BLAS/TLAS: enable ALLOW_UPDATE and compact BLAS; prefer PREFER_FAST_TRACE steady state.
- LOD: cluster far particles; per‑cluster BLAS; drop reservoir K and Δs with distance.

## Sampling and performance

- Per‑pixel reservoir (K≈4–8) of emissive indices updated temporally; reuse across steps.
- Blue‑noise sequence for candidate rays; early‑out when T < threshold.
- Async compute: overlap AABB/BLAS refit and denoiser with physics.

## Debug/validation checklist

- Heatmaps: σ_t, albedo, g; reservoir size; occlusion hit/miss; T(s) over path.
- A/B: scalar pre‑pass vs single‑scatter reservoir path.
- Visual goals: shafts through dense regions, layered depth, anisotropic halos.

## Suggested next edits

- Implement single‑scatter in volumetric march and remove scalar multiply by g_rtLighting.
- Add reservoir buffers and minimal update step; cast short RayQuery occlusion rays.
- Wire SER flags; switch AS to ALLOW_UPDATE + compaction.
- Expose new particle fields; start with temperature, albedo, g, jetAxis.
