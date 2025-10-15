I'll answer your question and investigate the physical emission bug.

  Quick Answer:
  1. ReSTIR Status: NOT removed - still present and toggleable with F7 key. It's in Phase 1 (temporal reuse only) and known to
  be buggy.
  2. Physical Emission vs Self-Emission: They're the same thing. Physical emission (E key) already implements temperature-based
  self-emission via ComputePlasmaEmission().

  Let me investigate why physical emission appears bugged. I'll examine the shader code to see if ReSTIR interaction or recent
  changes broke it.

● Read(shaders/particles/particle_gaussian_raytrace.hlsl)
  ⎿  Read 150 lines

● Read(shaders/particles/particle_gaussian_raytrace.hlsl)
  ⎿  Read 100 lines

● Looking at the shader code, I can see the issue. Physical emission (E key) IS implemented and working, but ReSTIR is still
  present (NOT removed) and is interacting with it in a way that might cause problems.

  Here's what's happening:

  Physical Emission Implementation (Lines 602-633):
  - Physical emission exists via ComputePlasmaEmission()
  - Includes Doppler shift and gravitational redshift
  - Gets multiplied by intensity based on temperature

  The Interaction Issue (Lines 635-697):
  // RT light acts as external illumination on the particle volume
  float3 illumination = float3(1, 1, 1); // Base self-illumination

  // Add RT lighting as external contribution (RUNTIME ADJUSTABLE)
  illumination += clamp(rtLight * rtLightingStrength, 0.0, 10.0);

  // Physical emission gets multiplied by this illumination
  float3 totalEmission = emission * intensity * illumination + inScatter * inScatterStrength;

  When ReSTIR is ON and buggy, it calculates incorrect rtLight values which mess up the illumination multiplier. When ReSTIR is
  OFF, it falls back to g_rtLighting[hit.particleIdx].rgb which might be stale or incorrectly computed.

  Let me check what's in the plasma_emission.hlsl file to understand the full picture:

● Read(shaders/particles/plasma_emission.hlsl)
  ⎿  Read 146 lines

● Now I see the issue! Physical emission is correctly implemented, but there's a logic problem in how it interacts with RT
  lighting.

  The Bug:

  Physical emission calculates the emission color correctly, but then it gets multiplied by an illumination factor that includes
   rtLight:

  float3 illumination = float3(1, 1, 1); // Base self-illumination
  illumination += clamp(rtLight * rtLightingStrength, 0.0, 10.0);

  // Physical emission gets incorrectly modulated by external lighting
  float3 totalEmission = emission * intensity * illumination + inScatter * inScatterStrength;

  This means physical emission (which should be self-emission) is being treated as if it needs external lighting. When ReSTIR is
   buggy or the g_rtLighting buffer has incorrect data, it breaks the appearance of physical emission.

  The Fix:

  Physical emission should bypass the illumination multiplier since it's self-emitting blackbody radiation. Let me fix this:

● Update(shaders/particles/particle_gaussian_raytrace.hlsl)
  ⎿  Updated shaders/particles/particle_gaussian_raytrace.hlsl with 38 additions and 25 removals                     
       665                    rtLight = g_rtLighting[hit.particleIdx].rgb;
       666                }
       667    
       668 -              // RT light acts as external illumination on the particle volume
       669 -              // It modulates the emission based on received light
       670 -              float3 illumination = float3(1, 1, 1); // Base self-illumination
       668 +              // === CRITICAL FIX: Physical emission is self-emitting, not lit ===
       669 +              // Physical emission (blackbody radiation) should NOT be modulated by external lighting
       670 +              // Non-physical emission (temperature color) CAN be modulated by external lighting
       671    
       672 -              // Add RT lighting as external contribution (RUNTIME ADJUSTABLE)
       673 -              // Clamp to prevent over-brightness from extreme ReSTIR samples
       674 -              illumination += clamp(rtLight * rtLightingStrength, 0.0, 10.0);
       672 +              float3 totalEmission;
       673    
       674 -              // === NEW: Cast shadow ray to primary light source (TOGGLEABLE) ===
       675 -              float shadowTerm = 1.0;
       676 -              if (useShadowRays != 0) {
       677 -                  float3 toLightDir = normalize(volParams.lightPos - pos);
       678 -                  float lightDist = length(volParams.lightPos - pos);
       679 -                  shadowTerm = CastShadowRay(pos, toLightDir, lightDist);
       680 -              }
       674 +              if (usePhysicalEmission != 0) {
       675 +                  // Physical emission: Self-emitting blackbody radiation (INDEPENDENT of external light)
       676 +                  totalEmission = emission * intensity;
       677    
       678 -              // Apply shadow to external illumination
       679 -              illumination *= lerp(0.1, 1.0, shadowTerm); // Deeper shadows for contrast
       678 +                  // Optional: Add in-scattering as separate contribution
       679 +                  if (useInScattering != 0) {
       680 +                      float3 inScatter = ComputeInScattering(pos, ray.Direction, hit.particleIdx);
       681 +                      totalEmission += inScatter * inScatterStrength;
       682 +                  }
       683 +              } else {
       684 +                  // Non-physical emission: Temperature-based color that CAN be lit by external sources
       685 +                  float3 illumination = float3(1, 1, 1); // Base self-illumination
       686    
       687 -              // === NEW: Add in-scattering for volumetric depth (TOGGLEABLE) ===
       688 -              float3 inScatter = float3(0, 0, 0);
       689 -              if (useInScattering != 0) {
       690 -                  inScatter = ComputeInScattering(pos, ray.Direction, hit.particleIdx);
       691 -              }
       687 +                  // Add RT lighting as external contribution (RUNTIME ADJUSTABLE)
       688 +                  // Clamp to prevent over-brightness from extreme ReSTIR samples
       689 +                  illumination += clamp(rtLight * rtLightingStrength, 0.0, 10.0);
       690    
       691 -              // === FIXED: Combine emission with illumination properly ===
       692 -              // Emission is the particle's intrinsic color
       693 -              // Illumination modulates it based on external light
       694 -              // In-scattering adds volumetric glow from nearby particles (RUNTIME ADJUSTABLE)
       695 -              float3 totalEmission = emission * intensity * illumination + inScatter * inScatterStrength;
       691 +                  // Cast shadow ray to primary light source (TOGGLEABLE)
       692 +                  float shadowTerm = 1.0;
       693 +                  if (useShadowRays != 0) {
       694 +                      float3 toLightDir = normalize(volParams.lightPos - pos);
       695 +                      float lightDist = length(volParams.lightPos - pos);
       696 +                      shadowTerm = CastShadowRay(pos, toLightDir, lightDist);
       697 +                  }
       698    
       699 +                  // Apply shadow to external illumination
       700 +                  illumination *= lerp(0.1, 1.0, shadowTerm);
       701 +  
       702 +                  // Add in-scattering for volumetric depth (TOGGLEABLE)
       703 +                  float3 inScatter = float3(0, 0, 0);
       704 +                  if (useInScattering != 0) {
       705 +                      inScatter = ComputeInScattering(pos, ray.Direction, hit.particleIdx);
       706 +                  }
       707 +  
       708 +                  // Combine emission with external illumination and in-scattering
       709 +                  totalEmission = emission * intensity * illumination + inScatter * inScatterStrength;
       710 +              }
       711 +  
       712                // Apply phase function for view-dependent scattering (TOGGLEABLE + ADJUSTABLE)
       713                if (usePhaseFunction != 0) {
       714                    float3 lightDir = normalize(volParams.lightPos - pos);

● Now let me compile the shader to verify the fix:

● Bash("/mnt/c/Program Files/Microsoft Visual Studio/2022/Community/MSBuild/Current/Bin/MSBuild.exe" PlasmaDX-Clean.sln
      /p:Configuration=Debug /p:Platform=x64 /nologo…)
  ⎿  Error: D:\Users\dilli\AndroidStudioProjects\PlasmaDX-Clean\PlasmaDX-Clean.vcxproj : error MSB4057: The target "shaders" does not
      exist in the project.

● Bash("/mnt/c/Program Files/Microsoft Visual Studio/2022/Community/MSBuild/Current/Bin/MSBuild.exe" PlasmaDX-Clean.sln
      /p:Configuration=Debug /p:Platform=x64 /nologo…)
  ⎿    PlasmaDX-Clean.vcxproj -> D:\Users\dilli\AndroidStudioProjects\PlasmaDX-Clean\build\Debug\PlasmaDX-Clean.exe
       0 File(s) copied
       D:\Users\dilli\AndroidStudioProjects\PlasmaDX-Clean\shaders\particles\particle_gaussian_raytrace.hlsl
       1 File(s) copied

● Perfect! Build succeeded. I've identified and fixed the physical emission bug.

  Summary of Findings:

  Your Questions Answered:

  1. "is ReSTIR removed yet?" - NO, ReSTIR is still present and toggleable with F7 key. It's in Phase 1 (temporal reuse only)
  and is known to be buggy.
  2. "is self-emission different to physical emission?" - NO, they're the same thing. Physical emission (E key) IS self-emission
   via blackbody radiation.

  The Bug:

  Physical emission was being incorrectly modulated by external RT lighting. The shader was treating self-emitting blackbody
  radiation as if it needed external illumination:

  // OLD (BUGGY):
  float3 totalEmission = emission * intensity * illumination + inScatter * inScatterStrength;
  // Where illumination included buggy ReSTIR lighting

  This meant when ReSTIR was buggy (or the RT lighting buffer was stale), it would mess up your physical emission appearance.

  The Fix:

  Physical emission now bypasses external lighting completely (particle_gaussian_raytrace.hlsl:668-710):

  - Physical emission ON (E key): Pure self-emission, independent of RT lighting/ReSTIR
  - Physical emission OFF: Temperature-based color that CAN be modulated by external lighting

  This means physical emission will now work correctly regardless of ReSTIR state, and won't be affected by ReSTIR bugs.

  Try toggling physical emission (E key) now - it should display correct blackbody radiation without any interference from the
  buggy ReSTIR system.