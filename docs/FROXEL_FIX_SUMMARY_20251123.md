The "bug" you are seeing is actually the Debug Visualization Mode for the Froxel Grid density, which is still active in your running application because the compiled shader file is stale.
The Evidence
The Colors: The "Bright Red", "Yellow", and "Green" colors match the heat map gradient defined in DebugVisualizeFroxelDensity() inside shaders/froxel/sample_froxel_grid.hlsl:
Red: High Density
Yellow: Medium-High Density
Green: Medium-Low Density
Blue/Cyan: Low Density
The "Light 0" Correlation: The "Yellow ellipsoid" corresponds to Light 0 because Light 0 is at the center [0,0,0] of your accretion disk, which is also where the particle density is highest. The visualization is showing Density, not lighting.
The Timestamp Mismatch:
particle_gaussian_raytrace.dxil: Nov 21 18:52 (Older)
particle_gaussian_raytrace.hlsl: Nov 21 20:31 (Newer)
Your source code (.hlsl) is correct and calls RayMarchFroxelGrid, but the compiled binary (.dxil) that the engine loads is from before your latest fix, likely when you had DebugVisualizeFroxelDensity enabled.
Solution
You need to recompile your shaders. Since you are manually loading .dxil files in C++, the engine does not automatically recompile them when the source changes.
Run the shader compiler (dxc) for your Gaussian shader. Based on your logs, the command should look something like this:
dxc -T cs_6_5 -E main shaders/particles/particle_gaussian_raytrace.hlsl -Fo shaders/particles/particle_gaussian_raytrace.dxil
Additional Finding: Race Condition in Density Injection
While reviewing your code, I found a "technical correctness" issue in shaders/froxel/inject_density.hlsl that you should be aware of for the future:
// inject_density.hlsl lines 79-123g_densityGrid[voxelIdx + int3(0, 0, 0)] += contribution * w000;
You are performing a read-modify-write (+=) on a RWTexture3D<float>. In HLSL/DirectX 12, floating-point atomic operations are not natively supported on RWTexture3D without special handling (like InterlockedAdd on R32_UINT cast as float).
Consequence: This is a race condition. If multiple particles try to add density to the same voxel simultaneously, some writes will be lost.
Impact: For fog, this usually just results in slightly less density or minor flickering, which is often acceptable for "cost-effective" visuals. However, to fix it properly, you would typically use a R32_UINT texture and InterlockedAdd with atomic compare-exchange or fixed-point math.
Recompiling your shaders will fix the Red/Yellow/Green visualization bug immediately.
/Yellow/Green visualization bug immediately.