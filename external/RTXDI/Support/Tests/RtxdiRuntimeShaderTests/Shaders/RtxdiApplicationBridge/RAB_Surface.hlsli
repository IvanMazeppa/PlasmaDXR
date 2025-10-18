#ifndef RTXDI_RAB_SURFACE_HLSLI
#define RTXDI_RAB_SURFACE_HLSLI

#include "RAB_RandomSamplerState.hlsli"
#include "RAB_Material.hlsli"

static const bool kSpecularOnly = false;

struct RAB_Surface
{
    uint unused;
};

RAB_Surface RAB_EmptySurface()
{
    RAB_Surface surface;
    return surface;
}

bool RAB_IsSurfaceValid(RAB_Surface surface)
{
    return true;
}

float3 RAB_GetSurfaceWorldPos(RAB_Surface surface)
{
    return float3(0.0, 0.0, 0.0);
}

RAB_Material RAB_GetMaterial(RAB_Surface surface)
{
    return RAB_EmptyMaterial();
}

float3 RAB_GetSurfaceNormal(RAB_Surface surface)
{
    return float3(0.0, 0.0, 0.0);
}

float RAB_GetSurfaceLinearDepth(RAB_Surface surface)
{
    return 0.0;
}

RAB_Surface RAB_GetGBufferSurface(int2 pixelPosition, bool previousFrame)
{
    return RAB_EmptySurface();
}

bool RAB_GetSurfaceBrdfSample(RAB_Surface surface, inout RAB_RandomSamplerState rng, out float3 dir)
{
    dir = float3(0.0, 0.0, 0.0);
    return false;
}

float RAB_GetSurfaceBrdfPdf(RAB_Surface surface, float3 dir)
{
    return 0.0;
}

#endif // RTXDI_RAB_SURFACE_HLSLI
