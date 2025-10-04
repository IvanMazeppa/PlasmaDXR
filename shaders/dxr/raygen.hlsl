// Raygen shader - entry point for each pixel
// Procedural lit unit box geometry for testing

struct RayPayload {
    float4 color;
};

// Global root signature
RaytracingAccelerationStructure g_scene : register(t0);
RWTexture2D<float4> g_output : register(u0);

[shader("raygeneration")]
void RayGen() {
    uint2 index = DispatchRaysIndex().xy;
    uint2 dims  = DispatchRaysDimensions().xy;
    float2 uv = float2(index)/float2(dims);

    // Procedural test: lit unit box
    float aspect = (float)dims.x / (float)dims.y;
    float2 ndc = uv*2.0-1.0; ndc.y = -ndc.y;
    float3 ro = float3(0,0,-3);
    float3 rd = normalize(float3(ndc.x*aspect, ndc.y, 1.0));
    float3 bmin=float3(-1,-1,-1), bmax=float3(1,1,1);
    float3 invD=1.0/rd; float3 t0=(bmin-ro)*invD, t1=(bmax-ro)*invD;
    float3 tmin=min(t0,t1), tmax=max(t0,t1);
    float tN=max(max(tmin.x,tmin.y),tmin.z), tF=min(min(tmax.x,tmax.y),tmax.z);
    float3 col=0;
    if (tF>=max(tN,0.0)) {
        float3 p=ro+rd*max(tN,0.0), n=0, L=normalize(float3(0.3,0.8,0.2));
        if (abs(p.x-(-1))<1e-3) n=float3(-1,0,0); else if (abs(p.x-1)<1e-3) n=float3(1,0,0);
        else if (abs(p.y-(-1))<1e-3) n=float3(0,-1,0); else if (abs(p.y-1)<1e-3) n=float3(0,1,0);
        else if (abs(p.z-(-1))<1e-3) n=float3(0,0,-1); else n=float3(0,0,1);
        float diff=max(0.0,dot(n,L)); col=diff*float3(1.0,0.95,0.85)*(0.7+0.3*abs(n));
    }
    g_output[index]=float4(col,1);
}