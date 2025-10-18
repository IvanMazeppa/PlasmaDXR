#ifndef RTXDI_RAB_RANDOM_SAMPLER_STATE_HLSLI
#define RTXDI_RAB_RANDOM_SAMPLER_STATE_HLSLI

struct RAB_RandomSamplerState
{
    uint unused;
};

RAB_RandomSamplerState RAB_InitRandomSampler(uint2 index, uint pass)
{
    RAB_RandomSamplerState rng;
    return rng;
}

float RAB_GetNextRandom(inout RAB_RandomSamplerState rng)
{
    return 0.0;
}

#endif // RTXDI_RAB_RANDOM_SAMPLER_STATE_HLSLI
