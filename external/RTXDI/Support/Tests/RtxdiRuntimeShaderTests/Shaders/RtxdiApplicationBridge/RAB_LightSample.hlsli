#ifndef RTXDI_RAB_LIGHT_INFO_HLSLI
#define RTXDI_RAB_LIGHT_INFO_HLSLI

struct RAB_LightSample
{
    float unused;
};

RAB_LightSample RAB_EmptyLightSample()
{
    RAB_LightSample lightSample;

    return lightSample;
}

bool RAB_IsAnalyticLightSample(RAB_LightSample lightSample)
{
    return true;
}

float RAB_LightSampleSolidAnglePdf(RAB_LightSample lightSample)
{
    return 0.0;
}

#endif // RTXDI_RAB_LIGHT_INFO_HLSLI
