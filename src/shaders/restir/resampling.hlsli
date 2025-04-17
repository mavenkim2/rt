#ifndef RESAMPLING_HLSLI_
#define RESAMPLING_HLSLI_

#include "../common.hlsli"

struct Reservoir 
{
    float totalWeight;
    float sourcePdf;
    float M;

    void Add(float u, float weight, float inSourcePdf)
    {
        M++;
        totalWeight += weight;

        if (u * totalWeight < weight)
        {
            sample = sample;
            sourcePdf = inSourcePdf;
        }
    }
};

struct AliasEntry 
{
    float threshold;
    float p;
    float pAlias;
    uint alias;
};

uint SampleAlias(StructuredBuffer<AliasEntry> aliasTable, float u, inout float pdf)
{
    uint numValues, stride;
    aliasTable.GetDimensions(numValues, stride);

    float index             = u * numValues;
    uint flooredIndex       = floor(index);
    uint lookupIndex        = min(flooredIndex, numValues - 1);
    AliasEntry entry = &entries[lookupIndex];

    float q = min(index - flooredIndex, OneMinusEpsilon);
    if (q < entry.threshold)
    {
        pdf = entry.p;
        return lookupIndex;
    }
    pdf = entry.pAlias;
    return entry.alias;
}

struct LightSample 
{
    float3 L;
    float3 samplePoint;
    float3 wi;
    float3 n;
    float pdf;
};

struct EnvironmentMap 
{
    float3x4 renderFromLight;
    Texture2D<float4> image;

    LightSample Sample(float3 pos, float2 u)
    {
        uint entry = SampleAlias(aliasTable, u, pdf);
    }
};

void GenerateRestirInitialCandidates(inout Reservoir reservoir, inout RNG rng, float3 pos) 
{
    // Generate canonical light samples for each pixel
    for (int lightSampleIndex = 0; lightSampleIndex < numLightSamples; lightSampleIndex++)
    {
        float pdf;
        float2 lightU = rng.Uniform2D();

        LightSample lightSample;

        float reservoirU = rng.Uniform();

        float3 bsdfValue = EvaluateSample();

        bool visible = VisibilityRay(rts, pos, lightSample.wi, FLT_MAX);
        // f * G * V * Le is the target function used
        float risWeight = visible * Luminance(bsdfValue * lightSample.L) / lightSample.pdf;
        reservoir.Add(reservoirU, risWeight, lightSample.pdf);
    }

    // For now, using 1/M mis weights
    float Wx = (reservoir.totalWeight / M) / reservoir.sourcePdf;

    bsdfValue * lightSample.L * Wx;
}

}

#endif
