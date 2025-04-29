#include "../common.hlsli"

// Samples a texture with Catmull-Rom filtering, using 9 texture fetches instead of 16.
// Ref: https://gist.github.com/TheRealMJP/c83b8c0f46b63f3a88a5986f4fa982b1

float3 LinearToGamma(float3 rgb)
{
    return pow(max(rgb, 0), 1 / 2.2);
}

float3 GammaToLinear(float3 rgb)
{
    return pow(max(rgb, 0), 2.2);
}

struct Gamma 
{
    float3 Apply(float3 rgb)
    {
        return GammaToLinear(rgb);
    }
};

struct CatmullRomPositionsAndWeights 
{
    float2 texPos0;
    float2 texPos12;
    float2 texPos3;

    float2 w0, w12, w3;
};

CatmullRomPositionsAndWeights Create(float2 texPos0, float2 texPos12, float2 texPos3,
                                     float2 w0, float2 w12, float2 w3)
{
    CatmullRomPositionsAndWeights result;
    result.texPos0 = texPos0;
    result.texPos12 = texPos12;
    result.texPos3 = texPos3;

    result.w0 = w0;
    result.w12 = w12;
    result.w3 = w3;
    return result;
}

CatmullRomPositionsAndWeights GenerateCatmullRomWeights(in float2 uv, in float2 texSize)  
{
    float2 samplePos = uv * texSize;
    float2 texPos1 = floor(samplePos - 0.5f) + 0.5f;

    float2 f = samplePos - texPos1;

    float2 w0 = f * (-0.5f + f * (1.0f - 0.5f * f));
    float2 w1 = 1.0f + f * f * (-2.5f + 1.5f * f);
    float2 w2 = f * (0.5f + f * (2.0f - 1.5f * f));
    float2 w3 = f * f * (-0.5f + 0.5f * f);

    float2 w12 = w1 + w2;
    float2 offset12 = w2 / (w1 + w2);

    float2 texPos0 = texPos1 - 1;
    float2 texPos3 = texPos1 + 2;
    float2 texPos12 = texPos1 + offset12;

    texPos0 /= texSize;
    texPos3 /= texSize;
    texPos12 /= texSize;

    return Create(texPos0, texPos12, texPos3, w0, w12, w3);
}

template <typename Func>
float3 SampleTextureCatmullRom(in Texture2D tex, in SamplerState linearSampler, in float2 uv, in float2 texSize)
{
    Func func;

    CatmullRomPositionsAndWeights weights = GenerateCatmullRomWeights(uv, texSize);

    float3 result = 0.0f;
    result += func.Apply(tex.SampleLevel(linearSampler, float2(weights.texPos0.x, weights.texPos0.y), 0.0f).rgb) * weights.w0.x * weights.w0.y;
    result += func.Apply(tex.SampleLevel(linearSampler, float2(weights.texPos12.x, weights.texPos0.y), 0.0f).rgb) * weights.w12.x * weights.w0.y;
    result += func.Apply(tex.SampleLevel(linearSampler, float2(weights.texPos3.x, weights.texPos0.y), 0.0f).rgb) * weights.w3.x * weights.w0.y;

    result += func.Apply(tex.SampleLevel(linearSampler, float2(weights.texPos0.x, weights.texPos12.y), 0.0f).rgb) * weights.w0.x * weights.w12.y;
    result += func.Apply(tex.SampleLevel(linearSampler, float2(weights.texPos12.x, weights.texPos12.y), 0.0f).rgb) * weights.w12.x * weights.w12.y;
    result += func.Apply(tex.SampleLevel(linearSampler, float2(weights.texPos3.x, weights.texPos12.y), 0.0f).rgb) * weights.w3.x * weights.w12.y;

    result += func.Apply(tex.SampleLevel(linearSampler, float2(weights.texPos0.x, weights.texPos3.y), 0.0f).rgb) * weights.w0.x * weights.w3.y;
    result += func.Apply(tex.SampleLevel(linearSampler, float2(weights.texPos12.x, weights.texPos3.y), 0.0f).rgb) * weights.w12.x * weights.w3.y;
    result += func.Apply(tex.SampleLevel(linearSampler, float2(weights.texPos3.x, weights.texPos3.y), 0.0f).rgb) * weights.w3.x * weights.w3.y;

    return result;
}

float3 SampleTextureCatmullRom(in Texture2DArray tex, in SamplerState linearSampler, in float3 uv, in float2 texSize)
{
    CatmullRomPositionsAndWeights weights = GenerateCatmullRomWeights(uv.xy, texSize);

    float3 result = 0.0f;
    result += tex.SampleLevel(linearSampler, float3(weights.texPos0.x, weights.texPos0.y, uv.z), 0.0f).rgb  * weights.w0.x * weights.w0.y;
    result += tex.SampleLevel(linearSampler, float3(weights.texPos12.x, weights.texPos0.y, uv.z), 0.0f).rgb * weights.w12.x * weights.w0.y;
    result += tex.SampleLevel(linearSampler, float3(weights.texPos3.x, weights.texPos0.y, uv.z), 0.0f).rgb * weights.w3.x * weights.w0.y;

    result += tex.SampleLevel(linearSampler, float3(weights.texPos0.x, weights.texPos12.y, uv.z), 0.0f).rgb * weights.w0.x * weights.w12.y;
    result += tex.SampleLevel(linearSampler, float3(weights.texPos12.x, weights.texPos12.y, uv.z), 0.0f).rgb * weights.w12.x * weights.w12.y;
    result += tex.SampleLevel(linearSampler, float3(weights.texPos3.x, weights.texPos12.y, uv.z), 0.0f).rgb * weights.w3.x * weights.w12.y;

    result += tex.SampleLevel(linearSampler, float3(weights.texPos0.x, weights.texPos3.y, uv.z), 0.0f).rgb * weights.w0.x * weights.w3.y;
    result += tex.SampleLevel(linearSampler, float3(weights.texPos12.x, weights.texPos3.y, uv.z), 0.0f).rgb * weights.w12.x * weights.w3.y;
    result += tex.SampleLevel(linearSampler, float3(weights.texPos3.x, weights.texPos3.y, uv.z), 0.0f).rgb * weights.w3.x * weights.w3.y;

    return result;
}

float3 IntersectRayPlane(float3 planeN, float3 planeP, float3 rayP, float3 rayD)
{
    float d   = dot(planeN, planeP);
    float t   = (d - dot(planeN, rayP)) / dot(planeN, rayD);
    float3 p = rayP + t * rayD;
    return p;
}
