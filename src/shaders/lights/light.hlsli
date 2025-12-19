#ifndef LIGHT_HLSLI_
#define LIGHT_HLSLI_

#include "../common.hlsli"
#include "../../rt/shader_interop/hit_shaderinterop.h"
#include "envmap.hlsli"
#include "../sampling.hlsli"

StructuredBuffer<GPULight> lights : register(t30);

struct LightSample 
{
    float3 dir;
    float tMax; // float3 lightPos;
    float3 radiance;
    float pdf;

    void Init(float3 dir_, float tMax_, float3 radiance_, float pdf_)
    {
        dir = dir_;
        tMax = tMax_;
        radiance = radiance_;
        pdf = pdf_;
    }
};

// TODO: better sampling
LightSample SampleEnvmapLight(GPULight light, float2 u, float3 hitPos, float3 hitN)
{
    float3 lightSampleDirection = SampleUniformSphere(u);
    float pdf = 1.f/ (4 * PI); 
    bool deltaLight = true;

    float3 L = EnvMapLe(light.transform, light.bindlessIndex, lightSampleDirection);
    LightSample sample;
    sample.Init(lightSampleDirection, FLT_MAX, L, pdf);
    return sample;
}

LightSample SampleDirectionalLight(GPULight light, float2 u, float3 hitPos, float3 hitN)
{
    LightSample sample;
    sample.Init(light.dir, FLT_MAX, light.color, 1.f);
    return sample;
}

LightSample SampleAreaLight(GPULight light, float2 u, float3 hitPos, float3 hitN)
{
    float2 areaLightDim = light.dim;
    float3 p[4] = 
    {
        float3(areaLightDim.x, areaLightDim.y, 0.f) / 2,
        float3(-areaLightDim.x, areaLightDim.y, 0.f) / 2,
        float3(-areaLightDim.x, -areaLightDim.y, 0.f) / 2,
        float3(areaLightDim.x, -areaLightDim.y, 0.f) / 2,
    };
    
    float3x4 areaLightTransform = light.transform;

    // TODO: the camera transform should just be baked into this
    float3 cameraBase;
    Translate(areaLightTransform, cameraBase);
    
    p[0] = mul(areaLightTransform, float4(p[0], 1));
    p[1] = mul(areaLightTransform, float4(p[1], 1));
    p[2] = mul(areaLightTransform, float4(p[2], 1));
    p[3] = mul(areaLightTransform, float4(p[3], 1));
    
    float3 v00 = normalize(p[0] - hitPos);
    float3 v10 = normalize(p[1] - hitPos);
    float3 v01 = normalize(p[3] - hitPos);
    float3 v11 = normalize(p[2] - hitPos);
    
    float3 p01        = p[1] - p[0];
    float3 p02        = p[2] - p[0];
    float3 p03        = p[3] - p[0];
    float3 lightSamplePos;
    float area0        = 0.5f * length(cross(p01, p02));
    float area1        = 0.5f * length(cross(p02, p03));
    
    float div  = 1.f / (area0 + area1);
    float prob = area0 * div;
    // Then sample the triangle by area
    if (u[0] < prob)
    {
        u[0]       = u[0] / prob;
        float3 bary = SampleUniformTriangle(u);
        lightSamplePos = bary[0] * p[0] + bary[1] * p[1] + bary[2] * p[2];
    }
    else
    {
        u[0]       = (u[0] - prob) / (1 - prob);
        float3 bary = SampleUniformTriangle(u);
        lightSamplePos = bary[0] * p[0] + bary[1] * p[2] + bary[2] * p[3];
    }
    float3 lightSampleDirection = normalize(lightSamplePos - hitPos);
    float samplePointPdf = div * length2(hitPos - lightSamplePos) / abs(dot(hitN, lightSampleDirection));
    float pdf = samplePointPdf;
    float tMax = length(hitPos - lightSamplePos) * .99f;

    LightSample sample;
    sample.Init(lightSampleDirection, tMax, light.color, pdf);
    return sample;
}

LightSample SampleLightDir(GPULight light, float2 u, float3 hitPos, float3 hitN)
{
    switch (light.lightType)
    {
        case GPULightType::Envmap:
        {
            return SampleEnvmapLight(light, u, hitPos, hitN);
        }
        case GPULightType::Directional:
        {
            return SampleDirectionalLight(light, u, hitPos, hitN);
        }
        case GPULightType::Area:
        {
            return SampleAreaLight(light, u, hitPos, hitN);
        }
    }
    return (LightSample)0;
}

int PowerSampleLight(float u, out float pdf)
{
    // TODO: num lights
    uint numLights = 0;

    int lightSampleIndex = 0;
    float chosenImportance = 0.f;
    float weightTotal = 0.f;

    int envMapIndex = -1;

    for (int i = 0; i < numLights; i++)
    {
        if (lights[i].lightType == GPULightType::Envmap)
        {
            envMapIndex = i;
            continue;
        }

        float3 color = lights[i].color;

        float importance = .3 * color.x + .6 * color.y + .1 * color.z;
        weightTotal += importance;
        float prob = importance / weightTotal;

        if (u < prob)
        {
            u /= prob;
            lightSampleIndex = i;
            chosenImportance = importance;
        }
        else 
        {
            u = (u - prob) / (1 - prob);
        }
    }

    pdf = chosenImportance / weightTotal;

    if (envMapIndex != -1)
    {
        if (u < .1)
        {
            pdf = .1f;
            lightSampleIndex = envMapIndex;
        }
        else 
        {
            pdf *= .9f;
        }
    }

    return lightSampleIndex;
}

#if 0
            float lightSample = rng.Uniform();
            float weightTotal = 0.f;
            uint lightSampleIndex = 0;
            float chosenImportance = 0.f;
            // TODO hardcoded
            for (int i = 0; i < 22; i++)
            {
                float4 color = areaLightColors[i];

                float importance = .3 * color.x + .6 * color.y + .1 * color.z;
                weightTotal += importance;
                float prob = importance / weightTotal;

                if (lightSample < prob)
                {
                    lightSample /= prob;
                    lightSampleIndex = i;
                    chosenImportance = importance;
                }
                else 
                {
                    lightSample = (lightSample - prob) / (1 - prob);
                }
            }
            float lightPdf = chosenImportance / weightTotal;
            float3 lightSampleDirection;

            float2 lightDirSample = rng.Uniform2D();
            float tMax = FLT_MAX;
            bool deltaLight = false;

            if (lightSample < .1)
            {
                lightSampleDirection = SampleUniformSphere(lightDirSample);
                lightPdf = .1f / (4 * PI);
                deltaLight = true;
            }
            else 
            {
                float2 areaLightDim = areaLightDims[lightSampleIndex];

                float3 p[4] = 
                {
                    float3(areaLightDim.x, areaLightDim.y, 0.f) / 2,
                    float3(-areaLightDim.x, areaLightDim.y, 0.f) / 2,
                    float3(-areaLightDim.x, -areaLightDim.y, 0.f) / 2,
                    float3(areaLightDim.x, -areaLightDim.y, 0.f) / 2,
                };

                float3x4 areaLightTransform = areaLightTransforms[lightSampleIndex];
                Translate(areaLightTransform, -scene.cameraBase);

                p[0] = mul(areaLightTransform, float4(p[0], 1));
                p[1] = mul(areaLightTransform, float4(p[1], 1));
                p[2] = mul(areaLightTransform, float4(p[2], 1));
                p[3] = mul(areaLightTransform, float4(p[3], 1));

                float3 v00 = normalize(p[0] - origin);
                float3 v10 = normalize(p[1] - origin);
                float3 v01 = normalize(p[3] - origin);
                float3 v11 = normalize(p[2] - origin);

                float3 p01        = p[1] - p[0];
                float3 p02        = p[2] - p[0];
                float3 p03        = p[3] - p[0];
                float3 lightSamplePos;
                float area0        = 0.5f * length(cross(p01, p02));
                float area1        = 0.5f * length(cross(p02, p03));

                float div  = 1.f / (area0 + area1);
                float prob = area0 * div;
                // Then sample the triangle by area
                if (lightDirSample[0] < prob)
                {
                    lightDirSample[0]       = lightDirSample[0] / prob;
                    float3 bary = SampleUniformTriangle(lightDirSample);
                    lightSamplePos = bary[0] * p[0] + bary[1] * p[1] + bary[2] * p[2];
                }
                else
                {
                    lightDirSample[0]       = (lightDirSample[0] - prob) / (1 - prob);
                    float3 bary = SampleUniformTriangle(lightDirSample);
                    lightSamplePos = bary[0] * p[0] + bary[1] * p[2] + bary[2] * p[3];
                }
                lightSampleDirection = normalize(lightSamplePos - origin);
                float samplePointPdf = div * length2(origin - lightSamplePos) / abs(dot(hitInfo.n, lightSampleDirection));
                lightPdf *= .9f * samplePointPdf;
                tMax = length(origin - lightSamplePos) * .99f;
            }
#endif

#endif
