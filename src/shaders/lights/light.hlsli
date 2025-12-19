#ifndef LIGHT_HLSLI_
#define LIGHT_HLSLI_

#include "../../rt/shader_interop/hit_shaderinterop.h"
#include "../sampling.hlsli"

float3 SampleAreaLight(GPULight light, float2 u, float3 hitPos, float3 hitN, out float pdf)
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
    pdf *= .9f * samplePointPdf;
    float tMax = length(hitPos - lightSamplePos) * .99f;
    return lightSampleDirection;
}

float3 SampleLight(GPULight light, float2 u, float3 hitPos, float3 hitN, out float pdf)
{
    switch (light.lightType)
    {
        case GPULightType::Directional:
        {
        }
        case GPULightType::Area:
        {
            return SampleAreaLight(light, u, hitPos, hitN, pdf);
        }
    }
    return 0;
}

#endif
