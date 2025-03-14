#include "sampling.hlsli"
#include "../rt/shader_interop/gpu_scene_shaderinterop.h"

void GenerateRay(GPUScene scene, float2 pFilm, float2 pLens, out float3 origin, out float3 dir, 
                 out float3 dpdx, out float3 dpdy, out float3 dddx, out float3 dddy)
{
    float4 homogeneousPCamera = mul(scene.cameraFromRaster, float4(pFilm, 0.f, 1.f));
    float3 pCamera            = homogeneousPCamera.xyz / homogeneousPCamera.w;
    origin = float3(0.f, 0.f, 0.f);
    dir = normalize(pCamera);
    if (scene.lensRadius > 0.f)
    {
        pLens = scene.lensRadius * SampleUniformDiskConcentric(pLens);

        DefocusBlur(dir, pLens, scene.focalLength, origin, dir);
        DefocusBlur(normalize(pCamera + scene.dxCamera), pLens, scene.focalLength, dpdx, dddx);
        DefocusBlur(normalize(pCamera + scene.dyCamera), pLens, scene.focalLength, dpdy, dddy);
    }
    origin = TransformP(scene.renderFromCamera, origin);
    dir = TransformV(scene.renderFromCamera, dir);

    dpdx = TransformP(scene.renderFromCamera, dpdx);
    dpdy = TransformP(scene.renderFromCamera, dpdy);
    dddx = TransformV(scene.renderFromCamera, dddx);
    dddy = TransformV(scene.renderFromCamera, dddy);
}

// See Ray Tracing Gems chapter 6
float3 OffsetRayOrigin(float3 p, float3 gn)
{
    static const float intScale = 256.f;
    static const float floatScale = 1.f / 65536;
    static const float origin = 1.f / 32;

    int3 of_i = intScale * gn;

    float3 p_i = float3(asfloat(asint(p.x) + copysign(of_i.x, p.x)),
        asfloat(asint(p.y) + copysign(of_i.y, p.y)),
        asfloat(asint(p.z) + copysign(of_i.z, p.z)));

    return float3(abs(p.x) < origin ? p.x + floatScale * gn.x : p_i.x,
                  abs(p.y) < origin ? p.y + floatScale * gn.y : p_i.y,
                  abs(p.z) < origin ? p.z + floatScale * gn.z : p_i.z);
}
