#ifndef GPU_SCENE_SHADERINTEROP_H_
#define GPU_SCENE_SHADERINTEROP_H_

// HLSL code
#ifdef __cplusplus
namespace rt
{
#endif

struct GPUScene
{
    float4x4 cameraFromRaster;
    float3x4 renderFromCamera;
    float3 dxCamera;
    float3 dyCamera;
    float lensRadius;
    float focalLength;

#ifndef __cplusplus
    void GenerateRay(float2 pFilm, float2 pLens, inout RayDesc desc, out RayPayload payload)
    {
        float4 homogeneousPCamera = mul(cameraFromRaster, float4(pFilm, 0.f, 1.f));
        float3 pCamera = homogeneousPCamera.xyz / homogeneousPCamera.w;
        desc.Origin    = float3(0.f, 0.f, 0.f);
        desc.Direction = normalize(pCamera);
        if (lensRadius > 0.f)
        {
            pLens = lensRadius * SampleUniformDiskConcentric(pLens);

            DefocusBlur(desc.Direction, pLens, focalLength, desc.Origin, desc.Direction);
            DefocusBlur(normalize(pCamera + dxCamera), pLens, focalLength, payload.pxOffset,
                        payload.dxOffset);
            DefocusBlur(normalize(pCamera + dyCamera), pLens, focalLength, payload.pyOffset,
                        payload.dyOffset);
        }
        Transform(desc, renderFromCamera);
        Transform(payload, renderFromCamera);
    }
#endif
};

#ifdef __cplusplus
}
#endif

#endif
