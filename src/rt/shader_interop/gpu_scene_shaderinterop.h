#ifndef GPU_SCENE_SHADERINTEROP_H_
#define GPU_SCENE_SHADERINTEROP_H_

#include "hlsl_cpp_compat.h"

// HLSL code
#ifdef __cplusplus
namespace rt
{
#endif

struct GPUScene
{
    float4x4 cameraFromRaster;
    float4x4 clipFromRender;
    float4x4 prevClipFromClip;
    float4x4 clipFromPrevClip;
    float3x4 renderFromCamera;
    float3x4 cameraFromRender;
    float3x4 lightFromRender;

    float3 cameraP;
    float lodScale;

    float3 dxCamera;
    float lensRadius;
    float3 dyCamera;
    float focalLength;

    float fov;
    float width;
    float height;
    uint dispatchDimX;
    uint dispatchDimY;

    float p22;
    float p23;

    float jitterX;
    float jitterY;
};

#ifdef __cplusplus
}
#endif

#endif
