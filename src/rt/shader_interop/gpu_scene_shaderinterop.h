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
    float3x4 renderFromCamera;
    float3x4 cameraFromRender;
    float3x4 lightFromRender;
    float3 dxCamera;
    float lensRadius;
    float3 dyCamera;
    float focalLength;

    float fov;
    float height;
    uint dispatchDimX;
    uint dispatchDimY;
};

#ifdef __cplusplus
}
#endif

#endif
