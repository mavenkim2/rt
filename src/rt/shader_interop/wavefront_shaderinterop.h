#ifndef WAVEFRONT_SHADERINTEROP_H_
#define WAVEFRONT_SHADERINTEROP_H_

#include "hlsl_cpp_compat.h"

#ifdef __cplusplus
namespace rt
{
#endif

struct PixelInfo
{
    float3 radiance;
    float3 throughput;
    uint pixelLocation_specularBounce;
    float rayConeWidth;
    float rayConeSpread;
};

struct WavefrontDescriptors
{
    // TODO: AOS vs SOA for pixel state?

    // Pixel state
    // int pixelLocationIndex;
    // int throughputIndex;
    // int radianceIndex;
    // int depthIndex;
    //
    // int throughputRWIndex;
    // int radianceRWIndex;
    // int depthRWIndex;

    // Ray kernel
    int rayQueuePosIndex;
    int rayQueueDirIndex;
    int rayQueuePixelIndex;

    int rayQueueRWPosIndex;
    int rayQueueRWDirIndex;
    int rayQueueRWPixelIndex;

    // Hit shading
    int hitShadingQueueClusterIDIndex;
    int hitShadingQueueInstanceIDIndex;
    int hitShadingQueueBaryIndex;
    int hitShadingQueuePixelIndex;
    int hitShadingQueueRNGIndex;
    int hitShadingQueueRayTIndex;
    int hitShadingQueueDirIndex;

    int hitShadingQueueRWClusterIDIndex;
    int hitShadingQueueRWInstanceIDIndex;
    int hitShadingQueueRWBaryIndex;
    int hitShadingQueueRWPixelIndex;
    int hitShadingQueueRWRNGIndex;
    int hitShadingQueueRWRayTIndex;
    int hitShadingQueueRWDirIndex;
};

#define WAVEFRONT_QUEUE_SIZE        (1u << 20u)
#define WAVEFRONT_RAY_QUEUE_INDEX   1
#define WAVEFRONT_SHADE_QUEUE_INDEX 2

#ifdef __cplusplus
}
#endif

#endif
