#ifndef WAVEFRONT_SHADERINTEROP_H_
#define WAVEFRONT_SHADERINTEROP_H_

#include "hlsl_cpp_compat.h"

#ifdef __cplusplus
namespace rt
{
#endif

struct WavefrontQueue
{
    uint readOffset;
    uint writeOffset;
};

struct GenerateRayPushConstant
{
    int imageWidth;
    int imageHeight;
    int frameNum;
    float filterIntegral;
};

struct WavefrontPushConstant
{
    int finishedQueueIndex;
    int dispatchQueueIndex;
    int flush;
};

struct PixelInfo
{
    float3 radiance;
    float3 throughput;
    uint pixelLocation_specularBounce;
    float rayConeWidth;
    float rayConeSpread;
    uint rngState;
    uint depth;
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

    // Ray kernel sorting
    int rayQueueMinPosIndex;
    int rayQueueMaxPosIndex;

    // Miss kernel
    int missQueuePixelIndex;
    int missQueueDirIndex;

    // Hit shading
    int hitShadingQueueClusterIDIndex;
    int hitShadingQueueInstanceIDIndex;
    int hitShadingQueueBaryIndex;
    int hitShadingQueuePixelIndex;
    int hitShadingQueueRNGIndex;
    int hitShadingQueueRayTIndex;
    int hitShadingQueueDirIndex;
};

#define WAVEFRONT_QUEUE_SIZE                 (1u << 20u)
#define WAVEFRONT_WORKING_SET_SIZE           (1u << 19u)
#define WAVEFRONT_RAY_QUEUE_INDEX            0
#define WAVEFRONT_SHADE_QUEUE_INDEX          1
#define WAVEFRONT_MISS_QUEUE_INDEX           2
#define WAVEFRONT_GENERATE_CAMERA_RAYS_INDEX 3
#define WAVEFRONT_RAY_SORT_INDEX             4
#define WAVEFRONT_NUM_QUEUES                 5

#ifdef __cplusplus
}
#endif

#endif
