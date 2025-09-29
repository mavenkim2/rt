#include "../../rt/shader_interop/gpu_scene_shaderinterop.h"
#include "../../rt/shader_interop/as_shaderinterop.h"
#include "../../rt/shader_interop/hierarchy_traversal_shaderinterop.h"
#include "../../rt/shader_interop/dense_geometry_shaderinterop.h"

RWStructuredBuffer<uint> globals : register(u0);
RWStructuredBuffer<BLASData> blasDatas : register(u1);
StructuredBuffer<GPUInstance> instances : register(t2);
RWStructuredBuffer<uint> resourceBitVector : register(u3);

[numthreads(128, 1, 1)]
void main(uint3 dtID : SV_DispatchThreadID) 
{
    uint instanceIndex = dtID.x;
    if (instanceIndex >= 1u << 22u) return;

    GPUInstance instance = instances[instanceIndex];
    if (instance.flags & (GPU_INSTANCE_FLAG_FREED | GPU_INSTANCE_FLAG_MERGED)) return;

    //if (instance.resourceID > 8 || instance.resourceID == 2) return;
    //if (instanceIndex > 0) return;// && instance.resourceID != 18 && instance.resourceID != 19) return;

    // Share
    BLASData blasData = (BLASData)0;
    blasData.instanceID = instanceIndex;
    uint blasIndex;
    InterlockedAdd(globals[GLOBALS_BLAS_COUNT_INDEX], 1, blasIndex);
    //uint sharedInstance = maxMinLodLevel[instance.resourceID].x & ((1u << 27u) - 1u);
    //uint minMaxLodLevel = maxMinLodLevel[instance.resourceID].y;

    // Merge
    {
        blasData.addressIndex = GPU_INSTANCE_FLAG_MERGED_INSTANCE;

        //if (instance.minLodLevel < minMaxLodLevel)
        {
            uint wasSet;
            uint bit = 1u << (instance.resourceID & 31u);
            InterlockedOr(resourceBitVector[instance.resourceID >> 5u], bit, wasSet);
        }
    }
    blasDatas[blasIndex] = blasData;
}
