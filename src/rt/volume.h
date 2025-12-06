#ifndef VOLUMES_H_
#define VOLUMES_H_

#include "containers.h"
#include "graphics/vulkan.h"

namespace rt
{
struct Arena;
struct CommandBuffer;

struct GPUOctreeNode
{
    float minValue;
    float maxValue;
    int childIndex;
    int parentIndex;
};

struct VolumeData 
{
    StaticArray<GPUOctreeNode> octree;
    TransferBuffer vdbDataBuffer;
};

VolumeData Volumes(CommandBuffer *cmd, Arena *arena);
} // namespace rt

#endif
