#ifndef VOLUMES_H_
#define VOLUMES_H_

#include "containers.h"

namespace rt
{
struct Arena;

struct GPUOctreeNode
{
    float minValue;
    float maxValue;
    int childIndex;
    int parentIndex;
};

StaticArray<GPUOctreeNode> Volumes(Arena *arena);
} // namespace rt

#endif
