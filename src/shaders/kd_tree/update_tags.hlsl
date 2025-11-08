#include "../../rt/shader_interop/kd_tree_shaderinterop.h"
int SubtreeSize(uint tag, uint N)
{
    int leftChildRoot = 2 * tag + 1;
    if (leftChildRoot >= N) return 0;

    // L in original paper
    int numLevelsTree = firstbithigh(N) + 1;
    // l in original paper
    int levelOfSubtree = firstbithigh(leftChildRoot) + 1;

    // L - l
    int numLevelsSubtree = numLevelsTree - levelOfSubtree;

    int first = (leftChildRoot + 1) << (numLevelsSubtree - 1);
    int onLevel = (1l << (numLevelsSubtree - 1)) - 1;
    int lastOnLastLevel = first + onLevel;
    int numMissingOnLastLevel = clamp(lastOnLastLevel - (int)N, 0, (1l << (numLevelsSubtree - 1)));

    int result = (1l << (numLevelsSubtree)) - 1 - numMissingOnLastLevel;
    return result;
}

int SegmentBegin(uint L, uint N, uint tag)
{
    int numSettled = (1l << L) - 1;
    int numLevelsTotal = firstbithigh(N) + 1;
    int numLevelsRemaining = numLevelsTotal - L;
    
    int firstNodeInThisLevel = numSettled;
    int numEarlierSubtreesOnSameLevel = tag - firstNodeInThisLevel;

    int numToLeftIfFull
        = numEarlierSubtreesOnSameLevel * ((1l << numLevelsRemaining) - 1);

    int numToLeftOnLastIfFull
        = numEarlierSubtreesOnSameLevel * (1l << (numLevelsRemaining - 1));

    int numTotalOnLastLevel
        = N - ((1l << (numLevelsTotal - 1)) - 1);

    int numReallyToLeftOnLast
        = min(numTotalOnLastLevel, numToLeftOnLastIfFull);
    int numMissingOnLast
        = numToLeftOnLastIfFull - numReallyToLeftOnLast;

    int result = numSettled + numToLeftIfFull - numMissingOnLast;
    return result;
}

int GetPivotPos(uint L, uint N, uint tag)
{
    return SegmentBegin(L, N, tag) + SubtreeSize(tag, N);
}

// Build temp
RWStructuredBuffer<uint64_t> tags : register(u0);
StructuredBuffer<uint> indices : register(t1);

// Data
RWStructuredBuffer<uint> dims : register(u2);
StructuredBuffer<float3> points : register(t3);

StructuredBuffer<float3> sceneBounds : register(t4);

struct Push 
{
    uint L;
};

[[vk::push_constant]] Push pc;

[numthreads(KD_TREE_WORKGROUP_SIZE, 1, 1)]
void main(uint3 dtID : SV_DispatchThreadID)
{
    // TODO IMPORTANT
    uint num = 8;//1u << 20u;
    uint L = pc.L;

    if (dtID.x >= num) return;

    uint tag = uint(tags[dtID.x] >> 32u);
    uint pivotPos = GetPivotPos(L, num, tag);

    // Get bounds by going up tree
    float3 minBounds = sceneBounds[0];
    float3 maxBounds = sceneBounds[1];

    int curr = tag;
    while (curr > 0)
    {
        int parent = (curr - 1) / 2;
        uint parentIndex = indices[parent];
        int parentDim = dims[parentIndex];
        float splitPos = points[parentIndex][parentDim];

        if (curr & 1)
        {
            maxBounds[parentDim] = min(maxBounds[parentDim], splitPos);
        }
        else 
        {
            minBounds[parentDim] = max(minBounds[parentDim], splitPos);
        }
        curr = parent;
    }

    uint index = indices[pivotPos];
    int pivotDim = dims[index];
    float pivotCoord = points[index][pivotDim];

    if (dtID.x < pivotPos)
    {
        tags[dtID.x] = 2 * tag + 1;
        maxBounds[pivotDim] = pivotCoord;
    }
    else if (dtID.x > pivotPos)
    {
        tags[dtID.x] = 2 * tag + 2;
        minBounds[pivotDim] = pivotCoord;
    }
    else 
    {
        tags[dtID.x] = tag;
    }

    if (dtID.x != pivotPos)
    {
        float3 extent = maxBounds - minBounds;
        int widestDimension = 0;
        if (extent.y > extent.z && extent.y > extent.x)
        {
            widestDimension = 1;
        }
        else if (extent.z > extent.x && extent.z > extent.y)
        {
            widestDimension = 2;
        }
        dims[indices[dtID.x]] = widestDimension;
    }
}
