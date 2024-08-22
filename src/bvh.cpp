#include "bvh.h"
void BVH::Build(Arena *arena, Scene *inScene, u32 primsPerLeaf)
{
    maxPrimitivesPerLeaf    = primsPerLeaf;
    scene                   = inScene;
    u32 totalPrimitiveCount = scene->totalPrimitiveCount;
    AABB *aabbs             = PushArray(arena, AABB, totalPrimitiveCount);
    scene->GetAABBs(aabbs);
    Build(arena, aabbs, totalPrimitiveCount);
}

void BVH::Build(Arena *arena, AABB *aabbs, u32 count)
{
    nodeCount = 0;
    assert(count != 0);
    const u32 nodeCapacity = count * 2 - 1;
    nodes                  = PushArray(arena, Node, nodeCapacity);
    leafIndices            = PushArray(arena, u32, count);

    Node &node = nodes[nodeCount++];
    node       = {};
    node.count = count;
    for (u32 i = 0; i < count; i++)
    {
        node.aabb      = AABB(node.aabb, aabbs[i]);
        leafIndices[i] = i;
    }
    Subdivide(0, aabbs);
}

void BVH::Subdivide(u32 nodeIndex, AABB *aabbs)
{
    Node &node      = nodes[nodeIndex];
    f32 surfaceArea = node.aabb.SurfaceArea();
    if (surfaceArea == 0 || node.count <= 2)
        return;

    i32 axis = node.aabb.MaxDimension();
    // Split into buckets
    const i32 numBuckets = 12;

    struct BVHSplitBucket
    {
        AABB bounds;
        i32 count;
    };

    BVHSplitBucket bucket[12];

    // for (u32 i = node.offset; i < node.count; i++)
    // {
    // Node &node = nodes[nodeIndex];
    // for (u32 i = 0; i < node.count; i++)
    // {
    //     node.aabb = AABB(node.aabb, aabbs[leafIndices[node.offset + i]]);
    // }
    //
    // }

    for (u32 i = node.offset; i < node.count; i++)
    {
        i32 b = i32(numBuckets * node.aabb.Offset(aabbs[leafIndices[i]].Centroid())[axis]);
        if (b == numBuckets) b--;
        assert(b >= 0 && b < numBuckets);
        bucket[b].count++;
        bucket[b].bounds = Union(bucket[b].bounds, aabbs[i]);
    }

    const i32 numSplits = numBuckets - 1;
    f32 cost[numSplits] = {};

    // Compute half of the SAH cost
    AABB boundsBelow;
    i32 countBelow = 0;

    for (i32 b = 0; b < numBuckets; b++)
    {
        boundsBelow = Union(boundsBelow, bucket[b].bounds);
        countBelow += 1;
        cost[b] += boundsBelow.SurfaceArea() * countBelow;
    }

    AABB boundsAbove;
    i32 countAbove = 0;
    for (i32 b = numSplits; b >= 1; b--)
    {
        boundsAbove = Union(boundsAbove, bucket[b].bounds);
        countAbove += 1;
        cost[b - 1] += boundsAbove.SurfaceArea() * countAbove;
    }

    f32 minCost      = infinity;
    i32 chosenBucket = -1;
    for (i32 b = 0; b < numBuckets; b++)
    {
        if (cost[b] < minCost)
        {
            chosenBucket = b;
            minCost      = cost[b];
        }
    }
    // Total cost: Traversal cost + SAH cost
    minCost      = 0.5f + minCost / surfaceArea;
    f32 leafCost = f32(node.count);

    if (node.count < maxPrimitivesPerLeaf && leafCost <= minCost)
        return;

    // vec3 extent = node.aabb.GetHalfExtent();
    // vec3 min    = node.aabb.minP;
    // int axis    = 0;
    // if (extent.y > extent[axis]) axis = 1;
    // if (extent.z > extent[axis]) axis = 2;
    // f32 splitPos = min[axis] + extent[axis];

    f32 splitPos = node.aabb.Centroid()[axis];

    int i = node.offset;
    int j = i + node.count - 1;
    while (i <= j)
    {
        vec3 center = aabbs[leafIndices[i]].Center();
        f32 value   = center[axis];
        if (value < splitPos)
        {
            i++;
        }
        else
        {
            u32 temp         = leafIndices[j];
            leafIndices[j--] = leafIndices[i];
            leafIndices[i]   = temp;
        }
    }
    u32 leftCount = i - node.offset;
    if (leftCount == 0 || leftCount == node.count)
        return;
    u32 leftChildIndex  = nodeCount++;
    u32 rightChildIndex = nodeCount++;
    node.left           = leftChildIndex;

    Node &leftChild  = nodes[leftChildIndex];
    leftChild        = {};
    leftChild.offset = node.offset;
    leftChild.count  = leftCount;

    Node &rightChild  = nodes[rightChildIndex];
    rightChild        = {};
    rightChild.offset = i;
    rightChild.count  = node.count - leftCount;

    // node.count = 0;
    node.isNode = 1;
    UpdateNodeBounds(leftChildIndex, aabbs);
    UpdateNodeBounds(rightChildIndex, aabbs);

    Subdivide(leftChildIndex, aabbs);
    Subdivide(rightChildIndex, aabbs);
}

inline void BVH::UpdateNodeBounds(u32 nodeIndex, AABB *aabbs)
{
    Node &node = nodes[nodeIndex];
    for (u32 i = 0; i < node.count; i++)
    {
        node.aabb = AABB(node.aabb, aabbs[leafIndices[node.offset + i]]);
    }
}

//////////////////////////////
// BVH4
//
BVH4 CreateBVH4(Arena *arena, BVH *bvh)
{
    // Convert BVH-2 to BVH-4 by skipping every other level
    u32 bvh4NodeCount = (bvh->nodeCount + 3) / 4;
    BVH4 bvh4;
    bvh4.scene       = bvh->scene;
    bvh4.nodes       = PushArray(arena, UncompressedBVHNode, bvh4NodeCount);
    bvh4.nodeCount   = bvh->nodeCount;
    bvh4.leafIndices = std::move(bvh->leafIndices);

    u32 stack[64];
    stack[0]                         = 0;
    i32 stackCount                   = 1;
    u32 runningBVH4NodeCount         = 1;
    bvh->nodes[0].compressedBVHIndex = 0;

    const AABB emptyAABB;

    if (bvh->nodes[0].IsLeaf())
    {
        f32 minX[4] = {
            bvh->nodes[0].aabb.minX,
            emptyAABB.minX,
            emptyAABB.minX,
            emptyAABB.minX,
        };
        bvh4.nodes[0].minP.x = Load(minX);

        f32 minY[4] = {
            bvh->nodes[0].aabb.minY,
            emptyAABB.minY,
            emptyAABB.minY,
            emptyAABB.minY,
        };
        bvh4.nodes[0].minP.y = Load(minY);

        f32 minZ[4] = {
            bvh->nodes[0].aabb.minZ,
            emptyAABB.minZ,
            emptyAABB.minZ,
            emptyAABB.minZ,
        };
        bvh4.nodes[0].minP.z = Load(minZ);

        f32 maxX[4] = {
            bvh->nodes[0].aabb.maxX,
            emptyAABB.maxX,
            emptyAABB.maxX,
            emptyAABB.maxX,
        };
        bvh4.nodes[0].maxP.x = Load(maxX);

        f32 maxY[4] = {
            bvh->nodes[0].aabb.maxY,
            emptyAABB.maxY,
            emptyAABB.maxY,
            emptyAABB.maxY,
        };
        bvh4.nodes[0].maxP.y = Load(maxY);

        f32 maxZ[4] = {
            bvh->nodes[0].aabb.maxZ,
            emptyAABB.maxZ,
            emptyAABB.maxZ,
            emptyAABB.maxZ,
        };
        bvh4.nodes[0].maxP.z = Load(maxZ);

        bvh4.nodes[0].offsetIndex[0] = bvh->nodes[0].offset;
        bvh4.nodes[0].count[0]       = SafeTruncateU32(bvh->nodes[0].count);
        bvh4.nodes[0].leafMask       = 2;
        return bvh4;
    }

    while (stackCount > 0)
    {
        assert(runningBVH4NodeCount < bvh4NodeCount);
        u32 top          = stack[--stackCount];
        BVH::Node *node  = &bvh->nodes[top];
        BVH::Node *left  = &bvh->nodes[node->left];
        BVH::Node *right = &bvh->nodes[node->left + 1];

        i32 grandChildrenIndices[] = {
            -1,
            -1,
            -1,
            -1,
        };

        u32 childrenCount = 0;
        if (left->IsLeaf())
        {
            grandChildrenIndices[childrenCount++] = node->left;
        }
        else
        {
            grandChildrenIndices[childrenCount++] = left->left;
            grandChildrenIndices[childrenCount++] = left->left + 1;
        }

        if (right->IsLeaf())
        {
            grandChildrenIndices[childrenCount++] = node->left + 1;
        }
        else
        {
            grandChildrenIndices[childrenCount++] = right->left;
            grandChildrenIndices[childrenCount++] = right->left + 1;
        }

        BVH::Node *grandChildren[4] = {
            grandChildrenIndices[0] != -1 ? &bvh->nodes[grandChildrenIndices[0]] : 0,
            grandChildrenIndices[1] != -1 ? &bvh->nodes[grandChildrenIndices[1]] : 0,
            grandChildrenIndices[2] != -1 ? &bvh->nodes[grandChildrenIndices[2]] : 0,
            grandChildrenIndices[3] != -1 ? &bvh->nodes[grandChildrenIndices[3]] : 0,
        };

        UncompressedBVHNode *currentNode = &bvh4.nodes[node->compressedBVHIndex];
        currentNode->leafMask            = 0;

        for (u32 i = 0; i < ArrayLength(grandChildren); i++)
        {
            BVH::Node *grandChild = grandChildren[i];
            assert(grandChild != &bvh->nodes[0]);
            if (!grandChild) continue;

            b8 isLeaf = grandChild->IsLeaf();
            if (!isLeaf)
            {
                stack[stackCount++]            = grandChildrenIndices[i];
                currentNode->childIndex[i]     = runningBVH4NodeCount;
                grandChild->compressedBVHIndex = runningBVH4NodeCount++;
            }
            else
            {
                currentNode->offsetIndex[i] = grandChild->offset;
                currentNode->count[i]       = SafeTruncateU32(grandChild->count);
            }
            b8 mask = isLeaf ? 2 : 1;
            currentNode->leafMask |= (mask << (2 * i));
        }

        f32 minX[4] = {
            grandChildren[0] ? grandChildren[0]->aabb.minX : emptyAABB.minX,
            grandChildren[1] ? grandChildren[1]->aabb.minX : emptyAABB.minX,
            grandChildren[2] ? grandChildren[2]->aabb.minX : emptyAABB.minX,
            grandChildren[3] ? grandChildren[3]->aabb.minX : emptyAABB.minX,
        };
        currentNode->minP.x = Load(minX);

        f32 minY[4] = {
            grandChildren[0] ? grandChildren[0]->aabb.minY : emptyAABB.minY,
            grandChildren[1] ? grandChildren[1]->aabb.minY : emptyAABB.minY,
            grandChildren[2] ? grandChildren[2]->aabb.minY : emptyAABB.minY,
            grandChildren[3] ? grandChildren[3]->aabb.minY : emptyAABB.minY,
        };
        currentNode->minP.y = Load(minY);

        f32 minZ[4] = {
            grandChildren[0] ? grandChildren[0]->aabb.minZ : emptyAABB.minZ,
            grandChildren[1] ? grandChildren[1]->aabb.minZ : emptyAABB.minZ,
            grandChildren[2] ? grandChildren[2]->aabb.minZ : emptyAABB.minZ,
            grandChildren[3] ? grandChildren[3]->aabb.minZ : emptyAABB.minZ,
        };
        currentNode->minP.z = Load(minZ);

        f32 maxX[4] = {
            grandChildren[0] ? grandChildren[0]->aabb.maxX : emptyAABB.maxX,
            grandChildren[1] ? grandChildren[1]->aabb.maxX : emptyAABB.maxX,
            grandChildren[2] ? grandChildren[2]->aabb.maxX : emptyAABB.maxX,
            grandChildren[3] ? grandChildren[3]->aabb.maxX : emptyAABB.maxX,
        };
        currentNode->maxP.x = Load(maxX);

        f32 maxY[4] = {
            grandChildren[0] ? grandChildren[0]->aabb.maxY : emptyAABB.maxY,
            grandChildren[1] ? grandChildren[1]->aabb.maxY : emptyAABB.maxY,
            grandChildren[2] ? grandChildren[2]->aabb.maxY : emptyAABB.maxY,
            grandChildren[3] ? grandChildren[3]->aabb.maxY : emptyAABB.maxY,
        };
        currentNode->maxP.y = Load(maxY);

        f32 maxZ[4] = {
            grandChildren[0] ? grandChildren[0]->aabb.maxZ : emptyAABB.maxZ,
            grandChildren[1] ? grandChildren[1]->aabb.maxZ : emptyAABB.maxZ,
            grandChildren[2] ? grandChildren[2]->aabb.maxZ : emptyAABB.maxZ,
            grandChildren[3] ? grandChildren[3]->aabb.maxZ : emptyAABB.maxZ,
        };
        currentNode->maxP.z = Load(maxZ);
    }
    return bvh4;
}

//////////////////////////////
// Compressed
//
CompressedBVH4 CreateCompressedBVH4(Arena *arena, BVH *bvh)
{
    // Convert BVH-2 to BVH-4 by skipping every other level
    CompressedBVH4 compressedBVH;
    compressedBVH.scene       = bvh->scene;
    compressedBVH.nodes       = PushArray(arena, CompressedBVHNode, bvh->nodeCount);
    compressedBVH.nodeCount   = bvh->nodeCount;
    compressedBVH.leafIndices = std::move(bvh->leafIndices);

    u32 stack[64];
    stack[0]                         = 0;
    i32 stackCount                   = 1;
    u32 runningCompressedNodeCount   = 1;
    bvh->nodes[0].compressedBVHIndex = 0;

    const AABB emptyAABB;

    if (bvh->nodes[0].IsLeaf())
    {
        Compress(&compressedBVH.nodes[0], bvh->nodes[0].aabb, emptyAABB, emptyAABB, emptyAABB);
        compressedBVH.nodes[0].offsetIndex[0] = bvh->nodes[0].offset;
        compressedBVH.nodes[0].count[0]       = SafeTruncateU32(bvh->nodes[0].count);
        compressedBVH.nodes[0].leafMask       = 2;
        compressedBVH.nodeCount               = runningCompressedNodeCount;
        return compressedBVH;
    }

    while (stackCount > 0)
    {
        // assert(runningCompressedNodeCount < compressedNodeCount);
        u32 top          = stack[--stackCount];
        BVH::Node *node  = &bvh->nodes[top];
        BVH::Node *left  = &bvh->nodes[node->left];
        BVH::Node *right = &bvh->nodes[node->left + 1];

        i32 grandChildrenIndices[] = {
            -1,
            -1,
            -1,
            -1,
        };

        u32 childrenCount = 0;
        if (left->IsLeaf())
        {
            grandChildrenIndices[childrenCount++] = node->left;
        }
        else
        {
            grandChildrenIndices[childrenCount++] = left->left;
            grandChildrenIndices[childrenCount++] = left->left + 1;
        }

        if (right->IsLeaf())
        {
            grandChildrenIndices[childrenCount++] = node->left + 1;
        }
        else
        {
            grandChildrenIndices[childrenCount++] = right->left;
            grandChildrenIndices[childrenCount++] = right->left + 1;
        }

        BVH::Node *grandChildren[4] = {
            grandChildrenIndices[0] != -1 ? &bvh->nodes[grandChildrenIndices[0]] : 0,
            grandChildrenIndices[1] != -1 ? &bvh->nodes[grandChildrenIndices[1]] : 0,
            grandChildrenIndices[2] != -1 ? &bvh->nodes[grandChildrenIndices[2]] : 0,
            grandChildrenIndices[3] != -1 ? &bvh->nodes[grandChildrenIndices[3]] : 0,
        };

        CompressedBVHNode *currentCompressedNode = &compressedBVH.nodes[node->compressedBVHIndex];
        currentCompressedNode->leafMask          = 0;

        for (u32 i = 0; i < ArrayLength(grandChildren); i++)
        {
            BVH::Node *grandChild = grandChildren[i];
            assert(grandChild != &bvh->nodes[0]);
            if (!grandChild) continue;

            b8 isLeaf = grandChild->IsLeaf();
            if (!isLeaf)
            {
                stack[stackCount++]                  = grandChildrenIndices[i];
                currentCompressedNode->childIndex[i] = runningCompressedNodeCount;
                grandChild->compressedBVHIndex       = runningCompressedNodeCount++;
            }
            else
            {
                currentCompressedNode->offsetIndex[i] = grandChild->offset;
                currentCompressedNode->count[i]       = SafeTruncateU32(grandChild->count);
            }
            b8 mask = isLeaf ? 2 : 1;
            currentCompressedNode->leafMask |= (mask << (2 * i));
        }

        Compress(currentCompressedNode,
                 grandChildren[0] ? grandChildren[0]->aabb : emptyAABB,
                 grandChildren[1] ? grandChildren[1]->aabb : emptyAABB,
                 grandChildren[2] ? grandChildren[2]->aabb : emptyAABB,
                 grandChildren[3] ? grandChildren[3]->aabb : emptyAABB);
    }
    compressedBVH.nodeCount = runningCompressedNodeCount;
    return compressedBVH;
}

//////////////////////////////
// Intersection Tests
//
bool BVH::Hit(const Ray &r, const f32 tMin, const f32 tMax, HitRecord &record) const
{
    TIMED_FUNCTION(primitiveIntersectionTime);
    u32 stack[64];
    u32 stackPtr      = 0;
    stack[stackPtr++] = 0;
    f32 closest       = tMax;
    bool hit          = false;
    HitRecord temp;

    int dirIsNeg[3] = {
        r.d.x < 0 ? 1 : 0,
        r.d.y < 0 ? 1 : 0,
        r.d.z < 0 ? 1 : 0,
    };

    while (stackPtr > 0)
    {
        assert(stackPtr < 64);
        const u32 nodeIndex = stack[--stackPtr];
        Node &node          = nodes[nodeIndex];

        bool result = node.aabb.Hit(r, 0, infinity, dirIsNeg);

        if (!result)
            continue;
        if (node.IsLeaf())
        {
            for (u32 i = 0; i < node.count; i++)
            {
                if (scene->Hit(r, tMin, tMax, temp, leafIndices[node.offset + i]))
                {
                    if (temp.t < closest)
                    {
                        closest = temp.t;
                        record  = temp;
                        hit     = true;
                    }
                }
            }
        }
        else
        {
            stack[stackPtr++] = node.left;
            stack[stackPtr++] = node.left + 1;
        }
    }
    return hit;
}

bool BVH4::Hit(const Ray &r, const f32 tMin, const f32 tMax, HitRecord &record) const
{
    TIMED_FUNCTION(primitiveIntersectionTime);

    f32 rdx = (r.d.x == -0.f) ? 0.f : r.d.x;
    f32 rdy = (r.d.y == -0.f) ? 0.f : r.d.y;
    f32 rdz = (r.d.z == -0.f) ? 0.f : r.d.z;

    vec3 oneOverDir    = vec3(1.f / rdx, 1.f / rdy, 1.f / rdz); // r.d.x, 1.f / r.d.y, 1.f / r.d.z);
    LaneVec3 rcpDir    = LaneV3FromV3(oneOverDir);
    LaneVec3 rayOrigin = LaneV3FromV3(r.o);

    u32 stack[64];
    u32 stackPtr      = 0;
    stack[stackPtr++] = 0;
    f32 closest       = tMax;
    bool hit          = false;
    HitRecord temp;

    while (stackPtr > 0)
    {
        assert(stackPtr < 64);
        const u32 nodeIndex       = stack[--stackPtr];
        UncompressedBVHNode &node = nodes[nodeIndex];

        i32 result = node.IntersectP(rayOrigin, rcpDir, 0, infinity);

        if (!result)
            continue;

        for (u32 childIndex = 0; childIndex < 4; childIndex++)
        {
            u32 mask = (node.leafMask >> (2 * childIndex)) & 3;
            if ((!(result & (1 << childIndex))) || mask == 0) continue;
            if (mask == 2)
            {
                for (u32 i = 0; i < node.count[childIndex]; i++)
                {
                    bool primitiveHit = scene->Hit(r, tMin, tMax, temp, leafIndices[node.offsetIndex[childIndex] + i]);

                    if (primitiveHit)
                    {
                        if (temp.t < closest)
                        {
                            closest = temp.t;
                            record  = temp;
                            hit     = true;
                        }
                    }
                }
            }
            else
            {
                stack[stackPtr++] = node.childIndex[childIndex];
            }
        }
    }
    return hit;
}

// https://research.nvidia.com/sites/default/files/publications/ylitie2017hpg-paper.pdf
bool CompressedBVH4::Hit(const Ray &r, const f32 tMin, const f32 tMax, HitRecord &record) const
{
    TIMED_FUNCTION(primitiveIntersectionTime);
    u32 stack[64];

    HitRecord temp;
    const u32 queueLength = 64;
    const u32 queueMask   = queueLength - 1;

    // for leaf primitives
    struct QueueEntry
    {
        u32 offset;
        u32 count;
    };
    QueueEntry queue[queueLength];

    u32 queueWritePos = 0;
    u32 queueReadPos  = 0;

    // traversal stack
    u32 stackPtr      = 0;
    stack[stackPtr++] = 0;

    f32 tClosest  = tMax;
    record.t      = tMax;

    temp.t = tMax;

    bool hit  = false;

    f32 rdx = (r.d.x == -0.f) ? 0.f : r.d.x;
    f32 rdy = (r.d.y == -0.f) ? 0.f : r.d.y;
    f32 rdz = (r.d.z == -0.f) ? 0.f : r.d.z;

    vec3 oneOverDir    = vec3(1.f / rdx, 1.f / rdy, 1.f / rdz);
    LaneVec3 rcpDir    = LaneV3FromV3(oneOverDir);
    LaneVec3 rayOrigin = LaneV3FromV3(r.o);

    const LaneU32 simdZeroI = LaneU32Zero();
    const LaneF32 simdZero  = LaneF32Zero();

    LaneF32 simdInfinity;
    simdInfinity.v       = SIMDInfinity;
    LaneF32 tLaneClosest = LaneF32FromF32(tClosest);

    while (stackPtr > 0)
    {
        assert(stackPtr < 64);
        const u32 nodeIndex     = stack[--stackPtr];
        CompressedBVHNode &node = nodes[nodeIndex];

        const f32 expX = BitsToFloat(node.scaleX << 23);
        const f32 expY = BitsToFloat(node.scaleY << 23);
        const f32 expZ = BitsToFloat(node.scaleZ << 23);

        const LaneVec3 minP = LaneV3FromV3(node.minP);
        const LaneVec3 exp  = LaneV3FromV3(expX, expY, expZ);

        // TODO: low efficiency?
        const LaneU32 minXminYminZmaxX = Load(node.minX, node.minY, node.minZ, node.maxX);
        const LaneU32 maxYmaxZ         = Load(node.maxY, 0, node.maxZ, 0);

        const LaneU32 minXminY = UnpackLowU32(minXminYminZmaxX, simdZeroI);
        const LaneU32 minZmaxX = UnpackHiU32(minXminYminZmaxX, simdZeroI);

        const LaneU32 minXCompressed_i = UnpackLowU32(minXminY, simdZeroI);
        const LaneF32 minXCompressed   = ConvertLaneU32ToLaneF32(SignExtendU8ToU32(minXCompressed_i));
        const LaneU32 minYCompressed_i = UnpackHiU32(minXminY, simdZeroI);
        const LaneF32 minYCompressed   = ConvertLaneU32ToLaneF32(SignExtendU8ToU32(minYCompressed_i));

        const LaneU32 minZCompressed_i = UnpackLowU32(minZmaxX, simdZeroI);
        const LaneF32 minZCompressed   = ConvertLaneU32ToLaneF32(SignExtendU8ToU32(minZCompressed_i));
        const LaneU32 maxXCompressed_i = UnpackHiU32(minZmaxX, simdZeroI);
        const LaneF32 maxXCompressed   = ConvertLaneU32ToLaneF32(SignExtendU8ToU32(maxXCompressed_i));

        const LaneU32 maxYCompressed_i = UnpackLowU32(maxYmaxZ, simdZeroI);
        const LaneF32 maxYCompressed   = ConvertLaneU32ToLaneF32(SignExtendU8ToU32(maxYCompressed_i));
        const LaneU32 maxZCompressed_i = UnpackHiU32(maxYmaxZ, simdZeroI);
        const LaneF32 maxZCompressed   = ConvertLaneU32ToLaneF32(SignExtendU8ToU32(maxZCompressed_i));

        LaneVec3 minCompressed;
        minCompressed.x = minXCompressed;
        minCompressed.y = minYCompressed;
        minCompressed.z = minZCompressed;

        LaneVec3 maxCompressed;
        maxCompressed.x = maxXCompressed;
        maxCompressed.y = maxYCompressed;
        maxCompressed.z = maxZCompressed;

        const LaneVec3 rayDPrime = exp * rcpDir;

        // TODO: see whether operator overloading abstraction has a cost
        const LaneVec3 rayOPrime = (minP - rayOrigin) * rcpDir;

        const LaneVec3 termMin = FMA(minCompressed, rayDPrime, rayOPrime);
        const LaneVec3 termMax = FMA(maxCompressed, rayDPrime, rayOPrime);

        const LaneF32 tEntryX = Min(termMax.x, termMin.x);
        const LaneF32 tLeaveX = Max(termMin.x, termMax.x);

        const LaneF32 tEntryY = Min(termMax.y, termMin.y);
        const LaneF32 tLeaveY = Max(termMin.y, termMax.y);

        const LaneF32 tEntryZ = Min(termMax.z, termMin.z);
        const LaneF32 tLeaveZ = Max(termMin.z, termMax.z);

        const LaneF32 tEntry    = Max(tEntryZ, Max(tEntryY, Max(tEntryX, simdZero)));
        const LaneF32 tLeaveRaw = Min(tLeaveZ, Min(tLeaveY, Min(tLeaveX, simdInfinity)));

        const LaneF32 tLeave = Min(tLeaveRaw, tLaneClosest);

        const LaneF32 intersectMask = tEntry <= tLeave;
        const i32 intersectFlags    = FlattenMask(intersectMask);

        LaneF32 t_dcba = Blend(simdInfinity, tLeaveRaw, intersectMask);

        const u32 childType0 = node.leafMask & 3;
        const u32 childType1 = (node.leafMask >> 2) & 3;
        const u32 childType2 = (node.leafMask >> 4) & 3;
        const u32 childType3 = (node.leafMask >> 6) & 3;

        // If child is a leaf or invalid (case 0 and 2), the upper bits are set according to the mask
        const u32 nodeBits0 = (0xffffffffu + !(childType0 & 1)) & 0x02a10;
        const u32 nodeBits1 = (0xffffffffu + !(childType1 & 1)) & 0x21420;
        const u32 nodeBits2 = (0xffffffffu + !(childType2 & 1)) & 0x48140;
        const u32 nodeBits3 = (0xffffffffu + !(childType3 & 1)) & 0x94080;

        const u32 leafBits0 = (1 - childType0) & 0x02a10;
        const u32 leafBits1 = (1 - childType1) & 0x21420;
        const u32 leafBits2 = (1 - childType2) & 0x48140;
        const u32 leafBits3 = (1 - childType3) & 0x94080;

        const u32 nodeBits = (nodeBits0 | nodeBits1 | nodeBits2 | nodeBits3) >> 4;
        const u32 leafBits = (leafBits0 | leafBits1 | leafBits2 | leafBits3) >> 4;

        // Third bit of nodeBits_n is either 1, 2, 4, 8, so can use that to determine wheter a node is invalid
        const u32 isKeptNode = (nodeBits & intersectFlags & 0xf);
        const u32 isKeptLeaf = (leafBits & intersectFlags & 0xf);

        const u32 numNodes  = PopCount(isKeptNode);
        const u32 numLeaves = PopCount(isKeptLeaf);

        // TODO: see if this branch is worth it
        if ((numNodes | numLeaves) <= 1)
        {
            // If numNodes <= 1, then numNode will be 0, 1, 2, 4, or 8. x/2 - x/8 maps to
            // 0, 0, 1, 2, 3
            stack[stackPtr] = node.childIndex[(isKeptNode >> 1) - (isKeptNode >> 3)];
            stackPtr += numNodes;

            const u32 leafIndex              = (isKeptLeaf >> 1) - (isKeptLeaf >> 3);
            queue[queueWritePos & queueMask] = {node.offsetIndex[leafIndex], node.count[leafIndex]};
            queueWritePos += numLeaves;
        }
        else
        {
            // Branchless adding leaf nodes
            const LaneF32 abac = PermuteReverseF32(t_dcba, 0, 1, 0, 2);
            const LaneF32 adcd = PermuteReverseF32(t_dcba, 0, 3, 2, 3);

            const u32 da_cb_ba_ac = FlattenMask(t_dcba < abac) & 0xe;
            const u32 aa_db_ca_dc = FlattenMask(adcd < abac);

            u32 da_cb_ba_db_ca_dc        = da_cb_ba_ac * 4 + aa_db_ca_dc;
            u32 da_cb_ba_db_ca_dc_nodes  = (da_cb_ba_db_ca_dc | (~nodeBits >> 4)) & (nodeBits >> 10);
            u32 da_cb_ba_db_ca_dc_leaves = (da_cb_ba_db_ca_dc | (~leafBits >> 4)) & (leafBits >> 10);

            u32 indexA = PopCount(da_cb_ba_db_ca_dc_nodes & 0x2a);
            u32 indexB = PopCount((da_cb_ba_db_ca_dc_nodes ^ 0x08) & 0x1c);
            u32 indexC = PopCount((da_cb_ba_db_ca_dc_nodes ^ 0x12) & 0x13);
            u32 indexD = PopCount((~da_cb_ba_db_ca_dc_nodes) & 0x25);

            stack[stackPtr + ((numNodes - 1 - indexA) & 3)] = node.childIndex[0];
            stack[stackPtr + ((numNodes - 1 - indexB) & 3)] = node.childIndex[1];
            stack[stackPtr + ((numNodes - 1 - indexC) & 3)] = node.childIndex[2];
            stack[stackPtr + ((numNodes - 1 - indexD) & 3)] = node.childIndex[3];

            stackPtr += numNodes;

            u32 childIndexA = PopCount(da_cb_ba_db_ca_dc_leaves & 0x2a);
            u32 childIndexB = PopCount((da_cb_ba_db_ca_dc_leaves ^ 0x08) & 0x1c);
            u32 childIndexC = PopCount((da_cb_ba_db_ca_dc_leaves ^ 0x12) & 0x13);
            u32 childIndexD = PopCount((~da_cb_ba_db_ca_dc_leaves) & 0x25);

            const u32 queueIndexA = (queueWritePos + childIndexA) & queueMask;
            const u32 queueIndexB = (queueWritePos + childIndexB) & queueMask;
            const u32 queueIndexC = (queueWritePos + childIndexC) & queueMask;
            const u32 queueIndexD = (queueWritePos + childIndexD) & queueMask;

            queue[queueIndexA].offset = node.offsetIndex[0];
            queue[queueIndexB].offset = node.offsetIndex[1];
            queue[queueIndexC].offset = node.offsetIndex[2];
            queue[queueIndexD].offset = node.offsetIndex[3];

            queue[queueIndexA].count = node.count[0];
            queue[queueIndexB].count = node.count[1];
            queue[queueIndexC].count = node.count[2];
            queue[queueIndexD].count = node.count[3];

            queueWritePos += numLeaves;
        }
        for (u32 queueIndex = queueReadPos; queueIndex < queueWritePos; queueIndex++)
        {
            QueueEntry *entry = &queue[queueIndex & queueMask];
            for (u32 i = 0; i < entry->count; i++)
            {
                bool primitiveHit = scene->Hit(r, tMin, tClosest, record, leafIndices[entry->offset + i]);
                hit |= primitiveHit;
                tClosest = record.t;
            }
        }
        queueReadPos = queueWritePos;
        tLaneClosest = LaneF32FromF32(tClosest);
    }
    return hit;
}

// bool CompressedBVH4::Hit(const Ray &r, const f32 tMin, const f32 tMax, HitRecord &record) const
// {
//     TIMED_FUNCTION(primitiveIntersectionTime);
//
//     f32 rdx = (r.d.x == -0.f) ? 0.f : r.d.x;
//     f32 rdy = (r.d.y == -0.f) ? 0.f : r.d.y;
//     f32 rdz = (r.d.z == -0.f) ? 0.f : r.d.z;
//
//     vec3 oneOverDir    = vec3(1.f / rdx, 1.f / rdy, 1.f / rdz); // r.d.x, 1.f / r.d.y, 1.f / r.d.z);
//     LaneVec3 rcpDir    = LaneV3FromV3(oneOverDir);
//     LaneVec3 rayOrigin = LaneV3FromV3(r.o);
//
//     u32 stack[64];
//     u32 stackPtr      = 0;
//     stack[stackPtr++] = 0;
//     f32 closest       = tMax;
//     bool hit          = false;
//     HitRecord temp;
//
//     while (stackPtr > 0)
//     {
//         assert(stackPtr < 64);
//         const u32 nodeIndex     = stack[--stackPtr];
//         CompressedBVHNode &node = nodes[nodeIndex];
//
//         UncompressedBVHNode uncompressedNode = node.Decompress();
//         // i32 result                           = uncompressedNode.IntersectP(r, 0, infinity);
//         i32 result = uncompressedNode.IntersectP(rayOrigin, rcpDir, 0, infinity);
//         if (!result)
//             continue;
//
//         for (u32 childIndex = 0; childIndex < 4; childIndex++)
//         {
//             u32 mask = (uncompressedNode.leafMask >> (2 * childIndex)) & 3;
//             if ((!(result & (1 << childIndex))) || mask == 0) continue;
//             if (mask == 2)
//             {
//                 for (u32 i = 0; i < uncompressedNode.count[childIndex]; i++)
//                 {
//                     bool primitiveHit = scene->Hit(r, tMin, tMax, temp, leafIndices[uncompressedNode.offsetIndex[childIndex] + i]);
//
//                     if (primitiveHit)
//                     {
//                         if (temp.t < closest)
//                         {
//                             closest = temp.t;
//                             record  = temp;
//                             hit     = true;
//                         }
//                     }
//                 }
//             }
//             else
//             {
//                 stack[stackPtr++] = uncompressedNode.childIndex[childIndex];
//             }
//         }
//     }
//     return hit;
// }

inline bool BVHHit(void *ptr, const Ray &r, const f32 tMin, const f32 tMax, HitRecord &record)
{
    BVH *bvh = (BVH *)ptr;
    return bvh->Hit(r, tMin, tMax, record);
}

inline bool BVH4Hit(void *ptr, const Ray &r, const f32 tMin, const f32 tMax, HitRecord &record)
{
    BVH4 *bvh = (BVH4 *)ptr;
    return bvh->Hit(r, tMin, tMax, record);
}
inline bool CompressedBVH4Hit(void *ptr, const Ray &r, const f32 tMin, const f32 tMax, HitRecord &record)
{
    CompressedBVH4 *bvh = (CompressedBVH4 *)ptr;
    return bvh->Hit(r, tMin, tMax, record);
}
