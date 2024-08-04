#include "bvh.h"
void BVH::Build(Scene *inScene, u32 primsPerLeaf)
{
    maxPrimitivesPerLeaf    = primsPerLeaf;
    scene                   = inScene;
    u32 totalPrimitiveCount = scene->totalPrimitiveCount;
    AABB *aabbs             = (AABB *)malloc(totalPrimitiveCount * sizeof(AABB));
    scene->GetAABBs(aabbs);
    Build(aabbs, totalPrimitiveCount);
}

void BVH::Build(AABB *aabbs, u32 count)
{
    nodeCount = 0;
    assert(count != 0);
    const u32 nodeCapacity = count * 2 - 1;
    nodes                  = (Node *)malloc(sizeof(Node) * nodeCapacity);
    leafIndices            = (u32 *)malloc(sizeof(u32) * count);

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

    for (u32 i = node.offset; i < node.count; i++)
    {
        i32 b = i32(numBuckets * node.aabb.Offset(aabbs[i].Centroid())[axis]);
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

    node.count = 0;
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

bool BVH::Hit(const Ray &r, const f32 tMin, const f32 tMax, HitRecord &record) const
{
    u32 stack[64];
    u32 stackPtr      = 0;
    stack[stackPtr++] = 0;
    f32 closest       = tMax;
    bool hit          = false;
    HitRecord temp;

    while (stackPtr > 0)
    {
        assert(stackPtr < 64);
        const u32 nodeIndex = stack[--stackPtr];
        Node &node          = nodes[nodeIndex];
        if (!node.aabb.Hit(r, 0, infinity))
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

//////////////////////////////
// Compressed
//
CompressedBVH CompressBVH(BVH *bvh)
{
    // Convert BVH-2 to BVH-4 by skipping every other level
    CompressedBVH compressedBVH;
    compressedBVH.scene       = bvh->scene;
    compressedBVH.nodes       = (CompressedBVHNode *)malloc(sizeof(CompressedBVHNode) * (bvh->nodeCount + 3) / 4);
    compressedBVH.nodeCount   = bvh->nodeCount;
    compressedBVH.leafIndices = std::move(bvh->leafIndices);

    u32 stack[64];
    stack[0]                         = 0;
    i32 stackCount                   = 1;
    u32 runningCompressedNodeCount   = 1;
    bvh->nodes[0].compressedBVHIndex = 0;

    const AABB emptyAABB;

    while (stackCount > 0)
    {
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
        currentCompressedNode->numChildren       = childrenCount;

        for (u32 i = 0; i < ArrayLength(grandChildren); i++)
        {
            BVH::Node *grandChild = grandChildren[i];
            assert(grandChild != &bvh->nodes[0]);
            if (!grandChild) continue;

            b8 isLeaf = grandChild->IsLeaf();
            if (!isLeaf)
            {
                stack[stackCount++] = grandChildrenIndices[i];
                currentCompressedNode->child[i] = runningCompressedNodeCount;
                grandChild->compressedBVHIndex  = runningCompressedNodeCount++;
            }
            else
            {
                currentCompressedNode->offset[i] = grandChild->offset;
                currentCompressedNode->count[i]  = SafeTruncateU32(grandChild->count);
            }
            currentCompressedNode->leafMask |= (isLeaf << i);
        }

        Compress(currentCompressedNode,
                 grandChildren[0] ? grandChildren[0]->aabb : emptyAABB,
                 grandChildren[1] ? grandChildren[1]->aabb : emptyAABB,
                 grandChildren[2] ? grandChildren[2]->aabb : emptyAABB,
                 grandChildren[3] ? grandChildren[3]->aabb : emptyAABB);
    }
    return compressedBVH;
}

bool CompressedBVH::Hit(const Ray &r, const f32 tMin, const f32 tMax, HitRecord &record) const
{
    u32 stack[64];
    u32 stackPtr      = 0;
    stack[stackPtr++] = 0;
    f32 closest       = tMax;
    bool hit          = false;
    HitRecord temp;

    while (stackPtr > 0)
    {
        assert(stackPtr < 64);
        const u32 nodeIndex     = stack[--stackPtr];
        CompressedBVHNode &node = nodes[nodeIndex];

        UncompressedBVHNode uncompressedNode = node.Decompress();

        i32 result = uncompressedNode.IntersectP(r, 0, infinity);
        if (!result)
            continue;

        for (u32 childIndex = 0; childIndex < node.numChildren; childIndex++)
        {
            if (node.IsLeaf(childIndex))
            {
                for (u32 i = 0; i < node.count[childIndex]; i++)
                {
                    if (scene->Hit(r, tMin, tMax, temp, leafIndices[node.offset[childIndex] + i]))
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
                stack[stackPtr++] = node.child[childIndex];
            }
        }
    }
    return hit;
}
