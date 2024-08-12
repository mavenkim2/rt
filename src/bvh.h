#ifndef BVH_H
#define BVH_H

#include "math.h"

struct BVH
{
    struct Scene *scene;
    struct Node
    {
        AABB aabb;
        u32 left;
        u32 offset;
        u32 count;

        u32 compressedBVHIndex;
        bool IsLeaf() { return count > 0; }
    };
    Node *nodes;
    u32 *leafIndices;
    u32 nodeCount;
    u32 maxPrimitivesPerLeaf;

    void Build(Scene *inScene, u32 primsPerLeaf);
    void Build(AABB *aabbs, u32 count);
    void Subdivide(u32 nodeIndex, AABB *aabbs);
    inline void UpdateNodeBounds(u32 nodeIndex, AABB *aabbs);
    bool Hit(const Ray &r, const f32 tMin, const f32 tMax, HitRecord &record) const;
};

struct UncompressedBVHNode
{
    LaneVec3 minP;
    LaneVec3 maxP;

    union
    {
        // Internal nodes
        u32 child[4];
        // Offset into primitive array
        u32 offset[4];
    };
    // Number of primitives
    u16 count[4];
    u32 leafMask : 4;
    u32 numChildren : 4;

    i32 IntersectP(const Ray &r, const f32 tMinHit, const f32 tMaxHit) const
    {
        LaneVec3 rayOrigin = LaneV3FromV3(r.o);

        LaneF32 result = CastLaneF32FromLaneU32(LaneU32FromU32(0xffffffff));

        vec3 oneOverDir              = vec3(1.f / r.d.x, 1.f / r.d.y, 1.f / r.d.z);
        LaneVec3 oneOverRayDirection = LaneV3FromV3(oneOverDir);

        // Slab test
        // p = o + d * t
        // t = (p - o) p -
        LaneF32 tNearAxis[3] = {
            (minP.x - rayOrigin.x) * oneOverRayDirection.x,
            (minP.y - rayOrigin.y) * oneOverRayDirection.y,
            (minP.z - rayOrigin.z) * oneOverRayDirection.z,
        };
        LaneF32 tFarAxis[3] = {
            (maxP.x - rayOrigin.x) * oneOverRayDirection.x,
            (maxP.y - rayOrigin.y) * oneOverRayDirection.y,
            (maxP.z - rayOrigin.z) * oneOverRayDirection.z,
        };

        LaneF32 testMinT = LaneF32FromF32(tMinHit);
        LaneF32 testMaxT = LaneF32FromF32(tMaxHit);
        for (u32 i = 0; i < ArrayLength(tNearAxis); i++)
        {
            LaneF32 swapMask = tNearAxis[i] > tFarAxis[i];

            LaneF32 tMin = AndNot(swapMask, tNearAxis[i]) | (swapMask & tFarAxis[i]);
            LaneF32 tMax = AndNot(swapMask, tFarAxis[i]) | (swapMask & tNearAxis[i]);

            // TODO: error correction? (see chapter 6.8 PBRT)
            // tMax *= 1 + 2 * gamma(3)

            LaneF32 minMask = tMin > testMinT;
            LaneF32 maxMask = tMax < testMaxT;

            ConditionalAssign(testMinT, tMin, minMask);
            ConditionalAssign(testMaxT, tMax, maxMask);

            LaneF32 intersectionTest = testMinT > testMaxT;
            result                   = AndNot(intersectionTest, result);

            if (MaskIsZeroed(result))
                return 0;
        }

        i32 outcome = FlattenMask(result);
        return outcome;
    }

    i32 IntersectP(const Ray &r, const f32 tMinHit, const f32 tMaxHit, const int dirIsNeg[3]) const
    {
        LaneVec3 rayOrigin = LaneV3FromV3(r.o);

        LaneF32 result = CastLaneF32FromLaneU32(LaneU32FromU32(0xffffffff));

        vec3 oneOverDir              = vec3(1.f / r.d.x, 1.f / r.d.y, 1.f / r.d.z);
        LaneVec3 oneOverRayDirection = LaneV3FromV3(oneOverDir);

        LaneF32 xMask = CastLaneF32FromLaneU32(LaneU32FromU32(dirIsNeg[0] ? 0xffffffff : 0));
        LaneF32 yMask = CastLaneF32FromLaneU32(LaneU32FromU32(dirIsNeg[1] ? 0xffffffff : 0));
        LaneF32 zMask = CastLaneF32FromLaneU32(LaneU32FromU32(dirIsNeg[2] ? 0xffffffff : 0));

        // Slab test
        // p = o + d * t
        // t = (p - o) p -

        LaneVec3 min = minP;
        LaneVec3 max = maxP;

        ConditionalAssign(min.x, maxP.x, xMask);
        ConditionalAssign(min.y, maxP.y, yMask);
        ConditionalAssign(min.z, maxP.z, zMask);

        ConditionalAssign(max.x, minP.x, xMask);
        ConditionalAssign(max.y, minP.y, yMask);
        ConditionalAssign(max.z, minP.z, zMask);

        LaneF32 tNearAxis[3] = {
            (min.x - rayOrigin.x) * oneOverRayDirection.x,
            (min.y - rayOrigin.y) * oneOverRayDirection.y,
            (min.z - rayOrigin.z) * oneOverRayDirection.z,
        };
        LaneF32 tFarAxis[3] = {
            (max.x - rayOrigin.x) * oneOverRayDirection.x,
            (max.y - rayOrigin.y) * oneOverRayDirection.y,
            (max.z - rayOrigin.z) * oneOverRayDirection.z,
        };

        LaneF32 testMinT = LaneF32FromF32(tMinHit);
        LaneF32 testMaxT = LaneF32FromF32(tMaxHit);
        for (u32 i = 0; i < ArrayLength(tNearAxis); i++)
        {
            // LaneF32 swapMask = tNearAxis[i] > tFarAxis[i];

            // LaneF32 tMin = AndNot(swapMask, tNearAxis[i]) | (swapMask & tFarAxis[i]);
            // LaneF32 tMax = AndNot(swapMask, tFarAxis[i]) | (swapMask & tNearAxis[i]);

            // TODO: error correction? (see chapter 6.8 PBRT)
            // tMax *= 1 + 2 * gamma(3)

            LaneF32 minMask = tNearAxis[i] > testMinT;
            LaneF32 maxMask = tFarAxis[i] < testMaxT;

            ConditionalAssign(testMinT, tNearAxis[i], minMask);
            ConditionalAssign(testMaxT, tFarAxis[i], maxMask);

            LaneF32 intersectionTest = testMinT > testMaxT;
            result                   = AndNot(intersectionTest, result);

            if (MaskIsZeroed(result))
                return 0;
        }

        i32 outcome = FlattenMask(result);
        return outcome;
    }
    inline b8 IsLeaf(i32 index)
    {
        b8 result = leafMask & (1 << index);
        return result;
    }
};

struct alignas(32) CompressedBVHNode
{
    vec3 minP;
    struct
    {
        u32 scaleX : 8;
        u32 scaleY : 8;
        u32 scaleZ : 8;
        // Set bit = Leaf
        u32 leafMask : 4;
        // Set bit = Valid
        u32 numChildren : 4;
    };
    u8 minX[4];
    u8 minY[4];
    u8 minZ[4];

    u8 maxX[4];
    u8 maxY[4];
    u8 maxZ[4];

    union
    {
        // Internal nodes
        u32 child[4];
        // Offset into primitive array
        u32 offset[4];
    };
    // Number of primitives
    u16 count[4];

    UncompressedBVHNode Decompress()
    {
        UncompressedBVHNode node;

        f32 expX = BitsToFloat(scaleX << 23);
        f32 expY = BitsToFloat(scaleY << 23);
        f32 expZ = BitsToFloat(scaleZ << 23);

        f32 uncompressedMinX[4];
        uncompressedMinX[0] = minP.x + expX * minX[0];
        uncompressedMinX[1] = minP.x + expX * minX[1];
        uncompressedMinX[2] = minP.x + expX * minX[2];
        uncompressedMinX[3] = minP.x + expX * minX[3];

        f32 uncompressedMinY[4];
        uncompressedMinY[0] = minP.y + expY * minY[0];
        uncompressedMinY[1] = minP.y + expY * minY[1];
        uncompressedMinY[2] = minP.y + expY * minY[2];
        uncompressedMinY[3] = minP.y + expY * minY[3];

        f32 uncompressedMinZ[4];
        uncompressedMinZ[0] = minP.z + expZ * minZ[0];
        uncompressedMinZ[1] = minP.z + expZ * minZ[1];
        uncompressedMinZ[2] = minP.z + expZ * minZ[2];
        uncompressedMinZ[3] = minP.z + expZ * minZ[3];

        f32 uncompressedMaxX[4];
        uncompressedMaxX[0] = minP.x + expX * maxX[0];
        uncompressedMaxX[1] = minP.x + expX * maxX[1];
        uncompressedMaxX[2] = minP.x + expX * maxX[2];
        uncompressedMaxX[3] = minP.x + expX * maxX[3];

        f32 uncompressedMaxY[4];
        uncompressedMaxY[0] = minP.y + expY * maxY[0];
        uncompressedMaxY[1] = minP.y + expY * maxY[1];
        uncompressedMaxY[2] = minP.y + expY * maxY[2];
        uncompressedMaxY[3] = minP.y + expY * maxY[3];

        f32 uncompressedMaxZ[4];
        uncompressedMaxZ[0] = minP.z + expZ * maxZ[0];
        uncompressedMaxZ[1] = minP.z + expZ * maxZ[1];
        uncompressedMaxZ[2] = minP.z + expZ * maxZ[2];
        uncompressedMaxZ[3] = minP.z + expZ * maxZ[3];

        node.minP.x = Load(uncompressedMinX);
        node.minP.y = Load(uncompressedMinY);
        node.minP.z = Load(uncompressedMinZ);

        node.maxP.x = Load(uncompressedMaxX);
        node.maxP.y = Load(uncompressedMaxY);
        node.maxP.z = Load(uncompressedMaxZ);

        return node;
    }

    inline b8 IsLeaf(i32 index)
    {
        b8 result = leafMask & (1 << index);
        return result;
    }
};

void Compress(CompressedBVHNode *node, const AABB &child1, const AABB &child2, const AABB &child3, const AABB &child4)
{
    node->minP = Min(Min(child1, child2), Min(child3, child4));

    vec3 maxP = Max(Max(child1, child2), Max(child3, child4));
    f32 expX  = std::ceil(log2f((maxP.x - node->minP.x) / 255.f));
    f32 expY  = std::ceil(log2f((maxP.y - node->minP.y) / 255.f));
    f32 expZ  = std::ceil(log2f((maxP.z - node->minP.z) / 255.f));

    f32 powX = powf(2.f, expX);
    f32 powY = powf(2.f, expY);
    f32 powZ = powf(2.f, expZ);

    node->minX[0] = u8((child1.minX - node->minP.x) / powX);
    node->minX[1] = u8((child2.minX - node->minP.x) / powX);
    node->minX[2] = u8((child3.minX - node->minP.x) / powX);
    node->minX[3] = u8((child4.minX - node->minP.x) / powX);

    node->minY[0] = u8((child1.minY - node->minP.y) / powY);
    node->minY[1] = u8((child2.minY - node->minP.y) / powY);
    node->minY[2] = u8((child3.minY - node->minP.y) / powY);
    node->minY[3] = u8((child4.minY - node->minP.y) / powY);

    node->minZ[0] = u8((child1.minZ - node->minP.z) / powZ);
    node->minZ[1] = u8((child2.minZ - node->minP.z) / powZ);
    node->minZ[2] = u8((child3.minZ - node->minP.z) / powZ);
    node->minZ[3] = u8((child4.minZ - node->minP.z) / powZ);

    node->maxX[0] = u8((child1.maxX - node->minP.x) / powX);
    node->maxX[1] = u8((child2.maxX - node->minP.x) / powX);
    node->maxX[2] = u8((child3.maxX - node->minP.x) / powX);
    node->maxX[3] = u8((child4.maxX - node->minP.x) / powX);

    node->maxY[0] = u8((child1.maxY - node->minP.y) / powY);
    node->maxY[1] = u8((child2.maxY - node->minP.y) / powY);
    node->maxY[2] = u8((child3.maxY - node->minP.y) / powY);
    node->maxY[3] = u8((child4.maxY - node->minP.y) / powY);

    node->maxZ[0] = u8((child1.maxZ - node->minP.z) / powZ);
    node->maxZ[1] = u8((child2.maxZ - node->minP.z) / powZ);
    node->maxZ[2] = u8((child3.maxZ - node->minP.z) / powZ);
    node->maxZ[3] = u8((child4.maxZ - node->minP.z) / powZ);

    node->scaleX = u8(FloatToBits(powX) >> 23);
    node->scaleY = u8(FloatToBits(powY) >> 23);
    node->scaleZ = u8(FloatToBits(powZ) >> 23);
}

struct BVH4
{
    struct Scene *scene;
    UncompressedBVHNode *nodes;
    u32 *leafIndices;
    u32 nodeCount;
    bool Hit(const Ray &r, const f32 tMin, const f32 tMax, HitRecord &record) const;
};

struct CompressedBVH4
{
    struct Scene *scene;
    CompressedBVHNode *nodes;
    u32 *leafIndices;
    u32 nodeCount;
    bool Hit(const Ray &r, const f32 tMin, const f32 tMax, HitRecord &record) const;
};

inline bool BVHHit(void *ptr, const Ray &r, const f32 tMin, const f32 tMax, HitRecord &record);
inline bool BVH4Hit(void *ptr, const Ray &r, const f32 tMin, const f32 tMax, HitRecord &record);
inline bool CompressedBVH4Hit(void *ptr, const Ray &r, const f32 tMin, const f32 tMax, HitRecord &record);

#endif
