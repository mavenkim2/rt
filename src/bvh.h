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
        // TODO: this is really wasteful and bad
        b8 isNode;
        bool IsLeaf() { return isNode == 0; }
    };
    Node *nodes;
    u32 *leafIndices;
    u32 nodeCount;
    u32 maxPrimitivesPerLeaf;

    void Build(Arena *arena, Scene *inScene, u32 primsPerLeaf);
    void Build(Arena *arena, AABB *aabbs, u32 count);
    void Subdivide(u32 nodeIndex, AABB *aabbs);
    inline void UpdateNodeBounds(u32 nodeIndex, AABB *aabbs);
    bool Hit(const Ray &r, const f32 tMin, const f32 tMax, HitRecord &record) const;
};

struct alignas(64) UncompressedBVHNode
{
    LaneVec3 minP;
    LaneVec3 maxP;

    union
    {
        // Internal nodes
        u32 childIndex[4];
        // Offset into primitive array
        u32 offsetIndex[4];
    };
    // Number of primitives
    u16 count[4];
    // invalid = 0, child = 1, leaf = 2
    u32 leafMask : 8;

    // https://www.youtube.com/watch?v=6BIfqfC1i7U
    i32 IntersectP(const LaneVec3 rayOrigin, const LaneVec3 rcpDir, const f32 tMinHit, const f32 tMaxHit) const
    {
        const LaneF32 termMinX = (minP.x - rayOrigin.x) * rcpDir.x;
        const LaneF32 termMaxX = (maxP.x - rayOrigin.x) * rcpDir.x;

        __m128 resultTest = _mm_mul_ps(_mm_sub_ps(maxP.x.v, rayOrigin.x.v), rcpDir.x.v);

        const LaneF32 termMinY = (minP.y - rayOrigin.y) * rcpDir.y;
        const LaneF32 termMaxY = (maxP.y - rayOrigin.y) * rcpDir.y;

        const LaneF32 termMinZ = (minP.z - rayOrigin.z) * rcpDir.z;
        const LaneF32 termMaxZ = (maxP.z - rayOrigin.z) * rcpDir.z;

        // NOTE: the order matters here. If one of the two values are NaN, SSE returns the second one.
        // For example, if rayOrigin.x = minP.x, and rcpDir.x = 0, then termMinX = NaN and termMaxX = +infinity
        // (because -0 is converted to +0). Therefore, tLeave won't be automatically set to -infinity, causing
        // the intersection to fail in this case. This prevents cracks between edges.

        const LaneF32 tMinX = Min(termMaxX, termMinX);
        const LaneF32 tMaxX = Max(termMinX, termMaxX);

        const LaneF32 tMinY = Min(termMaxY, termMinY);
        const LaneF32 tMaxY = Max(termMinY, termMaxY);

        const LaneF32 tMinZ = Min(termMaxZ, termMinZ);
        const LaneF32 tMaxZ = Max(termMinZ, termMaxZ);

        const LaneF32 tEntry = Max(tMinX, Max(tMinY, Max(tMinX, LaneF32FromF32(tMinHit))));
        const LaneF32 tLeave = Min(tMaxZ, Min(tMaxY, Min(tMaxX, LaneF32FromF32(tMaxHit))));

        // TODO: maybe this should be <=, in the case that we're intersecting something infinitely thin
        // (i.e. min = max)
        return FlattenMask(tEntry <= tLeave);
    }

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
};

struct alignas(64) CompressedBVHNode
{
    vec3 minP;
    u32 minX;
    u32 minY;
    u32 minZ;
    // u8 minX[4];
    // u8 minY[4];
    // u8 minZ[4];

    struct
    {
        u32 scaleX : 8;
        u32 scaleY : 8;
        u32 scaleZ : 8;
        // Set bit = Leaf
        u32 leafMask : 8;
        // Set bit = Valid
        // u32 numChildren : 4;
    };

    u32 maxX;
    u32 maxY;
    u32 maxZ;
    // u8 maxX[4];
    // u8 maxY[4];
    // u8 maxZ[4];

    union
    {
        // Internal nodes
        u32 childIndex[4];
        // Offset into primitive array
        u32 offsetIndex[4];
    };
    // Number of primitives
    u16 count[4];

    // TODO: actually deal with floating point precision systematically (see 6.8 pbrt)
    UncompressedBVHNode Decompress()
    {
        UncompressedBVHNode node;

        f32 expX = BitsToFloat(scaleX << 23);
        f32 expY = BitsToFloat(scaleY << 23);
        f32 expZ = BitsToFloat(scaleZ << 23);

        f32 uncompressedMinX[4];
        uncompressedMinX[0] = minP.x + expX * (minX & 0xff);         // 0.001f;
        uncompressedMinX[1] = minP.x + expX * ((minX >> 8) & 0xff);  // 0.001f;
        uncompressedMinX[2] = minP.x + expX * ((minX >> 16) & 0xff); // 0.001f;
        uncompressedMinX[3] = minP.x + expX * ((minX >> 24) & 0xff); // 0.001f;

        f32 uncompressedMinY[4];
        uncompressedMinY[0] = minP.y + expY * (minY & 0xff);         // 0.001f;
        uncompressedMinY[1] = minP.y + expY * ((minY >> 8) & 0xff);  // 0.001f;
        uncompressedMinY[2] = minP.y + expY * ((minY >> 16) & 0xff); // 0.001f;
        uncompressedMinY[3] = minP.y + expY * ((minY >> 24) & 0xff); // 0.001f;

        f32 uncompressedMinZ[4];
        uncompressedMinZ[0] = minP.z + expZ * (minZ & 0xff);         // 0.001f;
        uncompressedMinZ[1] = minP.z + expZ * ((minZ >> 8) & 0xff);  // 0.001f;
        uncompressedMinZ[2] = minP.z + expZ * ((minZ >> 16) & 0xff); // 0.001f;
        uncompressedMinZ[3] = minP.z + expZ * ((minZ >> 24) & 0xff); // 0.001f;

        f32 uncompressedMaxX[4];
        uncompressedMaxX[0] = minP.x + expX * (maxX & 0xff);         // 0.001f;
        uncompressedMaxX[1] = minP.x + expX * ((maxX >> 8) & 0xff);  // 0.001f;
        uncompressedMaxX[2] = minP.x + expX * ((maxX >> 16) & 0xff); // 0.001f;
        uncompressedMaxX[3] = minP.x + expX * ((maxX >> 24) & 0xff); // 0.001f;

        f32 uncompressedMaxY[4];
        uncompressedMaxY[0] = minP.y + expY * (maxY & 0xff);         // 0.001f;
        uncompressedMaxY[1] = minP.y + expY * ((maxY >> 8) & 0xff);  // 0.001f;
        uncompressedMaxY[2] = minP.y + expY * ((maxY >> 16) & 0xff); // 0.001f;
        uncompressedMaxY[3] = minP.y + expY * ((maxY >> 24) & 0xff); // 0.001f;

        f32 uncompressedMaxZ[4];
        uncompressedMaxZ[0] = minP.z + expZ * (maxZ & 0xff);         // + 0.001f;
        uncompressedMaxZ[1] = minP.z + expZ * ((maxZ >> 8) & 0xff);  // + 0.001f;
        uncompressedMaxZ[2] = minP.z + expZ * ((maxZ >> 16) & 0xff); // + 0.001f;
        uncompressedMaxZ[3] = minP.z + expZ * ((maxZ >> 24) & 0xff); //+ 0.001f;

        node.minP.x = Load(uncompressedMinX);
        node.minP.y = Load(uncompressedMinY);
        node.minP.z = Load(uncompressedMinZ);

        node.maxP.x = Load(uncompressedMaxX);
        node.maxP.y = Load(uncompressedMaxY);
        node.maxP.z = Load(uncompressedMaxZ);

        node.childIndex[0] = childIndex[0];
        node.childIndex[1] = childIndex[1];
        node.childIndex[2] = childIndex[2];
        node.childIndex[3] = childIndex[3];

        node.count[0] = count[0];
        node.count[1] = count[1];
        node.count[2] = count[2];
        node.count[3] = count[3];

        node.leafMask = leafMask;
        return node;
    }
};

void Compress(CompressedBVHNode *node, const AABB &child0, const AABB &child1, const AABB &child2, const AABB &child3)
{
    node->minP = Min(Min(child0, child1), Min(child2, child3));

    vec3 maxP = Max(Max(child0, child1), Max(child2, child3));
    f32 expX  = std::ceil(log2f((maxP.x - node->minP.x) / 255.f));
    f32 expY  = std::ceil(log2f((maxP.y - node->minP.y) / 255.f));
    f32 expZ  = std::ceil(log2f((maxP.z - node->minP.z) / 255.f));

    f32 powX = powf(2.f, expX);
    f32 powY = powf(2.f, expY);
    f32 powZ = powf(2.f, expZ);

    LaneF32 powXLane = LaneF32FromF32(powX);
    LaneF32 powYLane = LaneF32FromF32(powY);
    LaneF32 powZLane = LaneF32FromF32(powZ);

    LaneF32 minX = Load(child0.minX, child1.minX, child2.minX, child3.minX);
    LaneF32 minY = Load(child0.minY, child1.minY, child2.minY, child3.minY);
    LaneF32 minZ = Load(child0.minZ, child1.minZ, child2.minZ, child3.minZ);

    LaneF32 maxX = Load(child0.maxX, child1.maxX, child2.maxX, child3.maxX);
    LaneF32 maxY = Load(child0.maxY, child1.maxY, child2.maxY, child3.maxY);
    LaneF32 maxZ = Load(child0.maxZ, child1.maxZ, child2.maxZ, child3.maxZ);

    LaneF32 nodeMinX = LaneF32FromF32(node->minP.x);
    LaneF32 nodeMinY = LaneF32FromF32(node->minP.y);
    LaneF32 nodeMinZ = LaneF32FromF32(node->minP.z);

    node->minX = ExtractU32(TruncateU32ToU8(ConvertLaneF32ToLaneU32((minX - nodeMinX) / powXLane)), 0);
    node->minY = ExtractU32(TruncateU32ToU8(ConvertLaneF32ToLaneU32((minY - nodeMinY) / powYLane)), 0);
    node->minZ = ExtractU32(TruncateU32ToU8(ConvertLaneF32ToLaneU32((minZ - nodeMinZ) / powZLane)), 0);

    node->maxX = ExtractU32(TruncateU32ToU8(ConvertLaneF32ToLaneU32((maxX - nodeMinX) / powXLane)), 0);
    node->maxY = ExtractU32(TruncateU32ToU8(ConvertLaneF32ToLaneU32((maxY - nodeMinY) / powYLane)), 0);
    node->maxZ = ExtractU32(TruncateU32ToU8(ConvertLaneF32ToLaneU32((maxZ - nodeMinZ) / powZLane)), 0);

    node->scaleX = u8(FloatToBits(powX) >> 23);
    node->scaleY = u8(FloatToBits(powY) >> 23);
    node->scaleZ = u8(FloatToBits(powZ) >> 23);
}
// {
//     node->minP = Min(Min(child0, child1), Min(child2, child3));
//
//     vec3 maxP = Max(Max(child0, child1), Max(child2, child3));
//     f32 expX  = std::ceil(log2f((maxP.x - node->minP.x) / 255.f));
//     f32 expY  = std::ceil(log2f((maxP.y - node->minP.y) / 255.f));
//     f32 expZ  = std::ceil(log2f((maxP.z - node->minP.z) / 255.f));
//
//     f32 powX = powf(2.f, expX);
//     f32 powY = powf(2.f, expY);
//     f32 powZ = powf(2.f, expZ);
//
//     node->minX = (u32)((child0.minX - node->minP.x) / powX) & 0xff;
//     node->minX |= ((u32)(((child1.minX - node->minP.x) / powX)) & 0xff) << 8;
//     node->minX |= ((u32)(((child2.minX - node->minP.x) / powX)) & 0xff) << 16;
//     node->minX |= ((u32)(((child3.minX - node->minP.x) / powX)) & 0xff) << 24;
//
//     node->minY = (u32)((child0.minY - node->minP.y) / powY) & 0xff;
//     node->minY |= ((u32)(((child1.minY - node->minP.y) / powY)) & 0xff) << 8;
//     node->minY |= ((u32)(((child2.minY - node->minP.y) / powY)) & 0xff) << 16;
//     node->minY |= ((u32)(((child3.minY - node->minP.y) / powY)) & 0xff) << 24;
//
//     node->minZ = (u32)((child0.minZ - node->minP.z) / powZ) & 0xff;
//     node->minZ |= ((u32)(((child1.minZ - node->minP.z) / powZ)) & 0xff) << 8;
//     node->minZ |= ((u32)(((child2.minZ - node->minP.z) / powZ)) & 0xff) << 16;
//     node->minZ |= ((u32)(((child3.minZ - node->minP.z) / powZ)) & 0xff) << 24;
//
//     node->maxX = (u32)((child0.maxX - node->minP.x) / powX) & 0xff;
//     node->maxX |= ((u32)(((child1.maxX - node->minP.x) / powX)) & 0xff) << 8;
//     node->maxX |= ((u32)(((child2.maxX - node->minP.x) / powX)) & 0xff) << 16;
//     node->maxX |= ((u32)(((child3.maxX - node->minP.x) / powX)) & 0xff) << 24;
//
//     node->maxY = (u32)((child0.maxY - node->minP.y) / powY) & 0xff;
//     node->maxY |= ((u32)(((child1.maxY - node->minP.y) / powY)) & 0xff) << 8;
//     node->maxY |= ((u32)(((child2.maxY - node->minP.y) / powY)) & 0xff) << 16;
//     node->maxY |= ((u32)(((child3.maxY - node->minP.y) / powY)) & 0xff) << 24;
//
//     node->maxZ = (u32)((child0.maxZ - node->minP.z) / powZ) & 0xff;
//     node->maxZ |= ((u32)(((child1.maxZ - node->minP.z) / powZ)) & 0xff) << 8;
//     node->maxZ |= ((u32)(((child2.maxZ - node->minP.z) / powZ)) & 0xff) << 16;
//     node->maxZ |= ((u32)(((child3.maxZ - node->minP.z) / powZ)) & 0xff) << 24;
//
//     node->scaleX = u8(FloatToBits(powX) >> 23);
//     node->scaleY = u8(FloatToBits(powY) >> 23);
//     node->scaleZ = u8(FloatToBits(powZ) >> 23);
// }

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
