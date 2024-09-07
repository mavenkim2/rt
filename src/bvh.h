#ifndef BVH_H
#define BVH_H

#include "math.h"

namespace rt
{

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
    Lane4Vec3f minP;
    Lane4Vec3f maxP;

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
    i32 IntersectP(const Lane4Vec3f rayOrigin, const Lane4Vec3f rcpDir, const f32 tMinHit, const f32 tMaxHit) const
    {
        const Lane4F32 termMinX = (minP.x - rayOrigin.x) * rcpDir.x;
        const Lane4F32 termMaxX = (maxP.x - rayOrigin.x) * rcpDir.x;

        const Lane4F32 termMinY = (minP.y - rayOrigin.y) * rcpDir.y;
        const Lane4F32 termMaxY = (maxP.y - rayOrigin.y) * rcpDir.y;

        const Lane4F32 termMinZ = (minP.z - rayOrigin.z) * rcpDir.z;
        const Lane4F32 termMaxZ = (maxP.z - rayOrigin.z) * rcpDir.z;

        // NOTE: the order matters here. If one of the two values are NaN, SSE returns the second one.
        // For example, if rayOrigin.x = minP.x, and rcpDir.x = 0, then termMinX = NaN and termMaxX = +infinity
        // (because -0 is converted to +0). Therefore, tLeave won't be automatically set to -infinity, causing
        // the intersection to fail in this case. This prevents cracks between edges.

        const Lane4F32 tMinX = Min(termMaxX, termMinX);
        const Lane4F32 tMaxX = Max(termMinX, termMaxX);

        const Lane4F32 tMinY = Min(termMaxY, termMinY);
        const Lane4F32 tMaxY = Max(termMinY, termMaxY);

        const Lane4F32 tMinZ = Min(termMaxZ, termMinZ);
        const Lane4F32 tMaxZ = Max(termMinZ, termMaxZ);

        const Lane4F32 tEntry = Max(tMinX, Max(tMinY, Max(tMinX, tMinHit)));
        const Lane4F32 tLeave = Min(tMaxZ, Min(tMaxY, Min(tMaxX, tMaxHit)));

        return Movemask(tEntry <= tLeave);
    }

    i32 IntersectP(const Ray &r, const f32 tMinHit, const f32 tMaxHit) const
    {
        Lane4Vec3f rayOrigin = r.o;

        Lane4F32 result = Lane4F32::Mask(true);

        Vec3f oneOverDir               = Vec3f(1.f / r.d.x, 1.f / r.d.y, 1.f / r.d.z);
        Lane4Vec3f oneOverRayDirection = oneOverDir;

        // Slab test
        // p = o + d * t
        // t = (p - o) p -
        Lane4F32 tNearAxis[3] = {
            (minP.x - rayOrigin.x) * oneOverRayDirection.x,
            (minP.y - rayOrigin.y) * oneOverRayDirection.y,
            (minP.z - rayOrigin.z) * oneOverRayDirection.z,
        };
        Lane4F32 tFarAxis[3] = {
            (maxP.x - rayOrigin.x) * oneOverRayDirection.x,
            (maxP.y - rayOrigin.y) * oneOverRayDirection.y,
            (maxP.z - rayOrigin.z) * oneOverRayDirection.z,
        };

        Lane4F32 testMinT = tMinHit;
        Lane4F32 testMaxT = tMaxHit;
        for (u32 i = 0; i < ArrayLength(tNearAxis); i++)
        {
            Lane4F32 swapMask = tNearAxis[i] > tFarAxis[i];

            Lane4F32 tMin = Select(swapMask, tFarAxis[i], tNearAxis[i]);
            Lane4F32 tMax = Select(swapMask, tNearAxis[i], tFarAxis[i]);

            // TODO: error correction? (see chapter 6.8 PBRT)
            // tMax *= 1 + 2 * gamma(3)

            Lane4F32 minMask = tMin > testMinT;
            Lane4F32 maxMask = tMax < testMaxT;

            testMinT = Select(minMask, tMin, testMinT);
            testMaxT = Select(maxMask, tMax, testMaxT);

            Lane4F32 intersectionTest = testMinT <= testMaxT;
            result &= intersectionTest;

            if (None(result))
                return 0;
        }

        i32 outcome = Movemask(result);
        return outcome;
    }

    i32 IntersectP(const Ray &r, const f32 tMinHit, const f32 tMaxHit, const int dirIsNeg[3]) const
    {
        Lane4Vec3f rayOrigin = r.o;

        Lane4F32 result = Lane4F32::Mask(true);

        Vec3f oneOverDir               = Vec3f(1.f / r.d.x, 1.f / r.d.y, 1.f / r.d.z);
        Lane4Vec3f oneOverRayDirection = oneOverDir;

        Lane4F32 xMask = Lane4F32::Mask(dirIsNeg[0] ? false : true);
        Lane4F32 yMask = Lane4F32::Mask(dirIsNeg[1] ? false : true);
        Lane4F32 zMask = Lane4F32::Mask(dirIsNeg[2] ? false : true);

        // Slab test
        // p = o + d * t
        // t = (p - o) p -

        Lane4Vec3f min;
        Lane4Vec3f max;
        min.x = Select(xMask, minP.x, maxP.x);
        min.y = Select(yMask, minP.y, maxP.y);
        min.z = Select(zMask, minP.z, maxP.z);

        max.x = Select(xMask, maxP.x, minP.x);
        max.y = Select(yMask, maxP.y, minP.y);
        max.z = Select(zMask, maxP.z, minP.z);

        Lane4F32 tNearAxis[3] = {
            (min.x - rayOrigin.x) * oneOverRayDirection.x,
            (min.y - rayOrigin.y) * oneOverRayDirection.y,
            (min.z - rayOrigin.z) * oneOverRayDirection.z,
        };
        Lane4F32 tFarAxis[3] = {
            (max.x - rayOrigin.x) * oneOverRayDirection.x,
            (max.y - rayOrigin.y) * oneOverRayDirection.y,
            (max.z - rayOrigin.z) * oneOverRayDirection.z,
        };

        Lane4F32 testMinT(tMinHit);
        Lane4F32 testMaxT(tMaxHit);
        for (u32 i = 0; i < ArrayLength(tNearAxis); i++)
        {
            // Lane4F32 swapMask = tNearAxis[i] > tFarAxis[i];

            // Lane4F32 tMin = AndNot(swapMask, tNearAxis[i]) | (swapMask & tFarAxis[i]);
            // Lane4F32 tMax = AndNot(swapMask, tFarAxis[i]) | (swapMask & tNearAxis[i]);

            // TODO: error correction? (see chapter 6.8 PBRT)
            // tMax *= 1 + 2 * gamma(3)

            Lane4F32 minMask = tNearAxis[i] > testMinT;
            Lane4F32 maxMask = tFarAxis[i] < testMaxT;

            testMinT = Select(minMask, tNearAxis[i], testMinT);
            testMaxT = Select(maxMask, tFarAxis[i], testMaxT);

            Lane4F32 intersectionTest = testMinT <= testMaxT;
            result &= intersectionTest;

            if (None(result))
                return 0;
        }

        i32 outcome = Movemask(result);
        return outcome;
    }
};

struct alignas(64) CompressedBVHNode
{
    Vec3f minP;
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

        node.minP.x = Lane4F32::Load(uncompressedMinX);
        node.minP.y = Lane4F32::Load(uncompressedMinY);
        node.minP.z = Lane4F32::Load(uncompressedMinZ);

        node.maxP.x = Lane4F32::Load(uncompressedMaxX);
        node.maxP.y = Lane4F32::Load(uncompressedMaxY);
        node.maxP.z = Lane4F32::Load(uncompressedMaxZ);

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
    const f32 MIN_QUAN = 0.f;
    const f32 MAX_QUAN = 255.f;

    node->minP = Min(Min(child0, child1), Min(child2, child3));

    Vec3f maxP = Max(Max(child0, child1), Max(child2, child3));
    f32 expX   = Ceil(Log2f((maxP.x - node->minP.x) / 255.f));
    f32 expY   = Ceil(Log2f((maxP.y - node->minP.y) / 255.f));
    f32 expZ   = Ceil(Log2f((maxP.z - node->minP.z) / 255.f));

    f32 powX = Pow(2.f, expX);
    f32 powY = Pow(2.f, expY);
    f32 powZ = Pow(2.f, expZ);

    Lane4F32 powXLane(powX);
    Lane4F32 powYLane(powY);
    Lane4F32 powZLane(powZ);

    Lane4F32 minX(child0.minX, child1.minX, child2.minX, child3.minX);
    Lane4F32 minY(child0.minY, child1.minY, child2.minY, child3.minY);
    Lane4F32 minZ(child0.minZ, child1.minZ, child2.minZ, child3.minZ);

    Lane4F32 maxX(child0.maxX, child1.maxX, child2.maxX, child3.maxX);
    Lane4F32 maxY(child0.maxY, child1.maxY, child2.maxY, child3.maxY);
    Lane4F32 maxZ(child0.maxZ, child1.maxZ, child2.maxZ, child3.maxZ);

    Lane4F32 nodeMinX(node->minP.x);
    Lane4F32 nodeMinY(node->minP.y);
    Lane4F32 nodeMinZ(node->minP.z);

    Lane4F32 qNodeMinX = Floor((minX - nodeMinX) / powXLane);
    Lane4F32 qNodeMinY = Floor((minY - nodeMinY) / powYLane);
    Lane4F32 qNodeMinZ = Floor((minZ - nodeMinZ) / powZLane);

    Lane4F32 qNodeMaxX = Ceil((maxX - nodeMinX) / powXLane);
    Lane4F32 qNodeMaxY = Ceil((maxY - nodeMinY) / powYLane);
    Lane4F32 qNodeMaxZ = Ceil((maxZ - nodeMinZ) / powZLane);

    Lane4F32 maskMinX = FMA(powXLane, qNodeMinX, nodeMinX) > minX;
    node->minX        = TruncateToU8(Max(Select(maskMinX, qNodeMinX - 1, qNodeMinX), MIN_QUAN));
    Lane4F32 maskMinY = FMA(powYLane, qNodeMinY, nodeMinY) > minY;
    node->minY        = TruncateToU8(Max(Select(maskMinY, qNodeMinY - 1, qNodeMinY), MIN_QUAN));
    Lane4F32 maskMinZ = FMA(powZLane, qNodeMinZ, nodeMinZ) > minZ;
    node->minZ        = TruncateToU8(Max(Select(maskMinZ, qNodeMinZ - 1, qNodeMinZ), MIN_QUAN));

    Lane4F32 maskMaxX = FMA(powXLane, qNodeMaxX, nodeMinX) < maxX;
    node->maxX        = TruncateToU8(Min(Select(maskMaxX, qNodeMaxX + 1, qNodeMaxX), MAX_QUAN));
    Lane4F32 maskMaxY = FMA(powYLane, qNodeMaxY, nodeMinY) < maxY;
    node->maxY        = TruncateToU8(Min(Select(maskMaxY, qNodeMaxY + 1, qNodeMaxY), MAX_QUAN));
    Lane4F32 maskMaxZ = FMA(powZLane, qNodeMaxZ, nodeMinZ) < maxZ;
    node->maxZ        = TruncateToU8(Min(Select(maskMaxZ, qNodeMaxZ + 1, qNodeMaxZ), MAX_QUAN));

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
} // namespace rt

#endif
