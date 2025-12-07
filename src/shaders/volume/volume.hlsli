#ifndef VOLUME_HLSLI_
#define VOLUME_HLSLI_

#include "../intersect_ray_aabb.hlsli"

struct OctreeNode 
{
    float minValue;
    float maxValue;
    int childIndex;
    int parentIndex;
};

float NextFloatUp(float v)
{
    if (isinf(v) && v > 0.f) return v;
    if (v == -0.f) v = 0.f;

    uint ui = asuint(v);
    if (v >= 0) ++ui;
    else --ui;
    return asfloat(ui);
}

StructuredBuffer<OctreeNode> nodes : register(t29);

#define PNANOVDB_HLSL
#include "PNanoVDB.h"


struct VolumeIterator
{
    float currentT;
    float tMax;

    float3 rayO;
    float3 rayDir;
    float3 invDir;

    float3 octreeBoundsMin;
    float3 octreeBoundsMax;

    uint raySignMask;
    uint crossingAxis;

    int current;

    void Start(float3 mins, float3 maxs, float3 o, float3 d)
    {
        octreeBoundsMin = float3(0, 0, 0);
        octreeBoundsMax = float3(1, 1, 1);

        float3 diag = maxs - mins;
        rayO = (o - mins) / diag;
        rayDir = d / diag;

        invDir        = 1.f / rayDir;

        float3 tIntersectMin = (octreeBoundsMin - rayO) * invDir;
        float3 tIntersectMax = (octreeBoundsMax - rayO) * invDir;

        float3 tMin = min(tIntersectMin, tIntersectMax);
        float3 tMax_ = max(tIntersectMin, tIntersectMax);

        float tEntry = max(tMin.x, max(tMin.y, max(tMin.z, 0.f)));
        float tLeave = min(tMax_.x, min(tMax_.y, tMax_.z));

        bool intersects = tEntry < tLeave;
        current = intersects ? 0 : -1;
        currentT = min(tEntry, tLeave);

        // Traverse to child
        raySignMask = (rayDir.x < 0.f ? 1 : 0) | 
                      (rayDir.y < 0.f ? 2 : 0) | 
                      (rayDir.z < 0.f ? 4 : 0);
        TraverseToChild();

        tIntersectMin = (octreeBoundsMin - rayO) * invDir;
        tIntersectMax = (octreeBoundsMax - rayO) * invDir;

        tMin = min(tIntersectMin, tIntersectMax);
        tMax_ = max(tIntersectMin, tIntersectMax);

        tEntry = max(tMin.x, max(tMin.y, max(tMin.z, 0.f)));
        tLeave = min(tMax_.x, min(tMax_.y, tMax_.z));

        crossingAxis = tMax_.x == tLeave ? 0 : (tMax_.y == tLeave ? 1 : 2);
        tMax = tLeave;
        currentT = min(currentT, min(tEntry, tLeave));
    }

    uint CalculateAxisMask()
    {
        return (current - 1) & 0x7;
    }

    bool Next()
    {
        uint rayCode = (~raySignMask) & 0x7;
        uint axisMask = CalculateAxisMask();

        while (current != -1 && ((rayCode ^ axisMask) & (1u << crossingAxis)) == 0)
        {
            BoundsToParent();

            current = current == 0 ? -1 : nodes[current].parentIndex;
            axisMask = CalculateAxisMask();
        }

        if (current == -1) return false;

        // Traverse to the neighbor
        BoundsToParent();
        current -= axisMask;
        axisMask ^= (1u << crossingAxis);
        current += axisMask;
        uint closestChild = BoundsToChild();

        // Traverse to child
        TraverseToChild();

        float3 tIntersectMin = (octreeBoundsMin - rayO) * invDir;
        float3 tIntersectMax = (octreeBoundsMax - rayO) * invDir;

        float3 tMin = min(tIntersectMin, tIntersectMax);
        float3 tMax_ = max(tIntersectMin, tIntersectMax);

        float tEntry = max(tMin.x, max(tMin.y, max(tMin.z, 0.f)));
        float tLeave = min(tMax_.x, min(tMax_.y, tMax_.z));

        // Prepare next traversal
        crossingAxis = tMax_.x == tLeave ? 0 : (tMax_.y == tLeave ? 1 : 2);

        currentT += tEntry > tLeave ? .0001f : 0.f;
        tMax = max(currentT, tLeave);
        // currentT = Min(currentT, Min(tEntry, tLeave));
        return true;
    }

    void BoundsToParent()
    {
        uint axisMask = CalculateAxisMask();
        float3 extent = (octreeBoundsMax - octreeBoundsMin) / 2.f;
        octreeBoundsMin = float3((axisMask & 0x1) ? octreeBoundsMin.x - 2.f * extent.x : octreeBoundsMin.x,
                (axisMask & 0x2) ? octreeBoundsMin.y - 2.f * extent.y : octreeBoundsMin.y,
                (axisMask & 0x4) ? octreeBoundsMin.z - 2.f * extent.z : octreeBoundsMin.z);
        octreeBoundsMax = float3((axisMask & 0x1) ? octreeBoundsMax.x : octreeBoundsMax.x + 2.f * extent.x,
                (axisMask & 0x2) ? octreeBoundsMax.y : octreeBoundsMax.y + 2.f * extent.y,
                (axisMask & 0x4) ? octreeBoundsMax.z : octreeBoundsMax.z + 2.f * extent.z);
    }

    uint BoundsToChild()
    {
        float3 center = (octreeBoundsMax + octreeBoundsMin) / 2.f;
        float3 tPlanes = (center - rayO) * invDir;
        uint closestChild = (tPlanes.x <= currentT) | ((tPlanes.y <= currentT) << 1) | ((tPlanes.z <= currentT) << 2);
        closestChild ^= raySignMask;
        octreeBoundsMin = float3((closestChild & 0x1) ? center.x : octreeBoundsMin.x,
                (closestChild & 0x2) ? center.y : octreeBoundsMin.y,
                (closestChild & 0x4) ? center.z : octreeBoundsMin.z);
        octreeBoundsMax = float3((closestChild & 0x1) ? octreeBoundsMax.x : center.x,
                (closestChild & 0x2) ? octreeBoundsMax.y : center.y,
                (closestChild & 0x4) ? octreeBoundsMax.z : center.z);
        return closestChild;
    }

    void TraverseToChild()
    {
        while (nodes[current].childIndex != ~0u)
        {
            current = nodes[current].childIndex + BoundsToChild();
        }
    }

    float GetCurrentT() { return currentT; }
    float GetTMax() { return tMax; }

    void GetSegmentProperties(out float tMin, out float tFar, out float minor, out float major)
    {
        tMin = GetCurrentT();
        tFar = GetTMax();
        minor = nodes[current].minValue;
        major = nodes[current].maxValue;
    }

    void Step(float deltaT)
    {
        currentT += deltaT;
    }
};
#endif
