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

    float3 boundsMin;
    float3 boundsMax;

    int prev;
    int current;

#ifdef __SLANG__
    [mutating]
#endif
    void Start(float3 mins, float3 maxs, float3 o, float3 d)
    {
        boundsMin = mins;
        boundsMax = maxs;
        rayO = o;
        rayDir = d;

        float tLeave;
        bool intersects = IntersectRayAABB(boundsMin, boundsMax, o, d, currentT, tLeave);
        current = intersects ? 0 : -1;
    }

    // internal node
    //      - if current ray pos is outside bounds, go to parent
    //      - find closest child that is in front of the ray
    //      - go to that child
    // leaf node
    //      - Next() terminates
    //      - return majorant and minorant

#ifdef __SLANG__
    [mutating]
#endif
    bool Next()
    {
        prev = -1;
        if (current == -1) return false;
        int start = current;
        for (;;)
        {
            float3 currentPos = rayO + currentT * rayDir;
            float3 center = (boundsMin + boundsMax) / 2.f;
            currentPos -= center;
            float3 extent = (boundsMax - boundsMin) / 2.f;
            int next;

            uint childIndex = nodes[current].childIndex;
            // go to parent
            if (childIndex == ~0u || currentPos.x > extent.x || currentPos.y > extent.y ||
                currentPos.z > extent.z || currentPos.x < -extent.x ||
                currentPos.y < -extent.y || currentPos.z < -extent.z)
            {
                // go to parent
                if (current == 0)
                {
                    current = -1;
                    break;
                }
                int axisMask = (current - 1) & 0x7;
                next         = nodes[current].parentIndex;

                boundsMin = float3((axisMask & 0x1) ? boundsMin.x - 2.f * extent.x : boundsMin.x,
                                   (axisMask & 0x2) ? boundsMin.y - 2.f * extent.y : boundsMin.y,
                                   (axisMask & 0x4) ? boundsMin.z - 2.f * extent.z : boundsMin.z);
                boundsMax = float3((axisMask & 0x1) ? boundsMax.x : boundsMax.x + 2.f * extent.x,
                                   (axisMask & 0x2) ? boundsMax.y : boundsMax.y + 2.f * extent.y,
                                   (axisMask & 0x4) ? boundsMax.z : boundsMax.z + 2.f * extent.z);

                currentT += 0.0001f;
#if 0
                for (;;)
                {
                    float3 currentPos = rayO + currentT * rayDir - center;
                    currentT = NextFloatUp(currentT);
                    if (currentPos.x > extent.x || currentPos.y > extent.y ||
                        currentPos.z > extent.z || currentPos.x < -extent.x ||
                        currentPos.y < -extent.y || currentPos.z < -extent.z)
                    {
                        break;
                    }
                }
#endif
            }
            // go to child
            else 
            {
                uint closestChild = (currentPos.x >= 0.f) | ((currentPos.y >= 0.f) << 1) | ((currentPos.z >= 0.f) << 2);
                next = childIndex + closestChild;

                boundsMin = float3((closestChild & 0x1) ? center.x : boundsMin.x,
                                   (closestChild & 0x2) ? center.y : boundsMin.y,
                                   (closestChild & 0x4) ? center.z : boundsMin.z);
                boundsMax = float3((closestChild & 0x1) ? boundsMax.x : center.x,
                                   (closestChild & 0x2) ? boundsMax.y : center.y,
                                   (closestChild & 0x4) ? boundsMax.z : center.z);
            }

            prev = current;
            current = next;

            if (nodes[current].childIndex == ~0u)
            {
                float newT;
                float tLeave;
                bool intersects =
                    IntersectRayAABB(boundsMin, boundsMax, rayO, rayDir, newT, tLeave);

                if (!intersects || start == current)
                {
                    currentT = max(tLeave, currentT);
                    currentT += 0.0001f;
                }
                else 
                {
                    //currentT = newT;
                    tMax     = tLeave;

                    return true;
                }
            }
        }
        return false;
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

#ifdef __SLANG__
    [mutating]
#endif
    void Step(float deltaT)
    {
        currentT += deltaT;
    }
};

#endif
