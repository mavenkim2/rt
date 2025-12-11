#ifndef VOLUME_HLSLI_
#define VOLUME_HLSLI_

#include "../intersect_ray_aabb.hlsli"
#include "../sampling.hlsli"

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

        if (current == 0)
        {
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
    }

    uint CalculateAxisMask()
    {
        return (current - 1) & 0x7;
    }

    bool Done()
    {
        return current == -1 || current == 0;
    }

    bool Next()
    {
        if (current == -1) return false;

        int start = current;

        uint rayCode = (~raySignMask) & 0x7;
        uint axisMask = CalculateAxisMask();

        while (current != 0 && ((rayCode ^ axisMask) & (1u << crossingAxis)) == 0)
        {
            BoundsToParent();

            current = nodes[current].parentIndex;
            axisMask = CalculateAxisMask();
        }

        if (current == 0) return false;

        // Traverse to the neighbor
        BoundsToParent();
        current = nodes[current].parentIndex;
        //current -= axisMask;
        //axisMask ^= (1u << crossingAxis);
        //current += axisMask;
        //uint closestChild = BoundsToChild();

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

        float3 prevMin = octreeBoundsMin;
        float3 prevMax = octreeBoundsMax;

        octreeBoundsMin = float3((axisMask & 0x1) ? octreeBoundsMin.x - 2.f * extent.x : octreeBoundsMin.x,
                (axisMask & 0x2) ? octreeBoundsMin.y - 2.f * extent.y : octreeBoundsMin.y,
                (axisMask & 0x4) ? octreeBoundsMin.z - 2.f * extent.z : octreeBoundsMin.z);
        octreeBoundsMax = float3((axisMask & 0x1) ? octreeBoundsMax.x : octreeBoundsMax.x + 2.f * extent.x,
                (axisMask & 0x2) ? octreeBoundsMax.y : octreeBoundsMax.y + 2.f * extent.y,
                (axisMask & 0x4) ? octreeBoundsMax.z : octreeBoundsMax.z + 2.f * extent.z);

        if (any(octreeBoundsMin < 0.f) || any(octreeBoundsMax > 1.f))
        {
            printf("prev: %f %f %f %f %f %f\nnext: %f %f %f %f %f %f %u %u\n",
            prevMin.x, prevMin.y, prevMin.z,
            prevMax.x, prevMax.y, prevMax.z,
            octreeBoundsMin.x, octreeBoundsMin.y, octreeBoundsMin.z, 
                   octreeBoundsMax.x, octreeBoundsMax.y, octreeBoundsMax.z, current, axisMask);
        }
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

struct VolumeVertexData
{
    float transmittance;
    float density;
    float majorant;
};

bool GetNextVolumeVertex(inout VolumeIterator iterator, inout RNG rng, out VolumeVertexData data, 
                         float3 pos, float3 dir)
{
    bool done = false;
    float transmittance = 1.f;
    // TODO hardcoded
    const float densityScale = 4.f;

    if (!iterator.Done())
    {
        int count = 0;
        do
        {
            count++;
            float tMin, tMax, minorant, majorant;
            iterator.GetSegmentProperties(tMin, tMax, minorant, majorant);
            majorant *= densityScale;

            uint test = 0;

            for (;;)
            {
                float u = rng.Uniform();
                float t = iterator.GetCurrentT();
                float tStep = majorant == 0.f ? tMax - t : SampleExponential(u, majorant);

                if (t + tStep >= tMax)
                {
                    float deltaT = tMax - t;
                    iterator.Step(deltaT);

                    transmittance *= exp(-deltaT * majorant);
                    break;
                }
                else 
                {
                    iterator.Step(tStep);

                    pnanovdb_grid_handle_t grid = {0};

                    // TODO: hardcoded
                    int nanovdbIndex = 0;
                    pnanovdb_tree_handle_t tree = pnanovdb_grid_get_tree(nanovdbIndex, grid);
                    pnanovdb_root_handle_t root = pnanovdb_tree_get_root(nanovdbIndex, tree);
                    pnanovdb_uint32_t gridType = pnanovdb_grid_get_grid_type(nanovdbIndex, grid);

                    pnanovdb_readaccessor_t accessor;
                    pnanovdb_readaccessor_init(accessor, root);

                    float3 gridPos = pos + iterator.GetCurrentT() * dir;
                    float3 indexSpacePosition = pnanovdb_grid_world_to_indexf(nanovdbIndex, grid, gridPos);
                    pnanovdb_coord_t coord = pnanovdb_hdda_pos_to_ijk(indexSpacePosition);

                    pnanovdb_address_t valueAddr = pnanovdb_readaccessor_get_value_address(gridType, nanovdbIndex, accessor, coord);
                    float density = pnanovdb_read_float(nanovdbIndex, valueAddr);
                    // TODO hardcoded
                    density *= densityScale;

                    transmittance *= exp(-tStep * majorant);

                    data.transmittance = transmittance;
                    data.density = density;
                    data.majorant = majorant;
                    return false;
                }
            }
        } while (iterator.Next());
    }
    return true;
}

bool SpeculativelyDuplicateRays(inout uint N, bool terminated, inout uint laneToGo)
{
    uint laneCount = WaveGetLaneCount();
    uint stride = laneCount >> N;

    uint waveIndex = WaveGetLaneIndex();
    uint numAlive = WaveActiveCountBits(!terminated && waveIndex < stride);
    if (numAlive <= stride)
    {
        N++;
        uint stride = (laneCount >> N);
        if (stride == 0) return true;

        waveIndex &= stride - 1;
        // Compact state to N lowest threads
        uint laneToGo = WavePrefixCountBits(!terminated);
        for (uint i = waveIndex; i < laneCount; i++)
        {
            uint lanePrefix = WaveReadLaneAt(laneToGo, i);
            if (lanePrefix == waveIndex)
            {
                laneToGo = i;
                return false;
            }
        }
        return true;
    }
    return terminated;
}

// Implements speculative path execution
// section 4.6.3 https://graphics.pixar.com/library/RenderManXPU/paper.pdf

bool GetNextVolumeVertexSpeculative(inout VolumeIterator iterator, inout RNG rng, out VolumeVertexData data, 
                                    float3 pos, float3 dir, uint N)
{
    bool done = false;
    float transmittance = 1.f;
    // TODO hardcoded
    const float densityScale = 4.f;

    for (uint i = 0; i < N; i++)
    {
        float tMin, tMax, minorant, majorant;
        iterator.GetSegmentProperties(tMin, tMax, minorant, majorant);
        majorant *= densityScale;

        float u = rng.Uniform();
        float t = iterator.GetCurrentT();
        float tStep = majorant == 0.f ? tMax - t : SampleExponential(u, majorant);

        if (t + tStep >= tMax)
        {
            float deltaT = tMax - t;
            iterator.Step(deltaT);

            transmittance *= exp(-deltaT * majorant);
            done = iterator.Next();
        }
        else 
        {
            // We don't know whether this is a null or real collision. Make sure RNG 
            // advances in either case.
            rng.Uniform();
            rng.Uniform2D();
            iterator.Step(tStep);
            transmittance *= exp(-tStep * majorant);
        }
    }
    data.transmittance = transmittance;
    return done;
}

#endif
