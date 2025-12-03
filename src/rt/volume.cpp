#include "containers.h"
#include "math/basemath.h"
#include "math/math_include.h"
#include "math/bounds.h"
#include "memory.h"
#include "parallel.h"
#include "string.h"
#include <atomic>
#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/io/IO.h>
#include <nanovdb/GridHandle.h>

namespace rt
{

struct OctreeNode
{
    float minValue;
    float maxValue;
    OctreeNode *children;
    // int childIndex;
};

struct GPUOctreeNode
{
    float minValue;
    float maxValue;
    int childIndex;
    int parentIndex;
};

static const float T = Abs(1.f / std::log(0.5f));

static OctreeNode CreateOctree(Arena **arenas, const Bounds &bounds,
                               const nanovdb::FloatGrid *grid, std::atomic<u32> &nodeCount)
{
    Arena *arena = arenas[GetThreadIndex()];

    nanovdb::Vec3d i0 =
        grid->worldToIndexF(nanovdb::Vec3d(bounds.minP[0], bounds.minP[1], bounds.minP[2]));
    nanovdb::Vec3d i1 =
        grid->worldToIndexF(nanovdb::Vec3d(bounds.maxP[0], bounds.maxP[1], bounds.maxP[2]));
    auto indexBBox = grid->indexBBox();

    Vec3i indexMin(Max((int)Floor(i0[0]), indexBBox.min()[0]),
                   Max((int)Floor(i0[1]), indexBBox.min()[1]),
                   Max((int)Floor(i0[2]), indexBBox.min()[2]));
    Vec3i indexMax(Min((int)Ceil(i1[0]), indexBBox.max()[0]),
                   Min((int)Ceil(i1[1]), indexBBox.max()[1]),
                   Min((int)Ceil(i1[2]), indexBBox.max()[2]));

    int count = indexMax[2] - indexMin[2] + 1;

    struct Output
    {
        float min;
        float max;
    };

    ScratchArena scratch;

    // TODO: verify that the extents of these are correct
    ParallelForOutput output = ParallelFor<Output>(
        scratch.temp, 0, count, 1, [&](Output &output, u32 jobID, u32 start, u32 count) {
            int s          = indexMin[2] + start;
            float maxValue = 0.f;
            float minValue = 1.f;
            auto accessor  = grid->getAccessor();
            for (int z = s; z < s + count; z++)
            {
                for (int y = indexMin[1]; y <= indexMax[1]; y++)
                {
                    for (int x = indexMin[0]; x <= indexMax[0]; x++)
                    {
                        float value = accessor.getValue({x, y, z});
                        if (value == 0.f && minValue != 0.f && minValue != 1.f)
                        {
                            int stop = 5;
                        }
                        maxValue = Max(value, maxValue);
                        minValue = Min(value, minValue);
                    }
                }
            }
            output.min = minValue;
            output.max = maxValue;
        });

    Output minMax;
    Reduce(minMax, output, [&](Output &l, const Output &r) {
        l.min = Min(l.min, r.min);
        l.max = Max(l.max, r.max);
    });

    float maxValue = minMax.max;
    float minValue = minMax.min;

    f32 diag = Length(ToVec3f(bounds.Diagonal()));

    OctreeNode node = {};
    node.minValue   = minValue;
    node.maxValue   = maxValue;
    // Print("i0: %i %i %i, i1: %i %i %i, size: %u, min: %f, max: %f\n", (int)i0[0],
    // (int)i0[1],
    //       (int)i0[2], (int)i1[0], (int)i1[1], (int)i1[2], nodes.size(), minValue, maxValue);
    // if (minValue != 0.f || maxValue != 1.f)
    // {
    //     Print("min: %f, max: %f\n", minValue, maxValue);
    // }

    nodeCount.fetch_add(1, std::memory_order_relaxed);

    if ((maxValue - minValue) * diag > T)
    {
        node.children = PushArray(arena, OctreeNode, 8);
        // node.childIndex = nodes.size() + 1 + queueEnd - queueStart;
        ParallelForLoop(0, 8, 1, 1, [&](u32 jobID, u32 i) {
            Vec3f minP   = ToVec3f(bounds.minP);
            Vec3f maxP   = ToVec3f(bounds.maxP);
            Vec3f center = ToVec3f(bounds.Centroid());

            Vec3f newMinP((i & 1) ? center.x : minP.x, (i & 2) ? center.y : minP.y,
                          (i & 4) ? center.z : minP.z);
            Vec3f newMaxP((i & 1) ? maxP.x : center.x, (i & 2) ? maxP.y : center.y,
                          (i & 4) ? maxP.z : center.z);

            Bounds newBounds(newMinP, newMaxP);
            node.children[i] = CreateOctree(arenas, newBounds, grid, nodeCount);
        });
    }
    // else Print("%f %f\n", minValue, maxValue);

    return node;
}

static bool IntersectRayAABB(const Vec3f &boundsMin, const Vec3f &boundsMax, const Vec3f &o,
                             const Vec3f &d, float &tEntry, float &tLeave)
{
    Vec3f invDir        = 1.f / d;
    Vec3f tIntersectMin = (boundsMin - o) * invDir;
    Vec3f tIntersectMax = (boundsMax - o) * invDir;

    Vec3f tMin = Min(tIntersectMin, tIntersectMax);
    Vec3f tMax = Max(tIntersectMin, tIntersectMax);

    tEntry = Max(tMin.x, Max(tMin.y, tMin.z));
    tLeave = Min(tMax.x, Min(tMax.y, tMax.z));

    return tEntry <= tLeave;
}

struct VolumeIterator
{
    GPUOctreeNode *nodes;
    float currentT;
    float tMax;

    Vec3f rayO;
    Vec3f rayDir;

    Vec3f boundsMin;
    Vec3f boundsMax;

    int prev;
    int current;

    VolumeIterator(const Vec3f &mins, const Vec3f &maxs, const Vec3f &o, const Vec3f &d,
                   GPUOctreeNode *nodes)
        : boundsMin(mins), boundsMax(maxs), rayO(o), rayDir(d), nodes(nodes), current(0),
          prev(-1)
    {
        float tLeave;
        bool intersects = IntersectRayAABB(boundsMin, boundsMax, o, d, currentT, tLeave);
        Print("t start: %f %f\n", currentT, tLeave);
        Assert(intersects);
    }

    // internal node
    //      - if current ray pos is outside bounds, go to parent
    //      - find closest child that is in front of the ray
    //      - go to that child
    // leaf node
    //      - Next() terminates
    //      - return majorant and minorant

    bool Next()
    {
        for (;;)
        {
            // TODO floating point precision
            Vec3f currentPos = rayO + currentT * rayDir;
            // Print("pos: %f %f %f min: %f %f %f max: %f %f %f\n", currentPos.x, currentPos.y,
            //       currentPos.z, boundsMin.x, boundsMin.y, boundsMin.z, boundsMax.x,
            //       boundsMax.y, boundsMax.z);

            Vec3f center = (boundsMin + boundsMax) / 2.f;
            currentPos -= center;
            Vec3f extent = (boundsMax - boundsMin) / 2.f;
            int next;

            u32 childIndex = nodes[current].childIndex;
            // go to parent
            if (childIndex == ~0u || currentPos.x > extent.x || currentPos.y > extent.y ||
                currentPos.z > extent.z || currentPos.x < -extent.x ||
                currentPos.y < -extent.y || currentPos.z < -extent.z)
            {
                // go to parent
                if (current == 0)
                {
                    current = -1;
                    return false;
                }
                int axisMask = (current - 1) & 0x7;
                next         = nodes[current].parentIndex;

                boundsMin =
                    Vec3f((axisMask & 0x1) ? boundsMin.x - 2.f * extent.x : boundsMin.x,
                          (axisMask & 0x2) ? boundsMin.y - 2.f * extent.y : boundsMin.y,
                          (axisMask & 0x4) ? boundsMin.z - 2.f * extent.z : boundsMin.z);
                boundsMax =
                    Vec3f((axisMask & 0x1) ? boundsMax.x : boundsMax.x + 2.f * extent.x,
                          (axisMask & 0x2) ? boundsMax.y : boundsMax.y + 2.f * extent.y,
                          (axisMask & 0x4) ? boundsMax.z : boundsMax.z + 2.f * extent.z);

                // offset outside of
                u32 times = 0;
                for (;;)
                {
                    Vec3f currentPos = rayO + currentT * rayDir - center;
                    // Print("help me: %f %f %f, %f %f %f\n\n", currentPos.x, currentPos.y,
                    //       currentPos.z, extent.x, extent.y, extent.z);
                    times++;
                    currentT = NextFloatUp(currentT);
                    if (currentPos.x > extent.x || currentPos.y > extent.y ||
                        currentPos.z > extent.z || currentPos.x < -extent.x ||
                        currentPos.y < -extent.y || currentPos.z < -extent.z)
                    {
                        break;
                    }
                }

                // Print("curr: %u, next: %u, times %u\n", current, next, times);
            }
            // go to child
            else
            {
                u32 closestChild = (currentPos.x >= 0.f) | ((currentPos.y >= 0.f) << 1) |
                                   ((currentPos.z >= 0.f) << 2);
                next = childIndex + closestChild;

                // Print("curr: %u, next: %u, childIndex: %u\n", current, next, childIndex);

                boundsMin = Vec3f((closestChild & 0x1) ? center.x : boundsMin.x,
                                  (closestChild & 0x2) ? center.y : boundsMin.y,
                                  (closestChild & 0x4) ? center.z : boundsMin.z);
                boundsMax = Vec3f((closestChild & 0x1) ? boundsMax.x : center.x,
                                  (closestChild & 0x2) ? boundsMax.y : center.y,
                                  (closestChild & 0x4) ? boundsMax.z : center.z);
            }

            prev    = current;
            current = next;

            if (nodes[current].childIndex == ~0u)
            {
                float newT;
                float tLeave;
                bool intersects =
                    IntersectRayAABB(boundsMin, boundsMax, rayO, rayDir, newT, tLeave);
                // Assert(newT >= currentT);

                currentT = newT;
                tMax     = tLeave;
                Print("final t: %f %f, node data: %f %f\n", newT, tMax,
                      nodes[current].minValue, nodes[current].maxValue);

                return true;
            }
        }
    }
};

void Volumes(Arena *arena)
{
    // build the octree over all volumes...
    string filename = "../../data/wdas_cloud/wdas_cloud.nvdb";
    auto handle     = nanovdb::io::readGrid(std::string((char *)filename.str, filename.size));
    auto grid       = handle.grid<float>();

    nanovdb::Vec3dBBox bbox = grid->worldBBox();
    Bounds rootBounds(Vec3f(bbox.min()[0], bbox.min()[1], bbox.min()[2]),
                      Vec3f(bbox.max()[0], bbox.max()[1], bbox.max()[2]));

    ScratchArena scratch;
    Arena **arenas = GetArenaArray(scratch.temp.arena);

    std::atomic<u32> count = 1;
    OctreeNode rootNode    = {};
    rootNode.minValue      = grid->tree().root().minimum();
    rootNode.maxValue      = grid->tree().root().maximum();

    if ((rootNode.maxValue - rootNode.minValue) * Length(ToVec3f(rootBounds.Diagonal())) > T)
    {
        rootNode.children = PushArray(arena, OctreeNode, 8);
        // node.childIndex = nodes.size() + 1 + queueEnd - queueStart;
        ParallelForLoop(0, 8, 1, 1, [&](u32 jobID, u32 i) {
            Vec3f minP   = ToVec3f(rootBounds.minP);
            Vec3f maxP   = ToVec3f(rootBounds.maxP);
            Vec3f center = ToVec3f(rootBounds.Centroid());

            Vec3f newMinP((i & 1) ? center.x : minP.x, (i & 2) ? center.y : minP.y,
                          (i & 4) ? center.z : minP.z);
            Vec3f newMaxP((i & 1) ? maxP.x : center.x, (i & 2) ? maxP.y : center.y,
                          (i & 4) ? maxP.z : center.z);

            Bounds newBounds(newMinP, newMaxP);
            rootNode.children[i] = CreateOctree(arenas, newBounds, grid, count);
        });
    }

    u32 numNodes = count.load();
    Print("num nodes: %u\n", numNodes);

    // Flatten
    StaticArray<GPUOctreeNode> newNodes(arena, numNodes);
    StaticArray<OctreeNode> oldNodes(scratch.temp.arena, numNodes);
    oldNodes.Push(rootNode);
    GPUOctreeNode newNode;
    newNode.minValue    = rootNode.minValue;
    newNode.maxValue    = rootNode.maxValue;
    newNode.parentIndex = -1;
    newNodes.Push(newNode);

    // TODO: implicit representation? essentially, find where ray intersects
    // the aggregate, construct a location code based on this position, then lookup
    // the code in a hash table.

    for (int i = 0; i < numNodes; i++)
    {
        OctreeNode &node          = oldNodes[i];
        GPUOctreeNode &octreeNode = newNodes[i];
        octreeNode.childIndex     = -1;

        if (node.children)
        {
            octreeNode.childIndex = newNodes.Length();
            for (int childIndex = 0; childIndex < 8; childIndex++)
            {
                GPUOctreeNode newNode;
                newNode.minValue    = node.children[childIndex].minValue;
                newNode.maxValue    = node.children[childIndex].maxValue;
                newNode.parentIndex = i;
                newNodes.Push(newNode);

                oldNodes.Push(node.children[childIndex]);
            }
        }
    }

    ReleaseArenaArray(arenas);

    Vec3f rayDirection = Normalize(ToVec3f(rootBounds.Centroid()));
    Vec3f rayOrigin(-rayDirection * 1000.f);

    VolumeIterator iterator(ToVec3f(rootBounds.minP), ToVec3f(rootBounds.maxP), rayOrigin,
                            rayDirection, newNodes.data);

    u32 times = 0;
    for (;;)
    {
        bool more = iterator.Next();
        times++;
        if (!more) break;
        iterator.currentT = iterator.tMax;
    }
    Print("times %u\n", times);

    int stop = 5;
}
} // namespace rt
