#include "containers.h"
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
};

static OctreeNode CreateOctree(Arena **arenas, const Bounds &bounds,
                               const nanovdb::FloatGrid *grid, std::atomic<u32> &nodeCount)
{
    const float T = Abs(1.f / std::log(0.5f));
    Arena *arena  = arenas[GetThreadIndex()];

    nanovdb::Vec3d i0 =
        grid->worldToIndexF(nanovdb::Vec3d(bounds.minP[0], bounds.minP[1], bounds.minP[2]));
    nanovdb::Vec3d i1 =
        grid->worldToIndexF(nanovdb::Vec3d(bounds.maxP[0], bounds.maxP[1], bounds.maxP[2]));

    int count = (int)Ceil(i1[2]) - (int)Floor(i0[2]) + 1;

    struct Output
    {
        float min;
        float max;
    };

    ScratchArena scratch;

    // TODO: verify that the extents of these are correct
    ParallelForOutput output = ParallelFor<Output>(
        scratch.temp, 0, count, 1, [&](Output &output, u32 jobID, u32 start, u32 count) {
            int s          = (int)Floor(i0[2]) + start;
            float maxValue = 0.f;
            float minValue = 1.f;
            auto accessor  = grid->getAccessor();
            for (int z = s; z < s + count; z++)
            {
                for (int y = (int)Floor(i0[1]); y <= (int)Ceil(i1[1]); y++)
                {
                    for (int x = (int)Floor(i0[0]); x <= (int)Ceil(i1[0]); x++)
                    {
                        float value = accessor.getValue({x, y, z});
                        maxValue    = Max(value, maxValue);
                        minValue    = Min(value, minValue);
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

    return node;
}

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

    std::atomic<u32> count = {};
    OctreeNode rootNode    = CreateOctree(arenas, rootBounds, grid, count);

    u32 numNodes = count.load();
    Print("num nodes: %u\n", numNodes);

    // Flatten
    StaticArray<GPUOctreeNode> newNodes(arena, numNodes);
    StaticArray<OctreeNode> oldNodes(scratch.temp.arena, numNodes);
    oldNodes.Push(rootNode);
    GPUOctreeNode newNode;
    newNode.minValue = rootNode.minValue;
    newNode.maxValue = rootNode.maxValue;
    newNodes.Push(newNode);

    for (int i = 0; i < numNodes; i++)
    {
        OctreeNode &node          = oldNodes[i];
        GPUOctreeNode &octreeNode = newNodes[i];
        octreeNode.childIndex     = -1;

        if (node.children)
        {
            octreeNode.childIndex = newNodes.Length();
            for (int i = 0; i < 8; i++)
            {
                GPUOctreeNode newNode;
                newNode.minValue = node.children[i].minValue;
                newNode.maxValue = node.children[i].maxValue;
                newNodes.Push(newNode);

                oldNodes.Push(node.children[i]);
            }
        }
    }

    ReleaseArenaArray(arenas);

    int stop = 5;
}
} // namespace rt
