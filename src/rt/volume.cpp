#include "volume.h"
#include "bxdf.h"
#include "containers.h"
#include "math/basemath.h"
#include "math/math_include.h"
#include "math/bounds.h"
#include "math/vec3.h"
#include "memory.h"
#include "parallel.h"
#include "string.h"
#include <atomic>
#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/io/IO.h>
#include <nanovdb/GridHandle.h>
#include "random.h"

#define PNANOVDB_C
#define PNANOVDB_HDDA
#include <nanovdb/PNanoVDB.h>

namespace rt
{

struct OctreeNode
{
    float minValue;
    float maxValue;
    OctreeNode *children;
    // int childIndex;
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

    int groupSize = Max(1, count / 128);

    // TODO: verify that the extents of these are correct
    ParallelForOutput output = ParallelFor<Output>(
        scratch.temp, 0, count, groupSize,
        [&](Output &output, u32 jobID, u32 start, u32 count) {
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

struct TetrahedralCell
{
    int neighbors[4];
    Vec3f points[4];
};

static void AdaptiveSplitTetrahedra(std::vector<TetrahedralCell> &tets, int index)
{
    TetrahedralCell &tet = tets[index];
}

static void BuildAdaptiveTetrahedralGrid(CommandBuffer *cmd, Arena *arena)
{
    // Base 24 tetrahedra
    std::vector<TetrahedralCell> tets;

    static const f32 next[4]     = {0.f, 0.f, 1.f, 1.f};
    static const f32 nextNext[4] = {0.f, 1.f, 1.f, 0.f};

    const Vec3f cubeCenter(0.5f);

    // Four for each face
    for (int face = 0; face < 6; face++)
    {
        Vec3f center;
        int plane             = face / 2;
        float planeValue      = (face & 1) ? 1.f : 0.f;
        int planeNext         = (plane + 1) % 3;
        int planeNextNext     = (plane + 2) % 3;
        center[plane]         = planeValue;
        center[planeNext]     = 0.5f;
        center[planeNextNext] = 0.5f;

        for (int i = 0; i < 4; i++)
        {
            TetrahedralCell tet;
            int first                        = face & 1;
            int second                       = !(face & 1);
            tet.points[first][plane]         = planeValue;
            tet.points[first][planeNext]     = next[i];
            tet.points[first][planeNextNext] = nextNext[i];

            tet.points[second][plane]         = planeValue;
            tet.points[second][planeNext]     = next[(i + 1) & 3];
            tet.points[second][planeNextNext] = nextNext[(i + 1) & 3];

            tet.points[2] = center;
            tet.points[3] = cubeCenter;
            tets.push_back(tet);
        }
    }

    for (int i = 0; i < 24; i++)
    {
    }
}

VolumeData Volumes(CommandBuffer *cmd, Arena *arena)
{
    // build the octree over all volumes...
    string filename = "../../data/wdas_cloud/wdas_cloud.nvdb";
    auto handle     = nanovdb::io::readGrid(std::string((char *)filename.str, filename.size));
    auto grid       = handle.grid<float>();

    TransferBuffer vdbDataBuffer = cmd->SubmitBuffer(
        handle.buffer().data(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, handle.buffer().size());

    nanovdb::Vec3dBBox bbox = grid->worldBBox();
    Bounds rootBounds(Vec3f(bbox.min()[0], bbox.min()[1], bbox.min()[2]),
                      Vec3f(bbox.max()[0], bbox.max()[1], bbox.max()[2]));

    BuildAdaptiveTetrahedralGrid(cmd, arena);

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

    VolumeData data;
    data.octree        = newNodes;
    data.vdbDataBuffer = vdbDataBuffer;

    return data;
}
} // namespace rt
