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
        scratch.temp, 0, count, groupSize, [&](Output &output, u32 jobID, u32 start, u32 count) {
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

    tEntry = Max(tMin.x, Max(tMin.y, Max(tMin.z, 0.f)));
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
    Vec3f invDir;

    Vec3f boundsMin;
    Vec3f boundsMax;

    Vec3f octreeBoundsMin;
    Vec3f octreeBoundsMax;

    u32 raySignMask;
    u32 crossingAxis;

    int prev;
    int current;

    VolumeIterator(const Vec3f &mins, const Vec3f &maxs, const Vec3f &o, const Vec3f &d,
                   GPUOctreeNode *nodes)
        : boundsMin(mins), boundsMax(maxs), nodes(nodes), current(0),
          prev(-1), octreeBoundsMin(0), octreeBoundsMax(1)
    {
        Vec3f diag = boundsMax - boundsMin;
        rayO = (o - boundsMin) / diag;
        rayDir = d / diag;

        invDir        = 1.f / rayDir;

        Vec3f tIntersectMin = (octreeBoundsMin - rayO) * invDir;
        Vec3f tIntersectMax = (octreeBoundsMax - rayO) * invDir;

        Vec3f tMin = Min(tIntersectMin, tIntersectMax);
        Vec3f tMax_ = Max(tIntersectMin, tIntersectMax);

        float tEntry = Max(tMin.x, Max(tMin.y, Max(tMin.z, 0.f)));
        float tLeave = Min(tMax_.x, Min(tMax_.y, tMax_.z));

        bool intersects = tEntry < tLeave;
        current = intersects ? 0 : -1;
        currentT = Min(tEntry, tLeave);

        Print("t start: %f %f\n", currentT, tLeave);
        Assert(intersects);

        // Traverse to child
        raySignMask = (rayDir.x < 0.f ? 1 : 0) | 
                      (rayDir.y < 0.f ? 2 : 0) | 
                      (rayDir.z < 0.f ? 4 : 0);
        TraverseToChild();

        tIntersectMin = (octreeBoundsMin - rayO) * invDir;
        tIntersectMax = (octreeBoundsMax - rayO) * invDir;

        tMin = Min(tIntersectMin, tIntersectMax);
        tMax_ = Max(tIntersectMin, tIntersectMax);

        tEntry = Max(tMin.x, Max(tMin.y, Max(tMin.z, 0.f)));
        tLeave = Min(tMax_.x, Min(tMax_.y, tMax_.z));

        intersects = tEntry < tLeave;
        Assert(intersects);

        crossingAxis = tMax_.x == tLeave ? 0 : (tMax_.y == tLeave ? 1 : 2);
        tMax = tLeave;
        currentT = Min(currentT, Min(tEntry, tLeave));
    }

#ifdef __SLANG__
    [mutating]
#endif

    u32 CalculateAxisMask() const
    {
        return (current - 1) & 0x7;
    }
    bool TestNext()
    {
        u32 rayCode = (~raySignMask) & 0x7;
        u32 axisMask = CalculateAxisMask();

        while (current != -1 && ((rayCode ^ axisMask) & (1 << crossingAxis)) == 0)
        {
            BoundsToParent();

            current = current == 0 ? -1 : nodes[current].parentIndex;
            axisMask = CalculateAxisMask();
        }

        if (current == -1) return false;

        // Traverse to the neighbor
        BoundsToParent();
        current -= axisMask;
        axisMask ^= (1 << crossingAxis);
        current += axisMask;
        u32 closestChild = BoundsToChild();
        Assert(closestChild == axisMask);

        // Traverse to child
        TraverseToChild();

        Vec3f tIntersectMin = (octreeBoundsMin - rayO) * invDir;
        Vec3f tIntersectMax = (octreeBoundsMax - rayO) * invDir;

        Vec3f tMin = Min(tIntersectMin, tIntersectMax);
        Vec3f tMax_ = Max(tIntersectMin, tIntersectMax);

        float tEntry = Max(tMin.x, Max(tMin.y, Max(tMin.z, 0.f)));
        float tLeave = Min(tMax_.x, Min(tMax_.y, tMax_.z));

        Assert(tEntry <= tLeave);

        // Prepare next traversal
        crossingAxis = tMax_.x == tLeave ? 0 : (tMax_.y == tLeave ? 1 : 2);
        tMax = Max(currentT, Max(tEntry, tLeave));
        // currentT = Min(currentT, Min(tEntry, tLeave));
        return true;
    }

    void BoundsToParent()
    {
        u32 axisMask = CalculateAxisMask();
        Vec3f extent = (octreeBoundsMax - octreeBoundsMin) / 2.f;
        octreeBoundsMin = Vec3f((axisMask & 0x1) ? octreeBoundsMin.x - 2.f * extent.x : octreeBoundsMin.x,
                (axisMask & 0x2) ? octreeBoundsMin.y - 2.f * extent.y : octreeBoundsMin.y,
                (axisMask & 0x4) ? octreeBoundsMin.z - 2.f * extent.z : octreeBoundsMin.z);
        octreeBoundsMax = Vec3f((axisMask & 0x1) ? octreeBoundsMax.x : octreeBoundsMax.x + 2.f * extent.x,
                (axisMask & 0x2) ? octreeBoundsMax.y : octreeBoundsMax.y + 2.f * extent.y,
                (axisMask & 0x4) ? octreeBoundsMax.z : octreeBoundsMax.z + 2.f * extent.z);
    }

    u32 BoundsToChild()
    {
        Vec3f center = (octreeBoundsMax + octreeBoundsMin) / 2.f;
        Vec3f tPlanes = (center - rayO) * invDir;
        u32 closestChild = (tPlanes.x <= currentT) | ((tPlanes.y <= currentT) << 1) | ((tPlanes.z <= currentT) << 2);
        closestChild ^= raySignMask;
        octreeBoundsMin = Vec3f((closestChild & 0x1) ? center.x : octreeBoundsMin.x,
                (closestChild & 0x2) ? center.y : octreeBoundsMin.y,
                (closestChild & 0x4) ? center.z : octreeBoundsMin.z);
        octreeBoundsMax = Vec3f((closestChild & 0x1) ? octreeBoundsMax.x : center.x,
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

    bool Next()
    {
        u32 code = (rayDir.x >= 0.f) | ((rayDir.y >= 0.f) << 1) | ((rayDir.z >= 0.f) << 2);
        if (current == -1) return false;
        for (;;)
        {
            Vec3f currentPos = rayO + currentT * rayDir;
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
                    break;
                }
                int axisMask = (current - 1) & 0x7;
                next         = nodes[current].parentIndex;

                boundsMin = Vec3f((axisMask & 0x1) ? boundsMin.x - 2.f * extent.x : boundsMin.x,
                                   (axisMask & 0x2) ? boundsMin.y - 2.f * extent.y : boundsMin.y,
                                   (axisMask & 0x4) ? boundsMin.z - 2.f * extent.z : boundsMin.z);
                boundsMax = Vec3f((axisMask & 0x1) ? boundsMax.x : boundsMax.x + 2.f * extent.x,
                                   (axisMask & 0x2) ? boundsMax.y : boundsMax.y + 2.f * extent.y,
                                   (axisMask & 0x4) ? boundsMax.z : boundsMax.z + 2.f * extent.z);
                currentT += 0.0001f;
                // currentT += gamma(2) * Max(rayDir
                // for (;;)
                // {
                //     Vec3f currentPos = rayO + currentT * rayDir - center;
                //     currentT = NextFloatUp(currentT);
                //     if (currentPos.x > extent.x || currentPos.y > extent.y ||
                //         currentPos.z > extent.z || currentPos.x < -extent.x ||
                //         currentPos.y < -extent.y || currentPos.z < -extent.z)
                //     {
                //         break;
                //     }
                // }
            }
            // go to child
            else 
            {
                u32 closestChild = (currentPos.x >= 0.f) | ((currentPos.y >= 0.f) << 1) | ((currentPos.z >= 0.f) << 2);
                next = childIndex + closestChild;

                boundsMin = Vec3f((closestChild & 0x1) ? center.x : boundsMin.x,
                                   (closestChild & 0x2) ? center.y : boundsMin.y,
                                   (closestChild & 0x4) ? center.z : boundsMin.z);
                boundsMax = Vec3f((closestChild & 0x1) ? boundsMax.x : center.x,
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

                currentT = newT;
                tMax     = tLeave;

                return true;
            }
        }
        return false;
    }

    float GetCurrentT() { return currentT; }
    float GetTMax() { return tMax; }

    void GetSegmentProperties(float& tMin, float& tFar, float& minor, float& major)
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

VolumeData Volumes(CommandBuffer *cmd, Arena *arena)
{
    // build the octree over all volumes...
    string filename = "../../data/wdas_cloud/wdas_cloud.nvdb";
    auto handle     = nanovdb::io::readGrid(std::string((char *)filename.str, filename.size));
    auto grid       = handle.grid<float>();

    TransferBuffer vdbDataBuffer = 
        cmd->SubmitBuffer(handle.buffer().data(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, handle.buffer().size());

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

#if 1
    Vec3f pos(648.064, -82.473, -63.856);
    Vec3f dir = Normalize(ToVec3f(rootBounds.Centroid()) - pos);
    RNG rng;
    int i = 0;
    for (; i < 128; i++)
    {
        VolumeIterator iterator(ToVec3f(rootBounds.minP), ToVec3f(rootBounds.maxP), pos, dir, 
                                newNodes.data);
        bool done = false;

        do 
        {
            float tMin, tMax, minorant, majorant;
            iterator.GetSegmentProperties(tMin, tMax, minorant, majorant);

            for (;;)
            {
                float u = rng.Uniform<f32>();
                float tStep = majorant == 0.f ? tMax - tMin : SampleExponential(u, majorant);
                u = rng.Uniform<f32>();

                float t = iterator.GetCurrentT();
                // TODO: if majorant is 0, could this be false due to floating point precision?
                if (t + tStep >= tMax)
                {
                    float deltaT = tMax - t;
                    iterator.Step(deltaT);
                    //throughput *= exp(-deltaT * majorant);
                    break;
                }
                else 
                {
                    iterator.Step(tStep);

                    pnanovdb_grid_handle_t gridHandle = {0};

                    // TODO: hardcoded
                    pnanovdb_buf_t buf = pnanovdb_make_buf((u32 *)handle.buffer().data(), handle.buffer().size());
                    pnanovdb_tree_handle_t tree = pnanovdb_grid_get_tree(buf, gridHandle);
                    pnanovdb_root_handle_t root = pnanovdb_tree_get_root(buf, tree);
                    pnanovdb_uint32_t gridType = pnanovdb_grid_get_grid_type(buf, gridHandle);

                    pnanovdb_readaccessor_t accessor;
                    pnanovdb_readaccessor_init(&accessor, root);

                    Vec3f gridP = pos + iterator.GetCurrentT() * dir;
                    pnanovdb_vec3_t gridPos = {gridP.x, gridP.y, gridP.z};

                    nanovdb::Vec3d i0 =
                        grid->worldToIndexF(nanovdb::Vec3d(gridP.x, gridP.y, gridP.z));

                    pnanovdb_vec3_t indexSpacePosition = pnanovdb_grid_world_to_indexf(buf, gridHandle, &gridPos);
                    pnanovdb_coord_t coord = pnanovdb_hdda_pos_to_ijk(&indexSpacePosition);

                    // Clamp to valid range, it can't get outside the bounding box
                    //coord = clamp(coord, bboxMin, bboxMax);

                    // Get the address of the value at the coordinate
                    pnanovdb_address_t valueAddr = pnanovdb_readaccessor_get_value_address(gridType, buf, &accessor, &coord);
                    float density = pnanovdb_read_float(buf, valueAddr);

                    //throughput *= exp(-tStep * majorant);

                    // scatter
                    Assert(density <= majorant);
                    if (u < density / majorant)
                    {
                        done = true;
                        break;
                    }
                }
            }
        } while (!done && iterator.TestNext());

        if (iterator.current == -1)
        {
            int stop = 5;
            break;
        }

        float t = iterator.GetCurrentT();
        pos += t * dir;

        Vec2f u(rng.Uniform<f32>(), rng.Uniform<f32>());
        // TODO hardcoded
        float g = .877;
        dir = SampleHenyeyGreenstein(-dir, g, u);

        //radiance += beta * float3(0.03, 0.07, 0.23);
        //LightSource "distant"
        //"point3 to" [-0.5826 -0.7660 -0.2717]
        //"rgb L" [2.6 2.5 2.3]
    }
#endif

    VolumeData data;
    data.octree = newNodes;
    data.vdbDataBuffer = vdbDataBuffer;

    return data;
}
} // namespace rt
