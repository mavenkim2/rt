#include "containers.h"
#include "math/math_include.h"
#include "math/bounds.h"
#include "string.h"
#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/io/IO.h>
#include <nanovdb/GridHandle.h>

namespace rt
{

struct OctreeNode
{
    float minValue;
    float maxValue;
};

void Volumes()
{
    // build the octree over all volumes...
    string filename = "../../data/wdas_cloud/wdas_cloud.nvdb";
    auto handle     = nanovdb::io::readGrid(std::string((char *)filename.str, filename.size));
    auto grid       = handle.grid<float>();
    auto accessor   = grid->getAccessor();

    nanovdb::Vec3dBBox bbox = grid->worldBBox();
    Bounds rootBounds(Vec3f(bbox.min()[0], bbox.min()[1], bbox.min()[2]),
                      Vec3f(bbox.max()[0], bbox.max()[1], bbox.max()[2]));
    FixedArray<Bounds, 128> stack;
    stack.Push(rootBounds);

    struct StackEntry 
    {
        OctreeNode node;
    };

    std::vector<OctreeNode> nodes;
    nodes.reserve(64);

    const float T = Abs(1.f / std::log(0.5f));

    while (stack.Length())
    {
        Bounds bounds = stack.Pop();

        nanovdb::Vec3d i0 = grid->worldToIndexF(
            nanovdb::Vec3d(bounds.minP[0], bounds.minP[1], bounds.minP[2]));
        nanovdb::Vec3d i1 = grid->worldToIndexF(
            nanovdb::Vec3d(bounds.maxP[0], bounds.maxP[1], bounds.maxP[2]));

        float maxValue = 0.f;
        float minValue = pos_inf;
        for (int x = (int)i0[0]; x <= (int)i1[0]; x++)
        {
            for (int y = -100; y <= 100; y++)
            {
                for (int z = -100; z <= 100; z++)
                {
                    float value = accessor.getValue({x, y, z});
                    maxValue    = Max(maxValue, value);
                    minValue    = Min(minValue, value);
                }
            }
        }

        f32 diag = Length(ToVec3f(bounds.Diagonal()));

        if ((maxValue - minValue) * diag > T)
        {
            Vec3f minP   = ToVec3f(bounds.minP);
            Vec3f maxP   = ToVec3f(bounds.maxP);
            Vec3f center = ToVec3f(bounds.Centroid());
            for (int z = 0; z < 2; z++)
            {
                for (int y = 0; y < 2; y++)
                {
                    for (int x = 0; x < 2; x++)
                    {
                        Vec3f newMinP((x & 1) ? center.x : minP.x, (y & 1) ? center.y : minP.y,
                                      (z & 1) ? center.z : minP.z);
                        Vec3f newMaxP((x & 1) ? maxP.x : center.x, (y & 1) ? maxP.y : center.y,
                                      (z & 1) ? maxP.z : center.z);
                        Bounds newBounds(newMinP, newMaxP);

                        stack.Push(newBounds);
                    }
                }
            }
        }
    }
}
} // namespace rt
