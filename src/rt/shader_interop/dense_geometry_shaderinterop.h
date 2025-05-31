#ifndef DENSE_GEOMETRY_SHADERINTEROP_H_
#define DENSE_GEOMETRY_SHADERINTEROP_H_

#include "hlsl_cpp_compat.h"

#ifdef __cplusplus
namespace rt
{
#endif

#define LOG2_TRIANGLES_PER_LEAF 0
#define TRIANGLES_PER_LEAF      (1 << LOG2_TRIANGLES_PER_LEAF)

#define ANCHOR_WIDTH              24
#define MAX_CLUSTER_TRIANGLES_BIT 7
#define MAX_CLUSTER_TRIANGLES     (1 << MAX_CLUSTER_TRIANGLES_BIT)

#define MAX_CLUSTER_VERTICES_BIT 8
#define MAX_CLUSTER_VERTICES     (1 << MAX_CLUSTER_VERTICES_BIT)

#define CLUSTER_MIN_PRECISION -20

struct PackedDenseGeometryHeader
{
    uint z;
    uint a;
    uint b;
    uint c;
    uint d;
    uint e;
    uint f;
    uint g;
    uint h;
    uint i;
    uint j;
};

struct DGFGeometryInfo
{
    uint headerOffset;
};

struct AABB
{
    float minX;
    float minY;
    float minZ;
    float maxX;
    float maxY;
    float maxZ;
};

#ifdef __cplusplus
}
#endif

#endif
