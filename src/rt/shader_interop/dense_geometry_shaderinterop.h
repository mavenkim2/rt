#ifndef DENSE_GEOMETRY_SHADERINTEROP_H_
#define DENSE_GEOMETRY_SHADERINTEROP_H_

#include "hlsl_cpp_compat.h"

#ifdef __cplusplus
namespace rt
{
#endif

#define USE_PROCEDURAL_CLUSTER_INTERSECTION

#define ANCHOR_WIDTH              24
#define MAX_CLUSTER_TRIANGLES_BIT 7
#define MAX_CLUSTER_TRIANGLES     (1 << MAX_CLUSTER_TRIANGLES_BIT)

#define MAX_CLUSTER_VERTICES_BIT 8
#define MAX_CLUSTER_VERTICES     (1 << MAX_CLUSTER_VERTICES_BIT)

#define CLUSTER_DATA_PADDING_BYTES 4

#define CLUSTER_MIN_PRECISION -20

struct PackedDenseGeometryHeader
{
    uint a;
    uint b;
    uint c;
    uint d;
    uint e;
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
