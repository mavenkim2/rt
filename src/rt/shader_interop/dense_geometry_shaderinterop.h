#ifndef DENSE_GEOMETRY_SHADERINTEROP_H_
#define DENSE_GEOMETRY_SHADERINTEROP_H_

#include "hlsl_cpp_compat.h"

#ifdef __cplusplus
namespace rt
{
#endif

#define ANCHOR_WIDTH 24
#define MAX_CLUSTER_TRIANGLES_BIT 7
#define MAX_CLUSTER_TRIANGLES (1 << MAX_CLUSTER_TRIANGLES_BIT)

#ifdef __cplusplus
}
#endif

#endif
