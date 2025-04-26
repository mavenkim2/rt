#ifndef HLSL_CPP_COMPAT_H
#define HLSL_CPP_COMPAT_H

#ifdef __cplusplus
#include "../math/math_include.h"
namespace rt
{

typedef Vec3f float3;
typedef Vec4f float4;
typedef Mat4 float4x4;
typedef AffineSpace float3x4;

typedef unsigned int uint;
typedef Vec2u uint2;

typedef Vec2i int2;
typedef Vec3i int3;

} // namespace rt
#endif

#endif
