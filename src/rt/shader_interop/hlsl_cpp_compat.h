#ifndef HLSL_CPP_COMPAT_H
#define HLSL_CPP_COMPAT_H

#ifdef __cplusplus
namespace rt
{
#endif

typedef Mat4 float4x4;
typedef Vec3f float3;
typedef AffineSpace float3x4;
typedef Vec4f float4;
typedef unsigned int uint;

#ifdef __cplusplus
}
#endif

#endif
