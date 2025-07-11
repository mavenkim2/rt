#ifndef BIT_TWIDDLING_SHADERINTEROP_H_
#define BIT_TWIDDLING_SHADERINTEROP_H_

#include "hlsl_cpp_compat.h"

#ifdef __cplusplus
namespace rt
{
#endif

inline uint BitFieldExtractU32(uint data, uint size, uint offset)
{
    size &= 31;
    offset &= 31;
    return (data >> offset) & ((1u << size) - 1u);
}

#ifdef __cplusplus
inline u32 BitFieldPackU32(u32 val, u32 data, u32 &offset, u32 size)
{
    Assert(size == 32 || data < (1u << size));
    u32 o = offset & 31u;
    data  = data & ((1u << size) - 1u);
    val |= data << o;
    offset += size;
    return val;
}
#else
inline uint BitFieldPackU32(uint val, uint data, inout uint offset, uint size)
{
    uint o = offset & 31u;
    data   = data & ((1u << size) - 1u);
    val |= data << o;
    offset += size;
    return val;
}
#endif

#ifdef __cplusplus
}
#endif

#endif
