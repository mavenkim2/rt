#ifndef TRANSPOSE_H_
#define TRANSPOSE_H_

#include "math_include.h"

namespace rt
{

template <uint32_t N>
__forceinline void Transpose(const Lane4F32 lanes[N], Vec3lf<N> &out)
{
    if constexpr (N == 1) out = ToVec3f(lanes[0]);
    else if constexpr (N == 4)
        Transpose4x3(lanes[0], lanes[1], lanes[2], lanes[3], out.x, out.y, out.z);
    else if constexpr (N == 8)
        Transpose8x3(lanes[0], lanes[1], lanes[2], lanes[3], lanes[4], lanes[5], lanes[6],
                     lanes[7], out.x, out.y, out.z);
    else Assert(0);
}

} // namespace rt
#endif
