#ifndef BIT_PACKING_H_
#define BIT_PACKING_H_

#include "base.h"
#include "math/math_include.h"
#include "shader_interop/bit_twiddling_shaderinterop.h"

namespace rt
{
struct Arena;

struct BitVector
{
    u32 *bits;
    u32 maxNumBits;
    BitVector(Arena *arena, u32 maxNumBits);
    void SetBit(u32 bit);
    void UnsetBit(u32 bit);
    bool GetBit(u32 bit);
    void WriteBits(u32 &position, u32 value, u32 numBits);
};

inline u32 BitFieldPackI32(u32 bits, i32 data, u32 &offset, u32 size)
{
    u32 signBit = (data & 0x80000000) >> 31;
    return BitFieldPackU32(bits, data | (signBit << size), offset, size);
}

inline u32 BitFieldPackU32(Vec4u &bits, u32 data, u32 offset, u32 size)
{
    u32 d = data;
    Assert(offset < 128);
    u32 o     = offset & 31u;
    u32 index = o >> 5u;
    bits[index] |= data << o;
    if (o + size > 32u)
    {
        Assert(index + 1 < 4);
        bits[index + 1] |= (data >> (32u - o));
    }
    return offset + size;
}

inline u32 BitFieldPackI32(Vec4u &bits, i32 data, u32 offset, u32 size)
{
    u32 signBit = (data & 0x80000000) >> 31;
    return BitFieldPackU32(bits, data | (signBit << size), offset, size);
}

inline u32 BitAlignU32(u32 high, u32 low, u32 shift)
{
    shift &= 31u;

    u32 result = low >> shift;
    result |= shift > 0u ? (high << (32u - shift)) : 0u;
    return result;
}

inline void WriteBits(u32 *data, u32 &position, u32 value, u32 numBits)
{
    if (numBits == 0) return;
    Assert(numBits <= 32);
    u32 dwordIndex = position >> 5;
    u32 bitIndex   = position & 31;

    Assert(numBits == 32 || ((value & ((1u << numBits) - 1)) == value));

    data[dwordIndex] |= value << bitIndex;
    if (bitIndex + numBits > 32)
    {
        data[dwordIndex + 1] |= value >> (32 - bitIndex);
    }

    position += numBits;
}

} // namespace rt

#endif
