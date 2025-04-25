#include "memory.h"
#include "bit_packing.h"

namespace rt
{
BitVector::BitVector(Arena *arena, u32 maxNumBits) : maxNumBits(maxNumBits)
{
    bits = PushArray(arena, u32, (maxNumBits + 31) >> 5);
}
void BitVector::SetBit(u32 bit)
{
    Assert(bit < maxNumBits);
    bits[bit >> 5] |= 1ull << (bit & 31);
}

void BitVector::UnsetBit(u32 bit)
{
    Assert(bit < maxNumBits);
    bits[bit >> 5] &= ~(1ull << (bit & 31));
}

bool BitVector::GetBit(u32 bit)
{
    Assert(bit < maxNumBits);
    return bits[bit >> 5] & (1 << (bit & 31));
}
} // namespace rt
