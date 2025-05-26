#ifndef RADIX_SORT_H_
#define RADIX_SORT_H_

#include "base.h"
#include "memory.h"
#include "thread_context.h"

namespace rt
{

template <typename Handle, bool ascending = true>
void SortHandles(Handle *shadingHandles, u32 count)
{
    TempArena temp    = ScratchStart(0, 0);
    size_t handleSize = sizeof(shadingHandles[0].sortKey);

    Handle *keys0 = (Handle *)shadingHandles;
    Handle *keys1 = PushArrayNoZero(temp.arena, Handle, count);

    // Radix sort
    for (int iter = 0; iter < (int)handleSize; iter++)
    {
        u32 shift = iter * 8;
        Assert(shift < 64);
        u32 buckets[256] = {};
        // Calculate # in each radix
        for (u32 i = 0; i < count; i++)
        {
            const Handle &key = keys0[i];
            u32 bucket        = (key.sortKey >> shift) & 0xff;
            Assert(bucket < 256);
            buckets[bucket]++;
        }
        // Prefix sum

        if constexpr (ascending)
        {
            u32 total = 0;
            for (u32 i = 0; i < 256; i++)
            {
                u32 bucketCount = buckets[i];
                buckets[i]      = total;
                total += bucketCount;
            }
        }
        else
        {
            u32 total = 0;
            for (int i = 255; i >= 0; i--)
            {
                u32 bucketCount = buckets[i];
                buckets[i]      = total;
                total += bucketCount;
            }
        }

        // Place in correct position
        for (u32 i = 0; i < count; i++)
        {
            const Handle &key = keys0[i];
            u32 bucket        = (key.sortKey >> shift) & 0xff;
            u32 index         = buckets[bucket]++;
            Assert(index < count);
            keys1[index] = key;
        }
        Swap(keys0, keys1);
    }

    if (sizeof(shadingHandles[0].sortKey) & 1)
    {
        MemoryCopy(keys1, keys0, sizeof(Handle) * count);
    }

    ScratchEnd(temp);
}

} // namespace rt

#endif
