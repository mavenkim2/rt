// MIT License
// 
// Copyright (c) 2023 Mirco Werner
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

/**
* VkRadixSort written by Mirco Werner: https://github.com/MircoWerner/VkRadixSort
* Based on implementation of Intel's Embree: https://github.com/embree/embree/blob/v4.0.0-ploc/kernels/rthwif/builder/gpu/sort.h
*/

#define WORKGROUP_SIZE 256
#define RADIX_SORT_BINS 256
#define SUBGROUP_SIZE 32

#include "../../rt/shader_interop/radix_sort_shaderinterop.h"
#include "../../rt/shader_interop/wavefront_shaderinterop.h"

[[vk::push_constant]] RadixSortPushConstant pc;

StructuredBuffer<SortKey> g_elements_in : register(t0);
RWStructuredBuffer<SortKey> g_elements_out : register(u1);
StructuredBuffer<uint> g_histograms : register(t2);

StructuredBuffer<WavefrontQueue> queues : register(t3);
StructuredBuffer<uint> indirectBuffer : register(t4);

groupshared uint sums[RADIX_SORT_BINS / SUBGROUP_SIZE];// subgroup reductions
groupshared uint global_offsets[RADIX_SORT_BINS];// global exclusive scan (prefix sum)

struct BinFlags 
{
    uint flags[WORKGROUP_SIZE / 32];
};

groupshared BinFlags bin_flags[RADIX_SORT_BINS];

[numthreads(WORKGROUP_SIZE, 1, 1)]
void main(uint3 dtID : SV_DispatchThreadID, uint3 groupID : SV_GroupID, uint groupIndex : SV_GroupIndex) 
{
    const uint g_num_blocks_per_workgroup = 1;
    const uint g_num_elements = queues[pc.queueIndex].writeOffset - queues[pc.queueIndex].readOffset;
    const uint g_shift = pc.g_shift;
    const uint g_num_workgroups = indirectBuffer[3 * pc.queueIndex];

    uint sID = groupIndex / WaveGetLaneCount();
    uint maxWavesInGroup = WORKGROUP_SIZE / WaveGetLaneCount();
    uint waveIndex = WaveGetLaneIndex();

    uint local_histogram = 0;
    uint prefix_sum = 0;
    uint histogram_count = 0;

    if (groupIndex < RADIX_SORT_BINS) 
    {
        uint count = 0;
        for (uint j = 0; j < g_num_workgroups; j++) {
            const uint t = g_histograms[RADIX_SORT_BINS * j + groupIndex];
            local_histogram = (j == groupID.x) ? count : local_histogram;
            count += t;
        }
        histogram_count = count;
        const uint sum = WaveActiveSum(histogram_count);
        prefix_sum = WavePrefixSum(histogram_count);
        if (WaveIsFirstLane())
        {
            sums[sID] = sum;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (groupIndex < RADIX_SORT_BINS) 
    {
        uint sums_prefix_sum = 0;
        if (waveIndex < maxWavesInGroup)
        {
            sums_prefix_sum = WavePrefixSum(sums[waveIndex]);
        }
        sums_prefix_sum = WaveReadLaneAt(sums_prefix_sum, sID);
        const uint global_histogram = sums_prefix_sum + prefix_sum;
        global_offsets[groupIndex] = global_histogram + local_histogram;
    }

    const uint flags_bin = groupIndex / 32;
    const uint flags_bit = 1u << (groupIndex % 32);

    for (uint index = 0; index < g_num_blocks_per_workgroup; index++) 
    {
        uint elementId = groupID.x * g_num_blocks_per_workgroup * WORKGROUP_SIZE + index * WORKGROUP_SIZE + groupIndex;

        // initialize bin flags
        if (groupIndex < RADIX_SORT_BINS) {
            for (int i = 0; i < WORKGROUP_SIZE / 32; i++) {
                bin_flags[groupIndex].flags[i] = 0U;// init all bin flags to 0
            }
        }
        GroupMemoryBarrierWithGroupSync();

        SortKey element_in = (SortKey)0;
        uint binID = 0;
        uint binOffset = 0;
        if (elementId < g_num_elements) {
            element_in = g_elements_in[elementId];
            binID = uint(element_in.key >> g_shift) & uint(RADIX_SORT_BINS - 1);
            binOffset = global_offsets[binID];
            InterlockedAdd(bin_flags[binID].flags[flags_bin], flags_bit);
        }
        GroupMemoryBarrierWithGroupSync();

        if (elementId < g_num_elements) 
        {
            // calculate output index of element
            uint prefix = 0;
            uint count = 0;
            for (uint i = 0; i < WORKGROUP_SIZE / 32; i++) {
                const uint bits = bin_flags[binID].flags[i];
                const uint full_count = countbits(bits);
                const uint partial_count = countbits(bits & (flags_bit - 1));
                prefix += (i < flags_bin) ? full_count : 0U;
                prefix += (i == flags_bin) ? partial_count : 0U;
                count += full_count;
            }
            g_elements_out[binOffset + prefix] = element_in;
            if (prefix == count - 1) {
                InterlockedAdd(global_offsets[binID], count);
            }
        }

        GroupMemoryBarrierWithGroupSync();
    }
}
