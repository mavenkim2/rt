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
#define WORKGROUP_SIZE 256 // assert WORKGROUP_SIZE >= RADIX_SORT_BINS
#define RADIX_SORT_BINS 256

#include "../../rt/shader_interop/radix_sort_shaderinterop.h"
#include "../../rt/shader_interop/wavefront_shaderinterop.h"

[[vk::push_constant]] RadixSortPushConstant pc;

StructuredBuffer<SortKey> g_elements_in : register(t0);
RWStructuredBuffer<uint> g_histograms : register(u1);

StructuredBuffer<uint> numElements : register(t2);

groupshared uint histogram[RADIX_SORT_BINS];

[numthreads(WORKGROUP_SIZE, 1, 1)]
void main(uint groupIndex : SV_GroupIndex, uint3 groupID : SV_GroupID) 
{
    const uint g_num_blocks_per_workgroup = 32;
    const uint g_num_elements = numElements[0];

    if (groupID.x * g_num_blocks_per_workgroup + groupIndex > g_num_elements) return;

    uint g_shift = pc.g_shift;

    // initialize histogram
    if (groupIndex < RADIX_SORT_BINS) 
    {
        histogram[groupIndex] = 0U;
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint index = 0; index < g_num_blocks_per_workgroup; index++) 
    {
        uint elementId = groupID.x * g_num_blocks_per_workgroup * WORKGROUP_SIZE + index * WORKGROUP_SIZE + groupIndex;
        if (elementId < g_num_elements) 
        {
            // determine the bin
            const uint bin = uint(g_elements_in[elementId].key >> g_shift) & uint(RADIX_SORT_BINS - 1);
            // increment the histogram
            InterlockedAdd(histogram[bin], 1U);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (groupIndex < RADIX_SORT_BINS) 
    {
        g_histograms[RADIX_SORT_BINS * groupID.x + groupIndex] = histogram[groupIndex];
    }
}
