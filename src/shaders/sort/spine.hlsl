#include "../../rt/shader_interop/sort_shaderinterop.h"
#include "../../rt/shader_interop/bit_twiddling_shaderinterop.h"

StructuredBuffer<uint> elementCounts : register(t0);
RWStructuredBuffer<uint> globalHistogram : register(u1);
RWStructuredBuffer<uint> partitionHistogram : register(u2);

groupshared uint reduction;
groupshared uint intermediate[MAX_SUBGROUP_SIZE];

struct PushConstant 
{
    uint pass;
};

[[vk::push_constant]] PushConstant pc;

// dispatch this shader (RADIX, 1, 1), so that gl_WorkGroupID.x is radix
[shader("compute")]
[numthreads(SORT_WORKGROUP_SIZE, 1, 1)]
void main(uint3 groupThreadID: SV_GroupThreadID, uint3 groupId: SV_GroupID,
          uint groupIndex: SV_GroupIndex) {
  uint pass = pc.pass;
  // TODO IMPORTANT HARDCODED
  uint elementCount = 1u << 20u;//elementCounts[0];

  uint laneIndex = WaveGetLaneIndex();  // 0..31
  uint laneCount = WaveGetLaneCount();  // 32
  uint waveIndex = groupIndex / laneCount;
  uint waveCount = SORT_WORKGROUP_SIZE / laneCount;
  uint index = waveIndex * laneCount + laneIndex;

  uint radix = groupId.x;

  uint partitionCount = (elementCount + PARTITION_SIZE - 1) / PARTITION_SIZE;

  if (index == 0) {
    reduction = 0;
  }
  GroupMemoryBarrierWithGroupSync();

  for (uint i = 0; SORT_WORKGROUP_SIZE * i < partitionCount; ++i) {
    uint partitionIndex = SORT_WORKGROUP_SIZE * i + index;
    uint value =
        partitionIndex < partitionCount ? partitionHistogram[RADIX * partitionIndex + radix] : 0;
    uint excl = WavePrefixSum(value) + reduction;
    uint sum = WaveActiveSum(value);

    if (WaveIsFirstLane()) {
      intermediate[waveIndex] = sum;
    }
    GroupMemoryBarrierWithGroupSync();

    if (index < waveCount) {
      uint excl = WavePrefixSum(intermediate[index]);
      uint sum = WaveActiveSum(intermediate[index]);
      intermediate[index] = excl;

      if (index == 0) {
        reduction += sum;
      }
    }
    GroupMemoryBarrierWithGroupSync();

    if (partitionIndex < partitionCount) {
      excl += intermediate[waveIndex];
      partitionHistogram[RADIX * partitionIndex + radix] = excl;
    }
    GroupMemoryBarrierWithGroupSync();
  }

  if (radix == 0) {
    // one workgroup is responsible for global histogram prefix sum
    if (index < RADIX) {
      uint value = globalHistogram[RADIX * pass + index];
      uint excl = WavePrefixSum(value);
      uint sum = WaveActiveSum(value);

      if (WaveIsFirstLane()) {
        intermediate[waveIndex] = sum;
      }
      GroupMemoryBarrierWithGroupSync();

      if (index < RADIX / laneCount) {
        uint excl = WavePrefixSum(intermediate[index]);
        intermediate[index] = excl;
      }
      GroupMemoryBarrierWithGroupSync();

      excl += intermediate[waveIndex];
      globalHistogram[RADIX * pass + index] = excl;
    }
  }
}

