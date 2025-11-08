#include "../../rt/shader_interop/sort_shaderinterop.h"
#include "../../rt/shader_interop/bit_twiddling_shaderinterop.h"

StructuredBuffer<uint> elementCounts : register(t0);
RWStructuredBuffer<uint> globalHistogram : register(u1);
RWStructuredBuffer<uint> partitionHistogram : register(u2);
StructuredBuffer<uint64_t> keys : register(t3);

groupshared uint localHistogram[RADIX];

struct PushConstant 
{
    uint pass;
};

[[vk::push_constant]] PushConstant pc;

[shader("compute")]
[numthreads(SORT_WORKGROUP_SIZE, 1, 1)]
void main(uint3 groupThreadID: SV_GroupThreadID, uint3 groupId: SV_GroupID) {
  uint pass = pc.pass;
  // TODO hardcoded
  uint elementCount = 1u << 20u;

  uint index = groupThreadID.x;
  uint partitionIndex = groupId.x;
  uint partitionStart = partitionIndex * PARTITION_SIZE;

  // discard all workgroup invocations
  if (partitionStart >= elementCount) {
    return;
  }

  if (index < RADIX) {
    localHistogram[index] = 0;
  }
  GroupMemoryBarrierWithGroupSync();

  // local histogram
  for (int i = 0; i < PARTITION_DIVISION; ++i) {
    uint keyIndex = partitionStart + SORT_WORKGROUP_SIZE * i + index;
    uint64_t key = keyIndex < elementCount ? keys[keyIndex] : ~0ull;
    uint radix = uint(key >> (pass * 8)) & 0xff;
    InterlockedAdd(localHistogram[radix], 1);
  }
  GroupMemoryBarrierWithGroupSync();

  if (index < RADIX) {
    // set to partition histogram
    partitionHistogram[RADIX * partitionIndex + index] = localHistogram[index];

    // add to global histogram
    InterlockedAdd(globalHistogram[RADIX * pass + index], localHistogram[index]);
  }
}

