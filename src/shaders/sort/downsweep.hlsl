#include "../../rt/shader_interop/sort_shaderinterop.h"
#include "../../rt/shader_interop/bit_twiddling_shaderinterop.h"

#define KEY_VALUE

StructuredBuffer<uint> elementCounts : register(t0);
StructuredBuffer<uint> globalHistogram : register(t1);
StructuredBuffer<uint> partitionHistogram : register(t2);
StructuredBuffer<uint64_t> keysIn : register(t3);
RWStructuredBuffer<uint64_t> keysOut : register(u4);
#ifdef KEY_VALUE
StructuredBuffer<uint> valuesIn : register(t5);
RWStructuredBuffer<uint> valuesOut : register(u6);
#endif  // KEY_VALUE

groupshared uint64_t localHistogram[PARTITION_SIZE];  // (R, S=16)=4096, (P) for alias. take maximum.
groupshared uint localHistogramSum[RADIX];

// returns 0b00000....11111, where msb is id-1.
uint4 GetExclusiveWaveMask(uint id) {
  // clamp bit-shift right operand between 0..31 to avoid undefined behavior.
  uint shift = (1u << BitFieldExtractU32(id, 5, 0)) - 1;  //  (1 << (id % 32)) - 1
  // right shift operation on signed integer copies sign bit, use the trick for masking.
  // (negative)     >> 31 = 111...111
  // (non-negative) >> 31 = 000...000
  int x = int(id) >> 5;
  return uint4((shift & ((-1 - x) >> 31)) | ((0 - x) >> 31),  //
               (shift & ((0 - x) >> 31)) | ((1 - x) >> 31),   //
               (shift & ((1 - x) >> 31)) | ((2 - x) >> 31),   //
               (shift & ((2 - x) >> 31)) | ((3 - x) >> 31)   //
  );
}

uint GetBitCount(uint4 value) {
  uint4 result = countbits(value);
  return result[0] + result[1] + result[2] + result[3];
}

struct PushConstant 
{
    uint pass;
};

[[vk::push_constant]] PushConstant pc;

[numthreads(SORT_WORKGROUP_SIZE, 1, 1)]
void main(uint3 groupThreadID: SV_GroupThreadID, uint3 groupId: SV_GroupID,
          uint groupIndex: SV_GroupIndex) {
// TODO IMPORTANT
  uint elementCount = 1u << 20u;//elementCounts[0];
  uint pass = pc.pass;

  uint laneIndex = WaveGetLaneIndex();          // 0..31 or 0..63
  uint laneCount = WaveGetLaneCount();          // 32 or 64
  uint waveIndex = groupIndex / laneCount;      // 0..15 or 0..7
  uint waveCount = SORT_WORKGROUP_SIZE / laneCount;  // 32 or 16
  uint index = waveIndex * laneCount + laneIndex;

  uint4 waveMask = GetExclusiveWaveMask(laneIndex);

  uint partitionIndex = groupId.x;
  uint partitionStart = partitionIndex * PARTITION_SIZE;

  if (partitionStart >= elementCount)
    return;

  if (index < RADIX) {
    for (int i = 0; i < waveCount; ++i) {
      localHistogram[waveCount * index + i] = 0;
    }
  }
  GroupMemoryBarrierWithGroupSync();

  // load from global memory, local histogram and offset
  uint64_t localKeys[PARTITION_DIVISION];
  uint localRadix[PARTITION_DIVISION];
  uint localOffsets[PARTITION_DIVISION];
  uint waveHistogram[PARTITION_DIVISION];
#ifdef KEY_VALUE
  uint localValues[PARTITION_DIVISION];
#endif  // KEY_VALUE

  [unroll]
  for (int i = 0; i < PARTITION_DIVISION; ++i) {
    uint keyIndex =
        partitionStart + (PARTITION_DIVISION * laneCount) * waveIndex + i * laneCount + laneIndex;
    uint64_t key = keyIndex < elementCount ? keysIn[keyIndex] : ~0ull;
    localKeys[i] = key;

#ifdef KEY_VALUE
    localValues[i] = keyIndex < elementCount ? valuesIn[keyIndex] : 0;
#endif  // KEY_VALUE

    uint radix = uint(key >> (pass * 8)) & 0xff;
    localRadix[i] = radix;

    // mask per digit
    uint4 mask = WaveActiveBallot(true);
    [unroll]
    for (int j = 0; j < 8; ++j) {
      uint digit = (radix >> j) & 1;
      uint4 ballot = WaveActiveBallot(digit == 1);
      // digit - 1 is 0 or 0xffffffff. xor to flip.
      mask &= uint4(digit - 1, digit - 1, digit - 1, digit - 1) ^ ballot;
    }

    // wave level offset for radix
    uint waveOffset = GetBitCount(waveMask & mask);
    uint radixCount = GetBitCount(mask);

    // elect a representative per radix, add to histogram
    if (waveOffset == 0) {
      // accumulate to local histogram
      InterlockedAdd(localHistogram[waveCount * radix + waveIndex], radixCount);
      waveHistogram[i] = radixCount;
    } else {
      waveHistogram[i] = 0;
    }

    localOffsets[i] = waveOffset;
  }
  GroupMemoryBarrierWithGroupSync();

  // local histogram reduce 4096 or 2048
  for (uint i = index; i < RADIX * waveCount; i += SORT_WORKGROUP_SIZE) {
    uint v = uint(localHistogram[i]);
    uint sum = WaveActiveSum(v);
    uint excl = WavePrefixSum(v);
    localHistogram[i] = excl;
    if (laneIndex == 0) {
      localHistogramSum[i / laneCount] = sum;
    }
  }
  GroupMemoryBarrierWithGroupSync();

  // local histogram reduce 128 or 32
  uint intermediateOffset0 = RADIX * waveCount / laneCount;
  if (index < intermediateOffset0) {
    uint v = localHistogramSum[index];
    uint sum = WaveActiveSum(v);
    uint excl = WavePrefixSum(v);
    localHistogramSum[index] = excl;
    if (laneIndex == 0) {
      localHistogramSum[intermediateOffset0 + index / laneCount] = sum;
    }
  }
  GroupMemoryBarrierWithGroupSync();

  // local histogram reduce 4 or 1
  uint intermediateSize1 = max(RADIX * waveCount / laneCount / laneCount, 1);
  if (index < intermediateSize1) {
    uint v = localHistogramSum[intermediateOffset0 + index];
    uint excl = WavePrefixSum(v);
    localHistogramSum[intermediateOffset0 + index] = excl;
  }
  GroupMemoryBarrierWithGroupSync();

  // local histogram add 128
  if (index < intermediateOffset0) {
    localHistogramSum[index] += localHistogramSum[intermediateOffset0 + index / laneCount];
  }
  GroupMemoryBarrierWithGroupSync();

  // local histogram add 4096
  for (uint i = index; i < RADIX * waveCount; i += SORT_WORKGROUP_SIZE) {
    localHistogram[i] += localHistogramSum[i / laneCount];
  }
  GroupMemoryBarrierWithGroupSync();

  // post-scan stage
  [unroll]
  for (int i = 0; i < PARTITION_DIVISION; ++i) {
    uint radix = localRadix[i];
    localOffsets[i] += uint(localHistogram[waveCount * radix + waveIndex]);

    GroupMemoryBarrierWithGroupSync();
    if (waveHistogram[i] > 0) {
      InterlockedAdd(localHistogram[waveCount * radix + waveIndex], waveHistogram[i]);
    }
    GroupMemoryBarrierWithGroupSync();
  }

  // after atomicAdd, localHistogram contains inclusive sum
  if (index < RADIX) {
    uint v = index == 0 ? 0 : uint(localHistogram[waveCount * index - 1]);
    localHistogramSum[index] = globalHistogram[RADIX * pass + index] +
                               partitionHistogram[RADIX * partitionIndex + index] - v;
  }
  GroupMemoryBarrierWithGroupSync();

  // rearrange keys. grouping keys together makes dstOffset to be almost sequential, grants huge
  // speed boost. now localHistogram is unused, so alias memory.

  [unroll]
  for (int i = 0; i < PARTITION_DIVISION; ++i) {
    localHistogram[localOffsets[i]] = localKeys[i];
  }
  GroupMemoryBarrierWithGroupSync();

  // binning
  for (uint i = index; i < PARTITION_SIZE; i += SORT_WORKGROUP_SIZE) {
    uint64_t key = localHistogram[i];
    uint radix = uint(key >> (pass * 8)) & 0xff;
    uint dstOffset = localHistogramSum[radix] + i;
    if (dstOffset < elementCount) {
      keysOut[dstOffset] = key;
    }

#ifdef KEY_VALUE
    localKeys[i / SORT_WORKGROUP_SIZE] = dstOffset;
#endif  // KEY_VALUE
  }

#ifdef KEY_VALUE
  GroupMemoryBarrierWithGroupSync();

  [unroll]
  for (int i = 0; i < PARTITION_DIVISION; ++i) {
    localHistogram[localOffsets[i]] = localValues[i];
  }
  GroupMemoryBarrierWithGroupSync();

  for (uint i = index; i < PARTITION_SIZE; i += SORT_WORKGROUP_SIZE) {
    uint value = uint(localHistogram[i]);
    uint dstOffset = uint(localKeys[i / SORT_WORKGROUP_SIZE]);
    if (dstOffset < elementCount) {
      valuesOut[dstOffset] = value;
    }
  }
#endif  // KEY_VALUE
}

