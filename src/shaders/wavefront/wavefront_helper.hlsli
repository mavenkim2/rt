#ifndef WAVEFRONT_HELPER_HLSLI
#define WAVEFRONT_HELPER_HLSLI

#include "../common.hlsli"

struct WavefrontQueue 
{
    uint readOffset;
    uint writeOffset;
};

void StoreFloat3(float3 f, uint descriptorIndex, uint index)
{
    bindlessRWFloat3s[descriptorIndex][index] = f;
}

float3 GetFloat3(uint descriptorIndex, uint index)
{
    return bindlessFloat3s[descriptorIndex][index];
}

void StoreUint(uint u, uint descriptorIndex, uint index)
{
    bindlessRWUints[descriptorIndex][index] = u;
}

uint GetUint(uint descriptorIndex, uint index)
{
    return bindlessUints[descriptorIndex][index];
}

void StoreFloat2(float2 f, uint descriptorIndex, uint index)
{
    bindlessRWFloat2s[descriptorIndex][index] = f;
}

float2 GetFloat2(uint descriptorIndex, uint index)
{
    return bindlessFloat2s[descriptorIndex][index];
}

void StoreFloat(float f, uint descriptorIndex, uint index)
{
    bindlessRWFloats[descriptorIndex][index] = f;
}

float GetFloat(uint descriptorIndex, uint index)
{
    return bindlessFloats[descriptorIndex][index];
}

#endif
