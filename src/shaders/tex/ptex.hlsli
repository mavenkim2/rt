#ifndef PTEX_HLSLI_
#define PTEX_HLSLI_

#include "../common.hlsli"
#include "../bit_twiddling.hlsli"
#include "virtual_textures.hlsli"
#include "../../rt/shader_interop/hit_shaderinterop.h"

const float2x3 uvRotations[16];

namespace Ptex
{
    struct FaceData 
    {
        uint2 faceOffset;
        uint2 log2Dim;
        int rotateIndex;
        bool rotate;
    };

    int GetNeighborIndexFromUV(float2 texCoord, float2 faceDim, inout float2 uv)
    {
        int neighborIndex = -1;

        // TODO: how do I handle corners and different resolution faces?
        if (texCoord.x >= faceDim.x)
        {
            // Bottom right corner
            if (texCoord.y >= faceDim.y)
            {
                uv = (faceDim - 0.5) / faceDim;
            }
            // Top right corner
            else if (texCoord.y < 0)
            {
                uv = float2(faceDim.x - 0.5, 0.5) / faceDim;
            }
            else 
            {
                neighborIndex = 1;
            }
        }
        else if (texCoord.x < 0)
        {
            // Bottom left corner
            if (texCoord.y >= faceDim.y)
            {
                uv = float2(0.5, faceDim.y - 0.5) / faceDim;
            }
            // Top left corner
            else if (texCoord.y < 0)
            {
                uv = float2(0.5, 0.5) / faceDim;
            }
            else 
            {
                neighborIndex = 3;
            }
        }
        else if (texCoord.y >= faceDim.y)
        {
            neighborIndex = 2;
        }
        else if (texCoord.y < 0)
        {
            neighborIndex = 0;
        }
        return neighborIndex;
    }

    FaceData GetFaceData(GPUMaterial material, uint faceID, int neighborIndex = -1)
    {
        // Load face data
        ByteAddressBuffer faceDataBuffer = bindlessBuffer[material.bindlessFaceDataIndex];
        uint faceDataStride = 5 * 2 * (material.numVirtualOffsetBits + material.numFaceDimBits) + 13;
        uint neighborDataStride = 2 * (material.numVirtualOffsetBits + material.numFaceDimBits) + 3;
        uint bufferOffset = faceDataStride * faceID;

        bufferOffset += neighborIndex == -1 ? 0 : neighborDataStride * neighborIndex;

        BitStreamReader bitStreamReader = CreateBitStreamReader(faceDataBuffer, 0, bufferOffset, neighborDataStride);

        FaceData result;
        result.faceOffset.x = bitStreamReader.Read<32>(material.numVirtualOffsetBits);
        result.faceOffset.y = bitStreamReader.Read<32>(material.numVirtualOffsetBits);
        result.log2Dim.x = material.minLog2Dim + bitStreamReader.Read<4>(material.numFaceDimBits);
        result.faceOffset.y = material.minLog2Dim + bitStreamReader.Read<4>(material.numFaceDimBits);
        result.rotateIndex = neighborIndex = -1 ? -1 : bitStreamReader.Read<2>(2);
        result.rotate = (bool)bitStreamReader.Read<1>(1);

        return result;
    }
}

//struct FaceData 
//{
//    uint2 faceOffsets[5];
//    uint neighborFaceSize;
//    // Bottom 8 bits is 4 bits each for log2 dimensions, then 2 bits for each neighbor for 
//    // uv rotation
//    uint faceSize_rotate;
//
//    uint2 GetFaceSize()
//    {
//        uint width = 1u << BitFieldExtractU32(faceSize_rotate, 4, 0);
//        uint height = 1u << BitFieldExtractU32(faceSize_rotate, 4, 4);
//        uint2 faceDim = uint2(width, height);
//        return faceDim;
//    }
//
//    uint2 GetNeighborFaceSize(uint neighbor)
//    {
//        uint width = 1u << BitFieldExtractU32(neighborFaceSize, 4, 8 * neighbor);
//        uint height = 1u << BitFieldExtractU32(neighborFaceSize, 4, 8 * neighbor + 4);
//        uint2 faceDim = uint2(width, height);
//        return faceDim;
//    }
//};

// https://research.nvidia.com/labs/rtr/publication/pharr2024stochtex/stochtex.pdf
float4 SampleStochasticCatmullRomBorderless(Texture2DArray tex, Ptex::FaceData faceData, GPUMaterial material, uint faceID, float2 uv, uint mipLevel, inout float u) 
{
    float2 texSize = float2(1u << faceData.log2Dim.x, 1u << faceData.log2Dim.y);
    float2 w[4];
    float2 texPos[4];

    float2 samplePos = uv * texSize;
    texPos[1] = floor(samplePos - 0.5f) + 0.5f;

    float2 f = samplePos - texPos[1];

    w[0] = f * (-0.5f + f * (1.0f - 0.5f * f));
    w[1] = 1.0f + f * f * (-2.5f + 1.5f * f);
    w[2] = f * (0.5f + f * (2.0f - 1.5f * f));
    w[3] = f * f * (-0.5f + 0.5f * f);

    texPos[0] = texPos[1] - 1;
    texPos[2] = texPos[1] + 1;
    texPos[3] = texPos[1] + 2;

    for (int i = 0; i < 4; i++)
    {
        texPos[i] /= texSize;
    }

    // Weighted reservoir sampling
    int2 reservoir[2] = {int2(0, 0), int2(0, 0)};
    float weightsSum[2] = {0, 0};

    for (int y = 0; y < 4; y++)
    {
        for (int x = 0; x < 4; x++)
        {
            float weight = w[x].x * w[y].y;
            int index = weight >= 0.f; 
            weightsSum[index] += abs(weight);
            float p = abs(weight) / weightsSum[index];

            if (u <= p)
            {
                reservoir[index] = int2(x, y);
                u /= p;
            }
            else 
            {
                u = (u - p) / (1 - p);
            }
        }
    }

    // Get texels
    float4 result = 0.f;
    {
        float2 posCoord = float2(texPos[reservoir[0].x].x, texPos[reservoir[0].y].y);

        // Lookup adjacency information if necessary
        int neighborIndex = Ptex::GetNeighborIndexFromUV(posCoord, texSize, uv);
        Ptex::FaceData neighborData = faceData;
        if (neighborIndex != -1)
        {
            neighborData = Ptex::GetFaceData(material, faceID, neighborIndex);
        }

        const int rotateIndex = neighborIndex * 4 + neighborData.rotateIndex;
        const float2 newUv = neighborIndex == -1 ? uv : mul(uvRotations[rotateIndex], float3(uv.x, uv.y, 1));
        const uint2 faceSize = uint2(1u << neighborData.log2Dim.x, 1u << neighborData.log2Dim.y);

        // Virtual texture lookup
        const uint2 baseOffset = material.baseVirtualPage * PAGE_WIDTH + neighborData.faceOffset;
        const float3 fullUv = VirtualTexture::GetPhysicalUV(baseOffset, newUv, faceSize, mipLevel);
        result += weightsSum[0] * tex.SampleLevel(samplerLinearClamp, fullUv, 0.f);
    }

    if (weightsSum[1] != 0.f)
    {
        float2 negCoord = float2(texPos[reservoir[1].x].x, texPos[reservoir[1].y].y);

        // Lookup adjacency information if necessary
        int neighborIndex = Ptex::GetNeighborIndexFromUV(negCoord, texSize, uv);
        Ptex::FaceData neighborData = faceData; 
        if (neighborIndex != -1)
        {
            neighborData = Ptex::GetFaceData(material, faceID, neighborIndex);
        }

        const int rotateIndex = neighborIndex * 4 + neighborData.rotateIndex;
        const float2 newUv = neighborIndex == -1 ? uv : mul(uvRotations[rotateIndex], float3(uv.x, uv.y, 1));
        const uint2 faceSize = uint2(1u << neighborData.log2Dim.x, 1u << neighborData.log2Dim.y);

        // Virtual texture lookup
        const uint2 baseOffset = material.baseVirtualPage * PAGE_WIDTH + neighborData.faceOffset;
        const float3 fullUv = VirtualTexture::GetPhysicalUV(baseOffset, newUv, faceSize, mipLevel);
        result += weightsSum[1] * tex.SampleLevel(samplerLinearClamp, fullUv, 0.f);
    }

    return result;
}

#endif
