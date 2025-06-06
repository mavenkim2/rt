#ifndef PTEX_HLSLI_
#define PTEX_HLSLI_

#include "../common.hlsli"
#include "../bit_twiddling.hlsli"
#include "virtual_textures.hlsli"
#include "../../rt/shader_interop/hit_shaderinterop.h"

ByteAddressBuffer faceDataBuffer : register(t11);

static const float2x3 uvRotations[16] = {
    float2x3(1, 0, 0, 0, 1, 1), 
    float2x3(0, 1, 1, -1, 0, 1),
    float2x3(-1, 0, 1, 0, -1, 0),
    float2x3(0, -1, 0, 1, 0, 0),
    float2x3(1, 0, -1, 0, 1, 0),
    float2x3(0, 1, 0, -1, 0, 2), 
    float2x3(-1, 0, 2, 0, -1, -1), 
    float2x3(0, -1, 1, 1, 0, -1), 
    float2x3(1, 0, 0, 0, 1, -1), 
    float2x3(0, 1, -1, -1, 0, 1),
    float2x3(-1, 0, 1, 0, -1, 2),
    float2x3(0, -1, 2, 1, 0, 0), 
    float2x3(1, 0, 1, 0, 1, 0), 
    float2x3(0, 1, 0, -1, 0, 0), 
    float2x3(-1, 0, 0, 0, -1, 1), 
    float2x3(0, -1, 1, 1, 0, 1),
};

DefineBitStreamReader(Ptex, faceDataBuffer)

namespace Ptex
{
    struct FaceData 
    {
        int4 neighborFaces;
        uint2 faceOffset;
        uint2 log2Dim;
        uint rotateIndices;
        bool rotate;
    };

    int GetNeighborIndexFromUV(inout float2 texCoord, float2 faceDim)
    {
        int neighborIndex = -1;

        // TODO: how do I handle corners and different resolution faces?
        if (texCoord.x >= faceDim.x && texCoord.y >= 0 && texCoord.y < faceDim.y)
        {
            neighborIndex = 1;
        }
        else if (texCoord.x < 0 && texCoord.y >= 0 && texCoord.y < faceDim.y)
        {
            neighborIndex = 3;
        }
        else if (texCoord.y >= faceDim.y && texCoord.x >= 0 && texCoord.x < faceDim.x)
        {
            neighborIndex = 2;
        }
        else if (texCoord.y < 0 && texCoord.x >= 0 && texCoord.x < faceDim.x)
        {
            neighborIndex = 0;
        }
        else 
        {
            // Clamp to corner
            texCoord = clamp(texCoord, 0.5, faceDim - 0.5);
        }
        return neighborIndex;
    }

    FaceData GetFaceData(GPUMaterial material, uint faceID)
    {
        // Load face data
        uint faceDataStride = 2 * (material.numVirtualOffsetBits + material.numFaceDimBits) + 4 * material.numFaceIDBits + 9;
        uint bufferOffset = faceDataStride * faceID;
        uint2 offsets = GetAlignedAddressAndBitOffset(material.faceDataOffset, bufferOffset);

        FaceData result;
        BitStreamReader_Ptex reader = CreateBitStreamReader_Ptex(offsets.x, offsets.y, 32 * 4 + 24 * 2 + 4 * 2 + 9);

        result.faceOffset.x = reader.Read<24>(material.numVirtualOffsetBits);
        result.faceOffset.y = reader.Read<24>(material.numVirtualOffsetBits);
        result.log2Dim.x = material.minLog2Dim + reader.Read<4>(material.numFaceDimBits);
        result.log2Dim.y = material.minLog2Dim + reader.Read<4>(material.numFaceDimBits);
        result.rotateIndices = reader.Read<8>(8);
        result.rotate = reader.Read<1>(1);

        result.neighborFaces.x = reader.Read<32>(material.numFaceIDBits);
        result.neighborFaces.y = reader.Read<32>(material.numFaceIDBits);
        result.neighborFaces.z = reader.Read<32>(material.numFaceIDBits);
        result.neighborFaces.w = reader.Read<32>(material.numFaceIDBits);

        const uint shift = 32u - material.numFaceIDBits;
        result.neighborFaces = (result.neighborFaces << shift) >> shift;

        return result;
    }

}

float4 StochasticCatmullRomBorderlessHelper(GPUMaterial material, Ptex::FaceData faceData, float2 texCoord, float2 texSize, uint mipLevel, float reservoirWeightSum, bool debug = false) 
{
    // Lookup adjacency information if necessary
    int neighborIndex = Ptex::GetNeighborIndexFromUV(texCoord, texSize);
    Ptex::FaceData neighborData = faceData;

    if (neighborIndex != -1)
    {
        int neighborFaceID = faceData.neighborFaces[neighborIndex];
        if (neighborFaceID == -1)
        {
            texCoord = clamp(texCoord, 0.5, texSize - 0.5);
        }
        else 
        {
            neighborData = Ptex::GetFaceData(material, neighborFaceID);
        }
    }

    const uint2 faceSize = uint2(1u << neighborData.log2Dim.x, 1u << neighborData.log2Dim.y);
    const int rotateIndex = neighborIndex * 4 + BitFieldExtractU32(faceData.rotateIndices, 2, 2 * max(neighborIndex, 0));
    float2 newUv = texCoord / texSize;
    newUv = neighborIndex == -1 ? newUv : mul(uvRotations[rotateIndex], float3(newUv.x, newUv.y, 1));
    newUv = neighborData.rotate ? float2(1 - newUv.y, newUv.x) : newUv;

    // Virtual texture lookup
    const uint2 baseOffset = material.baseVirtualPage * PAGE_WIDTH + neighborData.faceOffset;
    const float3 fullUv = VirtualTexture::GetPhysicalUV(baseOffset, newUv, faceSize, mipLevel, debug);
    float4 result = reservoirWeightSum * physicalPages.SampleLevel(samplerNearestClamp, fullUv, 0.f);

    if (0)
    {
        printf("faceOffset: %u %u, log2Dim: %u %u\nmip: %u, face: %u, uv: %f %f\n "
                "neighborIndex: %i, weight sum: %f rgb: %f %f %f", 
                faceData.faceOffset.x, faceData.faceOffset.y, faceData.log2Dim.x, faceData.log2Dim.y, 
                mipLevel, 0, newUv.x, newUv.y, neighborIndex, reservoirWeightSum, result.x, result.y, result.z);
    }

    return result;
}

// https://research.nvidia.com/labs/rtr/publication/pharr2024stochtex/stochtex.pdf
float4 SampleStochasticCatmullRomBorderless(Ptex::FaceData faceData, GPUMaterial material, uint faceID, float2 uv, uint mipLevel, inout float u, bool debug = false) 
{
    float2 texSize = float2(1u << (faceData.log2Dim.x - mipLevel), 1u << (faceData.log2Dim.y - mipLevel));
    texSize = faceData.rotate ? texSize.yx : texSize;
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

    // Weighted reservoir sampling
    int2 reservoir[2] = {int2(0, 0), int2(0, 0)};
    float weightsSum[2] = {0, 0};

    for (int y = 0; y < 4; y++)
    {
        for (int x = 0; x < 4; x++)
        {
            float weight = w[x].x * w[y].y;
            int index = weight < 0.f; 
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
    float2 posCoord = float2(texPos[reservoir[0].x].x, texPos[reservoir[0].y].y);
    result += StochasticCatmullRomBorderlessHelper(material, faceData, posCoord, 
                                                   texSize, mipLevel, weightsSum[0], debug);
    if (weightsSum[1] != 0.f)
    {
        float2 negCoord = float2(texPos[reservoir[1].x].x, texPos[reservoir[1].y].y);
        result -= StochasticCatmullRomBorderlessHelper(material, faceData, negCoord, 
                                                       texSize, mipLevel, weightsSum[1], debug);
    }

    return result;
}

#endif
