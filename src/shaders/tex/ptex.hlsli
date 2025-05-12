#ifndef PTEX_HLSLI_
#define PTEX_HLSLI_

#include "../common.hlsli"

const float2x3 uvRotations[16];

struct FaceData 
{
    uint2 faceOffsets[5];
    uint neighborFaceSize;
    // Bottom 8 bits is 4 bits each for log2 dimensions, then 2 bits for each neighbor for 
    // uv rotation
    uint faceSize_rotate;

    uint2 GetFaceSize()
    {
        uint width = 1u << BitFieldExtractU32(faceSize_rotate, 4, 0);
        uint height = 1u << BitFieldExtractU32(faceSize_rotate, 4, 4);
        uint2 faceDim = uint2(width, height);
        return faceDim;
    }

    uint2 GetNeighborFaceSize(uint neighbor)
    {
        uint width = 1u << BitFieldExtractU32(neighborFaceSize, 4, 8 * neighbor);
        uint height = 1u << BitFieldExtractU32(neighborFaceSize, 4, 8 * neighbor + 4);
        uint2 faceDim = uint2(width, height);
        return faceDim;
    }

    int GetUV(float2 texCoord, inout float2 uv)
    {
        float2 faceDim = (float2)GetFaceSize();
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
};

// https://research.nvidia.com/labs/rtr/publication/pharr2024stochtex/stochtex.pdf
float4 SampleStochasticCatmullRomBorderless(Texture2DArray tex, uint2 basePage, FaceData faceData, float2 uv, uint mipLevel, inout float u) 
{
    float width, height, layers;
    tex.GetDimensions(width, height, layers);
    float2 texSize = float2(width, height);

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
        const int neighborIndex = faceData.GetUV(posCoord, uv);
        const uint2 faceOffset = neighborIndex == -1 ? faceData.faceOffsets[0] : faceData.faceOffsets[neighborIndex + 1];
        const int rotateIndex = neighborIndex == -1 ? -1 : neighborIndex * 4 + BitFieldExtractU32(faceData.faceSize_rotate, 2, 8 + 2 * neighborIndex);
        const float2 newUv = rotateIndex == -1 ? uv : mul(uvRotations[rotateIndex], float3(uv.x, uv.y, 1));
        const uint2 faceSize = neighborIndex == -1 ? faceData.GetFaceSize() : faceData.GetNeighborFaceSize((uint)neighborIndex);

        // Virtual texture lookup
        const uint2 baseOffset = basePage * VirtualTexture::pageWidth + faceOffset;
        const float3 fullUv = VirtualTexture::GetPhysicalUV(baseOffset, newUv, faceSize, mipLevel);
        result += weightsSum[0] * tex.SampleLevel(samplerLinearClamp, fullUv, 0.f);
    }

    if (weightsSum[1] != 0.f)
    {
        float2 negCoord = float2(texPos[reservoir[1].x].x, texPos[reservoir[1].y].y);

        // Lookup adjacency information if necessary
        const int neighborIndex = faceData.GetUV(negCoord, uv);
        const uint2 faceOffset = neighborIndex == -1 ? faceData.faceOffsets[0] : faceData.faceOffsets[neighborIndex + 1];
        const int rotateIndex = neighborIndex == -1 ? -1 : neighborIndex * 4 + BitFieldExtractU32(faceData.faceSize_rotate, 2, 8 + 2 * neighborIndex);
        const float2 newUv = rotateIndex == -1 ? uv : mul(uvRotations[rotateIndex], float3(uv.x, uv.y, 1));
        const uint2 faceSize = neighborIndex == -1 ? faceData.GetFaceSize() : faceData.GetNeighborFaceSize((uint)neighborIndex);

        // Virtual texture lookup
        const uint2 baseOffset = basePage * VirtualTexture::pageWidth + faceOffset;
        const float3 fullUv = VirtualTexture::GetPhysicalUV(baseOffset, newUv, faceSize, mipLevel);
        result += weightsSum[1] * tex.SampleLevel(samplerLinearClamp, fullUv, 0.f);
    }

    return result;
}

#endif
