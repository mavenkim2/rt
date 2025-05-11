#ifndef PTEX_HLSLI_
#define PTEX_HLSLI_

StructuredBuffer<FaceData> faceMetadata : register(t8);

// https://research.nvidia.com/labs/rtr/publication/pharr2024stochtex/stochtex.pdf
float4 SampleStochasticCatmullRomBorderless(Texture2DArray tex, float2 uv, uint faceID, inout float u) 
{
    float width, height, layers;
    tex.GetDimensions(width, height, layers);
    float2 texSize = float2(width, height);

    float2 w[4];
    float2 texPos[4];

    float2 samplePos = uv * texSize;
    texPos[1] = floor(samplePos - 0.5f) + 0.5f;

    float2 f = samplePos - texPos1;

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

    float4 result = 0;

    int2 reservoir[2] = {};
    float weightsSum[2] = {};

    for (int y = 0; y < 4; y++)
    {
        for (int x = 0; x < 4; x++)
        {
            float weight = w[x] * w[y];
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

    float4 result = 0.f;
    float2 posCoord = float2(texPos[reservoir[0].x].x, texPos[reservoir[0].y].y);

    float2 faceDim;
    // TODO: need to get this from dgf
    uint4 neighborFaces;

    uint posNeighborFace = posCoord.x >= faceDim.x ? 
                                                   (posCoord.y < faceDim.x ? faceID : (: neighborFaces.x ;

    // Virtual texture lookup
    if (weightsSum.y != 0.f)
    {
        float2 negCoord = float2(texPos[reservoir[1].x].x, texPos[reservoir[1].y].y);
        // Virtual texture lookup
    }

    return result;
}

#endif
