#ifndef SOMETHING_HLSLI
#define SOMETHING_HLSLI

#include "../common.hlsli"

#define MC_ADAPTIVE_PROB 0.7
#define MC_NUM_SAMPLES 5
#define MC_ADAPTIVE_BUFFER_SIZE (1u << 25)

#define GRID_PRIME_X 1
#define GRID_PRIME_Y 2654435761
#define GRID_PRIME_Z 805459861

const float adaptiveGridMinWidth = .01f;
const float adaptiveGridStepsPerUnitSize = 4.f;
const float adaptiveGridPower = 4.f;

struct MCState 
{
    float3 w_tgt;
    float sum_w;
    float w_cos;

    half3 mv;
    float T;

    uint16_t N;
    uint16_t hash; // grid_idx and level
};

void GetAdaptiveLevel(inout RNG rng, float3 pos)
{
    // TODO
    float tanAlphaHalf = 0.003f;
    float gridWidth = 2 * tanAlphaHalf * length(scene.cameraP - pos);

    float levelU = rng.Uniform();
    uint level = uint(round(adaptiveGridStepsPerUnitSize * log(max(gridWidth, adaptiveGridMinWidth) / adaptiveGridMinWidth) / log(adaptiveGridPower)))
    uint levelOffset = uint(-log(1 - levelU));
    return level + levelOffset;
}

int3 GetGridIndex(float3 pos, float3 cellWidth, float u)
{
    float3 gridPos = fract(pos / cellWidth);

    uint chosenIndex = 0;
    float sum = 0.f;

    for (uint i = 0; i < 8; i++)
    {
        float x = (i & 1) ? (1 - gridPos.x) : gridPos.x;
        float y = (i & 2) ? (1 - gridPos.y) : gridPos.y;
        float z = (i & 4) ? (1 - gridPos.z) : gridPos.z;

        sum += x * y * z;
        if (u < sum)
        {
            chosenIndex = i;
            break;
        }
    }

    float3 t = float3(chosenIndex & 1, chosenIndex & 2, chosenIndex & 4);

    return int3(lerp(ceil(pos / cellWidth), floor(pos / cellWidth), t));
}

int3 GetAdaptiveGridIndex(float3 pos, uint level, float u)
{
    float gridWidth = (adaptiveGridMinWidth * pow(adaptiveGridPower, level / adaptiveGridStepsPerUnitSize))
    int3 p = GetGridIndex(pos, gridWidth, u);
    return p;
}

uint HashGridNormalLevel(uint3 index, float3 normal, uint level, uint modulus) 
{
    uint cube = abs(normal.x) > abs(normal.y) && abs(normal.x) > abs(normal.z) ? (normal.x >= 0 ? 0 : 1) : 
                (abs(normal.y) > abs(normal.x) && abs(normal.y) > abs(normal.z) ? (normal.y >= 0 ? 2 : 3) :
                (normal.z >= 0 ? 4 : 5));

    uint cube = cubemap_side(normal);
    return ((index.x * GRID_PRIME_X) ^ (index.y * GRID_PRIME_Y) ^ (index.z * GRID_PRIME_Z) ^ ((cube | (level << 16)) * 723850877)) % modulus;
}

#define GRID_PRIME_X_2 74763401
#define GRID_PRIME_Y_2 2254437599
#define GRID_PRIME_Z_2 508460413

uint Hash2GridLevel(const uint3 index, const uint level) 
{
    return (index.x * GRID_PRIME_X_2) ^ (index.y * GRID_PRIME_Y_2) ^ (index.z * GRID_PRIME_Z_2) ^ (9351217 * level + 13 * level);
}

// VMF
float4 GetMCStateVMF(MCState state, float3 pos)
{
    float3 statePos = state.sum_w > 0.0 ? state.w_tgt / state.sum_w : state.w_tgt;
    float3 stateDir = statePos - pos;
    float Np = .2f / length2(stateDir);
    stateDir = normalize(stateDir);
    // rp = 0
    float r = ((state.N * state.N * clamp(state.w_cos / state.sum_w, 0.0, 0.9999999)) / (state.N * state.N + Np))
    return float4(stateDir, (3.0 * r - r * r * r) / (1.0 - r * r));
}

void SampleHashGrids(inout RNG rng, float3 pos, float3 normal)
{
    float scoreSum = 0.f;
    MCState chosenState;
    float scores[MC_NUM_SAMPLES];
    float4 vmfs[MC_NUM_SAMPLES];
    uint chosenIndex = 0;

    for (int i = 0; i < MC_NUM_SAMPLES; i++)
    {
        float u = rng.Uniform();

        // TODO: source uses log2?
        if (u < ADAPTIVE_PROB)
        {
            uint level = GetAdaptiveLevel(rng, pos);
            int3 gridIndex = GetAdaptiveGridIndex(pos, level, rng.Uniform());
            uint bufferIndex = HashGridNormalLevel(gridIndex, normal, level, MC_ADAPTIVE_BUFFER_SIZE);

            uint16_t hash = uint16_t(Hash2GridLevel(gridIndex, level));

            MCState mcState = mcStates[bufferIndex];
            if (mcState.sum_w < 0 || hash != mcState.hash)
            {
                mcState.sum_w = 0;
            }
            //mcState.w_tgt += ;
            scoreSum += mcState.sum_w;

            if (rng.Uniform() < mcState.sum_w / scoreSum)
            {
                chosenState = mcState;
                chosenIndex = i;
            }
            vmfs[i] = GetMCStateVMF(mcState, pos);
            scores[i] = mcState.sum_w;
        }
        else 
        {
        }
    }

    // Sample VMF or BSDF
    float pdf = 0.f;
    if (scoreSum > 0)
    {
        for (int i = 0; i < MC_NUM_SAMPLES; i++)
        {
            pdf += VmfPdf(vmfs[i]) * scores[i];
        }
        pdf /= scoreSum;
    }
}

#endif
