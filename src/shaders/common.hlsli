#ifndef COMMON_HLSLI_
#define COMMON_HLSLI_

static const float PI = 3.1415926535f;
static const float FLT_MAX = asfloat(0x7F800000);
static const float InvPi = 0.31830988618379067154f;
static const float OneMinusEpsilon = 0x1.fffffep-1;

#define texture2DSpace space1
#define structuredBufferSpace space2

Texture2D bindlessTextures[] : register(t0, texture2DSpace);
Texture2D<float3> bindlessFloat3Textures[] : register(t0, texture2DSpace);
StructuredBuffer<float3> bindlessFloat3s[] : register(t0, structuredBufferSpace);
StructuredBuffer<uint> bindlessUints[] : register(t0, structuredBufferSpace);
ByteAddressBuffer bindlessBuffer[] : register(t0, structuredBufferSpace);

SamplerState samplerLinearClamp : register(s50);

// Alan wake 2 random numbers 
uint Hash(uint x)
{
    x ^= x >> 16;
    x *= 0x7FEB352D;
    x ^= x >> 15;
    x *= 0x846CA68B;
    x ^= x >> 16;
    return x;
}

uint GetRandom(inout uint x)
{
    x = Hash(x);
    return 2.0 - asfloat((x >> 9) | 0x3F800000);
}

uint2 Get2Random(inout uint x)
{
    uint a = GetRandom(x);
    uint b = GetRandom(x);
    return uint2(a, b);
}

template <typename T>
void swap(T a, T b)
{
    T temp = a;
    a = b;
    b = temp;
}

float copysign(float a, float b)
{
    uint signBit = 0x80000000;
    return asfloat((asuint(a) & ~signBit) | (asuint(b) & signBit));
}

template <int N>
vector<float, N> flipsign(vector<float, N> mag, vector<float, N> sign)
{
    vector<float, N> result;
    for (int i = 0; i < N; i++)
    {
        result[i] = asfloat(asuint(mag[i]) ^ (asuint(sign[i]) & 0x80000000));
    }
    return result;
}

template <uint N>
float Dequantize(uint n)
{
    return float(n) / ((1u << N) - 1);
}

float3 UnpackOctahedral(float2 v)
{
    float3 normal = float3(v.xy, 1 - abs(v.x) - abs(v.y));
    float t = saturate(-normal.z);
    normal.xy += select(normal.xy >= 0.f, -t, t);
    return normalize(normal);
}

float3 DecodeOctahedral(uint n) 
{
    float2 decoded = float2(Dequantize<16>((n >> 16) & 0xffff),
            Dequantize<16>((n & 0xffff))) * 2 - 1; 
    return UnpackOctahedral(decoded);
}

float3 Transform(float4x4 m, float3 p)
{
    float4 result = mul(m, float4(p, 1.f));
    result /= result.w;
    return result.xyz;
}

void DefocusBlur(float3 dIn, float2 pLens, float focalLength, out float3 o,
                 out float3 d)
{
    float t       = focalLength / -dIn.z;
    float3 pFocus = dIn * t;
    o             = float3(pLens.x, pLens.y, 0.f);
    d             = normalize(pFocus - o);
}

float3 TransformP(float3x4 m, float3 p)
{
    return mul(m, float4(p, 1));
}

float3 TransformV(float3x4 m, float3 v)
{
    return mul(m, float4(v, 0));
}

uint2 SwizzleThreadGroup(uint3 dispatchThreadID, uint3 groupID, uint3 GTid, uint2 groupDim, uint dispatchDimX, uint tileWidth,
    uint log2TileWidth, uint numGroupsInTile, out uint2 swizzledGid)
{
    const uint groupIDFlattened = groupID.y * dispatchThreadID.x + groupID.x;
    const uint tileID = groupIDFlattened / numGroupsInTile;
    const uint groupIDinTileFlattened = groupIDFlattened % numGroupsInTile;

    const uint numFullTiles = dispatchDimX / tileWidth;
    const uint numGroupsInFullTiles = numFullTiles * numGroupsInTile;

    uint2 groupIDinTile;
    if (groupIDFlattened >= numGroupsInFullTiles)
    {
        // DispatchDimX & NumGroupsInTile
        const uint lastTileDimX = dispatchDimX - tileWidth * numFullTiles;
        groupIDinTile = uint2(groupIDinTileFlattened % lastTileDimX, groupIDinTileFlattened / lastTileDimX);
    }
    else
    {
        groupIDinTile = uint2(
            groupIDinTileFlattened & (tileWidth - 1),
            groupIDinTileFlattened >> log2TileWidth);
    }

    const uint swizzledGidFlattened = groupIDinTile.y * dispatchDimX + tileID * tileWidth + groupIDinTile.x;
    swizzledGid = uint2(swizzledGidFlattened % dispatchDimX, swizzledGidFlattened / dispatchDimX);
    const uint2 swizzledDTid = swizzledGid * groupDim + GTid.xy;

    return swizzledDTid;
}

float ReciprocalPow2(int power)
{
    return asfloat((127 - power) << 23);
}

float2x3 BuildOrthonormalBasis(float3 n)
{
    const float s = n.z >= 0.f ? 1.f : -1.f;
    const float a = -1.0 / (s + n.z);
    const float b = n.x * n.y * a;

    float2x3 result;
    result[0] = float3(mad(n.x * a, n.x * s, 1.0f), s * b, -s * n.x);
    result[1] = float3(b, mad(n.y * a, n.y, s), -n.y);

    return result;
}

float Luminance(float3 rgb)
{
    return dot(float3(0.2126f, 0.7152f, 0.0722f), rgb);
}

#endif
