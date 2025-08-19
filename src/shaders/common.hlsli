#ifndef COMMON_HLSLI_
#define COMMON_HLSLI_

static const float PI = 3.1415926535f;
static const float FLT_MAX = asfloat(0x7F800000);
static const float InvPi = 0.31830988618379067154f;
static const float OneMinusEpsilon = 0x1.fffffep-1;

#define texture2DSpace space1
#define structuredBufferSpace space2
#define rwTexture2DSpace space3

Texture2D bindlessTextures[] : register(t0, texture2DSpace);
Texture2D<float3> bindlessFloat3Textures[] : register(t0, texture2DSpace);
StructuredBuffer<float3> bindlessFloat3s[] : register(t0, structuredBufferSpace);
StructuredBuffer<uint> bindlessUints[] : register(t0, structuredBufferSpace);

ByteAddressBuffer bindlessBuffer[] : register(t0, structuredBufferSpace);
RWTexture2D<uint> bindlessRWTextureUint[] : register(u0, rwTexture2DSpace);
RWTexture2D<uint2> bindlessRWTextureUint2[] : register(u0, rwTexture2DSpace);
RWTexture2D<float4> bindlessRWTexture2D[] : register(u0, rwTexture2DSpace);

SamplerState samplerLinearClamp : register(s50);
SamplerState samplerNearestClamp : register(s51);

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
    const uint signBit = 0x80000000;
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

// https://developer.nvidia.com/blog/optimizing-compute-shaders-for-l2-locality-using-thread-group-id-swizzling/
// https://github.com/LouisBavoil/ThreadGroupIDSwizzling/blob/master/ThreadGroupTilingX.hlsl
uint2 SwizzleThreadGroup(
	const uint2 dispatchGridDim,		// Arguments of the Dispatch call (typically from a ConstantBuffer)
	const uint2 ctaDim,			// Already known in HLSL, eg:[numthreads(8, 8, 1)] -> uint2(8, 8)
	const uint maxTileWidth,		// User parameter (N). Recommended values: 8, 16 or 32.
	const uint2 groupThreadID,		// SV_GroupThreadID
	const uint2 groupId			// SV_GroupID
)
{
	// A perfect tile is one with dimensions = [maxTileWidth, dispatchGridDim.y]
	const uint Number_of_CTAs_in_a_perfect_tile = maxTileWidth * dispatchGridDim.y;

	// Possible number of perfect tiles
	const uint Number_of_perfect_tiles = dispatchGridDim.x / maxTileWidth;

	// Total number of CTAs present in the perfect tiles
	const uint Total_CTAs_in_all_perfect_tiles = Number_of_perfect_tiles * maxTileWidth * dispatchGridDim.y;
	const uint vThreadGroupIDFlattened = dispatchGridDim.x * groupId.y + groupId.x;

	// Tile_ID_of_current_CTA : current CTA to TILE-ID mapping.
	const uint Tile_ID_of_current_CTA = vThreadGroupIDFlattened / Number_of_CTAs_in_a_perfect_tile;
	const uint Local_CTA_ID_within_current_tile = vThreadGroupIDFlattened % Number_of_CTAs_in_a_perfect_tile;
	uint Local_CTA_ID_y_within_current_tile;
	uint Local_CTA_ID_x_within_current_tile;

	if (Total_CTAs_in_all_perfect_tiles <= vThreadGroupIDFlattened)
	{
		// Path taken only if the last tile has imperfect dimensions and CTAs from the last tile are launched. 
		uint X_dimension_of_last_tile = dispatchGridDim.x % maxTileWidth;
	#ifdef DXC_STATIC_DISPATCH_GRID_DIM
		X_dimension_of_last_tile = max(1, X_dimension_of_last_tile);
	#endif
		Local_CTA_ID_y_within_current_tile = Local_CTA_ID_within_current_tile / X_dimension_of_last_tile;
		Local_CTA_ID_x_within_current_tile = Local_CTA_ID_within_current_tile % X_dimension_of_last_tile;
	}
	else
	{
		Local_CTA_ID_y_within_current_tile = Local_CTA_ID_within_current_tile / maxTileWidth;
		Local_CTA_ID_x_within_current_tile = Local_CTA_ID_within_current_tile % maxTileWidth;
	}

	const uint Swizzled_vThreadGroupIDFlattened =
		Tile_ID_of_current_CTA * maxTileWidth +
		Local_CTA_ID_y_within_current_tile * dispatchGridDim.x +
		Local_CTA_ID_x_within_current_tile;

	uint2 SwizzledvThreadGroupID;
	SwizzledvThreadGroupID.y = Swizzled_vThreadGroupIDFlattened / dispatchGridDim.x;
	SwizzledvThreadGroupID.x = Swizzled_vThreadGroupIDFlattened % dispatchGridDim.x;

	uint2 SwizzledvThreadID;
	SwizzledvThreadID.x = ctaDim.x * SwizzledvThreadGroupID.x + groupThreadID.x;
	SwizzledvThreadID.y = ctaDim.y * SwizzledvThreadGroupID.y + groupThreadID.y;

	return SwizzledvThreadID.xy;
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

float length2(float3 v)
{
    return dot(v, v);
}

void Translate(inout float3x4 transform, float3 p)
{
    transform[0].w += p.x;
    transform[1].w += p.y;
    transform[2].w += p.z;
}

#endif
