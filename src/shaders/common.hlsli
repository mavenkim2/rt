static const float PI = 3.1415926535f;
static const float FLT_MAX = asfloat(0x7F800000);
#
#define texture2DSpace space1
#define structuredBufferSpace space2

#ifdef __spirv__
Texture2D bindlessTextures[] : register(t0, texture2DSpace);
StructuredBuffer<float3> bindlessFloat3s[] : register(t0, structuredBufferSpace);
StructuredBuffer<uint> bindlessUints[] : register(t0, structuredBufferSpace);
ByteAddressBuffer bindlessBuffer[] : register(t0, structuredBufferSpace);
#else
#error
#endif

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

struct RayPayload 
{
    float3 pxOffset;
    float3 pyOffset;
    float3 dxOffset;
    float3 dyOffset;
    uint seed;

    float3 radiance;
    float3 throughput;
};

template <typename T>
void swap(T a, T b)
{
    T temp = a;
    a = b;
    b = temp;
}

float copysign(float a, float b)
{
    return asfloat(abs(asint(a)) | (asuint(b) & 0x80000000));
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

float3 DecodeOctahedral(uint n) 
{
    float2 decoded = float2(float((n >> 16) & 0xff), float(n & 0xff)) * 2 - 1;
    float3 normal = float3(decoded.xy, 1 - abs(decoded.x) - abs(decoded.y));
    if (normal.z < 0) normal.xy = flipsign((1 - abs(normal.yx)), normal.xy);
    return normalize(normal);
}

float3 Transform(float4x4 m, float3 p)
{
    float4 result = mul(m, float4(p, 1.f));
    result /= result.w;
    return result.xyz;
}

float2 SampleUniformDiskConcentric(float2 u)
{
    float2 uOffset = 2 * u - 1;

    bool mask    = abs(uOffset.x) > abs(uOffset.y);
    float r      = select(mask, uOffset.x, uOffset.y);
    float theta  = select(mask, PI / 4 * (uOffset.y / uOffset.x),
                          PI / 2 - PI / 4 * (uOffset.x / uOffset.y));

    float2 result = select(uOffset.x == 0 && uOffset.y == 0, float2(0, 0),
                            r * float2(cos(theta), sin(theta)));
    return result;
}

void DefocusBlur(float3 dIn, float2 pLens, float focalLength, out float3 o,
                 out float3 d)
{
    float t       = focalLength / -dIn.z;
    float3 pFocus = dIn * t;
    o             = float3(pLens.x, pLens.y, 0.f);
    d             = normalize(pFocus - o);
}

void Transform(inout RayPayload payload, float3x4 m)
{
    payload.pxOffset = mul(m, float4(payload.pxOffset, 1));
    payload.pyOffset = mul(m, float4(payload.pyOffset, 1));
    payload.dxOffset = mul(m, float4(payload.dxOffset, 0));
    payload.dyOffset = mul(m, float4(payload.dyOffset, 0));
}

void Transform(inout RayDesc desc, float3x4 m)
{
    desc.Origin = mul(m, float4(desc.Origin, 1));
    desc.Direction = mul(m, float4(desc.Direction, 0));
}
