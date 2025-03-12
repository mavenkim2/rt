static const float PI = 3.1415926535f;
static const float FLT_MAX = asfloat(0x7F800000);

#ifdef __spirv__
[[vk::binding(0, 1)]] Texture2D bindlessTextures[];
#else
#error
#endif

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
    return asfloat(asuint(a) ^ (asuint(b) & 0x80000000));
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
