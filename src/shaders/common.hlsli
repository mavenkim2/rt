static const float PI = 3.1415926535f;
static const float FLT_MAX = asfloat(0x7F800000);

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

struct GPUScene
{
    float4x4 cameraFromRaster;
    float3x4 renderFromCamera;
    float3 dxCamera;
    float3 dyCamera;
    float lensRadius;
    float focalLength;

    void GenerateRay(float2 pFilm, float2 pLens, out RayDesc desc, out RayPayload payload)
    {
        float3 pCamera = Transform(cameraFromRaster, float3(pFilm, 0.f));
        desc.Origin = float3(0.f, 0.f, 0.f);
        desc.Direction = normalize(pCamera);
        if (lensRadius > 0.f)
        {
            pLens = lensRadius * SampleUniformDiskConcentric(pLens);

            DefocusBlur(desc.Direction, pLens, focalLength, desc.Origin, 
                        desc.Direction);
            DefocusBlur(normalize(pCamera + dxCamera), pLens, focalLength, payload.pxOffset,
                        payload.dxOffset);
            DefocusBlur(normalize(pCamera + dyCamera), pLens, focalLength, payload.pyOffset,
                        payload.dyOffset);
        }
        Transform(desc, renderFromCamera);
        Transform(payload, renderFromCamera);
    }
};
