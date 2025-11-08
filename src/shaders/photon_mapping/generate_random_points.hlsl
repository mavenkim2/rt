#include "../sampling.hlsli"

RWStructuredBuffer<float3> points : register(u0);
RWStructuredBuffer<float3> bounds : register(u1);

[numthreads(32, 1, 1)]
void main(uint3 dtID : SV_DispatchThreadID)
{
    if (dtID.x == 0)
    {
        bounds[0] = FLT_MAX;
        bounds[1] = -FLT_MAX;
    }

    RNG rng = RNG::Init(dtID.x, 0);
    float3 pt = float3(rng.Uniform2D(), rng.Uniform()) * 1000.f;
    points[dtID.x] = pt;
}
