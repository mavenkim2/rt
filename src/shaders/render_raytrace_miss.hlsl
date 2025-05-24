#include "common.hlsli"
#include "sampling.hlsli"
#include "payload.hlsli"
#include "../rt/shader_interop/gpu_scene_shaderinterop.h"
#include "../rt/shader_interop/ray_shaderinterop.h"

ConstantBuffer<GPUScene> scene : register(b2);

[[vk::push_constant]] RayPushConstant push;

[shader("miss")]
void main(inout RayPayload payload) 
{
    float3 d = normalize(mul(scene.lightFromRender, float4(WorldRayDirection(), 0)).xyz);

    // Equal area sphere to square
    float x = abs(d.x), y = abs(d.y), z = abs(d.z);

    // Compute the radius r
    float r = sqrt(1 - z); // r = sqrt(1-|z|)

    // Compute the argument to atan (detect a=0 to avoid div-by-zero)
    float a = max(x, y), b = min(x, y);
    b = a == 0 ? 0 : b / a;

    // Polynomial approximation of atan(x)*2/pi, x=b
    // Coefficients for 6th degree minimax approximation of atan(x)*2/pi,
    // x=[0,1].
    const float t1 = 0.406758566246788489601959989e-5;
    const float t2 = 0.636226545274016134946890922156;
    const float t3 = 0.61572017898280213493197203466e-2;
    const float t4 = -0.247333733281268944196501420480;
    const float t5 = 0.881770664775316294736387951347e-1;
    const float t6 = 0.419038818029165735901852432784e-1;
    const float t7 = -0.251390972343483509333252996350e-1;
    float phi      = mad(b, mad(b, mad(b, mad(b, mad(b, mad(b, t7, t6), t5), t4), 
                            t3), t2), t1);

    // Extend phi if the input is in the range 45-90 degrees (u<v)
    if (x < y) phi = 1 - phi;

    // Find (u,v) based on (r,phi)
    float v = phi * r;
    float u = r - v;

    if (d.z < 0)
    {
        // southern hemisphere -> mirror u,v
        swap(u, v);
        u = 1 - u;
        v = 1 - v;
    }

    // Move (u,v) to the correct quadrant based on the signs of (x,y)
    u = copysign(u, d.x);
    v = copysign(v, d.y);
    float2 uv = float2(0.5f * (u + 1), 0.5f * (v + 1));

    float3 imageLe = bindlessTextures[push.envMap].SampleLevel(samplerLinearClamp, uv, 0).rgb;
    //payload.radiance = payload.throughput * imageLe;
    //payload.missed = true;
#if 0
    int2 p = int2(int(uv[0] * push.width), int(uv[1] * push.height));
    if (p[0] < 0)
    {
        p[0] = -p[0];                    // mirror across u = 0
        p[1] = push.height - 1 - p[1]; // mirror across v = 0.5
    }
    else if (p[0] >= push.width)
    {
        p[0] = 2 * push.width - 1 - p[0]; // mirror across u = 1
        p[1] = push.height - 1 - p[1];    // mirror across v = 0.5
    }

    if (p[1] < 0)
    {
        p[0] = push.width - 1 - p[0]; // mirror across u = 0.5
        p[1] = -p[1];                   // mirror across v = 0;
    }
    else if (p[1] >= push.height)
    {
        p[0] = push.width - 1 - p[0];      // mirror across u = 0.5
        p[1] = 2 * push.height - 1 - p[1]; // mirror across v = 1
    }

    if (push.width == 1) p[0] = 0;
    if (push.height == 1) p[1] = 0;

    float3 imageLe = bindlessTextures[push.envMap].Load(int3(p, 0)).rgb;
    payload.radiance = payload.throughput * imageLe;
#endif
}
