#include "common.hlsli"
#include "bxdf.hlsli"
#include "rt.hlsli"
#include "sampling.hlsli"
#include "../rt/shader_interop/ray_shaderinterop.h"
#include "../rt/shader_interop/hit_shaderinterop.h"

RaytracingAccelerationStructure accel : register(t0);
RWTexture2D<float4> image : register(u1);
ConstantBuffer<GPUScene> scene : register(b2);
StructuredBuffer<RTBindingData> rtBindingData : register(t3);
StructuredBuffer<GPUMaterial> materials : register(t4);
ByteAddressBuffer denseGeometry : register(t5);

[[vk::push_constant]] RayPushConstant push;

struct BitStreamReader 
{
    uint4 buffer;
    uint bitOffset;
};

uint ReadBits(inout BitStreamReader reader, uint bitSize)
{
    uint result = 0;

    uint bitOffset = reader.bitOffset & 31;
    uint remaining = min(32, bitOffset + bitSize);
    uint remaining = min(bitSize, 32 - bitOffset);
    uint mask = uint((1ul << remaining) - 1) & ~((1u << (reader.bitOffset & 31)) - 1); 
    uint index = reader.bitOffset >> 5;

    result |= buffer[index] & mask;
    result |= buffer[index + 1]

    reader.bitOffset += remaining;

    uint prevBitOffset = bitOffset;
    reader.bitOffset += ;
    result |= ;

    if (bitSize + reader.bitOffset > 128)
    {
    }
}

#if 0
float3 DecodeTriangle(uint offset, int primitiveIndex)
{
    enum 
    {
        Restart = 0,
        Edge1 = 1, 
        Edge2 = 2, 
        Backtrack = 3,
    };

    // per scene index Dense Geometry
    uint blockIndex = primitiveIndex >> blockShift;
    uint triangleIndex = primitiveIndex & triangleMask;

    DenseGeometry dg = denseGeometry.Load<DenseGeometryHeader>(offset);
    uint3 indexAddress = uint3(0, 1, 2);
    uint r = 1;
    uint bt = 0;
    uint prevCtrl = Restart;
    for (uint k = 1; k < triangleIndex; k++)
    {
        uint ctrl = somehow;
        uint3 prev = indexAddress;
        switch (ctrl)
        {
            case Restart:
            {
                r++;
                // what is i?
                indexAddress = uint3(2 * r + i - 2, 2 * r + i - 1, 2 * r + i);
            }
            break;
            case Edge1: 
            {
                indexAddress = uint3(prev[2], prev[1], 2 * r + k);
                bt = prev[0];
            }
            break;
            case Edge2: 
            {
                indexAddress = uint3(prev[0], prev[2], 2 * r + k);
                bt = prev[1];
            }
            break;
            case Backtrack: 
            {
                indexAddress = prevCtrl == Edge1 ? uint3(bt, prev[0], 2 * r + k) 
                                                 : uint3(prev[1], bt, 2 * r + k);
            }
            break;
        }
        prevCtrl = ctrl;
    }

    [[unroll]]

    float3 result;
    for (int k = 0; k < 3; k++)
    {
        uint vid = countbits(indexAddress[k]);
        if ()
        {
            // use reuse buffer
        }
        result[k] = dg.anchor[k] + offset
    }

    f32 x = (dg.anchor.x + offsetX) * scene.scale;
    f32 y = (dg.anchor.y + offsetY) * scene.scale;
    f32 z = (dg.anchor.z + offsetZ) * scene.scale;
    return float3(x, y, z);
}
#endif

[numthreads(PATH_TRACE_NUM_THREADS_X, PATH_TRACE_NUM_THREADS_Y, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    RNG rng = RNG::Init(RNG::PCG3d(DTid.xyx).zy, push.frameNum);

    // Generate Ray
    float2 sample = rng.Uniform2D();
    const float2 filterRadius = float2(0.5, 0.5);
    float2 filterSample = float2(lerp(-filterRadius.x, filterRadius.x, sample[0]), 
                                 lerp(-filterRadius.y, filterRadius.y, sample[1]));
    filterSample += float2(0.5, 0.5) + float2(DTid.xy);
    float2 pLens = rng.Uniform2D();

    float3 throughput = float3(1, 1, 1);
    float3 radiance = float3(0, 0, 0);

    const int maxDepth = 2;
    int depth = 0;

    float3 pos;
    float3 dir;
    float3 dpdx, dpdy, dddx, dddy;
    GenerateRay(scene, filterSample, pLens, pos, dir, dpdx, dpdy, dddx, dddy);

    while (true)
    {
        RayDesc desc;
        desc.Origin = pos;
        desc.Direction = dir;
        desc.TMin = 0;
        desc.TMax = FLT_MAX;
        
        RayQuery<RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES | RAY_FLAG_FORCE_OPAQUE> query;
        query.TraceRayInline(accel, RAY_FLAG_NONE, 0xff, desc);
        query.Proceed();

        // TODO: emitter intersection
        if (depth++ >= maxDepth) break;

        if (query.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
        {
            uint bindingDataIndex = query.CommittedInstanceID() + query.CommittedGeometryIndex();
            uint vertexBufferIndex = 3 * bindingDataIndex;
            uint indexBufferIndex = 3 * bindingDataIndex + 1;
            uint normalBufferIndex = 3 * bindingDataIndex + 2;
            uint primID = query.CommittedPrimitiveIndex();

            uint index0 = bindlessUints[NonUniformResourceIndex(indexBufferIndex)][NonUniformResourceIndex(3 * primID + 0)];
            uint index1 = bindlessUints[NonUniformResourceIndex(indexBufferIndex)][NonUniformResourceIndex(3 * primID + 1)];
            uint index2 = bindlessUints[NonUniformResourceIndex(indexBufferIndex)][NonUniformResourceIndex(3 * primID + 2)];

            uint normal0 = bindlessUints[NonUniformResourceIndex(normalBufferIndex)][NonUniformResourceIndex(index0)];
            uint normal1 = bindlessUints[NonUniformResourceIndex(normalBufferIndex)][NonUniformResourceIndex(index1)];
            uint normal2 = bindlessUints[NonUniformResourceIndex(normalBufferIndex)][NonUniformResourceIndex(index2)];

            float3 p0 = bindlessFloat3s[NonUniformResourceIndex(vertexBufferIndex)][NonUniformResourceIndex(index0)];
            float3 p1 = bindlessFloat3s[NonUniformResourceIndex(vertexBufferIndex)][NonUniformResourceIndex(index1)];
            float3 p2 = bindlessFloat3s[NonUniformResourceIndex(vertexBufferIndex)][NonUniformResourceIndex(index2)];

            float3 n0 = DecodeOctahedral(normal0);
            float3 n1 = DecodeOctahedral(normal1);
            float3 n2 = DecodeOctahedral(normal2);

            float3 gn = normalize(cross(p0 - p2, p1 - p2));
            float2 bary = query.CommittedTriangleBarycentrics();

            float3 n = normalize(n0 + (n1 - n0) * bary[0] + (n2 - n0) * bary[1]);

            // Get material
            RTBindingData bindingData = rtBindingData[bindingDataIndex];
            GPUMaterial material = materials[bindingData.materialIndex];
            float eta = material.eta;

            float3 wo = -normalize(query.CandidateObjectRayDirection());

            float u = rng.Uniform();
            float R = FrDielectric(dot(wo, n), eta);

            float T = 1 - R;
            float pr = R, pt = T;

            float3 origin = p0 + (p1 - p0) * bary[0] + (p2 - p0) * bary[1];

            if (u < pr / (pr + pt))
            {
                dir = Reflect(wo, n);
            }
            else
            {
                float etap;
                bool valid = Refract(wo, n, eta, etap, dir);
                if (!valid)
                {
                    radiance = float3(0, 0, 0);
                    break;
                }
            
                throughput /= etap * etap;
            }

            origin = TransformP(query.CandidateObjectToWorld3x4(), origin);
            dir = TransformV(query.CandidateObjectToWorld3x4(), normalize(dir));
            pos = OffsetRayOrigin(origin, gn);
        }
        else 
        {
            float3 d = normalize(mul(scene.lightFromRender, 
                                 float4(query.WorldRayDirection(), 0)).xyz);

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
            radiance += throughput * imageLe;
            break;
        }
    }
    image[DTid.xy] = float4(radiance, 1);
}
