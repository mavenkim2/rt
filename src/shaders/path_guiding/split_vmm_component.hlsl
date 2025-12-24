#include "path_guiding.hlsl"

StructuredBuffer<VMM> vmms : register(t0);

float3 Map2DTo3D(float2 dir)
{
    float3 vec3D = 0.f;
    float norm2 = vec2D.x * vec2D.x + vec2D.y * vec2D.y;
    float length = norm2 > 0.f ? sqrt(norm2) : 0.f;
    float sinc = length > FLT_EPSILON ? sin(length) / length : 0.f;

    vec3D.x = length > 0.0f ? vec2D.x * sinc : vec3D.x;
    vec3D.y = length > 0.0f ? vec2D.y * sinc : vec3D.y;
    vec3D.z = cos(length);

    return vec3D;
}

[numthreads(PATH_GUIDING_GROUP_SIZE, 1, 1)]
void main(uint3 groupID : SV_GroupID, uint groupIndex : SV_GroupIndex)
{
    uint vmmIndex = groupID.x;
    VMM vmm = vmms[vmmIndex];

    // Principal component analysis
    uint componentIndex = groupIndex;
    float covX, covY, covXY;

    // TODO: always zero??
    const float2 mean = 0.f;

    float negB = covX + covY;

    float discriminant = sqrt(Sqr(covX - covY) - 4 * Sqr(covXY));
    float eigenValue0 = 0.5f * (negB + discriminant);
    float2 eigenVector0 = float2(-covXY, covX - eigenValue0);

    // NOTE: eigenvector1 is never used. also, eigenVector1.x may have wrong sign
    //float eigenValue1 = 0.5f * (negB - discriminant);
    //float2 eigenVector1 = float2(covXY, covX - eigenValue1);

    float norm0 = dot(eigenVector0, eigenVector0);
    norm0 = norm0 > FLT_EPSILON ? rsqrt(norm0) : 1.f;
    eigenVector0 *= norm0;

    float newWeight0 = ;
    float3 meanDirection = vmm.directions[componentIndex];

    if (discriminant > 1e-8f)
    {
        float2 temp = eigenValue0 * eigenVector0 * 0.5f;
        float2 meanDir0 = mean + temp;
        float2 meanDIr1 = mean - temp;

        float3 meanDirection = 
        float2x3 basis = BuildOrthonormalBasis(vmm.directions[componentIndex]);

        float3 meanDirection0 = mul(basis, Map2DTo3D(meanDir0));
        float meanCosine0 = meanCosine / abs(dot(meanDirection0, meanDirection));
    }
}
