#ifndef LOD_ERROR_TEST_HLSLI
#define LOD_ERROR_TEST_HLSLI

#include "../common.hlsli"

// TODO: don't hardcode this 
static const float zNear = 10.f;

// See if child should be visited
float2 TestNode(float3x4 renderFromObject, float3x4 cameraFromRender, float4 lodBounds, float maxScale, out float test, bool culled)
{
    // Find length to cluster center
    float3 center = mul(renderFromObject, float4(lodBounds.xyz, 1.f));

    float radius = lodBounds.w * maxScale;
    float distSqr = length2(center);

    // Find angle between vector to cluster center and view vector
    float3 cameraForward = -cameraFromRender[2].xyz;

    float z = dot(cameraForward, center);
    if (culled)
    {
        float zf = abs(dot(cameraForward, center));
        float zr = abs(dot(cameraFromRender[0].xyz, center));
        float zu = abs(dot(cameraFromRender[1].xyz, center));
        z = max(zf, max(zr, zu));
    }

    float x = distSqr - z * z;
    x = sqrt(max(0.f, x));

    // Find angle between vector to cluster center and vector to tangent point on sphere
    float distTangentSqr = distSqr - radius * radius;

    float distTangent = sqrt(max(0.f, distTangentSqr));

    // Find cosine of the above angles subtracted/added
    float invDistSqr = rcp(distSqr);
    float cosSub = (z * distTangent + x * radius) * invDistSqr;
    float cosAdd = (z * distTangent - x * radius) * invDistSqr;

    test = cosAdd;

    // Clipping
    float depth = z - zNear;
    if (distTangentSqr < 0.f || cosSub * distTangent < zNear)
    {
        float cosSubX = max(0.f, x - sqrt(radius * radius - depth * depth));
        cosSub = zNear * rsqrt(cosSubX * cosSubX + zNear * zNear);
    }
    if (distTangentSqr < 0.f || cosAdd * distTangent < zNear)
    {
        float cosAddX = x + sqrt(radius * radius - depth * depth);
        cosAdd = zNear * rsqrt(cosAddX * cosAddX + zNear * zNear);
    }

    float minZ = max(z - radius, zNear);
    float maxZ = max(z + radius, zNear);

    return z + radius > zNear ? float2(minZ * cosAdd, maxZ * cosSub) : float2(0, 0);
}
#endif
