#ifndef RAY_CONES_HLSLI
#define RAY_CONES_HLSLI

struct RayCone 
{
    float width;
    float spreadAngle;

    void Propagate(float surfaceSpreadAngle, float tHit)
    {
        width += spreadAngle * tHit;
        spreadAngle += surfaceSpreadAngle;
    }

    float ComputeTextureLOD(float3 p0, float3 p1, float3 p2, 
                            float2 uv0, float2 uv1, float2 uv2, 
                            float3 dir, float3 normal, int2 dimensions)
    {
        const float textureArea = (uv1.x - uv0.x) * (uv2.y - uv0.y) + (uv2.x - uv0.x) * (uv1.y - uv0.y);
        const float triangleArea = length(cross(p1 - p0, p2 - p0));

        float lambda = 0.f;
        lambda += 0.5f * log2(textureArea/triangleArea);
        lambda += log2(abs(width));
        lambda += 0.5f * log2(dimensions.x * dimensions.y);
        lambda -= log2(abs(dot(dir, normal)));
        return lambda;
    }

    float3 CalculateEdgeCurvatures(float3 p0, float3 p1, float3 p2, float3 n0, float3 n1, float3 n2)
    {
        float3 p1p0 = p1 - p0;
        float3 p2p1 = p2 - p1;
        float3 p0p2 = p0 - p2;
        float k01 = dot(n1 - n0, p1p0) / dot(p1p0, p1p0);
        float k12 = dot(n2 - n1, p2p1) / dot(p2p1, p2p1);
        float k20 = dot(n0 - n2, p0p2) / dot(p0p2, p0p2);
        return float3(k01, k12, k20);
    }

    float CalculatePrimaryHitComboSurfaceSpreadAngle()
    {
        return 0.f;
    }

    float CalculatePrimaryHitUnifiedSurfaceSpreadAngle(float3 d, float3 n, float3 p0, float3 p1, float3 p2, 
                                                       float3 n0, float3 n1, float3 n2) 
    {
        float3 k = CalculateEdgeCurvatures(p0, p1, p2, n0, n1, n2);
        float minK = min(k.x, min(k.y, k.z));
        float maxK = max(k.x, max(k.y, k.z));
        float spreadMinK = CalculateSurfaceSpreadAngle(minK, d, n);
        float spreadMaxK = CalculateSurfaceSpreadAngle(maxK, d, n);
        float test0 = abs(spreadAngle + spreadMinK);
        float test1 = abs(spreadAngle + spreadMaxK);
        float result = test0 >= test1 ? spreadMinK : spreadMaxK;
        return result;
    }

    float CalculateSecondaryHitSurfaceSpreadAngle(float3 d, float3 n, float3 p0, float3 p1, float3 p2, 
                                                  float3 n0, float3 n1, float3 n2)
    {
        float3 k = CalculateEdgeCurvatures(p0, p1, p2, n0, n1, n2);
        float result = (k.x + k.y + k.z) / 3.f;
        return result;
    }

    float CalculateSurfaceSpreadAngle(float k, float3 d, float3 n) 
    {
        float bc = -2.f * k * width / dot(d, n);
        return bc;
    }
};

#endif
