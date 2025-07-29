#include "../common.hlsli"
#include "../sampling.hlsli"

struct Brick 
{
    uint64_t bitMask;
    int3 minP;
};

float3 SampleUniformSphere(float2 u)
{
    float z   = 1 - 2 * u[0];
    float r   = sqrt(1 - z * z);
    float phi = 2 * PI * u[1];
    return float3(r * cos(phi), r * sin(phi), z);
}

void CoordinateSystem(float3 v1, out float3 v2, out float v3)
{
    float sign = v1.z > 0.f ? 1.f * -1.f;
    float a    = -1 / (sign + v1.z);
    float b    = v1.x * v1.y * a;

    v2      = float3(1 + sign * v1.x * v1.x * a, sign * b, -sign * v1.x);
    v3      = float3(b, sign + v1.y * v1.y * a, -v1.y);
}

struct SGGX 
{
    float nxx;
    float nxy;
    float nxz;

    float nyy;
    float nyz;

    float nzz;

    float ProjectedArea(float3 wi)
    {
        float3 temp;
        temp.x = nxy * wi.y
        temp.y = nyz * wi.z;
        temp.z = nxz * wi.x;

        temp *= 2.f;

        temp.x += nxx * wi.x;
        temp.y += nyy * wi.y;
        temp.z += nzz * wi.z;
        
        return dot(temp, wi);
    }

    float3 SampleSGGX(float3 w, float2 u) 
    {
        float3 wk;
        float3 wj;
        CoordinateSystem(w, wk, wj);

        float3x3 S = float3x3(nxx, nxy, nxz, nxy, nyy, nyz, nxz, nyz, nzz);

        float skk = ProjectedArea(wk);
        float sjj = ProjectedArea(wj);
        float sii = ProjectedArea(w);

        float3 sw = mul(S, w);

        float ski = dot(wk, sw);
        float skj = dot(wk, mul(S, wj));
        float sji = dot(wj, sw);

        float div = rsqrt(sii);

        float det = determinant(S);
        float val = sqrt(sjj * sii - sji * sji);
        float mjxDiv = 1.f / val;

        float3 mk = sqrt(det) * mjxDiv;
        float3 mj = div * float3(-(ski * sji - skj * sii) * mjxDiv, val, 0.f);
        float3 mi = div * float3(ski, sji, sii);

        float3 point = SampleUniformSphere(u);
        float3 wm = point.x * mk + point.y * mj + point.z * mi;

        wm = wm.x * wk + wm.y * wj + wm.z * w;
        return wm;
    }

    float PhaseFunction(float3 wm, float3 wi)
    {
        return InvPi * dot(wm, wi);
    }
};


void Something(float3 pos, float3 dir, float t, uint primitiveIndex, RNG &rng, inouot float3 throughput)
{
    // dir must be in object space

    uint pageIndex = primitiveIndex >> ?;
    uint clusterIndex;
    uint brickIndex;

    uint basePageAddress = GetClusterPageBaseAddress(pageIndex);
    uint numClusters = GetNumClustersInPage(basePageAddress);
    DenseGeometry dg = GetDenseGeometryHeader(basePageAddress, numClusters, clusterIndex);

    Brick brick = dg.DecodeBrick(brickIndex);

    float density;

    // DDA
    float3 voxelSize;
    float3 rcpVoxelSize = 1.f / voxelSize;

    float3 intersectPos = pos + t * dir;

    // DDA start position
    int3 voxel = int3(intersectPos * rcpVoxelSize) - brick.minP;
    float startT = 0.f;

    int3 step = 0;
    step.x = dir.x > 0.f ? 1 : -1;
    step.y = dir.y > 0.f ? 1 : -1;
    step.z = dir.z > 0.f ? 1 : -1;

    int3 checkStep = 0;
    checkStep.x = dir.x > 0.f ? 1 : 0;
    checkStep.y = dir.y > 0.f ? 1 : 0;
    checkStep.z = dir.z > 0.f ? 1 : 0;

    float3 invDir;

    float3 deltaT = 1.f / (abs(dir) * );
    float3 nextT = float3(voxel + checkStep) * invDir;

    bool terminated = false;

    for (int test = 0; test < 10; test++)
    {
        int voxelIndex = (voxel.z * 4 + voxel.y) * 4 + voxel.x;
        SGGX sggx;

        uint64_t bitMask = brick.bitMask;

        if ((bitMask >> voxelIndex) & 1) break;

        float minT = nextCrossingT.x;
        int axis = 0;
        if (nextCrossingT.y > minT)
        {
            minT = nextCrossingT.y;
            axis = 1;
        }
        if (nextCrossingT.z > minT)
        {
            minT = nextCrossingT.z;
            axis = 2;
        }

        voxel[axis] += step[axis];
        nextCrossingT[axis] += deltaT[axis];
    }

#if 0
        for (;;)
        {
            float nextT = min(next.x, next.y, next.z);

            float density;

            float projectedArea = sggx.ProjectedArea(dir);

            float u = rng.Uniform();
            float newT = t + -log(1 - u) / extinction;

            if (newT < tMax)
            {
                // Sample either scattering or absorption (termination)
                u = rng.Uniform();
                float scattering = projectedArea * density * albedo;
                float extinction = projectedArea * density;

                float p = scattering / extinction;
                if (u < p)
                {
                    // Sample SGGX phase function
                    float3 wm = sggx.SampleSGGX(-dir, rng.Uniform);
                    float3 newDir = SampleCosineHemisphere(rng.Uniform2D());

                    // TODO: I'm pretty sure the PDF and the phase function cancel?
                    //float phase = sggx.PhaseFunction(wm, newDir);

                    //float3 wx, wy;
                    //CoordinateSystem(wm, wx, wy);

                    //throughput *= phase;
                }
                else 
                {
                    terminated = true;
                    break;
                }
            }
            else 
            {
                break;
            }
        }
#endif
}
