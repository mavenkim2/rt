#ifndef SAMPLING_HLSI_
#define SAMPLING_HLSI_

#include "common.hlsli"

struct RNG
{
    static uint PCG(uint x)
    {
        uint state = x * 747796405u + 2891336453u;
        uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
        return (word >> 22u) ^ word;
    }

    // Ref: M. Jarzynski and M. Olano, "Hash Functions for GPU Rendering," Journal of Computer Graphics Techniques, 2020.
    static uint3 PCG3d(uint3 v)
    {
        v = v * 1664525u + 1013904223u;
        v.x += v.y * v.z;
        v.y += v.z * v.x;
        v.z += v.x * v.y;
        v ^= v >> 16u;
        v.x += v.y * v.z;
        v.y += v.z * v.x;
        v.z += v.x * v.y;
        return v;
    }

    // Ref: M. Jarzynski and M. Olano, "Hash Functions for GPU Rendering," Journal of Computer Graphics Techniques, 2020.
    static uint4 PCG4d(uint4 v)
    {
        v = v * 1664525u + 1013904223u;
        v.x += v.y * v.w; 
        v.y += v.z * v.x; 
        v.z += v.x * v.y; 
        v.w += v.y * v.z;
        v ^= v >> 16u;
        v.x += v.y * v.w; 
        v.y += v.z * v.x; 
        v.z += v.x * v.y; 
        v.w += v.y * v.z;
        return v;
    }

    static RNG Init(uint2 pixel, uint frame)
    {
        RNG rng;
#if 0
        rng.State = RNG::PCG(pixel.x + RNG::PCG(pixel.y + RNG::PCG(frame)));
#else
        rng.State = RNG::PCG3d(uint3(pixel, frame)).x;
#endif

        return rng;
    }

    static RNG Init(uint2 pixel, uint frame, uint idx)
    {
        RNG rng;
        rng.State = RNG::PCG4d(uint4(pixel, frame, idx)).x;

        return rng;
    }

    static RNG Init(uint idx, uint frame)
    {
        RNG rng;
        rng.State = rng.PCG(idx + PCG(frame));

        return rng;
    }

    static RNG Init(uint seed)
    {
        RNG rng;
        rng.State = seed;

        return rng;
    }

    uint UniformUint()
    {
        this.State = this.State * 747796405u + 2891336453u;
        uint word = ((this.State >> ((this.State >> 28u) + 4u)) ^ this.State) * 277803737u;

        return (word >> 22u) ^ word;
    }

    float Uniform() 
    {
#if 0
    	return asfloat(0x3f800000 | (UniformUint() >> 9)) - 1.0f;
#else
        // For 32-bit floats, any integer in [0, 2^24] can be represented exactly and 
        // there may be rounding errors for anything larger, e.g. 2^24 + 1 is rounded 
        // down to 2^24. 
        // Given random integers, we can right shift by 8 bits to get integers in 
        // [0, 2^24 - 1]. After division by 2^-24, we have uniform numbers in [0, 1).
        // Ref: https://prng.di.unimi.it/
        return float(UniformUint() >> 8) * 0x1p-24f;
#endif
    }

    // Returns samples in [0, bound)
    uint UniformUintBounded(uint bound)
    {
        uint32_t threshold = (~bound + 1u) % bound;

        for (;;) 
        {
            uint32_t r = UniformUint();

            if (r >= threshold)
                return r % bound;
        }        
    }

    // Returns samples in [0, bound). Biased but faster than #UniformUintBounded(): 
    // https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle
    uint UniformUintBounded_Faster(uint bound)
    {
        return (uint)(Uniform() * float(bound));
    }

    float2 Uniform2D()
    {
        float u0 = Uniform();
        float u1 = Uniform();

        return float2(u0, u1);
    }

    float3 Uniform3D()
    {
        float u0 = Uniform();
        float u1 = Uniform();
        float u2 = Uniform();

        return float3(u0, u1, u2);
    }

    float4 Uniform4D()
    {
        float u0 = Uniform();
        float u1 = Uniform();
        float u2 = Uniform();
        float u3 = Uniform();

        return float4(u0, u1, u2, u3);
    }

    uint State;
};

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

float2 SampleUniformDiskPolar(float2 u)
{
    float r     = sqrt(u[0]);
    float theta = 2 * PI * u[1];
    return float2(r * cos(theta), r * sin(theta));
}

float3 SampleUniformTriangle(float2 u)
{
    float3 result;
    if (u[0] < u[1])
    {
        result[0] = u[0] / 2;
        result[1] = u[1] - result[0];
    }
    else
    {
        result[1] = u[1] / 2;
        result[0] = u[0] - result[1];
    }
    result[2] = 1 - result[0] - result[1];
    return result;
}

float SampleLinear(float u, float a, float b)
{
    static float oneMinusEpsilon = 0x1.fffffep-1;
    float mask = u == 0 && a == 0;
    float x    = mask ? 0 : u * (a + b) / (a + sqrt(lerp(a * a, b * b, u)));
    return min(x, oneMinusEpsilon);
}

float Bilerp(float2 u, float4 w)
{
    float result = lerp(lerp(w[0], w[2], u[1]), lerp(w[1], w[3], u[1]), u[0]);
    return result;
}

float BilinearPDF(float2 u, float4 w)
{
    float zeroMask = u.x < 0 || u.x > 1 || u.y < 0 || u.y > 1;
    float denom    = w[0] + w[1] + w[2] + w[3];
    float oneMask  = denom == 0;
    float result   = zeroMask ? 0 : (oneMask ? 1 :  4 * Bilerp(u, w) / denom);
    return result;
}

float2 SampleBilinear(float2 u, float4 w)
{
    float2 result;
    result.y = SampleLinear(u[1], w[0] + w[1], w[2] + w[3]);
    result.x = SampleLinear(u[0], lerp(w[0], w[2], result.y), lerp(w[1], w[3], result.y));
    return result;
}


float SphericalQuadArea(float3 a, float3 b, float3 c, float3 d)
{
    float3 axb = normalize(cross(a, b));
    float3 bxc = normalize(cross(b, c));
    float3 cxd = normalize(cross(c, d));
    float3 dxa = normalize(cross(d, a));

    float g0 = AngleBetween(-axb, bxc);
    float g1 = AngleBetween(-bxc, cxd);
    float g2 = AngleBetween(-cxd, dxa);
    float g3 = AngleBetween(-dxa, axb);
    return abs(g0 + g1 + g2 + g3 - 2 * PI);
}

float3 SampleSphericalRectangle(float3 p, float3 base, float3 eu,
                                float3 ev, float2 samples, out float outPdf)
{
    float euLength = length(eu);
    float evLength = length(ev);

    // Calculate local coordinate system where sampling is done
    // NOTE: rX and rY must be perpendicular
    float3 rX = eu / euLength;
    float3 rY = ev / evLength;
    float3 rZ = cross(rX, rY);

    float3 d0  = base - p;
    float x0 = dot(d0, rX);
    float y0 = dot(d0, rY);
    float z0 = dot(d0, rZ);
    if (z0 > 0)
    {
        z0 *= -1.f;
        rZ *= float(-1.f);
    }

    float x1 = x0 + euLength;
    float y1 = y0 + evLength;

    float3 v00 = float3(x0, y0, z0);
    float3 v01 = float3(x0, y1, z0);
    float3 v10 = float3(x1, y0, z0);
    float3 v11 = float3(x1, y1, z0);

    // Compute normals to edges (i.e, normal of plane containing edge and p)
    float3 n0 = normalize(cross(v00, v10));
    float3 n1 = normalize(cross(v10, v11));
    float3 n2 = normalize(cross(v11, v01));
    float3 n3 = normalize(cross(v01, v00));

    // Calculate the angle between the plane normals
    float g0 = AngleBetween(-n0, n1);
    float g1 = AngleBetween(-n1, n2);
    float g2 = AngleBetween(-n2, n3);
    float g3 = AngleBetween(-n3, n0);

    // Compute solid angle subtended by rectangle
    float k = 2 * PI - g2 - g3;
    float S = g0 + g1 - k;

    if (S <= 0)
    {
        outPdf = 0.f;
        return 0;
    }
    outPdf = 1.f / S;

    float b0 = n0.z;
    float b1 = n2.z;

    // Compute cu
    // float au = samples[0] * S + k;
    float au = samples[0] * (g0 + g1 - 2 * PI) + (samples[0] - 1) * (g2 + g3);
    float fu = (cos(au) * b0 - b1) / sin(au);
    float cu = clamp(copysign(1 / sqrt(fu * fu + b0 * b0), fu), -1.f, 1.f);

    // Compute xu
    float xu = -(cu * z0) / sqrt(1.f - cu * cu);
    xu          = clamp(xu, x0, x1);
    // Compute yv
    float d  = sqrt(xu * xu + z0 * z0);
    float h0 = y0 / sqrt(d * d + y0 * y0);
    float h1 = y1 / sqrt(d * d + y1 * y1);
    // Linearly interpolate between h0 and h1
    float hv   = h0 + (h1 - h0) * samples[1];
    float hvsq = hv * hv;
    float yv   = (hvsq < 1 - 1e-6f) ? (hv * d / sqrt(1 - hvsq)) : y1;
    // Convert back to world space
    return p + rX * xu + rY * yv + rZ * z0;
}

float3 SampleUniformSphere(float2 u)
{
    float z   = 1 - 2 * u[0];
    float r   = sqrt(1 - z * z);
    float phi = 2 * PI * u[1];
    return float3(r * cos(phi), r * sin(phi), z);
}

float3 SampleCosineHemisphere(float2 u)
{
    float2 d = SampleUniformDiskConcentric(u);
    float z = sqrt(1 - d.x * d.x - d.y * d.y);
    return float3(d.x, d.y, z);
}

float CosineHemispherePDF(float cosTheta)
{
    return cosTheta * InvPi;
}
#endif
