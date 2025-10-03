#ifndef SAMPLING_HLSI_
#define SAMPLING_HLSI_

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
