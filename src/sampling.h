inline vec2 SampleUniformDiskPolar(vec2 u)
{
    f32 r     = sqrtf(u[0]);
    f32 theta = 2 * PI * u[1];
    return vec2(r * cosf(theta), r * sinf(theta));
}

inline vec2 InvertUniformDiskPolarSample(vec2 sample)
{
    f32 theta = atan2f(sample.y, sample.x);
    if (theta < 0) theta += 2 * PI;
    return vec2(sample.x * sample.x + sample.y * sample.y, theta / (2 * PI));
}

inline vec2 SampleUniformDiskConcentric(vec2 u)
{
    vec2 uOffset = 2 * u - vec2(1, 1);
    if (uOffset.x == 0 && uOffset.y == 0)
    {
        return vec2(0, 0);
    }

    f32 theta, r;
    if (std::abs(uOffset.x) > std::abs(uOffset.y))
    {
        r     = uOffset.x;
        theta = PI / 4 * (uOffset.y / uOffset.x);
    }
    else
    {
        r     = uOffset.y;
        theta = PI / 2 - PI / 4 * (uOffset.x / uOffset.y);
    }
    return r * vec2(std::cos(theta), std::sin(theta));
}

inline vec3 SampleUniformHemisphere(vec2 u)
{
    f32 z   = u[0];
    f32 r   = SafeSqrt(1 - z * z);
    f32 phi = 2 * PI * u[1];
    return {r * std::cos(phi), r * std::sin(phi), z};
}

inline f32 UniformHemispherePDF()
{
    return Inv2Pi;
}

inline vec2 InvertUniformHemisphereSample(vec3 w)
{
    f32 phi = atan2f(w.y, w.x);
    if (phi < 0) phi += 2 * PI;
    return vec2(w.z, phi / (2 * PI));
}

inline vec3 SampleUniformSphere(vec2 u)
{
    f32 z   = 1 - 2 * u[0];
    f32 r   = SafeSqrt(1 - z * z);
    f32 phi = 2 * PI * u[1];
    return {r * cosf(phi), r * sinf(phi), z};
}

inline f32 UniformSpherePDF()
{
    return Inv4Pi;
}

inline vec2 InvertUniformSphereSample(vec3 w)
{
    f32 phi = atan2f(w.y, w.x);
    if (phi < 0) phi += 2 * PI;
    return vec2((1 - w.z) / 2.f, phi / (2 * PI));
}

// using malley method
inline vec3 SampleCosineHemisphere(vec2 u)
{
    vec2 d = SampleUniformDiskConcentric(u);
    f32 z  = SafeSqrt(1 - d.x * d.x - d.y * d.y);
    return vec3(d.x, d.y, z);
}

inline f32 CosineHemispherePDF(f32 cosTheta)
{
    return cosTheta * InvPi;
}

// inline vec2 InvertCosineHemisphereSample(vec3 w)
// {
// return InvertUniformDisk
// }
