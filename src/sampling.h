namespace rt 
{
inline Vec2f SampleUniformDiskPolar(Vec2f u)
{
    f32 r     = sqrtf(u[0]);
    f32 theta = 2 * PI * u[1];
    return Vec2f(r * cosf(theta), r * sinf(theta));
}

inline Vec2f InvertUniformDiskPolarSample(Vec2f sample)
{
    f32 theta = atan2f(sample.y, sample.x);
    if (theta < 0) theta += 2 * PI;
    return Vec2f(sample.x * sample.x + sample.y * sample.y, theta / (2 * PI));
}

inline Vec2f SampleUniformDiskConcentric(Vec2f u)
{
    Vec2f uOffset = 2.f * u - Vec2f(1, 1);
    if (uOffset.x == 0 && uOffset.y == 0)
    {
        return Vec2f(0, 0);
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
    return r * Vec2f(std::cos(theta), std::sin(theta));
}

inline Vec3f SampleUniformHemisphere(Vec2f u)
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

inline Vec2f InvertUniformHemisphereSample(Vec3f w)
{
    f32 phi = atan2f(w.y, w.x);
    if (phi < 0) phi += 2 * PI;
    return Vec2f(w.z, phi / (2 * PI));
}

inline Vec3f SampleUniformSphere(Vec2f u)
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

inline Vec2f InvertUniformSphereSample(Vec3f w)
{
    f32 phi = atan2f(w.y, w.x);
    if (phi < 0) phi += 2 * PI;
    return Vec2f((1 - w.z) / 2.f, phi / (2 * PI));
}

// using malley method
inline Vec3f SampleCosineHemisphere(Vec2f u)
{
    Vec2f d = SampleUniformDiskConcentric(u);
    f32 z  = SafeSqrt(1 - d.x * d.x - d.y * d.y);
    return Vec3f(d.x, d.y, z);
}

inline f32 CosineHemispherePDF(f32 cosTheta)
{
    return cosTheta * InvPi;
}

// inline Vec2f InvertCosineHemisphereSample(Vec3f w)
// {
// return InvertUniformDisk
// }
}
