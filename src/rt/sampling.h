namespace rt
{

template <typename T>
inline Vec2<T> SampleUniformDiskPolar(Vec2<T> u)
{
    T r     = Sqrt(u[0]);
    T theta = 2 * PI * u[1];
    return Vec2<T>(r * Cos(theta), r * Sin(theta));
}

template <typename T>
inline Vec2<T> InvertUniformDiskPolarSample(const Vec2<T> &sample)
{
    T theta = Atan(sample.y, sample.x);
    theta += Select(theta < 0, T(2 * PI), T(0));
    return Vec2<T>(sample.x * sample.x + sample.y * sample.y, theta / (2 * PI));
}

template <typename T>
inline Vec2<T> SampleUniformDiskConcentric(const Vec2<T> &u)
{
    Vec2<T> uOffset = T(2.f * u) - Vec2<T>(1, 1);
    if (uOffset.x == 0 && uOffset.y == 0)
    {
        return Vec2<T>(0, 0);
    }

    Mask<T> mask = Abs(uOffset.x) > Abs(uOffset.y);
    T r          = Select(mask, uOffset.x, uOffset.y);
    T theta      = Select(mask, PI / 4 * (uOffset.y / uOffset.x), PI / 2 - PI / 4 * (uOffset.x / uOffset.y));

    Vec2<T> result = Select(uOffset.x == 0 && uOffset.y == 0, Vec2<T>(0, 0), r * Vec2<T>(Cos(theta), Sin(theta)););
    return result;
}

template <typename T>
inline Vec3<T> SampleUniformHemisphere(const Vec2<T> &u)
{
    T z   = u[0];
    T r   = SafeSqrt(1 - z * z);
    T phi = 2 * PI * u[1];
    return {r * Cos(phi), r * Sin(phi), z};
}

inline f32 UniformHemispherePDF()
{
    return Inv2Pi;
}

template <typename T>
inline Vec2<T> InvertUniformHemisphereSample(const Vec3<T> &w)
{
    T phi = Atan2(w.y, w.x);
    phi += Select(phi < 0, T(2 * PI), T(0));
    return Vec2<T>(w.z, phi / (2 * PI));
}

template <typename T>
inline Vec3<T> SampleUniformSphere(const Vec2<T> &u)
{
    T z   = 1 - 2 * u[0];
    T r   = SafeSqrt(1 - z * z);
    T phi = 2 * PI * u[1];
    return {r * Cos(phi), r * Sin(phi), z};
}

inline f32 UniformSpherePDF()
{
    return Inv4Pi;
}

template <typename T>
inline Vec2<T> InvertUniformSphereSample(const Vec3<T> &w)
{
    T phi = Atan2(w.y, w.x);
    phi += Select(phi < 0, T(2 * PI), T(0));
    return Vec2<T>((1 - w.z) / 2, phi / (2 * PI));
}

// using malley method
template <typename T>
inline Vec3<T> SampleCosineHemisphere(const Vec2<T> &u)
{
    Vec2<T> d = SampleUniformDiskConcentric(u);
    T z       = SafeSqrt(1 - d.x * d.x - d.y * d.y);
    return Vec3<T>(d.x, d.y, z);
}

template <typename T>
inline T CosineHemispherePDF(T cosTheta)
{
    return cosTheta * InvPi;
}

// inline Vec2f InvertCosineHemisphereSample(Vec3f w)
// {
// return InvertUniformDisk
// }
} // namespace rt
