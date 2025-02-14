namespace rt
{

Vec3f SampleUniformTriangle(const Vec2f &u)
{
    Vec3f result;
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

template <typename T>
T BilinearPDF(const Vec2<T> &u, const Vec4<T> &w)
{
    T zeroMask = u.x < 0 || u.x > 1 || u.y < 0 || u.y > 1;
    T denom    = w[0] + w[1] + w[2] + w[3];
    T oneMask  = denom == 0;
    T result   = Select(zeroMask, T(0), Select(oneMask, T(1), 4 * Bilerp(u, w) / denom));
    return result;
}

template <typename T>
Vec2<T> SampleBilinear(const Vec2<T> &u, const Vec4<T> &w)
{
    Vec2<T> result;
    result.y = SampleLinear(u[1], w[0] + w[1], w[2] + w[3]);
    result.x = SampleLinear(u[0], Lerp(result.y, w[0], w[2]), Lerp(result.y, w[1], w[3]));
    return result;
}

LaneNF32 SphericalQuadArea(const Vec3lfn &a, const Vec3lfn &b, const Vec3lfn &c,
                           const Vec3lfn &d)
{
    Vec3lfn axb = Normalize(Cross(a, b));
    Vec3lfn bxc = Normalize(Cross(b, c));
    Vec3lfn cxd = Normalize(Cross(c, d));
    Vec3lfn dxa = Normalize(Cross(d, a));

    LaneNF32 g0 = AngleBetween(-axb, bxc);
    LaneNF32 g1 = AngleBetween(-bxc, cxd);
    LaneNF32 g2 = AngleBetween(-cxd, dxa);
    LaneNF32 g3 = AngleBetween(-dxa, axb);
    return Abs(g0 + g1 + g2 + g3 - 2 * PI);
}

Vec3lfn SampleSphericalRectangle(const Vec3lfn &p, const Vec3lfn &base, const Vec3lfn &eu,
                                 const Vec3lfn &ev, const Vec2lfn &samples, LaneNF32 *pdf)
{
    LaneNF32 euLength = Length(eu);
    LaneNF32 evLength = Length(ev);

    // Calculate local coordinate system where sampling is done
    // NOTE: rX and rY must be perpendicular
    Vec3lfn rX = eu / euLength;
    Vec3lfn rY = ev / evLength;
    Vec3lfn rZ = Cross(rX, rY);

    Vec3lfn d0  = base - p;
    LaneNF32 x0 = Dot(d0, rX);
    LaneNF32 y0 = Dot(d0, rY);
    LaneNF32 z0 = Dot(d0, rZ);
    if (z0 > 0)
    {
        z0 *= -1.f;
        rZ *= LaneNF32(-1.f);
    }

    LaneNF32 x1 = x0 + euLength;
    LaneNF32 y1 = y0 + evLength;

    Vec3lfn v00(x0, y0, z0);
    Vec3lfn v01(x0, y1, z0);
    Vec3lfn v10(x1, y0, z0);
    Vec3lfn v11(x1, y1, z0);

    // Compute normals to edges (i.e, normal of plane containing edge and p)
    Vec3lfn n0 = Normalize(Cross(v00, v10));
    Vec3lfn n1 = Normalize(Cross(v10, v11));
    Vec3lfn n2 = Normalize(Cross(v11, v01));
    Vec3lfn n3 = Normalize(Cross(v01, v00));

    // Calculate the angle between the plane normals
    LaneNF32 g0 = AngleBetween(-n0, n1);
    LaneNF32 g1 = AngleBetween(-n1, n2);
    LaneNF32 g2 = AngleBetween(-n2, n3);
    LaneNF32 g3 = AngleBetween(-n3, n0);

    // Compute solid angle subtended by rectangle
    LaneNF32 k = TwoPi * PI - g2 - g3;
    LaneNF32 S = g0 + g1 - k;
    *pdf       = 1.f / S;

    LaneNF32 b0 = n0.z;
    LaneNF32 b1 = n2.z;

    // Compute cu
    // LaneNF32 au = samples[0] * S + k;
    LaneNF32 au = samples[0] * (g0 + g1 - TwoPi) + (samples[0] - 1) * (g2 + g3);
    LaneNF32 fu = (Cos(au) * b0 - b1) / Sin(au);
    LaneNF32 cu = Clamp(Copysign(1 / Sqrt(fu * fu + b0 * b0), fu), -1.f, 1.f);

    // Compute xu
    LaneNF32 xu = -(cu * z0) / Sqrt(1.f - cu * cu);
    xu          = Clamp(xu, x0, x1);
    // Compute yv
    LaneNF32 d  = Sqrt(xu * xu + z0 * z0);
    LaneNF32 h0 = y0 / Sqrt(d * d + y0 * y0);
    LaneNF32 h1 = y1 / Sqrt(d * d + y1 * y1);
    // Linearly interpolate between h0 and h1
    LaneNF32 hv   = h0 + (h1 - h0) * samples[1];
    LaneNF32 hvsq = hv * hv;
    LaneNF32 yv   = (hvsq < 1 - 1e-6f) ? (hv * d / Sqrt(1 - hvsq)) : y1;
    // Convert back to world space
    return p + rX * xu + rY * yv + rZ * z0;
}

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
    Vec2<T> uOffset = Vec2<T>(2.f * u) - Vec2<T>(1);

    if (All(uOffset.x == 0) && All(uOffset.y == 0))
    {
        return Vec2<T>(0, 0);
    }

    Mask<T> mask = Abs(uOffset.x) > Abs(uOffset.y);
    T r          = Select(mask, uOffset.x, uOffset.y);
    T theta      = Select(mask, PI / 4 * (uOffset.y / uOffset.x),
                          PI / 2 - PI / 4 * (uOffset.x / uOffset.y));

    Vec2<T> result = Select(uOffset.x == 0 && uOffset.y == 0, Vec2<T>(0, 0),
                            r * Vec2<T>(Cos(theta), Sin(theta)));
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

inline f32 UniformHemispherePDF() { return Inv2Pi; }

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

inline f32 UniformSpherePDF() { return Inv4Pi; }

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

inline Vec3f RandomUnitVector(Vec2f u) { return Normalize(SampleUniformSphere(u)); }

inline Vec3f RandomUnitVector()
{
    Vec2f u = Vec2f(RandomFloat(), RandomFloat());
    return RandomUnitVector(u);
}

inline Vec3f RandomOnHemisphere(const Vec3f &normal)
{
    // NOTE: why can't you just normalize a vector that has a length > 1?
    Vec3f result = RandomUnitVector();
    result       = Dot(normal, result) > 0 ? result : -result;
    return result;
}

inline Vec3f RandomInUnitDisk()
{
    Vec2f u = Vec2f(RandomFloat(), RandomFloat());
    return Vec3f(SampleUniformDiskConcentric(u), 0.f);
}

// inline Vec2f InvertCosineHemisphereSample(Vec3f w)
// {
// return InvertUniformDisk
// }
} // namespace rt
