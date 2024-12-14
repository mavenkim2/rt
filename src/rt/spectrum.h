#ifndef SPECTRUM_H
#define SPECTRUM_H

// #include "../build/rgbspectrum_srgb.cpp"
namespace rt
{

constexpr f32 LambdaMin           = 360;
constexpr f32 LambdaMax           = 830;
constexpr u32 NSampledWavelengths = 4;

static constexpr f32 CIE_Y_integral = 106.856895f;

namespace Spectra
{
const DenselySampledSpectrum &X();
const DenselySampledSpectrum &Y();
const DenselySampledSpectrum &Z();
} // namespace Spectra

// Compute emitted radiance of a blackbody emitter given an input temperature T and wavelength
// lambda
f32 Blackbody(f32 lambda, f32 T)
{
    if (T <= 0) return 0;
    // Speed of light
    const f32 c = 299792458.f;
    // Planck's constant
    const f32 h = 6.62606957e-34f;
    // Boltzmann constant
    const f32 kb = 1.3806488e-23f;
    f32 l        = lambda * 1e-9f;
    f32 Le       = (2 * h * c * c) / (Pow<5>(l) * (FastExp((h * c) / (l * kb * T)) - 1));
    assert(!IsNaN(Le));
    return Le;
}

struct Spectrum;
struct RGBColorSpace;

template <typename T>
struct SampledSpectrumBase
{
    SampledSpectrumBase() : SampledSpectrumBase(0.f) {}
    explicit SampledSpectrumBase(const T &c)
    {
        for (u32 i = 0; i < NSampledWavelengths; i++)
        {
            values[i] = c;
        }
    }
    explicit SampledSpectrumBase(const T *v)
    {
        for (u32 i = 0; i < NSampledWavelengths; ++i)
        {
            values[i] = v[i];
        }
    }

    Vec3f ToXYZ(const SampledWavelengths &lambda) const;
    Vec3f ToRGB(const RGBColorSpace &space, const SampledWavelengths &lambda) const;
    T operator[](i32 i) const { return values[i]; }

    T &operator[](i32 i) { return values[i]; }
    SampledSpectrumBase &operator+=(const SampledSpectrumBase<T> &s)
    {
        for (i32 i = 0; i < NSampledWavelengths; i++)
        {
            values[i] += s.values[i];
        }
        return *this;
    }
    SampledSpectrumBase operator+(const SampledSpectrumBase<T> &s) const
    {
        SampledSpectrumBase ret = *this;
        return ret += s;
    }
    SampledSpectrumBase &operator-=(const SampledSpectrumBase<T> &s)
    {
        for (i32 i = 0; i < NSampledWavelengths; i++)
        {
            values[i] -= s.values[i];
        }
        return *this;
    }
    SampledSpectrumBase operator-(const SampledSpectrumBase<T> &s) const
    {
        SampledSpectrumBase ret = *this;
        return ret -= s;
    }
    SampledSpectrumBase operator+(f32 a) const
    {
        assert(!IsNaN(a));
        SampledSpectrumBase ret;
        for (i32 i = 0; i < NSampledWavelengths; i++)
        {
            ret.values[i] += a;
        }
        return ret;
    }
    friend SampledSpectrumBase operator-(f32 a, const SampledSpectrumBase<T> &s)
    {
        assert(!IsNaN(a));
        SampledSpectrumBase ret;
        for (i32 i = 0; i < NSampledWavelengths; i++)
        {
            ret.values[i] = a - s.values[i];
        }
        return ret;
    }
    SampledSpectrumBase &operator*=(const SampledSpectrumBase<T> &s)
    {
        for (i32 i = 0; i < NSampledWavelengths; i++)
        {
            values[i] *= s.values[i];
        }
        return *this;
    }
    SampledSpectrumBase operator*(const SampledSpectrumBase<T> &s) const
    {
        SampledSpectrumBase ret = *this;
        return ret *= s;
    }
    SampledSpectrumBase operator*(f32 a) const
    {
        assert(!IsNaN(a));
        SampledSpectrumBase ret = *this;
        for (i32 i = 0; i < NSampledWavelengths; i++)
        {
            ret.values[i] *= a;
        }
        return ret;
    }
    SampledSpectrumBase &operator*=(f32 a)
    {
        assert(!IsNaN(a));
        for (i32 i = 0; i < NSampledWavelengths; i++)
        {
            values[i] *= a;
        }
        return *this;
    }
    friend SampledSpectrumBase operator*(f32 a, const SampledSpectrumBase<T> &s)
    {
        return s * a;
    }
    friend SampledSpectrumBase operator+(f32 a, const SampledSpectrumBase<T> &s)
    {
        return s + a;
    }

    SampledSpectrumBase &operator/=(const SampledSpectrumBase<T> &s)
    {
        for (int i = 0; i < NSampledWavelengths; ++i)
        {
            assert(s.values[i] != 0.f);
            values[i] /= s.values[i];
        }
        return *this;
    }
    SampledSpectrumBase operator/(const SampledSpectrumBase<T> &s) const
    {
        SampledSpectrumBase ret = *this;
        return ret /= s;
    }
    SampledSpectrumBase &operator/=(f32 a)
    {
        assert(a != 0.f);
        assert(!IsNaN(a));
        for (i32 i = 0; i < NSampledWavelengths; ++i) values[i] /= a;
        return *this;
    }
    SampledSpectrumBase operator/(f32 a) const
    {
        SampledSpectrumBase ret = *this;
        return ret /= a;
    }
    SampledSpectrumBase operator-() const
    {
        SampledSpectrumBase ret;
        for (i32 i = 0; i < NSampledWavelengths; ++i) ret.values[i] = -values[i];
        return ret;
    }
    bool operator==(const SampledSpectrumBase<T> &s) const { return values == s.values; }
    bool operator!=(const SampledSpectrumBase<T> &s) const { return values != s.values; }
    bool HasNaNs() const
    {
        for (i32 i = 0; i < NSampledWavelengths; ++i)
            if (IsNaN(values[i])) return true;
        return false;
    }
    // XYZ ToXYZ(const SampledWavelengths &lambda) const;
    // RGB ToRGB(const SampledWavelengths &lambda, const RGBColorSpace &cs) const;
    // f32 y(const SampledWavelengths &lambda) const;

    explicit operator MaskF32() const
    {
        MaskF32 mask(False);
        for (u32 i = 0; i < NSampledWavelengths; ++i)
        {
            mask |= (values[i] != 0);
        }
        return mask;
        // if (values[i] != 0) return true;
        // return false;
    }
    T MinComponentValue() const
    {
        T m = values[0];
        for (u32 i = 1; i < NSampledWavelengths; ++i) m = Min(m, values[i]);
        return m;
    }
    T MaxComponentValue() const
    {
        T m = values[0];
        for (u32 i = 1; i < NSampledWavelengths; ++i) m = Max(m, values[i]);
        return m;
    }
    T Average() const
    {
        T sum = values[0];
        for (u32 i = 1; i < NSampledWavelengths; ++i) sum += values[i];
        return sum / NSampledWavelengths;
    }
    T Sum() const
    {
        T sum = values[0];
        for (u32 i = 1; i < NSampledWavelengths; ++i) sum += values[i];
        return sum;
    }
    void SetInf()
    {
        for (u32 i = 0; i < NSampledWavelengths; ++i) values[i] = pos_inf;
    }

    T values[NSampledWavelengths];
};

typedef SampledSpectrumBase<f32> SampledSpectrum;
typedef SampledSpectrumBase<LaneNF32> SampledSpectrumN;

template <typename T>
SampledSpectrumBase<T> Clamp(const SampledSpectrumBase<T> &s, T &u, T &v)
{
    SampledSpectrumBase<T> result;
    for (u32 i = 0; i < NSampledWavelengths; i++)
    {
        result.values[i] = Clamp(s[i], u, v);
    }
    return result;
}
template <typename T>
SampledSpectrumBase<T> FastExp(const SampledSpectrumBase<T> &a)
{
    SampledSpectrumBase<T> result;
    for (u32 i = 0; i < NSampledWavelengths; i++)
    {
        result.values[i] = FastExp(a.values[i]);
    }
    return result;
}

template <typename T>
SampledSpectrumBase<T> Max(const SampledSpectrumBase<T> &a, const SampledSpectrumBase<T> &b)
{
    SampledSpectrumBase<T> result;
    for (u32 i = 0; i < NSampledWavelengths; ++i)
        result.values[i] = Max(a.values[i], b.values[i]);
    return result;
}

template <typename T>
SampledSpectrumBase<T> SafeDiv(const SampledSpectrumBase<T> &a,
                               const SampledSpectrumBase<T> &b)
{
    SampledSpectrumBase<T> ret;
    for (u32 i = 0; i < NSampledWavelengths; i++)
    {
        ret[i] = (b[i] != 0) ? a[i] / b[i] : 0.f;
    }
    return ret;
}

template <typename T>
SampledSpectrumBase<T> Select(const Mask<T> &mask, const SampledSpectrumBase<T> &a,
                              const SampledSpectrumBase<T> &b)
{
    SampledSpectrumBase<T> ret;
    for (u32 i = 0; i < NSampledWavelengths; i++)
    {
        ret[i] = Select(mask, a[i], b[i]);
    }
    return ret;
}

template <typename T>
struct SampledWavelengthsBase
{
    T lambda[NSampledWavelengths];
    T pdf[NSampledWavelengths];

    SampledWavelengthsBase() {}
    bool operator==(const SampledWavelengthsBase &swl) const
    {
        for (u32 i = 0; i < NSampledWavelengths; i++)
        {
            if (!All(lambda[i] == swl.lambda[i]) || !All(pdf[i] == swl.pdf[i])) return false;
        }
        return true;
    }
    bool operator!=(const SampledWavelengthsBase &swl) const
    {
        for (u32 i = 0; i < NSampledWavelengths; i++)
        {
            if (!All(lambda[i] == swl.lambda[i]) || !All(pdf[i] == swl.pdf[i])) return true;
        }
        return false;
    }
    static SampledWavelengthsBase SampleUniform(f32 u, f32 lambdaMin = LambdaMin,
                                                f32 lambdaMax = LambdaMax)
    {
        SampledWavelengthsBase swl;
        swl.lambda[0] = Lerp(u, lambdaMin, lambdaMax);
        f32 delta     = (lambdaMax - lambdaMin) / NSampledWavelengths;
        for (u32 i = 1; i < NSampledWavelengths; i++)
        {
            swl.lambda[i] = swl.lambda[i - 1] + delta;
            swl.lambda[i] = swl.lambda[i] > lambdaMax ? lambdaMin + swl.lambda[i] - lambdaMax
                                                      : swl.lambda[i];
        }
        for (u32 i = 0; i < NSampledWavelengths; i++)
        {
            swl.pdf[i] = 1 / (lambdaMax - lambdaMin);
        }
        return swl;
    }
    SampledSpectrum PDF() const { return SampledSpectrum(pdf); }
    // NOTE: for dispersion
    void TerminateSecondary()
    {
        MaskF32 mask = SecondaryTerminated();
        if (All(mask)) return;
        for (u32 i = 1; i < NSampledWavelengths; i++)
        {
            pdf[i] = 0.f;
        }
        pdf[0] *= Select(mask, 1.f, 1.f / NSampledWavelengths);
    }
    MaskF32 SecondaryTerminated() const
    {
        MaskF32 mask(true);
        for (u32 i = 1; i < NSampledWavelengths; i++)
        {
            mask &= pdf[i] != 0.f;
        }
        return mask;
    }
    // TODO:
    // void SampleVisible() const {}
    T &operator[](i32 i) { return lambda[i]; }
    T operator[](i32 i) const { return lambda[i]; }
};

typedef SampledWavelengthsBase<f32> SampledWavelengths;
typedef SampledWavelengthsBase<LaneNF32> SampledWavelengthsN;

//////////////////////////////
// Spectrum Implementations
//

struct ConstantSpectrum
{
    ConstantSpectrum() {}
    ConstantSpectrum(f32 c) : c(c) {}
    f32 operator()(f32 lambda) const { return Evaluate(lambda); }
    f32 Evaluate(f32 lambda) const { return c; }
    f32 MaxValue() const { return c; }
    SampledSpectrum Sample(const SampledWavelengths &lambda) const
    {
        return SampledSpectrum(c);
    }
    f32 c;
};

// Spectrum sampled at 1nm increments
struct DenselySampledSpectrum
{
    DenselySampledSpectrum() = default;
    DenselySampledSpectrum(Spectrum spec, i32 lambdaMin = (i32)LambdaMin,
                           i32 lambdaMax = (i32)LambdaMax);
    template <typename F>
    static DenselySampledSpectrum SampleFunction(Arena *arena, F func,
                                                 i32 lambdaMin = (i32)LambdaMin,
                                                 i32 lambdaMax = (i32)LambdaMax)
    {
        DenselySampledSpectrum s(arena, lambdaMin, lambdaMax);
        for (i32 lambda = lambdaMin; lambda <= lambdaMax; lambda++)
        {
            s.values[lambda - lambdaMin] = func(lambda);
        }
    }
    f32 operator()(f32 lambda) const;
    f32 Evaluate(f32 lambda) const;
    f32 MaxValue() const;
    SampledSpectrum Sample(const SampledWavelengths &lambda) const;
    bool operator==(const DenselySampledSpectrum &spec) const;

    // f32 *values;
    u32 numValues;
    f32 values[u32(LambdaMax - LambdaMin) + 1];
    u16 lambdaMin;
    u16 lambdaMax;
};

struct PiecewiseLinearSpectrum
{
    PiecewiseLinearSpectrum() = default;
    PiecewiseLinearSpectrum(f32 *lambdas, f32 *values, u32 numValues)
        : lambdas(lambdas), values(values), numValues(numValues)
    {
    }
    void Scale(f32 s)
    {
        for (u32 i = 0; i < numValues; i++)
        {
            values[i] *= s;
        }
    }
    f32 operator()(f32 lambda) const;
    f32 Evaluate(f32 lambda) const;
    f32 MaxValue() const;
    SampledSpectrum Sample(const SampledWavelengths &lambda) const;
    static PiecewiseLinearSpectrum *FromInterleaved(Arena *arena, const f32 *samples,
                                                    u32 numSamples, bool normalize);

    f32 *values;
    f32 *lambdas;
    u32 numValues;
};

struct BlackbodySpectrum
{
    BlackbodySpectrum(f32 T) : T(T)
    {
        f32 lambdaMax       = 2.89777721e-3f / T;
        normalizationFactor = 1.f / Blackbody(lambdaMax * 1e9f, T);
    }
    f32 operator()(f32 lambda) { return Evaluate(lambda); }
    f32 Evaluate(f32 lambda) const { return Blackbody(lambda, T) * normalizationFactor; }
    SampledSpectrum Sample(const SampledWavelengths &lambda);
    f32 MaxValue() const { return 1.f; }

    f32 T;
    // Using Wien's displacement law, find the wavelength where emission is maximum
    f32 normalizationFactor;
};

struct RGBToSpectrumTable
{
    static constexpr int res = 64;
    using CoefficientArray   = f32[3][res][res][res][3];
    RGBToSpectrumTable(const float *zNodes, const CoefficientArray *coeffs)
        : zNodes(zNodes), coeffs(coeffs)
    {
    }

    Vec3f operator()(Vec3f rgb) const;
    Vec3lfn GetCoeffs(const Vec3lfn &rgb) const;

    static const RGBToSpectrumTable *sRGB;
    static void Init(Arena *arena);

    const f32 *zNodes;
    const CoefficientArray *coeffs;
};

struct RGBColorSpace
{
    Mat3 RGBToXYZ;
    Mat3 XYZToRGB;

    // Color primaries (xy chromaticity coordinates)
    Vec2f r, g, b, w;

    // White point
    DenselySampledSpectrum illuminant;
    const RGBToSpectrumTable *rgbToSpec;

    static const RGBColorSpace *sRGB;
    static void Init(Arena *arena);

    RGBColorSpace(Vec2f r, Vec2f g, Vec2f b, Spectrum illuminant,
                  const RGBToSpectrumTable *rgbToSpec);
    Vec3f ToRGB(Vec3f xyz) const { return Mul(XYZToRGB, xyz); }
    Vec3f ToXYZ(Vec3f rgb) const { return Mul(RGBToXYZ, rgb); }
    Vec3f ToRGBCoeffs(Vec3f rgb) const;
};

// Converting from RGB to spectral:
// https://rgl.s3.eu-central-1.amazonaws.com/media/papers/Jakob2019Spectral_3.pdf
// inline f32 EvaluateSpectral(const Vec3f &c, const f32 wl)
// {
//     // f(x) = S(c0x^2 + c1x + c2), S is a sigmoid, x is the wavelength
//     // S(x) = 1/2 + x/(2 * sqrt(1 + x^2))
//     f32 x = FMA(FMA(c.x, wl, c.y), wl, c.z);
//     return FMA(.5f * x, 1.f / Sqrt(FMA(x, x, 1)), .5f);
// }

template <i32 K>
inline LaneF32<K> EvaluateSpectral(const Vec3<LaneF32<K>> &c, const LaneF32<K> &wl)
{
    // f(x) = S(c0x^2 + c1x + c2), S is a sigmoid, x is the wavelength
    // S(x) = 1/2 + x/(2 * sqrt(1 + x^2))
    LaneF32<K> x      = FMA(FMA(c.x, wl, c.y), wl, c.z);
    LaneF32<K> result = FMA(.5f * x, Rsqrt(FMA(x, x, 1)), .5f);
    return Select(IsInf(result), Select(result > 0, 1.f, 0.f), result);
}

inline f32 SigmoidPolynomialMaxValue(const Vec3f &c)
{
    f32 result = Max(EvaluateSpectral<1>(c, LambdaMin), EvaluateSpectral<1>(c, LambdaMax));
    f32 lambda = -c.y / (2 * c.x);
    if (lambda >= LambdaMin && lambda <= LambdaMax)
    {
        result = Max(result, EvaluateSpectral<1>(c, lambda));
    }
    return result;
}

struct RGBAlbedoSpectrum
{
    f32 operator()(f32 lambda) const { return Evaluate(lambda); }
    f32 Evaluate(f32 lambda) const { return EvaluateSpectral<1>(coeffs, lambda); }
    f32 MaxValue() const { return SigmoidPolynomialMaxValue(coeffs); }
    SampledSpectrum Sample(const SampledWavelengths &lambda) const
    {
        SampledSpectrum s;
        for (i32 i = 0; i < NSampledWavelengths; i++)
        {
            s[i] = EvaluateSpectral<1>(coeffs, lambda[i]);
        }
        return s;
    }
    static SampledSpectrumN Sample(const RGBColorSpace &cs, const Vec3lfn &rgb,
                                   const SampledWavelengthsN &lambda)
    {
        Vec3lfn coeffs;
        for (u32 i = 0; i < IntN; i++)
        {
            Set(coeffs, i) = cs.ToRGBCoeffs(Get(rgb, i));
        }
        SampledSpectrumN s;
        for (i32 i = 0; i < NSampledWavelengths; i++)
        {
            s[i] = EvaluateSpectral<IntN>(coeffs, lambda[i]);
        }
        return s;
    }
    RGBAlbedoSpectrum(const RGBColorSpace &cs, Vec3f rgb) { coeffs = cs.ToRGBCoeffs(rgb); }

    Vec3f coeffs;
};

// For unbounded positive RGB values (remapped to necessary range)
struct RGBUnboundedSpectrum
{
    f32 operator()(f32 lambda) const { return Evaluate(lambda); }
    f32 Evaluate(f32 lambda) const { return scale * EvaluateSpectral<1>(coeffs, lambda); }
    f32 MaxValue() const { return scale * SigmoidPolynomialMaxValue(coeffs); }
    SampledSpectrum Sample(const SampledWavelengths &lambda) const
    {
        SampledSpectrum s;
        for (i32 i = 0; i < NSampledWavelengths; i++)
        {
            s[i] = scale * EvaluateSpectral<1>(coeffs, lambda[i]);
        }
        return s;
    }
    static SampledSpectrumN Sample(const RGBColorSpace &cs, Vec3lfn rgb,
                                   const SampledWavelengthsN &lambda)
    {
        Vec3lfn coeffs;
        LaneNF32 m     = Max(rgb.x, Max(rgb.y, rgb.z));
        LaneNF32 scale = 2 * m;
        rgb            = Select(scale, rgb / scale, Vec3lfn(0));

        for (u32 i = 0; i < IntN; i++)
        {
            Set(coeffs, i) = cs.ToRGBCoeffs(Get(rgb, i));
        }
        SampledSpectrumN s;
        for (i32 i = 0; i < NSampledWavelengths; i++)
        {
            s[i] = EvaluateSpectral<IntN>(coeffs, lambda[i]);
        }
        return s;
    }
    RGBUnboundedSpectrum(const RGBColorSpace &cs, Vec3f rgb)
    {
        f32 m  = Max(rgb.x, Max(rgb.y, rgb.z));
        scale  = 2 * m;
        coeffs = cs.ToRGBCoeffs(scale ? rgb / scale : Vec3f(0, 0, 0));
    }

    f32 scale = 1;
    Vec3f coeffs;
};

struct RGBIlluminantSpectrum
{
    f32 operator()(f32 lambda) const { return Evaluate(lambda); }
    f32 Evaluate(f32 lambda) const
    {
        if (!illuminant) return 0.f;
        return scale * EvaluateSpectral<1>(coeffs, lambda) * (*illuminant)(lambda);
    }
    f32 MaxValue() const
    {
        if (!illuminant) return 0.f;
        return scale * SigmoidPolynomialMaxValue(coeffs) * illuminant->MaxValue();
    }
    SampledSpectrum Sample(const SampledWavelengths &lambda) const
    {
        if (!illuminant) return SampledSpectrum(0.f);
        SampledSpectrum s;
        for (i32 i = 0; i < NSampledWavelengths; i++)
        {
            s[i] = scale * EvaluateSpectral<1>(coeffs, lambda[i]);
        }
        return s * illuminant->Sample(lambda);
    }

    RGBIlluminantSpectrum(const RGBColorSpace &cs, Vec3f rgb) : illuminant(&cs.illuminant)
    {
        f32 m  = Max(rgb.x, Max(rgb.y, rgb.z));
        scale  = 2 * m;
        coeffs = cs.ToRGBCoeffs(scale ? rgb / scale : Vec3f(0, 0, 0));
    }

    f32 scale = 1;
    Vec3f coeffs;
    const DenselySampledSpectrum *illuminant;
};
} // namespace rt

#endif
