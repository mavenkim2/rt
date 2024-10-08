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

// Compute emitted radiance of a blackbody emitter given an input temperature T and wavelength lambda
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

struct SampledSpectrum
{
    SampledSpectrum() : SampledSpectrum(0.f) {}
    explicit SampledSpectrum(f32 c)
    {
        for (u32 i = 0; i < NSampledWavelengths; i++)
        {
            values[i] = c;
        }
    }
    SampledSpectrum(const f32 *v)
    {
        for (u32 i = 0; i < NSampledWavelengths; ++i)
        {
            values[i] = v[i];
        }
    }

    Vec3f ToXYZ(const SampledWavelengths &lambda) const;
    Vec3f ToRGB(const RGBColorSpace &space, const SampledWavelengths &lambda) const;
    f32 operator[](i32 i) const
    {
        return values[i];
    }

    f32 &operator[](i32 i)
    {
        return values[i];
    }
    SampledSpectrum &operator+=(const SampledSpectrum &s)
    {
        for (i32 i = 0; i < NSampledWavelengths; i++)
        {
            values[i] += s.values[i];
        }
        return *this;
    }
    SampledSpectrum operator+(const SampledSpectrum &s) const
    {
        SampledSpectrum ret = *this;
        return ret += s;
    }
    SampledSpectrum &operator-=(const SampledSpectrum &s)
    {
        for (i32 i = 0; i < NSampledWavelengths; i++)
        {
            values[i] -= s.values[i];
        }
        return *this;
    }
    SampledSpectrum operator-(const SampledSpectrum &s) const
    {
        SampledSpectrum ret = *this;
        return ret -= s;
    }
    SampledSpectrum operator+(f32 a) const
    {
        assert(!IsNaN(a));
        SampledSpectrum ret;
        for (i32 i = 0; i < NSampledWavelengths; i++)
        {
            ret.values[i] += a;
        }
        return ret;
    }
    friend SampledSpectrum operator-(f32 a, const SampledSpectrum &s)
    {
        assert(!IsNaN(a));
        SampledSpectrum ret;
        for (i32 i = 0; i < NSampledWavelengths; i++)
        {
            ret.values[i] = a - s.values[i];
        }
        return ret;
    }
    SampledSpectrum &operator*=(const SampledSpectrum &s)
    {
        for (i32 i = 0; i < NSampledWavelengths; i++)
        {
            values[i] *= s.values[i];
        }
        return *this;
    }
    SampledSpectrum operator*(const SampledSpectrum &s) const
    {
        SampledSpectrum ret = *this;
        return ret *= s;
    }
    SampledSpectrum operator*(f32 a) const
    {
        assert(!IsNaN(a));
        SampledSpectrum ret = *this;
        for (i32 i = 0; i < NSampledWavelengths; i++)
        {
            ret.values[i] *= a;
        }
        return ret;
    }
    SampledSpectrum &operator*=(f32 a)
    {
        assert(!IsNaN(a));
        for (i32 i = 0; i < NSampledWavelengths; i++)
        {
            values[i] *= a;
        }
        return *this;
    }
    friend SampledSpectrum operator*(f32 a, const SampledSpectrum &s)
    {
        return s * a;
    }
    friend SampledSpectrum operator+(f32 a, const SampledSpectrum &s)
    {
        return s + a;
    }

    SampledSpectrum &operator/=(const SampledSpectrum &s)
    {
        for (int i = 0; i < NSampledWavelengths; ++i)
        {
            assert(s.values[i] != 0.f);
            values[i] /= s.values[i];
        }
        return *this;
    }
    SampledSpectrum operator/(const SampledSpectrum &s) const
    {
        SampledSpectrum ret = *this;
        return ret /= s;
    }
    SampledSpectrum &operator/=(f32 a)
    {
        assert(a != 0.f);
        assert(!IsNaN(a));
        for (i32 i = 0; i < NSampledWavelengths; ++i)
            values[i] /= a;
        return *this;
    }
    SampledSpectrum operator/(f32 a) const
    {
        SampledSpectrum ret = *this;
        return ret /= a;
    }
    SampledSpectrum operator-() const
    {
        SampledSpectrum ret;
        for (i32 i = 0; i < NSampledWavelengths; ++i)
            ret.values[i] = -values[i];
        return ret;
    }
    bool operator==(const SampledSpectrum &s) const { return values == s.values; }
    bool operator!=(const SampledSpectrum &s) const { return values != s.values; }
    bool HasNaNs() const
    {
        for (i32 i = 0; i < NSampledWavelengths; ++i)
            if (IsNaN(values[i]))
                return true;
        return false;
    }
    // XYZ ToXYZ(const SampledWavelengths &lambda) const;
    // RGB ToRGB(const SampledWavelengths &lambda, const RGBColorSpace &cs) const;
    // f32 y(const SampledWavelengths &lambda) const;

    explicit operator bool() const
    {
        for (u32 i = 0; i < NSampledWavelengths; ++i)
            if (values[i] == 0) return false;
        return true;
    }
    f32 MinComponentValue() const
    {
        f32 m = values[0];
        for (u32 i = 1; i < NSampledWavelengths; ++i)
            m = Min(m, values[i]);
        return m;
    }
    f32 MaxComponentValue() const
    {
        f32 m = values[0];
        for (u32 i = 1; i < NSampledWavelengths; ++i)
            m = Max(m, values[i]);
        return m;
    }
    f32 Average() const
    {
        f32 sum = values[0];
        for (u32 i = 1; i < NSampledWavelengths; ++i)
            sum += values[i];
        return sum / NSampledWavelengths;
    }

    f32 values[NSampledWavelengths];
};

SampledSpectrum SafeDiv(SampledSpectrum a, SampledSpectrum b)
{
    SampledSpectrum ret;
    for (u32 i = 0; i < NSampledWavelengths; i++)
    {
        ret[i] = (b[i] != 0) ? a[i] / b[i] : 0.f;
    }
    return ret;
}

struct SampledWavelengths
{
    bool operator==(const SampledWavelengths &swl) const
    {
        for (u32 i = 0; i < NSampledWavelengths; i++)
        {
            if (lambda[i] != swl.lambda[i] || pdf[i] != swl.pdf[i]) return false;
        }
        return true;
    }
    bool operator!=(const SampledWavelengths &swl) const
    {
        for (u32 i = 0; i < NSampledWavelengths; i++)
        {
            if (lambda[i] != swl.lambda[i] || pdf[i] != swl.pdf[i]) return true;
        }
        return false;
    }
    static SampledWavelengths SampleUniform(f32 u, f32 lambdaMin = LambdaMin, f32 lambdaMax = LambdaMax)
    {
        SampledWavelengths swl;
        swl.lambda[0] = Lerp(u, lambdaMin, lambdaMax);
        f32 delta     = (lambdaMax - lambdaMin) / NSampledWavelengths;
        for (u32 i = 1; i < NSampledWavelengths; i++)
        {
            swl.lambda[i] = swl.lambda[i - 1] + delta;
            swl.lambda[i] = swl.lambda[i] > lambdaMax ? lambdaMin + swl.lambda[i] - lambdaMax : swl.lambda[i];
        }
        for (u32 i = 0; i < NSampledWavelengths; i++)
        {
            swl.pdf[i] = 1 / (lambdaMax - lambdaMin);
        }
        return swl;
    }
    SampledSpectrum PDF() const
    {
        return SampledSpectrum(pdf);
    }
    // NOTE: for dispersion
    void TerminateSecondary()
    {
        if (SecondaryTerminated()) return;
        for (u32 i = 1; i < NSampledWavelengths; i++)
        {
            pdf[i] = 0.f;
        }
        pdf[0] *= 1 / NSampledWavelengths;
    }
    bool SecondaryTerminated() const
    {
        for (u32 i = 1; i < NSampledWavelengths; i++)
        {
            if (pdf[i] != 0.f) return false;
        }
        return true;
    }
    // TODO:
    // void SampleVisible() const {}
    f32 &operator[](i32 i)
    {
        return lambda[i];
    }
    f32 operator[](i32 i) const
    {
        return lambda[i];
    }
    f32 lambda[NSampledWavelengths];
    f32 pdf[NSampledWavelengths];
};

//////////////////////////////
// Spectrum Implementations
//

struct ConstantSpectrum : SpectrumCRTP<ConstantSpectrum>
{
    ConstantSpectrum(f32 c) : c(c) {}
    f32 operator()(f32 lambda) const
    {
        return Evaluate(lambda);
    }
    f32 Evaluate(f32 lambda) const
    {
        return c;
    }
    f32 MaxValue() const
    {
        return c;
    }
    SampledSpectrum Sample(const SampledWavelengths &lambda) const
    {
        return SampledSpectrum(c);
    }
    f32 c;
};

// Spectrum sampled at 1nm increments
struct DenselySampledSpectrum : SpectrumCRTP<DenselySampledSpectrum>
{
    DenselySampledSpectrum(Arena *arena, Spectrum spec, i32 lambdaMin = (i32)LambdaMin, i32 lambdaMax = (i32)LambdaMax);
    template <typename F>
    static DenselySampledSpectrum SampleFunction(Arena *arena, F func, i32 lambdaMin = (i32)LambdaMin, i32 lambdaMax = (i32)LambdaMax)
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

    f32 *values;
    u32 numValues;
    u16 lambdaMin;
    u16 lambdaMax;
};

struct PiecewiseLinearSpectrum : SpectrumCRTP<PiecewiseLinearSpectrum>
{
    PiecewiseLinearSpectrum() = default;
    PiecewiseLinearSpectrum(f32 *lambdas, f32 *values, u32 numValues) : lambdas(lambdas), values(values), numValues(numValues) {}
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
    // static PiecewiseLinearSpectrum *FromInterleaved(Arena *arena, f32 *samples, u32 numSamples, bool normalize);

    f32 *values;
    f32 *lambdas;
    u32 numValues;
};

struct BlackbodySpectrum : SpectrumCRTP<BlackbodySpectrum>
{
    BlackbodySpectrum(f32 T) : T(T)
    {
        f32 lambdaMax       = 2.89777721e-3f / T;
        normalizationFactor = 1.f / Blackbody(lambdaMax * 1e9f, T);
    }
    f32 operator()(f32 lambda)
    {
        return Evaluate(lambda);
    }
    f32 Evaluate(f32 lambda) const
    {
        return Blackbody(lambda, T) * normalizationFactor;
    }
    SampledSpectrum Sample(const SampledWavelengths &lambda);
    f32 MaxValue() const
    {
        return 1.f;
    }

    f32 T;
    // Using Wien's displacement law, find the wavelength where emission is maximum
    f32 normalizationFactor;
};

struct RGBToSpectrumTable
{
    static constexpr int res = 64;
    using CoefficientArray   = f32[3][res][res][res][3];
    RGBToSpectrumTable(const float *zNodes, const CoefficientArray *coeffs) : zNodes(zNodes), coeffs(coeffs) {}

    Vec3f operator()(Vec3f rgb) const;

    static const RGBToSpectrumTable *sRGB;

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

    RGBColorSpace(Arena *arena, Vec2f r, Vec2f g, Vec2f b, Spectrum illuminant, const RGBToSpectrumTable *rgbToSpec);
    Vec3f ToRGB(Vec3f xyz) const
    {
        return Mul(XYZToRGB, xyz);
    }
    Vec3f ToXYZ(Vec3f rgb) const
    {
        return Mul(RGBToXYZ, rgb);
    }
    Vec3f ToRGBCoeffs(Vec3f rgb) const;
};

// Converting from RGB to spectral: https://rgl.s3.eu-central-1.amazonaws.com/media/papers/Jakob2019Spectral_3.pdf
inline f32 EvaluateSpectral(const Vec3f &c, const f32 wl)
{
    // f(x) = S(c0x^2 + c1x + c2), S is a sigmoid, x is the wavelength
    // S(x) = 1/2 + x/(2 * sqrt(1 + x^2))
    f32 x = FMA(FMA(c.x, wl, c.y), wl, c.z);
    return FMA(.5f * x, 1.f / sqrtf(FMA(x, x, 1)), .5f);
}

inline Lane4F32 EvaluateSpectral(const Vec3f &c, const Lane4F32 &wl)
{
    // f(x) = S(c0x^2 + c1x + c2), S is a sigmoid, x is the wavelength
    // S(x) = 1/2 + x/(2 * sqrt(1 + x^2))
    Lane4F32 x = FMA(FMA(c.x, wl, c.y), wl, c.z);
    return FMA(.5f * x, Rsqrt(FMA(x, x, 1)), .5f);
}

inline f32 SigmoidPolynomialMaxValue(const Vec3f &c)
{
    f32 result = Max(EvaluateSpectral(c, LambdaMin), EvaluateSpectral(c, LambdaMax));
    f32 lambda = -c.y / (2 * c.x);
    if (lambda >= LambdaMin && lambda <= LambdaMax)
    {
        result = Max(result, EvaluateSpectral(c, lambda));
    }
    return result;
}

struct RGBAlbedoSpectrum : SpectrumCRTP<RGBAlbedoSpectrum>
{
    f32 operator()(f32 lambda) const
    {
        return Evaluate(lambda);
    }
    f32 Evaluate(f32 lambda) const
    {
        return EvaluateSpectral(coeffs, lambda);
    }
    f32 MaxValue() const
    {
        return SigmoidPolynomialMaxValue(coeffs);
    }
    SampledSpectrum Sample(const SampledWavelengths &lambda) const
    {
        SampledSpectrum s;
        for (i32 i = 0; i < NSampledWavelengths; i++)
        {
            s[i] = EvaluateSpectral(coeffs, lambda[i]);
        }
        return s;
    }
    RGBAlbedoSpectrum(const RGBColorSpace &cs, Vec3f rgb)
    {
        coeffs = cs.ToRGBCoeffs(rgb);
    }

    Vec3f coeffs;
};

// For unbounded positive RGB values (remapped to necessary range)
struct RGBUnboundedSpectrum : SpectrumCRTP<RGBUnboundedSpectrum>
{
    f32 operator()(f32 lambda) const
    {
        return Evaluate(lambda);
    }
    f32 Evaluate(f32 lambda) const
    {
        return scale * EvaluateSpectral(coeffs, lambda);
    }
    f32 MaxValue() const
    {
        return scale * SigmoidPolynomialMaxValue(coeffs);
    }
    SampledSpectrum Sample(const SampledWavelengths &lambda) const
    {
        SampledSpectrum s;
        for (i32 i = 0; i < NSampledWavelengths; i++)
        {
            s[i] = scale * EvaluateSpectral(coeffs, lambda[i]);
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

struct RGBIlluminantSpectrum : SpectrumCRTP<RGBIlluminantSpectrum>
{
    f32 operator()(f32 lambda) const
    {
        return Evaluate(lambda);
    }
    f32 Evaluate(f32 lambda) const
    {
        if (!illuminant) return 0.f;
        return scale * EvaluateSpectral(coeffs, lambda) * (*illuminant)(lambda);
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
            s[i] = scale * EvaluateSpectral(coeffs, lambda[i]);
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
