#ifndef SPECTRUM_H
#define SPECTRUM_H

constexpr f32 LambdaMin           = 360;
constexpr f32 LambdaMax           = 830;
constexpr u32 NSampledWavelengths = 4;

// Compute emitted radiance of a blackbody emitter given an input temperature T and wavelength lambda
f32 Blackbody(i32 lambda, f32 T)
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
    return Le;
}

struct Spectrum;

struct SampledSpectrum
{
    SampledSpectrum() : SampledSpectrum(0.f) {}
    SampledSpectrum(f32 c)
    {
        for (u32 i = 0; i < NSampledWavelengths; i++)
        {
            values[i] = c;
        }
    }
    f32 &operator[](i32 i)
    {
        return values[i];
    }

    const f32 &operator[](i32 i) const
    {
        return values[i];
    }

    f32 values[NSampledWavelengths];
};

struct SampledWavelengths
{
    f32 &operator[](i32 i)
    {
        return lambda[i];
    }
    const f32 &operator[](i32 i) const
    {
        return lambda[i];
    }
    f32 lambda[NSampledWavelengths];
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

struct PiecewiseLinearSpectrum
{
};

struct BlackbodySpectrum
{
};

#endif
