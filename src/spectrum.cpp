DenselySampledSpectrum::DenselySampledSpectrum(Arena *arena, Spectrum spec, i32 lambdaMin, i32 lambdaMax)
    : lambdaMin((u16)lambdaMin), lambdaMax((u16)lambdaMax)
{
    assert(lambdaMin >= LambdaMin && lambdaMax <= LambdaMax);
    numValues = lambdaMax + lambdaMin + 1;
    values    = PushArray(arena, f32, numValues);
    if (spec)
    {
        for (i32 lambda = lambdaMin; lambda <= lambdaMax; lambda++)
        {
            values[lambda - lambdaMin] = spec((f32)lambda);
        }
    }
}

f32 DenselySampledSpectrum::operator()(f32 lambda) const
{
    return Evaluate(lambda);
}

f32 DenselySampledSpectrum::Evaluate(f32 lambda) const
{
    if (lambda < lambdaMin || lambda > lambdaMax) return 0;
    i32 offset = std::lround(lambda) - lambdaMin;
    return values[offset];
}

f32 DenselySampledSpectrum::MaxValue() const
{
    f32 maxValue = 0.f;
    for (u32 i = 0; i < numValues; i++)
    {
        if (values[i] > maxValue) maxValue = values[i];
    }
    return maxValue;
}

SampledSpectrum DenselySampledSpectrum::Sample(const SampledWavelengths &lambda) const
{
    SampledSpectrum s;
    for (i32 i = 0; i < NSampledWavelengths; i++)
    {
        s[i] = Evaluate(lambda[i]);
    }
    return s;
}

bool DenselySampledSpectrum::operator==(const DenselySampledSpectrum &spec) const
{
    if (spec.numValues != numValues || spec.lambdaMin != lambdaMin || spec.lambdaMax != lambdaMax) return false;
    for (u32 i = 0; i < numValues; i++)
    {
        if (values[i] != spec.values[i]) return false;
    }
    return true;
}
