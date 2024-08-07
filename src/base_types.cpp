//////////////////////////////
// Primitive Methods
//

PrimitiveMethods primitiveMethods[] = {
    {BVHHit},
    {BVH4Hit},
    {CompressedBVH4Hit},
};

//////////////////////////////
// Spectrum Methods
//

f32 Spectrum::operator()(f32 lambda) const
{
    void *ptr  = GetPtr();
    f32 result = spectrumMethods[GetTag()].Evaluate(ptr, lambda);
    return result;
}

f32 Spectrum::MaxValue() const
{
    void *ptr  = GetPtr();
    f32 result = spectrumMethods[GetTag()].MaxValue(ptr);
    return result;
}

SampledSpectrum Spectrum::Sample(const SampledWavelengths &lambda) const
{
    void *ptr              = GetPtr();
    SampledSpectrum result = spectrumMethods[GetTag()].Sample(ptr, lambda);
    return result;
}

template <class T>
f32 SpectrumCRTP<T>::Evaluate(void *ptr, f32 lambda)
{
    return static_cast<T *>(ptr)->Evaluate(lambda);
}

template <class T>
f32 SpectrumCRTP<T>::MaxValue(void *ptr)
{
    return static_cast<T *>(ptr)->MaxValue();
}

template <class T>
SampledSpectrum SpectrumCRTP<T>::Sample(void *ptr, const SampledWavelengths &lambda)
{
    return static_cast<T *>(ptr)->Sample(lambda);
}
