#ifndef BASE_TYPES_H
#define BASE_TYPES_H

//////////////////////////////
// Primitive
//
struct BVH;
struct BVH4;
struct CompressedBVH4;

struct PrimitiveMethods
{
    bool (*Hit)(void *ptr, const Ray &r, const f32 tMin, const f32 tMax, HitRecord &record);
};

extern PrimitiveMethods primitiveMethods[];

struct Primitive : TaggedPointer<BVH, BVH4, CompressedBVH4>
{
    using TaggedPointer::TaggedPointer;
    inline bool Hit(const Ray &r, const f32 tMin, const f32 tMax, HitRecord &record) const
    {
        void *ptr   = GetPtr();
        bool result = primitiveMethods[GetTag()].Hit(ptr, r, tMin, tMax, record);
        return result;
    }
};

//////////////////////////////
// Sampler
//
struct IndependentSampler;
struct StratifiedSampler;
struct SobolSampler;
struct PaddedSobolSampler;
struct ZSobolSampler;

using SamplerTaggedPointer = TaggedPointer<IndependentSampler, StratifiedSampler, SobolSampler, PaddedSobolSampler, ZSobolSampler>;

struct SamplerMethods
{
    i32 (*SamplesPerPixel)(void *);
    void (*StartPixelSample)(void *, vec2i p, i32 index, i32 dimension);
    f32 (*Get1D)(void *);
    vec2 (*Get2D)(void *);
    vec2 (*GetPixel2D)(void *);
};

static SamplerMethods samplerMethods[SamplerTaggedPointer::MaxTag()] = {};

struct Sampler : SamplerTaggedPointer
{
    using TaggedPointer::TaggedPointer;
    inline i32 SamplesPerPixel() const
    {
        void *ptr  = GetPtr();
        u32 tag    = GetTag();
        i32 result = samplerMethods[tag].SamplesPerPixel(ptr);
        return result;
    }
    inline void StartPixelSample(vec2i p, i32 index, i32 dimension = 0)
    {
        void *ptr = GetPtr();
        u32 tag   = GetTag();
        samplerMethods[tag].StartPixelSample(ptr, p, index, dimension);
    }
    inline f32 Get1D()
    {
        void *ptr  = GetPtr();
        u32 tag    = GetTag();
        f32 result = samplerMethods[tag].Get1D(ptr);
        return result;
    }
    inline vec2 Get2D()
    {
        void *ptr   = GetPtr();
        u32 tag     = GetTag();
        vec2 result = samplerMethods[tag].Get2D(ptr);
        return result;
    }
    inline vec2 GetPixel2D()
    {
        void *ptr   = GetPtr();
        u32 tag     = GetTag();
        vec2 result = samplerMethods[tag].GetPixel2D(ptr);
        return result;
    }
};

template <class T>
struct SamplerCRTP
{
    static const i32 samplerID;
    static i32 SamplesPerPixel(void *ptr)
    {
        return static_cast<T *>(ptr)->SamplesPerPixel();
    }
    static void StartPixelSample(void *ptr, vec2i p, i32 index, i32 dimension)
    {
        return static_cast<T *>(ptr)->StartPixelSample(p, index, dimension);
    }
    static f32 Get1D(void *ptr)
    {
        return static_cast<T *>(ptr)->Get1D();
    }
    static vec2 Get2D(void *ptr)
    {
        return static_cast<T *>(ptr)->Get2D();
    }
    static vec2 GetPixel2D(void *ptr)
    {
        return static_cast<T *>(ptr)->GetPixel2D();
    }
    static constexpr i32 Register()
    {
        samplerMethods[SamplerTaggedPointer::TypeIndex<T>()] = {SamplesPerPixel, StartPixelSample, Get1D, Get2D, GetPixel2D};

        return SamplerTaggedPointer::TypeIndex<T>();
    }
};

template <class T>
const i32 SamplerCRTP<T>::samplerID = SamplerCRTP<T>::Register();

template struct SamplerCRTP<IndependentSampler>;
template struct SamplerCRTP<StratifiedSampler>;
template struct SamplerCRTP<SobolSampler>;
template struct SamplerCRTP<PaddedSobolSampler>;
template struct SamplerCRTP<ZSobolSampler>;

//////////////////////////////
// Spectrum
//
struct ConstantSpectrum;
struct DenselySampledSpectrum;

struct SampledSpectrum;
struct SampledWavelengths;

using SpectrumTaggedPointer = TaggedPointer<ConstantSpectrum, DenselySampledSpectrum>;

struct SpectrumMethods
{
    f32 (*Evaluate)(void *, f32);
    f32 (*MaxValue)(void *);
    SampledSpectrum (*Sample)(void *, const SampledWavelengths &);
};

static SpectrumMethods spectrumMethods[SpectrumTaggedPointer::MaxTag()] = {};

struct Spectrum : SpectrumTaggedPointer
{
    using TaggedPointer::TaggedPointer;

    f32 operator()(f32 lambda) const;
    f32 MaxValue() const;
    SampledSpectrum Sample(const SampledWavelengths &lambda) const;
};

template <class T>
struct SpectrumCRTP
{
    static const i32 id;
    static f32 Evaluate(void *ptr, f32 lambda);
    static f32 MaxValue(void *ptr);
    static SampledSpectrum Sample(void *ptr, const SampledWavelengths &lambda);
    static constexpr i32 Register()
    {
        spectrumMethods[SpectrumTaggedPointer::TypeIndex<T>()] = {Evaluate, MaxValue, Sample};
        return SpectrumTaggedPointer::TypeIndex<T>();
    }
};

template <class T>
const i32 SpectrumCRTP<T>::id = SpectrumCRTP<T>::Register();

template struct SpectrumCRTP<ConstantSpectrum>;
template struct SpectrumCRTP<DenselySampledSpectrum>;

#endif
