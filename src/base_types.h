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
// struct SOAZSobolSampler;

using SamplerTaggedPointer = TaggedPointer<IndependentSampler, StratifiedSampler, SobolSampler,
                                           PaddedSobolSampler, ZSobolSampler>; //, SOAZSobolSampler>;

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
// template struct SamplerCRTP<SOAZSobolSampler>;

//////////////////////////////
// Spectrum
//
struct ConstantSpectrum;
struct DenselySampledSpectrum;
struct PiecewiseLinearSpectrum;
struct BlackbodySpectrum;

struct SampledSpectrum;
struct SampledWavelengths;

using SpectrumTaggedPointer = TaggedPointer<ConstantSpectrum, DenselySampledSpectrum, PiecewiseLinearSpectrum, BlackbodySpectrum>;

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
template struct SpectrumCRTP<PiecewiseLinearSpectrum>;
template struct SpectrumCRTP<BlackbodySpectrum>;

//////////////////////////////
// BSDF
//
struct DiffuseBSDF;
struct ConductorBSDF;
struct DielectricBSDF;
struct ThinDielectricBSDF;

enum class TransportMode;
struct BSDFSample;
enum class BSDFFlags;

using BSDFTaggedPointer = TaggedPointer<DiffuseBSDF, ConductorBSDF, DielectricBSDF, ThinDielectricBSDF>;
struct BSDFMethods
{
    SampledSpectrum (*f)(void *, vec3, vec3, TransportMode);
    BSDFSample (*Sample_f)(void *, vec3, f32, vec2, TransportMode, BSDFFlags);
    f32 (*PDF)(void *, vec3 wo, vec3 wi, TransportMode mode, BSDFFlags flags);
    BSDFFlags (*Flags)(void *);
};

static BSDFMethods bsdfMethods[BSDFTaggedPointer::MaxTag()] = {};

// TODO: because of the way I made this it's probably only valid to use bsdfs through this class, not through any
// of the child bsdfs
struct BSDF : BSDFTaggedPointer
{
    BSDF() = default;
    Frame frame;

    BSDF(vec3 ns, vec3 dpdus) : frame(Frame::FromXZ(Normalize(dpdus), ns)) {}

    SampledSpectrum f(vec3 wo, vec3 wi, TransportMode mode) const;
    BSDFSample Sample_f(vec3 wo, f32 uc, vec2 u, TransportMode mode, BSDFFlags inFlags) const;
    f32 PDF(vec3 wo, vec3 wi, TransportMode mode, BSDFFlags inFlags) const;
    // Hemispherical directional function
    SampledSpectrum rho(vec3 wo, f32 *uc, vec2 *u, u32 numSamples) const;
    // Hemispherical hemispherical function
    SampledSpectrum rho(vec2 *u1, f32 *uc, vec2 *u2, u32 numSamples) const;
    BSDFFlags Flags() const;
};

template <class T>
struct BSDFCRTP
{
    static const i32 id;
    static SampledSpectrum f(void *ptr, vec3 wo, vec3 wi, TransportMode mode);
    static BSDFSample Sample_f(void *ptr, vec3 wo, f32 uc, vec2 u, TransportMode mode, BSDFFlags flags);
    static f32 PDF(void *ptr, vec3 wo, vec3 wi, TransportMode mode, BSDFFlags flags);
    static BSDFFlags Flags(void *ptr);
    static constexpr i32 Register()
    {
        bsdfMethods[BSDFTaggedPointer::TypeIndex<T>()] = {f, Sample_f, PDF, Flags};
        return BSDFTaggedPointer::TypeIndex<T>();
    }
};

template <class T>
const i32 BSDFCRTP<T>::id = BSDFCRTP<T>::Register();

template struct BSDFCRTP<DiffuseBSDF>;
template struct BSDFCRTP<ConductorBSDF>;
template struct BSDFCRTP<DielectricBSDF>;
template struct BSDFCRTP<ThinDielectricBSDF>;

#endif
