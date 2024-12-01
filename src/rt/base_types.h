#ifndef BASE_TYPES_H
#define BASE_TYPES_H

//////////////////////////////
// Primitive
//
namespace rt
{
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
struct ScenePacket;
// struct SOAZSobolSampler;

using SamplerTaggedPointer = TaggedPointer<IndependentSampler, StratifiedSampler, SobolSampler,
                                           PaddedSobolSampler, ZSobolSampler>; //, SOAZSobolSampler>;

struct SamplerMethods
{
    i32 (*SamplesPerPixel)(void *);
    void (*StartPixelSample)(void *, Vec2i p, i32 index, i32 dimension);
    f32 (*Get1D)(void *);
    Vec2f (*Get2D)(void *);
    Vec2f (*GetPixel2D)(void *);
};

static SamplerMethods samplerMethods[SamplerTaggedPointer::MaxTag()] = {};

struct Sampler : SamplerTaggedPointer
{
    using TaggedPointer::TaggedPointer;
    static Sampler Create(Arena *arena, const ScenePacket *packet, const Vec2i fullResolution);
    inline i32 SamplesPerPixel() const
    {
        void *ptr  = GetPtr();
        u32 tag    = GetTag();
        i32 result = samplerMethods[tag].SamplesPerPixel(ptr);
        return result;
    }
    inline void StartPixelSample(Vec2i p, i32 index, i32 dimension = 0)
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
    inline Vec2f Get2D()
    {
        void *ptr    = GetPtr();
        u32 tag      = GetTag();
        Vec2f result = samplerMethods[tag].Get2D(ptr);
        return result;
    }
    inline Vec2f GetPixel2D()
    {
        void *ptr    = GetPtr();
        u32 tag      = GetTag();
        Vec2f result = samplerMethods[tag].GetPixel2D(ptr);
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
    static void StartPixelSample(void *ptr, Vec2i p, i32 index, i32 dimension)
    {
        return static_cast<T *>(ptr)->StartPixelSample(p, index, dimension);
    }
    static f32 Get1D(void *ptr)
    {
        return static_cast<T *>(ptr)->Get1D();
    }
    static Vec2f Get2D(void *ptr)
    {
        return static_cast<T *>(ptr)->Get2D();
    }
    static Vec2f GetPixel2D(void *ptr)
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
struct RGBAlbedoSpectrum;
struct RGBUnboundedSpectrum;
struct RGBIlluminantSpectrum;

struct SampledSpectrum;
struct SampledWavelengths;

using SpectrumTaggedPointer = TaggedPointer<ConstantSpectrum, DenselySampledSpectrum, PiecewiseLinearSpectrum, BlackbodySpectrum,
                                            RGBAlbedoSpectrum, RGBUnboundedSpectrum, RGBIlluminantSpectrum>;

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
template struct SpectrumCRTP<RGBAlbedoSpectrum>;
template struct SpectrumCRTP<RGBUnboundedSpectrum>;
template struct SpectrumCRTP<RGBIlluminantSpectrum>;

//////////////////////////////
// BSDF
//
struct DiffuseBxDF;
struct ConductorBxDF;
struct DielectricBxDF;
struct ThinDielectricBxDF;

enum class TransportMode;
struct BSDFSample;
enum class BxDFFlags;

using BxDFTaggedPointer = TaggedPointer<DiffuseBxDF, ConductorBxDF, DielectricBxDF, ThinDielectricBxDF>;
struct BxDFMethods
{
    SampledSpectrum (*EvaluateSample)(void *, Vec3f, Vec3f, f32 &, TransportMode);
    BSDFSample (*GenerateSample)(void *, Vec3f, f32, Vec2f, TransportMode, BxDFFlags);
    f32 (*PDF)(void *, Vec3f wo, Vec3f wi, TransportMode mode, BxDFFlags flags);
    BxDFFlags (*Flags)(void *);
};

static BxDFMethods bxdfMethods[BxDFTaggedPointer::MaxTag()] = {};

// TODO: because of the way I made this it's probably only valid to use bsdfs through this class, not through any
// of the child bsdfs

struct BxDF : BxDFTaggedPointer
{
    BxDF() {}
    SampledSpectrum EvaluateSample(Vec3f wo, Vec3f wi, f32 &pdf, TransportMode mode) const { return SampledSpectrum(0.f); }
    BSDFSample GenerateSample(Vec3f wo, f32 uc, Vec2f u, TransportMode mode, BxDFFlags inFlags) const { return {}; }
    f32 PDF(Vec3f wo, Vec3f wi, TransportMode mode, BxDFFlags inFlags) const { return 0.f; }
    BxDFFlags Flags() const { return {}; }
};

template <class T>
struct BxDFCRTP
{
    static const i32 id;
    static SampledSpectrum EvaluateSample(void *ptr, Vec3f wo, Vec3f wi, f32 &pdf, TransportMode mode);
    static BSDFSample GenerateSample(void *ptr, Vec3f wo, f32 uc, Vec2f u, TransportMode mode, BxDFFlags flags);
    static f32 PDF(void *ptr, Vec3f wo, Vec3f wi, TransportMode mode, BxDFFlags flags);
    static BxDFFlags Flags(void *ptr);
    static constexpr i32 Register()
    {
        bxdfMethods[BxDFTaggedPointer::TypeIndex<T>()] = {EvaluateSample, GenerateSample, PDF, Flags};
        return BxDFTaggedPointer::TypeIndex<T>();
    }
};

template <class T>
const i32 BxDFCRTP<T>::id = BxDFCRTP<T>::Register();

template struct BxDFCRTP<DiffuseBxDF>;
template struct BxDFCRTP<ConductorBxDF>;
template struct BxDFCRTP<DielectricBxDF>;
template struct BxDFCRTP<ThinDielectricBxDF>;

} // namespace rt
#endif
