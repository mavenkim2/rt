#ifndef SHADERS_H
#define SHADERS_H

#ifndef LANE_WIDTH
#define LANE_WIDTH 1
#endif

#ifndef CONCAT
#define CONCAT(a, b) a##b
#endif

// #define COMBINE(a, b) CONCAT(a, b)

#ifndef EXPAND
#define EXPAND(x) x
#endif

#define Vector3              Vec3<LaneF32<EXPAND(LANE_WIDTH)>>
#define Vector4              Vec4<LaneF32<EXPAND(LANE_WIDTH)>>
#define LaneFloat            LaneF32<EXPAND(LANE_WIDTH)>
#define LaneUInt32           LaneU32<EXPAND(LANE_WIDTH)>
#define SurfaceInteractionsN SurfaceInteractions<EXPAND(LANE_WIDTH)>
#define SampledWavelengthsN  SampledWavelengthsBase<EXPAND(LANE_WIDTH)>
#define SampledSpectrumN     SampledSpectrumBase<EXPAND(LANE_WIDTH)>

typedef void *(*MaterialFunc)(Arena *, SurfaceInteractionsN &, LaneUInt32 &, Vector4 &,
                              SampledWavelengthsN &);

#define CREATE_MATERIAL_FUNC(name) void *name(Scene *scene, u32 index, ScenePacket **packets)
typedef CREATE_MATERIAL_FUNC((*CreateMaterialFunc));
// TODO: how do I pass data to this fucker?

struct MaterialFuncs
{
    MaterialFunc eval;
    CreateMaterialFunc create;
};

static MaterialFuncs materialFuncs[NUM_MATERIALS];

#define DIFFUSE_MATERIAL(id, RflShader)                                                       \
    struct CONCAT(DiffuseMaterial, id)                                                        \
    {                                                                                         \
        RflShader reflectance;                                                                \
    };                                                                                        \
    EXPAND(CREATE_MATERIAL_FUNC(CONCAT(CreateFunc, id)))                                      \
    {                                                                                         \
        auto *mt        = (DiffuseMaterial##id *)scene->materials[id] + index;                \
        mt->reflectance = rflShader;                                                          \
    }                                                                                         \
    DiffuseBxDF *CONCAT(DiffuseMaterialFunc, id)(                                             \
        Arena * arena, SurfaceInteractionsN & intr, const LaneUInt32 &mtOffsets,              \
        Vector4 &filterWidths, SampledWavelengthsN &lambda)                                   \
    {                                                                                         \
        RflShader *shaders[LANE_WIDTH];                                                       \
        for (u32 i = 0; i < LANE_WIDTH; i++)                                                  \
        {                                                                                     \
            shaders[i] = &materials[i]->reflectance;                                          \
        }                                                                                     \
        SampledSpectrumN sampledSpectra =                                                     \
            RflShader##Func(intr, shaders, filterWidths, lambda);                             \
        return PushStructConstruct(arena, DiffuseBxDF)(sampledSpectra);                       \
    }                                                                                         \
    static constexpr int Register##id()                                                       \
    {                                                                                         \
        materialFuncs[id] = {CONCAT(DiffuseMaterialFunc, id), CONCAT(CreateFunc, id)};        \
        return id;                                                                            \
    }                                                                                         \
    static const int dummy##id = Register##id();

DIFFUSE_MATERIAL(0, alb);

#define DIFFUSE_TRMT_MATERIAL(id, RflShader, TrmShader)                                       \
    struct CONCAT(DiffuseTransmissionMaterial, id)                                            \
    {                                                                                         \
        RflShader rflShader;                                                                  \
        TrmShader trmShader;                                                                  \
    };                                                                                        \
    void CONCAT(CREATE_FUNC, id)(Scene * scene, u32 index, ScenePacket * *packets) {}         \
    DiffuseTransmissionBxDF *CONCAT(DiffuseTransmissionMaterialFunc, id)(                     \
        Arena * arena, SurfaceInteractions<LANE_WIDTH> & intr,                                \
        DiffuseTransmissionMaterial##id * *mt, Vec4lfn & filterWidths,                        \
        SampledWavelengths<LANE_WIDTH> & lambda)                                              \
    {                                                                                         \
        RflShader *rflShaders[LANE_WIDTH];                                                    \
        TrmShader *trmShaders[LANE_WIDTH];                                                    \
        for (u32 i = 0; i < LANE_WIDTH; i++)                                                  \
        {                                                                                     \
            rflShaders[i] = &mt[i]->rflShader;                                                \
            trmShaders[i] = &mt[i]->trmShaders;                                               \
        }                                                                                     \
        SampledSpectrumN r = RflShader##Func(intr, rflShaders, filterWidths, lambda);         \
        SampledSpectrumN t = TrmShader##Func(intr, trmShaders, filterWidths, lambda);         \
        return DiffuseTransmissionBxDF(r, t);                                                 \
    }                                                                                         \
    static constexpr int Register##id()                                                       \
    {                                                                                         \
        materialFuncs[id] = {CONCAT(DiffuseTransmissionMaterialFunc, id),                     \
                             CONCAT(CreateFunc, id)};                                         \
        return id;                                                                            \
    }                                                                                         \
    static const int dummy##id = Register##id();

#define DIELECTRIC_MATERIAL(id, RoughnessTexture, IORShader)                                  \
    struct CONCAT(DielectricMaterial, id)                                                     \
    {                                                                                         \
        RoughnessTexture roughness;                                                           \
        SpectrumIn ior;                                                                       \
    };                                                                                        \
    DielectricBxDF *CONCAT(DielectricMaterialFunc, id)(                                       \
        Arena * arena, SurfaceInteractions<LANE_WIDTH> & intr, DielectricMaterial##id * *mt,  \
        Vector4 & filterWidths, SampledWavelengths<LANE_WIDTH> & lambda)                      \
    {                                                                                         \
        RoughnessTexture *rghShaders[LANE_WIDTH];                                             \
        LaneFloat eta;                                                                        \
        for (u32 i = 0; i < LANE_WIDTH; i++)                                                  \
        {                                                                                     \
            rghShaders[i] = &mt[i]->rghShader;                                                \
            Set(eta, i)   = &mt[i]->ior(Get(lambda[0], i));                                   \
        }                                                                                     \
        if constexpr (!std::is_same_v<IORShader, ConstantSpectrum>)                           \
        {                                                                                     \
            lambda.TerminateSecondary();                                                      \
        }                                                                                     \
        LaneFloat roughness = RghShader##Func(intr, rghShaders, filterWidths, lambda);        \
        roughness           = TrowbridgeReitzDistribution::RoughnessToAlpha(roughness);       \
        TrowbridgeReitzDistribution distrib(roughness, roughness);                            \
        return PushStructConstruct(arena, DielectricBxDF)(eta, distrib);                      \
    }                                                                                         \
    static constexpr int Register##id()                                                       \
    {                                                                                         \
        materialFuncs[id] = {CONCAT(DielectricMaterialFunc, id), CONCAT(CreateFunc, id)};     \
        return id;                                                                            \
    }                                                                                         \
    static const int dummy##id = Register##id();

#define COATED_DIFFUSE_MATERIAL(id, RghShader, RflShader, AlbedoShader, IORShader)            \
    struct CONCAT(CoatedDiffuseMaterial, id)                                                  \
    {                                                                                         \
        RghShader rghShader;                                                                  \
        RflShader rflShader;                                                                  \
        AlbedoShader albShader;                                                               \
        IORShader ior;                                                                        \
        f32 thickness, g;                                                                     \
        u32 maxDepth, nSamples;                                                               \
    };                                                                                        \
    CoatedDiffuseBxDF *CONCAT(CoatedDiffuseMaterialFunc, id)(                                 \
        Arena * arena, SurfaceInteractionsN & intr, CoatedDiffuseMaterial##id * *mt,          \
        Vec4lfn & filterWidths, SampledWavelengthsN & lambda)                                 \
    {                                                                                         \
        RghShader *rghShaders[IntN];                                                          \
        RflShader *rflShaders[IntN];                                                          \
        AlbedoShader *albShaders[IntN];                                                       \
        LaneNF32 eta, gg, thickness;                                                          \
        LaneNU32 maxDepth, nSamples;                                                          \
        for (u32 i = 0; i < IntN; i++)                                                        \
        {                                                                                     \
            rghShaders[i]     = &mt[i]->rghShader;                                            \
            rflShaders[i]     = &mt[i]->rflShader;                                            \
            albShaders[i]     = &mt[i]->albShader;                                            \
            Set(eta, i)       = mt[i]->ior(Get(lambda[0], i));                                \
            Set(gg, i)        = mt[i]->g;                                                     \
            Set(thickness, i) = mt[i]->thickness;                                             \
            Set(maxDepth, i)  = mt[i]->maxDepth;                                              \
            Set(nSamples, i)  = mt[i]->nSamples;                                              \
        }                                                                                     \
        eta = Select(eta == 0.f, 1.f, eta);                                                   \
        if constexpr (!std::is_same_v<IORShader, ConstantSpectrum>)                           \
        {                                                                                     \
            lambda.TerminateSecondary();                                                      \
        }                                                                                     \
        SampledSpectrumN reflectance =                                                        \
            RflShader##Func(intr, rflShaders, filterWidths, lambda);                          \
        SampledSpectrumN a = SampledSpectrumN(                                                \
            Clamp(AlbedoShader##Func(intr, albShaders, filterWidths, lambda), 0.f, 1.f));     \
        LaneFloat roughness = RghShader##Func(intr, rghShaders, filterWidths, lambda);        \
        roughness           = TrowbridgeReitzDistribution::RoughnessToAlpha(roughness);       \
        TrowbridgeReitzDistribution distrib(roughness, roughness);                            \
        return PushStructConstruct(arena, CoatedDiffuseBxDF)(DielectricBxDF(eta, distrib),    \
                                                             DiffuseBxDF(reflectance), a, gg, \
                                                             thickness, maxDepth, nSamples);  \
    }                                                                                         \
    static constexpr int Register##id()                                                       \
    {                                                                                         \
        materialFuncs[id] = {CONCAT(CoatedDiffuseMaterialFunc, id), CONCAT(CreateFunc, id)};  \
        return id;                                                                            \
    }                                                                                         \
    static const int dummy##id = Register##id();

#define IMAGE_TEXTURE(OutType, id, SpectrumIn, textureFunc)                                   \
    OutType CONCAT(ImageTextureShaderFunc,                                                    \
                   id)(SurfaceInteractionsN & intrs, LaneUInt32 & mtOffsets,                  \
                       Vec4lfn & filterWidths, SampledWavelengthsN & lambda)                  \
    {                                                                                         \
        OutType result;                                                                       \
        for (u32 i = 0; i < LANE_WIDTH; i++)                                                  \
        {                                                                                     \
            textureFunc(data + Get(mtOffsets, i));                                            \
        }                                                                                     \
        return CONCAT(SpectrumIn, ::Sample)(*RGBColorSpace::sRGB, results, lambda);           \
    }

#define IMAGE_TEXTURE_1(id, Spectrum, textureFunc)                                            \
    IMAGE_TEXTURE(LaneFloat, id, Spectrum, textureFunc)

DIFFUSE_MATERIAL(0, adsf, asdf);
IMAGE_TEXTURE_1(0, asdf, asdf);
DIFFUSE_MATERIAL(0, ImageTextureShader0, textureFuncs[0]);
ImageTextureShader(0, RGBAlbedoSpectrum, whatever)

#endif
