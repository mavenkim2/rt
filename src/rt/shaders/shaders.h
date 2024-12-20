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

#define Vector3    Vec3<LaneF32<EXPAND(LANE_WIDTH)>>
#define Vector4    Vec4<LaneF32<EXPAND(LANE_WIDTH)>>
#define LaneFloat  LaneF32<EXPAND(LANE_WIDTH)>
#define LaneUInt32 LaneU32<EXPAND(LANE_WIDTH)>

#define DIFFUSE_MATERIAL(id, textureType, textureFunc)                                        \
    struct CONCAT(DiffuseMaterial, id)                                                        \
    {                                                                                         \
        textureType reflectance;                                                              \
    };                                                                                        \
    DiffuseBxDF CONCAT(DiffuseMaterialFunc,                                                   \
                       id)(SurfaceInteractions & intr, const LaneUInt32 &mtOffsets,           \
                           Vector4 &filterWidths, SampledWavelengthsN &lambda)                \
    {                                                                                         \
        SampledSpectrumN sampledSpectra = textureFunc(intr, mtOffsets, filterWidths, lambda); \
        return DiffuseBxDF(sampledSpectra);                                                   \
    }

#define DIELECTRIC_MATERIAL(id, textureType, Spectrum)                                        \
    struct CONCAT(DielectricMaterial, id)                                                     \
    {                                                                                         \
        textureType roughness;                                                                \
        Spectrum ior;                                                                         \
    };                                                                                        \
    DielectricBxDF

#define IMAGE_TEXTURE(OutType, id, Spectrum, textureFunc)                                     \
    OutType CONCAT(ImageTextureShaderFunc,                                                    \
                   id)(SurfaceInteractionsN & intrs, LaneUInt32 & mtOffsets,                  \
                       Vec4lfn & filterWidths, SampledWavelengthsN & lambda)                  \
    {                                                                                         \
        OutType result;                                                                       \
        for (u32 i = 0; i < LANE_WIDTH; i++)                                                  \
        {                                                                                     \
            textureFunc(data + Get(mtOffsets, i));                                            \
        }                                                                                     \
        return CONCAT(Spectrum, ::Sample)(*RGBColorSpace::sRGB, results, lambda);             \
    }

#define IMAGE_TEXTURE_1(id, Spectrum, textureFunc)                                            \
    IMAGE_TEXTURE(LaneFloat, id, Spectrum, textureFunc)

IMAGE_TEXTURE_1(0, asdf, asdf);
DIFFUSE_MATERIAL(0, ImageTextureShader0, textureFuncs[0]);
ImageTextureShader(0, RGBAlbedoSpectrum, whatever)

#endif
