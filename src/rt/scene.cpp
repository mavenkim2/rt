#include "scene.h"
#include "bvh/bvh_types.h"
#include "macros.h"
#include "integrate.h"
#include "math/matx.h"
#include "memory.h"
#include "scene_load.h"
#include "simd_integrate.h"
#include "spectrum.h"
#include <cwchar>
namespace rt
{

//////////////////////////////
// Scene
//

struct SceneLoadTable
{
    struct Node
    {
        string filename;
        // Scheduler::Counter counter = {};
        ScenePrimitives *scene;
        Node *next;
    };

    std::atomic<Node *> *nodes;
    u32 count;
};

void Texture::Start(ShadingThreadState *) {}

struct PtexData
{
    Ptex::PtexTexture *texture;
    Ptex::PtexFilter *filter;
};

struct PtexCache
{
    std::atomic<int> count;
    Mutex mutex;

    void Load(string filename, ShadingThreadState *state, PtexTexture *parent)
    {
        count.fetch_add(1, std::memory_order_acquire);
    }
    void Release(Ptex::PtexTexture *texture)
    {
        int result = count.fetch_sub(1, std::memory_order_release);
        if (result == 1) texture->release();
    }
};

enum FilterType
{
    Bspline,
    CatmullRom,
};
struct PtexTexture : Texture
{
    string filename;
    f32 scale = 1.f;
    ColorEncoding encoding;
    FilterType filterType;

    bool simd;

    PtexCache ptexCache;

    PtexTexture(string filename, FilterType filterType = FilterType::Bspline,
                ColorEncoding encoding = ColorEncoding::Gamma, f32 scale = 1.f)
        : filename(filename), filterType(filterType), encoding(encoding), scale(scale)
    {
    }

    f32 EvaluateFloat(SurfaceInteraction &si, const Vec4f &filterWidths) override
    {
        f32 result = 0.f;
        if (simd)
        {
            // PtexData *d = (PtexData *)data;
            Ptex::String error;
            Ptex::PtexTexture *texture = cache->get((char *)filename.str, error);
            if (!texture)
            {
                printf("%s\n", error.c_str());
                Assert(0);
            }
            EvaluateHelper<1>(texture, si, filterWidths, &result);
        }
        else
        {
            EvaluateHelper<1>(si, filterWidths, &result);
        }
        return result * scale;
    }

    SampledSpectrum EvaluateAlbedo(SurfaceInteraction &si, SampledWavelengths &lambda,
                                   const Vec4f &filterWidths) override
    {
        Vec3f result = {};
        if (simd)
        {
            Ptex::String error;
            Ptex::PtexTexture *texture = cache->get((char *)filename.str, error);
            if (!texture)
            {
                printf("%s\n", error.c_str());
                Assert(0);
            }
            EvaluateHelper<3>(texture, si, filterWidths, result.e);
        }
        else
        {
            EvaluateHelper<3>(si, filterWidths, result.e);
        }
        Assert(!IsNaN(result[0]) && !IsNaN(result[1]) && !IsNaN(result[2]));
        return Texture::EvaluateAlbedo(result * scale, lambda);
    }

    virtual void Start(ShadingThreadState *state) override
    {
        GetDebug()->texture = this;
        simd                = true;
        ptexCache.Load(filename, state, this);
    }

    virtual void Stop() override
    {
        Ptex::String error;
        Ptex::PtexTexture *texture = cache->get((char *)filename.str, error);
        if (!texture)
        {
            printf("%s\n", error.c_str());
            Assert(0);
        }
        ptexCache.Release(texture);
    }

    template <i32 c>
    void EvaluateHelper(Ptex::PtexTexture *texture, SurfaceInteraction &intr,
                        const Vec4f &filterWidths, f32 *result) const
    {
        GetDebug()->filename = filename;
        Ptex::PtexFilter::FilterType fType;
        switch (filterType)
        {
            case FilterType::Bspline:
            {
                fType = Ptex::PtexFilter::FilterType::f_bspline;
            }
            break;
            case FilterType::CatmullRom:
            {
                fType = Ptex::PtexFilter::FilterType::f_catmullrom;
            }
            break;
        }

        Ptex::PtexFilter::Options opts(fType);
        Ptex::PtexFilter *filter = Ptex::PtexFilter::getFilter(texture, opts);
        const Vec2f &uv          = intr.uv;
        u32 faceIndex            = intr.faceIndices;

        if (!texture)
        {
            Print("ptex filename: %S\n", filename);
            Print("scene filename: %S\n", GetDebug()->filename);
            Print("geomID: %u\n", GetDebug()->geomID);
            Assert(0);
        }
        u32 numFaces = texture->getInfo().numFaces;
        if (faceIndex >= numFaces)
        {
            Print("faceIndex: %u, numFaces: %u\n", faceIndex, numFaces);
            Print("scene filename: %S\n", GetDebug()->filename);
            Print("geomID: %u\n", GetDebug()->geomID);
            Print("filename: %S\n", filename);
            Assert(0);
        }

        i32 nc = texture->numChannels();
        Assert(nc == c);

        f32 out[3] = {};
        filter->eval(out, 0, c, faceIndex, uv[0], uv[1], filterWidths[0], filterWidths[1],
                     filterWidths[2], filterWidths[3]);

        Assert(!IsNaN(out[0]) && !IsNaN(out[1]) && !IsNaN(out[2]));
        filter->release();

        // Convert to srgb

        if constexpr (c == 1) *result = out[0];
        else
        {
            if (encoding == ColorEncoding::SRGB)
            {
                u8 rgb[3];
                for (i32 i = 0; i < nc; i++)
                {
                    rgb[i] = u8(Clamp(out[i] * 255.f + 0.5f, 0.f, 255.f));
                }
                Vec3f rgbF = SRGBToLinear(rgb);
                out[0]     = rgbF.x;
                out[1]     = rgbF.y;
                out[2]     = rgbF.z;
            }
            else if (encoding == ColorEncoding::Gamma)
            {
                out[0] = Pow(Max(out[0], 0.f), 2.2f);
                out[1] = Pow(Max(out[1], 0.f), 2.2f);
                out[2] = Pow(Max(out[2], 0.f), 2.2f);
            }

            Assert(!IsNaN(out[0]) && !IsNaN(out[1]) && !IsNaN(out[2]));

            result[0] = out[0];
            result[1] = out[1];
            result[2] = out[2];
        }
    }

    template <i32 c>
    void EvaluateHelper(SurfaceInteraction &intr, const Vec4f &filterWidths, f32 *result)
    {
        GetDebug()->filename = filename;
        Assert(cache);
        Ptex::String error;
        Ptex::PtexTexture *texture = cache->get((char *)filename.str, error);
        if (!texture)
        {
            printf("%s\n", error.c_str());
            Assert(0);
        }

        EvaluateHelper<c>(texture, intr, filterWidths, result);

        texture->release();
    }
};

struct ConstantTexture : Texture
{
    f32 constant;

    ConstantTexture(f32 constant) : constant(constant) {}

    f32 EvaluateFloat(SurfaceInteraction &si, const Vec4f &filterWidths) override
    {
        return constant;
    }
    SampledSpectrum EvaluateAlbedo(SurfaceInteraction &si, SampledWavelengths &lambda,
                                   const Vec4f &filterWidths) override
    {
        return SampledSpectrum(constant);
    }
};

struct ConstantVectorTexture : Texture
{
    Vec3f constant;

    ConstantVectorTexture(Vec3f constant) : constant(constant) {}

    SampledSpectrum EvaluateAlbedo(SurfaceInteraction &si, SampledWavelengths &lambda,
                                   const Vec4f &filterWidths) override
    {
        return Texture::EvaluateAlbedo(constant, lambda);
    }
};

struct NullMaterial : Material
{
    NullMaterial() {}
    BxDF Evaluate(Arena *arena, SurfaceInteraction &si, SampledWavelengths &lambda,
                  const Vec4f &filterWidths) override
    {
        return {};
    }
};

struct DiffuseMaterial : Material
{
    Texture *reflectance;
    DiffuseMaterial(Texture *reflectance) : reflectance(reflectance) {}
    BxDF Evaluate(Arena *arena, SurfaceInteraction &si, SampledWavelengths &lambda,
                  const Vec4f &filterWidths) override
    {
        DiffuseBxDF *bxdf = PushStruct(arena, DiffuseBxDF);
        *bxdf             = EvaluateHelper(si, lambda, filterWidths);
        return bxdf;
    }
    DiffuseBxDF EvaluateHelper(SurfaceInteraction &si, SampledWavelengths &lambda,
                               const Vec4f &filterWidths)

    {
        SampledSpectrum s = reflectance->EvaluateAlbedo(si, lambda, filterWidths);

        return DiffuseBxDF(s);
    }
    virtual void Start(ShadingThreadState *state) override { reflectance->Start(state); }
    virtual void Stop() override { reflectance->Stop(); }
};

struct DiffuseTransmissionMaterial : Material
{
    Texture *reflectance;
    Texture *transmittance;
    f32 scale;
    DiffuseTransmissionMaterial(Texture *reflectance, Texture *transmittance, f32 scale)
        : reflectance(reflectance), transmittance(transmittance), scale(scale)
    {
    }
    BxDF Evaluate(Arena *arena, SurfaceInteraction &si, SampledWavelengths &lambda,
                  const Vec4f &filterWidths) override
    {
        DiffuseTransmissionBxDF *bxdf = PushStruct(arena, DiffuseTransmissionBxDF);
        *bxdf                         = EvaluateHelper(si, lambda, filterWidths);
        return bxdf;
    }

    DiffuseTransmissionBxDF EvaluateHelper(SurfaceInteraction &si, SampledWavelengths &lambda,
                                           const Vec4f &filterWidths)
    {
        SampledSpectrum r = reflectance->EvaluateAlbedo(si, lambda, filterWidths);
        SampledSpectrum t = transmittance->EvaluateAlbedo(si, lambda, filterWidths);

        return DiffuseTransmissionBxDF(r, t);
    }
    virtual void Start(ShadingThreadState *state) override
    {
        reflectance->Start(state);
        transmittance->Start(state);
    }
    virtual void Stop() override
    {
        reflectance->Stop();
        transmittance->Stop();
    }
};

struct DielectricMaterial : Material
{
    Texture *uRoughnessTexture;
    Texture *vRoughnessTexture;
    // ConstantSpectrum eta;
    f32 eta;
    // Spectrum eta;

    DielectricMaterial() = default;
    DielectricMaterial(Texture *u, Texture *v, f32 eta)
        : uRoughnessTexture(u), vRoughnessTexture(v), eta(eta)
    {
    }

    BxDF Evaluate(Arena *arena, SurfaceInteraction &si, SampledWavelengths &lambda,
                  const Vec4f &filterWidths) override
    {
        DielectricBxDF *bxdf = PushStruct(arena, DielectricBxDF);
        *bxdf                = EvaluateHelper(si, lambda, filterWidths);
        return bxdf;
    }

    DielectricBxDF EvaluateHelper(SurfaceInteraction &si, SampledWavelengths &lambda,
                                  const Vec4f &filterWidths)
    {
        f32 uRoughness, vRoughness;
        if (uRoughnessTexture == vRoughnessTexture)
        {
            uRoughness = vRoughness = uRoughnessTexture->EvaluateFloat(si, filterWidths);
        }
        else
        {
            uRoughness = uRoughnessTexture->EvaluateFloat(si, filterWidths);

            vRoughness = vRoughnessTexture->EvaluateFloat(si, filterWidths);
        }

        uRoughness = TrowbridgeReitzDistribution::RoughnessToAlpha(uRoughness);
        vRoughness = TrowbridgeReitzDistribution::RoughnessToAlpha(vRoughness);

        // if (eta.TypeIndex<ConstantSpectrum>() == eta.GetTag())
        // {
        //     lambda.TerminateSecondary();
        // }
        f32 ior = eta; // eta(lambda[0]);
        return DielectricBxDF(ior, TrowbridgeReitzDistribution(uRoughness, vRoughness));
    }
    virtual void Start(ShadingThreadState *state) override
    {
        uRoughnessTexture->Start(state);
        if (uRoughnessTexture != vRoughnessTexture) vRoughnessTexture->Start(state);
    }
    virtual void Stop() override
    {
        uRoughnessTexture->Stop();
        if (uRoughnessTexture != vRoughnessTexture) vRoughnessTexture->Stop();
    }
};

struct CoatedDiffuseMaterial : Material
{
    DielectricMaterial dielectric;
    DiffuseMaterial diffuse;
    Texture *albedo;
    Texture *g;
    i32 maxDepth;
    i32 nSamples;
    f32 thickness;

    CoatedDiffuseMaterial(DielectricMaterial die, DiffuseMaterial diff, Texture *albedo,
                          Texture *g, i32 maxDepth, i32 nSamples, f32 thickness)
        : dielectric(die), diffuse(diff), albedo(albedo), g(g), maxDepth(maxDepth),
          nSamples(nSamples), thickness(thickness)
    {
    }

    BxDF Evaluate(Arena *arena, SurfaceInteraction &si, SampledWavelengths &lambda,
                  const Vec4f &filterWidths) override
    {
        CoatedDiffuseBxDF *bxdf = PushStruct(arena, CoatedDiffuseBxDF);
        *bxdf                   = EvaluateHelper(si, lambda, filterWidths);
        return bxdf;
    }

    CoatedDiffuseBxDF EvaluateHelper(SurfaceInteraction &si, SampledWavelengths &lambda,
                                     const Vec4f &filterWidths)

    {
        SampledSpectrum albedoValue = albedo->EvaluateAlbedo(si, lambda, filterWidths);

        f32 gValue = g->EvaluateFloat(si, filterWidths);

        return CoatedDiffuseBxDF(dielectric.EvaluateHelper(si, lambda, filterWidths),
                                 diffuse.EvaluateHelper(si, lambda, filterWidths), albedoValue,
                                 gValue, thickness, maxDepth, nSamples);
    }
    virtual void Start(ShadingThreadState *state) override
    {
        dielectric.Start(state);
        diffuse.Start(state);
    }
    virtual void Stop() override
    {
        dielectric.Stop();
        diffuse.Stop();
    }
};

struct MaterialNode
{
    string str;
    MaterialHandle handle;

    u32 Hash() const { return rt::Hash(str); }
    bool operator==(const MaterialNode &m) const { return str == m.str; }
    bool operator==(string s) const { return s == str; }
};

typedef HashMap<MaterialNode> MaterialHashMap;

struct RTSceneLoadState
{
    SceneLoadTable table;
    MaterialHashMap *map = 0;

    Mutex mutex;
    std::vector<ScenePrimitives *> scenes;

    Mutex lightMutex;
    std::vector<Light *> *lights;
};

i32 ReadIntBytes(Tokenizer *tokenizer)
{
    i32 value = *(i32 *)(tokenizer->cursor);
    tokenizer->cursor += sizeof(i32);
    SkipToNextChar(tokenizer);
    return value;
}
f32 ReadFloatBytes(Tokenizer *tokenizer)
{
    f32 value = *(f32 *)(tokenizer->cursor);
    tokenizer->cursor += sizeof(f32);
    SkipToNextChar(tokenizer);
    return value;
}

Vec3f ReadVec3Bytes(Tokenizer *tokenizer)
{
    Vec3f value = *(Vec3f *)(tokenizer->cursor);
    tokenizer->cursor += sizeof(Vec3f);
    SkipToNextChar(tokenizer);
    return value;
}

Texture *ReadTexture(Arena *arena, Tokenizer *tokenizer, string directory,
                     FilterType type        = FilterType::CatmullRom,
                     ColorEncoding encoding = ColorEncoding::None)
{
    string textureType = ReadWord(tokenizer);
    if (textureType == "ptex")
    {
        f32 scale   = 1.f;
        bool result = Advance(tokenizer, "filename ");
        Assert(result);
        string filename = ReadWord(tokenizer);
        if (Advance(tokenizer, "scale "))
        {
            scale = ReadFloatBytes(tokenizer);
        }
        if (Advance(tokenizer, "encoding "))
        {
            ErrorExit(0, "Not supported yet.\n");
        }
        encoding = encoding == ColorEncoding::None ? ColorEncoding::Gamma : encoding;
        return PushStructConstruct(arena, PtexTexture)(
            PushStr8F(arena, "%S%S\0", directory, filename), type, encoding, scale);
    }
    else
    {
        ErrorExit(0, "Texture type not supported yet");
    }
    return 0;
}

Texture *ParseTexture(Arena *arena, Tokenizer *tokenizer, string directory,
                      FilterType type        = FilterType::CatmullRom,
                      ColorEncoding encoding = ColorEncoding::None)
{
    if (!CharIsDigit(*tokenizer->cursor)) return 0;
    DataType dataType = (DataType)ReadInt(tokenizer);
    SkipToNextChar(tokenizer);

    switch (dataType)
    {
        case DataType::Float:
        {
            f32 value = ReadFloatBytes(tokenizer);
            return PushStructConstruct(arena, ConstantTexture)(value);
        }
        break;
        case DataType::Vec3:
        {
            Vec3f value = ReadVec3Bytes(tokenizer);
            return PushStructConstruct(arena, ConstantVectorTexture)(value);
        }
        break;
        case DataType::Texture:
        {
            return ReadTexture(arena, tokenizer, directory, type, encoding);
        }
        break;
        default: Assert(0); return 0;
    }
}

Texture *ParseDisplacement(Arena *arena, Tokenizer *tokenizer, string directory)
{
    if (Advance(tokenizer, "displacement "))
        return ParseTexture(arena, tokenizer, directory, FilterType::Bspline,
                            ColorEncoding::Linear);
    return 0;
}

DiffuseAreaLight *ParseAreaLight(Arena *arena, Tokenizer *tokenizer, AffineSpace *space,
                                 int sceneID, int geomID)
{
    DiffuseAreaLight *light = PushStruct(arena, DiffuseAreaLight);

    if (Advance(tokenizer, "filename "))
    {
        ErrorExit(0, "sorry, not supported yet\n");
    }
    if (Advance(tokenizer, "L "))
    {
        // TODO: right now this only supportrs a one quad mesh with constant RGB
        DataType dataType = (DataType)ReadInt(tokenizer);
        SkipToNextChar(tokenizer);

        ErrorExit(dataType == DataType::Vec3,
                  "sorry, only constant radiance lights supported\n");

        Vec3f value = ReadVec3Bytes(tokenizer);

        // RGBIlluminantSpectrum *spec =
        RGBIlluminantSpectrum spec(*RGBColorSpace::sRGB, value);
        DenselySampledSpectrum *dss =
            PushStructConstruct(arena, DenselySampledSpectrum)(&spec);
        light->Lemit           = dss;
        light->scale           = 1.f / SpectrumToPhotometric(RGBColorSpace::sRGB->illuminant);
        light->sceneID         = sceneID;
        light->geomID          = geomID;
        light->renderFromLight = space;
        // light->area  = Length(Cross(mesh->p[1] - mesh->p[0], mesh->p[3] - mesh->p[0]));
        // light->renderFromLight = 0;
    }
    if (Advance(tokenizer, "twosided "))
    {
        ErrorExit(0, "sorry, not supported yet\n");
    }
    return light;
}

void ReadDielectricMaterial(Arena *arena, Tokenizer *tokenizer, string directory,
                            DielectricMaterial *mat)
{
    bool isIsotropic = Advance(tokenizer, "roughness ");
    Texture *uRoughness;
    Texture *vRoughness;
    if (isIsotropic)
    {
        uRoughness = vRoughness = ParseTexture(arena, tokenizer, directory);
    }
    else
    {
        bool result = Advance(tokenizer, "uroughness ");
        if (!result)
        {
            uRoughness = vRoughness = PushStructConstruct(arena, ConstantTexture)(0.f);
        }
        else
        {
            uRoughness = ParseTexture(arena, tokenizer, directory);

            result = Advance(tokenizer, "vroughness ");
            Assert(result);
            vRoughness = ParseTexture(arena, tokenizer, directory);
        }
    }
    bool result   = Advance(tokenizer, "eta ");
    DataType type = (DataType)ReadInt(tokenizer);
    Assert(type == DataType::Float);
    SkipToNextChar(tokenizer);
    f32 eta = 1.5f;
    if (result) eta = ReadFloatBytes(tokenizer);
    new (mat) DielectricMaterial(uRoughness, vRoughness, eta);
}

MaterialHashMap *CreateMaterials(Arena *arena, Arena *tempArena, Tokenizer *tokenizer,
                                 string directory)
{
    TempArena temp = ScratchStart(&tempArena, 1);
    Scene *scene   = GetScene();

    ChunkedLinkedList<Material *, 1024> materialsList(temp.arena);
    NullMaterial *nullMaterial = PushStructConstruct(arena, NullMaterial)();
    materialsList.AddBack()    = nullMaterial;
    MaterialHashMap *table = PushStructConstruct(tempArena, MaterialHashMap)(tempArena, 8192);

    std::vector<MaterialTypes> types;
    types.push_back(MaterialTypes::Interface);

    while (!Advance(tokenizer, "MATERIALS_END "))
    {
        bool advanceResult = Advance(tokenizer, "m ");
        Assert(advanceResult);
        string materialName = ReadWord(tokenizer);

        SkipToNextChar(tokenizer);

        // Get the type of material
        MaterialTypes materialTypeIndex = MaterialTypes::Max;
        string materialType             = ReadWord(tokenizer);
        for (u32 m = 0; m < (u32)MaterialTypes::Max; m++)
        {
            if (materialType == materialTypeNames[m])
            {
                materialTypeIndex = (MaterialTypes)m;
                break;
            }
        }

        SkipToNextChar(tokenizer);
        Assert(materialTypeIndex != MaterialTypes::Max);

        // Add to hash table
        table->Add(tempArena,
                   MaterialNode{PushStr8Copy(tempArena, materialName),
                                MaterialHandle(materialTypeIndex, materialsList.totalCount)});

        Material **material = &materialsList.AddBack();

        switch (materialTypeIndex)
        {
            case MaterialTypes::Diffuse:
            {
                bool result = Advance(tokenizer, "reflectance ");
                Texture *reflectance;
                if (!result) reflectance = PushStructConstruct(arena, ConstantTexture)(0.5f);
                else reflectance = ParseTexture(arena, tokenizer, directory);

                *material = PushStructConstruct(arena, DiffuseMaterial)(reflectance);
                (*material)->displacement = ParseDisplacement(arena, tokenizer, directory);
            }
            break;
            case MaterialTypes::DiffuseTransmission:
            {
                bool result = Advance(tokenizer, "reflectance ");
                Texture *reflectance;
                if (!result) reflectance = PushStructConstruct(arena, ConstantTexture)(0.25f);
                else reflectance = ParseTexture(arena, tokenizer, directory);

                result = Advance(tokenizer, "transmittance ");
                Texture *transmittance;
                if (!result)
                    transmittance = PushStructConstruct(arena, ConstantTexture)(0.25f);
                else transmittance = ParseTexture(arena, tokenizer, directory);

                result    = Advance(tokenizer, "scale ");
                f32 scale = 1.f;
                if (result) scale = ReadFloatBytes(tokenizer);

                *material = PushStructConstruct(arena, DiffuseTransmissionMaterial)(
                    reflectance, transmittance, scale);
                (*material)->displacement = ParseDisplacement(arena, tokenizer, directory);
            }
            break;
            case MaterialTypes::Dielectric:
            {
                *material = (Material *)PushStruct(arena, DielectricMaterial);
                ReadDielectricMaterial(arena, tokenizer, directory,
                                       (DielectricMaterial *)(*material));
                (*material)->displacement = ParseDisplacement(arena, tokenizer, directory);
            }
            break;
            case MaterialTypes::CoatedDiffuse:
            {
                DielectricMaterial dm;
                ReadDielectricMaterial(arena, tokenizer, directory, &dm);

                bool result = Advance(tokenizer, "reflectance ");
                Assert(result);
                Texture *reflectance = ParseTexture(arena, tokenizer, directory);

                Texture *albedo = ParseTexture(arena, tokenizer, directory);
                if (!albedo)
                    albedo = PushStructConstruct(arena, ConstantVectorTexture)(Vec3f(0.f));

                Texture *g = ParseTexture(arena, tokenizer, directory);
                if (!g) g = PushStructConstruct(arena, ConstantTexture)(0.f);

                i32 maxDepth = 10;
                if (Advance(tokenizer, "maxdepth ")) maxDepth = ReadIntBytes(tokenizer);
                i32 nSamples = 1;
                if (Advance(tokenizer, "nsamples ")) nSamples = ReadIntBytes(tokenizer);
                f32 thickness = 0.01f;
                if (Advance(tokenizer, "thickness ")) thickness = ReadFloatBytes(tokenizer);

                // Texture if (Advance(tokenizer, "albedo "))
                *material = PushStructConstruct(arena, CoatedDiffuseMaterial)(
                    dm, DiffuseMaterial(reflectance), albedo, g, maxDepth, nSamples,
                    thickness);
                (*material)->displacement = ParseDisplacement(arena, tokenizer, directory);
            }
            break;
            default: Assert(0);
        }
        types.push_back(materialTypeIndex);
    }

    // Join
    scene->materials = StaticArray<Material *>(arena, materialsList.totalCount);

    materialsList.Flatten(scene->materials);

    // Create SIMD queues
    ShadingGlobals *globals   = GetShadingGlobals();
    globals->numShadingQueues = scene->materials.Length();
    globals->shadingQueues    = PushArray(arena, ShadingQueue, globals->numShadingQueues);
    for (u32 i = 0; i < scene->materials.Length(); i++)
    {
        globals->shadingQueues[i].material = scene->materials[i];
        switch (types[i])
        {
            case MaterialTypes::Diffuse:
            {
                globals->shadingQueues[i].handler = ShadingQueueHandler<DiffuseMaterial>;
            }
            break;
            case MaterialTypes::DiffuseTransmission:
            {
                globals->shadingQueues[i].handler =
                    ShadingQueueHandler<DiffuseTransmissionMaterial>;
            }
            break;
            case MaterialTypes::CoatedDiffuse:
            {
                globals->shadingQueues[i].handler = ShadingQueueHandler<CoatedDiffuseMaterial>;
            }
            break;
            case MaterialTypes::Dielectric:
            {
                globals->shadingQueues[i].handler = ShadingQueueHandler<DielectricMaterial>;
            }
            break;
            default:
            {
                globals->shadingQueues[i].handler = ShadingQueueHandler<NullMaterial>;
            }
        }
    }

    ScratchEnd(temp);
    return table;
}

u8 *ReadDataPointer(Tokenizer *tokenizer, Tokenizer *dataTokenizer, string p)
{
    if (Advance(tokenizer, p))
    {
        u32 offset = ReadInt(tokenizer);
        SkipToNextChar(tokenizer);
        return dataTokenizer->input.str + offset;
    }
    return 0;
}

Vec2f *ReadVec2Pointer(Tokenizer *tokenizer, Tokenizer *dataTokenizer, string p)
{
    return (Vec2f *)ReadDataPointer(tokenizer, dataTokenizer, p);
}

Vec3f *ReadVec3Pointer(Tokenizer *tokenizer, Tokenizer *dataTokenizer, string p)
{
    return (Vec3f *)ReadDataPointer(tokenizer, dataTokenizer, p);
}

i32 ReadInt(Tokenizer *tokenizer, string p)
{
    if (Advance(tokenizer, p))
    {
        i32 result = ReadInt(tokenizer);
        SkipToNextChar(tokenizer);
        return result;
    }
    return 0;
}

void LoadRTScene(Arena **arenas, Arena **tempArenas, RTSceneLoadState *state,
                 ScenePrimitives *scene, string directory, string filename,
                 Scheduler::Counter *counter, AffineSpace *renderFromWorld = 0,
                 bool baseFile = false)

{
    // TODO: add totals so we don't have to use linked lists
    scene->filename = filename;
    Assert(GetFileExtension(filename) == "rtscene");

    BeginMutex(&state->mutex);
    int sceneID = (int)state->scenes.size();
    state->scenes.push_back(scene);
    EndMutex(&state->mutex);

    TempArena temp = ScratchStart(0, 0);

    u32 threadIndex  = GetThreadIndex();
    Arena *arena     = arenas[threadIndex];
    Arena *tempArena = tempArenas[threadIndex];

    auto *table           = &state->table;
    auto *materialHashMap = state->map;

    string fullFilePath = StrConcat(temp.arena, directory, filename);
    string dataPath =
        PushStr8F(temp.arena, "%S%S.rtdata", directory, RemoveFileExtension(filename));
    Tokenizer tokenizer;
    tokenizer.input  = OS_MapFileRead(fullFilePath);
    tokenizer.cursor = tokenizer.input.str;

    bool hasMagic = Advance(&tokenizer, "RTSCENE_START ");
    ErrorExit(hasMagic, "RTScene file missing magic.\n");

    string data = OS_ReadFile(arena, dataPath);

    Tokenizer dataTokenizer;
    dataTokenizer.input  = data; // OS_MapFileRead(dataPath); // data;
    dataTokenizer.cursor = dataTokenizer.input.str;

    hasMagic = Advance(&dataTokenizer, "DATA_START ");
    ErrorExit(hasMagic, "RTScene data section missing magic.\n");
    bool hasTransforms = Advance(&dataTokenizer, "TRANSFORM_START ");
    if (hasTransforms)
    {
        Advance(&dataTokenizer, "Count ");
        u32 count            = ReadInt(&dataTokenizer);
        scene->numTransforms = count;
        SkipToNextChar(&dataTokenizer);
        // scene->affineTransforms     = PushArrayNoZero(arena, AffineSpace, count);
        scene->affineTransforms     = (AffineSpace *)(dataTokenizer.cursor);
        AffineSpace *dataTransforms = (AffineSpace *)(dataTokenizer.cursor);
        if (baseFile && renderFromWorld)
        {
            for (u32 i = 0; i < count; i++)
            {
                scene->affineTransforms[i] = *renderFromWorld * dataTransforms[i];
            }
        }
    }
    else if (FindSubstring(filename, "_rtshape_", 0, MatchFlag_CaseInsensitive) !=
             filename.size)
    {
        scene->affineTransforms = GetScene()->scene.affineTransforms;
    }

    bool isLeaf       = true;
    GeometryType type = GeometryType::Max;
    ChunkedLinkedList<ScenePrimitives *, 32, MemoryType_Instance> files(temp.arena);

    bool hasMaterials = false;
    for (;;)
    {
        if (Advance(&tokenizer, "RTSCENE_END")) break;
        if (Advance(&tokenizer, "INCLUDE_START "))
        {
            type = GeometryType::Instance;
            Advance(&tokenizer, "Count: ");
            u32 instanceCount = ReadInt(&tokenizer);
            SkipToNextChar(&tokenizer);

            scene->primitives   = PushArrayNoZero(arena, Instance, instanceCount);
            Instance *instances = (Instance *)scene->primitives;
            u32 instanceOffset  = 0;

            isLeaf = false;
            while (!Advance(&tokenizer, "INCLUDE_END "))
            {
                Advance(&tokenizer, "File: ");
                string includeFile = ReadWord(&tokenizer);
                // TODO: this is a band aid until I get curves working
#if 1
                if (!OS_FileExists(StrConcat(temp.arena, directory, includeFile)) ||
                    !OS_FileExists(PushStr8F(temp.arena, "%S%S.rtdata", directory,
                                             RemoveFileExtension(includeFile))))
                {
                    Print("Skipped %S\n", includeFile);
                    while (CharIsDigit(*tokenizer.cursor))
                    {
                        ReadInt(&tokenizer);
                        Assert(*tokenizer.cursor == '-');
                        tokenizer.cursor++;
                        ReadInt(&tokenizer);
                        SkipToNextChar(&tokenizer);
                    }
                    continue;
                }
#endif
                StringId hash = Hash(includeFile);
                u32 index     = hash & (table->count - 1);

                u32 id                     = files.Length();
                SceneLoadTable::Node *head = table->nodes[index].load();
                for (;;)
                {
                    auto *node = head;
                    while (node)
                    {
                        if (node->filename.size && node->filename == includeFile)
                        {
                            files.AddBack() = node->scene;
                            break;
                        }
                        node = node->next;
                    }

                    u64 pos = ArenaPos(arena);

                    auto *newNode                 = PushStruct(arena, SceneLoadTable::Node);
                    newNode->filename             = PushStr8Copy(arena, includeFile);
                    ScenePrimitives *includeScene = PushStruct(arena, ScenePrimitives);
                    newNode->scene                = includeScene;
                    newNode->next                 = head;

                    if (table->nodes[index].compare_exchange_weak(head, newNode))
                    {
                        files.AddBack() = includeScene;
                        scheduler.Schedule(counter, [=](u32 jobID) {
                            LoadRTScene(arenas, tempArenas, state, includeScene, directory,
                                        newNode->filename, counter);
                        });
                        break;
                    }
                    ArenaPopTo(arena, pos);
                }

                // Load instances
                while (CharIsDigit(*tokenizer.cursor))
                {
                    u32 transformIndexStart = ReadInt(&tokenizer);
                    Assert(*tokenizer.cursor == '-');
                    tokenizer.cursor++;
                    u32 transformIndexEnd = ReadInt(&tokenizer);
                    for (u32 i = transformIndexStart; i <= transformIndexEnd; i++)
                    {
                        Assert(instanceOffset < instanceCount);
                        instances[instanceOffset++] = {id, i};
                    }
                    SkipToNextChar(&tokenizer);
                }
            }
            scene->numPrimitives = instanceOffset;
            scene->childScenes   = PushArrayNoZero(arena, ScenePrimitives *, files.totalCount);
            scene->numChildScenes = files.totalCount;
        }
        else if (Advance(&tokenizer, "SHAPE_START "))
        {
            Assert(isLeaf);
            ChunkedLinkedList<Mesh, 1024, MemoryType_Shape> shapes(temp.arena);
            ChunkedLinkedList<PrimitiveIndices, 1024, MemoryType_Shape> indices(temp.arena);
            ChunkedLinkedList<Light *, 1024, MemoryType_Light> lights(temp.arena);
            // ChunkedLinkedList<Texture *, 1024, MemoryType_Texture>
            // alphaTextures(temp.arena);

            int noMaterialCount       = 0;
            auto AddMaterialAndLights = [&](Mesh &mesh) {
                PrimitiveIndices &primIndices = indices.AddBack();

                MaterialHandle materialHandle;
                LightHandle lightHandle;
                Texture *alphaTexture   = 0;
                DiffuseAreaLight *light = 0;

                // TODO: handle instanced mesh lights
                AffineSpace *transform = 0;

                // Check for material
                if (Advance(&tokenizer, "m "))
                {
                    Assert(materialHashMap);
                    string materialName      = ReadWord(&tokenizer);
                    const MaterialNode *node = materialHashMap->Get(materialName);
                    materialHandle           = node->handle;
                }
                else
                {
                    noMaterialCount++;
                }

                if (Advance(&tokenizer, "transform "))
                {
                    u32 transformIndex = ReadInt(&tokenizer);
                    transform          = &scene->affineTransforms[transformIndex];
                    SkipToNextChar(&tokenizer);
                }

                // Check for area light
                if (Advance(&tokenizer, "a "))
                {
                    ErrorExit(type == GeometryType::QuadMesh,
                              "Only quad area lights supported for now\n");
                    Assert(transform);
                    DiffuseAreaLight *areaLight = ParseAreaLight(
                        arena, &tokenizer, transform, sceneID, shapes.totalCount - 1);
                    lightHandle = LightHandle(LightClass::DiffuseAreaLight, lights.totalCount);
                    lights.AddBack() = areaLight;
                    light            = areaLight;
                }

                // Check for alpha
                if (Advance(&tokenizer, "alpha "))
                {
                    Texture *alphaTexture = ParseTexture(arena, &tokenizer, directory);

                    // TODO: this is also a hack: properly evaluate whether the alpha is
                    // always 0
                    if (lightHandle)
                    {
                        light->type = LightType::DeltaPosition;
                    }
                }
                primIndices = PrimitiveIndices(lightHandle, materialHandle, alphaTexture);
            };

            auto AddMesh = [&](Mesh &mesh, int numVerticesPerFace) {
                mesh.p       = ReadVec3Pointer(&tokenizer, &dataTokenizer, "p ");
                mesh.n       = ReadVec3Pointer(&tokenizer, &dataTokenizer, "n ");
                mesh.uv      = ReadVec2Pointer(&tokenizer, &dataTokenizer, "uv ");
                mesh.indices = (u32 *)ReadDataPointer(&tokenizer, &dataTokenizer, "indices ");
                mesh.numVertices = ReadInt(&tokenizer, "v ");
                mesh.numIndices  = ReadInt(&tokenizer, "i ");
                mesh.numFaces    = mesh.numIndices ? mesh.numIndices / numVerticesPerFace
                                                   : mesh.numVertices / numVerticesPerFace;
            };

            while (!Advance(&tokenizer, "SHAPE_END "))
            {
                if (Advance(&tokenizer, "Quad "))
                {
                    // MOANA: everything should be catclark
                    // Assert(0);
                    Assert(type == GeometryType::Max || type == GeometryType::QuadMesh);
                    type       = GeometryType::QuadMesh;
                    Mesh &mesh = shapes.AddBack();
                    AddMesh(mesh, 4);

                    // TODO: this is a hack
                    if (mesh.numFaces == 1 && mesh.numVertices == 4 && mesh.numIndices == 6)
                    {
                        mesh.indices    = 0;
                        mesh.numIndices = 0;
                    }

                    AddMaterialAndLights(mesh);

                    threadMemoryStatistics[threadIndex].totalShapeMemory +=
                        mesh.numVertices * (sizeof(Vec3f) * 2 + sizeof(Vec2f)) +
                        mesh.numIndices * sizeof(u32);
                }
                else if (Advance(&tokenizer, "Tri "))
                {
                    Assert(type == GeometryType::Max || type == GeometryType::TriangleMesh);
                    type       = GeometryType::TriangleMesh;
                    Mesh &mesh = shapes.AddBack();
                    AddMesh(mesh, 3);
                    AddMaterialAndLights(mesh);

                    threadMemoryStatistics[threadIndex].totalShapeMemory +=
                        mesh.numVertices * (sizeof(Vec3f) * 2 + sizeof(Vec2f)) +
                        mesh.numIndices * sizeof(u32);
                }
                else if (Advance(&tokenizer, "Catclark "))
                {
                    Assert(type == GeometryType::Max || type == GeometryType::CatmullClark);
                    type       = GeometryType::CatmullClark;
                    Mesh &mesh = shapes.AddBack();
                    AddMesh(mesh, 4);
                    AddMaterialAndLights(mesh);
                }
                else
                {
                    Assert(0);
                }
            }

            scene->numPrimitives = shapes.totalCount;
            Assert(shapes.totalCount == indices.totalCount);
            scene->primitives = PushArrayNoZero(arena, Mesh, shapes.totalCount);
            shapes.Flatten((Mesh *)scene->primitives);

            if (lights.totalCount)
            {
                int size = state->lights->size();
                BeginMutex(&state->lightMutex);
                state->lights->resize(size + lights.totalCount);
                lights.Flatten(state->lights->data() + size);
                EndMutex(&state->lightMutex);
                for (auto *node = indices.first; node != 0; node = node->next)
                {
                    for (int i = 0; i < node->count; i++)
                    {
                        if (node->values[i].lightID)
                        {
                            int index       = size + node->values[i].lightID.GetIndex();
                            LightClass type = node->values[i].lightID.GetType();
                            node->values[i].lightID = LightHandle(type, index + size);
                        }
                    }
                }
            }

            scene->primIndices = PushArrayNoZero(arena, PrimitiveIndices, indices.totalCount);
            indices.Flatten(scene->primIndices);
        }
        else if (Advance(&tokenizer, "MATERIALS_START "))
        {
            hasMaterials    = true;
            materialHashMap = CreateMaterials(arena, tempArena, &tokenizer, directory);
            state->map      = materialHashMap;
        }
        else
        {
            ErrorExit(0, "Invalid section header.\n");
        }
    }
    files.Flatten(scene->childScenes);

    scene->dataTokenizer = dataTokenizer;

    OS_UnmapFile(tokenizer.input.str);

    // If there are no transforms and no two level-bvhs, then we need to manually convert
    // meshes to render space.
    if (baseFile && isLeaf)
    {
        Mesh *meshes = (Mesh *)scene->primitives;
        for (u32 i = 0; i < scene->numPrimitives; i++)
        {
            Mesh *mesh = &meshes[i];
            for (u32 v = 0; v < mesh->numVertices; v++)
            {
                mesh->p[v] = TransformP(*renderFromWorld, mesh->p[v]);
            }
        }
    }

    if (type == GeometryType::CatmullClark)
    {
        scene->tessellationParams =
            PushArray(tempArena, TessellationParams, scene->numPrimitives);

        ComputeTessellationParams<GeometryType::QuadMesh>(
            (Mesh *)scene->primitives, scene->tessellationParams, 0, scene->numPrimitives);
    }

    scene->geometryType = type;

    ScratchEnd(temp);
}

// TODO IMPORTANT: there's a race condition in here that occurs very rarely, where some of the
// counters never reach zero, leading to a deadlock.
void BuildSceneBVHs(Arena **arenas, ScenePrimitives *scene, const Mat4 &NDCFromCamera,
                    const Mat4 &cameraFromRender, int screenHeight)
{
    u32 expected = 0;
    // TODO: circular dependencies?
    if (scene->counter.count.compare_exchange_strong(expected, 1, std::memory_order_acquire))
    {
        if (scene->nodePtr.data == 0)
        {
            switch (scene->geometryType)
            {
                case GeometryType::Instance:
                {
                    if (!scene->nodePtr.data)
                    {
                        ParallelFor(0, scene->numChildScenes, 1,
                                    [&](int jobID, int start, int count) {
                                        for (int i = start; i < start + count; i++)
                                        {
                                            BuildSceneBVHs(arenas, scene->childScenes[i],
                                                           NDCFromCamera, cameraFromRender,
                                                           screenHeight);
                                        }
                                    });
                        BuildTLASBVH(arenas, scene);
                    }
                    break;
                    case GeometryType::QuadMesh:
                    {
                        BuildQuadBVH(arenas, scene);
                    }
                    break;
                    case GeometryType::TriangleMesh:
                    {
                        BuildTriangleBVH(arenas, scene);
                    }
                    break;
                    case GeometryType::CatmullClark:
                    {
                        scene->primitives = AdaptiveTessellation(
                            arenas, scene, NDCFromCamera, cameraFromRender, screenHeight,
                            scene->tessellationParams, (Mesh *)scene->primitives,
                            scene->numPrimitives);
                        BuildCatClarkBVH(arenas, scene);
                    }
                    break;
                    default: Assert(0);
                }
            }
        }
        scene->counter.count.fetch_sub(1, std::memory_order_release);
    }
    else
    {
        scheduler.Wait(&scene->counter);
    }
}

enum class TessellationStyle
{
    ClosestInstance,
    PerInViewInstancePerEdge,
};

void ComputeEdgeRates(ScenePrimitives *scene, const AffineSpace &transform,
                      const Vec4f *planes, TessellationStyle style)
{
    switch (scene->geometryType)
    {
        case GeometryType::Instance:
        {
            Instance *instances = (Instance *)scene->primitives;
            ParallelFor(
                0, scene->numPrimitives, PARALLEL_THRESHOLD, PARALLEL_THRESHOLD,
                [&](int jobID, int start, int count) {
                    for (int i = start; i < start + count; i++)
                    {
                        const Instance &instance = instances[i];
                        AffineSpace t =
                            transform * scene->affineTransforms[instance.transformIndex];
                        ComputeEdgeRates(scene->childScenes[instance.id], t, planes, style);
                    }
                });
        }
        break;
        case GeometryType::CatmullClark:
        {
            Mesh *controlMeshes = (Mesh *)scene->primitives;
            ParallelFor(0, scene->numPrimitives, PARALLEL_THRESHOLD, PARALLEL_THRESHOLD,
                        [&](int jobID, int start, int count) {
                            for (int i = start; i < start + count; i++)
                            {
                                TessellationParams &params = scene->tessellationParams[i];

                                switch (style)
                                {
                                    case TessellationStyle::ClosestInstance:
                                    {
                                        BeginRMutex(&params.mutex);
                                        Bounds bounds  = Transform(transform, params.bounds);
                                        Vec3f centroid = ToVec3f(bounds.Centroid());
                                        f32 currentMinDistance = params.currentMinDistance;
                                        EndRMutex(&params.mutex);

                                        // NOTE: this skips the far plane test
                                        int result =
                                            IntersectFrustumAABB<1>(planes, &bounds, 5);
                                        f32 distance = Length(centroid);

                                        // Camera is at origin in this coordinate space
                                        if (result && distance < currentMinDistance)
                                        {
                                            BeginWMutex(&params.mutex);
                                            if (distance < params.currentMinDistance)
                                            {
                                                params.transform          = transform;
                                                params.currentMinDistance = distance;
                                            }
                                            EndWMutex(&params.mutex);
                                        }
                                    }
                                    break;
                                    case TessellationStyle::PerInViewInstancePerEdge:
                                    {
                                        Assert(0);
                                    }
                                    break;
                                }
                            }
                        });
        }
        break;
        default:
        {
        }
        break;
    }
}

void LoadScene(Arena **arenas, Arena **tempArenas, string directory, string filename,
               const Mat4 &NDCFromCamera, const Mat4 &cameraFromRender, int screenHeight,
               AffineSpace *t)
{
    TempArena temp = ScratchStart(0, 0);
    Arena *arena   = arenas[GetThreadIndex()];

    u32 numProcessors      = OS_NumProcessors();
    RTSceneLoadState state = {};
    state.table.count      = 1024;
    state.table.nodes =
        PushArray(temp.arena, std::atomic<SceneLoadTable::Node *>, state.table.count);
    state.lights = &GetScene()->lights;

    Scene *scene = GetScene();

    Scheduler::Counter counter = {};
    LoadRTScene(arenas, tempArenas, &state, &scene->scene, directory, filename, &counter, t,
                true);
    scheduler.Wait(&counter);

    ScenePrimitives **scenes = PushArrayNoZero(arena, ScenePrimitives *, state.scenes.size());
    MemoryCopy(scenes, state.scenes.data(), sizeof(ScenePrimitives *) * state.scenes.size());
    state.scenes.clear();
    SetScenes(scenes);

    AffineSpace space = AffineSpace::Identity();

    Vec4f planes[6];
    ExtractPlanes(planes, NDCFromCamera * cameraFromRender);

    ComputeEdgeRates(&scene->scene, space, planes, TessellationStyle::ClosestInstance);
    BuildSceneBVHs(arenas, &scene->scene, NDCFromCamera, cameraFromRender, screenHeight);

    for (u32 i = 0; i < numProcessors; i++)
    {
        ArenaClear(tempArenas[i]);
    }

    ScratchEnd(temp);
}

void BuildTLASBVH(Arena **arenas, ScenePrimitives *scene)
{
    TempArena temp = ScratchStart(0, 0);
    BuildSettings settings;
    // build tlas
    RecordAOSSplits record(neg_inf);

    BRef *refs = GenerateBuildRefs(scene, temp.arena, record);
    Bounds b   = Bounds(record.geomBounds);
    Assert((Movemask(b.maxP >= b.minP) & 0x7) == 0x7);

    // NOTE: record is being corrupted somehow during this routine.
    scene->nodePtr = BuildTLASQuantized(settings, arenas, scene, refs, record);
    using IntersectorType =
        typename IntersectorHelper<GeometryType::Instance, BRef>::IntersectorType;
    scene->intersectFunc = &IntersectorType::Intersect;
    scene->occludedFunc  = &IntersectorType::Occluded;

    b = Bounds(record.geomBounds);
    scene->SetBounds(b);
    Assert((Movemask(b.maxP >= b.minP) & 0x7) == 0x7);
    ScratchEnd(temp);
}

template <>
void BuildBVH<GeometryType::CatmullClark>(Arena **arenas, ScenePrimitives *scene)
{
    TempArena temp = ScratchStart(0, 0);
    BuildSettings settings;
    OpenSubdivMesh *meshes = (OpenSubdivMesh *)scene->primitives;

    int untessellatedPatchCount = 0;
    int tessellatedPatchCount   = 0;

    for (u32 i = 0; i < scene->numPrimitives; i++)
    {
        untessellatedPatchCount += (int)meshes[i].untessellatedPatches.Length();
        tessellatedPatchCount += (int)meshes[i].patches.Length();
    }

    int size = untessellatedPatchCount + tessellatedPatchCount;
    std::vector<PrimRef> refs;
    refs.reserve(size);
    refs.resize(untessellatedPatchCount);

    int untessellatedRefOffset = 0;

    Bounds geomBounds;
    Bounds centBounds;

    Arena *arena = arenas[GetThreadIndex()];

    for (u32 i = 0; i < scene->numPrimitives; i++)
    {
        auto *mesh = &meshes[i];

        std::vector<BVHPatch> bvhPatches;
        bvhPatches.reserve((int)mesh->patches.Length());

        std::vector<BVHEdge> bvhEdges;
        bvhEdges.reserve((int)mesh->patches.Length() * 4);

        const auto &indices  = mesh->stitchingIndices;
        const auto &vertices = mesh->vertices;
        for (u32 j = 0; j < mesh->untessellatedPatches.Length(); j++)
        {
            int indexStart = 4 * j;
            Vec3f p0       = vertices[indices[indexStart + 0]];
            Vec3f p1       = vertices[indices[indexStart + 1]];
            Vec3f p2       = vertices[indices[indexStart + 2]];
            Vec3f p3       = vertices[indices[indexStart + 3]];

            Vec3f minP = Min(Min(p0, p1), Min(p2, p3));
            Vec3f maxP = Max(Max(p0, p1), Max(p2, p3));

            geomBounds.Extend(Lane4F32(minP), Lane4F32(maxP));
            centBounds.Extend(Lane4F32(minP + maxP));

            PrimRef *ref = &refs[untessellatedRefOffset++];
            ref->minX    = -minP.x;
            ref->minY    = -minP.y;
            ref->minZ    = -minP.z;
            ref->geomID  = CreatePatchID(CatClarkTriangleType::Untess, 0, i);
            ref->maxX    = maxP.x;
            ref->maxY    = maxP.y;
            ref->maxZ    = maxP.z;
            ref->primID  = j;
        }
        for (u32 j = 0; j < mesh->patches.Length(); j++)
        {
            OpenSubdivPatch *patch = &mesh->patches[j];

            // Individually split each edge into smaller triangles
            for (int edgeIndex = 0; edgeIndex < 4; edgeIndex++)
            {
                // EdgeInfo &currentEdge = patch->edgeInfo[edgeIndex];
                EdgeInfo currentEdge = patch->edgeInfos.GetEdgeInfo(edgeIndex);

                auto itr = patch->CreateIterator(edgeIndex);

                while (itr.IsNotFinished())
                {
                    Vec3f minP(pos_inf);
                    Vec3f maxP(neg_inf);

                    // save start state
                    BVHEdge bvhEdge;
                    bvhEdge.patchIndex = j;
                    bvhEdge.steps      = itr.steps;

                    for (int triIndex = 0; triIndex < 8 && itr.Next(); triIndex++)
                    {
                        minP = Min(Min(minP, vertices[itr.indices[0]]),
                                   Min(vertices[itr.indices[1]], vertices[itr.indices[2]]));
                        maxP = Max(Max(maxP, vertices[itr.indices[0]]),
                                   Max(vertices[itr.indices[1]], vertices[itr.indices[2]]));
                    }

                    geomBounds.Extend(Lane4F32(minP), Lane4F32(maxP));
                    centBounds.Extend(Lane4F32(minP + maxP));

                    int bvhEdgeIndex = (int)bvhEdges.size();
                    bvhEdges.push_back(bvhEdge);

                    refs.emplace_back();
                    PrimRef *ref = &refs.back();
                    ref->minX    = -minP.x;
                    ref->minY    = -minP.y;
                    ref->minZ    = -minP.z;
                    ref->geomID =
                        CreatePatchID(CatClarkTriangleType::TessStitching, edgeIndex, i);
                    ref->maxX   = maxP.x;
                    ref->maxY   = maxP.y;
                    ref->maxZ   = maxP.z;
                    ref->primID = bvhEdgeIndex;
                }
            }

            // Split internal grid into smaller grids
            int edgeRateU     = patch->GetMaxEdgeFactorU();
            int edgeRateV     = patch->GetMaxEdgeFactorV();
            int bvhPatchIndex = (int)bvhPatches.size();
            int bvhPatchStart = bvhPatchIndex;
            {
                BVHPatch bvhPatch;
                bvhPatch.patchIndex = j;
                bvhPatch.grid       = UVGrid::Compress(
                    Vec2i(0, 0), Vec2i(Max(edgeRateU - 2, 0), Max(edgeRateV - 2, 0)));

                bvhPatches.push_back(bvhPatch);
            }

            while (bvhPatchIndex < (int)bvhPatches.size())
            {
                const BVHPatch &bvhPatch = bvhPatches[bvhPatchIndex];
                BVHPatch patch0, patch1;
                if (bvhPatch.Split(patch0, patch1))
                {
                    bvhPatches.push_back(patch1);
                    bvhPatches[bvhPatchIndex] = patch0;
                }
                else
                {
                    bvhPatchIndex++;
                }
            }

            for (int k = bvhPatchStart; k < (int)bvhPatches.size(); k++)
            {
                const BVHPatch &bvhPatch = bvhPatches[k];
                Vec3f minP(pos_inf);
                Vec3f maxP(neg_inf);

                Vec2i uvStart, uvEnd;
                bvhPatch.grid.Decompress(uvStart, uvEnd);
                for (int v = uvStart[1]; v <= uvEnd[1]; v++)
                {
                    for (int u = uvStart[0]; u <= uvEnd[0]; u++)
                    {
                        int index = patch->GetGridIndex(u, v);
                        minP      = Min(minP, vertices[index]);
                        maxP      = Max(maxP, vertices[index]);
                    }
                }

                geomBounds.Extend(Lane4F32(minP), Lane4F32(maxP));
                centBounds.Extend(Lane4F32(minP + maxP));

                // TODO: make it so that I can't forget to make these negative
                refs.emplace_back();
                PrimRef *ref = &refs.back();
                ref->minX    = -minP.x;
                ref->minY    = -minP.y;
                ref->minZ    = -minP.z;
                ref->geomID  = CreatePatchID(CatClarkTriangleType::TessGrid, 0, i);
                ref->maxX    = maxP.x;
                ref->maxY    = maxP.y;
                ref->maxZ    = maxP.z;
                ref->primID  = k;
            }
        }
        mesh->bvhPatches = StaticArray<BVHPatch>(arena, bvhPatches);
        mesh->bvhEdges   = StaticArray<BVHEdge>(arena, bvhEdges);

        threadMemoryStatistics[GetThreadIndex()].totalBVHMemory +=
            sizeof(BVHPatch) * mesh->bvhPatches.Length();
        threadMemoryStatistics[GetThreadIndex()].totalBVHMemory +=
            sizeof(BVHEdge) * mesh->bvhEdges.Length();
    }
    // });

    RecordAOSSplits record;
    record.geomBounds = Lane8F32(-geomBounds.minP, geomBounds.maxP);
    record.centBounds = Lane8F32(-centBounds.minP, centBounds.maxP);
    record.SetRange(0, (int)refs.size());

    scene->nodePtr =
        BuildQuantizedCatmullClarkBVH(settings, arenas, scene, refs.data(), record);
    using IntersectorType =
        typename IntersectorHelper<GeometryType::CatmullClark, PrimRef>::IntersectorType;
    scene->intersectFunc = &IntersectorType::Intersect;
    scene->occludedFunc  = &IntersectorType::Occluded;
    Bounds b(record.geomBounds);
    Assert((Movemask(b.maxP >= b.minP) & 0x7) == 0x7);
    scene->SetBounds(b);
    scene->numFaces = size;
    ScratchEnd(temp);
}

template <GeometryType type>
void BuildBVH(Arena **arenas, ScenePrimitives *scene)

{
    TempArena temp = ScratchStart(0, 0);
    BuildSettings settings;
    Mesh *meshes = (Mesh *)scene->primitives;
    RecordAOSSplits record(neg_inf);

    u32 totalNumFaces = 0;
    if (scene->numPrimitives > 1)
    {
        PrimRef *refs;
        u32 extEnd;
        if (scene->numPrimitives > PARALLEL_THRESHOLD)
        {
            ParallelForOutput output =
                ParallelFor<u32>(temp, 0, scene->numPrimitives, PARALLEL_THRESHOLD,
                                 [&](u32 &faceCount, u32 jobID, u32 start, u32 count) {
                                     u32 outCount = 0;

                                     for (u32 i = start; i < start + count; i++)
                                     {
                                         Mesh &mesh = meshes[i];
                                         outCount += mesh.GetNumFaces();
                                     }
                                     faceCount = outCount;
                                 });
            Reduce(totalNumFaces, output, [&](u32 &l, const u32 &r) { l += r; });

            u32 offset   = 0;
            u32 *offsets = (u32 *)output.out;
            for (u32 i = 0; i < output.num; i++)
            {
                u32 numFaces = offsets[i];
                offsets[i]   = offset;
                offset += numFaces;
            }
            Assert(totalNumFaces == offset);
            extEnd = u32(totalNumFaces * GROW_AMOUNT);

            // Generate PrimRefs
            refs = PushArrayNoZero(temp.arena, PrimRef, extEnd);

            ParallelReduce<RecordAOSSplits>(
                &record, 0, scene->numPrimitives, PARALLEL_THRESHOLD,
                [&](RecordAOSSplits &record, u32 jobID, u32 start, u32 count) {
                    GenerateMeshRefs<type>(meshes, refs, offsets[jobID],
                                           jobID == output.num - 1 ? totalNumFaces
                                                                   : offsets[jobID + 1],
                                           start, count, record);
                },
                [&](RecordAOSSplits &l, const RecordAOSSplits &r) { l.Merge(r); });
        }
        else
        {
            for (u32 i = 0; i < scene->numPrimitives; i++)
            {
                Mesh &mesh = meshes[i];
                totalNumFaces += mesh.GetNumFaces();
            }
            extEnd = u32(totalNumFaces * GROW_AMOUNT);
            refs   = PushArrayNoZero(temp.arena, PrimRef, extEnd);
            GenerateMeshRefs<type>(meshes, refs, 0, totalNumFaces, 0, scene->numPrimitives,
                                   record);
        }
        record.SetRange(0, totalNumFaces, extEnd);
        scene->nodePtr = BuildQuantizedSBVH<type>(settings, arenas, scene, refs, record);
        using IntersectorType = typename IntersectorHelper<type, PrimRef>::IntersectorType;
        scene->intersectFunc  = &IntersectorType::Intersect;
        scene->occludedFunc   = &IntersectorType::Occluded;
    }
    else
    {
        totalNumFaces           = meshes->GetNumFaces();
        u32 extEnd              = u32(totalNumFaces * GROW_AMOUNT);
        PrimRefCompressed *refs = PushArrayNoZero(temp.arena, PrimRefCompressed, extEnd);
        GenerateMeshRefs<type>(meshes, refs, 0, totalNumFaces, 0, 1, record);
        record.SetRange(0, totalNumFaces, extEnd);
        scene->nodePtr = BuildQuantizedSBVH<type>(settings, arenas, scene, refs, record);
        using IntersectorType =
            typename IntersectorHelper<type, PrimRefCompressed>::IntersectorType;
        scene->intersectFunc = &IntersectorType::Intersect;
        scene->occludedFunc  = &IntersectorType::Occluded;
    }
    Bounds b(record.geomBounds);
    Assert((Movemask(b.maxP >= b.minP) & 0x7) == 0x7);
    scene->SetBounds(b);
    scene->numFaces = totalNumFaces;
    ScratchEnd(temp);
}

void BuildTriangleBVH(Arena **arenas, ScenePrimitives *scene)
{
    BuildBVH<GeometryType::TriangleMesh>(arenas, scene);
}

void BuildQuadBVH(Arena **arenas, ScenePrimitives *scene)
{
    BuildBVH<GeometryType::QuadMesh>(arenas, scene);
}

void BuildCatClarkBVH(Arena **arenas, ScenePrimitives *scene)
{
    BuildBVH<GeometryType::CatmullClark>(arenas, scene);
}

template <GeometryType type, typename PrimRef>
void GenerateMeshRefs(Mesh *meshes, PrimRef *refs, u32 offset, u32 offsetMax, u32 start,
                      u32 count, RecordAOSSplits &record)
{
    RecordAOSSplits r(neg_inf);
    for (u32 i = start; i < start + count; i++)
    {
        Mesh *mesh = &meshes[i];

        u32 numFaces = mesh->GetNumFaces();
        RecordAOSSplits tempRecord(neg_inf);
        if (numFaces > PARALLEL_THRESHOLD)
        {
            ParallelReduce<RecordAOSSplits>(
                &tempRecord, 0, numFaces, PARALLEL_THRESHOLD,
                [&](RecordAOSSplits &record, u32 jobID, u32 start, u32 count) {
                    Assert(offset + start < offsetMax);
                    GenerateMeshRefsHelper<type, PrimRef>{mesh->p, mesh->indices}(
                        refs, offset + start, i, start, count, record);
                },
                [&](RecordAOSSplits &l, const RecordAOSSplits &r) { l.Merge(r); });
        }
        else
        {
            Assert(offset < offsetMax);
            GenerateMeshRefsHelper<type, PrimRef>{mesh->p, mesh->indices}(
                refs, offset, i, 0, numFaces, tempRecord);
        }
        r.Merge(tempRecord);
        offset += numFaces;
    }
    Assert(offsetMax == offset);
    record = r;
}

template <GeometryType type>
void ComputeTessellationParams(/*Arena *tempArena,*/ Mesh *meshes, TessellationParams *params,
                               u32 start, u32 count)
{
    for (u32 i = start; i < start + count; i++)
    {
        Bounds bounds;
        Mesh *mesh = &meshes[i];

        u32 numFaces = mesh->GetNumFaces();
        if (numFaces > PARALLEL_THRESHOLD)
        {
            ParallelReduce<Bounds>(
                &bounds, 0, numFaces, PARALLEL_THRESHOLD,
                [&](Bounds &b, int jobID, int start, int count) {
                    b = GenerateMeshRefsHelper<type>{mesh->p, mesh->indices}(start, count);
                },
                [&](Bounds &l, const Bounds &r) { l.Extend(r); });
        }
        else
        {
            bounds = GenerateMeshRefsHelper<type>{mesh->p, mesh->indices}(0, numFaces);
        }
        params[i].bounds = bounds;
        // TODO: hardcoded for quads
        // params[i].edgeIndexMap = AtomicHashIndex(tempArena, mesh->numFaces * 4);
        // params[i].edgeKeyValues =
        //     PushArray(tempArena, TessellationParams::EdgeKeyValue, mesh->numFaces * 4);
        params[i].currentMinDistance = pos_inf;
    }
}

// NOTE: this assumes the quad is planar
ShapeSample ScenePrimitives::SampleQuad(SurfaceInteraction &intr, Vec2f &u,
                                        AffineSpace *transform, int geomID)
{
    static const f32 MinSphericalSampleArea = 3e-4;
    static const f32 MaxSphericalSampleArea = 6.22;
    Mesh *mesh                              = ((Mesh *)primitives) + geomID;

    Vec3f p[4];

    // TODO: handle mesh lights
    Assert(mesh->GetNumFaces() == 1);
    int primID = 0;

    if (mesh->indices)
    {
        p[0] = mesh->p[mesh->indices[4 * primID + 0]];
        p[1] = mesh->p[mesh->indices[4 * primID + 1]];
        p[2] = mesh->p[mesh->indices[4 * primID + 2]];
        p[3] = mesh->p[mesh->indices[4 * primID + 3]];
    }
    else
    {
        p[0] = mesh->p[4 * primID + 0];
        p[1] = mesh->p[4 * primID + 1];
        p[2] = mesh->p[4 * primID + 2];
        p[3] = mesh->p[4 * primID + 3];
    }

    if (transform)
    {
        for (int i = 0; i < 4; i++)
        {
            p[i] = TransformP(*transform, p[i]);
        }
    }

    Vec3lfn v00 = Normalize(p[0] - Vec3f(intr.p));
    Vec3lfn v10 = Normalize(p[1] - Vec3f(intr.p));
    Vec3lfn v01 = Normalize(p[3] - Vec3f(intr.p));
    Vec3lfn v11 = Normalize(p[2] - Vec3f(intr.p));

    Vec3lfn eu = p[1] - p[0];
    Vec3lfn ev = p[3] - p[0];
    Vec3lfn n  = Normalize(Cross(eu, ev));

    ShapeSample result;
    // If the solid angle is small
    f32 area = SphericalQuadArea(v00, v10, v01, v11);
    // Vec3lfn wi   = intr.p - result.samplePoint;
    if (area < MinSphericalSampleArea || area > MaxSphericalSampleArea)
    {
        // First, sample a triangle based on area
        bool isSecondTri = false;
        Vec3f p01        = p[1] - p[0];
        Vec3f p02        = p[2] - p[0];
        Vec3f p03        = p[3] - p[0];
        f32 area0        = Length(Cross(p01, p02));
        f32 area1        = Length(Cross(p02, p03));

        f32 div  = 1.f / (area0 + area1);
        f32 prob = area0 * div;
        // Then sample the triangle by area
        if (u[0] < prob)
        {
            u[0]       = u[0] / area0;
            Vec3f bary = SampleUniformTriangle(u);
            result.p   = bary[0] * p[0] + bary[1] * p[1] + bary[2] * p[2];
        }
        else
        {
            u[0]       = (1 - u[0]) / (1 - prob);
            Vec3f bary = SampleUniformTriangle(u);
            result.p   = bary[0] * p[0] + bary[1] * p[2] + bary[2] * p[3];
        }
        result.n   = n;
        Vec3f wi   = Normalize(intr.p - result.p);
        result.w   = wi;
        result.pdf = div * LengthSquared(intr.p - result.p) / AbsDot(intr.n, wi);
    }
    else
    {
        f32 pdf;
        result.p = SampleSphericalRectangle(intr.p, p[0], eu, ev, u, &pdf);
        result.n = n;
        result.w = Normalize(intr.p - result.p);

        // add projected solid angle measure (n dot wi) to pdf
        Vec4f w(AbsDot(v00, intr.shading.n), AbsDot(v10, intr.shading.n),
                AbsDot(v01, intr.shading.n), AbsDot(v11, intr.shading.n));
        Vec2f uNew = SampleBilinear(u, w);
        pdf *= BilinearPDF(uNew, w);
        result.pdf = pdf;
    }
    return result;
}

ShapeSample ScenePrimitives::Sample(SurfaceInteraction &intr, AffineSpace *space, Vec2f &u,
                                    int geomID)
{
    switch (geometryType)
    {
        case GeometryType::QuadMesh:
        {
            return SampleQuad(intr, u, space, geomID);
        }
        break;
        default: Assert(0); return {};
    }
}

} // namespace rt
