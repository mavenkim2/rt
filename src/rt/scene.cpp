#include "scene.h"
#include "bvh/bvh_types.h"
#include "macros.h"
#include "integrate.h"
#include "scene_load.h"
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
        Scheduler::Counter counter = {};
        ScenePrimitives *scene;
        Node *next;
    };

    std::atomic<Node *> *nodes;
    u32 count;
};

struct Texture
{
    virtual f32 EvaluateFloat(SurfaceInteraction &si, SampledWavelengths &lambda,
                              const Vec4f &filterWidths)
    {
        Error(0, "EvaluateFloat is not defined for sub class \n");
        return 0.f;
    }
    virtual SampledSpectrum EvaluateAlbedo(SurfaceInteraction &si, SampledWavelengths &lambda,
                                           const Vec4f &filterWidths)
    {
        Error(0, "EvaluateAlbedo is not defined for sub class\n");
        return {};
    }
    SampledSpectrum EvaluateAlbedo(const Vec3f &color, SampledWavelengths &lambda)
    {
        if (color == Vec3f(0.f)) return SampledSpectrum(0.f);
        Assert(!IsNaN(color[0]) && !IsNaN(color[1]) && !IsNaN(color[2]));
        // GetDebug()->color = color;
        return RGBAlbedoSpectrum(*RGBColorSpace::sRGB, Clamp(color, Vec3f(0.f), Vec3f(1.f)))
            .Sample(lambda);
    }
};

struct PtexTexture : Texture
{
    string filename;
    f32 scale = 1.f;
    // TODO: encoding
    PtexTexture(string filename, f32 scale = 1.f) : filename(filename), scale(scale) {}

    f32 EvaluateFloat(SurfaceInteraction &si, SampledWavelengths &lambda,
                      const Vec4f &filterWidths) override
    {
        f32 result = 0.f;
        EvaluateHelper<1>(si, filterWidths, &result);
        return result * scale;
    }

    SampledSpectrum EvaluateAlbedo(SurfaceInteraction &si, SampledWavelengths &lambda,
                                   const Vec4f &filterWidths) override
    {
        Vec3f result = {};
        EvaluateHelper<3>(si, filterWidths, result.e);
        Assert(!IsNaN(result[0]) && !IsNaN(result[1]) && !IsNaN(result[2]));
        return Texture::EvaluateAlbedo(result * scale, lambda);
    }

    template <i32 c>
    void EvaluateHelper(SurfaceInteraction &intr, const Vec4f &filterWidths, f32 *result)
    {
        const Vec2f &uv = intr.uv;
        u32 faceIndex   = intr.faceIndices;

        // GetDebug()->filename = filename;
        Assert(cache);
        Ptex::String error;
        Ptex::PtexTexture *texture = cache->get((char *)filename.str, error);
        if (!texture)
        {
            Print("ptex filename: %S\n", filename);
            Print("scene filename: %S\n", GetDebug()->filename);
            Print("geomID: %u\n", GetDebug()->geomID);
            Assert(0);
        }
        u32 numFaces = texture->getInfo().numFaces;
        // TODO: some of the pbrt material -> shape pairings don't match?
        // if (faceIndex >= numFaces)
        // {
        //     Print("faceIndex: %u, numFaces: %u\n", faceIndex, numFaces);
        //     Print("scene filename: %S\n", GetDebug()->filename);
        //     Print("geomID: %u\n", GetDebug()->geomID);
        //     Print("filename: %S\n", filename);
        //     Assert(0);
        // }
        Ptex::PtexFilter::Options opts(Ptex::PtexFilter::FilterType::f_bspline);
        Ptex::PtexFilter *filter = Ptex::PtexFilter::getFilter(texture, opts);

        i32 nc = texture->numChannels();
        Assert(nc == c);

        // TODO: ray differentials
        // Vec2f uv(0.5f, 0.5f);

        f32 out[3] = {};
        filter->eval(out, 0, c, faceIndex, uv[0], uv[1], filterWidths[0], filterWidths[1],
                     filterWidths[2], filterWidths[3]);

        Assert(!IsNaN(out[0]) && !IsNaN(out[1]) && !IsNaN(out[2]));

        texture->release();
        filter->release();

        // Convert to srgb

        if constexpr (c == 1) *result = out[0];
        else
        {
            // if (tex->encoding == ColorEncoding::SRGB)
            // {
            //     u8 rgb[3];
            //     for (i32 i = 0; i < nc; i++)
            //     {
            //         rgb[i] = u8(Clamp(out[i] * 255.f + 0.5f, 0.f, 255.f));
            //     }
            //     Vec3f rgbF = SRGBToLinear(rgb);
            //     out[0]     = rgbF.x;
            //     out[1]     = rgbF.y;
            //     out[2]     = rgbF.z;
            // }
            // else
            // {
            out[0] = Pow(Max(out[0], 0.f), 2.2f);
            out[1] = Pow(Max(out[1], 0.f), 2.2f);
            out[2] = Pow(Max(out[2], 0.f), 2.2f);

            Assert(!IsNaN(out[0]) && !IsNaN(out[1]) && !IsNaN(out[2]));

            result[0] = out[0];
            result[1] = out[1];
            result[2] = out[2];
        }
    }
};

struct ConstantTexture : Texture
{
    f32 constant;

    ConstantTexture(f32 constant) : constant(constant) {}

    f32 EvaluateFloat(SurfaceInteraction &si, SampledWavelengths &lambda,
                      const Vec4f &filterWidths) override
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
            uRoughness = vRoughness =
                uRoughnessTexture->EvaluateFloat(si, lambda, filterWidths);
        }
        else
        {
            uRoughness = uRoughnessTexture->EvaluateFloat(si, lambda, filterWidths);
            vRoughness = vRoughnessTexture->EvaluateFloat(si, lambda, filterWidths);
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

        f32 gValue = g->EvaluateFloat(si, lambda, filterWidths);

        return CoatedDiffuseBxDF(dielectric.EvaluateHelper(si, lambda, filterWidths),
                                 diffuse.EvaluateHelper(si, lambda, filterWidths), albedoValue,
                                 gValue, thickness, maxDepth, nSamples);
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

Texture *ReadTexture(Arena *arena, Tokenizer *tokenizer, string directory)
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
        return PushStructConstruct(arena, PtexTexture)(
            PushStr8F(arena, "%S%S\0", directory, filename), scale);
    }
    else
    {
        Error(0, "Texture type not supported yet");
    }
    return 0;
}

Texture *ParseTexture(Arena *arena, Tokenizer *tokenizer, string directory)
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
            return ReadTexture(arena, tokenizer, directory);
        }
        break;
        default: Assert(0); return 0;
    }
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
    MaterialHashMap *table = PushStructConstruct(tempArena, MaterialHashMap)(tempArena, 8192);

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
        // Add to hash table
        table->Add(tempArena,
                   MaterialNode{materialName,
                                MaterialHandle(materialTypeIndex, materialsList.totalCount)});

        SkipToNextChar(tokenizer);
        Assert(materialTypeIndex != MaterialTypes::Max);

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
            }
            break;
            case MaterialTypes::Dielectric:
            {
                *material = (Material *)PushStruct(arena, DielectricMaterial);
                ReadDielectricMaterial(arena, tokenizer, directory,
                                       (DielectricMaterial *)(*material));
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
            }
            break;
            default: Assert(0);
        }
    }

    // Join
    scene->materials = StaticArray<Material *>(arena, materialsList.totalCount);

    materialsList.Flatten(scene->materials);

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

void LoadRTScene(Arena **arenas, RTSceneLoadState *state, ScenePrimitives *scene,
                 string directory, string filename, Scheduler::Counter *c = 0,
                 AffineSpace *renderFromWorld = 0, bool baseFile = false)

{
    // TODO: add totals so we don't have to use linked lists
    scene->filename = filename;
    Assert(GetFileExtension(filename) == "rtscene");
    TempArena temp = ScratchStart(0, 0);

    u32 threadIndex = GetThreadIndex();
    Arena *arena    = arenas[threadIndex];

    auto *table           = &state->table;
    auto *materialHashMap = state->map;

    string fullFilePath = StrConcat(temp.arena, directory, filename);
    string dataPath =
        PushStr8F(temp.arena, "%S%S.rtdata", directory, RemoveFileExtension(filename));
    Tokenizer tokenizer;
    tokenizer.input  = OS_MapFileRead(fullFilePath);
    tokenizer.cursor = tokenizer.input.str;

    bool hasMagic = Advance(&tokenizer, "RTSCENE_START ");
    Error(hasMagic, "RTScene file missing magic.\n");

    string data = OS_ReadFile(arena, dataPath);

    Tokenizer dataTokenizer;
    dataTokenizer.input  = data;
    dataTokenizer.cursor = dataTokenizer.input.str;

    hasMagic = Advance(&dataTokenizer, "DATA_START ");
    Error(hasMagic, "RTScene data section missing magic.\n");
    bool hasTransforms = Advance(&dataTokenizer, "TRANSFORM_START ");
    if (hasTransforms)
    {
        Advance(&dataTokenizer, "Count ");
        u32 count            = ReadInt(&dataTokenizer);
        scene->numTransforms = count;
        SkipToNextChar(&dataTokenizer);
        scene->affineTransforms = (AffineSpace *)(dataTokenizer.cursor);
        if (baseFile && renderFromWorld)
        {
            for (u32 i = 0; i < count; i++)
            {
                scene->affineTransforms[i] = *renderFromWorld * scene->affineTransforms[i];
            }
        }
    }

    Scheduler::Counter counter = {};
    bool isLeaf                = true;
    GeometryType type          = GeometryType::Max;
    ChunkedLinkedList<ScenePrimitives *, 32, MemoryType_Instance> files(temp.arena);

    bool hasMaterials = false;
    for (;;)
    {
        if (Advance(&tokenizer, "RTSCENE_END")) break;
        if (Advance(&tokenizer, "INCLUDE_START "))
        {
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
                            scheduler.Wait(&node->counter);
                            break;
                        }
                        node = node->next;
                    }

                    u64 pos = ArenaPos(arena);

                    auto *newNode                 = PushStruct(arena, SceneLoadTable::Node);
                    newNode->filename             = PushStr8Copy(arena, includeFile);
                    ScenePrimitives *includeScene = PushStruct(arena, ScenePrimitives);
                    newNode->counter.count        = 1;
                    newNode->scene                = includeScene;
                    newNode->next                 = head;

                    if (table->nodes[index].compare_exchange_weak(head, newNode))
                    {
                        files.AddBack() = includeScene;
                        scheduler.Schedule(&counter, [=](u32 jobID) {
                            LoadRTScene(arenas, state, includeScene, directory,
                                        newNode->filename, &newNode->counter);
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

            auto AddMaterial = [&]() {
                PrimitiveIndices &ids = indices.AddBack();
                if (Advance(&tokenizer, "m "))
                {
                    Assert(materialHashMap);
                    string materialName      = ReadWord(&tokenizer);
                    const MaterialNode *node = materialHashMap->Get(materialName);
                    ids                      = PrimitiveIndices(LightHandle(), node->handle);
                }
                else
                {
                    ids = PrimitiveIndices(LightHandle(), MaterialHandle());
                }
            };

            while (!Advance(&tokenizer, "SHAPE_END "))
            {
                if (Advance(&tokenizer, "Quad "))
                {
                    type             = GeometryType::QuadMesh;
                    Mesh &mesh       = shapes.AddBack();
                    mesh.p           = ReadVec3Pointer(&tokenizer, &dataTokenizer, "p ");
                    mesh.n           = ReadVec3Pointer(&tokenizer, &dataTokenizer, "n ");
                    mesh.numVertices = ReadInt(&tokenizer, "v ");
                    mesh.numFaces    = mesh.numVertices / 4;
                    threadMemoryStatistics[threadIndex].totalShapeMemory +=
                        sizeof(Vec3f) * mesh.numVertices * 2;
                    AddMaterial();
                }
                else if (Advance(&tokenizer, "Tri "))
                {
                    type       = GeometryType::TriangleMesh;
                    Mesh &mesh = shapes.AddBack();
                    mesh.p     = ReadVec3Pointer(&tokenizer, &dataTokenizer, "p ");
                    mesh.n     = ReadVec3Pointer(&tokenizer, &dataTokenizer, "n ");
                    mesh.uv    = ReadVec2Pointer(&tokenizer, &dataTokenizer, "uv ");
                    mesh.indices =
                        (u32 *)ReadDataPointer(&tokenizer, &dataTokenizer, "indices ");
                    mesh.numVertices = ReadInt(&tokenizer, "v ");
                    mesh.numIndices  = ReadInt(&tokenizer, "i ");
                    mesh.numFaces =
                        mesh.numIndices ? mesh.numIndices / 3 : mesh.numVertices / 3;
                    AddMaterial();

                    threadMemoryStatistics[threadIndex].totalShapeMemory +=
                        mesh.numVertices * (sizeof(Vec3f) * 2 + sizeof(Vec2f)) +
                        mesh.numIndices * sizeof(u32);
                }
                else
                {
                    Assert(0);
                }

                PrimitiveIndices &ids = indices.Last();
                // if (baseScene->materials[ids.GetIndex()])
                // {
                //     Displace
                // }
            }
            scene->numPrimitives = shapes.totalCount;
            scene->primitives    = PushArrayNoZero(arena, Mesh, shapes.totalCount);
            scene->primIndices = PushArrayNoZero(arena, PrimitiveIndices, indices.totalCount);
            shapes.Flatten((Mesh *)scene->primitives);
            indices.Flatten(scene->primIndices);
        }
        else if (Advance(&tokenizer, "MATERIALS_START "))
        {
            hasMaterials = true;
            // TODO: multithread this
            materialHashMap = CreateMaterials(arena, temp.arena, &tokenizer, directory);
            state->map      = materialHashMap;
        }
        else
        {
            Error(0, "Invalid section header.\n");
        }
    }
    files.Flatten(scene->childScenes);

    scheduler.Wait(&counter);

    OS_UnmapFile(tokenizer.input.str);
    BuildSettings settings;

    {
        if (baseFile && !hasTransforms)
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
    }

    if (!isLeaf)
    {
        BuildTLASBVH(arenas, settings, scene);
    }
    else
    {
        Assert(!hasTransforms);
        Assert(scene->numPrimitives);
        if (type == GeometryType::QuadMesh)
        {
            BuildQuadBVH(arenas, settings, scene);
        }
        else if (type == GeometryType::TriangleMesh)
        {
            BuildTriangleBVH(arenas, settings, scene);
        }
        else
        {
            Error(0, "No shapes specified\n");
        }
    }

    ScratchEnd(temp);
    if (c) c->count.fetch_sub(1, std::memory_order_acq_rel);
}

void LoadScene(Arena **arenas, string directory, string filename, AffineSpace *t)
{
    TempArena temp = ScratchStart(0, 0);
    Arena *arena   = arenas[GetThreadIndex()];

    RTSceneLoadState state;
    state.table.count = 1024;
    state.table.nodes =
        PushArray(temp.arena, std::atomic<SceneLoadTable::Node *>, state.table.count);

    Scene *scene = GetScene();
    LoadRTScene(arenas, &state, &scene->scene, directory, filename, 0, t, true);
    ScratchEnd(temp);
}

void BuildTLASBVH(Arena **arenas, BuildSettings &settings, ScenePrimitives *scene)
{
    TempArena temp = ScratchStart(0, 0);
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

template <GeometryType type>
void BuildBVH(Arena **arenas, BuildSettings &settings, ScenePrimitives *scene)

{
    TempArena temp = ScratchStart(0, 0);
    Mesh *meshes   = (Mesh *)scene->primitives;
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

void BuildTriangleBVH(Arena **arenas, BuildSettings &settings, ScenePrimitives *scene)
{
    BuildBVH<GeometryType::TriangleMesh>(arenas, settings, scene);
}

void BuildQuadBVH(Arena **arenas, BuildSettings &settings, ScenePrimitives *scene)
{
    BuildBVH<GeometryType::QuadMesh>(arenas, settings, scene);
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

} // namespace rt
