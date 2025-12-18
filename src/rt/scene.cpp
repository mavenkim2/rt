#include "scene.h"
#include "hash.h"
#include "macros.h"
#include "integrate.h"
#include "memory.h"
#include "parallel.h"
#include "radix_sort.h"
#include "simd_integrate.h"
#include "spectrum.h"
#include "shader_interop/hit_shaderinterop.h"
#include "graphics/ptex.h"
#include "scene_load.h"
#include "string.h"
#ifdef USE_GPU
#include "gpu_scene.h"
#else
#include "cpu_scene.h"
#endif
// #include <nanovdb/NanoVDB.h>
// #include <nanovdb/util/GridHandle.h>
// #include <nanovdb/util/IO.h>
// #include <nanovdb/util/SampleFromVoxels.h>
#include <atomic>
#include <cwchar>
#include <iterator>
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

struct RTSceneLoadState
{
    SceneLoadTable table;
    HashIndex materialHashMap;
    std::vector<string> materialNames;
    std::vector<MaterialHandle> materialHandles;

    Mutex mutex;
    std::vector<ScenePrimitives *> scenes;

    Mutex lightMutex;
    std::vector<Light *> *lights;
};

Scene *scene_;

GPUMaterial Material::ConvertToGPU() { return {}; }

GPUMaterial DiffuseMaterial::ConvertToGPU()
{
    GPUMaterial result = {};
    result.type        = GPUMaterialType::Diffuse;
    return result;
}

GPUMaterial DielectricMaterial::ConvertToGPU()
{
    GPUMaterial result = {};
    result.type        = GPUMaterialType::Dielectric;
    result.ior         = eta;
    return result;
}

GPUMaterial DisneyMaterial::ConvertToGPU()
{
    GPUMaterial result     = {};
    result.type            = GPUMaterialType::Disney;
    result.diffTrans       = diffTrans;
    result.baseColor       = baseColor;
    result.specTrans       = specTrans;
    result.clearcoatGloss  = clearcoatGloss;
    result.scatterDistance = scatterDistance;
    result.clearcoat       = clearcoat;
    result.specularTint    = specularTint;
    result.ior             = ior;
    result.metallic        = metallic;
    result.flatness        = flatness;
    result.sheen           = sheen;
    result.sheenTint       = sheenTint;
    result.anisotropic     = anisotropic;
    result.alpha           = alpha;
    result.roughness       = roughness;
    result.thin            = thin;
    return result;
}

GPUMediumType NanovdbMedium::GetType() { return GPUMediumType::Nanovdb; }

void Texture::Start(ShadingThreadState *) {}

PtexTexture::PtexTexture(string filename, FilterType filterType, ColorEncoding encoding,
                         f32 scale)
    : filename(filename), filterType(filterType), encoding(encoding), scale(scale)
{
}

f32 PtexTexture::EvaluateFloat(SurfaceInteraction &si, const Vec4f &filterWidths)
{
    f32 result = 0.f;
    EvaluateHelper<1>(si, filterWidths, &result);
    return result * scale;
}

SampledSpectrum PtexTexture::EvaluateAlbedo(SurfaceInteraction &si, SampledWavelengths &lambda,
                                            const Vec4f &filterWidths)
{
    Vec3f result = {};
    EvaluateHelper<3>(si, filterWidths, result.e);
    Assert(!IsNaN(result[0]) && !IsNaN(result[1]) && !IsNaN(result[2]));
    return Texture::EvaluateAlbedo(result * scale, lambda);
}

void PtexTexture::Start(ShadingThreadState *state)
{
    GetDebug()->texture = this;

    Ptex::String error;
    Ptex::PtexTexture *texture = cache->get((char *)filename.str, error);
    if (!texture)
    {
        printf("%s\n", error.c_str());
        Assert(0);
    }
}

void PtexTexture::Stop()
{
    Ptex::String error;
    Ptex::PtexTexture *texture = cache->get((char *)filename.str, error);
    if (!texture)
    {
        printf("%s\n", error.c_str());
        Assert(0);
    }
    texture->release();
    texture->release();
}

template <i32 c>
void PtexTexture::EvaluateHelper(Ptex::PtexTexture *texture, SurfaceInteraction &intr,
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
void PtexTexture::EvaluateHelper(SurfaceInteraction &intr, const Vec4f &filterWidths,
                                 f32 *result)
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

Texture *ReadTexture(Arena *arena, Tokenizer *tokenizer, string directory, int *index = 0,
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

        PtexTexture tex(PushStr8F(arena, "%S%S\0", directory, filename), type, encoding,
                        scale);

        Scene *scene = GetScene();
        scene->ptexTextures.push_back(tex);

        if (index) *index = (int)(scene->ptexTextures.size() - 1);
        return &scene->ptexTextures[scene->ptexTextures.size() - 1];
    }
    else
    {
        ErrorExit(0, "Texture type not supported yet");
    }
    return 0;
}

Texture *ParseTexture(Arena *arena, Tokenizer *tokenizer, string directory, int *index,
                      FilterType type, ColorEncoding encoding)
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
            return ReadTexture(arena, tokenizer, directory, index, type, encoding);
        }
        break;
        default: Assert(0); return 0;
    }
}

Texture *ParseDisplacement(Arena *arena, Tokenizer *tokenizer, string directory)
{
    if (Advance(tokenizer, "displacement "))
        return ParseTexture(arena, tokenizer, directory, 0, FilterType::Bspline,
                            ColorEncoding::Linear);
    return 0;
}

DiffuseAreaLight *ParseAreaLight(Arena *arena, Tokenizer *tokenizer, AffineSpace *space,
                                 int sceneID, int geomID)
{
    DiffuseAreaLight *light = 0;

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
        f32 scale = 1.f / SpectrumToPhotometric(RGBColorSpace::sRGB->illuminant);
        light     = PushStructConstruct(arena, DiffuseAreaLight)(dss, space, scale, geomID,
                                                             sceneID, LightType::Area);
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

void CreateMaterials(RTSceneLoadState *state, Arena *arena, Arena *tempArena,
                     Tokenizer *tokenizer, string directory)
{
    Scene *scene = GetScene();

    std::vector<string> &materialNames           = state->materialNames;
    std::vector<MaterialHandle> &materialHandles = state->materialHandles;

    std::vector<Material *> materialsList;
    scene->ptexTextures                = StaticArray<PtexTexture>(arena, 50000);
    NullMaterial *nullMaterial         = PushStructConstruct(arena, NullMaterial)();
    nullMaterial->ptexReflectanceIndex = -1;
    materialsList.push_back(nullMaterial);

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

        materialNames.push_back(PushStr8Copy(tempArena, materialName));
        materialHandles.push_back(MaterialHandle(materialTypeIndex, materialsList.size()));

        materialsList.emplace_back();

        Material **material      = &materialsList.back();
        int ptexReflectanceIndex = -1;
        switch (materialTypeIndex)
        {
            case MaterialTypes::Interface:
            {
                *material = PushStructConstruct(arena, NullMaterial)();
            }
            break;
            case MaterialTypes::Disney:
            {
                *material    = PushStructConstruct(arena, DisneyMaterial)();
                bool success = Advance(tokenizer, "t ");
                Assert(success);
                string ptexFilename = ReadWord(tokenizer);
                if (!(ptexFilename == "none"))
                {
                    PtexTexture tex(PushStr8F(arena, "%S%S\0", directory, ptexFilename));

                    Scene *scene = GetScene();
                    scene->ptexTextures.push_back(tex);

                    ptexReflectanceIndex = (int)(scene->ptexTextures.size() - 1);
                }
                SkipToNextChar(tokenizer);
                struct DiskDisneyMaterial
                {
                    float diffTrans;
                    Vec4f baseColor;
                    float specTrans;
                    float clearcoatGloss;
                    Vec3f scatterDistance;
                    float clearcoat;
                    float specularTint;
                    float ior;
                    float metallic;
                    float flatness;
                    float sheen;
                    float sheenTint;
                    float anisotropic;
                    float alpha;
                    float roughness;
                    bool thin;
                };
                DiskDisneyMaterial diskMaterial;
                GetPointerValue(tokenizer, &diskMaterial);

                DisneyMaterial *disneyMaterial = (DisneyMaterial *)(*material);

                disneyMaterial->diffTrans       = diskMaterial.diffTrans;
                disneyMaterial->baseColor       = diskMaterial.baseColor;
                disneyMaterial->specTrans       = diskMaterial.specTrans;
                disneyMaterial->clearcoatGloss  = diskMaterial.clearcoatGloss;
                disneyMaterial->scatterDistance = diskMaterial.scatterDistance;
                disneyMaterial->clearcoat       = diskMaterial.clearcoat;
                disneyMaterial->specularTint    = diskMaterial.specularTint;
                disneyMaterial->ior             = diskMaterial.ior;
                disneyMaterial->metallic        = diskMaterial.metallic;
                disneyMaterial->flatness        = diskMaterial.flatness;
                disneyMaterial->sheen           = diskMaterial.sheen;
                disneyMaterial->sheenTint       = diskMaterial.sheenTint;
                disneyMaterial->anisotropic     = diskMaterial.anisotropic;
                disneyMaterial->alpha           = diskMaterial.alpha;
                disneyMaterial->roughness       = diskMaterial.roughness;
                disneyMaterial->thin            = diskMaterial.thin;
                SkipToNextChar(tokenizer);
            }
            break;
            case MaterialTypes::Diffuse:
            {
                bool result = Advance(tokenizer, "reflectance ");
                Texture *reflectance;

                if (!result) reflectance = PushStructConstruct(arena, ConstantTexture)(0.5f);
                else
                    reflectance =
                        ParseTexture(arena, tokenizer, directory, &ptexReflectanceIndex);

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
        (*material)->ptexReflectanceIndex = ptexReflectanceIndex;
    }

    // Join
    scene->materials = StaticArray<Material *>(arena, materialsList.size());
    MemoryCopy(scene->materials.data, materialsList.data(),
               sizeof(Material *) * materialsList.size());
    scene->materials.size_ = materialsList.size();
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

Mesh ProcessMesh(Arena *arena, Mesh &mesh)
{
    Mesh newMesh        = {};
    newMesh.numVertices = mesh.numVertices;
    newMesh.numIndices  = mesh.numIndices;
    newMesh.numFaces    = mesh.numFaces;
    newMesh.p = PushArrayNoZeroTagged(arena, Vec3f, mesh.numVertices, MemoryType_Shape);
    MemoryCopy(newMesh.p, mesh.p, sizeof(Vec3f) * mesh.numVertices);
    if (mesh.n)
    {
        newMesh.n = PushArrayNoZeroTagged(arena, Vec3f, mesh.numVertices, MemoryType_Shape);
        MemoryCopy(newMesh.n, mesh.n, sizeof(Vec3f) * mesh.numVertices);
    }
    if (mesh.uv)
    {
        newMesh.uv = PushArrayNoZeroTagged(arena, Vec2f, mesh.numVertices, MemoryType_Shape);
        MemoryCopy(newMesh.uv, mesh.uv, sizeof(Vec2f) * mesh.numVertices);
    }
    if (mesh.indices)
    {
        newMesh.indices = PushArrayNoZero(arena, u32, mesh.numIndices);
        MemoryCopy(newMesh.indices, mesh.indices, sizeof(u32) * mesh.numIndices);
    }
    return newMesh;
}

void AddMaterialAndLights(Arena *arena, ScenePrimitives *scene, int sceneID, GeometryType type,
                          string directory, AffineSpace &worldFromRender,
                          AffineSpace &renderFromWorld, Tokenizer &tokenizer,
                          RTSceneLoadState *state, Mesh &mesh,
                          ChunkedLinkedList<Mesh, MemoryType_Shape> &shapes,
                          ChunkedLinkedList<PrimitiveIndices, MemoryType_Shape> &indices,
                          ChunkedLinkedList<Light *, MemoryType_Light> &lights)
{
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
        Assert(state->materialHandles.size() && state->materialNames.size());
        string materialName = ReadWord(&tokenizer);

        int hash       = Hash(materialName);
        materialHandle = MaterialHandle(MaterialTypes::Diffuse, 1);
        for (int hashIndex = state->materialHashMap.FirstInHash(hash); hashIndex != -1;
             hashIndex     = state->materialHashMap.NextInHash(hashIndex))
        {
            if (state->materialNames[hashIndex] == materialName)
            {
                materialHandle = state->materialHandles[hashIndex];
                break;
            }
        }
    }

    if (Advance(&tokenizer, "transform "))
    {
        u32 transformIndex = ReadInt(&tokenizer);
        SkipToNextChar(&tokenizer);
    }

    // Check for area light
    if (Advance(&tokenizer, "a "))
    {
        ErrorExit(type == GeometryType::QuadMesh, "Only quad area lights supported for now\n");
        Assert(transform);

        DiffuseAreaLight *areaLight =
            ParseAreaLight(arena, &tokenizer, transform, sceneID, shapes.totalCount - 1);
        lightHandle      = LightHandle(LightClass::DiffuseAreaLight, lights.totalCount);
        lights.AddBack() = areaLight;
        light            = areaLight;
    }

    // Check for medium
    if (Advance(&tokenizer, "medium "))
    {
        Assert(0);
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
}

void LoadRTScene(Arena **arenas, Arena **tempArenas, RTSceneLoadState *state,
                 ScenePrimitives *scene, string directory, string filename,
                 Scheduler::Counter *counter, AffineSpace *renderFromWorld = 0,
                 bool baseFile = false)

{
    scene->filename = filename;
    Assert(GetFileExtension(filename) == "rtscene");

    BeginMutex(&state->mutex);
    int sceneID = (int)state->scenes.size();
    state->scenes.push_back(scene);
    EndMutex(&state->mutex);

    scene->sceneIndex = sceneID;
    TempArena temp    = ScratchStart(0, 0);

    u32 threadIndex  = GetThreadIndex();
    Arena *arena     = arenas[threadIndex];
    Arena *tempArena = tempArenas[threadIndex];

    auto *table = &state->table;

    string fullFilePath = StrConcat(temp.arena, directory, filename);
    string dataPath =
        PushStr8F(temp.arena, "%S%S.rtdata", directory, RemoveFileExtension(filename));
    Tokenizer tokenizer;
    tokenizer.input  = OS_MapFileRead(fullFilePath);
    tokenizer.cursor = tokenizer.input.str;

    bool hasMagic = Advance(&tokenizer, "RTSCENE_START ");
    ErrorExit(hasMagic, "RTScene file missing magic.\n");

    Tokenizer dataTokenizer;
    dataTokenizer.input  = OS_MapFileRead(dataPath);
    dataTokenizer.cursor = dataTokenizer.input.str;

    hasMagic = Advance(&dataTokenizer, "DATA_START ");
    ErrorExit(hasMagic, "RTScene data section missing magic.\n");
    bool hasTransforms = Advance(&dataTokenizer, "TRANSFORM_START ");
    if (hasTransforms)
    {
        Advance(&dataTokenizer, "Count ");
        u32 count            = ReadInt(&dataTokenizer);
        scene->numTransforms = count;
        dataTokenizer.cursor++;
        scene->affineTransforms =
            PushArrayNoZeroTagged(arena, AffineSpace, count, MemoryType_Instance);
        AffineSpace *dataTransforms = reinterpret_cast<AffineSpace *>(dataTokenizer.cursor);
        if (baseFile && renderFromWorld)
        {
            ParallelForLoop(0, count, 1024, 1024, [&](u32 jobID, u32 i) {
                scene->affineTransforms[i] = *renderFromWorld * dataTransforms[i];
            });
        }
        else
        {
            MemoryCopy(scene->affineTransforms, dataTransforms, sizeof(AffineSpace) * count);
        }
    }
    else if (FindSubstring(filename, "_rtshape_", 0, MatchFlag_CaseInsensitive) !=
             filename.size)
    {
        scene->affineTransforms = GetScene()->scene.affineTransforms;
        scene->numTransforms    = GetScene()->scene.numTransforms;
    }

    bool isLeaf       = true;
    GeometryType type = GeometryType::Max;
    ChunkedLinkedList<ScenePrimitives *, MemoryType_Instance> files(temp.arena, 32);

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

            scene->primitives =
                PushArrayNoZeroTagged(arena, Instance, instanceCount, MemoryType_Instance);
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

                u32 id = files.Length();
                SceneLoadTable::Node *head =
                    table->nodes[index].load(std::memory_order_acquire);
                for (;;)
                {
                    auto *node = head;
                    while (node)
                    {
                        if (node->filename.size && node->filename == includeFile)
                        {
                            files.AddBack() = node->scene;

                            int depth = node->scene->depth.load(std::memory_order_acquire);
                            while (depth > scene->depth + 1 &&
                                   !node->scene->depth.compare_exchange_weak(
                                       depth, scene->depth + 1, std::memory_order_release,
                                       std::memory_order_relaxed));
                            break;
                        }
                        node = node->next;
                    }
                    if (node) break;

                    u64 pos = ArenaPos(arena);

                    auto *newNode                 = PushStruct(arena, SceneLoadTable::Node);
                    newNode->filename             = PushStr8Copy(arena, includeFile);
                    ScenePrimitives *includeScene = PushStruct(arena, ScenePrimitives);
                    includeScene->depth           = scene->depth + 1;
                    newNode->scene                = includeScene;
                    newNode->next                 = head;

                    if (table->nodes[index].compare_exchange_weak(head, newNode,
                                                                  std::memory_order_release,
                                                                  std::memory_order_relaxed))
                    {
                        files.AddBack() = includeScene;
                        scheduler.Schedule(counter, [=](u32 jobID) {
                            LoadRTScene(arenas, tempArenas, state, includeScene, directory,
                                        newNode->filename, counter, renderFromWorld);
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
            ChunkedLinkedList<Mesh, MemoryType_Shape> shapes(temp.arena, 1024);
            ChunkedLinkedList<PrimitiveIndices, MemoryType_Shape> indices(temp.arena, 1024);
            ChunkedLinkedList<Light *, MemoryType_Light> lights(temp.arena);

            AffineSpace worldFromRender = Inverse(*renderFromWorld);

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
                    type = GeometryType::QuadMesh;
                    Mesh mesh;
                    AddMesh(mesh, 4);

                    Assert(mesh.numVertices == 4);
                    // TODO: this is a hack
                    if (mesh.numFaces == 1 && mesh.numVertices == 4 && mesh.numIndices == 6)
                    {
                        mesh.indices    = 0;
                        mesh.numIndices = 0;
                    }

                    shapes.AddBack() = ProcessMesh(arena, mesh);
                    AddMaterialAndLights(arena, scene, sceneID, type, directory,
                                         worldFromRender, *renderFromWorld, tokenizer, state,
                                         shapes.Last(), shapes, indices, lights);

                    threadMemoryStatistics[threadIndex].totalShapeMemory +=
                        mesh.numVertices * (sizeof(Vec3f) * 2 + sizeof(Vec2f)) +
                        mesh.numIndices * sizeof(u32);
                }
                else if (Advance(&tokenizer, "Tri "))
                {
                    Assert(type == GeometryType::Max || type == GeometryType::TriangleMesh);
                    type = GeometryType::TriangleMesh;
                    Mesh mesh;
                    AddMesh(mesh, 3);

                    shapes.AddBack() = ProcessMesh(arena, mesh);
                    AddMaterialAndLights(arena, scene, sceneID, type, directory,
                                         worldFromRender, *renderFromWorld, tokenizer, state,
                                         shapes.Last(), shapes, indices, lights);

                    threadMemoryStatistics[threadIndex].totalShapeMemory +=
                        mesh.numVertices * (sizeof(Vec3f) * 2 + sizeof(Vec2f)) +
                        mesh.numIndices * sizeof(u32);
                }
                else if (Advance(&tokenizer, "Catclark "))
                {
                    Assert(type == GeometryType::Max || type == GeometryType::CatmullClark);
                    type = GeometryType::CatmullClark;
                    Mesh mesh;
                    AddMesh(mesh, 4);

                    shapes.AddBack() = ProcessMesh(tempArena, mesh);
                    AddMaterialAndLights(arena, scene, sceneID, type, directory,
                                         worldFromRender, *renderFromWorld, tokenizer, state,
                                         shapes.Last(), shapes, indices, lights);
                }
                else
                {
                    Assert(0);
                }
            }

            scene->numPrimitives = shapes.totalCount;
            Assert(shapes.totalCount == indices.totalCount);
            scene->primitives = PushArrayNoZero(arena, Mesh, scene->numPrimitives);
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

            scene->primIndices = PushArrayNoZeroTagged(arena, PrimitiveIndices,
                                                       indices.totalCount, MemoryType_Shape);
            indices.Flatten(scene->primIndices);
        }
        else if (Advance(&tokenizer, "Geo Filename "))
        {
            string geoFilename        = ReadWord(&tokenizer);
            type                      = GeometryType::TriangleMesh;
            scene->virtualGeoFilename = PushStr8Copy(arena, geoFilename);
            SkipToNextChar(&tokenizer);
        }
        else if (Advance(&tokenizer, "MATERIALS_START "))
        {
            hasMaterials = true;
            CreateMaterials(state, arena, tempArena, &tokenizer, directory);
        }
        else
        {
            ErrorExit(0, "Invalid section header.\n");
        }
    }
    files.Flatten(scene->childScenes);

    OS_UnmapFile(tokenizer.input.str);
    OS_UnmapFile(dataTokenizer.input.str);
    ScratchEnd(temp);

#ifndef USE_GPU
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
#endif

    scene->geometryType = type;
}

int LoadScene(RenderParams2 *params, Arena **tempArenas, string directory, string filename)
{
    TempArena temp = ScratchStart(0, 0);
    Arena *arena   = params->arenas[GetThreadIndex()];

    u32 numProcessors      = OS_NumProcessors();
    RTSceneLoadState state = {};
    state.table.count      = 1024;
    state.table.nodes =
        PushArray(temp.arena, std::atomic<SceneLoadTable::Node *>, state.table.count);
    state.lights = &GetScene()->lights;

    Scene *scene = GetScene();

    Scheduler::Counter counter = {};
    LoadRTScene(params->arenas, tempArenas, &state, &scene->scene, directory, filename,
                &counter, &params->renderFromWorld, true);
    scheduler.Wait(&counter);

    struct SceneHandle
    {
        u64 sortKey;
        u32 index;
    };

    SceneHandle *handles = PushArrayNoZero(temp.arena, SceneHandle, state.scenes.size());
    for (u32 handleIndex = 0; handleIndex < state.scenes.size() - 1; handleIndex++)
    {
        ScenePrimitives *scene = state.scenes[handleIndex + 1];
        SceneHandle handle;
        handle.sortKey       = MurmurHash64A(scene->filename.str, scene->filename.size, 0);
        handle.index         = handleIndex + 1;
        handles[handleIndex] = handle;
    }
    SortHandles(handles, state.scenes.size() - 1);

    int numScenes            = (int)state.scenes.size();
    ScenePrimitives **scenes = PushArrayNoZero(arena, ScenePrimitives *, state.scenes.size());

    scenes[0] = state.scenes[0];
    for (u32 handleIndex = 0; handleIndex < state.scenes.size() - 1; handleIndex++)
    {
        scenes[handleIndex + 1] = state.scenes[handles[handleIndex].index];
    }

    state.scenes.clear();
    SetScenes(scenes);

    // for (u32 i = 0; i < numProcessors; i++)
    // {
    //     ArenaRelease(tempArenas[i]);
    // }

    ScratchEnd(temp);
    return numScenes;
}

template <GeometryType type>
void ComputeTessellationParams(Mesh *meshes, TessellationParams *params, u32 start, u32 count)
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
        params[i].bounds             = bounds;
        params[i].currentMinDistance = pos_inf;
    }
}

Material *Scene::GetMaterial(SurfaceInteraction &si)
{
    return materials[MaterialHandle(si.materialIDs).GetIndex()];
}

} // namespace rt
