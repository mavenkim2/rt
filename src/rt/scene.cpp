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
    Node *nodes;
    u32 count;
    TicketMutex *mutexes;
};

// f32 ProcessFloatTexture(AttributeIterator *iterator, Vec2f uv, const Vec4f &filterWidths)
TEXTURE_CALLBACK(ProcessFloatTexture) { *result = iterator->ReadFloat(); }

__forceinline SampledSpectrum ProcessAlbedoTexture(TextureCallback &callback,
                                                   AttributeIterator *iterator,
                                                   SurfaceInteraction &intr,
                                                   SampledWavelengths &lambda,
                                                   const Vec4f &filterWidths)
{
    Vec3f result;
    callback(iterator, intr, lambda, filterWidths, result.e);
    return RGBAlbedoSpectrum(*RGBColorSpace::sRGB, result).Sample(lambda);
}

TEXTURE_CALLBACK(ProcessIOR)
{
    lambda.TerminateSecondary();
    auto spec = (PiecewiseLinearSpectrum *)iterator->ReadPointer();
    *result   = spec->Evaluate(lambda[0]);
}

__forceinline void ProcessRoughness(TextureCallback *callbacks, AttributeIterator *iterator,
                                    SurfaceInteraction &intr, SampledWavelengths &lambda,
                                    const Vec4f &filterWidths, f32 &u, f32 &v)
{
    callbacks[iterator->callbackCount++](iterator, intr, lambda, filterWidths, &u);
    f32 vR = -1.f;
    callbacks[iterator->callbackCount++](iterator, intr, lambda, filterWidths, &vR);
    if (vR == -1.f) vR = u;
}

TEXTURE_CALLBACK(ProcessPtexTexture)
{
    string filename = iterator->ReadString();
    const Vec2f &uv = intr.uv;
    u32 faceIndex   = intr.faceIndices;

    Assert(cache);
    Ptex::String error;
    Ptex::PtexTexture *texture = cache->get((char *)filename.str, error);
    Assert(texture);
    u32 numFaces = texture->getInfo().numFaces;
    Assert(faceIndex < numFaces);
    Ptex::PtexFilter::Options opts(Ptex::PtexFilter::FilterType::f_bspline);
    Ptex::PtexFilter *filter = Ptex::PtexFilter::getFilter(texture, opts);

    i32 nc = texture->numChannels();

    // TODO: ray differentials
    // Vec2f uv(0.5f, 0.5f);

    f32 out[3];
    filter->eval(out, 0, nc, faceIndex, uv[0], uv[1], filterWidths[0], filterWidths[1],
                 filterWidths[2], filterWidths[3]);

    texture->release();
    filter->release();

    // Convert to srgb

    if (nc == 1) *result = out[0];
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
        out[0] = Pow(out[0], 2.2f);
        out[1] = Pow(out[1], 2.2f);
        out[2] = Pow(out[2], 2.2f);
        // }
        for (i32 i = 0; i < nc; i++)
        {
            out[i] *= iterator->ReadFloat(1.f);
        }

        result[0] = out[0];
        result[1] = out[1];
        result[2] = out[2];
    }
}

// TODO: instead of this junk I should just create a preprocessor (i.e. write a text file
// that is code)
//////////////////////////////////////////////////////////////////////////////////////////////

#if 0
{
#define ROUGHNESS_TMPL(...)    TextureType ur, TextureType vr, TextureType r
#define IOR_TMPL(...)          typename Spectrum __VA_ARGS__
#define DISPLACEMENT_TMPL(...) TextureType disp __VA_ARGS__

#define EXPAND_2(...)         __VA_ARGS__
#define IIF(c, falseVal, ...) CONCAT(IF_, c)(falseVal, __VA_ARGS__)
#define IF_0(falseVal, ...)   falseVal
#define IF_1(falseVal, ...)   __VA_ARGS__

#define CHECK_HAS_COMMA(...) BOOL_ARGS(__VA_ARGS__)
#define IS_EMPTY(...)        IS_EMPTY_HELPER(__VA_ARGS__)
#define IS_EMPTY_HELPER(...) CHECK_HAS_COMMA(__VA_ARGS__)
#define CHECK_TMPL(x)        IS_EMPTY(CONCAT(x, _TMPL)(, ))

static_assert(CHECK_TMPL(IOR) == 1, "CHECK_TMPL SHOULD BE 1");

#define TMPL_DECL(x) IIF(CHECK_TMPL(x), TextureType x, CONCAT(x, _TMPL)())

#define MATERIAL_TEMPLATE_DECL_HELPER(x, ...)                                                 \
    EXPAND(CONCAT(RECURSE__, x)(TMPL_DECL, __VA_ARGS__))
#define MATERIAL_TEMPLATE_DECL(...)                                                           \
    MATERIAL_TEMPLATE_DECL_HELPER(COUNT_ARGS(__VA_ARGS__), __VA_ARGS__)

//////////////////////////////////////////////////////////////////////////////////////////////

#define SHADER_PARAMETER_FUNC_ROUGHNESS(...)                                                  \
    static_assert(                                                                            \
        (ur != TextureType::None && vr != TextureType::None && r == TextureType::None) ||     \
            (ur == TextureType::None && vr == TextureType::None && r != TextureType::None),   \
        "Cannot specify both anisotropic and isotropic roughness");                           \
    f32 uRoughness;                                                                           \
    f32 vRoughness;                                                                           \
    if constexpr (r == TextureType::None)                                                     \
    {                                                                                         \
        uRoughness = ProcessTexture<r>(itr);                                                  \
        uRoughness = Sqr(uRoughness);                                                         \
        vRoughness = uRoughness;                                                              \
    }                                                                                         \
    else                                                                                      \
    {                                                                                         \
        uRoughness = Sqr(ProcessTexture<ur>(itr));                                            \
        vRoughness = Sqr(ProcessTexture<vr>(itr));                                            \
    }                                                                                         \
    __VA_ARGS__

#define SHADER_PARAMETER_FUNC_IOR(...)                                                        \
    if constexpr (!std::is_same_v<Spectrum, ConstantSpectrum>)                                \
    {                                                                                         \
        lambda.TerminateSecondary();                                                          \
    }                                                                                         \
    f32 eta = ProcessIOR<Spectrum>(itr);                                                      \
    __VA_ARGS__

#define SHADER_PARAMETER_FUNC_DISPLACEMENT(...)                                               \
    (void)0;                                                                                  \
    __VA_ARGS__

// #define DEFAULT_SHADER_PARAMETER_FUNC(x) auto x##_var = ProcessTexture<x>(itr);

#define CHECK_FUNC(x) IS_EMPTY(CONCAT(SHADER_PARAMETER_FUNC_, x)(, ))

static_assert(CHECK_FUNC(ROUGHNESS) == 1, "CHECK_FUNC SHOULD BE 1");

#define FUNC_DECL(x)                                                                          \
    IIF(CHECK_FUNC(x), DEFAULT_SHADER_PARAMETER_FUNC(x), CONCAT(SHADER_PARAMETER_FUNC_, x)())
#define MATERIAL_FUNC_DECL_HELPER(x, ...) EXPAND(CONCAT(RECURSE_, x)(FUNC_DECL, __VA_ARGS__))
#define MATERIAL_FUNC_DECL(...)           MATERIAL_FUNC_DECL_HELPER(COUNT_ARGS(__VA_ARGS__), __VA_ARGS__)

//////////////////////////////////////////////////////////////////////////////////////////////

#define RETURN_TYPE_ROUGHNESS(...)                                                            \
    TrowbridgeReitzDistribution(uRoughness, vRoughness) __VA_ARGS__
#define RETURN_TYPE_IOR(...) eta __VA_ARGS__
#define RETURN_TYPE_DISPLACEMENT(...)

#define CHECK_RETURN(x) IS_EMPTY(CONCAT(RETURN_TYPE_, x)(, ))

static_assert(CHECK_RETURN(ROUGHNESS) == 1, "CHECK_RETURN SHOULD BE 1");

#define RETURN_DECL(x, count)                                                                 \
    IIF(CHECK_RETURN(x), , IIF(count, CONCAT(RETURN_TYPE_, x)(), CONCAT(RETURN_TYPE_, x)(, )))

#define MATERIAL_RETURN_DECL_HELPER(x, ...)                                                   \
    EXPAND(CONCAT(RECURSE2_, x)(RETURN_DECL, __VA_ARGS__))
#define MATERIAL_RETURN_DECL(...)                                                             \
    MATERIAL_RETURN_DECL_HELPER(COUNT_ARGS(__VA_ARGS__), __VA_ARGS__)

//////////////////////////////////////////////////////////////////////////////////////////////
#define MATERIAL_FUNCTION_HEADER(name)                                                        \
    void name(Arena *arena, AttributeIterator *itr, SurfaceInteraction &si,                   \
              SampledWavelengths &lambda, BSDFBase<BxDF> *result)

#define START_MATERIAL_SHADER(name, BxDFType, ...)                                            \
    template <MATERIAL_TEMPLATE_DECL(__VA_ARGS__)>                                            \
    MATERIAL_FUNCTION_HEADER(CONCAT(ShaderEvaluate_, name))

#define DEFINE_MATERIAL_SHADER(name, BxDFType, ...)                                           \
    START_MATERIAL_SHADER(name, BxDFType, __VA_ARGS__)                                        \
    {                                                                                         \
        MATERIAL_FUNC_DECL(__VA_ARGS__)                                                       \
        BxDFType *bxdf = PushStruct(arena, BxDFType);                                         \
        Vec4lfn filterWidths(.75f);                                                           \
        new (bxdf) BxDFType(MATERIAL_RETURN_DECL(__VA_ARGS__));                               \
        new (result) BSDF(bxdf, si.shading.dpdu, si.shading.n);                               \
    }

// DEFINE_MATERIAL_SHADER(Dielectric, DielectricBxDF, DISPLACEMENT, IOR, ROUGHNESS)
}
#endif

__forceinline void Material::Shade(Arena *arena, SurfaceInteraction &si,
                                   SampledWavelengths &lambda, BSDF *result)
{
    AttributeIterator itr(key);
    BxDF bxdf;
    shade(funcs, arena, &itr, si, lambda, bxdf);
    new (result) BSDF(bxdf, si.shading.dpdu, si.shading.n);
}

MATERIAL_FUNCTION_HEADER(ShaderEvaluate_Null) { result = {}; }
MATERIAL_FUNCTION_HEADER(ShaderEvaluate_Diffuse)
{
    Vec4f filterWidths(.75f);
    SampledSpectrum s = ProcessAlbedoTexture(callbacks[0], itr, si, lambda, filterWidths);

    DiffuseBxDF *bxdf = PushStruct(arena, DiffuseBxDF);
    new (bxdf) DiffuseBxDF(s);
    result = bxdf;
}

MATERIAL_FUNCTION_HEADER(ShaderEvaluate_DiffuseTransmission)
{
    Vec4f filterWidths(.75f);
    SampledSpectrum r =
        ProcessAlbedoTexture(callbacks[itr->callbackCount++], itr, si, lambda, filterWidths);
    SampledSpectrum t =
        ProcessAlbedoTexture(callbacks[itr->callbackCount++], itr, si, lambda, filterWidths);

    DiffuseTransmissionBxDF *bxdf = PushStruct(arena, DiffuseTransmissionBxDF);
    new (bxdf) DiffuseTransmissionBxDF(r, t);
    result = bxdf;
}

void ShaderEvaluate_DielectricHelper(TextureCallback *callbacks, Arena *arena,
                                     AttributeIterator *itr, SurfaceInteraction &si,
                                     SampledWavelengths &lambda, DielectricBxDF *result)
{
    Vec4lfn filterWidths(.75f);

    f32 uRoughness, vRoughness;
    ProcessRoughness(callbacks, itr, si, lambda, filterWidths, uRoughness, vRoughness);

    bool remapRoughness = itr->ReadBool(true);

    uRoughness = remapRoughness ? TrowbridgeReitzDistribution::RoughnessToAlpha(uRoughness)
                                : uRoughness;
    vRoughness = remapRoughness ? TrowbridgeReitzDistribution::RoughnessToAlpha(vRoughness)
                                : vRoughness;
    f32 eta;
    callbacks[itr->callbackCount++](itr, si, lambda, filterWidths, &eta);

    new (result) DielectricBxDF(eta, TrowbridgeReitzDistribution(uRoughness, vRoughness));
}

MATERIAL_FUNCTION_HEADER(ShaderEvaluate_Dielectric)
{
    DielectricBxDF *bxdf = PushStruct(arena, DielectricBxDF);
    ShaderEvaluate_DielectricHelper(callbacks, arena, itr, si, lambda, bxdf);
    result = bxdf;
}

MATERIAL_FUNCTION_HEADER(ShaderEvaluate_CoatedDiffuse)
{
    Vec4f filterWidths(.75f);
    DielectricBxDF diBxDF;
    ShaderEvaluate_DielectricHelper(callbacks, arena, itr, si, lambda, &diBxDF);

    SampledSpectrumN reflectance =
        ProcessAlbedoTexture(callbacks[itr->callbackCount++], itr, si, lambda, filterWidths);
    SampledSpectrumN albedo =
        ProcessAlbedoTexture(callbacks[itr->callbackCount++], itr, si, lambda, filterWidths);

    f32 g;
    callbacks[itr->callbackCount++](itr, si, lambda, filterWidths, &g);

    i32 maxDepth  = itr->ReadInt(10);
    i32 nSamples  = itr->ReadInt(1);
    f32 thickness = itr->ReadFloat(.01);

    CoatedDiffuseBxDF *bxdf = PushStruct(arena, CoatedDiffuseBxDF);
    new (bxdf) CoatedDiffuseBxDF(diBxDF, DiffuseBxDF(reflectance), albedo, g, thickness,
                                 maxDepth, nSamples);
    result = bxdf;
}

struct MaterialNode
{
    string str;
    u32 index;

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

MaterialHashMap *CreateMaterials(Arena *arena, Arena *tempArena, Tokenizer *tokenizer)
{
    TempArena temp = ScratchStart(&tempArena, 1);
    Scene *scene   = GetScene();

    ChunkedLinkedList<Material, 1024> materialsList(temp.arena);
    MaterialHashMap *table = PushStructConstruct(tempArena, MaterialHashMap)(tempArena, 8192);

    StringBuilder builders[(u32)(MaterialTypes::Max)];
    for (u32 i = 0; i < (u32)MaterialTypes::Max; i++)
    {
        builders[i].arena = temp.arena;
    }

    while (!Advance(tokenizer, "MATERIALS_END "))
    {
        bool advanceResult = Advance(tokenizer, "m ");
        Assert(advanceResult);
        string materialName = ReadWord(tokenizer);

        // Add to hash table
        table->Add(tempArena, MaterialNode{materialName, materialsList.totalCount});

        SkipToNextChar(tokenizer);

        // Get the type of material
        i32 materialTypeIndex = -1;
        for (u32 m = 0; m < (u32)MaterialTypes::Max; m++)
        {
            if (Advance(tokenizer, materialTypeNames[m]))
            {
                materialTypeIndex = m;
                break;
            }
        }
        SkipToNextChar(tokenizer);
        Assert(materialTypeIndex != -1);

        Material *material     = &materialsList.AddBack();
        StringBuilder *builder = &builders[materialTypeIndex];

        const string *materialParamNames = materialParameterNames[materialTypeIndex];
        u32 parameterCount               = materialParameterCounts[materialTypeIndex];

        material->shade      = materialFuncs[materialTypeIndex];
        material->key.offset = SafeTruncateU64ToU32(builder->totalSize);
        material->funcs      = PushArray(arena, TextureCallback, parameterCount);
        // TODO: it's not this exactly
        material->count = parameterCount;

        // NOTE: 0 means float, 1 means spectrum
        auto ParseDataType = [&](StringBuilder *builder, DataType type, u32 p) {
            switch (type)
            {
                case DataType::Float:
                {
                    u8 *start = tokenizer->cursor;
                    while (!CharIsBlank(*tokenizer->cursor))
                    {
                        tokenizer->cursor += sizeof(f32);
                    }
                    Put(builder, start, (u64)(tokenizer->cursor - start));
                }
                break;
                case DataType::String:
                {
                    u32 strSize = *(u32 *)tokenizer->cursor;
                    tokenizer->cursor += sizeof(u32);
                    Put(builder, tokenizer->cursor, strSize);
                    tokenizer->cursor += strSize;
                }
                break;
                default: Error(0, "forgot");
            }
        };

        auto ParseTexture = [&](StringBuilder *builder, u32 p, u32 textureTypeIndex) {
            const DataType *types = textureDataTypes[textureTypeIndex];
            const string *params  = textureParameterArrays[textureTypeIndex];
            u32 count             = textureParameterCounts[textureTypeIndex];
            for (u32 i = 0; i < count; i++)
            {
                if (Advance(tokenizer, params[i]))
                {
                    SkipToNextChar(tokenizer);
                    ParseDataType(builder, types[i], p);
                    SkipToNextChar(tokenizer);
                }
            }
        };

        // convert the file description into the appropriate byte format, join into the
        // attribute table, write material callbacks that properly process the attribute table.
        // problems:
        // - need to actually create the material tables
        // - fix string builder to be more efficient (less allocations)

        for (u32 i = 0; i < parameterCount; i++)
        {
            if (!Advance(tokenizer, materialParamNames[i])) continue;

            if (Advance(tokenizer, "t "))
            {
                // Parse a texture
                for (u32 textureTypeIndex = 0; textureTypeIndex < (u32)TextureType::Max;
                     textureTypeIndex++)
                {
                    if (Advance(tokenizer, textureTypeNames[textureTypeIndex]))
                    {
                        material->funcs[i] = textureFuncs[textureTypeIndex];
                        ParseTexture(builder, i, textureTypeIndex);
                        break;
                    }
                }
            }
            else if (Advance(tokenizer, "s "))
            {
                u8 *start = tokenizer->cursor;
                u32 count = 0;
                while (!CharIsBlank(*tokenizer->cursor)) tokenizer->cursor += sizeof(f32);
                Assert((count & 1) == 0);

                PiecewiseLinearSpectrum *spec = PiecewiseLinearSpectrum::FromInterleaved(
                    arena, (f32 *)start, count, false);
                Put(builder, &spec, sizeof(spec));
                if (materialParamNames[i] == "eta")
                {
                    material->funcs[i] = ProcessIOR;
                }
                else
                {
                    Error(0, "%S \n", materialParamNames[i]);
                }
            }
            SkipToNextChar(tokenizer);
        }
        material->key.SetIndexAndSize(
            materialTypeIndex,
            SafeTruncateU64ToU32(builder->totalSize - material->key.offset));
    }

    // Join
    scene->materialTables = StaticArray<AttributeTable>(arena, (u32)MaterialTypes::Max);
    scene->materials      = StaticArray<Material>(arena, materialsList.totalCount);

    materialsList.Flatten(scene->materials);

    for (u32 index = 0; index < (u32)MaterialTypes::Max; index++)
    {
        scene->materialTables[index].buffer = CombineBuilderNodes(arena, &builders[index]).str;
    }

    ScratchEnd(temp);
    return table;
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

                u32 id = files.Length();
                BeginTicketMutex(&table->mutexes[index]);
                auto *node = &table->nodes[index];
                while (node->next)
                {
                    if (node->filename.size && node->filename == includeFile) break;
                    node = node->next;
                }
                if (!node->next)
                {
                    Assert(node->filename.size == 0);
                    node->filename                = PushStr8Copy(arena, includeFile);
                    ScenePrimitives *includeScene = PushStruct(arena, ScenePrimitives);
                    node->counter.count           = 1;
                    node->scene                   = includeScene;
                    node->next                    = PushStruct(arena, SceneLoadTable::Node);
                    scheduler.Schedule(&counter, [=](u32 jobID) {
                        LoadRTScene(arenas, state, includeScene, directory, node->filename,
                                    &node->counter);
                    });
                    EndTicketMutex(&table->mutexes[index]);
                    files.AddBack() = includeScene;
                }
                else
                {
                    EndTicketMutex(&table->mutexes[index]);
                    files.AddBack() = node->scene;
                    scheduler.Wait(&node->counter);
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
            while (!Advance(&tokenizer, "SHAPE_END "))
            {
                if (Advance(&tokenizer, "m "))
                {
                    Assert(materialHashMap);
                    string materialName      = ReadWord(&tokenizer);
                    const MaterialNode *node = materialHashMap->Get(materialName);
                    PrimitiveIndices &ids    = indices.AddBack();
                    ids                      = PrimitiveIndices(LightHandle(), node->index);
                }
                if (Advance(&tokenizer, "Quad "))
                {
                    type       = GeometryType::QuadMesh;
                    Mesh &mesh = shapes.AddBack();
                    for (;;)
                    {
                        if (Advance(&tokenizer, "p "))
                        {
                            u32 pOffset = ReadInt(&tokenizer);
                            mesh.p      = dataTokenizer.input.str + pOffset;
                        }
                        else if (Advance(&tokenizer, "n "))
                        {
                            u32 nOffset = ReadInt(&tokenizer);
                            mesh.n      = dataTokenizer.input.str + nOffset;
                        }
                        else if (Advance(&tokenizer, "v "))
                        {
                            u32 num          = ReadInt(&tokenizer);
                            mesh.numVertices = num;
                            mesh.numFaces    = num / 4;
                        }
                        else
                        {
                            break;
                        }
                        SkipToNextChar(&tokenizer);
                    }
                }
                else if (Advance(&tokenizer, "Tri "))
                {
                    type       = GeometryType::TriangleMesh;
                    Mesh &mesh = shapes.AddBack();
                    for (;;)
                    {
                        if (Advance(&tokenizer, "p "))
                        {
                            u32 pOffset = ReadInt(&tokenizer);
                            mesh.p      = dataTokenizer.input.str + pOffset;
                        }
                        else if (Advance(&tokenizer, "n "))
                        {
                            u32 nOffset = ReadInt(&tokenizer);
                            mesh.n      = dataTokenizer.input.str + nOffset;
                        }
                        else if (Advance(&tokenizer, "uv "))
                        {
                            u32 uvOffset = ReadInt(&tokenizer);
                            mesh.uv      = dataTokenizer.input.str + uvOffset;
                        }
                        else if (Advance(&tokenizer, "v "))
                        {
                            u32 num          = ReadInt(&tokenizer);
                            mesh.numVertices = num;
                        }
                        else if (Advance(&tokenizer, "i "))
                        {
                            u32 num         = ReadInt(&tokenizer);
                            mesh.numIndices = num;
                            mesh.numFaces   = num / 3;
                        }
                        else if (Advance(&tokenizer, "indices "))
                        {
                            u32 indOffset = ReadInt(&tokenizer);
                            mesh.indices  = dataTokenizer.input.str + indOffset;
                        }
                        else
                        {
                            break;
                        }
                        SkipToNextChar(&tokenizer);
                    }
                }
                else
                {
                    Assert(0);
                }
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
            materialHashMap = CreateMaterials(arena, temp.arena, &tokenizer);
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
    state.table.count   = 1024;
    state.table.nodes   = PushArray(temp.arena, SceneLoadTable::Node, state.table.count);
    state.table.mutexes = PushArray(temp.arena, TicketMutex, state.table.count);

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
