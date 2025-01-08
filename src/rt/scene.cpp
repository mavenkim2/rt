#include "scene.h"
#include "bvh/bvh_types.h"
#include "macros.h"
#include "integrate.h"
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

enum class TextureType
{
    None,
    ConstantFloat,
    ConstantSpectrum,
    Ptex,
};

TextureType GetTextureType(Tokenizer *tokenizer)
{
    TextureType type;
    if (Advance(tokenizer, "cf "))
    {
        type = TextureType::ConstantFloat;
    }
    else if (Advance(tokenizer, "ptex "))
    {
        type = TextureType::Ptex;
    }
    else if (Advance(tokenizer, "cs "))
    {
        type = TextureType::ConstantSpectrum;
    }
    return type;
}

template <TextureType type>
auto ProcessTexture(AttributeIterator *iterator)
{
    if constexpr (type == TextureType::None)
    {
        return 0.f;
    }
    else if constexpr (type == TextureType::ConstantFloat)
    {
        return iterator->ReadFloat();
    }
    else if constexpr (type == TextureType::ConstantSpectrum)
    {
    }
    else if constexpr (type == TextureType::Ptex)
    {
        string filename = iterator->ReadString();
    }
}

template <TextureType u, TextureType r>
f32 ProcessRoughness(AttributeIterator *iterator)
{
    if constexpr (u == TextureType::None)
    {
        return ProcessTexture<r>(iterator);
    }
    return ProcessTexture<u>(iterator);
}

template <typename Spectrum>
f32 ProcessIOR(AttributeIterator *iterator)
{
    return 1.5f;
}

// TODO: instead of this junk I should just create a preprocessor (i.e. write a text file
// that is code)
//////////////////////////////////////////////////////////////////////////////////////////////

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

DEFINE_MATERIAL_SHADER(Dielectric, DielectricBxDF, DISPLACEMENT, IOR, ROUGHNESS)

enum class MaterialTypes
{
    Interface,
    Diffuse,
    DiffuseTransmission,
    CoatedDiffuse,
    Dielectric,
    Max,
};

template <>
MaterialCallback GenerateMaterialCallback()
{
    // i feel like literally the logical conclusion of what I'm doing is dr. jit, no? like,
    // i want to construct megakernel callbacks at runtime for each material type.
    // why not just go all the way and do it for literally every computation? for simd
    // queues, why have per type queues? it really should just be one massive kernel for
    // each type no?
}

void CreateMaterials(Arena *arena, Tokenizer *tokenizer)
{
    TempArena temp = ScratchStart(0, 0);
    Scene *scene   = GetScene();
    // ChunkedLinkedList<Material, 1024> materialsList(temp.arena);
    ChunkedLinkedList<ScenePacket, 1024> materialPackets(temp.arena);
    while (!Advance(tokenizer, "MATERIALS_END "))
    {
        string materialName = ReadWord(tokenizer);
        SkipToNextChar(tokenizer);

        // Get the type of material
        i32 index = -1;
        for (u32 materialTypeIndex = 0; materialTypeIndex < (u32)MaterialTypes::Max;
             materialTypeIndex++)
        {
            if (Advance(tokenizer, materialTypeNames[materialTypeIndex]))
            {
                index = materialTypeIndex;
                break;
            }
        }
        SkipToNextChar(tokenizer);

        const string *materialParamNames = materialParameterNames[index];
        u32 parameterCount               = materialParameterCounts[index];
        Assert(index != -1);

        TextureType *types      = PushArray(temp.arena, TextureType, parameterCount);
        TextureEvalFunc **funcs = PushArray(arena, TextureEvalFunc *, parameterCount);

        for (u32 i = 0; i < parameterCount; i++)
        {
            bool result = Advance(tokenizer, materialParamNames[i]);
            Assert(result);
            SkipToNextChar(tokenizer);

            i32 textureIndex = -1;

            // Parse a texture
            for (u32 textureTypeIndex = 0; textureTypeIndex < (u32)TextureType::Max;
                 textureTypeIndex++)
            {
                if (Advance(tokenizer, textureTypeNames[textureTypeIndex]))
                {
                    textureIndex        = textureTypeNames[textureTypeIndex];
                    funcs[textureIndex] = ;
                    break;
                }
            }

            // get the function pointer
        }
        for (;;)
        {
            if (Advance(tokenizer, "ur "))
            {
                Error(isAnisotropic != 1, "Cannot specify both roughness and "
                                          "anisotropic roughness\n");
                isAnisotropic = 2;
                ur            = GetTextureType(tokenizer);
            }
            else if (Advance(tokenizer, "vr "))
            {
                Error(isAnisotropic != 1, "Cannot specify both roughness and "
                                          "anisotropic roughness\n");
                isAnisotropic = 2;
                vr            = GetTextureType(tokenizer);
            }
            else if (Advance(tokenizer, "r "))
            {
                Error(isAnisotropic != 2, "Cannot specify both roughness and "
                                          "anisotropic roughness\n");
                isAnisotropic = 1;
                r             = GetTextureType(tokenizer);
            }
            else if (Advance(tokenizer, "ior "))
            {
                eta = GetTextureType(tokenizer);
            }
            else
            {
                break;
            }
        }
        // TODO: surely there's a better way of doing this than just if elsing
        // every single combination
        if (ur == TextureType::None && vr == TextureType::None &&
            r == TextureType::ConstantFloat && eta == TextureType::ConstantFloat &&
            disp == TextureType::None)
        {
            Material &mat = materialsList.AddBack();
            mat.eval      = ShaderEvaluate_Dielectric<TextureType::None, ConstantSpectrum,
                                                      TextureType::None, TextureType::None,
                                                      TextureType::ConstantFloat>;
        }
        else
        {
            Error(0, "Dielectric version not supported.\n");
        }
        // Create appropriate texture
        // Material *material = CreateDielectricMaterial(arena, Tokenizer *
        // tokenizer);
    }
    scene->materials = StaticArray<Material>(arena, materialsList.totalCount);
    materialsList.Flatten(scene->materials);

    ScratchEnd(temp);
}

void LoadRTScene(Arena **arenas, SceneLoadTable *table, ScenePrimitives *scene,
                 string directory, string filename, Scheduler::Counter *c = 0,
                 AffineSpace *renderFromWorld = 0, bool baseFile = false)

{
    scene->filename = filename;
    Assert(GetFileExtension(filename) == "rtscene");
    TempArena temp = ScratchStart(0, 0);
    Arena *arena   = arenas[GetThreadIndex()];

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
                        LoadRTScene(arenas, table, includeScene, directory, node->filename,
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
            // Error(instanceOffset == instanceCount, "inst offset %u\n",
            // instanceOffset);
            scene->numPrimitives = instanceOffset;
            scene->childScenes   = PushArrayNoZero(arena, ScenePrimitives *, files.totalCount);
            scene->numChildScenes = files.totalCount;
        }
        else if (Advance(&tokenizer, "SHAPE_START "))
        {
            Assert(isLeaf);
            ChunkedLinkedList<Mesh, 1024, MemoryType_Shape> shapes(temp.arena);
            while (!Advance(&tokenizer, "SHAPE_END "))
            {
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
            shapes.Flatten((Mesh *)scene->primitives);
        }
        else if (Advance(&tokenizer, "TEXTURES_START"))
        {
        }
        else if (Advance(&tokenizer, "MATERIALS_START"))
        {
            // CreateMaterials(arena);
        }
        else
        {
            Error(0, "Invalid section header.\n");
        }
    }
    files.Flatten(scene->childScenes);

    scheduler.Wait(&counter);

    OS_UnmapFile(tokenizer.input.str);
    PrimitiveIndices *ids = PushStructConstruct(arena, PrimitiveIndices)(
        LightHandle(), MaterialHandle(MaterialType::MSDielectricMaterial1, 0));
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
        scene->primIndices = ids;
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
    SceneLoadTable table;
    table.count   = 1024;
    table.nodes   = PushArray(temp.arena, SceneLoadTable::Node, table.count);
    table.mutexes = PushArray(temp.arena, TicketMutex, table.count);

    Scene *scene = GetScene();
    LoadRTScene(arenas, &table, &scene->scene, directory, filename, 0, t, true);
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
