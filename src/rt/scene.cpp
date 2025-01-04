#include "scene.h"
#include "bvh/bvh_types.h"
#include "integrate.h"
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

enum class AttributeType
{
    Float,
    // Spectrum,
    RGB,
    String,
};

struct AttributeTable
{
    u8 *buffer;

#ifdef DEBUG
    AttributeType *types;
    u32 attributeCount;
#endif
};

struct AttributeIterator
{
    AttributeTable *table;
    u64 offset;
#ifdef DEBUG
    u32 countOffset = 0;
#endif

    AttributeIterator(AttributeTable *table) : table(table) {}
    AttributeIterator(AttributeTable *table, u64 offset) : table(table), offset(offset) {}
    void DebugCheck(AttributeType type)
    {
#ifdef DEBUG
        Assert(table->types);
        Assert(countOffset < table->attributeCount);
        Assert(table->types[countOffset++] == type);
#endif
    }
    f32 ReadFloat()
    {
        u64 o = offset;
        offset += sizeof(f32);
        DebugCheck(AttributeType::Float);
        return *(f32 *)(table->buffer + o);
    }
    string ReadString()
    {
        u32 size = *(u32 *)(table->buffer + offset);
        offset += sizeof(size);
        string result = Str8(table->buffer + offset, size);
        offset += size;
        DebugCheck(AttributeType::String);
        return result;
    }
};

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
    static_assert(u == TextureType::None || r == TextureType::None,
                  "Cannot specify both anisotropic and isotropic roughness\n");
    if constexpr (u == TextureType::None)
    {
        return ProcessTexture<r>(iterator);
    }
    return ProcessTexture<u>(iterator);
}

//////////////////////////////////////////////////////////////////////////////////////////////

#define MATERIAL_TEMPLATE_DECL(...)                                                           \
    MATERIAL_TEMPLATE_DECL_HELPER(COUNT_ARGS(__VA_ARGS__), __VA_ARGS__)

#define USE_ROUGHNESS_TMPL(...)    TextureType ur, TextureType vr, TextureType r
#define USE_IOR_TMPL(...)          TextureType ior __VA_ARGS__
#define USE_DISPLACEMENT_TMPL(...) TextureType disp __VA_ARGS__

#define IIF(c, falseVal, ...) CONCAT(IF_, c)(falseVal, __VA_ARGS__)
#define IF_0(falseVal, ...)   EXPAND(falseVal)
#define IF_1(falseVal, ...)   __VA_ARGS__

#define CHECK_HAS_COMMA(...)                            CHECK_HAS_COMMA_HELPER(__VA_ARGS__, 1, 1, 0)
#define CHECK_HAS_COMMA_HELPER(_1, _2, _3, result, ...) result
#define IS_EMPTY(...)                                   CHECK_HAS_COMMA(__VA_ARGS__)
#define CHECK(x)                                        IS_EMPTY(CONCAT(x, _TMPL)(, ))

#define TMPL_DECL(x) IIF(CHECK(x), TextureType x, CONCAT(x, _TMPL)())

#define MATERIAL_TEMPLATE_DECL_HELPER(x, ...)                                                 \
    EXPAND(CONCAT(RECURSE__, x)(TMPL_DECL, __VA_ARGS__))

//////////////////////////////////////////////////////////////////////////////////////////////

#define MATERIAL_FUNC_DECL(...) MATERIAL_FUNC_DECL_HELPER(COUNT_ARGS(__VA_ARGS__), __VA_ARGS__)

#define START_MATERIAL_SHADER(name, BxDFType, ...)                                            \
    template <MATERIAL_TEMPLATE_DECL(__VA_ARGS__)>                                            \
    void ShaderEvaluate_##name(AttributeTable *table)

#define DEFINE_MATERIAL_SHADER(name, BxDFType, ...)                                           \
    START_MATERIAL_SHADER(name, BxDFType, __VA_ARGS__)                                        \
    {                                                                                         \
        typedef BxDFType BxDF;                                                                \
        AttributeIterator itr(table);                                                         \
    }

// template <TextureType ur, TextureType vr, TextureType r, TextureType eta, TextureType disp>
// DielectricBxDF EvaluateDielectric(AttributeTable *mat)

DEFINE_MATERIAL_SHADER(Dielectric, DielectricBxDF, //
                       USE_ROUGHNESS,              //
                       USE_IOR,                    //
                       USE_DISPLACEMENT, foo)
// {
//     AttributeIterator itr(mat);
//     f32 urResult = ProcessRoughness<ur, r>(&itr);
//     f32 vrResult = ProcessRoughness<vr, r>(&itr);
//
//     auto etaResult  = ProcessTexture<eta>(&itr);
//     auto dispResult = ProcessDisplacement<disp>(&itr);
// }

// Material CreateDielectricMaterial(TextureType ur, TextureType vr, TextureType r,
//                                   TextureType eta, TextureType displacement)
// {
//
//     u32 foo = CHECK(foo);
//     if (ur == TextureType::None && vr == TextureType::None && r == TextureType::None &&
//         eta == TextureType::ConstantFloat && displacement == TextureType::None)
//     {
//         Dielectric DielectricConstant
//     }
// }

void CreateMaterials(Arena *arena, Tokenizer *tokenizer)
{
    u32 foo      = CHECK(USE_IOR);
    Scene *scene = GetScene();
    while (!Advance(tokenizer, "MATERIALS_END "))
    {
        // the fun begins :)
        if (Advance(tokenizer, "dielectric "))
        {
            u32 isAnisotropic = 0;
            TextureType ur    = TextureType::None;
            TextureType vr    = TextureType::None;
            TextureType r     = TextureType::None;
            TextureType eta   = TextureType::None;
            for (;;)
            {
                if (Advance(tokenizer, "ur "))
                {
                    Error(isAnisotropic != 1,
                          "Cannot specify both roughness and anisotropic roughness\n");
                    isAnisotropic = 2;
                    ur            = GetTextureType(tokenizer);
                }
                else if (Advance(tokenizer, "vr "))
                {
                    Error(isAnisotropic != 1,
                          "Cannot specify both roughness and anisotropic roughness\n");
                    isAnisotropic = 2;
                    vr            = GetTextureType(tokenizer);
                }
                else if (Advance(tokenizer, "r "))
                {
                    Error(isAnisotropic != 2,
                          "Cannot specify both roughness and anisotropic roughness\n");
                    isAnisotropic = 1;
                    r             = GetTextureType(tokenizer);
                }
                else if (Advance(tokenizer, "eta "))
                {
                    eta = GetTextureType(tokenizer);
                }
                else
                {
                    break;
                }
            }
            // Create appropriate texture
            // Material *material = CreateDielectricMaterial(arena, Tokenizer * tokenizer);
        }
    }
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
            // Error(instanceOffset == instanceCount, "inst offset %u\n", instanceOffset);
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
