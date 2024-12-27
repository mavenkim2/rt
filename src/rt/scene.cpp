#include "scene.h"
#include "bvh/bvh_types.h"
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

void LoadRTScene(Arena **arenas, SceneLoadTable *table, ScenePrimitives *scene,
                 string directory, string filename, Scheduler::Counter *c = 0,
                 AffineSpace *renderFromWorld = 0, bool baseFile = false)

{
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
        u32 count = ReadInt(&dataTokenizer);
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
    bool isLeaf = true;
    for (;;)
    {
        if (Advance(&tokenizer, "RTSCENE_END")) break;
        if (Advance(&tokenizer, "INCLUDE_START "))
        {
            Advance(&tokenizer, "Count: ");
            u32 instanceCount = ReadInt(&tokenizer);
            SkipToNextChar(&tokenizer);

            ChunkedLinkedList<ScenePrimitives *, 32, MemoryType_Instance> files(temp.arena);
            scene->numPrimitives = instanceCount;
            scene->primitives    = PushArrayNoZero(arena, Instance, instanceCount);
            Instance *instances  = (Instance *)scene->primitives;
            u32 instanceOffset   = 0;

            isLeaf = false;
            while (!Advance(&tokenizer, "INCLUDE_END "))
            {
                Advance(&tokenizer, "File: ");
                string includeFile = ReadWord(&tokenizer);
                // TODO: this is a band aid until I get curves working
#if 1
                if (!OS_FileExists(StrConcat(temp.arena, directory, includeFile)))
                {
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
            scene->childScenes = PushArrayNoZero(arena, ScenePrimitives *, files.totalCount);

            files.Flatten(scene->childScenes);
        }
        else if (Advance(&tokenizer, "SHAPE_START "))
        {
            Assert(isLeaf);
            ChunkedLinkedList<QuadMesh, 1024, MemoryType_Shape> shapes(temp.arena);
            while (!Advance(&tokenizer, "SHAPE_END "))
            {
                if (Advance(&tokenizer, "Quad "))
                {
                    QuadMesh &mesh = shapes.AddBack();
                    for (;;)
                    {
                        if (Advance(&tokenizer, "p "))
                        {
                            u32 pOffset = ReadInt(&tokenizer);
                            mesh.p      = (Vec3f *)(dataTokenizer.input.str + pOffset);
                        }
                        else if (Advance(&tokenizer, "n "))
                        {
                            u32 nOffset = ReadInt(&tokenizer);
                            mesh.n      = (Vec3f *)(dataTokenizer.input.str + nOffset);
                        }
                        else if (Advance(&tokenizer, "c "))
                        {
                            u32 num          = ReadInt(&tokenizer);
                            mesh.numVertices = num;
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
            scene->primitives    = PushArrayNoZero(arena, QuadMesh, shapes.totalCount);
            shapes.Flatten((QuadMesh *)scene->primitives);
        }
        else
        {
            Error(0, "Invalid section header.\n");
        }
    }

    scheduler.Wait(&counter);

    OS_UnmapFile(tokenizer.input.str);
    PrimitiveIndices *ids = PushStructConstruct(arena, PrimitiveIndices)(
        LightHandle(), MaterialHandle(MaterialType::CoatedDiffuseMaterial2, 0));
    BuildSettings settings;
    if (!isLeaf)
    {
        BuildTLASBVH(arenas, settings, scene);
    }
    else
    {
        Assert(!hasTransforms);
        scene->primIndices = ids;
        // TODO: hardcoded
        if (scene->numPrimitives)
        {
            BuildQuadBVH(arenas, settings, scene);
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
    BRef *refs            = GenerateBuildRefs(scene, temp.arena, record);
    scene->nodePtr        = BuildTLASQuantized(settings, arenas, scene, refs, record);
    using IntersectorType = typename IntersectorHelper<Instance, BRef>::IntersectorType;
    scene->intersectFunc  = &IntersectorType::Intersect;
    scene->occludedFunc   = &IntersectorType::Occluded;

    scene->SetBounds(Bounds(record.geomBounds));
    ScratchEnd(temp);
}

template <typename Mesh>
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
                    GenerateMeshRefs(meshes, refs, offsets[jobID],
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
            GenerateMeshRefs(meshes, refs, 0, totalNumFaces, 0, scene->numPrimitives, record);
        }
        record.SetRange(0, totalNumFaces, extEnd);
        scene->nodePtr = BuildQuantizedSBVH<Mesh>(settings, arenas, scene, refs, record);
        using IntersectorType = typename IntersectorHelper<Mesh, PrimRef>::IntersectorType;
        scene->intersectFunc  = &IntersectorType::Intersect;
        scene->occludedFunc   = &IntersectorType::Occluded;
    }
    else
    {
        totalNumFaces           = meshes->GetNumFaces();
        u32 extEnd              = u32(totalNumFaces * GROW_AMOUNT);
        PrimRefCompressed *refs = PushArrayNoZero(temp.arena, PrimRefCompressed, extEnd);
        GenerateMeshRefs<PrimRefCompressed>(meshes, refs, 0, totalNumFaces, 0, 1, record);
        record.SetRange(0, totalNumFaces, extEnd);
        scene->nodePtr = BuildQuantizedSBVH<Mesh>(settings, arenas, scene, refs, record);
        using IntersectorType =
            typename IntersectorHelper<Mesh, PrimRefCompressed>::IntersectorType;
        scene->intersectFunc = &IntersectorType::Intersect;
        scene->occludedFunc  = &IntersectorType::Occluded;
    }
    scene->SetBounds(Bounds(record.geomBounds));
    scene->numFaces = totalNumFaces;
    ScratchEnd(temp);
}

void BuildTriangleBVH(Arena **arenas, BuildSettings &settings, ScenePrimitives *scene)
{
    BuildBVH<TriangleMesh>(arenas, settings, scene);
}

void BuildQuadBVH(Arena **arenas, BuildSettings &settings, ScenePrimitives *scene)
{
    BuildBVH<QuadMesh>(arenas, settings, scene);
}

template <typename PrimRef, typename Mesh>
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
                    mesh->GenerateMeshRefs(refs, offset + start, i, start, count, record);
                },
                [&](RecordAOSSplits &l, const RecordAOSSplits &r) { l.Merge(r); });
        }
        else
        {
            Assert(offset < offsetMax);
            mesh->GenerateMeshRefs(refs, offset, i, 0, numFaces, tempRecord);
        }
        r.Merge(tempRecord);
        offset += numFaces;
    }
    Assert(offsetMax == offset);
    record = r;
}

} // namespace rt
