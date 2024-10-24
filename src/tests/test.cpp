namespace rt
{
TriangleMesh *GenerateMesh(Arena *arena, u32 count, f32 min = -100.f, f32 max = 100.f)
{
    arena->align       = 64;
    TriangleMesh *mesh = PushStruct(arena, TriangleMesh);
    mesh->p            = PushArray(arena, Vec3f, count);
    mesh->numVertices  = count;
    mesh->indices      = PushArray(arena, u32, count);
    mesh->numIndices   = count;

    for (u32 i = 0; i < count; i++)
    {
        mesh->indices[i] = RandomInt(0, count);
    }

    for (u32 i = 0; i < count / 3; i++)
    {
        mesh->p[i * 3]     = RandomVec3(min, max);
        mesh->p[i * 3 + 1] = RandomVec3(min, max);
        mesh->p[i * 3 + 2] = RandomVec3(min, max);
    }
    return mesh;
}

PrimRefCompressed *GenerateAOSData(Arena *arena, TriangleMesh *mesh, u32 numFaces, Bounds &geomBounds, Bounds &centBounds)
{
    arena->align            = 64;
    PrimRefCompressed *refs = PushArray(arena, PrimRefCompressed, u32(numFaces * GROW_AMOUNT) + 1);
    for (u32 i = 0; i < numFaces; i++)
    {
        u32 i0 = mesh->indices[i * 3];
        u32 i1 = mesh->indices[i * 3 + 1];
        u32 i2 = mesh->indices[i * 3 + 2];

        PrimRefCompressed *prim = &refs[i];
        Vec3f v0                = mesh->p[i0];
        Vec3f v1                = mesh->p[i1];
        Vec3f v2                = mesh->p[i2];

        Vec3f min = Min(Min(v0, v1), v2);
        Vec3f max = Max(Max(v0, v1), v2);

        Lane4F32 mins = Lane4F32(min.x, min.y, min.z, 0);
        Lane4F32 maxs = Lane4F32(max.x, max.y, max.z, 0);
        Lane4F32::StoreU(prim->min, -mins);
        prim->maxX = max.x;
        prim->maxY = max.y;
        prim->maxZ = max.z;
        // prim->m256 = Lane8F32(-mins, maxs);

        prim->primID = i;

        geomBounds.Extend(mins, maxs);
        centBounds.Extend((maxs + mins)); //* 0.5f);
    }
    return refs;
}

PrimRefCompressed *GenerateQuadData(Arena *arena, QuadMesh *mesh, u32 numFaces, Bounds &geomBounds, Bounds &centBounds)
{
    arena->align            = 64;
    PrimRefCompressed *refs = PushArray(arena, PrimRefCompressed, u32(numFaces * GROW_AMOUNT) + 1);
    for (u32 i = 0; i < numFaces; i++)
    {
        PrimRefCompressed *prim = &refs[i];
        Vec3f v0                = mesh->p[4 * i + 0];
        Vec3f v1                = mesh->p[4 * i + 1];
        Vec3f v2                = mesh->p[4 * i + 2];
        Vec3f v3                = mesh->p[4 * i + 3];

        Vec3f min = Min(Min(v0, v1), Min(v2, v3));
        Vec3f max = Max(Max(v0, v1), Min(v2, v3));

        Lane4F32 mins = Lane4F32(min.x, min.y, min.z, 0);
        Lane4F32 maxs = Lane4F32(max.x, max.y, max.z, 0);
        Lane4F32::StoreU(prim->min, -mins);
        prim->maxX = max.x;
        prim->maxY = max.y;
        prim->maxZ = max.z;

        prim->primID = i;

        geomBounds.Extend(mins, maxs);
        centBounds.Extend((maxs + mins)); //* 0.5f);
    }
    return refs;
}

void AOSSBVHBuilderTest(Arena *arena, TriangleMesh *mesh)
{
    arena->align = 64;

    const u32 numFaces = mesh->numIndices / 3;
    Bounds geomBounds;
    Bounds centBounds;
    PrimRefCompressed *refs = GenerateAOSData(arena, mesh, numFaces, geomBounds, centBounds);

    BuildSettings settings;
    settings.intCost = 0.3f;

    RecordAOSSplits record;
    record.geomBounds = Lane8F32(-geomBounds.minP, geomBounds.maxP);
    record.centBounds = Lane8F32(-centBounds.minP, centBounds.maxP);
    record.SetRange(0, numFaces, u32(numFaces * GROW_AMOUNT));
    u32 numProcessors = OS_NumProcessors();
    Arena **arenas    = PushArray(arena, Arena *, numProcessors);
    for (u32 i = 0; i < numProcessors; i++)
    {
        arenas[i] = ArenaAlloc(16); // ArenaAlloc(ARENA_RESERVE_SIZE, LANE_WIDTH * 4);
    }

    PerformanceCounter counter = OS_StartCounter();
    BVHNodeType bvh            = BuildQuantizedTriSBVH(settings, arenas, mesh, refs, record);
    f32 time                   = OS_GetMilliseconds(counter);
    printf("num faces: %u\n", numFaces);
    printf("Build time: %fms\n", time);

    f64 totalMiscTime     = 0;
    u64 numNodes          = 0;
    u64 totalNodeMemory   = 0;
    u64 totalRecordMemory = 0;
    for (u32 i = 0; i < numProcessors; i++)
    {
        totalMiscTime += threadLocalStatistics[i].miscF;
        totalNodeMemory += threadMemoryStatistics[i].totalBVHMemory;
        totalRecordMemory += threadMemoryStatistics[i].totalRecordMemory;
        printf("thread time %u: %fms\n", i, threadLocalStatistics[i].miscF);
        numNodes += threadLocalStatistics[i].misc;
    }
    printf("total time: %fms \n", totalMiscTime);
    printf("num nodes: %llu\n", numNodes);               // num nodes: %llu\n", numNodes);
    printf("node kb: %llu\n", totalNodeMemory / 1024);   // num nodes: %llu\n", numNodes);
    printf("record kb: %llu", totalRecordMemory / 1024); // num nodes: %llu\n", numNodes);
}

void QuadSBVHBuilderTest(Arena *arena, QuadMesh *mesh)
{
    arena->align = 64;

    const u32 numFaces = mesh->numVertices / 4;
    Bounds geomBounds;
    Bounds centBounds;
    PrimRefCompressed *refs = GenerateQuadData(arena, mesh, numFaces, geomBounds, centBounds);

    BuildSettings settings;
    settings.intCost = 0.3f;

    RecordAOSSplits record;
    record.geomBounds = Lane8F32(-geomBounds.minP, geomBounds.maxP);
    record.centBounds = Lane8F32(-centBounds.minP, centBounds.maxP);
    record.SetRange(0, numFaces, u32(numFaces * GROW_AMOUNT));
    u32 numProcessors = OS_NumProcessors();
    Arena **arenas    = PushArray(arena, Arena *, numProcessors);
    for (u32 i = 0; i < numProcessors; i++)
    {
        arenas[i] = ArenaAlloc(16); // ArenaAlloc(ARENA_RESERVE_SIZE, LANE_WIDTH * 4);
    }

    PerformanceCounter counter = OS_StartCounter();
    BVHNodeType bvh            = BuildQuantizedQuadSBVH(settings, arenas, mesh, refs, record);
    f32 time                   = OS_GetMilliseconds(counter);
    printf("num faces: %u\n", numFaces);
    printf("Build time: %fms\n", time);

    f64 totalMiscTime     = 0;
    u64 numNodes          = 0;
    u64 totalNodeMemory   = 0;
    u64 totalRecordMemory = 0;
    for (u32 i = 0; i < numProcessors; i++)
    {
        totalMiscTime += threadLocalStatistics[i].miscF;
        totalNodeMemory += threadMemoryStatistics[i].totalBVHMemory;
        totalRecordMemory += threadMemoryStatistics[i].totalRecordMemory;
        printf("thread time %u: %fms\n", i, threadLocalStatistics[i].miscF);
        numNodes += threadLocalStatistics[i].misc;
    }
    printf("total time: %fms \n", totalMiscTime);
    printf("num nodes: %llu\n", numNodes);               // num nodes: %llu\n", numNodes);
    printf("node kb: %llu\n", totalNodeMemory / 1024);   // num nodes: %llu\n", numNodes);
    printf("record kb: %llu", totalRecordMemory / 1024); // num nodes: %llu\n", numNodes);
}

void PartialRebraidBuilderTest(Arena *arena)
{
    u32 numProcessors = OS_NumProcessors();
    Arena **arenas    = PushArray(arena, Arena *, numProcessors);
    for (u32 i = 0; i < numProcessors; i++)
    {
        arenas[i] = ArenaAlloc(16); // ArenaAlloc(ARENA_RESERVE_SIZE, LANE_WIDTH * 4);
    }

    PerformanceCounter counter = OS_StartCounter();
    Scene2 *scenes             = InitializeScene(arenas, "data/island/pbrt-v4/meshes/", "data/island/pbrt-v4/instances.inst");
    printf("scene initialization + blas build time: %fms\n", OS_GetMilliseconds(counter));

#if 0
    RecordAOSSplits record;
    counter         = OS_StartCounter();
    BRef *buildRefs = GenerateBuildRefs(scenes, 0, arena, record);
    printf("time to generate build refs: %fms\n", OS_GetMilliseconds(counter));

    BuildSettings settings;

    counter          = OS_StartCounter();
    BVHNodeType node = BuildTLAS(settings, arenas, buildRefs, record);
    printf("time to generate tlas: %fms\n", OS_GetMilliseconds(counter));
#endif

    f64 totalMiscTime     = 0;
    u64 numNodes          = 0;
    u64 totalNodeMemory   = 0;
    u64 totalRecordMemory = 0;
    for (u32 i = 0; i < numProcessors; i++)
    {
        totalMiscTime += threadLocalStatistics[i].miscF;
        totalNodeMemory += threadMemoryStatistics[i].totalBVHMemory;
        totalRecordMemory += threadMemoryStatistics[i].totalRecordMemory;
        // printf("thread time %u: %fms\n", i, threadLocalStatistics[i].miscF);
        numNodes += threadLocalStatistics[i].misc;
    }
    printf("total misc time: %fms \n", totalMiscTime);
    printf("num nodes: %llu\n", numNodes);
    printf("node kb: %llu\n", totalNodeMemory / 1024);
    printf("record kb: %llu", totalRecordMemory / 1024);
}

void PartitionFix()
{
    Arena *arena = ArenaAlloc();
    string data  = OS_ReadFile(arena, "build/parallel_mid.txt");
    Tokenizer tokenizer;
    tokenizer.input  = data;
    tokenizer.cursor = tokenizer.input.str;
    bool *testArray  = PushArray(arena, bool, 291916);
    u32 numLines     = 0;
    while (!EndOfBuffer(&tokenizer))
    {
        // string line = ReadLine(&tokenizer);
        // u32 number  = 0;
        // u32 i       = 0;
        // while (line.str[i] != ' ')
        // {
        //     number *= 10;
        //     number += (line.str[i++] - 48);
        // }
        // if (testArray[number])
        // {
        //     // Assert(!"don't think this should happen\n");
        //     string word      = SkipToNextWord(SkipToNextWord(line));
        //     Split::Type type = (Split::Type)(word.str[0] - 48);
        //     Assert(type == Split::Type::Object);
        // }
        // testArray[number] = true;
        numLines++;
        SkipToNextLine(&tokenizer);
    }
    printf("num lines %u\n", numLines);
}

// void SceneLoadTest()
// {
//     Arena *arena      = ArenaAlloc();
//     u32 numProcessors = OS_NumProcessors();
//     Arena **arenas    = PushArray(arena, Arena *, numProcessors);
//     for (u32 i = 0; i < numProcessors; i++)
//     {
//         arenas[i] = ArenaAlloc(16); // ArenaAlloc(ARENA_RESERVE_SIZE, LANE_WIDTH * 4);
//     }
//     Scene2 scene;
//
//     PerformanceCounter counter = OS_StartCounter();
//     ReadSerializedData(arenas, &scene, "data/island/pbrt-v4/meshes/", "data/island/pbrt-v4/instances.inst");
//     printf("time: %fms\n", OS_GetMilliseconds(counter));
// }

} // namespace rt
