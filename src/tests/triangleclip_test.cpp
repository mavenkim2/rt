namespace rt
{
TriangleMesh *GenerateMesh(Arena *arena, u32 count, f32 min = -100.f, f32 max = 100.f)
{
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

PrimData *GeneratePrimData(Arena *arena, TriangleMesh *mesh, u32 count, u32 numFaces, Bounds &bounds)
{
    PrimData *data = PushArray(arena, PrimData, numFaces);
    for (u32 i = 0; i < numFaces; i++)
    {
        u32 i0 = mesh->indices[i * 3];
        u32 i1 = mesh->indices[i * 3 + 1];
        u32 i2 = mesh->indices[i * 3 + 2];

        PrimData *prim = &data[i];
        Vec3f v0       = mesh->p[i0];
        Vec3f v1       = mesh->p[i1];
        Vec3f v2       = mesh->p[i2];

        Vec3f min = Min(Min(v0, v1), v2);
        Vec3f max = Max(Max(v0, v1), v2);

        Lane4F32 lMin(min);
        Lane4F32 lMax(max);
        prim->minP = lMin;
        prim->maxP = lMax;
        prim->SetPrimID(i);
        prim->SetGeomID(0);

        bounds.Extend(lMin, lMax);
    }
    return data;
}

void TriangleClipTest()
{
    Arena *arena = ArenaAlloc();
    TriangleMesh mesh;
    const u32 count  = 30000000;
    mesh.p           = PushArray(arena, Vec3f, count);
    mesh.numVertices = count;
    mesh.indices     = PushArray(arena, u32, count);
    mesh.numIndices  = count;

    PrimRef *refs = PushArray(arena, PrimRef, count / 3);

    for (u32 i = 0; i < count / 3; i++)
    {
        mesh.p[i * 3]           = RandomVec3(-100.f, 100.f);
        mesh.p[i * 3 + 1]       = RandomVec3(-100.f, 100.f);
        mesh.p[i * 3 + 2]       = RandomVec3(-100.f, 100.f);
        mesh.indices[i * 3]     = i * 3;
        mesh.indices[i * 3 + 1] = i * 3 + 1;
        mesh.indices[i * 3 + 2] = i * 3 + 2;

        Vec3f min      = Min(Min(mesh.p[i * 3], mesh.p[i * 3 + 1]), mesh.p[i * 3 + 2]);
        Vec3f max      = Max(Max(mesh.p[i * 3], mesh.p[i * 3 + 1]), mesh.p[i * 3 + 2]);
        refs[i].minX   = min.x;
        refs[i].minY   = min.y;
        refs[i].minZ   = min.z;
        refs[i].geomID = 0;
        refs[i].maxX   = max.x;
        refs[i].maxY   = max.y;
        refs[i].maxZ   = max.z;
        refs[i].primID = i;
    }

    u32 *faceIndices = PushArray(arena, u32, count / 3);
    for (u32 i = 0; i < count / 3; i++)
    {
        faceIndices[i] = RandomInt(0, count / 3);
    }
#if 1
    {
        PerformanceCounter perf = OS_StartCounter();
        Bounds l;
        Bounds r;
        for (u32 i = 0; i < count / 3; i += 8)
        {
            Bounds left;
            Bounds right;
            ClipTriangle(&mesh, faceIndices + i, 1, 1.f, left, right);
            l.Extend(left);
            r.Extend(right);
        }
        f32 time = OS_GetMilliseconds(perf);
        printf("Time elapsed AVX: %fms\n", time);
        printf("L Bounds: %f %f %f - %f %f %f\n", l.minP[0], l.minP[1], l.minP[2], l.maxP[0], l.maxP[1], l.maxP[2]);
        printf("R Bounds: %f %f %f - %f %f %f\n", r.minP[0], r.minP[1], r.minP[2], r.maxP[0], r.maxP[1], r.maxP[2]);
        printf("Gather Time: %f\n", OS_GetMilliseconds(threadLocalStatistics->misc));
    }
#endif

#if 0
    {
        PerformanceCounter perf = OS_StartCounter();
        Bounds l                = Bounds();
        Bounds r                = Bounds();
        for (u32 i = 0; i < count / 3; i++)
        {
            Bounds left;
            Bounds right;
            ClipTriangleSimple(&mesh, faceIndices[i], 1, 1.f, left, right);
            l.Extend(left);
            r.Extend(right);
        }
        f32 time = OS_GetMilliseconds(perf);

        printf("Time elapsed simple: %fms\n", time);
        printf("L Bounds: %f %f %f - %f %f %f\n", l.minP[0], l.minP[1], l.minP[2], l.maxP[0], l.maxP[1], l.maxP[2]);
        printf("R Bounds: %f %f %f - %f %f %f\n", r.minP[0], r.minP[1], r.minP[2], r.maxP[0], r.maxP[1], r.maxP[2]);
    }
#endif
}

void TriangleClipTestSOA(TriangleMesh *mesh, u32 count = 0)
{
    Arena *arena = ArenaAlloc();

    if (!mesh)
    {
        Assert(count != 0);
        mesh = GenerateMesh(arena, count);
    }
    else
    {
        count = mesh->numIndices;
    }

    const u32 numFaces = count / 3;
    PrimDataSOA soa;
    soa.minX    = PushArray(arena, f32, numFaces);
    soa.minY    = PushArray(arena, f32, numFaces);
    soa.minZ    = PushArray(arena, f32, numFaces);
    soa.geomIDs = PushArray(arena, u32, numFaces);
    soa.maxX    = PushArray(arena, f32, numFaces);
    soa.maxY    = PushArray(arena, f32, numFaces);
    soa.maxZ    = PushArray(arena, f32, numFaces);
    soa.primIDs = PushArray(arena, u32, numFaces);

    Bounds geomBounds;

    for (u32 i = 0; i < numFaces; i++)
    {
        u32 i0 = mesh->indices[i * 3 + 0];
        u32 i1 = mesh->indices[i * 3 + 1];
        u32 i2 = mesh->indices[i * 3 + 2];

        Vec3f v0 = mesh->p[i0];
        Vec3f v1 = mesh->p[i1];
        Vec3f v2 = mesh->p[i2];

        Vec3f min = Min(Min(v0, v1), v2);
        Vec3f max = Max(Max(v0, v1), v2);

        Lane4F32 lMin(min);
        Lane4F32 lMax(max);

        geomBounds.Extend(lMin, lMax);

        soa.minX[i]    = min.x;
        soa.minY[i]    = min.y;
        soa.minZ[i]    = min.z;
        soa.geomIDs[i] = 0;
        soa.maxX[i]    = max.x;
        soa.maxY[i]    = max.y;
        soa.maxZ[i]    = max.z;
        soa.primIDs[i] = i;
    }

    HeuristicSOASplitBinning<16> heuristic(geomBounds);
    PerformanceCounter start = OS_StartCounter();
    // heuristic.Bin(mesh, &soa, 0, numFaces);
    heuristic.BinDiffTest(mesh, &soa, 0, numFaces);
    f32 time = OS_GetMilliseconds(start);

    Split split = SpatialSplitBest(heuristic.finalBounds, heuristic.entryCounts, heuristic.exitCounts);

    printf("Time elapsed: %fms\n", time);
    printf("Num faces: %u\n", numFaces);
    printf("Split value: %u\n", split.bestPos);
    printf("Split SAH: %f\n", split.bestSAH);
    printf("Split dim: %u\n", split.bestDim);

    printf("Load time: %llums\n", threadLocalStatistics->misc);
    // printf("Clip # calls: %llu\n", threadLocalStatistics->misc);
}

void TriangleClipTestAOS(TriangleMesh *mesh, u32 count = 0)
{
    Arena *arena = ArenaAlloc();

    if (!mesh)
    {
        Assert(count != 0);
        mesh = GenerateMesh(arena, count);
    }
    else
    {
        count = mesh->numIndices;
    }
    const u32 numFaces = count / 3;

    PrimRef *refs = PushArray(arena, PrimRef, numFaces);

    Bounds geomBounds;
    for (u32 i = 0; i < numFaces; i++)
    {
        u32 i0 = mesh->indices[i * 3 + 0];
        u32 i1 = mesh->indices[i * 3 + 1];
        u32 i2 = mesh->indices[i * 3 + 2];

        Vec3f v0 = mesh->p[i0];
        Vec3f v1 = mesh->p[i1];
        Vec3f v2 = mesh->p[i2];

        Vec3f min = Min(Min(v0, v1), v2);
        Vec3f max = Max(Max(v0, v1), v2);

        Lane4F32 lMin(min);
        Lane4F32 lMax(max);

        geomBounds.Extend(lMin, lMax);

        PrimRef *ref = &refs[i];
        ref->minX    = min.x;
        ref->minY    = min.y;
        ref->minZ    = min.z;
        ref->geomID  = 0;
        ref->maxX    = max.x;
        ref->maxY    = max.y;
        ref->maxZ    = max.z;
        ref->primID  = i;
    }

    HeuristicSOASplitBinning heuristic(geomBounds);

    PerformanceCounter start = OS_StartCounter();
    heuristic.BinTest(mesh, refs, 0, numFaces);
    f32 time = OS_GetMilliseconds(start);

    Split split = SpatialSplitBest(heuristic.finalBounds, heuristic.entryCounts, heuristic.exitCounts);

    printf("Time elapsed: %fms\n", time);
    printf("Split value: %u\n", split.bestPos);
    printf("Split SAH: %f\n", split.bestSAH);
    printf("Split dim: %u\n", split.bestDim);
}

void TriangleClipBinTestDefault(TriangleMesh *mesh, u32 count = 0)
{
    Arena *arena = ArenaAlloc();

    if (!mesh)
    {
        Assert(count != 0);
        mesh = GenerateMesh(arena, count);
    }
    else
    {
        count = mesh->numIndices;
    }
    const u32 numFaces = count / 3;

    Bounds geomBounds;
    PrimData *data = GeneratePrimData(arena, mesh, count, numFaces, geomBounds);

    TestSplitBinningBase<16> heuristic(geomBounds);

    PerformanceCounter start = OS_StartCounter();
    heuristic.Bin(mesh, data, 0, numFaces);
    f32 time = OS_GetMilliseconds(start);

    Split split = SpatialSplitBest(heuristic.bins, heuristic.numBegin, heuristic.numEnd);

    printf("Time elapsed: %fms\n", time);
    printf("Split value: %u\n", split.bestPos);
    printf("Split SAH: %f\n", split.bestSAH);
    printf("Split dim: %u\n", split.bestDim);
}

} // namespace rt
