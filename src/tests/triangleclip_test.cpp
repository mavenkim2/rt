namespace rt
{
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

#if 1
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

void TriangleClipTestSOA(const u32 count)
{

    Arena *arena = ArenaAlloc();
    TriangleMesh mesh;
    mesh.p           = PushArray(arena, Vec3f, count);
    mesh.numVertices = count;
    mesh.indices     = PushArray(arena, u32, count);
    mesh.numIndices  = count;

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
        mesh.p[i * 3]     = RandomVec3(-100.f, 100.f);
        mesh.p[i * 3 + 1] = RandomVec3(-100.f, 100.f);
        mesh.p[i * 3 + 2] = RandomVec3(-100.f, 100.f);
        geomBounds.Extend(Lane4F32(mesh.p[i * 3]));
        geomBounds.Extend(Lane4F32(mesh.p[i * 3 + 1]));
        geomBounds.Extend(Lane4F32(mesh.p[i * 3 + 2]));
        mesh.indices[i * 3]     = i * 3;
        mesh.indices[i * 3 + 1] = i * 3 + 1;
        mesh.indices[i * 3 + 2] = i * 3 + 2;
    }

    TestSOASplitBinning heuristic(geomBounds);

    for (u32 i = 0; i < numFaces; i++)
    {
        u32 primID     = RandomInt(0, numFaces);
        Vec3f min      = Min(Min(mesh.p[primID * 3], mesh.p[primID * 3 + 1]), mesh.p[primID * 3 + 2]);
        Vec3f max      = Max(Max(mesh.p[primID * 3], mesh.p[primID * 3 + 1]), mesh.p[primID * 3 + 2]);
        soa.minX[i]    = min.x;
        soa.minY[i]    = min.y;
        soa.minZ[i]    = min.z;
        soa.geomIDs[i] = 0;
        soa.maxX[i]    = max.x;
        soa.maxY[i]    = max.y;
        soa.maxZ[i]    = max.z;
        soa.primIDs[i] = primID;
    }

    PerformanceCounter start = OS_StartCounter();
    heuristic.Bin(&mesh, &soa, 0, numFaces);
    f32 time = OS_GetMilliseconds(start);

    printf("Time elapsed: %fms\n", time);
}

void TriangleClipBinTestDefault(const u32 count)
{
    Arena *arena = ArenaAlloc();
    TriangleMesh mesh;
    mesh.p           = PushArray(arena, Vec3f, count);
    mesh.numVertices = count;
    mesh.indices     = PushArray(arena, u32, count);
    mesh.numIndices  = count;

    const u32 numFaces = count / 3;
    PrimData *data     = PushArray(arena, PrimData, numFaces);

    Bounds geomBounds;
    for (u32 i = 0; i < numFaces; i++)
    {
        mesh.p[i * 3]     = RandomVec3(-100.f, 100.f);
        mesh.p[i * 3 + 1] = RandomVec3(-100.f, 100.f);
        mesh.p[i * 3 + 2] = RandomVec3(-100.f, 100.f);

        geomBounds.Extend(Lane4F32(mesh.p[i * 3]));
        geomBounds.Extend(Lane4F32(mesh.p[i * 3 + 1]));
        geomBounds.Extend(Lane4F32(mesh.p[i * 3 + 2]));

        mesh.indices[i * 3]     = i * 3;
        mesh.indices[i * 3 + 1] = i * 3 + 1;
        mesh.indices[i * 3 + 2] = i * 3 + 2;
    }

    TestSplitBinningBase heuristic(geomBounds);

    for (u32 i = 0; i < numFaces; i++)
    {
        PrimData *prim = &data[i];
        u32 primID     = RandomInt(0, numFaces);
        Vec3f min      = Min(Min(mesh.p[primID * 3], mesh.p[primID * 3 + 1]), mesh.p[primID * 3 + 2]);
        Vec3f max      = Max(Max(mesh.p[primID * 3], mesh.p[primID * 3 + 1]), mesh.p[primID * 3 + 2]);
        prim->minP     = Min(prim->minP, Lane4F32(min));
        prim->maxP     = Max(prim->maxP, Lane4F32(max));
        prim->SetPrimID(primID);
        prim->SetGeomID(0);
    }

    PerformanceCounter start = OS_StartCounter();
    heuristic.Bin(&mesh, data, 0, numFaces);
    f32 time = OS_GetMilliseconds(start);

    printf("Time elapsed: %fms\n", time);
}

} // namespace rt
