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

PrimData *GeneratePrimData(Arena *arena, TriangleMesh *mesh, u32 count, u32 numFaces, Bounds &bounds, bool grow = false)
{
    PrimData *data;
    if (grow)
    {
        data = PushArray(arena, PrimData, u32(numFaces * GROW_AMOUNT));
    }
    else
    {
        data = PushArray(arena, PrimData, numFaces);
    }
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
    arena->align       = 64;
    PrimDataSOA soa;
    soa.minX    = PushArray(arena, f32, u32(numFaces * GROW_AMOUNT));
    soa.minY    = PushArray(arena, f32, u32(numFaces * GROW_AMOUNT));
    soa.minZ    = PushArray(arena, f32, u32(numFaces * GROW_AMOUNT));
    soa.geomIDs = PushArray(arena, u32, u32(numFaces * GROW_AMOUNT));
    soa.maxX    = PushArray(arena, f32, u32(numFaces * GROW_AMOUNT));
    soa.maxY    = PushArray(arena, f32, u32(numFaces * GROW_AMOUNT));
    soa.maxZ    = PushArray(arena, f32, u32(numFaces * GROW_AMOUNT));
    soa.primIDs = PushArray(arena, u32, u32(numFaces * GROW_AMOUNT));

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

        soa.minX[i]    = -min.x;
        soa.minY[i]    = -min.y;
        soa.minZ[i]    = -min.z;
        soa.geomIDs[i] = 0;
        soa.maxX[i]    = max.x;
        soa.maxY[i]    = max.y;
        soa.maxZ[i]    = max.z;
        soa.primIDs[i] = i;
    }

    HeuristicSOASplitBinning heuristic(geomBounds);
    PerformanceCounter start = OS_StartCounter();
    heuristic.BinDiffTest(mesh, &soa, 0, numFaces);
    f32 time = OS_GetMilliseconds(start);
    printf("Time elapsed binning: %fms\n", time);

#if 1
    Split split = SpatialSplitBest(heuristic.finalBounds, heuristic.entryCounts, heuristic.exitCounts);
    printf("Num faces: %u\n", numFaces);
    printf("Split pos: %u\n", split.bestPos);
    printf("Split dim: %u\n", split.bestDim);
    printf("Split SAH: %f\n", split.bestSAH);
    // printf("Misc: %llu\n\n", threadLocalStatistics->misc);

    f32 invScale    = (Lane8F32::LoadU((f32 *)(&heuristic.invScale[split.bestDim])))[0];
    f32 base        = (Lane8F32::LoadU((f32 *)(&heuristic.base[split.bestDim])))[0];
    split.bestValue = ((split.bestPos + 1) * invScale) + base;
    printf("Split value: %f\n", split.bestValue);

    ExtRangeSOA range(&soa, 0, numFaces, u32(numFaces * GROW_AMOUNT));
    start = OS_StartCounter();
    Bounds left;
    Bounds right;
    u32 mid = heuristic.SplitSOA(arena, mesh, range, split, left, right);
    time    = OS_GetMilliseconds(start);
    printf("Mid: %u\n", mid);
#endif

    printf("Split time: %fms\n", time);
#if 1
    printf("Left bounds min: %f %f %f\n", left.minP[0], left.minP[1], left.minP[2]);
    printf("Left bounds max: %f %f %f\n", left.maxP[0], left.maxP[1], left.maxP[2]);
    printf("Right bounds min: %f %f %f\n", right.minP[0], right.minP[1], right.minP[2]);
    printf("Right bounds max: %f %f %f\n", right.maxP[0], right.maxP[1], right.maxP[2]);
    // Tests to ensure that the partitioning is valid
    u32 errors        = 0;
    f32 *minStream    = ((f32 **)(&soa.minX))[split.bestDim];
    f32 *maxStream    = ((f32 **)(&soa.minX))[split.bestDim + 4];
    u32 firstBadIndex = 0;
    u32 lastBadIndex  = 0;
    for (u32 i = 0; i < numFaces; i++)
    {
        f32 min      = minStream[i];
        f32 max      = maxStream[i];
        f32 centroid = (max - min) * 0.5f;
        if (i < mid)
        {
            u32 value = (u32)Floor((centroid - heuristic.base[split.bestDim][0]) * heuristic.scale[split.bestDim][0]);
            if (centroid >= split.bestValue || value > split.bestPos)
            {
                if (firstBadIndex == 0)
                {
                    firstBadIndex = i;
                }
                // Assert(false);
                errors += 1;
            }
        }
        else
        {
            u32 value = (u32)Floor((centroid - heuristic.base[split.bestDim][0]) * heuristic.scale[split.bestDim][0]);
            if (centroid < split.bestValue || value <= split.bestPos)
            {
                if (firstBadIndex == 0)
                {
                    firstBadIndex = i;
                }
                lastBadIndex = i;
                // Assert(false);
                errors += 1;
            }
        }
    }
    printf("first bad index: %u\n", firstBadIndex);
    printf("last bad index: %u\n", lastBadIndex);
    printf("Num errors: %u\n", errors);
#endif
}

void TriangleClipBinTestDefault(TriangleMesh *mesh, u32 count = 0)
{
    Arena *arena = ArenaAlloc();
    arena->align = 64;

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
    PrimData *data = GeneratePrimData(arena, mesh, count, numFaces, geomBounds, true);

    TestSplitBinningBase heuristic(geomBounds);

    PerformanceCounter start = OS_StartCounter();
    heuristic.Bin(mesh, data, 0, numFaces);
    f32 time = OS_GetMilliseconds(start);

    Split split = SpatialSplitBest(heuristic.bins, heuristic.numBegin, heuristic.numEnd);

    printf("Time elapsed binning: %fms\n", time);
    printf("Split pos: %u\n", split.bestPos);
    printf("Split dim: %u\n", split.bestDim);
    printf("Split SAH: %f\n", split.bestSAH);

    split.bestValue = ((split.bestPos + 1) * heuristic.invScale[split.bestDim]) + heuristic.base[split.bestDim];
    printf("Split value: %f\n", split.bestValue);
    ExtRange range(data, 0, numFaces, u32(numFaces * GROW_AMOUNT));
    start = OS_StartCounter();

    Bounds left;
    Bounds right;
    u32 mid = heuristic.Split(mesh, data, range, split, left, right);
    time    = OS_GetMilliseconds(start);
    printf("Mid: %u\n", mid);

    printf("Time elapsed splitting: %fms\n", time);
#if 0
    printf("Left bounds min: %f %f %f\n", left.minP[0], left.minP[1], left.minP[2]);
    printf("Left bounds max: %f %f %f\n", left.maxP[0], left.maxP[1], left.maxP[2]);
    printf("Right bounds min: %f %f %f\n", right.minP[0], right.minP[1], right.minP[2]);
    printf("Right bounds max: %f %f %f\n", right.maxP[0], right.maxP[1], right.maxP[2]);

    // Correctness test
    u32 errors = 0;
    for (u32 i = 0; i < numFaces; i++)
    {
        PrimData *prim = &data[i];
        f32 centroid   = (prim->maxP + prim->minP)[split.bestDim] * 0.5f;
        if (i < mid)
        {
            u32 value = (u32)Floor((centroid - heuristic.base[split.bestDim]) * heuristic.scale[split.bestDim]);
            if (centroid >= split.bestValue || value > split.bestPos)
            {
                // Assert(false);
                errors += 1;
            }
        }
        else
        {
            u32 value = (u32)Floor((centroid - heuristic.base[split.bestDim]) * heuristic.scale[split.bestDim]);
            if (centroid < split.bestValue || value <= split.bestPos)
            {
                // Assert(false);
                errors += 1;
            }
        }
    }
    printf("Num errors: %u\n", errors);
#endif
}

} // namespace rt
