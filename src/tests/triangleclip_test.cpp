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

PrimData *GeneratePrimData(Arena *arena, TriangleMesh *mesh, u32 count, u32 numFaces, Bounds &bounds, Bounds &centBounds, bool grow = false)
{
    arena->align = 32;
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
        centBounds.Extend((lMin + lMax) * 0.5f);
    }
    return data;
}

PrimDataSOA GenerateSOAData(Arena *arena, TriangleMesh *mesh, u32 numFaces, Bounds &geomBounds, Bounds &centBounds)
{
    arena->align = 64;
    PrimDataSOA soa;
    soa.minX    = PushArray(arena, f32, u32(numFaces * GROW_AMOUNT));
    soa.minY    = PushArray(arena, f32, u32(numFaces * GROW_AMOUNT));
    soa.minZ    = PushArray(arena, f32, u32(numFaces * GROW_AMOUNT));
    soa.geomIDs = PushArray(arena, u32, u32(numFaces * GROW_AMOUNT));
    soa.maxX    = PushArray(arena, f32, u32(numFaces * GROW_AMOUNT));
    soa.maxY    = PushArray(arena, f32, u32(numFaces * GROW_AMOUNT));
    soa.maxZ    = PushArray(arena, f32, u32(numFaces * GROW_AMOUNT));
    soa.primIDs = PushArray(arena, u32, u32(numFaces * GROW_AMOUNT));

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
        centBounds.Extend((lMin + lMax) * 0.5f);

        soa.minX[i]    = -min.x;
        soa.minY[i]    = -min.y;
        soa.minZ[i]    = -min.z;
        soa.geomIDs[i] = 0;
        soa.maxX[i]    = max.x;
        soa.maxY[i]    = max.y;
        soa.maxZ[i]    = max.z;
        soa.primIDs[i] = i;
    }
    return soa;
}

PrimRef *GenerateAOSData(Arena *arena, TriangleMesh *mesh, u32 numFaces, Bounds &geomBounds, Bounds &centBounds)
{
    arena->align  = 64;
    PrimRef *refs = PushArray(arena, PrimRef, u32(numFaces * GROW_AMOUNT));
    for (u32 i = 0; i < numFaces; i++)
    {
        u32 i0 = mesh->indices[i * 3];
        u32 i1 = mesh->indices[i * 3 + 1];
        u32 i2 = mesh->indices[i * 3 + 2];

        PrimRef *prim = &refs[i];
        Vec3f v0      = mesh->p[i0];
        Vec3f v1      = mesh->p[i1];
        Vec3f v2      = mesh->p[i2];

        Vec3f min = Min(Min(v0, v1), v2);
        Vec3f max = Max(Max(v0, v1), v2);

        Lane4F32 mins = Lane4F32(min.x, min.y, min.z, 0);
        Lane4F32 maxs = Lane4F32(max.x, max.y, max.z, 0);
        prim->m256    = Lane8F32(-mins, maxs);

        prim->primID = i;

        geomBounds.Extend(mins, maxs);
        centBounds.Extend((maxs + mins) * 0.5f);
    }
    return refs;
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
    Bounds geomBounds;
    Bounds centBounds;
    PrimDataSOA soa = GenerateSOAData(arena, mesh, numFaces, geomBounds, centBounds);
    PrimDataSOA out;
    out.minX    = PushArrayNoZero(arena, f32, u32(numFaces * GROW_AMOUNT));
    out.minY    = PushArrayNoZero(arena, f32, u32(numFaces * GROW_AMOUNT));
    out.minZ    = PushArrayNoZero(arena, f32, u32(numFaces * GROW_AMOUNT));
    out.geomIDs = PushArrayNoZero(arena, u32, u32(numFaces * GROW_AMOUNT));
    out.maxX    = PushArrayNoZero(arena, f32, u32(numFaces * GROW_AMOUNT));
    out.maxY    = PushArrayNoZero(arena, f32, u32(numFaces * GROW_AMOUNT));
    out.maxZ    = PushArrayNoZero(arena, f32, u32(numFaces * GROW_AMOUNT));
    out.primIDs = PushArrayNoZero(arena, u32, u32(numFaces * GROW_AMOUNT));

    // SplitBinner binner(geomBounds);
    // HeuristicSOASplitBinning heuristic(&binner);
    // PerformanceCounter start = OS_StartCounter();
    // heuristic.Bin(mesh, &soa, 0, numFaces);
    //
    // f32 time = OS_GetMilliseconds(start);
    // printf("Time elapsed binning: %fms\n", time);

#if 1
#if 0
    ObjectBinner<32> binner(centBounds);
    HeuristicSOAObjectBinning<32> heuristic(&binner);
    PerformanceCounter start = OS_StartCounter();
    heuristic.Bin(&soa, 0, numFaces);
    f32 time = OS_GetMilliseconds(start);
    printf("Time elapsed binning: %fms\n", time);

    Split split = BinBest(heuristic.finalBounds, heuristic.counts, &binner);
    printf("Split pos: %u\n", split.bestPos);
    printf("Split dim: %u\n", split.bestDim);
    printf("Split SAH: %f\n", split.bestSAH);
    start   = OS_StartCounter();
    u32 mid = PartitionSerial<true>(&soa, split.bestDim, split.bestValue, 0, numFaces, [&](u32 index) { return index; });
    time    = OS_GetMilliseconds(start);
    printf("Time elapsed partition: %fms\n", time);

#else
    SplitBinner binner(geomBounds);
    // HeuristicSOASplitBinning heuristic(&binner);
    PerformanceCounter start = OS_StartCounter();
    // heuristic.Bin(mesh, &soa, 0, numFaces);
    using HSplit = HeuristicSOASplitBinning<16>;
    SplitBinner splitBinner(geomBounds);

    HSplit heuristic(&splitBinner);
    const u32 groupSize = 32 * 1024;

    ParallelForOutput output = ParallelFor<HSplit>(
        0, numFaces, groupSize,
        [&](HSplit &binner, u32 start, u32 count) { binner.Bin(mesh, &soa, start, count); },
        &splitBinner);

    Reduce<HSplit>(
        heuristic, output,
        [&](HSplit &l, const HSplit &r) { l.Merge(r); },
        &splitBinner);

    f32 time = OS_GetMilliseconds(start);
    printf("Time elapsed binning: %fms\n", time);
    Split split = BinBest(heuristic.bins8, heuristic.entryCounts, heuristic.exitCounts, &binner);
    printf("Num faces: %u\n", numFaces);
    printf("Split pos: %u\n", split.bestPos);
    printf("Split dim: %u\n", split.bestDim);
    printf("Split SAH: %f\n", split.bestSAH);
    // printf("Misc: %llu\n\n", threadLocalStatistics->misc);

    printf("Split value: %f\n", split.bestValue);

    u32 *lOffsets = PushArrayNoZero(arena, u32, output.num);
    u32 *rOffsets = PushArrayNoZero(arena, u32, output.num);
    u32 *lCounts  = PushArrayNoZero(arena, u32, output.num);
    u32 *rCounts  = PushArrayNoZero(arena, u32, output.num);

    u32 lOffset = 0;
    u32 rOffset = u32(numFaces * GROW_AMOUNT);
    for (u32 i = 0; i < output.num; i++)
    {
        HSplit *h       = &((HSplit *)output.out)[i];
        lOffsets[i]     = lOffset;
        u32 entryCounts = 0;
        u32 exitCounts  = 0;
        for (u32 bin = 0; bin < split.bestPos; bin++)
        {
            entryCounts += h->entryCounts[bin][split.bestDim];
        }
        for (u32 bin = split.bestDim; bin < 16; bin++)
        {
            exitCounts += h->exitCounts[bin][split.bestDim];
        }
        lOffset += entryCounts;
        rOffset -= exitCounts;
        rOffsets[i] = rOffset;
        lCounts[i]  = entryCounts;
        rCounts[i]  = exitCounts;
    }

    ExtRange range(0, numFaces, u32(numFaces * GROW_AMOUNT));

    start = OS_StartCounter();
    scheduler.ScheduleAndWait(output.num, 1, [&](u32 jobID) {
        RecordSOASplits left;
        RecordSOASplits right;
        u32 threadStart = range.start + output.groupSize * jobID;
        u32 groupSize   = jobID == output.num - 1 ? range.End() - threadStart : output.groupSize;
        heuristic.Split(mesh, &soa, &out,
                        lOffsets[jobID], lCounts[jobID],
                        rOffsets[jobID], rCounts[jobID],
                        threadStart, groupSize,
                        split, left, right);
    });
    time = OS_GetMilliseconds(start);
    printf("Split time: %fms\n", time);
    // for (u32 i = 0; i < OS_NumProcessors(); i++)
    // {
    //     printf("Thread %u time: %llu\n", i, threadLocalStatistics[i].misc);
    // }

    // unless I'm misunderstanding something, none of the primitives' bboxes should cross the split plane
    // f32 *minStream = ((f32 **)(&soa.minX))[split.bestDim];
    // f32 *maxStream = ((f32 **)(&soa.minX))[split.bestDim + 4];
    // u32 errors     = 0;
    // for (u32 i = 0; i < numFaces; i++)
    // {
    //     f32 min = minStream[i];
    //     f32 max = maxStream[i];
    //
    //     if (!((min > -split.bestValue && max <= split.bestValue) || (min <= -split.bestValue && max >= split.bestValue)))
    //     {
    //         printf("prim ID: %u\n", i);
    //         printf("min: %f, max: %f\n", min, max);
    //         u32 binMin = (u32)Floor((binner.base[split.bestDim][0] + min) * binner.scaleNegArr[split.bestDim][0]);
    //         u32 binMax = (u32)Floor((max - binner.base[split.bestDim][0]) * binner.scale[split.bestDim][0]);
    //         printf("min bin: %u, max bin: %u\n", binMin, binMax);
    //         errors += 1;
    //     }
    // }
    // printf("mismatches: %u\n", errors);
#endif

#if 0
    // printf("\n");
    // printf("Left bounds min: %f %f %f\n", left.geomBounds.minP[0], left.geomBounds.minP[1], left.geomBounds.minP[2]);
    // printf("Left bounds max: %f %f %f\n", left.geomBounds.maxP[0], left.geomBounds.maxP[1], left.geomBounds.maxP[2]);
    // printf("Right bounds min: %f %f %f\n", right.geomBounds.minP[0], right.geomBounds.minP[1], right.geomBounds.minP[2]);
    // printf("Right bounds max: %f %f %f\n", right.geomBounds.maxP[0], right.geomBounds.maxP[1], right.geomBounds.maxP[2]);
    //
    // printf("\n");
    // printf("Left cent bounds min: %f %f %f\n", left.centBounds.minP[0], left.centBounds.minP[1], left.centBounds.minP[2]);
    // printf("Left cent bounds max: %f %f %f\n", left.centBounds.maxP[0], left.centBounds.maxP[1], left.centBounds.maxP[2]);
    // printf("Right cent bounds min: %f %f %f\n", right.centBounds.minP[0], right.centBounds.minP[1], right.centBounds.minP[2]);
    // printf("Right cent bounds max: %f %f %f\n", right.centBounds.maxP[0], right.centBounds.maxP[1], right.centBounds.maxP[2]);
    // Tests to ensure that the partitioning is valid
    errors            = 0;
    u32 firstBadIndex = 0;
    u32 lastBadIndex  = 0;
    for (u32 i = 0; i < numFaces; i++)
    {
        f32 min      = minStream[i];
        f32 max      = maxStream[i];
        f32 centroid = (max - min) * 0.5f;
        if (i < mid)
        {
            u32 value = (u32)Floor((centroid - binner.base[split.bestDim][0]) * binner.scale[split.bestDim][0]);
            if (centroid >= split.bestValue || value >= split.bestPos)
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
            u32 value = (u32)Floor((centroid - binner.base[split.bestDim][0]) * binner.scale[split.bestDim][0]);
            if (centroid < split.bestValue || value < split.bestPos)
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
#endif
}

void TriangleClipTestAOS(TriangleMesh *mesh)
{
    Arena *arena = ArenaAlloc();
    arena->align = 64;

    Bounds centBounds;
    Bounds geomBounds;
    u32 numFaces     = mesh->numIndices / 3;
    u32 extEnd       = u32(numFaces * GROW_AMOUNT);
    PrimRef *refs    = GenerateAOSData(arena, mesh, numFaces, geomBounds, centBounds);
    PrimRef *outRefs = PushArrayNoZero(arena, PrimRef, extEnd);

    SplitBinner<16> binner(geomBounds);
    using Heuristic            = HeuristicAOSSplitBinning<16>;
    PerformanceCounter counter = OS_StartCounter();

    Heuristic heuristic; //(&binner);
    // heuristic.Bin(mesh, refs, 0, numFaces);
    ParallelForOutput output = ParallelFor<Heuristic>(
        0, numFaces, PARALLEL_THRESHOLD,
        [&](Heuristic &heuristic, u32 start, u32 count) { heuristic.Bin(mesh, refs, start, count); },
        &binner);

    Reduce(
        heuristic, output,
        [&](Heuristic &l, const Heuristic &r) { l.Merge(r); },
        &binner);

    f32 time = OS_GetMilliseconds(counter);
    printf("bin time: %fms\n", time);
    Split split = BinBest(heuristic.bins, heuristic.entryCounts, heuristic.exitCounts, &binner);
    printf("Split pos: %u\n", split.bestPos);
    printf("Split dim: %u\n", split.bestDim);
    printf("Split SAH: %f\n", split.bestSAH);

#if 1
    u32 *lOffsets = PushArrayNoZero(arena, u32, output.num);
    u32 *rOffsets = PushArrayNoZero(arena, u32, output.num);
    u32 *lCounts  = PushArrayNoZero(arena, u32, output.num);
    u32 *rCounts  = PushArrayNoZero(arena, u32, output.num);

    u32 lOffset = 0;
    u32 rOffset = extEnd;
    for (u32 i = 0; i < output.num; i++)
    {
        Heuristic *h    = &((Heuristic *)output.out)[i];
        lOffsets[i]     = lOffset;
        u32 entryCounts = 0;
        u32 exitCounts  = 0;
        for (u32 bin = 0; bin < split.bestPos; bin++)
        {
            entryCounts += h->entryCounts[bin][split.bestDim];
        }
        for (u32 bin = split.bestDim; bin < 16; bin++)
        {
            exitCounts += h->exitCounts[bin][split.bestDim];
        }
        lOffset += entryCounts;
        rOffset -= exitCounts;
        rOffsets[i] = rOffset;
        lCounts[i]  = entryCounts;
        rCounts[i]  = exitCounts;
    }
    // SplitPayload payload = SplitPayload(splitOffsets, refOffsets, output.num, output.groupSize);
    counter = OS_StartCounter();

    scheduler.ScheduleAndWait(output.num, 1, [&](u32 jobID) {
        RecordAOSSplits recordL;
        RecordAOSSplits recordR;
        u32 threadStart = 0 + jobID * output.groupSize;
        u32 count       = jobID == output.num - 1 ? numFaces - threadStart : output.groupSize;
        heuristic.Split(mesh, refs, outRefs, lOffsets[jobID], rOffsets[jobID],
                        threadStart, count, split, recordL, recordR);
    });
    time = OS_GetMilliseconds(counter);
    printf("split time: %fms\n", time);

    EndReduce(output);
#endif
#if 0
    ObjectBinner binner(centBounds);
    using Heuristic = HeuristicAOSObjectBinning<32>;
    printf("size: %llu\n", sizeof(Heuristic));

    // HeuristicAOSObjectBinning heuristic(&binner);
    // heuristic.Bin(refs, 0, numFaces);
    PerformanceCounter counter = OS_StartCounter();
    ParallelForOutput output   = ParallelFor<Heuristic>(
        0, numFaces, PARALLEL_THRESHOLD,
        [&](Heuristic &heuristic, u32 start, u32 count) { heuristic.Bin(refs, refRefs, start, count); },
        &binner);
    Heuristic heuristic;
    Reduce(
        heuristic, output, [&](Heuristic &l, const Heuristic &r) { l.Merge(r); }, &binner);
    f32 time = OS_GetMilliseconds(counter);

    printf("bin time: %fms\n", time);

    Split split = BinBest(heuristic.bins, heuristic.counts, &binner);
    printf("Split pos: %u\n", split.bestPos);
    printf("Split dim: %u\n", split.bestDim);
    printf("Split SAH: %f\n", split.bestSAH);

    u32 *lOffsets = PushArrayNoZero(arena, u32, output.num);
    u32 *rOffsets = PushArrayNoZero(arena, u32, output.num);
    u32 lOffset   = 0;
    u32 rOffset   = numFaces;
    for (u32 i = 0; i < output.num; i++)
    {
        Heuristic &h = ((Heuristic *)output.out)[i];
        lOffsets[i]  = lOffset;
        for (u32 bin = 0; bin < split.bestPos; bin++)
        {
            lOffset += h.counts[bin][split.bestDim];
        }
        for (u32 bin = split.bestDim; bin < 32; bin++)
        {
            rOffset -= h.counts[bin][split.bestDim];
        }
        rOffsets[i] = rOffset;
    }
    Assert(lOffset == rOffset);

    PartitionPayload payload(lOffsets, rOffsets, output.num, output.groupSize);

    counter = OS_StartCounter();
    ExtRange range(0, numFaces, numFaces);
    // u32 mid = PartitionParallel(split, range, refs);
    // u32 mid = Partition(split, 0, numFaces, refs);
    // u32 mid = Partition2(split, 0, numFaces, refs, refRefs, refRefs2);
    PartitionParallel(payload, split, range, refs, refRefs, refRefs2);

    time = OS_GetMilliseconds(counter);
    printf("Time elapsed partition: %fms\n", time);

#endif
    for (u32 i = 0; i < OS_NumProcessors(); i++)
    {
        printf("thread time %u: %fms\n", i, threadLocalStatistics[i].miscF);
    }
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
    Bounds centBounds;
    PrimData *data = GeneratePrimData(arena, mesh, count, numFaces, geomBounds, centBounds, true);

    // PerformanceCounter start = OS_StartCounter();
    // HeuristicSAHBinned<32> heuristic(centBounds);
    // heuristic.Bin(data, 0, numFaces);
    // f32 time = OS_GetMilliseconds(start);
    // printf("Time elapsed binning: %fms\n", time);

#if 1
    TestSplitBinningBase heuristic(geomBounds);

    PerformanceCounter start = OS_StartCounter();
    heuristic.Bin(mesh, data, 0, numFaces);
    f32 time = OS_GetMilliseconds(start);
    printf("Time elapsed binning: %fms\n", time);
#endif

#if 1

    TestHeuristic test(heuristic.base, heuristic.scale, heuristic.invScale);
    Split split = BinBest(heuristic.bins, heuristic.numBegin, heuristic.numEnd, &test);

    printf("Split pos: %u\n", split.bestPos);
    printf("Split dim: %u\n", split.bestDim);
    printf("Split SAH: %f\n", split.bestSAH);

    split.bestValue = ((split.bestPos) * heuristic.invScale[split.bestDim]) + heuristic.base[split.bestDim];
    printf("Split value: %f\n", split.bestValue);
    ExtRange range(0, numFaces, u32(numFaces * GROW_AMOUNT));
    start = OS_StartCounter();

    Bounds left;
    Bounds right;
    Bounds centLeft;
    Bounds centRight;
    u32 mid = heuristic.Split(mesh, data, range, split, left, right, centLeft, centRight);
    time    = OS_GetMilliseconds(start);
    printf("Mid: %u\n", mid);

    printf("Time elapsed splitting: %fms\n", time);
    printf("Left bounds min: %f %f %f\n", left.minP[0], left.minP[1], left.minP[2]);
    printf("Left bounds max: %f %f %f\n", left.maxP[0], left.maxP[1], left.maxP[2]);
    printf("Right bounds min: %f %f %f\n", right.minP[0], right.minP[1], right.minP[2]);
    printf("Right bounds max: %f %f %f\n", right.maxP[0], right.maxP[1], right.maxP[2]);

    printf("Left cent bounds min: %f %f %f\n", centLeft.minP[0], centLeft.minP[1], centLeft.minP[2]);
    printf("Left cent bounds max: %f %f %f\n", centLeft.maxP[0], centLeft.maxP[1], centLeft.maxP[2]);
    printf("Right cent bounds min: %f %f %f\n", centRight.minP[0], centRight.minP[1], centRight.minP[2]);
    printf("Right cent bounds max: %f %f %f\n", centRight.maxP[0], centRight.maxP[1], centRight.maxP[2]);

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

// void SOASBVHBuilderTest(TriangleMesh *mesh)
// {
//     Arena *arena = ArenaAlloc();
//     arena->align = 64;
//
//     const u32 numFaces = mesh->numIndices / 3;
//     Bounds geomBounds;
//     Bounds centBounds;
//     PrimDataSOA soa = GenerateSOAData(arena, mesh, numFaces, geomBounds, centBounds);
//
//     BuildSettings settings;
//     settings.intCost = 0.3f;
//
//     RecordSOASplits record;
//     record.geomBounds.FromBounds(geomBounds);
//     record.centBounds.FromBounds(centBounds);
//     record.range      = ExtRange(0, numFaces, u32(numFaces * GROW_AMOUNT));
//     u32 numProcessors = OS_NumProcessors();
//     Arena **arenas    = PushArray(arena, Arena *, numProcessors);
//     for (u32 i = 0; i < numProcessors; i++)
//     {
//         arenas[i] = ArenaAlloc(ARENA_RESERVE_SIZE, LANE_WIDTH * 4);
//     }
//
//     PerformanceCounter counter = OS_StartCounter();
//     BVH4Quantized bvh          = BuildQuantizedSBVH<4>(settings, arenas, mesh, &soa, record);
//     f32 time                   = OS_GetMilliseconds(counter);
//     printf("num faces: %u\n", numFaces);
//     printf("Build time: %fms\n", time);
//
//     f64 totalMiscTime     = 0;
//     u64 numNodes          = 0;
//     u64 totalNodeMemory   = 0;
//     u64 totalRecordMemory = 0;
//     for (u32 i = 0; i < numProcessors; i++)
//     {
//         totalMiscTime += threadLocalStatistics[i].miscF;
//         totalNodeMemory += threadMemoryStatistics[i].totalNodeMemory;
//         totalRecordMemory += threadMemoryStatistics[i].totalRecordMemory;
//         printf("thread time %u: %fms\n", i, threadLocalStatistics[i].miscF);
//         numNodes += threadLocalStatistics[i].misc;
//     }
//     printf("total time: %fms \n", totalMiscTime);
//     printf("num nodes: %llu\n", numNodes);               // num nodes: %llu\n", numNodes);
//     printf("node kb: %llu\n", totalNodeMemory / 1000);   // num nodes: %llu\n", numNodes);
//     printf("record kb: %llu", totalRecordMemory / 1000); // num nodes: %llu\n", numNodes);
// }
void AOSSBVHBuilderTest(TriangleMesh *mesh)
{
    Arena *arena = ArenaAlloc();
    arena->align = 64;

    const u32 numFaces = mesh->numIndices / 3;
    Bounds geomBounds;
    Bounds centBounds;
    PrimRef *refs    = GenerateAOSData(arena, mesh, numFaces, geomBounds, centBounds);
    PrimRef *outRefs = PushArrayNoZero(arena, PrimRef, u32(numFaces * GROW_AMOUNT));

    BuildSettings settings;
    settings.intCost = 0.3f;

    RecordAOSSplits record;
    record.geomBounds = Lane8F32(-geomBounds.minP, geomBounds.maxP);
    record.centBounds = Lane8F32(-centBounds.minP, centBounds.maxP);
    record.SetRange(0, 0, numFaces, u32(numFaces * GROW_AMOUNT));
    u32 numProcessors = OS_NumProcessors();
    Arena **arenas    = PushArray(arena, Arena *, numProcessors);
    for (u32 i = 0; i < numProcessors; i++)
    {
        arenas[i] = ArenaAlloc(ARENA_RESERVE_SIZE, LANE_WIDTH * 4);
    }

    PerformanceCounter counter = OS_StartCounter();
    BVH4Quantized bvh          = BuildQuantizedSBVH<4>(settings, arenas, mesh, refs, outRefs, record);
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
        totalNodeMemory += threadMemoryStatistics[i].totalNodeMemory;
        totalRecordMemory += threadMemoryStatistics[i].totalRecordMemory;
        printf("thread time %u: %fms\n", i, threadLocalStatistics[i].miscF);
        numNodes += threadLocalStatistics[i].misc;
    }
    printf("total time: %fms \n", totalMiscTime);
    printf("num nodes: %llu\n", numNodes);               // num nodes: %llu\n", numNodes);
    printf("node kb: %llu\n", totalNodeMemory / 1000);   // num nodes: %llu\n", numNodes);
    printf("record kb: %llu", totalRecordMemory / 1000); // num nodes: %llu\n", numNodes);
}

} // namespace rt
