#ifndef BVH_SOA_H
#define BVH_SOA_H
namespace rt
{
struct PrimDataSOA
{
    f32 *minX;
    f32 *minY;
    f32 *minZ;
    u32 *geomIDs;
    f32 *maxX;
    f32 *maxY;
    f32 *maxZ;
    u32 *primIDs;
};

// ways of doing this
// what I have now: data is SOA. find the bin index min and max in each dimension. increment the count of the bin and place the
// face index alongside. when the count reaches the max, start working on the 8 triangle/plane test. the code is arranged
// so that the bins are computed and then the previous triangles are added to the bins, counts are incremented, etc. this is
// for scalar vector overlap.
//
// 1. need to test whether this overlap is actually helping
//      results: definitely is not

// 2. test this with AoS? (the 8-wide prim data version). if this is just as fast as soa, then

// 3. test finding all of the bin indices first, and then work on each really really fast. the problem with this is that
// when there are a lot of triangles (e.g like 300 mill), allocation size gets really huge, even if it's temporary.
// how would you handle allocating the bins? if you worst case allocate (i.e. each bin gets the number of primitives),
// allocation sizes get large. if you allocate only the number of primitives and then subdivide, ... . How would this case
// handle primitives that aren't split?

// 4. maybe instead of finding all of the bin indices first, we could tweak the count before work is started on the tests
//      results: makes it slower

__forceinline void ClipTriangleTest(const TriangleMesh *mesh, const u32 faceIndices[8], const f32 leftBound,
                                    const f32 rightBound, Bounds8F32 &out)
{

    Lane8F32 clipLeft(leftBound);
    Lane8F32 clipRight(rightBound);

    Vec3f v0[8];
    Vec3f v1[8];
    Vec3f v2[8];

    u32 faceIndexA = faceIndices[0];
    u32 faceIndexB = faceIndices[1];
    u32 faceIndexC = faceIndices[2];
    u32 faceIndexD = faceIndices[3];

    Lane4F32 v0a = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexA * 3]]));
    Lane4F32 v1a = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexA * 3 + 1]]));
    Lane4F32 v2a = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexA * 3 + 2]]));

    Lane4F32 v0b = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexB * 3]]));
    Lane4F32 v1b = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexB * 3 + 1]]));
    Lane4F32 v2b = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexB * 3 + 2]]));

    Lane4F32 v0c = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexC * 3]]));
    Lane4F32 v1c = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexC * 3 + 1]]));
    Lane4F32 v2c = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexC * 3 + 2]]));

    Lane4F32 v0d = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexD * 3]]));
    Lane4F32 v1d = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexD * 3 + 1]]));
    Lane4F32 v2d = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexD * 3 + 2]]));

    Vec3lf4 p0;
    Vec3lf4 p1;
    Vec3lf4 p2;

    Transpose4x3(v0a, v0b, v0c, v0d, p0.x, p0.y, p0.z);
    Transpose4x3(v1a, v1b, v1c, v1d, p1.x, p1.y, p1.z);
    Transpose4x3(v2a, v2b, v2c, v2d, p2.x, p2.y, p2.z);

    faceIndexA = faceIndices[4];
    faceIndexB = faceIndices[5];
    faceIndexC = faceIndices[6];
    faceIndexD = faceIndices[7];
    v0a        = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexA * 3]]));
    v1a        = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexA * 3 + 1]]));
    v2a        = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexA * 3 + 2]]));

    v0b = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexB * 3]]));
    v1b = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexB * 3 + 1]]));
    v2b = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexB * 3 + 2]]));

    v0c = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexC * 3]]));
    v1c = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexC * 3 + 1]]));
    v2c = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexC * 3 + 2]]));

    v0d = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexD * 3]]));
    v1d = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexD * 3 + 1]]));
    v2d = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexD * 3 + 2]]));

    Vec3lf4 p3;
    Vec3lf4 p4;
    Vec3lf4 p5;

    Transpose4x3(v0a, v0b, v0c, v0d, p3.x, p3.y, p3.z);
    Transpose4x3(v1a, v1b, v1c, v1d, p4.x, p4.y, p4.z);
    Transpose4x3(v2a, v2b, v2c, v2d, p5.x, p5.y, p5.z);

    Lane8F32 v0u(p0[0], p3[0]);
    Lane8F32 v1u(p1[0], p4[0]);
    Lane8F32 v2u(p2[0], p5[0]);

    Lane8F32 v0v(p0[1], p3[1]);
    Lane8F32 v1v(p1[1], p4[1]);
    Lane8F32 v2v(p2[1], p5[1]);

    Lane8F32 v0w(p0[2], p3[2]);
    Lane8F32 v1w(p1[2], p4[2]);
    Lane8F32 v2w(p2[2], p5[2]);

    Lane8F32 minU = Min(v0u, Min(v1u, v2u));
    out.minU      = Min(out.minU, Max(minU, clipLeft));

    Lane8F32 maxU = Max(v0u, Max(v1u, v2u));
    out.maxU      = Max(out.maxU, Min(maxU, clipRight));

    // Edge 0: v0 to v1
    Lane8F32 div0    = Rcp(v1u - v0u);
    Lane8F32 t0Left  = Clamp(Lane8F32(0.f), Lane8F32(1.f), (clipLeft - v0u) * div0);
    Lane8F32 t0Right = Clamp(Lane8F32(0.f), Lane8F32(1.f), (clipRight - v0u) * div0);

    Lane8F32 edge0ClippedVLeft  = FMA(t0Left, v1v - v0v, v0v);
    Lane8F32 edge0ClippedVRight = FMA(t0Right, v1v - v0v, v0v);

    Lane8F32 edge0ClippedWLeft  = FMA(t0Left, v1w - v0w, v0w);
    Lane8F32 edge0ClippedWRight = FMA(t0Right, v1w - v0w, v0w);

    out.minV = Min(out.minV, Min(edge0ClippedVLeft, edge0ClippedVRight));
    out.maxV = Max(out.maxV, Max(edge0ClippedVLeft, edge0ClippedVRight));

    out.minV = Min(out.minV, Min(edge0ClippedVLeft, edge0ClippedVRight));
    out.maxV = Max(out.maxV, Max(edge0ClippedVLeft, edge0ClippedVRight));

    out.minW = Min(out.minW, Min(edge0ClippedWLeft, edge0ClippedWRight));
    out.maxW = Max(out.maxW, Max(edge0ClippedWLeft, edge0ClippedWRight));

    out.minW = Min(out.minW, Min(edge0ClippedWLeft, edge0ClippedWRight));
    out.maxW = Max(out.maxW, Max(edge0ClippedWLeft, edge0ClippedWRight));

    // Edge 1 : v1 to v2
    Lane8F32 div1    = Rcp(v2u - v1u);
    Lane8F32 t1Left  = Clamp(Lane8F32(0.f), Lane8F32(1.f), (clipLeft - v1u) * div1);
    Lane8F32 t1Right = Clamp(Lane8F32(0.f), Lane8F32(1.f), (clipRight - v1u) * div1);

    Lane8F32 edge1ClippedVLeft  = FMA(t1Left, v2v - v1v, v1v);
    Lane8F32 edge1ClippedVRight = FMA(t1Right, v2v - v1v, v1v);

    Lane8F32 edge1ClippedWLeft  = FMA(t1Left, v2w - v1w, v1w);
    Lane8F32 edge1ClippedWRight = FMA(t1Right, v2w - v1w, v1w);

    out.minV = Min(out.minV, Min(edge1ClippedVLeft, edge1ClippedVRight));
    out.maxV = Max(out.maxV, Max(edge1ClippedVLeft, edge1ClippedVRight));

    out.minV = Min(out.minV, Min(edge1ClippedVLeft, edge1ClippedVRight));
    out.maxV = Max(out.maxV, Max(edge1ClippedVLeft, edge1ClippedVRight));

    out.minW = Min(out.minW, Min(edge1ClippedWLeft, edge1ClippedWRight));
    out.maxW = Max(out.maxW, Max(edge1ClippedWLeft, edge1ClippedWRight));

    out.minW = Min(out.minW, Min(edge1ClippedWLeft, edge1ClippedWRight));
    out.maxW = Max(out.maxW, Max(edge1ClippedWLeft, edge1ClippedWRight));

    // Edge 2 : v2 to v0
    Lane8F32 div2    = Rcp(v2u - v1u);
    Lane8F32 t2Left  = Clamp(Lane8F32(0.f), Lane8F32(1.f), (clipLeft - v2u) * div2);
    Lane8F32 t2Right = Clamp(Lane8F32(0.f), Lane8F32(1.f), (clipRight - v2u) * div2);

    Lane8F32 edge2ClippedVLeft  = FMA(t2Left, v0v - v2v, v2v);
    Lane8F32 edge2ClippedVRight = FMA(t2Right, v0v - v2v, v2v);

    Lane8F32 edge2ClippedWLeft  = FMA(t2Left, v0w - v2w, v2w);
    Lane8F32 edge2ClippedWRight = FMA(t2Right, v0w - v2w, v2w);

    out.minV = Min(out.minV, Min(edge2ClippedVLeft, edge2ClippedVRight));
    out.maxV = Max(out.maxV, Max(edge2ClippedVLeft, edge2ClippedVRight));

    out.minV = Min(out.minV, Min(edge2ClippedVLeft, edge2ClippedVRight));
    out.maxV = Max(out.maxV, Max(edge2ClippedVLeft, edge2ClippedVRight));

    out.minW = Min(out.minW, Min(edge2ClippedWLeft, edge2ClippedWRight));
    out.maxW = Max(out.maxW, Max(edge2ClippedWLeft, edge2ClippedWRight));

    out.minW = Min(out.minW, Min(edge2ClippedWLeft, edge2ClippedWRight));
    out.maxW = Max(out.maxW, Max(edge2ClippedWLeft, edge2ClippedWRight));
}

template <i32 numBins = 16>
struct TestSOASplitBinning
{
    Lane8F32 baseX;
    Lane8F32 baseY;
    Lane8F32 baseZ;

    Lane8F32 scaleX;
    Lane8F32 scaleY;
    Lane8F32 scaleZ;

    Lane8F32 base;
    Lane8F32 scale;

    static const u32 LANE_WIDTH = 8;

    u32 faceIndices[3][numBins][LANE_WIDTH];
    u32 binCounts[3][numBins];

    Lane4U32 entryCounts[numBins];
    Lane4U32 exitCounts[numBins];

    Bounds8F32 bins[3][numBins];

    Bounds finalBounds[3][numBins];

    f32 splitPositions[3][numBins + 1];

    TestSOASplitBinning(Bounds &bounds)
    {
        const Lane4F32 eps = 1e-34f;

        Lane8F32 minP(bounds.minP);
        baseX = Shuffle<0>(minP);
        baseY = Shuffle<1>(minP);
        baseZ = Shuffle<2>(minP);

        base = Lane8F32(bounds.minP, bounds.minP);

        const Lane4F32 diag = Max(bounds.maxP - bounds.minP, eps);

        Lane4F32 scale4 = Select(diag > eps, Lane4F32((f32)numBins) / diag, Lane4F32(0.f));

        Lane8F32 scale8(scale4);
        scaleX = Shuffle<0>(scale8);
        scaleY = Shuffle<1>(scale8);
        scaleZ = Shuffle<2>(scale8);

        scale = Lane8F32(scale4, scale4);

        for (u32 dim = 0; dim < 3; dim++)
        {
            for (u32 i = 0; i < numBins; i++)
            {
                binCounts[dim][i]      = 0;
                splitPositions[dim][i] = bounds.minP[dim] + ((i + 1) * diag[dim] / (f32)numBins);
            }
            splitPositions[dim][numBins] = bounds.maxP[dim];
        }
    }

    void BinTest2(TriangleMesh *mesh, PrimRef *refs, u32 start, u32 count)
    {
        Lane8U32 z = Lane8U32(0);
        Lane8U32 e = Lane8U32(numBins - 1);

        u32 binIndices[8];

        for (u32 i = start; i < start + count; i++)
        {
            PrimRef *ref      = &refs[i];
            Lane8U32 binIndex = Clamp(z, e, Flooru((ref->m256 - base) * scale));

            Lane8U32::Store(binIndices, binIndex);

            u32 indexMinX = binIndices[0];
            u32 indexMaxX = binIndices[4];

            Lane8F32 minX = Shuffle<0>(ref->m256);
            Lane8F32 minY = Shuffle<1>(ref->m256);
            Lane8F32 minZ = Shuffle<2>(ref->m256);

            Lane8F32 maxX = Shuffle<4>(ref->m256);
            Lane8F32 maxY = Shuffle<5>(ref->m256);
            Lane8F32 maxZ = Shuffle<6>(ref->m256);

            Lane8F32 maskX(indexMinX == indexMaxX);
            bins[0][indexMinX].minU = MaskMin(maskX, bins[0][indexMinX].minU, minX);
            bins[0][indexMinX].minV = MaskMin(maskX, bins[0][indexMinX].minV, minY);
            bins[0][indexMinX].minW = MaskMin(maskX, bins[0][indexMinX].minW, minZ);

            bins[0][indexMinX].maxU = MaskMax(maskX, bins[0][indexMinX].maxU, maxX);
            bins[0][indexMinX].maxV = MaskMax(maskX, bins[0][indexMinX].maxV, maxY);
            bins[0][indexMinX].maxW = MaskMax(maskX, bins[0][indexMinX].maxW, maxZ);

            entryCounts[indexMinX][0] += 1;
            exitCounts[indexMaxX][0] += 1;

            u32 faceID = ref->primID;

            for (u32 index = indexMinX; index < indexMaxX; index++)
            {
                faceIndices[0][index][binCounts[0][index]++] = faceID;
                if (binCounts[0][index] == LANE_WIDTH)
                {
                    Bounds8F32 out;
                    ClipTriangleTest(mesh, faceIndices[0][index], splitPositions[0][index], splitPositions[0][index + 1], out);
                    binCounts[0][index] = 0;
                    bins[0][index].Extend(out);
                }
            }

            u32 indexMinY = binIndices[1];
            u32 indexMaxY = binIndices[5];
            Lane8F32 maskY(indexMinY == indexMaxY);

            bins[1][indexMinY].minU = MaskMin(maskY, bins[1][indexMinY].minU, minY);
            bins[1][indexMinY].minV = MaskMin(maskY, bins[1][indexMinY].minV, minZ);
            bins[1][indexMinY].minW = MaskMin(maskY, bins[1][indexMinY].minW, minX);

            bins[1][indexMinY].maxU = MaskMax(maskY, bins[1][indexMinY].maxU, maxY);
            bins[1][indexMinY].maxV = MaskMax(maskY, bins[1][indexMinY].maxV, maxZ);
            bins[1][indexMinY].maxW = MaskMax(maskY, bins[1][indexMinY].maxW, maxX);

            entryCounts[indexMinX][1] += 1;
            exitCounts[indexMaxX][1] += 1;

            for (u32 index = indexMinY; index < indexMaxY; index++)
            {
                faceIndices[1][index][binCounts[1][index]++] = faceID;
                if (binCounts[1][index] == LANE_WIDTH)
                {
                    Bounds8F32 out;
                    ClipTriangleTest(mesh, faceIndices[1][index],
                                     splitPositions[1][index], splitPositions[1][index + 1], out);
                    binCounts[1][index] = 0;
                    bins[1][index].Extend(out);
                }
            }

            u32 indexMinZ = binIndices[2];
            u32 indexMaxZ = binIndices[6];
            Lane8F32 maskZ(indexMinZ == indexMaxZ);

            bins[2][indexMinZ].minU = MaskMin(maskZ, bins[2][indexMinZ].minU, minZ);
            bins[2][indexMinZ].minV = MaskMin(maskZ, bins[2][indexMinZ].minV, minX);
            bins[2][indexMinZ].minW = MaskMin(maskZ, bins[2][indexMinZ].minW, minY);

            bins[2][indexMinZ].maxU = MaskMax(maskZ, bins[2][indexMinZ].maxU, maxZ);
            bins[2][indexMinZ].maxV = MaskMax(maskZ, bins[2][indexMinZ].maxV, maxX);
            bins[2][indexMinZ].maxW = MaskMax(maskZ, bins[2][indexMinZ].maxW, maxY);

            entryCounts[indexMinX][2] += 1;
            exitCounts[indexMaxX][2] += 1;

            for (u32 index = indexMinZ; index < indexMaxZ; index++)
            {
                faceIndices[2][index][binCounts[2][index]++] = faceID;
                if (binCounts[2][index] == LANE_WIDTH)
                {
                    Bounds8F32 out;
                    ClipTriangleTest(mesh, faceIndices[2][index],
                                     splitPositions[2][index], splitPositions[2][index + 1], out);
                    binCounts[2][index] = 0;
                    bins[2][index].Extend(out);
                }
            }
        }
    }

    void Bin(TriangleMesh *mesh, PrimDataSOA *soa, u32 start, u32 count)
    {
        u32 faceIDPrev[8];

        u32 indexMinXPrev[8];
        u32 indexMaxXPrev[8];

        u32 indexMinYPrev[8];
        u32 indexMaxYPrev[8];

        u32 indexMinZPrev[8];
        u32 indexMaxZPrev[8];

        Lane8U32 z = Lane8U32(0);
        Lane8U32 e = Lane8U32(numBins - 1);

        u32 i = start;
        // Initial run
#if 0
        Lane8U32 faceIDs = Lane8U32::LoadU(soa->primIDs + i);

        Lane8F32 prevMinX = Lane8F32::LoadU(soa->minX + i);
        Lane8F32 prevMinY = Lane8F32::LoadU(soa->minY + i);
        Lane8F32 prevMinZ = Lane8F32::LoadU(soa->minZ + i);

        Lane8F32 prevMaxX = Lane8F32::LoadU(soa->maxX + i);
        Lane8F32 prevMaxY = Lane8F32::LoadU(soa->maxY + i);
        Lane8F32 prevMaxZ = Lane8F32::LoadU(soa->maxZ + i);

        // X
        Lane8U32 binIndexMinX0 = Clamp(z, e, Flooru((prevMinX - baseX) * scaleX));
        Lane8U32 binIndexMaxX0 = Clamp(z, e, Flooru((prevMaxX - baseX) * scaleX));
        Lane8U32::Store(indexMinXPrev, binIndexMinX0);
        Lane8U32::Store(indexMaxXPrev, binIndexMaxX0);

        // Y
        Lane8U32 binIndexMinY0 = Clamp(z, e, Flooru((prevMinY - baseY) * scaleY));
        Lane8U32 binIndexMaxY0 = Clamp(z, e, Flooru((prevMaxY - baseY) * scaleY));
        Lane8U32::Store(indexMinYPrev, binIndexMinY0);
        Lane8U32::Store(indexMaxYPrev, binIndexMaxY0);

        // Z
        Lane8U32 binIndexMinZ0 = Clamp(z, e, Flooru((prevMinZ - baseZ) * scaleZ));
        Lane8U32 binIndexMaxZ0 = Clamp(z, e, Flooru((prevMaxZ - baseZ) * scaleZ));
        Lane8U32::Store(indexMinZPrev, binIndexMinZ0);
        Lane8U32::Store(indexMaxZPrev, binIndexMaxZ0);

        Lane8U32::Store(faceIDPrev, faceIDs);

        i += 8;
#endif
        for (; i < start + count; i += 8)
        {
            Lane8U32 faceIds = Lane8U32::LoadU(soa->primIDs + i);

#if 0
            Lane8F32 minX = Lane8F32::LoadU(soa->minX + i);
            Lane8F32 minY = Lane8F32::LoadU(soa->minY + i);
            Lane8F32 minZ = Lane8F32::LoadU(soa->minZ + i);

            Lane8F32 maxX = Lane8F32::LoadU(soa->maxX + i);
            Lane8F32 maxY = Lane8F32::LoadU(soa->maxY + i);
            Lane8F32 maxZ = Lane8F32::LoadU(soa->maxZ + i);

            // X
            Lane8U32 binIndexMinX = Clamp(z, e, Flooru((minX - baseX) * scaleX));
            Lane8U32 binIndexMaxX = Clamp(z, e, Flooru((maxX - baseX) * scaleX));
            Lane8U32 faceIDs = Lane8U32::LoadU(soa->primIDs + i);
#endif
#if 1

            Lane8F32 prevMinX = Lane8F32::LoadU(soa->minX + i);
            Lane8F32 prevMinY = Lane8F32::LoadU(soa->minY + i);
            Lane8F32 prevMinZ = Lane8F32::LoadU(soa->minZ + i);

            Lane8F32 prevMaxX = Lane8F32::LoadU(soa->maxX + i);
            Lane8F32 prevMaxY = Lane8F32::LoadU(soa->maxY + i);
            Lane8F32 prevMaxZ = Lane8F32::LoadU(soa->maxZ + i);

            Lane8U32 binIndexMinX = Clamp(z, e, Flooru((prevMinX - baseX) * scaleX));
            Lane8U32 binIndexMaxX = Clamp(z, e, Flooru((prevMaxX - baseX) * scaleX));

            Lane8U32::Store(indexMinXPrev, binIndexMinX);
            Lane8U32::Store(indexMaxXPrev, binIndexMaxX);
            Lane8U32::Store(faceIDPrev, faceIds);
#endif

            for (u32 prevIndex = 0; prevIndex < 8; prevIndex++)
            {
                u32 faceID    = faceIDPrev[prevIndex];
                u32 indexMinX = indexMinXPrev[prevIndex];
                u32 indexMaxX = indexMaxXPrev[prevIndex];

                bool noSplit  = indexMinX == indexMaxX;
                Lane8F32 mask = Lane8F32::Mask(noSplit);
                Lane8U32 perm(i);

                bins[0][indexMinX].minU = MaskMin(mask, bins[0][indexMinX].minU, Shuffle(prevMinX, perm));
                bins[0][indexMinX].minV = MaskMin(mask, bins[0][indexMinX].minV, Shuffle(prevMinY, perm));
                bins[0][indexMinX].minW = MaskMin(mask, bins[0][indexMinX].minW, Shuffle(prevMinZ, perm));

                bins[0][indexMinX].maxU = MaskMax(mask, bins[0][indexMinX].maxU, Shuffle(prevMaxX, perm));
                bins[0][indexMinX].maxV = MaskMax(mask, bins[0][indexMinX].maxV, Shuffle(prevMaxY, perm));
                bins[0][indexMinX].maxW = MaskMax(mask, bins[0][indexMinX].maxW, Shuffle(prevMaxZ, perm));

                entryCounts[indexMinX][0] += 1;
                exitCounts[indexMaxX][0] += 1;

                for (u32 index = indexMinX; index <= indexMaxX; index++)
                {
                    faceIndices[0][index][binCounts[0][index]++] = faceID;
                    if (binCounts[0][index] == LANE_WIDTH)
                    {
                        Bounds8F32 out;
                        ClipTriangleTest(mesh, faceIndices[0][index], splitPositions[0][index], splitPositions[0][index + 1], out);
                        binCounts[0][index] = 0;
                        bins[0][index].Extend(out);
                    }
                }
            }

#if 0
            Lane8U32::Store(indexMinXPrev, binIndexMinX);
            Lane8U32::Store(indexMaxXPrev, binIndexMaxX);
#endif

            // Y
#if 0
            Lane8U32 binIndexMinY = Clamp(z, e, Flooru((minY - baseY) * scaleY));
            Lane8U32 binIndexMaxY = Clamp(z, e, Flooru((maxY - baseY) * scaleY));
#endif
#if 1
            Lane8U32 binIndexMinY = Clamp(z, e, Flooru((prevMinY - baseY) * scaleY));
            Lane8U32 binIndexMaxY = Clamp(z, e, Flooru((prevMaxY - baseY) * scaleY));

            Lane8U32::Store(indexMinYPrev, binIndexMinY);
            Lane8U32::Store(indexMaxYPrev, binIndexMaxY);
#endif

            for (u32 prevIndex = 0; prevIndex < 8; prevIndex++)
            {
                u32 faceID    = faceIDPrev[prevIndex];
                u32 indexMinY = indexMinYPrev[prevIndex];
                u32 indexMaxY = indexMaxYPrev[prevIndex];

                bool noSplit  = indexMinY == indexMaxY;
                Lane8F32 mask = Lane8F32::Mask(noSplit);
                Lane8U32 perm(i);

                bins[1][indexMinY].minU = MaskMin(mask, bins[1][indexMinY].minU, Shuffle(prevMinY, perm));
                bins[1][indexMinY].minV = MaskMin(mask, bins[1][indexMinY].minV, Shuffle(prevMinZ, perm));
                bins[1][indexMinY].minW = MaskMin(mask, bins[1][indexMinY].minW, Shuffle(prevMinX, perm));

                bins[1][indexMinY].maxU = MaskMax(mask, bins[1][indexMinY].maxU, Shuffle(prevMaxY, perm));
                bins[1][indexMinY].maxV = MaskMax(mask, bins[1][indexMinY].maxV, Shuffle(prevMaxZ, perm));
                bins[1][indexMinY].maxW = MaskMax(mask, bins[1][indexMinY].maxW, Shuffle(prevMaxX, perm));

                entryCounts[indexMinY][1] += 1;
                exitCounts[indexMinY][1] += 1;

                for (u32 index = indexMinY; index <= indexMaxY; index++)
                {
                    faceIndices[1][index][binCounts[1][index]++] = faceID;
                    if (binCounts[1][index] == LANE_WIDTH)
                    {
                        Bounds8F32 out;
                        ClipTriangleTest(mesh, faceIndices[1][index], splitPositions[1][index], splitPositions[1][index + 1], out);
                        binCounts[1][index] = 0;
                        bins[1][index].Extend(out);
                    }
                }
            }

#if 0
            Lane8U32::Store(indexMinYPrev, binIndexMinY);
            Lane8U32::Store(indexMaxYPrev, binIndexMaxY);
#endif

            // Z
#if 0
            Lane8U32 binIndexMinZ = Clamp(z, e, Flooru((minZ - baseZ) * scaleZ));
            Lane8U32 binIndexMaxZ = Clamp(z, e, Flooru((maxZ - baseZ) * scaleZ));
#endif

#if 1
            Lane8U32 binIndexMinZ = Clamp(z, e, Flooru((prevMinZ - baseZ) * scaleZ));
            Lane8U32 binIndexMaxZ = Clamp(z, e, Flooru((prevMaxZ - baseZ) * scaleZ));

            Lane8U32::Store(indexMinZPrev, binIndexMinZ);
            Lane8U32::Store(indexMaxZPrev, binIndexMaxZ);
#endif

            for (u32 prevIndex = 0; prevIndex < 8; prevIndex++)
            {
                u32 faceID    = faceIDPrev[prevIndex];
                u32 indexMinZ = indexMinZPrev[prevIndex];
                u32 indexMaxZ = indexMaxZPrev[prevIndex];

                bool noSplit = indexMinZ == indexMaxZ;
                Lane8F32 mask = Lane8F32::Mask(noSplit);
                Lane8U32 perm(i);

                bins[2][indexMinZ].minU = MaskMin(mask, bins[2][indexMinZ].minU, Shuffle(prevMinZ, perm));
                bins[2][indexMinZ].minV = MaskMin(mask, bins[2][indexMinZ].minV, Shuffle(prevMinX, perm));
                bins[2][indexMinZ].minW = MaskMin(mask, bins[2][indexMinZ].minW, Shuffle(prevMinY, perm));

                bins[2][indexMinZ].maxU = MaskMax(mask, bins[2][indexMinZ].maxU, Shuffle(prevMaxZ, perm));
                bins[2][indexMinZ].maxV = MaskMax(mask, bins[2][indexMinZ].maxV, Shuffle(prevMaxX, perm));
                bins[2][indexMinZ].maxW = MaskMax(mask, bins[2][indexMinZ].maxW, Shuffle(prevMaxY, perm));

                entryCounts[indexMinZ][2] += 1;
                exitCounts[indexMaxZ][2] += 1;

                for (u32 index = indexMinZ; index <= indexMaxZ; index++)
                {
                    faceIndices[2][index][binCounts[2][index]++] = faceID;
                    if (binCounts[2][index] == LANE_WIDTH)
                    {
                        Bounds8F32 out;
                        ClipTriangleTest(mesh, faceIndices[2][index], splitPositions[2][index], splitPositions[2][index + 1], out);
                        binCounts[2][index] = 0;
                        bins[2][index].Extend(out);
                    }
                }
            }

#if 0
            Lane8U32::Store(indexMinZPrev, binIndexMinZ);
            Lane8U32::Store(indexMaxZPrev, binIndexMaxZ);

            Lane8U32::Store(faceIDPrev, faceIds);
#endif

#if 0
            prevMinX = minX;
            prevMinY = minY;
            prevMinZ = minZ;

            prevMaxX = maxX;
            prevMaxY = maxY;
            prevMaxZ = maxZ;
#endif
        }

        // Finish the final prev
#if 0
        for (u32 prevIndex = 0; prevIndex < 8; prevIndex++)
        {
            u32 faceID    = faceIDPrev[prevIndex];
            u32 indexMinX = indexMinXPrev[prevIndex];
            u32 indexMaxX = indexMaxXPrev[prevIndex];

            Lane8F32 mask(indexMinX == indexMaxX);
            Lane8U32 perm(i);

            bins[0][indexMinX].minU = MaskMin(mask, bins[0][indexMinX].minU, Shuffle(prevMinX, perm));
            bins[0][indexMinX].minV = MaskMin(mask, bins[0][indexMinX].minV, Shuffle(prevMinY, perm));
            bins[0][indexMinX].minW = MaskMin(mask, bins[0][indexMinX].minW, Shuffle(prevMinZ, perm));

            bins[0][indexMinX].maxU = MaskMax(mask, bins[0][indexMinX].maxU, Shuffle(prevMaxX, perm));
            bins[0][indexMinX].maxV = MaskMax(mask, bins[0][indexMinX].maxV, Shuffle(prevMaxY, perm));
            bins[0][indexMinX].maxW = MaskMax(mask, bins[0][indexMinX].maxW, Shuffle(prevMaxZ, perm));

            for (u32 index = indexMinX; index < indexMaxX; index++)
            {
                faceIndices[0][index][binCounts[0][index]++] = faceID;
                if (binCounts[0][index] == LANE_WIDTH)
                {
                    Bounds8F32 out;
                    ClipTriangleTest(mesh, faceIndices[0][index], splitPositions[0][index], splitPositions[0][index + 1], out);
                    binCounts[0][index] = 0;
                    bins[0][index].Extend(out);
                }
            }
        }
        for (u32 prevIndex = 0; prevIndex < 8; prevIndex++)
        {
            u32 faceID    = faceIDPrev[prevIndex];
            u32 indexMinY = indexMinYPrev[prevIndex];
            u32 indexMaxY = indexMaxYPrev[prevIndex];

            Lane8F32 mask(indexMinY == indexMaxY);
            Lane8U32 perm(i);

            bins[1][indexMinY].minU = MaskMin(mask, bins[1][indexMinY].minU, Shuffle(prevMinY, perm));
            bins[1][indexMinY].minV = MaskMin(mask, bins[1][indexMinY].minV, Shuffle(prevMinZ, perm));
            bins[1][indexMinY].minW = MaskMin(mask, bins[1][indexMinY].minW, Shuffle(prevMinX, perm));

            bins[1][indexMinY].maxU = MaskMax(mask, bins[1][indexMinY].maxU, Shuffle(prevMaxY, perm));
            bins[1][indexMinY].maxV = MaskMax(mask, bins[1][indexMinY].maxV, Shuffle(prevMaxZ, perm));
            bins[1][indexMinY].maxW = MaskMax(mask, bins[1][indexMinY].maxW, Shuffle(prevMaxX, perm));

            for (u32 index = indexMinY; index < indexMaxY; index++)
            {
                faceIndices[1][index][binCounts[1][index]++] = faceID;
                if (binCounts[1][index] == LANE_WIDTH)
                {
                    Bounds8F32 out;
                    ClipTriangleTest(mesh, faceIndices[1][index], splitPositions[1][index], splitPositions[1][index + 1], out);
                    binCounts[1][index] = 0;
                    bins[1][index].Extend(out);
                }
            }
        }
        for (u32 prevIndex = 0; prevIndex < 8; prevIndex++)
        {
            u32 faceID    = faceIDPrev[prevIndex];
            u32 indexMinZ = indexMinZPrev[prevIndex];
            u32 indexMaxZ = indexMaxZPrev[prevIndex];

            Lane8F32 mask(indexMinZ == indexMaxZ);
            Lane8U32 perm(i);

            bins[1][indexMinZ].minU = MaskMin(mask, bins[1][indexMinZ].minU, Shuffle(prevMinZ, perm));
            bins[1][indexMinZ].minV = MaskMin(mask, bins[1][indexMinZ].minV, Shuffle(prevMinX, perm));
            bins[1][indexMinZ].minW = MaskMin(mask, bins[1][indexMinZ].minW, Shuffle(prevMinY, perm));

            bins[1][indexMinZ].maxU = MaskMax(mask, bins[1][indexMinZ].maxU, Shuffle(prevMaxZ, perm));
            bins[1][indexMinZ].maxV = MaskMax(mask, bins[1][indexMinZ].maxV, Shuffle(prevMaxX, perm));
            bins[1][indexMinZ].maxW = MaskMax(mask, bins[1][indexMinZ].maxW, Shuffle(prevMaxY, perm));

            for (u32 index = indexMinZ; index < indexMaxZ; index++)
            {
                faceIndices[2][index][binCounts[2][index]++] = faceID;
                if (binCounts[2][index] == LANE_WIDTH)
                {
                    Bounds8F32 out;
                    ClipTriangleTest(mesh, faceIndices[2][index], splitPositions[2][index], splitPositions[2][index + 1], out);
                    binCounts[2][index] = 0;
                    bins[2][index].Extend(out);
                }
            }
        }
#endif

        Lane8F32 posInf(pos_inf);
        Lane8F32 negInf(neg_inf);
        for (u32 dim = 0; dim < 3; dim++)
        {
            for (u32 binIndex = 0; binIndex < numBins; binIndex++)
            {
                u32 remainingCount = binCounts[0][binIndex];
                u32 bitMask        = (1 << remainingCount) - 1;
                Lane8F32 mask      = Lane8F32::Mask(bitMask);
                Bounds8F32 out;
                ClipTriangleTest(mesh, faceIndices[dim][binIndex], splitPositions[dim][binIndex],
                                 splitPositions[dim][binIndex + 1], out);

                out.minU = Select(mask, out.minU, posInf);
                out.minV = Select(mask, out.minV, posInf);
                out.minW = Select(mask, out.minW, posInf);
                out.maxU = Select(mask, out.maxU, negInf);
                out.maxV = Select(mask, out.maxV, negInf);
                out.maxW = Select(mask, out.maxW, negInf);

                binCounts[dim][binIndex] = 0;
                bins[dim][binIndex].Extend(out);
            }
        }

        for (i = 0; i < numBins; i++)
        {
            Bounds8F32 &bX = bins[0][i];
            f32 bXMinX     = ReduceMin(bX.minU);
            f32 bXMinY     = ReduceMin(bX.minV);
            f32 bXMinZ     = ReduceMin(bX.minW);

            f32 bXMaxX = ReduceMax(bX.maxU);
            f32 bXMaxY = ReduceMax(bX.maxV);
            f32 bXMaxZ = ReduceMax(bX.maxW);

            Lane4F32 xMinP(bXMinX, bXMinY, bXMinZ, 0.f);
            Lane4F32 xMaxP(bXMaxX, bXMaxY, bXMaxZ, 0.f);

            finalBounds[0][i].Extend(xMinP, xMaxP);

            Bounds8F32 &bY = bins[1][i];
            f32 bYMinX     = ReduceMin(bY.minW);
            f32 bYMinY     = ReduceMin(bY.minU);
            f32 bYMinZ     = ReduceMin(bY.minV);

            f32 bYMaxX = ReduceMax(bY.maxW);
            f32 bYMaxY = ReduceMax(bY.maxU);
            f32 bYMaxZ = ReduceMax(bY.maxV);

            Lane4F32 yMinP(bYMinX, bYMinY, bYMinZ, 0.f);
            Lane4F32 yMaxP(bYMaxX, bYMaxY, bYMaxZ, 0.f);

            finalBounds[1][i].Extend(yMinP, yMaxP);

            Bounds8F32 &bZ = bins[2][i];
            f32 bZMinX     = ReduceMin(bZ.minV);
            f32 bZMinY     = ReduceMin(bZ.minW);
            f32 bZMinZ     = ReduceMin(bZ.minU);

            f32 bZMaxX = ReduceMax(bZ.maxU);
            f32 bZMaxY = ReduceMax(bZ.maxV);
            f32 bZMaxZ = ReduceMax(bZ.maxW);

            Lane4F32 zMinP(bZMinX, bZMinY, bZMinZ, 0.f);
            Lane4F32 zMaxP(bZMaxX, bZMaxY, bZMaxZ, 0.f);

            finalBounds[2][i].Extend(zMinP, zMaxP);
        }
    }
};

// NOTE: this is embree's implementation of split binning for SBVH
template <i32 numBins = 16>
struct TestSplitBinningBase
{
    Lane4F32 base;
    Lane4F32 scale;
    Lane4F32 invScale;

    Lane4U32 numEnd[numBins];
    Lane4U32 numBegin[numBins];
    Bounds bins[numBins][3];

    TestSplitBinningBase(Bounds &bounds)
    {
        const Lane4F32 eps = 1e-34f;

        base = bounds.minP;

        const Lane4F32 diag = Max(bounds.maxP - bounds.minP, eps);
        scale               = Select(diag > eps, Lane4F32((f32)numBins) / diag, Lane4F32(0.f));
        invScale            = (diag / (f32)numBins);

        for (u32 i = 0; i < numBins; i++)
        {
            numEnd[0]   = 0;
            numBegin[0] = 0;
        }
    }
    __forceinline void Add(const u32 dim, const u32 beginID, const u32 endID, const u32 binID, const Bounds &b, const u32 n = 1)
    {
        numEnd[endID][dim] += n;
        numBegin[endID][dim] += n;
        bins[binID][dim].Extend(b);
    }
    __forceinline Lane4U32 GetBin(const Lane4F32 &l)
    {
        return Clamp(Lane4U32(0), Lane4U32(numBins - 1), Flooru((l - base) * scale));
    }
    __forceinline f32 GetPos(u32 bin, u32 dim)
    {
        return FMA((f32)bin, invScale[dim], base[dim]);
    }
    __forceinline void Bin(TriangleMesh *mesh, const PrimData *source, u32 start, u32 count)
    {
        for (u32 i = start; i < start + count; i++)
        {
            const PrimData &prim = source[i];

            // if (unlikely(splits <= 1))
            // {
            //     const vint4 bin = mapping.bin(center(prim.bounds()));
            //     for (size_t dim = 0; dim < 3; dim++)
            //     {
            //         assert(bin[dim] >= (int)0 && bin[dim] < (int)BINS);
            //         add(dim, bin[dim], bin[dim], bin[dim], prim.bounds());
            //     }
            // }
            const Lane4U32 bin0 = GetBin(prim.minP);
            const Lane4U32 bin1 = GetBin(prim.maxP);

            for (u32 dim = 0; dim < 3; dim++)
            {
                u32 bin;
                u32 l = bin0[dim];
                u32 r = bin1[dim];

                // same bin optimization
                if (l == r)
                {
                    Add(dim, l, l, l, prim.GetBounds());
                    continue;
                }
                u32 bin_start = bin0[dim];
                u32 bin_end   = bin1[dim];
                Bounds rest   = prim.GetBounds();

                /* assure that split position always overlaps the primitive bounds */
                while (bin_start < bin_end && GetPos(bin_start + 1, dim) <= rest.minP[dim]) bin_start++;
                while (bin_start < bin_end && GetPos(bin_end, dim) >= rest.maxP[dim]) bin_end--;

                for (bin = bin_start; bin < bin_end; bin++)
                {
                    const float pos = GetPos(bin + 1, dim);
                    Bounds left, right;
                    ClipTriangleSimple(mesh, prim.PrimID(), dim, pos, left, right);

                    if (left.Empty()) l++;
                    bins[bin][dim].Extend(left);
                    rest = right;
                }
                if (rest.Empty()) r--;
                Add(dim, l, r, bin, rest);
            }
        }
    }
};

template <i32 numBins = 16>
__forceinline Split SpatialSplitBest(const Bounds bounds[3][numBins],
                                     const Lane4U32 *entryCounts,
                                     const Lane4U32 *exitCounts,
                                     const u32 blockShift = 0)
{
    Bounds boundsDimX;
    Bounds boundsDimY;
    Bounds boundsDimZ;

    Lane4U32 count = 0;
    Lane4U32 lCounts[numBins];
    Lane4F32 area[numBins] = {};

    const u32 blockAdd = (1 << blockShift) - 1;

    for (u32 i = 0; i < numBins - 1; i++)
    {
        count += entryCounts[i];
        lCounts[i] = count;

        boundsDimX.Extend(bounds[0][i]);
        boundsDimY.Extend(bounds[1][i]);
        boundsDimZ.Extend(bounds[2][i]);

        Lane4F32 minX, minY, minZ;
        Lane4F32 maxX, maxY, maxZ;
        Transpose3x3(boundsDimX.minP, boundsDimY.minP, boundsDimZ.minP, minX, minY, minZ);
        Transpose3x3(boundsDimX.maxP, boundsDimY.maxP, boundsDimZ.maxP, maxX, maxY, maxZ);

        Lane4F32 extentX = maxX - minX;
        Lane4F32 extentY = maxY - minY;
        Lane4F32 extentZ = maxZ - minZ;

        area[i] = FMA(extentX, extentY + extentZ, extentY * extentZ);
    }

    count = 0;

    boundsDimX        = Bounds();
    boundsDimY        = Bounds();
    boundsDimZ        = Bounds();
    Lane4F32 lBestSAH = pos_inf;
    Lane4U32 lBestPos = 0;
    for (u32 i = numBins - 1; i >= 1; i--)
    {
        count += exitCounts[i];

        boundsDimX.Extend(bounds[0][i]);
        boundsDimY.Extend(bounds[1][i]);
        boundsDimZ.Extend(bounds[2][i]);

        Lane4F32 minX, minY, minZ;
        Lane4F32 maxX, maxY, maxZ;
        Transpose3x3(boundsDimX.minP, boundsDimY.minP, boundsDimZ.minP, minX, minY, minZ);
        Transpose3x3(boundsDimX.maxP, boundsDimY.maxP, boundsDimZ.maxP, maxX, maxY, maxZ);

        Lane4F32 extentX = maxX - minX;
        Lane4F32 extentY = maxY - minY;
        Lane4F32 extentZ = maxZ - minZ;

        const Lane4F32 rArea = FMA(extentX, extentY + extentZ, extentY * extentZ);

        const Lane4U32 lCount = (lCounts[i - 1] + blockAdd) >> blockShift;
        const Lane4U32 rCount = (count + blockAdd) >> blockShift;
        const Lane4F32 lArea  = area[i - 1];

        // TODO: consider increasing the cost of having empty children/leaves
        // (lCount & (blockAdd - 1));
        const Lane4F32 sah = FMA(rArea, Lane4F32(rCount), lArea * Lane4F32(lCount));

        lBestPos = Select(sah < lBestSAH, Lane4U32(i), lBestPos);
        lBestSAH = Select(sah < lBestSAH, sah, lBestSAH);
    }

    f32 bestArea = pos_inf;
    u32 bestPos  = 0;
    u32 bestDim  = 0;
    for (u32 dim = 0; dim < 3; dim++)
    {
        // if (scale[dim] == 0.f) continue;

        if (lBestSAH[dim] < bestArea)
        {
            bestArea = lBestSAH[dim];
            bestPos  = lBestPos[dim];
            bestDim  = dim;
        }
    }
    // f32 bestValue = splitPositions[bestDim][bestPos + 1];
    return Split(bestArea, bestPos, bestDim, 0.f); // bestValue);
}

} // namespace rt

#endif
