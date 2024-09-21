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

__forceinline void ClipTriangleTest(const TriangleMesh *mesh, const u32 faceIndices[8], const u32 dim,
                                    const f32 leftBound, const f32 rightBound, Bounds8F32 &out)
{

    Assert(leftBound < rightBound);
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

    u32 v = LUTAxis[dim];
    u32 w = LUTAxis[v];
    Lane8F32 v0u(p0[dim], p3[dim]);
    Lane8F32 v1u(p1[dim], p4[dim]);
    Lane8F32 v2u(p2[dim], p5[dim]);

    Lane8F32 v0v(p0[v], p3[v]);
    Lane8F32 v1v(p1[v], p4[v]);
    Lane8F32 v2v(p2[v], p5[v]);

    Lane8F32 v0w(p0[w], p3[w]);
    Lane8F32 v1w(p1[w], p4[w]);
    Lane8F32 v2w(p2[w], p5[w]);

    Lane8F32 minU = Min(v0u, Min(v1u, v2u));
    out.minU      = Min(out.minU, Max(minU, clipLeft));

    Lane8F32 maxU = Max(v0u, Max(v1u, v2u));
    out.maxU      = Max(out.maxU, Min(maxU, clipRight));

    Lane8F32 v0uClipLeftMask = v0u >= clipLeft;
    Lane8F32 v1uClipLeftMask = v1u >= clipLeft;
    Lane8F32 v2uClipLeftMask = v2u >= clipLeft;

    Lane8F32 v0uClipRightMask = v0u <= clipRight;
    Lane8F32 v1uClipRightMask = v1u <= clipRight;
    Lane8F32 v2uClipRightMask = v2u <= clipRight;

    out.MaskExtendVW(v0uClipLeftMask & v0uClipRightMask, v0v, v0w);
    out.MaskExtendVW(v1uClipLeftMask & v1uClipRightMask, v1v, v1w);
    out.MaskExtendVW(v2uClipLeftMask & v2uClipRightMask, v2v, v2w);

    // Edge 0: v0 to v1
    Lane8F32 div0    = Select(v1u == v0u, Lane8F32(zero), Rcp(v1u - v0u));
    Lane8F32 t0Left  = (clipLeft - v0u) * div0;
    Lane8F32 t0Right = (clipRight - v0u) * div0;

    Lane8F32 sub0v              = v1v - v0v;
    Lane8F32 sub0w              = v1w - v0w;
    Lane8F32 edge0ClippedVLeft  = FMA(t0Left, sub0v, v0v);
    Lane8F32 edge0ClippedVRight = FMA(t0Right, sub0v, v0v);

    Lane8F32 edge0ClippedWLeft  = FMA(t0Left, sub0w, v0w);
    Lane8F32 edge0ClippedWRight = FMA(t0Right, sub0w, v0w);

    Lane8F32 edgeIsClippedLeftMask0  = v0uClipLeftMask ^ v1uClipLeftMask;
    Lane8F32 edgeIsClippedRightMask0 = v0uClipRightMask ^ v1uClipRightMask;

    out.MaskExtendVW(edgeIsClippedLeftMask0, edge0ClippedVLeft, edge0ClippedWLeft);
    out.MaskExtendVW(edgeIsClippedRightMask0, edge0ClippedVRight, edge0ClippedWRight);

    // Edge 1 : v1 to v2
    Lane8F32 div1    = Select(v2u == v1u, Lane8F32(zero), Rcp(v2u - v1u));
    Lane8F32 t1Left  = (clipLeft - v1u) * div1;
    Lane8F32 t1Right = (clipRight - v1u) * div1;

    Lane8F32 sub1v              = v2v - v1v;
    Lane8F32 sub1w              = v2w - v1w;
    Lane8F32 edge1ClippedVLeft  = FMA(t1Left, sub1v, v1v);
    Lane8F32 edge1ClippedVRight = FMA(t1Right, sub1v, v1v);

    Lane8F32 edge1ClippedWLeft  = FMA(t1Left, sub1w, v1w);
    Lane8F32 edge1ClippedWRight = FMA(t1Right, sub1w, v1w);

    Lane8F32 edgeIsClippedLeftMask1  = v2uClipLeftMask ^ v1uClipLeftMask;
    Lane8F32 edgeIsClippedRightMask1 = v2uClipRightMask ^ v1uClipRightMask;

    out.MaskExtendVW(edgeIsClippedLeftMask1, edge1ClippedVLeft, edge1ClippedWLeft);
    out.MaskExtendVW(edgeIsClippedRightMask1, edge1ClippedVRight, edge1ClippedWRight);

    // Edge 2 : v2 to v0
    Lane8F32 div2    = Select(v0u == v2u, Lane8F32(zero), Rcp(v0u - v2u));
    Lane8F32 t2Left  = (clipLeft - v2u) * div2;
    Lane8F32 t2Right = (clipRight - v2u) * div2;

    Lane8F32 sub2v              = v0v - v2v;
    Lane8F32 sub2w              = v0w - v2w;
    Lane8F32 edge2ClippedVLeft  = FMA(t2Left, sub2v, v2v);
    Lane8F32 edge2ClippedVRight = FMA(t2Right, sub2v, v2v);

    Lane8F32 edge2ClippedWLeft  = FMA(t2Left, sub2w, v2w);
    Lane8F32 edge2ClippedWRight = FMA(t2Right, sub2w, v2w);

    Lane8F32 edgeIsClippedLeftMask2  = v0uClipLeftMask ^ v2uClipLeftMask;
    Lane8F32 edgeIsClippedRightMask2 = v0uClipRightMask ^ v2uClipRightMask;

    out.MaskExtendVW(edgeIsClippedLeftMask2, edge2ClippedVLeft, edge2ClippedWLeft);
    out.MaskExtendVW(edgeIsClippedRightMask2, edge2ClippedVRight, edge2ClippedWRight);
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
                splitPositions[dim][i] = bounds.minP[dim] + (i * diag[dim] / (f32)numBins);
            }
            splitPositions[dim][numBins] = bounds.maxP[dim];
        }
    }

    // void BinTest2(TriangleMesh *mesh, PrimRef *refs, u32 start, u32 count)
    // {
    //     Lane8U32 z = Lane8U32(0);
    //     Lane8U32 e = Lane8U32(numBins - 1);
    //
    //     u32 binIndices[8];
    //
    //     for (u32 i = start; i < start + count; i++)
    //     {
    //         PrimRef *ref      = &refs[i];
    //         Lane8U32 binIndex = Clamp(z, e, Flooru((ref->m256 - base) * scale));
    //
    //         Lane8U32::Store(binIndices, binIndex);
    //
    //         u32 indexMinX = binIndices[0];
    //         u32 indexMaxX = binIndices[4];
    //
    //         Lane8F32 minX = Shuffle<0>(ref->m256);
    //         Lane8F32 minY = Shuffle<1>(ref->m256);
    //         Lane8F32 minZ = Shuffle<2>(ref->m256);
    //
    //         Lane8F32 maxX = Shuffle<4>(ref->m256);
    //         Lane8F32 maxY = Shuffle<5>(ref->m256);
    //         Lane8F32 maxZ = Shuffle<6>(ref->m256);
    //
    //         Lane8F32 maskX(indexMinX == indexMaxX);
    //         bins[0][indexMinX].minU = MaskMin(maskX, bins[0][indexMinX].minU, minX);
    //         bins[0][indexMinX].minV = MaskMin(maskX, bins[0][indexMinX].minV, minY);
    //         bins[0][indexMinX].minW = MaskMin(maskX, bins[0][indexMinX].minW, minZ);
    //
    //         bins[0][indexMinX].maxU = MaskMax(maskX, bins[0][indexMinX].maxU, maxX);
    //         bins[0][indexMinX].maxV = MaskMax(maskX, bins[0][indexMinX].maxV, maxY);
    //         bins[0][indexMinX].maxW = MaskMax(maskX, bins[0][indexMinX].maxW, maxZ);
    //
    //         entryCounts[indexMinX][0] += 1;
    //         exitCounts[indexMaxX][0] += 1;
    //
    //         u32 faceID = ref->primID;
    //
    //         for (u32 index = indexMinX; index < indexMaxX; index++)
    //         {
    //             faceIndices[0][index][binCounts[0][index]++] = faceID;
    //             if (binCounts[0][index] == LANE_WIDTH)
    //             {
    //                 Bounds8F32 out;
    //                 ClipTriangleTest(mesh, faceIndices[0][index], splitPositions[0][index], splitPositions[0][index + 1], out);
    //                 binCounts[0][index] = 0;
    //                 bins[0][index].Extend(out);
    //             }
    //         }
    //
    //         u32 indexMinY = binIndices[1];
    //         u32 indexMaxY = binIndices[5];
    //         Lane8F32 maskY(indexMinY == indexMaxY);
    //
    //         bins[1][indexMinY].minU = MaskMin(maskY, bins[1][indexMinY].minU, minY);
    //         bins[1][indexMinY].minV = MaskMin(maskY, bins[1][indexMinY].minV, minZ);
    //         bins[1][indexMinY].minW = MaskMin(maskY, bins[1][indexMinY].minW, minX);
    //
    //         bins[1][indexMinY].maxU = MaskMax(maskY, bins[1][indexMinY].maxU, maxY);
    //         bins[1][indexMinY].maxV = MaskMax(maskY, bins[1][indexMinY].maxV, maxZ);
    //         bins[1][indexMinY].maxW = MaskMax(maskY, bins[1][indexMinY].maxW, maxX);
    //
    //         entryCounts[indexMinY][1] += 1;
    //         exitCounts[indexMaxY][1] += 1;
    //
    //         for (u32 index = indexMinY; index < indexMaxY; index++)
    //         {
    //             faceIndices[1][index][binCounts[1][index]++] = faceID;
    //             if (binCounts[1][index] == LANE_WIDTH)
    //             {
    //                 Bounds8F32 out;
    //                 ClipTriangleTest(mesh, faceIndices[1][index],
    //                                  splitPositions[1][index], splitPositions[1][index + 1], out);
    //                 binCounts[1][index] = 0;
    //                 bins[1][index].Extend(out);
    //             }
    //         }
    //
    //         u32 indexMinZ = binIndices[2];
    //         u32 indexMaxZ = binIndices[6];
    //         Lane8F32 maskZ(indexMinZ == indexMaxZ);
    //
    //         bins[2][indexMinZ].minU = MaskMin(maskZ, bins[2][indexMinZ].minU, minZ);
    //         bins[2][indexMinZ].minV = MaskMin(maskZ, bins[2][indexMinZ].minV, minX);
    //         bins[2][indexMinZ].minW = MaskMin(maskZ, bins[2][indexMinZ].minW, minY);
    //
    //         bins[2][indexMinZ].maxU = MaskMax(maskZ, bins[2][indexMinZ].maxU, maxZ);
    //         bins[2][indexMinZ].maxV = MaskMax(maskZ, bins[2][indexMinZ].maxV, maxX);
    //         bins[2][indexMinZ].maxW = MaskMax(maskZ, bins[2][indexMinZ].maxW, maxY);
    //
    //         entryCounts[indexMinZ][2] += 1;
    //         exitCounts[indexMaxZ][2] += 1;
    //
    //         for (u32 index = indexMinZ; index < indexMaxZ; index++)
    //         {
    //             faceIndices[2][index][binCounts[2][index]++] = faceID;
    //             if (binCounts[2][index] == LANE_WIDTH)
    //             {
    //                 Bounds8F32 out;
    //                 ClipTriangleTest(mesh, faceIndices[2][index],
    //                                  splitPositions[2][index], splitPositions[2][index + 1], out);
    //                 binCounts[2][index] = 0;
    //                 bins[2][index].Extend(out);
    //             }
    //         }
    //     }
    // }

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
        // find greatest multiple of 8 less than the end

        u32 alignedCount = count - count % 8;
        u32 end          = start + count;
        for (; i < start + alignedCount; i += 8)
        {
            Lane8U32 faceIds = Lane8U32::LoadU(soa->primIDs + i);

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

            for (u32 prevIndex = 0; prevIndex < 8; prevIndex++)
            {
                u32 faceID    = faceIDPrev[prevIndex];
                u32 indexMinX = indexMinXPrev[prevIndex];
                u32 indexMaxX = indexMaxXPrev[prevIndex];

                bool noSplit  = indexMinX == indexMaxX;
                Lane8F32 mask = Lane8F32::Mask(noSplit);
                Lane8U32 perm(prevIndex);

                bins[0][indexMinX].minU = MaskMin(mask, bins[0][indexMinX].minU, Shuffle(prevMinX, perm));
                bins[0][indexMinX].minV = MaskMin(mask, bins[0][indexMinX].minV, Shuffle(prevMinY, perm));
                bins[0][indexMinX].minW = MaskMin(mask, bins[0][indexMinX].minW, Shuffle(prevMinZ, perm));

                bins[0][indexMinX].maxU = MaskMax(mask, bins[0][indexMinX].maxU, Shuffle(prevMaxX, perm));
                bins[0][indexMinX].maxV = MaskMax(mask, bins[0][indexMinX].maxV, Shuffle(prevMaxY, perm));
                bins[0][indexMinX].maxW = MaskMax(mask, bins[0][indexMinX].maxW, Shuffle(prevMaxZ, perm));

                entryCounts[indexMinX][0] += 1;
                exitCounts[indexMaxX][0] += 1;

                if (indexMinX != indexMaxX)
                {
                    for (u32 index = indexMinX; index <= indexMaxX; index++)
                    {
                        faceIndices[0][index][binCounts[0][index]++] = faceID;
                        if (binCounts[0][index] == LANE_WIDTH)
                        {
                            Bounds8F32 out;
                            ClipTriangleTest(mesh, faceIndices[0][index], 0,
                                             splitPositions[0][index], splitPositions[0][index + 1], out);
                            binCounts[0][index] = 0;
                            bins[0][index].Extend(out);
                        }
                    }
                }
            }

            // Y
            Lane8U32 binIndexMinY = Clamp(z, e, Flooru((prevMinY - baseY) * scaleY));
            Lane8U32 binIndexMaxY = Clamp(z, e, Flooru((prevMaxY - baseY) * scaleY));

            Lane8U32::Store(indexMinYPrev, binIndexMinY);
            Lane8U32::Store(indexMaxYPrev, binIndexMaxY);

            for (u32 prevIndex = 0; prevIndex < 8; prevIndex++)
            {
                u32 faceID    = faceIDPrev[prevIndex];
                u32 indexMinY = indexMinYPrev[prevIndex];
                u32 indexMaxY = indexMaxYPrev[prevIndex];

                bool noSplit  = indexMinY == indexMaxY;
                Lane8F32 mask = Lane8F32::Mask(noSplit);
                Lane8U32 perm(prevIndex);

                bins[1][indexMinY].minU = MaskMin(mask, bins[1][indexMinY].minU, Shuffle(prevMinY, perm));
                bins[1][indexMinY].minV = MaskMin(mask, bins[1][indexMinY].minV, Shuffle(prevMinZ, perm));
                bins[1][indexMinY].minW = MaskMin(mask, bins[1][indexMinY].minW, Shuffle(prevMinX, perm));

                bins[1][indexMinY].maxU = MaskMax(mask, bins[1][indexMinY].maxU, Shuffle(prevMaxY, perm));
                bins[1][indexMinY].maxV = MaskMax(mask, bins[1][indexMinY].maxV, Shuffle(prevMaxZ, perm));
                bins[1][indexMinY].maxW = MaskMax(mask, bins[1][indexMinY].maxW, Shuffle(prevMaxX, perm));

                entryCounts[indexMinY][1] += 1;
                exitCounts[indexMaxY][1] += 1;

                if (indexMinY != indexMaxY)
                {
                    for (u32 index = indexMinY; index <= indexMaxY; index++)
                    {
                        faceIndices[1][index][binCounts[1][index]++] = faceID;
                        if (binCounts[1][index] == LANE_WIDTH)
                        {
                            Bounds8F32 out;
                            ClipTriangleTest(mesh, faceIndices[1][index], 1,
                                             splitPositions[1][index], splitPositions[1][index + 1], out);
                            binCounts[1][index] = 0;
                            bins[1][index].Extend(out);
                        }
                    }
                }
            }

            // Z
            Lane8U32 binIndexMinZ = Clamp(z, e, Flooru((prevMinZ - baseZ) * scaleZ));
            Lane8U32 binIndexMaxZ = Clamp(z, e, Flooru((prevMaxZ - baseZ) * scaleZ));

            Lane8U32::Store(indexMinZPrev, binIndexMinZ);
            Lane8U32::Store(indexMaxZPrev, binIndexMaxZ);

            for (u32 prevIndex = 0; prevIndex < 8; prevIndex++)
            {
                u32 faceID    = faceIDPrev[prevIndex];
                u32 indexMinZ = indexMinZPrev[prevIndex];
                u32 indexMaxZ = indexMaxZPrev[prevIndex];

                bool noSplit  = indexMinZ == indexMaxZ;
                Lane8F32 mask = Lane8F32::Mask(noSplit);
                Lane8U32 perm(prevIndex);

                bins[2][indexMinZ].minU = MaskMin(mask, bins[2][indexMinZ].minU, Shuffle(prevMinZ, perm));
                bins[2][indexMinZ].minV = MaskMin(mask, bins[2][indexMinZ].minV, Shuffle(prevMinX, perm));
                bins[2][indexMinZ].minW = MaskMin(mask, bins[2][indexMinZ].minW, Shuffle(prevMinY, perm));

                bins[2][indexMinZ].maxU = MaskMax(mask, bins[2][indexMinZ].maxU, Shuffle(prevMaxZ, perm));
                bins[2][indexMinZ].maxV = MaskMax(mask, bins[2][indexMinZ].maxV, Shuffle(prevMaxX, perm));
                bins[2][indexMinZ].maxW = MaskMax(mask, bins[2][indexMinZ].maxW, Shuffle(prevMaxY, perm));

                entryCounts[indexMinZ][2] += 1;
                exitCounts[indexMaxZ][2] += 1;

                if (indexMinZ != indexMaxZ)
                {
                    for (u32 index = indexMinZ; index <= indexMaxZ; index++)
                    {
                        faceIndices[2][index][binCounts[2][index]++] = faceID;
                        if (binCounts[2][index] == LANE_WIDTH)
                        {
                            Bounds8F32 out;
                            ClipTriangleTest(mesh, faceIndices[2][index], 2,
                                             splitPositions[2][index], splitPositions[2][index + 1], out);
                            binCounts[2][index] = 0;
                            bins[2][index].Extend(out);
                        }
                    }
                }
            }
        }
        f32 baseXScalar = baseX[0];
        f32 baseYScalar = baseY[0];
        f32 baseZScalar = baseZ[0];

        f32 scaleXScalar = scaleX[0];
        f32 scaleYScalar = scaleY[0];
        f32 scaleZScalar = scaleZ[0];

        // Add the remaining triangles
        for (; i < end; i++)
        {
            f32 prevMinX = soa->minX[i];
            f32 prevMinY = soa->minY[i];
            f32 prevMinZ = soa->minZ[i];

            f32 prevMaxX = soa->maxX[i];
            f32 prevMaxY = soa->maxY[i];
            f32 prevMaxZ = soa->maxZ[i];

            u32 faceID = soa->primIDs[i];

            // X
            u32 indexMinX = Clamp(0u, numBins - 1u, (u32)Floor((prevMinX - baseXScalar) * scaleXScalar));
            u32 indexMaxX = Clamp(0u, numBins - 1u, (u32)Floor((prevMaxX - baseXScalar) * scaleXScalar));
            bool noSplit  = indexMinX == indexMaxX;
            Lane8F32 mask = Lane8F32::Mask(noSplit);

            bins[0][indexMinX].minU = MaskMin(mask, bins[0][indexMinX].minU, Lane8F32(prevMinX));
            bins[0][indexMinX].minV = MaskMin(mask, bins[0][indexMinX].minV, Lane8F32(prevMinY));
            bins[0][indexMinX].minW = MaskMin(mask, bins[0][indexMinX].minW, Lane8F32(prevMinZ));

            bins[0][indexMinX].maxU = MaskMax(mask, bins[0][indexMinX].maxU, Lane8F32(prevMaxX));
            bins[0][indexMinX].maxV = MaskMax(mask, bins[0][indexMinX].maxV, Lane8F32(prevMaxY));
            bins[0][indexMinX].maxW = MaskMax(mask, bins[0][indexMinX].maxW, Lane8F32(prevMaxZ));

            entryCounts[indexMinX][0] += 1;
            exitCounts[indexMaxX][0] += 1;

            if (indexMinX != indexMaxX)
            {
                for (u32 index = indexMinX; index <= indexMaxX; index++)
                {
                    faceIndices[0][index][binCounts[0][index]++] = faceID;
                    if (binCounts[0][index] == LANE_WIDTH)
                    {
                        Bounds8F32 out;
                        ClipTriangleTest(mesh, faceIndices[0][index], 0,
                                         splitPositions[0][index], splitPositions[0][index + 1], out);
                        binCounts[0][index] = 0;
                        bins[0][index].Extend(out);
                    }
                }
            }

            // Y
            u32 indexMinY  = Clamp(0u, numBins - 1u, (u32)Floor((prevMinY - baseYScalar) * scaleYScalar));
            u32 indexMaxY  = Clamp(0u, numBins - 1u, (u32)Floor((prevMaxY - baseYScalar) * scaleYScalar));
            bool noSplitY  = indexMinY == indexMaxY;
            Lane8F32 maskY = Lane8F32::Mask(noSplitY);

            bins[1][indexMinY].minU = MaskMin(maskY, bins[1][indexMinY].minU, Lane8F32(prevMinY));
            bins[1][indexMinY].minV = MaskMin(maskY, bins[1][indexMinY].minV, Lane8F32(prevMinZ));
            bins[1][indexMinY].minW = MaskMin(maskY, bins[1][indexMinY].minW, Lane8F32(prevMinX));

            bins[1][indexMinY].maxU = MaskMax(maskY, bins[1][indexMinY].maxU, Lane8F32(prevMaxY));
            bins[1][indexMinY].maxV = MaskMax(maskY, bins[1][indexMinY].maxV, Lane8F32(prevMaxZ));
            bins[1][indexMinY].maxW = MaskMax(maskY, bins[1][indexMinY].maxW, Lane8F32(prevMaxX));

            entryCounts[indexMinY][1] += 1;
            exitCounts[indexMaxY][1] += 1;

            if (indexMinY != indexMaxY)
            {
                for (u32 index = indexMinY; index <= indexMaxY; index++)
                {
                    faceIndices[1][index][binCounts[1][index]++] = faceID;
                    if (binCounts[1][index] == LANE_WIDTH)
                    {
                        Bounds8F32 out;
                        ClipTriangleTest(mesh, faceIndices[1][index], 1,
                                         splitPositions[1][index], splitPositions[1][index + 1], out);
                        binCounts[1][index] = 0;
                        bins[1][index].Extend(out);
                    }
                }
            }

            // Z
            u32 indexMinZ = Clamp(0u, numBins - 1u, (u32)Floor((prevMinZ - baseZScalar) * scaleZScalar));
            u32 indexMaxZ = Clamp(0u, numBins - 1u, (u32)Floor((prevMaxZ - baseZScalar) * scaleZScalar));

            bool noSplitZ  = indexMinZ == indexMaxZ;
            Lane8F32 maskZ = Lane8F32::Mask(noSplitZ);

            bins[2][indexMinZ].minU = MaskMin(maskZ, bins[2][indexMinZ].minU, Lane8F32(prevMinZ));
            bins[2][indexMinZ].minV = MaskMin(maskZ, bins[2][indexMinZ].minV, Lane8F32(prevMinX));
            bins[2][indexMinZ].minW = MaskMin(maskZ, bins[2][indexMinZ].minW, Lane8F32(prevMinY));

            bins[2][indexMinZ].maxU = MaskMax(maskZ, bins[2][indexMinZ].maxU, Lane8F32(prevMaxZ));
            bins[2][indexMinZ].maxV = MaskMax(maskZ, bins[2][indexMinZ].maxV, Lane8F32(prevMaxX));
            bins[2][indexMinZ].maxW = MaskMax(maskZ, bins[2][indexMinZ].maxW, Lane8F32(prevMaxY));

            entryCounts[indexMinZ][2] += 1;
            exitCounts[indexMaxZ][2] += 1;

            if (indexMinZ != indexMaxZ)
            {
                for (u32 index = indexMinZ; index <= indexMaxZ; index++)
                {
                    faceIndices[2][index][binCounts[2][index]++] = faceID;
                    if (binCounts[2][index] == LANE_WIDTH)
                    {
                        Bounds8F32 out;
                        ClipTriangleTest(mesh, faceIndices[2][index], 2,
                                         splitPositions[2][index], splitPositions[2][index + 1], out);
                        binCounts[2][index] = 0;
                        bins[2][index].Extend(out);
                    }
                }
            }
        }

        // Empty the bins
        Lane8F32 posInf(pos_inf);
        Lane8F32 negInf(neg_inf);
        for (u32 dim = 0; dim < 3; dim++)
        {
            for (u32 binIndex = 0; binIndex < numBins; binIndex++)
            {
                u32 remainingCount = binCounts[dim][binIndex];
                u32 bitMask        = (1 << remainingCount) - 1;
                Lane8F32 mask      = Lane8F32::Mask(bitMask);
                Bounds8F32 out;
                ClipTriangleTest(mesh, faceIndices[dim][binIndex], dim,
                                 splitPositions[dim][binIndex], splitPositions[dim][binIndex + 1], out);

                out.minU = Select(mask, out.minU, posInf);
                out.minV = Select(mask, out.minV, posInf);
                out.minW = Select(mask, out.minW, posInf);
                out.maxU = Select(mask, out.maxU, negInf);
                out.maxV = Select(mask, out.maxV, negInf);
                out.maxW = Select(mask, out.maxW, negInf);

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

            finalBounds[0][i] = Bounds(xMinP, xMaxP);

            Bounds8F32 &bY = bins[1][i];
            f32 bYMinX     = ReduceMin(bY.minW);
            f32 bYMinY     = ReduceMin(bY.minU);
            f32 bYMinZ     = ReduceMin(bY.minV);

            f32 bYMaxX = ReduceMax(bY.maxW);
            f32 bYMaxY = ReduceMax(bY.maxU);
            f32 bYMaxZ = ReduceMax(bY.maxV);

            Lane4F32 yMinP(bYMinX, bYMinY, bYMinZ, 0.f);
            Lane4F32 yMaxP(bYMaxX, bYMaxY, bYMaxZ, 0.f);

            finalBounds[1][i] = Bounds(yMinP, yMaxP);

            Bounds8F32 &bZ = bins[2][i];
            f32 bZMinX     = ReduceMin(bZ.minV);
            f32 bZMinY     = ReduceMin(bZ.minW);
            f32 bZMinZ     = ReduceMin(bZ.minU);

            f32 bZMaxX = ReduceMax(bZ.maxV);
            f32 bZMaxY = ReduceMax(bZ.maxW);
            f32 bZMaxZ = ReduceMax(bZ.maxU);

            Lane4F32 zMinP(bZMinX, bZMinY, bZMinZ, 0.f);
            Lane4F32 zMaxP(bZMaxX, bZMaxY, bZMaxZ, 0.f);

            finalBounds[2][i] = Bounds(zMinP, zMaxP);
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
    Bounds bins[3][numBins];

    TestSplitBinningBase(Bounds &bounds)
    {
        const Lane4F32 eps = 1e-34f;

        base = bounds.minP;

        const Lane4F32 diag = Max(bounds.maxP - bounds.minP, eps);
        scale               = Select(diag > eps, Lane4F32((f32)numBins) / diag, Lane4F32(0.f));
        invScale            = (diag / (f32)numBins);

        for (u32 i = 0; i < numBins; i++)
        {
            numEnd[i]   = 0;
            numBegin[i] = 0;
        }
    }
    __forceinline void Add(const u32 dim, const u32 beginID, const u32 endID, const u32 binID, const Bounds &b, const u32 n = 1)
    {
        numEnd[endID][dim] += n;
        numBegin[beginID][dim] += n;
        bins[dim][binID].Extend(b);
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
                    ClipTriangleSimple(mesh, rest, prim.PrimID(), dim, pos, left, right);

                    if (left.Empty()) l++;
                    bins[dim][bin].Extend(left);
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
