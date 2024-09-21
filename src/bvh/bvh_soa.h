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
// 1. data is AoS. find the bin index min and max in each dimension, increment the count of the bin and place the
// face index alongside. when the count reaches the max, start working on the 8 triangle/plane test
// 2. data is SoA, same as above. makes finding the bins more efficient (i think), but storing the results to memory
// could be problematic.

// could also find all of the bin indices first and then work on the triangles

__forceinline void ClipTriangleTest(const TriangleMesh *mesh, const u32 faceIndices[8], const f32 leftBound,
                                    const f32 rightBound, Bounds8F32 &out)
{

    Lane8F32 clipLeft(leftBound);
    Lane8F32 clipRight(leftBound);

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

    static const u32 LANE_WIDTH = 8;

    u32 faceIndices[3][numBins][LANE_WIDTH];
    u32 binCounts[3][numBins];
    Bounds8F32 bins[3][numBins];
    f32 splitPositions[3][numBins + 1];

    TestSOASplitBinning(Bounds &bounds)
    {
        const Lane4F32 eps = 1e-34f;

        Lane8F32 minP(bounds.minP, bounds.minP);
        baseX = Permute<0>(minP);
        baseY = Permute<1>(minP);
        baseZ = Permute<2>(minP);

        const Lane4F32 diag = Max(bounds.maxP - bounds.minP, eps);

        Lane4F32 scale = Select(diag > eps, Lane4F32((f32)numBins) / diag, Lane4F32(0.f));

        Lane8F32 scale8(scale, scale);
        scaleX = Permute<0>(scale8);
        scaleY = Permute<1>(scale8);
        scaleZ = Permute<2>(scale8);

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
        for (; i < start + count; i += 8)
        {
            Lane8U32 faceIds = Lane8U32::LoadU(soa->primIDs + i);

            Lane8F32 minX = Lane8F32::LoadU(soa->minX + i);
            Lane8F32 minY = Lane8F32::LoadU(soa->minY + i);
            Lane8F32 minZ = Lane8F32::LoadU(soa->minZ + i);

            Lane8F32 maxX = Lane8F32::LoadU(soa->maxX + i);
            Lane8F32 maxY = Lane8F32::LoadU(soa->maxY + i);
            Lane8F32 maxZ = Lane8F32::LoadU(soa->maxZ + i);

            // X
            Lane8U32 binIndexMinX = Clamp(z, e, Flooru((minX - baseX) * scaleX));
            Lane8U32 binIndexMaxX = Clamp(z, e, Flooru((maxX - baseX) * scaleX));

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

            Lane8U32::Store(indexMinXPrev, binIndexMinX);
            Lane8U32::Store(indexMaxXPrev, binIndexMaxX);

            // Y
            Lane8U32 binIndexMinY = Clamp(z, e, Flooru((minY - baseY) * scaleY));
            Lane8U32 binIndexMaxY = Clamp(z, e, Flooru((maxY - baseY) * scaleY));

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

            Lane8U32::Store(indexMinYPrev, binIndexMinY);
            Lane8U32::Store(indexMaxYPrev, binIndexMaxY);

            // Z
            Lane8U32 binIndexMinZ = Clamp(z, e, Flooru((minZ - baseZ) * scaleZ));
            Lane8U32 binIndexMaxZ = Clamp(z, e, Flooru((maxZ - baseZ) * scaleZ));

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

            Lane8U32::Store(indexMinZPrev, binIndexMinZ);
            Lane8U32::Store(indexMaxZPrev, binIndexMaxZ);

            Lane8U32::Store(faceIDPrev, faceIds);

            prevMinX = minX;
            prevMinY = minY;
            prevMinZ = minZ;

            prevMaxX = maxX;
            prevMaxY = maxY;
            prevMaxZ = maxZ;
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

} // namespace rt

#endif
