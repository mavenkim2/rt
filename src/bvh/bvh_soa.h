#ifndef BVH_SOA_H
#define BVH_SOA_H
namespace rt
{
struct Bounds8
{
    Lane8F32 v;

    Bounds8() : v(neg_inf) {}
    Bounds8(EmptyTy) : v(neg_inf) {}
    Bounds8(PosInfTy) : v(pos_inf) {}
    __forceinline Bounds8(Lane8F32 in)
    {
        v = in ^ signFlipMask;
    }

    __forceinline operator const __m256 &() const { return v; }
    __forceinline operator __m256 &() { return v; }

    __forceinline explicit operator Bounds() const
    {
        Bounds out;
        out.minP = FlipSign(Extract4<0>(v));
        out.maxP = Extract4<1>(v);
        return out;
    }

    __forceinline void Extend(const Lane8F32 &other)
    {
        v = Max(v, other.v);
    }
    __forceinline void Extend(const Bounds8 &other)
    {
        v = Max(v, other.v);
    }
    __forceinline void Intersect(const Bounds8 &other)
    {
        v = Min(v, other.v);
    }
};

struct Triangle8
{
    union
    {
        struct
        {
            Lane8F32 v0u;
            Lane8F32 v0v;
            Lane8F32 v0w;

            Lane8F32 v1u;
            Lane8F32 v1v;
            Lane8F32 v1w;

            Lane8F32 v2u;
            Lane8F32 v2v;
            Lane8F32 v2w;
        };
        Lane8F32 v[9];
    };
    Triangle8() {}
    Triangle8(const Triangle8 &other) : v0u(other.v0u), v0v(other.v0v), v0w(other.v0w),
                                        v1u(other.v1u), v1v(other.v1v), v1w(other.v1w),
                                        v2u(other.v2u), v2v(other.v2v), v2w(other.v2w) {}

    __forceinline const Lane8F32 &operator[](i32 i) const
    {
        Assert(i < 9);
        return v[i];
    }
    __forceinline Lane8F32 &operator[](i32 i)
    {
        Assert(i < 9);
        return v[i];
    }

    static Triangle8 Load(TriangleMesh *mesh, const u32 dim, const u32 faceIndices[8])
    {
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

        // primBounds[0] = Lane8F32(Min(Min(v0a, v1a), v2a), Max(Max(v0a, v1a), v2a));
        // primBounds[1] = Lane8F32(Min(Min(v0b, v1b), v2b), Max(Max(v0b, v1b), v2b));
        // primBounds[2] = Lane8F32(Min(Min(v0c, v1c), v2c), Max(Max(v0c, v1c), v2c));
        // primBounds[3] = Lane8F32(Min(Min(v0d, v1d), v2d), Max(Max(v0d, v1d), v2d));

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

        // primBounds[4] = Lane8F32(Min(Min(v0a, v1a), v2a), Max(Max(v0a, v1a), v2a));
        // primBounds[5] = Lane8F32(Min(Min(v0b, v1b), v2b), Max(Max(v0b, v1b), v2b));
        // primBounds[6] = Lane8F32(Min(Min(v0c, v1c), v2c), Max(Max(v0c, v1c), v2c));
        // primBounds[7] = Lane8F32(Min(Min(v0d, v1d), v2d), Max(Max(v0d, v1d), v2d));

        Vec3lf4 p3;
        Vec3lf4 p4;
        Vec3lf4 p5;

        Transpose4x3(v0a, v0b, v0c, v0d, p3.x, p3.y, p3.z);
        Transpose4x3(v1a, v1b, v1c, v1d, p4.x, p4.y, p4.z);
        Transpose4x3(v2a, v2b, v2c, v2d, p5.x, p5.y, p5.z);

        u32 v = LUTAxis[dim];
        u32 w = LUTAxis[v];

        Triangle8 out;
        out.v0u = Lane8F32(p0[dim], p3[dim]);
        out.v1u = Lane8F32(p1[dim], p4[dim]);
        out.v2u = Lane8F32(p2[dim], p5[dim]);

        out.v0v = Lane8F32(p0[v], p3[v]);
        out.v1v = Lane8F32(p1[v], p4[v]);
        out.v2v = Lane8F32(p2[v], p5[v]);

        out.v0w = Lane8F32(p0[w], p3[w]);
        out.v1w = Lane8F32(p1[w], p4[w]);
        out.v2w = Lane8F32(p2[w], p5[w]);

        return out;
    }
};

void ClipTriangle(const TriangleMesh *mesh, const u32 dim, const Triangle8 &tri, const Lane8F32 &splitPos,
                  Lane8F32 &leftMinX, Lane8F32 &leftMinY, Lane8F32 &leftMinZ,
                  Lane8F32 &leftMaxX, Lane8F32 &leftMaxY, Lane8F32 &leftMaxZ,
                  Lane8F32 &rightMinX, Lane8F32 &rightMinY, Lane8F32 &rightMinZ,
                  Lane8F32 &rightMaxX, Lane8F32 &rightMaxY, Lane8F32 &rightMaxZ)
{
    static const u32 LUTX[] = {0, 2, 1};
    static const u32 LUTY[] = {1, 0, 2};
    static const u32 LUTZ[] = {2, 1, 0};

    // Lane8F32 clipMasksL[] = {
    //     tri.v0u <= splitPos,
    //     tri.v1u <= splitPos,
    //     tri.v2u <= splitPos,
    // };
    //
    // Lane8F32 clipMasksR[] = {
    //     tri.v0u >= splitPos,
    //     tri.v1u >= splitPos,
    //     tri.v2u >= splitPos,
    // };

    Bounds8F32 left;
    left.maxU = splitPos;
    Bounds8F32 right;
    right.minU = splitPos;

    u32 first = 0;
    u32 next  = LUTAxis[first];
    for (u32 edgeIndex = 0; edgeIndex < 3; edgeIndex++)
    {
        const u32 v0IndexStart = 3 * first;
        const u32 v1IndexStart = 3 * next;

        const Lane8F32 &v0u = tri[v0IndexStart];
        const Lane8F32 &v1u = tri[v1IndexStart];

        const Lane8F32 &v0v = tri[v0IndexStart + 1];
        const Lane8F32 &v1v = tri[v1IndexStart + 1];

        const Lane8F32 &v0w = tri[v0IndexStart + 2];
        const Lane8F32 &v1w = tri[v1IndexStart + 2];

        // const Lane8F32 &clipMaskL = clipMasksL[first];

        left.MaskExtendL(v0u <= splitPos, v0u, v0v, v0w);
        right.MaskExtendR(v0u >= splitPos, v0u, v0v, v0w);

        // const Lane8F32 div = Select(v1u == v0u, Lane8F32(zero), Rcp(v1u - v0u));
        const Lane8F32 t = (splitPos - v0u) / (v1u - v0u); //* div;

        const Lane8F32 subV = v1v - v0v;
        const Lane8F32 subW = v1w - v0w;

        const Lane8F32 clippedV = FMA(t, subV, v0v);
        const Lane8F32 clippedW = FMA(t, subW, v0w);

        const Lane8F32 edgeIsClipped = ((v0u < splitPos) & (v1u > splitPos)) | ((v0u > splitPos) & (v1u < splitPos));

        left.MaskExtendVW(edgeIsClipped, clippedV, clippedW);
        right.MaskExtendVW(edgeIsClipped, clippedV, clippedW);

        first = next;
        next  = LUTAxis[next];
    }

    leftMinX = *((Lane8F32 *)(&left) + LUTX[dim]);
    leftMinY = *((Lane8F32 *)(&left) + LUTY[dim]);
    leftMinZ = *((Lane8F32 *)(&left) + LUTZ[dim]);

    leftMaxX = *((Lane8F32 *)(&left) + 3 + LUTX[dim]);
    leftMaxY = *((Lane8F32 *)(&left) + 3 + LUTY[dim]);
    leftMaxZ = *((Lane8F32 *)(&left) + 3 + LUTZ[dim]);

    rightMinX = *((Lane8F32 *)(&right) + LUTX[dim]);
    rightMinY = *((Lane8F32 *)(&right) + LUTY[dim]);
    rightMinZ = *((Lane8F32 *)(&right) + LUTZ[dim]);

    rightMaxX = *((Lane8F32 *)(&right) + 3 + LUTX[dim]);
    rightMaxY = *((Lane8F32 *)(&right) + 3 + LUTY[dim]);
    rightMaxZ = *((Lane8F32 *)(&right) + 3 + LUTZ[dim]);
}

void ClipTriangle(const TriangleMesh *mesh, const u32 dim, const Triangle8 &tri, const Lane8F32 &splitPos,
                  Bounds8 *l, Bounds8 *r)
{
    Lane8F32 lOut[8];
    Lane8F32 rOut[8];
    Lane8F32 leftMinX, leftMinY, leftMinZ, leftMaxX, leftMaxY, leftMaxZ,
        rightMinX, rightMinY, rightMinZ, rightMaxX, rightMaxY, rightMaxZ;

    ClipTriangle(mesh, dim, tri, splitPos,
                 leftMinX, leftMinY, leftMinZ,
                 leftMaxX, leftMaxY, leftMaxZ,
                 rightMinX, rightMinY, rightMinZ,
                 rightMaxX, rightMaxY, rightMaxZ);

    Transpose6x8(leftMinX, leftMinY, leftMinZ, leftMaxX, leftMaxY, leftMaxZ,
                 lOut[0], lOut[1], lOut[2], lOut[3], lOut[4], lOut[5], lOut[6], lOut[7]);
    Transpose6x8(rightMinX, rightMinY, rightMinZ, rightMaxX, rightMaxY, rightMaxZ,
                 rOut[0], rOut[1], rOut[2], rOut[3], rOut[4], rOut[5], rOut[6], rOut[7]);

    for (u32 i = 0; i < 8; i++)
    {
        Bounds8 lBounds8(lOut[i]);
        Bounds8 rBounds8(rOut[i]);

        lBounds8.Intersect(l[i]);
        rBounds8.Intersect(l[i]);

        l[i] = lBounds8;
        r[i] = rBounds8;
    }
}

void ClipTriangle(const TriangleMesh *mesh, const u32 dim, const Triangle8 &tri, const Lane8F32 &splitPos,
                  Bounds8 *l, Bounds8 *r, Bounds8 *outLGeomBounds, Bounds8 *outRGeomBounds, Lane8F32 *outCentroidBounds)
{
    Lane8F32 lOut[8];
    Lane8F32 rOut[8];
    Lane8F32 leftMinX, leftMinY, leftMinZ, leftMaxX, leftMaxY, leftMaxZ,
        rightMinX, rightMinY, rightMinZ, rightMaxX, rightMaxY, rightMaxZ;

    ClipTriangle(mesh, dim, tri, splitPos,
                 leftMinX, leftMinY, leftMinZ,
                 leftMaxX, leftMaxY, leftMaxZ,
                 rightMinX, rightMinY, rightMinZ,
                 rightMaxX, rightMaxY, rightMaxZ);

    Transpose6x8(leftMinX, leftMinY, leftMinZ, leftMaxX, leftMaxY, leftMaxZ,
                 lOut[0], lOut[1], lOut[2], lOut[3], lOut[4], lOut[5], lOut[6], lOut[7]);
    Transpose6x8(rightMinX, rightMinY, rightMinZ, rightMaxX, rightMaxY, rightMaxZ,
                 rOut[0], rOut[1], rOut[2], rOut[3], rOut[4], rOut[5], rOut[6], rOut[7]);

    Lane8F32 centroidLeftX = (leftMaxX + leftMinX) * 0.5f;
    Lane8F32 centroidLeftY = (leftMaxY + leftMinY) * 0.5f;
    Lane8F32 centroidLeftZ = (leftMaxZ + leftMinZ) * 0.5f;

    Lane8F32 centroidRightX = (rightMaxX + rightMinX) * 0.5f;
    Lane8F32 centroidRightY = (rightMaxY + rightMinY) * 0.5f;
    Lane8F32 centroidRightZ = (rightMaxZ + rightMinZ) * 0.5f;

    Transpose6x8(centroidLeftX, centroidLeftY, centroidLeftZ, centroidRightX, centroidRightY, centroidRightZ,
                 outCentroidBounds[0], outCentroidBounds[1], outCentroidBounds[2], outCentroidBounds[3],
                 outCentroidBounds[4], outCentroidBounds[5], outCentroidBounds[6], outCentroidBounds[7]);

    for (u32 i = 0; i < 8; i++)
    {
        Bounds8 lBounds8(lOut[i]);
        Bounds8 rBounds8(rOut[i]);

        outLGeomBounds[i] = lBounds8;
        outRGeomBounds[i] = rBounds8;

        lBounds8.Intersect(l[i]);
        rBounds8.Intersect(l[i]);

        l[i] = lBounds8;
        r[i] = rBounds8;
    }
}

template <i32 numBins = 32>
struct ObjectBinner
{
    Lane8F32 base[3];
    Lane8F32 scale[3];

    ObjectBinner(const Bounds &centroidBounds)
    {
        const f32 eps = 1e-34f;
        Lane8F32 minP(centroidBounds.minP);
        base[0] = Shuffle<0>(minP);
        base[1] = Shuffle<1>(minP);
        base[2] = Shuffle<2>(minP);

        const Lane4F32 diag = Max(centroidBounds.maxP - centroidBounds.minP, eps);
        Lane4F32 scale4     = Select(diag > eps, Lane4F32((f32)numBins) / diag, Lane4F32(0.f));

        Lane8F32 scale8(scale4);
        scale[0] = Shuffle<0>(scale8);
        scale[1] = Shuffle<1>(scale8);
        scale[2] = Shuffle<2>(scale8);
    }
    Lane8U32 Bin(const Lane8F32 &in, const u32 dim) const
    {
        Lane8F32 result = (in - base[dim]) * scale[dim];
        return Clamp(Lane8U32(zero), Lane8U32(numBins - 1), Flooru(result));
    }
    u32 Bin(const f32 in, const u32 dim) const
    {
        Lane8U32 result = Bin(Lane8F32(in), dim);
        return result[0];
    }
    f32 GetSplitValue(u32 pos, u32 dim) const
    {
        f32 invScale = scale[dim][0] == 0.f ? 0.f : 1 / scale[dim][0];
        return (pos * invScale) + base[dim][0];
    }
};

template <i32 numBins = 32>
struct HeuristicSOAObjectBinning
{
    Bounds8 bins[3][numBins];

    Lane4U32 counts[numBins];
    Bounds finalBounds[3][numBins];
    ObjectBinner<numBins> *binner;

    HeuristicSOAObjectBinning(ObjectBinner<numBins> *binner) : binner(binner)
    {
        for (u32 dim = 0; dim < 3; dim++)
        {
            for (u32 i = 0; i < numBins; i++)
            {
                bins[dim][i]        = Bounds8();
                finalBounds[dim][i] = Bounds();
            }
        }
        for (u32 i = 0; i < numBins; i++)
        {
            counts[i] = 0;
        }
    }

    void Bin(PrimDataSOA *data, u32 start, u32 count)
    {
        u32 alignedCount = count - count % LANE_WIDTH;
        u32 i            = start;
        Lane8F32 lanes[8];
        for (; i < start + alignedCount; i += LANE_WIDTH)
        {
            Lane8F32 mins[] = {
                Lane8F32::LoadU(data->minX + i),
                Lane8F32::LoadU(data->minY + i),
                Lane8F32::LoadU(data->minZ + i),
            };
            Lane8F32 maxs[] = {
                Lane8F32::LoadU(data->maxX + i),
                Lane8F32::LoadU(data->maxY + i),
                Lane8F32::LoadU(data->maxZ + i),
            };
            Lane8F32 centroids[] = {
                (maxs[0] - mins[0]) * 0.5f,
                (maxs[1] - mins[1]) * 0.5f,
                (maxs[2] - mins[2]) * 0.5f,
            };
            Lane8U32 binIndices[] = {binner->Bin(centroids[0], 0), binner->Bin(centroids[1], 1), binner->Bin(centroids[2], 2)};

            Transpose6x8(mins[0], mins[1], mins[2], maxs[0], maxs[1], maxs[2],
                         lanes[0], lanes[1], lanes[2], lanes[3], lanes[4], lanes[5], lanes[6], lanes[7]);

            for (u32 dim = 0; dim < 3; dim++)
            {
                alignas(32) u32 binDimIndices[LANE_WIDTH];
                Lane8U32::Store(binDimIndices, binIndices[dim]);
                for (u32 b = 0; b < LANE_WIDTH; b++)
                {
                    if (Any(lanes[b] == pos_inf))
                    {
                        int stop = 5;
                        Assert(false);
                    }
                    u32 bin = binDimIndices[b];
                    bins[dim][bin].Extend(lanes[b]);
                    counts[bin][dim]++;
                }
            }
        }
        for (; i < start + count; i++)
        {
            f32 min[3] = {
                data->minX[i],
                data->minY[i],
                data->minZ[i],
            };
            f32 max[3] = {
                data->maxX[i],
                data->maxY[i],
                data->maxZ[i],
            };
            f32 centroids[3] = {
                (max[0] - min[0]) * 0.5f,
                (max[1] - min[1]) * 0.5f,
                (max[2] - min[2]) * 0.5f,
            };
            Lane8F32 lane(min[0], min[1], min[2], pos_inf, max[0], max[1], max[2], pos_inf);
            for (u32 dim = 0; dim < 3; dim++)
            {
                u32 bin = binner->Bin(centroids[dim], dim);
                bins[dim][bin].Extend(lane);
                counts[bin][dim]++;
            }
        }

        for (u32 dim = 0; dim < 3; dim++)
        {
            for (i = 0; i < numBins; i++)
            {
                finalBounds[dim][i] = Bounds(bins[dim][i]);
            }
        }
    }

    void Merge(const HeuristicSOAObjectBinning<numBins> &other)
    {
        for (u32 i = 0; i < numBins; i++)
        {
            counts[i] += other.counts[i];
            for (u32 dim = 0; dim < 3; dim++)
            {
                finalBounds[dim][i].Extend(other.finalBounds[dim][i]);
            }
        }
    }
};

template <i32 numBins = 16>
struct SplitBinner
{
    Lane8F32 base[3];
    Lane8F32 invScale[3];
    Lane8F32 scale[3];
    Lane8F32 scaleNegArr[3];

    SplitBinner(const Bounds &bounds)
    {
        const Lane4F32 eps = 1e-34f;

        Lane8F32 minP(bounds.minP);
        base[0] = Shuffle<0>(minP);
        base[1] = Shuffle<1>(minP);
        base[2] = Shuffle<2>(minP);

        const Lane4F32 diag = Max(bounds.maxP - bounds.minP, eps);

        Lane4F32 scale4 = Select(diag > eps, Lane4F32((f32)numBins) / diag, Lane4F32(0.f));

        Lane8F32 scale8(scale4);
        scale[0] = Shuffle<0>(scale8);
        scale[1] = Shuffle<1>(scale8);
        scale[2] = Shuffle<2>(scale8);

        // test
        Lane4F32 invScale4 = Select(scale4 == 0.f, 0.f, 1.f / scale4);
        Lane8F32 invScale8(invScale4);
        invScale[0] = Shuffle<0>(invScale8);
        invScale[1] = Shuffle<1>(invScale8);
        invScale[2] = Shuffle<2>(invScale8);

        scaleNegArr[0] = FlipSign(scale[0]);
        scaleNegArr[1] = FlipSign(scale[1]);
        scaleNegArr[2] = FlipSign(scale[2]);
    };
    __forceinline Lane8U32 BinMin(const Lane8F32 &min, const u32 dim) const
    {
        return Clamp(Lane8U32(zero), Lane8U32(numBins - 1), Flooru((base[dim] + min) * scaleNegArr[dim]));
    }
    __forceinline u32 BinMin(const f32 min, const u32 dim) const
    {
        Lane8U32 result = BinMin(Lane8F32(min), dim);
        return result[0];
    }
    __forceinline Lane8U32 BinMax(const Lane8F32 &max, const u32 dim) const
    {
        return Clamp(Lane8U32(zero), Lane8U32(numBins - 1), Flooru((max - base[dim]) * scale[dim]));
    }
    __forceinline u32 BinMax(const f32 max, const u32 dim) const
    {
        Lane8U32 result = BinMax(Lane8F32(max), dim);
        return result[0];
    }
    __forceinline Lane8F32 GetSplitValue(const Lane8U32 &bins, u32 dim) const
    {
        return Lane8F32(bins) * invScale[dim] + base[dim];
    }
    // NOTE: scalar and AVX floating point computations produce different results, so it's important that this uses AVX.
    __forceinline f32 GetSplitValue(u32 bin, u32 dim) const
    {
        Lane8F32 result = GetSplitValue(Lane8U32(bin), dim);
        return result[0];
    }
};

template <i32 numBins = 16>
struct alignas(32) HeuristicSOASplitBinning
{
    Bounds8 bins8[3][numBins];

    // result data
    Lane4U32 entryCounts[numBins];
    Lane4U32 exitCounts[numBins];

    Bounds finalBounds[3][numBins];

    SplitBinner<numBins> *binner;

    HeuristicSOASplitBinning(SplitBinner<numBins> *binner) : binner(binner)
    {
        for (u32 dim = 0; dim < 3; dim++)
        {
            for (u32 i = 0; i < numBins; i++)
            {
                finalBounds[dim][i] = Bounds();
                bins8[dim][i]       = Bounds8();
            }
        }
        for (u32 i = 0; i < numBins; i++)
        {
            entryCounts[i] = 0;
            exitCounts[i]  = 0;
        }
    }

    void Bin(TriangleMesh *mesh, PrimDataSOA *soa, u32 start, u32 count)
    {
        u32 binCounts[3][numBins]                                 = {};
        alignas(32) u32 binIndexStart[3][numBins][2 * LANE_WIDTH] = {};
        u32 faceIndices[3][numBins][2 * LANE_WIDTH]               = {};

        u32 i            = start;
        u32 alignedCount = count - count % LANE_WIDTH;
        f32 totalTime    = 0.f;

        Lane8F32 lanes[8];

        for (; i < start + alignedCount; i += LANE_WIDTH)
        {
            u32 *faceIDs = soa->primIDs + i;

            Lane8F32 prevMin[] = {
                Lane8F32::LoadU(soa->minX + i),
                Lane8F32::LoadU(soa->minY + i),
                Lane8F32::LoadU(soa->minZ + i),
            };

            Lane8F32 prevMax[] = {
                Lane8F32::LoadU(soa->maxX + i),
                Lane8F32::LoadU(soa->maxY + i),
                Lane8F32::LoadU(soa->maxZ + i),
            };

            Lane8U32 indexMinArr[3] = {
                binner->BinMin(prevMin[0], 0),
                binner->BinMin(prevMin[1], 1),
                binner->BinMin(prevMin[2], 2),
            };
            Lane8U32 indexMaxArr[3] = {
                binner->BinMax(prevMax[0], 0),
                binner->BinMax(prevMax[1], 1),
                binner->BinMax(prevMax[2], 2),
            };

            Lane8U32 binDiffX = indexMaxArr[0] - indexMinArr[0];
            Lane8U32 binDiffY = indexMaxArr[1] - indexMinArr[1];
            Lane8U32 binDiffZ = indexMaxArr[2] - indexMinArr[2];
            // x0 x1 x2 x3 y0 y1 y2 y3 x4 x5 x6 x7 y4 y5 y6 y7
            Lane8U32 out0 = PackU32(binDiffX, binDiffY);
            // x0 x1 x2 x3 |  y0 y1 y2 y3 |  z0 00 z1 00 |  z2 00 z3 00 |  x4 x5 x6 x7 |  y4 y5 y6 y7 |  z4 00 z5 00 | z6 00 z7 00

            alignas(32) u8 bytes[32];
            Lane8U32 out1 = PackU16(out0, binDiffZ);
            Lane8U32::Store(bytes, out1);

            u32 bitMask[3] = {};

            static const u32 order[3][LANE_WIDTH] = {
                {0, 1, 2, 3, 16, 17, 18, 19},
                {4, 5, 6, 7, 20, 21, 22, 23},
                {8, 10, 12, 14, 24, 26, 28, 30},
            };
            Transpose6x8(prevMin[0], prevMin[1], prevMin[2], prevMax[0], prevMax[1], prevMax[2],
                         lanes[0], lanes[1], lanes[2], lanes[3], lanes[4], lanes[5], lanes[6], lanes[7]);

            for (u32 dim = 0; dim < 3; dim++)
            {
                alignas(32) u32 indexMins[8];
                Lane8U32::Store(indexMins, indexMinArr[dim]);
                for (u32 b = 0; b < LANE_WIDTH; b++)
                {
                    u32 diff     = bytes[order[dim][b]];
                    u32 indexMin = indexMins[b];
                    entryCounts[indexMin][dim] += 1;
                    exitCounts[indexMin + diff][dim] += 1;

                    switch (diff)
                    {
                        case 0:
                        {
                            bins8[dim][indexMin].Extend(lanes[b]);
                        }
                        break;
                        default:
                        {
                            bitMask[dim] |= (1 << diff);
                            faceIndices[dim][diff][binCounts[dim][diff]]   = faceIDs[b];
                            binIndexStart[dim][diff][binCounts[dim][diff]] = indexMin;
                            binCounts[dim][diff]++;
                        }
                    }
                }
            }
            for (u32 dim = 0; dim < 3; dim++)
            {
                u32 numIters = PopCount(bitMask[dim]);
                for (u32 iter = 0; iter < numIters; iter++)
                {
                    u32 bin = Bsf(bitMask[dim]);
                    if (binCounts[dim][bin] >= LANE_WIDTH)
                    {
                        Assert(binCounts[dim][bin] <= ArrayLength(binIndexStart[0][0]));
                        binCounts[dim][bin] -= LANE_WIDTH;
                        u32 binCount = binCounts[dim][bin];

                        Bounds8 bounds[2][LANE_WIDTH];
                        Triangle8 tri     = Triangle8::Load(mesh, dim, faceIndices[dim][bin] + binCount); //, bounds[0]);
                        Lane8U32 startBin = Lane8U32::LoadU(binIndexStart[dim][bin] + binCount);

                        for (u32 boundIndex = 0; boundIndex < LANE_WIDTH; boundIndex++)
                        {
                            bounds[0][boundIndex] = Bounds8(pos_inf);
                        }
                        alignas(32) u32 binIndices[LANE_WIDTH];

                        u32 current = 0;
                        for (u32 d = 0; d < bin; d++)
                        {
                            Lane8U32::Store(binIndices, startBin);
                            startBin += 1u;
                            Lane8F32 splitPos = binner->GetSplitValue(startBin, dim);

                            ClipTriangle(mesh, dim, tri, splitPos, bounds[current], bounds[!current]);

                            for (u32 b = 0; b < LANE_WIDTH; b++)
                            {
                                u32 binIndex = binIndices[b];
                                bins8[dim][binIndex].Extend(bounds[current][b]);
                            }
                            current = !current;
                        }
                        for (u32 b = 0; b < LANE_WIDTH; b++)
                        {
                            u32 binIndex = binIndices[b] + 1;
                            bins8[dim][binIndex].Extend(bounds[current][b]);
                        }
                        binCounts[dim][bin] = 0;
                    }
                    bitMask[dim] &= bitMask[dim] - 1;
                }
            }
        }
        // Finish the remaining primitives
        for (; i < start + count; i++)
        {
            f32 min[3] = {
                soa->minX[i],
                soa->minY[i],
                soa->minZ[i],
            };

            f32 max[3] = {
                soa->maxX[i],
                soa->maxY[i],
                soa->maxZ[i],
            };
            f32 centroids[3] = {
                (max[0] - min[0]) * 0.5f,
                (max[1] - min[1]) * 0.5f,
                (max[2] - min[2]) * 0.5f,
            };
            u32 faceID = soa->primIDs[i];

            Lane8F32 prim(min[0], min[1], min[2], pos_inf, max[0], max[1], max[2], pos_inf);

            for (u32 dim = 0; dim < 3; dim++)
            {
                u32 binIndexMin = binner->BinMin(min[dim], dim);
                u32 binIndexMax = binner->BinMax(max[dim], dim);

                Assert(binIndexMax >= binIndexMin);

                u32 diff = binIndexMax - binIndexMin;
                entryCounts[binIndexMin][dim] += 1;
                exitCounts[binIndexMax][dim] += 1;
                switch (diff)
                {
                    case 0:
                    {
                        bins8[dim][binIndexMin].Extend(prim);
                    }
                    break;
                    default:
                    {
                        faceIndices[dim][diff][binCounts[dim][diff]]   = faceID;
                        binIndexStart[dim][diff][binCounts[dim][diff]] = binIndexMin;
                        binCounts[dim][diff]++;
                    }
                }
            }
        }
        // Empty the bins
        Lane8F32 posInf(pos_inf);
        Lane8F32 negInf(neg_inf);
        for (u32 dim = 0; dim < 3; dim++)
        {
            for (u32 diff = 1; diff < numBins; diff++)
            {
                u32 remainingCount = binCounts[dim][diff];
                Assert(remainingCount <= ArrayLength(binIndexStart[0][0]));

                const u32 numIters = ((remainingCount + 7) >> 3);
                for (u32 remaining = 0; remaining < numIters; remaining++)
                {
                    u32 numPrims = Min(remainingCount, 8u);
                    Bounds8 bounds[2][8];
                    for (u32 boundIndex = 0; boundIndex < LANE_WIDTH; boundIndex++)
                    {
                        bounds[0][boundIndex] = Bounds8(pos_inf);
                    }
                    Triangle8 tri     = Triangle8::Load(mesh, dim, faceIndices[dim][diff] + remaining * LANE_WIDTH); //, bounds[0]);
                    Lane8U32 startBin = Lane8U32::LoadU(binIndexStart[dim][diff] + remaining * LANE_WIDTH);

                    alignas(32) u32 binIndices[8];

                    u32 current = 0;
                    for (u32 d = 0; d < diff; d++)
                    {
                        Lane8U32::Store(binIndices, startBin);
                        startBin += 1u;
                        Lane8F32 splitPos = binner->GetSplitValue(startBin, dim);
                        ClipTriangle(mesh, dim, tri, splitPos, bounds[current], bounds[!current]);
                        for (u32 b = 0; b < numPrims; b++)
                        {
                            u32 binIndex = binIndices[b];
                            bins8[dim][binIndex].Extend(bounds[current][b]);
                        }
                        current = !current;
                    }
                    for (u32 b = 0; b < numPrims; b++)
                    {
                        u32 binIndex = binIndices[b] + 1;
                        bins8[dim][binIndex].Extend(bounds[current][b]);
                    }
                    remainingCount -= LANE_WIDTH;
                }
            }
        }

        for (u32 dim = 0; dim < 3; dim++)
        {
            for (i = 0; i < numBins; i++)
            {
                finalBounds[dim][i] = Bounds(bins8[dim][i]);
            }
        }
    }

    u32 Split(Arena *arena, TriangleMesh *mesh, PrimDataSOA *data, ExtRange range, Split split,
              RecordSOASplits &outLeft, RecordSOASplits &outRight, u32 *outSplitCount = 0)
    {
        // partitioning
        u32 dim          = split.bestDim;
        f32 *minStream   = data->arr[dim];
        f32 *maxStream   = data->arr[dim + 4];
        u32 alignedCount = range.count - range.count % LANE_WIDTH;

        u32 count = 0;

        u32 refIDQueue[LANE_WIDTH * 2]  = {};
        u32 faceIDQueue[LANE_WIDTH * 2] = {};

        u32 splitCount      = 0;
        u32 totalSplitCount = 0;
        u32 splitBegin      = range.End();
        u32 splitMax        = range.ExtSize();
        PrimDataSOA out;
        out.minX         = data->minX + splitBegin;
        out.minY         = data->minY + splitBegin;
        out.minZ         = data->minZ + splitBegin;
        out.geomIDs      = data->geomIDs + splitBegin;
        out.maxX         = data->maxX + splitBegin;
        out.maxY         = data->maxY + splitBegin;
        out.maxZ         = data->maxZ + splitBegin;
        out.primIDs      = data->primIDs + splitBegin;
        f32 test         = 0.f;
        f32 negBestValue = -split.bestValue;
        u32 i            = range.start;
        for (; i < range.start + alignedCount; i += LANE_WIDTH)
        {
            Lane8F32 min       = Lane8F32::LoadU(minStream + i);
            Lane8F32 max       = Lane8F32::LoadU(maxStream + i);
            Lane8U32 faceIDs   = Lane8U32::LoadU(data->primIDs + i);
            Lane8U32 refIDs    = Lane8U32::Step(i);
            Lane8F32 splitMask = (min > negBestValue & max > split.bestValue);
            u32 mask           = Movemask(splitMask);
            Lane8U32::StoreU(refIDQueue + count, MaskCompress(mask, refIDs));
            Lane8U32::StoreU(faceIDQueue + count, MaskCompress(mask, faceIDs));
            count += PopCount(mask);

            if (count >= LANE_WIDTH)
            {
                Assert(count <= ArrayLength(refIDQueue));
                count -= LANE_WIDTH;
                // Lane8U32::Store((int *)faceIDQueue, _mm256_i32gather_epi32((int *)range.data->primIDs,
                //                                                            Lane8U32::LoadU(refIDQueue + count), 4));
                // for (u32 refIDIndex = 0; refIDIndex < 8; refIDIndex++)
                // {
                //     faceIDQueue[refIDIndex] = range.data->primIDs[refIDQueue[refIDIndex + count]];
                // }
                splitCount += LANE_WIDTH;
                totalSplitCount += LANE_WIDTH;
                Assert(splitCount <= splitMax);
                Bounds8 boundsLeft[LANE_WIDTH];
                for (u32 b = 0; b < LANE_WIDTH; b++)
                {
                    boundsLeft[b] = pos_inf;
                }
                Triangle8 tri = Triangle8::Load(mesh, dim, faceIDQueue + count); //, boundsLeft);
                Bounds8 boundsRight[LANE_WIDTH];
                ClipTriangle(mesh, dim, tri, split.bestValue, boundsLeft, boundsRight);
                for (u32 queueIndex = 0; queueIndex < LANE_WIDTH; queueIndex++)
                {
                    const u32 refID = refIDQueue[count + queueIndex];

                    alignas(32) f32 valuesLeft[8];
                    alignas(32) f32 valuesRight[8];
                    Lane8F32::Store(valuesLeft, boundsLeft[queueIndex].v);
                    Lane8F32::Store(valuesRight, boundsRight[queueIndex].v);

                    data->minX[refID] = valuesLeft[0];
                    data->minY[refID] = valuesLeft[1];
                    data->minZ[refID] = valuesLeft[2];
                    data->maxX[refID] = valuesLeft[4];
                    data->maxY[refID] = valuesLeft[5];
                    data->maxZ[refID] = valuesLeft[6];
                    data->geomIDs[refID]++;

                    out.minX[splitCount - 8 + queueIndex]    = valuesRight[0];
                    out.minY[splitCount - 8 + queueIndex]    = valuesRight[1];
                    out.minZ[splitCount - 8 + queueIndex]    = valuesRight[2];
                    out.maxX[splitCount - 8 + queueIndex]    = valuesRight[4];
                    out.maxY[splitCount - 8 + queueIndex]    = valuesRight[5];
                    out.maxZ[splitCount - 8 + queueIndex]    = valuesRight[6];
                    out.primIDs[splitCount - 8 + queueIndex] = data->primIDs[refID];
                    out.geomIDs[splitCount - 8 + queueIndex] = data->geomIDs[refID];
                }
            }
        }
        for (; i < range.End(); i++)
        {
            f32 min            = minStream[i];
            f32 max            = maxStream[i];
            refIDQueue[count]  = i;
            faceIDQueue[count] = data->primIDs[i];
            count += (min > -split.bestValue && max > split.bestValue);
        }
        Assert(splitCount + count <= splitMax);
        // Flush the queue
        u32 remainingCount = count;
        const u32 numIters = (remainingCount + 7) >> 3;
        for (u32 remaining = 0; remaining < numIters; remaining++)
        {
            u32 numPrims = Min(remainingCount, LANE_WIDTH);
            Bounds8 bounds[2][8];
            for (u32 b = 0; b < LANE_WIDTH; b++)
            {
                bounds[0][b] = pos_inf;
            }
            Triangle8 tri = Triangle8::Load(mesh, dim, faceIDQueue + remaining * LANE_WIDTH); //, bounds[0]);

            ClipTriangle(mesh, dim, tri, split.bestValue, bounds[0], bounds[1]);
            for (u32 b = 0; b < numPrims; b++)
            {
                const u32 refID = refIDQueue[remaining * LANE_WIDTH + b];
                // const u32 refID = faceIDQueue[count + queueIndex];
                alignas(32) f32 valuesLeft[8];
                alignas(32) f32 valuesRight[8];
                Lane8F32::Store(valuesLeft, bounds[0][b].v);
                Lane8F32::Store(valuesRight, bounds[1][b].v);

                data->minX[refID] = valuesLeft[0];
                data->minY[refID] = valuesLeft[1];
                data->minZ[refID] = valuesLeft[2];
                data->maxX[refID] = valuesLeft[4];
                data->maxY[refID] = valuesLeft[5];
                data->maxZ[refID] = valuesLeft[6];
                data->geomIDs[refID]++;

                out.minX[splitCount]    = valuesRight[0];
                out.minY[splitCount]    = valuesRight[1];
                out.minZ[splitCount]    = valuesRight[2];
                out.maxX[splitCount]    = valuesRight[4];
                out.maxY[splitCount]    = valuesRight[5];
                out.maxZ[splitCount]    = valuesRight[6];
                out.primIDs[splitCount] = data->primIDs[refID];
                out.geomIDs[splitCount] = data->geomIDs[refID];
                totalSplitCount++;
                splitCount++;
            }

            remainingCount -= LANE_WIDTH;
        }
        Assert(totalSplitCount <= range.ExtSize());
        Assert(splitCount == totalSplitCount);
        if (outSplitCount)
        {
            *outSplitCount = splitCount;
        }

        u32 mid = PartitionParallel(split, range, data);
        return mid;
    }
    void Merge(const HeuristicSOASplitBinning<numBins> &other)
    {
        for (u32 i = 0; i < numBins; i++)
        {
            entryCounts[i] += other.entryCounts[i];
            exitCounts[i] += other.exitCounts[i];
            for (u32 dim = 0; dim < 3; dim++)
            {
                finalBounds[dim][i].Extend(other.finalBounds[dim][i]);
            }
        }
    }
};

// SBVH
static const f32 sbvhAlpha = 1e-5;
template <i32 numObjectBins = 32, i32 numSpatialBins = 16>
struct HeuristicSpatialSplits
{
    using Record = RecordSOASplits;
    using HSplit = HeuristicSOASplitBinning<numSpatialBins>;
    using OBin   = HeuristicSOAObjectBinning<numObjectBins>;

    Arena **arenas;
    TriangleMesh *mesh;
    f32 rootArea;
    PrimDataSOA *soa;
    HeuristicSpatialSplits() {}
    HeuristicSpatialSplits(Arena **arenas, PrimDataSOA *soa, TriangleMesh *mesh, f32 rootArea) :
        arenas(arenas), soa(soa), mesh(mesh), rootArea(rootArea) {}

    void CalculateBounds(u32 start, u32 count, Bounds8F32 &geom, Bounds8F32 &cent)
    {
        Lane8F32 lanes[8];

        u32 i            = start;
        u32 alignedCount = count - count % LANE_WIDTH;
        for (; i < start + alignedCount; i += LANE_WIDTH)
        {
            Lane8F32 min[] = {
                Lane8F32::LoadU(soa->minX + i),
                Lane8F32::LoadU(soa->minY + i),
                Lane8F32::LoadU(soa->minZ + i),
            };

            Lane8F32 max[] = {
                Lane8F32::LoadU(soa->maxX + i),
                Lane8F32::LoadU(soa->maxY + i),
                Lane8F32::LoadU(soa->maxZ + i),
            };
            Lane8F32 centroids[] = {
                (max[0] - min[0]) * 0.5f,
                (max[1] - min[1]) * 0.5f,
                (max[2] - min[2]) * 0.5f,
            };
            geom.ExtendNegativeMin(min[0], min[1], min[2], max[0], max[1], max[2]);
            cent.Extend(centroids[0], centroids[1], centroids[2]);
        }
        for (; i < start + count; i++)
        {
            f32 min[] = {
                soa->minX[i],
                soa->minY[i],
                soa->minZ[i],
            };
            f32 max[] = {
                soa->maxX[i],
                soa->maxY[i],
                soa->maxZ[i],
            };
            f32 centroid[] = {
                ((Lane8F32(max[0]) - min[0]) * 0.5f)[0],
                ((Lane8F32(max[1]) - min[1]) * 0.5f)[0],
                ((Lane8F32(max[2]) - min[2]) * 0.5f)[0],
            };
            geom.ExtendNegativeMin(min[0], min[1], min[2], max[0], max[1], max[2]);
            cent.Extend(centroid[0], centroid[1], centroid[2]);
        }
    }

    Split Bin(const Record &record, u32 blockSize = 1)
    {
        // Object splits
        TempArena temp    = ScratchStart(0, 0);
        u64 popPos        = ArenaPos(temp.arena);
        temp.arena->align = 32;

        // Stack allocate the heuristics since they store the centroid and geom bounds we'll need later
        ObjectBinner<numObjectBins> *objectBinner = PushStructConstruct(temp.arena, ObjectBinner<numObjectBins>)(record.centBounds.ToBounds());
        OBin *objectBinHeuristic                  = PushStructNoZero(temp.arena, OBin);
        if (record.range.count > PARALLEL_THRESHOLD)
        {
            const u32 groupSize = PARALLEL_THRESHOLD;
            *objectBinHeuristic = ParallelReduce<OBin>(
                record.range.start, record.range.count, groupSize,
                [&](OBin &binner, u32 start, u32 count) { binner.Bin(soa, start, count); },
                [&](OBin &l, const OBin &r) { l.Merge(r); },
                objectBinner);
        }
        else
        {
            *objectBinHeuristic = OBin(objectBinner);
            objectBinHeuristic->Bin(soa, record.range.start, record.range.count);
        }
        struct Split objectSplit = BinBest(objectBinHeuristic->finalBounds,
                                           objectBinHeuristic->counts, objectBinner);
        objectSplit.type         = Split::Object;

        Bounds geomBoundsL;
        for (u32 i = 0; i < objectSplit.bestPos; i++)
        {
            geomBoundsL.Extend(objectBinHeuristic->finalBounds[objectSplit.bestDim][i]);
        }
        Bounds geomBoundsR;
        for (u32 i = objectSplit.bestPos; i < numObjectBins; i++)
        {
            geomBoundsR.Extend(objectBinHeuristic->finalBounds[objectSplit.bestDim][i]);
        }

        f32 lambda = HalfArea(Intersect(geomBoundsL, geomBoundsR));
        if (lambda > sbvhAlpha * rootArea)
        {
            // Spatial splits
            SplitBinner<numSpatialBins> *splitBinner = PushStructConstruct(temp.arena, SplitBinner<numSpatialBins>)(record.geomBounds.ToBounds());

            HSplit *splitHeuristic = PushStructNoZero(temp.arena, HSplit);
            if (record.range.count > PARALLEL_THRESHOLD)
            {
                const u32 groupSize = 4 * 1024; // PARALLEL_THRESHOLD;
                *splitHeuristic     = ParallelReduce<HSplit>(
                    record.range.start, record.range.count, groupSize,
                    [&](HSplit &binner, u32 start, u32 count) { binner.Bin(mesh, soa, start, count); },
                    [&](HSplit &l, const HSplit &r) { l.Merge(r); },
                    splitBinner);
            }
            else
            {
                *splitHeuristic = HSplit(splitBinner);
                splitHeuristic->Bin(mesh, soa, record.range.start, record.range.count);
            }
            struct Split spatialSplit = BinBest(splitHeuristic->finalBounds,
                                                splitHeuristic->entryCounts, splitHeuristic->exitCounts, splitBinner);
            spatialSplit.type         = Split::Spatial;
            u32 lCount                = 0;
            for (u32 i = 0; i < spatialSplit.bestPos; i++)
            {
                lCount += splitHeuristic->entryCounts[i][spatialSplit.bestDim];
            }
            u32 rCount = 0;
            for (u32 i = spatialSplit.bestPos; i < numSpatialBins; i++)
            {
                rCount += splitHeuristic->exitCounts[i][spatialSplit.bestDim];
            }
            if (spatialSplit.bestSAH < objectSplit.bestSAH && lCount + rCount - record.range.count <= record.range.ExtSize())
            {
                spatialSplit.ptr      = (void *)splitHeuristic;
                spatialSplit.allocPos = popPos;
                return spatialSplit;
            }
        }
        objectSplit.ptr      = (void *)objectBinHeuristic;
        objectSplit.allocPos = popPos;
        return objectSplit;
    }
    void Split(struct Split split, const Record &record, Record &outLeft, Record &outRight)
    {
        // NOTE: Split must be called from the same thread as Bin
        TempArena temp = ScratchStart(0, 0);

        const ExtRange &range = record.range;
        PrimDataSOA *data     = soa;
        u32 mid;
        u32 splitCount = 0;

        if (split.bestSAH == f32(pos_inf))
        {
            mid = record.range.start + (record.range.count / 2);
            Bounds geomLeft;
            Bounds centLeft;
            Bounds geomRight;
            Bounds centRight;
            for (u32 i = record.range.start; i < record.range.End(); i++) // mid; i++)
            {
                f32 min[3] = {
                    data->minX[i],
                    data->minY[i],
                    data->minZ[i],
                };
                f32 max[3] = {
                    data->maxX[i],
                    data->maxY[i],
                    data->maxZ[i],
                };
                f32 centroid[3] = {
                    (max[0] - min[0]) * 0.5f,
                    (max[1] - min[1]) * 0.5f,
                    (max[2] - min[2]) * 0.5f,
                };
                if (i < mid)
                {
                    geomLeft.Extend(Lane4F32(min[0], min[1], min[2], 0.f), Lane4F32(max[0], max[1], max[2], 0.f));
                    centLeft.Extend(Lane4F32(centroid[0], centroid[1], centroid[2], 0.f));
                }
                else
                {
                    geomRight.Extend(Lane4F32(min[0], min[1], min[2], 0.f), Lane4F32(max[0], max[1], max[2], 0.f));
                    centRight.Extend(Lane4F32(centroid[0], centroid[1], centroid[2], 0.f));
                }
            }
            outLeft.geomBounds.FromBounds(geomLeft);
            outLeft.centBounds.FromBounds(centLeft);
            outRight.geomBounds.FromBounds(geomRight);
            outRight.centBounds.FromBounds(centRight);
        }
        else
        {
            // PerformanceCounter perCounter = OS_StartCounter();
            switch (split.type)
            {
                case Split::Object:
                {
                    OBin *heuristic = (OBin *)(split.ptr);
                    mid             = PartitionParallelCentroids(split, record.range, soa);
                }
                break;
                case Split::Spatial:
                {
                    HSplit *heuristic = (HSplit *)(split.ptr);
                    mid               = heuristic->Split(arenas[GetThreadIndex()], mesh, soa, record.range,
                                                         split, outLeft, outRight, &splitCount);
                }
                break;
            }
            // threadLocalStatistics[GetThreadIndex()].miscF += OS_GetMilliseconds(perCounter);
        }

        OBin *oHeuristic   = (OBin *)(split.ptr);
        HSplit *sHeuristic = (HSplit *)(split.ptr);

        struct CentGeomLane8F32
        {
            Bounds8F32 geom;
            Bounds8F32 cent;
            CentGeomLane8F32() : geom(neg_inf), cent() {}
        };

        u32 numLeft  = mid - range.start;
        u32 numRight = range.End() - mid + splitCount;

        CentGeomLane8F32 left;
        if (numLeft > PARALLEL_THRESHOLD)
        {
            left = ParallelReduce<CentGeomLane8F32>(
                record.range.start, numLeft, PARALLEL_THRESHOLD,
                [&](CentGeomLane8F32 &bounds, u32 start, u32 count) { CalculateBounds(start, count, bounds.geom, bounds.cent); },
                [&](CentGeomLane8F32 &l, const CentGeomLane8F32 &r) { l.geom.ExtendNegativeMin(r.geom); l.cent.Extend(r.cent); });
            outLeft.geomBounds.FromBounds(left.geom.ToBoundsNegMin());
            outLeft.centBounds.FromBounds(left.cent.ToBounds());
        }
        else
        {
            CalculateBounds(record.range.start, numLeft, left.geom, left.cent);
            outLeft.geomBounds.FromBounds(left.geom.ToBoundsNegMin());
            outLeft.centBounds.FromBounds(left.cent.ToBounds());
        }
        CentGeomLane8F32 right;
        if (numRight > PARALLEL_THRESHOLD)
        {
            right = ParallelReduce<CentGeomLane8F32>(
                mid, numRight, PARALLEL_THRESHOLD,
                [&](CentGeomLane8F32 &bounds, u32 start, u32 count) { CalculateBounds(start, count, bounds.geom, bounds.cent); },
                [&](CentGeomLane8F32 &l, const CentGeomLane8F32 &r) { l.geom.ExtendNegativeMin(r.geom); l.cent.Extend(r.cent); });
            outRight.geomBounds.FromBounds(right.geom.ToBoundsNegMin());
            outRight.centBounds.FromBounds(right.cent.ToBounds());
        }
        else
        {
            CalculateBounds(mid, numRight, right.geom, right.cent);
            outRight.geomBounds.FromBounds(right.geom.ToBoundsNegMin());
            outRight.centBounds.FromBounds(right.cent.ToBounds());
        }

        f32 weight         = (f32)(numLeft) / (numLeft + numRight);
        u32 remainingSpace = (range.ExtSize() - splitCount);
        u32 extSizeLeft    = Min((u32)(remainingSpace * weight), remainingSpace);
        u32 extSizeRight   = remainingSpace - extSizeLeft;

        u32 shift      = Max(extSizeLeft, numRight);
        u32 numToShift = Min(extSizeLeft, numRight);

        if (numToShift != 0)
        {
            if (numToShift > PARALLEL_THRESHOLD)
            {
                u32 groupSize = PARALLEL_THRESHOLD;
                u32 taskCount = (numToShift + groupSize - 1) / groupSize;
                taskCount     = Min(taskCount, scheduler.numWorkers);
                u32 stepSize  = numToShift / taskCount;

                scheduler.ScheduleAndWait(taskCount, 1, [&](u32 jobID) {
                    u32 start = mid + jobID * stepSize;
                    u32 end   = start + stepSize; // Min(mid + numToShift, start + groupSize);
                    if (jobID == taskCount - 1)
                    {
                        Assert(mid + numToShift >= end);
                        end = mid + numToShift;
                    }
                    for (u32 i = start; i < end; i++)
                    {
                        Assert(i + shift < range.extEnd);
                        data->Set(*data, i + shift, i);
                    }
                });
            }
            else
            {
                for (u32 i = mid; i < mid + numToShift; i++)
                {
                    Assert(i + shift < range.extEnd);
                    data->Set(*data, i + shift, i);
                }
            }
        }
        Assert(numLeft <= record.range.count);
        Assert(numRight <= record.range.count);

        outLeft.range    = ExtRange(range.start, numLeft, range.start + numLeft + extSizeLeft);
        outRight.range   = ExtRange(outLeft.range.extEnd, numRight, range.extEnd);
        u32 rightExtSize = outRight.range.ExtSize();
        Assert(rightExtSize == extSizeRight);

        ArenaPopTo(temp.arena, split.allocPos);
    }
};

// NOTE: this is embree's implementation of split binning for SBVH
template <i32 numBins = 16>
struct TestHeuristic
{
    Lane4F32 base;
    Lane4F32 scale;
    Lane4F32 invScale;

    TestHeuristic(Lane4F32 base, Lane4F32 scale, Lane4F32 invScale) : base(base), scale(scale), invScale(invScale) {}
    f32 GetSplitValue(u32 bin, u32 dim) const
    {
        return bin * invScale[dim] + base[dim];
    }
};
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
    __forceinline u32 Split(TriangleMesh *mesh, PrimData *prims, ExtRange range, Split split, Bounds &outLeft, Bounds &outRight,
                            Bounds &centLeft, Bounds &centRight)
    {
        u32 dim                         = split.bestDim;
        const size_t max_ext_range_size = range.ExtSize();
        const size_t ext_range_start    = range.End();

        /* atomic counter for number of primref splits */
        // std::atomic<size_t> ext_elements;
        // ext_elements.store(0);

        const float fpos = (split.bestPos * invScale[dim]) + base[dim];

        u32 ID = 0;

        for (size_t i = range.start; i < range.End(); i++)
        {
            // const unsigned int splits = prims[i].GeomID();

            const u32 bin0 = (u32)Floor((prims[i].minP[dim] - base[dim]) * scale[dim]);
            const u32 bin1 = (u32)Floor((prims[i].maxP[dim] - base[dim]) * scale[dim]);
            if (bin0 < split.bestPos && bin1 >= split.bestPos)
            {
                Bounds left, right;
                Bounds primBounds;
                primBounds.minP = prims[i].minP;
                primBounds.maxP = prims[i].maxP;
                ClipTriangleSimple(mesh, primBounds, prims[i].PrimID(), dim, fpos, left, right);

                // no empty splits
                // if (left.Empty() || right.Empty()) continue;

                /* break if the number of subdivided elements are greater than the maximum allowed size */
                if (ID++ >= max_ext_range_size) break;

                /* only write within the correct bounds */
                prims[i].minP                    = left.minP;
                prims[i].maxP                    = left.maxP;
                prims[ext_range_start + ID].minP = right.minP;
                prims[ext_range_start + ID].maxP = right.maxP;
            }
        }

        PartitionResult result;
        PartitionParallel(split, prims, range.start, range.End(), &result);
        // PartitionSerial(split, prims, range.start, range.End(), &result);
        outLeft   = result.geomBoundsL;
        outRight  = result.geomBoundsR;
        centLeft  = result.centBoundsL;
        centRight = result.centBoundsR;
        return result.mid;
    }
};

template <typename Binner, i32 numBins>
Split BinBest(const Bounds bounds[3][numBins],
              const Lane4U32 *entryCounts,
              const Lane4U32 *exitCounts,
              const Binner *binner,
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
        // if (binner->scale[dim][0] == 0.f) continue;

        if (lBestSAH[dim] < bestArea)
        {
            bestArea = lBestSAH[dim];
            bestPos  = lBestPos[dim];
            bestDim  = dim;
        }
    }
    f32 bestValue = binner->GetSplitValue(bestPos, bestDim); // + 1, bestDim);
    return Split(bestArea, bestPos, bestDim, bestValue);
}

template <typename Binner, i32 numBins>
Split BinBest(const Bounds bounds[3][numBins],
              const Lane4U32 *counts,
              const Binner *binner,
              const u32 blockShift = 0)
{
    return BinBest(bounds, counts, counts, binner, blockShift);
}

} // namespace rt

#endif
