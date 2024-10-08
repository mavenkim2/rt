#ifndef BVH_AOS_H
#define BVH_AOS_H
#include <utility>
namespace rt
{

template <typename Binner>
void Partition(u32 lOffset, u32 rOffset, const Binner *binner, u32 l, u32 r, const PrimRef *data, PrimRef *outRefs, u32 dim,
               u32 bestPos, Lane8F32 &outLeft, Lane8F32 &outRight, u32 expectedLCount, u32 expectedRCount)
{
    u32 writeLocs[2]  = {lOffset, rOffset};
    Lane8F32 masks[2] = {Lane8F32::Mask(false), Lane8F32::Mask(true)};

    Lane8F32 lanes[8];

    Lane8F32 centLeft(neg_inf);
    Lane8F32 centRight(neg_inf);

    u32 i      = l;
    u32 totalL = 0;
    u32 totalR = 0;
    for (; i + (LANE_WIDTH - 1) < r; i += LANE_WIDTH)
    {
        Transpose8x6(data[i].m256, data[i + 1].m256, data[i + 2].m256, data[i + 3].m256,
                     data[i + 4].m256, data[i + 5].m256, data[i + 6].m256, data[i + 7].m256,
                     lanes[0], lanes[1], lanes[2], lanes[3], lanes[4], lanes[5]);
        Lane8F32 centroids[3] = {
            (lanes[3] - lanes[0]) * 0.5f,
            (lanes[4] - lanes[1]) * 0.5f,
            (lanes[5] - lanes[2]) * 0.5f,
        };
        Lane8U32 bin = binner->Bin(centroids[dim], dim);

        Lane8F32 mask = (AsFloat(bin) >= AsFloat(bestPos));
        u32 prevMask  = Movemask(mask);
        Transpose3x8(centroids[0], centroids[1], centroids[2],
                     lanes[0], lanes[1], lanes[2], lanes[3], lanes[4], lanes[5], lanes[6], lanes[7]);
        for (u32 b = 0; b < LANE_WIDTH; b++)
        {
            u32 select = (prevMask >> i) & 1;
            totalL += !select;
            totalR += select;
            centLeft  = MaskMax(masks[!select], centLeft, lanes[b] ^ signFlipMask);
            centRight = MaskMax(masks[select], centRight, lanes[b] ^ signFlipMask);
            _mm256_stream_ps((f32 *)&outRefs[writeLocs[select]++], data[i + b].m256);

            // left                         = MaskMax(masks[!select], left, data[inRefs[i + b]].m256);
            // right                        = MaskMax(masks[select], right, data[inRefs[i + b]].m256);
        }
    }
    for (; i < r; i++)
    {
        const PrimRef *primRef = &data[i];
        // f32 min                = primRef->min[dim];
        // f32 max                = primRef->max[dim];
        Lane4F32 min      = Extract4<0>(primRef->m256);
        Lane4F32 max      = Extract4<1>(primRef->m256);
        Lane4F32 centroid = (max - min) * 0.5f;
        Lane8F32 c(-centroid, centroid);
        u32 bin      = binner->Bin(centroid[dim], dim);
        bool isRight = bin >= bestPos;
        _mm256_stream_ps((f32 *)&outRefs[writeLocs[isRight]++], primRef->m256);
        if (isRight)
        {
            centRight = Max(centRight, c);
            totalR++;
        }
        else
        {
            centLeft = Max(centLeft, c);
            totalL++;
        }
    }
    Assert(expectedLCount == totalL);
    Assert(expectedRCount == totalR);
    outLeft  = centLeft;
    outRight = centRight;
}

template <i32 numBins = 32>
struct HeuristicAOSObjectBinning
{
    Bounds8 bins[3][numBins];
    Lane4U32 counts[numBins];
    ObjectBinner<numBins> *binner;

    HeuristicAOSObjectBinning() {}
    HeuristicAOSObjectBinning(ObjectBinner<numBins> *binner) : binner(binner)
    {
        for (u32 dim = 0; dim < 3; dim++)
        {
            for (u32 i = 0; i < numBins; i++)
            {
                bins[dim][i] = Bounds8();
            }
        }
        for (u32 i = 0; i < numBins; i++)
        {
            counts[i] = 0;
        }
    }
    void Bin(const PrimRef *data, u32 start, u32 count)
    {
        u32 alignedCount = count - count % LANE_WIDTH;
        u32 i            = start;

        Lane8F32 prevLanes[8];
        alignas(32) u32 prevBinIndices[3][8];
        if (count >= LANE_WIDTH)
        {
            Lane8F32 temp[6];
            Transpose8x6(data[i + 1].m256, data[i + 1].m256, data[i + 2].m256, data[i + 3].m256,
                         data[i + 4].m256, data[i + 5].m256, data[i + 6].m256, data[i + 7].m256,
                         temp[0], temp[1], temp[2], temp[3], temp[4], temp[5]);
            Lane8F32 centroids[3] = {
                (temp[3] - temp[0]) * 0.5f,
                (temp[4] - temp[1]) * 0.5f,
                (temp[5] - temp[2]) * 0.5f,
            };
            Lane8U32::Store(prevBinIndices[0], binner->Bin(centroids[0], 0));
            Lane8U32::Store(prevBinIndices[1], binner->Bin(centroids[1], 1));
            Lane8U32::Store(prevBinIndices[2], binner->Bin(centroids[2], 2));
            for (u32 c = 0; c < LANE_WIDTH; c++)
            {
                prevLanes[c] = data[i + c].m256;
            }
            i += LANE_WIDTH;
        }
        for (; i < start + alignedCount; i += LANE_WIDTH)
        {
            Lane8F32 temp[6];

            Transpose8x6(data[i + 1].m256, data[i + 1].m256, data[i + 2].m256, data[i + 3].m256,
                         data[i + 4].m256, data[i + 5].m256, data[i + 6].m256, data[i + 7].m256,
                         temp[0], temp[1], temp[2], temp[3], temp[4], temp[5]);
            Lane8F32 centroids[] = {
                (temp[3] - temp[0]) * 0.5f,
                (temp[4] - temp[1]) * 0.5f,
                (temp[5] - temp[2]) * 0.5f,
            };

            Lane8U32 indicesX = binner->Bin(centroids[0], 0);
            Lane8U32 indicesY = binner->Bin(centroids[1], 1);
            Lane8U32 indicesZ = binner->Bin(centroids[2], 2);

            for (u32 dim = 0; dim < 3; dim++)
            {
                for (u32 b = 0; b < LANE_WIDTH; b++)
                {
                    u32 bin = prevBinIndices[dim][b];
                    bins[dim][bin].Extend(prevLanes[b]);
                    counts[bin][dim]++;
                }
            }

            Lane8U32::Store(prevBinIndices[0], indicesX);
            Lane8U32::Store(prevBinIndices[1], indicesY);
            Lane8U32::Store(prevBinIndices[2], indicesZ);
            for (u32 c = 0; c < LANE_WIDTH; c++)
            {
                prevLanes[c] = data[i + c].m256;
            }
        }
        if (count >= LANE_WIDTH)
        {
            for (u32 dim = 0; dim < 3; dim++)
            {
                for (u32 b = 0; b < LANE_WIDTH; b++)
                {
                    u32 bin = prevBinIndices[dim][b];
                    bins[dim][bin].Extend(prevLanes[b]);
                    counts[bin][dim]++;
                }
            }
        }
        for (; i < start + count; i++)
        {
            Lane4F32 low      = Extract4<0>(data[i].m256);
            Lane4F32 hi       = Extract4<1>(data[i].m256);
            Lane4F32 centroid = (hi - low) * 0.5f;
            u32 indices[]     = {
                binner->Bin(centroid[0], 0),
                binner->Bin(centroid[1], 1),
                binner->Bin(centroid[2], 2),
            };
            for (u32 dim = 0; dim < 3; dim++)
            {
                u32 bin = indices[dim];
                bins[dim][bin].Extend(data[i].m256);
                counts[bin][dim]++;
            }
        }
    }
    void Merge(const HeuristicAOSObjectBinning &other)
    {
        for (u32 dim = 0; dim < 3; dim++)
        {
            for (u32 i = 0; i < numBins; i++)
            {
                bins[dim][i].Extend(other.bins[dim][i]);
            }
        }
        for (u32 i = 0; i < numBins; i++)
        {
            counts[i] += other.counts[i];
        }
    }
};

template <i32 numBins = 16>
struct alignas(32) HeuristicAOSSplitBinning
{
    Bounds8 bins[3][numBins];
    Lane4U32 entryCounts[numBins];
    Lane4U32 exitCounts[numBins];

    SplitBinner<numBins> *binner;

    HeuristicAOSSplitBinning() {}
    HeuristicAOSSplitBinning(SplitBinner<numBins> *binner) : binner(binner)
    {
        for (u32 dim = 0; dim < 3; dim++)
        {
            for (u32 i = 0; i < numBins; i++)
            {
                bins[dim][i] = Bounds8();
            }
        }
        for (u32 i = 0; i < numBins; i++)
        {
            entryCounts[i] = 0;
            exitCounts[i]  = 0;
        }
    }

    void Bin(TriangleMesh *mesh, const PrimRef *data, u32 start, u32 count)
    {
        u32 binCounts[3][numBins]                                 = {};
        alignas(32) u32 binIndexStart[3][numBins][2 * LANE_WIDTH] = {};
        u32 faceIndices[3][numBins][2 * LANE_WIDTH]               = {};

        u32 i            = start;
        u32 alignedCount = count - count % LANE_WIDTH;

        Lane8F32 lanes[6];

        for (; i < start + alignedCount; i += LANE_WIDTH)
        {
            Transpose8x6(data[i].m256, data[i + 1].m256, data[i + 2].m256, data[i + 3].m256,
                         data[i + 4].m256, data[i + 5].m256, data[i + 6].m256, data[i + 7].m256,
                         lanes[0], lanes[1], lanes[2], lanes[3], lanes[4], lanes[5]);

            Lane8U32 indexMinArr[3] = {
                binner->BinMin(lanes[0], 0),
                binner->BinMin(lanes[1], 1),
                binner->BinMin(lanes[2], 2),
            };
            Lane8U32 indexMaxArr[3] = {
                binner->BinMax(lanes[3], 0),
                binner->BinMax(lanes[4], 1),
                binner->BinMax(lanes[5], 2),
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
                            bins[dim][indexMin].Extend(data[i + b].m256);
                        }
                        break;
                        default:
                        {
                            bitMask[dim] |= (1 << diff);
                            faceIndices[dim][diff][binCounts[dim][diff]]   = data[i + b].primID;
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
                        Triangle8 tri     = Triangle8::Load(mesh, dim, faceIndices[dim][bin] + binCount);
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
                                bins[dim][binIndex].Extend(bounds[current][b]);
                            }
                            current = !current;
                        }
                        for (u32 b = 0; b < LANE_WIDTH; b++)
                        {
                            u32 binIndex = binIndices[b] + 1;
                            bins[dim][binIndex].Extend(bounds[current][b]);
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
            const PrimRef *ref = &data[i];

            for (u32 dim = 0; dim < 3; dim++)
            {
                u32 binIndexMin = binner->BinMin(ref->min[dim], dim);
                u32 binIndexMax = binner->BinMax(ref->max[dim], dim);

                Assert(binIndexMax >= binIndexMin);

                u32 diff = binIndexMax - binIndexMin;
                entryCounts[binIndexMin][dim] += 1;
                exitCounts[binIndexMax][dim] += 1;
                switch (diff)
                {
                    case 0:
                    {
                        bins[dim][binIndexMin].Extend(ref->m256);
                    }
                    break;
                    default:
                    {
                        faceIndices[dim][diff][binCounts[dim][diff]]   = ref->primID;
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
                            bins[dim][binIndex].Extend(bounds[current][b]);
                        }
                        current = !current;
                    }
                    for (u32 b = 0; b < numPrims; b++)
                    {
                        u32 binIndex = binIndices[b] + 1;
                        bins[dim][binIndex].Extend(bounds[current][b]);
                    }
                    remainingCount -= LANE_WIDTH;
                }
            }
        }
    }

    // Splits and partitions at the same time
    void Split(TriangleMesh *mesh, const PrimRef *data, PrimRef *outData, u32 outLStart, u32 outRStart,
               u32 start, u32 count, Split split, RecordAOSSplits &outLeft, RecordAOSSplits &outRight,
               u32 expectedL, u32 expectedR)
    {
        u32 dim                         = split.bestDim;
        u32 alignedCount                = count - count % LANE_WIDTH;
        u32 faceIDQueue[LANE_WIDTH * 2] = {};
        Lane8F32 masks[2]               = {Lane8F32::Mask(false), Lane8F32::Mask(true)};

        f32 negBestValue = -split.bestValue;
        u32 i            = start;
        Lane8F32 lanes[8];

        u32 splitCount = 0;

        Lane8F32 geomLeft;
        Lane8F32 geomRight;

        Bounds8F32 centLeft;
        Bounds8F32 centRight;

        u32 totalL = 0;
        u32 totalR = 0;

        u32 writeLocs[2] = {outLStart, outRStart};
        for (; i < start + alignedCount; i += LANE_WIDTH)
        {
            // Transposes the lanes in order to compute the bounds for the next generation
            // Lane8F32 fragments[] = {
            //     _mm256_castsi256_ps(_mm256_stream_load_si256((__m256i *)&data[i].m256)),
            //     _mm256_castsi256_ps(_mm256_stream_load_si256((__m256i *)&data[i + 1].m256)),
            //     _mm256_castsi256_ps(_mm256_stream_load_si256((__m256i *)&data[i + 2].m256)),
            //     _mm256_castsi256_ps(_mm256_stream_load_si256((__m256i *)&data[i + 3].m256)),
            //     _mm256_castsi256_ps(_mm256_stream_load_si256((__m256i *)&data[i + 4].m256)),
            //     _mm256_castsi256_ps(_mm256_stream_load_si256((__m256i *)&data[i + 5].m256)),
            //     _mm256_castsi256_ps(_mm256_stream_load_si256((__m256i *)&data[i + 6].m256)),
            //     _mm256_castsi256_ps(_mm256_stream_load_si256((__m256i *)&data[i + 7].m256)),
            // };
            // Transpose8x8(fragments[0], fragments[1], fragments[2], fragments[3],
            //              fragments[4], fragments[5], fragments[6], fragments[7],
            //              lanes[0], lanes[1], lanes[2], lanes[3], lanes[4], lanes[5], lanes[6], lanes[7]);

            Transpose8x8(data[i].m256, data[i + 1].m256, data[i + 2].m256, data[i + 3].m256,
                         data[i + 4].m256, data[i + 5].m256, data[i + 6].m256, data[i + 7].m256,
                         lanes[0], lanes[1], lanes[2], lanes[3], lanes[4], lanes[5], lanes[6], lanes[7]);
            // See if the primitive needs to be split
            // Only add the primitive to the left and right bounds if it doesn't need to be split
            Lane8U32 faceIDs = AsUInt(lanes[7]);

            Lane8F32 isFullyLeftV  = AsFloat(binner->BinMin(lanes[dim + 4], dim)) < AsFloat(split.bestPos);
            Lane8F32 isFullyRightV = AsFloat(binner->BinMax(lanes[dim], dim)) >= AsFloat(split.bestPos);

            Lane8F32 centroids[] = {
                (lanes[4] - lanes[0]) * 0.5f,
                (lanes[5] - lanes[1]) * 0.5f,
                (lanes[6] - lanes[2]) * 0.5f,
            };

            u32 lMaskBits = Movemask(isFullyLeftV);
            u32 rMaskBits = Movemask(isFullyRightV);
            // Extend next generation bounds
            centLeft.MaskExtend(isFullyLeftV, centroids[0], centroids[1], centroids[2]);
            centRight.MaskExtend(isFullyRightV, centroids[0], centroids[1], centroids[2]);

            for (u32 b = 0; b < LANE_WIDTH; b++)
            {
                u32 isFullyLeft  = (lMaskBits >> i) & 1;
                u32 isFullyRight = (rMaskBits >> i) & 1;

                geomLeft  = MaskMax(masks[isFullyLeft], geomLeft, data[i + b].m256);
                geomRight = MaskMax(masks[isFullyRight], geomRight, data[i + b].m256);

                if (isFullyRight)
                {
                    _mm256_stream_ps((f32 *)&outData[writeLocs[1]].m256, data[i + b].m256);
                    // Lane8F32::Store(&outData[writeLocs[1]], data[i + b].m256);
                    writeLocs[1]++;
                    totalR++;
                }
                else if (isFullyLeft)
                {
                    _mm256_stream_ps((f32 *)&outData[writeLocs[0]].m256, data[i + b].m256);
                    // Lane8F32::Store(&outData[writeLocs[0]], data[i + b].m256);
                    // outData[writeLocs[0]] = data[i + b];

                    writeLocs[0]++;
                    totalL++;
                }
                // outData[writeLocs[isFullyRight]] = data[i + b];
                // writeLocs[isFullyRight] += isFullyRight | isFullyLeft;
            }

            // Centroid bounds for next generation

            // Store primitives that need to be split
            u32 splitMask = (~(lMaskBits | rMaskBits)) & 0xff;
            Lane8U32::StoreU(faceIDQueue + splitCount, MaskCompress(splitMask, faceIDs));
            splitCount += PopCount(splitMask);

            if (splitCount >= LANE_WIDTH)
            {
                splitCount -= LANE_WIDTH;

                // Bounds8 boundsLeft[LANE_WIDTH];
                // Bounds8 boundsRight[LANE_WIDTH];
                Lane8F32 gL[LANE_WIDTH];
                Lane8F32 gR[LANE_WIDTH];
                Lane8F32 cL[3];
                Lane8F32 cR[3];
                Triangle8 tri = Triangle8::Load(mesh, dim, faceIDQueue + splitCount);
                ClipTriangle(mesh, dim, faceIDQueue + splitCount, tri, split.bestValue, gL, gR, cL, cR);

                centLeft.Extend(cL[0], cL[1], cL[2]);
                centRight.Extend(cR[0], cR[1], cR[2]);

                // TODO: there could be a false sharing problem here. not a major one, but these writes invalidate
                // any previously cached PrimRefs, so all reads at the beginning of the outer loop are a cache miss
                for (u32 queueIndex = 0; queueIndex < LANE_WIDTH; queueIndex++)
                {
                    geomLeft  = Max(geomLeft, gL[queueIndex]);
                    geomRight = Max(geomRight, gR[queueIndex]);

                    _mm256_stream_ps((f32 *)&outData[writeLocs[0]++].m256, gL[queueIndex]);
                    _mm256_stream_ps((f32 *)&outData[writeLocs[1]++].m256, gR[queueIndex]);
                    totalL++;
                    totalR++;
                    // Lane8F32::Store(&outData[writeLocs[0]++].m256, gL[queueIndex]);
                    // Lane8F32::Store(&outData[writeLocs[1]++].m256, gR[queueIndex]);
                }
            }
        }
        for (; i < start + count; i++)
        {
            const PrimRef *primRef = &data[i];
            Lane4F32 min           = Extract4<0>(primRef->m256);
            Lane4F32 max           = Extract4<1>(primRef->m256);

            bool isFullyLeft  = primRef->max[dim] < split.bestValue;
            bool isFullyRight = primRef->min[dim] <= -split.bestValue;
            Lane4F32 centroid = (max - min) * 0.5f;
            bool isSplit      = (~(isFullyLeft | isFullyRight)) & 0xff;

            Lane8F32 c(-centroid, centroid);
            faceIDQueue[splitCount] = primRef->primID;
            splitCount += isSplit;
            geomLeft  = MaskMax(masks[isFullyLeft], geomLeft, primRef->m256);
            geomRight = MaskMax(masks[isFullyRight], geomRight, primRef->m256);

            centLeft.MaskExtend(masks[isFullyLeft], centroid[0], centroid[1], centroid[2]);
            centRight.MaskExtend(masks[isFullyRight], centroid[0], centroid[1], centroid[2]);

            // outData[writeLocs[isFullyRight]] = data[i];
            // writeLocs[isFullyRight] += isFullyLeft | isFullyRight;
            if (isFullyRight)
            {
                // Lane8F32::Store(&outData[writeLocs[1]], data[i].m256);
                _mm256_stream_ps((f32 *)&outData[writeLocs[1]].m256, data[i].m256);
                // outData[writeLocs[1]] = data[i + b];
                writeLocs[1]++;
                totalR++;
            }
            else if (isFullyLeft)
            {
                // Lane8F32::Store(&outData[writeLocs[0]], data[i].m256);
                _mm256_stream_ps((f32 *)&outData[writeLocs[0]].m256, data[i].m256);
                // outData[writeLocs[0]] = data[i + b];
                writeLocs[0]++;
                totalL++;
            }
        }
        // Flush the queue
        u32 remainingCount = splitCount;
        const u32 numIters = (remainingCount + 7) >> 3;
        for (u32 remaining = 0; remaining < numIters; remaining++)
        {
            u32 qStart   = remaining * LANE_WIDTH;
            u32 numPrims = Min(remainingCount, LANE_WIDTH);
            Lane8F32 gL[LANE_WIDTH];
            Lane8F32 gR[LANE_WIDTH];
            Lane8F32 cL[3];
            Lane8F32 cR[3];
            Triangle8 tri = Triangle8::Load(mesh, dim, faceIDQueue + qStart);
            ClipTriangle(mesh, dim, faceIDQueue + qStart, tri, split.bestValue, gL, gR, cL, cR);

            Lane8F32 mask = Lane8F32::Mask((1 << numPrims) - 1u);
            centLeft.MaskExtend(mask, cL[0], cL[1], cL[2]);
            centRight.MaskExtend(mask, cR[0], cR[1], cR[2]);
            for (u32 queueIndex = 0; queueIndex < numPrims; queueIndex++)
            {
                geomLeft  = Max(geomLeft, gL[queueIndex]);
                geomRight = Max(geomRight, gR[queueIndex]);

                _mm256_stream_ps((f32 *)&outData[writeLocs[0]++].m256, gL[queueIndex]);
                _mm256_stream_ps((f32 *)&outData[writeLocs[1]++].m256, gR[queueIndex]);
                totalL++;
                totalR++;
                // Lane8F32::Store(&outData[writeLocs[0]++].m256, gL[queueIndex]);
                // Lane8F32::Store(&outData[writeLocs[1]++].m256, gR[queueIndex]);
            }

            remainingCount -= LANE_WIDTH;
        }
        Assert(totalL == expectedL);
        Assert(totalR == expectedR);
    }

    void Merge(const HeuristicAOSSplitBinning<numBins> &other)
    {
        for (u32 i = 0; i < numBins; i++)
        {
            entryCounts[i] += other.entryCounts[i];
            exitCounts[i] += other.exitCounts[i];
            for (u32 dim = 0; dim < 3; dim++)
            {
                bins[dim][i].Extend(other.bins[dim][i]);
            }
        }
    }
};

// SBVH
static const f32 sbvhAlpha = 1e-5;
template <i32 numObjectBins = 32, i32 numSpatialBins = 16>
struct HeuristicSpatialSplits
{
    using Record = RecordAOSSplits;
    using HSplit = HeuristicAOSSplitBinning<numSpatialBins>;
    using OBin   = HeuristicAOSObjectBinning<numObjectBins>;

    TriangleMesh *mesh;
    f32 rootArea;
    PrimRef *refs[2];

    HeuristicSpatialSplits() {}
    HeuristicSpatialSplits(PrimRef *data, PrimRef *data1, TriangleMesh *mesh, f32 rootArea)
        : refs{data, data1}, mesh(mesh), rootArea(rootArea) {}

    Split Bin(const Record &record, u32 primary, u32 blockSize = 1)
    {
        // Object splits
        TempArena temp    = ScratchStart(0, 0);
        u64 popPos        = ArenaPos(temp.arena);
        temp.arena->align = 32;

        // Stack allocate the heuristics since they store the centroid and geom bounds we'll need later
        ObjectBinner<numObjectBins> *objectBinner =
            PushStructConstruct(temp.arena, ObjectBinner<numObjectBins>)(record.centBounds);
        OBin *objectBinHeuristic = PushStructNoZero(temp.arena, OBin);
        ParallelForOutput objectOutput;
        if (record.count > PARALLEL_THRESHOLD)
        {
            const u32 groupSize = PARALLEL_THRESHOLD;
            objectOutput        = ParallelFor<OBin>(
                temp, record.start, record.count, groupSize,
                [&](OBin &binner, u32 start, u32 count) { binner.Bin(refs[primary], start, count); },
                objectBinner);
            Reduce<OBin>(
                *objectBinHeuristic, objectOutput,
                [&](OBin &l, const OBin &r) { l.Merge(r); },
                objectBinner);
        }
        else
        {
            *objectBinHeuristic = OBin(objectBinner);
            objectBinHeuristic->Bin(refs[primary], record.start, record.count);
            objectOutput.num       = 1;
            objectOutput.out       = objectBinHeuristic;
            objectOutput.groupSize = record.count;
        }
        struct Split objectSplit = BinBest(objectBinHeuristic->bins, objectBinHeuristic->counts, objectBinner);
        objectSplit.type         = Split::Object;

        Bounds8 geomBoundsL;
        for (u32 i = 0; i < objectSplit.bestPos; i++)
        {
            geomBoundsL.Extend(objectBinHeuristic->bins[objectSplit.bestDim][i]);
        }
        Bounds8 geomBoundsR;
        for (u32 i = objectSplit.bestPos; i < numObjectBins; i++)
        {
            geomBoundsR.Extend(objectBinHeuristic->bins[objectSplit.bestDim][i]);
        }

        f32 lambda = HalfArea(Intersect(geomBoundsL, geomBoundsR));
        if (lambda > sbvhAlpha * rootArea)
        {
            // Spatial splits
            SplitBinner<numSpatialBins> *splitBinner =
                PushStructConstruct(temp.arena, SplitBinner<numSpatialBins>)(record.geomBounds);

            HSplit *splitHeuristic = PushStructNoZero(temp.arena, HSplit);
            ParallelForOutput splitOutput;
            if (record.count > PARALLEL_THRESHOLD)
            {
                const u32 groupSize = PARALLEL_THRESHOLD;
                splitOutput         = ParallelFor<HSplit>(
                    temp, record.start, record.count, groupSize,
                    [&](HSplit &binner, u32 start, u32 count) { binner.Bin(mesh, refs[primary], start, count); },
                    splitBinner);
                Reduce<HSplit>(
                    *splitHeuristic, splitOutput,
                    [&](HSplit &l, const HSplit &r) { l.Merge(r); },
                    splitBinner);
            }
            else
            {
                *splitHeuristic = HSplit(splitBinner);
                splitHeuristic->Bin(mesh, refs[primary], record.start, record.count);
                splitOutput.out       = splitHeuristic;
                splitOutput.num       = 1;
                splitOutput.groupSize = record.count;
            }
            struct Split spatialSplit = BinBest(splitHeuristic->bins,
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
            u32 totalNumSplits = lCount + rCount - record.count;
            if (spatialSplit.bestSAH < objectSplit.bestSAH && totalNumSplits <= record.ExtSize())
            {
                u32 *lOffsets = PushArrayNoZero(temp.arena, u32, splitOutput.num);
                u32 *rOffsets = PushArrayNoZero(temp.arena, u32, splitOutput.num);
                u32 *lCounts  = PushArrayNoZero(temp.arena, u32, splitOutput.num);
                u32 *rCounts  = PushArrayNoZero(temp.arena, u32, splitOutput.num);

                u32 lOffset = record.extStart;
                u32 rOffset = record.extEnd;
                u32 rTotal  = 0;
                for (u32 i = 0; i < splitOutput.num; i++)
                {
                    HSplit *h       = &((HSplit *)splitOutput.out)[i];
                    lOffsets[i]     = lOffset;
                    u32 entryCounts = 0;
                    u32 exitCounts  = 0;
                    for (u32 bin = 0; bin < spatialSplit.bestPos; bin++)
                    {
                        entryCounts += h->entryCounts[bin][spatialSplit.bestDim];
                    }
                    for (u32 bin = spatialSplit.bestPos; bin < numSpatialBins; bin++)
                    {
                        exitCounts += h->exitCounts[bin][spatialSplit.bestDim];
                    }

                    lCounts[i] = entryCounts;
                    rCounts[i] = exitCounts;
                    lOffset += entryCounts;
                    rOffset -= exitCounts;
                    rTotal += exitCounts;
                    rOffsets[i] = rOffset;
                }
                // TODO: this function and Split() should probably just be combined instead of doing this
                spatialSplit.partitionPayload = PartitionPayload(lOffsets, rOffsets, lCounts, rCounts,
                                                                 splitOutput.num, splitOutput.groupSize);
                spatialSplit.ptr              = (void *)splitHeuristic;
                spatialSplit.allocPos         = popPos;
                spatialSplit.numLeft          = lOffset;
                spatialSplit.numRight         = rTotal;

                return spatialSplit;
            }
        }

        u32 *lOffsets = PushArrayNoZero(temp.arena, u32, objectOutput.num);
        u32 *rOffsets = PushArrayNoZero(temp.arena, u32, objectOutput.num);
        u32 *lCounts  = PushArrayNoZero(temp.arena, u32, objectOutput.num);
        u32 *rCounts  = PushArrayNoZero(temp.arena, u32, objectOutput.num);

        u32 lOffset = record.extStart;
        u32 rOffset = record.extEnd;
        u32 rTotal  = 0;
        for (u32 i = 0; i < objectOutput.num; i++)
        {
            OBin *h         = &((OBin *)objectOutput.out)[i];
            lOffsets[i]     = lOffset;
            u32 entryCounts = 0;
            u32 exitCounts  = 0;
            for (u32 bin = 0; bin < objectSplit.bestPos; bin++)
            {
                entryCounts += h->counts[bin][objectSplit.bestDim];
            }
            for (u32 bin = objectSplit.bestPos; bin < numObjectBins; bin++)
            {
                exitCounts += h->counts[bin][objectSplit.bestDim];
            }

            lCounts[i] = entryCounts;
            rCounts[i] = exitCounts;
            lOffset += entryCounts;
            rOffset -= exitCounts;
            rTotal += exitCounts;
            rOffsets[i] = rOffset;
        }

        objectSplit.partitionPayload = PartitionPayload(lOffsets, rOffsets, lCounts, rCounts,
                                                        objectOutput.num, objectOutput.groupSize);
        objectSplit.ptr              = (void *)objectBinHeuristic;
        objectSplit.allocPos         = popPos;
        objectSplit.numLeft          = lOffset;
        objectSplit.numRight         = rTotal;
        return objectSplit;
    }
    void FlushState(struct Split split)
    {
        TempArena temp = ScratchStart(0, 0);
        ArenaPopTo(temp.arena, split.allocPos);
    }
    void Split(struct Split split, u32 current, const Record &record, Record &outLeft, Record &outRight)
    {
        // NOTE: Split must be called from the same thread as Bin
        TempArena temp = ScratchStart(0, 0);

        PrimRef *inRefs  = refs[current];
        PrimRef *outRefs = refs[!current];

        if (split.bestSAH == f32(pos_inf))
        {
            u32 lCount = record.count / 2;
            u32 rCount = record.count - lCount;
            u32 mid    = record.start + lCount;
            Bounds8 geomLeft;
            Bounds8 centLeft;
            Bounds8 geomRight;
            Bounds8 centRight;
            u32 lOffset = record.extStart;
            u32 rOffset = record.extEnd - rCount;
            for (u32 i = record.start; i < record.start + record.count; i++)
            {
                PrimRef *ref      = &inRefs[i];
                Lane8F32 m256     = Lane8F32::Load(&ref->m256);
                Lane8F32 centroid = ((Shuffle4<1, 1>(m256) - Shuffle4<0, 0>(m256)) * 0.5f) ^ signFlipMask;
                if (i < mid)
                {
                    geomLeft.Extend(m256);
                    centLeft.Extend(centroid);
                    Lane8F32::Store(&outRefs[lOffset++], m256);
                }
                else
                {
                    geomRight.Extend(m256);
                    centRight.Extend(centroid);
                    Lane8F32::Store(&outRefs[rOffset++], m256);
                }
            }
            outLeft.geomBounds  = geomLeft.v;
            outLeft.centBounds  = centLeft.v;
            outRight.geomBounds = geomRight.v;
            outRight.centBounds = centRight.v;
        }
        else
        {
            // PerformanceCounter perCounter = OS_StartCounter();
            PartitionPayload &payload     = split.partitionPayload;
            RecordAOSSplits *leftRecords  = PushArrayNoZero(temp.arena, RecordAOSSplits, payload.count);
            RecordAOSSplits *rightRecords = PushArrayNoZero(temp.arena, RecordAOSSplits, payload.count);
            switch (split.type)
            {
                case Split::Object:
                {
                    OBin *heuristic = (OBin *)(split.ptr);
                    Bounds8 geomLeft;
                    Bounds8 geomRight;
                    for (u32 i = 0; i < split.bestPos; i++)
                    {
                        geomLeft.Extend(heuristic->bins[split.bestDim][i]);
                    }
                    for (u32 i = split.bestPos; i < numObjectBins; i++)
                    {
                        geomRight.Extend(heuristic->bins[split.bestDim][i]);
                    }
                    outLeft.geomBounds  = geomLeft.v;
                    outRight.geomBounds = geomRight.v;

                    if (payload.count > 1)
                    {
                        scheduler.ScheduleAndWait(payload.count, 1, [&](u32 jobID) {
                            u32 threadStart = record.start + payload.groupSize * jobID;
                            u32 count       = jobID == payload.count - 1 ? record.End() - threadStart : payload.groupSize;

                            Partition(payload.lOffsets[jobID], payload.rOffsets[jobID], heuristic->binner,
                                      threadStart, threadStart + count, inRefs, outRefs, split.bestDim, split.bestPos,
                                      leftRecords[jobID].centBounds, rightRecords[jobID].centBounds,
                                      payload.lCounts[jobID], payload.rCounts[jobID]);
                            // split.numLeft, split.numRight);
                        });
                    }
                    else
                    {
                        Assert(record.count == payload.groupSize);
                        Assert(record.extStart == payload.lOffsets[0]);
                        Assert(record.extEnd - split.numRight == payload.rOffsets[0]);
                        Partition(record.extStart, record.extEnd - split.numRight, heuristic->binner,
                                  record.start, record.count, inRefs, outRefs, split.bestDim, split.bestPos,
                                  leftRecords[0].centBounds, rightRecords[0].centBounds, split.numLeft, split.numRight);
                    }
                }
                break;
                case Split::Spatial:
                {
                    HSplit *heuristic = (HSplit *)(split.ptr);

                    if (payload.count > 1)
                    {
                        scheduler.ScheduleAndWait(payload.count, 1, [&](u32 jobID) {
                            u32 threadStart = record.start + payload.groupSize * jobID;
                            u32 count       = jobID == payload.count - 1 ? record.start + record.count - threadStart : payload.groupSize;
                            heuristic->Split(mesh, inRefs, outRefs, payload.lOffsets[jobID], payload.rOffsets[jobID],
                                             threadStart, count, split, leftRecords[jobID], rightRecords[jobID],
                                             payload.lCounts[jobID], payload.rCounts[jobID]);
                        });
                    }
                    else
                    {
                        Assert(record.count == payload.groupSize);
                        Assert(record.extStart == payload.lOffsets[0]);
                        Assert(record.extEnd - split.numRight == payload.rOffsets[0]);

                        heuristic->Split(mesh, inRefs, outRefs, payload.lOffsets[0], payload.rOffsets[0],
                                         record.start, record.count, split, leftRecords[0], rightRecords[0],
                                         split.numLeft, split.numRight);
                    }
                }
                break;
            }
            Lane8F32 geomLeft(neg_inf);
            Lane8F32 geomRight(neg_inf);
            Lane8F32 centLeft(neg_inf);
            Lane8F32 centRight(neg_inf);
            for (u32 i = 0; i < payload.count; i++)
            {
                geomLeft  = Max(geomLeft, leftRecords[i].geomBounds);
                centLeft  = Max(centLeft, leftRecords[i].centBounds);
                geomRight = Max(geomRight, rightRecords[i].geomBounds);
                centRight = Max(centRight, rightRecords[i].centBounds);
            }
            outLeft.geomBounds  = geomLeft;
            outLeft.centBounds  = centLeft;
            outRight.geomBounds = geomRight;
            outRight.centBounds = centRight;
            // threadLocalStatistics[GetThreadIndex()].miscF += OS_GetMilliseconds(perCounter);
        }
        u32 numLeft  = split.numLeft;
        u32 numRight = split.numRight;

        f32 weight         = (f32)(numLeft) / (numLeft + numRight);
        u32 remainingSpace = (record.extEnd - record.extStart - numLeft - numRight);
        u32 extSizeLeft    = Min((u32)(remainingSpace * weight), remainingSpace);
        u32 extSizeRight   = remainingSpace - extSizeLeft;

        Assert(numLeft <= record.count);
        Assert(numRight <= record.count);

        outLeft.SetRange(record.extStart, record.extStart, numLeft, record.extStart + numLeft + extSizeLeft);
        outRight.SetRange(outLeft.extEnd, outLeft.extEnd + extSizeRight, numRight, record.extEnd);
        ArenaPopTo(temp.arena, split.allocPos);

        // error check
        {
            for (u32 i = outLeft.start; i < outLeft.End(); i++)
            {
                PrimRef *ref = &outRefs[i];
                f32 min      = ref->min[split.bestDim];
                f32 max      = ref->max[split.bestDim];
                f32 centroid = (max - min) * 0.5f;
                Assert(centroid < split.bestValue);
            }
            for (u32 i = outRight.start; i < outRight.End(); i++)
            {
                PrimRef *ref = &outRefs[i];
                f32 min      = ref->min[split.bestDim];
                f32 max      = ref->max[split.bestDim];
                f32 centroid = (max - min) * 0.5f;
                Assert(centroid >= split.bestValue);
            }
        }

        // NOTE: for the nt stores during partitioning
        _mm_sfence();
    }
};

template <typename Binner, i32 numBins>
Split BinBest(const Bounds8 bounds[3][numBins],
              const Lane4U32 *entryCounts,
              const Lane4U32 *exitCounts,
              const Binner *binner)
{

    Lane4F32 areas[numBins];
    Lane4U32 counts[numBins];
    Lane4U32 currentCount = 0;

    Bounds8 boundsX;
    Bounds8 boundsY;
    Bounds8 boundsZ;
    for (u32 i = 0; i < numBins - 1; i++)
    {
        currentCount += entryCounts[i];
        counts[i] = currentCount;

        boundsX.Extend(bounds[0][i]);
        boundsY.Extend(bounds[1][i]);
        boundsZ.Extend(bounds[2][i]);

        Lane4F32 minX, minY, minZ;
        Lane4F32 maxX, maxY, maxZ;
        Transpose3x3(Extract4<0>(boundsX.v), Extract4<0>(boundsY.v), Extract4<0>(boundsZ.v), minX, minY, minZ);
        Transpose3x3(Extract4<1>(boundsX.v), Extract4<1>(boundsY.v), Extract4<1>(boundsZ.v), maxX, maxY, maxZ);

        Lane4F32 extentX = maxX + minX;
        Lane4F32 extentY = maxY + minY;
        Lane4F32 extentZ = maxZ + minZ;

        areas[i] = FMA(extentX, extentY + extentZ, extentY * extentZ);
    }
    boundsX      = Bounds8();
    boundsY      = Bounds8();
    boundsZ      = Bounds8();
    currentCount = 0;
    Lane4F32 bestSAH(pos_inf);
    Lane4U32 bestPos(0);
    for (u32 i = numBins - 1; i >= 1; i--)
    {
        currentCount += exitCounts[i];

        boundsX.Extend(bounds[0][i]);
        boundsY.Extend(bounds[1][i]);
        boundsZ.Extend(bounds[2][i]);

        Lane4F32 minX, minY, minZ;
        Lane4F32 maxX, maxY, maxZ;
        Transpose3x3(Extract4<0>(boundsX.v), Extract4<0>(boundsY.v), Extract4<0>(boundsZ.v), minX, minY, minZ);
        Transpose3x3(Extract4<1>(boundsX.v), Extract4<1>(boundsY.v), Extract4<1>(boundsZ.v), maxX, maxY, maxZ);

        Lane4F32 extentX = maxX + minX;
        Lane4F32 extentY = maxY + minY;
        Lane4F32 extentZ = maxZ + minZ;

        Lane4F32 rArea = FMA(extentX, extentY + extentZ, extentY * extentZ);

        Lane4F32 sah = FMA(rArea, Lane4F32(currentCount), areas[i - 1] * Lane4F32(counts[i - 1]));

        bestPos = Select(sah < bestSAH, Lane4U32(i), bestPos);
        bestSAH = Select(sah < bestSAH, sah, bestSAH);
    }

    u32 bestDim       = 0;
    f32 bestSAHScalar = pos_inf;
    u32 bestPosScalar = 0;
    for (u32 dim = 0; dim < 3; dim++)
    {
        if (binner->scale[dim][0] == 0) continue;
        if (bestSAH[dim] < bestSAHScalar)
        {
            bestPosScalar = bestPos[dim];
            bestDim       = dim;
            bestSAHScalar = bestSAH[dim];
        }
    }
    f32 bestValue = binner->GetSplitValue(bestPosScalar, bestDim);
    return Split(bestSAHScalar, bestPosScalar, bestDim, bestValue);
}

template <typename Binner, i32 numBins>
Split BinBest(const Bounds8 bounds[3][numBins],
              const Lane4U32 *counts,
              const Binner *binner)
{
    return BinBest(bounds, counts, counts, binner);
}

} // namespace rt
#endif
