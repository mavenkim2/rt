#ifndef BVH_AOS_H
#define BVH_AOS_H
namespace rt
{
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
    void Bin(PrimRef *data, u32 *refs, u32 start, u32 count)
    {
        u32 alignedCount = count - count % LANE_WIDTH;
        u32 i            = start;

        Lane8F32 prevLanes[8];
        alignas(32) u32 prevBinIndices[3][8];
        if (count >= LANE_WIDTH)
        {
            Lane8F32 temp[6];
            Transpose8x6(data[refs[i]].m256, data[refs[i] + 1].m256, data[refs[i] + 2].m256, data[refs[i] + 3].m256,
                         data[refs[i] + 4].m256, data[refs[i] + 5].m256, data[refs[i] + 6].m256, data[refs[i] + 7].m256,
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
                prevLanes[c] = data[refs[i] + c].m256;
            }
            i += LANE_WIDTH;
        }
        for (; i < start + alignedCount; i += LANE_WIDTH)
        {
            Lane8F32 temp[6];

            Transpose8x6(data[refs[i]].m256, data[refs[i] + 1].m256, data[refs[i] + 2].m256, data[refs[i] + 3].m256,
                         data[refs[i] + 4].m256, data[refs[i] + 5].m256, data[refs[i] + 6].m256, data[refs[i] + 7].m256,
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
                prevLanes[c] = data[refs[i] + c].m256;
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
            Lane4F32 low      = Extract4<0>(data[refs[i]].m256);
            Lane4F32 hi       = Extract4<1>(data[refs[i]].m256);
            Lane4F32 centroid = (hi - low) * 0.5f;
            u32 indices[]     = {
                binner->Bin(centroid[0], 0),
                binner->Bin(centroid[1], 1),
                binner->Bin(centroid[2], 2),
            };
            for (u32 dim = 0; dim < 3; dim++)
            {
                u32 bin = indices[dim];
                bins[dim][bin].Extend(data[refs[i]].m256);
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

    void Bin(TriangleMesh *mesh, PrimRef *data, u32 *refs, u32 start, u32 count)
    {
        u32 binCounts[3][numBins]                                 = {};
        alignas(32) u32 binIndexStart[3][numBins][2 * LANE_WIDTH] = {};
        u32 faceIndices[3][numBins][2 * LANE_WIDTH]               = {};

        u32 i            = start;
        u32 alignedCount = count - count % LANE_WIDTH;
        f32 totalTime    = 0.f;

        Lane8F32 lanes[6];

        for (; i < start + alignedCount; i += LANE_WIDTH)
        {
            Transpose8x6(data[refs[i]].m256, data[refs[i + 1]].m256, data[refs[i + 2]].m256, data[refs[i + 3]].m256,
                         data[refs[i + 4]].m256, data[refs[i + 5]].m256, data[refs[i + 6]].m256, data[refs[i + 7]].m256,
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
                            bins[dim][indexMin].Extend(data[refs[i + b]].m256);
                        }
                        break;
                        default:
                        {
                            bitMask[dim] |= (1 << diff);
                            faceIndices[dim][diff][binCounts[dim][diff]]   = data[refs[i + b]].primID;
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
            PrimRef *ref = &data[refs[i]];

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
    void Split(TriangleMesh *mesh, PrimRef *data, const u32 *inRefs, u32 *outRefs, u32 splitOffset, u32 outLStart, u32 outRStart,
               u32 start, u32 count, Split split, RecordAOSSplits &outLeft, RecordAOSSplits &outRight)
    {
        u32 dim                         = split.bestDim;
        u32 alignedCount                = count - count % LANE_WIDTH;
        u32 refIDQueue[LANE_WIDTH * 2]  = {};
        u32 faceIDQueue[LANE_WIDTH * 2] = {};
        Lane8F32 masks[2]               = {Lane8F32::Mask(false), Lane8F32::Mask(true)};

        f32 negBestValue = -split.bestValue;
        u32 i            = start;
        u32 v            = (dim + 1) % 3;
        u32 w            = (v + 1) % 3;
        Lane8F32 lanes[8];

        u32 splitCount = 0;

        Lane8F32 geomLeft;
        Lane8F32 geomRight;

        // Bounds8F32
        Lane8F32 centLeft;
        Lane8F32 centRight;

        u32 writeLocs[2] = {outLStart, outRStart};
        for (; i < start + alignedCount; i += LANE_WIDTH)
        {
            // Transposes the lanes in order to compute the bounds for the next generation
            Transpose8x8(data[inRefs[i]].m256, data[inRefs[i + 1]].m256, data[inRefs[i + 2]].m256, data[inRefs[i + 3]].m256,
                         data[inRefs[i + 4]].m256, data[inRefs[i + 5]].m256, data[inRefs[i + 6]].m256, data[inRefs[i + 7]].m256,
                         lanes[0], lanes[1], lanes[2], lanes[3], lanes[4], lanes[5], lanes[6], lanes[7]);

            // See if the primitive needs to be split
            // Only add the primitive to the left and right bounds if it doesn't need to be split
            Lane8U32 faceIDs = AsUInt(lanes[7]);

            Lane8F32 isFullyLeftV  = lanes[dim + 4] < split.bestValue;
            Lane8F32 isFullyRightV = lanes[dim] <= negBestValue;

            Lane8F32 centroid  = (lanes[dim + 4] - lanes[dim]) * 0.5f;
            Lane8F32 centroidV = (lanes[v + 4] - lanes[v]) * 0.5f;
            Lane8F32 centroidW = (lanes[w + 4] - lanes[w]) * 0.5f;

            Transpose3x8(centroid, centroidV, centroidW,
                         lanes[0], lanes[1], lanes[2], lanes[3], lanes[4], lanes[5], lanes[6], lanes[7]);

            u32 lMaskBits = Movemask(isFullyLeftV);
            u32 rMaskBits = Movemask(isFullyRightV);
            // Extend next generation bounds
            for (u32 b = 0; b < LANE_WIDTH; b++)
            {
                u32 isFullyLeft  = (lMaskBits >> i) & 1;
                u32 isFullyRight = (rMaskBits >> i) & 1;

                geomLeft                           = MaskMax(masks[isFullyLeft], geomLeft, data[inRefs[i + b]].m256);
                geomRight                          = MaskMax(masks[isFullyRight], geomRight, data[inRefs[i + b]].m256);
                centLeft                           = MaskMax(masks[isFullyLeft], centLeft, lanes[b] ^ signFlipMask);
                centRight                          = MaskMax(masks[isFullyRight], centRight, lanes[b] ^ signFlipMask);
                outRefs[writeLocs[isFullyRight]++] = inRefs[i + b];
            }

            // Centroid bounds for next generation

            // Store primitives that need to be split
            Lane8U32 refIDs = Lane8U32::LoadU(&inRefs[i]);
            u32 splitMask   = (~(lMaskBits | rMaskBits)) & 0xff;
            Lane8U32::StoreU(refIDQueue + splitCount, MaskCompress(splitMask, refIDs));
            Lane8U32::StoreU(faceIDQueue + splitCount, MaskCompress(splitMask, faceIDs));
            splitCount += PopCount(splitMask);

            if (splitCount >= LANE_WIDTH)
            {
                splitCount -= LANE_WIDTH;

                Bounds8 boundsLeft[LANE_WIDTH];
                Bounds8 boundsRight[LANE_WIDTH];

                Lane8F32 cL;
                Lane8F32 cR;
                Triangle8 tri = Triangle8::Load(mesh, dim, faceIDQueue + splitCount);
                ClipTriangle(mesh, dim, faceIDQueue + splitCount, tri, split.bestValue, boundsLeft, boundsRight, cL, cR);
                centLeft  = Max(centLeft, cL);
                centRight = Max(centRight, cR);

                // TODO: there could be a false sharing problem here. not a major one, but these writes invalidate
                // any previously cached PrimRefs, so all reads at the beginning of the outer loop are a cache miss
                for (u32 queueIndex = 0; queueIndex < LANE_WIDTH; queueIndex++)
                {
                    geomLeft        = Max(geomLeft, boundsLeft[queueIndex].v);
                    geomRight       = Max(geomRight, boundsRight[queueIndex].v);
                    const u32 refID = refIDQueue[splitCount + queueIndex];

                    Lane8F32::Store(&data[refID].m256, boundsLeft[queueIndex].v);
                    Lane8F32::Store(&data[splitOffset].m256, boundsRight[queueIndex].v);
                    outRefs[writeLocs[1]++] = splitOffset;
                    splitOffset++;
                }
            }
        }
        for (; i < start + count; i++)
        {
            u32 ref          = inRefs[i];
            PrimRef *primRef = &data[ref];
            Lane4F32 min     = Extract4<0>(primRef->m256);
            Lane4F32 max     = Extract4<1>(primRef->m256);

            bool isFullyLeft  = primRef->max[dim] < split.bestValue;
            bool isFullyRight = primRef->min[dim] <= -split.bestValue;
            Lane4F32 centroid = (max - min) * 0.5f;
            bool isSplit      = (~(isFullyLeft | isFullyRight)) & 0xff;

            Lane8F32 c(-centroid, centroid);
            refIDQueue[splitCount]  = ref;
            faceIDQueue[splitCount] = primRef->primID;
            splitCount += isSplit;
            geomLeft  = MaskMax(masks[isFullyLeft], geomLeft, primRef->m256);
            geomRight = MaskMax(masks[isFullyRight], geomRight, primRef->m256);

            centLeft  = MaskMax(masks[isFullyLeft], centLeft, c);
            centRight = MaskMax(masks[isFullyRight], centRight, c);

            outRefs[writeLocs[isFullyRight]++] = ref;
        }
        // Flush the queue
        u32 remainingCount = splitCount;
        const u32 numIters = (remainingCount + 7) >> 3;
        for (u32 remaining = 0; remaining < numIters; remaining++)
        {
            u32 qStart   = remaining * LANE_WIDTH;
            u32 numPrims = Min(remainingCount, LANE_WIDTH);
            Bounds8 boundsLeft[LANE_WIDTH];
            Bounds8 boundsRight[LANE_WIDTH];
            Lane8F32 cL;
            Lane8F32 cR;
            Triangle8 tri = Triangle8::Load(mesh, dim, faceIDQueue + qStart);

            ClipTriangle(mesh, dim, faceIDQueue + qStart, tri, split.bestValue, boundsLeft, boundsRight, cL, cR);

            centLeft  = Max(centLeft, cL);
            centRight = Max(centRight, cR);
            for (u32 queueIndex = 0; queueIndex < numPrims; queueIndex++)
            {
                geomLeft        = Max(geomLeft, boundsLeft[queueIndex].v);
                geomRight       = Max(geomRight, boundsRight[queueIndex].v);
                const u32 refID = refIDQueue[qStart + queueIndex];

                Lane8F32::Store(&data[refID].m256, boundsLeft[queueIndex].v);
                Lane8F32::Store(&data[splitOffset].m256, boundsRight[queueIndex].v);
                outRefs[writeLocs[1]++] = splitOffset;
                splitOffset++;
            }

            remainingCount -= LANE_WIDTH;
        }
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
// static const f32 sbvhAlpha = 1e-5;
// template <i32 numObjectBins = 32, i32 numSpatialBins = 16>
// struct HeuristicSpatialSplits
// {
//     using Record = RecordSOASplits;
//     using HSplit = HeuristicAOSSplitBinning<numSpatialBins>;
//     using OBin   = HeuristicAOSObjectBinning<numObjectBins>;
//
//     TriangleMesh *mesh;
//     f32 rootArea;
//     PrimRef *data;
//     u32 *refs[2];
//
//     struct alignas(CACHE_LINE_SIZE) ThreadSplits
//     {
//         u32 currentOffset;
//         u32 currentCount;
//     };
//
//     // TODO: hardcoded
//     ThreadSplits threadSplits[MAX_THREAD_COUNT];
//
//     alignas(CACHE_LINE_SIZE) std::atomic<u32> numSplits{0};
//
//     HeuristicSpatialSplits() {}
//     HeuristicSpatialSplits(PrimRef *data, TriangleMesh *mesh, u32 *refs0, u32 *refs1, f32 rootArea)
//         : data(data), mesh(mesh), refs{ref0s, refs1}, rootArea(rootArea) {}
//
//     Split Bin(const Record &record, u32 primary, u32 blockSize = 1)
//     {
//         // Object splits
//         TempArena temp    = ScratchStart(0, 0);
//         u64 popPos        = ArenaPos(temp.arena);
//         temp.arena->align = 32;
//
//         // Stack allocate the heuristics since they store the centroid and geom bounds we'll need later
//         ObjectBinner<numObjectBins> *objectBinner =
//             PushStructConstruct(temp.arena, ObjectBinner<numObjectBins>)(record.centBounds.ToBounds());
//         OBin *objectBinHeuristic = PushStructNoZero(temp.arena, OBin);
//         ParallelForOutput parallelOutput;
//         if (record.range.count > PARALLEL_THRESHOLD)
//         {
//             const u32 groupSize = PARALLEL_THRESHOLD;
//             parallelOutput      = ParallelFor<OBin>(
//                 record.range.start, record.range.count, groupSize,
//                 [&](OBin &binner, u32 start, u32 count) { binner.Bin(data, refs[primary], start, count); },
//                 objectBinner);
//             Reduce<OBin>(
//                 *objectBinHeuristic, parallelOutput,
//                 [&](OBin &l, const OBin &r) { l.Merge(r); },
//                 objectBinner);
//         }
//         else
//         {
//             *objectBinHeuristic = OBin(objectBinner);
//             objectBinHeuristic->Bin(data, refs[primary], record.range.start, record.range.count);
//         }
//         struct Split objectSplit = BinBest(objectBinHeuristic->bins, objectBinHeuristic->counts, objectBinner);
//         objectSplit.type         = Split::Object;
//
//         Bounds8 geomBoundsL;
//         for (u32 i = 0; i < objectSplit.bestPos; i++)
//         {
//             geomBoundsL.Extend(objectBinHeuristic->bins[objectSplit.bestDim][i]);
//         }
//         Bounds8 geomBoundsR;
//         for (u32 i = objectSplit.bestPos; i < numObjectBins; i++)
//         {
//             geomBoundsR.Extend(objectBinHeuristic->bins[objectSplit.bestDim][i]);
//         }
//
//         f32 lambda = HalfArea(Intersect(geomBoundsL, geomBoundsR));
//         if (lambda > sbvhAlpha * rootArea)
//         {
//             // Spatial splits
//             SplitBinner<numSpatialBins> *splitBinner =
//                 PushStructConstruct(temp.arena, SplitBinner<numSpatialBins>)(record.geomBounds.ToBounds());
//
//             HSplit *splitHeuristic = PushStructNoZero(temp.arena, HSplit);
//             ParallelForOutput splitOutput;
//             if (record.range.count > PARALLEL_THRESHOLD)
//             {
//                 const u32 groupSize = 4 * 1024; // PARALLEL_THRESHOLD;
//                 splitOutput         = ParallelFor<HSplit>(
//                     record.range.start, record.range.count, groupSize,
//                     [&](HSplit &binner, u32 start, u32 count) { binner.Bin(mesh, soa, start, count); },
//                     splitBinner);
//                 Reduce<HSplit>(
//                     *splitHeuristic, splitOutput,
//                     [&](HSplit &l, const HSplit &r) { l.Merge(r); },
//                     splitBinner);
//             }
//             else
//             {
//                 *splitHeuristic = HSplit(splitBinner);
//                 splitHeuristic->Bin(mesh, soa, record.range.start, record.range.count);
//             }
//             struct Split spatialSplit = BinBest(splitHeuristic->bins,
//                                                 splitHeuristic->entryCounts, splitHeuristic->exitCounts, splitBinner);
//             spatialSplit.type         = Split::Spatial;
//             u32 lCount                = 0;
//             for (u32 i = 0; i < spatialSplit.bestPos; i++)
//             {
//                 lCount += splitHeuristic->entryCounts[i][spatialSplit.bestDim];
//             }
//             u32 rCount = 0;
//             for (u32 i = spatialSplit.bestPos; i < numSpatialBins; i++)
//             {
//                 rCount += splitHeuristic->exitCounts[i][spatialSplit.bestDim];
//             }
//             u32 totalNumSplits = lCount + rCount - record.range.count;
//             if (spatialSplit.bestSAH < objectSplit.bestSAH && totalNumSplits <= record.range.ExtSize())
//             {
//                 u32 allocOffset = numSplits.fetch_add(totalNumSplits, std::memory_order_acq_rel);
//                 // TODO: assert less than max
//
//                 u32 *lOffsets = PushArrayNoZero(temp.arena, u32, output.num);
//                 u32 *rOffsets = PushArrayNoZero(temp.arena, u32, output.num);
//
//                 u32 *splitOffsets = PushArrayNoZero(temp.arena, u32, output.num);
//                 u32 *refOffsets   = PushArrayNoZero(temp.arena, u32, output.num);
//                 u32 lOffset       = record.range.start;
//                 u32 rOffset       = record.range.ExtEnd();
//                 u32 refOffset     = record.range.End();
//                 // TODO IMPORTANT: have to handle the left right shenanigans
//                 for (u32 i = 0; i < splitOutput.num; i++)
//                 {
//                     HSplit *h       = ((HSplit *)splitOutput.out)[i];
//                     lOffsets[i]     = lOffset;
//                     u32 entryCounts = 0;
//                     u32 exitCounts  = 0;
//                     for (u32 bin = 0; bin < split.bestPos; bin++)
//                     {
//                         entryCounts += h->entryCounts[bin][split.bestDim];
//                     }
//                     for (u32 bin = split.bestDim; bin < numSpatialBins; bin++)
//                     {
//                         exitCounts += h->exitCounts[bin][split.bestDim];
//                     }
//                     u32 groupSize      = i == output.num - 1
//                                              ? record.range.End() - (record.range.start + output.groupSize * (output.num - 1))
//                                              : output.groupSize;
//                     u32 taskSplitCount = entryCounts + exitCounts - groupSize;
//                     splitOffsets[i]    = allocOffset;
//                     refOffsets[i]      = refOffset;
//
//                     refOffset += taskSplitCount;
//                     allocOffset += taskSplitCount;
//
//                     lOffset += entryCounts;
//                     rOffset -= exitCounts;
//                     rOffsets[i] = rOffset;
//                 }
//                 spatialSplit.splitPayload = SplitPayload(splitOffsets, refOffsets, output.num, output.groupSize);
//                 spatialSplit.ptr          = (void *)splitHeuristic;
//                 spatialSplit.allocPos     = popPos;
//
//                 EndReduce(splitOutput);
//                 return spatialSplit;
//             }
//         }
//
//         objectSplit.pOutput  = parallelOutput;
//         objectSplit.ptr      = (void *)objectBinHeuristic;
//         objectSplit.allocPos = popPos;
//         EndReduce(parallelOutput);
//         return objectSplit;
//     }
//     void FlushState(struct Split split)
//     {
//         TempArena temp = ScratchStart(0, 0);
//         ArenaPopTo(temp.arena, split.allocPos);
//     }
//     void Split(struct Split split, const Record &record, Record &outLeft, Record &outRight)
//     {
//         // NOTE: Split must be called from the same thread as Bin
//         TempArena temp = ScratchStart(0, 0);
//
//         const ExtRange &range = record.range;
//         PrimDataSOA *data     = soa;
//         u32 mid;
//         u32 splitCount = 0;
//
//         if (split.bestSAH == f32(pos_inf))
//         {
//             mid = record.range.start + (record.range.count / 2);
//             Bounds geomLeft;
//             Bounds centLeft;
//             Bounds geomRight;
//             Bounds centRight;
//             for (u32 i = record.range.start; i < record.range.End(); i++) // mid; i++)
//             {
//                 f32 min[3] = {
//                     data->minX[i],
//                     data->minY[i],
//                     data->minZ[i],
//                 };
//                 f32 max[3] = {
//                     data->maxX[i],
//                     data->maxY[i],
//                     data->maxZ[i],
//                 };
//                 f32 centroid[3] = {
//                     (max[0] - min[0]) * 0.5f,
//                     (max[1] - min[1]) * 0.5f,
//                     (max[2] - min[2]) * 0.5f,
//                 };
//                 if (i < mid)
//                 {
//                     geomLeft.Extend(Lane4F32(min[0], min[1], min[2], 0.f), Lane4F32(max[0], max[1], max[2], 0.f));
//                     centLeft.Extend(Lane4F32(centroid[0], centroid[1], centroid[2], 0.f));
//                 }
//                 else
//                 {
//                     geomRight.Extend(Lane4F32(min[0], min[1], min[2], 0.f), Lane4F32(max[0], max[1], max[2], 0.f));
//                     centRight.Extend(Lane4F32(centroid[0], centroid[1], centroid[2], 0.f));
//                 }
//             }
//             outLeft.geomBounds.FromBounds(geomLeft);
//             outLeft.centBounds.FromBounds(centLeft);
//             outRight.geomBounds.FromBounds(geomRight);
//             outRight.centBounds.FromBounds(centRight);
//         }
//         else
//         {
//             // TODO:
//             // make the record 64 byte aligned, as follows
//             // struct
//             // {
//             //    f32 minX, minY, minZ; // geom
//             //    u32 extStart;
//             //    f32 maxX, maxY, maxZ;
//             //    u32 start;
//             //    f32 minX, minY, minZ; // cent
//             //    u32 end;
//             //    f32 maxX, maxY, maxZ;
//             //    u32 extEnd;
//             // }
//
//             // PerformanceCounter perCounter = OS_StartCounter();
//             switch (split.type)
//             {
//                 case Split::Object:
//                 {
//                     OBin *heuristic = (OBin *)(split.ptr);
//                     mid             = PartitionParallelCentroids(split, record.range, soa);
//                 }
//                 break;
//                 case Split::Spatial:
//                 {
//                     HSplit *heuristic = (HSplit *)(split.ptr);
//                     mid               = heuristic->Split(mesh, soa, record.range,
//                                                          split, outLeft, outRight, &splitCount);
//                 }
//                 break;
//             }
//             // threadLocalStatistics[GetThreadIndex()].miscF += OS_GetMilliseconds(perCounter);
//         }
//
//         OBin *oHeuristic   = (OBin *)(split.ptr);
//         HSplit *sHeuristic = (HSplit *)(split.ptr);
//
//         u32 numLeft  = mid - range.start;
//         u32 numRight = range.End() - mid + splitCount;
//
//         CentGeomLane8F32 left;
//
//         f32 weight         = (f32)(numLeft) / (numLeft + numRight);
//         u32 remainingSpace = (range.ExtSize() - splitCount);
//         u32 extSizeLeft    = Min((u32)(remainingSpace * weight), remainingSpace);
//         u32 extSizeRight   = remainingSpace - extSizeLeft;
//
//         Assert(numLeft <= record.range.count);
//         Assert(numRight <= record.range.count);
//
//         outLeft.range    = ExtRange(range.start, numLeft, range.start + numLeft + extSizeLeft);
//         outRight.range   = ExtRange(outLeft.range.extEnd, numRight, range.extEnd);
//         u32 rightExtSize = outRight.range.ExtSize();
//         Assert(rightExtSize == extSizeRight);
//
//         ArenaPopTo(temp.arena, split.allocPos);
//     }
// };

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

        Lane4F32 minX = Extract4<0>(boundsX.v);
        Lane4F32 maxX = Extract4<1>(boundsX.v);
        Lane4F32 minY = Extract4<0>(boundsY.v);
        Lane4F32 maxY = Extract4<1>(boundsY.v);
        Lane4F32 minZ = Extract4<0>(boundsZ.v);
        Lane4F32 maxZ = Extract4<1>(boundsZ.v);

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

        Lane4F32 minX = Extract4<0>(boundsX.v);
        Lane4F32 maxX = Extract4<1>(boundsX.v);
        Lane4F32 minY = Extract4<0>(boundsY.v);
        Lane4F32 maxY = Extract4<1>(boundsY.v);
        Lane4F32 minZ = Extract4<0>(boundsZ.v);
        Lane4F32 maxZ = Extract4<1>(boundsZ.v);

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