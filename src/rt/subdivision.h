#ifndef SUBDIVISION_H
#define SUBDIVISION_H
#include "base.h"
namespace rt
{

struct UntessellatedPatch
{
    int faceID;
    int stitchingStart;
};

struct EdgeInfo
{
    static const int reverseBit = 0x80000000;

    int indexStart;
    u32 edgeFactor;
    int id0, id1;

    int GetFirst(bool reversed) const { return reversed ? id1 : id0; }
    int GetLast(bool reversed) const { return reversed ? id0 : id1; }

    EdgeInfo Opposite() const
    {
        Assert(edgeFactor < 0xffffffff);
        return EdgeInfo{indexStart, edgeFactor | reverseBit, id0, id1};
    }

    int GetEdgeFactor() const { return edgeFactor & ~reverseBit; }
    int GetReversed() const { return (edgeFactor & reverseBit) >> 31; }

    int GetVertexID(u32 edgeStep) const
    {
        u32 ef = GetEdgeFactor();
        Assert(edgeStep >= 0 && edgeStep <= ef);
        return edgeStep == 0 ? id0 : (edgeStep == ef ? id1 : indexStart + edgeStep - 1);
    }
};

struct PatchItr;
struct OpenSubdivPatch
{
    // 3 --e2-- 2
    // |        |
    // e3       e1
    // |        |
    // 0 --e0-- 1

    int faceID;
    int gridIndexStart;
    FixedArray<EdgeInfo, 4> edgeInfo;

    // Generates stitching indices instead of manually having to store them

    OpenSubdivPatch() {}

    PatchItr CreateIterator(int edge) const;
    PatchItr GetUVs(int edge, int id, Vec2f uv[3]) const;

    __forceinline int GetMaxEdgeFactorU() const
    {
        return Max(edgeInfo[0].GetEdgeFactor(), edgeInfo[2].GetEdgeFactor());
    }

    __forceinline int GetMaxEdgeFactorV() const
    {
        return Max(edgeInfo[1].GetEdgeFactor(), edgeInfo[3].GetEdgeFactor());
    }

    __forceinline int GetGridIndex(int u, int v) const
    {
        int edgeU     = GetMaxEdgeFactorU();
        int edgeV     = GetMaxEdgeFactorV();
        int gridIndex = gridIndexStart + v * (edgeU - 1) + u;
        Assert(gridIndex < gridIndexStart + Max(1, (edgeU - 1)) * Max(1, (edgeV - 1)));
        return gridIndex;
    }
};

struct PatchItr
{
    static const FixedArray<int, 2> diff;
    static const FixedArray<int, 2> start;

    static const FixedArray<Vec2f, 4> uvTable;
    static const FixedArray<Vec2i, 4> uvDiffTable;

    const OpenSubdivPatch *patch;
    int edge;
    FixedArray<int, 3> indices;

    // Edge
    int edgeStart;
    int edgeStep;
    int edgeDiff;
    int edgeEnd;
    int maxEdgeFactor;

    // Inner grid
    Vec2i uvStart;
    Vec2i uvEnd;
    int q;
    int newIndex;

    PatchItr() {}
    PatchItr(const OpenSubdivPatch *patch, int edge) : patch(patch), edge(edge)
    {
        const EdgeInfo &edgeInfo = patch->edgeInfo[edge];
        int edgeFactor           = edgeInfo.GetEdgeFactor();
        int reversed             = edgeInfo.GetReversed();

        Assert(reversed == 0 || reversed == 1);

        edgeStep  = start[reversed] * edgeFactor;
        edgeStart = edgeStep;
        edgeDiff  = diff[reversed];
        edgeEnd   = start[!reversed] * edgeFactor;

        int edgeU     = patch->GetMaxEdgeFactorU();
        int edgeV     = patch->GetMaxEdgeFactorV();
        maxEdgeFactor = ((edge & 1) ? edgeV : edgeU);
        q             = maxEdgeFactor - 3 * edgeFactor;

        Vec2i gridStep = Vec2i(Max(edgeU - 2, 0), Max(edgeV - 2, 0));

        uvStart = Vec2i(uvTable[edge]) * gridStep;
        uvEnd   = uvStart + uvDiffTable[edge] * gridStep;
    }
    bool IsNotFinished() { return edgeStep != edgeEnd || uvStart != uvEnd; }
    bool Next()
    {
        if (!IsNotFinished()) return false;

        const EdgeInfo &edgeInfo = patch->edgeInfo[edge];
        indices.Clear();
        if (q >= 0 && uvStart != uvEnd)
        {
            int id0 = patch->GetGridIndex(uvStart[0], uvStart[1]);
            int id1 = edgeInfo.GetVertexID(edgeStep);
            uvStart += uvDiffTable[edge];
            int id2 = patch->GetGridIndex(uvStart[0], uvStart[1]);

            newIndex = 2;
            indices.Push(id0);
            indices.Push(id1);
            indices.Push(id2);
            q -= 2 * edgeInfo.GetEdgeFactor();
        }
        else
        {
            int id0 = edgeInfo.GetVertexID(edgeStep);
            edgeStep += edgeDiff;
            Assert(edgeStep <= edgeInfo.GetEdgeFactor() && edgeStep >= 0);
            int id1 = edgeInfo.GetVertexID(edgeStep);
            int id2 = patch->GetGridIndex(uvStart[0], uvStart[1]);

            newIndex = 1;
            indices.Push(id0);
            indices.Push(id1);
            indices.Push(id2);
            q += 2 * maxEdgeFactor;
        }
        return true;
    }

    __forceinline Vec2f GetGridUV(Vec2i gridLoc, Vec2f edgeDiv) const
    {
        return (Vec2f(gridLoc + Vec2i(1, 1))) * edgeDiv;
    }
    __forceinline Vec2f GetEdgeUV(int edgeStepCount, f32 edgeFactorInv) const
    {
        Vec2f result = Vec2f(uvDiffTable[edge] * edgeStepCount);
        result[edge & 1] *= edgeFactorInv;
        return uvTable[edge] + result;
    }

    void GetUVs(int id, Vec2f uv[3])
    {
        int edgeFactor = patch->edgeInfo[edge].GetEdgeFactor();
        for (int i = 0; i < id; i++)
        {
            ErrorExit(IsNotFinished(), "edge %u factor %u maxu %u maxv %u", edge, edgeFactor,
                      patch->GetMaxEdgeFactorU(), patch->GetMaxEdgeFactorV());
            Next();
        }
        Vec2f edgeDiv(1.f / Max(2, patch->GetMaxEdgeFactorU()),
                      1.f / Max(2, patch->GetMaxEdgeFactorV()));
        f32 edgeFactorInv = 1.f / edgeFactor;

        int edgeStepCount = Abs(edgeStep - edgeStart);
        if (q >= 0 && uvStart != uvEnd)
        {
            uv[0] = GetGridUV(uvStart, edgeDiv);
            uv[1] = GetEdgeUV(edgeStepCount, edgeFactorInv);
            uv[2] = GetGridUV(uvStart + uvDiffTable[edge], edgeDiv);
        }
        else
        {
            Assert(edgeStepCount < edgeFactor);
            uv[0] = GetEdgeUV(edgeStepCount, edgeFactorInv);
            uv[1] = GetEdgeUV(edgeStepCount + 1, edgeFactorInv);
            uv[2] = GetGridUV(uvStart, edgeDiv);
        }
        Next();
    }
};

const FixedArray<int, 2> PatchItr::diff  = {1, -1};
const FixedArray<int, 2> PatchItr::start = {0, 1};

const FixedArray<Vec2f, 4> PatchItr::uvTable = {
    Vec2f(0.f, 0.f),
    Vec2f(1.f, 0.f),
    Vec2f(1.f, 1.f),
    Vec2f(0.f, 1.f),
};
const FixedArray<Vec2i, 4> PatchItr::uvDiffTable = {
    Vec2i(1, 0),
    Vec2i(0, 1),
    Vec2i(-1, 0),
    Vec2i(0, -1),
};

struct BVHPatch
{
    OpenSubdivPatch *patch;
    // Lower left start
    Vec2i uvStart;
    // Upper right end
    Vec2i uvEnd;
    EdgeInfo edgeInfo[4] = {};

    int bitMask;

    static const FixedArray<u32, 4> edgeMasks;

    void SplitEdge(BVHPatch &patch0, BVHPatch &patch1, int edge) const
    {
        bool hasEdge                = bitMask & edgeMasks[edge];
        const EdgeInfo &currentEdge = edgeInfo[edge];

        if (hasEdge)
        {
            int indexStart = currentEdge.indexStart;
            int id0        = currentEdge.id0;
            int id1        = currentEdge.id1;

            u32 edgeFactor    = currentEdge.GetEdgeFactor();
            u32 edgeMid       = edgeFactor / 2;
            int oddEdgeFactor = edgeFactor & 1;
            int midIndex      = currentEdge.indexStart + edgeMid - 1;

            patch0.edgeInfo[edge] = EdgeInfo{indexStart, edgeMid, id0, midIndex};
            patch1.edgeInfo[edge] =
                EdgeInfo{indexStart + (int)edgeMid, edgeMid + oddEdgeFactor, midIndex, id1};

            if (currentEdge.GetReversed())
            {
                Swap(patch0.edgeInfo[edge], patch1.edgeInfo[edge]);
                patch0.edgeInfo[edge] = patch0.edgeInfo[edge].Opposite();
                patch1.edgeInfo[edge] = patch1.edgeInfo[edge].Opposite();
            }
        }
    }

    // i mean naively we just store 3 iterators
    bool Split(BVHPatch &patch0, BVHPatch &patch1) const
    {
        int edgeU = Max(1, Max(edgeInfo[0].GetEdgeFactor(), edgeInfo[2].GetEdgeFactor()));
        int edgeV = Max(1, Max(edgeInfo[1].GetEdgeFactor(), edgeInfo[3].GetEdgeFactor()));

        Vec2i diff   = uvEnd - uvStart;
        int numQuads = diff[0] * diff[1];
        // TODO: this is wrong but maybe it's ok
        int numTriangles = (edgeU - 1) + Max(edgeInfo[0].GetEdgeFactor() - 1, 0) +
                           (edgeV - 1) + Max(edgeInfo[1].GetEdgeFactor() - 1, 0) +
                           (edgeU - 1) + Max(edgeInfo[2].GetEdgeFactor() - 1, 0) +
                           (edgeV - 1) + Max(edgeInfo[3].GetEdgeFactor() - 1, 0) +
                           2 * numQuads;

        if (numTriangles <= 8 || (edgeU == 1 && edgeV == 1)) return false;

        patch0.patch = patch;
        patch1.patch = patch;

        // Vertical split
        if (edgeU + diff[0] > edgeV + diff[1])
        {
            SplitEdge(patch0, patch1, 0);
            SplitEdge(patch1, patch0, 2);

            patch0.uvStart = uvStart;
            patch0.uvEnd   = Vec2i((uvStart[0] + uvEnd[0]) / 2, uvEnd[1]);
            // Unset 2nd for edge 1
            patch0.bitMask     = bitMask & ~edgeMasks[1];
            patch0.edgeInfo[3] = edgeInfo[3];
            patch0.edgeInfo[1] = {};

            patch1.uvStart = Vec2i((uvStart[0] + uvEnd[0]) / 2, uvStart[1]);
            patch1.uvEnd   = uvEnd;
            // Unset 4th bit for edge 3
            patch1.bitMask     = bitMask & ~edgeMasks[3];
            patch1.edgeInfo[1] = edgeInfo[1];
            patch1.edgeInfo[3] = {};
        }
        // Horizontal split
        else
        {
            SplitEdge(patch0, patch1, 1);
            SplitEdge(patch1, patch0, 3);

            patch0.uvStart = uvStart;
            patch0.uvEnd   = Vec2i(uvEnd[0], (uvStart[1] + uvEnd[1]) / 2);
            // Unset 2nd for edge 1
            patch0.bitMask     = bitMask & ~edgeMasks[2];
            patch0.edgeInfo[0] = edgeInfo[0];
            patch0.edgeInfo[2] = {};

            patch1.uvStart = Vec2i(uvStart[0], (uvStart[1] + uvEnd[1]) / 2);
            patch1.uvEnd   = uvEnd;
            // Unset 1st bit for edge 0
            patch1.bitMask     = bitMask & ~edgeMasks[0];
            patch1.edgeInfo[2] = edgeInfo[2];
            patch1.edgeInfo[0] = {};
        }
        return true;
    }

    PatchItr CreateIterator(int edge) const
    {
        PatchItr itr;
        itr.patch = patch;
        itr.edge  = edge;

        int bit = (bitMask >> edge) & 1;
        if (!bit)
        {
            itr.uvStart  = Vec2i(0, 0);
            itr.uvEnd    = Vec2i(0, 0);
            itr.edgeStep = 0;
            itr.edgeEnd  = 0;
            return itr;
        }

        const EdgeInfo &currentEdge = edgeInfo[edge];
        int edgeFactor              = currentEdge.GetEdgeFactor();
        int reversed                = currentEdge.GetReversed();

        Assert(reversed == 0 || reversed == 1);
        itr.edgeStep  = PatchItr::start[reversed] * edgeFactor;
        itr.edgeStart = itr.edgeStep;
        itr.edgeDiff  = PatchItr::diff[reversed];
        itr.edgeEnd   = PatchItr::start[!reversed] * edgeFactor;

        int edgeU         = patch->GetMaxEdgeFactorU();
        int edgeV         = patch->GetMaxEdgeFactorV();
        itr.maxEdgeFactor = ((edge & 1) ? edgeV : edgeU);

        // this is conditional on whether the grid was split

        int prevEdgeExists = (bitMask >> ((edge + 3) & 3));
        itr.q = itr.maxEdgeFactor - edgeFactor - (prevEdgeExists ? 0 : 2 * edgeFactor);

        Vec2i gridStep = uvEnd - uvStart;

        // TODO: this is wrong
        itr.uvStart[0] = edge & 1 ? uvEnd[0] : uvStart[0];
        itr.uvStart[1] = edge > 1 ? uvEnd[1] : uvStart[1];
        itr.uvEnd[0]   = edge > 1 ? uvStart[0] : uvEnd[0];
        itr.uvEnd[1]   = edge & 1 ? uvStart[1] : uvEnd[1];

        return itr;
    }
    PatchItr GetUVs(int edge, int id, Vec2f uv[3]) const
    {
        PatchItr itr = CreateIterator(edge);
        itr.GetUVs(id, uv);
        return itr;
    }
};

const FixedArray<u32, 4> BVHPatch::edgeMasks = {
    0x1,
    0x2,
    0x4,
    0x8,
};

// TODO: can have stitching quads (for edges that match the
// maximum u/v edge rate), by specifying an edge index start, grid index start,
// and a grid offset step size. this may or may not be worth it
struct OpenSubdivMesh
{
    StaticArray<Vec3f> vertices;
    StaticArray<Vec3f> normals;
    StaticArray<int> stitchingIndices;

    StaticArray<UntessellatedPatch> untessellatedPatches;
    StaticArray<OpenSubdivPatch> patches;

    StaticArray<BVHPatch> bvhPatches;

    u32 GetNumFaces() const { return untessellatedPatches.Length() + patches.Length(); }
};

OpenSubdivMesh *AdaptiveTessellation(Arena *arena, ScenePrimitives *scene,
                                     const Mat4 &NDCFromCamera, int screenHeight,
                                     struct Mesh *controlMeshes, u32 numMeshes);

} // namespace rt
#endif
