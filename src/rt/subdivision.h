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

    OpenSubdivPatch() {}

    PatchItr CreateIterator(int edge) const;

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

enum class CatClarkTriangleType
{
    Untess,
    TessStitching,
    TessGrid,

    Max,
};
static_assert((u32)CatClarkTriangleType::Max <= 4, "enum is too large\n");

u32 CreatePatchID(CatClarkTriangleType type, int meta, int index)
{
    Assert(index >= 0 && index < 0x0fffffff);
    Assert(meta >= 0 && meta < 4);
    return ((u32)type << 30) | (meta << 28) | index;
}

CatClarkTriangleType GetPatchType(u32 val) { return CatClarkTriangleType(val >> 30); }
int GetTriangleIndex(u32 val) { return val & 0x0fffffff; }
int GetMeta(u32 val) { return (val >> 28) & 0x3; }

struct UVGrid
{
    Vec2<u8> uvStart;
    Vec2<u8> uvEnd;

    static UVGrid Compress(const Vec2i &start, const Vec2i &end)
    {
        UVGrid grid;
        grid.uvStart = Vec2<u8>(SafeTruncateU32ToU8(start[0]), SafeTruncateU32ToU8(start[1]));
        grid.uvEnd   = Vec2<u8>(SafeTruncateU32ToU8(end[0]), SafeTruncateU32ToU8(end[1]));
        return grid;
    }
    void Decompress(Vec2i &start, Vec2i &end) const
    {
        start[0] = uvStart[0];
        start[1] = uvStart[1];
        end[0]   = uvEnd[0];
        end[1]   = uvEnd[1];
    }
};

struct BVHEdge
{
    struct OpenSubdivPatch *patch;
    // Bit trail represents whether to advance on edge (0) or on grid (1)
    // for stitching index
    int bitTrail;

    UVGrid grid;

    int edgeStep;
    int edgeEnd;
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

    // Bit trail
    int bitTrail = 0;
    int steps    = 0;

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

    PatchItr(const OpenSubdivPatch *patch, const BVHEdge *bvhEdge, int edge)
        : patch(patch), edge(edge)
    {
        const EdgeInfo &edgeInfo = patch->edgeInfo[edge];
        int edgeFactor           = edgeInfo.GetEdgeFactor();
        int reversed             = edgeInfo.GetReversed();

        edgeStart = start[reversed] * edgeFactor;
        edgeStep  = bvhEdge->edgeStep;
        edgeDiff  = diff[reversed];
        edgeEnd   = bvhEdge->edgeEnd;

        bvhEdge->grid.Decompress(uvStart, uvEnd);

        bitTrail = bvhEdge->bitTrail;
    }

    int GetBitTrail()
    {
        Assert(steps <= 32);
        int result = bitTrail;
        bitTrail   = 0;
        steps      = 0;
        return result;
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

            indices.Push(id0);
            indices.Push(id1);
            indices.Push(id2);
            q += 2 * maxEdgeFactor;

            bitTrail |= (1 << steps);
        }
        steps++;
        return true;
    }

    bool EdgeItrNext()
    {
        if (!IsNotFinished()) return false;
        const EdgeInfo &edgeInfo = patch->edgeInfo[edge];
        indices.Clear();
        // Advance along edge
        if (!(bitTrail & 1))
        {
            int id0 = patch->GetGridIndex(uvStart[0], uvStart[1]);
            int id1 = edgeInfo.GetVertexID(edgeStep);
            uvStart += uvDiffTable[edge];
            int id2 = patch->GetGridIndex(uvStart[0], uvStart[1]);

            indices.Push(id0);
            indices.Push(id1);
            indices.Push(id2);
        }
        else
        {
            int id0 = edgeInfo.GetVertexID(edgeStep);
            edgeStep += edgeDiff;
            int id1 = edgeInfo.GetVertexID(edgeStep);
            int id2 = patch->GetGridIndex(uvStart[0], uvStart[1]);

            indices.Push(id0);
            indices.Push(id1);
            indices.Push(id2);
        }
        bitTrail >>= 1;
        return true;
    }

    __forceinline Vec2f GetGridUV(Vec2i gridLoc, Vec2f edgeDiv) const
    {
        return (Vec2f(gridLoc + Vec2i(1, 1))) * edgeDiv;
    }
    __forceinline Vec2f GetEdgeUV(int edgeStepCount, f32 edgeFactorInv) const
    {
        return uvTable[edge] + Vec2f(uvDiffTable[edge] * edgeStepCount) * edgeFactorInv;
    }

    void GetUV(int id, Vec2f uv[3])
    {
        int edgeFactor = patch->edgeInfo[edge].GetEdgeFactor();
        for (int i = 0; i < id; i++)
        {
            ErrorExit(IsNotFinished(), "edge %u factor %u maxu %u maxv %u", edge, edgeFactor,
                      patch->GetMaxEdgeFactorU(), patch->GetMaxEdgeFactorV());
            EdgeItrNext();
        }
        Vec2f edgeDiv(1.f / Max(2, patch->GetMaxEdgeFactorU()),
                      1.f / Max(2, patch->GetMaxEdgeFactorV()));
        f32 edgeFactorInv = 1.f / edgeFactor;

        int edgeStepCount = Abs(edgeStep - edgeStart);
        if (!(bitTrail & 1))
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
        EdgeItrNext();
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
    // TODO: reference this somehow implicitly
    OpenSubdivPatch *patch;

    UVGrid grid;

    // i mean naively we just store 3 iterators
    bool Split(BVHPatch &patch0, BVHPatch &patch1) const
    {
        Vec2<u8> uvStart = grid.uvStart;
        Vec2<u8> uvEnd   = grid.uvEnd;
        Vec2i diff       = uvEnd - uvStart;
        int numQuads     = diff[0] * diff[1];
        int numTriangles = 2 * numQuads;

        if (numTriangles <= 8) return false;

        patch0.patch = patch;
        patch1.patch = patch;

        // Vertical split
        if (diff[0] > diff[1])
        {
            if (diff[0] == 1) return false;
            patch0.grid.uvStart = uvStart;
            patch0.grid.uvEnd   = Vec2<u8>((uvStart[0] + uvEnd[0]) / 2, uvEnd[1]);

            patch1.grid.uvStart = Vec2<u8>((uvStart[0] + uvEnd[0]) / 2, uvStart[1]);
            patch1.grid.uvEnd   = uvEnd;
        }
        // Horizontal split
        else
        {
            if (diff[1] == 1) return false;
            patch0.grid.uvStart = uvStart;
            patch0.grid.uvEnd   = Vec2<u8>(uvEnd[0], (uvStart[1] + uvEnd[1]) / 2);

            patch1.grid.uvStart = Vec2<u8>(uvStart[0], (uvStart[1] + uvEnd[1]) / 2);
            patch1.grid.uvEnd   = uvEnd;
        }

        Assert(patch0.grid.uvStart[0] != patch0.grid.uvEnd[0]);
        Assert(patch0.grid.uvStart[1] != patch0.grid.uvEnd[1]);
        Assert(patch1.grid.uvStart[0] != patch1.grid.uvEnd[0]);
        Assert(patch1.grid.uvStart[1] != patch1.grid.uvEnd[1]);
        return true;
    }
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
    StaticArray<BVHEdge> bvhEdges;

    u32 GetNumFaces() const { return untessellatedPatches.Length() + patches.Length(); }
};

OpenSubdivMesh *AdaptiveTessellation(Arena *arena, ScenePrimitives *scene,
                                     const Mat4 &NDCFromCamera, int screenHeight,
                                     struct Mesh *controlMeshes, u32 numMeshes);

} // namespace rt
#endif
