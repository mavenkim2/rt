#ifndef SUBDIVISION_H
#define SUBDIVISION_H
namespace rt
{
void Subdivide(struct Mesh *mesh);

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
    const FixedArray<int, 2> diff  = {1, -1};
    const FixedArray<int, 2> start = {0, 1};

    const FixedArray<Vec2f, 4> uvTable = {
        Vec2f(0.f, 0.f),
        Vec2f(1.f, 0.f),
        Vec2f(1.f, 1.f),
        Vec2f(0.f, 1.f),
    };
    const FixedArray<Vec2i, 4> uvDiffTable = {
        Vec2i(1, 0),
        Vec2i(0, 1),
        Vec2i(-1, 0),
        Vec2i(0, -1),
    };

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

    u32 GetNumFaces() const { return untessellatedPatches.Length() + patches.Length(); }
};

OpenSubdivMesh *AdaptiveTessellation(Arena *arena, ScenePrimitives *scene,
                                     const Mat4 &NDCFromCamera, int screenHeight,
                                     Mesh *controlMeshes, u32 numMeshes);

} // namespace rt
#endif
