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
    int edgeFactor;
    int id0, id1;

    int GetFirst(bool reversed) const { return reversed ? id1 : id0; }
    int GetLast(bool reversed) const { return reversed ? id0 : id1; }

    EdgeInfo Opposite() const
    {
        Assert(edgeFactor < 0xffffffff);
        return EdgeInfo{indexStart, edgeFactor | reverseBit, id0, id1};
    }

    int GetEdgeFactor() const { return edgeFactor & ~reverseBit; }
    int GetReversed() const { return edgeFactor & reverseBit; }

    // int GetVertexId(int edgeStep) const
    // {
    //     Assert(edgeStep >= 0 && edgeStep <= edgeFactor);
    //     return edgeStep == 0 ? id0
    //                          : (edgeStep == edgeFactor ? id1 : indexStart + edgeStep - 1);
    // }

    int GetVertexID(int edgeStep) const
    {
        Assert(edgeStep >= 0 && edgeStep <= edgeFactor);
        return edgeStep == 0 ? id0
                             : (edgeStep == edgeFactor ? id1 : indexStart + edgeStep - 1);
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
    EdgeInfo edgeInfo[4];

    // Generates stitching indices instead of manually having to store them

    OpenSubdivPatch() {}

    PatchItr CreateIterator(int edge) const;
    PatchItr GetUVs(int id, Vec2f uv[3]) const;

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
        Assert(gridIndex < gridIndexStart + (edgeU - 1) * (edgeV - 1));
        return gridIndex;
    }

    // void Split4(OpenSubdivMesh *mesh, OpenSubdivPatch &out0, OpenSubdivPatch &out1,
    //             OpenSubdivPatch &out2, OpenSubdivPatch &out3)
    // {
    //     TempArena temp = ScratchStart(0, 0);
    //     int edgeRateU  = Max(edgeRates[0], edgeRates[2]);
    //     int edgeRateV  = Max(edgeRates[1], edgeRates[3]);
    //
    //     int numTriangles[] = {
    //         (edgeRates[0] - 1) + (edgeRateU - 1),
    //         (edgeRates[1] - 1) + (edgeRateV - 1),
    //         (edgeRates[2] - 1) + (edgeRateU - 1),
    //         (edgeRates[3] - 1) + (edgeRateV - 1),
    //     };
    //
    //     int totalNumTriangles = numTriangles0 + numTriangles1 + numTriangles2 +
    //     numTriangles3; Assert(3 * totalNumTriangles == stitchingCount);
    //
    //     int *tempIndices = PushArrayNoZero(temp.arena, int, totalNumTriangles);
    //
    //     int counts[4] = {
    //         numTriangles0 / 2 + numTriangles3 / 2 + numTriangles3 & 1,
    //         numTriangles0 / 2 + numTriangles0 & 1 + numTriangles1 / 2,
    //         numTriangles1 / 2 + numTriangles1 & 1 + numTriangles2 / 2,
    //         numTriangles2 / 2 + numTriangles2 & 1 + numTriangles3 / 2,
    //     };
    //     int offsets[4] = {};
    //
    //     int total = 0;
    //     for (int i = 0; i < 4; i++)
    //     {
    //         int offset = counts[i];
    //         offsets[i] = total;
    //         total += offset;
    //     }
    //     Assert(total == stitchingCount);
    //
    //     for (int side = 0; side < 4; side++)
    //     {
    //         int num = numTriangles[side];
    //         for (int i = 0; i < num; i++)
    //         {
    //             int quadrant        = (side + (i >= num / 2)) & 3;
    //             int offset          = offsets[quadrant]++;
    //             tempIndices[offset] = mesh->stitchingIndices[stitchingStart + offset];
    //         }
    //     }
    //
    //     Assert(offsets[3] == totalNumTriangles);
    //     out0.stitchingStart = 0;
    //     out0.stitchingCount = offsets[0];
    //     out1.stitchingStart = offsets[0];
    //     out1.stitchingCount = offsets[1] - offsets[0];
    //     out2.stitchingStart = offsets[1];
    //     out2.stitchingCount = offsets[2] - offsets[1];
    //     out3.stitchingCount = offsets[2];
    //     out3.stitchingCount = offsets[3] - offsets[2];
    //
    //     ScratchEnd(temp);
    // }
};

struct PatchItr
{
    const int diff[2]  = {1, -1};
    const int start[2] = {0, 1};

    const Vec2f uvTable[4] = {
        Vec2f(0.f, 0.f),
        Vec2f(1.f, 0.f),
        Vec2f(1.f, 1.f),
        Vec2f(0.f, 1.f),
    };
    const Vec2i uvDiffTable[4] = {
        Vec2i(1, 0),
        Vec2i(0, 1),
        Vec2i(-1, 0),
        Vec2i(0, -1),
    };

    const OpenSubdivPatch *patch;
    int edge;
    int indices[3];

    // Edge
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

        edgeStep = start[reversed] * edgeFactor;
        edgeDiff = diff[reversed];
        edgeEnd  = start[!reversed] * edgeFactor;

        int edgeU     = patch->GetMaxEdgeFactorU();
        int edgeV     = patch->GetMaxEdgeFactorV();
        maxEdgeFactor = ((edge & 1) ? edgeV : edgeU);
        q             = maxEdgeFactor - 3 * edgeFactor;

        Vec2i gridStep = Vec2i(Max(edgeU - 2, 0), Max(edgeV - 2, 0));

        uvStart = Vec2i(uvTable[edge]) * gridStep;
        uvEnd   = uvStart + uvDiffTable[edge] * gridStep;
    }
    bool IsNotFinished() { return edgeStep != edgeEnd || uvStart != uvEnd; }
    void Next()
    {
        const EdgeInfo &edgeInfo = patch->edgeInfo[edge];
        if (q >= 0 && uvStart != uvEnd)
        {
            int id0 = patch->GetGridIndex(uvStart[0], uvStart[1]);
            int id1 = edgeInfo.GetVertexID(edgeStep);
            uvStart += uvDiffTable[edge];
            int id2 = patch->GetGridIndex(uvStart[0], uvStart[1]);

            newIndex   = 2;
            indices[0] = id0;
            indices[1] = id1;
            indices[2] = id2;
            q -= 2 * edgeInfo.GetEdgeFactor();
        }
        else
        {
            int id0 = edgeInfo.GetVertexID(edgeStep);
            edgeStep += edgeDiff;
            Assert(edgeStep <= edgeInfo.GetEdgeFactor() && edgeStep >= 0);
            int id1 = edgeInfo.GetVertexID(edgeStep);
            int id2 = patch->GetGridIndex(uvStart[0], uvStart[1]);

            newIndex   = 1;
            indices[0] = id0;
            indices[1] = id1;
            indices[2] = id2;
            q += 2 * maxEdgeFactor;
        }
    }

    void GetUVs(int id, Vec2f uv[3])
    {
        for (int i = 0; i < id; i++)
        {
            Assert(IsNotFinished());
            Next();
        }
        Vec2f edgeDiv(1.f / patch->GetMaxEdgeFactorU(), 1.f / patch->GetMaxEdgeFactorV());
        int edgeFactor    = patch->edgeInfo[edge].GetEdgeFactor();
        f32 edgeFactorInv = 1.f / edgeFactor;
        if (q >= 0 && uvStart != uvEnd)
        {
            uv[0] = Vec2f(uvStart) * edgeDiv;
            uv[1] = (uvTable[edge] + Vec2f(uvDiffTable[edge] * edgeStep)) * edgeFactorInv;
            uv[2] = Vec2f(uvStart + uvDiffTable[edge]) * edgeDiv;
        }
        else
        {
            Assert(edgeStep < edgeFactor);
            uv[0] = (uvTable[edge] + Vec2f(uvDiffTable[edge] * edgeStep)) * edgeFactorInv;
            uv[1] =
                (uvTable[edge] + Vec2f(uvDiffTable[edge] * (edgeStep + 1))) * edgeFactorInv;
            uv[2] = Vec2f(uvStart) * edgeDiv;
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

    // Bounds GetPatchBounds(u32 primID) const
    // {
    //     Assert(primID < patches.Length());
    //     const OpenSubdivPatch *patch = &patches[primID];
    //
    //     Bounds bounds;
    //
    //     // Extend grid vertices
    //     for (int r = 0; r < patch->numRows; r++)
    //     {
    //         for (int c = 0; c < patch->numCols; c++)
    //         {
    //             Vec3f p = vertices[patch->gridIndexStart + patch->numCols * r + c];
    //             bounds.Extend(Lane4F32(p.x, p.y, p.z, 0.f));
    //         }
    //     }
    //     Assert(patch->stitchingCount % 3 == 0);
    //     // Extend stitching vertices
    //     for (int stitchIndex = 0; stitchIndex < patch->stitchingCount; stitchIndex++)
    //     {
    //         Vec3f p = vertices[stitchingIndices[patch->stitchingStart + stitchIndex]];
    //         bounds.Extend(Lane4F32(p.x, p.y, p.z, 0.f));
    //     }
    //     return bounds;
    // }

    u32 GetNumFaces() const { return untessellatedPatches.Length() + patches.Length(); }
};

OpenSubdivMesh *AdaptiveTessellation(Arena *arena, ScenePrimitives *scene,
                                     const Mat4 &NDCFromCamera, int screenHeight,
                                     Mesh *controlMeshes, u32 numMeshes);

} // namespace rt
#endif
