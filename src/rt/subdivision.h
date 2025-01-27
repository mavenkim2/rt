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

struct OpenSubdivPatch
{
    // 3 --e2-- 2
    // |        |
    // e3       e1
    // |        |
    // 0 --e0-- 1

    int faceID;
    int gridIndexStart;
    // Minimum 2 for each
    int stitchingStart, stitchingCount;
    int edgeRates[4];

    OpenSubdivPatch() {}

    OpenSubdivPatch(int faceID, int gridIndexStart, int edge0, int edge1, int edge2, int edge3,
                    int stitchingStart, int stitchingCount)
        : faceID(faceID), gridIndexStart(gridIndexStart),
          edgeRates{edge0, edge1, edge2, edge3}, stitchingStart(stitchingStart),
          stitchingCount(stitchingCount)
    {
    }

    __forceinline int GetGridIndex(int u, int v) const
    {
        int edgeU     = Max(edgeRates[0], edgeRates[2]);
        int edgeV     = Max(edgeRates[1], edgeRates[3]);
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
