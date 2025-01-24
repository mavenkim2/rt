#ifndef SUBDIVISION_H
#define SUBDIVISION_H
namespace rt
{
void Subdivide(struct Mesh *mesh);

struct OpenSubdivPatch
{
    // 3 --e2-- 2
    // |        |
    // e3       e1
    // |        |
    // 0 --e0-- 1

    int gridIndexStart;
    // Minimum 2 for each
    int numRows, numCols;
    int stitchingStart, stitchingCount;

    int GetNumPrims() const { return (numRows - 1) * (numCols - 1) + stitchingCount / 3; }
};

// TODO: can have stitching quads (for edges that match the
// maximum u/v edge rate), by specifying an edge index start, grid index start,
// and a grid offset step size. this may or may not be worth it
struct OpenSubdivMesh
{
    StaticArray<Vec3f> vertices;
    StaticArray<int> stitchingIndices;
    StaticArray<OpenSubdivPatch> patches;

    Bounds GetPatchBounds(u32 primID) const
    {
        Assert(primID < patches.Length());
        const OpenSubdivPatch *patch = &patches[primID];

        Bounds bounds;

        // Extend grid vertices
        for (int r = 0; r < patch->numRows; r++)
        {
            for (int c = 0; c < patch->numCols; c++)
            {
                Vec3f p = vertices[patch->gridIndexStart + patch->numCols * r + c];
                bounds.Extend(Lane4F32(p.x, p.y, p.z, 0.f));
            }
        }
        Assert(patch->stitchingCount % 3 == 0);
        // Extend stitching vertices
        for (int stitchIndex = 0; stitchIndex < patch->stitchingCount; stitchIndex++)
        {
            Vec3f p = vertices[stitchingIndices[patch->stitchingStart + stitchIndex]];
            bounds.Extend(Lane4F32(p.x, p.y, p.z, 0.f));
        }
        return bounds;
    }

    u32 GetNumFaces() const { return patches.Length(); }
};

} // namespace rt
#endif
