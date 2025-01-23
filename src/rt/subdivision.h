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
};

// TODO: can have stitching quads (for edges that match the
// maximum u/v edge rate), by specifying an edge index start, grid index start,
// and a grid offset step size. this may or may not be worth it
struct OpenSubdivMesh
{
    Vec3f *vertices;
    u32 *stitchingIndices;
    OpenSubdivPatch *patches;
    u32 numPatches;

    u32 numPatchFaces;
    u32 totalNumFaces;

    u32 GetVertices(u32 primID, Vec3f *v) const
    {
        if (primID > numPatchFaces)
        {
            // TODO
            return 4;
        }
        else
        {
            int triangleIndex = primID - numPatchFaces;
            v[0]              = vertices[stitchingIndices[triangleIndex * 3 + 0]];
            v[1]              = vertices[stitchingIndices[triangleIndex * 3 + 1]];
            v[2]              = vertices[stitchingIndices[triangleIndex * 3 + 2]];
            return 3;
        }
    }

    u32 GetNumFaces() const { return totalNumFaces; }
};

} // namespace rt
#endif
