#ifndef MESH_SIMPLIFICATION_H_
#define MESH_SIMPLIFICATION_H_

#include "../base.h"

namespace rt
{

struct Quadric
{
    static const u32 maxAttributes = 6;

    f32 c00;
    f32 c01;
    f32 c02;

    f32 c11;
    f32 c12;

    f32 c22;

    Vec3f dn;

    f32 d2;

    // Volume optimization
    // Dot(gVol, p) + dVol = 0
    Vec3f gVol;
    f32 dVol;

    f32 area;

    Vec3f gradients[maxAttributes];
    f32 d[maxAttributes];

    u32 numAttributes;

    Quadric(u32 numAttributes);
    Quadric(const Vec3f &p0, const Vec3f &p1, const Vec3f &p2, f32 *__restrict attr0,
            f32 *__restrict attr1, f32 *__restrict attr2, f32 *__restrict attributeWeights,
            u32 numAttributes);

    void InitializeEdge(const Vec3f &p0, const Vec3f &p1);
    f32 Evaluate(const Vec3f &p, f32 *__restrict attributes, f32 *__restrict attributeWeights);
    void Add(QuadricAttr<numAttributes> &other);

    bool Optimize(Vec3f &p, bool volume);
};

struct Pair
{
    int indexIndex0;
    int indexIndex1;

    float error;

    bool operator<(const Pair &other) { return error < other.error; }

    u32 GetIndex(u32 index)
    {
        Assert(index < 2);
        return index == 0 ? indexIndex0 : indexIndex1;
    }
};

struct MeshSimplifier
{
    struct VertexGraphNode
    {
        u32 offset;
        u32 count;
        int next;
    };

    // Constants
    f32 lockedPenaty     = 1e8f;
    f32 inversionPenalty = 100.f;

    // Contains position, and all attributes (normals, uvs, etc.)
    f32 *vertexData;
    u32 *indices;
    u32 numAttributes;

    // Quadrics
    StaticArray<Quadric> triangleQuadrics;

    // Graph mapping vertices to triangle faces
    VertexGraphNode *vertexNodes;
    VertexGraphNode *vertexToPairNodes;

    u32 *indexData;
    u32 *pairIndices;

    Vec3f &GetPosition(u32 vertexIndex);
    bool CheckInversion(const Vec3f &newPosition, u32 vertexIndex);
    f32 EvaluatePair(Pair &pair, Vec3f *newPosition);
    void Simplify(Mesh &mesh, u32 limitNumVerts, u32 limitNumTris, u32 targetError,
                  u32 limitError);
};

} // namespace rt

#endif
