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

    f32 area;

    // Volume optimization
    // Dot(gVol, p) + dVol = 0
    Vec3f gVol;
    f32 dVol;

    Quadric();
    Quadric(const Vec3f &p0, const Vec3f &p1, const Vec3f &p2);

    void InitializeEdge(const Vec3f &p0, const Vec3f &p1);
    void Add(Quadric &other);
};

struct QuadricGrad
{
    Vec3f g;
    f32 d;
};

struct Pair
{
    f32 error;

    Vec3f p0;
    Vec3f p1;

    bool operator<(const Pair &other) const { return error < other.error; }

    bool operator==(const Pair &other) { return p0 == other.p1 && p1 == other.p1; }

    int Hash();
};

struct VertexGraphNode
{
    u32 offset;
    u32 count;
    int next;
};

struct MeshSimplifier
{
    // Constants
    f32 lockedPenaty     = 1e8f;
    f32 inversionPenalty = 100.f;

    // Contains position, and all attributes (normals, uvs, etc.)
    f32 *attributeWeights;
    f32 *vertexData;
    u32 *indices;
    u32 numVertices;
    u32 numIndices;
    u32 numAttributes;

    // Quadrics
    Quadric *triangleQuadrics;
    QuadricGrad *triangleAttrQuadrics;

    StaticArray<Pair> pairs;
    HashIndex cornerHash;
    HashIndex pairHash0;
    HashIndex pairHash1;

    MeshSimplifier(f32 *vertexData, u32 numVertices, u32 *indices, u32 numIndices);

    Vec3f &GetPosition(u32 vertexIndex);
    f32 *GetAttributes(u32 vertexIndex);
    bool AddUniquePair(Pair &pair);
    bool CheckInversion(const Vec3f &newPosition, u32 vertexIndex0, u32 vertexIndex1);
    void CalculateTriQuadrics(u32 triIndex);

    f32 EvaluatePair(Pair &pair, Vec3f *newPosition = 0);
    f32 Simplify(Arena *arena, u32 targetNumVerts, u32 targetNumTris, f32 targetError,
                 u32 limitNumVerts, u32 limitNumTris, f32 limitError);
};

} // namespace rt

#endif
