#ifndef MESH_SIMPLIFICATION_H_
#define MESH_SIMPLIFICATION_H_

#include "../base.h"
#include "../bit_packing.h"

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
    Vec3f newP;
    f32 error;

    Vec3f p0;
    Vec3f p1;

    bool operator<(const Pair &other) const { return error < other.error; }

    bool operator==(const Pair &other) { return p0 == other.p0 && p1 == other.p1; }
};

struct MeshSimplifier
{
    Arena *arena;
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

    // Adjacency
    StaticArray<Pair> pairs;
    HashIndex cornerHash;
    HashIndex vertexHash;
    HashIndex pairHash0;
    HashIndex pairHash1;
    BitVector triangleIsRemoved;

    MeshSimplifier(Arena *arena, f32 *vertexData, u32 numVertices, u32 *indices,
                   u32 numIndices);

    Vec3f &GetPosition(u32 vertexIndex);
    const Vec3f &GetPosition(u32 vertexIndex) const;
    f32 *GetAttributes(u32 vertexIndex);
    bool AddUniquePair(Pair &pair, int pairIndex);
    bool CheckInversion(const Vec3f &newPosition, u32 *movedCorners, u32 count) const;
    void CalculateTriQuadrics(u32 triIndex);

    template <typename Func>
    void IterateCorners(const Vec3f &position, const Func &func);

    void EvaluatePair(Pair &pair);
    f32 Simplify(u32 targetNumVerts, u32 targetNumTris, f32 targetError, u32 limitNumVerts,
                 u32 limitNumTris, f32 limitError);
    void Finalize(u32 &finalNumVertices, u32 &finalNumIndices);
};

} // namespace rt

#endif
