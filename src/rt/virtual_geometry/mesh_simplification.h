#ifndef MESH_SIMPLIFICATION_H_
#define MESH_SIMPLIFICATION_H_

#include "../base.h"

namespace rt
{

struct Quadric
{
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

    f32 Evaluate(const Vec3f &p);
};

template <u32 numAttributes>
struct QuadricAttr : Quadric
{
    QuadricAttr();
    QuadricAttr(const Vec3f &p0, const Vec3f &p1, const Vec3f &p2, f32 *__restrict attr0,
                f32 *__restrict attr1, f32 *__restrict attr2,
                f32 *__restrict attributeWeights);

    void Zero();
    Vec3f gradients[numAttributes];
    f32 d[numAttributes];

    f32 Evaluate(const Vec3f &p);
    void Add(QuadricAttr<numAttributes> &other);

    bool Optimize(Vec3f &p);
    bool OptimizeVolume(Vec3f &p);
};

struct MeshSimplifier
{
    struct VertexGraphNode
    {
        u32 offset;
        u32 count;
        int next;
    };

    Vec3f GetPosition(u32 vertexIndex);
    bool CheckInversion(const Vec3f &newPosition, u32 vertexIndex);
    void Simplify(Mesh &mesh);

    // Contains position, and all attributes (normals, uvs, etc.)
    f32 *vertexData;
    u32 *indices;
    u32 numAttributes;

    // Graph mapping vertices to triangle faces
    VertexGraphNode *vertexNodes;
    u32 *indexData;
};

} // namespace rt

#endif
