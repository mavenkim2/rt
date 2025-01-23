#ifndef BVH_AOS_H
#define BVH_AOS_H
#include "bvh_types.h"
#include <atomic>
#include <utility>

// TODO:
// - stream (Fuetterling 2015, frusta)/packet traversal
// - ray sorting/binning(hyperion, or individually in each thread).
// - curves
//     - it seems that PBRT doesn't supported instanced curves, so the scene description files
//     handle these weirdly. look at converting these to instances?
// - support both BVH over all primitives and two level BVH.
//     - for BVH over all primitives, need polymorphism. will implement by indexing into an
//     array of indices,
//     - can instances contain multiple types of primitives?
// - expand current BVH4 traversal code to BVH8

// far future TODOs (after moana is rendered)
// - subdivision surfaces
// - systematically handling floating point error

// NOTEs on moana data set:
// - each triangle pair is a quad (not sure if coplanar, I think so?), representing the base
// cage of the subdivison surface. don't actually need the indices, uvs, or face indices

namespace rt
{
//////////////////////////////
// Clipping
//

struct Quad8
{
    static const u32 N = 4;
    static const u32 LUTNext[];
    union
    {
        struct
        {
            Lane8F32 v0u;
            Lane8F32 v0v;
            Lane8F32 v0w;

            Lane8F32 v1u;
            Lane8F32 v1v;
            Lane8F32 v1w;

            Lane8F32 v2u;
            Lane8F32 v2v;
            Lane8F32 v2w;

            Lane8F32 v3u;
            Lane8F32 v3v;
            Lane8F32 v3w;
        };
        Lane8F32 v[12];
    };
    Quad8() {}
    __forceinline const Lane8F32 &operator[](i32 i) const
    {
        Assert(i < 12);
        return v[i];
    }
    __forceinline Lane8F32 &operator[](i32 i)
    {
        Assert(i < 12);
        return v[i];
    }
    static void Load(const ScenePrimitives *scene, const u32 dim, const u32 faceIndices[8],
                     Quad8 *out)
    {
        Assert(scene->numPrimitives == 1);
        Mesh *mesh     = (Mesh *)scene->primitives;
        u32 faceIndexA = faceIndices[0];
        u32 faceIndexB = faceIndices[1];
        u32 faceIndexC = faceIndices[2];
        u32 faceIndexD = faceIndices[3];

        Lane4F32 v0a = Lane4F32::LoadU((float *)(&mesh->p[faceIndexA * 4 + 0]));
        Lane4F32 v1a = Lane4F32::LoadU((float *)(&mesh->p[faceIndexA * 4 + 1]));
        Lane4F32 v2a = Lane4F32::LoadU((float *)(&mesh->p[faceIndexA * 4 + 2]));
        Lane4F32 v3a = Lane4F32::LoadU((float *)(&mesh->p[faceIndexA * 4 + 3]));

        Lane4F32 v0b = Lane4F32::LoadU((float *)(&mesh->p[faceIndexB * 4 + 0]));
        Lane4F32 v1b = Lane4F32::LoadU((float *)(&mesh->p[faceIndexB * 4 + 1]));
        Lane4F32 v2b = Lane4F32::LoadU((float *)(&mesh->p[faceIndexB * 4 + 2]));
        Lane4F32 v3b = Lane4F32::LoadU((float *)(&mesh->p[faceIndexB * 4 + 3]));

        Lane4F32 v0c = Lane4F32::LoadU((float *)(&mesh->p[faceIndexC * 4 + 0]));
        Lane4F32 v1c = Lane4F32::LoadU((float *)(&mesh->p[faceIndexC * 4 + 1]));
        Lane4F32 v2c = Lane4F32::LoadU((float *)(&mesh->p[faceIndexC * 4 + 2]));
        Lane4F32 v3c = Lane4F32::LoadU((float *)(&mesh->p[faceIndexC * 4 + 3]));

        Lane4F32 v0d = Lane4F32::LoadU((float *)(&mesh->p[faceIndexD * 4 + 0]));
        Lane4F32 v1d = Lane4F32::LoadU((float *)(&mesh->p[faceIndexD * 4 + 1]));
        Lane4F32 v2d = Lane4F32::LoadU((float *)(&mesh->p[faceIndexD * 4 + 2]));
        Lane4F32 v3d = Lane4F32::LoadU((float *)(&mesh->p[faceIndexD * 4 + 3]));

        Vec3lf4 p0;
        Vec3lf4 p1;
        Vec3lf4 p2;
        Vec3lf4 p3;

        Transpose4x3(v0a, v0b, v0c, v0d, p0.x, p0.y, p0.z);
        Transpose4x3(v1a, v1b, v1c, v1d, p1.x, p1.y, p1.z);
        Transpose4x3(v2a, v2b, v2c, v2d, p2.x, p2.y, p2.z);
        Transpose4x3(v3a, v3b, v3c, v3d, p3.x, p3.y, p3.z);

        faceIndexA = faceIndices[4];
        faceIndexB = faceIndices[5];
        faceIndexC = faceIndices[6];
        faceIndexD = faceIndices[7];

        v0a = Lane4F32::LoadU((float *)(&mesh->p[faceIndexA * 4 + 0]));
        v1a = Lane4F32::LoadU((float *)(&mesh->p[faceIndexA * 4 + 1]));
        v2a = Lane4F32::LoadU((float *)(&mesh->p[faceIndexA * 4 + 2]));
        v3a = Lane4F32::LoadU((float *)(&mesh->p[faceIndexA * 4 + 3]));

        v0b = Lane4F32::LoadU((float *)(&mesh->p[faceIndexB * 4 + 0]));
        v1b = Lane4F32::LoadU((float *)(&mesh->p[faceIndexB * 4 + 1]));
        v2b = Lane4F32::LoadU((float *)(&mesh->p[faceIndexB * 4 + 2]));
        v3b = Lane4F32::LoadU((float *)(&mesh->p[faceIndexB * 4 + 3]));

        v0c = Lane4F32::LoadU((float *)(&mesh->p[faceIndexC * 4 + 0]));
        v1c = Lane4F32::LoadU((float *)(&mesh->p[faceIndexC * 4 + 1]));
        v2c = Lane4F32::LoadU((float *)(&mesh->p[faceIndexC * 4 + 2]));
        v3c = Lane4F32::LoadU((float *)(&mesh->p[faceIndexC * 4 + 3]));

        v0d = Lane4F32::LoadU((float *)(&mesh->p[faceIndexD * 4 + 0]));
        v1d = Lane4F32::LoadU((float *)(&mesh->p[faceIndexD * 4 + 1]));
        v2d = Lane4F32::LoadU((float *)(&mesh->p[faceIndexD * 4 + 2]));
        v3d = Lane4F32::LoadU((float *)(&mesh->p[faceIndexD * 4 + 3]));

        Vec3lf4 p4;
        Vec3lf4 p5;
        Vec3lf4 p6;
        Vec3lf4 p7;

        Transpose4x3(v0a, v0b, v0c, v0d, p4.x, p4.y, p4.z);
        Transpose4x3(v1a, v1b, v1c, v1d, p5.x, p5.y, p5.z);
        Transpose4x3(v2a, v2b, v2c, v2d, p6.x, p6.y, p6.z);
        Transpose4x3(v3a, v3b, v3c, v3d, p7.x, p7.y, p7.z);

        u32 v = LUTAxis[dim];
        u32 w = LUTAxis[v];

        out->v0u = Lane8F32(p0[dim], p4[dim]);
        out->v1u = Lane8F32(p1[dim], p5[dim]);
        out->v2u = Lane8F32(p2[dim], p6[dim]);
        out->v3u = Lane8F32(p3[dim], p7[dim]);

        out->v0v = Lane8F32(p0[v], p4[v]);
        out->v1v = Lane8F32(p1[v], p5[v]);
        out->v2v = Lane8F32(p2[v], p6[v]);
        out->v3v = Lane8F32(p3[v], p7[v]);

        out->v0w = Lane8F32(p0[w], p4[w]);
        out->v1w = Lane8F32(p1[w], p5[w]);
        out->v2w = Lane8F32(p2[w], p6[w]);
        out->v3w = Lane8F32(p3[w], p7[w]);
    }

    static void Load(const ScenePrimitives *scene, const u32 dim, const u32 geomIDs[8],
                     const u32 faceIndices[8], Quad8 *out)
    {
        Mesh *primitives = (Mesh *)scene->primitives;
        Mesh *meshes[8]  = {primitives + geomIDs[0], primitives + geomIDs[1],
                            primitives + geomIDs[2], primitives + geomIDs[3],
                            primitives + geomIDs[4], primitives + geomIDs[5],
                            primitives + geomIDs[6], primitives + geomIDs[7]};

        u32 faceIndexA = faceIndices[0];
        u32 faceIndexB = faceIndices[1];
        u32 faceIndexC = faceIndices[2];
        u32 faceIndexD = faceIndices[3];

        Lane4F32 v0a = Lane4F32::LoadU((float *)(&meshes[0]->p[faceIndexA * 4 + 0]));
        Lane4F32 v1a = Lane4F32::LoadU((float *)(&meshes[0]->p[faceIndexA * 4 + 1]));
        Lane4F32 v2a = Lane4F32::LoadU((float *)(&meshes[0]->p[faceIndexA * 4 + 2]));
        Lane4F32 v3a = Lane4F32::LoadU((float *)(&meshes[0]->p[faceIndexA * 4 + 3]));

        Lane4F32 v0b = Lane4F32::LoadU((float *)(&meshes[1]->p[faceIndexB * 4 + 0]));
        Lane4F32 v1b = Lane4F32::LoadU((float *)(&meshes[1]->p[faceIndexB * 4 + 1]));
        Lane4F32 v2b = Lane4F32::LoadU((float *)(&meshes[1]->p[faceIndexB * 4 + 2]));
        Lane4F32 v3b = Lane4F32::LoadU((float *)(&meshes[1]->p[faceIndexB * 4 + 3]));

        Lane4F32 v0c = Lane4F32::LoadU((float *)(&meshes[2]->p[faceIndexC * 4 + 0]));
        Lane4F32 v1c = Lane4F32::LoadU((float *)(&meshes[2]->p[faceIndexC * 4 + 1]));
        Lane4F32 v2c = Lane4F32::LoadU((float *)(&meshes[2]->p[faceIndexC * 4 + 2]));
        Lane4F32 v3c = Lane4F32::LoadU((float *)(&meshes[2]->p[faceIndexC * 4 + 3]));

        Lane4F32 v0d = Lane4F32::LoadU((float *)(&meshes[3]->p[faceIndexD * 4 + 0]));
        Lane4F32 v1d = Lane4F32::LoadU((float *)(&meshes[3]->p[faceIndexD * 4 + 1]));
        Lane4F32 v2d = Lane4F32::LoadU((float *)(&meshes[3]->p[faceIndexD * 4 + 2]));
        Lane4F32 v3d = Lane4F32::LoadU((float *)(&meshes[3]->p[faceIndexD * 4 + 3]));

        Vec3lf4 p0;
        Vec3lf4 p1;
        Vec3lf4 p2;
        Vec3lf4 p3;

        Transpose4x3(v0a, v0b, v0c, v0d, p0.x, p0.y, p0.z);
        Transpose4x3(v1a, v1b, v1c, v1d, p1.x, p1.y, p1.z);
        Transpose4x3(v2a, v2b, v2c, v2d, p2.x, p2.y, p2.z);
        Transpose4x3(v3a, v3b, v3c, v3d, p3.x, p3.y, p3.z);

        faceIndexA = faceIndices[4];
        faceIndexB = faceIndices[5];
        faceIndexC = faceIndices[6];
        faceIndexD = faceIndices[7];

        v0a = Lane4F32::LoadU((float *)(&meshes[4]->p[faceIndexA * 4 + 0]));
        v1a = Lane4F32::LoadU((float *)(&meshes[4]->p[faceIndexA * 4 + 1]));
        v2a = Lane4F32::LoadU((float *)(&meshes[4]->p[faceIndexA * 4 + 2]));
        v3a = Lane4F32::LoadU((float *)(&meshes[4]->p[faceIndexA * 4 + 3]));

        v0b = Lane4F32::LoadU((float *)(&meshes[5]->p[faceIndexB * 4 + 0]));
        v1b = Lane4F32::LoadU((float *)(&meshes[5]->p[faceIndexB * 4 + 1]));
        v2b = Lane4F32::LoadU((float *)(&meshes[5]->p[faceIndexB * 4 + 2]));
        v3b = Lane4F32::LoadU((float *)(&meshes[5]->p[faceIndexB * 4 + 3]));

        v0c = Lane4F32::LoadU((float *)(&meshes[6]->p[faceIndexC * 4 + 0]));
        v1c = Lane4F32::LoadU((float *)(&meshes[6]->p[faceIndexC * 4 + 1]));
        v2c = Lane4F32::LoadU((float *)(&meshes[6]->p[faceIndexC * 4 + 2]));
        v3c = Lane4F32::LoadU((float *)(&meshes[6]->p[faceIndexC * 4 + 3]));

        v0d = Lane4F32::LoadU((float *)(&meshes[7]->p[faceIndexD * 4 + 0]));
        v1d = Lane4F32::LoadU((float *)(&meshes[7]->p[faceIndexD * 4 + 1]));
        v2d = Lane4F32::LoadU((float *)(&meshes[7]->p[faceIndexD * 4 + 2]));
        v3d = Lane4F32::LoadU((float *)(&meshes[7]->p[faceIndexD * 4 + 3]));

        Vec3lf4 p4;
        Vec3lf4 p5;
        Vec3lf4 p6;
        Vec3lf4 p7;

        Transpose4x3(v0a, v0b, v0c, v0d, p4.x, p4.y, p4.z);
        Transpose4x3(v1a, v1b, v1c, v1d, p5.x, p5.y, p5.z);
        Transpose4x3(v2a, v2b, v2c, v2d, p6.x, p6.y, p6.z);
        Transpose4x3(v3a, v3b, v3c, v3d, p7.x, p7.y, p7.z);

        u32 v = LUTAxis[dim];
        u32 w = LUTAxis[v];

        out->v0u = Lane8F32(p0[dim], p4[dim]);
        out->v1u = Lane8F32(p1[dim], p5[dim]);
        out->v2u = Lane8F32(p2[dim], p6[dim]);
        out->v3u = Lane8F32(p3[dim], p7[dim]);

        out->v0v = Lane8F32(p0[v], p4[v]);
        out->v1v = Lane8F32(p1[v], p5[v]);
        out->v2v = Lane8F32(p2[v], p6[v]);
        out->v3v = Lane8F32(p3[v], p7[v]);

        out->v0w = Lane8F32(p0[w], p4[w]);
        out->v1w = Lane8F32(p1[w], p5[w]);
        out->v2w = Lane8F32(p2[w], p6[w]);
        out->v3w = Lane8F32(p3[w], p7[w]);
    }
};

struct Triangle8
{
    static const u32 N = 3;
    static const u32 LUTNext[];
    union
    {
        struct
        {
            Lane8F32 v0u;
            Lane8F32 v0v;
            Lane8F32 v0w;

            Lane8F32 v1u;
            Lane8F32 v1v;
            Lane8F32 v1w;

            Lane8F32 v2u;
            Lane8F32 v2v;
            Lane8F32 v2w;
        };
        Lane8F32 v[9];
    };
    Triangle8() {}
    Triangle8(const Triangle8 &other)
        : v0u(other.v0u), v0v(other.v0v), v0w(other.v0w), v1u(other.v1u), v1v(other.v1v),
          v1w(other.v1w), v2u(other.v2u), v2v(other.v2v), v2w(other.v2w)
    {
    }

    __forceinline const Lane8F32 &operator[](i32 i) const
    {
        Assert(i < 9);
        return v[i];
    }
    __forceinline Lane8F32 &operator[](i32 i)
    {
        Assert(i < 9);
        return v[i];
    }

    static void Load(const ScenePrimitives *scene, const u32 dim, const u32 geomIDs[8],
                     const u32 faceIndices[8], Triangle8 *out)
    {
        Mesh *primitives = (Mesh *)scene->primitives;
        u32 faceIndexA   = faceIndices[0];
        u32 faceIndexB   = faceIndices[1];
        u32 faceIndexC   = faceIndices[2];
        u32 faceIndexD   = faceIndices[3];

        Mesh *meshes[8] = {primitives + geomIDs[0], primitives + geomIDs[1],
                           primitives + geomIDs[2], primitives + geomIDs[3],
                           primitives + geomIDs[4], primitives + geomIDs[5],
                           primitives + geomIDs[6], primitives + geomIDs[7]};

        Lane4F32 v0a =
            Lane4F32::LoadU((float *)(&meshes[0]->p[meshes[0]->indices[faceIndexA * 3]]));
        Lane4F32 v1a =
            Lane4F32::LoadU((float *)(&meshes[0]->p[meshes[0]->indices[faceIndexA * 3 + 1]]));
        Lane4F32 v2a =
            Lane4F32::LoadU((float *)(&meshes[0]->p[meshes[0]->indices[faceIndexA * 3 + 2]]));

        Lane4F32 v0b =
            Lane4F32::LoadU((float *)(&meshes[1]->p[meshes[1]->indices[faceIndexB * 3]]));
        Lane4F32 v1b =
            Lane4F32::LoadU((float *)(&meshes[1]->p[meshes[1]->indices[faceIndexB * 3 + 1]]));
        Lane4F32 v2b =
            Lane4F32::LoadU((float *)(&meshes[1]->p[meshes[1]->indices[faceIndexB * 3 + 2]]));

        Lane4F32 v0c =
            Lane4F32::LoadU((float *)(&meshes[2]->p[meshes[2]->indices[faceIndexC * 3]]));
        Lane4F32 v1c =
            Lane4F32::LoadU((float *)(&meshes[2]->p[meshes[2]->indices[faceIndexC * 3 + 1]]));
        Lane4F32 v2c =
            Lane4F32::LoadU((float *)(&meshes[2]->p[meshes[2]->indices[faceIndexC * 3 + 2]]));

        Lane4F32 v0d =
            Lane4F32::LoadU((float *)(&meshes[3]->p[meshes[3]->indices[faceIndexD * 3]]));
        Lane4F32 v1d =
            Lane4F32::LoadU((float *)(&meshes[3]->p[meshes[3]->indices[faceIndexD * 3 + 1]]));
        Lane4F32 v2d =
            Lane4F32::LoadU((float *)(&meshes[3]->p[meshes[3]->indices[faceIndexD * 3 + 2]]));

        Vec3lf4 p0;
        Vec3lf4 p1;
        Vec3lf4 p2;

        Transpose4x3(v0a, v0b, v0c, v0d, p0.x, p0.y, p0.z);
        Transpose4x3(v1a, v1b, v1c, v1d, p1.x, p1.y, p1.z);
        Transpose4x3(v2a, v2b, v2c, v2d, p2.x, p2.y, p2.z);

        faceIndexA = faceIndices[4];
        faceIndexB = faceIndices[5];
        faceIndexC = faceIndices[6];
        faceIndexD = faceIndices[7];
        v0a = Lane4F32::LoadU((float *)(&meshes[4]->p[meshes[4]->indices[faceIndexA * 3]]));
        v1a =
            Lane4F32::LoadU((float *)(&meshes[4]->p[meshes[4]->indices[faceIndexA * 3 + 1]]));
        v2a =
            Lane4F32::LoadU((float *)(&meshes[4]->p[meshes[4]->indices[faceIndexA * 3 + 2]]));

        v0b = Lane4F32::LoadU((float *)(&meshes[5]->p[meshes[5]->indices[faceIndexB * 3]]));
        v1b =
            Lane4F32::LoadU((float *)(&meshes[5]->p[meshes[5]->indices[faceIndexB * 3 + 1]]));
        v2b =
            Lane4F32::LoadU((float *)(&meshes[5]->p[meshes[5]->indices[faceIndexB * 3 + 2]]));

        v0c = Lane4F32::LoadU((float *)(&meshes[6]->p[meshes[6]->indices[faceIndexC * 3]]));
        v1c =
            Lane4F32::LoadU((float *)(&meshes[6]->p[meshes[6]->indices[faceIndexC * 3 + 1]]));
        v2c =
            Lane4F32::LoadU((float *)(&meshes[6]->p[meshes[6]->indices[faceIndexC * 3 + 2]]));

        v0d = Lane4F32::LoadU((float *)(&meshes[7]->p[meshes[7]->indices[faceIndexD * 3]]));
        v1d =
            Lane4F32::LoadU((float *)(&meshes[7]->p[meshes[7]->indices[faceIndexD * 3 + 1]]));
        v2d =
            Lane4F32::LoadU((float *)(&meshes[7]->p[meshes[7]->indices[faceIndexD * 3 + 2]]));

        Vec3lf4 p3;
        Vec3lf4 p4;
        Vec3lf4 p5;

        Transpose4x3(v0a, v0b, v0c, v0d, p3.x, p3.y, p3.z);
        Transpose4x3(v1a, v1b, v1c, v1d, p4.x, p4.y, p4.z);
        Transpose4x3(v2a, v2b, v2c, v2d, p5.x, p5.y, p5.z);

        u32 v = LUTAxis[dim];
        u32 w = LUTAxis[v];

        out->v0u = Lane8F32(p0[dim], p3[dim]);
        out->v1u = Lane8F32(p1[dim], p4[dim]);
        out->v2u = Lane8F32(p2[dim], p5[dim]);

        out->v0v = Lane8F32(p0[v], p3[v]);
        out->v1v = Lane8F32(p1[v], p4[v]);
        out->v2v = Lane8F32(p2[v], p5[v]);

        out->v0w = Lane8F32(p0[w], p3[w]);
        out->v1w = Lane8F32(p1[w], p4[w]);
        out->v2w = Lane8F32(p2[w], p5[w]);
    }

    static void Load(const ScenePrimitives *scene, const u32 dim, const u32 faceIndices[8],
                     Triangle8 *out)
    {
        Assert(scene->numPrimitives == 1);
        Mesh *mesh     = (Mesh *)scene->primitives;
        u32 faceIndexA = faceIndices[0];
        u32 faceIndexB = faceIndices[1];
        u32 faceIndexC = faceIndices[2];
        u32 faceIndexD = faceIndices[3];

        Lane4F32 v0a = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexA * 3]]));
        Lane4F32 v1a = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexA * 3 + 1]]));
        Lane4F32 v2a = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexA * 3 + 2]]));

        Lane4F32 v0b = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexB * 3]]));
        Lane4F32 v1b = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexB * 3 + 1]]));
        Lane4F32 v2b = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexB * 3 + 2]]));

        Lane4F32 v0c = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexC * 3]]));
        Lane4F32 v1c = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexC * 3 + 1]]));
        Lane4F32 v2c = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexC * 3 + 2]]));

        Lane4F32 v0d = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexD * 3]]));
        Lane4F32 v1d = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexD * 3 + 1]]));
        Lane4F32 v2d = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexD * 3 + 2]]));

        Vec3lf4 p0;
        Vec3lf4 p1;
        Vec3lf4 p2;

        Transpose4x3(v0a, v0b, v0c, v0d, p0.x, p0.y, p0.z);
        Transpose4x3(v1a, v1b, v1c, v1d, p1.x, p1.y, p1.z);
        Transpose4x3(v2a, v2b, v2c, v2d, p2.x, p2.y, p2.z);

        faceIndexA = faceIndices[4];
        faceIndexB = faceIndices[5];
        faceIndexC = faceIndices[6];
        faceIndexD = faceIndices[7];
        v0a        = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexA * 3]]));
        v1a        = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexA * 3 + 1]]));
        v2a        = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexA * 3 + 2]]));

        v0b = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexB * 3]]));
        v1b = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexB * 3 + 1]]));
        v2b = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexB * 3 + 2]]));

        v0c = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexC * 3]]));
        v1c = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexC * 3 + 1]]));
        v2c = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexC * 3 + 2]]));

        v0d = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexD * 3]]));
        v1d = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexD * 3 + 1]]));
        v2d = Lane4F32::LoadU((float *)(&mesh->p[mesh->indices[faceIndexD * 3 + 2]]));

        Vec3lf4 p3;
        Vec3lf4 p4;
        Vec3lf4 p5;

        Transpose4x3(v0a, v0b, v0c, v0d, p3.x, p3.y, p3.z);
        Transpose4x3(v1a, v1b, v1c, v1d, p4.x, p4.y, p4.z);
        Transpose4x3(v2a, v2b, v2c, v2d, p5.x, p5.y, p5.z);

        u32 v = LUTAxis[dim];
        u32 w = LUTAxis[v];

        out->v0u = Lane8F32(p0[dim], p3[dim]);
        out->v1u = Lane8F32(p1[dim], p4[dim]);
        out->v2u = Lane8F32(p2[dim], p5[dim]);

        out->v0v = Lane8F32(p0[v], p3[v]);
        out->v1v = Lane8F32(p1[v], p4[v]);
        out->v2v = Lane8F32(p2[v], p5[v]);

        out->v0w = Lane8F32(p0[w], p3[w]);
        out->v1w = Lane8F32(p1[w], p4[w]);
        out->v2w = Lane8F32(p2[w], p5[w]);
    }
};

const u32 Quad8::LUTNext[]     = {1, 2, 3, 0};
const u32 Triangle8::LUTNext[] = {1, 2, 0};

template <typename Polygon8>
static void ClipPolygon(const u32 dim, const Polygon8 &poly, const Lane8F32 &splitPos,
                        Lane8F32 &leftMinX, Lane8F32 &leftMinY, Lane8F32 &leftMinZ,
                        Lane8F32 &leftMaxX, Lane8F32 &leftMaxY, Lane8F32 &leftMaxZ,
                        Lane8F32 &rightMinX, Lane8F32 &rightMinY, Lane8F32 &rightMinZ,
                        Lane8F32 &rightMaxX, Lane8F32 &rightMaxY, Lane8F32 &rightMaxZ)
{
    static const u32 LUTX[] = {0, 2, 1};
    static const u32 LUTY[] = {1, 0, 2};
    static const u32 LUTZ[] = {2, 1, 0};

    Bounds8F32 left;
    left.maxU = splitPos;
    Bounds8F32 right;
    right.minU = splitPos;

    u32 N     = Polygon8::N;
    u32 first = 0;
    u32 next  = Polygon8::LUTNext[first];
    for (u32 edgeIndex = 0; edgeIndex < N; edgeIndex++)
    {
        const u32 v0IndexStart = 3 * first;
        const u32 v1IndexStart = 3 * next;

        const Lane8F32 &v0u = poly[v0IndexStart];
        const Lane8F32 &v1u = poly[v1IndexStart];

        const Lane8F32 &v0v = poly[v0IndexStart + 1];
        const Lane8F32 &v1v = poly[v1IndexStart + 1];

        const Lane8F32 &v0w = poly[v0IndexStart + 2];
        const Lane8F32 &v1w = poly[v1IndexStart + 2];

        // const Lane8F32 &clipMaskL = clipMasksL[first];

        left.MaskExtendL(v0u <= splitPos, v0u, v0v, v0w);
        right.MaskExtendR(v0u >= splitPos, v0u, v0v, v0w);

        const Lane8F32 div = Rcp(v1u - v0u);
        // const Lane8F32 div = Select(v1u == v0u, Lane8F32(zero), Rcp(v1u - v0u));
        // TODO: this is a catastrophic cancellation
        const Lane8F32 t = (splitPos - v0u) * div;

        const Lane8F32 subV = v1v - v0v;
        const Lane8F32 subW = v1w - v0w;

        const Lane8F32 clippedV = FMA(t, subV, v0v);
        const Lane8F32 clippedW = FMA(t, subW, v0w);

        const Lane8F32 edgeIsClipped =
            ((v0u < splitPos) & (v1u > splitPos)) | ((v0u > splitPos) & (v1u < splitPos));

        left.MaskExtendVW(edgeIsClipped, clippedV, clippedW);
        right.MaskExtendVW(edgeIsClipped, clippedV, clippedW);

        first = next;
        next  = Polygon8::LUTNext[first];
    }

    leftMinX = *((Lane8F32 *)(&left) + LUTX[dim]);
    leftMinY = *((Lane8F32 *)(&left) + LUTY[dim]);
    leftMinZ = *((Lane8F32 *)(&left) + LUTZ[dim]);

    leftMaxX = *((Lane8F32 *)(&left) + 3 + LUTX[dim]);
    leftMaxY = *((Lane8F32 *)(&left) + 3 + LUTY[dim]);
    leftMaxZ = *((Lane8F32 *)(&left) + 3 + LUTZ[dim]);

    rightMinX = *((Lane8F32 *)(&right) + LUTX[dim]);
    rightMinY = *((Lane8F32 *)(&right) + LUTY[dim]);
    rightMinZ = *((Lane8F32 *)(&right) + LUTZ[dim]);

    rightMaxX = *((Lane8F32 *)(&right) + 3 + LUTX[dim]);
    rightMaxY = *((Lane8F32 *)(&right) + 3 + LUTY[dim]);
    rightMaxZ = *((Lane8F32 *)(&right) + 3 + LUTZ[dim]);
}

template <typename Polygon8>
__forceinline static void ClipPolygon(const u32 dim, const Polygon8 &tri,
                                      const Lane8F32 &splitPos, Bounds8 *l, Bounds8 *r)
{
    Bounds8 lOut[8];
    Lane8F32 leftMinX, leftMinY, leftMinZ, leftMaxX, leftMaxY, leftMaxZ, rightMinX, rightMinY,
        rightMinZ, rightMaxX, rightMaxY, rightMaxZ;

    ClipPolygon(dim, tri, splitPos, leftMinX, leftMinY, leftMinZ, leftMaxX, leftMaxY, leftMaxZ,
                rightMinX, rightMinY, rightMinZ, rightMaxX, rightMaxY, rightMaxZ);

    Transpose8x8(-rightMinX, -rightMinY, -rightMinZ, pos_inf, rightMaxX, rightMaxY, rightMaxZ,
                 pos_inf, r[0].v, r[1].v, r[2].v, r[3].v, r[4].v, r[5].v, r[6].v, r[7].v);
    for (u32 i = 0; i < 8; i++)
    {
        r[i].Intersect(l[i]);
    }

    Transpose8x8(-leftMinX, -leftMinY, -leftMinZ, pos_inf, leftMaxX, leftMaxY, leftMaxZ,
                 pos_inf, lOut[0].v, lOut[1].v, lOut[2].v, lOut[3].v, lOut[4].v, lOut[5].v,
                 lOut[6].v, lOut[7].v);
    for (u32 i = 0; i < 8; i++)
    {
        l[i].Intersect(lOut[i]);
    }
}

template <i32 numBins = 32>
struct ObjectBinner
{
    Lane8F32 base[3];
    Lane8F32 scale[3];

    ObjectBinner(const Bounds &centroidBounds)
    {
        const f32 eps = 1e-34f;
        Lane8F32 minP(centroidBounds.minP);
        base[0] = Shuffle<0>(minP);
        base[1] = Shuffle<1>(minP);
        base[2] = Shuffle<2>(minP);

        const Lane4F32 diag = Max(centroidBounds.maxP - centroidBounds.minP, eps);
        Lane4F32 scale4     = Select(diag > eps, Lane4F32((f32)numBins) / diag, Lane4F32(0.f));

        Lane8F32 scale8(scale4);
        scale[0] = Shuffle<0>(scale8);
        scale[1] = Shuffle<1>(scale8);
        scale[2] = Shuffle<2>(scale8);
    }
    // ObjectBinner(const Lane8F32 &l) : ObjectBinner(Bounds(-Extract4<0>(l), Extract4<1>(l)))
    // {}
    Lane8U32 Bin(const Lane8F32 &in, const u32 dim) const
    {
        Lane8F32 result = (in - base[dim]) * scale[dim];
        return Clamp(Lane8U32(zero), Lane8U32(numBins - 1), Flooru(result));
    }
    u32 Bin(const f32 in, const u32 dim) const
    {
        Lane8U32 result = Bin(Lane8F32(in), dim);
        return result[0];
        // return Clamp(0u, numBins - 1u, (u32)Floor((in - base[dim][0]) * scale[dim][0]));
    }
    f32 GetSplitValue(u32 pos, u32 dim) const
    {
        f32 invScale = scale[dim][0] == 0.f ? 0.f : 1 / scale[dim][0];
        return (pos * invScale) + base[dim][0];
    }
};

template <i32 numBins = 16>
struct SplitBinner
{
    Lane8F32 base[3];
    Lane8F32 invScale[3];
    Lane8F32 scale[3];
    Lane8F32 scaleNegArr[3];

    SplitBinner(const Bounds &bounds)
    {
        const Lane4F32 eps = 1e-34f;

        Lane8F32 minP(bounds.minP);
        base[0] = Shuffle<0>(minP);
        base[1] = Shuffle<1>(minP);
        base[2] = Shuffle<2>(minP);

        const Lane4F32 diag = Max(bounds.maxP - bounds.minP, eps);

        Lane4F32 scale4 = Select(diag > eps, Lane4F32((f32)numBins) / diag, Lane4F32(0.f));

        Lane8F32 scale8(scale4);
        scale[0] = Shuffle<0>(scale8);
        scale[1] = Shuffle<1>(scale8);
        scale[2] = Shuffle<2>(scale8);

        // test
        Lane4F32 invScale4 = Select(scale4 == 0.f, 0.f, 1.f / scale4);
        Lane8F32 invScale8(invScale4);
        invScale[0] = Shuffle<0>(invScale8);
        invScale[1] = Shuffle<1>(invScale8);
        invScale[2] = Shuffle<2>(invScale8);

        scaleNegArr[0] = FlipSign(scale[0]);
        scaleNegArr[1] = FlipSign(scale[1]);
        scaleNegArr[2] = FlipSign(scale[2]);
    };
    // SplitBinner(const Lane8F32 &l) : SplitBinner(Bounds(-Extract4<0>(l), Extract4<1>(l))) {}
    __forceinline Lane8U32 BinMin(const Lane8F32 &min, const u32 dim) const
    {
        return Clamp(Lane8U32(zero), Lane8U32(numBins - 1),
                     Flooru((base[dim] + min) * scaleNegArr[dim]));
    }
    __forceinline u32 BinMin(const f32 min, const u32 dim) const
    {
        Lane8U32 result = BinMin(Lane8F32(min), dim);
        return result[0];
    }
    __forceinline Lane8U32 BinMax(const Lane8F32 &max, const u32 dim) const
    {
        return Clamp(Lane8U32(zero), Lane8U32(numBins - 1),
                     Flooru((max - base[dim]) * scale[dim]));
    }
    __forceinline u32 BinMax(const f32 max, const u32 dim) const
    {
        Lane8U32 result = BinMax(Lane8F32(max), dim);
        return result[0];
    }
    __forceinline u32 Bin(const f32 val, const u32 dim) const
    {
        Lane8U32 result = BinMax(Lane8F32(val), dim);
        return result[0];
    }
    __forceinline Lane8F32 GetSplitValue(const Lane8U32 &bins, u32 dim) const
    {
        return Lane8F32(bins) * invScale[dim] + base[dim];
    }
    // NOTE: scalar and AVX floating point computations produce different results, so it's
    // important that this uses AVX.
    __forceinline f32 GetSplitValue(u32 bin, u32 dim) const
    {
        Lane8F32 result = GetSplitValue(Lane8U32(bin), dim);
        return result[0];
    }
};

//////////////////////////////
// AOS Partitions
//

template <typename Heuristic, typename PrimRef>
u32 PartitionParallel(Heuristic *heuristic, PrimRef *data, const Split &split, u32 start,
                      u32 count, RecordAOSSplits &outLeft, RecordAOSSplits &outRight)
{

    const u32 blockSize         = 512;
    const u32 blockMask         = blockSize - 1;
    const u32 blockShift        = Bsf(blockSize);
    const u32 numJobs           = Min(16u, (count + 511) / 512); // OS_NumProcessors();
    const u32 numBlocksPerChunk = numJobs;
    const u32 chunkSize         = blockSize * numBlocksPerChunk;

    const u32 PARTITION_PARALLEL_THRESHOLD = 32 * 1024;
    if (count) // < PARTITION_PARALLEL_THRESHOLD)
    {
        u32 mid = heuristic->Partition(
            data, split, start, start + count - 1, [&](u32 index) { return index; }, outLeft,
            outRight);
        return mid;
    }

    const u32 numChunks = (count + chunkSize - 1) / chunkSize;
    u32 end             = start + count;

    TempArena temp = ScratchStart(0, 0);
    u32 *outMid    = PushArrayNoZero(temp.arena, u32, numJobs);

    RecordAOSSplits *recordL = PushArrayNoZero(temp.arena, RecordAOSSplits, numJobs);
    RecordAOSSplits *recordR = PushArrayNoZero(temp.arena, RecordAOSSplits, numJobs);

    auto GetIndex = [&](u32 index, u32 group) {
        const u32 chunkIndex   = index >> blockShift;
        const u32 indexInBlock = index & blockMask;

        u32 outIndex = start + chunkIndex * chunkSize + (group << blockShift) + indexInBlock;
        return outIndex;
    };

    scheduler.ScheduleAndWait(numJobs, 1, [&](u32 jobID) {
        const u32 group = jobID;

        u32 l          = 0;
        u32 r          = (numChunks << blockShift) - 1;
        u32 lastRIndex = GetIndex(r, group);

        r = lastRIndex >= end ? (lastRIndex - end) < blockSize //(blockSize - 1)
                                    ? r - (lastRIndex - end) - 1
                                    : (r & ~(blockMask)) - 1
                              : r;

        u32 lIndex = GetIndex(l, group);
        u32 rIndex = GetIndex(r, group);
        Assert(lIndex >= start);
        Assert(rIndex < end);

        auto GetIndexGroup = [&](u32 index) { return GetIndex(index, group); };

        outMid[jobID] = heuristic->Partition(data, split, l, r, GetIndexGroup, recordL[jobID],
                                             recordR[jobID]);
    });

    // Partition the beginning and the end
    u32 minIndex  = start + count;
    u32 maxIndex  = start;
    u32 globalMid = start;
    u32 bestGroup = 0;
    for (u32 i = 0; i < numJobs; i++)
    {
        globalMid += outMid[i];
        minIndex = Min(GetIndex(outMid[i], i), minIndex);
        if (outMid[i]) maxIndex = Max(GetIndex(outMid[i] - 1, i), maxIndex);
    }

    maxIndex = Min(maxIndex, start + count - 1);
    minIndex = Max(minIndex, start);
    ErrorExit(maxIndex > minIndex, "max: %u, min %u, start %u, count %u\n", maxIndex, minIndex,
              start, count);
    // Assert(maxIndex < start + count);
    // Assert(minIndex >= start);
    u32 out = heuristic->Partition(data, split, minIndex, maxIndex);

    Assert(globalMid == out);
    Assert(out != start && out != start + count);

    Lane8F32 leftGeom(neg_inf);
    Lane8F32 leftCent(neg_inf);
    Lane8F32 rightGeom(neg_inf);
    Lane8F32 rightCent(neg_inf);
    for (u32 i = 0; i < numJobs; i++)
    {
        RecordAOSSplits &l = recordL[i];
        RecordAOSSplits &r = recordR[i];

        leftGeom = Max(leftGeom, l.geomBounds);
        leftCent = Max(leftCent, l.centBounds);

        rightGeom = Max(rightGeom, r.geomBounds);
        rightCent = Max(rightCent, r.centBounds);
    }
    outLeft.geomBounds  = leftGeom;
    outLeft.centBounds  = leftCent;
    outRight.geomBounds = rightGeom;
    outRight.centBounds = rightCent;

    ScratchEnd(temp);
    // printf("%u parallel %u\n", out, split.type);
    return out;
}

template <i32 numBins = 32, typename PrimRef = PrimRef>
struct HeuristicAOSObjectBinning
{
    Bounds8 bins[3][numBins];
    Lane4U32 counts[numBins];
    ObjectBinner<numBins> *binner;

    HeuristicAOSObjectBinning() {}
    HeuristicAOSObjectBinning(ObjectBinner<numBins> *binner) : binner(binner)
    {
        for (u32 dim = 0; dim < 3; dim++)
        {
            for (u32 i = 0; i < numBins; i++)
            {
                bins[dim][i] = Bounds8();
            }
        }
        for (u32 i = 0; i < numBins; i++)
        {
            counts[i] = 0;
        }
    }
    void Bin(const PrimRef *data, u32 start, u32 count)
    {
        u32 alignedCount = count - count % LANE_WIDTH;
        u32 i            = start;

        Lane8F32 lanes[2][8];
        u32 current = 0;
        alignas(32) u32 prevBinIndices[3][8];
        if (count >= LANE_WIDTH)
        {
            Lane8F32 *currentLanes = lanes[current];
            for (u32 laneIndex = 0; laneIndex < 8; laneIndex++)
            {
                currentLanes[laneIndex] = data[i + laneIndex].Load();
            }
            Lane8F32 temp[6];

            Transpose8x6(currentLanes[0], currentLanes[1], currentLanes[2], currentLanes[3],
                         currentLanes[4], currentLanes[5], currentLanes[6], currentLanes[7],
                         temp[0], temp[1], temp[2], temp[3], temp[4], temp[5]);

            Lane8F32 centroids[3] = {
                (temp[3] - temp[0]),
                (temp[4] - temp[1]),
                (temp[5] - temp[2]),
            };
            Assert(All(centroids[0] >= binner->base[0]));
            Assert(All(centroids[1] >= binner->base[1]));
            Assert(All(centroids[2] >= binner->base[2]));
            Lane8U32::Store(prevBinIndices[0], binner->Bin(centroids[0], 0));
            Lane8U32::Store(prevBinIndices[1], binner->Bin(centroids[1], 1));
            Lane8U32::Store(prevBinIndices[2], binner->Bin(centroids[2], 2));
            i += LANE_WIDTH;
            current = !current;
        }
        for (; i < start + alignedCount; i += LANE_WIDTH)
        {
            Lane8F32 *currentLanes = lanes[current];
            Lane8F32 *prevLanes    = lanes[!current];
            for (u32 laneIndex = 0; laneIndex < 8; laneIndex++)
            {
                currentLanes[laneIndex] = data[i + laneIndex].Load();
            }
            Lane8F32 temp[6];

            Transpose8x6(currentLanes[0], currentLanes[1], currentLanes[2], currentLanes[3],
                         currentLanes[4], currentLanes[5], currentLanes[6], currentLanes[7],
                         temp[0], temp[1], temp[2], temp[3], temp[4], temp[5]);

            Lane8F32 centroids[] = {
                (temp[3] - temp[0]),
                (temp[4] - temp[1]),
                (temp[5] - temp[2]),
            };

            Assert(All(centroids[0] >= binner->base[0]));
            Assert(All(centroids[1] >= binner->base[1]));
            Assert(All(centroids[2] >= binner->base[2]));
            Lane8U32 indicesX = binner->Bin(centroids[0], 0);
            Lane8U32 indicesY = binner->Bin(centroids[1], 1);
            Lane8U32 indicesZ = binner->Bin(centroids[2], 2);

            for (u32 dim = 0; dim < 3; dim++)
            {
                for (u32 b = 0; b < LANE_WIDTH; b++)
                {
                    u32 bin = prevBinIndices[dim][b];
                    bins[dim][bin].Extend(prevLanes[b]);
                    counts[bin][dim]++;
                }
            }

            Lane8U32::Store(prevBinIndices[0], indicesX);
            Lane8U32::Store(prevBinIndices[1], indicesY);
            Lane8U32::Store(prevBinIndices[2], indicesZ);
            current = !current;
        }
        if (count >= LANE_WIDTH)
        {
            Lane8F32 *prevLanes = lanes[!current];
            for (u32 dim = 0; dim < 3; dim++)
            {
                for (u32 b = 0; b < LANE_WIDTH; b++)
                {
                    u32 bin = prevBinIndices[dim][b];
                    bins[dim][bin].Extend(prevLanes[b]);
                    counts[bin][dim]++;
                }
            }
        }
        for (; i < start + count; i++)
        {
            Lane8F32 lane     = data[i].Load();
            Lane4F32 low      = Extract4<0>(lane);
            Lane4F32 hi       = Extract4<1>(lane);
            Lane4F32 centroid = (hi - low);
            u32 indices[]     = {
                binner->Bin(centroid[0], 0),
                binner->Bin(centroid[1], 1),
                binner->Bin(centroid[2], 2),
            };
            for (u32 dim = 0; dim < 3; dim++)
            {
                u32 bin = indices[dim];
                bins[dim][bin].Extend(lane);
                counts[bin][dim]++;
            }
        }
    }
    template <typename GetIndex>
    u32 Partition(PrimRef *data, const Split &split, i32 l, i32 r, GetIndex getIndex,
                  RecordAOSSplits &outLeft, RecordAOSSplits &outRight)
    {
        // TIMED_FUNCTION(miscF);
        u32 dim           = split.bestDim;
        u32 bestPos       = split.bestPos;
        Lane8F32 masks[2] = {Lane8F32::Mask(false), Lane8F32::Mask(true)};

        Lane8F32 centLeft(neg_inf);
        Lane8F32 centRight(neg_inf);
        Lane8F32 geomLeft(neg_inf);
        Lane8F32 geomRight(neg_inf);

        i32 start = l;
        i32 end   = r;

        Lane8F32 lanes[8];
        for (;;)
        {
            Lane8F32 lVal;
            Lane8F32 rVal;
            u32 lIndex;
            u32 rIndex;
            while (l <= r)
            {
                lIndex = getIndex(l);
                lVal   = data[lIndex].Load();
                Lane8F32 centroid =
                    ((Shuffle4<1, 1>(lVal) - Shuffle4<0, 0>(lVal))) ^ signFlipMask;

                u32 bin      = binner->Bin(centroid[4 + dim], dim);
                bool isRight = bin >= bestPos;
                if (isRight)
                {
                    centRight = Max(centRight, centroid);
                    geomRight = Max(geomRight, lVal);
                    break;
                }
                centLeft = Max(centLeft, centroid);
                geomLeft = Max(geomLeft, lVal);
                l++;
            }
            while (l <= r)
            {
                Assert(r >= 0);

                rIndex = getIndex(r);
                rVal   = data[rIndex].Load();
                Lane8F32 centroid =
                    ((Shuffle4<1, 1>(rVal) - Shuffle4<0, 0>(rVal))) ^ signFlipMask;

                u32 bin     = binner->Bin(centroid[4 + dim], dim);
                bool isLeft = bin < bestPos;
                if (isLeft)
                {
                    centLeft = Max(centLeft, centroid);
                    geomLeft = Max(geomLeft, rVal);
                    break;
                }
                centRight = Max(centRight, centroid);
                geomRight = Max(geomRight, rVal);
                r--;
            }
            if (l > r) break;

            Swap(data[lIndex], data[rIndex]);
            l++;
            r--;
        }

        outLeft.geomBounds  = geomLeft;
        outRight.geomBounds = geomRight;
        outLeft.centBounds  = centLeft;
        outRight.centBounds = centRight;
        return l;
    }
    u32 Partition(PrimRef *data, const Split &split, i32 l, i32 r)
    {
        u32 dim     = split.bestDim;
        u32 bestPos = split.bestPos;
        for (;;)
        {
            while (l <= r)
            {
                Lane8F32 lRef = data[l].Load();
                Lane8F32 centroid =
                    ((Shuffle4<1, 1>(lRef) - Shuffle4<0, 0>(lRef))) ^ signFlipMask;
                bool isRight = binner->Bin(centroid[4 + dim], dim) >= bestPos;
                if (isRight) break;
                l++;
            }
            while (l <= r)
            {
                Assert(r >= 0);
                Lane8F32 rRef = data[r].Load();
                Lane8F32 centroid =
                    ((Shuffle4<1, 1>(rRef) - Shuffle4<0, 0>(rRef))) ^ signFlipMask;
                bool isLeft = binner->Bin(centroid[4 + dim], dim) < bestPos;
                if (isLeft) break;
                r--;
            }
            if (l > r) break;

            Swap(data[l], data[r]);
            l++;
            r--;
        }
        return l;
    }

    void Merge(const HeuristicAOSObjectBinning &other)
    {
        for (u32 dim = 0; dim < 3; dim++)
        {
            for (u32 i = 0; i < numBins; i++)
            {
                bins[dim][i].Extend(other.bins[dim][i]);
            }
        }
        for (u32 i = 0; i < numBins; i++)
        {
            counts[i] += other.counts[i];
        }
    }
};

template <i32 numBins, typename PrimRefType, typename Polygon8>
struct alignas(32) HeuristicAOSSplitBinning
{
#define UseGeomIDs !std::is_same_v<PrimRefType, PrimRefCompressed>

    using ThisType = HeuristicAOSSplitBinning<numBins, PrimRefType, Polygon8>;
    using PrimRef  = PrimRefType;
    Bounds8 bins[3][numBins];
    Lane4U32 entryCounts[numBins];
    Lane4U32 exitCounts[numBins];

    SplitBinner<numBins> *binner;
    const ScenePrimitives *scene;

    HeuristicAOSSplitBinning() {}

    HeuristicAOSSplitBinning(SplitBinner<numBins> *binner, const ScenePrimitives *scene)
        : binner(binner), scene(scene)
    {
        for (u32 dim = 0; dim < 3; dim++)
        {
            for (u32 i = 0; i < numBins; i++)
            {
                bins[dim][i] = Bounds8();
            }
        }
        for (u32 i = 0; i < numBins; i++)
        {
            entryCounts[i] = 0;
            exitCounts[i]  = 0;
        }
    }

    void Bin(const PrimRef *data, u32 start, u32 count)
    {
        Lane8F32 masks[2]         = {Lane8F32::Mask(false), Lane8F32::Mask(true)};
        u32 binCounts[3][numBins] = {};
        alignas(32) u32 binIndexStart[3][numBins][2 * LANE_WIDTH] = {};

        // u32 faceIndices[3][numBins][2 * LANE_WIDTH]               = {};
        // u32 geomIDs[3][numBins][2 * LANE_WIDTH] = {};
        u32 refIDs[3][numBins][2 * LANE_WIDTH] = {};

        u32 i            = start;
        u32 alignedCount = count - count % LANE_WIDTH;

        Lane8F32 lanes[6];

        for (; i < start + alignedCount; i += LANE_WIDTH)
        {
            Transpose8x6(data[i + 0].Load(), data[i + 1].Load(), data[i + 2].Load(),
                         data[i + 3].Load(), data[i + 4].Load(), data[i + 5].Load(),
                         data[i + 6].Load(), data[i + 7].Load(), lanes[0], lanes[1], lanes[2],
                         lanes[3], lanes[4], lanes[5]);

            Assert(All(-lanes[0] >= binner->base[0]));
            Assert(All(-lanes[1] >= binner->base[1]));
            Assert(All(-lanes[2] >= binner->base[2]));
            Assert(All(lanes[3] >= binner->base[0]));
            Assert(All(lanes[4] >= binner->base[1]));
            Assert(All(lanes[5] >= binner->base[2]));

            Lane8U32 indexMinArr[3] = {
                binner->BinMin(lanes[0], 0),
                binner->BinMin(lanes[1], 1),
                binner->BinMin(lanes[2], 2),
            };
            Lane8U32 indexMaxArr[3] = {
                binner->BinMax(lanes[3], 0),
                binner->BinMax(lanes[4], 1),
                binner->BinMax(lanes[5], 2),
            };

            Assert(All(AsFloat(indexMinArr[0]) <= AsFloat(indexMaxArr[0])));
            Assert(All(AsFloat(indexMinArr[1]) <= AsFloat(indexMaxArr[1])));
            Assert(All(AsFloat(indexMinArr[2]) <= AsFloat(indexMaxArr[2])));

            u32 bitMask[3] = {};

            for (u32 dim = 0; dim < 3; dim++)
            {
                alignas(32) u32 indexMins[8];
                alignas(32) u32 indexMaxs[8];
                Lane8U32::Store(indexMins, indexMinArr[dim]);
                Lane8U32::Store(indexMaxs, indexMaxArr[dim]);
                for (u32 b = 0; b < LANE_WIDTH; b++)
                {
                    u32 indexMin = indexMins[b];
                    u32 diff     = indexMaxs[b] - indexMin;

                    switch (diff)
                    {
                        case 0:
                        {
                            entryCounts[indexMin][dim] += 1;
                            exitCounts[indexMin][dim] += 1;
                            bins[dim][indexMin].Extend(data[i + b].Load());
                        }
                        break;
                        default:
                        {
                            bitMask[dim] |= (1 << diff);
                            refIDs[dim][diff][binCounts[dim][diff]]        = i + b;
                            binIndexStart[dim][diff][binCounts[dim][diff]] = indexMin;
                            binCounts[dim][diff]++;
                        }
                    }
                }
            }
            for (u32 dim = 0; dim < 3; dim++)
            {
                u32 numIters = PopCount(bitMask[dim]);
                for (u32 iter = 0; iter < numIters; iter++)
                {
                    u32 bin = Bsf(bitMask[dim]);
                    if (binCounts[dim][bin] >= LANE_WIDTH)
                    {
                        Assert(binCounts[dim][bin] <= ArrayLength(binIndexStart[0][0]));
                        binCounts[dim][bin] -= LANE_WIDTH;
                        u32 binCount = binCounts[dim][bin];

                        u32 geomIDs[8];
                        u32 faceIndices[8];
                        Bounds8 bounds[2][LANE_WIDTH];
                        for (u32 b = 0; b < LANE_WIDTH; b++)
                        {
                            const PrimRef &ref = data[refIDs[dim][bin][binCount + b]];
                            if constexpr (UseGeomIDs) geomIDs[b] = ref.geomID;
                            faceIndices[b] = ref.primID;
                            bounds[0][b].v = ref.Load();
                        }

                        Polygon8 tri;
                        if constexpr (UseGeomIDs)
                            Polygon8::Load(scene, dim, geomIDs, faceIndices, &tri);
                        else Polygon8::Load(scene, dim, faceIndices, &tri);

                        Lane8U32 startBin =
                            Lane8U32::LoadU(binIndexStart[dim][bin] + binCount);

                        alignas(32) u32 binIndices[LANE_WIDTH];
                        alignas(32) u32 startBins[LANE_WIDTH];
                        Lane8U32::Store(startBins, startBin);

                        u32 current = 0;

                        u16 binMasks[LANE_WIDTH] = {};
                        for (u32 d = 0; d < bin; d++)
                        {
                            Lane8U32::Store(binIndices, startBin);
                            startBin += 1u;
                            Lane8F32 splitPos = binner->GetSplitValue(startBin, dim);

                            ClipPolygon(dim, tri, splitPos, bounds[current], bounds[!current]);

                            for (u32 b = 0; b < LANE_WIDTH; b++)
                            {
                                u32 bit      = !bounds[current][b].Empty();
                                u32 binIndex = binIndices[b];
                                bins[dim][binIndex].MaskExtend(masks[bit], bounds[current][b]);
                                binMasks[b] |= (bit << d);
                            }
                            current = !current;
                        }
                        for (u32 b = 0; b < LANE_WIDTH; b++)
                        {
                            u32 bit = !bounds[current][b].Empty();
                            binMasks[b] |= (bit << bin);
                            u32 binIndex = binIndices[b] + 1;
                            bins[dim][binIndex].MaskExtend(masks[bit], bounds[current][b]);

                            u32 tzcnt = _tzcnt_u16(binMasks[b]);
                            Assert(tzcnt != 16);
                            u32 startIndex = startBins[b] + tzcnt;
                            u32 lzcnt      = _lzcnt_u32(binMasks[b]);
                            Assert(lzcnt != 32);
                            u32 endIndex = startBins[b] + 31 - lzcnt;
                            Assert(startIndex >= 0 && startIndex < numBins);
                            Assert(endIndex >= 0 && endIndex < numBins &&
                                   endIndex >= startIndex);
                            entryCounts[startIndex][dim] += 1;
                            exitCounts[endIndex][dim] += 1;
                        }
                    }
                    bitMask[dim] &= bitMask[dim] - 1;
                }
            }
        }
        // Finish the remaining primitives
        for (; i < start + count; i++)
        {
            const PrimRef *ref = &data[i];

            for (u32 dim = 0; dim < 3; dim++)
            {
                u32 binIndexMin = binner->BinMin(ref->min[dim], dim);
                u32 binIndexMax = binner->BinMax(ref->max[dim], dim);

                Assert(binIndexMax >= binIndexMin);

                u32 diff = binIndexMax - binIndexMin;
                switch (diff)
                {
                    case 0:
                    {
                        entryCounts[binIndexMin][dim] += 1;
                        exitCounts[binIndexMax][dim] += 1;
                        bins[dim][binIndexMin].Extend(Lane8F32::LoadU(ref));
                    }
                    break;
                    default:
                    {
                        refIDs[dim][diff][binCounts[dim][diff]]        = i;
                        binIndexStart[dim][diff][binCounts[dim][diff]] = binIndexMin;
                        binCounts[dim][diff]++;
                    }
                }
            }
        }
        // Empty the bins
        Lane8F32 posInf(pos_inf);
        Lane8F32 negInf(neg_inf);
        for (u32 dim = 0; dim < 3; dim++)
        {
            for (u32 diff = 1; diff < numBins; diff++)
            {
                u32 remainingCount = binCounts[dim][diff];
                Assert(remainingCount <= ArrayLength(binIndexStart[0][0]));

                const u32 numIters = ((remainingCount + 7) >> 3);
                for (u32 remaining = 0; remaining < numIters; remaining++)
                {
                    u32 numPrims = Min(remainingCount, 8u);

                    u32 geomIDs[8]     = {};
                    u32 faceIndices[8] = {};
                    Bounds8 bounds[2][LANE_WIDTH];
                    u32 qStart = remaining * LANE_WIDTH;
                    for (u32 b = 0; b < numPrims; b++)
                    {
                        const PrimRef &ref = data[refIDs[dim][diff][qStart + b]];
                        if constexpr (UseGeomIDs) geomIDs[b] = ref.geomID;
                        faceIndices[b] = ref.primID;
                        bounds[0][b].v = ref.Load();
                    }

                    Polygon8 tri;
                    if constexpr (UseGeomIDs)
                        Polygon8::Load(scene, dim, geomIDs, faceIndices, &tri);
                    else Polygon8::Load(scene, dim, faceIndices, &tri);

                    Lane8U32 startBin = Lane8U32::LoadU(binIndexStart[dim][diff] + qStart);

                    alignas(32) u32 binIndices[8];
                    alignas(32) u32 startBins[LANE_WIDTH];
                    Lane8U32::Store(startBins, startBin);

                    u32 current              = 0;
                    u16 binMasks[LANE_WIDTH] = {};
                    for (u32 d = 0; d < diff; d++)
                    {
                        Lane8U32::Store(binIndices, startBin);
                        startBin += 1u;
                        Lane8F32 splitPos = binner->GetSplitValue(startBin, dim);
                        ClipPolygon(dim, tri, splitPos, bounds[current], bounds[!current]);
                        for (u32 b = 0; b < numPrims; b++)
                        {
                            u32 bit      = !bounds[current][b].Empty();
                            u32 binIndex = binIndices[b];
                            bins[dim][binIndex].MaskExtend(masks[bit], bounds[current][b]);
                            binMasks[b] |= (bit << d);
                        }
                        current = !current;
                    }
                    for (u32 b = 0; b < numPrims; b++)
                    {
                        u32 bit = !bounds[current][b].Empty();
                        binMasks[b] |= (bit << diff);
                        u32 binIndex = binIndices[b] + 1;
                        bins[dim][binIndex].MaskExtend(masks[bit], bounds[current][b]);

                        u32 tzcnt = _tzcnt_u16(binMasks[b]);
                        Assert(tzcnt != 16);
                        u32 startIndex = startBins[b] + tzcnt;
                        u32 lzcnt      = _lzcnt_u32(binMasks[b]);
                        Assert(lzcnt != 32);
                        u32 endIndex = startBins[b] + 31 - lzcnt;
                        Assert(startIndex >= 0 && startIndex < numBins);
                        Assert(endIndex >= 0 && endIndex < numBins && endIndex >= startIndex);
                        entryCounts[startIndex][dim]++;
                        exitCounts[endIndex][dim]++;
                    }
                    remainingCount -= LANE_WIDTH;
                }
            }
        }
    }
    void Split(PrimRef *data, const Split &split, u32 splitOffset, u32 splitMax, u32 start,
               u32 count)
    {
        // TIMED_FUNCTION(miscF);
        u32 dim                        = split.bestDim;
        u32 bestPos                    = split.bestPos;
        u32 refIDQueue[LANE_WIDTH * 2] = {};
        Lane8F32 masks[2]              = {Lane8F32::Mask(false), Lane8F32::Mask(true)};

        Lane8F32 lanes[8];

        u32 splitCount = 0;

        Lane8F32 geomLeft(neg_inf);
        Lane8F32 geomRight(neg_inf);

        Lane8F32 centLeft(neg_inf);
        Lane8F32 centRight(neg_inf);

        u32 totalL = 0;
        u32 totalR = 0;

        Lane8F32 splitValue = binner->GetSplitValue(Lane8U32(split.bestPos), dim);

        for (u32 i = start; i < start + count; i++)
        {
            PrimRef &primRef       = data[i];
            u32 minBin             = binner->BinMin(primRef.min[dim], dim);
            u32 maxBin             = binner->BinMax(primRef.max[dim], dim);
            bool isSplit           = minBin < split.bestPos && maxBin >= split.bestPos;
            refIDQueue[splitCount] = i;
            splitCount += isSplit;
            if (splitCount >= LANE_WIDTH)
            {
                splitCount -= LANE_WIDTH;

                u32 geomIDs[8];
                u32 faceIDQueue[8];
                Bounds8 gL[LANE_WIDTH];
                Bounds8 gR[LANE_WIDTH];
                for (u32 b = 0; b < LANE_WIDTH; b++)
                {
                    const PrimRef &ref = data[refIDQueue[splitCount + b]];
                    if constexpr (UseGeomIDs) geomIDs[b] = ref.geomID;
                    faceIDQueue[b] = ref.primID;
                    gL[b].v        = ref.Load();
                }

                Polygon8 tri;
                if constexpr (UseGeomIDs)
                    Polygon8::Load(scene, dim, geomIDs, faceIDQueue, &tri);
                else Polygon8::Load(scene, dim, faceIDQueue, &tri);
                ClipPolygon(dim, tri, splitValue, gL, gR);

                for (u32 queueIndex = 0; queueIndex < LANE_WIDTH; queueIndex++)
                {
                    const u32 refID = refIDQueue[splitCount + queueIndex];
                    bool notEmpty   = !(gR[queueIndex].Empty() | gL[queueIndex].Empty());
                    if (notEmpty)
                    {
                        const u32 splitLoc = splitOffset++;
                        Assert(splitLoc < splitMax);

                        data[refID]    = PrimRef(gL[queueIndex].v);
                        data[splitLoc] = PrimRef(gR[queueIndex].v);
                    }
                }
            }
        }

        // Flush the queue
        u32 remainingCount = splitCount;
        const u32 numIters = (remainingCount + 7) >> 3;
        for (u32 remaining = 0; remaining < numIters; remaining++)
        {
            u32 qStart   = remaining * LANE_WIDTH;
            u32 numPrims = Min(remainingCount, LANE_WIDTH);
            Bounds8 gL[LANE_WIDTH];
            Bounds8 gR[LANE_WIDTH];
            u32 geomIDs[8]     = {};
            u32 faceIDQueue[8] = {};
            for (u32 b = 0; b < numPrims; b++)
            {
                const PrimRef &ref = data[refIDQueue[qStart + b]];
                if constexpr (UseGeomIDs) geomIDs[b] = ref.geomID;
                faceIDQueue[b] = ref.primID;
                gL[b].v        = ref.Load();
            }
            Polygon8 tri;
            if constexpr (UseGeomIDs) Polygon8::Load(scene, dim, geomIDs, faceIDQueue, &tri);
            else Polygon8::Load(scene, dim, faceIDQueue, &tri);
            ClipPolygon(dim, tri, splitValue, gL, gR);

            for (u32 queueIndex = 0; queueIndex < numPrims; queueIndex++)
            {
                const u32 refID = refIDQueue[qStart + queueIndex];
                bool notEmpty   = !(gR[queueIndex].Empty() | gL[queueIndex].Empty());
                if (notEmpty)
                {
                    const u32 splitLoc = splitOffset++;
                    Assert(splitLoc < splitMax);

                    data[refID]    = PrimRef(gL[queueIndex].v);
                    data[splitLoc] = PrimRef(gR[queueIndex].v);
                }
            }

            remainingCount -= LANE_WIDTH;
        }
        Assert(splitOffset == splitMax);
    }
    template <typename GetIndex>
    u32 Partition(PrimRef *data, const struct Split &split, i32 l, i32 r, GetIndex getIndex,
                  RecordAOSSplits &outLeft, RecordAOSSplits &outRight)
    {
        // TIMED_FUNCTION(miscF);
        u32 dim           = split.bestDim;
        u32 bestPos       = split.bestPos;
        Lane8F32 masks[2] = {Lane8F32::Mask(false), Lane8F32::Mask(true)};

        Lane8F32 centLeft(neg_inf);
        Lane8F32 centRight(neg_inf);
        Lane8F32 geomLeft(neg_inf);
        Lane8F32 geomRight(neg_inf);

        i32 start = l;
        i32 end   = r;

        Lane8F32 lanes[8];
        for (;;)
        {
            Lane8F32 lVal;
            Lane8F32 rVal;
            u32 lIndex;
            u32 rIndex;
            while (l <= r)
            {
                lIndex       = getIndex(l);
                PrimRef &ref = data[lIndex];
                lVal         = ref.Load();
                Lane8F32 centroid =
                    ((Shuffle4<1, 1>(lVal) - Shuffle4<0, 0>(lVal))) ^ signFlipMask;
                f32 testCentroid = (ref.max[dim] - ref.min[dim]) * 0.5f;

                bool isRight = binner->Bin(testCentroid, dim) >= split.bestPos;
                if (isRight)
                {
                    centRight = Max(centRight, centroid);
                    geomRight = Max(geomRight, lVal);
                    break;
                }
                centLeft = Max(centLeft, centroid);
                geomLeft = Max(geomLeft, lVal);
                l++;
            }
            while (l <= r)
            {
                Assert(r >= 0);

                rIndex       = getIndex(r);
                PrimRef &ref = data[rIndex];
                rVal         = ref.Load();
                Lane8F32 centroid =
                    ((Shuffle4<1, 1>(rVal) - Shuffle4<0, 0>(rVal))) ^ signFlipMask;
                f32 testCentroid = (ref.max[dim] - ref.min[dim]) * 0.5f;

                // bool isLeft = ref.max[dim] <= split.bestValue;
                bool isLeft = binner->Bin(testCentroid, dim) < split.bestPos;
                if (isLeft)
                {
                    centLeft = Max(centLeft, centroid);
                    geomLeft = Max(geomLeft, rVal);
                    break;
                }
                centRight = Max(centRight, centroid);
                geomRight = Max(geomRight, rVal);
                r--;
            }
            if (l > r) break;

            Swap(data[lIndex], data[rIndex]);
            l++;
            r--;
        }

        outLeft.geomBounds  = geomLeft;
        outRight.geomBounds = geomRight;
        outLeft.centBounds  = centLeft;
        outRight.centBounds = centRight;
        return l;
    }
    u32 Partition(PrimRef *data, const struct Split &split, i32 l, i32 r)
    {
        u32 dim     = split.bestDim;
        u32 bestPos = split.bestPos;
        for (;;)
        {
            while (l <= r)
            {
                PrimRef &ref = data[l];
                // bool isRight  = binner->BinMin(ref.min[dim], dim) >= bestPos;
                f32 centroid = (ref.max[dim] - ref.min[dim]) * 0.5f;
                bool isRight = binner->Bin(centroid, dim) >= split.bestPos;
                if (isRight) break;
                l++;
            }
            while (l <= r)
            {
                Assert(r >= 0);
                PrimRef &ref = data[r];
                // bool isLeft   = binner->BinMin(ref.min[dim], dim) < bestPos;
                f32 centroid = (ref.max[dim] - ref.min[dim]) * 0.5f;
                bool isLeft  = binner->Bin(centroid, dim) < split.bestPos;
                if (isLeft) break;
                r--;
            }
            if (l > r) break;

            Swap(data[l], data[r]);
            l++;
            r--;
        }
        return l;
    }

    void Merge(const ThisType &other)
    {
        for (u32 i = 0; i < numBins; i++)
        {
            entryCounts[i] += other.entryCounts[i];
            exitCounts[i] += other.exitCounts[i];
            for (u32 dim = 0; dim < 3; dim++)
            {
                bins[dim][i].Extend(other.bins[dim][i]);
            }
        }
    }
#undef UseGeomIDs
};

template <i32 numObjectBins = 32, typename PrimRef = PrimRef>
Split SAHObjectBinning(const RecordAOSSplits &record, const PrimRef *primRefs,
                       HeuristicAOSObjectBinning<numObjectBins, PrimRef> *&objectBinHeuristic,
                       u64 &popPos, u32 logBlockSize)
{
    using OBin     = HeuristicAOSObjectBinning<numObjectBins, PrimRef>;
    TempArena temp = ScratchStart(0, 0);
    popPos         = ArenaPos(temp.arena);

    // Stack allocate the heuristics since they store the centroid and geom bounds we'll need
    // later
    ObjectBinner<numObjectBins> *objectBinner =
        PushStructConstruct(temp.arena, ObjectBinner<numObjectBins>)(record.centBounds);
    objectBinHeuristic = PushStructNoZero(temp.arena, OBin);
    if (record.count > PARALLEL_THRESHOLD)
    {
        const u32 groupSize = PARALLEL_THRESHOLD;
        ParallelReduce<OBin>(
            objectBinHeuristic, record.start, record.count, groupSize,
            [&](OBin &binner, u32 jobID, u32 start, u32 count) {
                binner.Bin(primRefs, start, count);
            },
            [&](OBin &l, const OBin &r) { l.Merge(r); }, objectBinner);
    }
    else
    {
        new (objectBinHeuristic) OBin(objectBinner);
        objectBinHeuristic->Bin(primRefs, record.start, record.count);
    }
    struct Split objectSplit = BinBest(objectBinHeuristic->bins, objectBinHeuristic->counts,
                                       objectBinner, logBlockSize);
    objectSplit.type         = Split::Object;
    return objectSplit;
}

template <i32 numObjectBins = 32, typename PrimRef = PrimRef>
void FinalizeObjectSplit(HeuristicAOSObjectBinning<numObjectBins, PrimRef> *objectBinHeuristic,
                         Split &objectSplit, u64 popPos)
{
    u32 lCount = 0;
    for (u32 i = 0; i < objectSplit.bestPos; i++)
    {
        lCount += objectBinHeuristic->counts[i][objectSplit.bestDim];
    }
    u32 rCount = 0;
    for (u32 i = objectSplit.bestPos; i < numObjectBins; i++)
    {
        rCount += objectBinHeuristic->counts[i][objectSplit.bestDim];
    }

    objectSplit.ptr      = (void *)objectBinHeuristic;
    objectSplit.allocPos = popPos;
    objectSplit.numLeft  = lCount;
    objectSplit.numRight = rCount;
}

template <typename PrimRef, typename Record>
void MoveExtendedRanges(const Split &split, const u32 newEnd, const Record &record,
                        PrimRef *primRefs, u32 mid, Record &outLeft, Record &outRight)
{
    u32 numLeft  = mid - record.start;
    u32 numRight = newEnd - mid;

    f32 weight         = (f32)(numLeft) / (numLeft + numRight);
    u32 remainingSpace = (record.extEnd - record.start - numLeft - numRight);
    u32 extSizeLeft    = Min((u32)(remainingSpace * weight), remainingSpace);
    u32 extSizeRight   = remainingSpace - extSizeLeft;

    u32 shift      = Max(extSizeLeft, numRight);
    u32 numToShift = Min(extSizeLeft, numRight);

    if (numToShift != 0)
    {
        if (numToShift > PARALLEL_THRESHOLD)
        {
            ParallelFor(mid, numToShift, PARALLEL_THRESHOLD,
                        [&](u32 jobID, u32 start, u32 count) {
                            for (u32 i = start; i < start + count; i++)
                            {
                                Assert(i + shift < record.extEnd);
                                primRefs[i + shift] = primRefs[i];
                            }
                        });
        }
        else
        {
            for (u32 i = mid; i < mid + numToShift; i++)
            {
                Assert(i + shift < record.extEnd);
                primRefs[i + shift] = primRefs[i];
            }
        }
    }

    Assert(numLeft <= record.count);
    Assert(numRight <= record.count);

    outLeft.SetRange(record.start, numLeft, record.start + numLeft + extSizeLeft);
    outRight.SetRange(outLeft.extEnd, numRight, record.extEnd);
    u32 rightExtSize = outRight.ExtSize();
    Assert(rightExtSize == extSizeRight);
}

template <typename Record, typename PrimRef>
u32 SplitFallback(const Record &record, Split &split, const PrimRef *primRefs, Record &outLeft,
                  Record &outRight)
{
    u32 lCount = record.count / 2;
    u32 rCount = record.count - lCount;
    u32 mid    = record.start + lCount;
    Bounds8 geomLeft;
    Bounds8 centLeft;
    Bounds8 geomRight;
    Bounds8 centRight;
    for (u32 i = record.start; i < mid; i++) // record.start + record.count; i++)
    {
        const PrimRef *ref = &primRefs[i];
        Lane8F32 m256      = Lane8F32::LoadU(ref);
        Lane8F32 centroid  = ((Shuffle4<1, 1>(m256) - Shuffle4<0, 0>(m256))) ^ signFlipMask;
        geomLeft.Extend(m256);
        centLeft.Extend(centroid);
    }
    for (u32 i = mid; i < record.End(); i++)
    {
        const PrimRef *ref = &primRefs[i];
        Lane8F32 m256      = Lane8F32::LoadU(ref);
        Lane8F32 centroid  = ((Shuffle4<1, 1>(m256) - Shuffle4<0, 0>(m256))) ^ signFlipMask;
        geomRight.Extend(m256);
        centRight.Extend(centroid);
    }
    outLeft.geomBounds  = geomLeft.v;
    outLeft.centBounds  = centLeft.v;
    outRight.geomBounds = geomRight.v;
    outRight.centBounds = centRight.v;
    split.numLeft       = lCount;
    split.numRight      = rCount;
    return mid;
}

// SBVH
static const f32 sbvhAlpha = 1e-5;
template <typename PrimRefType, typename Polygon8Type, i32 numObjectBins = 32,
          i32 numSpatialBins = 16>
struct HeuristicSpatialSplits
{
    using Record = RecordAOSSplits;

    using PrimRef  = PrimRefType;
    using Polygon8 = Polygon8Type;

    using HSplit = HeuristicAOSSplitBinning<numSpatialBins, PrimRef, Polygon8>;
    using OBin   = HeuristicAOSObjectBinning<numObjectBins, PrimRef>;

    const ScenePrimitives *scene;
    f32 rootArea;
    u32 logBlockSize;
    PrimRef *primRefs;

    HeuristicSpatialSplits() {}
    HeuristicSpatialSplits(PrimRef *data, const ScenePrimitives *scene, f32 rootArea,
                           u32 logBlockSize = 0)
        : primRefs(data), scene(scene), rootArea(rootArea), logBlockSize(logBlockSize)
    {
    }

    Split Bin(const Record &record, u32 blockSize = 1)
    {
        // Object splits
        TempArena temp = ScratchStart(0, 0);
        u64 popPos     = ArenaPos(temp.arena);
        OBin *objectBinHeuristic;

        struct Split objectSplit =
            SAHObjectBinning(record, primRefs, objectBinHeuristic, popPos, logBlockSize);

        // Stack allocate the heuristics since they store the centroid and geom bounds we'll
        // need later

        Bounds8 geomBoundsL;
        for (u32 i = 0; i < objectSplit.bestPos; i++)
        {
            geomBoundsL.Extend(objectBinHeuristic->bins[objectSplit.bestDim][i]);
        }
        Bounds8 geomBoundsR;
        for (u32 i = objectSplit.bestPos; i < numObjectBins; i++)
        {
            geomBoundsR.Extend(objectBinHeuristic->bins[objectSplit.bestDim][i]);
        }

        f32 lambda = HalfArea(Intersect(geomBoundsL, geomBoundsR));
        if (lambda > sbvhAlpha * rootArea)
        {
            // Spatial splits
            SplitBinner<numSpatialBins> *splitBinner = PushStructConstruct(
                temp.arena, SplitBinner<numSpatialBins>)(record.geomBounds);

            HSplit *splitHeuristic = PushStructNoZero(temp.arena, HSplit);
            ParallelForOutput output;
            if (record.count > PARALLEL_THRESHOLD)
            {
                output = ParallelFor<HSplit>(
                    temp, record.start, record.count, PARALLEL_THRESHOLD,
                    [&](HSplit &binner, u32 jobID, u32 start, u32 count) {
                        binner.Bin(primRefs, start, count);
                    },
                    splitBinner, scene);
                Reduce(
                    *splitHeuristic, output, [&](HSplit &l, const HSplit &r) { l.Merge(r); },
                    splitBinner, scene);
            }
            else
            {
                new (splitHeuristic) HSplit(splitBinner, scene);
                splitHeuristic->Bin(primRefs, record.start, record.count);
            }
            struct Split spatialSplit =
                BinBest(splitHeuristic->bins, splitHeuristic->entryCounts,
                        splitHeuristic->exitCounts, splitBinner, logBlockSize);
            spatialSplit.type = Split::Spatial;
            u32 lCount        = 0;
            for (u32 i = 0; i < spatialSplit.bestPos; i++)
            {
                lCount += splitHeuristic->entryCounts[i][spatialSplit.bestDim];
            }
            u32 rCount = 0;
            for (u32 i = spatialSplit.bestPos; i < numSpatialBins; i++)
            {
                rCount += splitHeuristic->exitCounts[i][spatialSplit.bestDim];
            }
            u32 totalNumSplits = lCount + rCount - record.count;
            if (spatialSplit.bestSAH < objectSplit.bestSAH &&
                totalNumSplits <= record.ExtSize())
            {
                if (record.count > PARALLEL_THRESHOLD)
                {
                    u32 *splitCounts = PushArrayNoZero(temp.arena, u32, output.num + 1);
                    u32 offset       = record.End();
                    for (u32 i = 0; i < output.num; i++)
                    {
                        HSplit *heur = &((HSplit *)output.out)[i];
                        lCount       = 0;
                        rCount       = 0;
                        for (u32 j = 0; j < spatialSplit.bestPos; j++)
                        {
                            lCount += heur->entryCounts[j][spatialSplit.bestDim];
                        }
                        for (u32 j = spatialSplit.bestPos; j < numSpatialBins; j++)
                        {
                            rCount += heur->exitCounts[j][spatialSplit.bestDim];
                        }
                        u32 groupSize =
                            i == output.num - 1
                                ? record.End() - (record.start + output.groupSize * i)
                                : output.groupSize;
                        Assert(lCount + rCount >= groupSize);
                        u32 splitCount = lCount + rCount - groupSize;
                        splitCounts[i] = offset;
                        offset += splitCount;
                    }
                    splitCounts[output.num]           = offset;
                    spatialSplit.payload.splitOffsets = splitCounts;
                    spatialSplit.payload.num          = output.num;
                }

                spatialSplit.ptr      = (void *)splitHeuristic;
                spatialSplit.allocPos = popPos;
                spatialSplit.numLeft  = lCount;
                spatialSplit.numRight = rCount;

                return spatialSplit;
            }
        }

        FinalizeObjectSplit(objectBinHeuristic, objectSplit, popPos);
        return objectSplit;
    }
    void FlushState(struct Split split)
    {
        TempArena temp = ScratchStart(0, 0);
        ArenaPopTo(temp.arena, split.allocPos);
    }
    void Split(struct Split split, const Record &record, Record &outLeft, Record &outRight)
    {
        // NOTE: Split must be called from the same thread as Bin
        TempArena temp = ScratchStart(0, 0);
        u32 mid;
        u32 newEnd;

        if (split.bestSAH == f32(pos_inf))
        {
            mid    = SplitFallback(record, split, primRefs, outLeft, outRight);
            newEnd = record.End();
        }
        else
        {
            switch (split.type)
            {
                case Split::Object:
                {
                    OBin *heuristic = (OBin *)(split.ptr);
                    mid    = PartitionParallel(heuristic, primRefs, split, record.start,
                                               record.count, outLeft, outRight);
                    newEnd = record.End();
                }
                break;
                case Split::Spatial:
                {
                    HSplit *heuristic = (HSplit *)(split.ptr);
                    if (record.count > PARALLEL_THRESHOLD)
                    {
                        u32 *offsets = split.payload.splitOffsets;
                        newEnd       = offsets[split.payload.num];
                        u32 num      = split.payload.num;
                        ParallelFor(record.start, record.count, PARALLEL_THRESHOLD,
                                    [&](u32 jobID, u32 start, u32 count) {
                                        u32 splitOffset = offsets[jobID];
                                        u32 splitMax    = offsets[jobID + 1];
                                        heuristic->Split(primRefs, split, splitOffset,
                                                         splitMax, start, count);
                                    });
                    }
                    else
                    {
                        u32 numSplits = split.numLeft + split.numRight - record.count;
                        newEnd        = record.End() + numSplits;
                        heuristic->Split(primRefs, split, record.End(), newEnd, record.start,
                                         record.count);
                    }
                    Assert(newEnd <= record.extEnd);
                    mid = PartitionParallel(heuristic, primRefs, split, record.start,
                                            newEnd - record.start, outLeft, outRight);
                }
                break;
            }
            Assert((Movemask(outLeft.geomBounds <= record.geomBounds) & 0x77) == 0x77);
            Assert((Movemask(outRight.geomBounds <= record.geomBounds) & 0x77) == 0x77);
        }

        MoveExtendedRanges(split, newEnd, record, primRefs, mid, outLeft, outRight);
        Assert(outLeft.count > 0);
        Assert(outRight.count > 0);

        // error check
#ifdef DEBUG
        {
            switch (split.type)
            {
                case Split::Object:
                {
                    OBin *heuristic = (OBin *)(split.ptr);
                    for (u32 i = outLeft.start; i < outLeft.End(); i++)
                    {
                        const PrimRef *ref = &primRefs[i];
                        Lane8F32 v         = Lane8F32::LoadU(ref);
                        Lane8F32 centroid =
                            ((Shuffle4<1, 1>(v) - Shuffle4<0, 0>(v))) ^ signFlipMask;
                        u32 pos =
                            heuristic->binner->Bin(centroid[4 + split.bestDim], split.bestDim);
                        Assert(pos < split.bestPos);
                        u32 gMask = Movemask(v <= outLeft.geomBounds) & 0x77;
                        Assert(gMask == 0x77);
                        u32 cMask = Movemask(centroid <= outLeft.centBounds) & 0x77;
                        Assert(cMask == 0x77);
                    }
                    for (u32 i = outRight.start; i < outRight.End(); i++)
                    {
                        const PrimRef *ref = &primRefs[i];
                        Lane8F32 v         = Lane8F32::LoadU(ref);
                        Lane8F32 centroid =
                            ((Shuffle4<1, 1>(v) - Shuffle4<0, 0>(v))) ^ signFlipMask;
                        u32 pos =
                            heuristic->binner->Bin(centroid[4 + split.bestDim], split.bestDim);
                        Assert(pos >= split.bestPos);
                        u32 gMask = Movemask(v <= outRight.geomBounds) & 0x77;
                        u32 cMask = Movemask(centroid <= outRight.centBounds) & 0x77;
                        Assert(gMask == 0x77);
                        Assert(cMask == 0x77);
                    }
                }
                break;

                case Split::Spatial:
                {
                    HSplit *heuristic = (HSplit *)(split.ptr);
                    for (u32 i = outLeft.start; i < outLeft.End(); i++)
                    {
                        const PrimRef *ref = &primRefs[i];
                        Lane8F32 v         = Lane8F32::LoadU(ref);
                        Lane8F32 centroid =
                            ((Shuffle4<1, 1>(v) - Shuffle4<0, 0>(v))) ^ signFlipMask;

                        f32 c = (ref->max[split.bestDim] - ref->min[split.bestDim]) * 0.5f;
                        Assert(heuristic->binner->Bin(c, split.bestDim) < split.bestPos);

                        u32 gMask = Movemask(v <= outLeft.geomBounds) & 0x77;
                        Assert(gMask == 0x77);

                        u32 cMask = Movemask(centroid <= outLeft.centBounds) & 0x77;
                        Assert(cMask == 0x77);
                    }
                    for (u32 i = outRight.start; i < outRight.End(); i++)
                    {
                        const PrimRef *ref = &primRefs[i];
                        Lane8F32 v         = Lane8F32::LoadU(ref);
                        Lane8F32 centroid =
                            ((Shuffle4<1, 1>(v) - Shuffle4<0, 0>(v))) ^ signFlipMask;

                        f32 c = (ref->max[split.bestDim] - ref->min[split.bestDim]) * 0.5f;
                        Assert(heuristic->binner->Bin(c, split.bestDim) >= split.bestPos);

                        u32 gMask = Movemask(v <= outRight.geomBounds) & 0x77;
                        Assert(gMask == 0x77);

                        u32 cMask = Movemask(centroid <= outRight.centBounds) & 0x77;
                        Assert(cMask == 0x77);
                    }
                }
                break;
            }
        }

#endif
        ArenaPopTo(temp.arena, split.allocPos);
    }
};

template <typename PrimRefType, i32 numObjectBins = 32>
struct HeuristicObjectBinning
{
    using Record  = RecordAOSSplits;
    using PrimRef = PrimRefType;
    using OBin    = HeuristicAOSObjectBinning<numObjectBins, PrimRef>;

    const ScenePrimitives *scene;
    u32 logBlockSize;
    PrimRef *primRefs;

    HeuristicObjectBinning() {}
    HeuristicObjectBinning(PrimRef *data, const ScenePrimitives *scene, u32 logBlockSize = 0)
        : primRefs(data), scene(scene), logBlockSize(logBlockSize)
    {
    }

    Split Bin(const Record &record, u32 blockSize = 1)
    {
        // Object splits
        TempArena temp = ScratchStart(0, 0);
        u64 popPos     = ArenaPos(temp.arena);
        OBin *objectBinHeuristic;

        struct Split objectSplit =
            SAHObjectBinning(record, primRefs, objectBinHeuristic, popPos, logBlockSize);

        Bounds8 geomBoundsL;
        for (u32 i = 0; i < objectSplit.bestPos; i++)
        {
            geomBoundsL.Extend(objectBinHeuristic->bins[objectSplit.bestDim][i]);
        }
        Bounds8 geomBoundsR;
        for (u32 i = objectSplit.bestPos; i < numObjectBins; i++)
        {
            geomBoundsR.Extend(objectBinHeuristic->bins[objectSplit.bestDim][i]);
        }

        FinalizeObjectSplit(objectBinHeuristic, objectSplit, popPos);
        return objectSplit;
    }
    void FlushState(struct Split split)
    {
        TempArena temp = ScratchStart(0, 0);
        ArenaPopTo(temp.arena, split.allocPos);
    }
    void Split(struct Split split, const Record &record, Record &outLeft, Record &outRight)
    {
        // NOTE: Split must be called from the same thread as Bin
        TempArena temp = ScratchStart(0, 0);
        u32 mid;
        u32 newEnd;

        if (split.bestSAH == f32(pos_inf))
        {
            mid    = SplitFallback(record, split, primRefs, outLeft, outRight);
            newEnd = record.End();
        }
        else
        {
            OBin *heuristic = (OBin *)(split.ptr);
            mid    = PartitionParallel(heuristic, primRefs, split, record.start, record.count,
                                       outLeft, outRight);
            newEnd = record.End();

            Assert((Movemask(outLeft.geomBounds <= record.geomBounds) & 0x77) == 0x77);
            Assert((Movemask(outRight.geomBounds <= record.geomBounds) & 0x77) == 0x77);
        }

        Assert(outLeft.count > 0);
        Assert(outRight.count > 0);

        // error check
#ifdef DEBUG
        {
            OBin *heuristic = (OBin *)(split.ptr);
            for (u32 i = outLeft.start; i < outLeft.End(); i++)
            {
                const PrimRef *ref = &primRefs[i];
                Lane8F32 v         = Lane8F32::LoadU(ref);
                Lane8F32 centroid  = ((Shuffle4<1, 1>(v) - Shuffle4<0, 0>(v))) ^ signFlipMask;
                u32 pos = heuristic->binner->Bin(centroid[4 + split.bestDim], split.bestDim);
                Assert(pos < split.bestPos);
                u32 gMask = Movemask(v <= outLeft.geomBounds) & 0x77;
                Assert(gMask == 0x77);
                u32 cMask = Movemask(centroid <= outLeft.centBounds) & 0x77;
                Assert(cMask == 0x77);
            }
            for (u32 i = outRight.start; i < outRight.End(); i++)
            {
                const PrimRef *ref = &primRefs[i];
                Lane8F32 v         = Lane8F32::LoadU(ref);
                Lane8F32 centroid  = ((Shuffle4<1, 1>(v) - Shuffle4<0, 0>(v))) ^ signFlipMask;
                u32 pos = heuristic->binner->Bin(centroid[4 + split.bestDim], split.bestDim);
                Assert(pos >= split.bestPos);
                u32 gMask = Movemask(v <= outRight.geomBounds) & 0x77;
                u32 cMask = Movemask(centroid <= outRight.centBounds) & 0x77;
                Assert(gMask == 0x77);
                Assert(cMask == 0x77);
            }
            break;
        }

#endif
        ArenaPopTo(temp.arena, split.allocPos);
    }
};

template <typename Binner, i32 numBins>
static Split BinBest(const Bounds8 bounds[3][numBins], const Lane4U32 *entryCounts,
                     const Lane4U32 *exitCounts, const Binner *binner,
                     const u32 blockShift = 0)
{

    Lane4F32 areas[numBins];
    Lane4U32 counts[numBins];
    Lane4U32 currentCount = 0;

    Bounds8 boundsX;
    Bounds8 boundsY;
    Bounds8 boundsZ;
    for (u32 i = 0; i < numBins - 1; i++)
    {
        currentCount += entryCounts[i];
        counts[i] = currentCount;

        boundsX.Extend(bounds[0][i]);
        boundsY.Extend(bounds[1][i]);
        boundsZ.Extend(bounds[2][i]);

        Lane4F32 minX, minY, minZ;
        Lane4F32 maxX, maxY, maxZ;
        Transpose3x3(Extract4<0>(boundsX.v), Extract4<0>(boundsY.v), Extract4<0>(boundsZ.v),
                     minX, minY, minZ);
        Transpose3x3(Extract4<1>(boundsX.v), Extract4<1>(boundsY.v), Extract4<1>(boundsZ.v),
                     maxX, maxY, maxZ);

        Lane4F32 extentX = maxX + minX;
        Lane4F32 extentY = maxY + minY;
        Lane4F32 extentZ = maxZ + minZ;

        areas[i] = FMA(extentX, extentY + extentZ, extentY * extentZ);
    }
    boundsX           = Bounds8();
    boundsY           = Bounds8();
    boundsZ           = Bounds8();
    currentCount      = 0;
    Lane4U32 blockAdd = (1 << blockShift) - 1;
    Lane4F32 bestSAH(pos_inf);
    Lane4U32 bestPos(0);
    for (u32 i = numBins - 1; i >= 1; i--)
    {
        currentCount += exitCounts[i];

        boundsX.Extend(bounds[0][i]);
        boundsY.Extend(bounds[1][i]);
        boundsZ.Extend(bounds[2][i]);

        Lane4F32 minX, minY, minZ;
        Lane4F32 maxX, maxY, maxZ;
        Transpose3x3(Extract4<0>(boundsX.v), Extract4<0>(boundsY.v), Extract4<0>(boundsZ.v),
                     minX, minY, minZ);
        Transpose3x3(Extract4<1>(boundsX.v), Extract4<1>(boundsY.v), Extract4<1>(boundsZ.v),
                     maxX, maxY, maxZ);

        Lane4F32 extentX = maxX + minX;
        Lane4F32 extentY = maxY + minY;
        Lane4F32 extentZ = maxZ + minZ;

        Lane4F32 rArea = FMA(extentX, extentY + extentZ, extentY * extentZ);

        Lane4U32 rCount = (currentCount + blockAdd) >> blockShift;
        Lane4U32 lCount = (counts[i - 1] + blockAdd) >> blockShift;

        Lane4F32 sah = FMA(rArea, Lane4F32(rCount), areas[i - 1] * Lane4F32(lCount));

        bestPos = Select(sah < bestSAH, Lane4U32(i), bestPos);
        bestSAH = Select(sah < bestSAH, sah, bestSAH);
    }

    u32 bestDim       = 0;
    f32 bestSAHScalar = pos_inf;
    u32 bestPosScalar = 0;
    for (u32 dim = 0; dim < 3; dim++)
    {
        if (binner->scale[dim][0] == 0) continue;
        if (bestSAH[dim] < bestSAHScalar)
        {
            bestPosScalar = bestPos[dim];
            bestDim       = dim;
            bestSAHScalar = bestSAH[dim];
        }
    }
    f32 bestValue = binner->GetSplitValue(bestPosScalar, bestDim);
    return Split(bestSAHScalar, bestPosScalar, bestDim, bestValue);
}

template <typename Binner, i32 numBins>
static Split BinBest(const Bounds8 bounds[3][numBins], const Lane4U32 *counts,
                     const Binner *binner, const u32 blockShift = 0)
{
    return BinBest(bounds, counts, counts, binner, blockShift);
}

} // namespace rt
#endif
