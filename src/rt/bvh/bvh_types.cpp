#include "bvh_types.h"
namespace rt
{
template <i32 N>
void Triangle<N>::GetData(Scene2Tri *scene, Lane4F32 v0[N], Lane4F32 v1[N], Lane4F32 v2[N],
                          u32 outGeomIDs[N], u32 outPrimIDs[N]) const
{
    for (u32 i = 0; i < N; i++)
    {
        Assert(geomIDs[i] < scene->numPrimitives);
        TriangleMesh *mesh = scene->primitives[geomIDs[i]];
        u32 indices[3];
        if (mesh->indices)
        {
            Assert(3 * primIDs[i] < mesh->numIndices);
            indices[0] = mesh->indices[3 * primIDs[i] + 0];
            indices[1] = mesh->indices[3 * primIDs[i] + 1];
            indices[2] = mesh->indices[3 * primIDs[i] + 2];
        }
        else
        {
            Assert(3 * primIDs[i] < mesh->numVertices);
            indices[0] = 3 * primIDs[i] + 0;
            indices[1] = 3 * primIDs[i] + 1;
            indices[2] = 3 * primIDs[i] + 2;
        }
        v0[i]         = Lane4F32::LoadU((f32 *)(mesh->p + indices[0]));
        v1[i]         = Lane4F32::LoadU((f32 *)(mesh->p + indices[1]));
        v2[i]         = Lane4F32::LoadU((f32 *)(mesh->p + indices[2]));
        outGeomIDs[i] = geomIDs[i];
        outPrimIDs[i] = primIDs[i];
    }
}
template <i32 N>
void TriangleCompressed<N>::GetData(Scene2Tri *scene, Lane4F32 v0[N], Lane4F32 v1[N],
                                    Lane4F32 v2[N], u32 outGeomIDs[N], u32 outPrimIDs[N]) const
{
    // TODO: reconsider this later
    TriangleMesh *mesh = scene->triangleMeshes[0];
    for (u32 i = 0; i < N; i++)
    {
        u32 indices[3];
        if (mesh->indices)
        {
            Assert(3 * this->primIDs[i] < mesh->numIndices);
            indices[0] = mesh->indices[3 * this->primIDs[i] + 0];
            indices[1] = mesh->indices[3 * this->primIDs[i] + 1];
            indices[2] = mesh->indices[3 * this->primIDs[i] + 2];
        }
        else
        {
            Assert(3 * this->primIDs[i] < mesh->numVertices);
            indices[0] = 3 * this->primIDs[i] + 0;
            indices[1] = 3 * this->primIDs[i] + 1;
            indices[2] = 3 * this->primIDs[i] + 2;
        }
        v0[i]         = Lane4F32::LoadU((f32 *)(mesh->p + indices[0]));
        v1[i]         = Lane4F32::LoadU((f32 *)(mesh->p + indices[1]));
        v2[i]         = Lane4F32::LoadU((f32 *)(mesh->p + indices[2]));
        outPrimIDs[i] = this->primIDs[i];
        outGeomIDs[i] = 0;
    }
}
template <i32 N>
void Quad<N>::GetData(Scene2Quad *scene, Lane4F32 v0[N], Lane4F32 v1[N], Lane4F32 v2[N],
                      Lane4F32 v3[N], u32 outGeomIDs[N], u32 outPrimIDs[N]) const
{
    for (u32 i = 0; i < N; i++)
    {
        Assert(this->geomIDs[i] < scene->numPrimitives);
        QuadMesh *mesh = scene->primitives[this->geomIDs[i]];
        u32 indices[4];
        if (mesh->indices)
        {
            Assert(4 * this->primIDs[i] < mesh->numIndices);
            indices[0] = mesh->indices[4 * this->primIDs[i] + 0];
            indices[1] = mesh->indices[4 * this->primIDs[i] + 1];
            indices[2] = mesh->indices[4 * this->primIDs[i] + 2];
            indices[3] = mesh->indices[4 * this->primIDs[i] + 3];
        }
        else
        {
            Assert(4 * this->primIDs[i] < mesh->numVertices);
            indices[0] = 4 * this->primIDs[i] + 0;
            indices[1] = 4 * this->primIDs[i] + 1;
            indices[2] = 4 * this->primIDs[i] + 2;
            indices[3] = 4 * this->primIDs[i] + 3;
        }
        v0[i]         = Lane4F32::LoadU((f32 *)(mesh->p + indices[0]));
        v1[i]         = Lane4F32::LoadU((f32 *)(mesh->p + indices[1]));
        v2[i]         = Lane4F32::LoadU((f32 *)(mesh->p + indices[2]));
        v3[i]         = Lane4F32::LoadU((f32 *)(mesh->p + indices[3]));
        outGeomIDs[i] = this->geomIDs[i];
        outPrimIDs[i] = this->primIDs[i];
    }
}

template <i32 N>
void QuadCompressed<N>::GetData(Scene2Quad *scene, Lane4F32 v0[N], Lane4F32 v1[N],
                                Lane4F32 v2[N], Lane4F32 v3[N], u32 outGeomIDs[N],
                                u32 outPrimIDs[N]) const
{
    // TODO: reconsider this later
    QuadMesh *mesh = scene->primitives;
    for (u32 i = 0; i < N; i++)
    {
        u32 indices[4];
        if (mesh->indices)
        {
            Assert(4 * this->primIDs[i] < mesh->numIndices);
            indices[0] = mesh->indices[4 * this->primIDs[i] + 0];
            indices[1] = mesh->indices[4 * this->primIDs[i] + 1];
            indices[2] = mesh->indices[4 * this->primIDs[i] + 2];
            indices[3] = mesh->indices[4 * this->primIDs[i] + 3];
        }
        else
        {
            Assert(4 * this->primIDs[i] < mesh->numVertices);
            indices[0] = 4 * this->primIDs[i] + 0;
            indices[1] = 4 * this->primIDs[i] + 1;
            indices[2] = 4 * this->primIDs[i] + 2;
            indices[3] = 4 * this->primIDs[i] + 3;
        }
        v0[i]         = Lane4F32::LoadU((f32 *)(mesh->p + indices[0]));
        v1[i]         = Lane4F32::LoadU((f32 *)(mesh->p + indices[1]));
        v2[i]         = Lane4F32::LoadU((f32 *)(mesh->p + indices[2]));
        v3[i]         = Lane4F32::LoadU((f32 *)(mesh->p + indices[3]));
        outPrimIDs[i] = this->primIDs[i];
        outGeomIDs[i] = 0;
    }
}
} // namespace rt
