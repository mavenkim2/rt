#include "bvh_types.h"
namespace rt
{
template <i32 N>
__forceinline void LeafPrim<N>::Fill(const ScenePrimitives *, PrimRef *refs, u32 &begin,
                                     u32 end)
{
    Assert(end > begin);
    for (u32 i = 0; i < N; i++)
    {
        if (begin < end)
        {
            PrimRef *ref = &refs[begin];
            geomIDs[i]   = ref->geomID;
            primIDs[i]   = ref->primID;
            begin++;
        }
        else
        {
            PrimRef *ref = &refs[begin - 1];
            geomIDs[i]   = ref->geomID;
            primIDs[i]   = ref->primID;
        }
    }
}

template <i32 N>
__forceinline void LeafPrimCompressed<N>::Fill(const ScenePrimitives *,
                                               PrimRefCompressed *refs, u32 &begin, u32 end)
{
    Assert(end > begin);
    for (u32 i = 0; i < N; i++)
    {
        if (begin < end)
        {
            PrimRefCompressed *ref = &refs[begin];
            primIDs[i]             = ref->primID;
            begin++;
        }
        else
        {
            PrimRefCompressed *ref = &refs[begin - 1];
            primIDs[i]             = ref->primID;
        }
    }
}

template <i32 N>
void Triangle<N>::GetData(const ScenePrimitives *scene, Lane4F32 v0[N], Lane4F32 v1[N],
                          Lane4F32 v2[N], u32 outGeomIDs[N], u32 outPrimIDs[N]) const
{
    Mesh *meshes = (Mesh *)scene->primitives;
    for (u32 i = 0; i < N; i++)
    {
        Assert(this->geomIDs[i] < scene->numPrimitives);
        Mesh *mesh = meshes + this->geomIDs[i];
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
        v0[i]         = Lane4F32::LoadU(mesh->p + indices[0]);
        v1[i]         = Lane4F32::LoadU(mesh->p + indices[1]);
        v2[i]         = Lane4F32::LoadU(mesh->p + indices[2]);
        outGeomIDs[i] = this->geomIDs[i];
        outPrimIDs[i] = this->primIDs[i];
    }
}
template <i32 N>
void TriangleCompressed<N>::GetData(const ScenePrimitives *scene, Lane4F32 v0[N],
                                    Lane4F32 v1[N], Lane4F32 v2[N], u32 outGeomIDs[N],
                                    u32 outPrimIDs[N]) const
{
    // TODO: reconsider this later
    Mesh *mesh = (Mesh *)scene->primitives;
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
        v0[i]         = Lane4F32::LoadU(mesh->p + indices[0]);
        v1[i]         = Lane4F32::LoadU(mesh->p + indices[1]);
        v2[i]         = Lane4F32::LoadU(mesh->p + indices[2]);
        outPrimIDs[i] = this->primIDs[i];
        outGeomIDs[i] = 0;
    }
}
template <i32 N>
void Quad<N>::GetData(const ScenePrimitives *scene, Lane4F32 v0[N], Lane4F32 v1[N],
                      Lane4F32 v2[N], Lane4F32 v3[N], u32 outGeomIDs[N],
                      u32 outPrimIDs[N]) const
{
    Mesh *meshes = (Mesh *)scene->primitives;
    for (u32 i = 0; i < N; i++)
    {
        Assert(this->geomIDs[i] < scene->numPrimitives);
        Mesh *mesh = meshes + this->geomIDs[i];
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
        v0[i]         = Lane4F32::LoadU(mesh->p + indices[0]);
        v1[i]         = Lane4F32::LoadU(mesh->p + indices[1]);
        v2[i]         = Lane4F32::LoadU(mesh->p + indices[2]);
        v3[i]         = Lane4F32::LoadU(mesh->p + indices[3]);
        outGeomIDs[i] = this->geomIDs[i];
        outPrimIDs[i] = this->primIDs[i];
    }
}

template <i32 N>
void QuadCompressed<N>::GetData(const ScenePrimitives *scene, Lane4F32 v0[N], Lane4F32 v1[N],
                                Lane4F32 v2[N], Lane4F32 v3[N], u32 outGeomIDs[N],
                                u32 outPrimIDs[N]) const
{
    Mesh *mesh = (Mesh *)scene->primitives;

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
        v0[i]         = Lane4F32::LoadU(mesh->p + indices[0]);
        v1[i]         = Lane4F32::LoadU(mesh->p + indices[1]);
        v2[i]         = Lane4F32::LoadU(mesh->p + indices[2]);
        v3[i]         = Lane4F32::LoadU(mesh->p + indices[3]);
        outPrimIDs[i] = this->primIDs[i];
        outGeomIDs[i] = 0;
    }
}

void TLASLeaf::GetData(const ScenePrimitives *scene, AffineSpace *&t,
                       ScenePrimitives *&childScene)
{
    t          = &scene->affineTransforms[transformIndex];
    childScene = scene->childScenes[sceneIndex];
}

} // namespace rt
