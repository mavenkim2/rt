#ifndef MESH_H_
#define MESH_H_

#include "base.h"
#include "handles.h"
#include "bvh/bvh_types.h"
#include "thread_context.h"
#include "math/math_include.h"

namespace rt
{
struct Arena;

// Megastruct for geometry
struct Mesh
{
    Vec3f *p;
    Vec3f *n;
    Vec2f *uv;
    u32 *indices;
    u32 *faceIDs;
    u32 numIndices;
    u32 numVertices;
    u32 numFaces;

    // PDF for area sampling
    // PiecewiseConstant1D areaPDF;
    u32 GetNumFaces() const { return numFaces; }

    void GetFaceIndices(int f, int outIndices[3])
    {
        if (indices)
        {
            outIndices[0] = indices[3 * f + 0];
            outIndices[1] = indices[3 * f + 1];
            outIndices[2] = indices[3 * f + 2];
        }
        else
        {
            outIndices[0] = 3 * f + 0;
            outIndices[1] = 3 * f + 1;
            outIndices[2] = 3 * f + 2;
        }
    }

#if 0
    void CreateTriangleAreaPDF(Arena *arena)
    {
        Assert(p && indices);
        f32 *areas = PushArrayNoZero(arena, f32, numFaces);
        for (int f = 0; f < numFaces; f++)
        {
            int triIndices[3];
            GetFaceIndices(f, triIndices);
            f32 area = 0.5f * Length(Cross(p[triIndices[1]] - p[triIndices[0]],
                                           p[triIndices[2]] - p[triIndices[0]]));
            areas[f] = area;
        }
        areaPDF = PiecewiseConstant1D(arena, areas, numFaces, 0.f, 1.f);
    }

    Vec3f SamplePosition(Vec2f u)
    {
        f32 pdf;
        u32 faceIndex = 0;
        f32 result    = areaPDF.Sample(u[0], &pdf, &faceIndex);

        int triIndices[3];
        GetFaceIndices(faceIndex, triIndices);

        Vec3f bary        = SampleUniformTriangle(u[1]);
        Vec3f samplePoint = bary[0] * p[triIndices[0]] + bary[1] * p[triIndices[1]] +
                            bary[2] * p[triIndices[2]];
        return samplePoint;
    }
#endif

    __forceinline const Vec3f &GetIndexedVertex(u32 index) const
    {
        if (indices)
        {
            Assert(index < numIndices);
            return p[indices[index]];
        }
        Assert(index < numVertices);
        return p[index];
    }
};

template <GeometryType type, typename PrimRefType = PrimRef>
struct GenerateMeshRefsHelper
{
    Vec3f *p;
    u32 *indices;

    __forceinline void operator()(PrimRefType *refs, u32 offset, u32 geomID, u32 start,
                                  u32 count, RecordAOSSplits &record)
    {
        static_assert(type == GeometryType::TriangleMesh || type == GeometryType::QuadMesh);
        constexpr u32 numVerticesPerFace = type == GeometryType::QuadMesh ? 4 : 3;
        Bounds geomBounds;
        Bounds centBounds;
        for (u32 i = start; i < start + count; i++, offset++)
        {
            PrimRefType *prim = &refs[offset];
            Vec3f v[numVerticesPerFace];
            if (indices)
            {
                for (u32 j = 0; j < numVerticesPerFace; j++)
                    v[j] = p[indices[numVerticesPerFace * i + j]];
            }
            else
            {
                for (u32 j = 0; j < numVerticesPerFace; j++)
                    v[j] = p[numVerticesPerFace * i + j];
            }

            Vec3f min(pos_inf);
            Vec3f max(neg_inf);
            for (u32 j = 0; j < numVerticesPerFace; j++)
            {
                min = Min(min, v[j]);
                max = Max(max, v[j]);
            }

            Assert(min != max);

            Lane4F32 mins = Lane4F32(min.x, min.y, min.z, 0);
            Lane4F32 maxs = Lane4F32(max.x, max.y, max.z, 0);
            Lane4F32::StoreU(prim->min, -mins);

            if constexpr (!std::is_same_v<PrimRefType, PrimRefCompressed>)
                prim->geomID = geomID;
            prim->maxX   = max.x;
            prim->maxY   = max.y;
            prim->maxZ   = max.z;
            prim->primID = i;

            geomBounds.Extend(mins, maxs);
            centBounds.Extend(maxs + mins);
        }
        record.geomBounds = Lane8F32(-geomBounds.minP, geomBounds.maxP);
        record.centBounds = Lane8F32(-centBounds.minP, centBounds.maxP);
    }
    // Computes bounds only
    __forceinline Bounds operator()(u32 start, u32 count)
    {
        static_assert(type == GeometryType::TriangleMesh || type == GeometryType::QuadMesh);
        constexpr u32 numVerticesPerFace = type == GeometryType::QuadMesh ? 4 : 3;
        Bounds geomBounds;
        for (u32 i = start; i < start + count; i++)
        {
            Vec3f v[numVerticesPerFace];
            if (indices)
            {
                for (u32 j = 0; j < numVerticesPerFace; j++)
                    v[j] = p[indices[numVerticesPerFace * i + j]];
            }
            else
            {
                for (u32 j = 0; j < numVerticesPerFace; j++)
                    v[j] = p[numVerticesPerFace * i + j];
            }

            Vec3f min(pos_inf);
            Vec3f max(neg_inf);
            for (u32 j = 0; j < numVerticesPerFace; j++)
            {
                min = Min(min, v[j]);
                max = Max(max, v[j]);
            }

            Assert(min != max);

            Lane4F32 mins = Lane4F32(min.x, min.y, min.z, 0);
            Lane4F32 maxs = Lane4F32(max.x, max.y, max.z, 0);

            geomBounds.Extend(mins, maxs);
        }
        return geomBounds;
    }
};

template <GeometryType type>
PrimRef *ParallelGenerateMeshRefs(Arena *arena, Mesh *meshes, u32 numPrimitives,
                                  RecordAOSSplits &record, bool spatialSplits)
{
    u32 totalNumFaces = 0;
    u32 extEnd        = 0;
    PrimRef *refs     = 0;
    if (numPrimitives > PARALLEL_THRESHOLD)
    {
        TempArena temp = ScratchStart(&arena, 1);

        ParallelForOutput output =
            ParallelFor<u32>(temp, 0, numPrimitives, PARALLEL_THRESHOLD,
                             [&](u32 &faceCount, u32 jobID, u32 start, u32 count) {
                                 u32 outCount = 0;

                                 for (u32 i = start; i < start + count; i++)
                                 {
                                     Mesh &mesh = meshes[i];
                                     outCount += mesh.GetNumFaces();
                                 }
                                 faceCount = outCount;
                             });
        Reduce(totalNumFaces, output, [&](u32 &l, const u32 &r) { l += r; });

        u32 offset   = 0;
        u32 *offsets = (u32 *)output.out;
        for (u32 i = 0; i < output.num; i++)
        {
            u32 numFaces = offsets[i];
            offsets[i]   = offset;
            offset += numFaces;
        }
        Assert(totalNumFaces == offset);
        u32 extEnd = u32(totalNumFaces * (spatialSplits ? GROW_AMOUNT : 1));

        // Generate PrimRefs
        refs = PushArrayNoZero(arena, PrimRef, extEnd);

        ParallelReduce<RecordAOSSplits>(
            &record, 0, numPrimitives, PARALLEL_THRESHOLD,
            [&](RecordAOSSplits &record, u32 jobID, u32 start, u32 count) {
                GenerateMeshRefs<type>(meshes, refs, offsets[jobID],
                                       jobID == output.num - 1 ? totalNumFaces
                                                               : offsets[jobID + 1],
                                       start, count, record);
            },
            [&](RecordAOSSplits &l, const RecordAOSSplits &r) { l.Merge(r); });

        ScratchEnd(temp);
    }
    else
    {
        for (u32 i = 0; i < numPrimitives; i++)
        {
            Mesh &mesh = meshes[i];
            totalNumFaces += mesh.GetNumFaces();
        }
        extEnd = u32(totalNumFaces * (spatialSplits ? GROW_AMOUNT : 1));
        refs   = PushArrayNoZero(arena, PrimRef, extEnd);
        GenerateMeshRefs<type>(meshes, refs, 0, totalNumFaces, 0, numPrimitives, record);
    }
    record.SetRange(0, totalNumFaces, extEnd);
    return refs;
}

template <GeometryType type, typename PrimRef>
void GenerateMeshRefs(Mesh *meshes, PrimRef *refs, u32 offset, u32 offsetMax, u32 start,
                      u32 count, RecordAOSSplits &record)
{
    RecordAOSSplits r(neg_inf);
    for (u32 i = start; i < start + count; i++)
    {
        Mesh *mesh = &meshes[i];

        u32 numFaces = mesh->GetNumFaces();
        RecordAOSSplits tempRecord(neg_inf);
        if (numFaces > PARALLEL_THRESHOLD)
        {
            ParallelReduce<RecordAOSSplits>(
                &tempRecord, 0, numFaces, PARALLEL_THRESHOLD,
                [&](RecordAOSSplits &record, u32 jobID, u32 start, u32 count) {
                    Assert(offset + start < offsetMax);
                    GenerateMeshRefsHelper<type, PrimRef>{mesh->p, mesh->indices}(
                        refs, offset + start, i, start, count, record);
                },
                [&](RecordAOSSplits &l, const RecordAOSSplits &r) { l.Merge(r); });
        }
        else
        {
            Assert(offset < offsetMax);
            GenerateMeshRefsHelper<type, PrimRef>{mesh->p, mesh->indices}(
                refs, offset, i, 0, numFaces, tempRecord);
        }
        r.Merge(tempRecord);
        offset += numFaces;
    }
    Assert(offsetMax == offset);
    record = r;
}

} // namespace rt
#endif
