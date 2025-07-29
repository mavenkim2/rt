#include "../scene.h"
#include "scene.h"
#include "../bvh/bvh_build.h"
#include "../bvh/bvh_intersect1.h"
#include "../math/math_include.h"
#include "../memory.h"
#include "../parallel.h"
#include "../sampling.h"
#include "../thread_context.h"

#ifndef USE_GPU
#include "../subdivision.h"
#endif

namespace rt
{

ScenePrimitives **scenes_;

PrimitiveIndices::PrimitiveIndices(LightHandle lightID, MaterialHandle materialID)
    : lightID(lightID), materialID(materialID), alphaTexture(0)
{
}

// TODO: ??
PrimitiveIndices::PrimitiveIndices(LightHandle lightID, MaterialHandle materialID,
                                   Texture *alpha)
    : lightID(lightID), materialID(materialID), alphaTexture(alpha)
{
}

void ScenePrimitives::GenerateBuildRefs(BRef *refs, u32 start, u32 count,
                                        RecordAOSSplits &record)
{
    Bounds geom;
    Bounds cent;
    const Instance *instances = (const Instance *)primitives;
    for (u32 i = start; i < start + count; i++)
    {
        const Instance &instance = instances[i];
        AffineSpace &transform   = affineTransforms[instance.transformIndex];
        u32 index                = instance.id;
        Assert(childScenes);
        ScenePrimitives *inScene = childScenes[index];
        BRef *ref                = &refs[i];

        Bounds bounds = Transform(transform, GetBounds());
        Assert((Movemask(bounds.maxP >= bounds.minP) & 0x7) == 0x7);

        ref->StoreBounds(bounds);
        geom.Extend(bounds);
        cent.Extend(bounds.minP + bounds.maxP);

        ref->instanceID = i;
#ifdef USE_QUANTIZE_COMPRESS
        ref->nodePtr = uintptr_t(nodePtr.GetPtr());
        ref->type    = nodePtr.GetType();
#else
        ref->nodePtr = nodePtr;
#endif
        ref->numPrims = numFaces;

        ErrorExit(ref->nodePtr.data, "Invalid scene: %u\n", index);
    }
    record.geomBounds = Lane8F32(-geom.minP, geom.maxP);
    record.centBounds = Lane8F32(-cent.minP, cent.maxP);
}

// NOTE: either Scene or Scene2Inst
BRef *ScenePrimitives::GenerateBuildRefs(Arena *arena, RecordAOSSplits &record)
{
    u32 numInstances = numPrimitives;
    u32 extEnd       = 4 * numInstances;
    BRef *b          = PushArrayNoZero(arena, BRef, extEnd);

    if (numInstances > PARALLEL_THRESHOLD)
    {
        ParallelReduce<RecordAOSSplits>(
            &record, 0, numInstances, PARALLEL_THRESHOLD,
            [&](RecordAOSSplits &record, u32 jobID, u32 start, u32 count) {
                GenerateBuildRefs(b, start, count, record);
            },
            [&](RecordAOSSplits &l, const RecordAOSSplits &r) { l.Merge(r); });
    }
    else
    {
        GenerateBuildRefs(b, 0, numInstances, record);
    }
    record.SetRange(0, numInstances, extEnd);
    return b;
}

#ifndef USE_GPU
template <>
void ScenePrimitives::BuildBVH<GeometryType::CatmullClark>(Arena **arenas)
{
    TempArena temp = ScratchStart(0, 0);
    BuildSettings settings;
    OpenSubdivMesh *meshes = (OpenSubdivMesh *)primitives;

    int untessellatedPatchCount = 0;
    int tessellatedPatchCount   = 0;

    for (u32 i = 0; i < numPrimitives; i++)
    {
        untessellatedPatchCount += (int)meshes[i].untessellatedPatches.Length();
        tessellatedPatchCount += (int)meshes[i].patches.Length();
    }

    int size = untessellatedPatchCount + tessellatedPatchCount;

    struct alignas(CACHE_LINE_SIZE) ThreadBounds
    {
        Bounds geomBounds;
        Bounds centBounds;

        void Merge(ThreadBounds &r)
        {
            geomBounds.Extend(r.geomBounds);
            centBounds.Extend(r.centBounds);
        }
    };

    struct alignas(CACHE_LINE_SIZE) ThreadData
    {
        ChunkedLinkedList<PrimRef> refs;
        ThreadBounds bounds;
        int refOffset;

        void Merge(ThreadData &r)
        {
            bounds.Merge(r.bounds);
            refs.Merge(&r.refs);
        }
    };

    Arena **tempArenas = GetArenaArray(temp.arena);

    ParallelForOutput output = ParallelFor<ThreadData>(
        temp, 0, numPrimitives, 1, [&](ThreadData &data, int jobID, int start, int count) {
            int threadIndex  = GetThreadIndex();
            Arena *arena     = arenas[threadIndex];
            Arena *tempArena = tempArenas[threadIndex];
            tempArena->align = 32;

            ThreadBounds threadBounds;
            auto &threadRefs = data.refs;
            threadRefs       = ChunkedLinkedList<PrimRef>(tempArena, 1024);

            for (int i = start; i < start + count; i++)
            {
                auto *mesh = &meshes[i];

                std::vector<BVHPatch> bvhPatches;
                std::vector<BVHEdge> bvhEdges;
                bvhPatches.reserve((int)mesh->patches.Length());
                bvhEdges.reserve((int)mesh->patches.Length() * 4);

                const auto &indices  = mesh->stitchingIndices;
                const auto &vertices = mesh->vertices;

                if (mesh->untessellatedPatches.Length())
                {
                    auto *threadRefsNode =
                        threadRefs.AddNode(mesh->untessellatedPatches.Length());

                    ThreadBounds untessellatedBounds;
                    ParallelReduce(
                        &untessellatedBounds, 0, mesh->untessellatedPatches.Length(), 512,
                        [&](ThreadBounds &bounds, int jobID, int start, int count) {
                            ThreadBounds threadBounds;
                            for (int j = start; j < start + count; j++)
                            {
                                int indexStart = 4 * j;
                                Vec3f p0       = vertices[indices[indexStart + 0]];
                                Vec3f p1       = vertices[indices[indexStart + 1]];
                                Vec3f p2       = vertices[indices[indexStart + 2]];
                                Vec3f p3       = vertices[indices[indexStart + 3]];

                                Vec3f minP = Min(Min(p0, p1), Min(p2, p3));
                                Vec3f maxP = Max(Max(p0, p1), Max(p2, p3));

                                threadBounds.geomBounds.Extend(Lane4F32(minP), Lane4F32(maxP));
                                threadBounds.centBounds.Extend(Lane4F32(minP + maxP));

                                Assert(minP != maxP);

                                threadRefsNode->values[j] =
                                    PrimRef(-minP.x, -minP.y, -minP.z,
                                            CreatePatchID(CatClarkTriangleType::Untess, 0, i),
                                            maxP.x, maxP.y, maxP.z, j);
                            }
                            bounds = threadBounds;
                        },
                        [&](ThreadBounds &l, ThreadBounds &r) { l.Merge(r); });

                    threadBounds.Merge(untessellatedBounds);
                }

                for (u32 j = 0; j < mesh->patches.Length(); j++)
                {
                    OpenSubdivPatch *patch = &mesh->patches[j];

                    // Individually split each edge into smaller triangles
                    for (int edgeIndex = 0; edgeIndex < 4; edgeIndex++)
                    {
                        // EdgeInfo &currentEdge = patch->edgeInfo[edgeIndex];
                        EdgeInfo currentEdge = patch->edgeInfos.GetEdgeInfo(edgeIndex);

                        auto itr = patch->CreateIterator(edgeIndex);

                        while (itr.IsNotFinished())
                        {
                            Vec3f minP(pos_inf);
                            Vec3f maxP(neg_inf);

                            // save start state
                            BVHEdge bvhEdge;
                            bvhEdge.patchIndex = j;
                            bvhEdge.steps      = itr.steps;

                            for (int triIndex = 0; triIndex < 8 && itr.Next(); triIndex++)
                            {
                                minP = Min(
                                    Min(minP, vertices[itr.indices[0]]),
                                    Min(vertices[itr.indices[1]], vertices[itr.indices[2]]));
                                maxP = Max(
                                    Max(maxP, vertices[itr.indices[0]]),
                                    Max(vertices[itr.indices[1]], vertices[itr.indices[2]]));
                            }

                            threadBounds.geomBounds.Extend(Lane4F32(minP), Lane4F32(maxP));
                            threadBounds.centBounds.Extend(Lane4F32(minP + maxP));

                            int bvhEdgeIndex = (int)bvhEdges.size();
                            bvhEdges.push_back(bvhEdge);

                            Assert(minP != maxP);

                            threadRefs.AddBack(
                                PrimRef(-minP.x, -minP.y, -minP.z,
                                        CreatePatchID(CatClarkTriangleType::TessStitching,
                                                      edgeIndex, i),
                                        maxP.x, maxP.y, maxP.z, bvhEdgeIndex));
                        }
                    }

                    // Split internal grid into smaller grids
                    int edgeRateU = patch->GetMaxEdgeFactorU();
                    int edgeRateV = patch->GetMaxEdgeFactorV();

                    if (edgeRateU <= 2 || edgeRateV <= 2) continue;
                    int bvhPatchIndex = (int)bvhPatches.size();
                    int bvhPatchStart = bvhPatchIndex;
                    {
                        BVHPatch bvhPatch;
                        bvhPatch.patchIndex = j;
                        bvhPatch.grid       = UVGrid::Compress(
                            Vec2i(0, 0), Vec2i(Max(edgeRateU - 2, 0), Max(edgeRateV - 2, 0)));

                        bvhPatches.push_back(bvhPatch);
                    }

                    while (bvhPatchIndex < (int)bvhPatches.size())
                    {
                        const BVHPatch &bvhPatch = bvhPatches[bvhPatchIndex];
                        BVHPatch patch0, patch1;
                        if (bvhPatch.Split(patch0, patch1))
                        {
                            bvhPatches.push_back(patch1);
                            bvhPatches[bvhPatchIndex] = patch0;
                        }
                        else
                        {
                            bvhPatchIndex++;
                        }
                    }

                    for (int k = bvhPatchStart; k < (int)bvhPatches.size(); k++)
                    {
                        const BVHPatch &bvhPatch = bvhPatches[k];
                        Vec3f minP(pos_inf);
                        Vec3f maxP(neg_inf);

                        Vec2i uvStart, uvEnd;
                        bvhPatch.grid.Decompress(uvStart, uvEnd);
                        for (int v = uvStart[1]; v <= uvEnd[1]; v++)
                        {
                            for (int u = uvStart[0]; u <= uvEnd[0]; u++)
                            {
                                int index = patch->GetGridIndex(u, v);
                                minP      = Min(minP, vertices[index]);
                                maxP      = Max(maxP, vertices[index]);
                            }
                        }

                        threadBounds.geomBounds.Extend(Lane4F32(minP), Lane4F32(maxP));
                        threadBounds.centBounds.Extend(Lane4F32(minP + maxP));

                        Assert(minP != maxP);

                        threadRefs.AddBack(
                            PrimRef(-minP.x, -minP.y, -minP.z,
                                    CreatePatchID(CatClarkTriangleType::TessGrid, 0, i),
                                    maxP.x, maxP.y, maxP.z, k));
                    }
                }
                mesh->bvhPatches = StaticArray<BVHPatch>(arena, bvhPatches);
                mesh->bvhEdges   = StaticArray<BVHEdge>(arena, bvhEdges);

                threadMemoryStatistics[GetThreadIndex()].totalBVHMemory +=
                    sizeof(BVHPatch) * mesh->bvhPatches.Length();
                threadMemoryStatistics[GetThreadIndex()].totalBVHMemory +=
                    sizeof(BVHEdge) * mesh->bvhEdges.Length();
            }
            data.bounds = threadBounds;
        });

    int totalRefCount = 0;
    ThreadBounds threadBounds;
    ThreadData *threadData = (ThreadData *)output.out;
    for (int i = 0; i < output.num; i++)
    {
        threadData[i].refOffset = totalRefCount;
        totalRefCount += threadData[i].refs.totalCount;
        threadBounds.Merge(threadData[i].bounds);
    }

    PrimRef *refs = PushArrayNoZero(temp.arena, PrimRef, totalRefCount);

    // Join
    ParallelFor(0, output.num, 1, [&](int jobID, int start, int count) {
        for (int i = start; i < start + count; i++)
        {
            ThreadData &data = threadData[i];
            data.refs.Flatten(refs + data.refOffset);
        }
    });

    ReleaseArenaArray(tempArenas);

    RecordAOSSplits record;
    record.geomBounds = Lane8F32(-threadBounds.geomBounds.minP, threadBounds.geomBounds.maxP);
    record.centBounds = Lane8F32(-threadBounds.centBounds.minP, threadBounds.centBounds.maxP);
    record.SetRange(0, totalRefCount);

    nodePtr = BuildQuantizedCatmullClarkBVH(settings, arenas, scene, refs, record);
    using IntersectorType =
        typename IntersectorHelper<GeometryType::CatmullClark, PrimRef>::IntersectorType;
    intersectFunc = &IntersectorType::Intersect;
    occludedFunc  = &IntersectorType::Occluded;
    bvhPrimSize   = (int)sizeof(typename IntersectorType::Prim);
    Bounds b(record.geomBounds);
    Assert((Movemask(b.maxP >= b.minP) & 0x7) == 0x7);
    SetBounds(b);
    numFaces = size;
    ScratchEnd(temp);
}

void ScenePrimitives::BuildCatClarkBVH(Arena **arenas)
{
    BuildBVH<GeometryType::CatmullClark>(arenas);
}
#endif

template <GeometryType type>
void ScenePrimitives::BuildBVH(Arena **arenas)
{
    TempArena temp = ScratchStart(0, 0);
    BuildSettings settings;
    Mesh *meshes = (Mesh *)primitives;
    RecordAOSSplits record(neg_inf);

    if (numPrimitives > 1)
    {
        PrimRef *refs         = ParallelGenerateMeshRefs<type>(temp.arena, (Mesh *)primitives,
                                                               numPrimitives, record, true);
        nodePtr               = BuildQuantizedSBVH<type>(settings, arenas, this, refs, record);
        using IntersectorType = typename IntersectorHelper<type, PrimRef>::IntersectorType;
        intersectFunc         = &IntersectorType::Intersect;
        occludedFunc          = &IntersectorType::Occluded;
        bvhPrimSize           = (int)sizeof(typename IntersectorType::Prim);
    }
    else
    {
        u32 totalNumFaces       = meshes->GetNumFaces();
        u32 extEnd              = u32(totalNumFaces * GROW_AMOUNT);
        PrimRefCompressed *refs = PushArrayNoZero(temp.arena, PrimRefCompressed, extEnd);
        GenerateMeshRefs<type>(meshes, refs, 0, totalNumFaces, 0, 1, record);
        record.SetRange(0, totalNumFaces, extEnd);
        nodePtr = BuildQuantizedSBVH<type>(settings, arenas, this, refs, record);
        using IntersectorType =
            typename IntersectorHelper<type, PrimRefCompressed>::IntersectorType;
        intersectFunc = &IntersectorType::Intersect;
        occludedFunc  = &IntersectorType::Occluded;
        bvhPrimSize   = (int)sizeof(typename IntersectorType::Prim);
    }
    Bounds b(record.geomBounds);
    Assert((Movemask(b.maxP >= b.minP) & 0x7) == 0x7);
    SetBounds(b);
    numFaces = record.Count();
    ScratchEnd(temp);
}

void ScenePrimitives::BuildTriangleBVH(Arena **arenas)
{
    BuildBVH<GeometryType::TriangleMesh>(arenas);
}

void ScenePrimitives::BuildQuadBVH(Arena **arenas)
{
    BuildBVH<GeometryType::QuadMesh>(arenas);
}

void ScenePrimitives::BuildTLASBVH(Arena **arenas)
{
    ScratchArena scratch;
    BuildSettings settings;
    // build tlas
    RecordAOSSplits record(neg_inf);

    BRef *refs = GenerateBuildRefs(scratch.temp.arena, record);
    Bounds b   = Bounds(record.geomBounds);
    Assert((Movemask(b.maxP >= b.minP) & 0x7) == 0x7);

    // NOTE: record is being corrupted somehow during this routine.
    nodePtr = BuildTLASQuantized(settings, arenas, this, refs, record);
    using IntersectorType =
        typename IntersectorHelper<GeometryType::Instance, BRef>::IntersectorType;
    intersectFunc = &IntersectorType::Intersect;
    occludedFunc  = &IntersectorType::Occluded;
    bvhPrimSize   = (int)sizeof(typename IntersectorType::Prim);

    b = Bounds(record.geomBounds);
    SetBounds(b);
    Assert((Movemask(b.maxP >= b.minP) & 0x7) == 0x7);
}

void ScenePrimitives::BuildSceneBVHs(Arena **arenas, const Mat4 &NDCFromCamera,
                                     const Mat4 &cameraFromRender, int screenHeight)
{
    switch (geometryType)
    {
        case GeometryType::Instance:
        {
            BuildTLASBVH(arenas);
        }
        break;
        case GeometryType::QuadMesh:
        {
            BuildQuadBVH(arenas);
        }
        break;
        case GeometryType::TriangleMesh:
        {
            BuildTriangleBVH(arenas);
        }
        break;
#ifndef USE_GPU
        case GeometryType::CatmullClark:
        {
            primitives = AdaptiveTessellation(arenas, this, NDCFromCamera, cameraFromRender,
                                              screenHeight, tessellationParams,
                                              (Mesh *)primitves, numPrimitives);
            BuildCatClarkBVH(arenas, scene);
        }
#endif
        break;
        default: Assert(0);
    }
}

Bounds ScenePrimitives::GetTLASBounds(u32 start, u32 count)
{
    Bounds geom;
    const Instance *instances = (const Instance *)primitives;
    for (u32 i = start; i < start + count; i++)
    {
        const Instance &instance = instances[i];
        AffineSpace &transform   = affineTransforms[instance.transformIndex];
        u32 index                = instance.id;
        Assert(childScenes);
        ScenePrimitives *inScene = childScenes[index];

        Bounds bounds = Transform(transform, inScene->GetBounds());
        Assert((Movemask(bounds.maxP >= bounds.minP) & 0x7) == 0x7);

        geom.Extend(bounds);
    }
    return geom;
}

Bounds ScenePrimitives::GetSceneBounds()
{
    Bounds b;
    switch (geometryType)
    {
        case GeometryType::Instance:
        {
            u32 numInstances = numPrimitives;

            if (numInstances > PARALLEL_THRESHOLD)
            {
                ParallelReduce<Bounds>(
                    &b, 0, numInstances, PARALLEL_THRESHOLD,
                    [&](Bounds &b, u32 jobID, u32 start, u32 count) {
                        b = GetTLASBounds(start, count);
                    },
                    [&](Bounds &l, const Bounds &r) { l.Extend(r); });
            }
            else
            {
                b = GetTLASBounds(0, numInstances);
            }
        }
        break;
        case GeometryType::TriangleMesh:
        case GeometryType::QuadMesh:
#ifndef USE_GPU
        case GeometryType::CatmullClark:
        {
            Mesh *meshes = (Mesh *)primitives;
            for (u32 i = 0; i < numPrimitives; i++)
            {
                Bounds bounds;
                Mesh *mesh = &meshes[i];

                u32 meshNumFaces = mesh->GetNumFaces();
                if (meshNumFaces > PARALLEL_THRESHOLD)
                {
                    ParallelReduce<Bounds>(
                        &bounds, 0, numFaces, PARALLEL_THRESHOLD,
                        [&](Bounds &b, int jobID, int start, int count) {
                            switch (geometryType)
                            {
                                case GeometryType::TriangleMesh:
                                {
                                    b = GenerateMeshRefsHelper<GeometryType::TriangleMesh>{
                                        mesh->p, mesh->indices}(start, count);
                                }
                                break;
                                case GeometryType::CatmullClark:
                                {
                                    b = GenerateMeshRefsHelper<GeometryType::QuadMesh>{
                                        mesh->p, mesh->indices}(start, count);
                                }
                                break;
                                default: Assert(0);
                            }
                        },
                        [&](Bounds &l, const Bounds &r) { l.Extend(r); });
                }
                else
                {
                    switch (geometryType)
                    {
                        case GeometryType::TriangleMesh:
                        {
                            bounds = GenerateMeshRefsHelper<GeometryType::TriangleMesh>{
                                mesh->p, mesh->indices}(0, numFaces);
                        }
                        break;
                        case GeometryType::CatmullClark:
                        {
                            b = GenerateMeshRefsHelper<GeometryType::QuadMesh>{
                                mesh->p, mesh->indices}(0, numFaces);
                        }
                        break;
                        default: Assert(0);
                    }
                }
                b.Extend(bounds);
            }
        }
#endif
        break;
        default: Assert(0);
    }

    return b;
}

bool ScenePrimitives::Occluded(Ray2 &ray)
{
    Assert(occludedFunc);
    return occludedFunc(this, StackEntry(nodePtr, ray.tFar), ray);
}

bool ScenePrimitives::Intersect(Ray2 &ray, SurfaceInteraction &si)
{
    Assert(intersectFunc);
    return intersectFunc(this, StackEntry(nodePtr, ray.tFar), ray, si);
}

// NOTE: this assumes the quad is planar
ShapeSample ScenePrimitives::SampleQuad(SurfaceInteraction &intr, Vec2f &u,
                                        AffineSpace *renderFromObject, int geomID)
{
    static const f32 MinSphericalSampleArea = 3e-4;
    static const f32 MaxSphericalSampleArea = 6.22;
    Mesh *mesh                              = ((Mesh *)primitives) + geomID;

    Vec3f p[4];

    // TODO: handle mesh lights
    Assert(mesh->GetNumFaces() == 1);
    int primID = 0;

    if (mesh->indices)
    {
        p[0] = mesh->p[mesh->indices[4 * primID + 0]];
        p[1] = mesh->p[mesh->indices[4 * primID + 1]];
        p[2] = mesh->p[mesh->indices[4 * primID + 2]];
        p[3] = mesh->p[mesh->indices[4 * primID + 3]];
    }
    else
    {
        p[0] = mesh->p[4 * primID + 0];
        p[1] = mesh->p[4 * primID + 1];
        p[2] = mesh->p[4 * primID + 2];
        p[3] = mesh->p[4 * primID + 3];
    }

    if (renderFromObject)
    {
        for (int i = 0; i < 4; i++)
        {
            p[i] = TransformP(*renderFromObject, p[i]);
        }
    }

    Vec3lfn v00 = Normalize(p[0] - Vec3f(intr.p));
    Vec3lfn v10 = Normalize(p[1] - Vec3f(intr.p));
    Vec3lfn v01 = Normalize(p[3] - Vec3f(intr.p));
    Vec3lfn v11 = Normalize(p[2] - Vec3f(intr.p));

    Vec3lfn eu = p[1] - p[0];
    Vec3lfn ev = p[3] - p[0];
    Vec3lfn n  = Normalize(Cross(eu, ev));

    ShapeSample result;
    // If the solid angle is small
    f32 area = SphericalQuadArea(v00, v10, v01, v11);
    // Vec3lfn wi   = intr.p - result.samplePoint;
    if (area < MinSphericalSampleArea || area > MaxSphericalSampleArea)
    {
        // First, sample a triangle based on area
        bool isSecondTri = false;
        Vec3f p01        = p[1] - p[0];
        Vec3f p02        = p[2] - p[0];
        Vec3f p03        = p[3] - p[0];
        f32 area0        = Length(Cross(p01, p02));
        f32 area1        = Length(Cross(p02, p03));

        f32 div  = 1.f / (area0 + area1);
        f32 prob = area0 * div;
        // Then sample the triangle by area
        if (u[0] < prob)
        {
            u[0]       = u[0] / prob;
            Vec3f bary = SampleUniformTriangle(u);
            result.p   = bary[0] * p[0] + bary[1] * p[1] + bary[2] * p[2];
        }
        else
        {
            u[0]       = (1 - u[0]) / (1 - prob);
            Vec3f bary = SampleUniformTriangle(u);
            result.p   = bary[0] * p[0] + bary[1] * p[2] + bary[2] * p[3];
        }
        result.n   = n;
        result.w   = Normalize(result.p - intr.p);
        result.pdf = div * LengthSquared(intr.p - result.p) / AbsDot(intr.n, result.w);
    }
    else
    {
        f32 pdf;
        result.p = SampleSphericalRectangle(intr.p, p[0], eu, ev, u, &pdf);
        result.n = n;
        result.w = Normalize(result.p - intr.p);

        // add projected solid angle measure (n dot wi) to pdf
        Vec4f w(AbsDot(v00, intr.shading.n), AbsDot(v10, intr.shading.n),
                AbsDot(v01, intr.shading.n), AbsDot(v11, intr.shading.n));
        Vec2f uNew = SampleBilinear(u, w);
        pdf *= BilinearPDF(uNew, w);
        result.pdf = pdf;
    }
    return result;
}

ShapeSample ScenePrimitives::Sample(SurfaceInteraction &intr, AffineSpace *space, Vec2f &u,
                                    int geomID)
{
    switch (geometryType)
    {
        case GeometryType::QuadMesh:
        {
            return SampleQuad(intr, u, space, geomID);
        }
        break;
        default: Assert(0); return {};
    }
}

} // namespace rt
