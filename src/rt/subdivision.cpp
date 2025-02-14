#include "math/matx.h"
#include "thread_context.h"
#define M_PI 3.1415926535897932
#include <opensubdiv/far/topologyDescriptor.h>
#include <opensubdiv/far/primvarRefiner.h>
#include <opensubdiv/far/patchTableFactory.h>
#include <opensubdiv/far/ptexIndices.h>
#include <opensubdiv/far/patchMap.h>

namespace rt
{

PatchItr OpenSubdivPatch::CreateIterator(int edge) const
{
    PatchItr itr = PatchItr(this, edge);
    return itr;
}

struct Vertex
{
    void Clear(void * = 0) { p = Vec3f(0.f); }
    void AddWithWeight(const Vertex &src, f32 weight) { p += weight * src.p; }
    Vec3f p;
};

using namespace OpenSubdiv;

u64 ComputeEdgeId(u32 id0, u32 id1)
{
    return id0 < id1 ? ((u64)id1 << 32) | id0 : ((u64)id0 << 32) | id1;
}

struct LimitSurfaceSample
{
    int faceID = -1;
    Vec2f uv;
    bool IsValid() const { return faceID != -1; }
};

void EvaluateLimitSurfacePosition(const Vertex *vertices, const Far::PatchMap *patchMap,
                                  const Far::PatchTable *patchTable,
                                  const LimitSurfaceSample &sample, Vec3f &outP,
                                  Vec3f &outDpdu, Vec3f &outDpdv)
{
    const Far::PatchTable::PatchHandle *handle =
        patchMap->FindPatch(sample.faceID, sample.uv[0], sample.uv[1]);

    Assert(handle);
    const auto &cvIndices = patchTable->GetPatchVertices(*handle);

    f32 pWeights[20];
    f32 duWeights[20];
    f32 dvWeights[20];
    patchTable->EvaluateBasis(*handle, sample.uv[0], sample.uv[1], pWeights, duWeights,
                              dvWeights);

    Vec3f pos  = {};
    Vec3f dpdu = {};
    Vec3f dpdv = {};

    for (int j = 0; j < cvIndices.size(); j++)
    {
        int index      = cvIndices[j];
        const Vec3f &p = vertices[cvIndices[j]].p;
        pos += pWeights[j] * p;
        dpdu += duWeights[j] * p;
        dpdv += dvWeights[j] * p;
    }
    outP    = pos;
    outDpdu = dpdu;
    outDpdv = dpdv;
}

void EvaluateLimitSurfacePosition(const Vertex *vertices, const Far::PatchMap *patchMap,
                                  const Far::PatchTable *patchTable,
                                  const LimitSurfaceSample &sample, Vec3f &outP,
                                  Vec3f &outDpdu, Vec3f &outDpdv, Vec3f &outDpduu,
                                  Vec3f &outDpduv, Vec3f &outDpdvv)
{
    const Far::PatchTable::PatchHandle *handle =
        patchMap->FindPatch(sample.faceID, sample.uv[0], sample.uv[1]);

    Assert(handle);
    const auto &cvIndices = patchTable->GetPatchVertices(*handle);

    f32 pWeights[20];
    f32 duWeights[20];
    f32 dvWeights[20];
    f32 duuWeights[20];
    f32 duvWeights[20];
    f32 dvvWeights[20];
    patchTable->EvaluateBasis(*handle, sample.uv[0], sample.uv[1], pWeights, duWeights,
                              dvWeights, duuWeights, duvWeights, dvvWeights);

    Vec3f pos   = {};
    Vec3f dpdu  = {};
    Vec3f dpdv  = {};
    Vec3f dpduu = {};
    Vec3f dpduv = {};
    Vec3f dpdvv = {};
    for (int j = 0; j < cvIndices.size(); j++)
    {
        const Vec3f &p = vertices[cvIndices[j]].p;
        pos += pWeights[j] * p;
        dpdu += duWeights[j] * p;
        dpdv += dvWeights[j] * p;

        dpduu += duuWeights[j] * p;
        dpduv += duvWeights[j] * p;
        dpdvv += dvvWeights[j] * p;
    }
    outP     = pos;
    outDpdu  = dpdu;
    outDpdv  = dpdv;
    outDpduu = dpduu;
    outDpduv = dpduv;
    outDpdvv = dpdvv;
}

struct FaceInfo
{
    int edgeInfoId[4];
    bool reversed[4];
};

struct TessellatedVertices
{
    std::vector<Vec3f> vertices;
    std::vector<Vec3f> normals;
    std::vector<Vec3f> dpdu;
    std::vector<Vec3f> dpdv;
    std::vector<Vec3f> dndu;
    std::vector<Vec3f> dndv;

    // Stitching triangles
    std::vector<int> stitchingIndices;

    void Resize(size_t size)
    {
        vertices.resize(size);
        normals.resize(size);
        dpdu.resize(size);
        dpdv.resize(size);
        dndu.resize(size);
        dndv.resize(size);
    }
    void Clear()
    {
        vertices.clear();
        vertices.shrink_to_fit();
        normals.clear();
        normals.shrink_to_fit();
        stitchingIndices.clear();
        stitchingIndices.shrink_to_fit();
        dpdu.clear();
        dpdu.shrink_to_fit();
        dpdv.clear();
        dpdv.shrink_to_fit();
        dndu.clear();
        dndu.shrink_to_fit();
        dndv.clear();
        dndv.shrink_to_fit();
    }
};

void CalculateWeingarten(const Vec3f &normal, const Vec3f &dpdu, const Vec3f &dpdv,
                         const Vec3f &dpduu, const Vec3f &dpduv, const Vec3f &dpdvv,
                         Vec3f &dndu, Vec3f &dndv)
{
    // Calculate dndu and dndv using weingarten equations
    f32 E = Dot(dpdu, dpdu);
    f32 F = Dot(dpdu, dpdv);
    f32 G = Dot(dpdv, dpdv);

    f32 e = Dot(normal, dpduu);
    f32 f = Dot(normal, dpduv);
    f32 g = Dot(normal, dpdvv);

    f32 denom = E * G - Sqr(F);
    denom     = denom == 0.f ? 0.f : 1.f / denom;

    dndu = denom * ((f * F - e * G) * dpdu + (e * F - f * E) * dpdv);
    dndv = denom * ((g * F - f * G) * dpdu + (f * F - g * E) * dpdv);
}

void EvaluateDisplacement(PtexTexture *texture, int faceID, const Vec2f &uv,
                          const Vec4f &filterWidths, Vec3f &pos, Vec3f &dpdu, Vec3f &dpdv,
                          const Vec3f &dndu, const Vec3f &dndv, Vec3f &normal)
{
    // Vector Displacement mapping
    Vec3f displacement, uDisplacement, vDisplacement;
    SurfaceInteraction intr;
    intr.uv          = uv;
    intr.faceIndices = faceID;
    texture->EvaluateHelper<3>(intr, filterWidths, displacement.e);

    f32 disp = (displacement[0] + displacement[1] + displacement[2]) / 3.f;

    f32 du  = filterWidths[0] + filterWidths[1];
    intr.uv = uv + Vec2f(du, 0.f);
    texture->EvaluateHelper<3>(intr, filterWidths, uDisplacement.e);
    f32 uDisp = (uDisplacement[0] + uDisplacement[1] + uDisplacement[2]) / 3.f;

    f32 dv  = filterWidths[2] + filterWidths[3];
    intr.uv = uv + Vec2f(0.f, dv);
    texture->EvaluateHelper<3>(intr, filterWidths, vDisplacement.e);
    f32 vDisp = (vDisplacement[0] + vDisplacement[1] + vDisplacement[2]) / 3.f;

    pos += disp * normal;
    dpdu += (uDisp - disp) / du * normal + dndu * disp;
    dpdv += (vDisp - disp) / dv * normal + dndv * disp;
    // normal = Normalize(Cross(dpdu, dpdv));
    normal = Cross(dpdu, dpdv);
}

// MORETON, H. 2001. Watertight tessellation using forward
// differencing. In HWWS ’01: Proceedings of the ACM SIG-
// GRAPH/EUROGRAPHICS workshop on Graphics hardware,
// ACM, New York, NY, USA, 25–32
// See Figure 7 from above for how this works
OpenSubdivMesh *AdaptiveTessellation(Arena **arenas, ScenePrimitives *scene,
                                     const Mat4 &NDCFromCamera, const Mat4 &cameraFromRender,
                                     int screenHeight, TessellationParams *params,
                                     Mesh *controlMeshes, u32 numMeshes)
{
    Scene *baseScene = GetScene();
    typedef Far::TopologyDescriptor Descriptor;
    Sdc::SchemeType type = OpenSubdiv::Sdc::SCHEME_CATMARK;
    Sdc::Options options;
    options.SetVtxBoundaryInterpolation(Sdc::Options::VTX_BOUNDARY_EDGE_AND_CORNER);

    u32 pixelsPerEdge        = 2;
    u32 edgesPerScreenHeight = screenHeight / pixelsPerEdge;

    Vec2f uvTable[] = {
        Vec2f(0.f, 0.f),
        Vec2f(1.f, 0.f),
        Vec2f(1.f, 1.f),
        Vec2f(0.f, 1.f),
    };
    Vec2i uvDiffTable[] = {
        Vec2i(1, 0),
        Vec2i(0, 1),
        Vec2i(-1, 0),
        Vec2i(0, -1),
    };

    OpenSubdivMesh *outputMeshes =
        PushArray(arenas[GetThreadIndex()], OpenSubdivMesh, numMeshes);

    ParallelFor(0, numMeshes, 1, 1, [&](u32 jobID, u32 start, u32 count) {
        for (u32 meshIndex = start; meshIndex < start + count; meshIndex++)
        {
            OpenSubdivMesh *outputMesh           = &outputMeshes[meshIndex];
            const TessellationParams *tessParams = params + meshIndex;

            TessellatedVertices tessellatedVertices;
            Mesh *controlMesh = &controlMeshes[meshIndex];
            tessellatedVertices.vertices.reserve(tessellatedVertices.vertices.capacity() +
                                                 controlMesh->numVertices);
            tessellatedVertices.normals.reserve(tessellatedVertices.normals.capacity() +
                                                controlMesh->numVertices);

            std::vector<FaceInfo> faceInfos(controlMesh->numFaces);
            std::vector<EdgeInfo> edgeInfos;
            edgeInfos.reserve(controlMesh->numIndices / 4);

            ScratchArena scratch;

            std::vector<LimitSurfaceSample> samples(controlMesh->numVertices);

            i32 *numVertsPerFace =
                PushArrayNoZero(scratch.temp.arena, i32, controlMesh->numFaces);
            for (u32 i = 0; i < controlMesh->numFaces; i++)
            {
                numVertsPerFace[i] = 4;
            }

            Descriptor desc;
            desc.numVertices        = controlMesh->numVertices;
            desc.numFaces           = controlMesh->numFaces;
            desc.numVertsPerFace    = numVertsPerFace;
            desc.vertIndicesPerFace = (int *)controlMesh->indices;

            int maxMeshEdgeRate = 1;

            // NOTE: this assumes everything is a quad
            // Compute edge rates
            std::unordered_map<u64, int> edgeIDMap;

            for (int f = 0; f < (int)controlMesh->numFaces; f++)
            {
                int indexOffset = 4 * f;

                // if (!tessParams->instanced)
                // {
                //     Bounds bounds;
                //     for (int i = 0; i < 4; i++)
                //     {
                //         int id = controlMesh->indices[indexOffset + i];
                //         bounds.Extend(
                //             Lane4F32(TransformP(tessParams->transform,
                //             controlMesh->p[id])));
                //     }
                //     // frustum cull
                //     if (FrustumCull(NDCFromCamera, bounds))
                //     {
                //         for (int i = 0; i < 4; i++)
                //         {
                //             int id0    = controlMesh->indices[indexOffset + i];
                //             int id1    = controlMesh->indices[indexOffset + ((i + 1) & 3)];
                //             u64 edgeId = ComputeEdgeId(id0, id1);
                //         }
                //         const auto &result = edgeIDMap.find(edgeId);
                //         int edgeIndex      = -1;
                //         if (result == edgeIDMap.end())
                //         {
                //         }
                //         continue;
                //     }
                // }
                // For uninstanced meshes,
                for (int i = 0; i < 4; i++)
                {
                    int id0       = controlMesh->indices[indexOffset + i];
                    int id1       = controlMesh->indices[indexOffset + ((i + 1) & 3)];
                    u64 edgeId    = ComputeEdgeId(id0, id1);
                    bool reversed = false;

                    // Add vertex samples for evaluation
                    Assert(id0 < (int)samples.size());
                    if (!samples[id0].IsValid())
                    {
                        samples[id0] = LimitSurfaceSample{f, uvTable[i]};
                    }
                    // Set edge rate
                    const auto &result = edgeIDMap.find(edgeId);
                    int edgeIndex      = -1;
                    if (result == edgeIDMap.end())
                    {
                        int edgeFactor = 1;
                        // Only set edge factor if there is a valid transform
                        if (tessParams->currentMinDistance != (f32)pos_inf)
                        {
                            // Compute tessellation factor
                            Vec3f p0 = controlMesh->p[id0];
                            Vec3f p1 = controlMesh->p[id1];

                            p0             = TransformP(tessParams->transform, p0);
                            p1             = TransformP(tessParams->transform, p1);
                            Vec3f midPoint = (p0 + p1) / 2.f;
                            f32 radius     = Length(p0 - p1) / 2.f;

                            f32 midPointW =
                                Transform(cameraFromRender, Vec4f(midPoint, 1.f)).z;
                            edgeFactor = int(edgesPerScreenHeight * radius *
                                             Abs(NDCFromCamera[1][1] / midPointW)) +
                                         1;

                            threadLocalStatistics[GetThreadIndex()].misc4 =
                                Max((u64)edgeFactor,
                                    threadLocalStatistics[GetThreadIndex()].misc4);

                            edgeFactor = Clamp(edgeFactor, 1, 64);

                            maxMeshEdgeRate = Max(maxMeshEdgeRate, edgeFactor);
                        }

                        edgeIndex = (int)edgeInfos.size();
                        // Assert(samples.size() - controlMesh->numVertices ==
                        // edgeInfos.size());
                        edgeInfos.emplace_back(
                            EdgeInfo{(int)samples.size(), (u32)edgeFactor, id0, id1});
                        edgeIDMap[edgeId] = edgeIndex;

                        // Add edge samples for evaluation
                        f32 stepSize = 1.f / edgeFactor;
                        for (int edgeVertexIndex = 1; edgeVertexIndex < edgeFactor;
                             edgeVertexIndex++)
                        {
                            Vec2f uv = uvTable[i] +
                                       Vec2f(uvDiffTable[i]) * (edgeVertexIndex * stepSize);
                            samples.push_back(LimitSurfaceSample{f, uv});
                        }
                    }
                    else
                    {
                        edgeIndex = edgeIDMap[edgeId];
                        reversed  = true;
                    }
                    Assert(edgeIndex != -1);
                    faceInfos[f].edgeInfoId[i] = edgeIndex;
                    faceInfos[f].reversed[i]   = reversed;
                }
            }

            // Create OpenSubdiv objects
            Far::TopologyRefiner *refiner = Far::TopologyRefinerFactory<Descriptor>::Create(
                desc, Far::TopologyRefinerFactory<Descriptor>::Options(type, options));

            int tessDepth = Clamp(Log2Int(maxMeshEdgeRate), 1, 10);
            Far::PatchTableFactory::Options patchOptions(tessDepth);
            patchOptions.SetEndCapType(Far::PatchTableFactory::Options::ENDCAP_GREGORY_BASIS);

            Far::TopologyRefiner::AdaptiveOptions adaptiveOptions =
                patchOptions.GetRefineAdaptiveOptions();
            refiner->RefineAdaptive(adaptiveOptions);

            auto *patchTable = Far::PatchTableFactory::Create(*refiner, patchOptions);
            OpenSubdiv::Far::PatchMap patchMap(*patchTable);

            // Feature adaptive refinment
            int size           = refiner->GetNumVerticesTotal();
            int numLocalPoints = patchTable->GetNumLocalPoints();
            Vertex *vertices =
                PushArrayNoZero(scratch.temp.arena, Vertex, size + numLocalPoints);
            MemoryCopy(vertices, controlMesh->p, sizeof(Vec3f) * controlMesh->numVertices);

            Vertex *src        = vertices;
            i32 nRefinedLevels = refiner->GetNumLevels();
            Far::PrimvarRefiner primvarRefiner(*refiner);

            for (i32 i = 1; i < nRefinedLevels; i++)
            {
                Vertex *dst = src + refiner->GetLevel(i - 1).GetNumVertices();
                primvarRefiner.Interpolate(i, src, dst);
                src = dst;
            }

            patchTable->ComputeLocalPointValues(&vertices[0], &vertices[size]);

            int numPatches = patchTable->GetNumPatchesTotal();

            MaterialHandle handle = scene->primIndices[meshIndex].materialID;
            PtexTexture *texture  = 0;
            if (handle)
            {
                int materialIndex = handle.GetIndex();
                texture = (PtexTexture *)baseScene->materials[materialIndex]->displacement;
            }

            tessellatedVertices.Resize(samples.size());

            // Evaluate limit surface positons for corners
            ParallelFor(
                0, (u32)samples.size(), 8092, 8092, [&](int jobID, int start, int count) {
                    for (int i = start; i < start + count; i++)
                    {
                        const LimitSurfaceSample &sample = samples[i];
                        Vec3f pos, dpdu, dpdv, dpduu, dpduv, dpdvv, dndu, dndv;
                        EvaluateLimitSurfacePosition(vertices, &patchMap, patchTable, sample,
                                                     pos, dpdu, dpdv, dpduu, dpduv, dpdvv);
                        Vec3f normal = Normalize(Cross(dpdu, dpdv));
                        CalculateWeingarten(normal, dpdu, dpdv, dpduu, dpduv, dpdvv, dndu,
                                            dndv);
                        dndu = Normalize(dndu - normal * Dot(dndu, normal));
                        dndv = Normalize(dndv - normal * Dot(dndv, normal));

                        tessellatedVertices.vertices[i] = pos;
                        tessellatedVertices.normals[i]  = normal;
                        tessellatedVertices.dpdu[i]     = dpdu;
                        tessellatedVertices.dpdv[i]     = dpdv;
                        tessellatedVertices.dndu[i]     = dndu;
                        tessellatedVertices.dndv[i]     = dndv;
                    }
                });

            // Displacement of shared vertices
            struct TempVertex
            {
                Vec3f pos;
                Vec3f normal;
                // Vec3f dpdu;
                // Vec3f dpdv;
                // f32 areaWeight;
                int count = 0;
            };

            // Evaluate displacements for shared vertices
            TempVertex *tempVertices = 0;
            if (texture)
            {
                tempVertices   = PushArray(scratch.temp.arena, TempVertex, samples.size());
                Mutex *mutexes = PushArray(scratch.temp.arena, Mutex, samples.size());

                auto EvaluateDisplacementHelper = [&](const Vec4f &filterWidths,
                                                      const Vec2f &uv, int f, int id) {
                    Vec3f pos    = tessellatedVertices.vertices[id];
                    Vec3f normal = tessellatedVertices.normals[id];
                    Vec3f dpdu   = tessellatedVertices.dpdu[id];
                    Vec3f dpdv   = tessellatedVertices.dpdv[id];
                    Vec3f dndu   = tessellatedVertices.dndu[id];
                    Vec3f dndv   = tessellatedVertices.dndv[id];

                    EvaluateDisplacement(texture, f, uv, filterWidths, pos, dpdu, dpdv, dndu,
                                         dndv, normal);

                    BeginMutex(&mutexes[id]);
                    TempVertex &vertex = tempVertices[id];
                    vertex.pos += pos;
                    vertex.normal += normal;
                    vertex.count++;
                    EndMutex(&mutexes[id]);
                };
                ParallelFor(
                    0, controlMesh->numFaces, 8092, 8092,
                    [&](int jobID, int start, int count) {
                        for (int f = start; f < start + count; f++)
                        {
                            const FaceInfo &faceInfo = faceInfos[f];

                            int edgeRates[4] = {
                                (int)edgeInfos[faceInfo.edgeInfoId[0]].edgeFactor,
                                (int)edgeInfos[faceInfo.edgeInfoId[1]].edgeFactor,
                                (int)edgeInfos[faceInfo.edgeInfoId[2]].edgeFactor,
                                (int)edgeInfos[faceInfo.edgeInfoId[3]].edgeFactor,
                            };

                            int edgeU  = Max(edgeRates[0], edgeRates[2]);
                            int edgeV  = Max(edgeRates[1], edgeRates[3]);
                            f32 scaleU = 1.f / edgeU;
                            f32 scaleV = 1.f / edgeV;

                            Vec4f filterWidths(.5f * scaleU, 0.f, 0.f, .5f * scaleV);
                            for (int i = 0; i < 4; i++)
                            {
                                int id = controlMesh->indices[4 * f + i];
                                EvaluateDisplacementHelper(filterWidths, uvTable[i], f, id);

                                const EdgeInfo &edgeInfo = edgeInfos[faceInfo.edgeInfoId[i]];
                                int reversed             = faceInfo.reversed[i];
                                int edgeFactor           = edgeInfo.edgeFactor;

                                int edgeStep = PatchItr::start[reversed] * edgeFactor;
                                int edgeDiff = PatchItr::diff[reversed];
                                int edgeEnd  = PatchItr::start[!reversed] * edgeFactor;

                                edgeStep += edgeDiff;
                                Vec2f uvDiff = Vec2f(uvDiffTable[i]) / (f32)edgeFactor;
                                Vec2f uv     = uvTable[i];

                                for (; edgeStep != edgeEnd; edgeStep += edgeDiff)
                                {
                                    uv += uvDiff;
                                    int edgeVertexId = edgeInfo.GetVertexID(edgeStep);
                                    Assert(edgeVertexId < samples.size());

                                    EvaluateDisplacementHelper(filterWidths, uv, f,
                                                               edgeVertexId);
                                }
                            }
                        }
                    });
            }

            // Allocate 1x1 and tessellated patches, calculate number of tessellated grid
            // vertices
            int totalNumPatchFaces = 0;
            std::vector<OpenSubdivPatch> patches;
            std::vector<UntessellatedPatch> untessellatedPatches;
            patches.reserve(controlMesh->numFaces);
            untessellatedPatches.reserve(controlMesh->numFaces);

            int totalSize = 0;

            int vSize = (int)tessellatedVertices.vertices.size();
            Assert(vSize == (int)tessellatedVertices.normals.size());

            for (int f = 0; f < (int)controlMesh->numFaces; f++)
            {
                int face = f;

                FaceInfo &faceInfo = faceInfos[f];
                // Compute limit surface samples
                int edgeRates[4] = {
                    (int)edgeInfos[faceInfo.edgeInfoId[0]].edgeFactor,
                    (int)edgeInfos[faceInfo.edgeInfoId[1]].edgeFactor,
                    (int)edgeInfos[faceInfo.edgeInfoId[2]].edgeFactor,
                    (int)edgeInfos[faceInfo.edgeInfoId[3]].edgeFactor,
                };
                int maxEdgeRates[2] = {
                    Max(edgeRates[0], edgeRates[2]),
                    Max(edgeRates[1], edgeRates[3]),
                };

                if (maxEdgeRates[0] == 1 && maxEdgeRates[1] == 1)
                {
                    untessellatedPatches.emplace_back(UntessellatedPatch{face});

                    for (int quadIndex = 0; quadIndex < 4; quadIndex++)
                    {
                        tessellatedVertices.stitchingIndices.push_back(
                            edgeInfos[faceInfo.edgeInfoId[quadIndex]].GetFirst(
                                faceInfo.reversed[quadIndex]));
                    }

                    continue;
                }

                // Effectively adds a center point
                if (maxEdgeRates[0] == 1) maxEdgeRates[0] = 2;
                if (maxEdgeRates[1] == 1) maxEdgeRates[1] = 2;

                f32 scale[2] = {
                    1.f / maxEdgeRates[0],
                    1.f / maxEdgeRates[1],
                };

                // Inner grid of vertices (local to a patch)
                OpenSubdivPatch patch;
                patch.faceID         = face;
                patch.gridIndexStart = totalSize + vSize;

                totalSize += (maxEdgeRates[1] - 1) * (maxEdgeRates[0] - 1);

                // Generate stitching triangles to compensate for differing edge rates
                for (u32 edgeIndex = 0; edgeIndex < 4; edgeIndex++)
                {
                    int edgeRate                = edgeRates[edgeIndex];
                    const EdgeInfo &currentEdge = edgeInfos[faceInfo.edgeInfoId[edgeIndex]];

                    patch.edgeInfos.indexStart[edgeIndex] = currentEdge.indexStart;
                    patch.edgeInfos.edgeFactors[edgeIndex] =
                        currentEdge.GetStoredEdgeFactor(faceInfo.reversed[edgeIndex]);
                    patch.edgeInfos.ids[edgeIndex] =
                        faceInfo.reversed[edgeIndex] ? currentEdge.id1 : currentEdge.id0;
                }
                patches.push_back(patch);
            }

            Arena *arena              = arenas[GetThreadIndex()];
            outputMesh->vertices      = StaticArray<Vec3f>(arena, vSize + totalSize);
            outputMesh->normals       = StaticArray<Vec3f>(arena, vSize + totalSize);
            outputMesh->vertices.size = vSize + totalSize;
            outputMesh->normals.size  = vSize + totalSize;

            // Resolve displacements now that full buffer size is known
            if (texture)
            {
                ParallelFor(0, (u32)samples.size(), 8092, 8092,
                            [&](int jobID, int start, int count) {
                                for (int i = start; i < start + count; i++)
                                {
                                    TempVertex &vertex      = tempVertices[i];
                                    Vec3f pos               = vertex.pos / (f32)vertex.count;
                                    Vec3f testPos           = tessellatedVertices.vertices[i];
                                    Vec3f testNormal        = tessellatedVertices.normals[i];
                                    Vec3f normal            = Normalize(vertex.normal);
                                    outputMesh->vertices[i] = pos;
                                    outputMesh->normals[i]  = normal;
                                }
                            });
            }
            else
            {
                MemoryCopy(outputMesh->vertices.data, tessellatedVertices.vertices.data(),
                           sizeof(Vec3f) * vSize);
                MemoryCopy(outputMesh->normals.data, tessellatedVertices.normals.data(),
                           sizeof(Vec3f) * vSize);
            }

            outputMesh->stitchingIndices =
                StaticArray<int>(arena, tessellatedVertices.stitchingIndices);
            outputMesh->untessellatedPatches =
                StaticArray<UntessellatedPatch>(arena, untessellatedPatches);
            outputMesh->patches = StaticArray<OpenSubdivPatch>(arena, patches);

            threadMemoryStatistics[GetThreadIndex()].totalShapeMemory +=
                sizeof(Vec3f) * 2 * outputMesh->vertices.Length();
            threadMemoryStatistics[GetThreadIndex()].totalShapeMemory +=
                sizeof(UntessellatedPatch) * outputMesh->untessellatedPatches.Length();
            threadMemoryStatistics[GetThreadIndex()].totalShapeMemory +=
                sizeof(OpenSubdivPatch) * outputMesh->patches.Length();
            threadMemoryStatistics[GetThreadIndex()].totalShapeMemory +=
                sizeof(int) * outputMesh->stitchingIndices.Length();

            tessellatedVertices.Clear();
            untessellatedPatches.clear();
            patches.clear();

            // Evaluate and displace grid vertices
            ParallelFor(
                0, outputMesh->patches.Length(), 8092, 8092,
                [&](int jobID, int start, int count) {
                    for (int i = start; i < start + count; i++)
                    {
                        OpenSubdivPatch *patch = &outputMesh->patches[i];
                        int f                  = patch->faceID;
                        FaceInfo &faceInfo     = faceInfos[f];
                        // Compute limit surface samples
                        int edgeRates[4] = {
                            (int)edgeInfos[faceInfo.edgeInfoId[0]].GetEdgeFactor(),
                            (int)edgeInfos[faceInfo.edgeInfoId[1]].GetEdgeFactor(),
                            (int)edgeInfos[faceInfo.edgeInfoId[2]].GetEdgeFactor(),
                            (int)edgeInfos[faceInfo.edgeInfoId[3]].GetEdgeFactor(),
                        };
                        int maxEdgeRates[2] = {
                            Max(edgeRates[0], edgeRates[2]),
                            Max(edgeRates[1], edgeRates[3]),
                        };

                        // Effectively adds a center point
                        if (maxEdgeRates[0] == 1) maxEdgeRates[0] = 2;
                        if (maxEdgeRates[1] == 1) maxEdgeRates[1] = 2;

                        f32 scale[2] = {
                            1.f / maxEdgeRates[0],
                            1.f / maxEdgeRates[1],
                        };
                        // Generate n x m interior grid of points
                        for (int vStep = 0; vStep < maxEdgeRates[1] - 1; vStep++)
                        {
                            for (int uStep = 0; uStep < maxEdgeRates[0] - 1; uStep++)
                            {
                                Vec2f uv(scale[0] * (uStep + 1), scale[1] * (vStep + 1));

                                Vec3f pos, dpdu, dpdv, dpduu, dpduv, dpdvv;
                                if (texture)
                                    EvaluateLimitSurfacePosition(
                                        vertices, &patchMap, patchTable,
                                        LimitSurfaceSample{f, uv}, pos, dpdu, dpdv, dpduu,
                                        dpduv, dpdvv);
                                else
                                    EvaluateLimitSurfacePosition(
                                        vertices, &patchMap, patchTable,
                                        LimitSurfaceSample{f, uv}, pos, dpdu, dpdv);

                                Vec3f normal = Normalize(Cross(dpdu, dpdv));

                                if (texture)
                                {
                                    Vec4f filterWidths(.5f * scale[0], 0, 0, .5f * scale[1]);

                                    Vec3f dndu, dndv;
                                    CalculateWeingarten(normal, dpdu, dpdv, dpduu, dpduv,
                                                        dpdvv, dndu, dndv);

                                    EvaluateDisplacement(texture, f, uv, filterWidths, pos,
                                                         dpdu, dpdv, dndu, dndv, normal);
                                    normal = Normalize(normal);
                                }

                                int index                  = patch->GetGridIndex(uStep, vStep);
                                outputMesh->normals[index] = normal;
                                outputMesh->vertices[index] = pos;
                            }
                        }
                    }
                });

            delete refiner;
            delete patchTable;
        }
    });
    return outputMeshes;
}

} // namespace rt
