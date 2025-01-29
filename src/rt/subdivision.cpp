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

// PatchItr OpenSubdivPatch::GetUVs(int edge, int id, Vec2f uv[3]) const
// {
//     PatchItr itr = PatchItr(this, edge);
//     itr.GetUVs(id, uv);
//     return itr;
// }

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

// Edge vertices (shared by edges)
// TODO: use half-edge structure?

struct FaceInfo
{
    int edgeInfoId[4];
    bool reversed[4];
};

struct TessellatedVertices
{
    std::vector<Vec3f> vertices;
    std::vector<Vec3f> normals;

    // Stitching triangles
    std::vector<int> stitchingIndices;

    // stack variables
    int currentGridOffset;
    int uMax;
    int vMax;

    // uMax is the max edge rate in the u direction, same for vMax
    // "Inner grid" contains the (uMax - 1) x (vMax - 1) inner tessellated vertices
    // specific to a patch
    // NOTE: this must be called AFTER all corner/edge vertices are added for a given patch
    void StartNewGrid(int maxU, int maxV)
    {
        uMax              = maxU;
        vMax              = maxV;
        currentGridOffset = (int)vertices.size();
        vertices.resize(vertices.size() + ((uMax - 1) * (vMax - 1)));
    }
    int GetInnerGridIndex(int u, int v) const
    {
        Assert(u >= 0 && v >= 0 && u < uMax - 1 && v < vMax - 1);
        int gridIndex = currentGridOffset + v * (uMax - 1) + u;
        Assert(gridIndex < (int)vertices.size());
        return gridIndex;
    }
    Vec3f &SetInnerGrid(int u, int v)
    {
        int gridIndex = GetInnerGridIndex(u, v);
        return vertices[gridIndex];
    }
};

// TODO:
// 1. patch bvh instead of triangles, fix uvs
// 2. reduce memory usage
// 3. set edge rates per instance

// MORETON, H. 2001. Watertight tessellation using forward
// differencing. In HWWS ’01: Proceedings of the ACM SIG-
// GRAPH/EUROGRAPHICS workshop on Graphics hardware,
// ACM, New York, NY, USA, 25–32
// See Figure 7 from above for how this works
OpenSubdivMesh *AdaptiveTessellation(Arena *arena, ScenePrimitives *scene,
                                     const Mat4 &NDCFromCamera, int screenHeight,
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

    // TODO: catmull clark primitive
    OpenSubdivMesh *outputMeshes = PushArray(arena, OpenSubdivMesh, numMeshes);

    for (u32 meshIndex = 0; meshIndex < numMeshes; meshIndex++)
    {
        OpenSubdivMesh *outputMesh = &outputMeshes[meshIndex];

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

        i32 *numVertsPerFace = PushArrayNoZero(scratch.temp.arena, i32, controlMesh->numFaces);
        for (u32 i = 0; i < controlMesh->numFaces; i++)
        {
            numVertsPerFace[i] = 4;
        }

        Descriptor desc;
        desc.numVertices        = controlMesh->numVertices;
        desc.numFaces           = controlMesh->numFaces;
        desc.numVertsPerFace    = numVertsPerFace;
        desc.vertIndicesPerFace = (int *)controlMesh->indices;

        int maxMeshEdgeRate = 0;

        // NOTE: this assumes everything is a quad
        // Compute edge rates
        std::unordered_map<u64, int> edgeIDMap;
        for (int f = 0; f < (int)controlMesh->numFaces; f++)
        {
            int indexOffset = 4 * f;
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
                    // Compute tessellation factor
                    Vec3f p0       = controlMesh->p[id0];
                    Vec3f p1       = controlMesh->p[id1];
                    Vec3f midPoint = (p0 + p1) / 2.f;
                    f32 radius     = Length(p0 - p1) / 2.f;

                    int edgeFactor = int(edgesPerScreenHeight * radius *
                                         Abs(NDCFromCamera[1][1] / midPoint.z)) -
                                     1;

                    edgeFactor = Clamp(edgeFactor, 1, 8);
                    // edgeFactor     = Clamp(edgeFactor, 1, 64);

                    maxMeshEdgeRate = Max(maxMeshEdgeRate, edgeFactor);

                    edgeIndex = (int)edgeInfos.size();
                    edgeInfos.emplace_back(
                        EdgeInfo{(int)samples.size(), (u32)edgeFactor, id0, id1});
                    edgeIDMap[edgeId] = edgeIndex;

                    // Add edge samples for evaluation
                    f32 stepSize = 1.f / edgeFactor;
                    for (int edgeVertexIndex = 1; edgeVertexIndex < edgeFactor;
                         edgeVertexIndex++)
                    {
                        Vec2f uv =
                            uvTable[i] + Vec2f(uvDiffTable[i]) * (edgeVertexIndex * stepSize);
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

        int maxLevel                    = refiner->GetMaxLevel();
        const Far::TopologyLevel &level = refiner->GetLevel(0);
        int faces                       = level.GetNumFaces();

        auto *patchTable = Far::PatchTableFactory::Create(*refiner, patchOptions);
        OpenSubdiv::Far::PatchMap patchMap(*patchTable);

        // Feature adaptive refinment
        int size           = refiner->GetNumVerticesTotal();
        int numLocalPoints = patchTable->GetNumLocalPoints();
        Vertex *vertices = PushArrayNoZero(scratch.temp.arena, Vertex, size + numLocalPoints);
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

        // Evaluate limit surface positons for corners
        for (auto &sample : samples)
        {
            Vec3f pos, dpdu, dpdv;
            EvaluateLimitSurfacePosition(vertices, &patchMap, patchTable, sample, pos, dpdu,
                                         dpdv);
            Vec3f normal = Normalize(Cross(dpdu, dpdv));
            tessellatedVertices.vertices.push_back(pos);
            tessellatedVertices.normals.push_back(normal);
        }

        int totalNumPatchFaces = 0;
        std::vector<OpenSubdivPatch> patches;
        std::vector<UntessellatedPatch> untessellatedPatches;
        patches.reserve(controlMesh->numFaces);
        untessellatedPatches.reserve(controlMesh->numFaces);
        for (int f = 0; f < (int)controlMesh->numFaces; f++)
        {
            int face = f;
            // int face =
            //     f >= (int)controlMesh->numFaces / 2 ? f - (int)controlMesh->numFaces / 2
            //     : f;

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
            f32 scale[2] = {
                1.f / maxEdgeRates[0],
                1.f / maxEdgeRates[1],
            };

            if (maxEdgeRates[0] == 1 && maxEdgeRates[1] == 1)
            {
                untessellatedPatches.emplace_back(UntessellatedPatch{
                    face, (int)tessellatedVertices.stitchingIndices.size()});

                for (int quadIndex = 0; quadIndex < 4; quadIndex++)
                {
                    tessellatedVertices.stitchingIndices.push_back(
                        edgeInfos[faceInfo.edgeInfoId[quadIndex]].GetFirst(
                            faceInfo.reversed[quadIndex]));
                }

                continue;
            }

            // Effectively adds a center point
            if (maxEdgeRates[0] == 1)
            {
                maxEdgeRates[0] = 2;
                scale[0]        = 0.5f;
            }
            if (maxEdgeRates[1] == 1)
            {
                maxEdgeRates[1] = 2;
                scale[1]        = 0.5f;
            }
            // Inner grid of vertices (local to a patch)
            tessellatedVertices.StartNewGrid(maxEdgeRates[0], maxEdgeRates[1]);

            OpenSubdivPatch patch;
            patch.faceID         = face;
            patch.gridIndexStart = tessellatedVertices.currentGridOffset;

            // Generate n x m interior grid of points
            for (int vStep = 0; vStep < maxEdgeRates[1] - 1; vStep++)
            {
                for (int uStep = 0; uStep < maxEdgeRates[0] - 1; uStep++)
                {
                    Vec2f uv(scale[0] * (uStep + 1), scale[1] * (vStep + 1));

                    Vec3f pos, dpdu, dpdv, dpduu, dpduv, dpdvv;
                    EvaluateLimitSurfacePosition(vertices, &patchMap, patchTable,
                                                 LimitSurfaceSample{f, uv}, pos, dpdu, dpdv,
                                                 dpduu, dpduv, dpdvv);
                    Vec3f normal = Normalize(Cross(dpdu, dpdv));

                    if (texture)
                    {
                        // Vector Displacement mapping
                        auto frame = LinearSpace3f::FromXZ(Normalize(dpdu), normal);
                        Vec3f displacement, uDisplacement, vDisplacement;
                        Vec4f filterWidths(.5f * scale[0], 0, 0, .5f * scale[1]);
                        SurfaceInteraction intr;
                        intr.uv          = uv;
                        intr.faceIndices = face;
                        texture->EvaluateHelper<3>(intr, filterWidths, displacement.e);

                        f32 du  = .5f * scale[0];
                        intr.uv = uv + Vec2f(du, 0.f);
                        texture->EvaluateHelper<3>(intr, filterWidths, uDisplacement.e);

                        f32 dv  = .5f * scale[1];
                        intr.uv = uv + Vec2f(0.f, dv);
                        texture->EvaluateHelper<3>(intr, filterWidths, vDisplacement.e);

                        // Calculate dndu and dndv using weingarten equations
#if 0
                        f32 E = Dot(dpdu, dpdu);
                        f32 F = Dot(dpdu, dpdv);
                        f32 G = Dot(dpdv, dpdv);

                        f32 e    = Dot(normal, dpduu);
                        f32 fvar = Dot(normal, dpduv);
                        f32 g    = Dot(normal, dpdvv);

                        f32 invEGF2 = Rcp(FMS(E, G, Sqr(F)));
                        Vec3f dndu =
                            invEGF2 * (dpdu * (fvar * F - e * G) + dpdv * (e * F - fvar * E));
                        Vec3f dndv =
                            invEGF2 * (dpdu * (g * F - fvar * G) + dpdv * (fvar * F - g * E));
                        dndu = Normalize(dndu - normal * Dot(dndu, normal));
                        dndv = Normalize(dndv - normal * Dot(dndv, normal));
#endif

                        displacement  = frame.FromLocal(displacement);
                        uDisplacement = frame.FromLocal(uDisplacement);
                        vDisplacement = frame.FromLocal(vDisplacement);

                        pos += displacement;
                        // dpdu = dpdu + (uDisplacement - displacement) / du * normal +
                        //        displacement * dndu;
                        // dpdv = dpdv + (vDisplacement - displacement) / dv * normal +
                        //        displacement * dndv;
                        dpdu += (uDisplacement - displacement) / du;
                        dpdv += (vDisplacement - displacement) / dv;
                        normal = Normalize(Cross(dpdu, dpdv));
                    }

                    tessellatedVertices.normals.push_back(normal);
                    tessellatedVertices.SetInnerGrid(uStep, vStep) = pos;
                }
            }

            // Generate stitching triangles to compensate for differing edge rates
            for (u32 edgeIndex = 0; edgeIndex < 4; edgeIndex++)
            {
                int edgeRate                = edgeRates[edgeIndex];
                const EdgeInfo &currentEdge = edgeInfos[faceInfo.edgeInfoId[edgeIndex]];

                patch.edgeInfo.Push(faceInfo.reversed[edgeIndex] ? currentEdge.Opposite()
                                                                 : currentEdge);
            }
            patches.push_back(patch);
        }

        outputMesh->vertices = StaticArray<Vec3f>(arena, tessellatedVertices.vertices);
        outputMesh->normals  = StaticArray<Vec3f>(arena, tessellatedVertices.normals);
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

        delete refiner;
        delete patchTable;
    }
    return outputMeshes;
}

} // namespace rt
