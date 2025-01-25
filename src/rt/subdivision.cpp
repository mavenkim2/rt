#include "math/matx.h"
#define M_PI 3.1415926535897932
#include <opensubdiv/far/topologyDescriptor.h>
#include <opensubdiv/far/primvarRefiner.h>
#include <opensubdiv/far/patchTableFactory.h>
#include <opensubdiv/far/ptexIndices.h>
#include <opensubdiv/far/patchMap.h>

namespace rt
{

struct Vertex
{
    void Clear(void * = 0) { p = Vec3f(0.f); }
    void AddWithWeight(Vertex &src, f32 weight) { p += weight * src.p; }
    Vec3f p;
};

using namespace OpenSubdiv;
void Subdivide(Mesh *mesh)
{
    TempArena temp = ScratchStart(0, 0);
    typedef Far::TopologyDescriptor Descriptor;
    typedef Far::TopologyDescriptor Descriptor;
    Sdc::SchemeType type = OpenSubdiv::Sdc::SCHEME_CATMARK;
    Sdc::Options options;
    options.SetVtxBoundaryInterpolation(Sdc::Options::VTX_BOUNDARY_EDGE_AND_CORNER);

    int maxLevel         = 3;
    i32 *numVertsPerFace = PushArrayNoZero(temp.arena, i32, mesh->numFaces);
    for (u32 i = 0; i < mesh->numFaces; i++)
    {
        numVertsPerFace[i] = 4;
    }

    Descriptor desc;
    desc.numVertices        = mesh->numVertices;
    desc.numFaces           = mesh->numFaces;
    desc.numVertsPerFace    = numVertsPerFace;
    desc.vertIndicesPerFace = (int *)mesh->indices;

    // TODOs:
    // 1. how do you interpolate the normals during subdivision?
    // 2. make this work with ptex
    // 3. adaptive subdivision based on screen space heuristics?
    // 4. multi resolution geometry cache?

    Far::TopologyRefiner *refiner = Far::TopologyRefinerFactory<Descriptor>::Create(
        desc, Far::TopologyRefinerFactory<Descriptor>::Options(type, options));

    Far::PatchTableFactory::Options patchOptions(maxLevel);
    patchOptions.SetEndCapType(Far::PatchTableFactory::Options::ENDCAP_GREGORY_BASIS);

    Far::TopologyRefiner::AdaptiveOptions adaptiveOptions =
        patchOptions.GetRefineAdaptiveOptions();
    refiner->RefineAdaptive(adaptiveOptions);

    u32 size         = refiner->GetNumVerticesTotal();
    Vertex *vertices = PushArrayNoZero(temp.arena, Vertex, size);
    MemoryCopy(vertices, mesh->p, sizeof(Vec3f) * mesh->numVertices);

    Vertex *src        = vertices;
    i32 nRefinedLevels = refiner->GetNumLevels();
    Far::PrimvarRefiner primvarRefiner(*refiner);

    for (i32 i = 1; i < nRefinedLevels; i++)
    {
        Vertex *dst = src + refiner->GetLevel(i - 1).GetNumVertices();
        primvarRefiner.Interpolate(i, src, dst);
        src = dst;
    }

    auto *patchTable = Far::PatchTableFactory::Create(*refiner, patchOptions);
    OpenSubdiv::Far::PatchMap patchMap(*patchTable);

    u32 numPatches = patchTable->GetNumPatchesTotal();

    Far::PtexIndices ptexIndices(*refiner);
    int numFaces = ptexIndices.GetNumFaces();

    for (int i = 0; i < numFaces; i++)
    {
        i32 faceID = ptexIndices.GetFaceId(i);
        f32 pWeights[20];
        f32 duWeights[20];
        f32 dvWeights[20];
        Vec3f cornerPos[4] = {};

        Vec2f uvs[4] = {
            Vec2f(0.f, 0.f),
            Vec2f(1.f, 0.f),
            Vec2f(1.f, 1.f),
            Vec2f(0.f, 1.f),
        };

        for (int c = 0; c < 4; c++)
        {
            const Far::PatchTable::PatchHandle *handle =
                patchMap.FindPatch(faceID, uvs[c][0], uvs[c][1]);
            Assert(handle);
            const auto &cvIndices = patchTable->GetPatchVertices(*handle);

            patchTable->EvaluateBasis(*handle, uvs[c][0], uvs[c][1], pWeights, duWeights,
                                      dvWeights);
            Vec3f pos = {};
            for (int j = 0; j < cvIndices.size(); j++)
            {
                int index = cvIndices[j];
                if (index == 25618 && c == 0)
                {
                    int stop = 5;
                }
                const Vec3f &p = vertices[index].p;
                pos += pWeights[j] * p;
            }
            cornerPos[c] = pos;
            int stop     = 5;
        }
    }
    delete refiner;
}

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
struct EdgeInfo
{
    int indexStart;
    int edgeFactor;
    int id0, id1;

    int GetFirst(bool reversed) const { return reversed ? id1 : id0; }
    int GetLast(bool reversed) const { return reversed ? id0 : id1; }

    int GetVertexId(int edgeStep) const
    {
        Assert(edgeStep >= 0 && edgeStep <= edgeFactor);
        return edgeStep == 0 ? id0
                             : (edgeStep == edgeFactor ? id1 : indexStart + edgeStep - 1);
    }
};

struct FaceInfo
{
    int edgeInfoId[4];
    bool reversed[4];
};

struct TessellatedVertices
{
    std::vector<Vec3f> vertices;
    std::vector<Vec3f> normals;
    std::vector<Vec2f> uvs;
    std::vector<int> faceIDs;
    // Stitching triangles
    std::vector<int> stitchingIndices;
    std::vector<EdgeInfo> edgeInfos;

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

// MORETON, H. 2001. Watertight tessellation using forward
// differencing. In HWWS ’01: Proceedings of the ACM SIG-
// GRAPH/EUROGRAPHICS workshop on Graphics hardware,
// ACM, New York, NY, USA, 25–32
// See Figure 7 from above for how this works
Mesh *AdaptiveTessellation(Arena *arena, ScenePrimitives *scene, const Mat4 &NDCFromCamera,
                           int screenHeight, Mesh *controlMeshes, u32 numMeshes)
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

    int quadTable[] = {
        0, 1, 2, 0, 2, 3,
    };

    // TODO: catmull clark primitive
    // OpenSubdivMesh *outputMeshes = PushArray(arena, OpenSubdivMesh, numMeshes);
    Mesh *outputMeshes = PushArray(arena, Mesh, numMeshes);

    for (u32 meshIndex = 0; meshIndex < numMeshes; meshIndex++)
    {
        // OpenSubdivMesh *outputMesh = &outputMeshes[meshIndex];
        Mesh *outputMesh = &outputMeshes[meshIndex];

        TessellatedVertices tessellatedVertices;
        Mesh *controlMesh = &controlMeshes[meshIndex];
        tessellatedVertices.vertices.reserve(tessellatedVertices.vertices.capacity() +
                                             controlMesh->numVertices);
        tessellatedVertices.normals.reserve(tessellatedVertices.normals.capacity() +
                                            controlMesh->numVertices);
        tessellatedVertices.uvs.reserve(tessellatedVertices.uvs.capacity() +
                                        controlMesh->numVertices);
        tessellatedVertices.faceIDs.reserve(tessellatedVertices.faceIDs.capacity() +
                                            controlMesh->numIndices / 3);

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
                                         Abs(NDCFromCamera[1][1] / midPoint.z));
                    edgeFactor     = Max(1, edgeFactor);

                    maxMeshEdgeRate = Max(maxMeshEdgeRate, edgeFactor);

                    edgeIndex = (int)edgeInfos.size();
                    edgeInfos.emplace_back(
                        EdgeInfo{(int)samples.size(), edgeFactor, id0, id1});
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

        Far::PatchTableFactory::Options patchOptions(maxMeshEdgeRate);
        patchOptions.SetEndCapType(Far::PatchTableFactory::Options::ENDCAP_GREGORY_BASIS);

        Far::TopologyRefiner::AdaptiveOptions adaptiveOptions =
            patchOptions.GetRefineAdaptiveOptions();
        refiner->RefineAdaptive(adaptiveOptions);

        auto *patchTable = Far::PatchTableFactory::Create(*refiner, patchOptions);
        OpenSubdiv::Far::PatchMap patchMap(*patchTable);

        // Feature adaptive refinment
        int size         = refiner->GetNumVerticesTotal();
        Vertex *vertices = PushArrayNoZero(scratch.temp.arena, Vertex, size);
        MemoryCopy(vertices, controlMesh->p, sizeof(Vec3f) * controlMesh->numVertices);

        Vertex *src        = vertices;
        i32 nRefinedLevels = refiner->GetNumLevels();
        Far::PrimvarRefiner primvarRefiner(*refiner);

        Assert(nRefinedLevels == 1);
        for (i32 i = 1; i < nRefinedLevels; i++)
        {
            Vertex *dst = src + refiner->GetLevel(i - 1).GetNumVertices();
            primvarRefiner.Interpolate(i, src, dst);
            src = dst;
        }

        PtexTexture *texture =
            (PtexTexture *)baseScene
                ->materials[scene->primIndices[meshIndex].materialID.GetIndex()]
                ->displacement;

        // Evaluate limit surface positons for corners
        for (auto &sample : samples)
        {
            Vec3f pos, dpdu, dpdv;
            EvaluateLimitSurfacePosition(vertices, &patchMap, patchTable, sample, pos, dpdu,
                                         dpdv);
            Vec3f normal = Normalize(Cross(dpdu, dpdv));
            tessellatedVertices.vertices.push_back(pos);
            tessellatedVertices.normals.push_back(normal);
            tessellatedVertices.uvs.push_back(sample.uv);
        }

        int totalNumPatchFaces = 0;
        std::vector<OpenSubdivPatch> patches;
        patches.reserve(controlMesh->numFaces);
        for (int f = 0; f < (int)controlMesh->numFaces; f++)
        {
            int face = f;
            // TODO THIS IS A HACK
            // int face =
            //     f >= (int)controlMesh->numFaces / 2 ? f - (int)controlMesh->numFaces / 2 :
            //     f;

            FaceInfo &faceInfo = faceInfos[f];
            // Compute limit surface samples
            int edgeRates[4] = {
                edgeInfos[faceInfo.edgeInfoId[0]].edgeFactor,
                edgeInfos[faceInfo.edgeInfoId[1]].edgeFactor,
                edgeInfos[faceInfo.edgeInfoId[2]].edgeFactor,
                edgeInfos[faceInfo.edgeInfoId[3]].edgeFactor,
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
                for (int triIndex = 0; triIndex < ArrayLength(quadTable); triIndex++)
                {
                    int index = quadTable[triIndex];
                    tessellatedVertices.stitchingIndices.push_back(
                        edgeInfos[faceInfo.edgeInfoId[index]].GetFirst(
                            faceInfo.reversed[index]));
                }

                tessellatedVertices.faceIDs.push_back(face);
                tessellatedVertices.faceIDs.push_back(face);

                continue;
            }

            // Effectively adds a center point
            if (maxEdgeRates[0] == 1 || maxEdgeRates[1] == 1)
            {
                maxEdgeRates[0] = 2;
                maxEdgeRates[1] = 2;
                scale[0]        = 0.5f;
                scale[1]        = 0.5f;
            }
            // Inner grid of vertices (local to a patch)
            tessellatedVertices.StartNewGrid(maxEdgeRates[0], maxEdgeRates[1]);
            if (maxEdgeRates[0] > 2 && maxEdgeRates[1] > 2)
            {
                patches.emplace_back(OpenSubdivPatch{tessellatedVertices.currentGridOffset,
                                                     maxEdgeRates[0] - 1,
                                                     maxEdgeRates[1] - 1});
                totalNumPatchFaces += (maxEdgeRates[0] - 2) * (maxEdgeRates[1] - 2);
            }

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
                    tessellatedVertices.uvs.push_back(uv);
                    tessellatedVertices.SetInnerGrid(uStep, vStep) = pos;

                    // TODO: create a new catmull clark primitive
                    if (uStep != maxEdgeRates[0] - 2 && vStep != maxEdgeRates[1] - 2)
                    {
                        int v0 = tessellatedVertices.GetInnerGridIndex(uStep, vStep);
                        int v1 = tessellatedVertices.GetInnerGridIndex(uStep + 1, vStep);
                        int v2 = tessellatedVertices.GetInnerGridIndex(uStep + 1, vStep + 1);
                        int v3 = tessellatedVertices.GetInnerGridIndex(uStep, vStep + 1);

                        tessellatedVertices.stitchingIndices.push_back(v0);
                        tessellatedVertices.stitchingIndices.push_back(v1);
                        tessellatedVertices.stitchingIndices.push_back(v2);

                        tessellatedVertices.faceIDs.push_back(face);

                        tessellatedVertices.stitchingIndices.push_back(v0);
                        tessellatedVertices.stitchingIndices.push_back(v2);
                        tessellatedVertices.stitchingIndices.push_back(v3);

                        tessellatedVertices.faceIDs.push_back(face);
                    }
                }
            }

            // Generate stitching triangles to compensate for differing edge rates
            for (u32 edgeIndex = 0; edgeIndex < 4; edgeIndex++)
            {
                int edgeRate                = edgeRates[edgeIndex];
                const EdgeInfo &currentEdge = edgeInfos[faceInfo.edgeInfoId[edgeIndex]];

                int maxEdgeRate = maxEdgeRates[edgeIndex & 1];
                // Generate indices
                int q = maxEdgeRate - edgeRate - 2 * edgeRate;

                int edgeStep = faceInfo.reversed[edgeIndex] ? currentEdge.edgeFactor : 0;
                int edgeDiff = faceInfo.reversed[edgeIndex] ? -1 : 1;
                int edgeEnd  = faceInfo.reversed[edgeIndex] ? 0 : currentEdge.edgeFactor;

                Vec2i gridStep =
                    Vec2i(Max(maxEdgeRates[0] - 2, 0), Max(maxEdgeRates[1] - 2, 0));

                Vec2i uvStartInnerGrid = Vec2i(uvTable[edgeIndex]) * gridStep;
                Vec2i uvEnd            = uvStartInnerGrid + uvDiffTable[edgeIndex] * gridStep;

                while (edgeStep != edgeEnd || uvStartInnerGrid != uvEnd)
                {
                    bool currentSide = q >= 0;
                    if (currentSide && uvStartInnerGrid != uvEnd)
                    {
                        int id0 = tessellatedVertices.GetInnerGridIndex(uvStartInnerGrid[0],
                                                                        uvStartInnerGrid[1]);
                        int id1 = currentEdge.GetVertexId(edgeStep);
                        uvStartInnerGrid += uvDiffTable[edgeIndex];
                        int id2 = tessellatedVertices.GetInnerGridIndex(uvStartInnerGrid[0],
                                                                        uvStartInnerGrid[1]);

                        tessellatedVertices.stitchingIndices.push_back(id0);
                        tessellatedVertices.stitchingIndices.push_back(id1);
                        tessellatedVertices.stitchingIndices.push_back(id2);

                        tessellatedVertices.faceIDs.push_back(face);

                        q -= 2 * edgeRate;
                    }
                    else
                    {
                        int id0 = currentEdge.GetVertexId(edgeStep);
                        edgeStep += edgeDiff;
                        Assert(edgeStep <= currentEdge.edgeFactor && edgeStep >= 0);
                        int id1 = currentEdge.GetVertexId(edgeStep);
                        int id2 = tessellatedVertices.GetInnerGridIndex(uvStartInnerGrid[0],
                                                                        uvStartInnerGrid[1]);

                        tessellatedVertices.stitchingIndices.push_back(id0);
                        tessellatedVertices.stitchingIndices.push_back(id1);
                        tessellatedVertices.stitchingIndices.push_back(id2);

                        tessellatedVertices.faceIDs.push_back(face);

                        q += 2 * maxEdgeRate;
                    }
                }
            }
        }

        outputMesh->p = PushArrayNoZero(arena, Vec3f, tessellatedVertices.vertices.size());
        MemoryCopy(outputMesh->p, tessellatedVertices.vertices.data(),
                   sizeof(Vec3f) * tessellatedVertices.vertices.size());
        outputMesh->numVertices = (int)tessellatedVertices.vertices.size();

        outputMesh->indices =
            PushArrayNoZero(arena, u32, tessellatedVertices.stitchingIndices.size());
        MemoryCopy(outputMesh->indices, tessellatedVertices.stitchingIndices.data(),
                   sizeof(int) * tessellatedVertices.stitchingIndices.size());
        outputMesh->numIndices = (u32)tessellatedVertices.stitchingIndices.size();
        outputMesh->numFaces   = outputMesh->numIndices / 3;

        Assert(tessellatedVertices.normals.size() == tessellatedVertices.vertices.size());
        outputMesh->n = PushArrayNoZero(arena, Vec3f, tessellatedVertices.normals.size());
        MemoryCopy(outputMesh->n, tessellatedVertices.normals.data(),
                   sizeof(Vec3f) * tessellatedVertices.normals.size());

        Assert(tessellatedVertices.vertices.size() == tessellatedVertices.uvs.size());
        outputMesh->uv = PushArrayNoZero(arena, Vec2f, tessellatedVertices.uvs.size());
        MemoryCopy(outputMesh->uv, tessellatedVertices.uvs.data(),
                   sizeof(Vec2f) * tessellatedVertices.uvs.size());

        // outputMesh->vertices = StaticArray<Vec3f>(arena, tessellatedVertices.vertices);
        // outputMesh->stitchingIndices =
        //     StaticArray<int>(arena, tessellatedVertices.stitchingIndices);
        // outputMesh->patches = StaticArray<OpenSubdivPatch>(arena, patches);

        Assert(tessellatedVertices.stitchingIndices.size() ==
               tessellatedVertices.faceIDs.size() * 3);
        outputMesh->faceIDs = PushArrayNoZero(arena, int, tessellatedVertices.faceIDs.size());
        MemoryCopy(outputMesh->faceIDs, tessellatedVertices.faceIDs.data(),
                   sizeof(int) * tessellatedVertices.faceIDs.size());

        delete refiner;
        delete patchTable;
    }
    return outputMeshes;
}

} // namespace rt
