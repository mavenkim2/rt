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
    // 1. how do you interpolate the normals durign subdivision?
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

// void IntersectCatmullClarkPatch(const Vec3f &dpdx, const Vec3f &dpdy)
// {
//     // Calculate level in the multi resolution geometry cache
//     // so many questions
// }

// MORETON, H. 2001. Watertight tessellation using forward
// differencing. In HWWS ’01: Proceedings of the ACM SIG-
// GRAPH/EUROGRAPHICS workshop on Graphics hardware,
// ACM, New York, NY, USA, 25–32

u64 ComputeEdgeId(u32 id0, u32 id1)
{
    return id0 < id1 ? ((u64)id1 << 32) | id0 : ((u64)id0 << 32) | id1;
}

// See Figure 7 from above for how this works
void AdaptiveTessellation(const Mat4 &c2s, Mesh *controlMeshes, u32 numMeshes)
{
    typedef Far::TopologyDescriptor Descriptor;
    Sdc::SchemeType type = OpenSubdiv::Sdc::SCHEME_CATMARK;
    Sdc::Options options;
    options.SetVtxBoundaryInterpolation(Sdc::Options::VTX_BOUNDARY_EDGE_AND_CORNER);

    u32 edgesPerHeight       = 1;
    u32 edgesPerScreenHeight = screenHeight / edgesPerHeight;

    // Edge vertices (shared by edges)
    struct EdgeInfo
    {
        int indexStart;
        int indexCount;
        int id0, id1;
    };

    // i have no idea what to do for creases/extraordinary points
    for (u32 meshIndex = 0; meshIndex < numMeshes; meshIndex++)
    {
        Mesh *controlMesh = &meshes[meshIndex];
        // Limit surface positions of control cage vertices
        std::vector<Vec3f> limitVertices;
        limitVertices.reserve(controlMesh->numVertices);

        std::vector<Vec3f> edgeVertices;
        std::vector<EdgeInfo> edgeInfo;
        int edgeCount = 0;

        ScratchArena scratch;

        i32 *numVertsPerFace = PushArrayNoZero(scratch.temp.arena, i32, controlMesh->numFaces);
        for (u32 i = 0; i < controlMesh->numFaces; i++)
        {
            numVertsPerFace[i] = 4;
        }

        Descriptor desc;
        desc.numVertices        = controlMesh->numVertices;
        desc.numFaces           = controlMesh->numFaces;
        desc.numVertsPerFace    = numVertsPerFace;
        desc.vertIndicesPerFace = (int *)mesh->indices;

        int maxEdgeRate = 0;

        // NOTE: this assumes everything is a quad
        // Compute edge rates
        std::unordered_map<u64, int> edgeRateMap;
        for (int f = 0; f < mesh->numFaces; f++)
        {
            int indexOffset = 4 * f;
            for (int i = 0; i < 4; i++)
            {
                int id0    = mesh->indices[indexOffset + i];
                int id1    = mesh->indices[indexOffset + ((i + 1) & 3)];
                u64 edgeId = ComputeEdgeId(id0, id1);

                if (const auto &result = edgeRateMap.find(edgeId); result == edgeRateMap.end())
                {
                    // Compute tessellation factor
                    Vec3f p0       = mesh->p[id0];
                    Vec3f p1       = mesh->p[id1];
                    Vec3f midPoint = (p0 + p1) / 2.f;
                    f32 radius     = Length(p0 - p1) / 2.f;

                    int edgeVertexCount =
                        edgesPerScreenHeight * radius * Abs(c2s[1][1] / midPoint.z);

                    edgeRateMap[edgeId] = edgeVertexCount;
                    maxEdgeRate         = Max(maxEdgeRate, edgeVertexCount);

                    int edgeIndex                  = edgeCount++;
                    edgeInfo[edgeIndex].indexStart = (int)edgeVertices.size();
                    edgeInfo[edgeIndex].indexCount = edgeVertexCount;
                    edgeInfo[edgeIndex].id0        = id0 < id1 ? id0 : id1;
                    edgeInfo[edgeIndex].id1        = id0 < id1 ? id1 : id0;
                }
            }
        }

        // Create OpenSubdiv objects
        Far::TopologyRefiner *refiner = Far::TopologyRefinerFactory<Descriptor>::Create(
            desc, Far::TopologyRefinerFactory<Descriptor>::Options(type, options));

        Far::PatchTableFactory::Options patchOptions(maxEdgeRate);
        patchOptions.SetEndCapType(Far::PatchTableFactory::Options::ENDCAP_GREGORY_BASIS);

        Far::TopologyRefiner::AdaptiveOptions adaptiveOptions =
            patchOptions.GetRefineAdaptiveOptions();
        refiner->RefineAdaptive(adaptiveOptions);

        auto *patchTable = Far::PatchTableFactory::Create(*refiner, patchOptions);
        OpenSubdiv::Far::PatchMap patchMap(*patchTable);

        // Feature adaptive refinment
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

        // Compute limit surface samples
        int maxEdgeRates[2] = {
            Max(edgeRates[0], edgeRates[2]),
            Max(edgeRates[1], edgeRates[3]),
        };
        f32 scale[2] = {
            1.f / maxEdgeRates[0],
            1.f / maxEdgeRates[1],
        };

        // Inner grid of vertices (local to a patch)
        std::vector<Vec3f> grid;
        // Maps edgeID to an index used to index an array

        std::unordered_map<>;

        grid.reserve(maxEdgeRates[0] * maxEdgeRates[1]);

        // Generate n x m grid of points
        for (int m = 0; m < maxEdgeRates[1]; m++)
        {
            for (int n = 0; n < maxEdgeRates[0]; n++)
            {
                Vec2f uv(scale[0] * n, scale[1] * m);
                const Far::PatchTable::PatchHandle *handle =
                    patchMap.FindPatch(faceID, uv[0], uv[1]);

                Assert(handle);
                const auto &cvIndices = patchTable->GetPatchVertices(*handle);

                f32 pWeights[20];
                f32 duWeights[20];
                f32 dvWeights[20];
                patchTable->EvaluateBasis(*handle, uv[0], uv[1], pWeights, duWeights,
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

                // fundamentally I just do't understand how to get the adjacency information
                // using opensubdiv. what patches are next to what other patches...
                // i need that so that I can average displacements to prevent cracks,
                // potentially to save memory (by not including edge vertices twice)

                // Vector Displacement mapping
                Vec3f normal = Normalize(Cross(dpdu, dpdv));
                auto frame   = LinearSpace3f::FromXZ(Normalize(dpdu), normal);
                // TODO get from texture. need footprints somehow
                Vec3f displacement;

                displacement = frame.FromLocal(displacement);
                pos += displacement;

                grid[m * maxEdgeRates[0] + n] = pos;
            }
        }

        // Generate connective triangle to compensate for differing edge rates
        for (u32 i = 0; i < 4; i++)
        {
            int edgeRate = edgeRates[i];

            // Stiching :)
            u32 u         = i & 1;
            u32 v         = (i + 1) & 1;
            int edgeRateU = maxEdgeRates[u];

            // Generate indices
            if (edgeRate < edgeRateU)
            {
                int edgeRateV = maxEdgeRates[v];
                f32 vMax      = scale[v] * (edgeRateV - 1);
                int q         = edgeRateU - edgeRate;

                f32 scaleN   = 1.f / edgeRate;
                int steps[2] = {};

                int stepM = 0;
                int stepN = 0;

                while (stepsM != maxEdgeRates[u] - 1 && stepN != edgeRate)
                {
                    bool currentSide = q >= 0;
                    if (currentSide)
                    {
                        int uOffset = u == 0 ? stepM : vMax;
                        int vOffset = v == 0 ? vMax : stepM;
                        u32 id0     = vOffset * maxEdgeRates[0] + uOffset;

                        vOffset += u == 0 ? 0 : 1;
                        uOffset += u == 0 ? 1 : 0;
                        u32 id1 = vOffset * maxEdgeRates[0] + uOffset;

                        int id2 = (int)grid.size();
                        Vec2f uv2(stepN * scaleN, 1.f);

                        q -= 2 * edgeRate;
                        stepM++;
                    }
                    else
                    {
                        int id0 = (int)grid.size();
                        grid.push_back();
                        int id1 = (int)grid.size();

                        Vec2f uv0(stepN * scaleN, 1.f);
                        Vec2f uv1((stepN + 1) * scaleN, 1.f);
                        // Vec2f uv2(stepM * scale[u], vMax);

                        q += 2 * edgeRateU;
                        stepN++;
                    }
                }
            }
        }
        delete refiner;
        delete patchTable;
    }
}

} // namespace rt
