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
    // first do everything up front, and then do caching maybe, see embree's paper
    typedef Far::TopologyDescriptor Descriptor;
    Sdc::SchemeType type = OpenSubdiv::Sdc::SCHEME_CATMARK;
    Sdc::Options options;
    options.SetVtxBoundaryInterpolation(Sdc::Options::VTX_BOUNDARY_EDGE_AND_CORNER);

    i32 *numVertsPerFace = PushArrayNoZero(temp.arena, i32, mesh->numFaces);
    for (u32 i = 0; i < mesh->numFaces; i++)
    {
        numVertsPerFace[i] = 4;
    }
    Descriptor desc;
    desc.numVertices        = mesh->numVertices;
    desc.numFaces           = mesh->numFaces;
    desc.numVertsPerFace    = numVertsPerFace;
    desc.vertIndicesPerFace = (i32 *)mesh->indices;

    Far::TopologyRefiner *refiner = Far::TopologyRefinerFactory<Descriptor>::Create(
        desc, Far::TopologyRefinerFactory<Descriptor>::Options(type, options));

    int maxLevel = 3;
    // refiner->RefineUniform(Far::TopologyRefiner::UniformOptions(maxLevel));

    // TODOs:
    // 1. how do you interpolate the normals durign subdivision?
    // 2. make this work with ptex
    // 3. adaptive subdivision based on screen space heuristics?
    // 4. multi resolution geometry cache?

    Far::PatchTableFactory::Options patchOptions(maxLevel);
    patchOptions.SetEndCapType(Far::PatchTableFactory::Options::ENDCAP_GREGORY_BASIS);

    Far::TopologyRefiner::AdaptiveOptions adaptiveOptions =
        patchOptions.GetRefineAdaptiveOptions();
    refiner->RefineAdaptive(adaptiveOptions);
    auto *patchTable = Far::PatchTableFactory::Create(*refiner, patchOptions);

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

    u32 numPatches = patchTable->GetNumPatchesTotal();
    Far::PtexIndices ptexIndices(*refiner);
    int numFaces = ptexIndices.GetNumFaces();
    OpenSubdiv::Far::PatchMap patchMap(*patchTable);
    Vec2f uv(0.5f);

    for (int i = 0; i < numFaces; i++)
    {
        i32 faceID                                 = ptexIndices.GetFaceId(i);
        const Far::PatchTable::PatchHandle *handle = patchMap.FindPatch(faceID, uv[0], uv[1]);
        Assert(handle);
        const auto &cvIndices = patchTable->GetPatchVertices(*handle);

        f32 pWeights[20];
        f32 duWeights[20];
        f32 dvWeights[20];
        patchTable->EvaluateBasis(*handle, uv[0], uv[1], pWeights, duWeights, dvWeights);

        Vec3f pos  = {};
        Vec3f dpdu = {};
        Vec3f dpdv = {};
        for (int j = 0; j < cvIndices.size(); j++)
        {
            u32 index      = cvIndices[j];
            const Vec3f &p = vertices[index].p;
            pos += pWeights[j] * p;
            dpdu += duWeights[j] * p;
            dpdv += dvWeights[j] * p;
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

// See Figure 7 from above for how this works
void Something(const Far::PatchTable *patchTable, u32 faceID)
{
    Vec4<int> edgeRates = {};

    int maxEdgeRates[2] = {
        Max(edgeRates[0], edgeRates[2]),
        Max(edgeRates[1], edgeRates[3]),
    };
    f32 scale[2] = {
        1.f / maxEdgeRates[0],
        1.f / maxEdgeRates[1],
    };

    std::vector<Vec3f> grid;
    grid.reserve(maxEdgeRates[0] * maxEdgeRates[1]);

    OpenSubdiv::Far::PatchMap patchMap(*patchTable);

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
            patchTable->EvaluateBasis(*handle, uv[0], uv[1], pWeights, duWeights, dvWeights);

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
}

} // namespace rt
